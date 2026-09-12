/*
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*
 * PATCH(fence-unify) [gt4o4 fork, vs upstream 615.71.09; first written against 610.43.03]
 *
 * This file exposes two dma-fence classes to userspace, both unchanged at the
 * ioctl/UAPI level:
 *
 *   - "prime" fences (nvidia.prime): implicit-sync fences attached to exported
 *     GEM objects (DRM_NVIDIA_GEM_PRIME_FENCE_ATTACH). A PRIME/reverse-PRIME
 *     display sink (e.g. amdgpu) blocks its atomic commit on these. Backed by a
 *     single persistent NVKMS channel event over an imported+mapped memory
 *     surface; 32-bit semaphore payload.
 *
 *   - "semsurf" fences (nvidia.semaphore_surface): explicit-sync timeline
 *     fences (Vulkan/Wayland, sync-fd create/wait). Backed by dynamically
 *     (re)registered NVKMS semaphore-surface callbacks; 64-bit payload with
 *     32-bit-GPU reconstruction.
 *
 * Both classes share ONE timeline engine, hoisted into struct
 * nv_drm_fence_context: a seqno-sorted pending list drained by the common
 * "signal while fence_seqno <= ctx payload, else break" loop, a per-context
 * nv_drm_workthread + one-shot nv_drm_timer that force-signals any fence past
 * its NV_DRM_FENCE_MAX_TIMEOUT_MS deadline with -ETIMEDOUT, and a shared
 * force-complete-on-teardown. The two backends differ only in a 2-op vtable:
 *
 *   .read_seqno - read the context's current completion payload.
 *   .update     - after draining, (re)arm the backend's wakeup. semsurf
 *                 (re)registers an RM callback + arms the timer; prime only
 *                 arms the timer, since its channel event is always live.
 *
 * A fence attached after its threshold has already passed (e.g. on an idle
 * timeline whose completion event fired long ago) is drained at zero latency:
 * .update reads the live payload at attach and arms the timeout for the rest.
 *
 * PATCH(prime-nonstall-wakeup) [gt4o4 fork]:
 *
 * Genuinely losing a channel event is real on GSP systems: the prime channel
 * event originates in GSP-RM firmware (POST_EVENT RPC), and losing the one
 * for the last frame before the timeline goes idle left the fence stalled on
 * the timeout timer (observed as occasional multi-second reverse-PRIME
 * freezes that self-recover; measured at ~1 lost event per 12,000 on GA104).
 * Two changes:
 *
 *   - The KAPI (nvkms-kapi-sync.c) now additionally registers a subdevice
 *     nonstall-interrupt callback (FIFO_EVENT_MTHD) for the same consumer,
 *     delivered with dataU32 == 1. The host nonstall interrupt is raised by
 *     the very same semaphore-release-with-AWAKEN and is serviced by CPU-RM
 *     directly, so it is a GSP-independent second wakeup. The prime event
 *     handler drains on it too (cheap empty-peek first, since it also fires
 *     for unrelated host traffic).
 *
 *   - Delivery diagnostics: per-context counters for channel-event vs
 *     nonstall deliveries, and a sparse log when fences complete only via the
 *     timer path (i.e. both event paths missed) -- previously silent.
 */

#include "nvidia-drm-conftest.h"

#if defined(NV_DRM_AVAILABLE)

#if defined(NV_DRM_DRMP_H_PRESENT)
#include <drm/drmP.h>
#endif

#include "nvidia-drm-priv.h"
#include "nvidia-drm-gem.h"
#include "nvidia-drm-fence.h"
#include "nvidia-dma-resv-helper.h"
#include "nv_drm_common_ioctl.h"

#include <linux/dma-fence.h>
#include <linux/atomic.h>

#ifndef READ_ONCE
#define READ_ONCE(x) ACCESS_ONCE(x)
#endif

/* Hard deadline after which a pending fence is force-signaled -ETIMEDOUT. */
#define NV_DRM_FENCE_MAX_TIMEOUT_MS 5000

#if defined(NV_DRM_SYNCOBJ_FEATURES_PRESENT)
#include <drm/drm_syncobj.h>
#endif

/*============================================================================
 * Shared fence-context base: types + vtable
 *==========================================================================*/

struct nv_drm_fence_context;

struct nv_drm_fence_context_ops {
    /* Release backend resources and free the context. */
    void (*destroy)(struct nv_drm_fence_context *nv_fence_context);

    /*
     * Read the context's current completion payload. Called with no lock held,
     * just before the drain takes ctx->lock; must be non-blocking (a plain
     * READ_ONCE) and safe from any context.
     */
    NvU64 (*read_seqno)(struct nv_drm_fence_context *nv_fence_context);

    /*
     * Drain completed/timed-out fences and (re)arm the backend's wakeup for the
     * next pending fence, if any. Runs in process context (workthread / ioctl);
     * never called with ctx->lock held. 'timer_initiated' is true when this
     * call was triggered by the timeout timer firing (as opposed to fence
     * creation or an event-deferred re-arm) -- diagnostics only; backends may
     * ignore it.
     */
    void (*update)(struct nv_drm_fence_context *nv_fence_context,
                   NvBool timer_initiated);
};

struct nv_drm_fence_context {
    struct nv_drm_gem_object base;

    const struct nv_drm_fence_context_ops *ops;

    struct nv_drm_device *nv_dev;
    uint64_t context;

    NvU64 fenceSemIndex; /* Index into semaphore surface */

    /*
     * Unified signaling/timeout engine. 'lock' protects 'pending_fences' and
     * 'current_wait_value'. The worker runs deferred timeout/registration work;
     * the one-shot timer is armed to the earliest pending deadline.
     */
    spinlock_t lock;
    struct list_head pending_fences;
    nv_drm_workthread worker;
    nv_drm_timer timer;
    nv_drm_work timeout_work;

    /*
     * Payload the backend's wakeup is (being) armed for; 0 if none needed. Used
     * to dedupe redundant re-arms. See __nv_drm_fence_context_process().
     */
    NvU64 current_wait_value;

    /*
     * Latched by the timer callback, consumed (xchg 0) by the worker before it
     * calls ops->update, so update() knows the drain was timer-initiated.
     */
    atomic_t timer_fired;
};

struct nv_drm_fence {
    struct dma_fence base;
    spinlock_t lock;

    /*
     * When unsignaled, node in the owning context's pending fence list, which
     * holds a reference to the fence.
     */
    struct list_head pending_node;

#if !defined(NV_DMA_FENCE_OPS_HAS_USE_64BIT_SEQNO)
    /* 64-bit version of base.seqno on kernels with 32-bit fence seqno */
    NvU64 wait_value;
#endif

    /* Absolute kernel-time (nv_drm_timer_now() domain) deadline. */
    unsigned long timeout;
};

static inline struct nv_drm_fence *to_nv_drm_fence(struct dma_fence *fence)
{
    return container_of(fence, struct nv_drm_fence, base);
}

static inline NvU64
__nv_drm_fence_seqno(const struct nv_drm_fence *nv_fence)
{
#if defined(NV_DMA_FENCE_OPS_HAS_USE_64BIT_SEQNO)
    return nv_fence->base.seqno;
#else
    return nv_fence->wait_value;
#endif
}

/*============================================================================
 * Shared dma_fence ops (behavior identical for both classes)
 *==========================================================================*/

static const char*
nv_drm_gem_fence_op_get_driver_name(struct dma_fence *fence)
{
    return "NVIDIA";
}

static bool __nv_drm_fence_op_enable_signaling(struct dma_fence *fence)
{
    /*
     * Nothing to do: both backends arm their wakeup when a fence is added to
     * the context, not lazily on first wait.
     */
    return true;
}

static void __nv_drm_fence_op_release(struct dma_fence *fence)
{
    struct nv_drm_fence *nv_fence = to_nv_drm_fence(fence);
    nv_drm_free(nv_fence);
}

/*============================================================================
 * Shared timeline engine
 *==========================================================================*/

/* Optional per-drain outcome counts, for delivery diagnostics. */
struct nv_drm_fence_drain_stats {
    unsigned int nSignaled;  /* payload reached: signaled success */
    unsigned int nTimedOut;  /* deadline passed: force-signaled -ETIMEDOUT */
};

/*
 * Drain the pending list: signal every head fence whose payload has been
 * reached (no error), and force-signal (-ETIMEDOUT) every head fence past its
 * deadline. The list is kept in increasing seqno order, so the walk stops at
 * the first fence that is neither complete nor timed out.
 *
 * If both out-params are non-NULL this establishes the re-arm contract: on
 * return *newWaitValueOut / *newTimeoutOut are the payload/deadline the caller
 * must arm a wakeup for, or 0/0 if no wakeup is needed (list empty, or a wakeup
 * for that payload is already registered per current_wait_value).
 *
 * Fences are collected under ctx->lock and signaled after dropping it, so
 * foreign dma_fence callbacks never run under our lock.
 */
static void
__nv_drm_fence_context_process(struct nv_drm_fence_context *ctx,
                               NvU64 *newWaitValueOut,
                               unsigned long *newTimeoutOut,
                               struct nv_drm_fence_drain_stats *statsOut)
{
    struct list_head finished;
    struct list_head timed_out;
    struct nv_drm_fence *nv_fence;
    struct dma_fence *fence;
    NvU64 currentSeqno = ctx->ops->read_seqno(ctx);
    NvU64 fenceSeqno = 0;
    unsigned long flags;
    unsigned long fenceTimeout = 0;
    unsigned long now = nv_drm_timer_now();

    INIT_LIST_HEAD(&finished);
    INIT_LIST_HEAD(&timed_out);

    if (statsOut) {
        statsOut->nSignaled = 0;
        statsOut->nTimedOut = 0;
    }

    spin_lock_irqsave(&ctx->lock, flags);

    while (!list_empty(&ctx->pending_fences)) {
        nv_fence = list_first_entry(&ctx->pending_fences,
                                    typeof(*nv_fence),
                                    pending_node);

        fenceSeqno = __nv_drm_fence_seqno(nv_fence);
        fenceTimeout = nv_fence->timeout;

        if (fenceSeqno <= currentSeqno) {
            list_move_tail(&nv_fence->pending_node, &finished);
        } else if (fenceTimeout <= now) {
            list_move_tail(&nv_fence->pending_node, &timed_out);
        } else {
            break;
        }
    }

    /*
     * See the re-arm contract in the function comment. NOTE: the timer deadline
     * (*newTimeoutOut) is decoupled from the RM-callback dedup (*newWaitValueOut):
     * the timeout timer must always track the earliest (head) pending fence, so
     * that a reordered stale mod_timer cannot permanently drop the backstop --
     * only the (expensive) backend wakeup registration is deduped.
     */
    if (newWaitValueOut && newTimeoutOut) {
        if (list_empty(&ctx->pending_fences)) {
            /* No pending fences: no wakeup and no timer. */
            ctx->current_wait_value = 0;
            *newWaitValueOut = 0;
            *newTimeoutOut = 0;
        } else {
            /* Timer always tracks the head deadline. */
            *newTimeoutOut = fenceTimeout;

            if (fenceSeqno == ctx->current_wait_value) {
                /*
                 * A backend wakeup is already registered, or in the process of
                 * being registered, for this fence -- no new one is needed.
                 */
                *newWaitValueOut = 0;
            } else {
                /* A new backend wakeup must be registered. Prep the context. */
                ctx->current_wait_value = fenceSeqno;
                *newWaitValueOut = fenceSeqno;
            }
        }
    }

    spin_unlock_irqrestore(&ctx->lock, flags);

    while (!list_empty(&finished)) {
        nv_fence = list_first_entry(&finished, typeof(*nv_fence), pending_node);
        list_del_init(&nv_fence->pending_node);
        fence = &nv_fence->base;
        dma_fence_signal(fence);
        dma_fence_put(fence); /* Drops the pending list's reference */
        if (statsOut) {
            statsOut->nSignaled++;
        }
    }

    while (!list_empty(&timed_out)) {
        nv_fence = list_first_entry(&timed_out, typeof(*nv_fence),
                                    pending_node);
        list_del_init(&nv_fence->pending_node);
        fence = &nv_fence->base;
        dma_fence_set_error(fence, -ETIMEDOUT);
        dma_fence_signal(fence);
        dma_fence_put(fence); /* Drops the pending list's reference */
        if (statsOut) {
            statsOut->nTimedOut++;
        }
    }
}

/*
 * Force-signal every pending fence and empty the list. 'error' is applied to
 * each fence when nonzero (semsurf teardown uses -ETIMEDOUT; prime uses 0 for
 * both teardown and seqno-wrap). Safe to call whether or not other references
 * to the context still exist: the list is spliced under the lock, then the
 * fences are signaled outside it.
 */
static void
__nv_drm_fence_context_force_complete(struct nv_drm_fence_context *ctx,
                                      int error)
{
    struct list_head local;
    struct nv_drm_fence *nv_fence;
    struct dma_fence *fence;
    unsigned long flags;

    INIT_LIST_HEAD(&local);

    spin_lock_irqsave(&ctx->lock, flags);
    list_splice_init(&ctx->pending_fences, &local);
    ctx->current_wait_value = 0;
    spin_unlock_irqrestore(&ctx->lock, flags);

    while (!list_empty(&local)) {
        nv_fence = list_first_entry(&local, typeof(*nv_fence), pending_node);
        list_del_init(&nv_fence->pending_node);
        fence = &nv_fence->base;
        if (error) {
            dma_fence_set_error(fence, error);
        }
        dma_fence_signal(fence);
        dma_fence_put(fence); /* Drops the pending list's reference */
    }
}

/*
 * Complete a fence's setup, insert it (seqno-sorted) into the pending list with
 * a reference, and arm the backend's wakeup. Can NOT be called from atomic
 * context: ops->update may call into RM.
 */
static void
__nv_drm_fence_context_add_pending(struct nv_drm_fence_context *ctx,
                                   struct nv_drm_fence *nv_fence,
                                   NvU64 timeoutMS)
{
    struct list_head *pending;
    unsigned long flags;

    if (timeoutMS == 0 || timeoutMS > NV_DRM_FENCE_MAX_TIMEOUT_MS) {
        timeoutMS = NV_DRM_FENCE_MAX_TIMEOUT_MS;
    }

    /* Add a reference to the fence for the list */
    dma_fence_get(&nv_fence->base);
    INIT_LIST_HEAD(&nv_fence->pending_node);

    nv_fence->timeout = nv_drm_timeout_from_ms(timeoutMS);

    spin_lock_irqsave(&ctx->lock, flags);

    list_for_each(pending, &ctx->pending_fences) {
        struct nv_drm_fence *pending_fence =
            list_entry(pending, typeof(*pending_fence), pending_node);
        if (__nv_drm_fence_seqno(pending_fence) >
            __nv_drm_fence_seqno(nv_fence)) {
            /* Inserts 'nv_fence->pending_node' before 'pending' */
            list_add_tail(&nv_fence->pending_node, pending);
            break;
        }
    }

    if (list_empty(&nv_fence->pending_node)) {
        /*
         * Inserts at the end of 'ctx->pending_fences', or as the head if the
         * list is empty.
         */
        list_add_tail(&nv_fence->pending_node, &ctx->pending_fences);
    }

    /* Fence is live starting... now! */
    spin_unlock_irqrestore(&ctx->lock, flags);

    /* Drain anything already complete and (re)arm the wakeup. */
    ctx->ops->update(ctx, NV_FALSE);
}

static void
__nv_drm_fence_context_timeout_work(void *data)
{
    struct nv_drm_fence_context *ctx = data;

    /*
     * Consume the timer latch: this pass is timer-initiated iff the timeout
     * timer fired since the last worker pass (as opposed to an event handler
     * deferring a re-arm here).
     */
    NvBool timer_initiated = (atomic_xchg(&ctx->timer_fired, 0) != 0);

    ctx->ops->update(ctx, timer_initiated);
}

static void
__nv_drm_fence_context_timeout_callback(nv_drm_timer *timer)
{
    struct nv_drm_fence_context *ctx =
        container_of(timer, typeof(*ctx), timer);

    /*
     * Defer the actual work (which may call into RM and must not run in the
     * timer softirq) to the worker. Failure is benign: either the work is
     * already scheduled (and will do at least as much), or the context is
     * shutting down (and will force-signal everything).
     *
     * The worker must be shut down before the timer during teardown so this
     * cannot re-arm a timer that is being idled.
     */
    atomic_set(&ctx->timer_fired, 1);
    nv_drm_workthread_add_work(&ctx->worker, &ctx->timeout_work);
}

/*
 * Initialize the shared engine. On success the caller owns a live workthread +
 * timer and must tear them down via __nv_drm_fence_context_teardown_engine().
 */
static bool
__nv_drm_fence_context_init(struct nv_drm_fence_context *ctx,
                            struct nv_drm_device *nv_dev,
                            const struct nv_drm_fence_context_ops *ops,
                            NvU64 fenceSemIndex)
{
    /* strlen("nvidia-drm/timeline-") + 16 for %llx + NUL */
    char worker_name[20+16+1];

    /*
     * dma_fence_context_alloc() cannot fail, so we do not need to check a
     * return value.
     */
    ctx->ops = ops;
    ctx->nv_dev = nv_dev;
    ctx->context = dma_fence_context_alloc(1);
    ctx->fenceSemIndex = fenceSemIndex;
    ctx->current_wait_value = 0;
    atomic_set(&ctx->timer_fired, 0);

    spin_lock_init(&ctx->lock);
    INIT_LIST_HEAD(&ctx->pending_fences);

    sprintf(worker_name, "nvidia-drm/timeline-%llx",
            (long long unsigned)ctx->context);
    if (!nv_drm_workthread_init(&ctx->worker, worker_name)) {
        return false;
    }

    nv_drm_workthread_work_init(&ctx->timeout_work,
                                __nv_drm_fence_context_timeout_work,
                                ctx);

    nv_drm_timer_setup(&ctx->timer, __nv_drm_fence_context_timeout_callback);

    return true;
}

/*
 * Idle the engine. The workthread must be shut down before the timer is stopped
 * so the timer cannot queue work that restarts itself.
 */
static void
__nv_drm_fence_context_teardown_engine(struct nv_drm_fence_context *ctx)
{
    nv_drm_workthread_shutdown(&ctx->worker);
    nv_timer_delete_sync(&ctx->timer.kernel_timer);
}

/*============================================================================
 * Shared fence-context-as-GEM plumbing
 *==========================================================================*/

int nv_drm_fence_supported_ioctl(struct drm_device *dev,
                                 void *data, struct drm_file *filep)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    return nv_dev->pDevice ? 0 : -EINVAL;
}

static inline struct nv_drm_fence_context *to_nv_fence_context(
    struct nv_drm_gem_object *nv_gem)
{
    if (nv_gem != NULL) {
        return container_of(nv_gem, struct nv_drm_fence_context, base);
    }

    return NULL;
}

/*
 * Tear down of the 'struct nv_drm_fence_context' object is not expected
 * to be happen from any worker thread, if that happen it causes dead-lock
 * because tear down sequence calls to flush all existing
 * worker thread.
 */
static void
__nv_drm_fence_context_gem_free(struct nv_drm_gem_object *nv_gem)
{
    struct nv_drm_fence_context *nv_fence_context = to_nv_fence_context(nv_gem);

    nv_fence_context->ops->destroy(nv_fence_context);
}

const struct nv_drm_gem_object_funcs nv_fence_context_gem_ops = {
    .free = __nv_drm_fence_context_gem_free,
};

static inline
struct nv_drm_fence_context *
__nv_drm_fence_context_lookup(
    struct drm_file *filp,
    u32 handle)
{
    struct nv_drm_gem_object *nv_gem =
            nv_drm_gem_object_lookup(filp, handle);

    if (nv_gem != NULL && nv_gem->ops != &nv_fence_context_gem_ops) {
        nv_drm_gem_object_unreference_unlocked(nv_gem);
        return NULL;
    }

    return to_nv_fence_context(nv_gem);
}

/*
 * Look up a fence context handle and verify it is of the expected backend type.
 * On success returns the context holding a reference the caller must drop with
 * nv_drm_gem_object_unreference_unlocked(&ctx->base); on failure logs and
 * returns NULL.
 */
static struct nv_drm_fence_context *
__nv_drm_fence_context_lookup_typed(
    struct nv_drm_device *nv_dev,
    struct drm_file *filep,
    u32 handle,
    const struct nv_drm_fence_context_ops *expected_ops)
{
    struct nv_drm_fence_context *nv_fence_context =
        __nv_drm_fence_context_lookup(filep, handle);

    if (nv_fence_context == NULL) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to lookup gem object for fence context: 0x%08x",
            handle);
        return NULL;
    }

    if (nv_fence_context->ops != expected_ops) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Wrong fence context type: 0x%08x",
            handle);
        nv_drm_gem_object_unreference_unlocked(&nv_fence_context->base);
        return NULL;
    }

    return nv_fence_context;
}

static int
__nv_drm_fence_context_gem_init(struct drm_device *dev,
                                struct nv_drm_fence_context *nv_fence_context,
                                u32 *handle,
                                struct drm_file *filep)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);

    nv_drm_gem_object_init(nv_dev,
                           &nv_fence_context->base,
                           &nv_fence_context_gem_ops,
                           0 /* size */,
                           NULL /* pMemory */);

    return nv_drm_gem_handle_create_drop_reference(filep,
                                                   &nv_fence_context->base,
                                                   handle);
}

static int __nv_drm_gem_attach_fence(struct nv_drm_gem_object *nv_gem,
                                     struct dma_fence *fence,
                                     bool shared)
{
    nv_dma_resv_t *resv = nv_drm_gem_res_obj(nv_gem);
    int ret;

    nv_dma_resv_lock(resv, NULL);

    ret = nv_dma_resv_reserve_fences(resv, 1, shared);
    if (ret == 0) {
        if (shared) {
            nv_dma_resv_add_shared_fence(resv, fence);
        } else {
            nv_dma_resv_add_excl_fence(resv, fence);
        }
    } else {
        NV_DRM_LOG_ERR("Failed to reserve fence. Error code: %d", ret);
    }

    nv_dma_resv_unlock(resv);

    return ret;
}

/*============================================================================
 * Prime fence backend (nvidia.prime)
 *==========================================================================*/

struct nv_drm_prime_fence_context {
    struct nv_drm_fence_context base;

    /* Mapped semaphore surface */
    struct NvKmsKapiMemory *pSemSurface;
    NvU32 *pLinearAddress;

    /*
     * Software signaling structures. __nv_drm_prime_fence_context_new()
     * allocates channel event and __nv_drm_prime_fence_context_destroy() frees
     * it. There are no simultaneous read/write access to 'cb', therefore it
     * does not require spin-lock protection.
     */
    struct NvKmsKapiChannelEvent *cb;

    /* Last seqno handed out, for 32-bit wrap detection. */
    unsigned last_seqno;

    /*
     * Delivery diagnostics (PATCH(prime-nonstall-wakeup)). The event handler
     * runs concurrently with the worker, so these are lock-free atomics.
     */
    atomic_long_t stat_channel_events;   /* GSP channel-event deliveries */
    atomic_long_t stat_nonstall_kicks;   /* nonstall deliveries, fences pending */
    atomic_long_t stat_nonstall_idle;    /* nonstall deliveries, list empty */
    atomic_long_t stat_timer_recoveries; /* fences completed only by the timer */
    atomic_long_t stat_timeouts;         /* fences force-completed -ETIMEDOUT */
    NvBool nonstall_live_logged;         /* one-time "path is live" log gate */
};

static inline struct nv_drm_prime_fence_context*
to_nv_prime_fence_context(struct nv_drm_fence_context *nv_fence_context) {
    return container_of(nv_fence_context, struct nv_drm_prime_fence_context, base);
}

static const char*
nv_drm_gem_prime_fence_op_get_timeline_name(struct dma_fence *fence)
{
    return "nvidia.prime";
}

static const struct dma_fence_ops nv_drm_gem_prime_fence_ops = {
    .get_driver_name = nv_drm_gem_fence_op_get_driver_name,
    .get_timeline_name = nv_drm_gem_prime_fence_op_get_timeline_name,
    .enable_signaling = __nv_drm_fence_op_enable_signaling,
    .release = __nv_drm_fence_op_release,
    .wait = dma_fence_default_wait,
    /*
     * NB: prime uses a 32-bit semaphore timeline and relies on wrap detection
     * at create time, so it must NOT set use_64bit_seqno.
     */
};

static NvU64
__nv_drm_prime_fence_ctx_read_seqno(struct nv_drm_fence_context *nv_fence_context)
{
    struct nv_drm_prime_fence_context *ctx =
        to_nv_prime_fence_context(nv_fence_context);

    /* Index into surface with 16 byte stride */
    return (NvU64) READ_ONCE(*(ctx->pLinearAddress +
                               (nv_fence_context->fenceSemIndex * 4)));
}

static void
__nv_drm_prime_fence_ctx_update(struct nv_drm_fence_context *nv_fence_context,
                                NvBool timer_initiated)
{
    struct nv_drm_prime_fence_context *prime_ctx =
        to_nv_prime_fence_context(nv_fence_context);
    struct nv_drm_fence_drain_stats stats;
    NvU64 newWaitValue;
    unsigned long newTimeout;

    /*
     * Drain anything the channel event / semaphore has completed, and (re)arm
     * the timeout timer to the earliest remaining fence's deadline. Prime's
     * channel event is always live, so the RM-registration signal
     * (newWaitValue) is unused here; the timer is keyed to the head deadline
     * (newTimeout), so a reordered stale arm self-heals on the next fire rather
     * than dropping the backstop. Called only from the worker and from create
     * (never the event handlers), so mod_timer cannot race teardown's
     * del_timer_sync.
     */
    __nv_drm_fence_context_process(nv_fence_context, &newWaitValue, &newTimeout,
                                   &stats);

    /*
     * Diagnostics: a fence completing in a TIMER-initiated drain means both
     * event paths (GSP channel event and CPU nonstall broadcast) missed it and
     * the waiting sink stalled until the timer deadline. Log sparsely -- this
     * was previously silent.
     */
    if (timer_initiated && stats.nSignaled != 0) {
        long n = atomic_long_add_return(stats.nSignaled,
                                        &prime_ctx->stat_timer_recoveries);
        if (n <= 8 || (n % 64) == 0) {
            NV_DRM_DEV_LOG_INFO(
                nv_fence_context->nv_dev,
                "prime fence: %u fence(s) completed only by the timeout timer "
                "(payload had advanced, both event paths missed); "
                "totals: timer=%ld ch_ev=%ld ns_kick=%ld ns_idle=%ld",
                stats.nSignaled, n,
                atomic_long_read(&prime_ctx->stat_channel_events),
                atomic_long_read(&prime_ctx->stat_nonstall_kicks),
                atomic_long_read(&prime_ctx->stat_nonstall_idle));
        }
    }

    if (stats.nTimedOut != 0) {
        long n = atomic_long_add_return(stats.nTimedOut,
                                        &prime_ctx->stat_timeouts);
        if (n <= 8 || (n % 64) == 0) {
            NV_DRM_DEV_LOG_ERR(
                nv_fence_context->nv_dev,
                "prime fence: %u fence(s) timed out (-ETIMEDOUT, payload never "
                "advanced); total: %ld",
                stats.nTimedOut, n);
        }
    }

    if (newTimeout != 0) {
        nv_drm_mod_timer(&nv_fence_context->timer, newTimeout);
    }
}

static void nv_drm_gem_prime_fence_event
(
    void *dataPtr,
    NvU32 dataU32
)
{
    struct nv_drm_prime_fence_context *nv_prime_fence_context = dataPtr;
    struct nv_drm_fence_context *ctx = &nv_prime_fence_context->base;

    if (dataU32 != 0) {
        /*
         * Redundant nonstall-broadcast delivery (PATCH(prime-nonstall-wakeup),
         * see nvkms-kapi-sync.c): raised by the same semaphore release as the
         * channel event, but serviced by CPU-RM independent of GSP. It also
         * fires for unrelated host nonstall traffic, so peek first: on an idle
         * timeline the cost is one spinlock + list_empty.
         */
        unsigned long flags;
        bool have_pending;

        if (!nv_prime_fence_context->nonstall_live_logged) {
            /* Benign race: at most a duplicate log line. */
            nv_prime_fence_context->nonstall_live_logged = NV_TRUE;
            NV_DRM_DEV_LOG_INFO(
                ctx->nv_dev,
                "prime fence: redundant nonstall wakeup path is live");
        }

        spin_lock_irqsave(&ctx->lock, flags);
        have_pending = !list_empty(&ctx->pending_fences);
        spin_unlock_irqrestore(&ctx->lock, flags);

        if (!have_pending) {
            atomic_long_inc(&nv_prime_fence_context->stat_nonstall_idle);
            return;
        }

        atomic_long_inc(&nv_prime_fence_context->stat_nonstall_kicks);
    } else {
        atomic_long_inc(&nv_prime_fence_context->stat_channel_events);
    }

    /*
     * Signal any completed fences immediately (low latency for the waiting
     * sink), but do NOT arm the timer from this NVKMS event context: defer the
     * (re-)arm to the worker, which teardown drains before del_timer_sync, so no
     * mod_timer can outlive the fence context. Mirrors the semsurf callback.
     */
    __nv_drm_fence_context_process(ctx, NULL, NULL, NULL);
    nv_drm_workthread_add_work(&ctx->worker, &ctx->timeout_work);
}

static void __nv_drm_prime_fence_context_destroy(
    struct nv_drm_fence_context *nv_fence_context)
{
    struct nv_drm_device *nv_dev = nv_fence_context->nv_dev;
    struct nv_drm_prime_fence_context *nv_prime_fence_context =
        to_nv_prime_fence_context(nv_fence_context);

    /*
     * Free the event callbacks before idling the engine, otherwise their
     * handlers (which drain the pending list and enqueue worker work, and
     * dereference this context) continue to get called.
     */
    nvKms->freeChannelEvent(nv_dev->pDevice, nv_prime_fence_context->cb);

    /* Idle the worker + timer, then force-signal whatever remains. */
    __nv_drm_fence_context_teardown_engine(nv_fence_context);
    __nv_drm_fence_context_force_complete(nv_fence_context, 0);

    /* Free nvkms resources */

    nvKms->unmapMemory(nv_dev->pDevice,
                       nv_prime_fence_context->pSemSurface,
                       NVKMS_KAPI_MAPPING_TYPE_KERNEL,
                       (void *) nv_prime_fence_context->pLinearAddress);

    nvKms->freeMemory(nv_dev->pDevice, nv_prime_fence_context->pSemSurface);

    nv_drm_free(nv_fence_context);
}

static struct nv_drm_fence_context_ops nv_drm_prime_fence_context_ops = {
    .destroy = __nv_drm_prime_fence_context_destroy,
    .read_seqno = __nv_drm_prime_fence_ctx_read_seqno,
    .update = __nv_drm_prime_fence_ctx_update,
};

static inline struct nv_drm_prime_fence_context *
__nv_drm_prime_fence_context_new(
    struct nv_drm_device *nv_dev,
    struct drm_nvidia_prime_fence_context_create_params *p)
{
    struct nv_drm_prime_fence_context *nv_prime_fence_context;
    struct NvKmsKapiMemory *pSemSurface;
    NvU32 *pLinearAddress;

    /* Allocate backup nvkms resources */

    pSemSurface = nvKms->importMemory(nv_dev->pDevice,
                                      p->size,
                                      p->import_mem_nvkms_params_ptr,
                                      p->import_mem_nvkms_params_size);
    if (!pSemSurface) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to import fence semaphore surface");

        goto failed;
    }

    if (!nvKms->mapMemory(nv_dev->pDevice,
                          pSemSurface,
                          NVKMS_KAPI_MAPPING_TYPE_KERNEL,
                          (void **) &pLinearAddress)) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to map fence semaphore surface");

        goto failed_to_map_memory;
    }

    /*
     * Allocate a fence context object, initialize it and allocate channel
     * event for it.
     */

    if ((nv_prime_fence_context = nv_drm_calloc(
                    1,
                    sizeof(*nv_prime_fence_context))) == NULL) {
        goto failed_alloc_fence_context;
    }

    if (!__nv_drm_fence_context_init(&nv_prime_fence_context->base,
                                     nv_dev,
                                     &nv_drm_prime_fence_context_ops,
                                     p->index)) {
        goto failed_ctx_init;
    }

    nv_prime_fence_context->pSemSurface = pSemSurface;
    nv_prime_fence_context->pLinearAddress = pLinearAddress;

    /*
     * The fence context should be completely initialized before channel event
     * allocation because the fence context may start receiving events
     * immediately after allocation.
     *
     * There are no simultaneous read/write access to 'cb', therefore it does
     * not require spin-lock protection.
     */
    nv_prime_fence_context->cb =
        nvKms->allocateChannelEvent(nv_dev->pDevice,
                                    nv_drm_gem_prime_fence_event,
                                    nv_prime_fence_context,
                                    p->event_nvkms_params_ptr,
                                    p->event_nvkms_params_size);
    if (!nv_prime_fence_context->cb) {
        NV_DRM_DEV_LOG_ERR(nv_dev,
                           "Failed to allocate fence signaling event");
        goto failed_to_allocate_channel_event;
    }

    return nv_prime_fence_context;

failed_to_allocate_channel_event:
    __nv_drm_fence_context_teardown_engine(&nv_prime_fence_context->base);

failed_ctx_init:
    nv_drm_free(nv_prime_fence_context);

failed_alloc_fence_context:

    nvKms->unmapMemory(nv_dev->pDevice,
                       pSemSurface,
                       NVKMS_KAPI_MAPPING_TYPE_KERNEL,
                       (void *) pLinearAddress);

failed_to_map_memory:
    nvKms->freeMemory(nv_dev->pDevice, pSemSurface);

failed:
    return NULL;
}

static struct dma_fence *__nv_drm_prime_fence_context_create_fence(
    struct nv_drm_prime_fence_context *nv_prime_fence_context,
    unsigned int seqno)
{
    struct nv_drm_fence_context *ctx = &nv_prime_fence_context->base;
    struct nv_drm_fence *nv_fence;
    struct nv_drm_fence *wrapped_fence;
    struct list_head wrapped;
    unsigned long flags;
    int ret = 0;

    if ((nv_fence = nv_drm_calloc(1, sizeof(*nv_fence))) == NULL) {
        ret = -ENOMEM;
        goto out;
    }

    INIT_LIST_HEAD(&wrapped);

    spin_lock_init(&nv_fence->lock);
#if !defined(NV_DMA_FENCE_OPS_HAS_USE_64BIT_SEQNO)
    nv_fence->wait_value = seqno;
#endif

    dma_fence_init(&nv_fence->base, &nv_drm_gem_prime_fence_ops,
                   &nv_fence->lock, ctx->context, seqno);

    /* The pending list holds a reference. */
    dma_fence_get(&nv_fence->base);
    INIT_LIST_HEAD(&nv_fence->pending_node);
    nv_fence->timeout = nv_drm_timeout_from_ms(NV_DRM_FENCE_MAX_TIMEOUT_MS);

    spin_lock_irqsave(&ctx->lock, flags);
    /*
     * If seqno wrapped, collect the outstanding fences so none get stuck behind
     * the new (smaller) seqno in the ordered list. The flush-collect and the
     * insert happen under one lock hold, so a concurrent creator cannot slip a
     * fence into the window and have it wrongly force-completed.
     */
    if (seqno < nv_prime_fence_context->last_seqno) {
        list_splice_init(&ctx->pending_fences, &wrapped);
        ctx->current_wait_value = 0;
    }
    nv_prime_fence_context->last_seqno = seqno;
    /* Prime seqnos are monotonic (post-flush), so the new fence is the tail. */
    list_add_tail(&nv_fence->pending_node, &ctx->pending_fences);
    spin_unlock_irqrestore(&ctx->lock, flags);

    /* Signal any wrapped-out fences (success, no error) outside the lock. */
    while (!list_empty(&wrapped)) {
        wrapped_fence = list_first_entry(&wrapped, typeof(*wrapped_fence),
                                         pending_node);
        list_del_init(&wrapped_fence->pending_node);
        dma_fence_signal(&wrapped_fence->base);
        dma_fence_put(&wrapped_fence->base); /* drops the pending list ref */
    }

    /*
     * The semaphore may already have passed this threshold (the channel event
     * that would have signaled us has fired, or will never fire again on an idle
     * timeline). update() drains immediately and arms the timeout, so the fence
     * cannot get stuck.
     */
    ctx->ops->update(ctx, NV_FALSE);

out:
    return ret != 0 ? ERR_PTR(ret) : &nv_fence->base;
}

int nv_drm_prime_fence_context_create_ioctl(struct drm_device *dev,
                                            void *data, struct drm_file *filep)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    struct drm_nvidia_prime_fence_context_create_params *p = data;
    struct nv_drm_prime_fence_context *nv_prime_fence_context;

    if (nv_dev->pDevice == NULL) {
        return -EOPNOTSUPP;
    }

    nv_prime_fence_context = __nv_drm_prime_fence_context_new(nv_dev, p);

    if (!nv_prime_fence_context) {
        return -ENOMEM;
    }

    /*
     * On failure, handle-create has already dropped the only reference, which
     * frees the context via gem_free -> ops->destroy. Do NOT destroy again here
     * (that would be a double-free); just return the error.
     */
    return __nv_drm_fence_context_gem_init(dev,
                                           &nv_prime_fence_context->base,
                                           &p->handle,
                                           filep);
}

int nv_drm_gem_prime_fence_attach_ioctl(struct drm_device *dev,
                                        void *data, struct drm_file *filep)
{
    int ret = -EINVAL;
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    struct drm_nvidia_gem_prime_fence_attach_params *p = data;

    struct nv_drm_gem_object *nv_gem;
    struct nv_drm_fence_context *nv_fence_context;
    struct dma_fence *fence;

    if (nv_dev->pDevice == NULL) {
        ret = -EOPNOTSUPP;
        goto done;
    }

    if (p->__pad != 0) {
        NV_DRM_DEV_LOG_ERR(nv_dev, "Padding fields must be zeroed");
        goto done;
    }

    nv_gem = nv_drm_gem_object_lookup(filep, p->handle);

    if (!nv_gem) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to lookup gem object for fence attach: 0x%08x",
            p->handle);

        goto done;
    }

    if ((nv_fence_context = __nv_drm_fence_context_lookup_typed(
                nv_dev,
                filep,
                p->fence_context_handle,
                &nv_drm_prime_fence_context_ops)) == NULL) {

        goto fence_context_lookup_failed;
    }

    fence = __nv_drm_prime_fence_context_create_fence(
                to_nv_prime_fence_context(nv_fence_context),
                p->sem_thresh);

    if (IS_ERR(fence)) {
        ret = PTR_ERR(fence);

        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to allocate fence: 0x%08x", p->handle);

        goto fence_context_create_fence_failed;
    }

    ret = __nv_drm_gem_attach_fence(nv_gem, fence, true /* exclusive */);

    dma_fence_put(fence);

fence_context_create_fence_failed:
    nv_drm_gem_object_unreference_unlocked(&nv_fence_context->base);

fence_context_lookup_failed:
    nv_drm_gem_object_unreference_unlocked(nv_gem);

done:
    return ret;
}

/*============================================================================
 * Semaphore-surface fence backend (nvidia.semaphore_surface)
 *==========================================================================*/

struct nv_drm_semsurf_fence_callback {
    struct nv_drm_semsurf_fence_ctx *ctx;
    nv_drm_work work;
    NvU64 wait_value;
};

struct nv_drm_sync_fd_wait_data {
    struct dma_fence_cb dma_fence_cb;
    struct nv_drm_semsurf_fence_ctx *ctx;
    nv_drm_work work; /* Deferred second half of fence wait callback */

    /* Could use a lockless list data structure here instead */
    struct list_head pending_node;

    NvU64 pre_wait_value;
    NvU64 post_wait_value;
};

struct nv_drm_semsurf_fence_ctx {
    struct nv_drm_fence_context base;

    /* The NVKMS KAPI reference to the context's semaphore surface */
    struct NvKmsKapiSemaphoreSurface *pSemSurface;

    /* CPU mapping of the semaphore slot values */
    union {
        volatile void *pVoid;
        volatile NvU32 *p32;
        volatile NvU64 *p64;
    } pSemMapping;
    volatile NvU64 *pMaxSubmittedMapping;

    /* List of pending fence wait operations */
    struct list_head pending_waits;

    /*
     * Tracking data for the single in-flight callback associated with this
     * context. Either both pointers will be valid, or both will be NULL.
     *
     * Note it is not safe to dereference these values outside of the context
     * lock unless it is certain the associated callback is not yet active,
     * or has been canceled. Their memory is owned by the callback itself as
     * soon as it is registered. Subtly, this means these variables can not
     * be used as output parameters to the function that registers the callback.
     *
     * The wait value the callback is (being) registered for lives in
     * base.current_wait_value.
     */
    struct {
        struct nv_drm_semsurf_fence_callback *local;
        struct NvKmsKapiSemaphoreSurfaceCallback *nvKms;
    } callback;
};

static inline struct nv_drm_semsurf_fence_ctx*
to_semsurf_fence_ctx(
    struct nv_drm_fence_context *nv_fence_context
)
{
    return container_of(nv_fence_context,
                        struct nv_drm_semsurf_fence_ctx,
                        base);
}

static NvU64
__nv_drm_semsurf_ctx_read_seqno(struct nv_drm_fence_context *nv_fence_context)
{
    struct nv_drm_semsurf_fence_ctx *ctx =
        to_semsurf_fence_ctx(nv_fence_context);
    NvU64 semVal;

    if (ctx->pMaxSubmittedMapping) {
        /* 32-bit GPU semaphores */
        NvU64 maxSubmitted = READ_ONCE(*ctx->pMaxSubmittedMapping);

        /*
         * Must happen after the max submitted read! See
         * NvTimeSemFermiGetPayload() for full details.
         */
        semVal = READ_ONCE(*ctx->pSemMapping.p32);

        if ((maxSubmitted & 0xFFFFFFFFull) < semVal) {
            maxSubmitted -= 0x100000000ull;
        }

        semVal |= (maxSubmitted & 0xffffffff00000000ull);
    } else {
        /* 64-bit GPU semaphores */
        semVal = READ_ONCE(*ctx->pSemMapping.p64);
    }

    return semVal;
}

/* Forward declaration */
static void
__nv_drm_semsurf_ctx_reg_callbacks(struct nv_drm_semsurf_fence_ctx *ctx);

static void
__nv_drm_semsurf_ctx_update(struct nv_drm_fence_context *nv_fence_context,
                            NvBool timer_initiated)
{
    /* timer_initiated is diagnostics-only; semsurf does not use it. */
    __nv_drm_semsurf_ctx_reg_callbacks(to_semsurf_fence_ctx(nv_fence_context));
}

static void
__nv_drm_semsurf_ctx_fence_callback_work(void *data)
{
    struct nv_drm_semsurf_fence_callback *callback = data;

    __nv_drm_semsurf_ctx_reg_callbacks(callback->ctx);

    nv_drm_free(callback);
}

static struct nv_drm_semsurf_fence_callback*
__nv_drm_semsurf_new_callback(struct nv_drm_semsurf_fence_ctx *ctx)
{
    struct nv_drm_semsurf_fence_callback *newCallback =
        nv_drm_calloc(1, sizeof(*newCallback));

    if (!newCallback) {
        return NULL;
    }

    newCallback->ctx = ctx;
    nv_drm_workthread_work_init(&newCallback->work,
                                __nv_drm_semsurf_ctx_fence_callback_work,
                                newCallback);

    return newCallback;
}

static void
__nv_drm_semsurf_ctx_callback(void *data)
{
    struct nv_drm_semsurf_fence_callback *callback = data;
    struct nv_drm_semsurf_fence_ctx *ctx = callback->ctx;
    unsigned long flags;

    spin_lock_irqsave(&ctx->base.lock, flags);
    /* If this was the context's currently registered callback, clear it. */
    if (ctx->callback.local == callback) {
        ctx->callback.local = NULL;
        ctx->callback.nvKms = NULL;
    }
    /* If storing of this callback may have been pending, prevent it. */
    if (ctx->base.current_wait_value == callback->wait_value) {
        ctx->base.current_wait_value = 0;
    }
    spin_unlock_irqrestore(&ctx->base.lock, flags);

    /*
     * This is redundant with the __nv_drm_semsurf_ctx_reg_callbacks() call from
     * __nv_drm_semsurf_ctx_fence_callback_work(), which will be called by the
     * work enqueued below, but calling it here as well allows unblocking
     * waiters with less latency.
     */
    __nv_drm_fence_context_process(&ctx->base, NULL, NULL, NULL);

    if (!nv_drm_workthread_add_work(&ctx->base.worker, &callback->work)) {
        /*
         * The context is shutting down. It will force-signal all fences when
         * doing so, so there's no need for any more callback handling.
         */
        nv_drm_free(callback);
    }
}

/*
 * Take spin lock, attempt to stash newNvKmsCallback/newCallback in ctx.
 * If current_wait_value in fence context != new_wait_value, we raced with
 * someone registering a newer waiter. Release spin lock, and unregister our
 * waiter. It isn't needed anymore.
 */
static bool
__nv_drm_semsurf_ctx_store_callback(
    struct nv_drm_semsurf_fence_ctx *ctx,
    NvU64 new_wait_value,
    struct NvKmsKapiSemaphoreSurfaceCallback *newNvKmsCallback,
    struct nv_drm_semsurf_fence_callback *newCallback)
{
    struct nv_drm_device *nv_dev = ctx->base.nv_dev;
    struct NvKmsKapiSemaphoreSurfaceCallback *oldNvKmsCallback;
    struct nv_drm_semsurf_fence_callback *oldCallback = NULL;
    NvU64 oldWaitValue;
    unsigned long flags;
    bool installed = false;

    spin_lock_irqsave(&ctx->base.lock, flags);
    if (ctx->base.current_wait_value == new_wait_value) {
        oldCallback = ctx->callback.local;
        oldNvKmsCallback = ctx->callback.nvKms;
        oldWaitValue = oldCallback ? oldCallback->wait_value : 0;
        ctx->callback.local = newCallback;
        ctx->callback.nvKms = newNvKmsCallback;
        installed = true;
    }
    spin_unlock_irqrestore(&ctx->base.lock, flags);

    if (oldCallback) {
        if (nvKms->unregisterSemaphoreSurfaceCallback(nv_dev->pDevice,
                                                      ctx->pSemSurface,
                                                      ctx->base.fenceSemIndex,
                                                      oldWaitValue,
                                                      oldNvKmsCallback)) {
            /*
             * The old callback was successfully canceled, and its NVKMS and RM
             * resources have been freed. Free its local tracking data.
             */
            nv_drm_free(oldCallback);
        } else {
            /*
             * The new callback is already running. It will do no harm, and free
             * itself.
             */
        }
    }

    return installed;
}

/*
 * Processes completed fences and registers an RM callback and a timeout timer
 * for the next incomplete fence, if any. To avoid calling in to RM while
 * holding a spinlock, this is done in a loop until the state settles.
 *
 * Can NOT be called from in an atomic context/interrupt handler.
 */
static void
__nv_drm_semsurf_ctx_reg_callbacks(struct nv_drm_semsurf_fence_ctx *ctx)

{
    struct nv_drm_device *nv_dev = ctx->base.nv_dev;
    struct nv_drm_semsurf_fence_callback *newCallback =
        __nv_drm_semsurf_new_callback(ctx);
    struct NvKmsKapiSemaphoreSurfaceCallback *newNvKmsCallback;
    NvU64 newWaitValue;
    unsigned long newTimeout;
    NvKmsKapiRegisterWaiterResult kapiRet;

    if (!newCallback) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to allocate new fence signal callback data");
        return;
    }

    do {
        /*
         * Process any completed or timed out fences. This returns the wait
         * value and timeout of the first remaining pending fence, or 0/0
         * if no pending fences remain. It will also tag the context as
         * waiting for the value returned.
         */
        __nv_drm_fence_context_process(&ctx->base,
                                       &newWaitValue,
                                       &newTimeout,
                                       NULL);

        /*
         * Keep the timeout timer armed to the head deadline whenever a fence is
         * pending, even when no NEW callback registration is needed (deduped) or
         * the list just drained. Decoupling the timer from the callback dedup
         * prevents a reordered stale mod_timer from dropping the backstop.
         */
        if (newTimeout != 0) {
            nv_drm_mod_timer(&ctx->base.timer, newTimeout);
        }

        if (newWaitValue == 0) {
            /* No new callback needed (list empty, or one is already registered). */
            nv_drm_free(newCallback);
            newCallback = NULL;
            return;
        }

        newCallback->wait_value = newWaitValue;

        /*
         * Attempt to register a callback for the remaining fences. Note this
         * code may be running concurrently in multiple places, attempting to
         * register a callback for the same value, a value greater than
         * newWaitValue if more fences have since completed, or a value less
         * than newWaitValue if new fences have been created tracking lower
         * values than the previously lowest pending one. Hence, even if this
         * registration succeeds, the callback may be discarded
         */
        kapiRet =
            nvKms->registerSemaphoreSurfaceCallback(nv_dev->pDevice,
                                                    ctx->pSemSurface,
                                                    __nv_drm_semsurf_ctx_callback,
                                                    newCallback,
                                                    ctx->base.fenceSemIndex,
                                                    newWaitValue,
                                                    0,
                                                    &newNvKmsCallback);
    } while (kapiRet == NVKMS_KAPI_REG_WAITER_ALREADY_SIGNALLED);

    /* Can't deref newCallback at this point unless kapiRet indicates failure */

    if (kapiRet != NVKMS_KAPI_REG_WAITER_SUCCESS) {
        /*
         * This is expected if another thread concurrently registered a callback
         * for the same value, which is fine. That thread's callback will do the
         * same work this thread's would have. Clean this one up and return.
         *
         * Another possibility is that an allocation or some other low-level
         * operation that can spuriously fail has caused this failure, or of
         * course a bug resulting in invalid usage of the
         * registerSemaphoreSurfaceCallback() API. There is no good way to
         * handle such failures, so the fence timeout will be relied upon to
         * guarantee forward progress in those cases.
         */
        nv_drm_free(newCallback);
        return;
    }

    /* The timeout timer was already (re)armed to the head deadline above. */

    if (!__nv_drm_semsurf_ctx_store_callback(ctx,
                                             newWaitValue,
                                             newNvKmsCallback,
                                             newCallback)) {
        /*
         * Another thread registered a callback for a different value before
         * this thread's callback could be stored in the context, or the
         * callback is already running. That's OK. One of the following is true:
         *
         * -A new fence with a lower value has been registered, and the callback
         *  associated with that fence is now active and associated with the
         *  context.
         *
         * -This fence has already completed, and a new callback associated with
         *  a higher value has been registered and associated with the context.
         *  This lower-value callback is no longer needed, as any fences
         *  associated with it must have been marked completed before
         *  registering the higher-value callback.
         *
         * -The callback started running and cleared ctx->current_wait_value
         *  before the callback could be stored in the context. Work to signal
         *  the fence is now pending.
         *
         * Hence, it is safe to request cancellation of the callback and free
         * the associated data if cancellation succeeds.
         */
        if (nvKms->unregisterSemaphoreSurfaceCallback(nv_dev->pDevice,
                                                      ctx->pSemSurface,
                                                      ctx->base.fenceSemIndex,
                                                      newWaitValue,
                                                      newNvKmsCallback)) {
            /* RM callback successfully canceled. Free local tracking data */
            nv_drm_free(newCallback);
        }
    }
}

static void
__nv_drm_semsurf_drain_pending_waits(struct nv_drm_semsurf_fence_ctx *ctx)
{
    unsigned long flags;

    /*
     * The pending waits are referenced by the fences they are waiting on, which
     * are guaranteed to complete in finite time. Keep the context alive until
     * they drain themselves.
     */
    spin_lock_irqsave(&ctx->base.lock, flags);
    while (!list_empty(&ctx->pending_waits)) {
        spin_unlock_irqrestore(&ctx->base.lock, flags);
        nv_drm_yield();
        spin_lock_irqsave(&ctx->base.lock, flags);
    }
    spin_unlock_irqrestore(&ctx->base.lock, flags);
}

static void __nv_drm_semsurf_fence_ctx_destroy(
    struct nv_drm_fence_context *nv_fence_context)
{
    struct nv_drm_device *nv_dev = nv_fence_context->nv_dev;
    struct nv_drm_semsurf_fence_ctx *ctx =
        to_semsurf_fence_ctx(nv_fence_context);
    struct NvKmsKapiSemaphoreSurfaceCallback *pendingNvKmsCallback;
    NvU64 pendingWaitValue;
    unsigned long flags;

    /*
     * Idle the worker + timer. The worker is shut down before the timer (inside
     * teardown_engine) to ensure the timer does not queue work that restarts
     * itself.
     */
    __nv_drm_fence_context_teardown_engine(nv_fence_context);

    /*
     * The semaphore surface could still be sending callbacks, so it is still
     * not safe to dereference the ctx->callback pointers. However,
     * unregistering a callback via its handle is safe, as that code in NVKMS
     * takes care to avoid dereferencing the handle until it knows the callback
     * has been canceled in RM. This unregistration must be done to ensure the
     * callback data is not leaked in NVKMS if it is still pending, as freeing
     * the semaphore surface only cleans up RM's callback data.
     */
    spin_lock_irqsave(&ctx->base.lock, flags);
    pendingNvKmsCallback = ctx->callback.nvKms;
    pendingWaitValue = ctx->callback.local ?
        ctx->callback.local->wait_value : 0;
    spin_unlock_irqrestore(&ctx->base.lock, flags);

    if (pendingNvKmsCallback) {
        WARN_ON(pendingWaitValue == 0);
        nvKms->unregisterSemaphoreSurfaceCallback(nv_dev->pDevice,
                                                  ctx->pSemSurface,
                                                  ctx->base.fenceSemIndex,
                                                  pendingWaitValue,
                                                  pendingNvKmsCallback);
    }

    nvKms->freeSemaphoreSurface(nv_dev->pDevice, ctx->pSemSurface);

    /*
     * Now that the semaphore surface, the timer, and the workthread are gone:
     *
     * -No more RM/NVKMS callbacks will arrive, nor are any in progress. Freeing
     *  the semaphore surface cancels all its callbacks associated with this
     *  instance of it, and idles any pending callbacks.
     *
     * -No more timer callbacks will arrive, nor are any in flight.
     *
     * -The workthread has been idled and is no longer running.
     *
     * Further, given the destructor is running, no other references to the
     * fence context exist, so this code can assume no concurrent access to the
     * fence context's data will happen from here on out.
     */

    if (ctx->callback.local) {
        nv_drm_free(ctx->callback.local);
        ctx->callback.local = NULL;
        ctx->callback.nvKms = NULL;
    }

    __nv_drm_fence_context_force_complete(&ctx->base, -ETIMEDOUT);
    __nv_drm_semsurf_drain_pending_waits(ctx);

    nv_drm_free(nv_fence_context);
}

static struct nv_drm_fence_context_ops
nv_drm_semsurf_fence_ctx_ops = {
    .destroy = __nv_drm_semsurf_fence_ctx_destroy,
    .read_seqno = __nv_drm_semsurf_ctx_read_seqno,
    .update = __nv_drm_semsurf_ctx_update,
};

static struct nv_drm_semsurf_fence_ctx*
__nv_drm_semsurf_fence_ctx_new(
    struct nv_drm_device *nv_dev,
    struct drm_nvidia_semsurf_fence_ctx_create_params *p
)
{
    struct nv_drm_semsurf_fence_ctx *ctx;
    struct NvKmsKapiSemaphoreSurface *pSemSurface;
    uint8_t *semMapping;
    uint8_t *maxSubmittedMapping;

    pSemSurface = nvKms->importSemaphoreSurface(nv_dev->pDevice,
                                                p->nvkms_params_ptr,
                                                p->nvkms_params_size,
                                                (void **)&semMapping,
                                                (void **)&maxSubmittedMapping);
    if (!pSemSurface) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to import semaphore surface");

        goto failed;
    }

    /*
     * Allocate a fence context object and initialize it.
     */

    if ((ctx = nv_drm_calloc(1, sizeof(*ctx))) == NULL) {
        goto failed_alloc_fence_context;
    }

    if (!__nv_drm_fence_context_init(&ctx->base,
                                     nv_dev,
                                     &nv_drm_semsurf_fence_ctx_ops,
                                     p->index)) {
        goto failed_ctx_init;
    }

    semMapping += (p->index * nv_dev->semsurf_stride);
    if (maxSubmittedMapping) {
        maxSubmittedMapping += (p->index * nv_dev->semsurf_stride) +
            nv_dev->semsurf_max_submitted_offset;
    }

    ctx->pSemSurface = pSemSurface;
    ctx->pSemMapping.pVoid = semMapping;
    ctx->pMaxSubmittedMapping = (volatile NvU64 *)maxSubmittedMapping;
    ctx->callback.local = NULL;
    ctx->callback.nvKms = NULL;

    INIT_LIST_HEAD(&ctx->pending_waits);

    return ctx;

failed_ctx_init:
    nv_drm_free(ctx);

failed_alloc_fence_context:
    nvKms->freeSemaphoreSurface(nv_dev->pDevice, pSemSurface);

failed:
    return NULL;

}

int nv_drm_semsurf_fence_ctx_create_ioctl(struct drm_device *dev,
                                          void *data,
                                          struct drm_file *filep)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    struct drm_nvidia_semsurf_fence_ctx_create_params *p = data;
    struct nv_drm_semsurf_fence_ctx *ctx;

    if (nv_dev->pDevice == NULL) {
        return -EOPNOTSUPP;
    }

    if (p->__pad != 0) {
        NV_DRM_DEV_LOG_ERR(nv_dev, "Padding fields must be zeroed");
        return -EINVAL;
    }

    ctx = __nv_drm_semsurf_fence_ctx_new(nv_dev, p);

    if (!ctx) {
        return -ENOMEM;
    }

    /*
     * On failure, handle-create has already dropped the only reference, which
     * frees the context via gem_free -> ops->destroy. Do NOT destroy again here
     * (that would be a double-free); just return the error.
     */
    return __nv_drm_fence_context_gem_init(dev, &ctx->base, &p->handle, filep);
}

static const char*
__nv_drm_semsurf_fence_op_get_timeline_name(struct dma_fence *fence)
{
    return "nvidia.semaphore_surface";
}

static const struct dma_fence_ops nv_drm_semsurf_fence_ops = {
    .get_driver_name = nv_drm_gem_fence_op_get_driver_name,
    .get_timeline_name = __nv_drm_semsurf_fence_op_get_timeline_name,
    .enable_signaling = __nv_drm_fence_op_enable_signaling,
    .release = __nv_drm_fence_op_release,
    .wait = dma_fence_default_wait,
#if defined(NV_DMA_FENCE_OPS_HAS_USE_64BIT_SEQNO)
    .use_64bit_seqno = true,
#endif
};

static struct dma_fence *__nv_drm_semsurf_fence_ctx_create_fence(
    struct nv_drm_device *nv_dev,
    struct nv_drm_semsurf_fence_ctx *ctx,
    NvU64 wait_value,
    NvU64 timeout_value_ms)
{
    struct nv_drm_fence *nv_fence;
    struct dma_fence *fence;
    int ret = 0;

    /* timeout_value_ms is clamped by __nv_drm_fence_context_add_pending(). */

    if ((nv_fence = nv_drm_calloc(1, sizeof(*nv_fence))) == NULL) {
        ret = -ENOMEM;
        goto out;
    }

    fence = &nv_fence->base;
    spin_lock_init(&nv_fence->lock);
#if !defined(NV_DMA_FENCE_OPS_HAS_USE_64BIT_SEQNO)
    nv_fence->wait_value = wait_value;
#endif

    /* Initializes the fence with one reference (for the caller) */
    dma_fence_init(fence, &nv_drm_semsurf_fence_ops,
                   &nv_fence->lock,
                   ctx->base.context, wait_value);

    __nv_drm_fence_context_add_pending(&ctx->base, nv_fence, timeout_value_ms);

out:
    /* Returned fence has one reference reserved for the caller. */
    return ret != 0 ? ERR_PTR(ret) : &nv_fence->base;
}

int nv_drm_semsurf_fence_create_ioctl(struct drm_device *dev,
                                      void *data,
                                      struct drm_file *filep)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    struct drm_nvidia_semsurf_fence_create_params *p = data;
    struct nv_drm_fence_context *nv_fence_context;
    struct dma_fence *fence;
    int ret = -EINVAL;
    int fd;

    if (nv_dev->pDevice == NULL) {
        ret = -EOPNOTSUPP;
        goto done;
    }

    if (p->__pad != 0) {
        NV_DRM_DEV_LOG_ERR(nv_dev, "Padding fields must be zeroed");
        goto done;
    }

    if ((nv_fence_context = __nv_drm_fence_context_lookup_typed(
                                nv_dev,
                                filep,
                                p->fence_context_handle,
                                &nv_drm_semsurf_fence_ctx_ops)) == NULL) {
        goto done;
    }

    fence = __nv_drm_semsurf_fence_ctx_create_fence(
        nv_dev,
        to_semsurf_fence_ctx(nv_fence_context),
        p->wait_value,
        p->timeout_value_ms);

    if (IS_ERR(fence)) {
        ret = PTR_ERR(fence);

        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to allocate fence: 0x%08x", p->fence_context_handle);

        goto fence_context_create_fence_failed;
    }

    if ((fd = nv_drm_create_sync_file(fence)) < 0) {
        ret = fd;

        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to create sync file from fence on ctx 0x%08x",
            p->fence_context_handle);

        goto fence_context_create_sync_failed;
    }

    p->fd = fd;
    ret = 0;

fence_context_create_sync_failed:
    /*
     * Release this function's reference to the fence.  If successful, the sync
     * FD will still hold a reference, and the pending list (if the fence hasn't
     * already been signaled) will also retain a reference.
     */
    dma_fence_put(fence);

fence_context_create_fence_failed:
    nv_drm_gem_object_unreference_unlocked(&nv_fence_context->base);

done:
    return ret;
}

static void
__nv_drm_semsurf_free_wait_data(struct nv_drm_sync_fd_wait_data *wait_data)
{
    struct nv_drm_semsurf_fence_ctx *ctx = wait_data->ctx;
    unsigned long flags;

    spin_lock_irqsave(&ctx->base.lock, flags);
    list_del(&wait_data->pending_node);
    spin_unlock_irqrestore(&ctx->base.lock, flags);

    nv_drm_free(wait_data);
}

static void
__nv_drm_semsurf_wait_fence_work_cb
(
    void *arg
)
{
    struct nv_drm_sync_fd_wait_data *wait_data = arg;
    struct nv_drm_semsurf_fence_ctx *ctx = wait_data->ctx;
    struct nv_drm_device *nv_dev = ctx->base.nv_dev;
    NvKmsKapiRegisterWaiterResult ret;

    /*
     * Note this command applies "newValue" immediately if the semaphore has
     * already reached "waitValue." It only returns NVKMS_KAPI_ALREADY_SIGNALLED
     * if a separate notification was requested as well.
     */
    ret = nvKms->registerSemaphoreSurfaceCallback(nv_dev->pDevice,
                                                  ctx->pSemSurface,
                                                  NULL,
                                                  NULL,
                                                  ctx->base.fenceSemIndex,
                                                  wait_data->pre_wait_value,
                                                  wait_data->post_wait_value,
                                                  NULL);

    if (ret != NVKMS_KAPI_REG_WAITER_SUCCESS) {
        NV_DRM_DEV_LOG_ERR(nv_dev,
                           "Failed to register auto-value-update on pre-wait value for sync FD semaphore surface");
    }

    __nv_drm_semsurf_free_wait_data(wait_data);
}

static void
__nv_drm_semsurf_wait_fence_cb
(
    struct dma_fence *fence,
    struct dma_fence_cb *cb
)
{
    struct nv_drm_sync_fd_wait_data *wait_data =
        container_of(cb, typeof(*wait_data), dma_fence_cb);
    struct nv_drm_semsurf_fence_ctx *ctx = wait_data->ctx;

    /*
     * Defer registering the wait with RM to a worker thread, since
     * this function may be called in interrupt context, which
     * could mean arriving here directly from RM's top/bottom half
     * handler when the fence being waited on came from an RM-managed GPU.
     */
    if (!nv_drm_workthread_add_work(&ctx->base.worker, &wait_data->work)) {
        /*
         * The context is shutting down. RM would likely just drop
         * the wait anyway as part of that, so do nothing. Either the
         * client is exiting uncleanly, or it is a bug in the client
         * in that it didn't consume its wait before destroying the
         * fence context used to instantiate it.
         */
        __nv_drm_semsurf_free_wait_data(wait_data);
    }

    /* Don't need to reference the fence anymore, just the fence context. */
    dma_fence_put(fence);
}

int nv_drm_semsurf_fence_wait_ioctl(struct drm_device *dev,
                                    void *data,
                                    struct drm_file *filep)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    struct drm_nvidia_semsurf_fence_wait_params *p = data;
    struct nv_drm_fence_context *nv_fence_context;
    struct nv_drm_semsurf_fence_ctx *ctx;
    struct nv_drm_sync_fd_wait_data *wait_data = NULL;
    struct dma_fence *fence;
    unsigned long flags;
    int ret = -EINVAL;

    if (nv_dev->pDevice == NULL) {
        return -EOPNOTSUPP;
    }

    if (p->pre_wait_value >= p->post_wait_value) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Non-monotonic wait values specified to fence wait: 0x%" NvU64_fmtu ", 0x%" NvU64_fmtu,
            p->pre_wait_value, p->post_wait_value);
        goto done;
    }

    if ((nv_fence_context = __nv_drm_fence_context_lookup_typed(
                                nv_dev,
                                filep,
                                p->fence_context_handle,
                                &nv_drm_semsurf_fence_ctx_ops)) == NULL) {
        goto done;
    }

    ctx = to_semsurf_fence_ctx(nv_fence_context);

    wait_data = nv_drm_calloc(1, sizeof(*wait_data));

    if (!wait_data) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to allocate callback data for sync FD wait: %d", p->fd);

        goto fence_context_sync_lookup_failed;
    }

    fence = nv_drm_sync_file_get_fence(p->fd);

    if (!fence) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Attempt to wait on invalid sync FD: %d", p->fd);

        goto fence_context_sync_lookup_failed;
    }

    wait_data->ctx = ctx;
    wait_data->pre_wait_value = p->pre_wait_value;
    wait_data->post_wait_value = p->post_wait_value;
    nv_drm_workthread_work_init(&wait_data->work,
                                __nv_drm_semsurf_wait_fence_work_cb,
                                wait_data);

    spin_lock_irqsave(&ctx->base.lock, flags);
    list_add(&wait_data->pending_node, &ctx->pending_waits);
    spin_unlock_irqrestore(&ctx->base.lock, flags);

    ret = dma_fence_add_callback(fence,
                                 &wait_data->dma_fence_cb,
                                 __nv_drm_semsurf_wait_fence_cb);

    if (ret) {
       if (ret == -ENOENT) {
           /* The fence is already signaled */
       } else {
           NV_DRM_LOG_ERR(
               "Failed to add dma_fence callback. Signaling early!");
           /* Proceed as if the fence wait succeeded */
       }

       /* Execute second half of wait immediately, avoiding the worker thread */
       dma_fence_put(fence);
        __nv_drm_semsurf_wait_fence_work_cb(wait_data);
    }

    ret = 0;

fence_context_sync_lookup_failed:
    if (ret && wait_data) {
        /*
         * Do not use __nv_drm_semsurf_free_wait_data() here, as the wait_data
         * has not been added to the pending list yet.
         */
        nv_drm_free(wait_data);
    }

    nv_drm_gem_object_unreference_unlocked(&nv_fence_context->base);

done:
    return 0;
}

int nv_drm_semsurf_fence_attach_ioctl(struct drm_device *dev,
                                      void *data,
                                      struct drm_file *filep)
{
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    struct drm_nvidia_semsurf_fence_attach_params *p = data;
    struct nv_drm_gem_object *nv_gem = NULL;
    struct nv_drm_fence_context *nv_fence_context = NULL;
    struct dma_fence *fence;
    int ret = -EINVAL;

    if (nv_dev->pDevice == NULL) {
        ret = -EOPNOTSUPP;
        goto done;
    }

    nv_gem = nv_drm_gem_object_lookup(filep, p->handle);

    if (!nv_gem) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to lookup gem object for fence attach: 0x%08x",
            p->handle);

        goto done;
    }

    nv_fence_context = __nv_drm_fence_context_lookup_typed(
        nv_dev,
        filep,
        p->fence_context_handle,
        &nv_drm_semsurf_fence_ctx_ops);

    if (!nv_fence_context) {
        goto done;
    }

    fence = __nv_drm_semsurf_fence_ctx_create_fence(
        nv_dev,
        to_semsurf_fence_ctx(nv_fence_context),
        p->wait_value,
        p->timeout_value_ms);

    if (IS_ERR(fence)) {
        ret = PTR_ERR(fence);

        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to allocate fence: 0x%08x", p->handle);

        goto done;
    }

    ret = __nv_drm_gem_attach_fence(nv_gem, fence, p->shared);

    dma_fence_put(fence);

done:
    if (nv_fence_context) {
        nv_drm_gem_object_unreference_unlocked(&nv_fence_context->base);
    }

    if (nv_gem) {
        nv_drm_gem_object_unreference_unlocked(nv_gem);
    }

    return ret;
}

int nv_drm_semsurf_export_to_syncobj_point_ioctl(struct drm_device *dev,
                                                  void *data,
                                                  struct drm_file *filep)
{
#if defined(NV_DRM_SYNCOBJ_FEATURES_PRESENT)
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    struct drm_nvidia_semsurf_export_to_syncobj_point_params *p = data;
    struct nv_drm_fence_context *nv_fence_context = NULL;
    struct dma_fence *fence = NULL;
    struct drm_syncobj *syncobj = NULL;
    struct dma_fence_chain *chain = NULL;
    int ret = -EINVAL;

    if (nv_dev->pDevice == NULL) {
        ret = -EOPNOTSUPP;
        goto done;
    }

    /*
     * Lookup the fence context. The typed lookup logs and drops the reference
     * itself on a type mismatch, so a NULL here needs no further cleanup.
     */
    nv_fence_context = __nv_drm_fence_context_lookup_typed(
        nv_dev,
        filep,
        p->fence_context_handle,
        &nv_drm_semsurf_fence_ctx_ops);

    if (nv_fence_context == NULL) {
        goto done;
    }

    /* Lookup the DRM syncobj */
    syncobj = drm_syncobj_find(filep, p->syncobj_handle);
    if (!syncobj) {
        ret = -ENOENT;
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to find syncobj: 0x%08x",
            p->syncobj_handle);
        goto done;
    }

#if defined(NV_DMA_FENCE_CHAIN_ALLOC_PRESENT)
    chain = dma_fence_chain_alloc();
#else
    chain = kzalloc(sizeof(struct dma_fence_chain), GFP_KERNEL);
#endif
    if (!chain) {
        ret = -ENOMEM;
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to allocate dmabuf chain");
        goto done;
    }

    /* Create a fence for the semsurf timeline point */
    fence = __nv_drm_semsurf_fence_ctx_create_fence(
        nv_dev,
        to_semsurf_fence_ctx(nv_fence_context),
        p->wait_value,
        NV_DRM_FENCE_MAX_TIMEOUT_MS);

    if (IS_ERR(fence)) {
        ret = PTR_ERR(fence);
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to create fence for wait_value: %llu",
            p->wait_value);
        goto done;
    }

    /* Replace the syncobj timeline point with our fence */
    drm_syncobj_add_point(syncobj, chain, fence, p->syncobj_point);

    ret = 0;

done:
    if (syncobj) {
        drm_syncobj_put(syncobj);
    }

    if (fence) {
        dma_fence_put(fence);
    }

    if (nv_fence_context) {
        nv_drm_gem_object_unreference_unlocked(&nv_fence_context->base);
    }

    return ret;
#else
    return -EOPNOTSUPP;
#endif /* NV_DRM_SYNCOBJ_FEATURES_PRESENT */
}

int nv_drm_syncobj_get_syncfd_ioctl(struct drm_device *dev,
                                    void *data,
                                    struct drm_file *filep)
{
#if defined(NV_DRM_SYNCOBJ_FEATURES_PRESENT)
    struct nv_drm_device *nv_dev = to_nv_device(dev);
    struct drm_nvidia_syncobj_get_syncfd_params *p = data;
    struct drm_syncobj *syncobj = NULL;
    struct dma_fence *fence = NULL;
    int ret = -EINVAL;
    int fd;

    if (nv_dev->pDevice == NULL) {
        ret = -EOPNOTSUPP;
        goto done;
    }

    /* Lookup the DRM syncobj */
    syncobj = drm_syncobj_find(filep, p->syncobj_handle);
    if (!syncobj) {
        ret = -ENOENT;
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to find syncobj: 0x%08x",
            p->syncobj_handle);
        goto done;
    }

    /* Find the fence at the specified timeline point */
    ret = drm_syncobj_find_fence(filep, p->syncobj_handle, p->syncobj_point,
                                  0 /* flags */, &fence);
    if (ret) {
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to find fence for syncobj 0x%08x at point %llu",
            p->syncobj_handle,
            p->syncobj_point);
        goto done;
    }

    /* Create a sync FD for the fence */
    fd = nv_drm_create_sync_file(fence);
    if (fd < 0) {
        ret = fd;
        NV_DRM_DEV_LOG_ERR(
            nv_dev,
            "Failed to create sync file from fence on syncobj 0x%08x",
            p->syncobj_handle);
        goto fence_found;
    }

    p->fd = fd;
    ret = 0;

fence_found:
    /* Release this function's reference to the fence */
    dma_fence_put(fence);

done:
    if (syncobj) {
        drm_syncobj_put(syncobj);
    }

    return ret;
#else
    return -EOPNOTSUPP;
#endif /* NV_DRM_SYNCOBJ_FEATURES_PRESENT */
}

#endif /* NV_DRM_AVAILABLE */
