#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/split_copy.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core::conversation_detail {

/// The private copy streams and ordering events (below) belong to a thread, but a thread can be short-lived: the layer
/// split's later stages run each prompt chunk on a fresh std::async thread, and a checkpoint part is saved on it.  The
/// handles used to be created per thread and never destroyed, so every such thread left one stream and one event per
/// device behind - a few MiB of VRAM per long cold prompt on the later cards, until a stream could not be created any
/// more and the copy failed with "out of memory" (#3350, #3470).  Now a thread hands its handles back to a process-wide
/// pool when it ends and the next thread takes them: the number of handles is the number of threads copying at once.
/// The pool is leaked on purpose (a thread may end after the statics are gone).
struct CopyHandlePool {
    std::mutex mu;
    std::vector<std::vector<cudaStream_t>> streams;   // free ones, by device
    std::vector<std::vector<cudaEvent_t>> events;
    long streams_created = 0, events_created = 0;
};
inline CopyHandlePool& copy_handle_pool() {
    static CopyHandlePool* p = new CopyHandlePool;
    return *p;
}
/// How many streams / events the process has created for these copies (tests: it stays at the peak number of threads
/// copying at once, not at the number of threads that ever copied).
inline long copy_streams_created() {
    CopyHandlePool& p = copy_handle_pool();
    std::lock_guard<std::mutex> lk(p.mu);
    return p.streams_created;
}
inline long copy_events_created() {
    CopyHandlePool& p = copy_handle_pool();
    std::lock_guard<std::mutex> lk(p.mu);
    return p.events_created;
}
struct ThreadCopyHandles {
    std::vector<cudaStream_t> streams;   // by device, nullptr = none yet
    std::vector<cudaEvent_t> events;
    ThreadCopyHandles() = default;
    ThreadCopyHandles(const ThreadCopyHandles&) = delete;
    ThreadCopyHandles& operator=(const ThreadCopyHandles&) = delete;
    ~ThreadCopyHandles() {
        CopyHandlePool& p = copy_handle_pool();
        std::lock_guard<std::mutex> lk(p.mu);
        if (p.streams.size() < streams.size()) p.streams.resize(streams.size());
        if (p.events.size() < events.size()) p.events.resize(events.size());
        for (size_t d = 0; d < streams.size(); ++d)
            if (streams[d]) p.streams[d].push_back(streams[d]);
        for (size_t d = 0; d < events.size(); ++d)
            if (events[d]) p.events[d].push_back(events[d]);
    }
};
inline ThreadCopyHandles& thread_copy_handles() {
    thread_local ThreadCopyHandles t;
    return t;
}

/// Conversation snapshot / checkpoint copies go through a private non-blocking stream of the current device
/// instead of the legacy (null) stream.  A plain cudaMemcpy is ordered against every blocking stream of the
/// device, and while another thread captures a graph on one (a drafter or a stage worker), HIP refuses it with
/// "operation would make the legacy stream depend on a capturing blocking stream".  The caller's contract is
/// unchanged: the device was synchronized before the copy (the copy itself is complete on return).
/// private_stream: this thread's stream on the current device: one a finished thread handed back, else a new one on
/// first use.
inline cudaError_t private_stream(cudaStream_t& s) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    std::vector<cudaStream_t>& streams = thread_copy_handles().streams;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= streams.size()) streams.resize((size_t) dev + 1, nullptr);
    if (!streams[(size_t) dev]) {
        CopyHandlePool& p = copy_handle_pool();
        std::lock_guard<std::mutex> lk(p.mu);
        if ((size_t) dev < p.streams.size() && !p.streams[(size_t) dev].empty()) {
            streams[(size_t) dev] = p.streams[(size_t) dev].back();
            p.streams[(size_t) dev].pop_back();
        } else {
            e = cudaStreamCreateWithFlags(&streams[(size_t) dev], cudaStreamNonBlocking);
            if (e != cudaSuccess) { streams[(size_t) dev] = nullptr; return e; }
            ++p.streams_created;
        }
    }
    s = streams[(size_t) dev];
    return cudaSuccess;
}

/// #2409: true when a copy on `dev` from this thread needs no new stream: the thread has one, or a thread that ended left
/// one in the pool. false: the next copy creates one (a small device allocation, which can fail on a full card).
inline bool copy_stream_ready(int dev) {
    if (dev < 0) dev = 0;
    const std::vector<cudaStream_t>& streams = thread_copy_handles().streams;
    if ((size_t) dev < streams.size() && streams[(size_t) dev]) return true;
    CopyHandlePool& p = copy_handle_pool();
    std::lock_guard<std::mutex> lk(p.mu);
    return (size_t) dev < p.streams.size() && !p.streams[(size_t) dev].empty();
}

/// SAGUARO_TEST_COPY_FAIL_AT=N (tests only, #3350): the N-th snapshot / checkpoint copy of the process fails with
/// cudaErrorMemoryAllocation, as when a card has no memory left for the copy.  Unset: nothing.
inline long test_copy_fail_at() {
    static const long v = [] { const char* e = std::getenv("SAGUARO_TEST_COPY_FAIL_AT"); return e ? std::atol(e) : 0L; }();
    return v;
}

/// `queued`: the copy is only queued on the private stream, and the caller synchronizes the device (or orders a
/// later launch after it with order_after_queued) before anything reads the target or writes the source.
inline cudaError_t copy_nonblocking(void* dst, const void* src, size_t bytes, bool queued = false) {
    if (const long at = test_copy_fail_at(); at > 0) {
        static std::atomic<long> copies{0};
        if (++copies == at) return cudaErrorMemoryAllocation;
    }
    cudaStream_t s = nullptr;
    cudaError_t e = private_stream(s);
    if (e != cudaSuccess) return e;
    e = strata::kernels::memcpy_async_split(dst, src, bytes, cudaMemcpyDefault, s);   // (SAGUARO_KV_HOST_ELASTIC)
    if (e != cudaSuccess || queued) return e;
    e = cudaStreamSynchronize(s);
    core::progress_beat();   // #2049: a parked conversation is many of these; each one that lands is progress
    return e;
}

/// The copies this thread queued on its private stream are complete: one wait instead of one per copy
/// (a checkpoint's running state is ~200 of them on a 62-layer model - each per-copy stream sync is a driver
/// round-trip the park paid without any clock covering it, PLAN-p124a).
inline cudaError_t copy_stream_done() {
    cudaStream_t s = nullptr;
    const cudaError_t e = private_stream(s);
    if (e != cudaSuccess) return e;
    const cudaError_t d = cudaStreamSynchronize(s);
    core::progress_beat();
    return d;
}

/// #2049 #2065 #1971: a host-to-host copy that tells the watchdog it is moving.  Parked state (the running
/// checkpoints, the K/V segments) sits in ordinary pageable memory for the whole session; when the OS has written
/// it to the pagefile, copying it back runs at the drive's page-in speed (measured in the reports: ~190 hard faults
/// a second, under 1 MB/s) and one memcpy of it stays silent past the 60 s limit, so the watchdog ended an engine
/// that was only slow.  One beat per 1 MiB slice keeps a copy that moves alive and still lets a real stall show.
/// SAGUARO_TEST_SLOW_COPY_US=N (tests only): every 1 MiB slice of a checkpoint copy takes N microseconds more, as when
/// the OS has paged the checkpoints out and each slice is read back from the drive.  Unset: no delay.
inline long test_slow_copy_us() {
    static const long v = [] { const char* e = std::getenv("SAGUARO_TEST_SLOW_COPY_US"); return e ? std::atol(e) : 0L; }();
    return v;
}

/// SAGUARO_PARK_FAST (on by default since the read-stall work of 11-19; =0 turns it back off): a parked K/V buffer
/// grows without the zero fill (the copy writes every new byte and is the pages' first touch, on the copy threads),
/// the threaded host copy starts at 4 MiB instead of 64 and its caller wakes when the last job lands instead of on
/// the next 20 ms tick, and a park's prompt checkpoints are copied on several threads.  The same bytes either way -
/// the single-core parks measured at 3-5 s for 5 GB (one half-busy thread) are the zero fill plus one memcpy; this
/// is the multi-threaded, fault-once copy the same evidence asked for.
inline bool park_fast() {
    static const bool on = [] {
        const char* e = std::getenv("SAGUARO_PARK_FAST");
        const bool v = e == nullptr || std::atoi(e) != 0;
        std::fprintf(stderr, "saguaro: conversation park: SAGUARO_PARK_FAST %s (%s)\n", v ? "on" : "off",
                     v ? "no zero fill, threaded copies from 4 MiB woken on completion, checkpoints copied in parallel"
                       : "the pre-11-19 path: zero-filled growth, 64 MiB threaded threshold");
        return v;
    }();
    return on;
}

/// The park/restore split (output only, #read-stall): how the snapshot copies spent their time -
/// alloc (buffer growth), copy (the threaded or single host transfers), sync (the device syncs around them) and
/// checkpoints - with the bytes moved and how many copies ran threaded vs as one runtime memcpy each.  The caller
/// (the serve loop's park / restore log) takes and prints it; every snapshot function adds into the current
/// thread's record.
struct SnapshotSplit {
    double alloc_ms = 0.0, copy_ms = 0.0, sync_ms = 0.0, ckpt_ms = 0.0;
    size_t copy_bytes = 0;
    long threaded = 0, plain = 0;
};
inline SnapshotSplit& snapshot_split() {
    thread_local SnapshotSplit s;
    return s;
}
inline SnapshotSplit take_snapshot_split() {
    SnapshotSplit s = snapshot_split();
    snapshot_split() = SnapshotSplit{};
    return s;
}
inline std::string snapshot_split_note(const SnapshotSplit& s) {
    char b[256];
    std::snprintf(b, sizeof b, "split: alloc %.0f ms, copy %.0f ms (%.2f GB, %ld threaded / %ld single memcpy), "
                               "sync %.0f ms, checkpoints %.0f ms",
                  s.alloc_ms, s.copy_ms, (double) s.copy_bytes / 1073741824.0, s.threaded, s.plain, s.sync_ms, s.ckpt_ms);
    return b;
}
/// ms between two clock points (the split's timers).
template<class T>
inline double split_ms(T a, T b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

template<class T>
inline std::vector<T> copy_beating(const std::vector<T>& src) {
    std::vector<T> out;
    out.reserve(src.size());
    const size_t step = std::max<size_t>(1, ((size_t) 1 << 20) / sizeof(T));
    for (size_t at = 0; at < src.size(); at += step) {
        out.insert(out.end(), src.begin() + (std::ptrdiff_t) at,
                   src.begin() + (std::ptrdiff_t) std::min(src.size(), at + step));
        if (const long us = test_slow_copy_us(); us > 0) std::this_thread::sleep_for(std::chrono::microseconds(us));
        core::progress_beat();
    }
    return out;
}

inline ConversationCheckpoint copy_checkpoint_beating(const ConversationCheckpoint& c) {
    ConversationCheckpoint o;
    o.ids = copy_beating(c.ids);
    o.imgs = copy_beating(c.imgs);
    o.gdn = copy_beating(c.gdn);
    o.ple = copy_beating(c.ple);
    o.tails = copy_beating(c.tails);
    o.dead = copy_beating(c.dead);
    o.block_pos = copy_beating(c.block_pos);
    o.rope_level = c.rope_level;   // the level of the K/V it continues: a parked image must not read as plain
    o.used = c.used;
    o.pinned = c.pinned;
    o.msg_tail = c.msg_tail;   // (a batch slot's message checkpoint stays the chain's tail kind after a park)
    o.stage_parts.reserve(c.stage_parts.size());
    for (const auto& part : c.stage_parts) o.stage_parts.push_back(copy_checkpoint_beating(part));
    return o;
}

inline std::vector<ConversationCheckpoint> copy_checkpoints_beating(const std::vector<ConversationCheckpoint>& cs) {
    std::vector<ConversationCheckpoint> out;
    if (park_fast() && cs.size() >= 2) {   // each checkpoint whole on one of up to 8 threads, into its own slot
        out.resize(cs.size());
        std::atomic<size_t> next{0};
        std::mutex mu;
        std::exception_ptr failed;   // (a host OOM in a thread is the park's bad_alloc, as on one thread)
        std::vector<std::thread> th;
        for (size_t t = 0; t < std::min<size_t>(8, cs.size()); ++t)
            th.emplace_back([&] {
                try {
                    for (size_t i; (i = next.fetch_add(1)) < cs.size();) out[i] = copy_checkpoint_beating(cs[i]);
                } catch (...) {
                    next.store(cs.size());
                    std::lock_guard<std::mutex> g(mu);
                    if (!failed) failed = std::current_exception();
                }
            });
        for (auto& x : th) x.join();
        if (failed) std::rethrow_exception(failed);
        return out;
    }
    out.reserve(cs.size());
    for (const auto& c : cs) out.push_back(copy_checkpoint_beating(c));
    return out;
}

/// Work queued on the legacy stream after this call starts after every copy this thread queued on the current
/// device's private stream (an event the legacy stream waits for; no host wait).
inline cudaError_t order_after_queued() {
    cudaStream_t s = nullptr;
    cudaError_t e = private_stream(s);
    if (e != cudaSuccess) return e;
    int dev = 0;
    e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    std::vector<cudaEvent_t>& events = thread_copy_handles().events;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= events.size()) events.resize((size_t) dev + 1, nullptr);
    if (!events[(size_t) dev]) {
        CopyHandlePool& p = copy_handle_pool();
        std::lock_guard<std::mutex> lk(p.mu);
        if ((size_t) dev < p.events.size() && !p.events[(size_t) dev].empty()) {
            events[(size_t) dev] = p.events[(size_t) dev].back();
            p.events[(size_t) dev].pop_back();
        } else {
            e = cudaEventCreateWithFlags(&events[(size_t) dev], cudaEventDisableTiming);
            if (e != cudaSuccess) { events[(size_t) dev] = nullptr; return e; }
            ++p.events_created;
        }
    }
    e = cudaEventRecord(events[(size_t) dev], s);
    if (e != cudaSuccess) return e;
    return cudaStreamWaitEvent(nullptr, events[(size_t) dev], 0);
}

} // namespace strata::core::conversation_detail
