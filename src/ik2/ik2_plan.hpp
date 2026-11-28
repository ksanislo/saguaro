// src/ik2/ik2_plan.hpp - the load-time sizing decision of saguaro-ik2 as pure functions (P125e a1, the elastic start
// step).  The engine's plan (src/ik2/ik2_main.cu, "experts ... dense ... K / V cache ... free:") used to count the
// whole --max-context of the global layers' K / V before a conversation had used a single position: a card holding
// every expert at 32768 moved to the offload path with NO expert in VRAM at a 262144 ceiling it did not use
// (ref/REPORT-p125b-ik2-vram.md, Ken's rule: the context ceiling must cost ~0 until used).  These helpers split the
// ceiling from the committed start step; tools/ik2_fit.py's plan() mirrors them for the setup and the Models page.
// tests/ik2_plan_test.cpp (no GPU) holds them to their arithmetic.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace ik2plan {

// SAGUARO_IK2_KV_START: the positions of the global (full-attention) layers' K / V the engine commits to VRAM at the
// load - a small start step, as the elastic QFN engine lays out its context; 0: the whole ceiling (today's layout).
constexpr int kKvStartDefault = 32768;

// The committed positions of a full-attention visit at the load: never more than the ceiling, and a kv_start of 0
// (or one past it) means the ceiling itself.
inline int kv_committed(int ctx, int kv_start) {
    return kv_start <= 0 || kv_start >= ctx ? ctx : kv_start;
}

// The engine's fits check: every expert beside the dense weights, the K / V cache of `kv_b` bytes, and the work
// reserve - in the free VRAM it measured - and the experts within the cache's own cap (--cache-gb).
inline bool experts_fit(size_t exp_b, size_t dense_b, size_t kv_b, size_t reserve, size_t fr0, double cache_gb) {
    return exp_b + dense_b + kv_b + reserve <= fr0 && exp_b <= static_cast<size_t>(cache_gb * 1073741824.0);
}

// The elastic start step: the ceiling's cache would crowd every expert out of VRAM (fits_all false) but the
// committed one leaves them room (fits_committed), so the experts stay and the global layers' K / V is laid out for
// the start step only.  The positions above the commit stay unbacked until the conversation needs them - the growth
// itself is step 2 (VMM-backed K / V taking VRAM back from the coldest experts), so until it lands the engine
// announces the committed context as the conversation's ceiling (it never writes past what it laid out).
// Not taken where KV streaming runs (its pinned host rows already hold the whole ceiling), not on q4 rows (they
// have their own VRAM / host split), not on two GPUs, not with --tp, --parallel or the expert-parallel modes.
inline bool elastic_start(bool fits_all, bool fits_committed, bool streaming_taken, int ctx, int kv_start,
                          bool two, bool q4, bool parallel, bool tp) {
    return !fits_all && fits_committed && !streaming_taken && kv_committed(ctx, kv_start) < ctx &&
           !two && !q4 && !parallel && !tp;
}

// ---- P125o (a2 / a3): the committed K / V in chunks of positions.  The elastic global K / V is one address range
// over the whole ceiling (src/ik2/ik2_vmm.hpp) with physical chunks behind the first committed positions; a request
// commits the chunks covering its positions before its first pass (host-side, a chunk at a time - a3), and the
// kernels keep addressing rows by absolute position, so the CUDA graphs over the stable base pointer stay valid.
// The arithmetic is here; the driver calls live beside it.

// positions a chunk covers before the granule rounds its bytes up (ref/REPORT-p125b-ik2-vram.md (a2): 4096)
constexpr int kKvChunkPos = 4096;

inline size_t align_up(size_t b, size_t g) { return g > 1 ? (b + g - 1) / g * g : b; }

// the positions one committed chunk covers: the wanted rows' bytes rounded up to the granule (a granule of 0:
// no VMM, the exact rows).  The rounding only ever makes a chunk bigger, so chunks stay whole and contiguous.
inline int kv_rows_per_chunk(int want, size_t row_bytes, size_t gran) {
    if (row_bytes == 0 || want <= 0) return 0;
    const size_t r = align_up((size_t) want * row_bytes, gran) / row_bytes;
    return (int) (r == 0 ? 1 : r);
}

// the chunk holding a position, and the chunks covering n positions.  The commit order is the position order:
// a request walks committed .. chunks_for(its last position) - 1, ascending, so a position's rows are never
// touched before the chunk holding them has physical pages.
inline int kv_chunk_of(int pos, int rows_per_chunk) {
    return pos <= 0 || rows_per_chunk <= 0 ? 0 : pos / rows_per_chunk;
}
inline int kv_chunks_for(int n_positions, int rows_per_chunk) {
    if (rows_per_chunk <= 0) return 0;
    return n_positions <= 0 ? 0 : (n_positions + rows_per_chunk - 1) / rows_per_chunk;
}

// ---- P125u (a4): the give-back.  A conversation that grew the commit past the start step and is over - session
// end, reset(), a snapshot rewind - leaves chunks mapped under rows nothing reads any more: attention reads
// [0, pos), a kept conversation holds its rows in host RAM and a snapshot its rewound state there (the report's
// point, and full-layer positions rewrite cleanly).  Everything above the live high-water goes back: the engine
// waits for the GPU, unmaps the chunks and marks them free, where a later grow - or the experts' demotion step
// (b4) - can take the VRAM again.  The unmap (cuMemUnmap / hipMemUnmap, src/ik2/ik2_vmm.hpp) happens in the
// engine, host-side between requests - never on a token's path.

// The live high-water across the slots: the largest position a live slot can still read (the conversation in the
// cache, its snapshots, and - where --parallel is taken, never on an elastic start - one conversation per slot).
// Positions at or below it may be read; an empty set or negative entries (no snapshot there) say nothing.
inline int kv_high_water(const std::vector<int>& live_positions) {
    int hw = 0;
    for (const int p : live_positions)
        if (p > hw) hw = p;
    return hw;
}

// The free-chunk list.  Every buffer of the elastic K / V commits in lockstep (one counter covers the K and the
// V of every global visit - the row is the same across them), so the list is the ceiling's chunks split into the
// committed prefix [0, used) and the free rest [used, n_total): commit() takes the next free chunks for a
// conversation growing, give_back_to() returns the ones above a position.  A returned chunk's physical VRAM goes
// back to the driver at the unmap; the list is the bookkeeping of which chunks of the reserved address range
// stand ready to hold pages again - re-committing one maps fresh pages the engine memsets (kv_grow).
struct KvFreeChunks {
    int n_total = 0;   // chunks in the ceiling's reserved range
    int used = 0;      // chunks holding VRAM now (the committed prefix)

    int free() const { return n_total - used; }
    // take chunks [used, n) from the free list for a grow; false when the ceiling's range does not reach that
    // far (nothing taken); a n at or below what is committed takes nothing and is satisfied.
    bool commit(int n) {
        if (n > n_total) return false;
        if (n > used) used = n;
        return true;
    }
    // return the chunks above position-count n to the free list: how many went back (0 when nothing above n
    // held VRAM; a negative n is 0: everything goes).  Above the committed prefix nothing needs returning.
    int give_back_to(int n) {
        const int keep = n < 0 ? 0 : std::min(n, used);
        const int gave = used - keep;
        used = keep;
        return gave;
    }
};

}  // namespace ik2plan
