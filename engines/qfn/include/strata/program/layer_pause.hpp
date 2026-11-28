// include/strata/program/layer_pause.hpp - P125j: where a prompt read that a STOP caught INSIDE a chunk
// pauses, where it goes on, and what saving it costs.
//
// A STOP reaches a running prompt read only between chunks today (Prefill::should_stop, checked at
// prefill.cpp's chunk-loop head), so a normal request that arrives mid-chunk waits the chunk out (up to
// ~10 s of an 8-16K chunk on the RX 9070 XT box).  Between two layers of a chunk the chunk's whole
// cross-layer state is its residual rows (m.R: T rows of hc*n_embd FP32 floats) - every layer done has
// already written this chunk's K/V cells, GDN/conv state and indexer tails INTO THE SESSION, and the
// layers not run hold nothing of these tokens - so pausing one layer deep means saving the rows and
// going on there: the K/V of the layers below the pause for this chunk is already written.
//
// Pure (no GPU, no clock, no environment read): the prompt path passes in what it knows and applies what
// comes back; layer_pause_test.cpp walks the scenarios by hand.
#pragma once

#include <cstdint>
#include <limits>

namespace strata::program::layer_pause {

/// Where a pause goes on.  All -1 / 0 tokens: no pause (the STOP stays a plain between-chunk cancel).
struct Point {
    int64_t chunk = -1;    ///< the paused chunk's first token, counted from the START OF THE RUN THAT
                           ///< paused - a log label; run-relative figures do not carry across runs (a
                           ///< continuation starts inside the old run's chunk grid), so nothing matches on it
    int64_t layer = -1;    ///< the first layer NOT yet run for this chunk
    int64_t tokens = 0;    ///< the chunk's token count (the saved rows' row count)
};

/// The pause taken after the layer pass `layer_done` of the chunk starting at run-relative token `c0`,
/// over a stage's layers [LB, LE), a run of `n` tokens in chunks of `C`.  No pause after the stage's
/// LAST layer (the chunk ends there anyway - the plain stop's job), none before its first, none outside
/// a chunk.  Pure.
inline Point pause_point(int64_t c0, int64_t layer_done, int64_t LB, int64_t LE, int64_t n, int64_t C) {
    if (c0 < 0 || n <= 0 || C <= 0 || c0 >= n) return {};
    if (LE - LB < 2) return {};                 // a one-layer stage has no "between layers"
    if (layer_done < LB) return {};             // nothing of the chunk is done
    if (layer_done + 1 >= LE) return {};        // the last layer: let the chunk end as before
    Point p;
    p.chunk = c0;
    p.layer = layer_done + 1;
    const int64_t left = n - c0;
    p.tokens = left < C ? left : C;
    return p;
}

/// Whether the run about to be made (`n` tokens in chunks of the same `C`) can go ON with this pause
/// rather than read the chunk again: the chunk its rows were saved for must be WHOLE in this run -
/// `c0` is where that chunk starts IN THIS RUN (the caller has already matched the pause's ABSOLUTE
/// first token against its own `pos0 + c0`, and compares the chunk's tokens themselves), and the saved
/// row count must be what this run reads there, `min(C, n - c0)`.  A `p` from another run's grid is
/// never wrong here, only unapplied: the chunk is then read again from its first layer, which rewrites
/// the session's cells with the same values (the PLE history advance over the same tokens is
/// idempotent).  `p.chunk` is the run-relative label of the run that PAUSED and takes no part - a
/// continuation starts inside that grid (its own c0 is 0 at the paused chunk).  Pure.
inline bool resumes(const Point& p, int64_t c0, int64_t n, int64_t C) {
    if (p.tokens <= 0 || c0 < 0 || n <= 0 || C <= 0) return false;
    const int64_t left = n - c0;
    if (left <= 0) return false;
    const int64_t want = left < C ? left : C;
    return p.tokens == want;
}

/// The size of a paused chunk's saved rows: `tokens` rows of `row_floats` FP32 floats (D = hc x n_embd:
/// 10240 = 40 KiB a token, so an 8192-token chunk is 320 MiB - the figure the park's budget and the
/// "X MiB of chunk rows kept" line show).  Saturates instead of wrapping.  Pure.
inline uint64_t state_bytes(int64_t tokens, int64_t row_floats) {
    if (tokens <= 0 || row_floats <= 0) return 0;
    if (row_floats > (int64_t) (std::numeric_limits<uint64_t>::max() / 4))
        return std::numeric_limits<uint64_t>::max();   // the row's own byte count would wrap (or zero)
    const uint64_t per_row = (uint64_t) row_floats * 4;
    if ((uint64_t) tokens > std::numeric_limits<uint64_t>::max() / per_row)
        return std::numeric_limits<uint64_t>::max();
    return (uint64_t) tokens * per_row;
}

}  // namespace strata::program::layer_pause
