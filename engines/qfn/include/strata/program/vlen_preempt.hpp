// include/strata/program/vlen_preempt.hpp - a pending VLEND / VRECLAIM pre-empts a long prompt read at its next piece.
//
// A VLEND raises CUDA0's VRAM floor and takes the room out of the expert cache; a VRECLAIM gives it back.  The step is
// one give / take-back on a QUIET CUDA0 (its pages cannot be unmapped under a window), so while a prompt read holds
// CUDA0 the step has to wait.  Live 11-20 15:57:49 said exactly that: "VLEND: the pause is refused this tick: a prompt
// read owns CUDA0 (its own settles answer the step)" - but a long read's settles do not reach the step (it gives no
// room mid-window).  Review 4 read the command off stdin the instant it arrived; review 4b makes it be ANSWERED: a
// pending step is treated like a waiting request for SAGUARO_READ_YIELD - the read holding CUDA0 stops at its next
// piece boundary, the give / take-back step runs there, and the read resumes from the next piece.
//
// Pure (no GPU, no clock, no environment read): the read loop passes in what it knows and applies what comes back.
// This is the one trigger the held-read path (cy_yield_line) and the solo read's stop-at-chunk share, so a pending
// step interrupts a read the same way at every boundary.
#pragma once

namespace strata::program::vlen_preempt {

/// Whether a step is pending and therefore should interrupt a read that holds CUDA0 at its next piece boundary.
inline bool interrupts(bool step_pending) {
    return step_pending;
}

/// What the read loop knows at one piece boundary.
struct In {
    bool step_pending = false;   ///< a VLEND awaits its VLENT, or a VRECLAIM's take-back is still growing the cache
    bool at_piece_boundary = false;  ///< the read has finished a piece (nothing of it in flight on CUDA0 here)
};

/// What the read loop should do at this boundary.
struct Out {
    bool stop = false;           ///< stop the read here so the step can run on the CUDA0 it just released
    bool run_step = false;       ///< run one quiet settle (vlen_tick: the give / take-back) now, then let the read go on
};

/// One boundary of the read's loop: stop the read (and step) only where it is safe - at a piece boundary, with a step
/// pending - so the CUDA0 the step unmaps is not under a read window.
inline Out decide(const In& s) {
    Out o;
    if (interrupts(s.step_pending) && s.at_piece_boundary) {
        o.stop = true;
        o.run_step = true;
    }
    return o;
}

}  // namespace strata::program::vlen_preempt
