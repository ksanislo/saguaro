// include/strata/program/vlen_pause.hpp - when the pump pauses CUDA0's pipeline for a pending VLEND / VRECLAIM.
//
// A VLEND raises CUDA0's VRAM floor for the picture encoder and takes the room out of the expert cache; a VRECLAIM
// gives it back.  Either step is one give / take-back on a QUIET CUDA0 (its pages cannot be unmapped under a window).
// The command arms a pause (PLAN-vision-fast-start); the pause holds new CUDA0 windows back at a window boundary,
// runs one quiet settle (vlen_tick: the give / take-back toward the target), and opens the pipeline again - the step
// came due, or 2 s passed at the most.
//
// Review 2 (live, 11-20): under SAGUARO_PIPED_MTP a CUDA0 window is in flight through the stage-0 launch gate almost
// always, so a pause armed only when that gate happens to find CUDA0 free never fired.  The decision is therefore made
// on every pump tick - arming is not gated on CUDA0 being free at the gate, only running the step is - and the hold it
// returns is what keeps new CUDA0 windows from launching until the step is due.
//
// Pure (no GPU, no clock, no environment read): the pump passes in what it knows and applies what comes back.
#pragma once

namespace strata::program::vlen_pause {

/// The most the pipeline is held for one pending step (after it the step continues on the opportunistic settles).
constexpr double cap_ms = 2000.0;

/// What the pump knows at the start of one tick.
struct In {
    bool pending = false;     ///< a VLEND awaits its VLENT, or a VRECLAIM's take-back is still growing the cache
    bool armed = false;       ///< a VLEND / VRECLAIM command armed the pause and no boundary has taken it yet
    bool is_open = false;     ///< the pause is already open (held back since open_ms ago)
    double open_ms = 0.0;     ///< how long the pause has been open (0 when it is closed)
    bool stage_quiet = false; ///< CUDA0 has no window in flight, so the step can run on it here
    bool may_step = false;    ///< nothing else is served on CUDA0 this tick (not the concurrent doorbell thread)
    bool blocked = false;     ///< a prompt read / concurrent serving holds the pools: its own settles answer, no pause
};

/// What the pump should do this tick.
struct Out {
    bool arm_now = false;     ///< take the boundary now: open the pause (the pump stamps its clock and clears armed)
    bool run_step = false;    ///< run one quiet settle (vlen_tick) on CUDA0 this tick
    bool hold = false;        ///< hold back new CUDA0 windows this tick (do not launch on the stage-0 gate)
    bool close = false;       ///< close the pause this tick (the step came due, or the cap passed)
    bool due = false;         ///< closing because the step is done (false: the cap passed, a quiet settle keeps at it)
};

/// One tick of the pause's state machine.
inline Out decide(const In& s) {
    Out o;
    if (s.blocked) return o;   // (a read owns CUDA0: it answers the pending step on its own settles)
    if (!s.is_open && s.armed && s.pending) o.arm_now = true;   // (the boundary is taken whenever a step is pending)
    if (s.is_open || o.arm_now) {
        if (s.pending && s.open_ms < cap_ms) {
            o.hold = true;
            o.run_step = s.stage_quiet && s.may_step;   // (the step only runs once CUDA0 has drained)
        } else {
            o.close = true;
            o.due = !s.pending;
        }
    }
    return o;
}

}  // namespace strata::program::vlen_pause
