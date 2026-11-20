// src/program/vlen_pause_test.cpp - the VLEND / VRECLAIM pause state machine (header-only, CPU).
//
//   1. a pending, armed step takes the boundary on the tick it is armed, even when CUDA0 has a window in flight
//      (review 2: arming must not depend on the stage-0 launch gate finding CUDA0 free);
//   2. while the step is pending the pause holds new CUDA0 windows and runs the step only once CUDA0 is quiet;
//   3. the pause closes when the step comes due; it also closes (not-done) when the 2 s cap passes;
//   4. a blocked tick (a prompt read / concurrent serving) neither arms, holds, steps, nor closes;
//   5. the regression: a fake decode loop of piped windows - where a CUDA0 window is in flight at every launch gate -
//      reaches the pause with this state machine, but would never have reached it through the old gate-only placement.
#include "strata/program/vlen_pause.hpp"

#include <cstdio>

namespace vp = strata::program::vlen_pause;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-84s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
vp::In base() {
    vp::In i;
    i.may_step = true;
    return i;
}

// One tick of a pump replica: feed decide() and advance a small pause state.  A step run on a quiet CUDA0 lands the
// room (the give succeeds there); a held tick launches no new CUDA0 window, so the next tick is quiet again.
struct FakePump {
    bool pending = true;
    bool armed = false;
    bool open = false;
    double open_ms = 0.0;
    bool in_flight = false;
    int printed = 0;
    int closes_due = 0, closes_cap = 0;
    bool ever_held = false, ever_stepped = false;

    void tick(double dt_ms, bool blocked) {
        if (open) open_ms += dt_ms;
        vp::In s = base();
        s.pending = pending;
        s.armed = armed;
        s.is_open = open;
        s.open_ms = open_ms;
        s.stage_quiet = !in_flight;
        s.blocked = blocked;
        const vp::Out o = vp::decide(s);
        if (o.arm_now) { armed = false; open = true; open_ms = 0.0; }
        if (o.hold) ever_held = true;
        if (o.run_step) { ever_stepped = true; pending = false; }   // (the give lands on the quiet card)
        if (o.close) { printed++; open = false; open_ms = 0.0; (o.due ? closes_due : closes_cap)++; }
        // (a held tick opens no new CUDA0 window; otherwise, when CUDA0 is free, the next piped window starts)
        if (!o.hold && !in_flight) in_flight = true;
        else if (in_flight) in_flight = false;   // (the window leaves CUDA0 at the end of its tick)
    }
};
}  // namespace

int main() {
    std::printf("vlen_pause_test\n");

    {   // 1. armed + pending takes the boundary even with a window in flight (the review-2 fix)
        vp::In s = base();
        s.pending = true; s.armed = true; s.is_open = false; s.stage_quiet = false;
        const auto o = vp::decide(s);
        check(o.arm_now && o.hold && !o.run_step && !o.close,
              "armed + pending with CUDA0 busy: opens and holds now, steps only when quiet (never fires only at the gate)");
    }
    {   // 2. an open pause with a window in flight holds but does not step
        vp::In s = base();
        s.pending = true; s.is_open = true; s.open_ms = 5.0; s.stage_quiet = false;
        const auto o = vp::decide(s);
        check(o.hold && !o.run_step && !o.arm_now && !o.close, "open, CUDA0 busy, step pending: hold, no step yet");
    }
    {   // 2b. an open pause on a quiet CUDA0 runs the step (and still holds until it is due)
        vp::In s = base();
        s.pending = true; s.is_open = true; s.open_ms = 5.0; s.stage_quiet = true;
        const auto o = vp::decide(s);
        check(o.run_step && o.hold && !o.close, "open, CUDA0 quiet, step pending: run the step, still hold");
    }
    {   // 3. the step done -> close as due; the step still pending past the cap -> close as not-done
        vp::In s = base();
        s.pending = false; s.is_open = true; s.open_ms = 30.0; s.stage_quiet = true;
        check(vp::decide(s).close && vp::decide(s).due, "step done while open: closes (the step came due)");
        vp::In c = base();
        c.pending = true; c.is_open = true; c.open_ms = vp::cap_ms + 1.0; c.stage_quiet = true;
        const auto co = vp::decide(c);
        check(co.close && !co.due && !co.run_step && !co.hold, "past the 2 s cap with the step pending: closes (not done)");
    }
    {   // 4. blocked (a read owns CUDA0): nothing happens
        vp::In s = base();
        s.pending = true; s.armed = true; s.is_open = true; s.open_ms = 1.0; s.stage_quiet = true; s.blocked = true;
        const auto o = vp::decide(s);
        check(!o.arm_now && !o.hold && !o.run_step && !o.close, "blocked tick: no arm, no hold, no step, no close");
    }

    {   // 5. the regression: a fake decode loop of piped windows, a VLEND armed mid-decode, must reach the pause
        // The OLD placement armed the pause only inside the stage-0 launch gate, and that gate runs only when CUDA0 is
        // free.  Nothing there ever held CUDA0 back, so each free CUDA0 relaunched the next piped window at once: at
        // every gate visit a CUDA0 window was in flight (the plan's live finding), the gate's stage-0 branch never ran,
        // and the armed pause was never taken.
        const bool old_reached = [] {
            const bool armed = true, pending = true;
            bool in_flight = true;   // (a CUDA0 window is decoding when the VLEND is armed)
            for (int t = 0; t < 5000; ++t) {
                const bool gate_free = !in_flight;   // (the stage-0 launch branch only runs when CUDA0 is free)
                if (gate_free && armed && pending) return true;
                in_flight = true;   // (CUDA0 is never held back, so the next piped window relaunches at once)
            }
            return false;
        }();
        FakePump f;
        f.armed = true;
        f.in_flight = true;   // (a CUDA0 window is in flight when the VLEND is armed)
        int t = 0;
        while (f.printed == 0 && t < 1000) { f.tick(1.0, false); ++t; }
        check(!old_reached, "the old gate-only placement never reaches the pause in a piped decode loop");
        check(f.armed == false && f.printed == 1 && f.closes_due == 1 && f.ever_held && f.ever_stepped,
              "the per-tick state machine reaches the pause, holds CUDA0, steps once quiet, and closes (came due)");
    }

    std::printf("%s\n", g_fail == 0 ? "vlen_pause_test: ok" : "vlen_pause_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
