// src/program/vlen_preempt_test.cpp - a pending lend pre-empts a long read at its next piece boundary (CPU, no GPU).
//
//   1. a step pending at a piece boundary stops the read and runs the step (the give / take-back); a step pending
//      mid-piece does not stop it (CUDA0 is under a read window - the pages cannot be unmapped there);
//   2. the regression that matters: a fake read of many pieces with a VLEND arriving mid-read - with this rule the
//      VLENT lands within ONE piece of the arrival and the step runs once; with the old rule (a read owns CUDA0 until
//      it ends, its settles never reaching the step) the VLENT never lands during the read at all.
#include "strata/program/vlen_preempt.hpp"

#include <cstdio>

namespace vp = strata::program::vlen_preempt;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-84s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// A fake read of `pieces` pieces.  A VLEND arrives (the step becomes pending) after `arrive_after` pieces have been
// read; at each boundary decide() says whether to stop, run the step (which lands the VLENT and clears the step) and
// resume.  The read always goes on to read all its pieces.  Returns the piece index the VLENT landed on (-1: never).
int fake_read(int pieces, int arrive_after, bool old_rule, int& steps) {
    int vlent_at = -1;
    steps = 0;
    bool pending = false;
    for (int p = 0; p < pieces; ++p) {
        if (p == arrive_after) pending = true;   // (the VLEND arrived here)
        if (old_rule) continue;                  // (a read owns CUDA0 until it ends; its settles never answer)
        // the read is between pieces here (nothing of it in flight on CUDA0 at a piece boundary)
        const vp::Out o = vp::decide({pending, /*at_piece_boundary*/ true});
        if (o.run_step) {
            ++steps;
            if (vlent_at < 0) vlent_at = p;   // (the give lands the VLENT on the CUDA0 the read just released)
            pending = false;                  // (the step is done; the read goes on)
        }
    }
    return vlent_at;
}
}  // namespace

int main() {
    std::printf("vlen_preempt_test\n");

    {   // 1. the predicate: a pending step interrupts a read; nothing else does
        check(vp::interrupts(true), "a pending step interrupts a read holding CUDA0");
        check(!vp::interrupts(false), "no pending step: the read is left alone");
    }
    {   // 1b. decide stops + steps at a piece boundary; a mid-piece step waits for the boundary
        const auto a = vp::decide({true, true});
        check(a.stop && a.run_step, "step pending at a piece boundary: stop the read, run the step");
        const auto b = vp::decide({true, false});
        check(!b.stop && !b.run_step, "step pending mid-piece: do not stop the read (CUDA0 is under a window)");
        const auto c = vp::decide({false, true});
        check(!c.stop && !c.run_step, "no pending step at a boundary: nothing to do");
    }

    {   // 2. a VLEND arriving mid-read lands its VLENT within one piece, and the step runs once
        int steps = 0;
        const int pieces = 200, arrive = 7;
        const int vlent = fake_read(pieces, arrive, /*old_rule*/ false, steps);
        check(vlent == arrive, "the VLENT lands on the boundary right after the lend arrives (within one piece)");
        check(steps == 1, "the give / take-back step runs exactly once for the pending lend");
    }
    {   // 2b. the regression: the old rule (a read owns CUDA0, settles never answer) leaves the lend unanswered
        int steps = 0;
        const int vlent = fake_read(200, 7, /*old_rule*/ true, steps);
        check(vlent == -1 && steps == 0,
              "the old rule: the read never yields, so the VLENT never lands during it (the 46 s of live 11-20)");
    }
    {   // 3. a multi-settle step (a VRECLAIM's take-back over a few settles) steps once per boundary while due; the
        // read goes on at the boundary the step is done (no pending step there)
        int done_at = -1, steps = 0;
        for (int p = 0; p < 50; ++p) {
            const bool pending = p < 5;   // (the reclaim needs five settles, then the cache is whole)
            const vp::Out o = vp::decide({pending, true});
            if (o.run_step)
                ++steps;
            else if (!pending && done_at < 0)
                done_at = p;   // (the first boundary with nothing pending: the read goes on)
        }
        check(steps == 5, "a multi-settle step (VRECLAIM) runs once per boundary while it is due");
        check(done_at == 5, "the read goes on the boundary the step is done");
    }

    std::printf("%s\n", g_fail == 0 ? "vlen_preempt_test: ok" : "vlen_preempt_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
