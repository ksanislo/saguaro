#include "strata/program/read_progress.hpp"

#include <cstdio>
#include <vector>

// U5b (PLAN-u5b-split-progress): the one progress figure of a read whose cards report out of order (the layer
// split's cards are pieces apart and their host points arrive in no order), the figure never decreasing, one card
// printing exactly what its own host points report, and two reads keeping separate figures.  Expected values
// computed by hand from the rule: the sum over the cards of (positions that card has fully passed, times its
// share of the layers) plus the part of each card's current chunk that runs up to the next boundary another card
// reported - over all layers.
int main() {
    using strata::program::ReadProgress;
    int checks = 0, bad = 0;
    auto eq = [&](long long got, long long want, const char* what) {
        ++checks;
        if (got != want) {
            ++bad;
            std::printf("FAIL %s: got %lld, want %lld\n", what, got, want);
        }
    };

    // four cards of two layers each (8 in all), a read of [0, 1000) in chunks of 100
    {
        ReadProgress p;
        p.reset(0, 1000, 8);
        // a pipeline step: card 0 works chunk [300,400) at pass 1/2, card 1 [200,300) at 1/2, card 2 [100,200) at
        // 1/2, card 3 [0,100) at 1/2 - reported in scrambled order (the slow cards can report first)
        p.report(6, 0, 1, 2);       // card 3
        p.report(4, 100, 1, 2);     // card 2
        p.report(0, 300, 1, 2);     // card 0 (ahead of the others: reported before card 1 catches up)
        // (300*2) + (100*2 + 200*1) + (0*2 + 100*1) = 600 + 400 + 100, over 8 layers: the fastest card's own
        // chunk gets no part-credit (no other card has reported its end), card 2's runs to the boundary at 300
        eq(p.done(), 138, "three behind, one ahead: 1100/8 rounds up");
        p.report(2, 200, 1, 2);     // card 1: the boundary at 200 becomes known, card 2's part shrinks to 100
        // 600 + (200*2 + 100*1) + (100*2 + 100*1) + (100*1) = 600 + 500 + 300 + 100 = 1500: 1500/8 = 187.5
        eq(p.done(), 188, "four cards, one chunk between each: (600+500+300+100)/8 rounds up");
        // the fastest card passes a chunk: full credit for the tokens up to its new start, its new chunk's end
        // is again unknown: 400*2 + 500 + 300 + 100 = 1700 over 8: 212.5
        p.report(0, 400, 1, 2);
        eq(p.done(), 213, "the ahead card overtaken: 1700/8 rounds up");
    }

    {   // one card: exactly the position its host points report (what it printed before U5b), passes = all layers
        ReadProgress p;
        p.reset(0, 1000, 8);
        p.report(0, 0, 3, 8);      // the first chunk, mid pass: nothing done yet
        eq(p.done(), 0, "one card, first chunk: 0");
        p.report(0, 200, 5, 8);    // two chunks fully passed, working on the third
        eq(p.done(), 200, "one card: its own chunk start, mid-pass credit never ahead");
        p.report_full(300);        // the chunk ended (the PP point): the figure never goes back
        eq(p.done(), 300, "one card: a chunk end reports full credit");
        p.report(0, 300, 0, 8);    // the next chunk, before any pass of it: the figure stands
        eq(p.done(), 300, "one card: back to a chunk start, never decreasing");
    }

    {   // the beats jump backwards: monotone over a pipeline's scramble (4 x T4's shape, small)
        ReadProgress p;
        p.reset(0, 800, 8);
        const int64_t cards[4] = {0, 2, 4, 6};
        long long prev = 0;
        bool mono = true;
        // four chunks of 100, the cards three chunks apart, reports interleaved round-robin with the pieces
        // moving: each card's chunk start walks forward as it passes pieces, in no lockstep order
        const int64_t seq[] = {0, 0, 0, 0, 100, 0, 0, 100, 200, 100, 100, 200, 300, 200, 200, 300, 400, 300,
                               300, 400, 500, 400, 400, 500, 600, 500, 500, 600, 700, 600, 600, 700};
        int k = 0;
        for (int step = 0; step < 8; ++step)
            for (int c = 0; c < 4; ++c) {   // the slowest card's host point can run last at every second
                p.report(cards[c], seq[k % 32], (k % 3), 2);
                ++k;
                if (p.done() < prev) mono = false;
                prev = p.done();
            }
        ++checks;
        if (!mono) { ++bad; std::printf("FAIL out-of-order beats went backwards\n"); }
        ++checks;
        if (prev <= 500) { ++bad; std::printf("FAIL out-of-order beat stuck at %lld\n", prev); }
    }

    {   // the figure ends at `new` when the read ends: every card at the last chunk, then the last stage's end
        ReadProgress p;
        p.reset(0, 1000, 4);
        p.report(0, 900, 1, 2);
        p.report(2, 900, 1, 2);
        p.report_full(1000);
        eq(p.done(), 1000, "ends at new");
    }

    {   // two reads beside each other keep separate figures (a solo read and a slot's, two slots)
        ReadProgress a, b;
        a.reset(0, 1000, 4);
        b.reset(500, 2500, 4);   // a read that resumed at 500: new counts from there
        a.report(0, 400, 1, 2);
        a.report(2, 200, 0, 2);
        b.report(0, 2000, 1, 2);   // absolute positions: [500,2000) fully passed, [0,500) never counts
        eq(a.done(), 300, "read a keeps its own figure: (400*2 + 200*2 + 100*0) / 4");
        eq(b.done(), 750, "read b counts from where IT started: (2000-500) * its 2 layers / 4");
        b.report_full(2500);
        eq(b.done(), 2000, "read b ends at its new");
        eq(a.done(), 300, "read a untouched by read b");
    }

    {   // positions outside the read clamp; a reset lays the next read out clean
        ReadProgress p;
        p.reset(100, 600, 4);
        p.report(0, 5000, 2, 2);   // past the read's end: clamped; one card of two: half the layers count
        eq(p.done(), 250, "clamped: 500 * 2 / 4");
        p.report(0, -10, 0, 2);    // behind the read's start: the figure never goes back
        eq(p.done(), 250, "never decreasing, clamped");
        p.reset(100, 600, 4);
        eq(p.done(), 0, "the next read starts over");
        p.reset(0, 0, 4);          // a warm run (nothing to read): every report is nothing
        p.report(0, 100, 1, 2);
        eq(p.done(), 0, "an empty read stays 0");
    }

    std::printf("%s: %d checks, %d bad\n", bad ? "FAILED" : "ok", checks, bad);
    return bad ? 1 : 0;
}
