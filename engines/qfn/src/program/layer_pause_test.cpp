#include "strata/program/layer_pause.hpp"

#include <cstdio>
#include <cstdint>

// P125j (PLAN-p125j-layer-pause): the three pure figures of a prompt read whose STOP lands BETWEEN the
// layers of a chunk - where it pauses (pause_point), which later run goes on with it (resumes), and what
// its saved rows cost (state_bytes).  Expected values computed by hand from the rules in the header: no
// pause after the stage's last layer or before its first, the last chunk keeps its short token count, a
// one-layer stage has no between-layers at all, a resume is the whole-chunk check at the run's own c0
// (the paused run's chunk label takes no part - a continuation starts inside that grid), and the byte
// figure saturates instead of wrapping.
int main() {
    using strata::program::layer_pause::Point;
    int checks = 0, bad = 0;
    auto eq = [&](long long got, long long want, const char* what) {
        ++checks;
        if (got != want) {
            ++bad;
            std::printf("FAIL %s: got %lld, want %lld\n", what, got, want);
        }
    };
    auto eq_point = [&](const Point& p, long long chunk, long long layer, long long tokens, const char* what) {
        ++checks;
        if (p.chunk != chunk || p.layer != layer || p.tokens != tokens) {
            ++bad;
            std::printf("FAIL %s: got {chunk %lld, layer %lld, tokens %lld}, want {%lld, %lld, %lld}\n",
                        what, (long long) p.chunk, (long long) p.layer, (long long) p.tokens, chunk, layer, tokens);
        }
    };
    auto on = [&](bool got, bool want, const char* what) { eq(got ? 1 : 0, want ? 1 : 0, what); };

    // ---- pause_point: a 61-layer stage (LB 0, LE 61) over a run of 12345 tokens, chunks of 1000 -------
    {
        eq_point(strata::program::layer_pause::pause_point(1000, 5, 0, 61, 12345, 1000), 1000, 6, 1000,
                 "mid read, layer 5 done: goes on at 6 with the chunk's 1000 rows");
        eq_point(strata::program::layer_pause::pause_point(0, 0, 0, 61, 12345, 1000), 0, 1, 1000,
                 "after the FIRST layer pass there is already a between");
        eq_point(strata::program::layer_pause::pause_point(12000, 30, 0, 61, 12345, 1000), 12000, 31, 345,
                 "short last chunk: 345 rows, not C");
        eq_point(strata::program::layer_pause::pause_point(1000, 60, 0, 61, 12345, 1000), -1, -1, 0,
                 "the LAST layer pass: the chunk ends, the plain between-chunk stop's job");
        eq_point(strata::program::layer_pause::pause_point(1000, 61, 0, 61, 12345, 1000), -1, -1, 0,
                 "a layer past the stage: no pause");
        eq_point(strata::program::layer_pause::pause_point(1000, -1, 0, 61, 12345, 1000), -1, -1, 0,
                 "before the stage's first layer: nothing of the chunk is done");
    }
    {   // a layer split's stage [20, 41): the same rule inside the stage's own layers
        eq_point(strata::program::layer_pause::pause_point(0, 20, 20, 41, 8000, 1000), 0, 21, 1000,
                 "stage [20,41): after its first pass, goes on at its second");
        eq_point(strata::program::layer_pause::pause_point(0, 19, 20, 41, 8000, 1000), -1, -1, 0,
                 "stage [20,41): a pass below the stage: no pause");
        eq_point(strata::program::layer_pause::pause_point(0, 40, 20, 41, 8000, 1000), -1, -1, 0,
                 "stage [20,41): its last pass: no pause");
    }
    {   // a one-layer stage has no "between two layers", whatever the pass
        eq_point(strata::program::layer_pause::pause_point(0, 0, 0, 1, 8000, 1000), -1, -1, 0,
                 "one-layer stage: no pause");
    }
    {   // degenerate runs and grids: never a pause outside the read
        eq_point(strata::program::layer_pause::pause_point(12345, 5, 0, 61, 12345, 1000), -1, -1, 0,
                 "c0 past the run: no pause");
        eq_point(strata::program::layer_pause::pause_point(-1, 5, 0, 61, 12345, 1000), -1, -1, 0,
                 "negative c0: no pause");
        eq_point(strata::program::layer_pause::pause_point(0, 5, 0, 61, 0, 1000), -1, -1, 0,
                 "empty run: no pause");
        eq_point(strata::program::layer_pause::pause_point(0, 5, 0, 61, 12345, 0), -1, -1, 0,
                 "zero chunk: no pause");
    }

    // ---- resumes: which run goes on with a saved pause (the caller matched the ABSOLUTE position; the
    //      tokens themselves it compares - this is the whole-chunk check at THIS run's c0) ----------------
    {
        const Point same{1000, 6, 1000};     // paused mid-run A at run-relative 1000, rows for 1000 tokens
        on(strata::program::layer_pause::resumes(same, 1000, 12345, 1000), true,
           "the same run shape: 1000 rows == min(C, n - c0)");
        on(strata::program::layer_pause::resumes(same, 11000, 11500, 1000), false,
           "a run ending in a short (500-token) chunk: 1000 rows do not go on there");
        on(strata::program::layer_pause::resumes(same, 12000, 12345, 1000), false,
           "1000 rows against the 345-token tail: no");
    }
    {   // the cross-run case (scratch/review_pausepoint's shape): run A paused mid-run at its c0 2000; the
        // continuation run starts its read AT the paused chunk (the caller's absolute pos0 match), so its
        // own c0 is 0 - the saved run-relative label is stale there, and must not veto the resume
        const Point cross{2000, 12, 1000};
        on(strata::program::layer_pause::resumes(cross, 0, 10345, 1000), true,
           "a continuation run c0 = 0 goes on with the saved 1000 rows (its grid, not the paused run's)");
        on(strata::program::layer_pause::resumes(cross, 0, 345, 1000), false,
           "a continuation that reads only the short tail: 1000 rows do not fit 345");
        const Point tail{12000, 12, 345};    // paused in the short last chunk
        on(strata::program::layer_pause::resumes(tail, 12000, 12345, 1000), true,
           "the short chunk's 345 rows on the same grid: goes on");
        on(strata::program::layer_pause::resumes(tail, 0, 345, 1000), true,
           "the same short chunk as a one-chunk run: goes on");
    }
    {   // a different chunk size (relayout, --prefill-pipe's planned chunk): the rows only go on where the
        // run reads exactly as many tokens there
        const Point p{-1, 6, 1000};          // a restored pause: no run-relative label at all
        on(strata::program::layer_pause::resumes(p, 0, 12345, 1000), true, "C = 1000: 1000 rows go on");
        on(strata::program::layer_pause::resumes(p, 0, 12345, 2000), false,
           "C = 2000: the run reads 2000 where the pause saved 1000 - re-read the chunk whole instead");
        on(strata::program::layer_pause::resumes(p, 0, 600, 1000), false,
           "the run's tail is shorter than the rows: no");
        on(strata::program::layer_pause::resumes(p, 0, 1000, 1000), true,
           "a run of exactly the paused chunk: goes on");
    }
    {   // degenerate: no pause, no run, no grid, position outside the run
        on(strata::program::layer_pause::resumes(Point{}, 0, 12345, 1000), false, "no pause never resumes");
        on(strata::program::layer_pause::resumes(Point{0, 1, 0}, 0, 12345, 1000), false, "zero rows: no");
        on(strata::program::layer_pause::resumes(Point{-1, 1, 1000}, 0, 12345, 1000), true,
           "a restored pause has no run-relative label (give_pause): the label takes no part in the check");
        on(strata::program::layer_pause::resumes(Point{0, 1, 1000}, -5, 12345, 1000), false,
           "negative c0: no");
        on(strata::program::layer_pause::resumes(Point{0, 1, 1000}, 12345, 12345, 1000), false,
           "c0 at the run's end: nothing to read there");
        on(strata::program::layer_pause::resumes(Point{0, 1, 1000}, 0, 0, 1000), false, "empty run: no");
        on(strata::program::layer_pause::resumes(Point{0, 1, 1000}, 0, 12345, 0), false, "zero chunk: no");
    }

    // ---- state_bytes: the park budget and the log's "X MiB of chunk rows" figure -------------------------
    {
        eq((long long) strata::program::layer_pause::state_bytes(8192, 10240), 8192LL * 10240 * 4,
           "an 8192-token chunk of a 40 KiB row: 320 MiB");
        eq((long long) strata::program::layer_pause::state_bytes(345, 10240), 345LL * 10240 * 4,
           "the short last chunk: its own rows");
        eq((long long) strata::program::layer_pause::state_bytes(0, 10240), 0, "no tokens: no bytes");
        eq((long long) strata::program::layer_pause::state_bytes(8192, 0), 0, "no row: no bytes");
        eq((long long) strata::program::layer_pause::state_bytes(-1, 10240), 0, "negative: no bytes");
        eq((long long) strata::program::layer_pause::state_bytes(8192, -4), 0, "negative row: no bytes");
    }
    {   // saturation, never a wrap: 2^60 rows of 2^60 floats (4 bytes each) is far past a 64-bit count
        const uint64_t sat = strata::program::layer_pause::state_bytes(1LL << 60, 1LL << 60);
        eq((long long) sat, -1 /* (long long) UINT64_MAX */, "2^60 x 4096 EiB row: saturates at max");
        // exactly at the edge: the largest count that still fits goes through unrounded - 2^64 is not far away
        const uint64_t edge = strata::program::layer_pause::state_bytes((1LL << 62) - 1, 1);   // 4 x (2^62 - 1)
        eq((long long) edge, (long long) (((uint64_t) (1LL << 62) - 1) * 4), "the largest fitting count: exact");
        const uint64_t over = strata::program::layer_pause::state_bytes(1LL << 62, 1);         // exactly 2^64
        eq((long long) over, -1 /* (long long) UINT64_MAX */, "one past the edge (2^64): saturates");
    }

    std::printf("%s: %d checks, %d bad\n", bad ? "FAILED" : "ok", checks, bad);
    return bad ? 1 : 0;
}
