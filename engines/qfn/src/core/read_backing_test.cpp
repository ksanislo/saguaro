// P124e (round 8, manager card): the staging size and the read's backing order for a prompt read on an ELASTIC
// conversation (include/strata/core/read_backing.hpp): the loan never promises cells past the backed end plus
// one grow-ahead step, and a read's pieces stop at what is backed and staged - never touching a cell whose
// host copy / follow state / RoPE rows are addresses without memory.  Pure arithmetic - no GPU, no allocation.
#include "strata/core/read_backing.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
constexpr int64_t FULL = 1048576;   // the card's --max-context
constexpr int64_t N = 546065;       // the card's prompt: the run that SIGSEGVed (scratch/r7-crash-tail.log)
}  // namespace

int main() {
    // ---- the plain grid (Prefill::stage_cells_for's rule, unchanged for a conversation nothing of is elastic)
    check(stage_grid_cells(0, FULL) == 0, "no reach: no staging");
    check(stage_grid_cells(30000, FULL) == 262144, "a 30K read lays out the base");
    check(stage_grid_cells(N, FULL) == 786432, "the card's 546K read: the 786432-cell loan of the crash log");
    check(stage_grid_cells(FULL + 7, FULL) == FULL, "a loan never passes the context");
    check(stage_grid_cells(300000, FULL, false) == 393216, "headroom off: the reach itself, rounded to the grid");

    // ---- the elastic size: a loan never out-promises the backing by more than the one step
    check(stage_cells_elastic(N, -1, FULL) == 786432, "nothing elastic: today's sizing word for word");
    check(stage_cells_elastic(N, FULL, FULL) == 786432, "all backed: today's sizing too");
    check(stage_cells_elastic(N, 546148, FULL) == 786432, "the card's numbers: backed+step rounds to the SAME loan");
    check(stage_cells_elastic(N, 100000, FULL) == 231072, "backed 100000: the loan stops at backed+step, not 786432");
    check(stage_cells_elastic(N, 0, FULL) == kStageBackAhead, "nothing backed: one step of staging, no more");
    check(stage_cells_elastic(N, 32767, FULL) == 163839, "below the grid's base the base does not out-promise either");
    check(stage_cells_elastic(N, 655361, FULL) == 786432, "a state bigger than the reach needs: the reach size rules (grid respected)");
    check(stage_cells_elastic(FULL, 655361, FULL) == 917504, "past the base the cap rides the grid");
    check(stage_cells_elastic(30000, 500000, FULL) == 262144, "a short read keeps its base when the state covers it");
    check(stage_cells_elastic(N, 310000, FULL, false) == 524288, "headroom off: the cap decides alone");
    check(stage_cells_elastic(N, FULL - 1, FULL) == 786432, "one cell short of the context: still the reach's size");
    for (int64_t backed = 0; backed < FULL; backed += 32768) {
        const int64_t loan = stage_cells_elastic(N, backed, FULL);
        check(loan <= 786432 && (loan > backed || loan == 786432) &&
                  (loan == 786432 || loan <= backed + kStageGridStep + kStageBackAhead),
              "the loan covers the backed end, rides at most one step past it, never past the reach's size");
    }
    for (int64_t backed = 0; backed <= FULL; backed += 65536)   // monotonic: a growing state never shrinks a fresh loan
        check(stage_cells_elastic(N, backed, FULL) <= stage_cells_elastic(N, backed + 1, FULL),
              "the elastic loan grows with the backing");

    // ---- the read's pieces: [q, end) stays inside what is backed AND staged
    check(read_piece_end(N, 546148, 786432) == N, "the card's read is ONE piece: backing covers the prompt");
    check(read_piece_end(N, 400000, 786432) == 400000, "a piece stops at the backed end");
    check(read_piece_end(N, 600000, 524288) == 524288, "a piece stops at the staging");
    check(read_piece_end(N, 400000, 786432) == 400000, "nothing runs before the backing grows");
    check(read_piece_end(N, 400001, 786432) == 400001, "one backed cell runs one cell");

    // the grow ahead of the piece: past `ahead`, never past the request's end, always past q
    const int64_t req_end = 546065 + 16 + 64;
    check(read_back_cells(0, 32768, req_end) == 32768, "the first grow backs the ahead-step");
    check(read_back_cells(530000, 32768, req_end) == req_end, "the last grow stops at the request's end");
    check(read_back_cells(100, 0, req_end) == 101, "a grow moves at least one cell");

    // the short simulation of the card's read on a card that backs 131072 cells and lends what that covers:
    // every touched cell is backed when touched, the run ends at the staging with an error - not a fault.
    int64_t backed = 131072;
    const int64_t staged = stage_cells_elastic(N, backed, FULL);
    int64_t q = 0, max_touched = 0, grows = 0;
    bool stuck = false;
    while (q < N && !stuck) {
        int64_t qe = read_piece_end(N, backed, staged);
        if (qe <= q) {
            if (staged <= q) { stuck = true; break; }   // the engine's error path, never an access
            const int64_t want = std::min(staged, std::max(read_back_cells(q, 32768, req_end), q + 1));
            backed = std::max(backed, want);            // (the fits succeed here; the caps below bound them)
            ++grows;
            qe = read_piece_end(N, backed, staged);
            if (qe <= q) { stuck = true; break; }
        }
        check(qe <= backed && qe <= staged, "a piece never reaches past the backed or the staged cells");
        max_touched = qe;
        q = qe;
    }
    check(max_touched == staged, "the short read runs to the staging's end and no further");
    check(stuck && q >= staged, "and ends on the error, not an access past the loan");
    check(grows > 0 && grows < 16, "the read grows the backing in ahead-steps, not per cell");

    // and the healthy card: the whole prompt is one piece with zero grows (the fix changes nothing there)
    q = 0;
    int64_t pieces = 0;
    int64_t healthy_back = 546148;
    while (q < N) { q = read_piece_end(N, healthy_back, 786432); ++pieces; }
    check(pieces == 1 && q == N, "the card's healthy read stays the single run it was");

    std::printf("read_backing_test: %d checks OK\n", checks);
    return 0;
}
