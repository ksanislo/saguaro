// strata/core/read_backing.hpp - P124e (round 8, manager card): the pure arithmetic of what a PROMPT READ may
// touch on an ELASTIC conversation, where its staging loan stops, and how far the backing must run ahead of
// the chunk the read is about to touch.  A read indexes the host copy, the device follow state and the RoPE
// rows by CELL; those arrays hold addresses for the whole context but physical memory only for their backed
// prefix, so a loan sized from the whole context (or from a reach headroom) that lays staging out past the
// backed end invites an access past a mapped page - which is not a refusal, it is a SIGSEGV.  Everything here
// is a function of its arguments; the engine (prefill.cpp's guard, generate.cpp's read loop) asks these
// questions, and src/core/read_backing_test.cpp checks the answers.  The crash this header exists for: a
// 546,065-cell prompt on a conversation backed for 546,148 cells got a 786,432-cell staging loan.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::core {

/// The staging loan's grid (Prefill::stage_cells_for's rule): up to `kStageGridBase` cells the loan is that
/// size; past it the read's reach (+ the quarter's re-lend headroom when `headroom`) rounded up to
/// `kStageGridStep`, capped by the context `full`.
inline constexpr int64_t kStageGridBase = 262144, kStageGridStep = 131072;
/// P124e round 8: how far past the BACKED end a loan may still be laid out - one growth step, the ahead-step
/// the read's per-chunk grow backs the conversation with before the piece that touches the cells.
inline constexpr int64_t kStageBackAhead = 131072;

/// The cells a loan for a read reaching `reach` cells lays out (0: `reach` <= 0 - no staging).
inline int64_t stage_grid_cells(int64_t reach, int64_t full, bool headroom = true) {
    if (reach <= 0 || full <= 0) return 0;
    const int64_t want = headroom ? std::min(full, reach + std::max<int64_t>(65536, reach / 4)) : reach;
    return want <= kStageGridBase ? std::min(full, kStageGridBase)
                                  : std::min(full, (want + kStageGridStep - 1) / kStageGridStep * kStageGridStep);
}

/// P124e round 8: the same on an ELASTIC conversation - a loan never promises cells the state is not backed
/// for.  `backed` < 0 or >= `full`: nothing elastic or all backed: the plain sizing.  Else the loan stops at
/// the backed end plus one step, on the same grid (so a conversation one cell past a step keeps its loan);
/// below the grid's base the base does not out-promise the backing either.
inline int64_t stage_cells_elastic(int64_t reach, int64_t backed, int64_t full, bool headroom = true) {
    const int64_t want = stage_grid_cells(reach, full, headroom);
    if (want == 0 || backed < 0 || backed >= full) return want;
    const int64_t past = backed + kStageBackAhead;
    const int64_t cap = past <= kStageGridBase ? std::min(kStageGridBase, past)
                                               : (past + kStageGridStep - 1) / kStageGridStep * kStageGridStep;
    return std::min(want, std::min(full, cap));
}

/// Where the read's next run may end: at most `r`, never past what is backed (`backed`) nor past what the
/// staging holds (`staged`, the whole context when there is none).  <= `q`: nothing can run until the
/// backing or the loan grows.  Never touches a cell outside [q, end).
inline int64_t read_piece_end(int64_t r, int64_t backed, int64_t staged) {
    return std::min(r, std::min(backed, staged));
}

/// The cells to back before the piece at `q` (the fits then give the card / RAM the room the same way the
/// request-start fit does): `ahead` cells past `q`, never past the request's own end.
inline int64_t read_back_cells(int64_t q, int64_t ahead, int64_t request_end) {
    return std::min(request_end, q + std::max<int64_t>(1, ahead));
}

}  // namespace strata::core
