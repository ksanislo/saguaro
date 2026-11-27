# FIX-REPORT - P125i review fixes (GLM memory report rows H and J)

Fixes against `REVIEW-RESULT.md` (verdict FIX-FIRST) on `PLAN-p125i-glm-small-buffers.md`.
Two commits: e20a3c4dc (findings 1 + 2, row J), f08362298 (nit 4). Findings 3 and 5 need no separate
change (why below). No commit touched `serve/qfn_engine.py` or `engines/qfn/serve/server.py` - the fixes
are all CUDA-side, nothing to re-extract.

## Finding 1 (HIGH) - rows_dsa grew the shared side buffer mid-loop while earlier rows' pointers were pending

FIXED (e20a3c4dc), the review's first remedy: `src/core/glm_rows.cu:555-577`. The growth is hoisted out of
the row loop - a first pass over the `n` rows takes `max(pool_done)`, the buffer grows at most once per
`rows_dsa` call (`glm_score_rows_grow` unchanged), and the row loop only computes per-row pointers from the
settled stride. No pointer in `sc.score[]` / `sr.score[]` can dangle or wear a stale stride: a verify window
crossing a `kpool` boundary above the carve, two sequences of different lengths in one batch, and the MTP
draft path (`glm_mtp_rows.cu:946`, same function) all grow before row 0 captures anything. Per-layer calls
are unaffected: positions do not change within a step, so after the frontier layer the capacity simply fits.

## Finding 2 (MEDIUM) - the J grow freed the old buffer before allocating the new one

FIXED in the same edit, mirroring `score_rows` (src/core/glm_model.cu:2277-2282): the new buffer is
`cudaMalloc`ed into a local first; only on success is the old `ascore_x` freed and the stride swapped
(glm_rows.cu:566-575). A failed growth leaves the old buffer and its stride in place - with finding 1 fixed
no row has captured anything yet, so the early return touches nothing freed, the capacity is kept for the
next step's retry, and `err` says the reason.

## Finding 3 (LOW) - the rows side buffer's grown bytes are on no bookkeeping line

NO CHANGE, deliberately, as the finding itself concludes. `RowsState::bytes` (src/core/glm_rows_state.hpp:27)
has no consumer outside the "(SAGUARO_GLM_ROWS_ATTN)" load line (src/core/glm_rows.cu:180-185), and that line
prints once at load, before any growth can happen - adding the side buffer's bytes there is impossible, and
adding them to `S->bytes` after growth would count bytes no line ever reads. A print at each grow would break
the review's own "no new prints" property for a handful-of-times event. The load line stays honest about the
carve it prices (plan: "keep the bookkeeping lines honest" - held for what the line claims), the pre-change
arena never reached VramReport either, and the gap stays recorded in REPORT open question 5.

## Nit 4 - kScoreGridPositions not reused by the elastic-ring inline

FIXED (f08362298): src/core/glm_model.cu:1602 takes `glmfast::kScoreGridPositions` (src/core/glm_memory.hpp:222)
instead of writing `8192 + 64` inline again; value unchanged.

## Nit 5 - mid-decode cudaFree on the grow frontier device-syncs

J half FIXED for free by finding 1: the `cudaFree` now sits ahead of the row loop, at most once per
`rows_dsa` call, not inside the per-row pass. H half UNCHANGED per the review's own judgement - the reference
slow path, a handful of doublings over a context's life (at most ~7 to 1M ctx per the helper), not worth
deferred-free machinery there.

## Verification

- `cmake --build scratch/build-sm75 --target maya_engine` - both changed .cu files recompile clean;
  `--target maya` relinks `saguaro-glm` (same sm_75 flags, incremental).
- `--target glm_memory_test` rebuilt and run (the plan's named module, the only one): "GLM memory policy
  checks passed (CPU only)". No GPU runs, per plan; finding 1 lives in CUDA buffer lifetime a CPU helper
  test cannot reach, as the review noted.
- Worktree clean apart from this report (force-added: FIX-REPORT.md sits in the main checkout's info/exclude,
  as REVIEW-RESULT.md does).
