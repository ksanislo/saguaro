# REVIEW-RESULT - P125i (sc_score / ascore by use, GLM memory report rows H and J)

FIX-FIRST

Reviewed `git diff d789228f..HEAD` (e844828de row H, 9794d7a81 row J, 68cddc147 report) against
`PLAN-p125i-glm-small-buffers.md` and `REPORT-p125i-glm-small-buffers.md`.

## Findings by severity

### 1. HIGH - rows_dsa grows the shared side buffer mid-loop, freeing and re-striding it while earlier
### rows' scores are still pending (correctness bug, use-after-free / wrong rows read)

`src/core/glm_rows.cu:564-578`. Inside one `rows_dsa` call each row's raw score pointer is captured at row
time - `sc.score[n_sc] = score` (glm_rows.cu:584) and `sr.score[n_sr] = score` (glm_rows.cu:598) - and both
are consumed only **after** the loop: the batched score launch `gf::dsa_score_rows` (glm_rows.cu:618) and the
selection `gf::dsa_select_rows` (glm_rows.cu:619). A later row can trigger the growth
(`pool_done > S->ascore_xn`, glm_rows.cu:565), which `cudaFree`s `S->ascore_x` (glm_rows.cu:567) and re-allocs
with a **new stride** `S->ascore_xn` (glm_rows.cu:575-576). Then:

- rows already captured from the old buffer keep dangling pointers; on the `SAGUARO_GLM_ROWS_SMALL` path the
  batched `dsa_score_rows` launch at glm_rows.cu:618 then **writes through freed pointers**; on the per-row
  path (`gf::dsa_score` launched at glm_rows.cu:587) the after-loop `dsa_select_rows` **reads freed memory**
  (or, when the allocator hands back the same base, reads row data under the old stride - silently wrong
  cells for the affected rows);
- with the new stride, `S->ascore_x + i * S->ascore_xn` no longer addresses the rows the pointers were
  captured for.

Reachable when, in one call, two rows have **different completed-pool counts above the carve**: a verify
window's rows of one conversation carry consecutive positions (glm_rows.cu:305-306, 527-530; the window
advances `p` by one - see `include/maya/core/glm_model.hpp:977-978`), so a window crossing a `kpool` boundary
has two `pool_done` values; also two sequences of different lengths in one batch. Conditions:
`SAGUARO_GLM_ROWS_ATTN` on, context past the 2064-pool carve (~8260 tokens at kpool 4), `n > 1`. The
first grow jumps exactly to the need (the helper's jump-to-need), so a second distinct `pool_done` in the
same call grows again one pool later; every doubling frontier hit mid-batch corrupts the rows ahead of the
frontier row. The MTP draft path calls the same function with advancing draft positions
(glm_mtp_rows.cu:946), so it is exposed the same way.

The old code was immune: one arena row of `attn_pools` (the maximum), constant stride, never reallocated
mid-call. Fix: hoist the growth out of the row loop - a first pass over the `n` rows (n <= rows_max(),
glm_rows.cu:76, hpp:978) takes `max(pool_done)`, grow once, then compute per-row pointers; or resolve rows
after the loop instead of capturing raw pointers. Note the H side does not have this problem (below).

### 2. MEDIUM - J's grow frees the old buffer before allocating the new one

`src/core/glm_rows.cu:567-574`: on a transient `cudaMalloc` failure the previously grown side buffer is
already gone (`ascore_x = nullptr`, `ascore_xn = 0`), so the step fails **and** the capacity is lost; rows
captured earlier in the same call point at the freed buffer (the early return skips the after-loop launches,
so no use occurs, but it is one abort away). H's `score_rows` does it in the safe order - allocate new
(`glm_model.cu:2278`), then uncount/free old (glm_model.cu:2279-2280) - J should mirror it.

### 3. LOW - the rows side buffer's grown bytes are on no bookkeeping line

`src/core/glm_rows_state.hpp:58` / `src/core/glm_rows.cu:568`: `S->bytes` and the "(SAGUARO_GLM_ROWS_ATTN)"
load line count the carve only (load line glm_rows.cu:180-185), and `RowsState::bytes` has no consumer
outside that line. Report open question 5 states this honestly, and the pre-change arena never reached
VramReport either, so the manager's "bookkeeping lines updated" holds for J (the line is honest about what
it prices). Recorded so the gap is on the ledger.

### 4. NIT - kScoreGridPositions not reused by the elastic-ring inline

`src/core/glm_model.cu:1602` still writes `8192 + 64` inline in `glm_search_bounds`; same grid as
`kScoreGridPositions` (src/core/glm_memory.hpp:222). Pre-existing line; tidiness only.

### 5. NIT - mid-decode cudaFree on the grow frontier device-syncs

`src/core/glm_model.cu:2280` and `src/core/glm_rows.cu:567`: `cudaFree` blocks for device completion, a stall
at each doubling frontier. Acceptable on the reference slow path / a handful of times overall; fixing #1 by
hoisting also moves J's sync out of the per-row loop.

## Checked clean

- **H correctness**: `score_rows` (src/core/glm_model.cu:2271-2285) allocates the new buffer before freeing
  the old, and a freed side buffer is never referenced: each layer's `glm_dsa_score`/`glm_dsa_select` pair is
  enqueued back-to-back on the legacy default stream (src/core/glm_model.cu:2401-2412), so a later layer's
  cudaFree (which device-syncs) cannot race an unfinished reader. The grow cap `max_ctx_ / g_.idx_kpool`
  (glm_model.cu:2276) equals load's `max_pools` (glm_model.cu:758, load_pack glm_model.cu:3483) and
  `max_ctx_` is set in both (glm_model.cu:723, 2940); if an elastic context changes `max_ctx_`
  (glm_multi.cu:725), `glm_score_rows_grow` still returns >= the step's need (src/core/glm_memory.hpp:226-228)
  - never short, and `std::max(1, kpool)` guards the divide. `pool_done` can exceed neither without also
  overflowing the old code, so the contract is unchanged.
- **H bookkeeping honesty**: the side buffer's bytes are added to `sc_bytes_` (glm_model.cu:2283) and removed
  on reallocation (glm_model.cu:2279) only after a successful alloc; `sc_bytes_` feeds the VramReport `act`
  line (src/core/glm_fast_path.cu:5292) and the state/KV INFO line's known total (glm_fast_path.cu:1539) -
  both now follow the longest context actually decoded, per the plan. `sc_bytes_` is bookkeeping-only (no
  memcpy/free sizes derive from it), so arena + side both counted is right.
- **Estimator claim verified**: the split-auto estimator's state term (glm_model.cu:1578-1611) prices the
  state arena (lat, rings, pools) only - no per-pool score term - so it needed no change, as the report says.
- **Destructors**: both side buffers freed (glm_model.cu:710, glm_rows.cu:203); no other path releases the
  arenas (only `~Glm5Model` frees `sc_`).
- **Feature OFF byte-identical**: `sc_score` has exactly one reader - `step_layers` (grep: glm_model.cu
  820/3608/2273/2403/2412 only) - and `step_layers` is the SAGUARO_GLM_SLOW reference decode; the fast path's
  own row `F->score` (glm_fast_path.cu:1304) is untouched, matching the manager's rows H+J scope. With
  SAGUARO_GLM_ROWS_ATTN off, rows_alloc's block is skipped (glm_rows.cu:102, 153) and rows_dsa keeps
  `score = F->score` (glm_rows.cu:563) - the OFF path is unchanged. The sc_/rows arena offset shifts are
  internal to the arenas.
- **Locks/threads/stdout**: no locks in the changed paths (no nested acquire anywhere); rows_dsa runs
  single-threaded per model from the decode thread (glm_rows.cu:429, glm_mtp_rows.cu:946); no new threads;
  no new prints (the rows load line is the pre-existing stderr line, glm_rows.cu:180).
- **Generated file**: `serve/qfn_engine.py` and `tools/saguaro/extract_qfn_engine.py` are not in the diff -
  the diff is exactly the six source/test files the report lists plus the report; nothing changed by hand.
- **Paths/secrets**: no absolute paths outside the repo and no secrets anywhere in the diff; builds and
  scratch live under relative `scratch/`.
- **Tests really exercise the change**: `check_score_rows` (tests/glm_memory_test.cpp:81-107) tests the exact
  helper both buffers use - grid caps at the GLM geometry and kpool 1/2, short-context passthrough, no-op
  when it fits, doubling, jump-to-need, `max_pools` cap, zero capacity, never-short/never-over sweeps at
  2064/65536/262144. Reran at HEAD: rebuilt `scratch/build-sm75 --target glm_memory_test` and ran it: "GLM
  memory policy checks passed (CPU only)". No GPU runs, per plan. (Finding 1 is precisely what a CPU helper
  test cannot reach: CUDA buffer lifetime, not sizing arithmetic.)
- **Build**: `scratch/build-sm75` binaries are current at HEAD; `saguaro-glm` carries `-arch sm_75` cubins,
  matching the report's sm_75 claim.
- **Report accuracy**: helper semantics, carves (8256/kpool = 2064 pools = 8 KB), per-buffer commits, the J
  load-line change, and open questions 1-5 all match the code; the J figures are self-corrected against the
  source report honestly (report lines 40-43).

## Manager check

Exactly rows H (`sc_score`) and J (`ascore`) changed - F->score (the report's row-adjacent fast-path buffer)
deliberately left, matching "rows H and J" scope; bookkeeping lines: H fully honest (sc_bytes_ tracks the
grown buffer; the split-auto estimator confirmed score-free), J's load line honest about the carve with the
side buffer's grown bytes uncounted as before (finding 3); builds: sm_75 target rebuilt clean, named test
module passes.

## Verdict

FIX-FIRST - finding 1 must be fixed (hoist the side-buffer growth ahead of the row loop, or defer pointer
resolution) before merge; the H half is mergeable as-is.
