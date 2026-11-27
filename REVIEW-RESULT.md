# REVIEW-RESULT - P125i re-review (sc_score / ascore by use, GLM memory report rows H and J)

MERGE

Re-reviewed `git diff d789228f..HEAD` (6e9dab30a = e844828de row H + helper + test, 9794d7a81 row J,
68cddc147 report, 1ea088862 prior review, e20a3c4dc fix of review findings 1+2, f08362298 nit 4,
6e9dab30a fix report) against `PLAN-p125i-glm-small-buffers.md`, `REPORT-p125i-glm-small-buffers.md`
and `FIX-REPORT.md`. The prior review's FIX-FIRST findings are all resolved; no new findings.

## The prior findings, re-checked

### 1. HIGH (was: rows_dsa grew the shared side buffer mid-loop) - FIXED, verified correct

`src/core/glm_rows.cu:562-577`: the growth is hoisted ahead of the row loop - a first pass takes
`max(pool_done)` over the `n` rows (glm_rows.cu:563-564), the buffer grows at most once per
`rows_dsa` call (glm_rows.cu:565-576), and the row loop only resolves pointers from the settled
stride (glm_rows.cu:583-586). Verified there is no remaining window for the old
use-after-free / stale-stride:

- every score pointer a row captures (`sc.score[n_sc]`, glm_rows.cu:591; `sr.score[n_sr]`,
  glm_rows.cu:605; per-row `gf::dsa_score`, glm_rows.cu:594-595) is computed after the buffer is
  settled, and the after-loop launches (`dsa_score_rows` glm_rows.cu:625, `dsa_select_rows`
  glm_rows.cu:626) can no longer race a `cudaFree` (glm_rows.cu:573) - nothing frees or re-strides
  inside the loop;
- the growth condition is airtight: a row takes the side buffer only when `pool_done > ascore_n`
  (glm_rows.cu:585); if any row does, `pools_max > ascore_n` and the pre-pass guarantees
  `ascore_x != nullptr && ascore_xn >= pools_max >= pool_done` (grow or the call returned false);
- mixing buffers in one call is safe: `DsaScoreRows` / `DsaSelectRows` carry per-row score pointers
  with per-row `n_pools` / `n_vis` (include/maya/kernels/glm_fast.hpp:198-213) - no shared-stride
  assumption;
- allocation covers any call: `xn * rows_max()` floats (glm_rows.cu:568) and every caller bounds
  `n <= rows_max()` (glm_rows.cu:902, 962; include/maya/core/glm_model.hpp:978), so row index `i`
  always fits; `int pools_max` cannot overflow (1M ctx at kpool 4 is 262145);
- the previously exposed paths - a verify window crossing a `kpool` boundary above the carve, two
  sequences of different lengths, the MTP draft path (src/core/glm_mtp_rows.cu:946, same
  `rows_dsa`) - all grow before row 0 captures anything. Per-layer calls re-run the pre-pass;
  positions do not change within a step, so past the frontier layer the capacity already fits and
  no growth happens.

### 2. MEDIUM (was: J freed the old buffer before allocating the new one) - FIXED

`src/core/glm_rows.cu:567-575`: `cudaMalloc` into a local first; the old `ascore_x` is freed only
on success, then the stride swaps (mirrors `score_rows`, src/core/glm_model.cu:2278-2283). A failed
growth (glm_rows.cu:568-572) clears the error, keeps the old buffer and stride, and returns false
with the reason - and with finding 1 fixed, no pointer is captured yet, so the early return leaves
nothing dangling; the capacity is kept for the next step's retry. `score_rows` itself keeps the
safe order unchanged.

### 3. LOW (rows side buffer uncounted in bookkeeping) - NO CHANGE, agreed

Re-verified `FIX-REPORT.md`'s reasoning: `RowsState::bytes` is written once (src/core/glm_rows.cu:176)
and has no other consumer (grep: no other `S->bytes` / `rows_->bytes` read anywhere); the
"(SAGUARO_GLM_ROWS_ATTN)" load line (glm_rows.cu:179-185) prints before any growth and is honest
about the carve it prices. Pre-existing gap, recorded in REPORT open question 5. Stays on the
ledger, not a blocker.

### 4. NIT (elastic-ring inline) - FIXED

`src/core/glm_model.cu:1602` takes `glmfast::kScoreGridPositions` (src/core/glm_memory.hpp:222,
= 8256) instead of the inline `8192 + 64`; value unchanged, matches the ik-ring cap
(glm_model.cu:3508, src/core/glm_multi.cu:407,757).

### 5. NIT (cudaFree syncs on the grow frontier) - J half fixed, H half accepted

J's `cudaFree` moved out of the row loop (glm_rows.cu:573), at most once per call; H's
(glm_model.cu:2280) stays - the SAGUARO_GLM_SLOW reference path, at most ~7 doublings per context
(src/core/glm_memory.hpp:226-229), as the prior review itself allowed.

## New behaviour re-checked at the fixed HEAD

- **Helper**: `glm_score_rows_grow` (src/core/glm_memory.hpp:226-229) returns `max(pools, ...)` -
  never below the step's need even when `max_pools` caps low (an elastic `max_ctx_` change,
  src/core/glm_multi.cu:725, cannot leave a step short); `std::max(1, kpool)` guards the divide
  (glm_memory.hpp:224). The H grow cap `max_ctx_ / g_.idx_kpool` (glm_model.cu:2276) equals load's
  `max_pools` (glm_model.cu:758, load_pack glm_model.cu:3483).
- **H bookkeeping**: side-buffer bytes join `sc_bytes_` only after a successful alloc
  (glm_model.cu:2279,2283); `sc_bytes_` is read only for reporting (src/core/glm_fast_path.cu:1539
  load line, :5292 VramReport `act`) - no memcpy/free size derives from it. The split-auto
  estimator (glm_model.cu:1578-1611) prices the state arena only - still no `sc_score` term.
- **Threads**: `sc_bytes_` gains a runtime writer (`score_rows`, glm_model.cu:2279-2283) while the
  VRAM beat thread reads it via `vram_report()` (glm_fast_path.cu:5292; generate.cpp:5013-5024) -
  checked exclusive: the beat runs only `if (model.fast())`, and `score_rows` runs only under
  `step_layers`, the SAGUARO_GLM_SLOW path where `fast_mode_` is off (glm_model.cu:1536,3140). No
  race introduced. No locks in any changed path (no nested acquire), no new threads, no new
  prints; the rows load line's single fprintf is pre-existing (glm_rows.cu:179).
- **Feature OFF byte-identical**: `sc_score`'s only reader is `step_layers` via `score_rows`
  (glm_model.cu:2403; grep shows no other `sc_score` use); the fast path's `F->score` is untouched;
  with `SAGUARO_GLM_ROWS_ATTN` off the rows carve block is skipped (glm_rows.cu:153) and the
  `rows_dsa` grow pre-pass is `if (attn)`-gated (glm_rows.cu:562) with `score = F->score`
  (glm_rows.cu:583) - identical output bytes; only the load-time VRAM figures change, which is the
  plan's bookkeeping intent. `ascore_n0` (glm_rows.cu:109) is a pure integer `min`, side-effect
  free when off.
- **Destructors**: both side buffers freed - glm_model.cu:710, glm_rows.cu:203.
- **Generated file**: `serve/qfn_engine.py`, `engines/qfn/serve/server.py`,
  `tools/saguaro/extract_qfn_engine.py` are not in the diff - no generated file touched.
- **Paths / secrets**: no absolute paths outside the repo, no secrets (the only "token" hits are
  prose about 1M-token contexts); builds and scratch stay under relative `scratch/`.
- **Tests really exercise the change**: `check_score_rows` (tests/glm_memory_test.cpp:83-108)
  asserts the exact helper both buffers use - grid caps at the GLM geometry and kpool 1/2,
  short-context passthrough, no-op / doubling / jump-to-need / cap / zero-capacity growth,
  never-short never-over sweeps at 2064 / 65536 / 262144. Rebuilt and ran the plan's named module
  at HEAD (`scratch/build-sm75/glm_memory_test`): "GLM memory policy checks passed (CPU only)".
  No GPU runs, per the plan; the CUDA buffer-lifetime property finding 1 was about is by nature
  outside a CPU sizing test, verified by reading instead.
- **Build**: `scratch/build-sm75` (CUDA, `arch=compute_75,code=[compute_75,sm_75]` in the
  maya_core and maya_engine flags) builds current at 6e9dab30a with no warnings; `saguaro-glm`
  links. HIP not compiled, as REPORT open question 2 says.

## Manager check

Exactly rows H (`sc_score`, ref/REPORT-p125c-glm-mem.md:64) and J (`ascore`, :66) - `F->score`
(the fast path's same-shaped row, glm_fast_path.cu:1304) deliberately left per the plan's "two
buffers" and REPORT open question 1; bookkeeping lines: H's `sc_bytes_` tracks the grown buffer
into the `act` / state-KV lines, J's load line counts the carve (glm_rows.cu:184), side-buffer
bytes on no line as before (finding 3, open question 5); builds: sm_75 clean at HEAD, named test
module passes; worktree clean; commits per buffer as the plan asked (e844828de, 9794d7a81).

## Verdict

MERGE - findings 1 and 2 are fixed as prescribed and verified against the pointer-capture sites;
what remains is the documented set of non-blocking choices (buffers never shrink, doubling rather
than per-step reallocs, HIP build, the pre-existing rows-arena reporting gap), none of which
changes correctness or the OFF paths.
