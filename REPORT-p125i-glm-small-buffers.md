# P125i - GLM: size sc_score and ascore by use, not by --max-context

Per `PLAN-p125i-glm-small-buffers.md`: rows H and J of `ref/REPORT-p125c-glm-mem.md` section 3 - the two small
indexer-score buffers that were sized for the whole `--max-context` at load. Both are now carved at one sub-batch
grid of pools and grow on use (doubling toward the context's own pool count), the plan's "size them per step from
the pools done (or cap at the sub-batch grid, as row I already is)". One commit per buffer, as the plan asked.

## What changed

**The shared pure helper** (`src/core/glm_memory.hpp`, new, in `maya::core::glmfast` next to the other CPU-only
policy helpers):

* `kScoreGridPositions = 8192 + 64` - the prompt path's largest sub-batch + slack, the same grid the ik/ig rings
  keep (row C), so the carve is `glm_score_rows_start(max_pools, kpool) = min(max_pools, 8256 / kpool)` pools:
  2064 pools at the GLM geometry (kpool 4), 8 KB of floats instead of the 32 KB..1 MB the report's row H carried.
* `glm_score_rows_grow(cap, pools, max_pools)` - a step whose completed pools pass the capacity doubles it toward
  `max_pools`, but never less than the step's need (so it is never short); zero capacity jumps straight to the
  need. From the grid that is ~7 reallocs to a 1M-token context.

**Row H - `sc_score`, the scratch arena** (`include/maya/core/glm_model.hpp`, `src/core/glm_model.cu`; the H line
glm_model.cu:3582 in the report's numbering):

* `load()` and `load_pack()` now carve `sc_score_cap_ = glm_score_rows_start(max_pools, idx_kpool)` floats, not
  `max_pools`.
* New `Glm5Model::score_rows(pools, out, err)`: while a step's `pool_done = (p + 1) / idx_kpool` fits the carve it
  returns the arena offset; past it a side buffer (`sc_score_x_`) doubles toward `max_ctx_ / idx_kpool`, and its
  bytes move into `sc_bytes_` as it grows. So the "activations" figure of the `glm fast: ... state/KV (N ctx)`
  line (glm_fast_path.cu, VRAM-before-the-pool) and the `VramReport` `act` line / `sc_bytes_` bookkeeping follow
  the longest context actually decoded, not `--max-context` - the arena alone shrinks 32 KB..1 MB per card at
  load, and the grown buffer only appears after that long a context has really run.
* `step_layers()` (the SAGUARO_GLM_SLOW decode, the only user of `sc_score`) takes the row from `score_rows()`;
  `glm_dsa_score` writes `pool_done` floats into it and `glm_dsa_select` reads them back - same kernels, same
  values, a bigger row when the context is bigger. The side buffer is freed in the destructor.
* The split-auto estimator (glm_model.cu, "the state at the context as load_pack lays it out") never priced
  `sc_score` - it prices the state arena only - so it needed no change to stay honest.

**Row J - `ascore`, the batched rows' score rows** (`src/core/glm_rows.cu`, `src/core/glm_rows_state.hpp`; the
J lines glm_rows.cu:104-106,157-158 in the report's numbering, opt-in `SAGUARO_GLM_ROWS_ATTN`):

* `rows_alloc()` carves `S->ascore_n = glm_score_rows_start(attn_pools, kpool)` floats a row instead of
  `attn_ctx / kpool + 1` - the same 1 B a position as row H: R x 32 KB at 32K ctx, R x 256 KB at 262K, R x 1 MB
  at 1M (the report's J row figures R x 1 MB / R x 4 MB there read 4 B a position, not the formula's); now R x
  8 KB a row at load (R = `rows_max()`). The "(SAGUARO_GLM_ROWS_ATTN) ... MB of the rows' MB" load line counts
  the carve, not the whole-context rows.
* `rows_dsa()`: a row whose completed pools pass the carve scores (and selects) from a side buffer `S->ascore_x`
  shared by all the rows, stride `S->ascore_xn` doubling toward `S->ascore_max` (= the longest sequence's pool
  count, the old `attn_pools`) through the same `glm_score_rows_grow`; a malloc failure there returns the honest
  "the indexer scores of N pools a row did not allocate" instead of a load-time context figure. `rows_destroy()`
  frees it. `ascore_max` also bounds a row's need by construction (`rw[i].p` < its sequence's `max_ctx`).

Both grow paths stay on the whole-pool contract the kernels demand (a selection ranks every completed pool, so
the row really is `pool_done` floats deep at a long position) - the win is at load, where the whole context was
paid before a single token ran, which is what rows H and J measured.

## Files

* `src/core/glm_memory.hpp` - the two pure helpers + the grid constant.
* `tests/glm_memory_test.cpp` - `check_score_rows()` (see tests below).
* `include/maya/core/glm_model.hpp` - `sc_score_cap_ / sc_score_xcap_ / sc_score_x_`, `score_rows()` decl.
* `src/core/glm_model.cu` - carve x2 (load, load_pack), `score_rows()`, `step_layers()` uses, dtor free.
* `src/core/glm_rows.cu` - carve, the load-line figure, grow-on-use in `rows_dsa`, `rows_destroy()` free.
* `src/core/glm_rows_state.hpp` - `ascore_x / ascore_xn / ascore_max`.

Commits (per buffer, on `saguaro/q1-p125i-glm-small-buffers`): step 1 `e844828de` (row H + helper + test),
step 2 `9794d7a81` (row J).

## Tests run

* `glm_memory_test` (the CPU-only policy module that already tests this header; the plan's "a unit test if the
  sizing is a pure helper" - the sizing is one) - `check_score_rows()`: the grid caps (262144 and 32768 ctx both
  carve 2064 pools, a shorter context its own `max_pools`, kpool 1/2 keep the grid in positions), the growth
  (no-op when it fits, double at the grid's edge, jump-to-need when a step is far ahead, capped at
  `max_pools`, zero capacity exact), and a never-short / never-over sweep at `max_pools` 2064 / 65536 / 262144.
  PASS ("GLM memory policy checks passed (CPU only)"), also built through CMake and run from
  `scratch/build-sm75/glm_memory_test` after both commits.
* `saguaro-glm` (the `maya` target) built for **CUDA sm_75** in a new dir `scratch/build-sm75`
  (`-DMAYA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75`, CUDA 13.1): clean link, the compile flags are
  `arch=compute_75,code=[compute_75,sm_75]` and the binary carries sm_75 cubins. No GPU runs, per the plan.
  (`maya_engine` rebuilt after each commit; the only warnings in the log are pre-existing
  glm_fast_path.cu / generate.cpp ones.)
* No other test modules: the plan names only the helper unit test, and the GLM parity/model tests
  (`glm_model_test`, `glm_pack_test`, the kernels' parities, the serve panels) are GPU runs this plan forbids.

## Open questions

1. The fast path's own decode keeps its score row in the `F` arena - `F->score = fa_take(at, max_pools)` at
   glm_fast_path.cu:1304 (same 1 B a position). The report's row H names only `sc_score` at glm_model.cu:3582 and
   the plan says the two buffers, so it is untouched; it is the same pattern one file over, ~2 KB of a helper
   call away from the same treatment if wanted.
2. HIP: the change is runtime-neutral (`cudaMalloc`/`cudaFree` are the HIP macros, the helper is header-only), but
   only the CUDA sm_75 build was compiled here, per the plan. A HIP build of the branch was not run.
3. Growth granularity: the side buffers double (never below the step's need) rather than reallocating to exactly
   `pool_done` every kpool tokens - a literal "per step" realloc would be one `cudaMalloc` per 4 positions on the
   reference path. The report's "size them per step from the pools done (or cap at the sub-batch grid)" reads
   satisfied either way; the doubling is the row-I-style reading.
4. The side buffers never shrink (a reset leaves the capacity in place) - like the ik/ig rings, the arena keeps
   what the longest conversation used. If the serve loop should hand the VRAM back when contexts release
   (`SAGUARO_GLM_ELASTIC_KEEP`-style), that trim is not implemented; it is outside rows H and J's numbers.
5. `ascore`'s side buffer: `RowsState::bytes` (the arena size) has no consumer outside the "(SAGUARO_GLM_ROWS_ATTN)"
   load line, and nothing in VramReport carries the rows arena at all - so the side buffer's grown bytes reach the
   books only through the card's used total, exactly as the arena's always did. If the context panel should itemize
   the rows arena, that is a pre-existing gap, not one this change widened.
