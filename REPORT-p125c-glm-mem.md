# P125c - GLM-5.3-Flash: what still scales with --max-context at load (REPORT ONLY)

Read-only investigation, 2026-11-26. No code changed. Sources: `ref/mortar-run-20261126.txt`
(the RX 9070 XT run: Maya-L `saguaro-l-hip` and Maya-S `saguaro-s-v2-iq2_xxs`, 32768 vs 262144),
the GLM engine under `src/` (`src/core/glm_*`, `src/program/generate.cpp`,
`include/maya/core/glm_model.hpp`, `include/maya/kernels/glm_kv_stream.hpp`), and the
CHANGELOG. The plan pointed at `logs/maya/PLAN-parallel-conversations.md` for the STRATA_GLM_SEQS /
elastic-ctx history; that file is **not in this checkout** (only `CHANGELOG.md:1756`, the #94
release note, and the code comments describe it), so section 2 is read off the code itself.

All line references are to this worktree at `17e6b5bb3`.

## Basis: geometry and bytes a position

`Glm5Geometry` (include/maya/core/glm_model.hpp:67-91, released GLM-5.3-Flash defaults, mirrored
by `ref/glm.py`): 45 layers, of which the run logs say **11 are DSA layers** (the rest KDA, whose
state is one fixed-size matrix + conv history per layer and never scales with the context);
`kv_lora 512` (a latent-cache row = 512 fp16 = **1024 B a position a layer**),
`idx_key 128`, `idx_kpool 4` (one pooled index key per 4 positions = 128 floats = 512 B =
**128 B a position a layer**). So for one conversation the context-proportional bytes are:

| per position | VRAM fast path | formula |
|---|---|---|
| DSA latent cache | 11 x 1024 B = 11,264 B | src/core/glm_model.cu:3468-3470 (`lat_floats`) |
| DSA pooled index keys | 11 x 128 B = 1,408 B | src/core/glm_model.cu:3491,3508 (`idx_key * max_pools`) |
| KV streaming host copy (when on) | 11,264 B | src/core/glm_model.cu:907-910 |
| KV streaming page tables | 11 B | src/core/glm_model.cu:920,938 (4 B a block of 4 positions) |
| prompt-path stage ceiling / score rows / scratch | 1,024 + 256 + 1 B | see rows G-I below |

A loaded NextN/MTP draft block adds one more DSA-like layer with the same three rates
(src/core/glm_model.cu:3483-3492). Maya-L and Maya-S share this geometry (the log's state/KV and
KV-streaming lines are identical for both).

The log's "state/KV" figure is `state_bytes_` printed verbatim (src/core/glm_fast_path.cu:1539-1544),
and the arithmetic below reproduces the measured 0.62 / 0.92 GB and 2.95 GB exactly, which is the
check that this enumeration is the whole set:

* 32768, streaming off (`kv_resident_cells` returns 0 when `cells >= max_ctx`,
  src/core/glm_model.cu:869-876): latent 369 MB + pools 46 MB + ik/ig rings 93 MB + KDA 153 MB
  = **0.62 GB**.
* 262144, streaming on (resident window 32768): slots 369 MB + **pools 369 MB** + rings 93 MB +
  KDA 153 MB = **0.92 GB**; host copy 262144 x 11,264 B = **2.95 GB pinned** - the log's line
  src/core/glm_model.cu:950-952. The 0.62 -> 0.92 growth is the **pooled index keys**, nothing else.

## 1. Every allocation sized by --max-context at load

`--max-context` (src/program/generate.cpp:5513) becomes `max_ctx` of `load_pack`
(src/program/generate.cpp:1135). VRAM rows are the fast path on one card; sizes at 32768 /
262144 / 1048576 positions, 11 DSA layers, streaming on above 32768 (`--kv-resident` default
window 32768, src/program/generate.cpp:5703-5725; min 20480,
include/maya/kernels/glm_kv_stream.hpp:34).

### VRAM (all in the state arena `state_`, one cudaMalloc at src/core/glm_model.cu:3545)

| | allocation | file:line | 32768 | 262144 | 1048576 |
|---|---|---|---|---|---|
| A | DSA latent cache **rows** - only when streaming is off (ctx <= window, or `--kv-streaming off`): full ctx rows | glm_model.cu:3466-3470 | 0.37 GB | 2.95 GB (off) / 0.37 GB slots | 11.8 GB (off) / 0.37 GB slots |
| E | KV streaming **VRAM slots** (= A capped at the resident window) | glm_model.cu:3465-3466, arena runs per glm_multi.cu:750-754 | - (off at <= window) | 0.37 GB | 0.37 GB |
| B | **DSA pooled index keys** (`dsa_pool_`, 128 floats per 4 positions) | glm_model.cu:3491,3508 | 0.046 GB | 0.37 GB | 1.48 GB |
| C | ik/ig indexer-key **rings** - `min(ctx, 8192+64)`: constant for any ctx >= 8256 (this is the elasticized-by-cap case) | glm_model.cu:3482,3503-3506 | 0.093 GB | 0.093 GB | 0.093 GB |
| D | KDA states + conv history (34 layers) - constant | glm_model.cu:3495-3499 | 0.15 GB | 0.15 GB | 0.15 GB |
| F | KV streaming **residency maps** `map_arena` (page_table = 4 B a block = 1 B a position a layer) | glm_model.cu:920-944 | - (off) | 2.9 MB | 11.5 MB |
| G | KV streaming **prompt staging ceiling** `kv_stage_bytes_` = one layer's whole cache (ctx x 1024 B). Not an extra allocation: the prompt path carves it from the expert pool's lendable tail per request, sized to the prompt's end (src/core/glm_prefill.cu:1635-1642) - but the startup chunk layout is sized with the *whole-context* ceiling (src/core/glm_prefill.cu:1600-1606), i.e. it costs expert slots at 1M | glm_model.cu:967 | 34 MB | 268 MB | 1.07 GB |
| H | scratch `sc_score` = `max_pools` floats (1 B a position) | glm_model.cu:3582 | 32 KB | 0.26 MB | 1 MB |
| I | prompt-path DSA **score rows** `sub x max_pools` floats in the borrowed region - the sub-batch starts at kSub=256 and only doubles while that stays under the MoE scratch (src/core/glm_prefill.cu:246-247,356-374), so its floor is 256 B a position | glm_prefill.cu:246-247,1582 | 8 MB | 67 MB | 268 MB |
| J | rows-path `ascore`, one `attn_pools`-float row per conversation row (opt-in `SAGUARO_GLM_ROWS_ATTN`) | src/core/glm_rows.cu:104-106,157-158 | R x 32 KB | R x 1 MB | R x 4 MB |
| K | extra sequence arenas, `SAGUARO_GLM_SEQS>1`: a **whole second state arena per extra conversation** for `SAGUARO_GLM_SEQ_CTX` (default = ctx) tokens, each with A(or E)+B+C | src/core/glm_multi.cu:787-827 (layout 745-782) | x(n-1) 0.55 GB | x(n-1) 0.92 GB | x(n-1) 2.0 GB |

`head_x`, the verify window (`win_alloc`, kWinMax=16, src/core/glm_window.cu:170), snapshots,
DFlash (fixed ~96 MB ring, the `dflash` estimate at glm_model.cu:1591) and the expert pool itself
are not ctx-sized. The reference path `load()` (SAGUARO_GLM_SLOW) sizes latent and ik/ig at the
**whole ctx in F32**, uncapped rings (glm_model.cu:757-781) - correctness path only.

### Pinned host RAM

| | allocation | file:line | 32768 | 262144 | 1048576 |
|---|---|---|---|---|---|
| M | **KV streaming latent cache** - the authoritative copy, one `cudaHostAlloc(Mapped|Portable)` of `n_blocks x page x row_bytes x 11` at load | glm_model.cu:907-918 | 0 (streaming off) | **2.95 GB** | **11.8 GB** |
| N | one more whole host copy **per extra SEQ** when streaming | glm_multi.cu:829-834 | - | x(n-1) 2.95 GB | x(n-1) 11.8 GB |

No other host pin in `src/core/` is ctx-sized (prompt staging, hop slots, rings and the expert
RAM tier are all chunk-, row- or RAM-sized; the tier budget is measured *after* these pins,
glm_fast_path.cu:1574-1589, which is why the run's RSS barely moved between contexts even as M
appeared). The last FAIL in `ref/mortar-run-20261126.txt` (qwen3.8-flash-next, "pinned RAM for the
K/V of every session at --max-context") is the **QFN engine** (`engines/qfn/src/program/generate.cpp:6199`),
not the GLM engine - out of scope here, but it is the same whole-max pinning pattern, one engine over.

## 2. What is already elastic, and how elastic-ctx relates

Already grow-with-use or capped:

* **C (ik/ig rings)** - fixed by cap `min(ctx, 8256)` on the fast path: constant since the ring
  only needs the prompt path's biggest sub-batch + slack (glm_model.cu:3480-3482).
* **G** - per request, not per context, *at use time* (`prefill_carve(Tn, end)` sizes the stage to
  the prompt's end, whole pages, glm_prefill.cu:1635-1642); only the startup sizing still assumes
  the 1M ceiling.
* **The expert pool** - when elastic-ctx is on (CUDA) the pool's main slots live in `ElPool`, one
  VMM reservation backed a granule per handle so a layer's top can be given up granule by granule
  (glm_multi.cu:124-183). Not ctx-sized itself, but it is the thing that "gives way".

Not elastic (whole amount allocated at load, on every platform unless noted): **B**, **F**, **M**,
**N**, **K** (on HIP), and **A** whenever streaming is off at ctx <= window (fine there: the whole
thing is what a 32K context costs).

The existing elastic-ctx (opt-in `SAGUARO_GLM_ELASTIC=1`, **CUDA only**) works exactly on B and A:

* `el_create` reserves the **whole context as address space** and backs only the fixed part
  (residual, rings, KDA states, snapshots); every DSA layer's latent cache and pooled keys are
  `Grow` regions with a bytes-per-position rate, backed from their start as positions are written
  (glm_multi.cu:392-495; rates at 403/440: the table's 1024 B and 128 B a position a layer). The
  VRAM cost at load is the fixed part (~250 MB) whatever `--max-context` says.
* Backing draws on a per-card budget (`el_budget_`, base `min(whole, 250 MB)`,
  glm_model.cu:3532-3539); when it runs short, `pool_give` unmaps the coldest layers' expert slots
  down to the route floor and adds them to the budget, `pool_take_back` returns the surplus when
  the engine idles (glm_multi.cu:234-323; the pool sizes keep the budget free at
  glm_fast_path.cu:1548-1551 and glm_prefill.cu:1203).
* Growth hooks: one position per decode token (glm_fast_path.cu:7562), one chunk per prompt
  (glm_prefill.cu:2115, whole prompt before the read at 4450); the serve loop reserves room before
  admitting/extending a request, releases idle contexts LRU-first, and trims to
  `SAGUARO_GLM_ELASTIC_KEEP` when idle (src/program/generate.cpp:3621-3694, idle_trim 3671-3685);
  `el_held`/`el_pool_mapped` keep the VRAM lines and context panel honest (glm_fast_path.cu:5300-5306,
  vram_fingerprint 5319-5334). SEQS under elastic: every extra conversation is a `back_fixed=false`
  reservation that pays only when a conversation is admitted (glm_multi.cu:804-813) - vs the
  static path's whole arenas (K).
* It is **mutually exclusive with KV streaming** - `--kv-resident` on prints
  "glm elastic: off with --kv-streaming" and keeps the static state (glm_model.cu:3518-3520) - and
  on **HIP the whole mechanism is stubbed out** ("glm elastic: not on HIP", glm_multi.cu:374-390,
  449-454, 499-503, 548-553, 630-659; the VMM entry-point table at 29-83 is the CUDA driver only).
  On the 9070 XT the *only* context-relief mechanism is therefore KV streaming (E + M), which is
  what the mortar run shows. The split-auto planner prices elastic vs static state with the same
  formulas used here (glm_model.cu:1595-1610).

## 3. Cost to make the rest grow with use, smallest first

Estimates assume one engineer who knows the engine, including the bookkeeping lines that must
stay honest (VramReport glm_fast_path.cu:5268-5317, the `INFO context=... kv_ram_bytes=` line
generate.cpp:1166-1171, the split-auto estimator glm_model.cu:1576-1610) and a parity check.

| allocation | what it would take | hours |
|---|---|---|
| H `sc_score`, J `ascore` | already tiny: size them per step from pools done (or cap at the sub-batch grid like I); mostly for tidiness | 1-2 |
| F residency maps | two-level page table (a leaf per 64K positions, allocated on first touch) or allocate maps for `resident + staged` and extend; touches only `kv_stream_make` + the kernels' page-table walk (include/maya/kernels/glm_kv_stream.hpp) | 3-5 |
| G/I prompt-path context ceilings | the per-request carve is already prompt-sized; size the startup layout and settle against `kv_resident + max chunk` instead of the whole ctx, re-settle (the smaller-chunk retry exists) when a longer prompt shows up; touches `prefill_bytes_for` default (glm_prefill.cu:1600-1606) and the fast_setup/prefill_setup chunk sizing | 4-8 |
| M pinned host latent cache (N follows) | replace the one `cudaHostAlloc` with an mmap plus `cudaHostRegister` per 64-256 MB granule on first write (the host arena's VA stays stable, the device reads unmapped-then-mapped pages as it writes them; nothing reads a position before it is written - the reset comment at glm_model.cu:971-972), un-register on reset/finish. Keep the WSL "the driver pins only about 1 GB" fallback. Check against `src/kernels/glm_kv_stream_parity.cpp` and the serve context-panel tests | 12-20 |
| B pooled index keys on HIP (and A at `--kv-streaming off`) | the CUDA answer exists (elastic-ctx); port `Vmm`/`Elastic`/`ElPool` to the HIP VMM API (`hipMemAddressReserve/Create/Map/SetAccess`) and un-stub glm_multi.cu's HIP branches; verify granule behaviour on gfx12; the elastic estimator, budget and serve loop then work on AMD unchanged | 16-30 |
| elastic-ctx *and* KV streaming together | make B (and, at `--kv-streaming off`, A) grow under streaming instead of refusing it (glm_model.cu:3518): the layout must interleave the slot runs with grow regions, the prompt stage and page tables size off the window, and serve's reserve/release has to account for the pinned copy as host budget. This is the "one mechanism grows everything from a small start" end-state | 20-40 |

After M + the HIP port, a 1M-context GLM start would hold only the fixed ~0.25 GB of context
memory beyond the resident window (state/KV growing at ~1.4 KB a position in VRAM + ~11.3 KB a
position pinned, both from zero), versus the ~13 GB (1.95 GiB VRAM + 11.8 GB pinned) allocated up
front today.

## What changed / files touched

Nothing but this report: `REPORT-p125c-glm-mem.md` (new). No code edited, per the plan
("change NO code; commit only the report").

## Tests run

None - the plan names no test modules and forbids code changes. The arithmetic in section 1 was
checked against the engine's own log figures in `ref/mortar-run-20261126.txt`
(state/KV 0.62/0.92 GB, KV streaming 0.37 GB VRAM slots, 2.95 GB pinned) rather than by running
the engine. Modules a future change would have to run: `src/kernels/glm_kv_stream_parity.cpp`,
`src/core/glm_model_test.cpp` / `glm_pack_test.cpp` (geometry reader), `serve/test_context_panel.py`,
`serve/test_dashboard.py` (both hard-code a K/V-bytes-per-position constant), `tools/mem_vs_ctx.py`.

## Open questions

1. The mortar run's "pinned MiB" column reads 0.0 for both contexts even though the engine printed
   2.95 GB pinned at 262144 - `tools/mem_vs_ctx.py:123` sums `VmLck+VmPin`, and ROCm's
   `hipHostMalloc` pages apparently show up in neither. GLM's PASS may be partly a measurement gap
   (its RSS stayed flat because the expert-RAM-tier budget adapts after the pin,
   glm_fast_path.cu:1574-1589); worth confirming on a CUDA box where VmPin does count.
2. The 1048576 figures are computed from the formulas, not measured (the run stopped at 262144);
   11.8 GB of one `cudaHostAlloc` will likely fail before the pool start on most boxes - does the
   "cannot pin ... lower the context" error (glm_model.cu:914-916) need a graceful degrade to
   VRAM-window-only mode?
3. `logs/maya/PLAN-parallel-conversations.md` is not in this checkout; if the Maya branches carry
   elastic-ctx notes beyond the code comments (e.g. intended HIP support), section 2's reading of
   intent should be re-checked against it.
4. Should elastic-ctx and KV streaming compose by capping B only (cheap, ~a day) or by the full
   unified design (last row of section 3)? The run suggests capping B first: on HIP it is the
   entire remaining VRAM growth with streaming on.
5. `--kv-resident`'s window (32768 default) is itself a fixed VRAM cost that does not shrink for
   short conversations; is it in scope to back the resident window's latent slots as written too,
   or is 0.37 GB flat acceptable as "the rule's small start"?
