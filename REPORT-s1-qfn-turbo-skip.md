# REPORT: S1 — QFN makes the live Turbo `skip` work (engines/qfn)

Plan: ./PLAN-s1-qfn-turbo-skip.md. Branch `saguaro/q1-s1-qfn-turbo-skip`, one commit per numbered step:

| step | commit | what |
|---|---|---|
| 1 | 8412dd12b | ./scratch/notes.md: every decode route path, the residency table's state on it, and where the RAM-tier fact lives |
| 2 | 34bea2c04 | skip=ram: the fused 512x10 router and `turbo_drop_apply` leave out picks that are not resident in VRAM; `SessionState::turbo_skip` rides the window's mapped upload |
| 3 | 99cbd9f79 | skip=ssd: a device mirror of the resident RAM mode's compact copy; the arena says "no-op", plain mmap / page cache says "no reliable fact" and runs exact |
| 4 | 7cfb2cd32 | skipped picks keep the tiers learning: the router counts them into a per-card usage table, folded into `drive.d.usage` at every adaptive round |
| 5 | 96b818b84 | `turbo drop=N skip=X: P picks, D dropped, S skipped`, one line per request (solo DONE, every BDONE); the old "skip is off" line is gone |
| 6 | 7e5b8a73b | `turbo_skip_parity` (ctest): off-bitwise, all-resident-bitwise, none-resident = top-1, synthetic residency vs a host replay; 512x10 and 256x8; counters and usage |
| fix | b559abcdd | bounds-check the counter bucket in the kernel |

No commit touched `serve/qfn_engine.py`, `engines/qfn/serve/server.py` (the Python path
already forwards `turbo_drop`/`turbo_skip` on the GEN line) or any other engine. `tools/saguaro/extract_qfn_engine.py`
did not need to run (nothing extracted changed).

## What changed

**Semantics (step 2/3).** After the pick and the drop, a row whose `turbo_skip` bar rejects a kept pick writes it
as the row's top-1 id at weight 0 and renormalises the kept sum exactly as the drop does (clamp 2^-14). Bar 1
(`ram`): not resident in VRAM (the window's device residency row, `VerifyHits::d_res`, `slot or -1`). Bar 2
(`ssd`): neither there nor in the RAM tier (`VerifyHits::d_ram`, the mirror of `FileExpertSource::has_resident`).
The row's top-1 always stays — it is fetched if it must be. A left-out pick is a repeated top-1 at weight 0, so
every existing drop guarantee holds for it unchanged: not planned (pool `dropped()`, `resident_plan`'s duplicate
check), not fetched (`copy_rows_from_mapped` id-pairing), not published over PCIe (`doorbell_publish_res`), not
computed (the combine weights it 0), and now also counted for usage (step 4). No new kernel launch and no new
upload: `skip` and the counter tag ride the window's existing mapped position copy (two more ints per row, at
`pos_[MT*(NH+NKV+IQ+3/4/5)]`).

**Files.** `native_router.hpp/.cu` (the `TurboRoute` bundle: drop/skip/res/ram/tag/stats/skip_usage/layer;
`route`, `route_multi` and `turbo_drop_k` fold the bar in; the 512x10 fused kernel stays one launch);
`session.hpp` (`turbo_skip`); `layer.hpp/.cpp` (`moe_route` takes a `TurboRoute`); `verify.hpp/.cpp`
(`VerifyHits::d_ram/d_skip_usage/turbo_stats`, the +6 upload, `record_window` builds the per-layer bundle from
the same residency row the E-6 plan reads, `stage_inputs`/`stage_batch` write the rows' drop/skip/tag);
`generate.cpp` (`turbo_skip_of` and its said-once refusals, the per-stage RAM mirror + `ram_upload`, the skip
usage table + `fold_skip_usage`, the counter buckets + `turbo_take`, the per-request lines, `slot_turbo`/
`SoloTurbo` set both options); `turbo_skip_parity.cpp` + CMake (ctest).

## Per-path table

| decode route path | skip=ram | skip=ssd | notes |
|---|---|---|---|
| solo verify windows (`run`, serial) | works | works | tables patched by the tier's own `res_upload`/`ram_upload` before the window |
| batch windows (`run_slots`/`batch_launch*`) | works | works | per-row tables: each slot routes with its own request's bar |
| pipelined windows (`--pipeline-windows`) | works, table may lag ≤1 window | works, same lag | these verifiers always publish and the residency table follows at the round/`res_upload` points rather than per launch; a lag only mis-skips (under-skip is the old behaviour, over-skip hits an expert the host already marked evicted — what the engine's own plan path already treats as gone) |
| multi-GPU layer-split stages | works | works | each stage routes by its own card's residency row and its own RAM mirror ("the expert cache on that card", as GLM's bar is) |
| gfx906 batch router (`moe_route_window`) | works (apply after the window router) | works | one `turbo_drop_apply` launch, already there for the drop |
| token graph / hit path (`moe_hit_select`) | exact — unchanged | unchanged | the CLI single-token loop and the prompt/hit path: no request line reaches it (the drop never did either). Its `d_res` is the same live table |
| prompt reads (Prefill windows) | exact | exact | Prefill routes outside the Verifier and never reads the tables (unchanged); `slot_turbo(b,0,0)` at read/admission holds the reply's session bars at 0 until the first reply token |
| MTP draft layer route | untouched | untouched | as the plan asked; drafts stay the drafter's, the (skipping) target verifies |
| 256-expert Coder (src/qwen35 engine) | exact | exact | that engine parses no turbo keys at all today (the drop applies there neither); out of scope, documented in scratch/notes.md |
| layouts | — | arena (`--resident-experts` class): **no-op, said once** — every expert byte is in RAM. Resident RAM mode with the full complement: mirrored, the bar skips nothing while the copy holds the GPU's complement — correct without a message. Budgeted (`--resident-budget-gib`) or RAM-adaptive (`SAGUARO_RAM_ADAPT`): mirrored and real (the copy can hold a subset and move). Plain mmap / GGUF-in-place / page cache: **not guessed, said once, runs exact**. `--no-pool`: said-exact as well (no bar without a table). With SAGUARO_ROUTE_PRIOR / SAGUARO_ROUTE_RESIDENT: drop and skip are both said-off (the re-pick would undo them) |

## Exactness with skip off

- skip absent/off: the row's `tskip` int is 0; the kernels branch `bar != 0 && ...` — residency is not read, no
  atomic (the counter requires a nonzero drop/skip in the row), and `alive == (rank < keep)` is the old drop
  predicate. Selected-sum accumulation, ids and weights are the identical expression, in the identical order; the
  fused launch's arguments differ only by pointer values. `native_multi_parity` (no table) and
  `turbo_skip_parity`'s zero-table case compare bitwise by memcmp, fused and apply, ids and weights.
- The upload is the same `copy_i32_from_mapped` (two ints wider; the mapped arena was grown in `init`).
- The pool, plan, doorbell and combine code paths: untouched.
- With the whole engine idle of turbo, `d_ram`/`d_skip_usage`/`turbo_stats` are written only by the tier's own
  `ram_upload` (version-checked: a no-op unless the RAM copy moved) — and the counters are only touched by
  turbo-on rows.

## How to measure on a card

1. A card that misses experts: QFN pack, `--expert-cache` sized below the expert set (e.g. 12–16 GB), with
   `--resident-experts` (or `--resident-budget-gib N` for a *working* ssd bar). Serve, `--spec`/MTP as usual.
2. Chat with Turbo skip=ram: watch the engine log line `turbo drop=0 skip=ram: P picks, D dropped, S skipped` —
   S > 0 and rising over the request is the feature working; the decode `tok/s` on the DONE line should rise and
   the "expert cache hit rate" line with it (the skipped misses never fetch or compute).
3. skip=ssd: on the full complement expect the once-said "no-op" line and S = 0; on a budgeted complement expect
   S > 0 only while hot experts sit outside VRAM *and* the RAM copy; on plain mmap expect the once-said
   "no reliable RAM-tier fact ... runs exact" line.
4. Learning (step 4): after a skipping chat, the next rounds should promote the hot skipped experts — watch the
   adaptive tier's swap counts (`swaps=` on the timing lines) with skip on vs off; hot experts should arrive in
   VRAM outside the skipping request's own windows (they are counted, just not fetched).
5. Regression: the ctest modules `native_multi_parity`, `router_top10_parity`, `route_resident_parity` and the new
   `turbo_skip_parity`; a live A/B of the same prompt with turbo off before/after this change must match token
   for token (the skip-off guarantee), and the server's `serve/test_turbo_live.py` for the wire half.

## Tests run in this session (no GPU on this box; the plan: no GPU runs)

- `nvcc -c engines/qfn/src/kernels/cuda/native_router.cu` — compiles (step 2 shape, re-checked after steps 4/5/fix).
- `g++ -fsyntax-only` (C++20, STRATA_VERSION defined, CUDA headers): `generate.cpp`, `verify.cpp`, `layer.cpp`,
  `mtp.cpp`, `session.cpp`, `turbo_skip_parity.cpp` — all clean; `route_resident_parity.cu` compiles (nvcc);
  `router_chain_bench.cpp` clean.
- `turbo_skip_parity` is registered in ctest but **was not run** (it needs a GPU; "unit tests only if they need
  no GPU"). No Python suite applies to these files (no serve/ or tools/ file changed).

## Risks

- **Residency lag** on the pipelined/NOWAIT paths: a skip decision can use a table up to one window/round old.
  Under-skip is benign; over-skip only hits experts the host already evicted (their bytes may still be valid in
  the slot — running them would be exact, skipping them is what the bar says). Same convention as the E-6 device
  plan; not silent — the counters report it (S higher than a fresh table would give).
- **ssd mirror windows**: soft-committed exchanges (SAGUARO_ADAPT_OVERLAP) show the incoming expert as
  not-yet-RAM for one round, and staged `out`s as not-in-RAM while their bytes sit in an exchange buffer: brief
  over-skip only. The mirror re-uploads at every round/upload point plus request starts and slot admissions;
  a missed point can stale it by one round — bounded, and it never plans or fetches for a skipped pick.
- **MTP acceptance** drops when the target skips what the (unsuppressed) drafter drafted for: fewer accepted
  tokens per window can eat the PCIe/CPU savings on miss-heavy chats. This is a speed question, not a correctness
  one; the counters + tok/s lines show it. (Leaving the drafter's own route untouched was the plan's instruction.)
- **Duplicate compute of aliased rows** on the device hit path (the top-1 is computed twice, weight-0 row
  included) — the drop already had this; skip=ram with many misses makes the top-1's extra hit-row visible.
- `fold_skip_usage`'s D2H can race an in-flight pipelined window's atomics (increments from that window are
  folded one round later — nothing corrupts).
- Counter buckets assume batch ≤ 15 slots (kVerifyMaxT is 8 today); bounded out-of-range tags are ignored
  (b559abcdd).

## Open questions

- Should the router lookahead/warm (`RouterLookahead::submit`) also see the original skipped ids (it reads the
  post-rewrite ids), so a skipping request pre-reads the hot experts it asked for? The adaptive tier learns them
  (step 4); this would make the bring-in faster but adds an id list to the doorbell.
- Should `skip=ram` treat the *page cache* as RAM opportunistically (`tier_of` samples four pages)? The plan
  says do not guess; kept exact, said once.
- Should drop-cut picks also count as usage the way skip-cut ones now do (step 4 asked only for skip; drop's
  today-not-counted rule was left as shipped with the drop)?
- Live measurement (this is a no-GPU change audit): the tok/s and acceptance numbers on a 12–16 GB card with the
  resident RAM mode will say whether the ssd mirror's upload points need a pre-launch hook too (they are set for
  NOWAIT/LAG modes only today).
