# S1 notes - every QFN decode route path, and the residency facts on it (step 1)

Repo-relative paths: the QFN engine is `engines/qfn/` (the plan's bare `src/...` / `include/...` paths read
against it). Line numbers are the worktree at e6d840db2.

## A. Where the ROUTER runs per decode path

The request's Turbo lives in `SessionState::turbo_drop` (include/strata/core/session.hpp:68) and rides the
window's mapped position upload as one int per row (`td[]`, verify.cpp stage_inputs ~2758, stage_batch ~3893;
uploaded with pos_ in record_window ~1199). The router kernels take the per-row table
(`route`/`route_multi`/`turbo_drop_k`, src/kernels/cuda/native_router.cu). With `drop = 0` / no table the
arithmetic is today's, bit for bit (08d672eda).

| # | decode route path | where it routes | residency table on device? | current at route time? |
|---|---|---|---|---|
| 1 | **solo verify window** (serve reply loop, serial: `ver.run`, generate.cpp ~19841/~22003; also the prompt-tail windows and reads run through windows with the session's turbo set to 0) | `Verifier::record_window` native branch verify.cpp:1959-1993 (512x10 fused) / per-token `moe_route` loop for any other (n_expert, k) | yes: `hits_.d_res + l*n_expert` (VerifyHits::d_res, verify.hpp:71; thits.d_res at generate.cpp:9728, the "token graph hit path" table) | yes: the pre-launch hook `res_sync` (set via `set_pre_launch`, run at verify.cpp:2920) patches d_res from `host_res` before every `run()` window |
| 2 | **batch window** (`run_slots`/`run_slot_rows` ~14124, `batch_launch`/`batch_launch_rows` ~15180): row t = slot `rows[t]`, each slot its own request | same `record_window`, per-row table read per row | yes, same table | yes: `pre_launch_` also runs at verify.cpp:3972/4151 (batch launches) |
| 3 | **pipelined windows** (`--pipeline-windows 2`, pl_launch/pl_stage ~21550; stage_ver_p per stage and parity) | same `record_window` on each stage's verifier; `pl_stage` -> `stage_inputs` (verify.cpp:4652), so the per-row table rides | yes (per stage: `stages[k]->d_res`, verify.cpp ~9887/9944 `vs.d_res = gs.d_res`) | LAGS: these verifiers run `set_always_publish(true)` and the table is not synced before every launch (verify.hpp:306-313: with windows in flight the device table may lag the host's evictions). Under-skip (stale resident) = the request's own loss; over-skip of a mid-eviction expert = what the engine already treats as non-resident. No launch is refused, but it is not exact per window. |
| 4 | **multi-GPU layer split stages** (each stage runs `record_window` for its layers with ITS card's table) | same `record_window` | yes, one per stage | as path 1/2/3 for the driving mode (serial: pre-launch; pipelined: lags) |
| 5 | **token graph / hit path** (`session_capture_token`/`session_run_token`, session.cpp:970-1049; `moe_hit_select` reads `hits->d_res` to split hits/misses ON THE DEVICE) | `block_layer_pre` -> `moe_route(drop=nullptr)`, session.cpp:940-1003 | yes (`thits.d_res`, allocate+fill generate.cpp:8104-8124, kept current by `res_patch`/`res_put` 11385/22814/23258) | yes for the hit split - BUT this path is the CLI single-token loop (generate.cpp:22982, `--no-capture`/tgraph) and the prompt path: **no request line, no Turbo exists there**. Nothing to apply; it stays exact. |
| 6 | **prompt path** (session_token / prompt windows `sp.run` 16412/19948/2019948) | `moe_route` with drop=nullptr / windows staged while the session's turbo is 0 | yes | N/A - prompt reads stay exact by construction (c03451d13: the drop starts with the reply) |
| 7 | **MTP draft layer route** (mtp.cpp record_rest ~1178; `native_route_resident` only with SAGUARO_ROUTE_RESIDENT) | its own single-token route | `rr_res_` = `thits.d_res + (n_layers-1)*n_expert` (generate.cpp:9737) when SAGUARO_ROUTE_RESIDENT | N/A - plan: leave the drafter's own route as it is. Drafts are verified by the (skipping) target, so output stays what the target's routing says. |
| 8 | **256-expert Coder (Qwen3.6-35B-A3B)**: a SEPARATE engine (`strata::qwen35::run`, q36_engine.cu; `native_router_topk` 256/8) | its own verify path (q36_engine.cu:1812/2015) | no d_res equivalent: it keeps its experts resident on the device (no VRAM cache tier at all) | N/A - the engine parses neither `turbo_drop` nor `turbo_skip` TODAY (drop was never applied there either, 08d672eda/c03451d13 touched only the main engine). Runs exact; out of this change's scope. |
| 9 | **gfx906 batch router** (verify.cpp:1971-1978, `moe_route_window` + `turbo_drop_apply`) | same `record_window` | yes | yes (serial hook) |

Slots being swapped in or lent while a window routes:

* **adaptive VRAM tier** (drive.d.usage -> `adapt()`): with SAGUARO_ADAPT_NOWAIT/LAG the HOST marks evicted
  experts -1 at once and the device table follows at the next pre-launch patch (res_sync, generate.cpp:11495;
  the copy's bytes land a window later - the engine's own convention already treats host -1 as "not resident").
* **elastic cache** (--vram-live / VRECLAIM): drops slots -> host_res -1 -> same patch channel.
* **prompt loan** (pf_borrow, generate.cpp:9239/22756): the TAIL slots' experts are marked non-resident for the
  prompt and restored after it; a loan is only up during prompt windows, which route exact (path 6).
* **RAM-tier swaps** (SAGUARO_RAM_ADAPT / RAM_ADAPT2): do not touch d_res at all - they only move the host
  complement's contents (below).

## B. Where the RAM-tier fact lives

* `FileExpertSource` (src/core/expert_source.cpp): the resident RAM mode's **compact copy** ("complement").
  `has_resident(layer, expert)` / `resident_blob(index)` (expert_source.hpp:781/816) read
  `complement_offsets_` (kNoComplement = not held); with SAGUARO_EXCHANGE_ROTATE the ownership table decides
  through `resident_blob`. `complement_ready()` says the copy exists at all.
  Membership changes ONLY through `detail::exchange_cache_complement` swaps, at four points:
  `commit_exchanges` (~4314), `commit_flip` (~4513, with the rotate arm at ~4506), `install_resident` (~4363,
  from `replace_resident` and `ram_install`). All run on the decode thread between windows
  (`commit_copies` runs on the async tier's thread but only memcpy's; the membership flips in `commit_flip`).
  Full mode (`pin_cache_complement` with `budget_bytes = 0`, no lend pairs dropped): the copy holds EXACTLY the
  experts the GPU does not and keeps doing so across exchanges - everything is always in VRAM or RAM.
  Budget mode (`--resident-budget-gib`): the copy holds the ranked subset; the rest is the file tier (SSD).
  RAM-adaptive mode (SAGUARO_RAM_ADAPT[2]) moves that subset between windows (`replace_resident`,
  `ram_stage`/`ram_install`).
* `ArenaExpertSource` (`arena_src`, generate.cpp:6364, `srcp = &arena_src` at 6622): the whole model's expert
  bytes live in the RAM arena. Every expert is in RAM; the files are only read at load. skip=ssd is a no-op here.
* **plain mmap / page cache** (FileExpertSource without the complement: experts.bin or the GGUF shards in place,
  `gguf_mode()`): whether a blob's pages are in the page cache is sampled guesswork (`tier_of` reads four
  pages; `cached_share`); the plan says do NOT guess -> skip=ssd runs exact and says so once.
* `--no-pool` (GPU-only floor): no CPU tier reads at all; misses go straight to the file reads - same
  no-fact case as plain mmap.
* `xcache` (ExpertCache) is the VRAM tier; its truth is `host_res` / `d_res` (`slot or -1`), NOT has_resident.

## C. What skip must mean here (from the plan and the GLM reference, src/core/glm_fast_path.cu:7150-7169)

* `ram`: leave out every routed expert with `d_res < 0` (not in VRAM). The engine's own bar = "is a cache miss".
* `ssd`: leave out every expert with `d_res < 0` AND not held by the RAM tier. Needs the RAM fact on the device.
* a left-out pick = the row's top-1 at weight 0, the kept weights renormalised as drop does; the row's top-1
  always stays (fetched if non-resident); nothing left out may be planned / fetched / promoted / computed -
  which all downstream drop machinery already guarantees for a repeated top-1 at weight 0:
  pool (`dropped()`, expert_source.cpp:4826-4861), device plan (`resident_plan` duplicate check,
  verify_kernels.cu:1306), doorbell (`doorbell_publish_res`, elementwise.cu:252), miss-row copy
  (`copy_rows_from_mapped(..., ids, k)`, verify.cpp ~2183).
* residency must not freeze: the pool counts usage per pick, and deliberately does NOT count the aliased
  repeats (expert_source.cpp:4839-4843) - a skipped expert would never be learned. The skipped ids must be
  counted where the rewrite happens (the router, device-side), and folded into `drive.d.usage` before the
  adaptive rounds consume it (batch round ~14159, solo round ~21338, async snapshot ~12003, RAM-adapt scans
  ~21888/~21952).
* counters: P picks / D dropped / S skipped per request; S is unknowable host-side after the rewrite (the ids
  are gone), so the router kernels count into device u64 triples indexed by a per-row tag (0 = solo, 1+slot for
  batch rows), read at the request's end from every device.
