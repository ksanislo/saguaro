# T6a report: QFN engine stats lines (engine C++ only)

Task 6 (part a) of the QFN-first project. All edits are host-side `printf` lines
and the host-side counters they read, inside `engines/qfn/src/program/generate.cpp`.
No kernel, arithmetic, GPU-order or token change. Three commits' worth of work; two
code commits (steps 1 and 2) plus this report. Step 3 was a deliberate no-op (below).

## What changed

### Step 1 - `mtp=` in the INFO line (commit `65f5d8134`)
The server's `speculation_modes` (`serve/server.py:2633`) lists a draft from the engine's
INFO `mtp=` key (`info.get("mtp")`, on when `float(v) > 0`). QFN printed `mtp_max=` (the
window cap) but no `mtp=`, so the Monitor's Speculation never showed that a QFN engine
drafts with MTP. Added `mtp=%d` next to `mtp_max=` on the main INFO line, printed as
`use_mtp ? 1 : 0` - the same condition the maya / GLM engine uses (`o.mtp.empty() ? 0 : 1`);
`use_mtp` is `!o.mtp.empty()` (`generate.cpp:9069`), i.e. on when `--mtp` loaded the drafter.

- Printed at `generate.cpp:12390` (the main `INFO logit_bias=1 ...` line); the `mtp=%d`
  field is on the format string at `12393`, its argument at `12407`.
- The `engine_kind=qwen` INFO line at `~12432` (which another task edits) was left
  untouched - the key went on the *other* INFO line to avoid that conflict. The server
  parses every INFO line into one dict, so `mtp` is picked up regardless of which line
  carries it.

Example (illustrative rendering; the engine was not run):
```
INFO logit_bias=1 context=262144 kv=q8_0 ... spec=5 mtp_max=5 mtp=1 lookup=64 vram_free_mib=... ... engine=0.1.49.2
```

### Step 2 - STAT in solo decode, and `promo=` (commit `811f7a93d`)

**(a) STAT in the solo path.** QFN printed its `STAT` line only from batch windows
(`batch_stat_line`, callers at `~14230 / ~14900 / ~14955`), so a solo (parallel 1) engine
showed no "where the experts live" tier card. The solo serial decode loop now feeds this
window's emitted tokens and accepted/offered drafts into the same `bstat_*` counters and
calls the **same** `batch_stat_line(1)` at its own cadence (the function self-throttles to
once a second and self-resets, matching the batch path exactly):

- baseline reset per solo request at `generate.cpp:21740` (`if (o.batch == 0) bstat_reset();`),
  as a batch request's is when its slot begins.
- per-window counters + emit at `22044-22053`: `bstat_toks += produced_n - produced_n0`
  (`produced_n0` captured at `22035`); for non-merged windows `bstat_off += T - 1` and
  `bstat_acc += a` (drafts offered / accepted, matching the DONE line and the maya engine's
  `spec_steps_` / `spec_hits_`).
- Guarded on `o.batch == 0`: a batch engine drives those counters from `batch_step`, so the
  two paths never share them.

**(b) `promo=` field.** `STAT` had no `promo` field, so the Monitor's "Promoted" tile was
blank. Added `promo=%.3f` (`generate.cpp:12969`), the experts the adaptive VRAM tier
promoted into VRAM since the last line, per token: `(promo_now() - bstat_promo0) / n`.
`promo_now()` (`12928`) returns `ajob ? a_swapped : adapt_nswaps` - the asynchronous tier's
cumulative swap counter under `--adapt-async`, else the blocking tier's - the same pick the
decode timing line already makes (`dt_sw0`, `generate.cpp:21062`). The baseline
`bstat_promo0` (`12932`) is refreshed in `bstat_reset`. A per-token *rate* (not a raw count)
matches the maya engine (`promo=%.3f`, `src/program/generate.cpp:2907`) and the UI's `/tok`
unit (`serve/web/monitor.js`). It reads existing host-side counters only; nothing in the
cache class or GPU path was touched (`expert_cache.hpp` lives under `include/`, off-limits).
Adding it to `batch_stat_line` also fills the Promoted tile for the batch path, which had
the same gap.

Example (illustrative rendering; the engine was not run):
```
STAT tok_s=21.40 ms_tok=46.73 vram_hit=0.9512 ram_fetch=0.487 disk=0.000 promo=0.031 vram_used=4096 vram_slots=5120 ram_used=113664 ram_gb=71.2 vram_gb=2.4 disk_gb=0.0 draft_acc=37 draft_off=58 active=1
```

### Step 3 - logprobs INFO key: NOT added (report only)
Per the plan ("if it already can return logprobs, print the key; if it can't, don't add the
feature, just report"), I grepped `engines/qfn`:
- No `logprobs=` request-key parsing in the GEN handler, and **no `LP` line emission** - the
  server protocol for logprobs is an `LP` line per generated token after a request carries
  `logprobs=<n>` (`serve/server.py:1673`). QFN writes none.
- The QFN server driver emits only empty logprobs: `logprobs=[]` / `"logprobs": []`
  (`engines/qfn/serve/responses.py:345,512,531,576,616`), and nothing fills them.
- QFN *does* compute log-probabilities internally - `Verifier::window_logprobs`
  (`src/core/verify.cpp:3361`) - but only for the `SAGUARO_LOGPOS=<path>` bench/scoring dump
  to a file (`generate.cpp:19839`); it is not wired to the request protocol.

So QFN cannot return logprobs to the server, and `INFO logprobs=1` must **not** be added:
`serve/server.py:2957` states an engine that advertises them but "would answer without them,
which a client that asked could not tell from a bug", and `2963` documents QFN explicitly as
the no-logprobs case (the server strips `logprobs`/`top_logprobs` from its requests). No code
change; nothing to commit for step 3.

## Why nothing on the GPU path changed
Every edit is a host-side `printf` and the host-side counters it reads:
- Step 1 reads `use_mtp` (a host bool) - no kernel, no arithmetic.
- Step 2 (a) reads `produced_n` (tokens already emitted), `T`, `a` (window width / accepted
  drafts) and writes host ints `bstat_*`; it calls an existing reporting function.
- Step 2 (b) reads the adaptive tier's existing cumulative counters (`a_swapped` /
  `adapt_nswaps`). No cache class, residency table, copy order or kernel launch changed.
The verify window still confirms every emitted token against the real model; the tokens QFN
produces are bit-for-bit what they were before. The `engine_kind=qwen` token at `~12432`
(another task's) was not touched.

## Files changed
- `engines/qfn/src/program/generate.cpp` - the only source file touched (`+26 / -3`).

## Tests / build run and results
The plan names one build (the engine was not run). No test modules were named by the plan for
this engine-C++-only task, so none were run; the build is the check.

1. First configure exactly as the plan wrote - `cmake -S . -B build-t6
   -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc -DSAGUARO_QFN=ON` - succeeded but
   created **no `saguaro-qfn` target**: that target is gated on `SAGUARO_QFN AND
   MAYA_ENABLE_CUDA` (`CMakeLists.txt:1891,1938`) and `MAYA_ENABLE_CUDA` defaults OFF, so the
   CUDA toolchain was never enabled (the `CMAKE_CUDA_COMPILER not used` warning). To get "the
   QFN engine's normal CUDA build" the plan asks for, I reconfigured adding
   **`-DMAYA_ENABLE_CUDA=ON`** (keeping the given flags). This is the one deviation from the
   plan's literal command; without it there is no target to build.
2. The `qfn_engine` external project then failed to *configure*: it needs llama.cpp's ggml at
   the pinned commit `third_party/llama.cpp/ggml` (`engines/qfn/CMakeLists.txt:1794`), which
   this worktree had not provisioned (`third_party/llama.cpp` absent; only `ggml-common.h` is
   vendored). I provisioned it the way `setup.py:get_llama_cpp` does - the pinned zip
   `3cf03257f...` extracted to `third_party/llama.cpp` (git-ignored, `.gitignore:32`; the
   commit matches the engine's own FetchContent pin). `git status` stays clean.
3. `cmake --build build-t6 --target saguaro-qfn -j 16` - **succeeded**. `generate.cpp`
   (step 171/178) compiled with no errors and no new warnings; the only `generate.cpp` warning
   is a pre-existing unused-variable at `15196` (untouched), and there are no `-Wformat`
   issues, so the new printf fields match their arguments. `strings` on the binary confirms
   both new format strings are linked in (`... mtp_max=%d mtp=%d lookup=%d ...` and
   `... disk=0.000 promo=%.3f ...`).

Last lines of the build:
```
[174/178] Linking CXX static library libstrata_kernels.a
[175/178] Linking CXX static library libstrata_engine.a
[176/178] Linking CXX static library libstrata_prefill.a
[177/178] Linking CXX static library libstrata_qwen35.a
[178/178] Linking CXX executable saguaro-qfn
[ 50%] No install step for 'qfn_engine'
[100%] Completed 'qfn_engine'
[100%] Built target qfn_engine
[100%] Built target saguaro-qfn
```

## Open questions / notes
- **Build-command flag.** The plan's build line omits `-DMAYA_ENABLE_CUDA=ON`; on a clean tree
  that leaves no `saguaro-qfn` target. Either the plan assumes an env where CUDA defaults on,
  or the flag belongs in the build recipe. Confirm which, so the documented one-liner works.
- **Provisioning.** A fresh worktree has no `third_party/llama.cpp` until `setup.py` runs; the
  engine build depends on it. Worth a precheck (`ggml/CMakeLists.txt` present) before the build.
- **Pipelined solo decode not covered.** The STAT emit is in the default serial loop. A solo
  engine run with `--pipeline-windows 2` (multi-GPU overlap, `generate.cpp:~21369`) uses the
  separate pipeline loop and still prints no STAT line. Left out of scope (the plan names the
  solo decode loop; the pipeline path is rare in parallel-1). Add it there if coverage is wanted.
- **`promo` source is the adaptive tier.** QFN has no counter literally named "promotions" for
  a normal run (the only `grep promot` hit is the opt-in `SAGUARO_RAM_ADAPT2` prefetch's
  `ram_adapt2.installed`). `a_swapped` / `adapt_nswaps` (the tier's experts swapped into VRAM)
  are the faithful host-side stand-ins and match the tile's "experts moved into VRAM". With no
  adaptive tier (`--adapt-every 0` / static residency) `promo=0.000`, which is correct.
- **`draft_off` meaning.** Solo uses drafts offered (`T-1`), matching maya's `spec_steps`; the
  batch path's existing `+= batch_mtp ? 1 : 0` counts windows-with-a-proposal. Left batch as-is.
