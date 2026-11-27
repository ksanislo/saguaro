# P125g - GLM-5.3-Flash: the pinned host latent cache grows with use (REPORT)

Item M of `ref/REPORT-p125c-glm-mem.md`: the KV-streaming latent cache was ONE `cudaHostAlloc`
sized by `--max-context` (2.95 GB pinned at 262144, 11.8 GB at 1048576), allocated at load whether
or not a conversation ever wrote a byte of it. It is now a **granule arena**: the whole range is
reserved as address space (`mmap` / `VirtualAlloc MEM_RESERVE`), and a 64-256 MB granule is
`cudaHostRegister`'ed (`hipHostRegister` through the HIP shim) the first time a position of it is
named, and unregistered + `MADV_DONTNEED`'ed (`MEM_DECOMMIT`) when its conversation resets, parks
or ends. The arena's addresses never move, so the layers' host-copy rows are still one pointer
each, valid across pin / give-back / re-pin. The bookkeeping lines keep the report's honesty: the
load line names what is pinned NOW (0 at the start) and the ceiling, `INFO context= ...
kv_ram_bytes=`, the VRAM lines' `kv_ram`/`kv_ram_room`, `vram_fingerprint` and `kv_stream_stats`'
`host_gb` all read the arena's current size.

## What changed (commits, oldest first)

* **8d2002089 - step 1, the granule arena as a unit**: `include/maya/core/granule_arena.hpp`
  (new) - `GranuleArena` + the `Pinner` seam (the driver is the caller's: the engine passes
  `cudaHostRegister`/`cudaHostUnregister`, the test a recording fake), with
  `tests/granule_arena_test.cpp` (new, 41 checks; host code, no GPU) and its CMake target
  (`granule_arena_test`, registered with `add_test`). Thread-safe; granules rounded to whole
  4 KB pages; a granule is zero-filled and backed before `pin` (a reader of an unwritten position
  still reads zeros, as the `cudaHostAlloc` arena gave); a refused pin un-backs that granule and
  leaves the earlier pins of the same call in place.
* **82851897c - step 2, the latent cache on it**: `kv_stream_make` (src/core/glm_model.cu) opens
  the arena over `host_bytes` instead of one `cudaHostAlloc`; the granule is
  `SAGUARO_GLM_KV_PIN_GRANULE_MB` MB, clamped to the report's 64-256 MB range (default 128; a
  cache smaller than one granule is one granule). `Glm5Model::kv_pin(K, end_pos)` registers every
  granule below `end_pos` (whole pages - a resolve reads whole blocks - plus `2*kWinMax + 2 +
  page` of slack for the verify window and the draft chain), per DSA slab. It sits beside every
  elastic grow hook: `fast_layers` and `fast_mtp` (src/core/glm_fast_path.cu:7564, 7926), the MTP
  cache rows (glm_mtp.cu:366, glm_mtp_rows.cu:841), the prefill chunk and the whole-prompt pass
  (glm_prefill.cu:2116, 4458), `slot_load` (glm_fast_path.cu:10033, 10346), and the rows' bind
  (glm_rows.cu:254). A device that cannot use the host pointer of registered memory (or where the
  reservation fails) keeps the old whole-max `cudaHostAlloc` (the eager fallback; `pin_slabs`
  stays empty, `kv_pin` is a no-op). Under the WSL ~1 GB pin cap a conversation now stops at the
  refusal (`cannot grow the pinned latent cache ...`) instead of failing the whole load.
* **ef5ef9b57 - step 3, the give-back paths**: `reset()` unpins the bound stream's granules after
  its stream sync (src/core/glm_model.cu:855, both the fast and the legacy branch);
  `seq_release(s)` unpins a released sequence's stream on **every platform** - CUDA and HIP alike
  - behind a card sync first (src/core/glm_multi.cu:631; HIP has no elastic VRAM to release, but
  since this change the host copy grows there too, so it must shrink there too); a parked
  conversation gives its pinned copy back from both slot_put paths - after the RAM slot takes it
  (src/program/generate.cpp:1634) and after the disk write (1689) - through
  `Glm5Model::kv_pinned_release()`, which syncs each part and skips a part whose sync failed;
  `slot_load` registers the restored positions' granules before copying rows into them
  (glm_fast_path.cu:10346), as the file slot's read path does.
* **d273a4849 - step 4, the lines stay honest**: the KV streaming start prints
  `pinned latent cache grows with use: N MiB of pinned RAM at start, up to M GB` - N is
  `pinned_now()` (0 for a fresh arena, the whole cache for the eager fallback;
  glm_model.cu:1009-1014), M is the ceiling `host_bytes`. `KvStream::pinned_now()`
  (src/core/glm_fast_state.hpp:1058) is the one reader of "now": the `INFO context= ...
  kv_ram_bytes=` line (src/program/generate.cpp:1171 via `kv_host_bytes()`), the VRAM lines'
  `kv_ram` and `kv_ram_room` (glm_fast_path.cu:5292, re-printed lines follow the growth and the
  give-back), `vram_fingerprint` (5329) and `kv_stream_stats`' `host_gb` (1105). serve's Context
  card already documents `kv_ram_bytes` as "the pinned host K/V it holds now ... 0 when allocated
  lazily" (serve/server.py:3796), so the panel tracks the growth without a serve change. The
  split-auto estimator (glm_model.cu:1704-1738) prices only the VRAM state (slots at the window,
  pools, rings) and never did price the host pin, so nothing there changed; the expert RAM tier's
  budget is measured from `MemAvailable` after the pins (glm_fast_path.cu:1584), so it simply
  sees the RAM back.
* **71c7b1058 - build fix**: the sm_75 build failed on the two rows call sites - `RowSeq::kv` is
  a `const KvStream*` (src/core/glm_rows_state.hpp:68: the rows only read the stream's page
  tables and host rows). `kv_pin` now takes a `const KvStream*` and const-casts the arena:
  pinning is the arena's mutex-guarded bookkeeping, not a write to the stream's data.
* **d6123587b + dc42c8723 - step 5, the card-test script**: `tools/card_tests/glm_pinned_start.sh`
  (new; see Files). `bash -n` clean.

## Files

New: `include/maya/core/granule_arena.hpp`, `tests/granule_arena_test.cpp`,
`tools/card_tests/glm_pinned_start.sh`, this report.
Changed: `CMakeLists.txt` (the `granule_arena_test` target), `include/maya/core/glm_model.hpp`
(`kv_pin`/`kv_unpin`/`kv_pinned_release`, the `pinned_now` contract), `src/core/glm_fast_state.hpp`
(`KvStream::arena`/`pin_slabs`/`slab_bytes`/`pinned_now`), `src/core/glm_model.cu` (the `KvPinner`,
`kv_pin_granule`, the arena in `kv_stream_make`, `kv_pin`/`kv_unpin`/`kv_pinned_release`, the load
line, the "now" figures), `src/core/glm_fast_path.cu`, `src/core/glm_prefill.cu`,
`src/core/glm_mtp.cu`, `src/core/glm_mtp_rows.cu`, `src/core/glm_rows.cu` (the `kv_pin` hooks),
`src/core/glm_multi.cu` (`seq_release`), `src/program/generate.cpp` (parking gives back).
serve/ and engines/ are untouched.

## Tests run

* `granule_arena_test` (the plan's unit-tested helper, built as its target in
  `scratch/build-p125g`): **41 checks passed**, exit 0. CPU only, fake `Pinner`.
* Build proof, **no GPU run** (per the plan): `saguaro-glm` (target `maya`) built clean for CUDA
  **sm_75** in the new build dir `scratch/build-p125g` (`cmake -DCMAKE_CUDA_ARCHITECTURES=75`,
  logs `scratch/cmake-configure.log`, `scratch/build-maya.log` [the pre-fix failure],
  `scratch/build-maya-2.log` [the clean build, exit 0, 8 warnings, all in code this task did not
  write: the `w_rq` #549-D and unused `g` #177-D in glm_fast_path.cu (blame: DFlash2, #236) and
  three `-Wmissing-field-initializers` on the slot structs in generate.cpp (blame: #236, 1.0.9)).
  The failed first
  build is what surfaced the `kv_pin` const fix above.
* `tools/card_tests/glm_pinned_start.sh`: `bash -n` clean; **not executed** - it loads the GLM
  model twice, and the plan says no GPU runs here. It takes `CFG=<glm card json>` (port 8092,
  own config copies / session dirs / engine logs under `--out`, api_key dropped,
  `SAGUARO_TEST_EXE=scratch/build-p125g/saguaro-glm` to test this binary), and checks exactly the
  plan's card test: the pinned figure on the engine's own start line is the **same at 32768 and
  262144** (and 0 - streaming is off at 32768 and the arena starts empty at 262144, where the old
  engine pinned 2.95 GB), and a fresh ~20K-token read's wall time at 262144 is within `RATIO`
  (default 1.30) of the 32768 one, warm-up request first, a different prompt seed per phase.

## Open questions

1. Granule accounting across slabs: the eleven DSA layer slabs sit end to end in the arena, so a
   first write at position p pins one granule **per slab** - ~11 x granule MB, not the bytes
   written (the default 128 MB granule: a 20K read at 262144 pins ~1.4 GB, not the ~230 MB it
   wrote; `SAGUARO_GLM_KV_PIN_GRANULE_MB=64` gives ~700 MB). That is the report's "at most one
   granule over what was written" per slab. If the card test finds it too coarse, interleaving
   the slabs (granule k holding byte k of every layer) would pin ~granule MB total per prefix -
   but it changes the resolve kernels' row addressing, so it is a follow-up, not this change.
2. HIP: the change compiles through the CUDA->HIP shim names and the give-back is wired for HIP
   (`seq_release` runs it on every platform), but the only build proof here is CUDA sm_75. The
   9070 XT box the memory report came from is the natural first target for
   `glm_pinned_start.sh` (hipHostRegister on a `MADV_DONTNEED`-then-repopulate range is the
   untested driver path).
3. The WSL ~1 GB cap now stops a *conversation* mid-run instead of failing the load
   (`cannot grow the pinned latent cache`); how the serve loop surfaces that error per request
   is worth a look on a WSL box.
4. Modules the memory report says a change like this "would have to run" that the plan did not
   name and that need the GPU or the serve test rigs: `src/kernels/glm_kv_stream_parity.cpp`,
   `serve/test_context_panel.py` / `serve/test_dashboard.py` (both read the VRAM lines' `kv_ram`,
   which is now the growing figure), `tools/mem_vs_ctx.py` (step 4's comment says its pinned
   column should now move with use - on CUDA via VmPin; the report's open question 1, ROCm not
   counting `hipHostRegister` pages in VmLck/VmPin, decides whether the GLM column moves there).
   Run those with the card test on the GPU box.
5. A mid-conversation unpin cannot be partial by design (`GranuleArena::release` is all of a
   stream's granules): a reset before 200K gives back the granules of the whole written prefix,
   including the part a *new* conversation in the same slot will re-pin. Releasing a suffix only
   (a context that shrank) would need per-granule refcounts - not in the plan, not done.
6. The report's open question 2 (the cannot-pin-at-load error needing a graceful degrade) is now
   mostly moot at the load - nothing is pinned there - but the eager fallback keeps it alive for
   devices without UVA-for-registered-memory, where the whole-max `cudaHostAlloc` still fails at
   the load exactly as before.
