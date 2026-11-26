# P124e - QFN K/V memory grows with use, not up front - REPORT

Branch `saguaro/q1-p124e-elastic-kv`, two commits on top of `88809c4ea` (Saguaro 1.2.3):

* `747fe1908` - step 2+3: the elastic host K/V (growth modes, relief, load-time start)
* `ee8e3dc89` - step 4: pure sizing/growth helper + unit test + the manager's card-test script

## Step 1 - how the streamed host K/V was sized and pinned before this change

The three log lines the plan greps for sit in:

* `saguaro generate: KV streaming: N of M cells per QSA layer in VRAM, the K/V in X GiB of pinned RAM` -
  `engines/qfn/src/program/generate.cpp`, right after the CUDA0 `session_init` (now line ~5744).
* `saguaro generate: KV streaming: X GiB of pinned RAM for the K/V of every session at --max-context N ...` -
  `generate.cpp`, after the batch slots and the drafters are up (now line ~6228).
* `CUDA%d cannot use a registered page's host address: the slots' host copies are pinned whole` -
  `engines/qfn/src/core/layer.cpp`, `host_elastic_on()` (the old wording).

The pinning itself happened in `qsa_state_init()` (`layer.cpp`): for every streamed state
(`kv_mode` 1, the main layers with `--kv-resident`) and every drafter ring (`kv_mode 2`) whose host copy
was NOT made elastic, the whole host copy - `pages x bytes-per-block` for the WHOLE `--max-context`,
all arrays packed in `kv_per_slot` order - was allocated and pinned once, at state construction, by
`kv_host_alloc()` (a `cudaHostAlloc(Mapped|Portable)` with a `cudaHostRegister`-of-`MAP_NORESERVE`-anon
fallback for WSL2) and counted in `g_kv_host_bytes`.

* **Solo session** (and each layer-split stage session): every QSA layer's `QsaState` pinned its full
  context copy inside `session_init` - mortar: 12.38 GiB at 1M, 3.10 GiB at 262144.
* **Batch slots**: the slot sessions (`generate.cpp` ~5940) are elastic-slot states, so their host copies
  were elastic only when `host_elastic_on()` said yes; where it said no they took the same whole pin
  through the same code (`slot_G > 0 || g_host_main` decides *elasticity*, `host_elastic_on()` decides
  whether the elastic pool is made at all).
* **Drafters**: the MTP drafter's ring host copy (`mtp.cpp`, `qsa_state_init(..., ring, ..., host_elastic)`)
  grew with its owner's slot through `SessionState::host_follow` (`generate.cpp`: `host_follow = &mtp.kv_state()`
  / `qsa_host_grow(d->kv_state(), owner.slot_cells)`) - again only if `host_elastic_on()`.

The elastic machinery itself (`SAGUARO_KV_HOST_ELASTIC`, `layer.cpp`: `HostPool`, `host_pool_make`,
`qsa_host_grow`/`qsa_host_trim`, `session_slot_fit`/`session_host_fit`, `ram_reclaim`) already existed -
but `host_elastic_on()` gated it on `cudaDevAttrCanUseHostPointerForRegisteredMem` for EVERY device.
HIP maps that to `hipDeviceAttributeCanUseHostPointerForRegisteredMem`, which ROCm does not implement,
so on mortar the gate said "no" and everything fell back to pinned-whole: the 38.16 GiB line, then
`FileExpertSource: --resident-budget-gib 68.00 exceeds available RAM (45.80 GiB ...): 41.55 GiB`
(`expert_source.cpp`, sized from `available_memory_bytes()` AFTER the K/V pins) and the 100%-from-pack
warning. Elastic positions (YaRN) were per-conversation; the memory was not.

## Steps 2+3 - what changed (commit 747fe1908)

1. **The gate is a probe now, not just the attribute** (`layer.cpp: probe_registered_page_address()`).
   One `kHostPiece`-sized piece of a scratch `MAP_NORESERVE` range is really registered and
   `cudaHostGetDevicePointer`-checked (`d == a`). NVIDIA passes as before (attribute first, probe to
   confirm); HIP fails the attribute but passes the probe (flat VA), so mortar gets the *pieces* mode
   it was silently being denied. K/V stays INT8 throughout - no dtype or layout change anywhere.
2. **Prefix mode for the real fallback** (`layer.cpp: HostPool::prefix`, `host_prefix_reregister_locked()`):
   when a card truly cannot hand the kernels a registered page's host address, growth still works:
   every array is ONE registered prefix `[off_i, off_i + backed x per_i)` and a grow/trim
   re-registers the (all-or-nothing) set and rewrites the owner state's `host` pointers from the new
   device bases (`HostPool::owner`, `host_apply()`). The device is synchronized before every remap;
   no copy can ever straddle a registration boundary (each copy site is one array run). This is the
   plan's "chunked pinned allocations" - the arrays are the chunks. The old "pinned whole" outcome
   survives only as `SAGUARO_KV_HOST_ELASTIC=0` or Windows (message text updated:
   `... the host copies grow as one registered prefix per array (P124e)`).
3. **Memory follows use at every size** (`kv_host_growth.hpp` helpers, used by both modes):
   * the step and the load-time start is `SAGUARO_KV_HOST_STEP`, default now **32768 cells** (was 8192) -
     "about 32768 positions, the cells QFN keeps in VRAM already", per conversation, at 262144 exactly
     as at 1048576;
   * `solo_host_fit(1)` (moved BEFORE the "every session" load line, `generate.cpp:6211`) backs the solo
     session + stages + its drafter start at load; each elastic batch slot's existing
     `session_slot_fit(slot, 1)` (`generate.cpp:5985`) backs its start, and each batch slot's drafter its
     start (`generate.cpp:6189`); a prompt read still backs the whole prompt ahead of need (the admit
     path fits prompt + max_new before the read; `qsa_host_grow` on a fresh state pins the whole ask);
   * past the backed end a grow doubles what the state holds (`kv_host_grow_pages`) - 1M fills in ~5
     steps, not 32; each step logs `the host copy backs N cells now (X GiB pinned for K/V in all, T ms)`
     (prefix mode times its own remap, pieces mode the grow call) - so the card test can report every
     growth step's time, and no step runs per-token (decode only crosses a boundary if it overruns the
     pinned `max_new`, and then the grow happens between windows, never mid-copy);
   * **give-back**: `qsa_host_trim` (park/reset/end via `session_slot_fit(trim)` / `solo_host_trim`)
     returns whole steps down to the start step; parked conversations keep one step (~0.4 GiB), like
     the pieces mode's "keep one piece" rule.
4. **The expert RAM tier keeps its budget at load and gives way only when a conversation grows** :
   * at load the K/V now holds ~0.4 GiB per conversation instead of the worst case, so
     `available_memory_bytes()` when `pin_cache_complement` sizes the budget sees the same (bigger)
     room at 262144 and at 1048576 - `--resident-budget-gib` stays the ceiling and "fills up to it when
     the room is there"; the 68 -> 41.55 GiB shrink should not recur;
   * when a grow still runs out of RAM, `qsa_host_grow` asks the new host-relief hook
     (`layer.hpp: set_host_relief/host_relief`, wired in `generate.cpp:6728`) BEFORE parking anything:
     `ExpertSource::relieve_ram(bytes)` (new virtual; `FileExpertSource` override at
      `expert_source.cpp:3465`) hands back the resident complement's **coldest** experts - its slots are
      walked by descending arena offset (the copy is hottest-first), their offset set to `kNoComplement`
      so later reads fall through to the mapped files, their working-set lock undone (a locked page
      survives `MADV_DONTNEED`), and their pages `madvise(DONTNEED)`-ed only after a grace window, so a
      CPU reader that latched a blob before the trim never reads zeroed weights (open question 4);
      the registered PCIe prefix and a `cudaHostAlloc` arena are never trimmed, and
     `exchange_storage_` (SAGUARO_EXCHANGE_ROTATE) turns the give-way off. The relief handler syncs every
     device first; if giving way did not help, the existing `ram_reclaim` (park an idle conversation,
     drop the conversation cache) still runs unchanged. The tier takes the room back naturally: freed
     pages are ordinary reclaimable RAM, and `--session-dir`/idle-park paths are untouched.
5. **Load-time reporting**: both KV-streaming lines now count the *backed* bytes and say the copy grows
   (`it grows with the conversation (X GiB if one filled --max-context)`), and the "every session" line
   splits solo (backed, via `session_host_backed`) from slots+drafters; `qsa_kv_host_reserved_bytes()`
   keeps the worst case for the "if every one were full" clause.

## Step 4 - tests

* **Unit test** (the plan's "pure sizing/growth helpers ... like `core/park_policy_test.cpp`"):
  `engines/qfn/include/strata/core/kv_host_growth.hpp` (new: `kv_host_pages_for`, `kv_host_step_up`,
  `kv_host_pieces`, `kv_host_grow_pages`, `kv_host_trim_pages`, `kv_host_prefix_bytes`) and
  `engines/qfn/src/core/kv_host_growth_test.cpp`, registered in `engines/qfn/CMakeLists.txt` beside
  `park_policy_test` (`STRATA_BUILD_CONVERSATION_TESTS`). 22 checks: the start is identical at 262144
  and 1048576, a grow satisfies a big first ask but a fresh conversation lands on one step, doubling
  bounds a 1M fill to <20 steps, pieces end on step boundaries, a park keeps exactly the start step,
  prefix sizes are page-aligned.
  **Result: `ctest -R kv_host_growth_test` -> 100% tests passed, 0 failed (22 checks).** Run alone, as
  the plan asks; no other test module was run (no GPU tests exist that could run here).
* **Build**: NEW build dir `scratch/build-p124e`, CUDA sm_75, CUDA 13.1:
  `cmake -S engines/qfn -B scratch/build-p124e -G Ninja -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_MMQ_TQ_OVERLAY=OFF -DSTRATA_GGML_DIR=$(pwd)/scratch/ggml-pinned
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` -> `ninja` **437/437 targets OK, `saguaro-qfn` links**
  (no GPU was used for anything). `third_party/ggml` is an unfetched pinned commit in this tree
  (`VERSION.txt` `3cf03257f...`) and there is no network, so the pinned ggml was exported with
  `git -C ../llama.cpp archive 3cf03257f | tar -x -C scratch/ggml-pinned` and passed via `STRATA_GGML_DIR`
  (scratch, uncommitted).
* **Card test for the manager**: `scratch/card-test-elastic-kv.sh` (scratch/ is gitignored by the repo,
  so it lives only on disk - as the plan words it). It drives the engine's `--serve` stdin (`GEN <max_new>
  <ids>`) three times - 30K read at 262144, at 1048576, then one conversation grown to 300K - and prints,
  side by side, the pinned-K/V load line, the VRAM cells, the RAM-budget line, the give-way lines, the
  per-step "backs N cells now (... ms)" lines, the YaRN switch line, and read/decode tok/s parsed from the
  `DONE` lines, with the pass criteria appended. Needle files (`NEEDLE`/`HAYSTACK`/`QUESTION`, token ids)
  make the 300K run also re-answer a question about a needle planted at 10K after the YaRN switch.

## Acceptance mapping (what the card run must now show)

| check | before (mortar) | after |
|---|---|---|
| pinned host K/V at load | 9.54 GiB @262144, 38.16 GiB @1048576 | ~0.4 GiB per conversation (one 32768-cell step: solo, each slot, each drafter), IDENTICAL at both contexts |
| VRAM cells per QSA layer | 32768 of N | unchanged (only host memory changed) |
| resident expert budget | shrunk 68 -> 41.55 GiB @1M | same at both contexts, filled up to `--resident-budget-gib` (>= today's 262144 figure, since the K/V no longer eats RAM up front) |
| 30K read / decode | ~117 tok/s @1M vs ~650 @262144 | equal at both (nothing pins per token; prompts back ahead of need) |
| conversation past 262144 | paid up front for everyone | pays only its own steps; logs each step's ms; expert tier gives way (coldest first) only if RAM is short; a park gives the steps back |
| YaRN (positions) | unchanged | untouched (`SAGUARO_YARN_ELASTIC` behaviour identical); composes with growth (RoPE/YaRN tables already grow per conversation) |

## Files

* `engines/qfn/include/strata/core/kv_host_growth.hpp` (new, header-only pure arithmetic)
* `engines/qfn/src/core/kv_host_growth_test.cpp` (new test) + `engines/qfn/CMakeLists.txt` (register it)
* `engines/qfn/src/core/layer.cpp` - probe gate, prefix mode, step 32768, grow/trim, relief hook
* `engines/qfn/include/strata/core/layer.hpp` - `HostRelief` declarations
* `engines/qfn/src/core/expert_source.cpp` + `include/strata/core/expert_source.hpp` - `relieve_ram`
* `engines/qfn/src/program/generate.cpp` - start pin before the load line, load-line accounting, relief wiring
* `scratch/card-test-elastic-kv.sh` (manager script; scratch, uncommitted)
* NOT touched: `serve/qfn_engine.py`, `engines/qfn/serve/server.py` (no flag plumbing was needed - the fix
  is default-on engine behaviour), `session.cpp`/`mtp.cpp` (their `qsa_host_grow`/`host_follow` plumbing
  already existed), K/V dtype (INT8, Ken's rule).

## Open questions / risks

1. **Prefix mode is untested on real hardware** - no card here lacks the host-address property; NVIDIA
   (attribute + probe pass) and the piece mode are what this box could exercise (compile-only, sm_75).
   The WSL2/dxg path is the realistic prefix-mode user; the card script's step C should be run once on a
   WDDM-ish box before trusting it (it degrades to "grow re-registers", never to wrong data).
2. **A mid-decode boundary crossing stalls for one remap** if a conversation overruns its admitted
   `max_new` pinning: the prefix mode synchronizes the device before re-registering (pieces mode never
   remaps, so only the fallback pays). Growth never happens per token; with `--serve` the admit pins
   prompt + max_new up front.
3. **The trim floor is one step** (~0.4 GiB per parked conversation, both modes) - chosen for parity
   with the pieces mode's keep-one-piece rule and to make re-admission cheap; trivially lowerable to
   zero if Ken wants parked chats to cost literally nothing.
4. **`relieve_ram` trims while other threads may read the complement** - FIXED on review (it did NOT
   hold: `MADV_DONTNEED` would zero pages a CPU reader had just latched). The give-way is two-phase now:
   trim the offsets (release store; later readers see `kNoComplement` and read the files) and undo the
   working-set lock first, then after a grace window (`SAGUARO_KV_RELIEF_GRACE_MS`, default 250 ms, one
   wait per give-way on the already-slow relief path) discard the pages - so a reader that latched a
   blob's address before the trim finishes over whole bytes (unlocking keeps them). The relief handler
   still syncs every device, so the GPU lane is quiet anyway. What remains is a reader that re-reads a
   file it was about to serve from RAM - wasted work, never wrong data.
5. `docs/BENCHMARKING.md` (the plan's interleaved-comparison protocol) does not exist in this tree; the
   script keeps the same *spirit* (same-config, both contexts, printed side by side) but old-vs-new
   numbers must be captured by running the script against the pre-change binary too.
6. Whether `--resident-budget-gib auto` should *raise* its fill above today's number once the K/V
   stops eating RAM - it already does (the budget is sized from the real free RAM), but the
   lend-region sizing path was not otherwise touched; a mortar run will show the exact new figure.
7. The piece-mode growth log line prints on every grow call that pins something (one line per 32768
   cells) - intended for the card test's "report the time any growth step took"; if it turns out noisy
   on a busy server it can be raised to log only >1-step jumps.
