# FIX-REPORT - REVIEW-RESULT findings, one commit each (round 6: `d94e12c4b..374bac012`)

The manager's card test (mortar, RX 9070 XT, 2026-11-26 20:11-20:33, run against `d94e12c4b`, `REVIEW-RESULT.md`
committed with this report) is the whole finding this round: LOAD and SPEED pass and stay untouched; GROWTH fails -
one 546,145-token conversation on the 1048576 config grew its host copy correctly through 44 steps to "546148 cells
now (4.91 GiB pinned for K/V in all)" and then died with `the context's device state could not be mapped on CUDA0:
no error` followed by `ERR no RAM is left to pin this request's context (546145 tokens)` - a VRAM failure reported as
"no error" and answered with a RAM message while 13 GiB sat free. The card's three numbered parts map to the three
sections below (`a83743418`, `59c4c6688`, `f8f6ff2c0`; the nit `374bac012`); the new build dir is
`scratch/build-p124e-r6` (CUDA sm_75, ggml from `scratch/ggml-pinned`), the acceptance stays
`scratch/card-test-elastic-kv.sh` (criterion F added this round), and nothing here is GPU-testable on this box.

**What the code actually did.** Two defects, both in the round-4 VRAM half (`follow_room` / `follow_map_locked`):

1. *The probe lied.* `follow_room` probed the room one chunk at a time - `cuMemCreate`, then `cuMemRelease`, then
   the next - so it proved only that ONE chunk fits at a time. With the card nearly full (weights + graphs + the
   7.3 GiB expert cache) it returned true for GiBs the card did not have, **never asked the relief** (a single
   chunk always fit, and the relief was only ever consulted on a per-chunk refusal), and `follow_map_locked` then
   failed mid-`map_range` - which is the card's "could not be mapped". The cache gave way only in the rope grow's
   loop (`qsa_rope_grow` keeps its chunks); the follow pools and the scratch probed-and-freed.
2. *The error was unread.* The driver API (`cuMemCreate`/`cuMemMap`/...) never sets `cudaGetLastError()`, so
   `cudaGetErrorString(cudaGetLastError())` printed "no error" over a genuine out-of-memory. And the caller -
   `solo_host_fit_r` failing, which covers BOTH halves since round 4 - answered with the RAM text, because the
   half that failed was never consulted: `qsa_dev_pin_failed()` was defined in round 4 and read nowhere.

## Finding 1 - the mapping failure reports its real cause (`a83743418`)

* `vmm.cpp` / `vmm.hpp`: every failing driver call records itself first - `cuMemAddressReserve`, `cuMemCreate`,
  `cuMemMap`, `cuMemSetAccess` (HIP branch: `hipMemAddressReserve`/`hipMemCreate`/`hipMemMap`/`hipMemSetAccess`,
  which already logged and now record too) - into a thread-local `vmm_last_error()`: the call, the driver's own
  string (`cuGetErrorString`, resolved optionally at startup; `CUresult %d` without it) and the bytes. That is
  what the failure line prints instead of `cudaGetLastError()`, so "no error" is gone by construction; a build
  with no VMM says so.
* `include/strata/core/vmm_grow.hpp` (new, pure): `vmm_grow_failure_text()` composes the line the card asked for -
  which step failed (`mapped` / `grown`), the driver call and its code, how much was asked (MiB and chunks), how
  much VRAM is free right now (the caller's `cudaMemGetInfo`), and whether the expert cache was asked to give way
  and what it gave ("was asked to give way of 776 MiB and gave 512 MiB" / "was never asked (the driver's free
  memory was not short)"). Pure, so the test reads it; it never contains "no error" and never the word RAM.
* `layer.cpp`: `report_vmm_grow_failure()` (the free-memory read + the line) is printed by the follow map's refusal
  (replacing the "no error" line), by the scratch's map refusal (silent before) and by both room-short paths
  (`follow_room` refusing said NOTHING at all before); `follow_room` now also records what the relief was asked
  and what it returned, for the line.
* Never a RAM message for a VRAM failure - `qsa_dev_pin_failed()` decides the wording at every site the grow can
  fail: the solo admit answers `ERR no VRAM is left for this request's context (N tokens): the card's expert cache
  gave all it can` (that ERR text already existed for the budget half; the RAM/VRAM choice is new); a batch slot's
  admit error and its `BDONE` say `no VRAM is left ...` / finish `vram`; the decode's pin ends `finish=vram`; and
  `engines/qfn/serve/server.py`'s `engine_refusal` maps the `no VRAM is left` prefix to `EngineOutOfVram` (retry
  once, else 503 - the same handling as a DONE/BDONE `vram`), never to `EngineOutOfRam`/`RAM_SHORT`.
  `serve/qfn_engine.py` re-extracted with `tools/saguaro/extract_qfn_engine.py` (never hand-edited).

## Finding 2 - the growth works: the cache gives way BEFORE the map (`59c4c6688`)

`vmm_grow_room()` (vmm_grow.hpp, the card's order, pure) - and `follow_room` is now just it plus the device
context: **take every chunk the grow needs from the driver and HOLD them**; the first refusal asks the card's
relief ONCE for the whole remaining shortfall (`(need - made) * G`, the expert cache's coldest residents first,
down to its floor - the plan's take-from-experts in VRAM); only with every chunk in hand is the map tried, and
`follow_map_locked` / `scores_grow_locked` map from `got` instead of taking fresh ones. So for the card's 546K
conversation: the free VRAM runs out after a few chunks, `vram_relief` → `free_to_driver(quiet)` → `give()`
evicts the coldest experts and frees their chunks down toward `give_floor`, and the map never starts before that
room is real - which it must be, since QFN's VRAM K/V is a 32K-cell window (the arrays that scale with the
context are the page table, the pooled indexer rows and the scratch: GiBs, not the K/V's tens of GiB - 7.3 GiB of
cache can free far more than 546K cells of index state costs). A conversation that TRULY cannot fit - the cache
gave to its floor and the driver still refuses - fails at the room step with the line above, and the request ends
`no VRAM is left` (503), never mid-map with a half-mapped state. The load's start step (`qsa_follow_start`, the
pool builder) and the fallback's whole-context backing (`qsa_follow_back_whole`, `qsa_scores_back_whole`) keep
mapping the driver's fresh chunks - no relief is installed at the load, and the fallback's numbers stay byte for
byte (criterion D untouched). The lock rule stands: the room is taken OUTSIDE `g_slot_mu`/`g_scores_mu` (the
relief takes the budget's lock), the map re-takes it; chunks the map never consumed go back to the driver.
`qsa_follow_grow_at` still runs measure - room (+relief) - map as three lock-scoped steps (round 4's rule).

## Finding 3 - the tests (`f8f6ff2c0`)

* `src/core/vmm_grow_test.cpp` (new ctest module, header-only/CPU like `kv_host_growth_test`): a fake allocator
  records the event order, and the card's order is asserted - with a short card, `give:` happens before `map:`
  (give-way-before-map), the relief is asked EXACTLY once and for exactly the shortfall (`(need - made) * G`),
  every chunk is held when the map starts, a room that never arrives maps NOTHING (no half-grown state),
  `may_relief=false` (the load's start step) never asks, and a card with the room never asks - the experts keep
  their room until a conversation needs it. The error-text half asserts the line carries the driver call and its
  code, "776 MiB asked (388 chunks of 2 MiB)", "2.00 of 15.00 GiB free on the card" and the cache clause, and that
  it never reads "no error" and never says RAM - with a recorded driver code and without one. 28 checks.
* `serve/test_vram_refusal.py` (runs against the DEPLOYED `serve/qfn_engine.py`): "no VRAM is left..." →
  `EngineOutOfVram` and not `EngineOutOfRam`, its text never says "no RAM"; "no RAM is left..." unchanged; an
  ordinary ERR stays the request's. The same VRAM case was added to the QFN-side `test_vram_finish.py` - that
  module cannot be imported in this checkout for a pre-existing reason unrelated to this round
  (`engines/qfn/serve/server.py`'s `from serve import turbo_live` names the root package's module, so neither
  sys.path order can satisfy it), which is why the runnable version lives at the root, next to the other
  qfn_engine tests.
* `scratch/card-test-elastic-kv.sh` (gitignored, the manager's copy): criterion F - the grow run IS the 546K
  conversation (`GROW_TO` now defaults to 546145), and it must reach a DONE at 546145 with the ~10K needle
  answered; the report blocks now grep the give-way line, the new failure lines and any "no error" (its presence
  is the finding).

## The nits

* Taken (`59c4c6688`, `374bac012`): the two map failures that quoted nothing - `qsa_slots_grow`'s "mapping N
  chunks failed" and the take-back's "could not map its chunks back" - and `ExpertCache`'s load-map line, which
  claimed "out of memory" whatever the driver said, now quote `vmm_last_error()`.
* Not taken: the rope grow (`qsa_rope_grow`) keeps its own per-chunk relief loop - it already holds its chunks
  (it cannot lie the way the probe did), and rewriting it now would move the RoPE path the round-5 card has not
  faulted; KMEM/HMEM stay --batch-budget-gated (round 4's stated gap, untouched here).

## Verification

* `cmake -S engines/qfn -B scratch/build-p124e-r6 -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
  -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON
  -DSTRATA_MMQ_TQ_OVERLAY=OFF -DSTRATA_GGML_DIR=$PWD/scratch/ggml-pinned`; `ninja -C scratch/build-p124e-r6` ->
  442/442, `saguaro-qfn` links. The only warnings in the touched files are the pre-existing ones the round-5
  report lists (layer.cpp's indentation/unused-var set, generate.cpp's `w`); nothing new from
  `vmm.cpp`/`vmm_grow.hpp`/`vmm_grow_test.cpp`/`expert_cache.cpp`. No GPU was touched (`sm_75` objects only).
* `ctest --test-dir scratch/build-p124e-r6 -R '^kv_host_growth_test$'` run ALONE (the module the plan names) ->
  passed, "22 checks OK" - its sizing helpers did not change; `ctest ... -R '^vmm_grow_test$'` run ALONE (the
  module this round's card names) -> passed, "28 checks OK";
  `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_vram_refusal` -> 3/3 OK. One at a time, nothing else
  run.
* No GPU here, so the card keeps the proof: `scratch/card-test-elastic-kv.sh` criterion F on mortar is the
  acceptance of finding 2 (546K answers, needle found, cache gave way, no "no error", and - if it ever fails
  again - a line that says which driver call refused, for how much, with how much free, and what the cache gave).
  `serve/qfn_engine.py`'s diff is only the `engine_refusal`/`VRAM_SHORT` change, re-extracted, never hand-edited.
