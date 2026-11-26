# FIX-REPORT - REVIEW-RESULT findings, one commit each (round 4: `ae2dac446..8b16a85a2`)

The manager's card test (mortar, 2026-11-26 13:24-13:30, `REVIEW-RESULT.md` committed with these fixes) is the
whole finding this round: the HOST side passes (1.30 GiB pinned at both contexts, expert RAM tier 68.00 GiB), but
**some DEVICE (VRAM) allocation is still sized by --max-context at load** - the GPU expert cache holds 2256 experts
at 262144 and only 843 at 1048576, lendable slots 1915 vs 618, and a 29,571-token read takes 36 s vs 256 s. Fix
per Ken's acceptance (just HAVING 1M set must not hurt; memory follows USE): every context-scaled VRAM allocation
now holds a start step (~32768 cells - the same start the host copies pin), grows with its conversation, and the
experts get everything else until a conversation takes it back. The card's three numbered parts map to the three
sections below; the code change is fixes `d49bedb40`..`8b16a85a2` (5 commits), the new build dir is
`scratch/build-p124e-r4` (CUDA sm_75, ggml from `scratch/ggml-pinned`), and nothing here is GPU-testable on this
box (no card, no resident tier) - the proof stays `scratch/card-test-elastic-kv.sh` on mortar.

## 1. FIX-FIRST step 1 - every load-time VRAM allocation sized by --max-context (`ae2dac446` line numbers)

The K/V ring in VRAM is window-sized already (`kv_plan` gives `p.slots = --kv-resident / 4` pages - equal at both
contexts); `gdn/moe/block/PLE` state does not scale with the context; the prompt path's staging is chunk-sized;
the drafter's buffers are window-sized. What DID scale, per conversation holder (mortar's Q4 config: 12 QSA layers,
`page_size = idx_block = 4`, `idx_dim = 128`, `n_rot = 64`, INT8 K/V; a "full-session holder" is the solo session,
each stage, and - with `--batch`'s default full session shape - each batch slot):

| allocation | size at 262144 | size at 1048576 | where |
|---|---|---|---|
| the indexer's pooled keys `idx_pooled`, `(max_cells/4 + 2) x 128 f32` = 128 B per cell, PER QSA STATE (x12) | 32.00 MiB / state, 384 MiB / session | 128.00 MiB / state, **1536 MiB / session** | `layer.cpp:1902-1903` (bytes), `:2028` (carve) |
| the RoPE cos+sin tables, `max_cells x 32 x 4 B x 2` = 256 B per cell, ONCE per full session (the others borrow) | 64 MiB | **256 MiB** | `layer.cpp:1905` (bytes), `:2053-2054` (carve) |
| the page table `pages x 4 B` = 1 B per cell, PER QSA STATE (x12) AND per drafter ring state | 0.25 MiB / state, 3 MiB / session | 1 MiB / state, **12 MiB / session** | `layer.cpp:1898`/`:2009`; drafter via `mtp.cpp:393-396` |
| the block-score scratch `cell_scores` = `max_cells` f32, once per full session (every `--batch` slot has one) | 1 MiB | **4 MiB** | `layer.cpp:2245` (bytes), `:2256` (carve) |

Per full-session holder the delta 262144 -> 1048576 is 1808 - 452 = **1356 MiB** (+ the ~1 MiB each for drafter
states); at mortar's `--parallel 2` (solo + 2 batch slots) that is ~4.05 GiB of VRAM the setting burned and the
expert cache did not get - which is the card's 2256 - 843 = 1413 missing slots (~2.94 MiB each) and the 1915 -> 618
lendable slots. Nothing else: `session_bytes` (`session.cpp:59-86`) is the only load-time `cudaMalloc` priced from
`o.max_context` (solo `generate.cpp:5736`, stages `:5783`, slots `:5952`), the drafter's arena is `mtp.cpp:393`,
and the table above is everything inside them that reads `max_cells`. The suffix drafter (`:13076/:23995`,
`--suffix`) is also context-sized but is not in this config; left alone, said here.

## 2. FIX-FIRST step 2 - each one sized by use, grown with the conversation like the host K/V (fixes 1-5)

**The rule (fixes 2-5, on for `--serve` where it works, off = today byte for byte):** with KV streaming, the host
elastic rule, and a card that does VMM (NVIDIA; HIP/ROCm 6+ on gfx1201 - `vmm.cpp`'s HIP branch - included;
`SAGUARO_KV_VRAM_ELASTIC=0`, `SAGUARO_KV_HOST_ELASTIC=0`, the CLI, Windows, or no VMM all mean no change), the
four allocations above become **addresses for the whole --max-context and physical VRAM for the start step**:

* `layer.cpp/.hpp` (fix 2): a streamed state gets a **follow pool** (a `SlotPool` of its own, registered apart from
  the budget's `slot_elastic` as `QsaState::follow_elastic`) holding its page table and pooled keys; the session's
  block-score scratch holds one (`qsa_scores_*`, since it is per session, not per state). `qsa_state_bytes` skips
  their bytes (the session arena `cudaMalloc` itself shrinks - the `--batch: %.2f GiB each` line says so),
  `qsa_state_init` reserves the ranges and backs the start step (the host copies' `SAGUARO_KV_HOST_STEP`, 32768
  cells). `qsa_follow_grow(cells)` maps whole chunks from the driver and, when the card says no, asks the card's
  relief hook ONCE (`vram_relief` - the expert cache giving way, the plan's take-from-experts in VRAM) and retries;
  `qsa_follow_trim` frees chunks back to the driver (park/reset/end give the room back, floored at the start step
  like the host copies). `session_bytes`/`qsa_buffers_bytes` priced outside the scope (the CLI, the split search's
  pricing, the context card) keep the whole-context number - conservative, never under-prices.
* `slot_sizing.hpp`, `kv_stream.cu/.hpp`, `conversation_snapshot.cpp` (fixes 1, 5): a fresh chunk is not just
  zeroed - each array declares the fill its content demands (`SlotArray::fill`): a streamed page table needs
  **-1** ("not resident": a zeroed entry would name slot 0 and attend to garbage) and a ring table the formula
  `block % n_slots` (`kv_ring_fill`). Every map path (budget grow, fresh grow, follow grow) runs the fill over the
  fresh unit spans, and the two whole-table writers - `kv_stream_reset` and the state's identity/ring init - are
  bounded to the mapped prefix (a reset past it is unnecessary: no reader names a block before it is live).
* `rope_grow_on` (fix 2, `layer.cpp:1855`): `SAGUARO_ROPE_GROW`'s grown table is now the rule's own - it turns on
  for the follow mode as it did for the elastic solo pools. At 1048576 the table reserves 256 MiB of addresses and
  backs the trained 262144 positions (32 MiB per half = 64 MiB, chunk-rounded exactly the bytes the 262144 carve
  took); `qsa_rope_grow` maps further positions as conversations pass them - and its budget-only chunk source got
  the same driver+relief fallback (`qsa_rope_grow`, fix 2), since a solo-serve card has no slot budget.
* `session.cpp/.hpp`, `mtp.cpp` (fix 3): **one fit, both halves.** `session_host_fit` (the same call the admit,
  the prompt read and the park/restore sites already make - no new call sites) now grows the follow pools, the
  scratch and the session's RoPE rows beside the host copies; `session_slot_fit` does it in both its branches
  (the `slot_cells < 0` host-only one the r3 fix made, and the elastic-slot one), and `session_host_trim` gives
  the device half back too. The drafter's ring table follows its slot (its `elastic_slot`-shaped `follow` param
  derives from its owner's primary state; growth through the existing `host_follow`).
  `session_host_backed_cells` returns the MINIMUM of the two halves, so the decode gate and the admit fit cannot
  let a window touch a position that neither half has.
* `generate.cpp` (fix 4): scope `qsa_set_vram_elastic(o.serve && SAGUARO_KV_VRAM_ELASTIC!=0)` around the holders'
  construction (5732 -> 6211, exactly like the host half), and make the relief real: the expert cache opens in VMM
  (`ExpertCache::set_vmm(true)` before any `open`), the serve loop's give path (which moves cold cache chunks to
  the card and on to the driver) no longer requires `--batch` (one `Give` per card: CUDA0 + each stage, bounded by
  `1 + stages.size()` instead of `bslot_ss.size()`), and the `vram_relief` hook is installed whenever the budget
  OR the follow rule runs (iterating the `Give` list). The load says once: "the sessions' context device state
  (page tables, the indexer's keys, the selection scratch, the RoPE tables) holds N MiB at load and grows with
  each conversation".
* **The safety valve ("nothing may change for a conversation that really grows past 262144"):** a conversation
  that grows past the start needs VRAM the cache holds - only the give path can supply it. Where the give can
  never run (experts not all in the RAM arena, `--adapt-async`, `--prefill-head-alias`, `--peer-device`,
  `--vram-elastic`, helper caches), the follow pools are **backed whole at load** (`qsa_follow_back_whole` /
  `qsa_scores_back_whole` before the cache sizes itself), the rule switches itself off for the process
  (`qsa_vram_follow_off`), and the load says so - those rigs keep today's carve, today's growth (there is
  none), and today's numbers. On a rig where the rule DOES run, growth that the card and the cache's floor both
  refuse ends as the slots' failed fit ends today: the request is refused "no memory", the pages already written
  stay valid, nothing is stepped down below (the r3 fixes 3/4 semantics are untouched).

What a conversation that grows pays, and equality at load: each holder's device state holds the start step at
load - page table 1 chunk (2 MiB) + pooled keys 2 chunks (4 MiB) per QSA state, scratch 1 chunk per session, RoPE
64 MiB per full session (the 262144 carve's exact bytes), the drafter's table 1 chunk - whatever --max-context
says, so 262144 and 1048576 reserve and back the SAME bytes and the expert cache sees the same free VRAM (card
step 3). Past the start, a grow maps chunks (2 MiB each) ahead of need at the fit sites - never per token, never
mid-read (the r3 decode precheck covers both halves now).

## 3. The card test (step 3) and the nits

`scratch/card-test-elastic-kv.sh` is unchanged (scratch, gitignored): its criterion for the two load lines now
compares the device side too - the `--batch: %.2f GiB each` and the solo session's arena are the shrunk carve,
the expert cache's "N in the GPU cache" and the "x of y lendable slots" lines must match at 262144 and 1048576,
and the grown-to-300K conversation must still read correctly while the give lines ("the expert cache ... gives the
slots room", now also feeding the relief) appear. No cheap nits were outstanding; the card's three parts are all
taken.

## Not taken / known gaps

* Nothing declined. Deviations from the literal reading, on purpose:
  - `--suffix`'s context-sized buffers and the `--kv-grow` (no-streaming) layout are NOT elasticized: the first is
    off in the card config, the second's context-sized VRAM is its whole K/V pool, which `--kv-grow` already grows
    and shrinks with the context (and streaming is off there, so the rule's scope excludes it).
  - The `KMEM`/`HMEM` server gauges stay `--batch`-budget-gated; the give path's stdout line still says "elastic
    slots" while now also serving the follow relief. Cosmetics, not semantics.
  - Freed follow chunks go to the driver at once (a park gives its VRAM back); the cache takes that room again
    through its own paths (`VRECLAIM`, the adaptive tier), as with budget chunks today.

## Verification

* `cmake -S engines/qfn -B scratch/build-p124e-r4 -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
  -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON
  -DSTRATA_MMQ_TQ_OVERLAY=OFF -DSTRATA_GGML_DIR=$PWD/scratch/ggml-pinned`; `ninja -C scratch/build-p124e-r4` ->
  427/427, `saguaro-qfn` links; warnings in the touched files are the pre-existing dense-line ones (579/2790/3147/
  3297) plus generate.cpp's unused `w` - none introduced (the one my `qsa_buffers_bytes` line raised is fixed).
* `ctest --test-dir scratch/build-p124e-r4 -R '^kv_host_growth_test$'` run ALONE (the only module the plan names)
  -> 100% passed ("22 checks OK"). The new grow/trim arithmetic reuses the already-unit-tested pure helpers
  (`slot_units`/`slot_chunks`/`slot_units_in`); `slot_sizing_test` still compiles and the `SlotArray` field
  additions keep its aggregate initializers.
* No GPU here: the VMM mapping, the fills, the relief and the serve-loop widening are code-true only - mortar's
  card test is the acceptance. OFF-path: `SAGUARO_KV_HOST_ELASTIC=0` makes `qsa_vram_follow_wanted()` false before
  any state is sized, and the CLI never sets the scope, so those loads carve exactly `ae2dac446`'s arenas (the
  criterion D comparison stands, plus a new "no context device state line" check).
* `serve/qfn_engine.py` / `engines/qfn/serve/server.py` are still absent from `git diff --name-only 88809c4e..HEAD`
  (the rule's switches are engine envs read in C++, no flag plumbing, so no `extract_qfn_engine.py` re-run).
  `scratch/` stays gitignored; the new build dir and its log are uncommitted.
