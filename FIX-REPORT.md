# FIX-REPORT - REVIEW-RESULT findings, one commit each (round 7: `e2684fc39..12dc848fb`)

The manager's round-7 card (`REVIEW-RESULT.md`, committed with this report): LOAD and SPEED pass unchanged (1.30 /
1.30 GiB host K/V, 68 / 68 GiB expert RAM, 2571 / 2509 cache slots, 30K read 35.0 / 37.0 s) and round 6's fix works
(the device state grows, 546148 cells, the cache gives way); GROWTH now fails at the NEXT allocation - the YaRN
switch of that same 546,145-token conversation, `ERR no VRAM for the YaRN RoPE rows`. The card's three numbered
parts map to the three sections below (`e758a5b07` + its follow-up `df1adca13`, `33e462af7`, `0434ba69d`; the nit
`12dc848fb`). New build dir `scratch/build-p124e-r7` (CUDA sm_75, ggml from `scratch/ggml-pinned`), 268/268,
`saguaro-qfn` links, no GPU touched.

**What the code actually did.** Round 6 put the give-before-map path (`vmm_grow_room`) in front of the follow-pool
and scratch maps only. The two context arrays that grow on use and did NOT go through it were exactly the two the
card hit: `qsa_rope_yarn_fit` took room ONLY from the card's slot budget - budget short, it returned false without
ever asking the expert cache; no budget at all (the follow-rule rigs), and `VmmRange::map_range` quietly made fresh
driver chunks mid-map, again with no relief. And its size came from `solo_yarn_rows`' `n + max_new + 64`, which for
a request whose `max_new` runs to the end of the context IS `--max-context` - so the switch tried to back the whole
0.75 GiB table in one step, through a path that could not ask for room.

## Finding 1 - EVERY growth-time VRAM allocation runs the give-way-before-map path (`e758a5b07`, `df1adca13`)

The card's list, from the round-5 accounting lines ("addresses for --max-context", `grep DeviceNote`), with what
each allocation does now:

| allocation (accounting line) | grow site | room path |
|---|---|---|
| streamed states' context device state - follow pools (`layer.cpp:2570` addresses, `layer.cpp:2763` start step) | `qsa_follow_grow_at` `layer.cpp:1727` | `follow_room` `layer.cpp:1716` -> `vmm_grow_room` (round 6) |
| block-score scratch (`layer.cpp:1938`) | `qsa_scores_grow` `layer.cpp:1957` | `follow_room` -> `vmm_grow_room` (round 6) |
| grown RoPE table past the trained 262144 (`layer.cpp:2611`) | `qsa_rope_grow` `layer.cpp:2139` | `vmm_grow_room_pooled` (round 7; the old loop asked the relief per refused chunk) |
| **YaRN level's RoPE rows** (`layer.cpp:2718`) | `qsa_rope_yarn_fit` `layer.cpp:2218` | `vmm_grow_room_pooled` (round 7; **was budget-only, no relief at all - this is the card's failure**) |

`vmm_grow_room_pooled` (new in `vmm_grow.hpp`, the same order as `vmm_grow_room`): the card's slot budget first
(its chunks cost the experts nothing), then the driver for the shortfall, the FIRST refusal asking the relief ONCE
for the whole remainder, and the map starting only with every chunk in hand. `qsa_rope_yarn_fit` also keeps its
chunks measure / room / map apart (round 4's rule - the relief's low-water check counts the RoPE chunks through the
very lock it holds), and a refused rope/YaRN map or room now prints the round-6 line (which driver call, the bytes
asked, free VRAM, what the cache was asked and gave) instead of silence, and records the refusal in
`qsa_dev_pin_failed()` so the solo admit and the decode's end answer `no VRAM`, not `no RAM` (`df1adca13` - the
round-6 wording bug at a new site).

Allocations examined and left as they are, with why: `qsa_slots_grow` (`layer.cpp:1548`, the slots' rings and pooled
rows) takes no driver allocation at all - it only remaps chunks the slot budget holds, and the budget is filled by
the same cache `give()` (`generate.cpp:13888`) that the `vram_relief` hook (`generate.cpp:18722`) drives, so its
room arrives through the same giving path. `qsa_slots_grow_fresh` and the `*_back_whole` fallbacks are the load's
start step and the feature-off fallback - no relief is installed at the load, and the fallback's numbers stay byte
for byte (criterion D). `mtp.cpp:1445`'s records buffer already asks the relief before retrying. The load-time
carves, head and drafters do not grow on use.

## Finding 2 - the YaRN rows follow the POSITIONS IN USE (`33e462af7`) - and which sizing, said out

**Which: "back them in chunks as the conversation grows", as a prefix over the positions the conversation has
reached (+ a fixed 32768-token ahead-step) - not "the window in use".** Reason, from how the kernel reads them:
`rope_tab_cs` (`include/strata/kernels/mrope.hpp:95-99`) reads row `p - kRopeYarnOff` for whatever position the
current kernel is ROTATING, and a prompt re-read, a checkpoint reuse or the drafter's verify window can re-rotate
any cell below the conversation's length - so the backed set must stay a gapless prefix [0, positions reached), and
a per-window backing would leave holes that the next re-read reads as unmapped VMM. What it does NOT owe is
positions the conversation has not reached:

* `solo_yarn_rows` (`generate.cpp:21218`) sizes `min(request end, at + kSoloPinAhead) + 64`, `at` = the positions
  in use at the call (the prompt `n` before a read, `21263`; the current cell `p` at a mid-reply crossing, `21238` -
  the crossing window lands inside the ahead-step). The old `n + max_new + 64` was `--max-context` itself when
  `max_new` runs to the end of the context: that is the card's "all 0.75 GiB at the switch".
* the request-start fit (`generate.cpp:19541`) passes `rope_cells = solo_pin` (prompt + ahead-step, the number the
  host   copies already use, `generate.cpp:19525`) to `session_vram_fit` (`session.cpp:297`) / `session_vram_need`
  (`session.cpp:310`), which cap the rope grows with it - the K/V state still takes the whole request (the plan:
  grow ahead of need once, at the read's start), the RoPE rows trail the positions.
* past that, the rows arrive chunk by chunk from the fits that track the conversation - the decode's 32K-ahead pin
  step (`generate.cpp:22492` -> `solo_host_fit_r` -> `session_host_fit`, whose rope step is `session.cpp:270`), the
  batch slots' `slot_fit` grow target (`generate.cpp:14039` -> `session_slot_fit` `session.cpp:240-241`) and the
  slot decode step (`yarn_fit` `generate.cpp:14519`, sized by the slot's own next cells).

On the card's run: the switch backs ~546K+33K positions (a read from 0 needs every row it re-rotates, and 546K rows
of cos+sin is what a 546K conversation costs - not the 1M table), the next 32K rows come with the reply's first
32K tokens, each step through the cache's give-way, and `qsa_rope_yarn_trim` still hands all of it back when no YaRN
conversation is live.

## Finding 3 - the tests, for each growth allocation (`0434ba69d`)

`src/core/vmm_grow_test.cpp` (the module the card names - run ALONE) now covers BOTH room paths, and its header
lists which growth allocation runs which (table above). Five new cases for the pooled path: the budget's chunks are
taken whole and cost the driver and the cache nothing ("unused room the experts already hold"); budget, then
driver, then the relief asked ONCE for the SHORTFALL after the budget, `give:` before `map:` - event by event, the
YaRN switch's order; a room that never arrives maps NOTHING and reports `from_pool` so the caller gives the
budget's chunks back; `may_relief=false` never asks; a need of 0 touches nothing. 44 checks (the round-6 28 pass
word for word). The pooled case caught a real bug in the helper fix 1 added: `vmm_grow_room` sized its one relief
ask by `got.size()`, which the pooled path seeds with the budget's chunks - a grow whose budget part landed and
whose card part refused asked the cache for ZERO bytes and died; the ask is now `need - chunks THIS call made`
(the plain path's numbers are unchanged - its `got` starts empty).

## The nits

* Taken (`12dc848fb`): `serve`'s `engine_refusal` matched only "no VRAM is left", so `ERR no VRAM for the YaRN RoPE
  rows` - the very line this card hit - stayed an ordinary request error; "no VRAM for..." now ends as
  `EngineOutOfVram` too (retry once / 503), `serve/test_vram_refusal.py` gains the case, `serve/qfn_engine.py`
  re-extracted with `tools/saguaro/extract_qfn_engine.py`, never hand-edited.
* Taken (`df1adca13`): the rope/YaRN grows record `qsa_dev_pin_failed()` (see finding 1).
* Not taken: a per-window YaRN backing (the card's "or"); finding 2 says why - the kernel reads rows for any
  position a later window re-rotates, so only the prefix rule is safe without hooking every window.

## Verification

* `cmake -S engines/qfn -B scratch/build-p124e-r7 -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
  -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON
  -DSTRATA_MMQ_TQ_OVERLAY=OFF -DSTRATA_GGML_DIR=$PWD/scratch/ggml-pinned`; `ninja -C scratch/build-p124e-r7` ->
  268/268, `saguaro-qfn` links, `sm_75` objects only, no GPU run. The warnings in the touched files are exactly the
  pre-existing set the round-5 report lists (layer.cpp's indentation/unused-var pair, generate.cpp's `w`); nothing
  new from the rope/session/test changes.
* `ctest --test-dir scratch/build-p124e-r7 -R '^vmm_grow_test$'` run ALONE -> passed, "44 checks OK";
  `ctest ... -R '^kv_host_growth_test$'` run ALONE -> passed, "22 checks OK" (its helpers did not change);
  `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_vram_refusal` -> 4/4 OK. One at a time, nothing else run.
* No GPU here, so the card keeps the proof: `scratch/card-test-elastic-kv.sh` (gitignored, the manager's copy) gained
  criterion G - the 546K run's log must show the YaRN rows climbing step by step (NOT 1048576 at the switch), each
  step preceded by the cache's give on a full card, no `ERR no VRAM for the YaRN RoPE rows` anywhere, and the
  round-7 load/speed figures of the card unchanged.
