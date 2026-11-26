# P124b - REPORT: a paused SOLO prompt read is parked, not thrown away (Saguaro 1.2.4)

Branch `saguaro/q1-p124b-lost-read`, base 1.2.3 (`88809c4ea`). Four commits + this report:
`ba69e3f59` (engine), `c93813b54` (server log line), `fd733e828` (tests), `90e807a79` (card script).

## What was wrong

`ref/mortar-paused-read-20261125.log`: a low read paused at 16,384 of 105,603 prompt tokens, the engine's
cancelled leg said `0 reused + 16384 of 105603 read ... (cancelled)`, and after the switch to the normal
request and back the same prompt came in as `new conversation, 105,603 tokens to read` and
`0 reused + 105603 read` - the 16,384 tokens read before the pause were gone. The key detail: the log has
NO `cancelled request keeps its conversation: N tokens live` line - #1620's keep never fired. It requires
`pp_exact` (the session sitting exactly at a reported position), and a read stopped INSIDE a batched or
pipelined part - which a 100K read on a slow card is, chunk after chunk - leaves `pp_exact` false
(`generate.cpp` sets it false before each part precisely because "a cancel may leave the state past
pp_reached"). So the cut read stayed `live_ok = false`, the next request's switch parked only the finished
turn that came in, and the cancelled read's own periodic checkpoints (`1 checkpoints` on the DONE line)
were erased by that request's start path. Meanwhile the server logged "the low prompt read continues from
where the engine kept it" - a claim with nothing behind it.

## What changed

**1. Engine - the cut read lands on its checkpoint and gets parked like a finished turn** (step 1,
`ba69e3f59`). `engines/qfn/src/program/generate.cpp`, right after the #1620 keep: a read that was cancelled
with `pp_exact` false (and is not a non-finite/#879 read, not a no-VRAM read, not a BYIELD - the part that
already waits in its slot) now takes the deepest checkpoint the read took at or below the position it
reached whose tokens are this prompt's, syncs and restores it into the session (main + every split stage,
the way `live_from_checks` does at request start), and makes it the live conversation - the same
`cancelled request keeps its conversation: N tokens live` state, with the line noting the checkpoint.
From there everything existing owns it: the next request of the same prompt continues from it
(`starts_with(live)` -> `RESUME N`), and a switch to another conversation parks it through `park_current`
- K/V to the position, the checkpoint chain, the prompt ids - under the park budget, the physical-RAM
admission and the disk spill like any parked conversation; a `--session-dir` box also has the turn save.
The pick itself is a new pure helper, `strata::program::conv_cache::cut_read_landing`
(`engines/qfn/include/strata/program/conv_cache.hpp`), so it is testable without a GPU. The #1620 exact-
position keep is untouched; if it fires, `live_ok` is already true and the landing stand aside.

**2. Server - the continue line carries the engine's real reused figure** (step 2, `c93813b54`).
`serve/server.py`, `Service._sliced`: a resumed read no longer prints
"prompt read continues from where the engine kept it" at the hand-back. It waits for the resumed engine
call's first output and prints `priority: the low prompt read continued: 16,384 of 105,603 tokens reused`
with the engine's own figure (`engine.reused` - QFN's `RESUME` line, the same number that leg's `DONE`
carries; the QFN driver resets it at every `generate()`, so it cannot be the previous leg's value). An
engine that reports no figure over the pipe (the line-protocol drivers, MockEngine) keeps the old
numberless wording. 0 reused prints as 0 - a lost read is now honest instead of claimed. The answer
line ("answer continues") is unchanged. No change to the QFN engine driver was needed, so
`serve/qfn_engine.py` was NOT regenerated (its source `engines/qfn/serve/server.py` is untouched; the
standalone QFN server has no read-preemption line to begin with).

**3. Tests** (step 3, `fd733e828`).
- `engines/qfn/src/program/conv_cache_test.cpp`: ten checks for `cut_read_landing` (deepest checkpoint at
  or below the position, the edge at exactly the position, nothing early, no checkpoints, unsorted chains,
  off-path checkpoints skipped, an empty checkpoint never a landing point).
- `serve/test_lost_read.py` (new, `serve/test_priority.py`'s race with the same fakes): the continue line
  carries the engine's reused figure and it equals where the read paused; a fake whose cut read is lost
  (`LostRead`) must log `0 ... reused` and never the old claim; a fake with no figure
  (`UnreportingRead`) keeps the numberless line. The `ReadEngine` fakes gained `reports_reused` /
  `self.reused = pos` (their RESUME figure); `PausingRead` (the GLM-style line engine) reports none.
  `serve/test_priority.py` follows the line's new wording in
  `test_a_qfn_read_is_cut_and_continues_from_where_it_was`.

**4. Card test for the manager** (step 4, `90e807a79`). `scratch/card-test-lost-read.sh` (force-added,
like the other scratch deliverables): a test copy of the card's config on port 8092 with its own session
dir under `scratch/lost-read/sessions`, `SAGUARO_PRIORITY_IDLE_S=2` for the resume; a fresh ~100K-token
low read cut at positions 20,000 / 40,000 / 60,000 by a short normal request; PASS requires three
`prompt read paused` lines, a `continued: M of N tokens reused` line per cut with M > 0 and growing, and
the interrupted read's summed engine read time within `RATIO` (default 1.30) of a second fresh
uninterrupted control prompt - the lost-read shape roughly doubles it. Its log parsers were checked
against `ref/mortar-paused-read-20261125.log` itself (max position 105,602, summed read ms 104,510 =
15,323 + 186 + 89,001). It has NOT been run on a card yet - that is the manager's step; nothing GPU was
run here.

## Files

- `engines/qfn/include/strata/program/conv_cache.hpp` - `cut_read_landing` (pure)
- `engines/qfn/src/program/generate.cpp` - the landing + keep at the cancelled-read point
- `engines/qfn/src/program/conv_cache_test.cpp` - landing scenarios
- `serve/server.py` - the resumed-read line with the engine's figure, in `Service._sliced`
- `serve/test_priority.py` - fake `reused` figures, `LostRead`, `UnreportingRead`, wording
- `serve/test_lost_read.py` - new
- `scratch/card-test-lost-read.sh` - new (force-added)

## Tests run (one at a time, as planned)

| what | how | result |
|---|---|---|
| `conv_cache_test` | `g++ -std=c++17 -Wall -Wextra -I engines/qfn/include` + run | PASS (all, incl. the 10 new landing checks) |
| `serve.test_lost_read` | `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_lost_read` | OK, 3 tests |
| `serve.test_priority` | `python -m unittest serve.test_priority` | OK, 53 tests |
| QFN engine, CUDA sm_75 | `cmake -S engines/qfn -B scratch/build-sm75 -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75` (CUDA 13.1) + `cmake --build scratch/build-sm75 --target strata -j 40` | exit 0; `generate.cpp` compiled clean (one pre-existing warning at `bt_windows`, not from this change), `saguaro-qfn` linked; nothing executed on a GPU |
| card script | `bash -n`; parsers smoke-checked against the ref log | syntax ok, parsers agree with the log's own figures |

## Open questions / for the manager

1. **The engine fix is compile-verified only.** The landing path (restore at request end, then
   park/continue) needs the card run of `scratch/card-test-lost-read.sh` to prove the `M > 0` each time and
   one-read's-worth of read time; it also should watch once for the new stderr line
   `cancelled request keeps its conversation: N tokens live (its checkpoint: the read stopped inside a part)`.
2. **How deep the landing can be** is bounded by `--prompt-cache-every` (16,384 by default): a read cut
   before its first checkpoint still has nothing to land on (the helper returns -1 and the old drop
   happens). On mortar's card the first chunk boundary is ~15 s in; a cut before it stays honest ("0
   reused") but still restarts. Raising the checkpoint frequency is a config call, not code.
3. **Per-request billing figures unchanged:** `_sliced`'s merged-segments `reused` still takes the FIRST
   leg's figure (0 for a cut read, as before - conservative, never overclaiming). The new priority line is
   the truthful figure; making the merged request stats add up across legs is a separate decision.
4. `CHANGELOG.md` / `VERSION` untouched - this is a fix branch; the 1.2.4 release note is yours.
5. The card script sets `SAGUARO_PRIORITY_IDLE_S=2` for the test server; the live box keeps its 30 s
   idle grace (mortar's own resume gap in the ref log was that grace).
