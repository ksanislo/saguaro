# FIX-REPORT - P125j re-review round 5 (FIX-FIRST, manager card FAIL on saguaro/rc124-cand3)

The review: `tools/card_tests/layer_pause.sh` on the live Q4 `--batch` config still FAILs cut 1 - no
"prompt read paused" line, the normal request's first token after 11.2 s, the low read ran 37632 tokens
straight through "on a thread of its own" (scratch/card-rc3-test.txt, scratch/card-rc3-engine.txt).
P124z's READ_TICK_S makes the server re-decide the preemption every 0.25 s and DS4 keep-read passes;
the QFN read never pauses. Fix items 1-3 below, each its own commit; line numbers as of this branch
after `git merge saguaro/rc124-cand3` (P124z's tick in, P124t2's readiness loops in); fix 1's numbers are that pre-fix state, fix 2's edits shift
the same sites by a few lines and its own numbers are the post-fix ones.

## Fix 1 (REPORT): the read-preemption STOP from the server to the engine's concurrent reader thread

Where the STOP is, and every place it is NOT seen, for a low prompt read on the QFN `--batch` driver.

**Server side - the STOP is never sent for a batch read.**

1. serve/server.py:6049-6052 - `Service._sliced`'s R122 read preemption fires only for
   `(can_pause and multi)` or `(keeps_stopped_read and not engine.batch)`. `QfnEngine` sets
   `keeps_stopped_read = True` (serve/server.py:10017) but `--batch` makes `engine.batch` non-zero
   (engines/qfn/serve/server.py:1131, set from the engine's `INFO batch_slots=` at :1321), so the
   condition is false for every batch read: no `cut`, no STOP, no `paused_for_priority` tag. The
   engine's own docstring says it out loud (serve/server.py:10014-10015): "the preemption never cuts
   a batch read". With P124z's read tick the decision is re-taken every 0.25 s at serve/server.py:6049
   and comes back false every time - which is why the card log has no `priority:` line at all while
   DS4 (keeps_stopped_read, not batch) passes with the same tick.
2. What the gate WOULD do on QFN: `cut = True; break` (serve/server.py:6062-6063; `can_pause` is
   False at :10016), `gen.close()` at :6121, and `Service._sliced` already knows how to log the cut
   with the layer figures: "prompt read paused at N of P prompt tokens, inside chunk c at layer l"
   (serve/server.py:6178-6192, reading `paused_layer` / `paused_chunk` / `paused_at` off the DONE).

**Driver side - the STOP itself works on this leg; nothing to add.**

3. A cut of a request whose read is on the control lines - which is what `--batch` does with the
   concurrent reader: the `--batch` driver runs a lone long read on the SOLO leg
   (engines/qfn/serve/server.py:2351-2364, `phase = "solo"`) - and an early consumer close sends
   `STOP` and drains to its DONE (engines/qfn/serve/server.py:2680-2684; the read-time sends at
   :1959/:1975/:1988 in `_control` are the client-cancel ones). `_parse_done` carries
   `paused_layer` / `paused_chunk` / `paused_at` into `self.last` already
   (engines/qfn/serve/server.py:1679-1681).

**Engine side - the STOP reaches the flags, but the concurrent reader never reaches a pause point,
and if it did, the leg would drop it.**

4. generate.cpp:12812-12814 - the stdin reader thread takes `STOP` into the shared
   `stop_req` (generate.cpp:12762) without the serve loop; the prompt path is wired to it at
   generate.cpp:12849 (`sp.should_stop`) and - P125j - at generate.cpp:12853
   (`sp.stop_between_layers`). So the flag IS set within a line-read of the STOP, on every leg.
5. The flag is never seen by the pause point on the live shape, for two independent reasons:
   - **the pause point excludes a layer split**: prefill.cpp:2978 gates the between-layer check
     (`lp_on`) on `hand_in_ == nullptr && next_ == nullptr` - the concurrent reader runs
     `sp.run` on the head stage of the `--layer-split 11,24,36` pipe (scratch/card-rc3-engine.txt:1-2),
     whose `next_` is the CUDA1 stage, so `lp_on` is false and the pause block at
     prefill.cpp:5516-5545 never runs. (The card's read leg itself is fine - the reader thread at
     generate.cpp:21130-21147 calls the same `sp.run`; only the pipe shape vetoes the pause.)
   - **the server never sends the STOP** (finding 1 above), so even a single-GPU box would only
     pause for a client cancel, never for a priority cut.
6. If a layer pause DID fire on the reader leg, the leg drops it:
   - the inline batch `run` lambda captures the pause (generate.cpp:20886-20890: `lp_pos/lp_layer/
     lp_tokens/lp_chunk` from `sp.paused_chunk()/paused_layer()/paused_tokens()/paused_chunk_index()`);
     the reader-thread `sp.run` (generate.cpp:21135) is not that lambda and sets none of them
     (`lp_*` are declared at generate.cpp:19746 and stay -1);
   - after the join, `read_ok == false` goes to generate.cpp:21229-21232 (`e = read_err;
     return false`), and generate.cpp:21210 computes `q = b0` (the read log at :21222-21228 claims
     "to b0 of n" even though the run stopped mid-chunk);
   - the piece loop's `!sp_ok` branch (generate.cpp:21907-21928) treats it as a cancel only because
     `stop_req` is set - `cancelled = true` (the error text itself would print `ERR layer paused`
     and kill the engine at :21915-21925 if the STOP had not been the trigger);
   - the P125j keep (generate.cpp:23128-23157) arms only `lp_pos >= 0 && sp.pause_rows_bytes() > 0`,
     so with `lp_pos == -1` it is refused and generate.cpp:23219-23221 drops the saved rows
     (`sp.clear_pause()`): DONE says cancel, the next request re-reads the chunk from layer 0 -
     the "read N tokens straight through" outcome the card saw.
   - (What DOES stop this leg today is only BYIELD at a chunk head - generate.cpp:21125-21128 wires
     `sp.stop_at_chunk` to the yield flags alone, and prefill.cpp:3164-3171 checks it (and
     `should_stop` as a plain cancel) at the chunk's TOP, never between layers.)

## Fix 2 (engine b1eb4ed3f + server 537d78db4): the concurrent reader takes the layer pause and keeps what it read

Engine (engines/qfn/src):

- prefill.cpp:2974-2987 - `lp_on` no longer demands `next_ == nullptr`, so the layer split's HEAD reaches the
  between-layer pause block (prefill.cpp:5517-5553) on the card's `--layer-split` box. Safe by construction:
  the pause is taken INSIDE a chunk whose rows have not left the head (the hand-off is at the chunk's end),
  returning ends the run before that hand-off and before any later launch, a later stage's run never consults
  `should_stop` (only the head's is wired, generate.cpp:12849), and `Prefill::run` drains the handed-on chunks
  (`drain_pipeline`, prefill.cpp:5875-5894) before the caller keeps the read - so every earlier chunk ends
  fully written through every layer, the paused chunk holds the layers below the saved one, and the resume
  feeds its rows back at the head at that layer and hands the chunk down the pipe on its chunk end (which the
  later stages then see for the first time). Queued jobs, batched chunks and stages reading handed-off rows
  stay excluded (hand_in_/job), exactly as the old comment demanded.
- generate.cpp:21213-21219, 21240-21248 - the SAGUARO_PIPED_CONCURRENT leg captures the pause after the
  reader joins (`lp_pos/lp_layer/lp_tokens/lp_chunk` from `sp.paused_*`, the same capture the inline lambda
  does at 20891-20895), and the read's own log line says where it got to (`q = lp_pos + lp_tokens`, not `b0`).
  With `lp_pos >= 0` the existing pieces do the rest: `stop_req` routes the failed run to `cancelled`
  (generate.cpp:21926-21946), the P125j keep arms (23149-23178: live through the paused chunk, `pp_reached`,
  the DONE's `paused_layer/paused_chunk/paused_at`), the rows stay with the conversation (23232-23240 no
  longer clears them), and the next request of the same prompt goes on at the saved layer.
- generate.cpp:13039-13042 - the INFO batch line now carries `layer_pause=1` while `SAGUARO_LAYER_PAUSE` is
  on: the fact the server needs before it cuts any batch read.

Server (serve/server.py, hand-edited; serve/qfn_engine.py regenerates byte-identical - the driver needed no
change, its phase=solo finally already sends the STOP and drains to DONE):

- :6041-6058 - the read-preemption gate: `keeps_stopped_read and (not batch or pauses_batch_read)`. A batch
  engine without the pause (older engine, or `SAGUARO_LAYER_PAUSE=0`) is still never cut: a cut read it cannot
  keep would run back from 0 on its next BGEN, so it runs on rather than lose what it read.
- :10009-10042 - `QfnEngine.pauses_batch_read` (:10038-10044): `batch and INFO layer_pause == 1`; the docstring that promised
  "the preemption never cuts a batch read" now describes the shipped shape. The pause line the server prints
  ("prompt read paused at N of P prompt tokens, inside chunk c at layer l ...") and the resume line with the
  engine's reused figure (:6189-6198, :6001-6014) already read the DONE's fields - the card's missing line is
  this gate never having fired.
- `SAGUARO_PRIORITY_IDLE_S` stays 0: with this fix the 0 is earned in the shipping `--batch` config too (the
  round-4 MAJOR's alternative, "keep 5 when the engine runs batch slots", is what the docs now tell operators
  to do when the engine does NOT report `layer_pause` - see the nits below).

## Fix 3 (test 42882ed25): fake batch engine for the concurrent-reader path

serve/test_priority.py: `QfnBatchRead` (batch=1, keeps_stopped_read, `pauses_batch_read` like the driver sets
it from the engine's INFO, can_pause=False) reads chunk by chunk on the cut leg, and the cut `GeneratorExit`
lands in the MIDDLE of the next chunk: the kept read counts that chunk through its end (the engine-side rows
resume inside it, invisible over the pipe), the DONE says cancel with `paused_layer/paused_chunk/paused_at`,
and the next call continues from the end of that chunk with the reused figure. Two tests in `ReadsPreempt`:

- `test_a_qfn_batch_read_pauses_between_layers_and_continues_at_the_pause`: normal-first order, exactly one
  pause inside a chunk, the server line naming `inside chunk ... at layer 7`, every chunk read exactly once,
  the resume carrying the engine's `M of P tokens reused`. RED before the gate change (verified by reverting
  it: no pause line, the read running to its end - the manager's card shape), GREEN after.
- `test_a_batch_read_of_an_engine_without_the_layer_pause_is_never_cut`: with `pauses_batch_read = False`
  nothing is cut, no pause line, one call per read - the old rule pinned for engines that would drop the read.

## Nits (d2f6a7653)

- round 4, NIT "card-script expectations for the batch lane": tools/card_tests/layer_pause.sh's header now
  states the batch lane must pass the SAME checks (its "on a thread of its own" line with no
  "prompt read paused" line is the rc124-cand3 FAIL) and phase L names the batch lane when the engine ran
  concurrent reads.
- round 4, NIT "REPORT scope wording": REPORT-p125j-layer-pause.md was dropped from the tree by cand3 itself
  ("candidate 3: drop task reports"), so the stale "the shape the priority cut actually uses" sentence no
  longer exists to reword - this FIX-REPORT is the live document instead.
- docs/SETTINGS.md: SAGUARO_LAYER_PAUSE's row said "a SOLO QFN prompt read" - it now covers `--batch` (the
  concurrent reader, the INFO `layer_pause=1`, the server cutting batch reads), and the grace-back-to-5
  caveat (NIT 7) widened to an older engine whose batch INFO says no `layer_pause`.
- Cheap-and-cheap-only: nothing else from round 4's list was open; the OFF path (SAGUARO_LAYER_PAUSE=0), the
  locks (no new lock), the generated file (extractor run: byte-identical) and the no-secrets rule were
  re-respected.

## What ran

- `scratch/build-p125j-r5` (new, engines/qfn standalone, CUDA 13.1, arch 75): `saguaro-qfn` links (one
  pre-existing warning at :15880); QFN only, no GPU run (plan step 4's rule).
- `layer_pause_test` 43 checks, 0 bad (ctest-registered, run from the build dir).
- `python -m unittest serve.test_priority` - 82 tests OK, one module at a time; RED (gate reverted) / GREEN
  both verified for the new cut test.
- `tools/saguaro/extract_qfn_engine.py`: serve/qfn_engine.py byte-identical (engines/qfn/serve/server.py
  untouched).
- `bash -n tools/card_tests/layer_pause.sh`, `py_compile` on the Python touched.
- NOT run: `tools/card_tests/layer_pause.sh` itself - no card here (the plan excludes GPU runs); the manager
  reruns it on the live Q4 `--batch` config, where the layer-pause lane now prints the engine's "the read
  paused at chunk ... layer ..." line from the concurrent reader and the server's "prompt read paused ...
  inside chunk c at layer l" line.
