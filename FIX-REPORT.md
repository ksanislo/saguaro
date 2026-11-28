# FIX-REPORT - P125j re-review round 5 (FIX-FIRST, manager card FAIL on saguaro/rc124-cand3)

The review: `tools/card_tests/layer_pause.sh` on the live Q4 `--batch` config still FAILs cut 1 - no
"prompt read paused" line, the normal request's first token after 11.2 s, the low read ran 37632 tokens
straight through "on a thread of its own" (scratch/card-rc3-test.txt, scratch/card-rc3-engine.txt).
P124z's READ_TICK_S makes the server re-decide the preemption every 0.25 s and DS4 keep-read passes;
the QFN read never pauses. Fix items 1-3 below, each its own commit; line numbers as of this branch
after `git merge saguaro/rc124-cand3` (P124z's tick in, P124t2's readiness loops in).

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

## Fix 2 (engine + server): the concurrent reader takes the layer pause and keeps what it read

## Fix 3 (test): fake batch engine for the concurrent-reader path
