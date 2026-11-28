# MANAGER: FIX-FIRST (card, mangonel, 2026-11-27 21:11-21:15 PST, candidate saguaro/rc124-cand3 = dev + this branch
# + P124z's read-preemption tick fix (READ_TICK_S, serve/server.py) + others)
tools/card_tests/layer_pause.sh on the live Q4 config, candidate 3: STILL FAIL at cut 1 (scratch/card-rc3-test.txt):
  -- cut 1 at read position 32256: a short normal request; its first token after 11.2 s
  FAIL: no 'prompt read paused' line for cut 1
P124z made the server's read-preemption decision tick every 0.25 s during a read - DS4 keep-read now PASSES with it
(3 cuts paused + resumed, reused 32768 / 49152 / 73728) - but QFN under --batch still reads a long prompt on the
CONCURRENT READER ("read N tokens on a thread of its own ... concurrent", scratch/card-rc3-engine.txt), and that read
never pauses: the same failure as the 11-27 10:35 card (rounds 5-6 passed review without reaching that path).
Fix (commit each):
1. FIX-REPORT: from serve/server.py's read-preemption STOP for a low request under the QFN batch driver
   (serve/qfn_engine.py) to the engine's concurrent reader thread (engines/qfn/src/program/generate.cpp - the
   "thread of its own" read): the exact file:line where the stop is (not) seen, with this candidate's code.
2. Make the concurrent reader take the layer-boundary stop (P125j's pause point) and keep what it read; the server
   logs "prompt read paused" and the normal request starts within ~one layer pass.
3. A test with the fake batch engine for the concurrent-reader path (fails before, passes after). Build QFN only.
Base your work on saguaro/rc124-cand3 (git merge saguaro/rc124-cand3 into this branch first) so P124z's tick is in.
