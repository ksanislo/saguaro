# REPORT - P124z: a LOW prompt READ no longer steps aside (candidate 2, 1.2.4-rc2)

Branch `saguaro/q1-p124z-rc2-read-preempt` off `saguaro/rc124-cand2`.
Card (mangonel 2026-11-27, `tools/card_tests`): candidate 1 (rc124-cand) DS4 and ik2 logged
`priority: a low prompt read paused at N of M prompt tokens for a normal-priority request`;
candidate 2 logs none at all on DS4 and cuts only low ANSWERS on QFN (`ref/rc1-ds4-engine-PAUSED.txt`,
`ref/rc1-ik2-engine-PAUSED.txt`, `ref/rc2-ds4-engine.txt`, `ref/rc2-qfn-layer-pause.out`).

## 1. The read-preemption decision in serve/server.py

The decision lives in `Service._sliced`, on the READ ticks (an engine call yielding `None`: a PP/BRP
line or a keep-alive), serve/server.py:6013-6035:

- the gate, serve/server.py:6021-6024 — a read may be stepped aside for a higher priority only when:
  `prio < PRIORITY_HIGH` (a high request is never cut) and `t0 is None` (nothing of the answer decoded
  yet) and not `pausing`/`emb`/`self.stopping`, and the ENGINE keeps a stopped read:
  * `(can_pause and multi)` - serve/server.py:6022: a PAUSE-speaking conversation engine (GLM: the
    READY `pause` word, parsed at serve/server.py:1584; a `PAUSE` stops the read at a chunk boundary
    and keeps the state, serve/server.py:6030-6032); or
  * `(keeps_stopped_read and not batch)` - serve/server.py:6023-6024: an engine whose STOP'd prompt
    read stays live at the token reached: the flag defaults False (serve/server.py:1525) and the READY
    parse gives it at serve/server.py:1587 (GLM line engine by name, maya-metal excepted),
    serve/server.py:1588-1589 (MM3, by its `INFO engine_kind=mm3`, solo and multi),
    serve/server.py:1590-1591 (any engine whose READY says the word `keepread` - that is saguaro-ds4,
    P124u/P124k; `src/ds4/ds4_main.cpp:757-770` prints it, and both its solo and its `--parallel`
    READY carry it), serve/server.py:1632-1636 (saguaro-ik2 SOLO only, P125l: its STOP lands between
    chunk passes and keeps the read; its `--parallel` line slot-resets a stopped reader, so multi-ik2
    reads run to their end) and serve/server.py:1637-1638 (hy4, solo).
  QFN (the batch engine) is EXCLUDED while `batch`: its class sets `keeps_stopped_read = True`
  (serve/server.py:9989 - the SOLO GEN path keeps the conversation at the token it reached,
  generate.cpp #1620) but the `not batch` term means a BATCH slot's read is never cut: a cancelled
  batch read is dropped ("a read that failed keeps none" - the BRA/BSTOP paths clear the slot), serve/server.py:9983-9997,
  pinned by `serve/test_priority.py::ReadsPreempt::test_a_batch_read_runs_to_its_end_even_for_a_high_request`
  (serve/test_priority.py:1026-1033). The cut itself: `cut = True; break` (STOP via `gen.close()`,
  serve/server.py:6034-6035) or `self.engine.pause()` for a PAUSE engine (6030-6032); the place is
  given up with `yield_place` (6143) and the pause line logs at 6150-6164 ("prompt read paused at N of
  M prompt tokens", P125j adds the chunk/layer suffix from the DONE's `paused_layer/paused_chunk/`
  `paused_at`, 6156-6162 - parsed by the generated driver at serve/qfn_engine.py:1236 from
  engines/qfn/serve/server.py:1681); the resumed call logs 5972-5992.
- how it KNOWS a low request is READING and where: `Service._read_position`, serve/server.py:5782-5790
  - the line engines' PP lines land in the engine's thread-local `_tls.pp` (the multi path,
  serve/server.py:2445-2459: one tick per PP line, `self._tls.pp = self.progress`, `(read, total,
  reuse, ms)`) and `_tls.own_pp` (the solo path, serve/server.py:2195-2217, and every driver's
  per-request read; QFN's batch slots report the same on BRP lines through the wrapper's
  `_admission_wait`, serve/server.py:10153-10233, QFN's solo on PP through `generate`, 10102-10151).
  `pos[0] < pos[1]` (the read unfinished) plus a `None` tick is what re-runs the decision.
- who it steps aside for: `FairLock.step_aside`, serve/server.py:605-630. A LOW runner steps aside
  whenever a normal/high request RUNS (`_idle_wait(PRIORITY_LOW)` returns -1 when a normal/high holds
  a place, serve/server.py:528-541) or WAITS (a higher entry in `_queue`), even beside a free place.
  A NORMAL one only for a waiting high that does not fit.
- the answer-tick cut (not a read) is ungated by the engine flags, serve/server.py:6059-6076 - which
  is why the QFN card run paused the low ANSWER while never touching the low READ
  (`ref/rc2-qfn-engine.txt:25`).

Commit (step 1) done.
