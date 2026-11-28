# REPORT - P124z: a LOW prompt READ no longer steps aside (candidate 2, 1.2.4-rc2)

Branch `saguaro/q1-p124z-rc2-read-preempt` off `saguaro/rc124-cand2`.
Card (mangonel 2026-11-27, `tools/card_tests`): candidate 1 (rc124-cand) DS4 and ik2 logged
`priority: a low prompt read paused at N of M prompt tokens for a normal-priority request`;
candidate 2 logs none at all on DS4 and cuts only low ANSWERS on QFN (`ref/rc1-ds4-engine-PAUSED.txt`,
`ref/rc1-ik2-engine-PAUSED.txt`, `ref/rc2-ds4-engine.txt`, `ref/rc2-qfn-layer-pause.out`).

## 1. The read-preemption decision in serve/server.py

The decision lives in `Service._sliced`, on the READ ticks (an engine call yielding `None`: a PP/BRP
line or a keep-alive), serve/server.py:6041-6063:

- the gate, serve/server.py:6049-6052 — a read may be stepped aside for a higher priority only when:
  `prio < PRIORITY_HIGH` (a high request is never cut) and `t0 is None` (nothing of the answer decoded
  yet) and not `pausing`/`emb`/`self.stopping`, and the ENGINE keeps a stopped read:
  * `(can_pause and multi)` - serve/server.py:6050: a PAUSE-speaking conversation engine (GLM: the
    READY `pause` word, parsed at serve/server.py:1593; a `PAUSE` stops the read at a chunk boundary
    and keeps the state, serve/server.py:6058-6060); or
  * `(keeps_stopped_read and not batch)` - serve/server.py:6051-6052: an engine whose STOP'd prompt
    read stays live at the token reached: the flag defaults False (serve/server.py:1534) and the READY
    parse gives it at serve/server.py:1596 (GLM line engine by name, maya-metal excepted),
    serve/server.py:1597-1598 (MM3, by its `INFO engine_kind=mm3`, solo and multi),
    serve/server.py:1599-1600 (any engine whose READY says the word `keepread` - that is saguaro-ds4,
    P124u/P124k; `src/ds4/ds4_main.cpp:757-770` prints it, and both its solo and its `--parallel`
    READY carry it), serve/server.py:1641-1645 (saguaro-ik2 SOLO only, P125l: its STOP lands between
    chunk passes and keeps the read; its `--parallel` line slot-resets a stopped reader, so multi-ik2
    reads run to their end) and serve/server.py:1646-1647 (hy4, solo).
  QFN (the batch engine) is EXCLUDED while `batch`: its class sets `keeps_stopped_read = True`
  (serve/server.py:10017 - the SOLO GEN path keeps the conversation at the token it reached,
  generate.cpp #1620) but the `not batch` term means a BATCH slot's read is never cut: a cancelled
  batch read is dropped ("a read that failed keeps none" - the BRA/BSTOP paths clear the slot), serve/server.py:10010-10025,
  pinned by `serve/test_priority.py::ReadsPreempt::test_a_batch_read_runs_to_its_end_even_for_a_high_request`
  (serve/test_priority.py:1026-1033). The cut itself: `cut = True; break` (STOP via `gen.close()`,
  serve/server.py:6062-6063) or `self.engine.pause()` for a PAUSE engine (6058-6060); the place is
  given up with `yield_place` (6171) and the pause line logs at 6179-6192 ("prompt read paused at N of
  M prompt tokens", P125j adds the chunk/layer suffix from the DONE's `paused_layer/paused_chunk/`
  `paused_at`, 6185-6190 - parsed by the generated driver at serve/qfn_engine.py:1236 from
  engines/qfn/serve/server.py:1681); the resumed call logs 6000-6020.
- how it KNOWS a low request is READING and where: `Service._read_position`, serve/server.py:5810-5818
  - the line engines' PP lines land in the engine's thread-local `_tls.pp` (the multi path,
  serve/server.py:2473-2487: one tick per PP line, `self._tls.pp = self.progress`, `(read, total,
  reuse, ms)`) and `_tls.own_pp` (the solo path, serve/server.py:2222-2244, and every driver's
  per-request read; QFN's batch slots report the same on BRP lines through the wrapper's
  `_admission_wait`, serve/server.py:10181-10261, QFN's solo on PP through `generate`, 10130-10179).
  `pos[0] < pos[1]` (the read unfinished) plus a `None` tick is what re-runs the decision.
- who it steps aside for: `FairLock.step_aside`, serve/server.py:605-630. A LOW runner steps aside
  whenever a normal/high request RUNS (`_idle_wait(PRIORITY_LOW)` returns -1 when a normal/high holds
  a place, serve/server.py:528-541) or WAITS (a higher entry in `_queue`), even beside a free place.
  A NORMAL one only for a waiting high that does not fit.
- the answer-tick cut (not a read) is ungated by the engine flags, serve/server.py:6087-6104 - which
  is why the QFN card run paused the low ANSWER while never touching the low READ
  (`ref/rc2-qfn-engine.txt:25`).

## 2. Which merge broke it - none: the decision replayed at every merge of rc124-cand2

The test (`serve/test_priority.py`, class `QuietReadsPreempt`, commit c32e0974e): a new fake LINE engine
(`QUIET_READ_ENGINE`, driven through the REAL pipe driver `StrataEngine` as a subprocess, like
`serve/fake_multi_engine.py` in test_multi): `READY 32768 stop [multi=2] [keepread]` (saguaro-ds4's words,
P124u/P124k), a long prompt read that says its FIRST `PP` line and then NOTHING for 3 s - saguaro-ds4's
DS4_READER hands its cards to a running answer between the read's pieces and emits no line of the read
while it waits (src/ds4/ds4_main.cpp:2149) - a STOP kept read (`STOP`/`STOP <rid>` ends the read at the
chunk, the next GEN of the same prompt says `RESUME <from>` and goes on there), conversations that
continue an answer past what was written. The test: a fresh low read, a short NORMAL request at a cut
that fits the engine's free place, runs and finishes entirely inside one silence - the card's
`ref/rc2-ds4-engine.txt` shape to the second - and asserts the read pauses for it and continues from the
engine's own reused figure.

* **rc124-cand2 (this branch's base): FAILS** - `AssertionError: 'priority: a low prompt read paused' not
  found` - exactly the card's "DS4 logs NO priority line at all".
* **Replayed at all eight merge commits** of saguaro/rc124-cand2 (c1f936711 mac/qfn-metal, 426721af5 p124v2,
  a446e35dc p125l, 1a0aeb418 p125e, 41fd2352c p125n, 32b2cf5a0 p125m, 21eafa313 p124k, 76637293b p125j)
  and at **saguaro/rc124-cand** (candidate 1): **it fails at every one**. There is no merge that broke it:
  a sha1 replay of the decision's own code across the whole series shows it byte-identical - the gate
  (`prio < PRIORITY_HIGH and t0 is None ...`, serve/server.py:6049-6052), `Service._read_position`
  (5782-5790), `FairLock.step_aside` (605-630) and the two engine-line waits (`self.lines.get(timeout=10)`
  solo, `q.get(timeout=poll_s)` multi) hash the same at all ten points (gate=8820dc11 readpos=cef251b3
  stepaside=2713f931 loops=84007dc1). The P124v2 lead does not hold: p124v2 reworked the read RATE
  (`_note_hb`/`_read_span_s`/`_own_read_rate`, serve/server.py:4623-4788) - nothing in the decision reads
  that state; the decision reads `_tls.pp`/`_tls.own_pp` only, which no merge touched.
* Why the card still differed: candidate 1's DS4 cut fired because its short normal **WAITED in the FIFO**
  for ~10 s across read ticks (`ref/rc1-ds4-engine-PAUSED.txt:9` "reading the prompt: 36 tokens, 10 s so
  far" - `step_aside` sees a waiter or a holder whenever it ticks at all); candidate 2's run
  (`ref/rc2-ds4-engine.txt`) shows the normal starting on a **free place** ~2 s before the read's next
  line, finishing ~3 s inside a silence, the next tick (6 s later) finding the FIFO idle with
  `SAGUARO_PRIORITY_IDLE_S=2` expired (`_idle_wait` 0, serve/server.py:528-541) - so the cut was never
  even attempted. The defect is the decision's DEPENDENCE on the engine's next line, and it predates the
  merge set (candidate 1 passes that card only by timing: its cut needed a tick to land while the normal
  was still around).
* The QFN half of the card (`ref/rc2-qfn-layer-pause.out`, TTFT 19.9 s, low ANSWER cut but never the READ)
  is not this race and not a merge regression either: with `--batch` the server never cuts a read at all
  - serve/server.py:6051-6052 `not batch`, serve/server.py:10010-10025 (a batch slot's cancelled read is
  dropped, the BRA/BSTOP paths keep none) - which is precisely the BLOCKER P125j's own re-review round 4
  recorded at its tip (d8e25fd95: "under --batch the server never cuts a read ... and the piped
  concurrent reader (generate.cpp:20892-21033) has no pause capture ... card FAIL: no STOP line, read ran
  37632 tokens straight through"). p125j merged into candidate 2 with that BLOCKER open; the engine-side
  piece (pause capture in the pipelined batch reader) is not in this tree, and cutting a batch read
  without it would make the engine abort the request instead. Out of reach here (no cards, no nvcc; the
  plan says no GPU runs), and the answer tick (`ref/rc2-qfn-engine.txt:25` the low answer did pause)
  shows the rest of the machinery intact.

## 3. The fix (in the candidate - it is the interaction), both behaviours kept

serve/server.py (the only code file changed; `serve/qfn_engine.py`/`engines/qfn` untouched, the extractor
had nothing to run):

* `READ_TICK_S = 0.25` (serve/server.py:1298-1306) - the cadence QFN's own read heartbeat already beats
  its reads with (an RHB line every ~0.23 s), so a line engine's quiet read now gets the same eyes on it.
* `StrataEngine._read_tick(otherwise)` (serve/server.py:2140-2156): the wait for the engine's next line,
  shortened to `READ_TICK_S` while THIS thread's prompt read is unfinished - the same `(_tls.)pp` /
  `own_pp` position `Service._read_position` looks at (5810-5818), so the wake and the decision agree by
  construction; the untouched `otherwise` keeps today's cadence everywhere else (the answer's token
  stream, reads nothing in flight, the `REQUEST_QUIET_S`/stall checks keep their same wall-time triggers
  - only the frequency of looking goes up, and what a look costs is what the 10 s Empty wake already
  paid: the cancel check, the throttled stall sample, one None keep-alive).
* The two waits: solo `self.lines.get(timeout=self._read_tick(10.0))` (serve/server.py:2187), multi
  `q.get(timeout=self._read_tick(poll_s))` (serve/server.py:2419). On the None tick the existing R122
  gate (6049-6063) re-decides and the cut proceeds exactly as it does for a chatty engine: STOP/PAUSE,
  the `prompt read paused at N of M prompt tokens` line (6183), `yield_place`, and the resumed call's
  `prompt read continued: K of M tokens reused` from the engine's own figure (6000-6020).

Behaviours kept:

* the gate itself is untouched - an engine that drops a stopped read (no keepread word, maya-metal,
  ik2 `--parallel`, the QFN batch slots) is still never cut: `test_a_quiet_low_read_runs_on_when_a_stopped_read_would_be_lost`
  and `test_a_batch_read_runs_to_its_end_even_for_a_high_request` both stay green;
* every pre-existing test of the four modules below passes unchanged (a faster wake moves no assertion:
  the keep-alive None ticks were already an allowed part of a quiet read - the card's own log shows
  second-spaced "reading the prompt" lines during silences).

## Files

* serve/server.py - READ_TICK_S + _read_tick + the two waits (step 3 commit 149bda5d3).
* serve/test_priority.py - QUIET_READ_ENGINE + class QuietReadsPreempt (3 tests; step 2 c32e0974e, pacing
  in step 3 149bda5d3).
* REPORT-p124z-rc2-read-preempt.md - this file.

## Tests run (one at a time, per the plan; no GPU anywhere)

* serve.test_priority: **80 OK** (77 before + the 3 new; the discriminator fails on the base and on all
  eight merges + candidate 1 without the fix, passes with it; verified RED again by reverting just the
  two waits on top of the fix).
* serve.test_read_rate: **34 OK**.  serve.test_read_rate_true: **10 OK**.  serve.test_server: **183 OK**.

## Open questions

1. QFN `--batch` low reads (the card's `qfn-layer-pause` FAIL, 19.9 s TTFT): server-side `not batch`
   stays until P125j's recorded BLOCKER gets its engine half - pause capture in the pipelined concurrent
   reader (generate.cpp:20892-21033 per d8e25fd95) plus a BRA line that says a stopped batch read is
   kept - and then a gate keyed to THAT report, not to `batch` itself. Needs a card and a CUDA build;
   nothing in this tree can test it.
2. `READ_TICK_S` is a constant, not a knob - the plan named no setting; if a quiet read's four wakes a
   second ever need tuning, it wants an env beside SAGUARO_FAIR_READ_S, and docs/SETTINGS.md a row.
3. The card host should rerun `tools/card_tests/lost_read.sh` (DS4 and ik2) and the RC gate's
   `layer_pause.sh` on this branch: 1 and 2 should now cut at the next quarter-second tick whatever the
   config; layer_pause stays FAIL on batch until question 1 lands - the candidate note should say so
   rather than let the merge re-litigate it.
