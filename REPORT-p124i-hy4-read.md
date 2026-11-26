# P124i - Hy4-Flash keeps a stopped prompt read (request priority, audit item 2)

ref/REPORT-p124c-prio-audit.md §5.2 and §8.2 item 2: the Hy4 engine already keeps a prompt read that a STOP cuts
between chunks, but `serve/server.py`'s read preemption never exercised it. Now the HY4 line driver reports
`keeps_stopped_read`, so a low HY4 prompt read steps aside for a waiting higher request and continues at the chunk
it kept. Two files changed; no C++, nothing on a GPU.

## 1. Verified: a STOP during a Hy4 prompt read keeps what was read, and the next GEN reuses it

(src/hy4/hy4_main.cpp, the `--serve` GEN path; the audit's §5.2 line refs hold up.)

- A `STOP` line on stdin sets the engine's stop flag: the reader thread, :430-432 (`stop.store(true)`).
- The prompt is read in chunks and every chunk fed to the caches is appended to `state` ("the tokens in the
  caches, positions 0 ..", :969): read loop :1079-1096, `state.insert(...)` per chunk at :1086.
- The STOP is honored **between chunks** and ends the read: :1092-1095 (`if (in.stop.load() && !last) {
  cancel = true; break; }` — a STOP inside the last chunk is not taken, the `&& !last`).
- The cancel path keeps the read: :1106-1110 prints `DONE ... cancel ...` and continues the request loop **with no
  `state.clear()`** — the clears are only the engine-error path (:1101) and a picture error (:1060). So the
  conversation stays live in the caches at the chunk boundary reached.
- The next GEN of the same prompt reuses it: `from` = the shared start of `state` and the prompt's ids, :1024-1025,
  and the read loop starts at `a = from` (:1079) — after a kept read `from` is the chunk boundary reached, so only
  the rest is read (unless the sliding windows are gone: `m.resumable(from)` check :1035).
- The DONE's reused figure: the protocol is `DONE ... <reused> 0 0` (:60-61); the finished request's DONE reports
  `from` there (:1273-1274), so the resumed GEN's DONE carries the chunk boundary as its reused figure (the PP
  lines report `from` the same way, :1089). The **cancelled** read's own DONE reports the old base as reused
  (:1107) — as the audit's §10.2 noted, not the partial end.

So it does keep the read: steps 2 and 3 proceed. (Nothing changed C++-side; the keep existed, unexercised — the
server's read preemption gated to `can_pause and multi` or `keeps_stopped_read and not batch`, and the HY4 driver
reported neither.)

## 2. Changed: the HY4 line driver reports `keeps_stopped_read` (serve/server.py)

serve/server.py:1550-1551, in `StrataEngine.__init__` — the line driver every HY4 start gets (:9822):

    if self.info.get("engine_kind") == "hy4":
        self.keeps_stopped_read = True   # R122: a STOP between its prompt's chunks keeps the conversation live at the chunk reached (solo, hy4_main.cpp:1086, :1106-1110)

(one if, one statement, one comment line; nothing else in the file changed.)

The signal is the engine's own INFO line (`INFO engine=hy4 engine_kind=hy4 ...`, hy4_main.cpp:932), which the
driver already parses into `self.info` (:1506-1509; the same mechanism the ik2 branch at :1545 uses). The gate in
`Service._sliced` (:5728-5731, the audit's :5726-5729 before these two lines landed) then fires its second arm —
`keeps_stopped_read and not batch` — for HY4 exactly as for solo QFN: `can_pause` is false (its READY says `stop`
only, hy4_main.cpp:958) so the read is cut with STOP (`gen.close()` drains to the DONE cancel, :5800) and the
resume re-sends the same prompt, which the engine continues from the chunk it kept. HY4 never sets `batch`, so the
`not batch` arm always applies; GLM/MM3/DS4/IK2 are untouched.

## 3. Tests (serve/test_priority.py)

- `Hy4Read` (after the `PausingRead` fake): `ReadEngine` shaped as the HY4 driver now is — `keeps_stopped_read`
  True, `can_pause` False, `multi` 0, `info = {"engine_kind": "hy4"}`.
- `ReadsPreempt.test_a_hy4_solo_read_is_cut_for_a_normal_request_and_resumed`: a low `long one` read in progress
  gives its place to a waiting **normal** `short one` (the gate needs only `prio < HIGH` plus the FIFO's
  step_aside rule); asserts the normal answer goes first, the low one completes with every prompt chunk read
  exactly once (`engine.ticks == CHUNKS` - what the engine kept was not read again), the low prompt arrives twice
  with the same ids (nothing written yet at the cut), and the log shows "priority: a low prompt read paused ...
  for a normal-priority request" and "the low prompt read continues". `race_read` grew a `waiter=` parameter
  (default HIGH, existing tests unchanged).
- `TheReadKeepDriver`: starts `serve/test_priority.INFO_KIND_ENGINE` (a fake line engine, the temp-exe pattern of
  serve/test_multi.py's `AWedgedRequest`) through the real `StrataEngine`: `test_the_hy4_driver_reports_keeps_stopped_read`
  (INFO engine_kind=hy4 → `keeps_stopped_read` True, solo: no PAUSE, multi 0) and
  `test_another_line_engine_is_not_given_it` (engine_kind=mm3 → not given it; MM3's turn is audit §8.3).
- Module docstring notes the HY4 case next to QFN's and GLM's.

## Tests run

- `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_priority` — **58 tests, OK** (78 s; was 55 before this
  plan, +3 new). Python 3.13.5.
- `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_fair` — **14 tests, OK** (10 s).
- One more run filtered to the two new classes (`ReadsPreempt`, `TheReadKeepDriver`): 9 tests, all ok.
- Only the two modules the plan names; no C++ built, no GPU used, no live HY4 run here.

## Files

- serve/server.py — +2 lines (the driver flag, §2). Commits: step 1 be7432909 (report: the verification), step 2
  222af3eb8 (the flag), step 3 0f19dafb9 (the tests).
- serve/test_priority.py — new fake, two tests + the driver test class + the `waiter=` parameter (§3).
- No C++ changed; nothing else touched.

## Open questions

1. Live-HY4 parity (audit §10.2): the flag was flipped on the code reading above plus the mock-engine round trip;
   a DONE-parity run of a real cut→resume on a GPU (the resumed GEN's `reused` figure at the chunk boundary, and
   whether penalty windows / drafts treat the partial state right, not just the KV) still belongs to the manager.
2. The `&& !last` gap (hy4_main.cpp:1092): a STOP landing during the prompt's LAST chunk is not taken — the read
   finishes and the cut lands on the decode's first tick instead. Costs ≤ one chunk of the ~1 s goal; the plan
   says note, don't fix.
3. A cut read whose sliding-window state no longer stands at the kept point (`m.resumable(from)` false, :1035)
   re-reads from 0 — the preemption still helps, the resume just costs the whole prompt again.
4. Audit §10.4 stands: HY4 has no in-RAM conversation keep — a paused request whose cache slot is taken by the
   next request re-reads from the HY4_SLOT_DIR folder (or cold). Acceptable for 1.2.4, or park-on-pause first?
5. The cancelled read's own DONE reports `reused=from` = the old base (:1107), so the web stats for a paused-then-
   resumed request show the base, and only the final DONE shows the chunk boundary — same as MM3, cosmetic.
