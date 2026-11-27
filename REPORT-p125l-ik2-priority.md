# REPORT - P125l: request priority for Inkling 2 Mini (saguaro-ik2)

Saguaro 1.2.5, deferred from 1.2.4. Worktree branch `saguaro/q1-p125l-ik2-priority`.

## 1. How the QFN and GLM drivers support step-aside, and what the ik2 engine offers

### 1a. The shared machinery (serve/server.py)

The decision is `Service._sliced` (serve/server.py:5754) on top of `FairLock.step_aside`
(serve/server.py:605): a LOW answer gives its place up whenever a normal/high request runs
or waits (a free place is no shield), a NORMAL one only for a waiting HIGH one that does not
fit, a HIGH one never.

- **An answer steps aside at its next token** (serve/server.py:5888-5899): no engine flag is
  consulted - on a token tick where `step_aside` says so, the driver either `pause()`s the
  engine (when `engine.can_pause`) or cuts the generator (`cut = True; break`), whose
  `finally: gen.close()` (serve/server.py:5916) sends the engine its STOP. The request queues
  again (keep-alives, the 10-min note), and resumes with `prompt + what it already wrote`, the
  budget and thinking budget reduced by what it got (serve/server.py:5912-5914, 6003-6035).
  Answer step-aside therefore already happens on every engine that honours a STOP - ik2 included.
- **A prompt read steps aside only where the engine keeps a stopped read**
  (serve/server.py:5844-5858): the gate is `(can_pause and multi)` or
  `(keeps_stopped_read and not batch)`; a PAUSE-able engine gets `pause()`, otherwise the read
  is cut (STOP, drain to its `DONE ... cancel`). An engine that would *lose* a stopped read
  gets no cut at all - its read runs to its end (serve/test_priority.py's
  `test_an_engine_that_would_lose_a_stopped_read_finishes_it`, the batch test).
  The driver flags: `can_pause` from READY's `pause` word, `keeps_stopped_read` from
  `pause` (not maya-metal, serve/server.py:1526), the INFO `engine_kind=` hy4 (:1560) and mm3
  (:1527), or the READY word `keepread` any engine may say itself (P124u, :1529).
  The resumed read logs the engine's own reused figure when it reports one
  (`engine.reused`, serve/server.py:5807-5815; the line drivers have none - numberless wording).

### 1b. QFN

Driver `_qfn_engine_class` (serve/server.py:9583): `can_pause = False`,
`keeps_stopped_read = True` - and `Service._sliced` cuts its read only while solo
(`not engine.batch`).

- **Pause/resume points**: an answer between tokens (the GEN loop), a solo read between the
  prompt's windows/chunks; a batch read never (its slot drops a cancelled read -
  "a read that failed keeps none", engines/qfn/src/program/generate.cpp:15135).
- **What the engine keeps**: the solo stop lands the session exactly at `pp_reached`
  ("cancelled request keeps its conversation: N tokens live", generate.cpp:22934-22941, #1620);
  a read cut INSIDE a window goes back to the deepest checkpoint of that read
  (P124b's `cut_read_landing`, generate.cpp:22945-22985). The K/V state, the read position and
  the checkpoints stay with the conversation; the next GEN with prompt + written tokens
  continues from it and prints `RESUME n` - the driver keeps it as `self.reused`
  (serve/qfn_engine.py:723, :2556-2559) for the "continued: N of M tokens reused" line.

### 1c. GLM (line driver with PAUSE)

READY carries `pause` -> `can_pause = True`; `pause()` (serve/server.py:2401-2412) writes
`PAUSE [rid][ keep]`, the engine ends the request a few steps later with `DONE ... cancel`,
keeping its state with what it read/wrote; the same prompt resumes from it. Solo GLM keeps a
cancelled read too (#573), which is what `keeps_stopped_read` at :1526 encodes; maya-metal
drops one, so its name is excluded.

### 1d. What the ik2 engine (src/ik2/ik2_main.cu, `saguaro-ik2 --serve`) offers

Its READY says only `stop` (solo) or `stop multi=N` (--parallel) - no `pause`, no `keepread`
(ik2_main.cu:10165-10166), so today the driver gives it neither flag (test_priority's
`test_another_line_engine_is_not_given_it`: "Inkling 2 drops a cut read in multi, audit 8.5").

**Solo (`READY <ctx> stop`)** - a STOP sets `g_cancel` (ik2_main.cu:10181-10182), looked at:

- **between tokens of an answer**: the decode loop checks it once per token and after each
  speculative window (ik2_main.cu:10968, :11029); the conversation `hist` already holds the
  prompt and every token pushed so far (:10894, :11015-11018, :11057), and its end-of-prompt
  snapshot survives, so a cut answer keeps the K/V through the prompt - the next request with
  prompt + written tokens re-reads only those written tokens (the fair-slice assumption,
  already how a cancelled ik2 answer behaves today).
- **between the chunks of a prompt read**: `g_cancel` is checked before each chunk pass, in the
  plain chunk loop and the layer-major (`T_sup`) path alike (ik2_main.cu:10815-10841) - a STOP
  never lands inside a chunk pass, so a read always stops at a chunk boundary. What is kept
  when it stops (:10870-10892): the latest snapshot at or below the prompt - snapshots exist at
  every finished turn's end and, on the resident one-conversation path (`keep_n > 0`:
  :9705-9707, `SAGUARO_IK2_KEEP` default 4, not with --parallel/KV-streaming/KV-Q4) with
  `SAGUARO_IK2_SNAP_MSG` on (default) and no drafter (or `SAGUARO_IK2_DRAFT_SNAP=1`), at the
  last user message's start `p_ms` (:10798-10809, taken at :10860-10866). So:
  - a read cut **inside the last user message** keeps everything before it (the snapshot) -
    the resume re-reads only that message;
  - a read cut **before the snapshot** keeps only what the conversation already held at the
    request's start - for the FIRST prompt of a new conversation (nothing held, no snapshot)
    the read starts from 0 again when it comes back;
  - the request never gets *wrong* results: the next GEN re-finds its resume point from the
    prefix + picture-sig match and the snapshots (:10676-10786), and its `DONE ... cancel`
    reports the `cached` figure of its own start (:10886). No re-read of what a snapshot holds.

**Multi (`--parallel`, `READY <ctx> stop multi=N`)** - `STOP <rid>`:

- a **writing** one keeps its slot's conversation: prompt + tokens already pushed
  (ik2_main.cu:10531-10534) - answer step-aside is safe there too (already today).
- a **reading** one is slot_reset: "a reader stopped: nothing of its prompt is kept"
  (ik2_main.cu:10494-10499). A read cut in multi LOSES the read (re-read from 0; harmless,
  but work lost) - so the read flag must NOT be given on a multi ik2.

*(Steps 2-4: implementation, tests and results appended below as they land.)*
