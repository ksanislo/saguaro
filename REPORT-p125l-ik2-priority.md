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

## 2. Implementation: the ik2 driver gets the flag (solo only)

One change, in serve/server.py (`StrataEngine.__init__`, the ik2 start-up block next to the
hy4 one): an engine whose INFO line says `engine_kind=ik2` is now given
`keeps_stopped_read = True` **when its READY said no `multi=`** (solo). With the flag, the
existing `Service._sliced` read preemption (serve/server.py:5844-5858) cuts a LOW ik2 prompt
read with a plain STOP tick at a `PP` line whenever `step_aside` says so, and the resumed call
re-sends the same prompt - the engine re-finds its resume point at its kept snapshot itself.
No engine change (this box has no card; the engine is used exactly as it is, "with what the
engine has").

What this means per mode, with what the engine really keeps:

- **Answer, solo and --parallel**: already worked (the token-tick step-aside is not gated on
  any flag, and saguaro-ik2 honours STOP between tokens in both loops); the cut keeps the K/V
  through the prompt (solo) / through the prompt plus what was written (multi slot hist), and
  the resume continues from there - no prompt re-read.
- **Read, solo**: the flag given. The engine sees STOP only *between* chunk passes - a chunk
  pass is never split, so a mid-read stop is always a between-chunks stop (said here per the
  plan). Resume continues from the latest snapshot (the last message's start once past it,
  else the conversation as it was at the request's start); the K/V held at that point is not
  re-read. **Where a mid-read stop loses work**: a read cut before any snapshot of this
  request was taken - the first prompt of a new conversation before its `p_ms` snapshot - comes
  back and reads again from 0; nothing breaks (the engine's prefix match recomputes honestly),
  the read is just not further along, and the log line keeps the numberless wording ("continues
  from where the engine kept it") - the driver never claims a reused figure the engine did not
  report (the engine reports none over the pipe: no RESUME line, and PP's `cached` is its
  start's figure).
- **Read, --parallel**: deliberately NOT given (`if not self.multi`). The engine's
  `STOP <rid>` on a reading request slot_resets its slot - "a reader stopped: nothing of its
  prompt is kept" (ik2_main.cu:10494-10499) - so cutting it would lose the read, the exact
  shape P124b says never to cut. Its reads run to their end, as the existing
  `test_another_line_engine_is_not_given_it` recorded.

Hy4 and MM3 are untouched (serve/server.py:1526-1530, :1560-1561 unchanged); no READY-word
(`keepread`) claim was added to the engine.

## 3. Tests (serve/test_priority.py) and results

New fake ik2 driver `Ik2Read` (serve/test_priority.py) models saguaro-ik2 solo as the line
driver sets it up (`READY ... stop`, no PAUSE, INFO `engine_kind=ik2`): a `PP` tick per chunk;
a close (the driver's STOP) is seen only between chunk passes; the engine rolls the read back
to the latest snapshot it kept (one every 2nd chunk here - the real engine's is the last user
message's start), never below what the conversation already held; no reused figure over the
pipe. New tests:

- `ReadsPreempt.test_an_ik2_solo_read_steps_aside_for_a_normal_and_continues_at_its_snapshot`
  - the low read pauses for a waiting NORMAL one, the normal answers first, the low answer
  arrives whole; the resume starts at the snapshot (> 0, on a chunk boundary, not at 0),
  the chunks the snapshot held were read exactly once, later ones at most twice; the pause
  line names the prompt tokens, the continue line keeps its numberless wording.
- `ReadsPreempt.test_a_high_request_goes_first_and_the_ik2_read_continues` - same race with a
  HIGH request: its answer ends before the low read's, which continues from its snapshot.
- `ReadsPreempt.test_an_ik2_read_cut_rolls_back_to_the_kept_snapshot` - the fake's engine
  semantics without a race: cut after chunk 3 -> back to chunk 2's snapshot (not 0, nothing
  half-read kept); the resume goes on from there; the snapshot's chunks read once, the dropped
  chunk exactly twice.
- `ReadsPreempt.test_an_ik2_request_that_never_waits_is_unchanged` - alone on an idle server:
  one engine call, every chunk exactly once, no `priority:` line.
- `ServicePreempts.test_a_low_ik2_answer_steps_aside_and_resumes_from_its_token` - the
  between-tokens half (never gated on a flag, and what the card test mirrors): a low ik2
  answer cuts at a token for a high one and resumes with prompt + written tokens.
- `TheReadKeepDriver.test_the_ik2_driver_is_given_it_solo_but_not_parallel` replaces
  `test_another_line_engine_is_not_given_it` (its old truth, audit 8.5, now lives in the
  multi leg of the new test): the stub engine saying `engine_kind=ik2` gets
  `keeps_stopped_read` on `READY ... stop` and does NOT on `READY ... stop multi=2`. An
  unnamed, un-worded engine still gets nothing (`some-new-name`).

Results (`/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_priority`):

- before the change: 68 tests, 1 failure - only `test_another_line_engine_is_not_given_it`
  (the ik2 solo leg), exactly the behavior the task changes.
- after: **74 tests, OK** (93 s); the three race-carrying classes re-run 3x: 36 tests OK each
  time. No ik2 driver's own test module exists (serve/test_inkling2_frontend.py is the chat
  template's, not the driver's), so serve.test_priority is the whole run the plan names.
- `py_compile serve/server.py` clean.

## 4. Open questions / owed

- The card test after (manager, mangonel inkling-pairs or mortar's Inkling config): a low
  answer steps aside for a normal one and resumes with no re-read. Not run here (no card).
- Whether --parallel Inkling reads should ever step aside needs an engine change (the
  `STOP <rid>` reader path would have to keep the slot's kept prefix instead of
  `slot_reset`, ik2_main.cu:10494-10499); out of scope, the driver deliberately does not cut
  them.
- The continue line for a resumed ik2 read stays numberless. The engine does print its
  `cached` figure on PP lines, so a follow-up could surface it as the reused figure; left
  alone (the P124b wording rule: never claim a figure the engine did not report as its
  resume).
- docs/SETTINGS.md's priority section still lists only QFN/GLM as the read-keeping engines
  (it already skipped hy4/mm3/ds4 from P124i/P124j/P124u); left untouched per the plan, same
  as those cards.
