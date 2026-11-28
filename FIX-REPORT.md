# P124v2 FIX-REPORT - manager review round 8 (FIX-FIRST: REVIEW-RESULT.md, reviewed at fde18a4a2/38b3ccce6, 2026-11-28)

Review: REVIEW-RESULT.md - FIX-FIRST with one blocker (finding 1: the round-6 numerator exclusion is
released by the next admission hook), one Low to state (finding 2), two Infos (3: the documented
OFF-path; 4: a stale test docstring), and two nits (5: the restart-tick blank; 6: the unlocked share
trackers). The FIX-FIRST is taken with a committed red-before-green-after gate; finding 4 taken;
finding 2 documented where the review said it was stated nowhere; 3, 5 and 6 need no code and are
answered below. Every step committed separately. Modules only the read-rate ones the plan names, one
at a time: `serve.test_read_rate_true` 18 + `serve.test_read_rate` 34 + `serve.test_read_heartbeat`
19, green at the code commit and again at this file's tree. `tools/saguaro/extract_qfn_engine.py`
re-run reproduces `serve/qfn_engine.py` byte-identically (`git status` clean; the driver was not
touched this round). No engine C++ touched, no `serve/qfn_engine.py` hand-edit. Line references are
re-grepped at THIS tree (the two `serve/server.py` commits shift the file by +13 and +11 lines; the
moved references are listed under finding 2's commit).

## Finding 1 (FIX-FIRST, P1): `_read_start` released the restart exclusion - TAKEN, committed gate

Confirmed independently: through the committed harness with the round-6 gate test's exact shape
(`scratch/r8_red_check.py`: 9,600 tokens / 400 tok/s restarted at the engine's own 300 tok/s), the
hook between the restart and the retry made the retry's six steps read **3,500 / 2,700 / 2,220 /
1,900 / 1,671 / 1,500 tok/s where the engine says 300** - the review's numbers exactly, re-derived
through `Service._own_read_rate` after `Service._read_start`.

Fix - the review's FIRST option, exactly as directed ("do not release the exclusion in `_read_start`"):

- serve/server.py `_read_start`: the `stat.read_count_reset = 0` line is GONE (the comment at the old
  site now says why: `_admission_wait` fires the hook on EVERY admission - `:9921-9923`, installed
  `:6340` - and a plain slice leg is unchanged anyway, the exclusion being 0 without a restart).
- Why it stays coherent (as the review argued): the snapshot and the closed intervals it counts
  against both survive the hook - `_read_start` only closes the open leg and opens the next one - and
  `min(excl, closed)` (the review's own phrase: "already tracks pruning") keeps the subtraction from
  over-reaching as intervals age out of `self.reads`; what ages out with the history then reads LOW,
  never the spike, and the engine's own pair floors the denominator so the readout stays under the
  engine's figure. A SECOND restart re-baselines the snapshot to the full closed total (the
  `_read_counted` it takes at that moment includes everything the previous snapshot covered), so the
  exclusion is monotone over a request's life, and the stat is a per-request object that dies with
  the request - no cross-request leak.
- `_own_read_rate`'s docstring and the `RequestStat` field comment now state the survival.
- Committed gate, `serve/test_read_rate_true.py`
  `CounterFallsRestartOrCorrectDown.test_the_restart_exclusion_survives_the_next_admission_hook`
  (verified RED before the fix - the hook zeroed `read_count_reset` and the steps read 3,500..1,500 -
  green after): exclusion alive after the admission, readout under 1,000 at every retry step, ending
  at 300.0 = the engine's own 2,400 tokens / 8.0 s, the chat row under 1,000 and no `peak_prefill`
  above it (the review's point that the live readout feeds the peaks - `read_eng_n` 2,400 >=
  PEAK_MIN_READ_TOKENS over the retry's pair - is gated here). Commit d4c2f245f.

## Finding 2 (Low, documented): sliced legs never get the in-flight share - TAKEN (documentation)

Agreed with the review's own grading: direction-safe (whole-unit booking, never above the engine's
pair, the same conservative class the report already owns in open question 3), but "not stated
anywhere" - and it is not the right trade to "fix" by shifting pos0/new/reused too (that would make
the guard claim a share for a unit the reporting card's line does not identify). Now stated in both
places the review's finding lives:

- a comment at the shift site, `serve/server.py:9951-9962`: only `done` shifts by base, the guard
  cannot hold once a slice's base > 0, the leg books whole units.
- REPORT-p124v2-read-rate-true.md open question 3: the same, plus the honest scope of the plan's
  never-0-after-the-first-RHB property (whole reads and a read's FIRST leg, not a slice's legs after
  their BRA).

Commit 50d521cf8 (same commit re-greps the drifted line references the round-8 fix moved: the hook
fire/install refs above, and the REPORT's `_note_hb` 4621->4635, `_own_read_rate` 4517->4519,
`_read_span_s` 4501->4503, `_read_close` 4474->4476, restart zeroing 4767->4781).

## Finding 3 (Info, documented, accepted in earlier rounds): the OFF-path is not byte-identical - NO ACTION

As the review records it: stated plainly in REPORT:152-159 already, every deviation safe in direction,
"recorded, not blocked". Nothing this round touches the field-gating or those four unconditional
paths; the round-6 answer stands (gating the readout semantics behind the fields would carry the
same class of bug twice).

## Finding 4 (Info): stale causation in the `CardTrueReadout` docstring - TAKEN

`serve/test_read_rate_true.py:373-383`: the docstring no longer claims its cut-end BRP steps land
below the booked share. It now states what the committed synthesis does (PASSES = 60, beats strictly
before each cut end: the booked share maxes ~77 tokens BELOW the cut, no step ever lands under
`read_pos`, the monotone / never-re-opened gates pass because no fall occurs, the capture held no RHB
lines - the 9,881's exact cause stays unproven, round 5), keeping the fall-vs-restart explanation as
the conditional it is ("a whole-cut step that DID fall below used to fire the restart branch"). The
fall-correct-down branch remains covered where the review says:
`test_a_step_lagging_a_booked_share_correct_down` and `SplitReadFollowsEngineFigure`. Docstring only.
Commit bac328d24.

## Finding 5 (Nit): transient blank at the restart tick - NO ACTION (by design, and the same one round 6 recorded)

A restart RE-OPENS the read: the counter resets, a fresh interval opens, and the retry's own pair is
0/None until its first step - the engine's own cumulative figure has none either at that instant. A
readout of 0.0 there is exactly what a brand-new read shows before its first beat/step, which is the
committed design: the readout "FALLS with the silence instead of freezing at the last beat" - holding
the pre-restart tick's figure across the restart would freeze a dead attempt's rate, and its honest
value is unknowable from the line stream (the review's own probe saw it last until the retry's first
step at the engine's ~0.23 s cadence). The round-6 FIX-REPORT accepted the identical transient ("for
one instant at the restarting step itself ... never an inflated figure"); finding 1's fix removes the
danger (the inflated figure), leaving only the drain-shaped blank the gates already tolerate by
polling where the retry's steps live. No change.

## Finding 6 (Nit): `read_share_unit`/`read_split_share` tracked in `_note_hb` without `status_lock` - NO ACTION (as assessed)

The review's analysis stands and nothing this round changes it: per-request beats and steps run on
one consumer thread, every write there is a single GIL-atomic attribute assignment, and the worst
case is one lost reset or one stale tracker - direction-safe. Making it lock-clean would need
`status_lock` inside `_note_hb` around trackers only, or re-plumbing the beat path, against the
file's verified invariant that the plain `status_lock` is NEVER nested (`_note_hb` calls
`_note_read`, which takes it internally). Benign, carried from round 7; no change.

## Manager gates (REVIEW-BRIEF line)

Not re-measured by hand this round: the committed replays ARE the gates and they are green in the
18-test module run at both the code commit and this tree - `MangonelTrueReadout` (never 0/blank after
the read's first RHB, time-average within 5% of the engine's 1,567.8, peak <= 1.25x, closes 27,254)
and `CardTrueReadout` (same shape on the card capture). The P1 fix cannot move them (no restart fires
in either replay - the monotone / never-re-opened assertions are part of what passes).

## Modules (one at a time, this checkout)

- serve.test_read_rate_true - 18 green (17 prior + the new admission-hook gate), ~352 s; re-run after
  the final comment/report commit, again 18 green
- serve.test_read_rate - 34 green (module untouched), re-run at the final tree: 34 green
- serve.test_read_heartbeat - 19 green (module untouched), re-run at the final tree: 19 green
- tools/saguaro/extract_qfn_engine.py - reproduces serve/qfn_engine.py byte-identically (clean status)
- scratch/: one new probe this round, `scratch/r8_red_check.py` (the pre-fix rate printout; `scratch/`
  is gitignored)
