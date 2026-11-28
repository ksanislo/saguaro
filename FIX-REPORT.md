# P124v2 FIX-REPORT - manager review round 6 (FIX-FIRST: REVIEW-RESULT.md, reviewed at c75f80547, 2026-11-28)

Review: REVIEW-RESULT.md - FIX-FIRST with one blocker (P1: after a mid-read restart the cumulative
readout divides the OLD attempt's tokens by the NEW read's span), two carried/documented findings
with no action (2, 3), one accepted-design observation (5), and a cheap nit (4: the report's line
references had drifted). Every FIX-FIRST finding taken; the nit taken; each fix committed
separately. Modules only the read-rate ones the plan names, one at a time:
`serve.test_read_rate_true` 17 + `serve.test_read_rate` 34, green at the code commit;
`tools/saguaro/extract_qfn_engine.py` re-run reproduces `serve/qfn_engine.py` byte-identically
(the driver was not touched this round, `git status` clean after). No engine C++ touched. No
`serve/qfn_engine.py` hand-edit.

## P1 (FIX-FIRST, blocker): post-restart the readout divided the abandoned attempt's tokens by the new span

Confirmed, and reproduced IN A COMMITTED TEST (the review's own ask: no committed test gated the
readout's VALUE after a restart): a 24 s / 400 tok/s read (9,600 tokens, its beats' own ms said
400 the whole way) retried at 300 tok/s showed, pre-fix, 6,800 tok/s 1.5 s after the restart -
`_own_read_rate`'s no-pair path `n_own / (now - engine_t0)` with engine_t0 just re-baselined and
`n_own = _read_counted` still summing the closed interval's 9,600 - and 3,500..1,500 on the
retry's own steps (the pair floor `max(read_eng_ms, span)` cannot hold a numerator that carries
tokens the pair never saw). Same failure class and shape as the review's through-the-real-Service
spike (49,000 at the retry's step, 3,363 six seconds in).

Fix - the review's FIRST option (baseline the numerator; the second, keeping the span consistent
with the counted, collides with the two committed assertions it names: the `engine_t0`
re-baseline pin and `LateStepAfterClose`):

- serve/server.py `RequestStat`: new `read_count_reset` (init 0) - the tokens the last restart
  dropped from the readout's numerator.
- The restart branch (`_note_read`): sets `stat.read_count_reset = self._read_counted(stat)` with
  the reset (the closed intervals' counted tokens at that moment - the failed attempt's included).
- `_own_read_rate`: subtracts `min(read_count_reset, the closed total still in the history)` from
  `n_own` (same scan as `_read_counted`). `min()`: a closed leg pruned from the history total
  AFTER the reset was inside the reset figure - subtracting the whole reset figure after a prune
  would read low; the closed total recomputed each call is the honest ceiling. The numerator and
  the restarted span now cover the same attempt: the pair-floor sentence in the docstring holds
  again after restarts.
- `_read_start` (the driver hook) CLEARS the exclusion: that hook does not zero `read_leg_ms`, so
  the closed legs' wall time stands in the denominator and their tokens must stand in the
  numerator too (slice legs / a hooked retry stay consistent). The exclusion belongs to the
  span-zeroing restart and dies with the next admission.
- `_read_counted` UNCHANGED - the history figure every test reads back stands (the new test pins
  it: 10,200 after a 9,600 + 600 restart); only the rate's numerator is rebaselined.
- Accepted as-is: for one instant at the restarting step itself (engine_t0 == now, the pair reset
  and not yet re-set by the step) the readout shows 0.0 - a transient blank that predates this
  fix (the branch has always reset the pair), never an inflated figure; the committed gate polls
  1.5 s later where the review's spike lived.
- Committed gate, `serve/test_read_rate_true.py`
  `CounterFallsRestartOrCorrectDown.test_a_restart_rebaselines_the_rate_numerator_with_the_span`
  (verified RED before the fix at 6,800): the readout under 1,000 and not blank 1.5 s after the
  restart, under 1,000 at every one of six retry steps, ending at 300.0 = the engine's own
  2,400 tokens / 8.0 s; the chat row's `prefill_tok_s` under 1,000; `peak_prefill` never above it
  (the review noted peak_prefill survived only by accident via `_note_peak_read`'s internal span
  check - it is now gated too).
- Commit 2c9b4719b.

## Finding 2 (carried, documented): the four unconditional changes vs a P124v (field-less) engine - no action

Per the review itself: "no action beyond the report's open question 1". REPORT-p124v2-read-rate-
true.md's open question 1 already states it plainly (the SHARE is field-gated, the beat belt, the
cumulative readout, the `engine_t0` re-baseline and the `read_hb` clear are unconditional; the old
engine's readout denominator/clock semantics shift, safe in direction) and stays open: the card
run that would close the byte-identical OFF check is still owed. Not gated behind the fields
either way: two readout-semantics paths would carry the exact class of bug P1 just was.

## Finding 3 (documented deviation): the cumulative readout lags the card's warm-up figure ~31% - no action

Accepted as documented (round-5 addendum, lines 130-136): the plan's 5% clause binds the STEADY
mangonel replay, where the manager gate itself confirms it (MangonelTrueReadout: sample average
within 5% of the engine's own 1,567.8, green); a cumulative figure lagging a ~711-then-~2,400
warm-up read is by construction, and the card's gates are the end value, the cap and the last-5-s
window, all green.

## Finding 4 (nit): report line references had drifted - TAKEN

REPORT-p124v2-read-rate-true.md updated: `_note_hb` 4582 -> 4621, `_own_read_rate` 4508 -> 4517,
`_read_span_s` 4495 -> 4501, plus the same clause's `read_close` 4476 -> 4474 and the restart
zeroing 4679 -> 4767; `EngineSideContract` 382 -> 421 (the printf-pinning method). All re-grepped
at this HEAD (the P1 fix shifts serve/server.py), so they are true against the tree they ship in;
the same list's `engines/qfn/serve/server.py:1727` `_rhb` was checked still true (driver
untouched). Commit 9cbffccb7.

## Finding 5 (observation, accepted design): a split read's counter runs on the leading card until the lagging fraction lands - no action

The review accepts it; the behavior is what the round-5 gate commits: `SplitReadFollowsEngineFigure`
(counted <= the engine's own cumulative figure at every sample once a cumulative figure exists,
the fraction-falling-back flag, correct-down, closes exact) is unchanged by this round, and the
P1 fix does not touch the share or the split trackers.

## Modules (one at a time, this checkout, at 2c9b4719b)

- serve.test_read_rate_true - 17 green (16 prior + the new restart-readout gate), ~281 s
- serve.test_read_rate - 34 green (module untouched this round)
- serve.test_read_heartbeat - not run (the plan names the read-rate modules; neither the driver
  nor the heartbeat module was touched this round)
- tools/saguaro/extract_qfn_engine.py re-run: serve/qfn_engine.py unchanged (clean `git status`)
- Nothing added under scratch/ this round (the probes behind P1 are the committed test itself).
