# FIX-REPORT: R122 review, second pass (REVIEW-RESULT.md) — finding by finding

The one FIX-FIRST blocker - the stepping-aside rules (finding 1) - is fixed, as are the cheap nits (4-7). The
two MEDIUMs are what the review itself says they are: a bigger change that belongs to Ken (2) and a behaviour
question Ken has not answered (3); both are answered below with why they stand. Commits `080fc24e0..8e778619e`
on top of the reviewed `0fc09a5c0`. `engines/`, `serve/qfn_engine.py` and `tools/` are untouched by all five
commits (`git diff --stat 0fc09a5c0..HEAD`: serve/server.py, serve/test_priority.py, docs/SETTINGS.md only); no
re-extraction was needed. The plan's five modules were run again after all fixes, one at a time:
serve.test_priority 46 OK, serve.test_fair 14 OK, serve.test_server 179 OK, serve.test_dashboard 85 OK,
serve.test_new_engines_ui 38 OK.

## HIGH — must fix

**1. The stepping-aside rules are not the plan's last CHANGED block. — Fixed (`080fc24e0`).**
- `FairLock.step_aside(p)` is now the runner's one tick decision, and both `_sliced` gates (decode token, read
  tick) ask it instead of `waiting(above=p) and claim_yield(0.0, above=p)`: a **low** runner gives its place up
  whenever `_idle_wait(LOW) != 0` (a normal or high holds a place) or any waiter is above it - **regardless of
  free places**, the review's fix as written; a **normal** one only for a waiting high that does not fit
  (`claim_yield`'s counting, the rules that worked, unchanged); a **high** one never. Equal priorities still cut
  nothing (`_idle_wait` ignores the runner's own LOW grant, `waiting(above=LOW)` ignores low waiters), and the
  read gates keep their engine-capability clause (solo QFN cut, multi can_pause PAUSE, batch/other reads run on).
- The low step-aside **claims** like a fair slice does (`_yields += 1`, symmetric `yield_place`/
  `unclaim_yield` - the test asserts `_yields` back to 0): a waiting high's counting (`need - free - _yields`)
  then sees the low place already on its way out, which is also what makes "low ones first" real rather than
  a tick race (finding 2 below).
- The line names who it steps aside for: `highest_active()` (running or waiting) replaces `highest_waiting()`
  at both sites - a low answer cut beside a RUNNING normal has no waiter to name, and said "for a low-priority
  request"; it now says "for a normal-priority request" (the `or prio` fallback for a vanished waiter stays).
- `docs/SETTINGS.md`: the paragraph the review quoted ("...arrives and no place is free") now teaches the
  plan's rule: low whenever a normal or high runs or waits, free place or not, every running low at once;
  normal only for a waiting high with every place taken, only as many as needed; high never.
- Tests: `test_a_free_place_nobody_yields` (the one that asserted the opposite rule) is flipped into
  `test_a_free_place_makes_the_low_one_step_aside_anyway` (free place: step_aside says no for normal, yes for
  low); `test_parallel_two_two_lows_and_a_high_both_lows_pause` replaces `test_parallel_two_cuts_only_as_many_
  as_needed` and asserts **both** lows pause and continue (2 cuts, 2 resumes, `_yields` 0); and the plan's
  four scenarios are all there: `test_a_normal_and_a_high_run_side_by_side_untouched`,
  `test_a_low_answer_steps_aside_for_a_normal_taking_the_free_place` (the fast-path normal the old gates never
  saw, the one that could keep a low company forever), `test_a_high_request_cuts_a_running_normal`, and the
  flipped both-lows test. Nothing else changed in the suite's behaviour: the old races (cap 1, waiter in the
  line) cut exactly where they cut before, and `serve.test_fair` passes unchanged (14 OK).

## MEDIUM

**2. The yielder choice ("low ones first, then the normal one that started last"). — Partly fixed, the rest waits for Ken.**
"The low ones first" is now structural, not a tick race: a running low steps aside unconditionally and counts
its claim while any normal or high runs or waits, so a waiting high's `claim_yield` sees its place covered and
a running normal is cut only when there is no low one left to give (and never more runners than the waiter
needs, as before - `test_only_the_waiters_above_cut_and_only_as_many_as_needed` still passes). "then the normal
one that started last" is **not** implemented: the FIFO does not know in what order the running requests
started - naming that victim means the FIFO nominates it or the tick consults the held-priority stack, a real
change to FairLock's contract. The review's own fix for finding 1 says "the normal/high rules stay as
implemented", so it stays as implemented; if Ken wants last-started-wins among normals, that is the follow-up.

**3. The step-3b padding/empty-event trigger. — Not changed; still the open question for Ken.**
This pass changed neither `_collect_wait`'s padding clock nor `wait_events`' empty-event gate, so the answer is
exactly what the review says it is: the literal step-3b text, tested and logged, with the one-line gate ("was
queued once or paused") recorded in `REPORT-r122-priority.md` waiting for Ken's word. Until he answers, the
"byte-identical when the feature is not used" caveat stands for those two paths and for nothing new: finding
1's fix touches no wire - the resumed stream, the keep-alives and the events are the same as before this pass.

## LOW / nits

**4. Aging vs the idle rule in `wait_turn`. — Said, not used (`9583f7ce0`).** The comment now spells the edge
out (an aged low waiter grants and cuts as a normal one but waits out the grace - the grace is the model's rest
after a person, not a queue position). "use `_eff`" is not the free option it looks: the grant appends its
level to `_held` and the request's `release(priority=)` hands back its RAW level, so granting at the aged level
would leak a NORMAL entry into `_held` for every aged low grant - and Ken's grace sentence covers waits, not
just starts, so waiting it out is the defensible reading.

**5. The queue watchdog vs Ken's no-aging rule. — Fixed (`b1db562ed`).** `FairLock.all_low_waiting()` (under
the lock, `_eff`: with aging on, an aged waiter counts as normal again) and the watchdog skips its "wedged
FIFO" WARNING while the line is all low requests; one normal or high waiter in it prints the line as before.

**6. `waiting(above=)` / `highest_waiting()` read `_queue` without `_cv`. — Fixed (`82e0b2220`).** Both take
the lock now, like `oldest_wait`/`held_by`; every caller was checked to hold nothing (the gates call them from
handler threads, `step_aside` from outside its own lock). Same values, same advisory semantics.

**7. No dedicated Turbo + priority test. — Added (`8e778619e`).** `test_turbo_one_at_a_time_respects_the_
priorities`: a two-place FIFO with `set_one(True)` (the Turbo flag the config actually sets) - a running low
answer is cut for a waiting high AND for a fresh normal request; the free second place is no shield in either
direction (`_fits` keeps one holder, `claim_yield`'s `free = 0 if self._one and self._used`), the cut answers
resume intact and the bookkeeping (`_yields`, queued, locked) ends at zero.

**8. `wait_event_seconds` duplicated. — Not done.** The copy exists because `serve/server.py` imports
`serve/responses.py` and not the other way round, and serve/ has no settings module a helper could live in
(`SERVER_ENV` is a tuple in server.py itself); creating a new module and re-homing one 8-line env read - or,
to do it properly, all the `SAGUARO_*` accessors - is a structural choice bigger than this pass and better
made with every sibling than for this one line. The duplication is documented at both copies; cheap says no.

## After the fixes

- The plan's LAST CHANGED block ("who steps aside, exactly") is what the code and the docs now do, and all
  four of its test scenarios run against real threads and a real `Service.run`.
- With equal priorities everything is still the plain FIFO (`step_aside` for a normal runner is the old
  two-call expression, `test_fair` unchanged); the wire question of the queued/padded paths is finding 3,
  open with Ken; the FIFO never drops a waiting or paused request, and the tokens of a cut one are exactly
  what the engine kept, as before this pass.
