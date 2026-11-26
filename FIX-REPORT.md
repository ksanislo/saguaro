# FIX-REPORT: R122 review (REVIEW-RESULT.md) — finding by finding

Every HIGH and MEDIUM finding and every nit was fixed, one commit per finding (`f001ece39..9c72c9e89`, on top of
the reviewed `ab685ff98`). One MEDIUM is a question for Ken and was left as the behaviour the plan's step 3b text
literally asks for — finding 5 says so itself. Nothing touched `engines/`, `saguaro.py` or the generated
`serve/qfn_engine.py`; no re-extraction was needed or run. The plan's five modules were run again after all
fixes, one at a time: serve.test_priority 42 OK, serve.test_fair 14 OK, serve.test_server 179 OK,
serve.test_dashboard 85 OK, serve.test_new_engines_ui 38 OK.

## HIGH — must fix

**1. The "names and the idle rule" and "ONE name" blocks; the report's wrong claim. — Fixed.**
- `f001ece39` (ONE name): `priority_level` accepts exactly `"high" | "normal" | "low"`; the 400 names the three
  (`priority: "high", "normal" or "low"`); `"background"` and `"idle"` fall through to the same 400 as any wrong
  value (a test loop now names both). `PRIORITY_BACKGROUND` → `PRIORITY_LOW`, `PRIORITY_NAMES` says `low`, and
  it is renamed end to end: the pause/paused/continues log lines, `/metrics` (live state, the "Right now" rows,
  the finished rows — they carry `PRIORITY_NAMES[prio]`), the Monitor tag (`serve/web/monitor.js` `PriorityTag`),
  `docs/SETTINGS.md`, and every name, string and comment in `serve/test_priority.py`. `grep -i background` over
  the feature: only the doc line explaining the refusal, the unrelated Responses `background` field, and the
  history in the plan/report files.
- `75847ab9b` (idle rule): `SAGUARO_PRIORITY_IDLE_S` (default 30, 0 = no grace) exists — accessor
  `priority_idle_seconds()`, in `SERVER_ENV`, documented (its settings-table row and the prose). `FairLock`
  tracks its holders (`_held`, one entry per grant) and stamps `_last_finish` when a request releases with its
  priority (`release(..., priority=)`, passed by `_turn`, which `run()` now always hands its level); `_idle_wait`
  gates every low grant — the fast path and `wait_turn` (the wait is capped by what is left of the grace, so a
  time-only block still wakes; keep-alive pings go on; a release notifies) — and `yield_place` counts the
  paused request as waiting, so a resumed low request is subject to the rule exactly like a fresh one. The three
  required tests exist (`IdleRule`: low starts after the grace; a normal arriving inside the grace keeps it
  waiting; grace 0), plus one through `run()`/`_turn` and one that a *running* normal holds the low off.
  `Env` pins the setting to 0 for the other classes so preemption races keep cutting at once.
- `4da93b48e`: the report — see finding 1's third item in the report itself: the two blocks are listed as
  applied *in this pass*, with the commits, and the "applied as written" sentence is gone, named as wrong.
- `9c72c9e89`: the bad-value test names `background`/`idle`.

**2. MANAGER — the llama-swap example was a second model entry. — Fixed (`655dc7aed`).**
`docs/SETTINGS.md` now shows ONE entry: `- id: glm` with `aliases: [glm-agent]` and
`filters: setParamsByID: glm-agent: priority: low` on that same entry, the one-line "why" (an alias is another
name of the same model, `setParamsByID` applies last — no swap, no second process; a second entry would swap to
its own process), and the version claim is what was checked: **v260** (re-checked: `llama-swap --version` →
`v260 (fcefa7b)` here). The example's `priority:` is `low` (finding 1).

**3. QFN batch reads drop a stopped read. — Fixed (`2a5881ed5`).**
The read preemption in `_sliced` fires for a `keeps_stopped_read` engine only with `not engine.batch` (the
decoder gate the review suggested); the decode-tick cut is untouched (a batch slot keeps a stopped *answer*).
The `QfnEngine.keeps_stopped_read` comment on the serve/server.py adapter now says solo-only with the corrected
provenance (#1620 is the solo GEN path; the batch `BRA 0 cancel` / BSTOP branches keep none and free the slot —
described, not line-numbered, since the generated driver moves). The test that repeated the batch claim became
`test_a_batch_read_runs_to_its_end_even_for_a_high_request` (engine with `batch = 4`: no "prompt read paused",
the read finishes, the high one goes after), the property test's comment says solo, and `ReadEngine`'s and the
module's docstrings no longer say "solo and batch alike".

## MEDIUM

**4. `_collect_wait` caught only `(ValueError, EngineDied)`. — Fixed (`8394767ca`).**
The worker catches `Exception` and records it; before the headers the caller re-raises it (the client gets its
truthful failure), after them it rides the body as the error JSON — no more thread traceback, no more
`KeyError: 'body'` reaching `do_POST` as a bogus "malformed request". New test:
`test_an_error_of_the_gpu_busy_kind_rides_the_body_after_the_headers` (a plain `RuntimeError` from `generate`,
raised after the early headers went out: 200 + the real error in the body).

**5. Byte-identity of the two step-3b default paths. — Not changed; open question for Ken.**
The finding is a behaviour question ("confirm with Ken"), and the current trigger is the literal step-3b text.
It is recorded in `REPORT-r122-priority.md` ("Decisions and open questions") with the exact one-line change
(gate the non-stream padding and the empty events on "was queued once or paused") ready for when he answers.

**6. `PRIORITY_NAMES[self.fifo.highest_waiting()]` race. — Fixed (`5bda4ef04`).**
Both sites (decode and read) use `self.fifo.highest_waiting() or prio` — the waiter's name if it is still there,
this request's own level if it slipped out; no `PRIORITY_NAMES[None]`, no dead answer thread.

## LOW / nits

**7. `preempt`/`pos` not reset per engine call. — Fixed (`80562911b`).** Reset with `claimed` at the top of each
`_sliced` engine call: a later plain fair slice logs "fair slice", not a priority pause.

**8. Read PAUSE fired for any `can_pause` engine. — Fixed (`3b2b2317c`).** Gated `can_pause and multi`, exactly
how the fair reads gate theirs; a solo can_pause engine's read runs on. Tests both ways: `PausingRead` now
carries `multi = 1` (like GLM's READY line), and a new test drops it to 0 and asserts no "prompt read paused".

**9. The read log asserted a precision the engine cannot guarantee. — Fixed (`de6246cea`).** The resume line for
a read says "the low prompt read continues **from where the engine kept it**" instead of promising the tick's
N; the tick position stays only in the paused-at line (what the server actually saw).

**10. HTTP/1.1 chunked answer to a 1.0 client. — Fixed (`12d3e601d`).** `_collect_wait` answers the
early-200-with-padding only for clients whose `request_version` is not HTTP/1.0; a 1.0 request waits with its
status still free (the pre-R122 behaviour — padding spaces cannot be framed without chunked transfer). Tested
with a raw HTTP/1.0 request against a blocked engine: `HTTP/1.0 200`, no `Transfer-Encoding`, no leading space.

**11. Misindented `live` continuation lines. — Fixed (`3e87495ab`).** Three lines to 16 spaces, the dict's own
indentation.

**12. `priority_age_seconds()` re-parsed the env under `_cv`. — Fixed (`dba8cc316`).** `_env_seconds(name,
default)` parses once per change of the raw string and caches `(raw, value)`; `priority_age_seconds` and the new
`priority_idle_seconds` use it. Same values as before (unset/empty/unparseable → default, floor 0).

## The brief's questions, after the fixes

- FIFO byte-for-byte with no priority used: unchanged — the idle rule consults `_idle_wait` (0 for p > low) and
  test_fair still passes 14; the wire-bytes question is finding 5, open.
- Deadlock/starvation/tokens: unchanged reasoning, plus the idle rule's own wait (capped sleeps, never dropped);
  tokens: the batch read is no longer cut at all (finding 3), the solo one still resumes from the kept token.
- QFN read keeps the stopped read also in batch? No — and the code no longer claims or assumes it.
- Generated file: `serve/qfn_engine.py` is still untouched by every commit of this pass; the property and its
  corrected provenance live on the adapter in `serve/server.py` and survive `extract_qfn_engine.py`.
- Test gaps the review named are covered: the `low` naming and the idle rule (renamed suite + `IdleRule`), the
  batch/solo QFN read distinction (new batch test, solo-pause test), and `_collect_wait` with a GpuBusy-class
  error. The Turbo-priority test the review called "worth adding" is not added (the review did not block on it).
