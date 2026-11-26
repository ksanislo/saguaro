# FIX-REPORT: R122 review, third pass (REVIEW-RESULT.md) - finding by finding

The round the MANAGER line asked for: findings 1, 2 and 4, plus the cheap nits 5, 7 and 9, plus the
"waiting for Ken" notes in REPORT / FIX-REPORT. All of it landed, one commit per finding:

| finding | where |
| --- | --- |
| 1 HIGH - padding / empty-event gate on "was queued once or paused" | `29cac889e` |
| 2 llama-swap example: the v260 map form, validated | `59a538e28` |
| 4 the normal that started LAST steps aside | `f2e46aaa7` |
| 5 REPORT's test count, and its "waiting for the word" note | `dfae79cda` |
| 7 a priority test through /api/chat and /api/generate | `07500a33b` |
| 9 wait_event_seconds twice | unchanged - declined, recorded below |

`engines/`, `serve/qfn_engine.py`, `tools/` and `saguaro.py` appear in none of the commits
(`git diff --name-only 47325dea3..HEAD`: serve/server.py, serve/responses.py, serve/test_priority.py,
docs/SETTINGS.md, REPORT-r122-priority.md); no re-extraction was needed. No Co-Authored-By /
Claude-Session / Assisted-by lines. The plan's five modules were run again after all fixes, one at a
time: serve.test_priority 51 OK, serve.test_fair 14 OK, serve.test_server 179 OK, serve.test_dashboard
85 OK, serve.test_new_engines_ui 38 OK.

## HIGH - 1. The padding / empty-event trigger: Ken's decision, implemented.

Gate = "this request was queued at least once, or paused", carried on the request's own live row:

- `RequestStat.had_to_wait` - set where the request really waits: `_queue_wait` marks it the first time
  a first-turn `wait_turn` does NOT come (an idle-server request is granted on the first look and never
  sets it), and `_sliced` sets it on every `yield_place` (a priority pause or a fair slice's re-queue).
  `paused_for_priority` stays as it was.
- `Handler._cid_waited(req)` finds the request's row by the `_cid` its rows already carry (both live
  sets), and is the gate: `wait_events(items, empty, waited)` only emits its empty event while it says
  yes; `response_events(..., waited=)` gates the extra `response.in_progress` the same way;
  `_collect_wait(..., paused, waited)` starts its `SAGUARO_WAIT_SPACE_S` clock at the first moment
  `waited()` is true (a PAUSE still sends the 200 + headers at once, per step 3b) instead of at request
  start - both callbacks are required parameters now, all four `_collect_wait` callers (openai,
  anthropic, responses, ollama) pass them.
- The un-queued wire is 1.2.1's: a non-streaming answer slower than `SAGUARO_WAIT_SPACE_S` gets no
  early 200, no chunked framing and no spaces - its status stays free, so a failure after the deadline
  is its real 400, not the old padded 200 carrying the error in the body; a stream whose prompt read
  takes longer than `SAGUARO_WAIT_EVENT_S` keeps its ": keep-alive" comments and gets no empty
  `delta: {}` / `ping` / `response.in_progress` (the PP ticks of a plain read are None items too -
  that was the hole).
- Tests (three new, all on an idle server with nothing ever queued): an un-queued ~2.4 s non-stream
  answer (status 200 with the JSON, no space prefix, `Transfer-Encoding` absent, the whole body behind
  its `Content-Length`, same content as the undisturbed run), an un-queued answer that fails after the
  deadline (real 400 naming the reason, and the padding log line never printed), and an un-queued
  ~0.45 s prompt read on the three event APIs (no empty chunk, no Anthropic ping, exactly the start's
  one `response.in_progress`; comments still there, text identical to the non-streamed answer). The
  queued/paused tests of the second pass all still pass: the gate opens the moment the request waits.
  SETTINGS.md (the settings row and the prose) teaches the gate.
## 2. The llama-swap example: the map form, validated on the version named.

docs/SETTINGS.md now shows v260's actual shape - `models:` as a MAP keyed by the model id, the entry
carrying its `cmd`, the alias and `filters: setParamsByID:` on that same entry (the mechanism the
review verified is unchanged). The printed block is what was validated - copied out of the doc and
run through `llama-swap -validate` here:

```
$ llama-swap -validate -config scratch/swaptest/config_docs.yaml
config is valid: 1 model(s), 0 peer(s)          (exit 0)
```

and the old list form now printed in the doc as little as it fits fails as the review said:

```
$ llama-swap -validate -config scratch/swaptest/config.yaml
config validation failed: yaml: unmarshal errors:
  line 2: cannot unmarshal !!seq into map[string]config.modelMacroConfig          (exit 1)
```

`llama-swap --version` here: `version: v260 (fcefa7b), built at 2026-09-26T15:25:50Z`. The doc's
claim names only v260, which is what was checked; nothing is claimed for v262.

## 4. "The normal one that started last" steps aside.

`FairLock._held` entries become `(level, thread)` pairs - the append order IS the start order, and a
request's whole run (FIFO turn included) lives on one thread, so the ident is a handle a runner can
name about itself without plumbing one through `_turn`/`_sliced`:

- `step_aside(p)`'s normal branch first checks `_last_normal()` (the newest NORMAL entry's thread)
  against its own: an earlier-started normal's tick returns False and cannot claim, so the LAST-started
  one gives its place up - Ken's victim choice, not first-tick-wins. The low branch and the
  claim/yield counting are untouched (lows still step aside first, structurally).
- `_unhold(p)` (used by `release` and `yield_place`) gives back the calling thread's OWN entry at that
  level first - the pairs make it findable; level-only and newest-last remain as fallbacks exactly for
  the paths that had them before (a level-less release pops the newest; a level with no entry leaves
  the list alone). `_idle_wait` and `highest_active` read the levels through the pairs.
- Test `test_the_normal_that_started_last_steps_aside_for_a_high_one`: parallel 2, two normals running
  (B started last, both past their first tokens), then a high that cannot fit - B is cut and resumed
  (>=2 engine calls, one "a normal answer paused ... for a high-priority request" + one "continues"),
  A runs on with exactly one engine call and no pause line; answers whole, `_yields` back to 0.
  SETTINGS.md says who gives the place up. `serve.test_fair` unchanged (14 OK).

## 5. REPORT's test count, and the stale open question. Fixed (`dfae79cda`).

serve.test_priority is counted at 42 in REPORT-r122-priority.md in both places; it is 46 with the
second pass's tests and 51 with this pass's. The "Open for Ken ... waiting for the word" bullet is
gone: the decision was made and is implemented (finding 1 above), and the bullet now says so and
describes the gate. This FIX-REPORT replaces the second-pass file, whose "still the open question for
Ken" and "the rest waits for Ken" lines are likewise closed by findings 1 and 4 (git history keeps
that pass as `1c0a5da40`).

## 7. The ollama endpoints' field. Tested (`07500a33b`).

`test_the_ollama_apis_take_the_field_too`: `model_api.chat_to_openai` / `generate_to_openai` copy the
body's `"priority"` into the OpenAI request they translate to, a request through `/api/chat` and one
through `/api/generate` with the field is accepted (200), the `X-Saguaro-Priority` header works there
too, and a bad value is the same 400 naming the three values - in ollama's own error shape (the
message as a plain string), as both endpoints answer it.

## 9. wait_event_seconds still exists twice. Not done - the second pass's reason stands.

serve/server.py imports serve/responses.py and not the other way round, and serve/ has no settings
module to share without a structural change this round's brief ("contained fixes") should not make.
The review itself closed the item "declined, noted"; both copies' docstrings now name the gate and the
import direction, so the duplication is documented at both ends.

## Nits 6 and 8, and finding 3: not in this round.

The MANAGER line scoped this round to 1, 2, 4 (and 5, 7, 9 if cheap) - "Nothing else". `highest_waiting`
(nit 6) and the one-line read-keep comment (nit 8) stay as the second pass left them; finding 3 was
already verified clean.

## After the fixes

- The brief's WIRE question is answered yes now: with the feature unused (no priority anywhere, nothing
  queued, nothing paused) the framing is byte-for-byte 1.2.1's - the ORDER was already unchanged and
  still is (serve.test_fair 14 OK unchanged), and finding 1's gate closes the two paths that leaked.
- Who steps aside is the plan's last CHANGED block end to end: low whenever a normal or high runs or
  waits, high never, and normals only for a waiting high that cannot fit - the last-started one first.
