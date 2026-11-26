# R122 code review, third pass (local agent) — verdict: FIX-FIRST

Reviewed `git diff 42aec008f..HEAD` (30 commits; serve/server.py, serve/responses.py, serve/model_api.py,
serve/web/monitor.js, serve/test_priority.py, docs/SETTINGS.md + the reports). Scope clean: `engines/`,
`serve/qfn_engine.py` and `tools/` untouched in every commit (the generated file needs no re-extraction, the
`keeps_stopped_read` property lives on the serve/server.py adapter :9089); no secrets, no paths outside the repo,
no Co-Authored-By/Claude-Session/Assisted-by trailers. All five plan modules re-run here, one at a time, all OK:
serve.test_priority 46, serve.test_fair 14, serve.test_server 179, serve.test_dashboard 85,
serve.test_new_engines_ui 38 — exactly FIX-REPORT's counts.

The pass-2 blocker (the stepping-aside rules, MANAGER item 3) is fixed and verified below. What blocks now is
Ken's recorded decision that was never implemented, and a llama-swap example that does not parse on the version
it claims.

## HIGH — must fix

1. **Ken's MANAGER decision (REVIEW-RESULT.md, top line) is not implemented: the padding/empty-event trigger.**
   The decision: gate BOTH on "this request was queued at least once, or paused", so an un-queued request is
   byte-for-byte as in 1.2.1, "with a test that an un-queued 30 s answer is unchanged". At HEAD the trigger is
   still the raw clock: `_collect_wait` starts its padding deadline at request start
   (serve/server.py:7172 `deadline = time.monotonic() + wait_space_seconds()`), so any non-streaming answer
   slower than `SAGUARO_WAIT_SPACE_S` (20) on an idle server that was never queued or paused still gets the
   early 200 + `Transfer-Encoding: chunked` + spaces (serve/server.py:7180-7198) and a post-header error rides
   the body as a 200 instead of its real status; `wait_events` (serve/server.py:753-772, applied at :8437 and
   :8526, and serve/responses.py:389-392 for /v1/responses) adds its empty event on ANY
   `SAGUARO_WAIT_EVENT_S` gap of keep-alive `None`s — and the pipe drivers yield `None` on PP lines during a
   plain prompt read (serve/server.py:2296, 2298), so a >15 s read on a stream that never queued gets an extra
   `delta:{}` / `ping` / `response.in_progress` every 15 s. FIX-REPORT.md:54-59 ("Not changed; still the open
   question for Ken") is stale against Ken's answer, and REPORT-r122-priority.md:120-125 still says "waiting
   for the word". None of the 46 tests asserts the un-queued case (the suite's streams all wait behind a
   blocker). Until this lands, the brief's "behaviour with the feature OFF must be unchanged (byte-identical
   paths)" does NOT hold for those paths. The queue/pause state to gate on is already visible:
   `_queue_wait`/`wait_turn` and `stat.paused_for_priority`.

## MANAGER

2. **llama-swap example (docs/SETTINGS.md:278-291): the STRUCTURE is right, the SYNTAX is wrong for the
   version claimed.** One model entry carrying the alias, `filters: setParamsByID` keyed to that alias on the
   SAME entry, no second process — verified correct in mechanism against llama-swap v260 (installed here,
   `llama-swap --version` → `v260 (fcefa7b)`): its embedded config schema documents `setParamsByID` as
   "mapping requested model IDs (**or aliases**) to parameters ... Applied after setParams", and the binary
   validates alias keys per model ("duplicate alias '%s' in model %s filters.setParamsByID"). BUT the YAML
   shown is the OLD list form (`- id: glm` under `models:`), and **v260 rejects it**:
   `llama-swap -validate` on the example fails — `yaml: cannot unmarshal !!seq into
   map[string]config.modelMacroConfig`. In v260 `models` is a MAP keyed by model id, each entry requiring
   `cmd`; the equivalent one-entry example validates clean here ("config is valid: 1 model(s)"):

   ```yaml
   models:
     glm:
       aliases:
         - glm-agent
       cmd: ... --port ${PORT}
       filters:
         setParamsByID:
           glm-agent:
             priority: low
   ```

   So the parenthetical "(checked on llama-swap v260)" (docs/SETTINGS.md:278) is false as written: the claim
   was checked against v260's setParamsByID schema, but the snippet printed cannot load on v260. Rewrite the
   example in the map form (verified above) or show both and say which was checked. v262 is not installed
   here and no Go module cache exists, so nothing can be claimed for it; the claim must name only what was
   checked (v260).
3. **The names — verified, nothing left.** Exactly `high`/`normal`/`low` end to end: `priority_level`
   (serve/server.py:377-382) refuses everything else with a 400 naming the three (serve/test_priority.py:987-
   994, :1017-1019); the logs (:5802-5804, :5819-5821, :5833-5835), the Monitor tag (serve/web/monitor.js:130-
   132 — `high`/`low` only, normal shows nothing), every `/metrics` value (serve/server.py:3129, 4582, 4814,
   6458), the tests and the docs all say `low`. The only remaining occurrences of "background" in the feature
   are the refusal itself (serve/server.py:379, docs/SETTINGS.md:242-243, serve/test_priority.py:1017) and the
   unrelated Responses `background` field — that is the plan's ONE-name rule, not residue. Grep clean.
4. **The stepping-aside rules — now the plan's last CHANGED block, one named deviation left.**
   `FairLock.step_aside` (serve/server.py:570-590) is the runner's tick decision used by both `_sliced` gates
   (read :5674-5688, decode :5718-5729): low steps aside whenever a normal or high runs OR waits — a free
   place is no shield, and it claims (`_yields += 1`, :587) so a waiting high's counting sees the place leaving
   (which also makes "low ones first" structural); normal only for a waiting high that does not fit
   (`waiting(above=)` + `claim_yield`'s counting — and `above=` excludes re-queued equals from `need`, so
   exactly as many runners as the waiter needs pause); high never. `highest_active()` names a RUNNING normal.
   The old opposite test is flipped (:275-283) and all four plan scenarios are real-thread tests (:483-578:
   both lows pause, normal+high side by side untouched, low steps for a fast-path normal, high cuts a running
   normal); `serve.test_fair` still passes unchanged. The deviation: "then the normal one that started last"
   (PLAN line 40) is not implemented — with two running normals and a waiting high, the first runner to tick
   claims, not the last-started one (FIX-REPORT.md:44-52 admits it). By the brief ("any difference is a
   finding") this is a finding: implement last-started, or have Ken's word recorded that first-tick-wins is
   accepted, and say it in the docs.

## LOW / nits

5. REPORT-r122-priority.md:98 still counts "42 tests"; the suite is 46 (FIX-REPORT has it right).
6. `highest_waiting()` (serve/server.py:690-693) is dead in production — only serve/test_priority.py:235 calls
   it; the preemption logs use `highest_active()`. Drop it or use it.
7. `/api/chat` and `/api/generate` carry the field (serve/model_api.py:146-148,
   `self._priority_header(oreq)` serve/server.py:7508) but no test sends a `priority` through either —
   RequestFields covers only the three OpenAI/Anthropic/Responses endpoints plus the header on openai.
8. The read preemption for a solo QFN cut assumes the engine's #1620 keep, which itself is conditional
   (`pp_exact && pp_reached > 0 && prompt_cache > 0 && (req_ckpt || req_cont) && !read_vram && !nf_hit`,
   engines/qfn/src/program/generate.cpp:22596-22597): where the engine keeps less (stopped mid-window, ckpt=0)
   the resume re-reads the prompt. Correctness holds (the prompt is re-fed whole) and the log already promises
   only "from where the engine kept it" — a one-line note in the property comment would close it.
9. `wait_event_seconds` still exists twice (serve/server.py:741-749, serve/responses.py:306-313) for the
   documented import direction; FIX-REPORT.md:84-88 declined a shared helper with a reason — declined, noted.

## The brief's questions

- **FIFO byte-for-byte with no priority used?** The ORDER: yes — with all-equal priorities `acquire`'s fast path
  reduces to `not self._queue and _fits` (serve/server.py:505-511), `_first(me)` to `queue[0] is me`
  (:482-496), `claim_yield(above=None)` to the old `need` (:544-563), `_held`/`_last_finish` gate only p==LOW
  (:520-533), and serve.test_fair passes unchanged (14 OK). The WIRE: no, for a queued stream, a >15 s read
  and a >20 s non-streaming answer with no priority anywhere — finding 1.
- **Can a low request deadlock, starve forever, or lose tokens?** Deadlock: no — `_cv` (a plain Lock) is never
  taken nested (`_eff`/`_first`/`_idle_wait` take nothing under it; `step_aside`'s normal branch calls
  `waiting` and `claim_yield` sequentially, each taking the lock once; no caller holds it — checked every one),
  `status_lock` and `_cv` never overlap (:5786-5828), every `step_aside`/`claim_yield` True is consumed by
  `yield_place` (:5796) or `unclaim_yield` (:5747-5748, also on exceptions via the finally), the pause-wait's
  finally leaves the line, `_turn`'s finally always leaves-or-releases (:5408-5413), and releases notify.
  Starve forever: yes, deliberately (Ken's no-aging rule, plus the idle grace) — and never dropped or errored
  for it (the 10-minute line only logs, tested serve/test_priority.py:450-481 with a faked clock). Tokens:
  never — a cut resumes with prompt + written tokens via `_resumed_sampling` (:5856-5857), the streamed text is
  asserted identical (:347-379, :737-766), the thinking budget counts on (:678-735).
- **Can a paused request hold a place, or leave the line twice?** No: `yield_place` frees and re-queues in one
  locked step and drops its `_held` entry (:599-614), `hold[0]` keeps `_turn` from releasing what is free,
  `leave` is idempotent (:648-652), cancel-while-paused drains (tested :427-448).
- **Does the QFN read path really keep the stopped read, also in batch?** No — SOLO only, and the code is now
  limited that way. Engine source checked at generate.cpp: the solo GEN path keeps the conversation live at
  `pp_reached` when stopped between windows (#1620, generate.cpp:22596-22605, the "cancelled request keeps its
  conversation: N tokens live" line); the batch path drops a cancelled read — `BRA <b> 0` clears the slot's ids
  and hands every checkpoint back ("a read that failed keeps none", :14894-14910) and a BSTOP-reached held read
  ends the same way (:16442-16463). `_sliced` fires the read cut only with `not engine.batch`
  (serve/server.py:5676-5677), the adapter property documents the solo provenance (:9089-9097), and both shapes
  are tested (serve/test_priority.py:609-665, batch included).
- **Images, the loop guard and Turbo unaffected?** Yes: both gates require `not emb` (:5674, :5718); the loop
  guard rides the unchanged resume path and produces the same answer and actions across a pause (tested
  :737-766); `_one` keeps one holder at a time whatever the places and a free place is no shield in either
  direction for the priorities (tested :404-425).
- **Generated file:** `serve/qfn_engine.py` (and engines/, tools/) appear in no commit of the range
  (`git diff --stat` empty); the adapter override survives a re-extraction.
- **Tests that really exercise the change:** yes — real threads against real `Service.run`, the seam asserted
  on recorded engine calls (prompt + written, budgets), `_yields` back to 0, each API's empty event parsed
  back with the client's SSE rules, HTTP/1.0 raw socket, cancel-while-paused, the plan's four step-aside
  scenarios, the idle rule through `run()`. Missing: the un-queued-answer case of finding 1, and the ollama-
  endpoints' field (nit 7).

## Verdict

FIX-FIRST — finding 1 (Ken's decision, already ruled: gate on "was queued once or paused", plus the un-queued
test) and MANAGER item 2 (v260-valid YAML for the llama-swap example, with the claim matching). Both are
contained; everything else above is a nit or a doc count, and the rest of the feature — the FIFO, the
stepping-aside rules, the read keeps, the wiring, the tests — verifies against the plan.
