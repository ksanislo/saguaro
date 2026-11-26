# R122: request priority — what was done

One server, one model, two kinds of clients: an agent's long answers and long prompt reads (low) step aside at
once for a person's question (normal or high), and continue from exactly where they were — no reload, no engine
swap, nothing re-read that the engine still holds. Everything in `serve/` only, one commit per step.

## What changed

**Step 1 — the FIFO's priorities** (`620c8dc7c`, serve/server.py). `FairLock` entries carry a priority
(`high` 2 / `normal` 1 = default / `low` 0 — the name it carries now, after Ken's 11-25 notes below; the first
cut called the bottom level `background`). The grant order is highest-first, FIFO inside one priority; with equal priorities the
order and every existing test are exactly as before (the arrival order, the all-places waiter holding back its own
and lower). `acquire`'s fast path only when nobody equal or higher waits; `enqueue(units, priority)`;
`waiting(above=p)` counts the waiters that outrank a runner at p; `highest_waiting()`;
`claim_yield(slice_s, above=p)` cuts only as many runners as the waiters above p need; `yield_place` keeps its
priority, so a resumed high answer does not land behind the low queue.

**Step 2 — the request's priority** (`876559d42`). Body field `"priority": "high" | "normal" | "low"`, header
`X-Saguaro-Priority` (the body wins), and the run config's `"priority"` as the default (a bad config value refuses
to start, a bad request value is a 400 that names the field and the three values). OpenAI's `service_tier` is
untouched. Resolved in `Service.check_request` to `req["_priority"]`; the header reaches it through the
`_priority_header` handler helper (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/api/chat`,
`/api/generate`; `serve/responses.py` normalizer and `serve/model_api.py` carry the field through). `run()`
queues at the level, `RequestStat` carries it, `/metrics` live rows, the "Right now" chats list and every
finished request's row say `"priority"` and (live) `"paused_for_priority"`.

**Step 3 — preemption** (`0bd421fc7`). In `_sliced`: a running answer of lower priority than a waiter that
does not fit gives its place up at once — on a decode token tick, as a fair slice does (PAUSE where
`engine.can_pause`, else the STOP cut), not on the `SAGUARO_FAIR_SLICE_S` timer; on a read tick only where the
engine keeps a stopped read — GLM by its PAUSE (a multi conversation engine, gated like the fair reads), QFN by
the cut **solo** (a batch slot's cancelled read is dropped, so a batch read runs to its end) — and every other
engine's read runs to its end. Images (`emb`) are never preempted, equal or lower priorities never cut, and with
parallel places only as many runners give way as needed (`claim_yield`'s counting). The resumed request:
prompt + written tokens, `_resumed_sampling`, stats summed over its segments as fair slices do, keep-alives to
the client while it waits again; a cancel while paused leaves the line. `QfnEngine.keeps_stopped_read` (the
adapter in serve/server.py, not the generated `serve/qfn_engine.py`) is the property saying where the keep comes
from — generate.cpp #1620, "cancelled request keeps its conversation: N tokens live", the solo GEN path; its
comment carries the corrected provenance and says the batch path drops a stopped read. Logs: `[saguaro]
priority: a low answer paused at N tokens for a high-priority request (k waiting)` / `... continues` (a resumed
read says it "continues from where the engine kept it").

**Step 3b — a waiting client is never left dark** (`58c1a4f9c`). The SSE comments stay; once
`SAGUARO_WAIT_EVENT_S` (15) goes by without a real event, each stream adds one empty event: chat completions a
chunk whose `delta` is `{}` (no content, no finish_reason), `/v1/messages` an Anthropic `ping`, `/v1/responses`
another `response.in_progress`; the ollama-style streams send nothing (an empty line would break them). Never
text in the answer. Non-streaming: nothing until the request was paused once or waited `SAGUARO_WAIT_SPACE_S`
(20); from then on the 200 + headers (chunked) go out and a space follows every `SAGUARO_WAIT_EVENT_S` until the
JSON — valid before a JSON body — and the log says once that an error after that point cannot change the status.
Applies to `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/api/chat` and `/api/generate`. An
HTTP/1.0 client gets none of this (it parses no chunked body): it waits with its status still free, as before.

**CHANGED 11-25** (`ee9fd808f`). No promotion by waiting by default: `SAGUARO_PRIORITY_AGE_S` is 0 = off
(above 0: that long a wait counts a low waiter as normal). `SAGUARO_PRIORITY_MAX_PAUSE_S` does not exist. A
paused or waiting request is never dropped or errored for waiting — it ends only when its client goes away, asks
to stop, or the server stops — and a pause past ten minutes says so once (`[saguaro] priority: a low answer has
been paused for N min`), nothing else.

**CHANGED 11-25 again — names, the idle rule, ONE name** (`f001ece39`, `75847ab9b`). These two blocks arrived
after the first cut of the feature and were NOT applied in it, whatever an earlier version of this report
claimed; the review said so, and they are in now. The levels are exactly `high`, `normal`, `low` — "background"
and "idle" are refused with a 400 that names the three like any other wrong value; every log line, the Monitor
tag, every `/metrics` value, every test and doc says `low` (no "background" left in the feature). And low means
"only while idle": a low request starts, or resumes after a pause, only when nothing normal or high runs or
waits AND none finished in the last `SAGUARO_PRIORITY_IDLE_S` seconds (30, 0 = no grace) — the FIFO counts the
places it holds per priority and stamps every finished normal or high release, and the low grant waits the grace
out with its keep-alives on (nothing is ever dropped for it).

**Review revision (11-25, this pass)** — `f001ece39..dba8cc316`, one commit per finding of `REVIEW-RESULT.md`:
the ONE-name rename end to end; the idle rule + `SAGUARO_PRIORITY_IDLE_S` (SERVER_ENV, docs, tests); the
llama-swap example as ONE entry with the alias on it, `setParamsByID` keyed to the alias on that same entry,
claimed only as "checked on llama-swap v260" (the binary installed here); the QFN read cut limited to solo with
the corrected property provenance and a batch test; `_collect_wait`'s worker catching anything (`GpuBusy`-class
errors no longer reach the client as a KeyError about a body); the preemption's `highest_waiting()` fallback;
and the nits — per-slice reset of `preempt`/`pos`, the read-pause gated to multi engines, the honest read-resume
log line, no chunked padding to HTTP/1.0, the `live` dict indent, cached seconds-setting reads. See
`FIX-REPORT.md` for finding by finding.

## Files

- serve/server.py — FairLock (priorities, the idle rule's `held` / `last_finish` / `_idle_wait`), the priority
  resolution and the handlers, `_sliced`'s preemption and its wait loops, `QfnEngine.keeps_stopped_read` (the
  adapter, solo-only use), the wait events, `_collect_wait`, `/metrics`, `SERVER_ENV`, `_env_seconds`.
- serve/responses.py — the `"priority"` field through the normalizer, the keep-alive `response.in_progress`.
- serve/model_api.py — the ollama-style requests carry the field through.
- serve/web/monitor.js — the `high` / `low` tags and the paused detail text.
- serve/test_priority.py — new: 51 tests (the grant order, preemption in answers and reads, the engines that
  keep a stopped read and the ones that would lose it (batch QFN, solo-pause, no-keep), equals never
  preempting, the thinking budget and the loop guard across a pause, cancel while paused, parallel two, aging
  off/on, a faked ten-minute pause, the idle rule (grace, a normal inside the grace, grace 0, it through
  `run()`), the fields and the 400 (with `background`/`idle` refused) including the two ollama endpoints,
  `/metrics`, each API's empty-event shape parsed back with the client's SSE rules, the non-streaming space
  path, a GpuBusy-class error riding the body, the HTTP/1.0 client getting no chunked padding, the trigger
  gate from the other side (an un-queued slow answer keeps its plain framing and its error keeps its real
  status, an un-queued long read gets no empty events), and the normal that started last stepping aside while
  the earlier one runs on untouched).
- docs/SETTINGS.md — the settings (incl. `SAGUARO_PRIORITY_IDLE_S`), the prose, and the one-entry llama-swap
  alias example checked on v260.

## Tests (python -m unittest, one module at a time, all OK)

- serve.test_priority: 51 tests OK (new)
- serve.test_fair: 14 tests OK (unchanged: equal priorities keep the old order exactly)
- serve.test_server: 179 tests OK (re-run after every step)
- serve.test_dashboard: 85 tests OK
- serve.test_new_engines_ui: 38 tests OK

## Decisions and open questions

- `/v1/completions` does not exist in this codebase (it answers 404), so steps 2 and 3b have nothing to cover
  there; if it is ever added, it should take the field and the empty-chunk keep-alive the same way.
- `/api/chat` and `/api/generate` are gated on model management, as before; the tests turn it on.
- `keeps_stopped_read` was added to the `QfnEngine` adapter in serve/server.py (with the generate.cpp #1620
  provenance, solo path, in its comment) rather than to engines/qfn/serve/server.py + `extract_qfn_engine.py`,
  because engines/ is out of scope and serve/qfn_engine.py is generated. One matching line in the QFN source
  and a re-extraction would supersede it. The property is true for the SOLO path only: a batch slot's cancelled
  read is dropped by the engine (BRA 0 cancel / BSTOP hand every checkpoint back), so `_sliced` fires the read
  preemption for QFN only while the engine runs without batch places.
- The empty stream events fire on the request's keep-alive pings (every `SAGUARO_QUEUE_PING_S`), so their
  arrival is the first ping at or past `SAGUARO_WAIT_EVENT_S`, then one per threshold.
- The padding answers are chunked HTTP/1.1 (`_collect_wait` raises `protocol_version` for them and is skipped
  whole for an HTTP/1.0 client, which could not parse it); the pause trigger recognizes the request by the
  conversation id its rows already carry.
- Decided and implemented (was the open question of the first two passes): the step 3b padding and the empty
  stream events gate on "this request was queued at least once, or paused" - Ken's recorded decision, now the
  code: `RequestStat.had_to_wait` marks the first real FIFO wait and every give-place-back, `_cid_waited` reads
  it off the request's own live row, and `_collect_wait`, `wait_events` and `response_events` open their gate on
  it. A request that never waited is framed byte-for-byte as in 1.2.1 (no early 200 + chunked spaces on a
  >`SAGUARO_WAIT_SPACE_S` answer - its late error keeps its real status; no empty event on a
  >`SAGUARO_WAIT_EVENT_S` prompt read), which is what the brief's "behaviour with the feature OFF is
  byte-identical" needed; the queued/paused paths keep the padding exactly as tested before, a pause still
  starting it at once.
- Ken's first 11-25 note (no aging by default, no `SAGUARO_PRIORITY_MAX_PAUSE_S`) was applied when it came in
  (`ee9fd808f`); his second and third ("names and the idle rule", "ONE name") were NOT applied in the first
  cut — the review caught that, and they are applied in this pass (`f001ece39`, `75847ab9b`). An earlier
  version of this report claimed all three had been applied as written; that claim was wrong and is gone.
