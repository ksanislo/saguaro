# R122: request priority — what was done

One server, one model, two kinds of clients: an agent's long answers and long prompt reads (background) step
aside at once for a person's question (normal or high), and continue from exactly where they were — no reload,
no engine swap, nothing re-read that the engine still holds. Everything in `serve/` only, one commit per step.

## What changed

**Step 1 — the FIFO's priorities** (`620c8dc7c`, serve/server.py). `FairLock` entries carry a priority
(`high` 2 / `normal` 1 = default / `background` 0). The grant order is highest-first, FIFO inside one priority;
with equal priorities the order and every existing test are exactly as before (the arrival order, the
all-places waiter holding back its own and lower). `acquire`'s fast path only when nobody equal or higher
waits; `enqueue(units, priority)`; `waiting(above=p)` counts the waiters that outrank a runner at p;
`highest_waiting()`; `claim_yield(slice_s, above=p)` cuts only as many runners as the waiters above p need;
`yield_place` keeps its priority, so a resumed high answer does not land behind the background queue.
The background waiter's aging into normal came in at 300 s and was replaced by Ken's 11-25 rule — see below.

**Step 2 — the request's priority** (`876559d42`). Body field `"priority": "high" | "normal" | "background"`,
header `X-Saguaro-Priority` (the body wins), and the run config's `"priority"` as the default (a bad config
value refuses to start, a bad request value is a 400 that names the field). OpenAI's `service_tier` is
untouched. Resolved in `Service.check_request` to `req["_priority"]`; the header reaches it through the
`_priority_header` handler helper (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/api/chat`,
`/api/generate`; `serve/responses.py` normalizer and `serve/model_api.py` carry the field through). `run()`
queues at the level, `RequestStat` carries it, `/metrics` live rows, the "Right now" chats list and every
finished request's row say `"priority"` and (live) `"paused_for_priority"`.

**Step 3 — preemption** (`0bd421fc7`). In `_sliced`: a running answer of lower priority than a waiter that
does not fit gives its place up at once — on a decode token tick, as a fair slice does (PAUSE where
`engine.can_pause`, else the STOP cut), not on the `SAGUARO_FAIR_SLICE_S` timer; on a read tick only for
engines that keep a stopped read (GLM by its PAUSE, QFN by the cut), every other engine's read runs to its
end. Images (`emb`) are never preempted, equal or lower priorities never cut, and with parallel places only as
many runners give way as needed (`claim_yield`'s counting). The resumed request: prompt + written tokens,
`_resumed_sampling`, stats summed over its segments as fair slices do, keep-alives to the client while it
waits again; a cancel while paused leaves the line. `QfnEngine.keeps_stopped_read` (the adapter in
serve/server.py) is the property saying where it comes from — generate.cpp #1620, "cancelled request keeps its
conversation: N tokens live", solo and batch slots alike. Logs: `[saguaro] priority: a background answer
paused at N tokens for a high-priority request (k waiting)` / `... continues` (and the prompt-read variants).

**Step 3b — a waiting client is never left dark** (`58c1a4f9c`). The SSE comments stay; once
`SAGUARO_WAIT_EVENT_S` (15) goes by without a real event, each stream adds one empty event: chat completions a
chunk whose `delta` is `{}` (no content, no finish_reason), `/v1/messages` an Anthropic `ping`, `/v1/responses`
another `response.in_progress`; the ollama-style streams send nothing (an empty line would break them). Never
text in the answer. Non-streaming: nothing until the request was paused once or waited `SAGUARO_WAIT_SPACE_S`
(20); from then on the 200 + headers (chunked) go out and a space follows every `SAGUARO_WAIT_EVENT_S` until
the JSON — valid before a JSON body — and the log says once that an error after that point cannot change the
status. Applies to `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/api/chat` and `/api/generate`.

**CHANGED 11-25** (`ee9fd808f`). No promotion by waiting by default: `SAGUARO_PRIORITY_AGE_S` is 0 = off
(above 0: that long a wait counts a background waiter as normal). `SAGUARO_PRIORITY_MAX_PAUSE_S` does not
exist. A paused or waiting request is never dropped or errored for waiting — it ends only when its client goes
away, asks to stop, or the server stops — and a pause past ten minutes says so once (`[saguaro] priority: a
background answer has been paused for N min`), nothing else.

**Step 4 — the Monitor** (`602562a9d`, serve/web/monitor.js). The running and queued rows show a small
`high` / `background` tag and a paused one says "paused for a higher-priority request"; no new page.

**Step 6 — docs** (`04a0ff261`, docs/SETTINGS.md). The field, the header, the config default,
`SAGUARO_PRIORITY_AGE_S`, `SAGUARO_WAIT_EVENT_S` / `SAGUARO_WAIT_SPACE_S`, and a llama-swap alias example
(`filters: setParamsByID: "<alias>": priority: background`, said to be the llama-swap >= v200 syntax) with
what it shows in the log.

## Files

- serve/server.py — FairLock, the priority resolution and the handlers, `_sliced`'s preemption and its
  wait loops, `QfnEngine.keeps_stopped_read`, the wait events, `_collect_wait`, `/metrics`, `SERVER_ENV`.
- serve/responses.py — the `"priority"` field through the normalizer, the keep-alive `response.in_progress`.
- serve/model_api.py — the ollama-style requests carry the field through.
- serve/web/monitor.js — the tags and the paused detail text.
- serve/test_priority.py — new: 34 tests (the grant order, preemption in answers and reads, the engines that
  keep a stopped read and the one that would lose it, equals never preempting, the thinking budget and the
  loop guard across a pause, cancel while paused, parallel two, aging off/on, a faked ten-minute pause, the
  fields and the 400, `/metrics`, and each API's empty-event shape parsed back with the client's SSE rules
  plus the non-streaming space path).
- docs/SETTINGS.md — the settings and the prose.

## Tests (python -m unittest, one module at a time, all OK)

- serve.test_priority: 34 tests OK (new)
- serve.test_fair: 14 tests OK (unchanged: equal priorities keep the old order exactly)
- serve.test_server: 179 tests OK (re-run after every step)
- serve.test_dashboard: 85 tests OK
- serve.test_new_engines_ui: 38 tests OK

## Decisions and open questions

- `/v1/completions` does not exist in this codebase (it answers 404), so steps 2 and 3b have nothing to cover
  there; if it is ever added, it should take the field and the empty-chunk keep-alive the same way.
- `/api/chat` and `/api/generate` are gated on model management, as before; the tests turn it on.
- `keeps_stopped_read` was added to the `QfnEngine` adapter in serve/server.py (with the generate.cpp #1620
  provenance in its comment) rather than to engines/qfn/serve/server.py + `extract_qfn_engine.py`, because
  engines/ is out of scope and serve/qfn_engine.py is generated. One matching line in the QFN source and a
  re-extraction would supersede it.
- The empty stream events fire on the request's keep-alive pings (every `SAGUARO_QUEUE_PING_S`), so their
  arrival is the first ping at or past `SAGUARO_WAIT_EVENT_S`, then one per threshold.
- The padding answers are chunked HTTP/1.1 (the server's own protocol_version is HTTP/1.0 and is left as it
  is everywhere else); the pause trigger recognizes the request by the conversation id its rows already carry.
- Ken's 11-25 notes were applied as written: they replace step 1's 300 s aging and step 3b's
  `SAGUARO_PRIORITY_MAX_PAUSE_S`.
