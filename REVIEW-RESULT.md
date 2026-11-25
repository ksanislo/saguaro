# Review: P2 - an open page must not keep a model loaded behind llama-swap

Verdict: FIX-FIRST (one high finding silently disables the whole feature; one medium retry bug; both share one small fix).

Reviewed `git diff c3844c07..HEAD` (0a2969c1 page, dc4a9823 tests, 1c12f7e7 report) against
PLAN-p2-llamaswap-polling.md and REPORT-p2-llamaswap-polling.md. Tests run here, one at a time:
`serve.test_dashboard` 63 OK (~28 s, node present, PageLogic not skipped), `serve.test_new_engines_ui` 35 OK.

## High

1. **A page first opened in a hidden tab never loads `/health` - so the llama-swap gate never engages, ever.**
   `loadHealth` returns at serve/web/server.js:10 when `document.hidden` *without* parking anything, and the
   `visibilitychange` resume (serve/web/server.js:106) calls `loadHealth` only when `healthWait` was set - which only
   the catch path at server.js:38 ever does. The boot chain's first health (serve/web/app.js:115) is therefore dropped
   for a background-tab open (ctrl-click, "open in new tab", session restore), and nothing retries: `poll()`'s
   gate needs `S.health.supervisor` (server.js:75), which only `/health` supplies, so `swapReady` stays `null` and
   the visible page polls `/metrics` every second behind llama-swap forever - exactly what this change was to stop -
   while the model's facts and defaults never load. Reproduced with the PageLogic stubs (scratch/gate_hidden_open.mjs):
   hidden at open, then shown: fetch log `["metrics"]`, `supervisor: null`. Regression too: before this change
   `loadHealth` fetched regardless of visibility. Fix: in the `document.hidden` branch set `healthWait = true` (keep
   `unloaded()` as a plain return - resuming into a closed gate must not fetch health), or have the resume path fetch
   health when the supervisor is still unknown.

## Medium

2. **A health retry whose 2 s timer fires while the tab is hidden kills the retry chain.** Same root cause:
   the armed `setTimeout(loadHealth, 2000)` (server.js:39) fires into the server.js:10 guard, which neither re-arms
   nor sets `healthWait`. Scenario: page opens while the server is starting (health fails), tab hidden before the
   retry fires - after that `/health` is never asked again, visible or not, until a gate reopen transition
   (server.js:81) that a plain setup can never produce. Reproduced (scratch/health_retry_lost.mjs): 1 health fetch
   total, none after the tab is shown again. Old code retried every 2 s regardless. The finding-1 fix closes this too.

## Low

3. **A transient `/running` error un-gates for one cycle and re-loads the model.** `swapRunning` returns `null` on
   any throw (server.js:60-66) and `null` opens the gate (server.js:75), so one network blip while the model is off
   the card leads to an immediate `/metrics` fetch - loading the model and evicting another. The plan prescribes
   "poll as today" when `/running` fails, but plainly means the persistent 404/key-refused case; treating a blip
   like "older llama-swap" is stricter than needed (e.g. keep the last known state on anything but 404).

4. **The answer-reconnect loop is the one repeating request under `/upstream/` that ignores both the gate and
   visibility** (brief's reconnect-loop question): `follow` (serve/web/chats.js:481-502) re-fetches
   `v1/strata/stream` every 2 s for up to 10 minutes, and `sleepOrVisible` (chats.js:370-374) resolves on its plain
   timeout while hidden. Behind llama-swap each such request is an upstream request. Bounded in practice - a
   swapped-out model has no live job, the first request loads the model and its 404 ends the loop - so it cannot
   keep a model loaded, but it does "ask while hidden" contrary to the plan's wording, and the report's open
   question 3 covers `watchReload`/`watchUpdate` but not this loop.

5. **`watchReload` (server.js:274-288, `/api/context` every 2 s) and `watchUpdate` (server.js:424-447, `health`
   every 3 s) loop unchecked while the tab is hidden and while the gate is closed.** Self-declared in the report
   (open q3); user-initiated, transient, and llama-swap-managed servers refuse those POSTs outright
   (serve/server.py:7298-7305), so accepted as a deviation from "stop all polling while hidden, in every setup".

6. **The loadMcp "servers starting" retry chain (server.js:172-173) is dropped by a hidden tab the same way as
   finding 2** (early return at server.js:165, no park, no re-arm). It self-heals from poll's 10 s mcpTick
   (server.js:100) whenever the Monitor or Settings > Tools is visible again, so minor.

7. **`api/tokens` bypasses the gate: opening or editing a chat loads the model while unloaded.** `updateCtxMeter`
   (serve/web/chats.js:284-309, also armed by the health hook at serve/web/app.js:114) POSTs `api/tokens` under the
   upstream path with a 400 ms debounce whenever a chat with messages is opened or typed in - so *browsing* chats,
   not just sending, reloads the model behind llama-swap. It is user-triggered and de-duplicated (ctxKey), not a
   poll, so arguably the same class as sending - but it does soften the Monitor's "the next message loads it".

## Nits

8. serve/web/monitor.js:822-824: the diff re-indents the pre-existing "key" and "Waiting" continuation lines
   (col 41 -> 39) while adding the "unloaded" branch - cosmetic churn on untouched lines; rendered text is unchanged.
9. The five new PageLogic tests exercise the real module (serve/test_dashboard.py:861-870 imports the live
   serve/web/server.js and asserts exact fake-fetch logs) and genuinely cover: gate closed -> only `/running`,
   `conn=unloaded`, `metrics=null`, loadHealth/loadMcp silent; ready -> `/metrics`; swap-back -> health-then-metrics;
   hidden -> zero requests in both setups; no-supervisor and path-without-supervisor -> `/metrics` only, never
   `/running`; `/running` 404/401 -> fallback visible-only. Not covered: the `visibilitychange` handler itself is
   unreachable in the harness (`document.addEventListener` is a noop, serve/test_dashboard.py:851) - findings 1-2
   live exactly there - plus the 5 s gate cadence and the mcpTick scheduling.

## The brief's two questions

- **A poll on any path while unloaded, incl. SSE/reconnect?** There is no EventSource anywhere; the stream is
  fetch + getReader (serve/web/chats.js:434). With health known, the standing paths are gated: poll (server.js:76-79,
  `/running` only, 5 s), loadHealth (server.js:10) and loadMcp (server.js:165). What can still ask while unloaded:
  finding 1 (gate never learns to close), the bounded reconnect loop (finding 4), the `api/tokens` meter
  (finding 7), and the acknowledged watchReload/watchUpdate flows (finding 5). None of them keeps a model loaded
  the way the old 1 s `/metrics` did, except finding 1, which restores the old behaviour entirely.
- **Does the chat still send, loading the model?** Yes. The send button gates only on text/busy/reload
  (serve/web/chat-view.js:381,425) - never on `S.conn` - and `send` -> `answer` POSTs `v1/chat/completions`
  (serve/web/chats.js:513) under `/upstream/<id>/`, which is what llama-swap loads. The "Unloaded" pill
  (serve/web/sidebar.js:21, serve/web/monitor.js:823) is display-only.

## Also checked

- Feature OFF (visible, no supervisor): fetch order, 401/`key` handling, `fails`-to-`down`, mcpTick and the 1 s
  cadence are unchanged (server.js:82-101); the only differences are the unconditional `emit()` (server.js:99,
  reachable-hidden only mid-await - immaterial) and the plan-mandated hidden-tab stop. `/running` is never asked
  when not behind llama-swap. `supervisor` is a real /health field (serve/server.py:7160).
- Locks/threads/stdout interleaving: not applicable - the diff is page-only JS; no Python, no server, no stdout.
  `poll`'s `polling` flag is a plain reentrancy guard cleared in `finally` (server.js:71,102); no nesting, and a
  dropped resume is re-armed by the in-flight poll (server.js:101).
- Generated file: `serve/qfn_engine.py` and `engines/qfn/serve/server.py` are untouched in the diff (git diff
  --stat), so no extract run was owed - consistent with the report.
- No paths outside the repo, no secrets: the diff adds only relative URLs and the origin-root `/running`; the
  hard-coded strings in the new tests are model ids and copy text.
- Report accuracy: file list, commit split, the five tests, both module counts and the four open questions match
  what was reviewed; the hidden-tab-at-open hole (finding 1) is not among them.
