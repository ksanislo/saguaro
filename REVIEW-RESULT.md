# Review (round 2): P2 - an open page must not keep a model loaded behind llama-swap

Verdict: MERGE-WITH-NITS - the round-1 high and medium are genuinely fixed and the standing polls are gated; one medium
hole is left (the first poll after a tab is revealed still runs ungated, and its test cannot see it), plus small items.

Reviewed `git diff c3844c07..HEAD` (0a2969c1 page, dc4a9823 tests, 1c12f7e7 report, 3f47ec23 round-1 review, 8afb9b06 /
b9c72083 / 55be4154 / 5a95efed / 1d2b5719 / 4c9f5c16 fixes, 42ca46ba FIX-REPORT) against
PLAN-p2-llamaswap-polling.md, REPORT-p2-llamaswap-polling.md and FIX-REPORT.md. Tests run here, one at a time:
`serve.test_dashboard` 68 OK (30.7 s, node v20.19.2, PageLogic not skipped - serve/test_dashboard.py:864),
`serve.test_new_engines_ui` 35 OK (3.7 s).

## Round-1 findings: verified

| # | where | state |
| --- | --- | --- |
| 1 hidden tab at open lost its /health | serve/web/server.js:12 parks it, :111 makes it up | FIXED - but see **M1**: the resume does not await it |
| 2 a health retry landing while hidden | serve/web/server.js:41 | FIXED, tested (serve/test_dashboard.py:1114) |
| 3 a `/running` blip un-gating | serve/web/server.js:63-70 (`was` kept; only 404/401/403 answer "no gate") | FIXED, tested (:1136, both halves) |
| 4 the reconnect loop | serve/web/chats.js:382-385 | FIXED for hidden (see **L1** for what it costs elsewhere); the visible loop is still ungated (**L4**) |
| 5 watchReload / watchUpdate | serve/web/server.js:277-297, :429-452 | unchanged, as agreed (REPORT open q3) - see **N1** |
| 6 the loadMcp "starting" retry | serve/web/server.js:170-172, :112 | FIXED, tested (:1172) |
| 7 `api/tokens` bypassing the gate | serve/web/chats.js:289-298 | FIXED, tested (:1192) - see **L3** for the figure it keeps |
| 8 monitor.js re-indent | `git diff c3844c07 -- serve/web/monitor.js` is one added line (:823) | FIXED |
| 9 the harness could not fire `visibilitychange` | serve/test_dashboard.py:851-856 (listener list + `fire`) | FIXED for the handler; cadence still untested |

## Medium

1. **The first poll after a tab is revealed runs ungated: it asks `/metrics` (and `/health`) before `/running`, so behind
   llama-swap a reveal loads the model.** The `visibilitychange` resume (serve/web/server.js:109-113) calls `loadHealth()`
   and `poll()` back to back and neither is awaited; `poll()`'s gate needs `S.health.supervisor` (serve/web/server.js:80),
   which only the `/health` response supplies, so at that instant it evaluates to "not behind llama-swap" →
   `swapReady = null` → the metrics fetch (serve/web/server.js:88) goes out under `/upstream/<id>/`. The order is
   deterministic (both calls run synchronously in the handler; `loadHealth` suspends at its `await fetch("health")` at
   server.js:15, then `poll` reaches its fetch), so it is not a flaky race but the normal path. FIX-REPORT.md:6-7 records
   exactly this - "shows `health, metrics`" - and treats it as the fix working. The harm is the two-model case the plan
   names: two Saguaro tabs, alt-tab to the other model's Monitor → llama-swap swaps this model back in and evicts that
   one, on every reveal, whatever the Monitor's "the next message loads it" says. (The same ordering exists at boot -
   serve/web/app.js:115 vs :125 - and is covered by REPORT open q4, because loading the page's own files already loaded
   the model; a reveal minutes or hours later is a fresh load with nothing asked of the model.) Its test cannot catch
   this: serve/test_dashboard.py:1091-1112 answers `/running` with the model **ready**, so a gated and an ungated poll
   produce the same log; the same is true of scratch/gate_hidden_open.mjs:17-18. Fix (either): make the resume
   `async` and `await loadHealth()` before `poll()` when `healthWait` was set (server.js:111), or have `poll()` ask
   `/running` whenever the path is `/upstream/<id>/` (serve/web/server.js:56) instead of waiting for the supervisor
   from `/health` - `/running` loads nothing, so the extra ask is free and a 404 falls back as it already does
   (server.js:67).

## Low

2. **The reconnect wait now stalls in a hidden tab in every setup, including plain Saguaro with no gate at all**
   (serve/web/chats.js:380-387): `go` returns while `document.hidden`, so `follow` (chats.js:494-514) stops retrying.
   The loop exists for "a phone's screen sleeping mid-answer" (chats.js:425-427, :493) - which *is* a hidden tab - and
   behind a plain server there is no upstream proxy to protect, so this is a behaviour change with the feature OFF: a
   break that lands in the dark is taken back up only when the tab is looked at, and a gap past the cap hits
   `Date.now() - t0 > 600000` at chats.js:499 on that first resume and reports the answer as cut off where it used to
   reconnect. The plan does say "stop all polling while the tab is hidden, in every setup", so this follows the letter
   of it; the harm it prevents only exists behind llama-swap, so `unloaded()` (or the supervisor) would be the better
   condition here.
3. **Opening Settings loads the model while unloaded.** `SettingsDialog`'s effect fires six requests on every open -
   `settings`, `mcp`, `api/context`, `api/port`, `api/thinking-budget`, `api/update` (serve/web/settings.js:485) - all
   under `/upstream/<id>/`; only `loadMcp` refuses while the gate is closed (serve/web/server.js:173). Same class as the
   api/tokens hole just fixed (round-1 finding 7): a click that wants the dialog, not the model.
4. **While unloaded the context meter keeps the previous chat's real figure.** serve/web/chats.js:289-296 only
   re-estimates when `S.ctx.tokens == null`, so with the gate closed, switching from a chat that was counted (exact,
   `approx: false`) to another chat with messages shows that count for the new chat and never re-asks; FIX-REPORT.md:49-50
   states this as intended. Comparing `ctxKey` (chats.js:302-303) with the new chat's body, or always re-estimating when
   the draft changed, would keep the estimate honest.
5. **The reconnect loop is still not gate-aware while the tab is visible** (serve/web/chats.js:494-514, the brief's
   reconnect question): the first attempt of a break while unloaded is an upstream request that swaps the model in; it
   is bounded by the 404 at chats.js:503 ending the loop, so it cannot keep a model loaded. Left as-is is defensible
   (the answer itself is what the user is waiting for), but it should be a known deviation rather than silence.

## Nits

6. serve/web/server.js:277-297 and :429-452: `watchReload` (`api/context` every 2 s) and `watchUpdate` (`health` every
   3 s) have no hidden or `unloaded()` check - the accepted deviation (REPORT open q3). Both are only entered from a
   POST the supervisor refuses outright (serve/server.py:7298-7305) or from a health/metrics reload state.
7. serve/web/server.js:80 with :106: while the model is ready `/running` is asked once a second and its round trip sits
   in front of every `/metrics` fetch (REPORT open q1). Caching it for a second would drop both the request and the
   added latency.
8. serve/web/server.js:65 sends `headers()` - Saguaro's own `Authorization: Bearer` (serve/web/lib.js:69-75) - to
   llama-swap's `/running`. Same origin, so nothing new leaves the machine, but llama-swap does not use that key: if
   *it* has a key configured the answer is a 401 (server.js:67) and the gate silently never engages, with nothing in
   the UI saying the guard is off.
9. serve/web/chats.js:382-387: when the wait lands in the dark, `go` keeps its `visibilitychange` listener on the
   document (and its resolved-once `ok`) until the next reveal; harmless, but a break in a tab nobody shows leaves it
   there for the life of the page.
10. serve/test_dashboard.py:1126 and :1179 park retries with 2.5-3.5 s of real sleep each (the module now runs 30.7 s
    here). Still not covered, as FIX-REPORT.md:33-35 and :57-59 say: `sleepOrVisible`/`follow` (closures inside
    `answer`, unexported) and the 5 s gate cadence and the `mcpTick` scheduling.

## The brief's two questions

- **A poll on any path (SSE / reconnect included) while the model is unloaded?** There is no `EventSource` and no
  `WebSocket` anywhere in serve/web; the answer stream is `fetch` + `getReader` (serve/web/chats.js:446-451), so there is
  no browser-side auto-reconnecting SSE to gate. With the supervisor known, the standing paths are gated: `poll`
  (serve/web/server.js:78-85, `/running` only, 5 s), `loadHealth` (:13), `loadMcp` (:173), the token meter
  (serve/web/chats.js:289, :298), and nothing at all is asked of a hidden tab (`poll` returns at server.js:75 *before*
  any fetch, so even the mid-await hide race makes no request in the dark). What can still ask while unloaded: the
  reveal/boot first poll (**M1**), the reconnect loop's first attempt (**L4**, then the 404 ends it), opening Settings
  (**L3**), and the acknowledged reload/update loops (**N1**). None of them is a standing poll; M1 is the one that
  repeats per user action.
- **Does the chat still send, loading the model?** Yes. `canSend` looks only at text/attachments/busy/reload
  (serve/web/chat-view.js:381, button at :425), `send` guards only on reload/busy (serve/web/chats.js:564-567), and the
  answer POSTs `v1/chat/completions` under `/upstream/<id>/` (serve/web/chats.js:526) - which is what llama-swap loads.
  Nothing in the send path consults `S.conn` or `unloaded()`; the "Unloaded" pill and card (serve/web/sidebar.js:21,
  serve/web/monitor.js:823) are display-only, and `S.metrics = null` (server.js:82) is safe for every reader of it
  (serve/web/monitor.js:788, serve/web/sidebar.js:17,250, serve/web/chat-view.js:445, serve/web/settings.js:57,399).

## Also checked

- **Feature OFF (visible, no supervisor): the paths are byte-identical.** `swapAt()` finds no `/upstream/` (or health
  says no supervisor) → `swapReady = null` (server.js:78-80) → the same `fetch("metrics")`/`metrics?requests=all` with
  the same `headers()` (:88), the same 401 → `key` + one toast (:89-91), the same `fails` → `down` (:101-103), the same
  `mcpTick % 10` (:105) and 1 s cadence (:106); the swap-back branch (:86) can never fire because `was` is always
  `null`. `/running` is never asked (tested, serve/test_dashboard.py:1053). The two real differences are the ones the
  plan asks for: the unconditional `emit()` (:99, reachable in a hidden tab only in the mid-await race) and the
  hidden-tab stop, which replaces the old 5 s background poll.
- **Locks, threads, stdout:** nothing here is Python - the diff is page JS, its tests and three markdown files, so no
  non-reentrant lock is nested and no server thread or stdout line is touched. The JS analogue is sound: `polling`
  (serve/web/server.js:48) is a single-threaded reentrancy flag released in `finally` (:107), so it cannot deadlock, and
  a resume swallowed by it is always re-armed by the in-flight poll (every path past the guard arms a timer, :83 or
  :106), so no chain dies; `timer` is cleared before every arm (:74) and on hide (:110), so one chain at a time.
  serve/test_dashboard.py runs node with `capture_output=True` (:871), so the new cases add no interleaved stdout.
- **Generated file:** `serve/qfn_engine.py` and `engines/qfn/serve/server.py` are absent from `git diff --stat` for the
  range, so nothing was owed from tools/saguaro/extract_qfn_engine.py - as the report says (REPORT:44-45). Nothing was
  hand-edited in serve/.
- **Tests really exercise it:** the ten llama-swap cases import the live modules (serve/test_dashboard.py:860 imports
  serve/web/server.js; :1200 imports serve/web/chats.js) and assert exact fake-fetch logs, and the four new ones do
  fail-open in the right direction - the blip case pins `["/running", "/running"]` (:1136), both parked retries pin
  "1 ask in the dark, 2 after the reveal" (:1114, :1172), and the meter case pins `["/running"]` with an approximate
  figure (:1192). The single blind spot is M1 (its `/running` answers `ready`).
- **No paths outside the repo, no secrets:** the diff adds only relative URLs plus the origin-root `/running` (a path,
  from `location.pathname`, server.js:56-57 - and the module's own no-absolute-URL guard at serve/test_dashboard.py:823
  still holds); the new test strings are model ids and copy text; `git status` is clean and scratch/ is gitignored
  (.gitignore:20), so the reviewer's scratch harness is not in the diff.
- **Report accuracy:** the file list, the commit split, 68 + 35, the monitor.js single-line and the four open questions
  all match what is here. Two claims overstate: FIX-REPORT.md:12 "the `/running` gate can engage" at the reveal (it
  engages one poll later - M1), and FIX-REPORT.md:49-50 presents the stale meter figure as intended rather than as the
  wrong-chat display it is (L3).
