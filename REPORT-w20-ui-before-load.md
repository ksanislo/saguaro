# REPORT: W20 — the web page up before the model loads (1.2.1 item 20, phase 1)

Five commits on this branch, one per plan step (`git log`):

| step | commit    | what |
|------|-----------|------|
| 1    | a0bab6052 | bind first: the port answers from the first second (done before this session, per the resume note) |
| 2    | 3cfb30cb9 | /health while loading = 503 + `{status, model, elapsed_s, line, supervisor}`; the /health readers checked (also done before) |
| 3    | 58acba53f | a load that FAILS: the standing failed page, the exit cases, the last-good config and its button |
| 4    | 4a4427af3 | the page: the "loading the model" screen that becomes the app on the first 200 |
| 5    | 5f6db8aab | serve.test_starting: the start order, 503 shapes, failure, hand-over, llama-swap, the screen |

## What changed

### Step 1/2 (already committed; read, not rewritten)
`serve/server.py`: `LoadState` + `make_loading_handler` + `serve_loading` bind the port in `main()`
before the engine loads (the bind replaces the old port-free preflight); the page's files, `/health`
open; every engine path answers `503 {"error": "the model is loading"}` with `Retry-After: 5`
(never queues). `take_over()` swaps `httpd.RequestHandlerClass` on the same listening socket — no
gap, no second bind (`socketserver` reads the handler class per connection).

### Step 3 — a load that fails (`serve/server.py`, `serve/models.py`, `.gitignore`)
- Finished the working tree's uncommitted step-3 code (`LoadState.fail()`, `read/write_last_good`,
  `ends_on_failed_load`, `start_last_good`, the `/api/last-good/start` POST with the key check and
  the same-origin check, `failed_seen`) and wired it into `main()`:
  - the whole engine-load block (tokenizer → vision → `StrataEngine`/QFN/mock) is wrapped; on any
    `Exception` the port keeps answering: `/health` → `503 {"status": "failed", "error": ...,
    line, elapsed_s, ...}` and the page shows it;
  - under a supervisor the process waits for one `/health` so llama-swap's check sees the failure
    (capped at 60 s, so a non-polling supervisor still ends the wait), then ends with rc 1;
    with `--exit-on-fail` (new flag) it ends at once; a direct start stands with the failed page
    until Ctrl+C/SIGTERM (rc 1);
  - a successful load records the config + time in `.maya-last-good.json` in the server's state
    folder (the one holding `models.SWITCH_FILE`; gitignored). The failed page then offers the
    server's own note as `last_good` — never under a supervisor (it owns starts) and never when it
    is the very config this start failed with;
  - the button `POST /api/last-good/start` leaves the same `models.SWITCH_FILE` note the Models
    page's Switch leaves and ends the process with `RESTART_EXIT`, so saguaro.py starts the config
    that worked; it refuses (409, with the reason) under a supervisor, with no note, for the same
    config, or when nothing started this server in a restarting loop (prints the hand-start line).
- `serve/models.py`: `LAST_GOOD = ".maya-last-good.json"`; `.gitignore`: the new note.

### Step 4 — the page (`serve/web/`)
- `starting.js` (new): `StartingPage` — model name, elapsed time, the engine's last line; a failed
  load shows the error, the last line and the "Start with <config> · the last config that worked"
  button when `/health` says a `last_good` (after the click: "Starting again with …"; 409s show
  inline). `startingView()` is the pure text-shaping. `starting.css` (new), linked from
  `index.html` (the starting service serves both).
- `server.js`: `watchStarting()` asks `/health` once a second while it says loading/failed.
  Today's rules hold: a hidden tab asks nothing (seen again, the watch resumes at once), and
  behind llama-swap off the card only its `/running` is asked (`/health` would load the model).
  `loadHealth()` was split into fetch + `applyHealth(h)` so the gate's first 200 body boots the app
  directly — a page load still asks `/health` exactly once (the dashboard review's nit holds).
- `app.js`: the screen stands on `#app` until `watchStarting` is ready; then `boot()` renders the
  app and runs today's chain (chats, health, MCP, Turbo, poll) with that health. Idle-unload and
  reload states are untouched (that path keeps `/health` 200 and the normal page, as today).
- `state.js`: `S.starting`.

### Step 5 — tests (`serve/test_starting.py`, new; 22 tests)
- Loading service in-process: page + screen files served, `/health` shape (trimmed log line),
  every engine path 503 + `Retry-After: 5` answering in <2 s (no unbounded queueing), OPTIONS,
  the API key guarding exactly what it guards today (incl. the button), the failed shape +
  `failed_seen`.
- Last-good: note round-trip (missing/corrupt/deleted config → nothing offered), the offer's
  exclusions, the button's 409s + 403 + the note it leaves (`why: switch`, port) + the
  restart-after-answer thread; `ends_on_failed_load` truth table.
- Hand-over: a 20 ms `/health` client across `take_over` sees 503→200 with zero refused
  connections and the same port; after it the real service answers (version, a mock chat, `/`).
- Real `main()` in children with the mock engine patched slow/dead (wrapper script): the page and
  the loading health answer while a 10 s engine loads (elapsed < 9 s), `Retry-After` on chat,
  the last-good note after success, SIGTERM while loading ends at once and closes the port;
  a failed load stands with `error`/`last_good`, `--exit-on-fail` exits 1 with the log line,
  under `SAGUARO_SUPERVISOR=llama-swap` it exits (rc 1) shortly after its `/health` was seen and
  never offers `last_good`, and the button ends with `RESTART_EXIT` + the switch note.
- The screen in Node (as test_dashboard's page-logic tests): `startingView` shapes, the one-second
  watch booting on the first 200 (and carrying its body — no second `/health`), the hidden tab,
  the llama-swap off-card `/running`-only rule.

## Tests run (the plan's list, one module at a time)

| module | result |
|--------|--------|
| serve.test_server          | 178 tests, OK |
| serve.test_dashboard       | 79 tests, OK |
| serve.test_models          | 63 tests, OK |
| serve.test_model_api       | 28 tests, OK |
| serve.test_new_engines_ui  | 38 tests, OK |
| serve.test_starting (new)  | 22 tests, OK |

Plus scratch smokes during development (`scratch/smoke_fail.py`, `scratch/smoke_up.py`: the
failure exits, a normal mock start answering 503-then-200 and SIGTERM ending it rc 0).
`serve/qfn_engine.py` was not touched (no engines/ change needed), no card runs, no network.

## Open questions
- **The supervisor's 60 s cap**: a supervisor that sets `SAGUARO_SUPERVISOR` but never polls
  `/health` makes a failed load linger up to 60 s before the process ends (so one that *does* poll
  sees `status: failed` rather than a bare closed port). 60 s is my pick; llama-swap's own health
  check interval decides in practice. `--exit-on-fail` skips the wait.
- **The button's reach**: it can only restart when `SAGUARO_RESTART_ON_UPDATE` is set (started via
  saguaro.sh / START-SAGUARO.bat); a bare `python serve/server.py` gets the refusal text with the
  hand-start command. Deliberate (the note is read by saguaro.py's RESTART_EXIT loop), but it means
  the button's visibility (`last_good` in `/health`) slightly overpromises in that one case.
- **The screen's elapsed counter** follows the server's `elapsed_s` at the 1 s poll, so it ticks in
  whole-server-seconds; no local ticker (kept simple, matches the server's clock).
- Phase 2 (cached `engine.info` so a restart serves the full UI at once) is out of this task,
  per the plan; the hand-over screen therefore only stands before the *first* load of a process.
