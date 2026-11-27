# REPORT - P125m: the page before the model, phase 2 (Saguaro 1.2.5)

Branch `saguaro/q1-p125m-page-phase2`. Three parts, one commit each, then this report. All work done in the
worktree at `/data/llm/saguaro-q1-p125m-page-phase2`; only relative paths used except the venv Python. No
sub-agents. serve/qfn_engine.py touched only via `tools/saguaro/extract_qfn_engine.py`.

---

## What changed

### Part 1 - switching without blanking  (commit df8093c52)

The second half of a switch (saguaro.py restarting with the chosen config) now says so, and a page that already
shows the app keeps it on screen and waits instead of dropping to the loading screen.

- `saguaro.py` `after_server`: when the switch note said why == "switch", the next server starts with
  `SAGUARO_SWITCH=1` (cleared to "" for the settings/port restarts, so a switch is never implied).
- `serve/server.py`:
  - `LoadState(..., switching=...)`; `main()` sets it from `SAGUARO_SWITCH=1`.
  - `/health` while loading gains `{"switch": true}`.
  - the starting service keeps `/v1/models` (and `/models`) answering **200** with the coming model listed
    `status {"value":"loading"}` (still behind the API key, still no key needed to read /health).
  - the chat routes (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`) answer **503 + Retry-After: 5**
    at once with `{"error": {...names the model...}, "loading": "<model>"}`.
  - a plain start (no switch) is byte-for-byte phase 1 everywhere (503 "the model is loading").
- `serve/web/server.js`: `switchingView(sw)` (name / NN% / step or line / failed), `watchSwitch()` (a /health a
  second, nothing asked in a hidden tab, reload on the first 200 after the gap = "the app returns to normal"),
  `probeSwitch()` (a booted app whose poll failed finds the starting service and sets S.switching); `poll()`
  asks nothing while the bar stands.
- `serve/web/state.js`: `S.switching`. `serve/web/models.js`: `watchRestart` sets the bar at once when the page
  asked (and clears it if Saguaro never comes back). `serve/web/chat-view.js`: a `.switchbar` above the box,
  `canSend`/`note()` follow it; `serve/web/chats.js`: `send()`/`regenerate()` wait. `serve/web/sidebar.js`: the
  status pill reads the bar's text. `serve/web/app.css`: the switchbar styles.

### Part 2 - real load progress  (commit df1a041e2)

The percent and the step's words come from the engine's own load lines, not a timer.

- `serve/server.py`: `LOAD_MILESTONES`, a small ordered table per family - glm, deepseek (ds4 through maya),
  qfn, ik2, hy4, minimax (mm3) - of (pattern, percentage, step words). A counter line "N / M" sits between its
  endpoints (`_between`); a line no table entry knows keeps the last value; the mock engine has no table and
  never moves the bar. `load_progress(family, line, pct, step)` applies one line.
- `LoadState` gains `family` and the progress reader: `_advance_progress()` reads the engine log from where
  this start began (only complete lines, a 256 KiB step a look), runs each line through the family's table, and
  keeps `pct` / `step`. `main()` picks the family with `maya_quants.family_of(cfg)`. `/health` gains
  `pct` (0-100 | null) and `step`; the phase-1 `line` and `elapsed_s` stay.
- `serve/testdata/load-{glm,ds4,qfn,ik2,hy4,mm3}.log`: recorded load logs, one per engine, the source of the
  table's lines. `.gitignore`: `!serve/testdata/*.log` (the `*.log` ignore would otherwise drop the fixtures).
- `serve/web/starting.js`: `startingView` passes `pct`/`step` through; the loading screen draws a real progress
  bar with the step's words when pct is present, and falls back to the phase-1 elapsed/line view before the
  first known line. `serve/web/starting.css`: `.start__prog`. The switch bar reads the same fields.

### Part 3 - a crash mid-load  (commit b99538ed5)

An engine that dies during a load shows a page with its exit code and last ~20 log lines and a choice; the port
stays up.

- `serve/server.py`:
  - `EngineStartFailed(RuntimeError)` carries `exit_code` and `lines`; `serve.server.StrataEngine` raises it on a
    start whose process died before READY, after reading the exit code (`exit_code()`) and the engine's own last
    lines from the log (`engine_log_tail(log, log_start)`), and closes the dead engine's pipes + log handle.
  - the batch driver `engines/qfn/serve/server.py` attaches the same two fields to its `EngineDied`
    (`start_log_tail_lines`); `serve/qfn_engine.py` re-extracted (extraction verified byte-stable).
  - `LoadState.fail()` reads `exit_code`/`lines` off the exception (falling back to the log tail if the engine
    died but attached no lines), `crashed`/`restartable()` mark the crash and whether a restart can happen.
  - `engine_exit_words`: signal named (SIGKILL says "the system's out-of-memory killer ends it this way"),
    Windows NTSTATUS decoded (`_WIN_CRASH`), else "exit code N".
  - `/health` of a failed load gains `crashed`, `exit_code`, `exit`, `lines`, and `again` (the config name, only
    where this page can actually restart: not under a supervisor, and a saguaro.sh / START-SAGUARO.bat start).
  - `start_again()` + `POST /api/load-again`: notes this very config in models.SWITCH_FILE and ends with
    RESTART_EXIT; key and same-origin checks apply as the last-good button's; a 409 explains every refusal. A
    plain failed load (a config error, no engine died) keeps phase 1's page exactly.
- `serve/web/starting.js`: the crash view draws the exit code, a scrollable `.start__log` box of the engine's
  lines, a "Load <config> again" button and the last-good button (first = primary, second = secondary); the
  plain-failed view unchanged. `serve/web/starting.css`: `.start__log`.

---

## Files

- changed: `saguaro.py`, `serve/server.py`, `engines/qfn/serve/server.py`, `serve/qfn_engine.py` (generated),
  `.gitignore`
- page: `serve/web/state.js`, `serve/web/server.js`, `serve/web/models.js`, `serve/web/chat-view.js`,
  `serve/web/chats.js`, `serve/web/sidebar.js`, `serve/web/starting.js`, `serve/web/app.css`,
  `serve/web/starting.css`
- new tests: `serve/test_switch.py`, `serve/test_load_progress.py`, `serve/test_load_crash.py`
- new fixtures: `serve/testdata/load-{glm,ds4,qfn,ik2,hy4,mm3}.log`

## Tests run and results

Run one at a time with `/opt/llm/saguaro/.venv/bin/python -m unittest <module>` (Node page tests included, and
the child-process tests that run `serve.server.main()` in a subprocess with the mock engine):

| module | result |
|---|---|
| `serve.test_switch` (new, part 1) | Ran 13 - OK |
| `serve.test_load_progress` (new, part 2) | Ran 12 - OK |
| `serve.test_load_crash` (new, part 3) | Ran 18 - OK |
| `serve.test_starting` (phase 1, affected) | Ran 24 - OK |
| `serve.test_server` (StrataEngine start path) | Ran 183 - OK |

The four page-facing modules run together as well: `serve.test_starting serve.test_switch
serve.test_load_progress serve.test_load_crash` -> Ran 67 - OK.

Headless page checks (`tools/page_check.py`, chromium is present on this PC) - one per part, against a live
fake-engine server on a scratch port:

- part 1: the switch's loading screen -> `PASS ... the page drew the loading screen`
- part 2: the bar screen (a glm log grown line by line) -> `PASS ...`
- part 3: the crash screen -> `PASS ...`; a DOM dump of the crash page showed `start__log`, `start__error`,
  "stopped while it was loading", "exit code 137", "out of memory" and the "Load crash-cfg.json again" button
  (no last-good button, correctly: nothing else loaded before).

`py_compile` on every changed .py, `node --check` on every changed .js, and
`tools/saguaro/extract_qfn_engine.py` re-run (no drift on `serve/qfn_engine.py`) all clean.

## Open questions

1. **The percent is an estimate of the shape of a load, not a byte ratio.** The milestone tables fix where each
   engine's known lines sit in a usual load of that family. Lines that are NOT in a table keep the last value, so
   an engine path this server has never seen reads as a stall rather than a wrong jump. Should each step's
   percent be tuned against the real logs on the reference machines (the docs' DGX/7900/RTX boards) before
   release, and the ik2/hy4/mm3 tables widened (those three log little while they load)?
2. **Which "loading" percent does the *real* service answer?** Part 2 reads the engine log through the starting
   service (phase 1's model). A context reload (`/api/context`, the `reloading` state) has its own progress that
   does not come from the LoadState reader - the switch bar does show the engine log lines (it polls /health of
   the starting service), but the "reloading" status still shows its own waiting/running state. Fold the reload
   into the same milestone reader, or leave them separate?
3. **The switch bar and a failed switch.** Part 3 built the crash page on the *starting* service (a page that
   opens the port fresh). A page that already shows the app and then loses the switch to a crash currently keeps
   its switch bar up ("GLM 5.3 Flash did not start") without the exit code / last-lines box. Should the booted
   page show the same crash details, and the "load again" / "last config that worked" buttons there too?
4. **The llama-swap (supervisor) shape.** Under a supervisor `again`/`last_good` are never offered (the
   supervisor owns starts) and the switch flag is unused; the crash page still names the exit code and the
   engine's lines. Is a supervisor's own retry UI wanted there, or is pointing at the supervisor the answer?
5. **CHANGELOG / release notes**: not touched here (the 1.2.5 notes section does not exist yet on this branch);
   phase-2's three parts will want lines when the release entry is written.
6. The `ResourceWarning: unclosed file <engine.log>` lines that show in some subprocess test runs are from the
   *other* pre-existing engine-start test paths (test_starting shows them at HEAD too), not from the crash path
   (which now closes the dead engine's handles).
