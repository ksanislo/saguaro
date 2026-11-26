# P124d - report: a real browser page load in the release checks

The plan: 1.2.2 shipped a page that died on load (`startingView(null)` threw on the first paint) while every API
smoke passed because none of them opened the page. A check that opens the page now exists, is tested, and is in
the per-machine release list.

## What changed

### 1. `tools/page_check.py` (new) — the check

`python3 tools/page_check.py <base-url> [--key-file F]` loads the page in headless Chromium and answers one
PASS/FAIL/SKIP line; exit 0 on PASS or SKIP, 1 on FAIL.

- **The browser**: `chromium`, `chromium-browser`, `google-chrome` from the PATH (plus the usual macOS/Windows
  install paths); a clear `SKIP ...` line and exit 0 when the PC has none.
- **The load**: `--headless --dump-dom --enable-logging=stderr --v=0 --virtual-time-budget=30000
  --timeout=30000`, a throwaway `--user-data-dir` profile (removed at the end); the virtual-time budget runs the
  page's timers and waits for its fetches, so the dumped DOM is the rendered one and the CONSOLE lines are in
  hand without a DevTools client. A subprocess timeout (45 s) is the safety net.
- **The render**: the shell is `class="... shell ..."` in the dump (`.shell` only appears once Preact drew the
  app — `index.html` ships an empty `#app`), or the loading screen `.start` while the model still loads (that is
  the first thing `app.js` renders). Class tokens are matched whole, so `start__card` (inside the screen) is never
  `.start` and `restart` is never `.start`; `shell--closed` is the shell.
- **The failure side**: every CONSOLE line in the chromium stderr log is an error — an uncaught throw ("Uncaught
  TypeError: ...") and a `console.error` both write one, in both formats chromium has used (`CONSOLE:1` new,
  `CONSOLE(34)` old); chromium's own WARNING lines (sandbox, D-Bus) are not console messages. This is exactly the
  1.2.2/W20 shape: a `startingView(null)` throw lands as `Uncaught ...` with `#app` empty → FAIL twice over.
- **The key**: the page and `/health` stay open behind an API key (verified against a `--api-key` mock server:
  `/` 200, `/health` 200, `/v1/models` 401 → 200 with the Bearer key), so the browser never sees the key — the
  page renders its shell anyway. `--key-file F` reads the install's key and proves it against `/v1/models`: a
  refusal (401/403) is a FAIL, a 503 means "the model is still loading" (then the page legitimately shows the
  loading screen, not a shell) and is noted, not failed.
- One fix found while testing: the URL only gets its `/` appended when the path is empty (`http://h:8095` →
  `http://h:8095/`; `file://.../page.html` and a llama-swap path prefix are left as given — a trailing slash
  there was an `ERR_FILE_NOT_FOUND`).

### 2. `tools/test_page_check.py` (new) — the tests

- **Parser** (fixed inputs, no browser): both CONSOLE formats, several lines in order, escaped quotes/newlines
  decoding, and chromium's own WARNING-only log → no errors.
- **Decision** (fixed inputs): the shell passes, the loading screen passes, nothing drawn fails (the reason names
  the 30 s), a console error fails whatever drew (the reason names the first), `start__card` alone is not `.start`.
- **Real run**: a local `python -m serve.server --engine mock` on a free port (started by the test, `/health`
  polled up to 30 s), then `sys.executable tools/page_check.py <base>` — exit 0 and a `PASS ... the page drew`
  line; `skipUnless` a chromium exists (the tool's own SKIP case).

### 3. `docs/RELEASE-CHECKS.md`

The per-machine list ("What still needs a person") gains the page check: `tools/page_check.py` against each
install's page, on llama-swap machines the page of the install it runs too (its own port, with llama-swap holding
it loaded — the page only answers while that Saguaro runs), `--key-file` for the install's key, and the SKIP case
(no chromium: open the page by hand there).

`serve/web/` was not touched; `serve/qfn_engine.py` and `extract_qfn_engine.py` are not involved in this plan.

## Files

- `tools/page_check.py` (new)
- `tools/test_page_check.py` (new)
- `docs/RELEASE-CHECKS.md` (one list item added)

## Tests run

- `python -m unittest tools.test_page_check -v` (the only module this plan names; venv python): **9 tests, OK** —
  run twice (once after the regex fix below, once against the final commits). The real-page test ran on the box's
  Chromium 156 against a live mock server.
- By hand while developing (not tests): PASS against the mock server with and without `--api-key` + key file;
  FAIL for a refused key; FAIL for a `file://` page whose script throws before anything draws ("1 console
  error(s), first: Uncaught ReferenceError: startingView is not defined"); FAIL for an unreachable port
  ("neither .shell nor .start rendered within 30 s"); SKIP with an empty PATH.
- One fix found by the test suite itself: the parser regex took the new `CONSOLE:1` format only — the old
  `CONSOLE(34)]` needs the `)` before the `]`; fixed and re-run.

## Commits (one per plan part, plus the fix)

1. `tools/page_check.py` — the check
2. `tools/test_page_check.py` — the tests
3. `docs/RELEASE-CHECKS.md` — the per-machine list
4. the `CONSOLE(34)]` regex fix (part 2's test caught it)

## Open questions

- **Console noise today**: the page logs nothing, so "any CONSOLE line fails" costs nothing now; if a future page
  adds an intentional `console.log/warn`, the check cannot tell console methods apart from the log line alone and
  would fail — decide then whether to filter severities or keep it strict.
- **The nightly does not run the page check** — the plan only puts it in the per-machine (person) list; the
  nightly box has chromium here but wiring a per-machine page URL into `saguaro-nightly.json` was out of scope.
- **Behind llama-swap**: the page is only served while that Saguaro instance is loaded (measured reasoning, not a
  llama-swap run — none on this box), so the docs tell the person to have it loaded; a llama-swap URL of its own
  (`/` on the swap port) serves the swap's UI, not Saguaro's page.
- The `--key-file` check needs `/v1/models` to be the right probe: it is the cheapest key-gated endpoint; no
  other endpoint is checked.
