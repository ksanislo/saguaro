# P1: three small UI / API fixes - report

Three commits, one per plan step. No sub-agents, relative paths only, no network, no
`engines/` / `saguaro.py` / `serve/turbo*` / Monitor edits. `serve/qfn_engine.py` was not
touched (no edit needed - the plan's changes live in the shared `serve/server.py`).

## What changed

### 1. Favicon cache-bust version comes from VERSION (commit 87ffd13af)
- `serve/web/index.html:9` - `web/icon.svg?v=1.2.1f` -> `web/icon.svg?v=__VERSION__`.
- `serve/server.py` (`do_GET`, the `path == ""` branch, ~line 7132) - after reading
  `index.html`, `body.replace(b"?v=__VERSION__", f"?v={SAGUARO_VERSION or ''}".encode())`.
  A no-op when the placeholder is absent, so a hand-served / cached `index.html` still works;
  when `SAGUARO_VERSION` is `None` it leaves `?v=`.
- Test `serve/test_dashboard.py` `Dashboard.test_index_busts_the_favicon_cache_with_the_version`
  - `GET /` carries `web/icon.svg?v=<SAGUARO_VERSION>` and no `?v=__VERSION__`.

### 2. Engine version dropped from About / labels (commit b382727fd)
- `serve/web/settings.js:426` - the About > Version row was
  `Saguaro v${ver}${eng.version ? ` · engine ${eng.version}` : ""}` -> now just
  `Saguaro v${ver}`. Saguaro's own version (`ver`) stays; the engine version is only dropped
  from what the user sees. The Model-card label at `settings.js:85` shows the engine *kind*
  (`ENGINES[eng.engine_kind]`), not a version, so it is left as is.
- The server's JSON is untouched - `/metrics`' `engine.version` is still returned.
- Test `serve/test_dashboard.py` `Offline.test_about_shows_only_saguaros_version` - the
  settings.js source keeps `Saguaro v${ver}` and no longer contains `engine ${eng.version}`.
  (A source-text check: `facts()` is not an exported page function, so a Node page-logic call
  would need the whole component module stubbed; this is the small test the plan allows.)

### 3. POST /models/load restart path answers 200 (commit 3878e1b6d)
- `serve/server.py` `_mapi_target(self, name, load=False)` - gained a `load` flag. In the
  switch-started branch (the old "starts again with it" 503), when `load=True` it now answers
  `200 {"success": true, "loading": true, "model": "<name>:latest", "message": ...}` with no
  `Retry-After`; the chat path (`load=False`) keeps its retryable 503 + `Retry-After`. The real
  failures (404 not found, 409 cannot-switch / already-switching) still return their error -
  they `return False` before this branch.
- `serve/server.py` `_llamacpp_model` (`/models/load` swap, ~line 6547) now calls
  `self._mapi_target(name, load=True)`.
- Page caller: the only in-page caller, `serve/web/models.js` `loadOrUnload`, posts
  `models/load` with an empty body, so it never reaches the switch/restart path and never
  showed the old error - no page change needed. (The page's own model switch uses
  `POST /api/models/switch`, a different route.)
- Test `serve/test_server.py` `ModelManagement.test_a_load_that_switches_models_answers_200_not_the_chats_503`
  - with `model_listing` stubbed to a running model, `find_config` stubbed to `other.json` and
    `model_restart` set, `POST /models/load {"model":"other"}` -> 200 with `success`, `loading`,
    `model == "other:latest"` and the message; `POST /api/generate {"model":"other", ...}` still
    -> 503 with the "starts again with it" error.

## Tests run (only the modules the plan names, one at a time)
- `python -m unittest serve.test_dashboard` -> 60 tests, OK (covers steps 1 + 2).
- `python -m unittest serve.test_models` -> 62 tests, OK (regression; touched shared server.py).
- `python -m unittest serve.test_server` -> 172 tests, OK (covers step 3).

## Open questions / notes
- `index.html`: when the `VERSION` file is missing (`SAGUARO_VERSION is None`) the favicon URL
  becomes `web/icon.svg?v=`; left as-is because a versionless favicon is harmless and the plan
  asked only to keep the page working when the placeholder is absent.
- Step-2 test asserts on settings.js source text rather than running `facts()` in Node; the
  function is private and would require stubbing the whole component module for little gain.
