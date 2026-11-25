# Fix report: REVIEW-RESULT.md findings (`git diff c3844c07..HEAD` review)

## High - stale 503 assertion in `serve/test_model_api.py` (FIX-FIRST) - FIXED (`73b9049bc`)

`Api.test_swap_on_request` line 141-142: the `POST /models/load {"model": other}` now expects
`(s, b["success"], b["loading"], b["model"]) == (200, True, True, other + ":latest")` instead of the old 503,
and its comment says why the load gets a 200. The chat's 503 assertions (137-140: `503`, `Retry-After: 30`,
`loading` in the body, `restart.assert_called_once()`) are untouched.

Verification: per the plan only `serve.test_dashboard` / `serve.test_models` / `serve.test_server` may run, so
`serve.test_model_api` was not run here. The expectation was established the way the review's High finding was -
by reading the path - and the file compiles (`py_compile`): `_restart_into` stays mocked so `svc.model_restart`
is unset, `_mapi_gate` passes (management on, loopback), `switch_model` succeeds again (`_restart_busy` sees no
restart/reload, `other` is not the running config), and the `load=True` branch at `serve/server.py:6729` writes
exactly that body with `model = model_api.public_name(p) + ":latest"` == `other + ":latest"`.

## Low - "page keeps working when the placeholder is absent" untested - FIXED (`51219a259`)

`serve/test_dashboard.py`:
- `Dashboard.test_index_still_serves_without_the_placeholder_or_the_version` (live `GET /`): with
  `serve.server.SAGUARO_VERSION` patched to `None` the page answers 200 with `href="web/icon.svg?v="` (the
  empty-version form), and an `index.html` without the placeholder (via a patched `serve.server.ROOT`) is
  served byte-identical - the `bytes.replace` no-op the plan asked for, now pinned by behaviour.
- `Offline.test_the_index_favicon_version_is_the_placeholder` (the review's suggested source-text check):
  `index.html` carries `?v=__VERSION__` and no hard-coded `1.2.1f`.

## Low - step-2 test asserts source text, not behaviour - NOT CHANGED (on purpose)

The review itself calls it acceptable ("the plan allowed a new small test"); `test_about_shows_only_saguaros_version`
stays as it is. Rewriting it as a Node page-logic test is not cheap, and the plan caps every change at minimal.

## Nit - the 200 message embedded `:latest` in the model name - FIXED (`fa74cc0c8`)

`serve/server.py:6731` now names the model `model_api.public_name(p)` - the same form the 503 message uses -
so the message reads `loading model other: ...`; the `model` field of the body still carries `other:latest`
(llama.cpp's name), which `serve/test_server.py:2526` pins. No test pinned the old `:latest`-in-message text
(only the `Saguaro starts again with it` substring).

## Nit - VERSION interpolated into HTML unsanitised - NOT CHANGED (on purpose)

The `VERSION` file is repo-controlled and same-origin, as the review notes ("not exploitable"); the same string
already ships in `/health` JSON. Escaping would add code the plan's minimal-change rule has no room for.

## Test runs (the plan's three modules, one at a time, after the commits above)

- `serve.test_server` - 172 tests, OK (after the server.py message change)
- `serve.test_dashboard` - 62 tests, OK (60 before + the 2 new)
- `serve.test_models` - 62 tests, OK

`serve.qfn_engine.py` was not touched and needs no regeneration: the extractor works from
`engines/qfn/serve/server.py`, which these fixes did not change (the plan's Don't list keeps `engines/` away).
