# P124s - Turbo is never on for an API request unless that request asks - REPORT

## Step 1 - how Turbo reaches a request today (all paths)

Per-request lossy Turbo = the engine line's `turbo_drop=` / `turbo_skip=` keys, set from `sampling["turbo"]`.

1. **The request's own `turbo` field** - checked/normalized in `check_request_sampling`
   (serve/server.py:1224-1225), kept by `Service.check_request` (serve/server.py:3362), turned into engine-line
   keys by `turbo_live.turbo_keys` (serve/server.py:1935, 1954; same in serve/qfn_engine.py's driver).
   Entry points that carry it: `/v1/chat/completions` (`_openai`, serve/server.py:8499-8503),
   `/v1/responses` (allowed parameter, serve/responses.py:109, 270; handler serve/server.py:8421-8429),
   `/v1/messages` (`_anthropic`, serve/server.py:8612-8616). Ollama's `/api/chat` / `/api/generate` rebuild the
   body in `model_api.chat_to_openai` / `generate_to_openai` (serve/model_api.py:172-193), so no `turbo` field
   can reach the pipeline from there.
2. **The model's live Turbo default `svc.turbo_default`** - applied to EVERY request that does not carry its own
   dict `turbo`, agents and API clients included: serve/server.py:3380-3382 before this change (3381-3382 now).
   This is the path that breaks Ken's rule. What can put it on:
   - the sidecar `<config>.turbo.json` read at start (serve/server.py:10039-10047, `turbo_path` at 10040);
   - POST `/api/turbo/options` - the control panel's "Turbo, live" (serve/server.py:8142-8143 -> `_turbo_options`
     8342-8359 -> `set_turbo_default` 3460-3476, which also stores the sidecar);
   - the Turbo button's `set_turbo` live half (serve/server.py:4042-4048) from `models.TURBO_LIVE_DEFAULT`
     (serve/models.py:1088 - empty today, so a leftover-state path only).
3. `NO_LIVE_TURBO` (serve/server.py:84, 3383-3384) forces `turbo=None` for ik2/hy4: a suppression, not a source.
4. Display only (never turns Turbo on): the Monitor badge `stat.turbo` (serve/server.py:6027, 4639, 6535).
5. Out of this path: the Turbo button's engine half (`models.turbo_enable`, serve/models.py:1056-1216) writes
   context/expert-skip env/drafting into the run config for the whole engine - Maya's Turbo, page-gated by
   `_own_page` (serve/server.py:8330), not a per-request skip/drop field.

## Step 2 - the change

One rule point: `Service.check_request` (serve/server.py) now feeds `turbo_default` only to the web app's own
chats - the requests `_openai` marks `req["_api"] = "web"` (their `strata_resume: true`, the same marker the
Monitor's table uses, serve/server.py:8505). Every dialect handler overwrites `_api` itself before
`check_request` (7618 openai/Ollama, 8429 responses, 8505 chat, 8618 anthropic), so a client cannot claim it.

- serve/server.py:3378-3382 - the default applies `if req.get("_api") == "web" and not isinstance(...)`;
  an API request without its own dict `turbo` keeps nothing: the engine line stays today's exact line.
  Its own `turbo` field (checked at 1224) is untouched - that stays the only way an API request gets Turbo on.
  `NO_LIVE_TURBO` (3383-3384) still forces exact for ik2/hy4 whatever the source.
- Doc words of the old rule corrected to the new one: `set_turbo_default`'s docstring (agents included -> the
  app's own chats), GET /api/turbo/options and `_turbo_options` comments + its print line, the start-up sidecar
  line, and `check_request_sampling`'s comment (absent = exact).
- serve/web/turbo.js - the "Turbo, live" panel words, its section comments and the toast no longer promise the
  apps and the agents: the setting is this app's own chats; API requests run exact unless they ask. The page
  needs no behavioural change: chats.js still sends the chat's own `turbo` every turn (serve/web/chats.js:436-437)
  and its requests carry `strata_resume: true`, so the panel's choice keeps applying to them server-side.

No config, no sidecar format, no engine line, no /api/turbo (the engine's Turbo button) change. serve/qfn_engine.py
untouched (nothing in engines/qfn/serve/server.py changed, so no re-extraction was due).

## Step 3 - tests

serve/test_turbo_live.py:

- new `ServerSide.test_the_api_never_borrows_the_pages_turbo`: sidecar ON and the page's last choice ON
  (POST /api/turbo/options stores both - it asserts `<config>.turbo.json` exists); a `/v1/chat/completions`, a
  `/v1/messages` and a `/v1/responses` request without Turbo fields reach the engine with none; each with its own
  `turbo` field reaches it with it on; the web app's own request (its `strata_resume` marker, new `page()`
  helper) still follows the panel's choice, its own `turbo` wins and an explicit off opts out; panel off, exact
  again.
- `test_the_live_global_setting_applies_and_persists` followed the rule: with the global on, a plain API `chat()`
  is now None (was the global), `page()` takes it; the sidecar persist/off lines unchanged.
- new `page()`, `messages_api()`, `responses_api()` helpers beside `chat()`.

Runs (one at a time, as the plan names them):

- `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_turbo_live -v` -> Ran 14 tests, OK (13 before, +1)
- `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_server` -> Ran 179 tests, OK

## Files changed

- serve/server.py - the guard (check_request) + the old-rule doc words (set_turbo_default, both
  /api/turbo/options comments and the two prints, check_request_sampling's comment)
- serve/web/turbo.js - the "Turbo, live" panel words, section comments and the toast
- serve/test_turbo_live.py - the tests above
- serve/qfn_engine.py untouched (nothing changed under engines/qfn/serve/, so extract_qfn_engine.py was not due)

## Open questions

- The page's marker is `strata_resume: true`, the server's existing own-page identifier (the Monitor's request
  table, serve/server.py:8505). An API client that deliberately sends the app's page marker could still inherit
  the panel's choice; sealing that would mean the page stamping its loaded choice onto every request body and
  dropping the server-side default entirely - a page-behaviour change this plan's "smallest change" rule
  excluded. Real agents (opencode, Hermes, /v1 callers) never send it.
- Ollama's /api/chat / /api/generate rebuild the body (serve/model_api.py:172-193), so they cannot carry their
  own `turbo`: they now always run exact. If an Ollama-dialect client should ever ask for Turbo, model_api needs
  a field mapping - deliberately out of scope.
- The Turbo button's engine half (config env knobs, drafting, models.py:1056-1216) still applies to the whole
  engine for every client when the page switches it on; it is page-gated (`_own_page`) and outside this plan's
  per-request rule, which covers the skip/drop fields.
