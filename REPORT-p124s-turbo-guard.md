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
