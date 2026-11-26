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
   dict `turbo`, agents and API clients included: serve/server.py:3380-3382 in `Service.check_request`.
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

## Step 3 - tests
