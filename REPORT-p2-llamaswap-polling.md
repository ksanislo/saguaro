# P2 report: an open Saguaro page no longer keeps a model loaded under llama-swap

Two commits, as planned: `0a2969cb1` (the page) and `dc4a98235` (its tests).

## What changed

**The polling loop (serve/web/server.js).** Before each poll the page now asks llama-swap's own
`GET <root>/running` - the origin two levels up (`location.pathname`'s part before `upstream/`, the same root the
Models page already uses for `/v1/models`), whose `{"running": [{"model": id, "state": "ready"|...}]}` loads
nothing. It is only asked when the setup is known to be llama-swap: the path matches `/upstream/<id>/` **and**
`/health` says `supervisor: "llama-swap"`.

- Our id not `"ready"`: no `/metrics`, no `/health`, no `/mcp` (all three refuse while the gate is closed:
  `loadHealth`/`loadMcp` return early on `unloaded()`). The Monitor's figures give way to a quiet state -
  `S.conn = "unloaded"`, `S.metrics = null` - and only `/running` is asked, every 5 s, until the model is back on
  the card. When it is ready again the page refetches `/health` once (the model's facts) and resumes the 1 s
  `/metrics` poll.
- `/running` does not answer (404, a key it refuses, no network): `swapReady = null`, no gate - poll as today.
- Not behind llama-swap (no `/upstream/` path, or no supervisor in health): the polls are exactly as before;
  `/running` is never asked.
- Hidden tab, in every setup: `poll()` asks nothing and schedules nothing; the health retry is parked
  (`healthWait`) instead of re-armed. The `visibilitychange` handler clears the timer when the tab hides and calls
  `poll()` (and a parked `loadHealth`) the moment it is visible again. A hidden tab no longer polls at 5 s as it
  did before - it stops.
- Also: one poll at a time (`polling` flag, single `timer`), so a `visibilitychange` resume cannot start a second
  chain. `stopPoll` is exported for the tests.

**The unloaded state shown (serve/web/sidebar.js, serve/web/monitor.js, serve/web/state.js).** The status pill says
"Unloaded" (muted, quiet) for the new `conn` value, and the Monitor's waiting card says
"Unloaded - the next message loads it. llama-swap keeps one model on the card." The comment on `S.conn` names the
new state.

The chat itself is untouched: a message still posts to `/v1/chat/completions` under `/upstream/<id>/`, which is
what loads the model - exactly the "the next message loads it" behavior.

## Files

- serve/web/server.js - the `/running` gate, the hidden-tab stop/resume, the `loadHealth`/`loadMcp` refusal
- serve/web/sidebar.js - the status pill's "Unloaded"
- serve/web/monitor.js - the Monitor's unloaded line
- serve/web/state.js - the `S.conn` comment
- serve/test_dashboard.py - five new `PageLogic` tests (node-run, fake fetch)

Nothing in `engines/`, `saguaro.py`, `serve/server.py` or the Turbo files was touched; `serve/qfn_engine.py` was
not needed at all (no extract run).

## Tests run (only the two the plan names, one at a time)

- `python -m unittest serve.test_dashboard` - 63 tests, **OK** (21.4 s), node 20.19 (not skipped)
- `python -m unittest serve.test_new_engines_ui` - 35 tests, **OK** (4.2 s)

The five new tests (serve/test_dashboard.py, `PageLogic`):

| test | asserts |
| --- | --- |
| `test_behind_llama_swap_the_page_asks_running_not_the_model` | model absent from `/running`: the fetch log is exactly `["/running"]` (a later `loadHealth`/`loadMcp` add nothing), `conn == "unloaded"`, `metrics == null` |
| `test_running_says_ready_the_page_polls_and_coming_back_reloads_health` | off the card: `/running` only; swapped back in: `["/running", "/running", "health", "metrics"]`, `conn == "ok"`, metrics filled |
| `test_a_hidden_tab_asks_nothing_in_every_setup` | hidden (plain): no fetch at all, `loadHealth`/`loadMcp` included; visible: `["metrics"]`; hidden again behind llama-swap: nothing new |
| `test_not_behind_llama_swap_polls_exactly_as_before` | no supervisor, and an `/upstream/` path with no supervisor: `["metrics"]` each time, never `/running` |
| `test_a_running_that_does_not_answer_polls_as_today_while_visible` | `/running` 404 and 401: `["/running", "metrics"]` while visible, and nothing further once hidden |

`node --check` on the four changed web files: clean. No network was used.

## Open questions

1. **`/running` is polled once per poll cycle (every second) when the model is ready.** It is a llama-swap-local
   request that loads nothing, but it is one more request per second than before. It could be cached for a second
   or asked at a slower cadence while ready; left as-is (the plan says "before each poll").
2. **The state string is assumed to be exactly `"ready"`.** llama-swap also has states like `"loading"`; anything
   but `"ready"` closes the gate, which seemed the safe reading of the plan.
3. **`watchReload` / `watchUpdate` keep their own 2-3 s loops while running** (they poll `/api/context`, `/health`).
   They are transient user-initiated flows that llama-swap's server refuses anyway (`/api/port`, `/api/update`,
   context changes return "under llama-swap" errors), so they were left alone - out of the plan's "metrics, health,
   mcp" scope.
4. **The very first `/health` at page open still loads the model** under llama-swap (it is how the page learns the
   supervisor, and opening the page loads it anyway). Only the standing polls are gated.
