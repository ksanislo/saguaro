# REPORT P125n - an open Metrics/Monitor page must not wake an unloaded model (11-27)

Card test (mangonel): the live QFN model was unloaded through llama-swap (`POST /api/models/unload`);
an open Monitor/Metrics page made llama-swap load it again within a minute. Investigated both sides,
fixed ours, tested. One commit for the fix (`36a98262a`), this report committed after it.

## 1. What OUR page asks (serve/web, the page llama-swap serves at `/upstream/<id>/`)

Every fetch of a page that is open and visible (code lines as they are now, post-fix):

| path (relative to `/upstream/<id>/`) | who sends it | how often | guard |
| --- | --- | --- | --- |
| `metrics` | `poll()` `serve/web/server.js:189` | every 1 s (`:211`) | the /running gate |
| `/running` (llama-swap's own, at the origin root) | `swapState()` `serve/web/server.js:144` | every poll; 5 s alone while unloaded (`:182`) | - |
| `health` | `loadHealth()` `:18`, `startingHealth()` `:68`, `loadNow()` `:166`, poll's `back` `:186`, `watchUpdate` `:554` | on load, on retries (2 s), during watches | `unloaded()` (`loadNow` is the deliberate one) |
| `mcp` | `loadMcp()` `:284` | every 10 s via poll | `unloaded()` |
| `api/context`, `api/port`, `api/update`, `settings`, `api/thinking-budget`, `api/savings`, `api/turbo/options`, `api/tokens` | their loaders (`server.js`, `settings.js`, `chats.js`, `turbo.js`) | while their page/dialog is open | `unloaded()` |
| `api/turbo` | `loadTurbo()` `serve/web/turbo.js:22`, called from `app.js` boot | once per page load | **was unguarded - fixed** |
| `v1/strata/stream?id=…` (SSE take-up) | the cut-off answer's `follow()` `serve/web/chats.js:517` | every 2 s for up to 10 min (and again on tab reveal) | **was unguarded - fixed** |
| `v1/strata/cancel` | Stop click `serve/web/chats.js:627` | once | **was unguarded off-card - fixed** |
| `api/models` | `loadModels()` `serve/web/models.js:25` (Models page open) | on open | **was unguarded when the gate closed unnamed - fixed** |
| `api/report` | Monitor's "Copy report" (`server.js` `fetchReport`) | on click | left as is: a labeled deliberate click, like sending a chat |
| `v1/chat/completions` | sending a chat (`chats.js:542`) | on send | by design: "the next message loads it" |
| `${root}v1/models` | the Models page under a *named* supervisor (`models.js` `SupervisedModels`) | once | goes to llama-swap's origin root, not under `/upstream/` |

## 2. llama-swap v260: which paths load a model, which do not

llama-swap is **not vendored** (third_party holds only `beellama.cpp`, `exllamav3`, `ggml`, `wheels`),
and the plan forbids network fetches - so this rests on what this repo already recorded against the
live stack, plus the card test's own use of the API:

* **Loads** - every request under `/upstream/<model>/...`, any path: `serve/web/server.js:119-121`
  ("EVERY request under that path LOADS the model - and evicts another one from the card") and
  `serve/web/models.js` "its model list - each model's own page loads it (llama-swap swaps it in)".
  That is what the wake-up is: the page's ordinary per-second `GET /upstream/<id>/metrics`.
  `/v1/chat/completions` etc. under `/upstream/` - or at the root with a `"model"` id - load by the
  same rule; agents picking a model by id swap it in (docs/SETTINGS.md's alias note, v260).
* **Does not load** - paths llama-swap answers itself at its origin root: `/running`
  (`{"running": [{model, state}]}`, what our gate already uses and P2 validated on the live box),
  `/api/models` and the management API (`/api/models/load`, `/api/models/unload` - the card test
  unloads through this and nothing loads back), and `/v1/models` (our Models page lists through it
  without swapping: "its own /v1/models two levels up").

## 3. The exact request that woke the model, and why the old gate let it through

The waker: **`GET /upstream/<model>/metrics`, once a second, sent by `poll()`** - the same symptom
the code already described ("one open Monitor tab's per-second /metrics keeps reloading it"), so the
gate was not closing on mangonel. Why: the old `poll()` only asked `/running` when
`S.health.supervisor === undefined || "llama-swap"`. `/health` sends that name from
`supervisor_name()` (`serve/server.py:9511`), which reads **only the immediate parent**
(`/proc/<ppid>/comm`) - a llama-swap `cmd:` that starts us through a wrapper script (`saguaro.sh`,
a shell chain) leaves the parent's name as the script, `/health` then carries `"supervisor": null`,
and `null` matches neither `undefined` nor `"llama-swap"`: the page decided "no gate here" and polled
`/metrics` under `/upstream/` every second - llama-swap loaded the model on the first one.
`watchStarting` already gated on the URL path alone, so the boot could even show the quiet unloaded
state and `poll()` then woke the model right back up.

The audit found three more asks of the same family (a page opened or restored off the card, or a
half-finished answer, could load the model even with the gate working) - all fixed below.
The one that cannot be avoided from the page: a proxy under `/upstream/` that does not answer
`/running` at all (pre-`/running` llama-swap) or refuses it - no path there can tell "off the card"
without loading, so the page keeps polling as before and says so in REPORT only. llama-swap v260
does answer `/running` - no config option is needed; the operational belt for wrapper-script starts
is `SAGUARO_SUPERVISOR: llama-swap` in the model's `env` (now documented, `docs/SETTINGS.md`), which
also restores the page's supervisor mode and the 409 refusals that the missed name disables.

## 4. What changed

Commit `36a98262a` (files; `serve/qfn_engine.py` untouched - nothing in engines/qfn was edited):

* `serve/web/server.js` - the gate no longer trusts `/health`'s supervisor name: under `/upstream/`,
  llama-swap's own `{"running": [...]}` answer decides (`swapState` treats a 200 of any other shape
  as "no gate here", 404/401/403 like before, a blip still keeps a closed gate). New
  `loadNow()`/`waking()`: the unloaded Monitor's Load button fires exactly one held `GET health` -
  the deliberate request - and nothing else; `poll()`'s and the reveal handler's conditions match.
* `serve/web/monitor.js` - the unloaded card: the sentence plus a `Load it now` button; while that
  request is in flight, "Loading the model - a minute or two…".
* `serve/web/turbo.js` + `serve/web/app.js` - `loadTurbo()` refuses while unloaded; when the boot
  happened off the card, app.js re-arms it through `whenHealth` with the first health after the gate
  opens.
* `serve/web/chats.js` - the cut-off answer's reconnect loop waits the closed gate out (5 s looks,
  no upstream request) instead of retrying `v1/strata/stream` every 2 s; `Stop` skips
  `v1/strata/cancel` off-card (nothing runs there to cancel).
* `serve/web/models.js` - `loadModels()` refuses while the gate is closed (a supervisor named by
  `/health` already gets llama-swap's own list instead) and says why.
* `docs/SETTINGS.md` - the `SAGUARO_SUPERVISOR` row (it existed in code only).

## 5. Tests (only the modules the plan names, one at a time)

Fake upstream (llama-swap) = the existing node page-logic harness in `serve/test_dashboard.py`
(`/upstream/` pathname + a `fetch` stub logging every asked URL):

* `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_dashboard` - **90 tests, OK** (was 85).
  New: `test_the_gate_needs_no_supervisor_name_only_lama_swap_running` (P125n regression: health says
  null supervisor, `/running` says another model has the card -> only `/running` asked, state
  unloaded); `test_the_load_button_is_the_one_deliberate_request_while_unloaded` (one `health`, a
  second click asks nothing, waking clears on ready); `test_the_unloaded_monitor_card_offers_the_load_button`;
  `test_booting_off_card_asks_no_turbo_and_asks_it_once_the_gate_opens`;
  `test_a_resumed_answer_waits_out_the_closed_gate` (stream asked again only after the gate opens).
  Updated `test_not_behind_llama_swap_polls_exactly_as_before`: under `/upstream/` the page probes
  `/running` once and, seeing a shape that is not llama-swap's, polls `metrics` as before.
* `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_starting` - **24 tests, OK** (the
  starting gate uses the same `swapState`; unchanged behavior on its shapes).
* Baselines before the change: both modules OK (85 / 24). All changed JS parses
  (`node --check`). `serve/qfn_engine.py` not touched, so no extract needed.

## 6. Open questions

1. **Which trigger fired on mangonel** needs its `/health` body (`supervisor: null`?) and the
   process tree (`ps -o comm= -p <pid>` up to PID 1) to confirm - the wrapper-script gate bypass is
   the only one that explains "per second, within a minute", but a card rerun on mangonel should
   verify the fix as shipped (unload via llama-swap, leave the Monitor open for ten minutes:
   llama-swap's log must show no load; the `Load it now` click must show exactly one).
2. `supervisor_name()` could walk a few ancestors instead of only the parent, so wrapper-script
   starts name the supervisor themselves (Models page mode + 409 refusals); left out here on purpose
   (server-side change, outside this plan's named tests) - until then the yaml env
   `SAGUARO_SUPERVISOR: llama-swap` covers it.
3. llama-swap v260's exact `/running` and `/v1/models` handlers were not re-read (not vendored, no
   network); the shape-check makes the gate fail *open to old behavior* - never onto a load - if a
   future llama-swap changes the answer's shape.
