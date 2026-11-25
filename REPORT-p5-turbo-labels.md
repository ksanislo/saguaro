# P5 report: clearer Turbo labels and a warning on the RAM skip

Plan: `PLAN-p5-turbo-labels.md`. Two commits on `saguaro/q1-p5-turbo-labels`:

- `03378f01a` step 1 - the plain words and the RAM warning
- `65804fd25` step 2 - the Skip control hides where the engine leaves skip off

## Step 1 - plain option words + the RAM warning

The words now live in one place, `serve/web/turbo-core.js`, and both pages read them:

- `TURBO_DROP_HINT` = "fewer of the 10 routed experts a token, the weakest dropped" - the Settings row's sub,
  each drop option's tooltip, and each Drop experts menu item's hint.
- `TURBO_SKIPS` hints: SSD = "an expert that would be read from the SSD is left out"; RAM = "an expert that is
  not on the card is left out"; the RAM entry also carries `warn` = "can break answers: only where almost
  everything is on the card". `turboSkipWarn(id)` returns it ("" for off and SSD).

`serve/web/turbo.js` (Settings > Turbo, live): the rows are "Drop experts" off / 1 / 2 / 3 and "Skip slow
experts" off / SSD / RAM (was "Skip experts", Seg label "Skip tier"). While RAM is chosen the warning shows
under the row in `callout callout--warn` - a class the page already has for warnings (settings.js' context
warnings use it). RAM stays selectable ("Do not disable it").

`serve/web/chat-view.js` (the chat's Turbo menu): the section is "Skip slow experts" (was "Skip"); each RAM
item's hint carries the warning in a `warn-text` span (components.css).

## Step 2 - the Skip control hides where skip is not applied

No capability/engine field told the page about skip (the `supported` field only says whether live Turbo
exists at all, for Inkling 2 / Hy4-Flash), so per the plan the server added one **in /api/turbo/options
only**: the GET body now carries `"turbo_skip": true|false`. `Service.turbo_skip_supported()` (serve/server.py,
beside `set_turbo_default`): true by default; false only for `engine_kind == "qfn"` while the engine has not
reported skip support itself - the convention taken is its `INFO turbo_skip=1`, which the QFN driver already
folds into `engine.info` from any INFO line. Today's QFN engine applies drop only and says so on its stderr
("this engine applies only drop; skip is off"), so it reports false; an engine that gains skip gains the line.
The POST body still says only `{"turbo": ...}` (its tests pin it), and no request's wire form or defaults
changed.

The page: `loadTurboLive` keeps the answer in `S.turboSkip` (missing field = true, "default true"); with it
false the Settings Skip row and the chat menu's Skip items disappear - Drop experts, the toggle and the
warning all stay. An older server (no /api/turbo/options) still hides the whole live-Turbo block as before.
`TurboMenu` in chat-view.js is now exported (it was module-private) so the page test can walk the rendered menu.

## Files

- serve/web/turbo-core.js - TURBO_DROP_HINT, the plain TURBO_SKIPS hints, RAM's `warn`, `turboSkipWarn`
- serve/web/turbo.js - the Settings rows' words, the callout--warn under the Skip row, `S.turboSkip`
- serve/web/chat-view.js - the menu's words, the warn-text in the RAM item hint, the hidden Skip section,
  `TurboMenu` exported
- serve/server.py - `Service.turbo_skip_supported()` + the `turbo_skip` field in the GET /api/turbo/options body
- serve/test_turbo_live.py - the words and the warn-only-for-RAM check (node, turbo-core.js); GET
  /api/turbo/options reports turbo_skip (true on the mock engine, false for QFN, true once
  `engine.info["turbo_skip"] = 1`), and the POST body still says only `{"turbo": ...}`
- serve/test_dashboard.py - PageLogic walks the rendered trees (vnode strings): the rows and the RAM warning
  (callout--warn only with RAM), and with `S.turboSkip = false` the Skip row and the menu's skip items are
  gone while Drop experts stay

Untouched, per the plan's Don't: what any request sends and the defaults, the top Turbo button, engines/,
saguaro.py, serve/qfn_engine.py. No network, no stash.

## Tests (each module run on its own, /opt/llm/saguaro/.venv/bin/python)

| module | result |
| --- | --- |
| serve.test_turbo_live | 13 tests, OK |
| serve.test_dashboard | 64 tests, OK |
| serve.test_new_engines_ui | 35 tests, OK |
| serve.test_model_api | 28 tests, OK |
| serve.test_server | 175 tests, OK |

## Open questions

- The sibling branch `saguaro/q1-s1-qfn-turbo-skip` teaches the QFN engine to apply skip but reports nothing
  on the INFO line, so on its own it would keep the page's Skip control hidden. When it lands it (or the
  extract of serve/qfn_engine.py from engines/qfn/serve/server.py) should emit `INFO ... turbo_skip=1` - or
  that merge flips `turbo_skip_supported()` for qfn itself. The convention here is deliberately the engine
  saying so, per the plan's "a QFN whose engine does not report it".
- The QFN line/solo driver path (the non-batch one in qfn_engine.py) has the same drop-only engine, and
  `engine_kind == "qfn"` covers both drivers, so both hide the Skip control today.
- A model whose saved sidecar already holds a skip (set when it ran a skip-applying engine) with `turbo_skip`
  false: the row disappears and the hidden skip still rides the model's live setting until some request or the
  sidecar changes it. Nothing is disabled per the plan, so the server keeps applying what was saved; the wire
  key stays sent as asked (the engine says it leaves it off).
