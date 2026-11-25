# P7 report: the rest of "an open page must not load a model under llama-swap"

Plan: `PLAN-p7-unloaded-rest.md` (the P2 review round-2 follow-ups: Settings' six upstream requests, L3, and the
stale meter figure, L4). Two commits on `saguaro/q1-p7-unloaded-rest`:

- `5070772be` step 1 - Settings over a closed gate asks nothing; its rows say so; they load when the gate opens
- `5575a3fee` step 2 - the context meter re-estimates when the chat or its messages change while unloaded

## Step 1 - opening Settings while unloaded asks only /running

Behind llama-swap every request under `/upstream/<id>/` loads the model, and `SettingsDialog`'s open effect fired
six of them - `settings`, `mcp`, `api/context`, `api/port`, `api/thinking-budget`, `api/update` - on every open
(L3). Only `/mcp` refused while the gate was closed. Now:

- **serve/web/server.js** - the five unguarded row loaders each start with an `unloaded()` return, like `loadMcp`
  already did: `loadShared` (nothing), `loadCtxInfo` (`S.ctxInfo`), `loadPortInfo` (`S.portInfo`),
  `loadThinkBudget` (`S.thinkBudget`), `loadUpdate` (`S.update`). With the gate closed the page asks only
  llama-swap's own `/running`, as the Monitor already did.
- **serve/web/server.js** - `whenHealth` is a list now: each subscriber gets called at the end of a `loadHealth`
  that brought the model's facts, and its return removes it. app.js's meter hook is unchanged; a second,
  temporary subscriber is what carries step 1's "load them when it is ready".
- **serve/web/settings.js** - `loadSettingsRows()` (exported: the dialog's six asks in one guarded call) is what
  the open effect runs; when `unloaded()` at open the effect also does `whenHealth(loadSettingsRows)` for the
  dialog's lifetime (unsubscribed on close), so a dialog left open loads its rows the moment `poll()` sees the
  model back on the card (that path's `loadHealth` fires the hooks).
- The rows that lost their server figures show **"unloaded - the next message loads it"** (a `UNLOADED` const):
  Tools instead of "No MCP servers are set up..." (which would be a lie about the run config), and Settings >
  Model's fallback note instead of "This server does not say its context range". The stored (local) chat
  settings - sliders, thinking, presets, instructions - show all along; the Model section's supervisor note and
  About's hidden update card are as they were. `Tools` is exported for the page test, as `TurboMenu` was in P5.

## Step 2 - the meter while unloaded re-estimates on chat / message change

With the gate closed `updateCtxMeter` re-estimated only when `S.ctx.tokens == null`, so a chat opened after one
the server had *counted* could go on showing that exact count as "This chat" (L4; P2's FIX-REPORT had presented
it as intended). The off-card branch now keys the estimate:

```js
const off = "off-card " + JSON.stringify(apiMessages(S.current, msgs, true));
if (S.ctx.tokens == null || ctxKey !== off) { ctxKey = off; ...estimateTokens(msgs), approx: true... }
```

- switching to another chat re-estimates that chat (whatever switched it - openChat resets `S.ctx`, and a chat
  set without a reset, as at restore, fails the key compare too);
- changing this chat's messages - typing, an edit, an attachment, its instructions - re-estimates;
- a chat counted before the swap keeps its exact figure while nothing about it changes (no churn, no re-ask);
- the key carries the `"off-card "` mark, so it can never satisfy the exact-count dedup: once the gate opens,
  the real `api/tokens` count is asked once more for the chat on screen. Nothing is ever asked while unloaded.

## Files

- serve/web/server.js - `whenHealth` subscribers (with removal) called from `loadHealth`; `unloaded()` guards in
  `loadShared` / `loadCtxInfo` / `loadPortInfo` / `loadThinkBudget` / `loadUpdate`
- serve/web/settings.js - `UNLOADED` note rows (Tools, Model's fallback), exported `loadSettingsRows()` +
  exported `Tools()`, the dialog's effect (guarded load + `whenHealth` while unloaded)
- serve/web/chats.js - the `unloaded()` branch of `updateCtxMeter` keyed on ctxKey ("off-card " mark)
- serve/test_dashboard.py - two PageLogic tests (see below)

Untouched, per the plan's Don't: the polling (`poll`, `swapRunning`, the gate itself), the server
(serve/server.py), engines/, saguaro.py. Nothing under engines/qfn/ changed, so `tools/saguaro/extract_qfn_engine.py`
was not run and serve/qfn_engine.py is unchanged. No stash, no cd.

## Tests (each module run on its own, /opt/llm/saguaro/.venv/bin/python)

| module | result |
| --- | --- |
| serve.test_dashboard | 77 tests, OK (~33 s, node v20.19.2 - PageLogic not skipped; baseline before the work: 75 OK) |
| serve.test_new_engines_ui | 35 tests, OK |

The two new cases drive the real modules with a fake fetch:

- `test_opening_settings_while_unloaded_asks_only_running` - renders the real `SettingsDialog` through preact
  (the PAGE_STUBS plus a `createElementNS`/`createTextNode` the render pass needs), with the gate closed: the
  fetch log is exactly `["/running"]`, and Tools' tree carries "unloaded - the next message loads it". When
  `/running` says ready the dialog's parked rows load: all six asked, once each, and every one lands in `S`.
- `test_the_meter_re_estimates_when_the_chat_changes_while_unloaded` - chat a counted exact (1234, `approx:
  false`); gate closes; opening chat b shows a different, approximate figure and asked nothing; typing in b
  re-estimates again and still asks nothing; back on the card the exact count is asked again (`api/tokens`
  asked twice in total, never while off-card).

## Open questions

- **Settings > Savings and the Chat-defaults Turbo-live row** ask `api/savings` and `api/turbo/options` from
  their section's own effects. The plan names exactly the six dialog-effect asks, so those two are untouched:
  while unloaded they only fire if the user walks to that section, and they land the model like the six did.
  Same guard fits if the next round wants them too.
- **A gate that closes while the dialog stands open**: the six rows keep their last loaded values (they were
  true a second ago; llama-swap owns context/port anyway, and Model says so in its supervisor note). Nothing
  re-clears them; the note rows appear only where there was never a value.
- **The gate with no supervisor told us** (an `/upstream/` path whose `/health` has not answered - the reveal
  case P2 tested): `unloaded()` is true there too, and that is exactly where Model's new fallback note shows
  instead of "does not say its context range".
- `whenHealth` now accumulates: app.js registers the meter once at boot; the dialog adds and removes its own.
  No third caller exists; a fourth that forgets the returned removal would simply load its rows on every
  health.
