# REPORT-n1-model-names

Consistent model names on every page, and labels that follow a model switch (ref/items.txt items 9 and 10).
One commit per plan step: 8b0a9d401 (step 1), 1bff6b8d5 (step 2), ff0b48d79 (step 3), 81d883f85 (step 4).

## What changed

### Step 1 - the server says the name in two parts (`model_name_pretty` + `model_quant`)
- `tools/maya_quants.py`: new `pretty_name()` (a family's SHOWN name with dashes as spaces: "Qwen3.8 Flash Next"),
  `quant_name()` (the files' name exactly as named: "Q8_0", "UD-Q4_K_XL", "Maya-L", "CYBER-FROST-3.8 Q5_K_M",
  never prettified), a `QUANT_SUFFIX` regex (UD-*/IQ*/Qn_K_M/Qn_0/Maya-*/SQ-n), and `display_names(cfg)`:
  the setup's names from the installer record (incl. an Inkling `ik2_file`'s own name), or - for a hand-made
  config with neither - the `model_name` with "-" as spaces minus a recognised quant suffix, that suffix or ""
  as the quant; nothing to say: `(None, None)`.
- `serve/server.py`: Service carries `model_name_pretty`/`model_quant` (set at start from the config via
  `display_names`); `/health` and `/metrics`' `engine` carry both fields (null when unsaid). Every existing
  field stays; `model_label` is untouched. The page never splits the model id itself - the split happens here,
  where the installer record is known.
- `serve/test_server.py`: new `test_the_model_is_named_in_two_parts` (QFN Q8_0/UD-Q4_K_XL, CYBER-FROST's
  Q5_K_M quant, GLM Maya-L, two hand-made configs, and the two endpoints).

### Step 2 - one label everywhere (`modelLabel`)
- `serve/web/lib.js`: `modelLabel(h, {compact})` -> `"Qwen3.8 Flash Next (Q8_0)"` compact, or `{name, quant}`
  for two lines. Prefers the server's two fields; a page row (Models list, Saved rows) uses its own `model` +
  `quant`; an old server without the fields keeps today's text (`model_label`, else `modelName(model)`).
  The quant is never prettified ("Q8_0", not "Q8 0").
- Used everywhere a model is named: `sidebar.js` (status card - the wrapping "· Q8_0 (saguaro-qfn)" line is gone,
  the id-tooltip dropped; the Saved popover's prices, "This start" and per-model rows), `chat-view.js` (chat
  header button, model menu, new-chat model card), `settings.js` (Settings > Model title, Chat-defaults intro,
  the Savings price cards), `models.js` (card title and aria-labels; quant now on a second line with the about
  line; deleted-model toast), `server.js` (the switch toast).
- No engine name or version beside a model: Settings > Model's id/engine/`Saguaro v...` line is replaced by
  `API id: <id>` with a Copy button - the only place the API id appears; the Models card's "API model name"
  detail row and the switch dialog's id mention are gone (the dialog now points at Settings > Model).
- `serve/test_server.py`: the sidebar-label assertion updated to the new form.

### Step 3 - the labels follow a switch, no reload
- `serve/web/server.js`: `identityChanged(health, engine)` (exported for the tests), and `poll()` re-reads
  `/health` when the connection comes back (was "down"/"key") or when `/metrics`' engine names another
  `model`/`model_label` than the page shows. `loadHealth()` updates `S.health`, so every label re-renders; the
  chat list and the open chat are untouched; the per-model sampling reset and the "its own settings" toast
  happen as they did at a page load.
- `serve/web/state.js`: `modelId()` also counts the two new fields.

### Step 4 - tests
- `serve/test_dashboard.py` (`PageLogic`): `test_one_model_label_names_the_model_and_its_files` - Q8_0 is not
  "Q8 0"; UD-Q4_K_XL; Maya-L; CYBER-FROST-3.8 Q5_K_M; a hand-made config (name only); an old server (label or
  id words); a row's {name, quant}.
- `serve/test_new_engines_ui.py` (`Page`): `test_an_open_page_follows_a_model_switch` - identityChanged (same
  config false, another model / another quant-config true, a silent engine never; old servers), the re-read
  label changing "Qwen3.8 Flash Next (Q8_0)" -> "GLM 5.3 (Maya-L)", the switch detected by `newModelCheck()`,
  and the chat list / open chat staying.

## Files changed
tools/maya_quants.py; serve/server.py; serve/test_server.py; serve/web/lib.js; serve/web/sidebar.js;
serve/web/chat-view.js; serve/web/settings.js; serve/web/models.js; serve/web/server.js; serve/web/state.js;
serve/test_dashboard.py; serve/test_new_engines_ui.py.

## Tests run (one module at a time, /opt/llm/saguaro/.venv/bin/python -m unittest)
- serve.test_dashboard: 59 tests, OK
- serve.test_new_engines_ui: 36 tests, OK
- serve.test_models: 62 tests, OK
- serve.test_server: 172 tests, OK

`tools/saguaro/extract_qfn_engine.py` was run as a check: `serve/qfn_engine.py` came out byte-identical
(no engine-driver change was needed). `engines/`, `saguaro.py`, the Monitor's live numbers and turbo* were
not touched; no network used.

## Open questions
- The hand-made fallback is literal, as the plan words it: model_name "minimax-m3.1-flash" -> pretty
  "minimax m3.1 flash" (lowercase kept), where the old page text (`modelName`) would have said
  "MiniMax M3.1 Flash". Real live hand-made configs carry proper names ("Qwen3.8-Flash-Next-UD-Q4_K_XL"),
  but say the word if the fallback should capitalise like the page did.
- `/api/models` and `/api/savings` rows still carry only `model` + `quant`, so Models-list and Saved rows name
  via `modelName(model)` ("GLM 5.3 Flash") where the header says the server's pretty ("GLM 5.3"). The plan
  required the new fields only on /health and /metrics; sending them per row too would make every line match.
- The chat header menu still shows Saguaro's own version ("· v1.2.1") - read as Saguaro's, not an engine
  name/version; only those were removed.
- The standalone server in engines/qfn/serve/server.py (no longer the path any start takes) carries neither
  `model_label` nor the new fields; left as it was. The Copy report's "Model: <id> (<label>)" line is server
  text for issues, not a page, and keeps the id.
