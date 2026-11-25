# FIX-REPORT - N1 model names (REVIEW-RESULT.md, manager's FIX-FIRST list)

Four commits, in the manager's order; one per list item (the cheap nits 2, 5, 7 went in one, as they are one list
item). Worked alone, read-and-edit only, in this worktree; only the plan's modules were run, one at a time
(`/opt/llm/saguaro/.venv/bin/python -m unittest ...`). `serve/qfn_engine.py` was not touched (its source
`engines/qfn/serve/server.py` was neither).

## Finding 1 (MUST) - the first page load after the update faked a model switch - FIXED (b3a0fb143)

- `serve/web/state.js` `newModelCheck()`: after storing the new four-part id (which seeds the new format once), a
  stored value equal to the old two-part `"${model}|${model_label}"` of the *current* health is not a switch - it
  returns false before touching anything, so every `maya.sampling*` key keeps its CHOSEN entries and `loadHealth`
  fires no "its own settings" toast.
- Test: `serve.test_new_engines_ui.test_the_first_load_after_the_update_is_no_switch` - an old-format stored value
  plus stored `sampling_qfn` choices: no switch, choices survive, the stored value comes back four-part, the next
  load stays no-switch, and a real switch (another model's stored value) still switches and still clears the
  choices.

## Finding 8 (MUST) - /api/models and /api/savings said nothing the header says - FIXED (3b72988c8)

- `serve/models.py` `describe()`: every row carries `model_name_pretty` / `model_quant` from
  `maya_quants.display_names(cfg)` - exactly what `/health` carries for a running config.
- `serve/savings.py` `Ledger`: a new `names` argument (the two parts, from the server) is recorded into the
  running model's entry (`name_pretty` / `name_quant`) at start and at reset; `report()` passes them through on the
  per-model rows and "this start", and `name_fields()` gives them to the `prices` rows and `price_list` from
  whatever the ledger recorded for that model id. `serve/server.py` passes
  `names=(svc.model_name_pretty, svc.model_quant)` when it builds the Ledger.
- Nothing was re-keyed: ledger rows stay keyed on the config file name and prices on the model id, so every
  existing Saved ledger entry and price entry still matches; an entry written before the fields simply carries no
  `model_name_pretty` and the page keeps today's text (old server alike - `modelLabel` falls back as before).
- The page needed no change: the Models cards (`models.js` `titleOf`/`modelLabel(m)`), the Saved popover/sheet rows
  and price lines (`sidebar.js` `SavedBody`) and the Settings > Savings price cards (`settings.js` `PriceCard`)
  already go through `modelLabel()`, which now reads the fields from these endpoints and matches the header.
- Tests: `serve.test_models.test_rows_carry_the_display_names` (every /api/models row equals its config's
  `display_names`; the setup's GLM row says "GLM 5.3"/"Maya-M"; a hand-made config splits its suffix) and
  `serve.test_server.test_savings_rows_carry_the_display_names` (rows, start, prices and price_list carry the two
  parts; a pre-fields ledger entry carries none; keys stay the file names).

## Finding 4 (MUST) - QUANT_SUFFIX missed llama.cpp's K-family names - FIXED (1f0149897)

- `tools/maya_quants.py`: the three `Q[0-9]_...` alternatives became `Q[0-9]_[0-9A-Z]+(?:_[0-9A-Z]+)*`, so
  `Q4_K_XL`, `Q2_K_XS`, `Q5_K_M`, `Q4_K_S` (and `Q4_0`, `Q5_K_M` as before) split; the `UD-` re-publishes
  (`UD-Q4_K_XL`) and the IQ names (`IQ3_S`, `IQ2_XXS`) keep splitting as they did.
- Tests (`serve.test_server.test_the_model_is_named_in_two_parts`, new subTests):
  `Qwen3.8-Flash-Next-UD-Q4_K_XL` -> ("Qwen3.8 Flash Next", "UD-Q4_K_XL"); `...-Q4_K_XL` -> (..., "Q4_K_XL");
  `...-Q2_K_XS` -> (..., "Q2_K_XS"); `...-IQ3_S` -> (..., "IQ3_S"); `GLM-5.3-Flash-Maya-L-IQ3_S` ->
  ("GLM 5.3 Flash Maya L", "IQ3_S") - the Maya branch's alphabet excludes `_`, so the IQ suffix is the split,
  as it already was before this change (pinned now); `gpt-oss-20b` -> ("gpt oss 20b", ""), no quant.
  Installer-record configs are untouched (they take the exact names, as the review noted).

## Finding 2 - identityChanged blind to a quant-only name change - FIXED (8f1e6ad27)

- `serve/web/server.js` `identityChanged()`: also compares `model_name_pretty` / `model_quant`, in symmetry with
  `modelId()`. A field the metrics dict does not carry at all is treated as "unsaid" (an old server's /metrics
  cannot fake a change, and the page cannot fall into re-reading /health every second).
- Test: the switch test (`serve.test_new_engines_ui.test_an_open_page_follows_a_model_switch`) gained a
  `quantOnly` case (same id and label, another quant: a change), and the old "says nothing" / "same model" cases
  still pass unchanged.

## Finding 5 - duplicate /health on every page load - FIXED (8f1e6ad27)

- `serve/web/server.js` `poll()`: the success path re-reads health only when coming back from `"down"` or `"key"`
  (or `identityChanged`), matching the comment; the first `"connecting"` -> `"ok"` poll no longer duplicates the
  `loadHealth()` app.js already makes at startup (checked: app.js:115).
- No test: the request-count effect only shows on a live page; the surrounding switch/reconnect paths stay
  covered by the page-logic test above.

## Finding 7 - "API id:" rendered empty when health is missing - FIXED (8f1e6ad27)

- `serve/web/settings.js`: the "API id" line, its value and its Copy button render only when `h.model` is said,
  as the `[...].filter(Boolean)` line did before the label change. No test (a live-server edge; one conditional).

## Findings left, as the manager said

- Finding 3: left functional (Settings > Connect your tools must show the id apps use).
- Finding 6: left (only the quant wrapping changes, and the manager said leave it).
- Finding 9: left (the /health re-indent stays; reverting it is churn).

## Runs (each module alone, /opt/llm/saguaro/.venv/bin/python -m unittest)

- serve.test_server: 173 OK
- serve.test_models: 63 OK
- serve.test_dashboard: 59 OK
- serve.test_new_engines_ui: 37 OK
- `node --check` on the three changed page files: clean.
