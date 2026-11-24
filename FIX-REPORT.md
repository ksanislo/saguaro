# FIX-REPORT - the REVIEW-RESULT findings, one by one (T8a, post-review)

Against REVIEW-RESULT.md (verdict FIX-FIRST); baseline a4db87c1a, four fix commits: 47fcaf42d, 947ec810e, 50161de50,
e8dafa9e8. Nothing touched `serve/qfn_engine.py` or `engines/` (no extraction needed); the generated file and the C++
are byte-identical to the baseline.

## 1. HIGH - `QFN_KV` referenced but never defined (FIXED - 47fcaf42d)

Changed: `serve/web/models.js` - re-added the deleted `QWEN_KV` list verbatim as `const QFN_KV = [{id: "fp16", ...},
{id: "int8", ...}, {id: "k8v4", ...}, {id: "q4_0", ...}];` with its KV-bytes-from-INFO comment, right above
`qfnFields` (the review's suggested spot, next to the old line 511 comment). The two `QFN_KV` uses (516, 558) now
resolve and every `family === "qfn"` settings card renders again.

Also in the same commit, so the slip cannot repeat: `fieldsOf` is exported (the only new export, same as `hy4TpOptions`
is exported for the tests) and `serve/test_new_engines_ui.py` gained `test_the_qfn_settings_card_renders` - it runs
`fieldsOf` -> `qfnFields` through the node harness and checks the three card groups. Verified red/green: against a copy
of `serve/web` with the new declaration line deleted it fails `ReferenceError: QFN_KV is not defined`; against the
tree it passes - exactly the hole the review named (`node --check` being a syntax-only check, and no test reaching
`qfnFields`).

## 2. LOW - `--calibrate` / `--speed-calibration` before the CYBER-FROST migration (FIXED - 50161de50)

Changed: `saguaro.py`, in the existing-config branch right after the Metal short-circuit - when `--calibrate` or
`--speed-calibration` is asked and `migrate_cyber_frost` recognises the picked config (called on a fresh `read_json`
that is thrown away, so nothing is rewritten early), it says the reason instead of measuring nothing: the config is
not migrated to `saguaro-qfn` yet, run `--setup` once first - it rewrites the config and builds the pack and drafter;
the tuning waits. The tuning flags are then cleared like the Metal block does, so the normal start follows and the
durable `cyber_pack_pending` refusal (saguaro.py:4730) stays the guard behind the note. `tools/test_qfn_calibrate.py`
is not plan-named and was not run; its routing pins live in `calibrate_config`'s dispatch, which this does not touch.

## 3. NIT - `tools/maya_quants.py` misindentation (FIXED - 947ec810e)

Changed: the `OTHERS` continuation lines of `("minimax", "exl3")` and `("ik2", "ud-q4_k_xl")` back to the 10 spaces
their neighbours use (they sat at 11 after the `("qwen", "q5_k_m")` pair's deletion). Cosmetic; `tools.test_maya_quants`
still 24 OK (skipped=1).

## 4. NIT - check_parallel's fallback prints the raw family name (NOT CHANGED - deliberate)

The review itself scopes it to "family names that no longer exist": every production call passes `config_family(cfg)`
(saguaro.py:4716, 8905) or `choose_family` (saguaro.py:5468-5469), and neither can return `"qwen"` any more -
`family_of` maps every old config to `"qfn"` (tools/maya_quants.py:278) and `choose_family`'s table has no qwen row.
So the fallback is only reachable with a family that does not exist at all, where printing the unknown name verbatim
is the honest default; making it print `Qwen3.8-Flash-Next` would mean re-adding the literal `"qwen"` key T8a exists
to delete, for an input nothing can produce. `tools/test_setup_qfn.py`'s `Parallel` case still asserts the refusal and
passes ("Qwen3.8-Flash-Next" in the refusal is the supported-engines list, which stays). Chose no change over dead code.

## 5. NOTE - one-time reset of stored Chat settings (FIXED - e8dafa9e8)

Changed: the T8a CHANGELOG entry's "Server and pages" bullet now says a Qwen/QFN chat's saved sampling and thinking
choices start from the defaults once after the update and are remembered again from then on - what the
`sampling_qwen` -> `sampling_qfn` prefsKey move (serve/web/state.js:223, no value copy) does. Docs only; the re-key
itself is unchanged, as the plan asked ("re-key it to 'qfn' only").

## Tests after the fixes (plan-named modules, one at a time, repo root)

tools.test_setup_qfn 40 OK · tools.test_maya_quants 24 OK (skipped=1 - node-gated) · tools.test_qfn_fit 20 OK ·
tools.test_setup_amd 29 OK · tools.test_setup_download 2 OK · tools.test_bench 33 OK · tools.test_calibrate 10 OK ·
serve.test_models 60 OK · serve.test_dashboard 58 OK · serve.test_new_engines_ui **35** OK (the table's 34 +
finding 1's card test) · serve.test_model_api 24 OK. `node --check serve/web/models.js` clean;
`python -m py_compile saguaro.py` clean.
