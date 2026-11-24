# T7m report: merging T7 (CYBER-FROST on QFN) into the QFN-first integration branch

Branch `saguaro/q1-t7m-merge`, one merge commit `49b3de52a` (message "Merge saguaro/q1-t7-cyberfrost
(QFN-first t7: CYBER-FROST on the QFN engine)") on top of `e1f43e79a`; this report is its own commit.
`git merge --no-ff saguaro/q1-t7-cyberfrost` stopped, exactly as the plan said, with conflicts in
`saguaro.py`, `tools/test_setup_qfn.py` and an add/add in `FIX-REPORT.md` (removed with `git rm -f
FIX-REPORT.md` - both sides had written a review-response file with that name; each survives in its own
branch's history). Every resolution keeps HEAD's features (T2 NVIDIA/qfn_fit layer split, T3 pictures +
drafting choice + parallel, steps 1-9) with T7's CYBER-FROST behaviour on top.

## Each conflict and how it was resolved

### FIX-REPORT.md (add/add)
`git rm -f FIX-REPORT.md` per the plan. HEAD's was the T2 review's response, T7's its own review's; both
remain reachable on their branches.

### saguaro.py - qfn_about
Kept HEAD's text (NVIDIA one card or the up-to-four-card split, AMD measured, pictures with the vision
tower) and appended T7's sentence: "... the model as Qwen released it (CYBER-FROST-3.8-PS is offered as a
quant of its own, in the quant list)." T7's older AMD-only / "text only" wording is gone (HEAD's).

### saguaro.py - the vision/drafter block
Kept HEAD's `qfn_vision_model`, `compile_vision_qfn` and `qfn_vision_step` (the real step: downloads the
QFN manifest's mmproj, compiles vision-qfn, writes the elastic entry) and added T7's `qfn_drafter_step`
(CYBER-FROST's NextN drafter into `<pack>/mtp`, kept while `QFN_DRAFTER_TOOLS` and the file stand).
Dropped T7's `qfn_vexe` and T7's `qfn_vision_step` (the download-only variant that pointed at a
`vision-qfn` beside the engine because its tree could not compile it) - HEAD compiles the reader, so
CYBER-FROST's pictures run through HEAD's `qfn_vision_step`, per the plan.

### saguaro.py - write_config_qfn
Signature/body: HEAD's (the `vision` parameter HEAD already had, `QF.plan` layer split, T4 env,
`layer_split` config entry, the parallel block) - auto-merged, with two fixes:
- docstring conflict: merged text - HEAD's NVIDIA/T4/split/pictures sentence plus T7's
  "maya-cyber-frost[-hip].json for CYBER-FROST, whose config keeps the file name it always had";
  kept `step(9, ...)`.
- the auto-merge had produced a duplicated `--vision` arg (T7 appended it after `--mtp`, HEAD after the
  RoPE args): removed T7's copy, kept HEAD's placement. The T7 config test's
  `assertIn("--vision", ...)` holds either way.
The file naming itself is `Q.config_stem`'s (tools/maya_quants.py, auto-merged): cyber -> maya-cyber-frost.

### saguaro.py - set_up_qfn
Docstring: HEAD's steps 1-9 text plus T7's sentence, renumbered (CYBER-FROST's drafter is step 8 in
HEAD's order, not 7). Body: HEAD's order - step 7 `qfn_vision_step` for every quant (CYBER-FROST's
pictures included, the mmproj is the base model's), step 8 the drafter - `qfn_drafter_step` for
CYBER-FROST, `qfn_mtp_step` (Lamina) otherwise, both under a `step(8, ...)` - step 9 the config, then
HEAD's drafting-choice block, whose object is named "the MTP drafter from the file's NextN block" for
CYBER-FROST instead of "the Lamina MTP draft layer". The `step()` calls stay consecutive 1-9.

### saguaro.py - family_unavailable / NO_ENGINE
HEAD's semantics (NO_ENGINE has no cuda/qfn entry since T2: Linux CUDA is offered; the `WIN and family
== "qfn"` line refuses on Windows) plus T7's None-normalization fix, kept: `NO_ENGINE[backend or
"cuda"][family]`. T7's migration guard (`migrate_cyber_frost` consults
`family_unavailable("qfn", cfg.get("backend"))`) stays and now means what the merged tree means: Metal
and Windows configs stay on the old engine, Linux CUDA configs migrate like AMD's (QFN runs there).
The auto-merged T7 pieces were checked whole: `server_command`'s migrate + durable
`cyber_pack_pending` stop, `migrate_cyber_frost`, `cyber_pack_pending`, and `main()`'s
`if not a.check and migrate_cyber_frost(prev)` (`--check` still rewrites nothing).

### tools/test_setup_qfn.py - Manifest class
Kept both sides' new tests: HEAD's `test_the_vision_tower_is_pinned_too` and T7's
`test_the_cyber_frost_quant`.

### tools/test_setup_qfn.py - Offered class
Kept HEAD's test (`test_amd_and_nvidia_offer_it_metal_refuses`); T7's CUDA-refusal assertions are wrong
now, so the merged test asserts HEAD's behaviour, keeping T7's extra None-backend coverage:
`assertIsNone(family_unavailable("qfn", "cuda"))` and `assertIsNone(family_unavailable("qfn", None))`
(no backend = CUDA default, offered too); Metal still refuses.

## T7 test changes outside the conflict hunks (each one, as the plan asked)
1. `Offered` (above): the two `assertIn("on AMD GPUs", family_unavailable("qfn", "cuda"/None))` became
   `assertIsNone` for cuda and for None.
2. `Drafter` docstring: "CYBER-FROST step 7" -> step 8 (pictures are HEAD's step 7 for every quant).
3. `Drafter.test_vision_entry`: retargeted at HEAD's `qfn_vision_step(a, pc, meta, llama, d, files)` -
   the no_vision and declined-download paths still give text only, and the entry is HEAD's (
   `qfn_vision_exe(True)`, the QFN manifest's `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, `**QFN_VISION`)
   instead of the removed `qfn_vexe` / `mmproj-F16.gguf` shape. Its entry-shape assertions duplicate
   HEAD's `Vision` tests now; the declined path (not covered there) is why it stays.
4. `Migration.test_left_on_the_old_engine_where_qfn_has_no_engine`: T7 asserted cuda, metal and
   no-backend all stay on the old engine. Now: metal stays (no QFN engine), every backend stays under
   `WIN` (patched), and cuda / no-backend migrate on Linux (offered since T2, `--ple-gguf` gone, exe
   `<build>/qfn/saguaro-qfn`). The hip sample's existing migration tests are unchanged.

## Files changed by the merge (all relative)
- `saguaro.py` - the six conflict resolutions above; the auto-merged T7 additions (`QFN_DRAFTER_TOOLS`,
  `qfn_entry` per-quant repo/revision, `server_command` + `main()` guards, `migrate_cyber_frost`,
  `cyber_pack_pending`) verified line by line.
- `tools/test_setup_qfn.py` - the test resolutions above.
- `tools/maya_quants.py`, `tools/qfn/download.json`, `tools/test_maya_quants.py`,
  `tools/test_setup_qwen.py`, `serve/model_pull.py`, `REPORT-t7-cyberfrost.md` - T7's, auto-merged
  cleanly (the manifest holds HEAD's `vision` + `mtp` blocks and T7's three quants; the JSON parses).
- `FIX-REPORT.md` - removed (the add/add above).
`engines/qfn/serve/server.py` and `serve/qfn_engine.py` are untouched by both sides (`git diff` clean),
so `tools/saguaro/extract_qfn_engine.py` needed no run and the engine was not touched by hand.

## Tests (repo root, one at a time, /opt/llm/saguaro/.venv/bin/python -m unittest ...)
| module | result |
|---|---|
| tools.test_setup_qfn | Ran 40, OK |
| tools.test_maya_quants | Ran 24, OK (skipped=1 - the known skip) |
| tools.test_setup_qwen | Ran 18, FAILED (failures=1) - only `Dashboard.test_the_web_apps_names`: node `ReferenceError: navigator is not defined` (serve/web/lib.js:152). Reproduced identically on the untouched HEAD `e1f43e79a` (throwaway worktree, since removed), already logged as environmental by REPORT-t1-identity.md and REPORT-t7-cyberfrost.md; unrelated to this merge. |
| tools.test_qfn_fit | Ran 20, OK |
| serve.test_models | Ran 60, OK |
| serve.test_dashboard | Ran 55, OK |

## Open questions
- The plan's "all must pass": the one failure is the pre-existing environmental node failure above
  (identical before and after the merge on this PC); every test this merge could influence passes.
  A browser-less `navigator` in lib.js or a skip marker there is a different task's call.
- `serve/model_pull.py`'s `qfn_fit` still says "on NVIDIA it is set up by hand" / "the setup offers it
  on AMD GPUs only" - HEAD's wording, unchanged by HEAD when T2 opened NVIDIA and reused now by T7's
  cyber-frost row. It reads wrong on the merged tree (a `fit` for an all-NVIDIA PC says AMD-only); no
  test asserts it, and the plan does not name the file, so it is left for a follow-up.
- A migrated CYBER-FROST config's `vision.exe` (T7's migration) points at `<build>/qfn/vision-qfn`,
  while HEAD's setup writes `qfn_vision_exe()` = `<build-vision-qfn>/bin/vision-qfn[-hip]`. Harmless in
  practice (the durable pack guard forces a `--setup`, which rewrites the config with HEAD's path), but
  the migration's target could be `qfn_vision_exe` for exactness.
- The drafting-choice block now asks about CYBER-FROST's NextN drafter too (the label was adapted); the
  "measure" path still defers to the drafter calibration of a later task, as it does for Lamina.
