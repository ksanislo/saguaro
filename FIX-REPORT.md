# FIX-REPORT — C122 review findings (REVIEW-RESULT.md, "FIX-FIRST 1, 2, 3, 4 and 7, the rest as they are")

Branch `saguaro/q1-c122-config-names` off `30fec98de`, one commit per numbered finding. Plan:
`./PLAN-c122-config-names.md`; review: `./REVIEW-RESULT.md`.

## The FIX-FIRST findings

### 1 - "never treat another config as a sidecar" — commit `9dcd37ad6`
`saguaro.py:migrate_config_names` no longer moves a file just because its name starts with `p.stem + "."`. The scan
now matches only the names that actually belong to a config: `run-<stem>.sh/.bat`, `<stem>.log`,
`<stem>.turbo.json`, `<stem>.shared-settings.json`, `<name>.prev`, `<name>.tmp`, `<name>.bak-*` - and an entry
that is itself a config (`CN.is_config_name`) is skipped outright (the review's letter; the exact patterns make it
impossible anyway, since a config's name is `<stem>.json` and `.json` is not one of the sidecar suffixes). A
dot-named variant (`maya-2.5.json` beside `maya-2.json`) therefore keeps its files to itself and gets its own
rename pass, run script moved and rewritten. Tests: `test_a_neighbouring_config_is_no_sidecar` (both variants move
with their own sidecars, nothing said) and `test_a_neighbour_does_not_drag_a_shadowed_pair_around` (the reachable
form: with the pair `maya-2.5.json` + `saguaro-2.5.json`, the neighbour used to drag `maya-2.5.turbo.json` into
`saguaro-2.5.turbo.json` - verified against the pre-fix code, it did; now it stays the shadow's own).

### 2 - "remove a leftover switch note of either family" on a non-restart exit — commit `8234fe052`
`after_server` now builds `notes = CN.note_paths(ROOT, SWITCH_FILE)` before anything else, reads it only when
`rc == RESTART_EXIT` (as before), and unlinks every note found - `.saguaro-switch.json` and a pre-rename
`.maya-switch.json` - on every exit, including the early return for a normal stop, so a stray note (a crash
between the server's write and its restart) no longer lingers through starts and update-exits. Test:
`test_a_stray_note_does_not_survive_the_server` over a normal exit (`rc = 0`) and an update-exit, both names.

### 3 - "a new-name config exists (or existed and was deleted): never restore the old one" — commit `3638c4cd1`
* Both names at once: the migration renames the old file aside **once** as **`maya-<model>.json.shadowed`** - the
  name nothing loads: it stops matching the `maya-*.json` glob every list/find path uses, is not a config name,
  and a repeat run sees only the new file. It is `warn`ed once ("the new one is used, and the old file is kept
  beside it as ..."), never lost. `--config`/llama-swap paths given the old name still resolve (unchanged
  `CN.resolve`).
* Deleting a model deletes its shadow: `serve/models.py:delete` removes the old-name twin and that name's run
  scripts / `.prev` / `.shared-settings.json` beside the config it deletes, so the next start's rename has nothing
  to move back and the deleted model stays deleted.
* The neighbour pass from finding 1 also never touches a shadowed pair's sidecars.
Tests: `test_both_names_the_new_one_and_the_old_goes_aside` (replaces the old-stays test - the review called that
very no-op the restore-after-delete path) and `test_a_deleted_model_does_not_return_from_its_shadow`.

### 4 - "the sidecars ... so a crash between can be finished at the next start (idempotent)" — commit `be6982ca5`
The beside-files move was extracted to `_finish_config_rename(old_stem, target)` (same moves, same warnings, now
also covering `<name>.tmp`), and `migrate_config_names` gained a second pass: for every new-name config it looks
for the old stem's leftovers beside it and finishes them - what a crash between the config move and its sidecars
left, and what a single locked sidecar left, moves at the next start; a rerun after that does nothing. The pass
skips a stem whose old config still exists (its own pass handles it) or that went aside as `.shadowed` (finding
3: a set-aside shadow keeps its files with the shadow). So the set is never permanently orphaned and the page's
shared-settings/Turbo reads recover by themselves once the locked file lets go. Tests:
`test_a_crash_between_the_moves_is_finished_at_the_next_start` (config moved with the sidecar pass disabled, next
start finishes: files, contents, the config's `"log"` key, the run script) and
`test_a_locked_sidecar_moves_at_the_next_start` (one file raises EACCES: said once, stays; next start moves it).

### 7 - "remove the dead helper" + the cheap text nits — commit `d94f1f002`
`tools/config_names.py:write_stem` removed (no production caller; `CN.locate` is the write path) and its test
lines with it. The stale texts now say the new family: `saguaro.py` `write_config_qfn`'s docstring
(`saguaro-qwen3.8-flash-next-<quant>[-hip].json`, `saguaro-cyber-frost...`), the GLM config-name comment beside
`BACKEND_SUFFIX`, `tools/maya_quants.py`'s `configs` line, `tools/maya_nightly.py`'s example config.

## The rest, as they are
* **5, 6** - notes on intentional deltas and migration scope; nothing to fix, nothing touched.
* **7's duplicate `.saguaro-switch.json`** string in `saguaro.py` / `serve/models.py` - the review itself calls it
  the pre-existing pattern; left.
* **8 - merge-time action, not mine to run here:** `tools/test_setup_ds4.py`, `tools/test_setup_hy4.py`,
  `tools/test_setup_ik2.py`, `tools/test_setup_mm3.py` and `tools/test_maya_models.py` are outside the plan's run
  list ("Run ONLY ..."), so they were not run; they (and `rename_upstream --check`) belong to the 1.2.1 merge.
  My `after_server` change keeps the restart path byte-for-byte (only the non-restart paths now also unlink), so
  `test_maya_models.py`'s switch-note tests see the same behaviour; the docstring/comment edits in finding 7 are
  text-only, so `test_maya_quants`/`test_maya_nightly` see the same code.
* `REPORT-c122-config-names.md` left as the step-5 record; note its helper list still names the now-removed
  `write_stem` (finding 7) - history, not corrected in place.

## One extra commit (needed to keep the plan's run list green) - `aee13b04b`
`tools.test_rename_upstream`'s `test_check_passes_over_the_worktree` failed on this worktree - on
`REVIEW-RESULT.md` itself, not on the branch's code: the walk reads the review's own note, whose `maya-<model>.json`
placeholders the C122 rule rewrites. `SKIP_NAME_RE` now skips `REVIEW-RESULT*.md` and `FIX-REPORT*.md` like the
`PLAN-*/REPORT-*` notes they are written from (this file included), plus a skip test. Nothing else in the tree
changed behaviour; the review's own CHECK OK predates the note file.

## Tests (plan's list, one module at a time, `/opt/llm/saguaro/.venv/bin/python -m unittest`)
serve.test_models 64 OK, serve.test_model_api 28 OK, serve.test_server 178 OK, tools.test_setup_qfn 147 OK
(1 skip), tools.test_rename_upstream 27 OK (26 + the new skip test; its in-test `--check` runs over this worktree
and passes), tools.test_docs_new_engines 7 OK, tools.test_config_names 22 OK (16 + 6 new/updated here). Worktree
clean after the runs; `scratch/` holds only throwaway copies.
