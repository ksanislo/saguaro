# REPORT-c122-config-names — saguaro-<model>.json config names (1.2.2)

Branch `saguaro/q1-c122-config-names`, one commit per plan step off `44df46fc9` plus one follow-up commit of the
same work (`293e9e320`, setup-test names) and this report.
Plan: `./PLAN-c122-config-names.md`.

## What changed

### 1. One name-family helper — commit `1e786fe86`
New `tools/config_names.py` is the single place the family lives: the two prefixes (`saguaro-` new, `maya-`
found-but-never-written), the globs (new family first), the sidecar suffixes that are not configs
(`<stem>.shared-settings.json`, `<stem>.turbo.json`), and the operations: `bare` (display name), `write_stem`,
`matches`, `is_config_name`, `list_configs` (both families, a model under both names is the new one), `locate`
(an existing config wherever it sits, the new path for a new one), `resolve` (the old name as the fallback) and
`note_paths` (step 3). Wired through: `serve/models.py` (`config_paths`, `quant_label`, module text),
`serve/model_api.py` (`find_config`, `match_name` — `public_name` unchanged), `serve/server.py` (`_model_path`
now accepts either family and no longer lets a Turbo sidecar through), `saguaro.py` (`configs()` — which also
stops counting `<stem>.turbo.json` as a config, the one the page's list already excluded), and the usage texts
of `tools/calibrate_glm.py`, `calibrate_draft.py`, `calibrate_ds4.py`, `calibrate_ds4_tp.py`,
`calibrate_mm3_prefill.py`, `qfn_calibrate.py`, `maya_golden.py` (they take a path, so "older maya- names work
too" is said where the usage line names one). No behaviour change: nothing was written under a new name yet.

### 2. Writes and the rename — commit `742afeca4`
* New setups write `saguaro-<model>.json`: every config path a setup builds (GLM, DeepSeek, MiniMax incl.
  `--model-dir` names, QFN incl. `maya-cyber-frost`, Hy4, Inkling 2 per-GGUF, and the GLM "before" reads) goes
  through `config_names.locate` — an existing install's file is updated under whatever name it has (so `--setup`
  on one continues its run), and only an absent config is created under the new name.
* `saguaro.py:migrate_config_names()` runs at a start and on the update path (the updater's re-exec lands in the
  same place), not under `--check`, `--report` or `--bench`: each `maya-<model>.json` moves to the new name
  once, atomically (`os.replace`) and only when the target does not exist, with its sidecars — the run scripts
  (contents rewritten: their `--config` line), `<stem>.shared-settings.json`, `<stem>.turbo.json`,
  `<name>.prev`, `tools/saguaro/migrate_configs.py`'s `<name>.bak-*`, and `<stem>.log` (with the config's own
  `"log"` key following it, mtime kept). A file that cannot move is said (`warn`) and stays; finding still
  works under the old name, so a read-only folder leaves everything working. It is idempotent.
* The running config's path keeps working as **the old name as a fallback** — the choice, and why: symlinks are
  not dependable on Windows (`os.symlink` needs elevated rights / developer mode) and the configs, run scripts
  and systemd-style launches live on Windows too. `config_names.resolve()` maps a missing `maya-X.json` to
  `saguaro-X.json` when that exists, and it is applied to `--config` in `saguaro.py main()` (before validation
  and again after the move), to `--config` in `serve/server.py` (a server started directly by llama-swap or a
  unit), to a switch note's config path in `after_server` and to a delete note's in `serve/server.py`.
  External commands that hard-code `./run-maya-<model>.sh` *by path* must be repointed once (the script itself
  moves and is rewritten by the migration); commands that pass `--config maya-<model>.json` need nothing.

### 3. The switch notes — commit `1dddbe5b7`
`SWITCH_FILE` / `DELETE_FILE` are `.saguaro-switch.json` / `.saguaro-delete.json` and are only ever written
there; a note left under `.maya-switch.json` / `.maya-delete.json` by a server from before the rename is read
once (new name first) and cleared under both names (`config_names.note_paths`), and a note's config path
resolves across the rename. **Browser keys deliberately did not change**: the `maya.*` localStorage keys
(`serve/web/lib.js`, `state.js`), the `maya.sampling*` per-model keys and the `maya` IndexedDB are untouched —
they hold users' saved settings, exactly as the plan says to report.

### 4. Docs and the rename tool — commit `79f16365d`
`docs/UPSTREAM.md` §2's keep-list now separates the concrete `maya-*.json` names (kept: running installs point
at them; `config_names` finds both) from the generic placeholder, and `tools/saguaro/rename_rules.py` gained the
two C122 rules: `maya-<model>.json` → `saguaro-<model>.json` and `run-maya-<model>.sh|.bat` → the `run-saguaro-`
forms (idempotent; a test added). The files that quote the old placeholder on purpose — the module that explains
both families (`tools/config_names.py`), `serve/models.py`, `saguaro.py` — are `EXC_TEXT` keeps, reported by
`--check`, and `rename_upstream.py --check` says CHECK OK over the worktree. User-facing text now says
`saguaro-<model>.json` / `run-saguaro-<model>.sh`: `README.md` (2 lines, the speed tables untouched),
`docs/SETUP.md` (incl. the systemd example and one line telling a 1.2.1 install keeps its name until the next
start renames it), `docs/SETTINGS.md`, `docs/HY4-FLASH.md`, `docs/INKLING2-MINI.md`, `docs/AMD_SAGUARO.md`,
`docs/RELEASE-CHECKS.md`, `CONTRIBUTING.md`, and `saguaro.py`'s own text (its `--help` step list, the `--config`
help, the ds4/mm3/ik2 config docstrings, the `--report` line) and one `serve/server.py` comment. `maya-setup.log`,
`maya-bench*` and the `maya-<engine>` program/quant names are not config names and stayed.

### 5. Tests — commit `0d28e96c0`
New `tools/test_config_names.py`, 16 tests: helper semantics; both families listed and shadowing (a model under
both names is the new one, the old file untouched, `locate`); switch = mtime touch ordering; shown names
(`quant_label` under either family); API (`find_config` / `match_name` old↔new, family name alone refused);
`resolve` after the move; the full move with every sidecar (incl. log-key update, run-script rewrite,
idempotence); a taken target; a read-only folder (skipped when run as root) leaving list/find/read/delete
working; update (`UPDATE_EXIT` re-exec passes the old `--config` path on, which resolves); delete with its
sidecars; `note_paths` ordering and a switch note under the old name followed to the moved file.

## Files changed (30)
`saguaro.py`, `serve/models.py`, `serve/model_api.py`, `serve/server.py`, `tools/config_names.py` (new),
`tools/test_config_names.py` (new), `tools/calibrate_glm.py`, `tools/calibrate_draft.py`, `tools/calibrate_ds4.py`,
`tools/calibrate_ds4_tp.py`, `tools/calibrate_mm3_prefill.py`, `tools/qfn_calibrate.py`, `tools/maya_golden.py`,
`tools/saguaro/rename_rules.py`, `tools/test_rename_upstream.py`, `tools/test_maya_models.py`,
`tools/test_setup_qfn.py`, `tools/test_setup_ds4.py`, `tools/test_setup_hy4.py`, `tools/test_setup_ik2.py`,
`tools/test_setup_mm3.py`, `README.md`, `CONTRIBUTING.md`, `docs/SETUP.md`, `docs/SETTINGS.md`, `docs/HY4-FLASH.md`,
`docs/INKLING2-MINI.md`, `docs/AMD_SAGUARO.md`, `docs/RELEASE-CHECKS.md`, `docs/UPSTREAM.md`.

Not touched, per the plan: `engines/`, the Turbo code, the README speed tables, and `serve/qfn_engine.py`
(no change was needed in `engines/qfn/serve/server.py`, so no extraction run; neither file contains
config-name logic).

## Tests run (one at a time, `/opt/llm/saguaro/.venv/bin/python -m unittest <mod>`)
| module | result |
| --- | --- |
| serve.test_models | 64 OK |
| serve.test_model_api | 28 OK |
| serve.test_server | 178 OK |
| tools.test_setup_qfn | 147 OK (1 skipped) |
| tools.test_rename_upstream | 26 OK (incl. the new C122 placeholder case and `--check` over the worktree) |
| tools.test_docs_new_engines | 7 OK |
| tools.test_config_names (new) | 16 OK |
| `tools/saguaro/rename_upstream.py --check` | CHECK OK |
Also run because their files were edited: tools.test_maya_models — 55 OK.

## Open questions / notes for the merge
1. `tools/test_setup_ds4.py`, `test_setup_hy4.py`, `test_setup_ik2.py`, `test_setup_mm3.py` assert the names a
   setup writes; I updated those assertions (and the ds4/mm3 helpers that clear or re-find the written file, the
   fresh-flow run-script name and one ik2 assertion) mechanically to the new names but could not run them — the
   plan says to run only the six named modules; commit `293e9e320`. Please run them in the branch merge.
2. The plan mentions `.lastgood` among the sidecars; this tree has no `.lastgood` (searched all py/js/md).
   The sidecars found beside a config are: `<stem>.shared-settings.json`, `<stem>.turbo.json`, `<name>.prev`,
   `<name>.tmp`, `<name>.bak-<date>-pre-env-rename` (migrate_configs.py), `run-<stem>.sh/.bat` and `<stem>.log`;
   all are moved by the migration. Sessions are not a config sidecar: they live in `<model folder>/var/sessions`
   and are not named after the config.
3. A model under both names is shown as the new one and the old shadowed file is never touched or deleted
   ("rename only when the target does not exist"); a re-`--setup` updates the existing file under whichever name
   it has. Cleaning up a hand-made shadow pair is left to the user — say if 1.2.2 should instead fold the old
   one in (e.g. at setup time) when the contents match.
4. The migration deliberately does not run for `--check`, `--report` or `--bench` (diagnostics change nothing);
   `--setup` and every start/update path do rename first.
5. The model API's JSON keys (`"maya"` in /api/tags and /api/show entries) and `maya-bench` /
   `maya-setup.log`-style program artefacts were left: they are API/program names, not config file names,
   and the plan only covered the config family, the notes and the browser keys.
6. `VERSION` still reads 1.2.0 and CHANGELOG has no 1.2.2 entry — the release commit is Ken's call after the
   1.2.1 merge (plan header: "merged after 1.2.1 is published").
7. Windows note for the release text: the rename is plain `os.replace` in the config folder; a run script that
   is open in an editor there simply stays and says so, and its model still starts via the old-name fallback.
