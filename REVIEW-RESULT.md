# REVIEW-RESULT — C122 saguaro-<model>.json config names (re-review of the full diff 44df46fc..6856e2c0, fix round included)

**MERGE-WITH-NITS**

Re-review of `git diff 44df46fc9578d6aae3ed28d37021cff8d5b2df82..HEAD` against PLAN-c122-config-names.md,
REPORT-c122-config-names.md and FIX-REPORT.md, replacing the previous round's note. All five manager findings
(FIX-FIRST 1, 2, 3, 4, 7) are fixed in `9dcd37ad6..d94f1f002` and hold; nothing in the fix round introduces a
blocking problem. The plan's modules were re-run one at a time, all green (table at the end). Nothing edited but
this file.

## The previous round's findings — each verified closed

- **1 (never sweep a neighbouring config as a sidecar)** — the beside-scan now matches only the names that belong
  to the config: the two run scripts by exact name, `<stem>.log`, `<stem>.turbo.json`,
  `<stem>.shared-settings.json`, `<name>.prev`, `<name>.tmp`, `<name>.bak-*`, and skips entries that are
  themselves configs (`saguaro.py:847-858`); the exact patterns make a `.json` match impossible anyway. Tested
  both in the plain form and in the reachable shadow-pair form (`tools/test_config_names.py:231-269`).
- **2 (a stray note must not outlive the server)** — `after_server` builds both note paths first, reads only on a
  restart, and unlinks every note of either family on *every* exit, the normal stop included
  (`saguaro.py:5297-5306`); tested over `rc = 0` and `UPDATE_EXIT`, both names
  (`tools/test_config_names.py:389-402`). Strictly a superset of 1.2.1's behaviour (44df46fc:saguaro.py:5192-5204
  unlinked only on the update/restart exits and left a stray through normal stops); with the 600 s gate no real
  switch is lost either way.
- **3 (never restore a deleted model from a shadow)** — both names: the old file goes aside **once** as
  `maya-<model>.json.shadowed`, warned once, never deleted (`saguaro.py:901-911`); the name matches no
  `maya-*.json` glob (`saguaro.py:898`) and is not a config name (`tools/config_names.py:32-35`), so nothing
  loads it; the delete of the model removes the old-name twin and that name's run scripts / `.prev` /
  `.shared-settings.json` too (`serve/models.py:1476-1479`). Tested end to end
  (`tools/test_config_names.py:271-311`).
- **4 (a crash or a locked file mid-move is finished at the next start)** — the beside-files move lives in
  `_finish_config_rename` (`saguaro.py:837-884`) and a second pass re-runs it for every new-name config whose old
  stem left leftovers, skipping a stem whose old config still exists (its own pass) or went aside
  (`saguaro.py:924-931`). Idempotent; tested with a crash between the halves and with one locked sidecar
  (`tools/test_config_names.py:180-229`).
- **7 (dead helper)** — `write_stem` is gone from `tools/config_names.py` and its test lines with it; the qfn
  docstring, the GLM comment, `maya_quants.py` and `maya_nightly.py` texts now say saguaro-.
- The extra commit `aee13b04b` (skip the review/fix notes in the rename tool, `rename_rules.py:253`, plus a test)
  is a reasonable extension of the existing PLAN/REPORT skip.

## The brief's questions

- **Can any path lose a config, or make an installed model vanish from the Models page?** No. Both families are
  listed, new first, sidecars filtered out *before* the shadow set is built, so a sidecar can never hide a model
  (`tools/config_names.py:54-66`); a rename lands only on an absent target (`saguaro.py:914-915`); a pair's old
  file goes aside, not away (`saguaro.py:901-911`); a file that cannot move is said and is still listed, shown,
  read, switched, deleted and updated under the old name (read-only-folder test ran here — non-root — and passed,
  `tools/test_config_names.py:313-332`); and every write path updates the existing file wherever it still sits
  (`CN.locate`, `tools/config_names.py:69-77`, called from `saguaro.py:4098-4099`, `6830`, `7408-7411`, `8206`,
  `8828`, `9281-9284`) — no write can create a competing new-name file beside an unmoved install.
- **Is the rename atomic and done at most once?** Per file, yes: same-folder `os.replace` onto a checked-absent
  target; reruns do nothing (`tools/test_config_names.py:163, 196, 244`). A config is never renamed twice: the
  finish pass moves only sidecars, never configs (`saguaro.py:855-856, 924-931`), and the shadow's set-aside is
  guarded by `aside.exists()` (`saguaro.py:903`). The config + sidecars *set* is not one atomic step — it is
  resumable instead, which is what finding 4 asked for.
- **Does a config path stored in llama-swap / systemd with the old name still work?** Yes: `resolve` maps a
  missing `maya-X.json` to the same folder's `saguaro-X.json` (`tools/config_names.py:89-97`) and is applied at
  both entry points before validation (`saguaro.py:9813`, `serve/server.py:8916-8917`), again after a move in the
  same start (`saguaro.py:9850`), and to switch/delete note paths (`saguaro.py:5313`, `serve/server.py:3817`);
  API model names given in either family match (`serve/model_api.py:45-55, 71-77`). The report's disclosed caveat
  stands and is correct: a unit that *executes* `./run-maya-<model>.sh` by path must be repointed once (the
  script itself moves and has its `--config` line rewritten); a `--config maya-<model>.json` unit needs nothing.
- **Are the sidecars all covered?** Everything this tree names after a config: `run-<stem>.sh/.bat` (contents
  rewritten), `<stem>.log` with the config's `"log"` key following it and its mtime kept,
  `<stem>.shared-settings.json`, `<stem>.turbo.json`, `<name>.prev`, `<name>.tmp`, `<name>.bak-*`
  (`saguaro.py:847-885`; the bak name comes from `tools/saguaro/migrate_configs.py:125`), tested
  (`tools/test_config_names.py:150-179`). The plan's `.lastgood` exists nowhere in the tree (grep: only the plan
  and report texts) — the report's inventory claim is right. Sessions live per model folder (`var/sessions`), not
  under the config name; `maya-setup.log` / `maya-bench*` are program artefacts, not sidecars.

## Findings (by severity; none blocks the merge)

1. **Nit — stale name text the finding-7 pass missed:** `saguaro.py:687` ("its config maya-minimax-maya-xs.json"),
   `saguaro.py:8822` (`write_config_hy4`'s docstring "maya-hy4-flash-maya-s.json and its start script"),
   `saguaro.py:9333` (`write_config_ik2`'s docstring "maya-inkling-2-mini[-<file>].json"). The functions write the
   saguaro- name via `CN.locate`; the rename tool's `--check` does not flag these (concrete names, not the
   `maya-<model>.json` placeholder), so they will sit silently.
2. **Nit — untested wiring on the server side:** `serve/server.py`'s `--config` resolve for a directly started
   server (`serve/server.py:8916-8917`, exactly the llama-swap/systemd entry) and `finish_delete` reading a
   pre-rename `.maya-delete.json` (`serve/server.py:3807-3819`) are exercised only through the helper's own tests
   (`tools/test_config_names.py:134-142, 373-387`), not at these call sites. One request each would pin them.
3. **Note — intentional deltas vs 1.2.1, all disclosed:** `_model_path` now refuses a `.turbo.json` the old regex
   let through (`serve/server.py:3659-3664`); `configs()` no longer counts a turbo sidecar as a CLI-selectable
   model (`saguaro.py:829-835`); the API and the page match names across both families
   (`serve/model_api.py:45-55, 64-77`); `after_server` clears notes on normal exits too (see above). For a
   maya-only install the list/run/read/save paths are otherwise the same code paths and sort identically, and
   every rewrite keeps the existing `json.dumps(..., indent=1)` form.
4. **Note — delete leaves the residue it always left, now under both names:** `.shadowed`, each stem's
   `.turbo.json`, `.log`, `.bak-*` stay (`serve/models.py:1474-1482`; the log/turbo leak predates C122). Nothing
   loads a `.shadowed` copy and the finish pass skips a stem that has one (`saguaro.py:930-931`), so it is
   kept-data, not a resurrection path — as FIX-REPORT intends ("kept, not lost").
5. **Note — a permanently read-only folder warns on every start** (`saguaro.py:917-919`: the retry says the failure
   each time); only the set-aside warns once. Consistent with "a file that cannot move is said".
6. **Note — merge-time action (unchanged from the last round, FIX-REPORT §"the rest"):** `tools.test_setup_ds4`,
   `tools.test_setup_hy4`, `tools.test_setup_ik2`, `tools.test_setup_mm3` and `tools.test_maya_models` are outside
   the plan's run list and were not run here either; I reviewed those hunks by eye — the assertions and the
   clear-both-families helpers match the write paths (e.g. `tools/test_setup_ds4.py:349-353`,
   `tools/test_setup_mm3.py:359-363`), and `write_config_*` returns the located path the tests read back. Run them
   when merging, as both reports ask.
7. **Note — process hygiene:** `FIX-REPORT.md` was force-added although named on the ignore list, and this note is
   excluded via `.git/info/exclude` (`:42`), so it commits only with `-f`; the rename tool skips both note families
   like the PLAN/REPORT notes (`rename_rules.py:253`). Works, just worth knowing for the merge.

## Checks that came back clean

- **Locks / threads / stdout:** no new or nested lock acquisition anywhere in the diff; `migrate_config_names` is
  plain file moves plus `warn`/`say` prints, running single-threaded in `main()` before the server subprocess
  starts (`saguaro.py:9843-9848`), so nothing interleaves with server output; `finish_delete` runs on the server's
  start path before threads; `delete`/`_model_path` add only file ops inside existing handlers.
- **Generated file:** `serve/qfn_engine.py` and its source `engines/qfn/serve/server.py` are untouched by the diff
  (and contain no config-name logic) — the source+generator rule holds vacuously, nothing to extract. `engines/`
  is untouched entirely, as the plan's Don'ts require.
- **Browser keys:** `serve/web/*` is not in the diff — `maya.*` localStorage and the `maya` IndexedDB kept exactly
  as the plan's step 3 and the report say.
- **Notes:** written only under `.saguaro-switch.json` / `.saguaro-delete.json` (`serve/server.py:3791, 3878`),
  read under both names new-first, cleared under both (`tools/config_names.py:80-86`, `serve/server.py:3807-3819`,
  `saguaro.py:5297-5306`).
- **Migration scope:** runs on every start/update and on `--setup` (`saguaro.py:9843`); `--report`, `--bench`,
  `--import-model` return before it (`saguaro.py:9827-9841`) and `--check` skips it — as the plan says; the
  updater's re-exec lands back in the same place.
- **Rename tool / keep-list:** the two C122 placeholder rules (`rename_rules.py:183-184` — a replacement is never
  a match) with the three EXC_TEXT keeps (`rename_rules.py:238-240`); worktree `--check` green inside the test.
- **Repo hygiene:** every touched path is inside the repo; no secrets; the only absolute paths added are the
  plan's own `/opt/llm/saguaro/.venv` mentions in the report notes' test tables.

## Tests (plan's list only, one module at a time, `/opt/llm/saguaro/.venv/bin/python -m unittest <mod>`)

| module | result |
| --- | --- |
| serve.test_models | 64 OK |
| serve.test_model_api | 28 OK |
| serve.test_server | 178 OK |
| tools.test_setup_qfn | 147 OK (1 skip) |
| tools.test_rename_upstream | 27 OK (worktree `--check` inside it: CHECK OK) |
| tools.test_docs_new_engines | 7 OK |
| tools.test_config_names | 22 OK — the read-only-folder test ran (user `ken`, not root) |

Worktree clean before and after the runs.
