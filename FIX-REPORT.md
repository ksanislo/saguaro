# FIX-REPORT — T7: CYBER-FROST runs on the QFN engine

Response to REVIEW-RESULT.md (verdict FIX-FIRST). Each finding below maps to what changed, in which file and
commit, or why it was left as it is. Fixes are one commit each; only the files each fix touched were staged.

Fixes applied: finding 1 (HIGH), finding 2 (MEDIUM) — the two FIX-FIRST items — and the one cheap NIT,
finding 3. The three INFO items (4, 5, 6) were reviewed and left unchanged, with reasons below.

## FIX-FIRST findings

### 1. HIGH — unconditional migration stranded existing CYBER-FROST installs on NVIDIA/Metal — FIXED
Commit `fc7625108`.

`migrate_cyber_frost` matched only the installer family and the files, never the backend, so an old-engine CUDA
install was rewritten to `<build>/qfn/saguaro-qfn` on its next start, stopped with "run `--setup` once", and then
`--setup` refused to build the engine there (`NO_ENGINE["cuda"]["qfn"]`) — dead at both ends, with the old config
gone from disk.

Change: after the existing predicate confirms an old-engine CYBER-FROST config, `migrate_cyber_frost` now
consults `family_unavailable("qfn", cfg.get("backend"))` and returns `False` when the backend has no QFN engine
(`saguaro.py:8674`). A CUDA or Metal install is left exactly as it was — family "qwen", the GLM exe, its
`--ple-gguf` — so it keeps starting on the old engine, which the plan keeps precisely until the NVIDIA route is
decided. Only a backend where saguaro-qfn exists (currently HIP) is migrated.

Incidental fix this surfaces: `family_unavailable(family, backend)` is typed `backend: str | None` and its first
line normalizes with `NO_ENGINE.get(backend or "cuda")`, but its second line indexed `NO_ENGINE[backend]`, so a
config with no `backend` field (a hand-made or very old one) raised `KeyError: None` the moment this path called
it. Corrected to `NO_ENGINE[backend or "cuda"]` (`saguaro.py:7670`), matching its own normalization and the
codebase's "missing backend = CUDA" convention. No behaviour changes for the non-None callers.

Tests (`tools/test_setup_qfn.py`): `Migration.test_left_on_the_old_engine_where_qfn_has_no_engine` asserts a
cuda/metal/no-backend old CYBER-FROST config is not migrated and keeps its GLM exe and `--ple-gguf`;
`Offered.test_amd_offers_nvidia_lists_with_the_reason` now also asserts `family_unavailable("qfn", None)` gives
the CUDA reason. The existing `Migration` tests use a `backend: "hip"` sample, so they still migrate as before.

### 2. MEDIUM — the one-time "its pack is still the other engine's" stop could be bypassed — FIXED
Commit `5e2a6ee1a`.

The stop lived only in the migrate-then-fail of `server_command`. `main()` ran the same `migrate_cyber_frost` on
the setup-mode path, and `--check` is a setup-mode command that reports and returns without building anything, so
`--check` rewrote the config to saguaro-qfn with no stop message; a later plain start saw an already-QFN config
(`migrate_cyber_frost` then returns `False`) and, where `build-<backend>/qfn/saguaro-qfn` already exists on the
same 96 GB AMD PCs, launched it against the pack that is still the GLM format — the misread the stop exists to
prevent (both packs say "index v3" while the FORM tables and dense.bin differ).

Change (the durable guard, the review's first suggested fix):
- New `cyber_pack_pending(cfg)` (`saguaro.py:8719`): for a family-qfn config whose `qwen_quant` is
  `Q.CYBER_FROST_QUANT`, returns its `--pack` path while that pack's `sources.json` does not name the QFN pack
  tools (`read_json(pack / PACK_SOURCES).get("tools") != S.source_hash(QFN_PACK_TOOLS)`), else `None`. A Qwen-path
  pack stamps `QWEN_PACK_TOOLS` (saguaro.py:7080) and a pack missing entirely has no record, so both read as
  pending; only `qfn_pack_step` writes the QFN tools stamp (saguaro.py:7443), which is what clears it.
- `server_command` now refuses on `cyber_pack_pending(cfg)` on every start (`saguaro.py:4584`), not once per
  migration. The "runs on its own engine" `ok` line still prints only on the migration itself.
- The setup path no longer migrates on `--check`, which builds nothing and so must not rewrite the config either
  (`saguaro.py:8962`, `if not a.check and migrate_cyber_frost(prev)`). A real `--setup` still migrates and then
  continues into `set_up_qfn`, which rebuilds the pack; a `--check` leaves the config on the old engine, and any
  later start is governed by the durable guard.

Together these close the vector twice over: `--check` no longer mutates the config, and even if a config reaches a
foreign pack by any route (a hand edit included), the start is refused until `--setup` rebuilds the pack.

Test (`tools/test_setup_qfn.py`): `Migration.test_pack_still_the_other_engines_blocks_the_start` drives the real
`cyber_pack_pending` — pending with no `sources.json`, pending with a `glm-tools` pack, cleared once the pack names
the QFN tools, and `None` for a non-CYBER-FROST QFN quant and for the old engine.

### 3. NIT — stray leading space in tools/qfn/download.json — FIXED
Commit `f5c70852e`. The Q8_0 `00006` file line (line 33) had gained a fifth leading space; restored to the four its
siblings use. The entry is otherwise untouched and the JSON validates (`quants`: ud-q4_k_xl, q8_0,
cyber-frost-q5_k_m).

## INFO items — reviewed, not changed

### 4. INFO — first `--setup` rebuilds the drafter even when `<pack>/mtp` is present
Left as is, as the review itself concludes. The old `qwen_pack_step` stamped only the pack root, so
`qfn_drafter_step`'s stamp check misses and rebuilds — re-checking a foreign-format artifact is the safe choice, at
a few minutes once. No code change would be correct here. (The branch already noted the "kept while unchanged"
wording; this report does not change that wording.)

### 5. INFO — the migrated config's `--expert-profile` initially points at the old engine's profile file
Left as is. It is harmless precisely because the stop is now durable (finding 2): the engine cannot start on a still
foreign pack, and the `--setup` that clears the guard rewrites the `--expert-profile` paths to
`QFN_PROFILE`/`var/expert-profile.bin` via `write_config_qfn` (saguaro.py:7545-7551). Rewriting the profile path
inside `migrate_cyber_frost` instead would change `--setup`'s owned files behind its back and duplicate that logic;
the durable guard is the intended single point of control.

### 6. INFO — vision-qfn is not built by this branch
Left as is; acknowledged in REPORT-t7-cyberfrost.md as follow-up territory. The QFN server adds `--vision` at most
once and starts the reader lazily, so only picture requests fail, not the start.

## The two report-accuracy gaps the review noted
Both described behaviours that are now fixed, so they no longer describe this branch: the NVIDIA open question no
longer strands existing CUDA installs (finding 1 — they are not migrated), and the one-time-stop open question no
longer allows a `--check` to migrate past the stop (finding 2 — `--check` does not migrate, and the start guard is
durable). REPORT-t7-cyberfrost.md itself was not edited here; these lines record the change.

## Tests (repo root, one module at a time, `/opt/llm/saguaro/.venv/bin/python -m unittest ...`)

| module | before | after | note |
|---|---|---|---|
| tools.test_setup_qfn | Ran 18, OK | Ran 20, OK | +3 tests: family_unavailable None, CUDA/Metal old config not migrated, cyber_pack_pending |
| tools.test_maya_quants | Ran 24, OK (skipped=1) | Ran 24, OK (skipped=1) | unchanged |
| tools.test_setup_qwen | Ran 18, FAILED (failures=1) | Ran 18, FAILED (failures=1) | the same environmental `Dashboard.test_the_web_apps_names` (`ReferenceError: navigator is not defined` in `serve/web/lib.js:152` under node); untouched by this task, identical before and after |

## Files changed (and only these staged per commit)
- `saguaro.py` — finding 1 (migration backend guard + `family_unavailable` None fix), finding 2
  (`cyber_pack_pending`, durable start guard, `--check` does not migrate)
- `tools/test_setup_qfn.py` — three tests for the above
- `tools/qfn/download.json` — finding 3 whitespace

No engine C++ was touched; `serve/qfn_engine.py` and `engines/qfn/serve/server.py` are unchanged, so no engine
regeneration was needed. No paths outside the repo, no secrets.
