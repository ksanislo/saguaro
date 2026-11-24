# Report: v1.2.1-C nits (PLAN-v121c-nits.md)

Branch `saguaro/q1-v121c-nits`, four commits, one per numbered step.

## Step 1 - the --bench result line names the QFN drafter by what --mtp is

`tools/maya_bench.py` (`drafting_choice`, the `family == "qfn"` branch) said `MTP (Lamina)`
whenever a drafter was loaded, for CYBER-FROST too. The result line now says

- `MTP (from the model's NextN block)` when the config's `--mtp` folder is the pack's own
  `mtp` folder (`<pack>/mtp`, what `qfn_drafter_step` builds and `keep_drafting` writes), and
- `MTP (Lamina)` otherwise (the downloaded `Qwen3.8-Flash-Next-MTP-Lamina` layer).

`MTP drafts off (--mtp-max-t 1)` is unchanged. Files: `tools/maya_bench.py`.
Commit `a3cab0d93`.

## Step 2 - one version everywhere

`--bench` (`maya_version()`), `--report` and `/health` (`SAGUARO_VERSION` in
`serve/server.py`), the banner and the dashboard's About (`saguaro_version.py`) all already
read the checkout's own `VERSION` file - the file itself was the odd one out, still at
`1.1.50` while the Saguaro section of the CHANGELOG is 1.2.0. `VERSION` is now `1.2.0`;
verified after the bump: `maya_version() == SAGUARO_VERSION == 1.2.0`, and the flat `/health`
`version` field (served straight from `SAGUARO_VERSION`, as `serve/test_dashboard.py`'s
`test_health_has_the_version` asserts) shows it. No runtime path prints `v1.1.50` any more.
Files: `VERSION`. Commit `003dbe8ca`.

Left alone deliberately: `CHANGELOG.md`'s and `docs/MI50_VALIDATION.md`'s `v1.1.50` lines
(history, and an upstream-commit reference), the engine's own facts (Monitor `Engine` line,
`/props` `build_info`: the engine's version, not Saguaro's), and the Ollama-API shim's
`/api/version` `0.0.0-maya` (an API-compatibility constant, not a displayed release).

## Step 3 - hipFree's nodiscard warning in the QFN HIP build

`engines/qfn/src/program/generate.cpp` ends the session with bare `cudaFree(...)` calls
(`cudaFree` -> `hipFree` through `include/strata/platform/hip_compat/strata_hip.h`), which
the HIP build warns about at ~24051-24056. Every bare `cudaFree`/`cudaFreeHost` (->
`hipHostFree`) in the file - 19 call sites, host code only - now takes the `(void)` cast the
file and the engine already use for the same calls (`(void) cudaFree(p)` at ~9993,
`(void) cudaFreeHost` in `layer.cpp`/`expert_cache.cpp`); the one bare `cudaGetLastError()`
in the same region also takes the cast the file's ~40 other sites use. No behaviour change;
no bare ignores of those calls remain in the file.
File: `engines/qfn/src/program/generate.cpp`. Commit `e3d707bea`.

## Step 4 - CYBER-FROST's picture tower is not downloaded twice

`saguaro.py`: new `qfn_vision_reuse(m, d)`, called from `qfn_vision_step` before it offers
the download. The base model's mmproj is one file every QFN quant shares, so step 7 now
first searches the folders the setup knows for a copy with the published size and a matching
sha256 (`sha256_checked`, the memoized checker): every QFN model folder under the models
root (its `vision/` subfolder first, then the folder itself) and the `vision` folder each
written QFN config (all quants, CUDA and HIP) reads. A found file is hard linked into the
model's own `vision` folder (no second copy of its bytes, its own name and VERIFIED record);
a file on another filesystem is used by path. Only when nothing whole is seen does the
setup offer the download, as before. Files: `saguaro.py`. Commit `dc151f0fa`.

## Tests run (one module at a time, as the plan names)

- `tools.test_setup_qfn`: 52 tests, OK (1 skipped) - after step 4.
- `tools.test_bench`: 33 tests, OK - after step 1 and again at the end.
- `serve.test_dashboard`: 58 tests, OK - after all code steps.
- Extra (scratch, not committed): `scratch/test_vision_reuse.py` exercises the new reuse
  scan directly: hard link from another quant's `vision/` folder, from a model folder
  without one, a whole own copy kept, a failed-download stub replaced by the link, and None
  (download offered) when no copy is anywhere - all pass.
- `python -m py_compile saguaro.py` clean.

## Open questions

- Step 3: the HIP build log the plan quotes is not in the repo, so "the same pattern
  nearby" was taken as every bare ignore of `hipFree`/`hipHostFree` (plus the lone bare
  `hipGetLastError`) in `generate.cpp`; the HIP engine could not be rebuilt here (no ROCm
  toolkit in this environment), so the warning silencing is by inspection, not a rebuild.
- Step 4: a path-reused (cross-filesystem) mmproj is referenced where it stands - if the
  folder of the other quant is later deleted, the vision entry points at a gone file until
  the next setup; hard-link reuse (the normal case) is unaffected.
- Step 2: `VERSION` was still 1.1.50 while the plan states the current version is 1.2.0;
  the bump follows the plan. Whether 1.2.0 also needs a CHANGELOG release heading is out of
  this plan's scope and was left to the release step.
