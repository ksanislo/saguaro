# T3: QFN setup features - pictures, drafting choice, answers at once, import - REPORT

QFN (saguaro-qfn, family "qfn") now offers the three things the other engines' setups offer - pictures, a drafting
choice and "answers at once" - and its files are in `--import-model`'s table. The engine already supported all three
(the live hand-made T4 configs in ./ref/ show what the server starts); this only taught the setup to write and
download them. No engine sources, `serve/` (incl. `serve/qfn_engine.py`), the web UI, CYBER-FROST's functions or the
NVIDIA build/fit code were touched; no engine was built.

## What changed, by step (one commit each)

### Step 1 - pictures  (commit `61a407d44`)
- `tools/qfn/download.json`: a new `"vision"` section - the model's vision tower as llama.cpp's mmproj, pinned to the
  same repo revision as the quants (`efc50b7c...`) with its sha256. The Hub file is `mmproj-BF16.gguf`; `hub_names`
  downloads it under the name the live configs use, `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (907,542,944 B, sha256
  `2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355` - read from the Hugging Face API, and the resolve
  URL + content-length were checked with curl).
- `saguaro.py` constants: `QFN_VBUILD` (`build-vision-qfn/`, its own folder: engines/qfn pins another llama.cpp
  commit than `tools/vision`), `QFN_VSTAMP`, `QFN_VISION_SRC`, `QFN_VISION` (the live entry's elastic keys) and
  `qfn_vision_exe(hip)` (`vision-qfn`, or `vision-qfn-hip` on AMD - how `engines/qfn/tools/vision/CMakeLists.txt`
  names the GPU reader; `serve/server.py` start_qfn and `serve/qfn_engine.py` Vision read the same section).
- `qfn_vision_model()` - the tower as offer_download's form (mirrors `qwen_vision_model`).
- `compile_vision_qfn()` - builds `strata-vision` from `engines/qfn/tools/vision` for this PC's GPUs/toolkit
  (HIP: `-DSTRATA_VISION_HIP=ON` + archs; CUDA: `-DSTRATA_VISION_CUDA=ON`; `-DSTRATA_PORTABLE=OFF`), mirroring
  `compile_vision`; `qfn_vision_step()` - step 7: download the mmproj (pinned, sha256-checked, into
  `<model folder>/vision`), build the reader, and return the config's `"vision"` entry with the live config's keys -
  exe, mmproj, model, gpu, max_tokens 1024, elastic, elastic_mib 1536, elastic_idle_s 30, elastic_keep_s 600.
- `write_config_qfn(..., vision=)` writes the entry and adds `--vision` to the engine args; `--no-vision` (or a failed
  download/compile) leaves both out. `set_up_qfn` calls the new step 7 (MTP moved to 8, config to 9) and passes the
  entry through; the setup's model row and `qfn_about()` no longer say "text only".

### Step 2 - drafting choice  (commit `673e4ca61`)
- `set_up_qfn`: after the config is written, if the Lamina draft layer is there it asks `drafting_choice(a, "the
  Lamina MTP draft layer", ...)`. `--drafting off` -> `keep_drafting(cfg, False)` = the engine arg `--mtp-max-t 1`
  (`keep_drafting` already handled the "qfn" family: it keeps `--mtp` loaded and records `installer.mtp_calibration`).
  `on` = same, drafting kept. `measure` (the default, so `--yes`) is accepted and drafting stays on as written - the
  measuring itself is task T4's step 4; nothing here builds or runs the engine.

### Step 3 - answers at once  (commit `2548fe36b`)
- `PARALLEL_FAMILIES` gains `"qfn"`, so `check_parallel` lets `--parallel` through for QFN (its refusal text now lists
  Qwen3.8-Flash-Next among the engines that support it).
- `write_config_qfn` writes the top-level `"parallel": N` (the QFN server's `parallel_args` turns it into the engine's
  `--batch N`). Default: off on the one 16 GB AMD card it was measured on (parallel decoding lost speed there - no
  key written, as the setup's AMD config before), 2 on several GPUs as the live T4 configs run it; `--parallel N`
  says, and a value the config before carried is kept. `--parallel 1` = off.

### Step 4 - `import_catalog`  (commit `3eb4b319c`->`3eb4b3199`)
- Entries for QFN, like the other engines': each quant's shards into `<name>/` (UD-Q4_K_XL 4 files, Q8_0 6), the Lamina
  draft layer into `Qwen3.8-Flash-Next-MTP-Lamina/` (its `LICENSE` has no published sha256, so it is not matched) and
  the shared picture tower into each model's `vision/`. 16 entries, all matched by size + sha256.

### Step 5 - tests  (commit `66161ba16`)
- `tools/test_setup_qfn.py` (12 -> 24 tests): the vision manifest is pinned + sha256 (with its `hub_names`);
  `qfn_vision_step` writes the live entry's keys and `--no-vision` leaves it out; the config gets the entry and
  `--vision`; the setup flow with the machine/build/downloads stepped around - drafting `off` adds `--mtp-max-t 1`
  (and keeps `--mtp`), `on` keeps the layer, `measure` (the `--yes` default) leaves drafting on with no calibration
  recorded yet; `check_parallel` accepts qfn; the parallel value written (off on one card, 4 with `--parallel 4`,
  2 on two GPUs, off with `--parallel 1`, kept from a config before); and the `--import-model` entries (both quants'
  shards, the draft layer minus LICENSE, the vision tower; every sha a real 64-hex).

## Files
- `saguaro.py`  (setup: vision build + download + entry, drafting choice, parallel, import_catalog)
- `tools/qfn/download.json`  (the pinned vision tower)
- `tools/test_setup_qfn.py`  (12 -> 24 tests)
- `REPORT-t3-setup-features.md`  (this)

## Tests run (repo root, one at a time, the three the plan names)
| module | before | after | result |
| --- | --- | --- | --- |
| `tools.test_setup_qfn` | 12 | 24 | OK |
| `tools.test_setup_download` | 2 | 2 | OK |
| `tools.test_setup_amd` | 29 | 29 | OK |

## Open questions
- The mmproj sha256/size come from the Hugging Face LFS oid of `mmproj-BF16.gguf` at the pinned revision (checked via
  the API + a content-length HEAD, not against the live hand-made file, which is not on this machine). The setup
  verifies the sha on download, so a mismatch fails loudly rather than loading a wrong tower.
- On AMD the reader is built as `vision-qfn-hip` (the GPU program). Its CPU fallback (`vision-qfn`, `cpu_exe` in the
  vision section) is not written - the live config's CPU-fallback path and the elastic `vision_reader` handling are
  the engine's, not the setup's; add `cpu_exe` (a second, non-HIP build) if AMD wants the no-GPU-runtime fallback.
- `measure` for drafting only accepts the choice (drafting on) and does not record `mtp_calibration`, so a later
  `--setup` asks again until task T4's step 4 records a measurement - intended here, but worth confirming that T4
  writes `mtp_calibration` so the question settles.
- `compile_vision_qfn`'s CUDA branch is written but the setup only offers QFN on AMD (`family_unavailable`), so it is
  exercised only once the NVIDIA build/fit task offers QFN; it mirrors the tested `compile_vision`.
- `QFN_VBUILD` (a fresh CMake folder, ~the mtmd build) is not cleaned up; vision-glm behaves the same.
