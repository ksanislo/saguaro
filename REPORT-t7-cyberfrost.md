# T7 report: CYBER-FROST runs on the QFN engine (QFN-first, task 7 of 9)

Branch `saguaro/q1-t7-cyberfrost`, five commits on top of `d2bdeabe6`:

| commit | step |
|---|---|
| `8204404b5` | 1. tools/qfn/download.json: CYBER-FROST as a QFN quant |
| `bb8e46edc` | 2. the QFN setup installs it (pack, drafter, pictures) |
| `d52e6886c` | 3. ids and the Models page |
| `b7e25427d` | 4. migration of old-engine configs |
| `c6f9e6471` | 5. tests |

## What changed, per step

### 1. tools/qfn/download.json
- New quant `cyber-frost-q5_k_m`: `"name": "CYBER-FROST-3.8-PS"` (the model folder, as the qwen setup's),
  `"repo": "peasantsmith/CYBER-FROST-3.8-PS-GUFF"`, `"revision": 32867bb9…`, the one file
  `CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf [112534603328, 1c03787f…ca9c5]` - the repo, revision, size and sha256
  copied byte for byte from `tools/qwen/download.json` (checked with grep, both files). Its `about` names the
  in-file NextN block (blk.48 of block_count 49, left out of the pack by `engines/qfn/tools/iq_pack.py`,
  commit `db2e65562`). It carries its own `repo`/`revision` because it is not in unsloth's repo; the top `_`
  line says so.
- `needs`: `{"vram_gb": 15, "ram_gb": 88}` - taken from UD-Q4_K_XL's measured numbers (the file is the same
  size class, 112.5 GB vs 111.3 GB). No `measured` line, so the setup and the Models page say "not measured on
  AMD yet".

### 2. The QFN setup can install it (saguaro.py, tools/maya_quants.py)
- `maya_quants`: `CYBER_FROST_QUANT = "cyber-frost-q5_k_m"`, added to `QFN_QUANTS` as "CYBER-FROST-3.8
  Q5_K_M" - so it appears in the quant question (`choose_qfn_quant` iterates `QFN_QUANTS`), `shown()` renders
  "Qwen3.8-Flash-Next · CYBER-FROST-3.8 Q5_K_M (saguaro-qfn)", `QFN_QUANTS`' new key does not disturb
  `family_of` (old configs' `qwen_quant` stays `"q5_k_m"`). `config_stem("qfn", key)` special-cases it:
  **"maya-cyber-frost"** - the config keeps the file name it always had (installs and run scripts point at
  it; also what the migration in step 4 rewrites in place).
- `qfn_entry()`: a quant may name its own `repo`/`revision` (CYBER-FROST's); unsloth's quants are unchanged.
- Pack: `qfn_pack_step` already runs `engines/qfn/tools/iq_pack.py`, which since `db2e65562` leaves
  blk.48+ out of the pack - it takes the one CYBER-FROST file as it takes the sharded quants.
- Drafter (step 7 for this quant): `qfn_drafter_step` builds it from the file's own NextN block with
  `tools/mtp_from_gguf.py` into `<pack>/mtp`. The folder layout was checked against the engine's `--mtp`
  loader (`engines/qfn/src/core/mtp.cpp` ~287-335: `dense.txt` index lines, `dense.bin`, `experts.bin`) -
  exactly what `mtp_from_gguf.py` writes (`draft_vocab.bin` is an optional extra, `tools/draft_vocab.py`).
  Kept while `QFN_DRAFTER_TOOLS` (`tools/mtp_from_gguf.py`, `tools/mtp_rt.py`) and the model file stand
  (sources.json stamp in `<pack>/mtp`), same pattern as `qwen_pack_step`.
- Pictures: `qfn_vision_step` downloads the base model's mmproj - the file the "qwen" setup downloads
  (`qwen_vision_model()`, tools/qwen/download.json's "vision") - into `<model folder>/vision`, and writes
  the config's `vision` entry **the way the live QFN config does** (`./ref/Qwen3.8-Flash-Next-UD-Q4_K_XL.json`):
  exe `vision-qfn` beside the engine, `mmproj`, `model`, `gpu`, `max_tokens` 1024, `elastic` + the three
  elastic keys. `write_config_qfn` gained a `vision` parameter (writes the entry and the engine's
  `--vision`; the QFN server would add the flag itself too - serve/qfn_engine.py ~3240, untouched).
  **This branch has no code to compile `vision-qfn`** (the source is there, `engines/qfn/tools/vision`,
  output name `vision-qfn`, but `saguaro.py` only builds `tools/vision` → `vision-glm`): noted under Open
  questions.
- `set_up_qfn`: for this quant step 7 is "the MTP drafter (from the file's NextN block) and pictures"; the
  Lamina download stays the other quants' step 7. `qfn_entry(quant)["name"]` = "CYBER-FROST-3.8-PS" makes
  the model folder `models/CYBER-FROST-3.8-PS` - the folder the old install used - and its var/ folder
  beside it. `qfn_about` gained "(CYBER-FROST-3.8-PS is offered as a quant of its own, in the quant list)".

### 3. Ids
- `maya_quants.resolve`: `QWEN_IDS` (all five: cyber-frost, -3.8, -3.8-ps, -3.8-ps-q5_k_m, -q5_k_m) now
  resolve to `("qfn", "cyber-frost-q5_k_m")` - every old id still works, none removed. `model_option`'s
  existing "qfn" branch writes `a.model = QFN, a.qfn_quant = cyber-frost-q5_k_m`, so
  `--model cyber-frost` sets up on QFN, and `choose_family` answers "qfn".
- `serve/model_pull.catalog`: the `"cyber-frost"` row now points at the QFN quant - family `"qfn"`, name
  `Q.shown("qfn", cyber-frost-q5_k_m)`, `size_gb` 112.5, `needs`/`fit` from `qfn_fit` with a leading
  "CYBER-FROST-3.8, Blackfrost-AI's fine-tune of Qwen3.8-Flash-Next, on its own engine saguaro-qfn". The
  qfn loop skips the cyber key so it is offered once; `installed` is true for the migrated config
  `("qfn", "cyber-frost-q5_k_m")` and for an old-engine one `("qwen", "q5_k_m")` not migrated yet.

### 4. Migration (saguaro.py)
- `migrate_cyber_frost(cfg) -> bool`: a config whose installer says family `"qwen"` with the `"q5_k_m"`
  files and that still reads as family `"qwen"` is rewritten to the QFN config of the same file -
  `--pack`/`--native` (the same GGUF), the pack's `--expert-profile` (now saved into), `--mtp` pointing at
  the same `<pack>/mtp` drafter (its layout is what QFN reads), `--vision` only when a `vision` entry is
  there; the QFN engine's measured settings added (`--spec 3 --spec-min-p 0.7`, `--kv int8`,
  `--kv-resident`, `--vram-reserve-mib 700`, `--resident-budget-gib` from `qfn_ram_budget`, YaRN,
  `--session-dir <model_dir>/var/sessions`); exe `<this backend's build>/qfn/saguaro-qfn` (chosen by the
  config's own `backend`, not by any global), cwd `engines/qfn`; a `vision`/`vision_off` entry's exe
  `vision-glm*` becomes `vision-qfn`; env keeps what it had and gains QFN's three defaults; the installer
  record becomes `{"family": "qfn", "qwen_quant": "cyber-frost-q5_k_m", "engine": "saguaro-qfn", "quant":
  "CYBER-FROST-3.8 Q5_K_M", "files_dir": …, "migrated": …}`. Context, port, host, sampling,
  reasoning_effort, model_name, tokenizer and gpu stay as they were. Idempotent: the rewritten config
  fails every condition (asserted in the tests).
- Where it runs: `server_command` - the start/update path where `fix_config_names` and `migrate_env_names`
  already rewrite configs once (the plan's "where the setup refreshes configs on update"; `setup.py` is
  the GPU/toolkit helper library, `refresh_engine` only rebuilds engines). After the rewrite it says so
  and **stops with "run saguaro.py --setup once"**: the old config's pack was written by `tools/iq_pack.py`
  for the GLM engine and is not the pack saguaro-qfn reads (both write "index.txt v3", the FORM tables and
  dense.bin conversions differ) - failing here beats starting an engine that misreads it. `--setup`
  rebuilds it: `qfn_pack_step`'s `QFN_PACK_TOOLS` source-hash check sees the foreign stamp and repacks,
  the drafter step builds `<pack>/mtp` with its own tools stamp. And `main()`'s setup path migrates the
  `prev` config the same way, so `--setup` on an old install goes on as family "qfn" with the old answers
  as defaults (quant question defaults to the quant it had, context to its `--max-context`, models and
  model folder to its installer's).

### 5. Tests
- `tools/test_setup_qfn.py`: id resolution (every QWEN id → qfn/cyber quant), `--model cyber-frost` →
  qfn_quant + family qfn, the manifest entry checked **against tools/qwen/download.json itself** (repo,
  revision, size, sha256 equal; pinned file URL), the written config (`maya-cyber-frost-hip.json`,
  `--mtp <pack>/mtp`, `--vision` + the live QFN configs' vision shape, installer qfn/cyber quant), the
  drafter step (kept when unchanged, rebuilt through mtp_from_gguf.py when the tools moved), the vision
  step (off/declined/entry), the Models page (family qfn, saguaro-qfn needs, listed once, fit at 60 GB
  RAM), and `Migration`: the sample old config dict keeps its files/context/port/sampling, reads as qfn
  under the same file name, and migrates exactly once; nothing else (QFN's own configs, other families,
  installer-less ones) is touched.
- `tools/test_maya_quants.py`: `QfnIdentity` follows the new resolution; `test_cyber_frost_stays_qwen`
  became `test_cyber_frost_stays_qwen_until_migrated` (the old-engine config shapes still read as "qwen" -
  that is what makes the migration's predicate work); the naming block checks shown/config_label/
  config_stem for the cyber quant ("maya-cyber-frost" on both families).
- `tools/test_setup_qwen.py`: two assertions named the OLD engine for cyber-frost and were updated -
  1. `Names.test_ids`: every cyber id resolved to `("qwen", "q5_k_m")` → now `("qfn", "cyber-frost-q5_k_m")`;
  2. `Names.test_model_option`: `--model cyber-frost` set `a.model == maya.QWEN` → now `maya.QFN` +
     `a.qfn_quant == "cyber-frost-q5_k_m"`. No other test_setup_qwen assertion changed: the "qwen" family,
     its picker row, its setup steps, its models-page paths and its vision step are the old engine's and
     stay (their configs still read as "qwen" until migrated).

## Tests (repo root, one module at a time, `/opt/llm/saguaro/.venv/bin/python -m unittest …`)

| module | before (untouched branch) | after |
|---|---|---|
| tools.test_setup_qfn | 12 ran, OK | **18 ran, OK** |
| tools.test_maya_quants | 24 ran, OK (skipped=1) | 24 ran, OK (skipped=1) |
| tools.test_setup_qwen | 18 ran, FAILED (failures=1) | 18 ran, FAILED (failures=1) - same failure |

The one `test_setup_qwen` failure, `Dashboard.test_the_web_apps_names`, is identical before and after
this branch: it runs `serve/web/lib.js` through node and dies on `ReferenceError: navigator is not
defined` (lib.js:152) - an environment problem, unrelated to this change (same finding as
REPORT-t1-identity.md).

## Open questions
- **vision-qfn is not built by this Saguaro yet.** The vision entry the setup writes points at
  `<build>/qfn/vision-qfn` (beside the engine, as the live install keeps it at `bin/`), and its shape is
  the live config's (`./ref/`) - but nothing in this branch compiles `engines/qfn/tools/vision`. Pictures
  therefore need the other task's build step; the QFN server fails only picture requests, not the start.
  The migration rewrites an existing `vision-glm*` exe to that path - harmless while the program is not
  there, wrong-looking if the owner wants the old GLM reader kept instead.
- **The one-time start stop.** After a migration, the first start deliberately refuses with "run saguaro.py
  --setup once" until the QFN pack/drafter are built (the two engines' pack formats differ behind the same
  "index.txt v3" header). Kept because starting anyway would feed saguaro-qfn a pack of the other engine.
  If the owner prefers an automatic rebuild at that point, that is a follow-up.
- **NVIDIA.** CYBER-FROST ran on NVIDIA through the old path; the QFN setup offers the family on AMD only
  (`NO_ENGINE["cuda"]["qfn"]`, untouched here), so `--model cyber-frost` on a CUDA PC is refused with the
  QFN line ("on NVIDIA it is set up by hand"), and the Models page's `fit` says "the setup offers it on AMD
  GPUs only". The old engine's `set_up_qwen` and the picker's "qwen" row are still in place (the plan says
  a later task removes them), so the NVIDIA route only returns with that task's decision.
- **needs {15 GB VRAM, 88 GB RAM}** for the cyber quant are borrowed from UD-Q4_K_XL's measurement, and
  there is no `measured` speed line yet; both say "not measured" where shown. Correct them when the file
  has run on the reference PC.
- The migrated config's env keeps a `SAGUARO_EXPERT_RAM_GB` a dashboard setting wrote for the GLM engine
  (harmless: saguaro-qfn ignores it; its RAM tier is `--resident-budget-gib`). Dropping such keys could
  also drop hand-set SAGUARO_* settings, so nothing is dropped.
- The model question's QFN row still says "Not CYBER-FROST: the model as Qwen released it" - true of the
  default quant and the row's about-text, with the new "(CYBER-FROST-3.8-PS is offered as a quant of its
  own, in the quant list)" appended; if the wording reads oddly the owner may want it rephrased.
- `serve/qfn_engine.py` was not touched and did not need regenerating (nothing in
  `engines/qfn/serve/server.py` changed); the engine C++ was not touched either.
