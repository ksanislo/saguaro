# v1.2.1-D setup fixes — report (2026-11-24/25)

Carried out from `./PLAN-v121d-setup-fixes.md` against `./ref/queue-notes.txt` (items a–f + the scale-down note)
and `./ref/REPORT-p40-batch.md`. ONE COMMIT PER NUMBERED STEP, six commits. No sub-agents. Relative paths only.
Nothing under `engines/` or `serve/` was touched (no `serve/qfn_engine.py` edit, so no `extract_qfn_engine.py`
run was needed); no setup, engine or network was run. All scratch work stayed out of the tree (no `./scratch/`
files were needed).

## Commits (oldest first)

1. **`cdf28f612`** — QFN setup: the shipped 48x512 expert profile is copied only when the quant's expert count matches.
2. **`6e62a4479`** — QFN fit: a quant with fewer experts than the measured PC's RAM is not slow for thin RAM alone; the
   scale-down line no longer claims a 32K context when `--context` was given.
3. **`3b5036151`** — Headless Pascal (all-sm_6x) cards keep 1024 MiB free, not 512.
4. **`6a40f36aa`** — The `--yes` skips of the picture reader and the draft layer name `--download-model`.
5. **`59bf92816`** — `--gguf-dir` accepts a merged single-file GGUF for a quant pinned as shards.
6. **`16a944220`** — Pascal and DS4 EXL3 on AMD: what is measured and what is not, never an experimental label.

## What changed, per plan item

### 1 — Expert profile (`write_config_qfn`)
- New `qfn_profile_shape(path)` reads a profile header (`"STRP"`, u32 version, u32 layers, u32 experts — the shape
  `engines/qfn/src/core/expert_cache.cpp` checks and refuses with "a profile for a different artifact").
- New `qfn_expert_shape(pack)` reads the quant's own `(layers, experts a layer)` from the pack's `index.txt`:
  `blk.0.ffn_gate_inp.weight`'s router rows (ne1) are the experts kept and the `blk.N.` rows give the layers — the
  Coder's 256 is **read, not hard-coded** (`iq_pack.py`: "router rows = experts kept - pruned models ship < 512").
- `write_config_qfn` copies `data/expert-profile-qfn.bin` into `var/` only when its shape equals the quant's; the
  `--expert-profile` / `--expert-profile-save` pair is written only for a matching file. With no matching profile the
  pair stays out and the setup says the engine starts and learns on its own; **unless the layout pins a RAM tier**
  (`lay["resident_budget_gib"] is not None`, the low-RAM complement layout that REQUIRES a static profile) — then it
  stops with a plain line naming the file, the shape it needs, what the shipped one is, and how to make one
  (`tools/make_profile.py --no-base --n-expert <N> --out <profile>`, per that tool's own "--n-expert 256 (the Qwen
  Coder keeps 256 of 512)"). A stale 48x512 copy already in `var/` (Paul's box) is left in place and never read.
- A pack whose `index.txt` does not read keeps the pre-(f) rule exactly (the copy + wiring), so nothing changes for a
  quant the setup can price the normal way.

### 2 — Scale-down (`tools/qfn_fit.py`, `set_up_qfn`)
- `comfortable(cards, ram_gb, quant=None)` — the RAM side is now quant-aware. Below `MIN_RAM_GB` a PC is slow for the
  RAM only when the quant's experts exceed the measured PC's whole RAM (UD-Q4_K_XL 104 GiB, Q8_0 175, CYBER-FROST
  105): the streaming the scale-down was tuned on. A smaller quant — GSQ IQ3_S (77.9 GiB), the Coder (54.4) — gets the
  layout as laid out, on the measured point: the 24 GB / 31 GB P40 box answered IQ3_S at ~15-16 tok/s at 262144
  **with drafting** (REPORT-p40-batch item 3), the pace of the measured Q8_0 box, so drafting stays on and the context
  / K/V window are not scaled down. No quant named, or an unread manifest: no waiver (rule as before). The card side
  (small cards) and every refusal under the true minimum are unchanged, so the v1.2.1-B matrix's other points stand.
- `plan()` passes the quant to `comfortable()`.
- `set_up_qfn`: the step-2 scale-down line drops the "a 32K context instead of 64K" entry when `--context` named a
  context (the setup writes that one) and says the named context is kept, not scaled down; the drafting and
  `--kv-resident` entries are unchanged.

### 3 — VRAM reserve (`gpu_reserve_mib` / `gpu_reserve_why`)
- New `gpu_sm6x(gpus)` (every card compute capability 6.x, AMD archs read as not-Pascal).
- A headless card keeps **1024 MiB instead of 512 when every card is sm_6x** (P40 batch: 512 fell to 264-302 MiB free
  in the warm-up graph captures; 1024 held above 760). The display rule is unchanged (a desktop card keeps 3072; a
  mixed list keeps 512), the Linux AMD APU 1024 unchanged, and `--vram-reserve-mib` still wins (it is read before the
  rule in `write_config_qfn`). The reserve `ok` line (`gpu_reserve_why`) carries the measured reason.

### 4 — `--yes` warnings (`qfn_mtp_step`, `qfn_vision_step`)
- Both warnings now name the flag: "drafting off: the draft layer was not downloaded (--yes answers the download
  question no; --download-model answers it yes)" and the same shape for the vision tower, so the `--yes` skip of the
  picture reader and the draft layer says what would have said yes.

### 5 — Merged single-file GGUF (`qfn_files_step`)
- New `gguf_tensor_slice(path, at, size)` (sha256 of one tensor's bytes, read at `data_start + offset`) and
  `qfn_merged_gguf(...)`. `--gguf-dir` holding ONE non-pinned `.gguf` (`gguf-split --merge`, an Ollama blob hard link)
  is now taken for a multi-shard quant when its tensor **names, shapes and types** are the pinned shards' index — and,
  where that index is on the PC (all pinned shards whole in the folder, in the folder a written config read them from
  `installer.files_dir`, or in the model folder), when the **bytes of a sample of tensors — the first, the last and 8
  spread between** — match the shards', tensor for tensor. Where the index is not on disk there is nothing to compare
  bytes against: the file is accepted on its own complete tensor directory (names, shapes, types, and the total tensor
  bytes its directory says reaching the end of the file — `incomplete()`) and said so plainly with
  **"not sha256-checked"**. A candidate that differs is refused with exactly what differed (missing/unknown tensor,
  a shape/type mismatch, or the first sampled tensor whose bytes differ). Accepted files then flow into the normal
  pack / drafter / vision steps (the packer and the engine read a single file the way they read the first shard).

### 6 — Wording (no experimental / not-validated labels)
- `check_pc`: the per-GPU line is now `can be used (Pascal: second class)`; the pre-Volta warning says second class
  and lists what was measured on a Tesla P40 (Inkling 2 Mini 8K reads ~384-458 tok/s; saguaro-qfn ~4-20 tok/s by quant,
  GSQ IQ3_S ~15-16 with drafting) and that the other engines are "not measured on a Pascal card yet". The none-usable
  refusal, the module header and the `VALIDATED_CC` / `arch_setting` docstrings follow.
- `tools/ik2_fit.unsupported` (Inkling's older-than-Pascal refusal line): "Pascal, second class: measured on a Tesla
  P40" instead of "Pascal, experimental".
- `check_pc` (DS4, `--backend hip`): the EXL3-on-AMD line drops "experimental" / "not validated" and states the
  measurement instead — the kernels pass their tests on AMD GPUs, the whole model "has not been measured on one yet",
  and the GGUF quants (`deepseek-maya-s` / `-xs`) "are the ones measured on AMD". The GGUFs still go through with no
  line of their own.

## Files

- `saguaro.py` — steps 1, 2 (set_up message), 3, 4, 5, 6.
- `tools/qfn_fit.py` — step 2 (`comfortable`, `plan`, the `MIN_RAM_GB` note).
- `tools/ik2_fit.py` — step 6 (one refusal line).
- `tools/test_qfn_fit.py`, `tools/test_setup_qfn.py`, `tools/test_setup_ds4.py`, `tools/test_setup_amd.py` — tests.

## Tests run (the six the plan names, one at a time, `/opt/llm/saguaro/.venv/bin/python -m unittest`)

| module | result |
|---|---|
| `tools.test_qfn_fit` | OK — 44 tests (was 39; +5 in `P40MeasurementPoint`) |
| `tools.test_setup_qfn` | OK — 98 tests, 1 skip (was 82, 1 skip; +6 `ExpertProfileShapes`, +6 `MergedGguf`, +2 `YesAloneNamesDownloadModel`, +1 headless-P40 reserve, +1 `--context` scale-down line) |
| `tools.test_setup_ds4` | OK — 81 tests (was 81: `test_exl3_on_amd` failed at baseline on the old label; now asserts the measurement line and no "experimental") |
| `tools.test_setup_amd` | OK — 35 tests (was 34; +1 `Reserve.test_headless_pascal_keeps_1024`) |
| `tools/test_setup_ik2` | OK — 54 tests |
| `tools.test_maya_quants` | OK — 25 tests, 1 skip |

Baseline before any change: `test_setup_ds4.test_exl3_on_amd` already failed (the pre-existing EXL3-on-AMD label the
queue note names); the other five modules passed. After: all six pass. New `MergedGguf` tests build real little GGUF
v3 files (`fake_gguf`) so the whole header/tensor-directory path is exercised, not stubbed.

## Open questions / notes for Ken

1. **The waiver line for item 2 is the measured PC's RAM (88 GiB).** IQ3_S (77.9) sits under it and CYBER-FROST
   (105) over it — the two are 15 GiB apart, so a re-quant whose experts land between ~80 and ~105 GiB would flip
   verdict on a thin-RAM PC. If a future quant lands in that band it should be measured on the P40 before its row is
   trusted; the constant is `MIN_RAM_GB`, deliberately the same figure the warning text cites.
2. **Item 1's stop condition is `resident_budget_gib is not None`** as the "low-RAM layout REQUIRES a static profile"
   proxy. The engine's own quoted error is for `--resident-cpu-experts` (`generate.cpp:3451`, the resident complement
   the setup's tier lays out); if that requirement is narrower than "a tier was written", the stop is slightly broad.
   It never writes a config that cannot start, which was the bug.
3. **No shipped Coder profile was created.** The notes offer "copy only when it matches (or ship a Coder profile)"; a
   `data/expert-profile-coder.bin` needs a Coder routing trace (`--dump-routing`) or Strata's own — out of this
   setup-only branch's reach (no engine, no network). The stop line points at `tools/make_profile.py --n-expert 256`.
4. **Item 5's "index on disk" is the set of pinned shard files**, located in `--gguf-dir`, the config's old
   `installer.files_dir`, or the model folder. An Ollama layout that keeps the shards as unnamed `sha256:*` blobs is
   not matched as an index, so its merged file falls to the weaker "not sha256-checked" acceptance (still
   self-consistent by `incomplete()`). Optionally pinning unsloth's Nov-4 UD-Q4 re-upload (queue note (e),
   `rev d9620119`, KL vs October) was NOT done — that is a manifest/network change and the plan forbids the network.
5. **Item 6 DS4:** kept one measurement-based warning and updated `test_exl3_on_amd` to it, rather than deleting the
   warning to match the old "no warning at all" assertion — item 6 says "state what was measured or say 'not measured
   yet'". If the intent was a silent AMD path (as GLM's/MiniMax's EXL3), the warn block can be dropped and the test
   reverted to `said == []`.
6. **Left for other tasks (per the plan's Don'ts):** the `vram_elastic` / pictures low-RAM workaround (queue note a /
   item d) and the `RAM_SPARE_GB` 11-GiB-on-a-31-GB-box tier question (queue note 21:35) — `RAM_SPARE_GB` is untouched.
