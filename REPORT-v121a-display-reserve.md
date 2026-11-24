# 1.2.1-A report: keep VRAM free for the desktop on a display GPU - every engine, NVIDIA and AMD

## What changed

### Step 1 - setup.py: the NVIDIA display fact
`gpus()` now asks `nvidia-smi --query-gpu=...,display_active` as a 7th column. `Enabled`/`Disabled` become the
card's `"display": True/False`, exactly like the AMD sysfs facts; `[N/A]`, `[Not Supported]`, or no field is
unknown - no key - and counts as a card that may drive the desktop. An `nvidia-smi` old enough not to know the
field fails the whole query, so `gpus()` falls back to the query without it (a failed query never loses cards).
Row parsing keeps the old shape: six fields parse as before, seven carry the display column, a GPU name with a
comma is skipped as it was.

### Step 2 - saguaro.py: one rule for both vendors
`hip_reserve_mib` / `hip_reserve_why` / `hip_headless` are renamed and generalised to `gpu_reserve_mib` /
`gpu_reserve_why` / `gpu_headless`. The rule is unchanged in shape - 3072 MiB where the desktop may run on one of
the cards, 1024 on a Linux AMD APU (the APU case kept verbatim, `hip_unified_memory` as it is), 512 on cards with
no display - and now reads the vendor's own fact: AMD's sysfs connectors + VRAM-in-use, NVIDIA's `display_active`
(which says it outright, so Windows NVIDIA cards get the headless 512 too; Windows AMD cards have no sysfs facts
and keep 3072). The old `not WIN` guard on the headless branch is gone: the facts decide. A card that says
`display: False` without a VRAM-used number (NVIDIA) is headless; AMD's sysfs keeps its "next to no VRAM in use"
condition.

Every engine's config writer that sets a VRAM reserve now uses the rule, on NVIDIA as on AMD, always honouring
`--vram-reserve-mib` and the config before's own value, and the setup's ok() line says which case and why:

- **GLM** (`write_config`): HIP keeps `SAGUARO_GLM_RESERVE_MB` (unchanged); NVIDIA configs now write
  `--vram-reserve-mib` from the rule (the engine's default 700 was gap 2's "engine's default margin"); a
  `--vram-reserve-mib`, an `--env SAGUARO_GLM_RESERVE_MB=`, or the config before's flag all come first.
- **DeepSeek** (`write_config_ds4`): `--reserve-mib` from the rule on both vendors (was HIP-only; NVIDIA had the
  engine's own 600); the user's value and a kept config value win.
- **MiniMax** (`write_config_mm3`): `--reserve-gb` from the rule on both vendors (was HIP-only) - and
  `--vram-reserve-mib`, which the MM3 writer ignored before, is honoured (written as its GB value).
- **QFN** (`write_config_qfn`): `--vram-reserve-mib` from the rule (3072 / 512 / the user's value) instead of a
  hard 700, on every vendor, with an ok() line; the 700 stands only where nothing is known about the cards.
- **`migrate_cyber_frost`**: keeps the old config's `--vram-reserve-mib` if it had one, else the rule on this
  PC's cards (new `gpu_facts(cfg)` looks the config's vendor and card indices up via `S.gpus()` /
  `S.amd_gpus()`; a PC that cannot be asked keeps the desktop's 3072). It runs once per migrating config.
- **HY4 / IK2 - checked, no writer to change**: `write_config_hy4` and `write_config_ik2` write no VRAM reserve
  of their own. Hy4's `--reserve-gb 0.5` is the *measured four-T4 layout's* constant (`tools/hy4_fit.py`
  `T4_ARGS`, written only for exactly four equal sub-22 GB cards, which are headless anyway - a headless T4 box
  gets what it gets today); Inkling's `SAGUARO_IK2_RESERVE_MB=1024` is the AMD small-card layout constant
  (`tools/ik2_fit.py`). Neither is a config-writer default the rule could feed; both are left alone.
- No engine C++ changed. `engines/qfn/serve/server.py` and `serve/qfn_engine.py` were not touched (their 700s are
  only the vision-fallback and the note for configs with no reserve at all - which a fresh config now always
  has), so `tools/saguaro/extract_qfn_engine.py` did not need to run.

### What a headless multi-T4 box gets (the plan asked to say so)
QFN's written config changes **700 -> 512** on the headless T4 boxes (their `display_active` is `Disabled`) - the
plan's one named exception; the split, contexts, env and the live-config parity tests are otherwise untouched
(`--vram-reserve-mib` is listed among LiveT4Parity's intended differences from the live ref file, which still
holds its setup-time 700). Note the intended consequence of closing gap 2: an NVIDIA card also gets a written
reserve for GLM / DeepSeek / MiniMax now (512 headless, 3072 driving the desktop) where those engines used their
own defaults (700 / 600 / 1 GB + 1 GB) - that is the fix, not a T4-only change.

## Files
- `setup.py` - the NVIDIA display fact.
- `saguaro.py` - the rule (renamed), its five users, and `gpu_facts`.
- `tools/test_setup_amd.py`, `tools/test_setup_ds4.py`, `tools/test_setup_mm3.py`, `tools/test_setup_qfn.py` -
  tests (step 3 + the mechanical consequences of the rename and the new values).
- `tools/test_maya_quants.py` - one patched name (`hip_reserve_mib` -> `gpu_reserve_mib`); not run (below).
- `REPORT-v121a-display-reserve.md` - this report.

## Tests (run with /opt/llm/saguaro/.venv/bin/python -m unittest, one module at a time)
- `tools.test_setup_qfn` - **Ran 55, OK (1 skipped)**. The skip is the pre-existing
  `test_the_ref_copy_still_matches_the_live_file_when_it_is_here` (needs `ref/live-ud-q4.json`, not in this
  worktree). New `ReserveRule` (NVIDIA display/unknown/headless/split-mixed/user value; AMD sysfs cases and the
  Linux APU) and `Migration.test_the_reserve_is_its_own_or_the_rule`; the T4 fixtures now carry
  `display: False` and the parity tests assert the written 512; the Migration class patches `S.gpus` /
  `S.amd_gpus` to stay GPU-free.
- `tools.test_setup_amd` - **Ran 34, OK**. `Reserve` renamed to `gpu_reserve_mib` plus the new
  `test_nvidia_display_active` (Disabled headless on Linux and Windows, Enabled and unknown keep 3072; a mixed
  box keeps the desktop's room).
- `tools.test_setup_ds4` - **Ran 81, 1 failure - pre-existing and unrelated**:
  `Quant35.test_exl3_on_amd` fails identically at the untouched HEAD baseline (it expects the experimental
  EXL3-on-AMD warning to be gone, but `saguaro.py` still warns; a leftover of the "Drop the experimental mark"
  commit, nothing to do with reserves). The reserve test is now `Config.test_reserve_mib`: AMD display /
  headless / APU / user / kept, NVIDIA display / headless / unknown (Windows too) / user / kept.
- `tools.test_setup_mm3` - **Ran 31, OK**. New `Reserve`: NVIDIA and AMD written `--reserve-gb` per case, the
  `--vram-reserve-mib` override, a kept `--reserve-gb`, the Linux-APU GB value.
- `py_compile` over the changed files. GLM's config writer is exercised only by `tools/test_maya_*` modules,
  which the plan does not list (and `tools/test_maya_hip` fails at the HEAD baseline too, so its numbers would
  not compare cleanly); its HIP path is text-unchanged apart from the rule's name, and its NVIDIA line is covered
  by reading the diff against the DS4 test's behaviour.

## Open questions
1. **NVIDIA unified-memory cards** (DGX Spark / GB10, `display_active [Not Supported]`) get the desktop's 3072 -
   the rule's APU case is still AMD-only (`hip_unified_memory`). Should `unified_memory` NVIDIA cards join the
   1024 case, or is 3072 right there?
2. **QFN keeps no user reserve across a setup again**: a hand-set `--vram-reserve-mib` in an existing QFN config
   is replaced by the rule / the flag given now (as the hard 700 was; GLM, DeepSeek and MiniMax keep theirs).
   Wanted for QFN too?
3. The pre-existing `tools.test_setup_ds4.Quant35.test_exl3_on_amd` failure (warning text vs the v1.1.53 drop)
   - fix in this branch, or its own change?
4. `test_maya_quants` / `test_maya_hip` were not run (outside the plan's list); only one patched name changed
   there - worth one full-suite run on the branch's next pass.
