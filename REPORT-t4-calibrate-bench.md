# T4 report: QFN --calibrate, --speed-calibration, --bench (QFN-first, task 4 of 9)

Branch `saguaro/q1-t4-calibrate-bench`, three commits on top of `d2bdeabe6` (the T6 merge):

| commit | step |
|---|---|
| `b1120fdbc` | 1. the QFN drafter check on the config's own `--mtp` folder; `--calibrate` / `--speed-calibration` for "qfn" |
| `23ba64486` | 2. `--bench` for QFN: keep saguaro-qfn, start the engine as its server does, speed lines from QFN's own output |
| `0f1d4db4c` | 3. tests: `tools/test_qfn_calibrate.py` |

## What changed, per step

### 1. The drafter check + `--calibrate` / `--speed-calibration` for "qfn"

**New `tools/qfn_calibrate.py`** (QFN's harness, the calibrate_ds4.py shape):

- `mtp_dir(cfg)` - the drafter is the folder the **config's own `--mtp`** arg names (the Lamina layer,
  `tools/qfn/download.json`'s "mtp", e.g. `./ref/Qwen3.8-Flash-Next-UD-Q4_K_XL.json`'s
  `/opt/llm/models/Qwen3.8-Flash-Next-MTP-Lamina`), not CYBER-FROST's `<pack>/mtp/dense.txt`;
  `has_drafter(cfg)` = that folder with its `dense.txt` (the file every Lamina download carries).
- `with_drafting(cfg, on)` - on: the config's `--mtp` args as they stand; off: `--mtp-max-t 1`
  (the drafter stays loaded, windows of one token, prompt-lookup drafts only).
- `decide(off, on)` - drafts kept only when faster by more than `MIN_GAIN` = 3% (CYBER-FROST's
  MTP check's rule); a tie decodes without drafts.
- `run(cfg, say, start_engine)` - reuses the other engines' speed-calibration harness
  (`calibrate_draft.py`'s `requests`/`measure`: warm-up, fresh questions, sustained tok/s), like
  DS4/MM3's DSpark check (`calibrate_ds4.py`): the config twice, one engine start each, `MIN_TOKENS`
  1024 / `DOC_TOKENS` 512 as there; it raises when a side wrote too few tokens.
  `profile_copy` points `--expert-profile(-save)` at copies so the calibration's answers do not
  lead the next start's placement.
- `start_qfn_engine(cfg)` - starts QFN the way the server does (`serve/server.py` start_qfn ->
  `serve/qfn_engine.py`: its `engine_args` - layer split, `--batch` from `parallel`, `--pool-cores`,
  the draft settings - and `child_env`); it takes the config dict (calibrate_draft's side) or the
  args list (calibrate.py's) so both harnesses share it. `serve/qfn_engine.py` is only imported,
  never touched.

**`saguaro.py`** (CYBER-FROST's `qwen_draft_calibration` untouched; `("qwen", "qfn")` branches split):

- `calibrate_config`: "qfn" now goes to `qfn_calibrate_config` = `qfn_hardware_calibration`
  (`CALH.run` = **`tools/calibrate.py`** - the PCIe share, the draft floor and the CPU worker count,
  measured through the engine's per-request `strata_tune` exactly like the old Strata
  `setup.py --calibrate` did - via `CQ.start_qfn_engine`) followed by the drafter check; the won
  settings land in the config's args (a setting that did not win returns to the product default)
  and `installer.calibration`. "qwen" still goes to `qwen_draft_calibration` alone.
- `qfn_draft_calibration(cfg_path)`: no `--mtp` / no Lamina files / GPUs busy -> says so, nothing
  changed (None); else measures on vs off and keeps the faster (on: the config's `--mtp` args, off:
  `--mtp-max-t 1`), records `installer.mtp_calibration` (off/on/kept) and **says which it chose**
  ("the Lamina drafter is faster on this PC - 14.0 against 10.0 tokens/s (+40%)" / "... is not
  faster ...: Saguaro decodes without drafting"); a failed measurement leaves the config as it was.
- `speed_calibration`: "qfn" -> `qfn_draft_calibration`, "qwen" -> `qwen_draft_calibration`.

**Prefill-pipe / PCIe, as asked:** `engines/qfn` has the knobs (`--prefill-pipe`, `--pcie-frac`,
per-request `strata_tune` keys `pcie_frac` / `spec_min_p` / `pipe` / `aux_cpus`,
`engines/qfn/src/prefill/prefill.cpp:2482`, `program/generate.cpp:18712`) and the engine's comment
at `prefill.cpp:2494` claims "setup's `--calibrate` measures it once (tools/calibrate.py) and writes
`--prefill-pipe <r>`" - but `tools/calibrate.py` sweeps only the PCIe share, the draft floor and
the workers, **no tool measures `--prefill-pipe` itself**. The first two are wired into
`--calibrate` for qfn as described; the pipe ratio is not measured anywhere (open question).

### 2. `--bench` for QFN

- `bench_exe`: kept saguaro-qfn - `family_target("qfn")` is QFN's own target since T1, and
  `exe_for_target` returns `qfn_exe()`; a test locks it in (the old failure mode was falling
  through to "maya" = saguaro-glm).
- new `bench_start_command(cfg, tok=None)`; `bench()` and `bench_compare`'s fallback start through
  it: for "qfn" the batch driver the server uses - `serve/qfn_engine.py`'s `engine_args` (so
  `parallel` -> `--batch N`, `pool_cores` -> `--pool-cores`, the layer split, the draft settings)
  plus its `effort_end_args` (with the tokenizer at hand) and its `child_env`; every other family
  still `serve/server.py`'s `engine_args`/`child_env`.
- **speed lines from QFN's own output**: the bench numbers were always the engine's own DONE
  fields (its solo GEN path answers the same serve contract under `--batch`), and now the
  engine-lines side works for QFN too: `MB.Session.gen` records the stdout lines between the
  answers (QFN prints its `STAT tok_s=...` speed line there, the maya engines keep theirs on
  stderr), `bench()` takes the non-compare "engine lines" from the session's lines (log + stdout),
  and `speed_lines`/`REPORT_LINES` gained QFN's own patterns (`^STAT tok_s=`,
  `^saguaro serve: prompt chunk`, `^saguaro batch:`). `MB.drafting_choice` learned "qfn":
  `mtp=1` + `mtp_max=` from the engine's INFO line (falls back to the config's `--mtp`/`--mtp-max-t`).

### 3. Tests

`tools/test_qfn_calibrate.py` (new, 25 tests, fakes only - no engine, no GPU, no network): the
drafter-path lookup (own `--mtp` folder, missing folder, no `--mtp`), `with_drafting` on/off, the
profile copy, the **calibration decision on fake timings** (`CQ.run` with a fake `measure`: both
sides started, tie -> off, clear win -> on, too-few-tokens raises; `decide` boundaries), the
saguaro routing (calibrate_config / speed_calibration: qfn -> the qfn check, qwen stays
CYBER-FROST's; kept-on removes `--mtp-max-t`, kept-off adds `--mtp-max-t 1`, records + the said
choice; no drafter / failed measurement change nothing; busy GPUs are said), the hardware
calibration writing only the won settings, and the bench part: `bench_exe` keeps
`/build-hip/qfn/saguaro-qfn` and never the GLM binary, `bench_start_command` for a qfn config
carries `--serve`, `--batch 2`, `--mtp` and the driver's `child_env`, other families keep the line
driver's command, `speed_lines` picks the STAT line, `drafting_choice` reads qfn's INFO/args.

## Tests run (repo root, one at a time, before -> after)

| module | before | after |
|---|---|---|
| `tools.test_setup_qfn` | 12 tests, OK | 12 tests, OK |
| `tools.test_qfn_calibrate` (new) | - | 25 tests, OK |
| `tools.test_bench` | 33 tests, OK | 33 tests, OK |

(`tools.test_setup_bench` does not exist; `ls tools/test_*bench*` names only `tools/test_bench.py`,
so that is the bench module that ran.)

## Open questions

1. **`--prefill-pipe` is measured by nothing.** The engine reads `pipe=` per request and its
   comment points at `tools/calibrate.py`, but that tool sweeps only PCIe share / draft floor /
   workers. Should `qfn_hardware_calibration` grow a pipe sweep (only meaningful on a layer
   split)? Left out here - nothing invented.
2. **`keep_drafting`** (setup's drafting-choice code, off-limits to this task) still adds
   `<pack>/mtp` when a qfn config has no `--mtp` - the Lamina folder lives in the models folder,
   so a "choose now, skip measuring" on such a config would point at the wrong folder. Another
   task's code, deliberately untouched.
3. **`--bench --compare` against a qfn config**: the other build's engine is started through
   `MB.start_command` (its own code) or, as fallback, `bench_start_command(ocfg)` without the
   tokenizer, so an `"effort_position": "end"` config loses the `--tail-role-token` there. The
   measured side (`here`, both plain and compare) starts fully.
4. **No auto-measure at a start**: `needs_speed_calibration` still returns False for "qfn" (as
   before T1) - the drafter is measured by `--calibrate` / `--speed-calibration` when someone
   waits for it, never unattended.
5. `busy_gpus` reads `nvidia-smi`; on an AMD qfn box it finds nothing busy and the checks pass
   vacuously - the same as every other hip calibration does today.
