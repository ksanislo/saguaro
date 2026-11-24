# T1 report: QFN gets its own family "qfn" (QFN-first, task 1 of 9)

Branch `saguaro/q1-t1-identity`, five commits on top of `e0146b3e8`:

| commit | step |
|---|---|
| `bafe938c8` | 1. tools/maya_quants.py: the family "qfn" |
| `24d1a0e91` | 2. saguaro.py: the setup writes `"family": "qfn"`; the qwen branches take "qfn" too |
| `83733d0b3` | 3. serve/models.py (+ model_pull.py): family(cfg) = "qfn"; the serve paths serve it |
| `1c2b09929` | 4. engine + web: engine_kind=qfn, TIER_SETS.qfn, qfn fields |
| `a5cb36b86` | 5. tests |

## What changed, per step

### 1. tools/maya_quants.py
- `resolve()`: QFN ids (`QFN_IDS`) and the quant file names (`Qwen3.8-Flash-Next-Q8_0`,
  `-UD-Q4_K_XL`) resolve to `("qfn", <quant>)`; CYBER-FROST's ids stay `("qwen", "q5_k_m")`.
- `family_of()`: "qfn" when `installer.family == "qfn"`, or `installer.family == "qwen"` with a
  `qwen_quant` in `QFN_QUANTS` (old configs, read-time migration), or - for the live hand-made
  configs of `./ref/*.json`, whose installer names no quant - when the exe name ends in
  `saguaro-qfn` (checked both inside the "qwen"-family branch and, when the installer names no
  family at all, before the `--pack` → deepseek fallback). CYBER-FROST (exe `saguaro-glm`,
  `--ple-gguf`, `qwen_quant q5_k_m` or none) stays "qwen".
- `MODEL_DEFAULTS["qfn"]` = the values QFN gets today (`{"temperature": 1.0, "top_p": 0.95,
  "top_k": 20}, "xhigh"`); `shown()` renders `("qfn", key)` exactly as it rendered `("qwen", key)`
  with a QFN key before (the old path kept for callers of the "qwen" shape); `config_stem()`,
  `config_key()` handle "qfn" (key = `installer.qwen_quant`; the hand-made configs name none:
  None, so the label is the model's own name - before, they were shown as CYBER-FROST's `Q5_K_M`,
  which is exactly the identity bug being fixed); `FAMILIES`, `MODEL["qfn"]`, `SHOWN["qfn"]`.

### 2. saguaro.py
- `write_config_qfn` writes `installer = {"family": "qfn", "qwen_quant": ..., "engine": "saguaro-qfn"}`
  (config file name unchanged: `maya-qwen3.8-flash-next-<quant>[-hip].json`).
- `is_qfn_config(cfg)` := `config_family(cfg) == "qfn"`; `setup_family(cfg)` := `config_family(cfg)`
  (maya_quants.family_of now tells the two apart itself, old configs included).
- Branches that today also served QFN now `in ("qwen", "qfn")`: `calibrate_config` (~4135),
  `needs_speed_calibration` (~4297), `speed_calibration` (~4317), `keep_drafting` (~4529),
  `server_command`'s loading note (~4686), `bench_chat` (~5052). Same code, same strings.
- `check_parallel`: its model-name map gained `"qfn": QFN_SHOWN` - a QFN setup/config no longer
  gets refused with CYBER-FROST's name; CYBER-FROST unchanged.
- QFN-only call sites named by the "qfn" family now: `model_option` (`family == "qfn"`),
  `qfn_entry` / `qfn_quant_fit` / `choose_qfn_quant` / `qfn_files_step` / `set_up_qfn`
  (`Q.shown("qfn", quant)` - the same string), `write_config_qfn`'s `config_stem("qfn", quant)`.
- Checked and deliberately left alone: `family_target` (already maps "qfn" → saguaro-qfn; via
  `config_family` it now fixes `bench_exe`/`--bench --compare` for QFN configs, which resolved to
  the GLM engine while QFN masqueraded as "qwen" - the docstring at `bench_exe` names exactly this
  drift as #350); `FAMILIES` / `choose_family` / `set_up` / `NO_ENGINE` (already keyed "qfn");
  `check_pc`'s two `family == "qwen"` spots (~1611 min CC, ~1645 the <23 GB note) are unreachable
  for "qfn" (a "qfn" setup fails on cuda at ~1582 before them and returns into `check_hip_pc` on
  hip), left as they are; `qwen_*`, `set_up_qwen`, CYBER-FROST's rows and its installer record
  untouched.
- `tools/maya_bench.py:drafting_choice` needs nothing: "qwen" fell through to its log-line
  fallbacks, "qfn" falls through the same way.

### 3. serve
- `serve/models.py`: `family()` is maya_quants.family_of, so QFN configs read as "qfn"; its
  `choices` / `settings_of` / `apply` Qwen branches, the `QWEN_KEYS` field set, the context list
  and `TURBO_DRAFT` take "qfn" beside "qwen" - same fields, same validation, same values.
- `serve/model_pull.py`: the Models page's QFN entries carry `family "qfn"`; their `installed`
  check matches `family_of`'s "qfn" (through the read-time migration also for configs of releases
  before the split). Without this the flag would have regressed for configs the setup writes now.
- `serve/frontend.py` and `serve/server.py`: no change needed - their `QWEN_EFFORT` mapping and
  the Qwen tool-call form key on the template's features (`forced_thinking`), not the family, so
  they apply to QFN's identical Qwen3.8 template as before; serve/server.py has no engine_kind
  "qwen" string a QFN start needs (QFN was and stays engine_kind "qfn" via the batch driver,
  ~8656; the engine INFO parse at ~1176 feeds the web, covered in step 4).

### 4. engine + web
- `engines/qfn/src/program/generate.cpp` ~12432: `INFO engine_kind=qfn` (one string; the comment
  above it at ~12425 fixed too). Not built here, per the plan - it builds at the gate.
- `serve/web/monitor.js`: `TIER_SETS.qfn` = TIER_SETS.qwen, `tierSet` picks it, and the "the CPU
  computes them" tip branch takes "qfn" beside "qwen".
- `serve/web/state.js`: `familyOf` routes "qfn" to the Qwen thinking levels (low/medium/xhigh);
  the QFN model names already matched /qwen/ so nothing user-visible changes.
- `serve/web/models.js`: `qwenFields` for family "qfn" too. `serve/web/lib.js`: Qwen's logo for
  "qfn" too. `serve/web/settings.js`: `ENGINES` entry `qfn: "saguaro-qfn"`.
- `tools/setup_dash.py` ~227: INFO engine_kind "qfn" gets the Qwen path's tier rows beside
  "mm3"/"qwen".
- `serve/qfn_engine.py` untouched (never hand-edited); nothing in `engines/qfn/serve/server.py`
  needed a change, so `tools/saguaro/extract_qfn_engine.py` had nothing to regenerate.
  `engines/qfn/serve/test_perreq_stats.py` (a fake engine echoing `engine_kind=qwen`) is not one
  of the named files - left as it is; its assertions don't read the string.

### 5. tests
- `tools/test_maya_quants.py`: new `QfnIdentity` class - every QFN id resolves to ("qfn", quant),
  CYBER-FROST's ids to ("qwen", "q5_k_m"); `family_of` for the three config shapes (new "qfn",
  pre-split "qwen"+QFN-quant, and a dict copy of `ref/Qwen3.8-Flash-Next-UD-Q4_K_XL.json`);
  CYBER-FROST's shapes stay "qwen"; shown/config_stem/config_key/model_defaults unchanged in value.
- `tools/test_setup_qfn.py`: the assertions that pinned QFN as "qwen" now pin "qfn" (ids,
  family_of, the installer the setup writes, the Models page's family); new hand-made-config
  test (the ref copy: family_of/is_qfn_config/setup_family "qfn", label "Qwen3.8-Flash-Next");
  the pre-split config's migration asserted too.

## Tests (repo root, one module at a time)

| module | before (untouched branch) | after |
|---|---|---|
| tools.test_maya_quants | 20 ran, OK (skipped=1) | 24 ran, OK (skipped=1) |
| tools.test_setup_qfn | 11 ran, OK | 12 ran, OK |
| tools.test_setup_qwen | 18 ran, FAILED (failures=1) | 18 ran, FAILED (failures=1) - same failure |
| tools.test_setup_amd | 29 ran, OK | 29 ran, OK |
| serve.test_models | 51 ran, OK | 51 ran, OK |
| serve.test_dashboard | 52 ran, OK | 52 ran, OK |

The one failure, `tools.test_setup_qwen.Dashboard.test_the_web_apps_names`, is identical before
and after: it runs `serve/web/lib.js` through node and dies on `ReferenceError: navigator is not
defined` (lib.js:152) - an environment problem (no browser globals in this node run), unrelated to
this change; the line is byte-identical before/after.

## Open questions
- `check_parallel` for QFN: the refusal text now names QFN (added to the family→name map) instead
  of falling back to CYBER-FROST's name - an identity improvement, but it is a visible string
  change for a QFN config given `--parallel`. Kept; flag it if T8's owner wants the old fallback.
- `bench_exe` / `--bench --compare` for QFN configs now resolve to saguaro-qfn (through
  `family_target("qfn")`) where they resolved to the GLM engine while QFN read as "qwen". This is
  the family↔engine consistency `bench_exe`'s docstring demands (#350), not a behaviour CYBER-FROST
  depends on - but "--bench on a QFN config" was arguably broken before and now runs, so someone
  should measure it once on mangonel's QFN.
- The hand-made `./ref/*.json` configs keep `installer.family = "qwen"` on disk (read-time
  migration; the plan's choice). `Q.config_label` shows them as bare "Qwen3.8-Flash-Next" (their
  installer names no quant) - before they showed as CYBER-FROST's "Q5_K_M", which was simply
  wrong. Rewriting the live configs belongs to a later task at most.
- The C++ string is unverified until the gate builds engines/qfn; nothing in the named test
  modules greps `engine_kind=qwen` from generate.cpp (only engines/qfn/serve/test_perreq_stats.py's
  fake engine echoes it, harmlessly).
