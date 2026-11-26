# REPORT-p124r-nits - small leftovers from the 1.2.4 reviews (Saguaro 1.2.4)

Branch `saguaro/q1-p124r-nits`, one commit per numbered plan step.

## Step 1 (e5f683c24) - the card-test scripts into the repo, fixed

New `tools/card_tests/park.sh` and `tools/card_tests/lost_read.sh`, from
`ref/card-test-park.sh` / `ref/card-test-lost-read.sh` with the four fixes:

* **(a) the test config drops `api_key`** - the readiness poll and the requests send none.
  park.sh took the line as the manager's hand-patch (`ref/card-test-park.mortar-patched.sh`)
  has it; lost_read.sh got the same `cfg.pop("api_key", None)`.
* **(b) the port is clean around the run** - before starting, both scripts wait (up to 60 s)
  until the port is free and stop with the holder's pids named if it is not; on exit (trap)
  they kill the server and whatever still holds the port - the server's child engine outlived
  the launcher before. Holders are found with `ss -ltnp`, `lsof` as the fallback. The kill on
  the port only ever runs once the script itself has started a server (a startup failure
  before that leaves a live server on the port alone).
* **(c) park.sh's RAM-restore check** now matches the engine's real line -
  `conversation cache: restored N tokens (checkpoint|live)` - whatever N the engine logs (it
  logs the conversation's real token count, not the requested size, and a park that kept its
  RAM restores as `(checkpoint)` as often as `(live)`; the old `restored $N tokens` grep
  printed CHECK although the log showed restores from RAM).
* **(d) each script writes only under its own `--out` folder** (default `scratch/park-card` /
  `scratch/lost-read`; a dedicated folder is demanded, `.` and `..` refused): its config copy,
  session dir, engine log, server log and request bodies all live there; the install's own
  config, log and sessions are only ever read. lost_read.sh additionally points the copy's
  `"log"` key at `<out>/engine.log` - the server sends the engine's stderr to exactly that
  path (serve/server.py:1473,1480), so leaving the live config's `"log"` in place is how the
  old script wrote into the install's own log. Its read-figure greps therefore read
  `engine.log` (the engine's `reused + N read in M ms` lines) and `serve.log` (the server's
  own progress / pause / continue lines) separately. Headers say what each checks and how to
  run it.

No test module for the shell scripts: both pass `bash -n`, and their argument handling
(`--out` validation, unknown args, missing `CFG` / missing `saguaro-*.json`) was smoke-run
safely in this checkout (all refuse cleanly, nothing started).

## Step 2 (7fdac6af0) - setup: Metal guard + the two wiring lines tested

* `choose_model`'s recorded-folder branch (saguaro.py:2290-2295) got the same Metal guard the
  `--model-dir` / `--model` branches use: a config whose installer records an EXL3 `model_dir`
  and a setup again with `--backend metal` now fail "an EXL3 quant runs on NVIDIA GPUs only"
  instead of returning `("exl3", folder)` (REVIEW-p124f finding 2). A recorded GGUF folder is
  still taken on Metal.
* Tests for the two installer->step wiring lines finding 1 names (both read correctly; the
  lines have moved since the review: 9449 -> saguaro.py:9465, 7567-7569 -> 7583-7585):
  `tools/test_setup_ik2.py::test_set_up_ik2_hands_the_recorded_file_to_its_steps` drives
  `set_up_ik2` itself (steps 2-8 patched off) and pins that `a.ik2_prev_file` carries the
  installer's `ik2_file` into `ik2_local_gguf` - and None without the record;
  `tools/test_maya_own_quants.py::test_set_up_mm3_wires_the_recorded_own_folder_to_its_steps`
  drives `set_up_mm3` to its first step after the wiring and pins that the installer's
  `own` + `model_dir` become the `own` folder (`mm3_memory_gb(quant, own)`) - and None for a
  config without the record.
* New choose_model guard test: `tools/test_maya_models.py::test_a_recorded_exl3_folder_is_not_taken_on_metal`.

## Step 3 (41da54ea7) - MiniMax: the cancelled read's real expert counters

`src/mm3/mm3_main.cpp`: the cancelled-read DONEs (solo :760-768, multi :1481-1490 - the
last-chunk STOPs of P124j and the between-chunk cancels beside them) printed a literal `0 0`
for the expert fields where the old fall-through-to-encode `finish()` reported real counters
(REVIEW-p124j finding 3). They now report the read's own routings since its start - in VRAM
and all - the way the normal DONE reports the decode's; the solo path finally uses the `s_pre`
snapshot it already took (its `(void) s_pre;` dropped), multi takes the same snapshot before
its read loop. The ERR-parity DONEs keep their zeros (the nit names the two cancel lines).
Telemetry only: the fields feed `StrataEngine._parse_done`'s hits/lookups (serve/server.py:1857).

Build-check: `mm3_main.cpp.o` compiles clean in a fresh `scratch/build-mm3-nit`
(CUDA 13.1.115, `CMAKE_CUDA_ARCHITECTURES=75`, Ninja; configure log and build log under
scratch/, zero warnings). The full link was not built (only the changed translation unit
matters; no GPU runs).

## Step 4 (9b512404d) - stale old-name texts for NEW files

saguaro.py, four texts that still said the pre-C122 family while meaning files written today:
the MiniMax Maya-GGUF note's "its config `maya-minimax-maya-xs.json`" (:688), `write_config_hy4`'s
docstring (:8869), `write_config_ik2`'s docstring (:9384) - both write through `CN.locate`, which
returns the `saguaro-` path when neither name exists - and the `--bench-smoke` refusal's hint,
whose log is `saguaro-bench.log`, the name the bench writes (:5752). Strings that mean the OLDER
files (`maya-<model>.json found too`, "before 1.2.2", the `maya-*.json` legacy lookups, the
rename tools' rules) keep their name; `tools/saguaro/*.py` grep clean - every `maya-` there is
a rename rule or a deliberate exclusion list.

## Tests run (one module at a time, /opt/llm/saguaro/.venv/bin/python tools/<module>)

| module | result |
|---|---|
| tools/test_maya_models.py | 58 OK (57 before, +1 new) |
| tools/test_maya_own_quants.py | 36 OK + 1 skip (35 before, +1 new) |
| tools/test_setup_ik2.py | 57 OK (56 before, +1 new) |
| tools/test_bench.py | 50 OK |
| mm3 build-check | `mm3_main.cpp.o` built, zero warnings (new scratch dir, sm_75) |

Steps 1 and 3 touch no Python test module; step 3's protocol (PP/DONE shapes) is unchanged,
so serve.test_priority was not named by the plan here and was not run.

## Files changed

* added: `tools/card_tests/park.sh`, `tools/card_tests/lost_read.sh` (executable)
* `saguaro.py` - Metal guard (step 2), four old-name texts (step 4)
* `tools/test_maya_models.py`, `tools/test_maya_own_quants.py`, `tools/test_setup_ik2.py` - one test each
* `src/mm3/mm3_main.cpp` - the two cancel DONEs
* this report

## Open questions

1. **lost_read.sh on an mm3 card**: the server prints the numberless "continues from where the
   engine kept it" wording for an engine that reports no reuse figure (serve/server.py:5706-5714,
   P124j review line 80), so the test's `prompt read continued: M of N` parse - and its M > 0
   gate - only pass on engines whose DONE carries `reused` (the QFN shape P124b was written
   for). Kept as the ref script intended; say so if it should learn the mm3 shape.
2. **park.sh (c)**: PASS is any restore line with the real count, `(checkpoint)` or `(live)`.
   The manager's hand-patched variant matched the requested size exactly; if the intent was
   "(checkpoint) means disk, keep FAILing it", the tag list needs a second look - the engine's
   own comment calls both restores RAM-side (generate.cpp:19938-19947).
3. **ERR-parity DONEs** (solo :754, multi :1469) still print `0 0` for the expert fields - the
   nit names only the two cancel lines. Same treatment if wanted.
4. `scratch/build-mm3-nit/` (configure + build logs, one object) left under the excluded
   scratch folder; remove it whenever.
