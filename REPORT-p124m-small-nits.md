# P124m - three small leftovers (Saguaro 1.2.3, small; one commit each)

Two of the three turned out to be already fixed on this branch (the plan describes the pre-Nov-24-13:58 state):
step 1's label fix is a3cab0d93 and step 3's free fix is e3d707bea, nine minutes later. What was missing was the
tests the plan asks for, the single-source plumbing for the version strings, and step 3's sm_75 build-check.
Python 3.13.5 (/opt/llm/saguaro/.venv/bin/python). serve/qfn_engine.py untouched, so extract_qfn_engine.py was
not run. No GPU was used anywhere.

## 1. --bench names the drafter the config loads (tools/maya_bench.py, tested in tools/test_bench.py)

The rule (tools/maya_bench.py:155-165, `drafting_choice`, the `family == "qfn"` arm; the fix itself is a3cab0d93,
already an ancestor of this branch): which drafter the config loads is the folder its `--mtp` names - CYBER-FROST's
own drafter sits beside its pack as `<pack>/mtp` (saguaro.py:8086-8110 `qfn_drafter_step`, setup step 8 for
`Q.CYBER_FROST_QUANT` at saguaro.py:8558-8560), and it is called "MTP (from the model's NextN block)"; every other
QFN config's `--mtp` is the Lamina layer folder `Qwen3.8-Flash-Next-MTP-Lamina` (saguaro.py:7909 `qfn_mtp_step`)
and stays "MTP (Lamina)". A `--mtp-max-t 1` config still says "MTP drafts off (--mtp-max-t 1)" either way.

New tests (the fix had none): tools/test_bench.py:247 `test_drafting_choice_qfn_names_the_drafter_the_config_loads`
- the CYBER-FROST shape (`--pack <p> --mtp <p>/mtp`) and the Lamina shape each through the config's args alone and
through an INFO `mtp=1` line: both cases, as the plan asks. `tools.test_bench` 40 OK (was 39).

Commit: 129e0e3e8 (tools/test_bench.py only).

## 2. The --bench header and /health get their version from saguaro_version.py

Before, both read the VERSION file themselves in their own way: saguaro.py's `maya_version` (the `version` that
--bench passes to `bench_text`'s header line "Saguaro v<version> benchmark, ..." at saguaro.py:5789 and to its
records) read `HERE / "VERSION"`, and serve/server.py read `ROOT / "VERSION"` into `SAGUARO_VERSION` (:7965 is
/health's `"version":`, and the report line and the page's cache-bust use it too).

- saguaro_version.py:13 `installed_version()` - the VERSION file's line (1.2.3), or None without one: the new one
  place that reads it. The module docstring says who asks it now.
- saguaro.py:89-90 imports it; `maya_version` (saguaro.py:9534-9536) is now `installed_version(HERE) or "?"`;
  the --compare "other build" line (saguaro.py:5848) reads the other checkout through `installed_version(other)`
  too.
- serve/server.py:199-202: `SAGUARO_VERSION = installed_version(ROOT)` - the try/that-reads-VERSION block is gone.

Tests, one assertion each that the printed version equals the VERSION file:
- tools/test_bench.py:238 `test_bench_header_names_the_version_file` - `maya_version()` equals the file, and the
  header line is `Saguaro v<file> benchmark`.
- serve/test_starting.py:293 `test_the_health_version_is_the_version_file` - /health's `version` equals the file.

`tools.test_bench` 41 OK; `serve.test_starting` 24 OK (33.6 s, the module's child-process starts included).
serve/test_dashboard.py was not run (the plan names no module for /health beyond the one touched; its SAGUARO_VERSION
assertions read the same value, only now obtained through installed_version).

Commit: 3399ecb70 (saguaro.py, saguaro_version.py, serve/server.py, serve/test_starting.py, tools/test_bench.py).

## 3. The free at generate.cpp ~24055: already fixed on the branch; build-check sm_75 green

engines/qfn/src/program/generate.cpp around line 24055 **of the plan's snapshot** is the session-end block - at
a3cab0d93 the file ended with bare `cudaFree(d_next); ... cudaFree(arena);` at :24050-24055 (their results neither
checked nor discarded; `hipFree` / `hipHostFree` are nodiscard, so the HIP build warned on them). e3d707bea
("QFN HIP build: every cudaFree/cudaFreeHost in generate.cpp takes (void) ... the way the file already writes the
same calls", Nov 24 13:58) applied the file's own idiom - the `(void)` cast every neighbouring free in the file
already used (:2033, :2443, :2489-2490, ...) - to all nineteen of them. It is an ancestor of this branch, and the
check confirms nothing is left: every cuda*/hip* free call site in the file now carries the cast (21 sites incl.
the session-end block at :24566-24571; grep for a bare one: none). No new source change was made for step 3, and
the neighbouring `strata::core::session_graphs_free` / `doorbell_free` helpers use the same idiom internally
(layer.cpp:2168), so "the same way the neighbouring frees are checked" needs no other macro/helper - the file has
none that inspects a free's result, then or now.

Build-check in a NEW build dir under scratch/, no GPU runs:
- pinned ggml: scratch/llama.cpp - a `--depth 1` fetch of the commit third_party/ggml/VERSION.txt pins
  (3cf03257f219); the tree has no llama.cpp checkout of its own.
- configure: `cmake -G Ninja -S engines/qfn -B scratch/build-qfn-sm75 -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_ENABLE_CUDA=ON
  -DSTRATA_ISA_FLOOR=avx -DSTRATA_BUILD_DSV41=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=OFF
  -DSTRATA_GGML_DIR=<repo>/scratch/llama.cpp -DSTRATA_MMQ_TAB=ON` (the top project's qfn_engine ExternalProject
  options, tests off): clean configure, ggml 0.24.0 @ 3cf0325.
- `ninja CMakeFiles/strata.dir/src/program/generate.cpp.o`: compiles; its only warning is the pre-existing unused
  `w` at :15580, from elsewhere in the file.
- `ninja strata -j 12`: 178/178, saguaro-qfn linked, no errors.

No commit (nothing changed); this report carries the verification.

## Tests run (only the modules touched, one at a time)

- tools.test_bench: 39 OK (baseline) -> 40 OK (step 1) -> 41 OK (step 2)
- serve.test_starting: 24 OK (step 2)
- engines/qfn sm_75 build under scratch/: generate.cpp.o compiles, `strata` 178/178 linked

## Open questions

- Step 1 kept a3cab0d93's folder-shape rule (`<pack>/mtp` names the NextN drafter, anything else the Lamina layer)
  rather than a strict "the folder is named Qwen3.8-Flash-Next-MTP-Lamina" test: tools/test_qfn_calibrate.py:276-282
  (not one of the modules this plan names, and not run here) pins "MTP (Lamina)" for a no-args INFO line and for a
  `/m/Lamina` folder, so tightening the rule would need that module touched and run too. The two real configs
  answer correctly either way.
- Step 2: the Ollama-compat /api/version still answers `0.0.0-maya` (serve/server.py:7402) - an upstream string on
  another endpoint, not /health; left alone as out of scope. The banner and About > Updates still name a git clone
  by its commit through `saguaro_version()` (by design; untouched). serve/test_dashboard.py's
  `SAGUARO_VERSION` assertions were not re-run here.
- Step 3: e3d707bea silences the nodiscard and matches the file's idiom, but the result at those sites is still not
  *handled* - a failed free at the session's end has nothing to do with it anyway, and no neighbouring site in the
  file handles one either. The build-check built only the `strata` target (the changed file's target); the
  production build's extra `strata_kq` / `strata_kq_tab` targets were not compiled. scratch/llama.cpp and
  scratch/build-qfn-sm75 are left in place (scratch/ is git-ignored).
