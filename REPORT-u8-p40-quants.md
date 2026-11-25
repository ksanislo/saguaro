# U8 report - Paul's P40 setup: the two GSQ quants, and the Pascal build recipe

## What changed

### Step 1 - the two GSQ-RCO quants (commit "U8 step 1: the two GSQ-RCO quants...")

* `tools/qfn/download.json` - two new quants beside `ud-q4_k_xl` (still the default), `q8_0` and
  `cyber-frost-q5_k_m`:
  * `gsq-iq3_s` - ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF @ `ed59f92082b1e93c0e96d60a8b11aab089b52f09`,
    folder `IQ3_S`: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` 54,817,524,224 B
    sha256 `4c1eb2ce...d2aca3`, `-00002-of-00002` 28,800,138,432 B sha256 `316b46f3...61e113` (83.6 GB).
  * `gsq-coder-iq1_m` - ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF @
    `5348543e0147355ac9cbcb031184a3546350988e`, folder `IQ1_M`:
    `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` 29,608,446,496 B sha256 `e11083ba...087fad`,
    `-00002-of-00002` the same 28,800,138,432 B / sha256 `316b46f3...61e113` as IQ3_S's shard 2 -
    the PLE table, the same bytes in every GSQ quant (a test pins this equality). 58.4 GB.
  * Both `needs` = {vram 15, ram 88} - estimated from the file sizes like the measured quants state
    the same pair; each entry carries a `_` comment saying so.  The coder's `about`/shown name say it
    is the code-tuned variant in the repo's own words ("half of its experts removed, targeted at
    code"), and its repo's mmproj is noted in the `_` comment.
  * The file names and the mmproj size were read from the Hugging Face API's tree listings **at the pinned
    revisions** (metadata GETs - no model bytes were downloaded).  All sizes/hashes are the plan's.
* `tools/maya_quants.py` - `QFN_QUANTS` gains `gsq-iq3_s: "GSQ-RCO IQ3_S"` and
  `gsq-coder-iq1_m: "GSQ-RCO Coder IQ1_M"`; `QFN_IDS` gains their ids (`qwen3.8-flash-next-iq3_s`,
  `qwen3.8-flash-next-coder-iq1_m`, the file stems, and `qfn-...` short forms); new `QFN_SETUP_IDS`
  map, so `qfn_setup_id()` gives the plan's ids - the GSQ keys drop their `gsq-` (the repos name the
  quant `IQ3_S` / `IQ1_M`) - which drives `config_stem` (`maya-qwen3.8-flash-next-iq3_s`,
  `maya-qwen3.8-flash-next-coder-iq1_m`) and the Models page rows.  Everything else (shown names,
  picker, import catalog, installer records) flows through `QFN_QUANTS` unchanged.
* `saguaro.py` - one line: `qfn_entry`'s about now reads a quant's own `license` first (the GSQ repos
  are apache-2.0; the manifest-level Qwen Community License stays for the rest).  `--gguf-dir`
  acceptance needed no change (`qfn_files_step` already uses any folder in place, size- and
  sha-checked) - now covered by tests for both GSQ quants, including the missing-shard refusal.
  Packing stays `engines/qfn/tools/iq_pack.py`'s as for the other QFN quants; GSQ files need no
  `--compat-bf16` (they already store in BF16 what the engine reads as BF16, so the flag the setup
  passes converts nothing for them).

### Step 2 - the Pascal build (commit "U8 step 2: the P40 build path...")

* `saguaro.py` `cuda_gcc_max()` - NVIDIA's supported-matrix: CUDA 12.0-12.3 compile with g++ <= 12,
  12.4-12.5 <= 13, 12.6+ <= 14; 13.x: no pinned ceiling (the probe tries).
* `saguaro.py` `gpp_versions()` / `gcc_major()` / `host_order()` / `missing_gpp_hint()` -
  `toolchain_for()` now carries (nvcc, version) pairs and orders the host compilers per toolkit:
  when the toolkit's ceiling is under the PC's default g++ and a fitting `g++-N` is installed it is
  tried **first** (the P40's CUDA 12.0 on Ubuntu 24.04's g++ 14 -> `g++-12`, no wasted "unsupported
  GNU version" round); when nothing fits, the closing line names the package
  (`sudo apt-get install -y g++-12`).  Windows and `--host-compiler` behaviour unchanged.
* `saguaro.py` `host_compiler_defs()` - a chosen host compiler now pins the whole CUDA build on it
  (`-DCMAKE_CUDA_HOST_COMPILER` + `-DCMAKE_C_COMPILER=gcc-N` + `-DCMAKE_CXX_COMPILER=g++-N`), as the
  box's known `build-qfn.sh` does (gcc-12/g++-12 for C, C++ and CUDA's host code); used by
  `compile_engine` (so engines/qfn's ExternalProject inherits it through the top project's
  `_qfn_extra`) and by `compile_vision_qfn` from the engine stamp.  The failed-build hint now names
  the matrix instead of a fixed g++-13.
* `CMakeLists.txt` - unchanged: it already gives engines/qfn `-DSTRATA_EXPERIMENTAL_SM60=ON` plus
  this build's C/C++/CUDA-host compilers when every arch is below sm_75 (`_qfn_old AND NOT _qfn_new`
  branch), and Pascal sm_6x gets `-Xptxas -dlcm=ca`.  A test pins the branch and the flag so a
  refactor cannot silently drop them.
* The fit policy is untouched (`tools/qfn_fit.py`, `MIN_CC` 70 - 1.2.1-B's), no engine C++, nothing
  downloaded.  `serve/qfn_engine.py` was not touched, so no `extract_qfn_engine.py` run.

## Files

`tools/qfn/download.json`, `tools/maya_quants.py`, `saguaro.py`, `tools/test_setup_qfn.py`
(`GsqQuants`, `PascalBuild`), `tools/test_maya_quants.py` (`test_the_gsq_quant_names_and_setup_ids`).

## Tests run (each module once, as the plan names)

| module | result |
| --- | --- |
| `tools.test_setup_qfn`    | Ran 62 tests, OK (1 skip - the pre-existing `ref/live-ud-q4.json` copy check) |
| `tools.test_maya_quants`  | Ran 25 tests, OK (1 skip - pre-existing, the screen's Textual libraries) |
| `tools.test_setup_download` | Ran 2 tests, OK |

New cases: manifest pins + hub URLs + apache-2.0 for both GSQ quants, the shared shard 2, ids / shown
names / config stems, `--gguf-dir` in place for both (with the missing-shard refusal) in
`GsqQuants`; the CUDA-gcc matrix, the host order, the install hint, `toolchain_for` picking `g++-12`
over the default g++ 14 in one probe, and the P40 `compile_engine` line (`arch 61`,
`-DSAGUARO_QFN=ON`, `/usr/bin/nvcc`, `g++-12`/`gcc-12`, target `saguaro-qfn`, and the top CMakeLists'
SM60 branch) in `PascalBuild`.  `qfn_fit --ram 96 --quant gsq-coder-iq1_m 24:61` also laid out from
the CLI as a smoke check.

## Open questions

1. **Paul's box still will not set up**: QFN keeps `QF.MIN_CC = 70` and the global 88 GB-RAM / 16 GB-VRAM
   fit gate (both 1.2.1-B's territory), so a 24 GB P40 with 32 GB of RAM is refused before the build -
   the quants and the build line are ready, the acceptance is 1.2.1-B's.  Whether the smaller GSQ
   quants may relax the RAM floor (32 GB nearly holds the Coder quant's 54 GiB with 24 GB in VRAM) is
   a fit decision this task must not make.
2. The new quants' `needs` stay {15, 88} so they agree with `qfn_quant_fit`'s fixed "about 96 GB" line
   and the global gate; giving them their own numbers means parameterizing that line (and its test
   expectations) - left for when the fit policy per quant lands.
3. The Coder repo's own `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (907,543,008 B, sha `b1a82259...`) is a
   different file from unsloth's pinned tower (907,542,944 B, `2e788f8c...`); the vision step still
   downloads unsloth's.  Accepting the coder repo's mmproj from `--gguf-dir` would mean pinning it as
   an alternative hash (`sha256_ok` already takes a list) - not asked for, not done.
4. The setup still passes `--compat-bf16` to `iq_pack.py` for every quant.  For the GSQ files it
   converts nothing (their small projections are already BF16), but if it should be dropped for them
   outright, `qfn_pack_step` needs a per-quant-kind flag.
5. Neither GSQ quant has a `measured` line, so the picker says "not measured on AMD yet" - on the P40
   box the wording (AMD-centric) may want a rethink together with the NVIDIA acceptance.
