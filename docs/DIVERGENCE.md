# DIVERGENCE: how Saguaro differs from upstream Project Maya / Strata

The register docs/UPSTREAM.md §3 asks for: per engine, each of our modifications - what, why, where (files/functions),
its switch, its measured gain, and whether upstream could take it - plus what we drop from upstream on purpose.
Resolve an intake conflict by reading the entry for the file it touches, not by guessing. Every topic branch adds its
line when it merges.

Entry format (2-4 lines): **what** and **why**; **where** (`file:function`) and **switch + default**; the **measured
gain** with the line of the source that states it, or "not measured"; and whether the change is **exact** (same tokens
/ bit-identical) or **answer-changing**. Numbers are quoted as written in the source (ref/PERF-LEDGER.md,
ref/ROLLBACK.md, ref/live-stack-notes.md, ref/reports/REPORT-*.md); nothing here is re-derived.

Hardware note: every gain below is on the machines in those sources - four Tesla T4s (mangonel), an RX 9070 XT
(mortar), or a Tesla P40 (paulai). Long QFN/GLM reads on the T4s are power-capped at 70 W/card, so a kernel that is
faster in isolation can be flat there (ref/PERF-LEDGER.md:456-465).

QFN (engines/qfn) is our own port of Strata, carried as its own CMake sub-project; the Maya engines (GLM, DS4, MM3,
HY4, IK2) are upstream Project Maya with our changes on top. Nothing here records a Strata sweep: the last swept
Strata commit is not yet written down in this register (docs/UPSTREAM.md §5 asks for it) - the port base in the tree
is the subtree of the live Strata tree 82c389f5e + server 0c800169d (ref/live-stack-notes.md:65).

## GLM

_(step 4)_

## DS4

_(step 3)_

## MM3

_(step 3)_

## HY4

_(step 4)_

## IK2

_(step 4)_

## QFN

_(step 2)_

## serve + web

_(step 4)_

## setup/tools

_(step 4)_

## Dropped from upstream on purpose (the Maya Qwen path)

**The Qwen path inside saguaro-glm (the "qwen4exp" engine) - kept in our tree, upstream's changes to it are dropped.**
- What: upstream's Qwen3.8-Flash-Next support lives in the GLM binary as its own engine: the `qwen4exp` driver in
  `src/program/generate.cpp` (~4.5K of its 10.7K lines) plus the Qwen-only files
  `src/core/{session,mtp,verify,layer,expert_source,native_dense,native_head,remote_experts}.cpp` and
  `src/prefill/prefill.cpp` - about 16K lines, which is Strata v0.1.27, the ancestor of engines/qfn. We serve Qwen on
  saguaro-qfn instead, so at every intake the upstream hunks in these paths are dropped by path, and our copy stays
  where it was: the v1.1.50 Qwen path (upstream v1.1.51-v1.1.76 Qwen work was never taken in).
  ref/reports/REPORT-qwen-engines.md:23, 113-119.
- Why: on the same CYBER-FROST file and the same card (RX 9070 XT) saguaro-qfn answers 1.2x faster (34.4 vs 28.3
  tok/s solo) and reads 1.4-2.0x faster (16K: 733-1,007 vs 422-428 tok/s; ~55K: 1,058 vs 527 tok/s, first token 50 s
  vs 106 s); the Maya path cannot load unsloth's UD-Q4_K_XL at all (no Q8_0 GPU dequantizer for its embedding) and dies
  with exit -4 on an AVX1-only host, where saguaro-qfn runs the same file at 43.7 tok/s.
  ref/reports/REPORT-qwen-engines.md:15-21. Exactness: neither is exact against the other - the two engines disagree
  KL 0.025 / top-1 95.55 % on the same file (rounding, not a defect), and saguaro-qfn on AMD is KL 0.006 from saguaro-qfn
  on the T4s. ref/reports/REPORT-qwen-engines.md:78-82.
- Where: `src/program/generate.cpp` (the qwen4exp driver), `src/core/session.cpp`, `src/core/mtp.cpp`,
  `src/core/verify.cpp`, `src/core/layer.cpp`, `src/core/expert_source.cpp`, `src/core/native_dense.cpp`,
  `src/core/native_head.cpp`, `src/core/remote_experts.cpp`, `src/prefill/prefill.cpp`.
  Switch and default: **none** - this is not switch-gated. Nothing is deleted yet; the drop happens at intake
  (path exclusions in `tools/saguaro/rename_internal_maya.py`) and `generate.cpp` is the one file shared with GLM, whose
  runner upstream already moved out. ref/reports/REPORT-qwen-engines.md:118.
- Removal (option 1 of that report, recommended to Ken, **his call**) is not done: setup (`saguaro.py`,
  `tools/maya_quants.py`, the family "qwen"), the server's "qwen" engine kind, ~2-4 h of engine deletion plus a GLM
  smoke gate, and CYBER-FROST moved onto saguaro-qfn (which needs `engines/qfn/tools/iq_pack.py` and the NextN-block
  drafter - both already built on branch qfn/in-file-nextn d025b3d9f, not merged).
  ref/reports/REPORT-qwen-engines.md:143-171.

**Skipped upstream changes are listed where they belong**: GLM "same answers on every start" (not ported, needs four
earlier upstream mechanisms and costs reads; see the GLM section) and GLM tensor parallel (opt-in upstream, needs every
expert slice in VRAM). ref/reports/REPORT-upstream-v1.1.75.md:102-109.

## Renamed

Everything upstream calls Maya / Strata / IK2_*, we call SAGUARO_*, saguaro-* or Saguaro. The tools do it by pattern,
so a new upstream setting is caught automatically; a missed name is a tool bug, fixed in the tool.

- **Run-time settings**: `MAYA_X` / `STRATA_X` -> `SAGUARO_X` by suffix pattern - `tools/saguaro/rename_env.py`
  (`--check` passes on the tree; "same suffix, one name, no aliases"), and `tools/saguaro/migrate_configs.py` for our
  own configs. The engine code reads the SAGUARO_ name. Upstream's v1.1.16 note that `MAYA_X` aliases `STRATA_X` still
  holds for old installs' configs (ref/live-stack-notes.md:82-83); a compatibility map for user settings is the open
  item in docs/UPSTREAM.md §2.
- **The v1.1.75 internal rename is adopted** (upstream dc6e746a4): `include/strata` -> `include/maya`,
  `namespace strata` -> `maya`, CMake targets `strata_*` -> `maya_*`, build options and defines `STRATA_X` -> `MAYA_X`.
  It is one generated commit (f784321fd) made by `tools/saguaro/rename_internal_maya.py`, which reproduces upstream's
  commit for 1292 of its 1315 files; `-DSTRATA_X` is still accepted, with one deprecation line, so deploy scripts
  outside the repo keep building. Two places are deliberately different from upstream: FetchContent is
  `maya_llamacpp` everywhere, and our copies of a92df34f6 / 7f8804760 test `MAYA_USE_HIP` (upstream's own test still
  tests `STRATA_USE_HIP` after its rename, which turns GLM decode graphs on by default in its HIP builds).
  ref/reports/REPORT-upstream-v1.1.75.md:34-68.
- **Left with the old names on purpose**: `engines/qfn`, `tools/qfn` and the generated `serve/qfn_engine.py` (QFN is
  Strata's own CMake project, so its `SAGUARO_QFN` sub-build keeps its `-DSTRATA_*` arguments); the Strata pack format
  (`src/strata_fmt`), `tools/strata/` and the Python module names; the quant names; the browser's `maya.*` keys; and
  `tools/maya_bench.py`, whose filename is kept to ease merges. ref/reports/REPORT-upstream-v1.1.75.md:50-57, 86.
- **Names that arrived bare upstream and were renamed here**: #1154's `IK2_*` -> `SAGUARO_IK2_*` (d700057c5),
  `MAYA_GLM_KDA_GRAPH` / `MAYA_GLM_DEFER_CPU` / `MAYA_GLM_SPEC_WIN` -> `SAGUARO_GLM_*`, `--bench`'s GPU-lock and
  hot-flag variables -> `SAGUARO_*` (`tools/maya_bench.py:34`), and the `MAYA_*` names the rename left behind in
  comments (e58d740a3). ref/reports/REPORT-upstream-v1.1.75.md:111-118.
- **Bare per-engine families stay bare, with the prefix accepted**: the DS4 engine reads each switch as `SAGUARO_DS4_*`
  first and the plain `DS4_*` second (`ds4_env`, `src/ds4/ds4_model.cu:15715-15718`), so its entries below name the
  `DS4_*` form the configs write; the HY4 engine reads `HY4_*` directly (e.g. `HY4_HOT_SEED`,
  `src/hy4/hy4_model.cu:6596`), like the other HY4_* settings.
- **Program and log names**: the engine programs are `saguaro-glm`, `saguaro-ds4`, `saguaro-mm3`, `saguaro-hy4`,
  `saguaro-ik2`, `saguaro-qfn` (upstream: `maya`, `maya-ds4`, ...), and the log lines start `saguaro <part>:`
  (`tools/saguaro/rename_log_prefixes.py`). Outside the repo, log parsers must match both old prefixes: the live stack
  changed `[maya]` to `[saguaro]` on 2026-11-19 and the bench scripts still grep for both (ref/ROLLBACK.md:315-316).
