# NEXT-u2.md - saguaro/upstream-u2 (upstream v1.1.74 fixes for GLM / DS4 / MM3 + the AMD commits)

Branch `saguaro/upstream-u2` off `saguaro/qfn-vram` 95fb5638. Worktree /data/llm/saguaro-upstream-u2.
Build: `build-u2/` (Release, sm_75, NCCL). Card-proof binaries: /data/llm/bench/upstream-u2/bin (commit in bin/COMMIT).

## State
- #1026 (VRAM reserve held for the whole run): ported for DS4, GLM, MM3 + server restart-once. DONE, built.
- MiniMax peer copies only where tested (2ac721f7d, 89f619e36): cherry-picked. DONE.
- DS4_AR2_DECODE=2 (e84874144): ported, opt-in (default off). DONE.
- Qwen Monitor promotions (71261c990): ported; touches only the in-tree Qwen path (engine_kind=qwen), not QFN.
- AMD/HIP commits since 10e69a0eb for GLM/DS4/MM3/shared: ported where they apply (list in the final report,
  `.port/done`, `.port/skipped`). check_hip_compat / test_hip_compat fixed (the rename had renamed the STRATA_USE_HIP
  macro in the checker only).
- Shared with saguaro/hy4-up: src/core/vram_late.hpp, tests/vram_late_test.cpp byte-identical to upstream fd601f298
  (MAYA_VRAM_LATE_* names inside: run tools/saguaro/rename_env.py after both merges);
  CMakeLists vram_late_test hunk identical; hip_compat/cuda_runtime.h merges cleanly with hy4-up (checked with
  git merge-tree).

## HIP (mortar, RX 9070 XT gfx1201, ROCm 7.1)
- 90fcee37 pushed as saguaro/test/u2-hip, built in mortar:/storage/llm/saguaro/u2/build-hip (Release, Makefiles,
  STRATA_PREFILL_MMQ=ON, -j12): saguaro-glm, saguaro-ds4, saguaro-mm3 all compile and link; --help runs. No GPU run.
- check_hip_compat now also finds inline PTX and missing CUDA headers in what an AMD device pass compiles (each file's
  #if chain evaluated for gfx1201 / gfx1100); it would have flagged main's ds4_vision names, the mm3 mma.sync and
  hy4's <cuda_bf16.h>. It cannot see a HIP stub struct missing members (the ds4 elastic pool): only a HIP compile can.

## Waiting
- Card proof queued (night-queue): bench/upstream-u2/run.py glm | ds4 | mm3 -> bench/upstream-u2/<eng>/summary.txt,
  results.jsonl, DONE. A = /opt/llm/saguaro/bin/saguaro-<eng>, B = bench/upstream-u2/bin/saguaro-<eng>; A B, then
  A B again when the expert figures differ. Late files go to bench/upstream-u2/<eng>/vram-late-B.txt.

## Open / unverified
- HIP build not compiled here (no ROCm): HIP paths only checked by check_hip_compat and reading.
- exl3.cu's HIP commits change CUDA device code too (identity rewrites: fsh32, bytesum2h); no live profile uses EXL3;
  exl3_test not run.
- Skipped: f692477ca (sm_80+ only), 57744879c (installer refactor), 653ef9b53 (Windows Radeon pin cap),
  8d50ceb32 / 40a05ba97 / e7d2a4d03 / 8c6b149c3 (Hy4's own ds4 copy), mixed-GPU (AMD+NVIDIA) and GLM link/helper
  series, v1.1.57 GLM-5.3 (744B), Inkling/Hy4 setup commits.
