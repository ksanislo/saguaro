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

## Card proof (done 2026-11-22, bench/upstream-u2/<eng>/summary.txt, results.jsonl)
- GLM: pools identical (1065/1159/1161/1031 slots), VRAM identical, greedy same; decode 17.1 -> 17.4 tok/s (A1/B1).
  Late figures 160-224 MiB < 572 (default 700 reserve - 128): the pool unchanged. Free once loaded 675-689 MiB.
- MM3: 1927 of 7296 experts in VRAM both, VRAM identical, greedy same; decode 7.4/7.6 -> 7.6/7.7. Late 224-256 MiB <
  896: unchanged. Peer copies GPU0<->1 / GPU2<->3 tested peer, cross-socket the driver's copy (as before).
- DS4 (ABAB): experts in VRAM within the run-to-run jitter (A 1278/1310/1258/1259, B2 1278/1310/1251/1260); late 256
  MiB < 472: unchanged. Decode B -4..-6% (21.2/21.0 vs 22.5/21.9): B is a new build, so its PCIe-lane times re-timed
  and came out "lane on" for CUDA2 (1 layer streams, a marginal 2.23 vs 2.33 ms call) where the deployed build's
  live times say "lane off" on every card - the calibration, not the ported code. Before install: re-time or carry
  "lane off" (DS4_LANE_CAL), then re-check decode.

## Open / unverified
- HIP build not compiled here (no ROCm): HIP paths only checked by check_hip_compat and reading.
- exl3.cu's HIP commits change CUDA device code too (identity rewrites: fsh32, bytesum2h); no live profile uses EXL3;
  exl3_test not run.
- Skipped: f692477ca (sm_80+ only), 57744879c (installer refactor), 653ef9b53 (Windows Radeon pin cap),
  8d50ceb32 / 40a05ba97 / e7d2a4d03 / 8c6b149c3 (Hy4's own ds4 copy), mixed-GPU (AMD+NVIDIA) and GLM link/helper
  series, v1.1.57 GLM-5.3 (744B), Inkling/Hy4 setup commits.
