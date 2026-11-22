# saguaro-ik2 on upstream Maya v1.1.74 (branch saguaro/ik2-up, worktree /data/llm/saguaro-ik2-up)

Base: saguaro/qfn-vram 95fb5638 (contains our v1.1.72 port). Upstream: origin/main fd601f298 (v1.1.74).
Build: build-ik2up (cmake line in the task; `nice ninja -C build-ik2up -j20 saguaro-ik2 ik2_kernels_test ik2_off_test ik2_cpu_test`).
Card runs only through /data/llm/bench/night-queue/queue.txt (IK2_NO_CLAIM=1, cards 2,3); outputs under runs/ (git-excluded).

## Done (commits on saguaro/ik2-up)
1. src/ik2/* + tests/ik2_{kernels,off}_test.cu + src/mm3/mm3_link.cuh + data/expert-profile-ik2mini.bin = upstream v1.1.74, unchanged;
   CMake: offload path (ik2_off.cu), lane on saguaro-hy4's lane threads + src/hy4/ds4 CPU rows, Volta on.
2. Renames: IK2_* -> SAGUARO_IK2_* (not the IK2_SUB macro), "maya-ik2:" -> "saguaro ik2:", program saguaro-ik2,
   calibration file ~/.saguaro/ik2-offload-calib.txt.
3. ik2_cpu.cpp/hpp: our AVX1 NVFP4 rows kept (upstream's file unchanged since v1.1.72). Upstream's new exact lane
   (ik2_lane.cpp) has AVX-512 / AVX2+FMA / scalar forms, chosen by cpuid: this host takes scalar (std::fma, libm) - correct.
4. Row check fix (88266f524) re-applied: upstream still had C % bv || F % bv.
5. AMD/HIP build branch kept (Ken: second machine, RX 9070 XT 16 GB); hip_compat headers taken from upstream (superset).
   build-ik2up: clean (one unused-function warning). ik2_cpu_test passes.

## Queued
- runs/r1/r1.sh (night-queue): kernels + off_test, t4_check 1+2 cards, KL (new forms: scalar ref vs tc / scalarattn /
  scalarexp, 1 and 2 cards; ours fast0/fast; cross new vs ours), bench 1 card UD (new vs ours, rotated x2), quant 1 card,
  2 cards 131072 depths 0/16K (rotated x2). Done marker runs/r1/DONE.

## Next
- Conversations (set aside / saved folder / rewind snapshot at last user message / cancel back to snapshot) ported into
  upstream's single-conversation serve loop (upstream has --parallel slots + 8 snapshots, no saving). Same env names as
  our port: SAGUARO_IK2_KEEP, _KEEP_GB, _SLOT_DIR, _SLOT_MIN, _SLOT_GB, _SLOT_DAYS, _SNAPS, _SNAP_MSG, --user-token.
- serve/server.py + web (fork, branch saguaro/ik2-up-serve), setup/Models/docs/configs (fork, branch saguaro/ik2-up-setup).
- AMD 16 GB assessment (offload path is NVIDIA-only upstream).
