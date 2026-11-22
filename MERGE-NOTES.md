# saguaro/hy4-maya: merge notes (2026-11-22)

Last merge into this branch: c55126b9 = saguaro/qfn-vram 7d98986b (setup, Models page, Monitor and docs for Hy4-Flash
and Inkling 2 Mini). The merge had no conflicts (the two sides touch different files).

## What this branch adds since the previous merge (d5d2ed84)
- Saved conversations (HY4_SLOT_DIR / HY4_SLOT_GB / HY4_SLOT_DAYS / HY4_SLOT_MIN), engine-side, maya-mm3's SlotStore.
  Conversations with pictures are not saved; off under KV streaming or a loaded drafter.
- HY4_VISION_F16=1: the vision tower's BF16 GEMMs on FP16 tensor cores (sm_75 has no BF16 ones).
- HY4_TP_HEAD_MEMBER=1: the output head's rows split over the tensor-parallel members.
- HY4_CREW_PIN=1|2: members' launch threads pinned to their GPU's NUMA node / one core.
- HY4_FUSE=1: rmsnorm / sigmoid gate / SwiGLU fused with the next GEMV's int8 rows (q_b, wo, shared down).
- HY4_TP_PART_NODES=1: each member's partial sums in its own GPU's NUMA node.
- SAGUARO_PCIE_SAFE=1 (host safety, opt-in): under tensor parallel no expert weights are streamed to any GPU during
  prompt reads; the experts no member holds run on the CPU lane. For PCs that reset when all cards stream from host
  memory at once (this box). ON in configs/Hy4-Flash-Maya-S-IQ2_S.json, off by default and in the setup's output.
- HY4_TP_HEAD_MEMBER=2: a test arm (the split head's VRAM planned, the head on GPU 0).
- The serve loop prints "streamed:" (MB a GPU, seconds of passes / bounce fills) after a request that streamed experts.
- bench/hy4/slot_check.py, run.py HY4_NO_CLAIM.
- configs/Hy4-Flash-Maya-S-IQ2_S.json: HY4_SLOT_DIR + HY4_SLOT_GB 16 + HY4_VISION_F16=1 + SAGUARO_PCIE_SAFE=1.
- tools/hy4_fit.py: the four-T4 layout writes HY4_VISION_F16=1; tests updated (test_setup_hy4, test_docs_new_engines).
- docs/HY4-FLASH.md: pictures measured, saved conversations (how to turn on).

## Tests
`/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_hy4_frontend serve.test_new_engines_ui serve.test_models
serve.test_server tools.test_setup_hy4 tools.test_docs_new_engines`: 305 tests OK (4 skipped), on HEAD.
Also tools.test_setup_ik2 tools.test_setup_dash tools.test_maya_quants: OK (after the merge).
Engine: build-hy4c builds clean (incremental from build-hy4b's configure, CUDA 13.1, sm_75).
On the cards (build-hy4b = before HY4_FUSE etc.): slot checks PASS (tp 4; layer split with a fixed placement), pictures
right on BF16 and FP16 paths, head split +5-6% decode (exactness not yet shown), pin1 / spin0 < 2%.

## Unverified
- NOTHING built after build-hy4b has run on the cards: HY4_FUSE, HY4_TP_PART_NODES, HY4_TP_HEAD_MEMBER=2,
  SAGUARO_PCIE_SAFE (ON in the config!) and the streamed report. Four-card Hy4 runs are held by Ken after the
  2026-11-22 host reset; the test lines are in NEXT-hy4.md (T0-T5). Run T1 before installing this config.
- SAGUARO_PCIE_SAFE's read cost (estimated 0 to -25%), and whether it removes the reset risk at all (one event, a lead).
- HY4_TP_HEAD_MEMBER=1 exactness (md5 differs from off because the head's VRAM moves 41 experts to the CPU lane).
- 262K+ context: held; 131K stays.
