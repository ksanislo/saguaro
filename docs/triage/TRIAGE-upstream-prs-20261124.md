# TRIAGE: upstream Project Maya open PRs (143) vs the Saguaro tree

Date 2026-11-24. Sources: `ref/prs/<n>.json` (title/body/files/commits), `ref/upstream-1176-1178-commits.txt`,
`ref/UPSTREAM.md` policy (take every supportable gain for ANY hardware; answer-changing ones opt-in or KL-checked).
Working notes: `scratch/notes.md`, condensed PR texts: `scratch/digest.txt`.

Already-in-release check: **none of the 143 open PR numbers appear in the v1.1.76-v1.1.78 intake** (every `(#…)` in
`upstream-1176-1178-commits.txt` is a merged/closed PR). Per the plan: **#1233 taken**, **#1225 planned** (marked
below). Tree checks found two more already ours: **#867** (DS4_EXL3_LEND, 9 refs in `ds4_model.cu`) and **#1204**
(AVX1 IQ rows wired into `ds4_iq_cpu.cpp`).

Rename layer applies to everything: `MAYA_*` -> `SAGUARO_*` via `tools/saguaro/rename_env.py`, `maya.py` ->
`saguaro.py`, `include/strata/*` -> `include/maya/*` (PRs rebased before the v1.1.75 rename still carry the old
paths). The Maya Qwen path is REMOVED here (no `src/ngram`, no `src/core/{layer,mtp,verify,session,native_*}.cpp`),
so Qwen hunks drop by path. Columns: **Verdict** TAKE / TAKE-OPT-IN (answer-changing, keep opt-in or KL-check) /
LATER / SKIP / MARK (already taken/planned); **Val** high/med/low; **Ours**: clean = applies with renames only,
adapt = rename + small path drift, careful = touches our heavily modified area (see end).

## Table

| PR | Verdict | Val | Area | What changes (1 line) | Own numbers | Exact? | Size | Ours |
|---|---|---|---|---|---|---|---|---|
| 1231 | TAKE | H | GLM | DSA indexer score+select fused over all SMs, same cells | 500K ctx 23.8->31.2 tok/s | yes | med | careful |
| 952 | TAKE | H | GLM/Pascal | sm_60 decode+prefill kernels (Q4_K GEMV, IQ dots, IQ GEMM, FP16 dense) | decode +5-9%, first token -20-39% | yes | large | careful |
| 630 | TAKE | H | GLM | decode route in one pass an expert: 3 block barriers not 9 (ROUTE_ONE on) | route -24%, decode -0.35% | yes | med | careful |
| 1219 | TAKE | H | IK2 | 6 more expert block types IQ2_XXS..Q4_K + row-width loader fix | UD-IQ3_XXS loads on 16 GB | n/a | med | careful |
| 1045 | TAKE | H | DS4 | RAM store slots of each expert's own size (size classes) | +24% experts/RAM, decode +10%, prompt +12% | yes | large | careful |
| 960 | TAKE | H | GLM | MAYA_GLM_DEMOTE=all / ADMIT=all tier policy | spilled decode +21% | yes | small | careful |
| 1166 | TAKE | H | GLM | prompt-lookup drafts beside DFlash (MAYA_GLM_LOOKUP) | agent code repeats verified | yes | med | careful |
| 712 | TAKE | H | DS4 | Q5_K int8 tensor-core GEMVs (IMMA), opt-in | 4090D decode 60.2->70.4 (+16.9%) | yes | med | careful |
| 1222 | TAKE | H | HY4 | HY4_PSTREAM: prompt experts from pinned RAM, 2 buffers, a pass ahead | prompt 1.3-1.8x | yes | med | careful |
| 1024 | TAKE | H | all | pinned host mem on explicit 2 MB huge pages (shared allocator) | fixes AMD compaction stalls | yes | large | careful |
| 896 | TAKE | H | MM3 | helper GPU for CPU-lane experts (MAYA_MM3_HELPER) + per-GPU lane timing | decode ~2x with cheap helper | yes | med | careful |
| 1192 | TAKE | H | IK2 | IK2_PAR_DECODE_PCT: decode steps interleaved during parallel prompt reads | decode median +42.6% | yes | small | careful |
| 636 | TAKE | H | DS4 | --parallel: measured cost policy decides who drafts (replaces 1-pass gate) | drafts where gate drafted nothing | yes | large | careful |
| 170 | TAKE | H | GLM | experts-only community EXL3: Q8_0 twins (1/row), packer, MTP from quants | serves neko 4bpw on 7x3090 | yes | large | careful |
| 1145 | TAKE | M | kernels | sampler chain parallel + multi-block vocab passes | hash-identical draws | yes | med | clean |
| 1054 | TAKE | M | GLM | Workers NUMA/cache-line isolated fast path (WORKERS_FAST) | lane call 297->173 us | yes | med | careful |
| 1158 | TAKE | M | DS4 | exact peer-VRAM all-reduce for decode windows (PEER_AR_DECODE) | NVLink-pair boxes off PCIe | yes | small | careful |
| 1123 | TAKE | M | HY4 | restore points --snap-every N (agent compaction resumes, no reread) | 10.3 MB/point | yes | med | careful |
| 953 | TAKE | M | DS4 | ordered CPU-lane worker affinity (DS4_CPU_LANE_CPUS) | EPYC+4090D decode +9.8% | yes | small | careful |
| 971 | TAKE | M | GLM | DFlash for the lone active request in multi-slot serve | 4x4090 measured | yes | small | careful |
| 911 | TAKE | M | DS4 | IQ1_S CPU rows on AVX-512/VNNI (DS4_IQ1S512) | decode +9% i7-11700KF | yes | med | careful |
| 1018 | TAKE | M | GLM | FETCH_EARLY + fixes upstream moe_wait_fetch seq bug (~1% wrong experts) | -10% ms/token on SSD rig | yes | small | careful |
| 931 | TAKE | M | DS4 | MXFP4 CPU: SSSE3 tier (exact) + DS4_MX_FAST_FMA (opt-in) | old Xeons 1.6->several tok/s | tier yes | med | careful |
| 1117 | TAKE | M | DS4 | --parallel: copy busy conversation state into idle lane (SLOT_COPY) | agent compaction no reread | yes | med | careful |
| 1022 | TAKE | M | DS4 | arm64 NEON+SDOT MXFP4 CPU tier (scalar bits) | 27.4->~40 tok/s Altra | yes | small | careful |
| 1141 | TAKE | M | DS4 | GEN line carries only new ids (GEN_KEEP + BLAKE2b verify, RESEND) | 1.4 MB -> KB per turn | yes | med | careful |
| 902 | TAKE-OPT-IN | H | SERVE | MAYA_DS4_PREREAD: read next turn start after thinking answers | first token 3.3-17.9s->0.4-1.7s | no | med | clean |
| 1194 | TAKE-OPT-IN | H | DS4 | window attention on tensor cores (DS4_ATTN_WIN_MMA) | large window-decode gain | no | med | careful |
| 821 | TAKE-OPT-IN | H | DS4 | cache-aware expert substitution for decode (opt-in) | 3xR9700 decode +24.5% | no | large | careful |
| 959 | TAKE-OPT-IN | H | DS4 | short-prompt least-picked RAM experts on CPU lane (PREFILL_CPU) | agent turn ~-40% PCIe3 | no | med | careful |
| 1236 | TAKE-OPT-IN | H | DS4 | auto layer split weighted by each GPU's host link (HIP default) | 7900XTX 14,13,13->18,11,11 | no | small | careful |
| 1105 | TAKE-OPT-IN | H | HY4 | MTP block no longer skipped for the dense span (TP 256K) | 92->203 tok/s | no | small | careful |
| 1223 | TAKE-OPT-IN | M | GLM | sm_80 prompt FP16 expert path by default | 8K 916->1027 CMP 170HX | no | small | careful |
| 1072 | TAKE-OPT-IN | M | DS4/AMD | DS4_AR_P2P on HIP via hipDeviceMallocUncached (lifts #809 lockout) | RDNA3 TP hang fixed | opt-in | small | careful |
| 1048 | TAKE-OPT-IN | M | GLM/AMD | MFMA wave64 MLA prompt attention for CDNA (opt-in) | MI350 prefill +28%, KL 0.018 | no | small | careful |
| 1065 | TAKE-OPT-IN | M | DS4 | resident EXL3 experts computed on helper during prefill (HELPER_PREFILL) | prefill +4.3-6.1% | no | med | careful |
| 1162 | TAKE-OPT-IN | M | DS4 | TP4/8 pair-halves prompt all-reduce over NVLink pairs | halves PCIe bytes | no (FP16) | med | careful |
| 967 | TAKE-OPT-IN | M | DS4 | DS4_PROMPT_LIGHT: few-row prompt experts on helper instead of copied | links 6.7s->~3s 2K read | no | small | careful |
| 904 | TAKE-OPT-IN | M | DS4 | helper pool takes the DSpark blocks' experts (HELPER_DRAFT) | drafter SSD reads gone | no | small | careful |
| 1132 | TAKE-OPT-IN | M | MM3/AMD | gfx906 FP16 draft-fc fallback (MM3_DSPARK_FC16) + Wave64 markov | MI50 MiniMax 0->20.5 tok/s | no | small | careful |
| 1153 | TAKE-OPT-IN | M | HY4 | HY4_VISION_FP16 vision fallback pre-sm80/gfx906 | vision cos 0.99998 | no | med | careful |
| 889 | TAKE-OPT-IN | M | DS4/AMD | gfx906 hardware v_dot4_i32_i8 index kernel (DS4_INDEX_SDOT4) | MI50 decode +10.8% | no | small | careful |
| 1217 | TAKE-OPT-IN | M | HY4 | mixed-quant GGUF experts on GPU + Q5 Engram (HY4_MIXED_GPU=1) | Mix-128G loads | no | large | careful |
| 305 | TAKE-OPT-IN | M | GLM | retimer may take its own all-PCIe zero plan | 4x3090 CPU-lane pathology | no | small | careful |
| 1002 | TAKE-OPT-IN | M | DS4 | idle settle: hot helper-pool experts into VRAM (SETTLE_IDLE_MS) | helper wait removed | no | med | careful |
| 1033 | TAKE-OPT-IN | L | DS4 | prompt picks steer pool's idle leg (stacked) | none shown | no | small | careful |
| 1034 | TAKE-OPT-IN | L | DS4 | smaller light cut for short chunks (stacked) | +2% short reads | no | small | careful |
| 1003 | TAKE-OPT-IN | M | DS4 | light experts through window kernels (LIGHT_WIN, stacked) | prefill 2K +7% | no | small | careful |
| 1032 | TAKE-OPT-IN | L | DS4 | idle settle third leg: hottest file experts into pool (stacked) | decode +0.5 | no | med | careful |
| 1088 | TAKE-OPT-IN | L | HY4 | runtime refusal-direction projection (HY4F_ABLATE) | n/a | no | small | careful |
| 1046 | TAKE | M | DS4 | Windows: recent-cache fills reuse RAM-store copies | warmed decode +42.9% (1 rig) | yes | med | careful |
| 1017 | TAKE | M | GLM | MAYA_GLM_EXPERTS_BIN aligned single experts file, unbuffered reads | -12% ms/token SSD rig | yes | small | careful |
| 1019 | TAKE | M | GLM | XSTREAM misses as they land + AHEAD_YIELD (stacked) | -5% ms/token | yes | med | careful |
| 1016 | TAKE | M | GLM | AHEAD_MIN_W skips picks the disk skip drops + read knobs | 90->~75 ms/token w/ 1017 | yes | small | careful |
| 1147 | TAKE | M | GLM | shared expert + KDA low-rank projections on side stream | decode gain on EXL3 | yes | small | careful |
| 1148 | TAKE | M | GLM | POOL_FILL: incomplete pool takes its full budget, tail interleaved | fewer evict/refetch loops | yes | small | careful |
| 1157 | TAKE | M | DS4 | peer AR as 2 MiB piece ring when full rows don't fit | VRAM-full pairs keep peer AR | yes | small | careful |
| 1165 | TAKE | M | GLM | PREFILL_P2P NVLink chunk hand-offs on layer splits | 4 boundaries off PCIe | yes | small | careful |
| 1226 | TAKE | M | GLM | GGUF SPLIT_SHARED: shared down on side stream, no IQ2_S stack spill | REG64 STACK8->0 | yes | small | careful |
| 1160 | TAKE | M | IK2/SERVE | tool_choice "required": engine starts= picks among tool names | 2/10 -> 10/10 calls | n/a | med | careful |
| 1130 | TAKE | M | IK2 | IK2_SESSION_CACHE_GB host-RAM session cache for taking turns | same tokens, no reread | yes | med | careful |
| 1142 | TAKE | M | SERVE | loop guard memo (MAYA_LOOP_GUARD_MEMO) | 34->5 ms @1700 msgs | yes | small | clean |
| 1134 | TAKE | M | SERVE | Inkling tool calls stream while the model writes them | minutes->immediate | n/a | med | clean |
| 947 | TAKE | M | DS4 | Engram row cache sized by leftover RAM, 16 GiB cap (was 256 MB) | long-quote hits | yes | small | careful |
| 1087 | TAKE | M | HY4 | HY4_ENGRAM_RAM=1 lock Engram table in RAM (Linux) | 262K/1M ctx no swap | n/a | small | careful |
| 985 | TAKE | M | DS4 | kept slots/prefix carry ablation identity | 12K conv 5.3s->1.0s | yes | small | careful |
| 1115 | TAKE | L | TOOLS | ik2 derisked-checkpoint -> inkling2 GGUF converter + validator | 734/734 tensors | n/a | med | clean |
| 950 | TAKE | M | SERVE | POST /v1/completions raw prompt + FIM markers | 94/127 HumanEval+ fills | n/a | small | clean |
| 922 | TAKE | M | SERVE | DS4 image UUID references (vLLM-compatible) | frame reuse | n/a | med | clean |
| 932 | TAKE | M | SERVE | /slots/export + /slots/import between servers | conv handoff | yes | med | clean |
| 891 | TAKE | M | SERVE | per-request thinking_budget | n/a | yes | small | clean |
| 680 | TAKE | M | SERVE | Responses: Hermes xhigh/summary/encrypted-content + validation | n/a | n/a | med | clean |
| 1050 | TAKE | M | SERVE | vision encoder pinned to a chosen card + setup vision_keep | prompt-cache hits back | n/a | small | adapt |
| 1125 | TAKE | M | SETUP | CUDA setup keeps manual GLM tuning on rerun | n/a | n/a | small | adapt |
| 784 | TAKE | M | DS4 | opt-in strict NUMA placement for locked Engram pages | fixes node-skew loads | yes | med | careful |
| 748 | TAKE | M | DS4/AMD | drain startup peer probes past sampling expiry | no more load fails on VMs | yes | med | careful |
| 872 | TAKE | M | DS4 | fail-closed GGUF prompt-row errors, end engine (no stranded TP peer) | 90s hang -> clean stop | n/a | med | careful |
| 876 | TAKE | M | DS4 | implicit AR2 default still enters peer calibration | V100 TP2 auto-calib | yes | small | careful |
| 901 | TAKE | L | GLM | residual-direction ablation harness (GLM_ABLATE), off unless file | n/a | yes | small | careful |
| 1029 | TAKE | L | DS4 | HC_MIX_ROWS: mHC mixes in one kernel not cuBLAS SGEMM | chunk path | no* | small | careful |
| 1030 | TAKE | M | DS4 | EG_LATE: one-chunk prompts start before all Engram rows land | prefill +3-5% cold | yes | small | careful |
| 1031 | TAKE | L | DS4 | LIGHT_X_ONCE: light-expert rows cross link once (stacked) | prefill 2K +4% | yes | small | careful |
| 1140 | TAKE | L | DS4 | ROUTER_ROWS=2: window router GEMV on one-token kernel | window latency | yes | small | careful |
| 1179 | TAKE | M | CPU | AVX-512 one-token IQ2_S/IQ3 rows for GLM CPU lane (opt-in) | CPU-lane decode | yes | small | clean |
| 1135 | TAKE | M | IK2/AMD | maya-ik2 HIP builds+runs on Windows (R9700) + build script | first Win AMD run | yes | small | careful |
| 1144 | TAKE | M | IK2 | IK2_BATCH_PREFILL batched prompt experts (NVIDIA) | prefill gain, same tokens | yes | small | careful |
| 995 | TAKE | M | CORE | Windows unbuffered expert reads + release mapped pages | >RAM prefill stall gone | yes | small | careful |
| 1209 | TAKE | M | GLM | CPU-lane calibration sample topped to 64 experts | DRAM-real timing | mostly | small | careful |
| 1146 | TAKE | L | GLM | GLM_DUMP_LOGITS test hook (bit-for-bit build compare) | n/a | yes | small | careful |
| 1175 | TAKE | L | HY4 | HY4_LAT16 prompt latents F16 once a layer | prefill | yes | small | careful |
| 1193 | TAKE | L | TOOLS | ds4 pack --drafter-experts 3-bit donor | smaller drafter VRAM | yes | small | clean |
| 941 | TAKE | M | METAL | keep prompt state on PAUSE + opt-in SSD snapshot slots | 60312/60313 tokens reused | yes | large | careful |
| 684 | TAKE | M | DS4/SERVE | report drafts by source (DP line, /metrics) | n/a | n/a | small | careful |
| 854 | TAKE | M | DS4 | accept exactly-representable F16 embeddings (65536-codec test) | packs load | n/a | small | careful |
| 1049 | TAKE | M | kernels | Q5_1 in native MMVQ + BF16 dequant | Q5_1 tensors load | n/a | med | clean |
| 687 | TAKE | L | DS4 | --ablate-sites/--ablate-layers narrow the projection | Huihui packs reproducible | n/a | small | clean |
| 912 | TAKE | M | SERVE | STRATA_SPEC_DEFAULT injects spec tune for plain clients (serve hunks only!) | 71.5->74.4, 40.7->46.0 | n/a | small | clean |
| 636-note | - | - | - | (see above; PR also deletes 3 .github workflows - do not take those) | | | | |
| 569 | TAKE | L | DS4 | DS4_PF_SAMPLES: sample residency before read-ahead hints | fewer wasted hints | n/a | med | careful |
| 1119 | TAKE | L | DS4/SERVE | Prefill VRAM hit tile separate from decode | n/a | n/a | small | careful |
| 820 | TAKE | L | SERVE | MAYA_GLM_NVML=0 NVML reader opt-out | diagnostics | n/a | small | clean |
| 829 | TAKE | L | SERVE | telemetry: close /proc files explicitly | no fd warnings | n/a | small | clean |
| 683 | TAKE | L | SERVE | Windows: FILE_APPEND_DATA log handles | no lost log lines | n/a | small | clean |
| 1210 | TAKE | L | GLM | MAYA_GLM_NO_REUSE parsed via env_flag (0/off = off) | n/a | yes | small | clean |
| 1129 | TAKE | L | SERVE | Inkling "minimal" effort -> template 0.1 | n/a | by design | small | clean |
| 885 | TAKE | L | TOOLS | ref_gguf Engram rows by table type (IQ4_NL/Q5_K) | ref forward fixed | n/a | small | clean |
| 1061 | TAKE | L | TOOLS | ref_gguf --workers N parallel expert dequant | big speedup offline | deterministic | med | clean |
| 933 | TAKE | L | TOOLS | drafter_from_gguf.py from existing DSpark GGUF | no double weights | n/a | med | clean |
| 1174 | TAKE | L | IK2 | name the conflicting tensor pair in resident-fallback log | n/a | n/a | small | careful |
| 1197 | TAKE | M | DS4 | feed_windows: logits for --ppl/--dump-logits | enables window-KL checks | n/a | small | careful |
| 909 | TAKE | M | AMD/HY4 | gfx1200/gfx1102 support + tuning tables + Hy4 context steps | 4x9060XT ~670 PP/s | n/a | med | careful |
| 460 | TAKE | M | GLM | inclusive RAM tier (GLM_RAM_INCLUSIVE): free evictions | decode +6% | yes | small | careful |
| 641 | LATER | - | DS4 | two verify windows in flight on 2-GPU split (draft, stacked on 636) | claimed pipeline win | no | huge | careful |
| 642 | LATER | - | GLM | paged latent cache + multi-seq scheduler + prefix store | n/a | mixed | huge | careful |
| 812 | LATER | - | DS4 | DS4_EP expert parallel over idle links on 4-GPU relay (draft) | appends -19% 4x3090 | no | med | careful |
| 847 | LATER | - | SERVE | web EN/zh localization | n/a | n/a | large | careful |
| 855 | LATER | - | SERVE | CUDA checkpoint sleep deep_sleep.py (Linux NVIDIA) | VRAM freed while idle | n/a | large | clean |
| 856 | LATER | - | DS4 | native GBNF constrained decoding (build opt-in) | n/a | n/a | large | careful |
| 879 | LATER | - | GLM/AMD | CUDA engine + HIP expert helper, mixed vendor, guided setup | RTX5090+8060S together | n/a | huge | careful |
| 962 | LATER | - | NEW | maya-k31 Kimi K3.1 engine (off by default) | 0.88-1.0 tok/s 3x3090 | n/a | huge | clean |
| 991 | LATER | - | GLM | FlashStates conversation-state cache (opt-in) | agent turns resume | n/a | huge | careful |
| 994 | LATER | - | GLM | opt-in RAM/disk slot spill for interleaved agents | cold-slot resumes | yes | med | careful |
| 1023 | LATER | - | SERVE | GLM-5.3 KV streaming opt-in in Models page | n/a | n/a | small | adapt |
| 1028 | LATER | - | SERVE | users: SQLite accounts, roles, per-user keys | n/a | n/a | large | careful |
| 1052 | LATER | - | GLM | GLM-5.3 KV streaming: indexer-group maps, identity window | split resolves 79->22 | n/a | large | careful |
| 1208 | LATER | - | IK2 | Q8_32 streaming KV + Quest sparse + GQA3 + Heat-LFU (multi-topic) | 3060/3070 long ctx | mixed | huge | careful |
| 1215 | LATER | - | GLM | two-rig split p1: link + roles (--role front\|back) | 24 KB/token hop | yes | large | careful |
| 1216 | LATER | - | GLM | two-rig p2: pipelined MTP over the link (stacked) | link-latency decode | yes | large | careful |
| 1228 | LATER | - | GLM | two-rig p3: --parallel over the link (stacked) | 2-8 conversations | mixed | large | careful |
| 1224 | LATER | - | NEW | maya-ling engine (Ling-3.2-air / bailingmoe3, draft) | parity-verified base | n/a | huge | clean |
| 863 | SKIP | - | kernels | Q8_0 Qwen n-gram (PLE) mmap tables | n/a | - | small | no: Qwen path removed |
| 897 | SKIP | - | DS4 | store layout-sized backing | - | - | large | no: superseded by #1045 |
| 910 | SKIP | - | DS4 | docs row for DS4_FLOW_COPY_DMA | - | - | small | no: setting not in our tree |
| 963 | SKIP | - | DS4 | store size classes v1 | - | - | med | no: superseded by #1045 |
| 984 | SKIP | - | MM3/AMD | RDNA4 MM3 MMQ uplift (draft) | own A/B -22.9% | no | small | no: regresses |
| 990 | SKIP | - | SETUP | GGUF setup + Qwen quant support mega-PR | - | - | huge | no: Qwen removed; setup targets upstream menus |
| 996 | SKIP | - | CPU | GLM_LANE_ROW4=auto | - | - | med | no: row4 kernels not in our tree |
| 1104 | SKIP | - | Qwen | Qwen RoPE/YaRN native controls | - | - | large | no: Qwen path removed (port to QFN later) |
| 1133 | SKIP | - | IK2 | 8 block types + CPU lane + deadlock (v1.1.73 base) | - | - | large | no: ours/superseded (#1219, SAGUARO_IK2_CPU, ik2_pascal) |
| 1170 | SKIP | - | docs | RTX 5060 Ti ik2 bench report | - | - | docs | no: docs/data only |
| 867 | MARK | - | DS4 | EXL3 lend in place | - | - | - | already in our tree |
| 1204 | MARK | - | DS4 | AVX1 IQ rows for Sandy/Ivy CPU lane | decode +13% | yes | - | already in our tree |
| 1225 | MARK | - | GLM | IQ2 FP16 dequant 2x ggml | 8K 1026->1074 | yes | - | planned (per plan) |
| 1233 | MARK | - | IK2 | RTX 20 Turing flash + GPU1 experts + NVLink | up to 7x prompts | mixed | - | taken (per plan) |

(641/1003/1031/1032/1033/1034 are stacked chains: take in order 967 -> 1003 -> 1031 and 1002 -> 1032 -> 1033;
1215 -> 1216 -> 1228 likewise; 636 before 641.)

## Top 10 to take next

1. **#1231** GLM DSA indexer fused - +31% decode at 500K context, same cells, small hook in `fast_dsa`/window; our
   GLM window path is ours so port carefully, but it is the single biggest exact long-context gain in the set.
2. **#630** GLM route-in-one-pass - bit-exact decode win on every GLM card; depends only on already-merged #486/#451.
3. **#712** DS4 Q5_K IMMA GEMVs - opt-in, bit-identical, +16.9% decode on consumer NVIDIA (our T4/4090 boxes).
4. **#952** Pascal kernels - matches our Pascal recipe (STRATA_EXPERIMENTAL_SM60); +5-9% decode, -20-39% first token.
5. **#1045** DS4 store per-expert slots - +24% cache capacity, decode +10%, prompt +12%, answers unchanged; replaces
   the two older store PRs (#897, #963) - take only this one.
6. **#960** GLM DEMOTE/ADMIT=all - +21% decode on spilled GLM, same bits, small host-side change.
7. **#1219** IK2 six block types - makes UD-IQ3_XXS load on 16 GB cards; small, well-tested, supersedes #1133's types.
8. **#1166** GLM prompt-lookup drafts - all drafts verified, big agent-decode win on repeat-heavy text.
9. **#1024** pinned-host huge pages - one allocator fixing compaction stalls for GLM/MM3/DS4; unset = unchanged.
10. **#1222** Hy4 HY4_PSTREAM - 1.3-1.8x prompt reading on VRAM-short Hy4 boxes; small, Strata-proven scheme.

Near-misses: #1145 (sampler, both engines, hash-exact), #953 (CPU-lane affinity, +9.8%), #1192 (IK2 parallel
fairness, huge), #1236/#1105 (AMD/Hy4 defaults - KL-check them), #821/#1194 (big opt-in, KL-check first).

## PRs touching our heavily modified areas - careful porting

Our divergence concentrates in: the **GLM CPU lane / fast path / window & shadow-cache code** (`src/core/glm_*`,
`glm_fast_state.hpp`), **DS4 reader/prefetch/NUMA + store + parallel serve** (`src/ds4/ds4_model.cu`, `ds4_os.hpp`,
`ds4_store.hpp`, `ds4_main.cpp`), **MM3** (`src/mm3/*`), **AMD hip_compat/hip kernels**, plus our own **IK2 CPU
lane/Pascal/WARM** and branded **serve/web + server.py**.

- GLM fast path / window / CPU lane: #305, #460, #630, #901, #952, #960, #971, #1016-1019, #1054, #1146-1148, #1165,
  #1166, #1209, #1226, #1231 (and the LATER set #642, #991, #994, #1052, #1215/1216/1228 - #1052 needs a rewrite
  because `src/core/glm_dsa_mixer.cu` does not exist here).
- DS4 reader/prefetch/NUMA/store/parallel: #569, #636, #748, #784, #821, #876, #897/963/1045, #904, #953, #959,
  #1002-1034, #1046, #1065, #1072, #1117, #1140, #1157, #1158, #1162, #1236.
- DS4 protocol/engine + our serve changes together: #684, #902, #932, #1141, #1160 - engine and serve sides must
  land as a pair over our `server.py`.
- MM3: #896, #909 (mm3_os.hpp), #1132; (#984 rejected).
- AMD hip_compat / hip kernels: #889, #909, #1048, #1072, #1132, #1135, #1153, #1236.
- IK2 (our CPU lane `SAGUARO_IK2_CPU`, `ik2_pascal.cu`, `SAGUARO_IK2_WARM`): #1133 is ours/superseded; #1135, #1144,
  #1174, #1192, #1208, #1219 must not collide with our lane - re-test lane parity after each.
- serve/web & server.py (our UI is rebranded/modified): #680, #683, #820, #829, #847, #855, #891, #912, #922, #932,
  #950, #985, #1050, #1125, #1129, #1134, #1142, #1160, #1210.

Gates per `ref/UPSTREAM.md` §6 for every take: T4 SASS/output-identity spot check for "same tokens" claims; KL vs
previous release on our T4s for anything flagged answer-changing (our TAKE-OPT-IN list); Q8_0 stays exact.
