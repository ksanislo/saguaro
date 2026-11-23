# Strata packs (.strata) in saguaro-ik2 - branch saguaro/strata-format (worktree /data/llm/saguaro-strataformat)

Base: saguaro/ik2-up 166a08add (coordinator: build on ik2-up, merge it in periodically; do not edit its worktree).
Assessment: /data/llm/logs/saguaro/STRATA-FORMAT-assessment.md. Pack: /opt/llm/models/Inkling-2-Mini-Strata-SQ-15/
(sha256 ok). Strata v0.2.19 source (read-only): /data/llm/strata-v0219.
Decision (coordinator 11-22): no partial converter; native support. Python tools/strata/ = reference decode + map.

## Design
- src/strata_fmt/ (shared, model-independent reader + codecs; Hy4 SQ packs later):
  strata_pack.{hpp,cpp} container/manifest(zstd)/directory, mmap; sf_json.h (Strata's parser);
  sf_codecs.h host/CUDA/HIP block types: kSH2..6 (half S_K super-block, 128 values, lossless re-layout), kQ8F
  (Q8_ROW as 32 x int8 + F32 row scale), kBF16 (30), kMIX (per-expert codec: MixHdr per role part, MACaQ s[F]
  FP16 after the up part).
  strata_inkling.{hpp,cpp}: InklingView = the pack as the master-layout GGUF ik2 reads (names, keys, types).
  All-SF4 layers -> NVFP4 + .scale (same bytes as UD-Q4_K_XL); else kMIX.
- tests/strata_fmt_test.cpp: every value vs Strata's scalar CPU decode (ported). Trunk 1.6 G values bit-equal.

## Engine hooks (ik2) - the points the ik2-up rebase must keep
- ik2_main.cu Shards: a .strata path -> InklingView (info/data/get/all/read_experts); g_sv; tensor_bytes(kMIX).
  The two read_into_strided expert sites -> S.read_experts. host_mmap / pin_later off for a pack.
- ik2_main.cu Mat::seg (MatSeg): a fused dense matrix of mixed block types; mat_rows/mv1/mv1c/matmul/tile_rows loop
  the parts; try_load_fused_raw builds them for a pack; the offload path's blocks() check accepts them.
- ik2_main.cu prompt path run_exp: kMIX decoded part by part, swiglu_cap_f32(..., mix_up) for MACaQ. TP: refused.
- ik2_kernels.cu: decode_value (strata types + kMIX), block_values/bytes, mix_role/mix_inv at the top of the
  namespace, kMIX resolution at entry of experts_gate_up_swiglu / down / _g / _mem kernels, row_dot / row_dot_2
  chunk paths (sf_chunk_dot), swiglu_cap_f32 mix_up.
- CMake: strata_fmt library (zstd), strata_fmt_test, strata_fmt_gpu_test; ik2_kernels includes src/strata_fmt.

## Status (11-22 ~16:10 PST)
- CPU exactness (strata_fmt_test, all 3072 experts + 646 trunk tensors): PASS, 26.07 G values, 0 mismatches
  (SF4 E2M1 -0 read as +0 by ggml's table: 423 M codes, same weight). Log /data/llm/logs/saguaro/strata-fmt-test-full.out.
- GPU (T4, runs/sf1): strata_fmt_gpu_test PASS (trunk 2.03 G values bit-exact, experts exact, token MIX kernels
  2.7e-4 rel vs double ref). Engine load failed: offload path needs fused dense blocks -> Mat::seg (runs/sf2 queued).
- mortar: HIP build clean; SQ-15 downloaded + sha ok; runs/sfm1 (gpu test, 1-card, KL, bench) started.
- MIX padding: slot = the layer's largest expert (SF4 / S5_K) -> 13.0 GiB of experts vs 11.4 GiB packed; on a T4
  that means the offload path, not resident. Fix = per-expert slot classes in the cache (follow-up, touches the
  cache pools the rebase moves).

## Results 11-22 evening (details: /data/llm/logs/saguaro/STRATA-FORMAT-assessment.md section 6)
- T4 (runs/sf2): SQ-15 loads and runs on one T4 (offload path, 2504/3072 experts in VRAM). KL vs UD-Q4_K_XL 0.0153,
  top-1 94.0 %. Decode 24.4 / 23.2 tok/s (~0 / 8K), 8K read 348 tok/s (UD same build 27.1 / 30.4, 535).
- RX 9070 XT (mortar, HIP, SAGUARO_IK2_OFF=1): gpu test PASS, KL 0.0152 / 94.05 %, 1-card check PASS,
  serve_check 20/20 PASS (runs/sfm4, merged tree), decode 55.9 / 50.5 / 46.3 tok/s at ~0 / 8K / 30K, reads 1275 / 1033.
- T4 runs/sf3 (fast FP16 decode): 1-card check PASS, decode 24.9 / 23.8 tok/s, 8K read 553 tok/s (was 348); ik2_kernels_test all passed.
- Next: per-expert slot classes (kMIX padding: 13.0 GiB vs ~11.6 packed - would make SQ-15 resident on 15-16 GB
  cards); MX8 experts (SQ-20); serve config entry for the pack (tokenizer dir = any Inkling tokenizer).
