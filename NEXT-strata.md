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

## Engine hooks (ik2) - planned
- Shards: a .strata path -> InklingView (info/data/get/read_experts); read_into_strided call sites -> S.read_experts.
- k::block_values/bytes/decode_value/decode_f16/decode_host: + strata_fmt types.
- kMIX: per-expert type resolution in the expert kernels (MixHdr), MACaQ applied to h in the gate_up kernel.

## Status
- see git log
