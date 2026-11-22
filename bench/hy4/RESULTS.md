# saguaro-hy4 on 4x T4, Hy4-Flash-Maya-S-IQ2_S: measurements of 2026-11-21

Harness: bench/hy4/run.py (one load, greedy requests, 200 tokens each; raw lines in results-20261121.jsonl; logs in
build-hy4/bench/logs/, not committed). Prompts: "short" 47 tokens, "10k" 10,059, "60k" 60,059, "v1" 40. Context
131072. One run each unless a pair is shown; another agent's builds ran on the box at times, so treat differences
under ~3% as noise. Nothing here is a KL-validated quality claim except the KL table.

| setting (each row = one change from the row above it unless said) | decode short | decode 10K | decode 60K | read 10K | read 60K | load | experts in VRAM |
|---|---|---|---|---|---|---|---|
| first-load config: layer split, chunk 4096, clocks UNLOCKED | 7.6 | 7.3 | - | 153 | - | 50 s | 70% |
| + GPU clocks locked (llm-gpu-clocks lock) | 15.4 | 16.5 | - | 277 | - | 27 s | 70% |
| layer split, chunk 1024 (lane nearly empty) | 18.4 | 17.8 | - | 246 | - | 20 s | 97% |
| layer split + --mtp (drafts kept 67-75%) | 16.6-22.1 | 20.6 | - | 127 | - | 57 s | 60% |
| --tp 4, chunk 4096, experts placed by index | 20.8 | 20.4 | - | 568 | - | 22 s | 71% |
| --tp 4, chunk 4096, placed by the learned scores (HY4_HOT=1 + file) | 30.3 | 30.1 | - | 561 | - | 22 s | 71% (94.6% of routes) |
| --tp 4, chunk 1024, learned placement | 33.7 | 33.0 | - | 486 | - | 20 s | 88% (98.9%) |
| --tp 4, chunk 2048, reserve 0.3 | 33.0-33.2 | 32.5 | - | 618 | - | 20 s | 87% (98.6%) |
| **--tp 4, chunk 2048, reserve 0.5, learned placement (the config)** | 32.5 | 32.1 | 31.4 | 589 | 459 | 20 s | 85% |
| the same + --mtp (chunk 1024, reserve 0.3; drafts kept 62-73%) | 32.3 | 34.2 | - | 383 | - | 27 s | 83% |

Decode in tok/s (200 generated), read in prompt tok/s. --kv auto takes the 8-bit cache rows by itself under tensor
parallel with the CPU lane (INFO the latent cache in 8-bit rows), so every --tp 4 row above is 8-bit rows.

A/B pairs on the config (ABAB loads): HY4_IX_FORK=1 + HY4_ATTN_FORK=1: decode +1.7% short, +1.7% v1, +0.3% 10K (33.80 /
33.55 vs 33.00 / 33.02 short) - exact, inside the noise band, NOT set. --cpu-threads 9: -1%; 4: -5%; default (19) kept.
DS4_IQ_PERSIST: not built into this engine's ds4_iq copy and only for codebooks of 8 KB up - nothing to test here.
--dfly2: not loadable on this layout (needs layers 25-41 and the head on one GPU, no tensor parallel).
--reserve-gb 0.3 with chunk 4096 + --mtp failed at warm-up ("the light sets' scratch (64 MB) failed"): 0.5 is the floor used.

Where a token's time goes (--tp 4, HY4_TP_PROF / HY4_TP_EV, 1-token windows): 28-29 ms to the last sum + 2.0 ms the
output head on GPU 0. Member 0's GPU: attention 14.0 ms, FFN 13.1 ms, the two sums' waits 2.2 ms, post-mixes 0.55 ms.
The host's sums 0.6-0.7 ms. Cards draw ~45 W in decode (70 W cap in reads): the token path is a chain of ~40 short
kernels a layer a member, not compute- or bandwidth-bound. Layer split (HY4_PROF, 300 tokens): the Q6_K attention
GEMVs are 60% of a token and run at ~200 GB/s of weights read - bandwidth-bound on one card, which tensor parallel splits.
Decode does not fall with context (short / 10K / 60K above); the first-load report's 4.0 tok/s at 10K was unlocked
clocks plus a cold Engram table.

KL against layer split F16 (bench/hy4/kld_seed.py, 8 x 512 tokens of the docs, 2040 scored positions):

| setting | mean KL | p99 | top-1 same | PPL (ref 31.89) |
|---|---|---|---|---|
| layer split F16 again | 0.000001 | 0.00008 | 99.95% | 31.89 |
| layer split, --kv q8 | 0.0577 | 0.597 | 88.6% | 32.64 |
| --tp 4, --kv f16 (two loads) | 0.0559 / 0.0564 | 0.531 | 89.7% | 33.15 / 33.24 |
| --tp 4, 8-bit rows, chunk 2048 (the config) | 0.0604 | 0.550 | 88.2% | 32.20 |

OPEN: tensor parallel with F16 rows differs from the layer split as much as 8-bit rows do. Not explained yet (the
prompt chunk's partial sums go through the host in F16 - HY4_TP_F32=1 makes them F32; or this 2-bit quant amplifies
last-bit differences). The floor arms were queued, not run: see NEXT-hy4.md item 1.

## KL floor arms (2026-11-21 ~18:30 MST, run after the hand-back; vs layer split F16, 2040 positions)
| Arm | Mean KL | Top-1 same | PPL (ref 31.89) |
|---|---|---|---|
| Layer split, half the experts on the CPU lane (--vram-experts 0.5) | 0.0581 | 89.4% | 32.02 |
| Layer split, --chunk 1024 | 0.0586 | 89.6% | 31.97 |
| --tp 4 --kv f16, F32 sums (HY4_TP_F32=1) | 0.0566 | 89.0% | 32.76 |
Any change of rounding order (chunk size, CPU-lane share) moves this quant by KL ~0.057: that is the floor, and
tensor parallel (0.056) and 8-bit cache rows (0.058-0.060) sit on it. Nothing to fix; --tp 4 with 8-bit rows stays.

## Phase 3 (2026-11-22, build-hy4b, --tp 4 chunk 2048 reserve 0.5 learned placement, 131K, clocks locked)
| run | result | decision |
|---|---|---|
| saved conversations, --tp 4 (slot_check.py) | ALL PASS: 10K saved 0.3 s (0.36 GB), read back 0.5 s, 1.0 s after a restart | ON (HY4_SLOT_DIR, HY4_SLOT_GB 16) |
| saved conversations, layer split | PASS with a fixed placement (HY4_HOT=0); the one FAIL was a learning placement moving experts between requests | - |
| HY4_TP_HEAD_MEMBER=1, 2 ABAB pairs x2 | decode +6.5 / +6.4 / +5% and +5.9 / +6.1 / +4.6% (short / 10K / v1) | md5 differs from off (41 experts fewer in VRAM); exactness test (=2 arm) held |
| HY4_CREW_PIN=1 / HY4_LANE_SPIN_US=0 vs base-3 | +0.9 / +0.1 / +0.3% and -0.7 / -1.0 / -1.2%, md5 equal | not set |
| HY4_VISION_F16=1 (server, the test picture, 2 pairs) | picture prompt 3105 -> 2679 ms and 2814 -> 2673 ms; descriptions right on both paths | ON |
| HY4_TP_EV=all, 1-token windows | member 0's FFN 12.2 ms vs 8.3 on members 1-3 (they wait 5.1 ms at its sum): the CPU lane's experts | lever: residency |
| --max-context 262144 | 75% experts in VRAM; decode 30.5 / 30.9 (-6 / -4%), 10K read 457 (-22%); the host RESET during the 60K read | 131K stays; 262K+ held |
