# DIVERGENCE: how Saguaro differs from upstream Project Maya / Strata

The register docs/UPSTREAM.md §3 asks for: per engine, each of our modifications - what, why, where (files/functions),
its switch, its measured gain, and whether upstream could take it - plus what we drop from upstream on purpose.
Resolve an intake conflict by reading the entry for the file it touches, not by guessing. Every topic branch adds its
line when it merges.

Entry format (2-4 lines): **what** and **why**; **where** (`file:function`) and **switch + default**; the **measured
gain** with the line of the source that states it, or "not measured"; and whether the change is **exact** (same tokens
/ bit-identical) or **answer-changing**. Numbers are quoted as written in the source (ref/PERF-LEDGER.md,
ref/ROLLBACK.md, ref/live-stack-notes.md, ref/reports/REPORT-*.md); nothing here is re-derived.

Hardware note: every gain below is on the machines in those sources - four Tesla T4s, an RX 9070 XT
or a Tesla P40. Long QFN/GLM reads on the T4s are power-capped at 70 W/card, so a kernel that is
faster in isolation can be flat there (ref/PERF-LEDGER.md:456-465).

QFN (engines/qfn) is our own port of Strata, carried as its own CMake sub-project; the Maya engines (GLM, DS4, MM3,
HY4, IK2) are upstream Project Maya with our changes on top. Nothing here records a Strata sweep: the last swept
Strata commit is not yet written down in this register (docs/UPSTREAM.md §5 asks for it) - the port base in the tree
is the subtree of the live Strata tree 82c389f5e + server 0c800169d (ref/live-stack-notes.md:65).

## GLM

The upstream GLM-5.3-Flash engine (saguaro-glm) with our parallel-conversation, CPU-lane and VRAM work on top.
Biggest first.

1. **Turbo: drop N of the routed experts while decoding.** Why: on a card whose experts stream from RAM/SSD the decode
   is bound by moving experts, not by arithmetic. Where: the engine's `res_k` route limit in
   `src/kernels/cuda/glm_fast.cu` (`SAGUARO_GLM_RESIDENT_TOPK` read there) and the per-request path in
   `src/core/glm_window.cu`; the request key comes from `serve/turbo_live.py`.
   Switch and default: per request `"turbo": {"drop": N}` (off); engine-wide `SAGUARO_GLM_RESIDENT_TOPK` default
   0 = off, 5 in the AMD KL runs (= drop 3-of-8, ref/reports/REPORT-9070xt-glm.md:20); decode
   only, prompt reads stay exact. Gain (RX 9070 XT): Maya-L +66 % (8.45 -> 13.85 tok/s), Maya-S +29 %
   (ref/reports/REPORT-9070xt-glm.md:27-28). Answer-changing: KL 0.192 (5x the engine's own floor) on Maya-L, and
   GPQA-Diamond lost no accuracy (70.7 % vs 66.7 %, p 0.29) but answers loop more (13 -> 21 cut-offs;
   ref/reports/REPORT-9070xt-glm.md:46, 132-149).
2. **Per-conversation shadow LRU caches with background expert moves.** Why: a conversation's next turn wants the
   experts its own earlier turns used, which the global LRU drops. Where: `conv_shadow_set()` and
   `Glm5Model::fast_boundary()` in `src/core/glm_fast_path.cu`. Switch and default (the live "sh50" set,
   ref/ROLLBACK.md:256): `SAGUARO_GLM_CONV_SHADOW=1` with `SAGUARO_GLM_CONV_EXPERTS`,
   `SAGUARO_GLM_CONV_PREFETCH_RATE=50`, `SAGUARO_GLM_TAB_ZC=1`, all off by default.
   Gain: solo +12.9 %, the switch (A resuming after B) +4.8 % on its first 256 tokens, beside +1.7 %, n2 +0.9 %,
   long late -0.1 % (ref/PERF-LEDGER.md:849-852). Answer-changing at the placement level (KL 0.0087 vs a placement
   reference 0.0105); no dropped compute.
3. **AVX1 multi-token CPU-lane rows (IQ2_XXS/XS/S, IQ3_XXS/S, Q5_K, IQ4_XS).** Why: our live host has AVX but no AVX2,
   and ggml's dots waste it. Where: `iq128_rows()` in `src/kernels/cpu/iq_avx1.cpp`, the shared kernels in
   `include/maya/kernels/cpu/iq_avx1_kern.inc` (also DeepSeek's lane and hy4). Switch and default: on where the CPU is
   AVX1-only; `SAGUARO_NO_IQ128=1` keeps ggml's dots (`src/kernels/cpu/native_expert.cpp:118`).
   Gain: ~+10 % (ref/live-stack-notes.md:73). Exact (the 931-case test says 0 differ from ggml's dot).
4. **Prompt CPU-lane gate 2.0 -> 1.** Why: with the old gate the lane refused prompt experts on PCIe that was only
   ~1.4x slower than computing them. Where: the `pf_cpu_gate` lambda in `Glm5Model::prefill_half`,
   `src/core/glm_prefill.cu`. Switch and default: `SAGUARO_GLM_PREFILL_CPU_GATE`, default 2; live 1.
   Gain: long reads 597 -> 650 tok/s (+8.9 %), J/read -8.6 %, n2 decode flat (ref/PERF-LEDGER.md:39); host energy
   +-0 % - the lane burns what the cards save (ref/PERF-LEDGER.md:495). Exact (a planning choice).
5. **RAM-tier row halves (each RAM slot's rows split across the two NUMA nodes).** Where: `numa_pinned()` /
   `row_halves_of` in `src/core/glm_fast_path.cu`. Switch and default: `SAGUARO_GLM_ROW_HALVES=1`, off by default,
   on in the live config since 11-03 (ref/ROLLBACK.md:128-132). Gain: +5..10 % decode
   (ref/live-stack-notes.md:73). Exact.
6. **Prompt read chunk 4096 (was 2048) and the graded-chunk read ramp OFF.** Why: with a CPU lane the fill/drain
   bubble at the start and end of a solo read costs more than the smaller chunks save. Where: `prefill_mode()` (the
   chunk knob) and the ramp in `Glm5Model::prefill_run`, `src/core/glm_prefill.cu`.
   Switch and default: `SAGUARO_GLM_PREFILL_CHUNK` (live 4096),
   `SAGUARO_GLM_PREFILL_RAMP=0` + `SAGUARO_GLM_PREFILL_RAMP_MIN=64` (live; ramp on with min 64 is the upstream
   default). Gain: chunk 4096 -13 % on a 65K read, decode -1 % (noise); ramp off -9 % / -14 % / -24 % at
   65K / 22K / 11K, paid by a short request landing mid-read waiting 38.8 vs 32.4 s to its first token
   (ref/PERF-LEDGER.md:210, 247; accepted losses ref/PERF-LEDGER.md:193-194). Exact.
7. **Even layer split 11,22,33 instead of the auto 14/12/10/9.** Where: `Glm5Model::load_pack_place`,
   `src/core/glm_model.cu`. Switch and default: `SAGUARO_GLM_SPLIT` (live "12,23,34" since 11-18 with the tail thread;
   0 forces one part). Gain: 65K read -10 %, decode same, idle even 15-25 % (ref/PERF-LEDGER.md:205). Exact.
8. **Prestage 64 expert slots on a layer split (off by default there).** Where: the prestage ring in
   `Glm5Model::prefill_setup`, `src/core/glm_prefill.cu`. Switch and default: `SAGUARO_GLM_PRESTAGE=64`
   (live; default 0 when the layers are split). Gain: 65K 88.3 vs 92.1 s, 22K 38.7 vs 41.4, 11K 23.3 vs 26.0, solo
   decode -1.6 % (inside 2 %, and later judged likely noise - the pools are identical at 0 and 64)
   (ref/PERF-LEDGER.md:245, 192). Exact.
9. **Load: the RAM arena faulted in by many threads, and the cards' pins unchained.** Where: `arena_par_threads()` /
   `numa_pinned()` in `src/core/glm_fast_path.cu`. Switch and default: `SAGUARO_GLM_ARENA_PAR=1` (code default off,
   on in the live config; `SAGUARO_GLM_ARENA_THREADS` for the count). Gain: load 141-157 -> 81 s
   (ref/PERF-LEDGER.md:418, ref/ROLLBACK.md:28). Exact.
10. **Slot save/take for all parts at once, through a reused pinned buffer, written in the background.**
    Where: `slot_fast_on()` and the slot writer in `src/core/glm_fast_path.cu`, the file writes in
    `src/program/generate.cpp`. Switch and default: `SAGUARO_GLM_SLOT_FAST=1` with `SAGUARO_GLM_SLOT_FAST_PIN_MB`
    (4096) / `_QUEUE_MB` (8192), off by default, on in the live config (ref/ROLLBACK.md:27).
    Gain: small saves 115-168 ms -> 7 ms warm, 0.15 GB in 45 ms, ~0.9 s/GB off the switch's time-to-first-token
    (ref/PERF-LEDGER.md:419-420). Exact (a first try was rolled back for breaking the elastic release, not for the
    slot path).
11. **The CPU lane's pool at 20 threads and only on CUDA0/3 for prompts.** Why: the lane saturates past 16 and its
    copies contend with the cards that stream experts. Where: `Glm5Model::fast_cpu_lane_setup` and
    `prompt_cpu_dev_ok()` in `src/core/glm_prefill.cu`. Switch and default: `SAGUARO_GLM_CPU_LANE=20` and
    `SAGUARO_GLM_PREFILL_CPU_DEVS=0,3` (live; all cards and 16 before), timings kept in
    `/opt/llm/var/<model>/cpu-lane-cal.txt` via `SAGUARO_GLM_CPU_CAL`.
    Gain: 2-chat decode +2.15 % from 16 -> 20 threads (reads -0.4 %); DEVS=0,3 long reads 654 -> 674 tok/s (+3.1 %),
    host J -4.1 % (ref/PERF-LEDGER.md:43, 41). Exact.
12. **Spec decode: the MTP tail enqueued on a second host thread (with split 12,23,34).** Where:
    `Glm5Model::decode_spec` in `src/core/glm_fast_path.cu`. Switch and default:
    `SAGUARO_GLM_SPEC_TAIL_THREAD=1` (off by default, on live since 11-18; ref/ROLLBACK.md:257).
    Gain: solo +2.7 % (under the +3 bar), beside +4.6 %, 65K read -5.4 % (split of that test not given in the sources here) - shipped at 12,23,34
    where reads were fixed (+0.6 %) and the decode gain fell to +1.2 %, on Ken's "ship the small wins that are ready"
    (ref/PERF-LEDGER.md:317-328). Exact, KL at the floor.
13. **Several conversations at once with elastic contexts, reads beside a decode, and checkpoints across chats.**
    Where: `Glm5Model::seq_alloc_arenas` (`src/core/glm_multi.cu`), the elastic budget in `src/program/generate.cpp`,
    `prefill_begin()` in `src/core/glm_prefill.cu`. Switch and default: `--parallel N` sets
    `SAGUARO_GLM_SEQS` (+ `SAGUARO_GLM_SEQ_CTX`), `SAGUARO_GLM_ELASTIC=1`,
    `SAGUARO_GLM_PREFILL_BESIDE=1024`, `SAGUARO_GLM_MULTI_CKPTS=1`; elastic and beside are off under
    `--kv-streaming` (ref/live-stack-notes.md:73-84). Gain: 8 conversations of up to 262K on four T4s, a prompt read
    beside running chats; the A/B numbers are in the upstream-intake ledgers, not measured as one change here.
    Exact per conversation.
14. **The tier fix (an expert promoted twice) plus a verifier, and the shadow-move audit.** Why: our shadow /
    `CONV_PREFETCH` moves could make a promoted expert be computed with another expert's weights.
    Where: `fast_boundary()` in `src/core/glm_fast_path.cu` (c38b9747). Switch and default:
    `SAGUARO_GLM_TIER_VERIFY=1` is a debug check, off by default (ref/PERF-LEDGER.md:67-68).
    Gain: no speed claim (a correctness fix); with the verifier on, 0 stale / double / wrong entries on all cards.
15. **Upstream's GLM decode graphs, pinned OFF on our live profile.** Where: `kda_graph_wanted()` in `src/core/glm_fast_path.cu` and
    the capture machinery in `src/core/glm_layer_graphs.hpp`. Switch and default: upstream v1.1.75 turns them on by default on NVIDIA; our
    GLM config sets `SAGUARO_GLM_KDA_GRAPH=0` (ref/live-stack-notes.md:26-27). Gain: measured flat on T4 -
    14.8-16.5 tok/s in both arms - and prefetch / lookahead switch them off anyway, and our config enables
    `CONV_PREFETCH` (ref/reports/REPORT-upstream-v1.1.75.md:150-157). Exact either way.
16. **Not taken, and the near-misses we keep warm.**
    - The shared CPU-lane pool's NUMA pinning (v1.1.77) stays off when the lane would leave fewer than 4 physical cores free
      (`lane_pin_cpus`, src/core/glm_fast_path.cu): 20 lane threads on 2 x 10 cores leave one core for four GPUs' service
      threads; measured 2026-11-25 on 4 x T4: decode 7.1 tok/s pinned, 16.7 not, 16.3 on v1.2.0. `SAGUARO_GLM_CPU_PIN=1` forces it.
    - The FP16 tensor-core prompt path (`STRATA_GLM_FP16TC` on branch pr/glm-fp16tc, not in this tree): reads
      +2.0..2.6 %, n2 -1.7 %, KL at the cap (0.0101 vs off 0.0094, every on-arm below every off-arm), no site carries
      the shift at n=1 - parked (ref/PERF-LEDGER.md:271-281, 540-548).
    - `SAGUARO_GLM_MULTI_BOUNDARY=8`: n2-n4 +4.0 % but beside -4.5 % and mean TTFT +1.5 % FAIL (ref/PERF-LEDGER.md:836-838).
    - Upstream's GLM "same answers on every start" was **not ported**: it needs four earlier upstream mechanisms
      (~a day) and costs prompt reads on every request (upstream measured 2-3 % on 3x V100, ~5 % on a Radeon APU;
      our cost not measured) for reproducibility, not speed (ref/reports/REPORT-upstream-v1.1.75.md:11-21).
    - Upstream's GLM tensor parallel was **not taken**: it needs every expert slice in VRAM, which GLM-5.3 does not
      fit on four T4s, and it is built on glm-dsa code we do not carry
      (ref/reports/REPORT-upstream-v1.1.75.md:104-105).

## DS4

The DeepSeek-V4.1-Flash engine (saguaro-ds4). Its switches are named bare below (`DS4_*`): the engine reads
`SAGUARO_DS4_*` first only where it goes through `ds4_env` (`src/ds4/ds4_model.cu:15715-15718`), and the live configs
write the bare form. Biggest first.

1. **The read helper: the idle card on the other NUMA node stages and computes a share of a prompt's streamed
   experts, with the prompt buffers lent to the expert VRAM between prompts.** Why: with 10x4 even layers two cards
   sit idle during a long read, and the read's buffers crowd the experts out of VRAM.
   Where: `read_help_pairs_env()` / `read_help_setup()` in `src/ds4/ds4_model.cu`, `read_help_borrow_env()` and
   `alloc_chunks()` (the borrow), `src/ds4/ds4_borrow.hpp`. Switch and default: `DS4_READ_HELP=0:2,1:3` with
   `DS4_READ_HELP_SHARE=0.5`, `DS4_BORROW=1`, `DS4_FAT_LENT=1`, `DS4_READ_HELP_RING` (all off by default; live since
   11-10). Gain: read -11.7 %, decode +3..11 % vs live, 7.2 GB more experts in VRAM from BORROW alone
   (ref/PERF-LEDGER.md:249, 208). Output shift of the config class (KL 0.0115 vs the all-FP16 reference, live's own
   0.0126; a routine `--parallel 5` shift is 0.022).
2. **The background prompt reader: a long prompt read on its own threads and streams beside running answers.**
   Why: before it, a chat beside a 61K read got 0.14 tok/s. Where: `reader_env()` and the reader in
   `src/ds4/ds4_model.cu`, `DS4_READER_MIN` / `DS4_READER_PIECE_TOKENS` in `src/ds4/ds4_main.cpp:2176`.
   Switch and default: `DS4_READER=1` (off by default, live since 11-13). Gain: the other chat during a 61K read
   9.9-10.5 tok/s (41-49 % of solo, was 0.15), that read's first token +23 %, arrivals during a 61K read 39-46 s vs
   74-83 s (ref/PERF-LEDGER.md:556-562). Exact with no concurrency; reads are not bit-repeatable even alone (KV hash).
3. **Elastic conversations, and idle conversation memory taken back by the experts on demand.**
   Where: `alloc_seqs()` and the elastic block in `src/ds4/ds4_model.cu` (`el_take_env()` at :10384, `el_take()` at
   :10630), `src/ds4/ds4_elastic.hpp`. Switch and default: `DS4_ELASTIC_CONV=1` (off; the pool sizes to
   `DS4_ELASTIC_CONV_TOKENS`, 1048576 positions), `DS4_ELASTIC_TAKE` on unless `=0` (`DS4_ELASTIC_BUDGET_MB`, 256 a
   card); shipped by the 11-11 / 11-12 installs (ref/ROLLBACK.md:258-259). Gain: n2 +4.6 % vs the elastic live build,
   +10.9 % vs pre-elastic, reads +0.4 %, experts 4.30 -> 4.93 GB (ref/PERF-LEDGER.md:45). Exact (a memory-layout
   change).
4. **NUMA staging: every card's pinned hand-off and staging buffers on its own NUMA node.**
   Where: `numa_stage()` in `src/ds4/ds4_model.cu`. Switch and default: `DS4_NUMA_STAGE=1` (off by default, live
   since 11-15; 56-60 pinned buffers, 18.1 GB with the reader and shadows). Gain: solo +4.4 %, n2 +6.3 %, reads
   +1.5 %, beside +5.8 %, reader TTFT x0.997 (ref/PERF-LEDGER.md:722-723).
   KL 0.0078 vs the live-x-live floor 0.0064 (bar 1.5x).
5. **Idle-link expert prefetch: the next layer's guessed CPU-lane experts staged during the current token.**
   Where: `xp_alloc()` and the prefetch ring in `src/ds4/ds4_model.cu`. Switch and default: `DS4_PREFETCH=1`, ring
   `DS4_PREFETCH_SLOTS=16`, `DS4_PREFETCH_MAX=2`, `DS4_PREFETCH_RANK=6` (defaults kept: 24 slots or rank 4/8 are
   inside the default's spread, ref/PERF-LEDGER.md:594-596). Gain: solo +3.1 %, n2 +1.1 %, reads flat, decode KL
   0.0082 vs the 0.0074 floor, ~25-28 % of CPU-lane misses served (ref/PERF-LEDGER.md:574-582). Answer-changing at
   near-tie splits (a shared greedy prefix ran 102 vs 196 tokens; waived by Ken).
6. **`DS4_PICK_MAX=128`: prompts under the cap take the flow path, not a verify window.**
   Where: `pick_path()` and the load-time read in `src/ds4/ds4_model.cu`. Switch and default: `DS4_PICK_MAX` (unset = auto, 2048;
   a literal 0 is clamped to 1: every prompt takes the flow path), live 128. Gain: short-prompt TTFT total -23.7 % (128-512 tokens 2-6 s faster), solo +6.9 %, n2 +6.0 %,
   57K read -0.6 %; the known cost is 32-96-token prompts warm +0.3-1.3 s (ref/PERF-LEDGER.md:552-555).
   Answer-changing at near-ties (KL <= 0.020 on 6/6).
7. **The indexer's scores and the router on int8 tensor cores at sm_75.** Why: the FP32 thread kernel is the prompt
   path's scorer. Where: `index_score()` in `src/ds4/ds4_kernels.cu` (and `src/ds4/ds4_kernels.cuh`).
   Switch and default: `DS4_INDEX_TC_SM75=1` (prompt slabs; `=2` adds the decode windows), `DS4_INDEX_TC=0` off;
   on by default on RDNA3, `DS4_INDEX_TC_GFX12=1` opts in RDNA4. Gain: shipped 11-11 with the router; cards -6.1 % /
   host -6.7 % energy per read (ref/PERF-LEDGER.md:35, 495). Answer-changing: KL 0.0151 accepted against the par5
   calibrator's 0.0258 (ref/PERF-LEDGER.md:30).
8. **Our AVX1 CPU-lane rows (and the PR #1204 scale table we took instead of its rows).** Where: `own()` in
   `src/ds4/ds4_iq_cpu.cpp` with the shared kernels in `include/maya/kernels/cpu/iq_avx1_kern.inc`
   (`OddScales` / `k_odd()`). Switch and default: on for AVX1-only hosts; `DS4_IQ_OWN=0` gives the rows back to
   ggml's / the GLM lane's (`src/ds4/ds4_iq_cpu.cpp:916`). Gain: live GSQ decode +2.5 % code and prose, n2 flat, from
   the scale table alone
   (ref/reports/REPORT-ds4-avx1-pr1204.md:101); our rows beat upstream PR #1204's by up to 30 % on IQ3_S at 20 threads
   (ref/reports/REPORT-ds4-avx1-pr1204.md:29-30, ref/PERF-LEDGER.md:883-884). Bit-exact (931 cases, 0 differ from
   ggml-cpu's dot).
9. **The T4 sparse-attention kernel (`sparse_attn_v7t4_kernel`).** Why: prompt attention ran on the FP32 fallback and
   was 36 % of a chunk's time; the v7 kernel needs 93 KB of shared memory, which a T4's 64 KB cannot hold.
   Where: `sparse_attn_v7_kernel` and `sparse_attn_v7t4_kernel` in `src/ds4/ds4_kernels.cu` (the v7 kernel serves
   compute capability 7.x generally, `DS4_ATTN_V7=0` off).
   Switch and default: `DS4_ATTN_V7_T4=1` (opt-in, for the 7.x cards where v7 does not fit - the T4s).
   Gain: shipped 11-03; the ledger records the share it removed, not a single A/B figure
   (ref/PERF-LEDGER.md:230). Same fetches, scalings and MMAs in the same order and the same one-pass FP32 softmax.
10. **Hand-back from a pinned RAM copy, and pinning beside the copy.** Where: `lend_shadow()` and `pin_overlap()` in
    `src/ds4/ds4_model.cu`. Switch and default: `DS4_LEND_SHADOW=1`, `DS4_PIN_OVERLAP=1`, `DS4_PIN_THREADS=16`
    (off by default; live since 11-10). Gain: reads -3.4 % (73.1 -> 70.65 s), a hand-back 4 s -> 0.38 s, load
    96.5 -> 79 s (ref/PERF-LEDGER.md:253). It does not shrink the CPU lane's RAM copies - host headroom only.
11. **The read-help ordering guard (a latent hang).** Why: a helper could be inside chunk k while the owner issued
    chunk k's shares, closing a cycle through the helper's shared launch queue; with the 7,7,13,13 split it hung 2 of
    3 loads.     Where: the helper's per-chunk wait in `run_dev_chunks()` (`src/ds4/ds4_model.cu:20346`).
    Switch and default: none (unconditional, 8d73f40, live 11-14). Gain: speed-neutral by design
    (ref/PERF-LEDGER.md:606-609, ref/live-stack-notes.md:71). Exact.
12. **Restore points at message starts, and a rewind that reuses them.** Where: `snap_into()` in
    `src/ds4/ds4_model.cu`, the cap in `src/ds4/ds4_marks.hpp:19`, the request path in `src/ds4/ds4_main.cpp:1531`.
    Switch and default: `SAGUARO_DS4_RESTORE_POINTS=16` (engine default 4) and `DS4_REWIND=1` (live since 11-09/11-10,
    ref/ROLLBACK.md:14-18, 48). Gain: not measured as a speed number (an edited message re-reads from a point instead
    of the whole conversation). Exact.
13. **Conversations kept across requests and across runs.** Where: the slot store read in
    `src/ds4/ds4_main.cpp:836`. Switch and default: `SAGUARO_DS4_SLOT_KEEP=1` with `SAGUARO_DS4_SLOT_GB` (live 64)
    and `SAGUARO_DS4_SLOT_DIR` (ref/ROLLBACK.md:19). Gain: not measured (a behaviour change: a slot each per
    conversation instead of one). Exact.
14. **The DSpark drafter loaded always, with a VRAM cap and a cost knob.** Where: `vram_cap()` in
    `src/ds4/ds4_model.cu`, the load-time defaults in `src/ds4/ds4_main.cpp:2010`. Switch and default:
    `SAGUARO_DS4_DRAFT_VRAM=32`, `SAGUARO_DS4_DRAFT_COST=1` (live on the GSQ profile; ref/ROLLBACK.md:18).
    Gain: always-loaded beats on-demand (ref/PERF-LEDGER.md:226, no single number). Exact.
15. **DeepSeek reads pictures: the vision tower's BF16 GEMMs on FP16 tensor cores.** Why: cuBLAS BF16 runs on sm_75
    but without tensor cores - 7-10x slower on vision shapes. Where: `VisionTower::load` and the FP16 path in
    `src/ds4/ds4_vision.cu`. Switch and default: on; `SAGUARO_DS4_VISION_F16=0` runs the BF16 reference.
    Gain: a 560x560 tower-only encode 683 -> 224 ms warm (ref/PERF-LEDGER.md:79-84). Answer-changing in the last
    bits: per-row cosine >= 0.991, mean 0.99976, ~2-4 BF16 ulps; saturation counted (0 seen), 547 tiny weights
    flushed.
16. **AMD (RX 9070 XT): the engine needed nothing new - the setup writes `--ssd-cache auto`.** Why: DeepSeek's RAM
    store (anonymous slots, O_DIRECT, LRU) already has the memory behaviour ROCm needs; without it the box swaps.
    Where: the store's sizing in `src/ds4/ds4_store.hpp:40`, the flag passed through by `saguaro.py:6557`.
    Switch and default: `--ssd-cache auto` on AMD GGUF configs (upstream default off); the setup rule is commit
    fbbf82c17 on the `saguaro/amd-ds4mm3` lineage - in this tree the flag exists but the setup does not write it, so
    an AMD DeepSeek config is still made by hand. Gain: decode 4.4 -> 11.1-13.2 tok/s on the 9070 XT
    (4.4 -> 11.1 at ref/reports/REPORT-amd-ds4-mm3.md:57-58, the 13.2 at :19 and :146). Exact for the logits that
    matter: AMD-vs-T4 KL 0.013-0.025 against a
    T4 run-to-run floor of 0.009-0.020 (ref/reports/REPORT-amd-ds4-mm3.md:14).
17. **Measured and not shipped** (so an intake does not re-do them). The switch names below are as the ledger wrote
    them; the arms ran on experiment branches, so most of them are not in this tree (noted where it is not the one
    that shipped). The 7,7,13,13 layer split (reads -4.2 % but solo
    -3.4 % / n2 -5.1 %, and it deadlocked the read helper before the guard - keep 10x4,
    ref/PERF-LEDGER.md:614-616, 802-805); uneven TP2 (`DS4_TP_GGUF_UNEVEN`, worse everywhere,
    ref/PERF-LEDGER.md:603-605); the Engram decode partition (`DS4_ENGRAM_DECODE_KEEP`, rejected,
    ref/PERF-LEDGER.md:563-568); the read share (`SAGUARO_DS4_READ_SHARE`, which *is* in the tree at
    `src/ds4/ds4_main.cpp:2166`, none of its arms met the 25 %-of-solo bar, ref/PERF-LEDGER.md:506-510);
    expert-parallel decode (`DS4_EP`, parked: -38 % solo vs live,
    ref/PERF-LEDGER.md:732-737); `--parallel 8` (no gain, "keep 4", ref/PERF-LEDGER.md:624-629); Engram O_DIRECT
    (slower on x86, ref/PERF-LEDGER.md:693-697); `DS4_MODULE_WARM` lazy modules (+5-6 experts, under the bar,
    ref/PERF-LEDGER.md:588-593); `DS4_SETTLE_FEED` (harmful: n2 -28 %, ref/PERF-LEDGER.md:535-539); card-local expert
    RAM (`DS4_COPY_LOCAL=1`, decode -3..-4 %, ref/PERF-LEDGER.md:238).

## MM3

The MiniMax-M3.1-Flash engine (saguaro-mm3), our proof-of-concept profile on Maya-S GGUFs. The engine reads
`SAGUARO_MM3_*` (`::mm3::env` tries the prefixed name). Biggest first.

1. **Prompt attention on Turing tensor cores (`sparse_attn_tc_kernel`).** Why: the CUDA-core kernel reloaded K/V per
   query and was 4.2 s of a 14.2 s 1024-token chunk. Where: `sparse_attn_tc_kernel` and `sparse_attn()` in
   `src/mm3/mm3_kernels.cu`. Switch and default: `SAGUARO_MM3_ATTN_TC=0` returns the CUDA-core kernel; on by default
   on NVIDIA (HIP keeps the CUDA cores: the `mma.sync` PTX is NVIDIA-only). Gain: the kernel 5.2x on random data,
   a live 25.7K read 378 -> 289 s (-23.5 %) (ref/PERF-LEDGER.md:102-108). Answer-changing but inside the model's own
   floor: final-logit KL 0.306 vs the CUDA-core kernel, where the exact-intent chunk change 1024 -> 2048 alone gives
   0.343.
2. **The deep-read fix: a chunk's host K/V blocks gathered into VRAM once.** Why: past the 32K KV-streaming ring every
   query read its kept host blocks over PCIe, so a 4096-chunk went 7.7 s at 33K to 23.3 s at 57K.
   Where: `kv_gather_mode()` and the gather in `src/mm3/mm3_model.cu`, given back in `Model::read_end`
   (`src/mm3/mm3_model.cu:681`). Switch and default: `SAGUARO_MM3_KV_GATHER` = 1, on by default with
   `SAGUARO_MM3_ELASTIC` (`SAGUARO_MM3_KV_GATHER_CHECK` compares). Gain: attention flat 8.0-8.3 s to 72K (was climbing
   to 23.3 s). Bit-identical (ref/PERF-LEDGER.md:178-187).
3. **Elastic index keys: each conversation's index cache is a virtual-memory range grown in 2 MB granules, taking the
   coldest resident experts when VRAM is short.** Where: the `SAGUARO_MM3_ELASTIC` block in `src/mm3/mm3_model.cu`
   (`ctx_take()`, `ic_rows()`). Switch and default: `SAGUARO_MM3_ELASTIC=32768` (off by default; without it the planner
   sizes the index for the whole context and 1468 experts stay in VRAM instead of 1929),
   `SAGUARO_MM3_ELASTIC_MIN_FREE_MB` (256 MB always free). Gain: solo 7.5 -> 7.9 tok/s (+5 %), two at once +10 %
   (ref/PERF-LEDGER.md:159-168). Exact (the kernels are unchanged); a taken expert has no pinned copy, so long
   conversations read it from the SSD mapping.
4. **The read helper on the partner card, and staging overlap on the copy stream.** Why: a prompt pass copied,
   computed, then waited on the host. Where: `experts()` in `src/mm3/mm3_model.cu` (`SAGUARO_MM3_READ_HELP`,
   `SAGUARO_MM3_STAGE_OVERLAP`). Switch and default: both =1 in the live config (off in the engine), `--chunk 4096`.
   Gain: STAGE_OVERLAP 12K read 88.4 -> 72.4 s (+22 %), READ_HELP 72.7 -> 62.5 s (+16 %); with ATTN_TC, chunk 4096 and
   these two the live 25.7K read went 378 -> 139.7 s (2.7x over the night) (ref/PERF-LEDGER.md:112-119).
   Both logit-BIT-IDENTICAL, decode flat.
5. **Two conversations at once with per-conversation KV streaming and per-conversation saved conversations.**
   Why: upstream turned streaming off under `--parallel`, which caps the context at what fits in VRAM.
   Where: the `SAGUARO_MM3_PAR_STREAM` branch in `Model::load`, `src/mm3/mm3_model.cu`.
   Switch and default: `SAGUARO_MM3_PAR_STREAM=1` with `--parallel 2` (live; upstream's behaviour without it).
   Gain: two chats 9.8 tok/s together vs 7.4 one after the other (+33 %), both done at 61 s instead of 81 s, 262K
   context kept (ref/PERF-LEDGER.md:143-148). Batched decode is not bit-identical to solo (upstream's path behaves the
   same).
6. **Saved conversations.** Why: llama-swap unloads the model between chats; a 25.7K read costs ~60 s of re-read.
   Where: the slot store in `src/mm3/mm3_main.cpp:389`/`:584` (`Model::state_ok` in `mm3_model.hpp`).
   Switch and default: `SAGUARO_MM3_SLOT_DIR=/opt/llm/var/<model>/sessions` with `SAGUARO_MM3_SLOT_GB` /
   `_MIN` / `_DAYS`; off when unset, and ignored without KV streaming or with several conversations.
   Gain: 9.8K tokens saved 0.5 s, read back 0.3 s (vs ~60 s of reading); 25.7K saved 1.2 s / 2.0 GB
   (ref/PERF-LEDGER.md:123-128). Greedy answer after a restore identical to the cold read.
7. **The learned expert profile (this box's routing).** Where: `--expert-profile <file>`, written after each request,
   read at load for placement only. Switch and default: the live config points at
   `/opt/llm/var/MiniMax-M3.1-Flash-Maya-S/expert-profile.bin` (ref/PERF-LEDGER.md:120-122).
   Gain: placement only and grows with use - "re-measure decode after a few days", no number yet. Exact.
8. **`--chunk 4096` and `--staging 96`.** Why: the fill/drain bubble dominates a 12K read at small chunks, and the
   staging slots double as the decode's promoted-expert cache. Where: `Model::load` in `src/mm3/mm3_model.cu`.
   Switch and default: CLI, chosen by the setup / the live config (`--chunk 4096`, `--staging 96`, was 48).
   Gain: 12K read 158 -> 89 s (+78 %) for the chunk, decode -6 % (8.56 -> 8.00 tok/s, an accepted loss,
   ref/PERF-LEDGER.md:109-111, 196); `--staging 96` decode +7 % calibrated, reads -2 % (ref/PERF-LEDGER.md:135-138).
   Exact per setting (a chunk-size change moves the rounding, which for MiniMax is not small - see item 1).
9. **262K context instead of 1M.** Where: `--max-context 262144` in the live config. Switch and default: CLI.
   Gain: none - a trade: decode -2 % (the 4 GB index cache displaces 157 VRAM experts), and 1M needs a streamed index
   cache (ref/PERF-LEDGER.md:132-133). The 1M attempt (`SAGUARO_MM3_IC_STREAM`, opt-in, never live) is recorded as a
   dead end: 0 of 146,400 host blocks pruned on real text, decode -16 % at 8K host positions
   (ref/PERF-LEDGER.md:150-155).
10. **Turbo drop/skip, DSpark, and the CPU lane.** MiniMax runs the GLM-style Turbo keys (`serve/turbo_live.py`) and
    the DSpark drafter from the pack's `companions/`; its CPU lane is the shared AVX1 rows (see the DS4 and GLM
    entries) with `SAGUARO_MM3_LANE_FREE` (`CpuLane::auto_workers`, `src/mm3/mm3_cpu.cpp:159`) choosing the cores.
    Gain for the lane: 20 / 30 / 38 threads were all flat on decode (5.53 / 5.39 / 5.51 tok/s unpinned,
    ref/PERF-LEDGER.md:129-131). Exact.
11. **Measured and not shipped**: `SAGUARO_MM3_DECODE_HELP` (the partner card takes some CPU-lane experts: decode
    +2 % but not bit-identical, and the CPU lane's per-layer cost is the real limit,
    ref/PERF-LEDGER.md:139-142); `SAGUARO_MM3_NUMA_POOL` (prompt +1.5 %, decode flat - noise,
    ref/PERF-LEDGER.md:156-157); `--reserve-gb 0` (OOM at load: the planner's estimate is short by > 1 GB,
    ref/PERF-LEDGER.md:134).
12. **Not in this tree yet (branch `saguaro/amd-ds4mm3`): MiniMax on AMD with a model bigger than RAM.** The pinned
    pool registered from an mmap instead of hipHostMalloc's TTM pages (`SAGUARO_MM3_POOL_REGISTER`), the file-served
    experts evicted from the page cache after use (`SAGUARO_MM3_FILE_EVICT`), unbuffered O_DIRECT reads through pinned
    blobs with 8 readers side by side (`SAGUARO_MM3_FILE_READERS`, `SAGUARO_MM3_DIRECT`), and 16 GB of RAM headroom
    instead of a tenth of RAM (`SAGUARO_MM3_RAM_HEADROOM_GB`); all AMD-default, NVIDIA unchanged.
    Gain: swap traffic 27 GB each way -> ~1 GB, a 1K prompt 3.5-7.0 -> 11.9 -> 45 tok/s, decode 0.39-0.60 -> ~4
    (ref/reports/REPORT-amd-ds4-mm3.md:26-58, commit messages 09367e7ed / 6e747b976 / 7e998686d). Prompt logits
    bit-identical. The names are quoted from that branch: none of them exists in this tree (checked with
    `grep -rn` over `src/mm3`), which is why they are absent from the list above.
13. **Open, and a warning for anyone touching MiniMax's router**: its prompt path is rounding-sensitive on every GPU -
    chunk 1024 vs 4096 moves the prompt logits by KL 0.50-0.69 (top-1 71-75 %) from route flips that start at the first
    MoE layer and compound to a 50-71 % different residual stream by layer 30, so KL is not a usable quality bar for
    this engine; it predicts the text equally well either way (ref/reports/REPORT-amd-ds4-mm3.md:118-133).
    Nothing here changed it.

## HY4

The Hy4-Flash engine (saguaro-hy4). Installed in the live stack since 11-23 with no profile of its own
(ref/live-stack-notes.md:30), so almost nothing here is measured.

1. **Tensor parallel across a group of cards.** Where: `src/hy4/hy4_tp.cpp` / `hy4_tp.hpp` (the host's side,
   `Options::tp`), the planning and the sums in `Model::Impl::plan()` in `src/hy4/hy4_model.cu`.
   Switches and defaults: `--tp N`; the knobs are bare `HY4_TP_*` - `HY4_TP_F32` (default: F16 partial sums,
   `hy4_model.cu:741`), `HY4_TP_ONE_BLOCK` (:743), `HY4_TP_EXPERTS` (:786), `HY4_TP_LANE` (the group takes the CPU lane
   only with `=1`, :2163), `HY4_TP_DOWN` (:2092), `HY4_TP_HELPER` (:2558), `HY4_TP_MIXED` (:2505),
   `HY4_TP_PART_NODES=1` (each member's slice of the partials on its own GPU's NUMA node, :664),
   `HY4_TP_DRAFT_HEAD` (:1797). Gain: not measured for Hy4 (the live four-T4 pair profile is Inkling's, not Hy4's).
   Exact within a group (the same terms; `HY4_TP_F32=1` asks for F32 accumulation).
2. **KV spill: the indexed layers' rows past what RAM holds go to a file beside the pack.** Where:
   `src/hy4/hy4_spill.hpp`, `Options::kv_spill` (`src/hy4/hy4_model.hpp:61`), `kv_spill_on_plan()`
   (`src/hy4/hy4_model.cu:1374`) and the auto rule inside `spill_layer()` (`:1458`). Switch and default:
   `--kv-spill on|off|auto` (auto = on where the RAM does not fit), `--kv-spill-dir` (the pack's directory).
   Gain: not measured. Exact (the same bytes, a different tier).
3. **The Engram n-gram table read from the pack and looked up on a side stream.** Where: the table's own format
   (`engram.bin`, `Engram::engram()` in `src/hy4/hy4_pack.hpp:192`), the kernels `engram_rows()` / `engram_mix()` in
   `src/hy4/hy4_kernels.cu:3004`/`:3011`, and the gather behind the previous layer's FFN partial
   (`src/hy4/hy4_model.cu:444`). Switches and defaults: the model's geometry decides; `HY4_ENGRAM_EPS` and `hy4.txt`'s
   `engram_scales` (`src/hy4/hy4_model.cu:969-976`), `HY4_EG_DUMP=<file>` in tests. Gain: not measured.
   Exact - with no eps set the norms take the type's machine epsilon, as the release's graph does.
4. **`SAGUARO_PCIE_SAFE=1`: a host-safety switch.** Why: some PCs reset when all cards stream weights from host memory
   at once. Where: `tp_share_on` in `src/hy4/hy4_model.cu:1678`. Switch and default: `=1` turns the TP share (and the
   lane's use of it, `:7065`) off; unset means the fast behaviour. It is in the Hy4 smoke line
   (`tp4, 131K, SAGUARO_PCIE_SAFE=1`, ref/reports/REPORT-upstream-v1.1.75.md:162). Gain: not measured. Exact.
5. **Upstream's HOT_SEED taken, our copy shipped.** Where: `hot_init()` (`src/hy4/hy4_model.cu:6558`, the
   `HY4_HOT_SEED` read at `:6596`) and the seed files `data/hy4-hot-seed/hy4-expert-hot-*.bin`.
   Switch and default: `HY4_HOT_SEED` names the file; without one for this PC the engine learns its own.
   Taken from upstream (1823471fb engine part, the seed files via up75-setup,
   ref/reports/REPORT-upstream-v1.1.75.md:81-82) - kept here because it is a per-PC file we ship. Exact.
6. **Not on AMD, not on Pascal, and one gap the Monitor shows.** `saguaro.py:772` keeps Hy4 out of the AMD picker
   ("its engine has not run on an AMD GPU yet"); it compiles for gfx1201 and would need the same card check DS4 and
   MM3 got, plus the two memory traps MiniMax hit (ref/reports/REPORT-amd-ds4-mm3.md:196-210). On Pascal it needs a
   `__nanosleep` guard and WMMA guards in `src/hy4/ds4/ds4_iq.cu` before it builds at all, and the model is 117 GB on
   four cards (ref/reports/REPORT-p40.md:76). Hy4 (and Inkling) never report `kv_bytes_token`, so the Monitor's
   Context bar has no K/V figure for them - true on main too (ref/reports/REPORT-upstream-v1.1.75.md:31-32).

## IK2

The Inkling 2 Mini engine (saguaro-ik2), our own port; live since 11-23 as the two `--tp 2` Inkling profiles
(ref/live-stack-notes.md:28). Biggest first.

1. **Drafting made a speed gain: our switches on top of upstream v1.1.75's window.** Why: upstream's drafting alone was
   flat or negative on our traffic and its gate is unstable on Turing. Where: the defaults block in `main()`,
   `src/ik2/ik2_main.cu:7727-7744`; the controller's reads at `backoff` :4889, `gate` :4982, `lk_match` :5026; the
   drafter's head in `set_head_q8()` (`src/ik2/ik2_draft.cu:1099`); the snapshot rings at `snap_rings` :9182.
   Switch and default: each default is set only where the environment does not set the switch, so
   `SAGUARO_IK2_<X>=0/1` still wins - T4 anywhere: `SAGUARO_IK2_DRAFT_GATE=0` + `DRAFT_BACKOFF=1`,
   `DRAFT_HEAD_TILES=1`, `BLOCK_SPLIT=1`, `DRAFT_SNAP=1`, `DRAFT_SNAP_RINGS=1`; plus `LOOKUP_MATCH=1` where every
   expert is resident and `BLOCK_EXACT=0` (the rounding-level window) on the T4 UD dynamic cache; AMD offload, Pascal,
   the two-GPU split and `--parallel 2`: drafting off. Gain: T4 UD +8.6 % (agent turns +23..+27 %), T4 resident
   +4.4 %, RX 9070 XT resident +11.3 %, and -1..-12 % on the AMD offload layout, so off there
   (ref/reports/REPORT-ik2-draft-gain.md:22-26; PERF-LEDGER.md:877-878 carries the preliminary +8.3 / +4.3 / +12.6 %).
   The resident gains are bit-identical;
   the +8.6 % is answer-changing at near-ties (window KL mean 5.3e-5, top-1 99.73 %).
2. **Pascal (Tesla P40): our own kernels, and the host memory released.** Why: the generic scalar kernels sit at
   45-50 % of the P40's bandwidth and ik2 pinned its whole GGUF. Where: `src/ik2/ik2_pascal.cu` / `.cuh` (the gates
   `expert_layout()`/`env_on()` at :498-534), hooked from `src/ik2/ik2_kernels.cu` and `src/ik2/ik2_main.cu`.
   Switch and default: `SAGUARO_IK2_PASCAL_EXPERTS`, `_PASCAL_Q8_EXPERTS`, `_PASCAL_Q8`, `_PASCAL_Q8_SPLIT`,
   `_PASCAL_ATTN` (on for the Pascal arch class, `=0` per path) and `SAGUARO_IK2_RELEASE_HOST` (default on for
   Pascal); `--chunk auto` is Pascal's default when `--chunk` is not given (8192, else 4096). Gain: decode
   36.2 / 32.4 / 28.7 -> 40.1 / 37.3 / 33.5 tok/s (0 / 8K / 30K deep, ECC on), `--chunk auto` reads +19 %
   (8K 384 -> 458), peak RSS 14-17 GB -> 2.2 GB with the same logits (ref/PERF-LEDGER.md:886-888,
   ref/reports/REPORT-ik2-p40-tuning.md:28-34, 62). Run to run bit-equal; KL vs the T4 reference 2.1e-3, the
   cross-kernel-class class. The bigger chunk does change answers after long prompts (chunk 2048 vs 4096/8192 alone:
   4e-3 on the one-T4 quant, 9.5e-3 on UD-Q4, top-1 93.7 %) - recorded as Ken's decision, and `--chunk 2048` restores
   the old numerics (ref/reports/REPORT-ik2-p40-tuning.md:48).
3. **Two cards as a pair, with drafting off under `--tp`.** Where: the same `--tp` path as Hy4's; the live profiles run
   `bin/saguaro-ik2 --tp 2` on one socket pair each, group `inkling-pairs` (ref/live-stack-notes.md:28-29).
   Switch and default: `--tp 2`; drafting is off there because `tp_forward` is the prompt path (a host sync every
   visit, no conversation records to roll back). Gain: the two instances together run 138 tok/s
   (ref/reports/REPORT-ik2-draft-gain.md:116). A verify window under `--tp` is queued (~+17 % on code on paper,
   ref/reports/REPORT-ik2-draft-gain.md:100-101).
4. **The drafter we dropped for upstream's.** `SAGUARO_IK2_DRAFT_Q8` (the layers as planar Q8_0 on q8_tile:
   7.35 -> 5.56 ms a round on UD, 0.62 instead of 0.85 GiB, and it fits beside the one-T4 quant at 8K where the BF16
   drafter never did) was dropped on the merge in favour of upstream's GGUF Q8_0 drafter, which does the same thing
   properly (its step recorded as a CUDA graph, the Markov fold int8) - the switch is gone from this tree, so what
   remains of it is the `SAGUARO_IK2_DRAFT_HEAD_TILES` path (`set_head_q8()`, `src/ik2/ik2_draft.cu:1099`).
   ref/reports/REPORT-ik2-draft-gain.md:89-95, 124-127. Acceptance unchanged (0.64 vs 0.59).
5. **AMD, Volta and the offload layout.** Upstream's Inkling-on-AMD commit (bdbc727ba) was merged with our AMD offload
   path, dynamic expert cache, drafter placement, `--tp` and Pascal spin rather than taken as-is, and upstream's
   drafting was ported to Volta and Turing (c2ca8a1a6) with our card checks kept (`ik2_kernels_test`: 397 checks on
   sm_75, 403 on HIP; `SAGUARO_IK2_DRAFT_FORCE=2` drafts anyway on the AMD offload path).
   ref/reports/REPORT-upstream-v1.1.75.md:79, 88, 128-132, 137, 175. Gain: item 1's numbers.
6. **Measured and not built**: drafting on Pascal (a window of 2 costs ~48 ms against 26 ms for one token; break-even
   needs ~85 % acceptance, which only document edits reach, ref/reports/REPORT-ik2-p40-tuning.md:64-68), the 4-card
   drafter split on the other socket pair (the drafter cannot overlap the verify: its taps and the bonus token come out
   of round n's, ref/reports/REPORT-ik2-draft-gain.md:108-114), and 8-bit activations in the inexact window (the dense
   part is not where the window's cost is, ref/reports/REPORT-ik2-draft-gain.md:54-55).

## QFN

engines/qfn is our own port of Strata (its own CMake project; `serve/qfn_engine.py` is generated from
`engines/qfn/serve/server.py` by `tools/saguaro/extract_qfn_engine.py`). Biggest first.

1. **The sm_75 exact kernel stack.** Why: the T4s are our live box and upstream's kernels leave 3-4x on the table there.
   Where: the umbrella is `strata::sm75_series()` / `sm75_default()` in
   `engines/qfn/include/strata/core/sm75_series.hpp:24-37` (reads `SAGUARO_SM75_PERF`; each kernel keeps its own
   switch, which wins in both directions); under it `kql_bits()` in `src/kernels/cuda/iq_kernels.cu`, `q8g_on()` in
   `src/kernels/cuda/native_mmvq.cu`, `gdn_keyhead_ok()` and `argmax` rows in `src/prefill/kernels.cu` /
   `src/kernels/cuda/verify_kernels.cu`. `engines/qfn/src/core/nv_defaults.cpp:nv_default_env` is not the umbrella:
   it only gates the separate `SAGUARO_SM75_STACK` opt-in table for sm_75 (`:37-40`).
   Switch and default: `SAGUARO_SM75_PERF=1` turns the family on (`SAGUARO_KQL=3`, `SAGUARO_SM75_Q8G`,
   `SAGUARO_GDN_KEYHEAD`, `SAGUARO_ARGMAX_ROWS_SM70`); all off without it; live configs set them
   (ref/live-stack-notes.md:68). Gain: Q4 ~4x / Q8 ~3.7x faster than llama.cpp (ref/live-stack-notes.md:65). Exact.
2. **Piped MTP drafting up to 4 decoding conversations, plus a Lamina draft layer.** Why: multi-chat decode.
   Where: `engines/qfn/src/program/generate.cpp:main` (the switch), the MTP layer from `tools/mtp_from_gguf.py`.
   Switch and default: `SAGUARO_PIPED_MTP_MAX=4` (engine default 2); live on Q4 and Q8.
   Gain: 4 chats +29 % (Q4) / +16 % (Q8), Lamina +0..7 % (ref/PERF-LEDGER.md:228). Exact.
3. **The prompt-MMQ stack: K-quant MMQ + in-place resident experts + K-tail skip + fused SwiGLU/q8_1.**
   Why: the prompt path gathered every routed expert into group buffers and ran K=640 as 768.
   Where: `direct_mode()` in `engines/qfn/src/prefill/moe_mmq.cu`, `src/prefill/mmq_kq_loader.cpp`,
   `src/prefill/moe_mmq_tab.cuh` (KTAIL). Switch and default: `SAGUARO_MMQ_KQ=1`, `SAGUARO_MMQ_DIRECT=1`,
   `SAGUARO_MMQ_KTAIL=1`, `SAGUARO_MMQ_SWIGLU_Q8=1`, off by default; live on Q4 (all) and Q8 (with
   `--vram-reserve-mib 689 / -later 687`). Gain: Q4 all reads +2.63 %, solo 99.98 %, n2 100.00 %
   (ref/PERF-LEDGER.md:583-587); Q8 +1.40 % (ref/PERF-LEDGER.md:756). Exact (KL 0, 0 flips; the tab library is ABI 2,
   binary + `libstrata_kq.so` + `libstrata_kq_tab.so` are one set).
4. **The QSA select scorer on Turing tensor cores (FP16 hi/lo).** Why: the FP32 warp scorer is 10 % of a read and the
   3xTF32 path refuses sm_75. Where: `scores_halve_on()` in `engines/qfn/src/kernels/cuda/qsa_select.cu`.
   Switch and default: `SAGUARO_SELECT_SM75=1`, default off, **on in the live configs** (shipped 11-11).
   Gain: reads +3.9 %, J/read -3.7 %, n2 +3.4 %, scorer 3.3x, select stage 12.9 -> 7.9 ms per 8K/256q
   (ref/PERF-LEDGER.md:36-37, 453). Answer-changing (near-ties): top-1 97.01 %, conf 99.79 %, KL 0.0059, PPL +0.07 %.
5. **Multi-conversation batch engine, elastic slots and elastic K/V.** Why: upstream serves one conversation; we run
   1-2 chats at a time on 4 T4s. Where: `host_elastic_on()` in `engines/qfn/src/core/layer.cpp`, the elastic
   slot/budget code in `src/program/generate.cpp`. Switch and default: `SAGUARO_KV_HOST_ELASTIC` and
   `SAGUARO_KV_VRAM_ELASTIC` default on with elastic slots, `SAGUARO_ELASTIC_IDLE=1 _S=300` in the configs
   (ref/ROLLBACK.md:95). Gain: 2 chats +40 % on the live T4 Q4 (ref/reports/REPORT-qwen-engines.md:101). Exact per
   conversation; batched decode is not bit-identical to solo.
6. **Conversations persisted to disk, parked in RAM, and a faster park/load.** Why: llama-swap unloads a model between
   chats, so a resumed conversation must not re-read. Where: `park_fast()` in
   `engines/qfn/src/core/conversation_copy.hpp`, `SAGUARO_PARALLEL_LOAD_REGISTER` in `src/core/expert_source.cpp`.
   Switch and default: `--session-dir <dir>` (no default), `SAGUARO_PARK_FAST=1`,
   `SAGUARO_PARALLEL_LOAD_REGISTER=overlap`; live on both QFN profiles (ref/ROLLBACK.md:45).
   Gain: park 1.39-1.51 -> 0.97-1.10 s (~0.4 s), warm load ~0.3 s over a 5 s fill (ref/PERF-LEDGER.md:424-425). Exact
   ("same bytes"; a session saved by an older engine version is not read).
7. **Elastic YaRN: 1M context, each conversation moved to YaRN only when it passes 262144.** Why: one box serving both
   short and 1M chats without paying YaRN's accuracy cost on short ones. Where:
   `engines/qfn/src/program/generate.cpp:main`. Switch and default: `SAGUARO_YARN_ELASTIC=1` with
   `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144`; live on Q4 and Q8 (ref/ROLLBACK.md:72-74).
   Gain: not measured as a speed change (a capability). Answer-changing for the one conversation that crosses the
   threshold; pictures move with it.
8. **Batch wrap-spill: what the engine generated past a budget's stop is kept instead of re-read.** Why: a wrapped-up
   answer's continuation re-read the whole conversation (31.8 s gap on a 22.8K conversation).
   Where: `serve/qfn_engine.py` (generated; `BATCH_WRAP_SPILL`) from `engines/qfn/serve/server.py`.
   Switch and default: `SAGUARO_BATCH_WRAP_SPILL=1`, off; live on the Q8_0 profile only (ref/ROLLBACK.md:267).
   Gain: wrap gap 36.7 s -> 0.97 s, 14/15 continuations clean (ref/PERF-LEDGER.md:759-763). Exact.
9. **Route tail skip (drop the 7th-9th of 10 routed experts in decode) - measured and left OFF.**
   Where: `tail_skip_rank()` in `engines/qfn/src/core/expert_source.cpp`. Switch and default:
   `SAGUARO_ROUTE_TAIL_SKIP=7` to turn on; the engine's own rank-7 default is disabled by `--batch 8`, so live runs
   with it off (ref/PERF-LEDGER.md:632-635, 688). Gain if on: Q8 solo +13.8 %, Q4 +2.1 % - and KL 0.0110 at parallel 1,
   several times that in live-shaped batch windows; GPQA-Diamond showed no accuracy difference (178/198 both arms).
   Closed 11-15 (ref/PERF-LEDGER.md:647-689). Answer-changing.
10. **The BF16 -> tensor-core prompt path, with a skip list that the KL harness justified.** Why: the T4's cuBLAS runs
    BF16 without tensor cores. Where: `bf16_path()` in `engines/qfn/src/prefill/gemm.cu`.
    Switch and default: `SAGUARO_BF16_TC=1` with `SAGUARO_BF16_TC_SKIP=indexer,router,ssm` (live).
    Gain: the skip list is the cost side - router alone on TC gives top-1 96.91 %, all three 96.58 % (the ledger's bar, 96.24 %, is a floor: both
    pass it); its recorded verdict is FAIL on the quality gate, so the skip list stays (ref/PERF-LEDGER.md:28, 451-452). Answer-changing.
11. **GDN chunked recurrence ported to Turing - correct, parked by the 70 W cap.** Where: `gdn_conv()` in
    `engines/qfn/src/prefill/kernels.cu`. Switch and default: `SAGUARO_GDN_CHUNKED_SM75=1`, off.
    Gain: the kernel 1.74-2.0x and its own phase 13.8 -> 8.3 s per 77K read, the read flat (+0.7 %; 2K +0.61 %,
    8K +1.2 %) because long reads are power-capped, and short/medium means are under the +2 % bar
    (ref/PERF-LEDGER.md:283-287, 456-465, 610-613). Answer-changing (FP16 hi/lo, KL 0.0061, top-1 96.78 %).
12. **CPU expert pool: the per-socket split stayed off; `--pool-workers 19` only on Q4.** Where:
    `ExpertPool::ExpertPool` in `engines/qfn/src/kernels/cpu/pool.cpp`. Switch and default:
    `SAGUARO_NUMA_POOL` (auto by placement; `=0` off - live uses the default one-socket pool on Q8).
    Gain: none - both tries lost: -7..-10 % (10-22) and 2 chats -11 % with `--pool-workers 19` on Q8, where it is noise
    on Q4, so Q4 keeps 19 workers and Q8 keeps the 9-worker default (ref/PERF-LEDGER.md:859-863).
13. **RDNA4 (RX 9070 XT): the shared CUDA-to-HIP layer, register warp sums, and the RAM tier sized by the setup.**
    Where: `include/maya/hip_compat` (QFN's own copy deleted; the CMake path `SAGUARO_HIP_COMPAT_DIR`,
   `engines/qfn/cmake/hip_backend.cmake:96`, for a tree built alone),
    `warp_sum()` in `engines/qfn/src/kernels/cuda/iq_kernels.cu`, the arch defaults in
    `src/core/arch_defaults.cpp`, `resident_budget_gib()` in `tools/qfn_fit.py`.
    Switch and default: HIP build via `-DSAGUARO_QFN=ON` / `tools/qfn/build_hip.sh`; `--resident-budget-gib` = RAM -
    20 GiB (`RAM_SPARE_GB = 20`) is the setup's default; the tuned hipBLASLt table is **not** shipped.
    Gain: RAM tier 72 vs 64 GiB: decode +32 %, 8K read +20 % (ref/PERF-LEDGER.md:866-867); the hipBLASLt table is
    +12 % on 8K reads (ref/reports/REPORT-qfn-amd.md:180) but makes two loads differ, and gfx1151's decode table is -13 % on gfx1201
    (ref/reports/REPORT-qfn-amd.md:169). Exact (the lane_xor warp sums are bit-for-bit; drafting on vs off gives
    identical greedy tokens).
14. **Pascal (Tesla P40): exact FP32 dense GEMMs and packed Q8_0 rows, offered by the setup.** Where:
    `dense_f32_on()` in `engines/qfn/src/prefill/gemm.cu`, the rows in `src/kernels/cuda/native_mmvq.cu`
    (`q8_sm60.cuh`), the env class in `tools/qfn_fit.py` (`PASCAL_ENV`, `PASCAL_ENV_QUANT`).
    Switch and default: `SAGUARO_DENSE_F32=1`, `SAGUARO_Q8_SM60=1`, `SAGUARO_MTP_Q8_SM60=1`, all opt-in; the setup
    writes the two Q8 ones on all-sm_6x PCs and `SAGUARO_DENSE_F32` only on GSQ IQ3_S.
    Gain: reads +4.9 % (10.5K) / +1.9 % (33.9K) with DENSE_F32 on a P40 (ref/reports/REPORT-p40-batch.md:45-46).
    Bit-exact: `gemm_f32_parity` 5,201 sizes all bitwise against that cuBLAS 12.0
    (ref/reports/REPORT-p40-batch.md:34) - re-run it if CUDA changes.
 15. **Turbo: drop and skip, per request.** Why: the request-level Turbo (the chat's bolt, Settings > Turbo, live)
     is a GLM feature; the QFN engine applies both since its s1 skip. Where: `turbo_drop_apply`
     (`src/core/verify.cpp`) for drop; for skip the wire value is translated to the kernel bar once in
     `turbo_skip_of` (`include/strata/program/turbo_skip.hpp`), the pick is left out by the residency table (and
     the RAM mirror for `ssd`), and the engine reports `INFO turbo_skip=1` (generate.cpp:12707) so the page
     offers Skip - a build from before it parses the key and leaves it off, which is the only case the page
     hides the choice for; what still runs exact (re-pick on, no residency table, `ssd` without a RAM-tier
     fact) says so once. Switch and default: request keys `turbo_drop=N` and `turbo_skip=ssd|ram`, both off.
     Gain: not measured on QFN. Answer-changing for both when on; absent/off is the exact line, bit for bit.
16. **Measured, exact, and left off** (near-misses kept in the ledger for a future pairing):
    `SAGUARO_SM75_Q8G_ROWS=4` (config-only; 77K reads -0.08 %, solo +0.61 % - no ship, ref/PERF-LEDGER.md:789-791) and
    the Q8_0 dense row kernels of branch pr/qfn-q8rows-sm75 (0.77x on the 1-column head, ~1 % of decode; the build
    that carried them was installed 11-18 and rolled back 16:17 on no measurable gain,
    ref/PERF-LEDGER.md:337-344, 359-379). Two more exact ports live on branches that never merged: the fused
    up+mix hyper-connection kernel (pr/qfn-upmix-sm75: -0.80 % reads, +0.55 % J/read, ref/PERF-LEDGER.md:764-768) and
    the PCIe fetch-overlap port (pr/qfn-fetch-overlap: exact, solo behind in 3/3 rotated pairs,
    ref/PERF-LEDGER.md:617-623). Parked for capacity rather than speed: conversation slots 8 -> 4 (+49 expert slots,
    n4 +3.6 %, ref/PERF-LEDGER.md:511-517).

## serve + web

One `serve/server.py` for all engines, plus the browser app in `serve/web`.

1. **One server for all six MoE profiles, with QFN's batch driver inside it.** Why: one install, one port, one UI;
   `--engine qfn` used to exec engines/qfn's own server. Where: `start_qfn()` (`serve/server.py:8193`) and
   `_qfn_engine_class()` (`:8255`), the engine-kind notes at `:29` and `:38-39`, and the generated
   `serve/qfn_engine.py` (from `engines/qfn/serve/server.py`, by `tools/saguaro/extract_qfn_engine.py`).
   Switch and default: `SAGUARO_QFN_SHIM=1` falls back to the old exec-the-other-server shim
   (ref/live-stack-notes.md:20-25). Gain: no speed change - QFN reads unchanged on the shared server
   (Q4 35.9K in 21.4-21.5 s, Q8 27.7 s, ref/PERF-LEDGER.md:94). Exact.
2. **Several chats answering at once from one page, per-chat and engine-total speeds labelled, `/health` advertising
   `capabilities`.** Where: `serve/web/chats.js` / `chat-view.js`, `Service.capabilities()`
   (`serve/server.py:2988`), the `/health` line (`serve/server.py:7147`). Switch and default: none.
   Gain (per chat vs engine total, one chat beside another): GLM 4.0 vs 10.3, DS4 7.5 vs 12.8, QFN Q8 9.3 vs
   15.6 tok/s (ref/PERF-LEDGER.md:97).
3. **Fixed on the way: a new web answer no longer cancels the other chats.** Why: the one-at-a-time rule in
   `jobs_put` took away every other place. Where: `jobs_put()` (`serve/server.py:6243`), called from the web path at
    `:7698`. Switch and default: none. Gain: not measured (a behaviour fix; the check was the live box's
    `bench/saguaro-one/multichat.py`, a script that is not in this repo - every engine streams the other chat during a
    read and cancels only the asked one, ref/PERF-LEDGER.md:95-96).
4. **Engines without logprobs answer instead of failing, and QFN's `/metrics` stopped crashing.** Why: clients ask for
   logprobs unconditionally; QFN and MiniMax report none, and `vram()` was read as a method.
   Where: the capabilities note at `serve/server.py:2981`. Switch and default: none. Gain: not measured
   (ref/PERF-LEDGER.md:98-100).
5. **The Saguaro rebrand, with the old names kept where a rename would break something.** Where: the `[saguaro]` log
   prefix (`serve/server.py:247`, and the engines' lines via `tools/saguaro/rename_log_prefixes.py`), the desert theme
   and logo in `serve/web/`. Switch and default: none. Kept on purpose: users' `MAYA_*` / `STRATA_*` setting names, the
   browser's `maya.*` / `strata.*` keys (`serve/web/chats.js:61`, `:88`) and the quant names
   (ref/live-stack-notes.md:24-25). Gain: not measured.
6. **The UI under llama-swap refuses what the supervisor owns.** Why: a page that switched models or rewrote a config
   fought the process manager. Where: `supervisor_name()` (`serve/server.py:8183`) and `svc.supervisor` (`:8738`).
   Switch and default: the refusal (409 on switch / update / port / context / Turbo / config writes) is on whenever a
   supervisor is detected; `SAGUARO_SUPERVISOR=""` says there is none and lifts it (ref/ROLLBACK.md:317-319).
   Gain: not measured (a safety change).
7. **The loop guard, and the fair read pause.** Where: `serve/loop_guard.py` (a request's guard, `:163`) and
   `fair_read_seconds()` / `fair_read_yield_seconds()` (`serve/server.py:536-549`). Switch and default: the guard is on
   (a repeated block ends the answer with finish `stop`, even with `ignore_eos`); `SAGUARO_FAIR_READ_S` (0 = off) and
   `SAGUARO_FAIR_READ_YIELD_S` (30). Effect: synthetic long decodes stop early by design, so a test that needs a fixed
   length must use varied prompts (ref/PERF-LEDGER.md:503-505); a long read pauses for a request waiting behind it.
   Answer-changing for the guarded answers - which is the point.
8. **DeepSeek's `gt=` render skipped, and the MiniMax/QFN gaps the one-server work left.** Where: the unified merge's
   server change (ref/live-stack-notes.md:71), `serve/ds41_encoding.py`. Known: QFN and MiniMax report no logprobs and
   MiniMax's saved conversations came later (see MM3 item 6), so the Monitor shows nothing for them
   (ref/PERF-LEDGER.md:100). Gain: not measured.

## setup/tools

The things that keep an intake routine (docs/UPSTREAM.md §2) and what our setup writes that upstream's does not.

1. **The rename layer, as tools.** Where: `main()` in `tools/saguaro/rename_env.py` (settings by suffix pattern,
   `--check`), `tools/saguaro/migrate_configs.py` (our configs), `rename_log_prefixes.py:rewrite`, and
   `rename_internal_maya.py:rename` (the internal `include/strata` -> `include/maya` rename, reproducing upstream's
   own commit). Switch and default: not runtime - each tool is idempotent and tested, and a missed name is a tool bug
   fixed in the tool (docs/UPSTREAM.md §2). Gain: not measured per run; the target is an intake of "a few hours, not a
   rewrite".
2. **`tools/saguaro/extract_qfn_engine.py`: the QFN server adapter is generated.** Where: `main()`,
   `tools/saguaro/extract_qfn_engine.py:43`, writing `serve/qfn_engine.py`. Switch and default: none - re-run it after
   changing QFN's server. `serve/test_dashboard.py` reads the generated file (`:429`), so a stale copy fails a test
   rather than the live box.
3. **The AMD compile checks.** Where: `tools/check_hip_compat.py` + its test `tools/test_hip_compat.py` (every CUDA
   name the HIP build compiles must have a mapping in `include/maya/hip_compat/`; since 90fcee373 it also finds inline
   PTX and CUDA headers an AMD device pass compiles, evaluating each file's `#if` chain for gfx1201 / gfx1100 - no
   ROCm or GPU needed), and `tools/hip_syntax_check.py` (`check()` at `:78`: the real thing - every HIP source through
   ROCm's clang with `-fsyntax-only`, host and device passes, `--arch gfx1201|gfx1100`). In this tree the syntax check
   is run by hand: its own header (`:12`) says `tools/test_hip_compat.py` runs it when `SAGUARO_HIP_SYNTAX_ROCM` is set,
   but no line of this tree's `tools/test_hip_compat.py` or CI refers to it - that wiring is on the
   `saguaro/amd-ds4mm3` lineage, together with the CONTRIBUTING change that makes AMD a supported platform. Reverting
   the 11-22 fixes, the syntax check reported exactly those 12 errors plus MiniMax's `'+f'` asm constraint, which the
   name scan missed (ref/reports/REPORT-amd-ds4-mm3.md:30).
4. **QFN built inside the top project.** Where: `CMakeLists.txt:1787-1808` - `-DSAGUARO_QFN=ON` builds `engines/qfn`
   with this configure's generator, compilers and CPU ISA floor (`-DSTRATA_ISA_FLOOR=avx` for the Ivy Bridge T4 box),
   and takes the pre-Turing path (`STRATA_EXPERIMENTAL_SM60`) only when every arch is < 75 - so a mixed `--gpus` list
   warns instead of building a binary with no kernel image for the old card, and the sub-build inherits the setup's
   ninja (`CMAKE_MAKE_PROGRAM`), which a PC without ninja on PATH needs. Helpers: `tools/qfn/build.sh`,
   `tools/qfn/build_hip.sh`, the `qfn-hip-compile` CI job (`.github/workflows/tests.yml:53`), `tools/qfn/kl.py` (the
   quality harness) and `tools/qfn/speed.py`. Gain: not measured (build fixes; the engines' own numbers are in the QFN
   section).
5. **The setup's QFN fit: what it refuses, warns about, and scales down.** Where: `plan()` and `best_gpus()` in
   `tools/qfn_fit.py` (`MIN_CC = 61` at `:33` - Pascal and newer), `resident_budget_gib()` at `:146` with
   `RAM_SPARE_GB = 20` at `:29`, and `PASCAL_ENV` / `PASCAL_ENV_QUANT` at `:93-94`. Switch and default: the RAM tier is
   RAM - 20 GiB (measured on 91.7 GiB: 71); on all-sm_6x cards the setup writes `SAGUARO_Q8_SM60=1` +
   `SAGUARO_MTP_Q8_SM60=1`, and `SAGUARO_DENSE_F32=1` only for GSQ IQ3_S (measured exact there). It refuses only what
   cannot run, says so with numbers, and warns then scales down (drafting off, `--kv-resident`, 32K) behind a confirm
   that `--yes` skips. Gain: the RAM tier is decode +32 % / 8K read +20 % (ref/PERF-LEDGER.md:866-867).
6. **Per-engine setup rules that differ from upstream.** Where: `NO_ENGINE` (`saguaro.py:771`: Hy4 not on AMD, Metal
   GLM-only, and saguaro-qfn on NVIDIA since T2 - one card, or the layers split across up to four - so the old
   "set up by hand there" note is out of date); `tools/ik2_fit.py:pascal()` (`:364`) with `layout()` (`:373`) leaving
   `--chunk` out on Pascal where one card holds every expert (the engine's `auto` then applies; the Models page reads
   `--chunk` as a number, so a literal `auto` would be dropped at the next save); and `serve/model_pull.py`'s measured
   AMD figures. Gain: on the P40 that choice is the +19 % read number in the IK2 entry above
   (ref/reports/REPORT-p40-batch.md:18-30).
7. **`--bench`, and the tree kept free of our box's paths.** Where: `tools/maya_bench.py` (the lock and hot-flag
   variables at `:34`; `--bench --compare <folder|git ref>` builds by CMake target) - upstream 3d1b67905 taken with
   our quiet-machine, warm-up and median rules (ref/reports/REPORT-upstream-v1.1.75.md:86). And the cleanup a fresh
   user's report asked for: `/opt/llm` and `/data/llm` paths out of `configs/`, `RUN-hy4-first-load.md` gone, the
   issue links pointed at ksanislo/saguaro (ref/reports/REPORT-9070xt-fresh-install.md:138-141, 195-212).
   Gain: not measured (the fresh-user reports are the test). Still open from those reports, and unfixed here: the AMD
   VRAM reserve (3072 MiB whether or not a display drives the card - `SAGUARO_GLM_RESERVE_MB=1024` gave Maya-S +9 %
   decode and Maya-L's reads 5.1 -> 107 tok/s, ref/reports/REPORT-9070xt-first-runs.md:78-88), the shipped gfx1201
   hipBLASLt tuning file not matching ROCm 7.1's 100100 (`:96-99`), `--yes` starting the 1.1 GB vision download without
   `--download-model` (`:111-115`), and switching Inkling quants carrying the old `--chunk` over
   (ref/reports/REPORT-9070xt-user-install.md:102).

## Dropped from upstream on purpose (the Maya Qwen path)

**The Qwen path inside saguaro-glm (the "qwen4exp" engine) - gone from our tree (T8a/T8b); upstream's changes to it
are dropped at intake.**
- What: upstream's Qwen3.8-Flash-Next support lived in the GLM binary as its own engine: the `qwen4exp` driver in
  `src/program/generate.cpp` (~4.5K of its then-10.7K lines) plus the Qwen-only files
  `src/core/{session,mtp,verify,layer,expert_source,native_dense,native_head,remote_experts}.cpp` and
  `src/prefill/prefill.cpp` - about 16K lines, which is Strata v0.1.27, the ancestor of engines/qfn. We serve Qwen on
  saguaro-qfn instead, and the path was removed on 2026-11-24, before this branch's base: the driver out of
  generate.cpp (37de16bb5 - a non-GLM pack now ends "this model runs on saguaro-qfn"; the file is 5,752 lines), the
  Qwen-only files, kernels and their tests out of the tree and CMake (730e785d0), and the Python side - family
  "qwen", the "qwen" engine kind, its web fields and tests - out of setup, tools, server and web (T8a, merged
  5bf926032). CHANGELOG.md:43-57. ref/reports/REPORT-qwen-engines.md:23, 113-119.
- Why: on the same CYBER-FROST file and the same card (RX 9070 XT) saguaro-qfn answers 1.2x faster (34.4 vs 28.3
  tok/s solo) and reads 1.4-2.0x faster (16K: 733-1,007 vs 422-428 tok/s; ~55K: 1,058 vs 527 tok/s, first token 50 s
  vs 106 s); the Maya path cannot load unsloth's UD-Q4_K_XL at all (no Q8_0 GPU dequantizer for its embedding) and dies
  with exit -4 on an AVX1-only host, where saguaro-qfn runs the same file at 43.7 tok/s.
  ref/reports/REPORT-qwen-engines.md:15-21. Exactness: neither is exact against the other - the two engines disagree
  KL 0.025 / top-1 95.55 % on the same file (rounding, not a defect), and saguaro-qfn on AMD is KL 0.006 from saguaro-qfn
  on the T4s. ref/reports/REPORT-qwen-engines.md:78-82.
- Where: none of those files exist at the top level any more, except the two the GLM build still uses:
  `src/core/expert_source.cpp` and `src/core/remote_experts.cpp` (730e785d0 keeps them: glm_fast_state.hpp and the HIP
  GLM helper include them); mrope (rope.cu), router_sigmoid (glm_model.cu), the qsa.hpp / kv_stream.hpp headers and
  strata-plan stay for the same reason. Where the driver was, `src/program/generate.cpp:5747-5751`, is one exit line.
  The removed path lives on only inside `engines/qfn/src/` (its own copy of core/ and prefill/).
  Switch and default: **none** - never switch-gated; the drop is by deletion.
- Intake: upstream hunks touching the removed paths are dropped, not re-added - the rule is written at
  CONTRIBUTING.md:29-33 (added with T8a, d8e8faf84); `tools/saguaro/rename_internal_maya.py` carries no per-file
  exclusion for them (it walks the tree, and the files are simply not there to walk). What is left of the path is the
  one-way read of old configs: `maya_quants.family_of` reads every old "qwen" config as "qfn" (`saguaro.py:2024`) and
  `migrate_cyber_frost` rewrites a CYBER-FROST old-engine config into the QFN one (`saguaro.py:8951`, `:706`); the
  `QWEN_IDS` aliases, the cyber-frost quant key and `QWEN_EFFORT` stay because the QFN template uses them.
  ref/reports/REPORT-qwen-engines.md:143-171 - the removal plan "if Ken picks option 1" - is what T8a/T8b executed.

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
  first and the plain `DS4_*` second (`ds4_env`, `src/ds4/ds4_model.cu:15715-15718`), so its entries above name the
  `DS4_*` form the configs write; the HY4 engine reads `HY4_*` directly (e.g. `HY4_HOT_SEED`,
  `src/hy4/hy4_model.cu:6596`), like the other HY4_* settings.
- **Program and log names**: the engine programs are `saguaro-glm`, `saguaro-ds4`, `saguaro-mm3`, `saguaro-hy4`,
  `saguaro-ik2`, `saguaro-qfn` (upstream: `maya`, `maya-ds4`, ...), and the log lines start `saguaro <part>:`
  (`tools/saguaro/rename_log_prefixes.py`). Outside the repo, log parsers must match both old prefixes: the live stack
  changed `[maya]` to `[saguaro]` on 2026-11-19 and the bench scripts still grep for both (ref/ROLLBACK.md:315-316).
