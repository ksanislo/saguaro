// src/program/generate.cpp - P2.S6: `strata generate`.
//
// THE DRIVER, and the first program in this project that answers a question.  Everything below it is a
// component; this is the thing that composes them into a token:
//
//     embed_row(token)  ->  48 captured layer graphs (the CPU expert pool behind the doorbell)  ->
//     lm_head(R)        ->  sample        ->  embed_row(next)  ->  ...
//
// WHAT IT IS NOT.  There is no tokenizer here.  `pack/full/tokenizer/` and `tools/strata_tokenizer.py` exist,
// and a C++ BPE is Phase 1's deliverable rather than this program's, so the prompt arrives as IDS via
// `--tokens`.  That is not a placeholder: it is exactly what Gate C1 needs, because C1 compares logits against
// llama.cpp on the SAME ids, and a tokenizer on only one side of that comparison is a second variable.
//
// AND IT IS PHASE 2, so hit rate is `h = 0` and the number it prints is slow on purpose
// (`phase-2-correct-engine.md:5-9`).  What it is FOR is the honest tok/s figure and the logit dump.

#include "maya/platform/idle_gate.hpp"
#include "maya/platform/env_report.hpp"
#include "maya/program/vram_live.hpp"
#include "maya/kernels/glm_fast.hpp"
#include "maya/core/layout.hpp"
#include "maya/kernels/sampler.hpp"
#include "maya/kernels/logprobs.hpp"
#include "maya/core/glm_model.hpp"
#include "maya/core/mtp_rows.hpp"
#include "maya/core/split_groups.hpp"
#include "maya/core/prefix_cut.hpp"
#include "maya/core/prefill_pieces.hpp"
#include "maya/core/prefill_gaps.hpp"
#include "maya/core/vram_mark.hpp"
#include "maya/program/logits_selection.hpp"
#include "maya/core/progress.hpp"
#ifndef NOMINMAX
#define NOMINMAX   // gguf_reader.hpp includes windows.h
#endif
#include "maya/artifact/gguf_reader.hpp"
#if defined(_WIN32)
#include <functional>
#include <windows.h>
#include <psapi.h>
#include <io.h>
#else
#include <unistd.h>
#include <cerrno>
#endif

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <charconv>
#include <climits>
#include <cmath>
#include <limits>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

struct Options {
    std::string pack = "pack/full";
    std::string glm_pack;             // --glm-pack: a GLM-5.3-Flash pack dir (the M3.1 path; see glm_pack_generate)
    std::vector<int64_t> tokens;      // the prompt, PRE-TOKENIZED
    int64_t max_new = 16;
    int64_t max_context = 4096;
    bool greedy = true;
    uint64_t seed = 0;
    int top_k = 20;
    float top_p = 0.95f;
    float temperature = 1.0f;
    std::string dump_logits;          // one line of logits per generated position
    int64_t logits_stride = 1;        // storage selection; all prompt tokens remain conditioned
    /// **THE RESIDUAL, SO THE HEAD CAN BE CHECKED WITHOUT THE LAYERS.**
    ///
    /// C1 fails (LEDGER L116) and the pipeline is `embed -> 48 layers -> head`.  Dumping `R` splits it in half:
    /// the head is one norm, two bf16 projections and one 794 MB GEMV, all of which can be recomputed in Python
    /// from the manifest.  If Python agrees with the engine on the same `R`, the head is right and the layers
    /// are wrong; if it disagrees, the head is wrong.  Nothing else in the engine can be split that cheaply.
    std::string ple_gguf;              // the ORIGINAL second GGUF shard: the PLE table is not in the pack
    bool no_ple = false;              // explicit diagnostic ablation; never a normal inference default
    bool stream_token = false;        // R2.6 experiment: ordered work on the session stream
    bool check_logits = false;        // optional full-vocabulary finite scan
    bool gr_fp32_activations = false;  // pinned CUDA single-token BF16 activation contract
    bool gr_native_mmvf = false;       // pinned projection reduction tree as well as FP32 inputs
    bool native_bf16 = false;          // SSM gates, router and indexer projections only
    bool native_bf16_extra = false;    // PLE value and shared expert scalar gate
    bool native_ple_key = false;       // unchanged Q2_0 key and CUDA Q8_1 activations
    bool native_moe_combine = false;   // pinned fused CUDA weighted reduction
    bool native_gdn = false;          // pinned CUDA recurrence and preprocessing
    bool native_flash_attn_short = false; // diagnostic pinned attention, context <=256
    bool native_qsa_indexer = false;  // pinned F16 key cache and F32 pooling
    bool native_qsa = false;          // pinned F32 QSA norms and gate
    bool native_rope = false;         // pinned text-only CUDA rotary arithmetic
    bool native_ple_postops = false;  // pinned PLE postprojection arithmetic
    bool native_router = false;       // pinned fused 512-expert top-10 router
    bool cpu_oracle_q8_0 = false;      // pinned x86 activation scales/codes at both expert stages
    std::string native_head_gguf;      // native output.weight experiment; same model shard as the pack
    std::vector<std::string> native_dense_gguf; // repeat for native GDN/QSA projection shards
    /// Plan v0.3 P1: the whole native arithmetic set as ONE switch (model shard 1). It enables exactly the
    /// combination recorded in bench/results/2026-09-23-attention-ple plus the native indexer, and never the
    /// <=256-token attention adapter. It becomes the default once P0 shows it is not slower.
    std::string native_preset;
    /// Plan v0.3 P2: how the n-gram table is read. Direct (default) = unbuffered SSD reads, table never in RAM.
    std::string ple_io = "direct";
    int64_t ple_row_cache = 1 << 20;   ///< bounded row cache (rows of 90 B); 0 disables
    int ple_inflight = 64;
    double ple_delay_us = 0;           ///< fault injection: every row read completes no earlier than this
    bool ple_sync_submit = false;      ///< A/B arm: submit reads on the token thread, no I/O worker
    std::string kv = "fp16";           ///< plan v0.3 P7: KV storage, fp16 (default) or int8 (half the VRAM)
    int64_t kv_resident = 0;           ///< KV streaming: resident cells per QSA layer (0: all in VRAM)
    std::string kv_streaming;          ///< --kv-streaming on|off: streaming with kv_resident's window (32768 if none)
    std::string dump_residual;
    /// The head input, `bb.mixed`.  It exists so the head can be SPLIT: steps 1-4 (the per-stream norm, the two
    /// bf16 projections and the stream mean) recompute cheaply in Python, and only the 794 MB GEMV does not.
    std::string dump_mixed;
    /// One residual snapshot per layer per position: `n_layers * hc * n_embd` floats per position, appended in
    /// position order.  This is the C1 BISECTION LADDER - it is what `llama-debug --tensor-filter l_last` prints
    /// for the reference, so the first layer whose `sum` diverges is the layer that holds the bug.  It needs the
    /// captured path (the expert pool only exists there), so it is refused with `--no-capture`.
    std::string dump_layers;
    /// `2 * n_embd + 2 * hc` floats per layer per position: the attention half's block output, the MoE half's,
    /// and the two injection vectors.  It separates `linear_attn_out-<l>` from `ffn_out-<l>`, which the residual
    /// ladder cannot.  **CAPTURED INTO THE LAYER GRAPHS**, so it must be armed before `session_capture`.
    std::string dump_halves;
    /// P0.S8's routing trace, and a prerequisite the Phase 3 plan names explicitly.  One record per layer per
    /// position: `int32 layer, int32 k, k int32 ids, k float weights`.  It is what a hit-rate curve for a
    /// candidate VRAM expert cache is computed from, and it needs no new kernels - the doorbell already
    /// publishes exactly this much to pinned memory.
    std::string dump_routing;
    bool no_capture = false;          // run the layers directly instead of replaying graphs
    bool no_pool = false;             // skip the CPU expert pool: the GPU-only floor
    bool sync_every_layer = false;
    /// Per-stage CUDA-event timings inside the layer halves.  `--no-capture` only: an event recorded inside a
    /// stream capture is silently dropped, so the captured path cannot carry this.
    bool stage_timing = false;
    /// Launch the 48 captured `pre` graphs back to back with no host work between them and report the pure GPU
    /// time per token.  This is the only measurement that separates host-bound from GPU-bound, because the
    /// stage events include every gap where the GPU waited for the host.
    bool graph_only = false;
    bool gpu_only_full = false;   ///< R0.3: pre + post + head, the true per-token GPU floor
    int pool_workers = 0;         ///< R2.2: 0 = "all physical cores minus the host's"; >0 overrides
    /// R2.2's first half, as an A/B arm.  **ON by default**, because the measurement that justifies it is the
    /// pool's own drain: 33.7 GB/s against 5/6 x 44.14 = 36.8 for five workers, on a machine whose sixth core
    /// is reserved for a host thread that has nothing to do while the drain runs.
    bool no_host_worker = false;
    bool mmap_experts = false;    ///< R2.1: opt OUT of the resident arena, back to MapViewOfFile
    /// v1.1.32: the experts straight from the model's GGUF - the hottest the GPUs leave in RAM as far as it goes, the
    /// rest read when routed.  "auto" (default): when this PC's RAM cannot hold them all; "on" / "off".
    std::string experts_from_gguf = "auto";
    bool resident_cpu_experts = false; ///< mmap-backed static-cache misses copied into ordinary RAM
    /// R4: slots of VRAM-resident experts.  **0 = off, and off is the default.**
    /// **THE COMMENT THAT USED TO BE HERE WAS FALSE AND ROUND 328 MEASURED IT.**  It said "the cache has no
    /// consumer yet - `moe_hit_grouped_s2` does not exist - so switching it on costs the fill traffic and
    /// saves nothing".  The kernel exists (`src/kernels/cuda/s2_expert_grouped.cu`), it is wired at line ~660
    /// via `expert_hit_run`, and switching the cache on **does** move work off the CPU pool: the drain fell
    /// **19.076 -> 10.312 ms/token** at 4096 per-layer slots, for **-2.7 ms/token** end to end.  What was
    /// true is that the ADMISSION POLICY gave every slot to the first position, which is why the earlier
    /// measurement found nothing - see `expert_cache_per_layer`.
    int expert_cache = 0;
    std::array<int, 3> expert_cache_remote{}; ///< CUDA1..3 slots; CUDA0 keeps dense/state/MTP
    std::string expert_cache_remote_placement = "stripe"; ///< stripe experts or assign complete layers to CUDA1..3
    bool expert_cache_cpu_order = false;
    /// **R4.2g.  ROUND 328 MEASURED THAT THE GLOBAL ADMISSION POLICY CANNOT WORK, AND THIS IS THE FIX.**
    /// The default policy hands out slots in arrival order from one counter shared by all 48 layers, so the
    /// first `n_slots` distinct pairs - about 26 LAYERS OF POSITION 0 - take every slot and hits are confined
    /// to them.  Measured at 256 slots: **1781 of 60000 = 2.97%**, against **21.4%** for 8 slots per layer and
    /// **70.4%** for 64, from `Memory/cache_allocation.py` on the same run's routing.  Off by default.
    bool expert_cache_per_layer = false;
    /// The PLE gather's prefetch, as an A/B arm.  The gather measured 2.10-2.61 ms/token because its sixteen
    /// row reads are sixteen SEPARATE page faults into a 26.8 GB mapping; see `ple_prefetch_enable`.
    bool no_ple_prefetch = false;
    /// R4.2e: a `profile.bin` from `tools/make_profile.py`.  **When given, it decides residency instead of the
    /// compulsory-miss policy**, which is the whole point: a profile ranked by routing frequency over a whole
    /// trace is what the plan's `h = 0.6447` refers to, and compulsory-miss measured 0.4864 because it fills
    /// with whatever the prompt touched FIRST.  Empty means no profile.  A GLM pack (--glm-pack) reads it too: its
    /// ranking decides each layer's VRAM slots and the warm-up's order (data/expert-profile-glm.bin).
    std::string expert_profile;
    /// #477 (--serve, opt-in): where to save what the expert tiers learned, as a profile `--expert-profile` reads
    /// (the resident experts first, then the routing counted); on QUIT and every `expert_profile_save_min` minutes
    /// between requests.  Empty (the default): nothing is written.  GLM packs (--glm-pack) only.
    std::string expert_profile_save;
    double expert_profile_save_min = 10.0;
    /// R4.2d: **ON by default**, because the measurement is unambiguous and the alternative is known-broken.
    /// Without it, 17 of 10,562 layers had the hit work done when the pool returned; with it, 9,190.  The
    /// A/B arm is `--no-hit-poke`.
    bool no_hit_poke = false;
    /// R0.9: capture each layer as THREE graphs and time them from outside the capture, which is the only
    /// valid way to get a per-stage table on the real graph.  Prints and exits; it is a measurement, not a run.
    bool gpu_stages = false;
    bool stats = false;
    bool shared_late = false;          ///< plan v0.3 P3 A/B: shared expert inside post[l] (old order)
    bool keep_canonical = false;       ///< plan v0.3 P1 A/B: load canonical copies of natively served tensors
    bool no_token_graph = false;       ///< plan v0.3 P3 A/B: two graphs per layer instead of one per token
    bool no_fused_gr = false;          ///< plan v0.3 P3 A/B: the six-kernel native gr_read + separate gr_write
    bool no_fast_attn = false;         ///< plan v0.3 P3 A/B: gather + one-block-per-head QSA attention
    bool no_publish_kernel = false;    ///< plan v0.3 P3 A/B: memcpy nodes for the doorbell and QSA step
    bool no_fused_gdn = false;         ///< plan v0.3 P3 A/B: llama.cpp-layout GDN step + separate out norm
    bool no_fast_select = false;       ///< plan v0.3 P7 A/B: FP64 row scores + bit-serial cell top-k
    /// Plan v0.3 P4: `--expert-cache auto` sizes the VRAM tier from what is free after the weights, the session
    /// and the KV state, minus this reserve for the graphs, the hit scratch and the head.
    int vram_reserve_mib = 700;
    bool vram_reserve_given = false;   ///< --vram-reserve-mib on the command line (--glm-pack: a small card's default gives way)
    /// `--vram-reserve-later-mib N` (--glm-pack): the reserve on a layer split's later GPUs (-1: --vram-reserve-mib's)
    int vram_reserve_later_mib = -1;
    /// Plan v0.3 P5: batched prompt processing in chunks of this many tokens (0 = the token path).
    int64_t prefill_chunk = 0;
    /// `--prefill auto`: the largest chunk (up to 8192) whose buffers the expert cache can lend.  Every expert a chunk
    /// routes to is streamed once per chunk, so a bigger chunk streams fewer bytes per token (the "ubatch" effect).
    bool prefill_auto = false;
    std::string prefill_arg;           ///< `--prefill` as given (the GLM engine reads it as SAGUARO_GLM_PREFILL)
    std::string dflash;                ///< `--dflash`: a DFlash2 draft GGUF for the GLM engine (SAGUARO_GLM_DFLASH)
    bool no_split_rows = false;        ///< plan v0.3 P4 A/B: one whole expert per pool thread
    /// Plan v0.3 P5: the prompt path borrows the top expert-cache slots for its buffers and refills them after
    /// the prompt (default); `--no-prefill-borrow` reserves the buffers' VRAM for the whole session instead.
    bool no_prefill_borrow = false;
    /// Plan v0.3 P5 validation: batch only positions [0, P) and run the rest of the prompt through the token path
    /// (teacher-forced), so the logits of positions >= P - which depend on the batched state - can be scored
    /// against the oracle at many positions.  0 = the whole prompt but the last position.
    int64_t prefill_until = 0;
    /// Plan v0.3 P6: after every processed position, append the residual after the last layer (hc x n_embd
    /// floats, the MTP draft head's input) to this file.  Token path only.
    std::string dump_final_r;
    /// Plan v0.3 P6: speculative decoding with a verify window of this many tokens (the last accepted token and
    /// spec-1 drafts); 0 = plain decode.  `spec_oracle` drafts from a token file (the expected continuation, for
    /// the exactness test); `spec_corrupt` N > 0 replaces every Nth draft with a wrong token.
    int spec = 0;
    std::string spec_oracle;
    std::string spec_follow, spec_follow_save;   // --glm-pack: SAGUARO_GLM_SPEC_FOLLOW / _SAVE (glm_pack_generate)
    int spec_corrupt = 0;
    /// Plan v0.3 P6: the MTP draft layer's runtime directory (tools/mtp_rt.py); drafts come from it.
    std::string mtp;
    int64_t mtp_window = 32768;   ///< the draft layer attends to the last N cells (0 = every cell)
    /// Plan v0.3 P6: the share (0..1) of each layer's distinct missed experts the GPU reads over PCIe from the
    /// pinned arena while the CPU computes the rest (verify windows).
    double pcie_frac = -1.0;   ///< < 0: the model's default (0.2 direct for the Q2_0 pack, 0.55 DMA for native packs)
    std::string pcie_mode = "auto";   ///< auto | dma | kernel | direct
    /// Plan v0.3 P6: every `adapt_every` rounds, swap up to `adapt_swaps` of the most-routed missing experts into
    /// the VRAM tier in place of the least-routed resident ones (decayed counts).  0 = static residency.
    int adapt_every = 4;
    /// Plan v0.3 P6: a draft enters the verify window only while every draft before it (and itself) has at least
    /// this probability under the draft layer; 0 = always --spec-1 drafts.
    double spec_min_p = 0.0;
    bool spec_min_p_set = false;   ///< --spec-min-p given (the DFlash decode's default cut is 0.9 otherwise)
    /// Stop when the model emits an end-of-turn token (<|endoftext|> 248044, <|im_end|> 248046, or --eos-ids).
    bool stop_eos = false;
    bool eos_given = false;    ///< --stop-eos or --eos-ids appeared: the glm path stops only when asked (defaults are qwen's)
    std::vector<int64_t> eos_ids = {248044, 248046};
    bool spec_split = false;   ///< opt-in split verify window (the overlap study: exact, ~7% slower)
    /// --serve, multi-GPU layer split: "K" or "K1,K2,.." (the first layer of each later stage) or "auto" (placed
    /// from each GPU's free VRAM); empty = one GPU
    std::string layer_split;
    /// the later stages' devices "D1,D2,.." (default: the next visible GPUs; "0" with one K: both stages on this
    /// GPU, sharing everything - the bit-exact A/B of the hand-off)
    std::string split_device;
    /// Plan v0.3 P8: stay resident and take requests on stdin (see the --serve block in main).
    bool serve = false;
    /// The vision path: keep a per-cell (t, h, w) rotary position table so --serve can take GENI requests.
    bool vision = false;
    int adapt_swaps = 96;
    /// --serve: how many conversation checkpoints to keep between requests (0 = every request reads its whole
    /// prompt again, the v0.1.2 behaviour).  One is the GDN recurrence of the 36 layers, the QSA indexer tails and
    /// the PLE history (~118 MB of host RAM); the KV cache itself is positional and stays where it is.
    int prompt_cache = 6;
    /// --serve: also keep a checkpoint every N freshly read prompt tokens (0 = only at the last turn boundary)
    int64_t prompt_cache_every = 16384;
    /// --serve: a prompt read from token 0 is also checkpointed at its first turn boundary - the end of the system
    /// prompt, which every chat of the same client shares - when that is at least N tokens (0 = never)
    int64_t prompt_cache_root = 2048;
    /// --serve: the token that opens a chat turn (<|im_start|>).  The last one in a prompt is where the chat's
    /// history ends and the new assistant turn begins, which is the checkpoint the next request can reuse.
    int64_t turn_token = 248045;
    /// --serve: a text part of the prompt of at most N tokens (a chat message, the assistant header, a short tool
    /// result) goes through the verify windows, S tokens at a time, instead of the batched prompt path (0 = always
    /// the batched path)
    int64_t short_read = 64;
    /// The suffix drafter (prompt lookup): when the text being written repeats an earlier stretch of the context (code
    /// edits, quoted input, tool-call JSON) by at least this many tokens, the window may be filled with what followed
    /// it there instead of the MTP's drafts, where the MTP's own first guess agrees and the draft policy expects it to
    /// pay (maya/spec/draft_policy.hpp).  On by default; 0 = MTP only.
    int suffix_draft = 3;
    /// The MTP's own window cap (0 = --spec): with --spec 6 --mtp-max-t 4 the long windows come from suffix matches.
    int mtp_max_t = 0;
    /// A control vector on the residual stream (maya/kernels/cvec.hpp), with llama.cpp's flags: the
    /// `experimental-speed-projection` profile passes `--control-vector-scaled FILE:1.0 --control-vector-layer-range
    /// 4 44 --cvec-mode project --cvec-dir per-layer`.  None by default; --serve switches a loaded one per request.
    std::vector<std::pair<std::string, float>> cvec_files;
    int cvec_first = -1, cvec_last = -1;   ///< llama.cpp's defaults: 1 .. the last layer
    int cvec_mode = 1;                     ///< 0 = project, 1 = add (llama.cpp's default)
    int cvec_single = -1;                  ///< --cvec-dir single:L (project mode): layer L's direction everywhere
};

void usage() {
    std::fprintf(stderr,
                 "saguaro-glm --pack DIR --tokens \"1,2,3\" [options]\n"
                 "\n"
                 "  --pack DIR           the pack directory (default pack/full)\n"
                 "  --glm-pack DIR       a GLM-5.3-Flash pack (M3.1): the verified Glm5Model runner behind the\n"
                 "                       same wire contract (T <id> / DONE; --serve reads GEN lines)\n"
                 "  --dflash GGUF        (--glm-pack) draft with a DFlash2 block drafter made for the model: each\n"
                 "                       round it drafts up to --spec T - 1 tokens (default its block size - 1), one\n"
                 "                       verify window checks them; --spec-min-p P stops a draft at a selector\n"
                 "                       confidence below P (default 0.5 on 3+ GPUs, 0.95 on one; two GPUs: at\n"
                 "                       least one draft, up to 4 while the confidences' product stays at 0.9 -\n"
                 "                       SAGUARO_GLM_DFLASH_MIN / _CUM).  The model's\n"
                 "                       MTP block is not loaded with it.  Env: SAGUARO_GLM_DFLASH\n"
                 "  --spec-follow PATH   (--glm-pack) speed A/B: every request decodes this continuation (token ids,\n"
                 "                       as --tokens-file takes them) instead of its own picks and ends with it; the\n"
                 "                       drafting decodes keep exactly the drafts that match it, so two builds or\n"
                 "                       settings process the same text.  Env: SAGUARO_GLM_SPEC_FOLLOW\n"
                 "  --spec-follow-save PATH  (--glm-pack) write each request's decoded token ids there (for the above).\n"
                 "                       Env: SAGUARO_GLM_SPEC_FOLLOW_SAVE\n"
                 "  --expert-cache auto|N  (--glm-pack) the expert pool in VRAM: auto (the default) takes the free\n"
                 "                       VRAM less the reserve and checks it again once written; N caps the first\n"
                 "                       GPU's at N of its largest experts.  Env: SAGUARO_GLM_EXPERT_CACHE\n"
                 "  --vram-reserve-mib N (--glm-pack) VRAM kept free beside the expert pool (default 700; a small\n"
                 "                       card's gives way to the pool's floor).  Env: SAGUARO_GLM_RESERVE_MB\n"
                 "  --vram-reserve-later-mib N  (--glm-pack) the same on a split's later GPUs (default: the above).\n"
                 "                       Env: SAGUARO_GLM_RESERVE_LATER_MB\n"
                 "  --glm-ram-resident   the pinned RAM tier holds every expert VRAM does not, and nothing is evicted\n"
                 "                       to disk. The start fails if the tier cannot hold them (raise SAGUARO_GLM_RAM_GB\n"
                 "                       or lower --max-context). Env: SAGUARO_GLM_RAM_RESIDENT=1\n"
                 "  --glm-ram-slack N    extra RAM-tier slots per MoE layer (SAGUARO_GLM_RAM_SLACK; default 16 when\n"
                 "                       resident, 0 otherwise)\n"
                 "  --parallel N         --serve: up to N requests decode at once (1-8, default 1). One GPU decodes\n"
                 "                       them together (each weight read once for all); a split pipelines them\n"
                 "                       across its cards. Each extra conversation keeps its own context state\n"
                 "  --tokens LIST        the prompt as comma-separated token IDS (required)\n"
                 "  --tokens-file PATH   pretokenized prompt, commas or whitespace (alternative to --tokens)\n"
                 "  --ple-gguf PATH      required PLE table (original second GGUF shard)\n"
                 "  --no-ple             explicit diagnostic ablation of the PLE layer\n"
                 "  --ple-io direct|mmap  n-gram table reads (plan v0.3 P2). direct (default): unbuffered SSD\n"
                 "                       reads, the table never enters RAM or the file cache; mmap: A/B arm\n"
                 "  --ple-row-cache N    bounded cache of fetched rows, 90 B each (default 1048576; 0 = off)\n"
                 "  --ple-inflight N     outstanding SSD reads (default 64)\n"
                 "  --ple-delay-us U     fault injection: each row read completes no earlier than U us\n"
                 "  --ple-sync-submit    A/B arm: submit table reads on the token thread (default: an I/O thread)\n"
                 "  --kv fp16|int8       KV storage (plan v0.3 P7): int8 codes + fp16 scale per 64 values, half the\n"
                 "                       VRAM; default fp16 until gate G-C accepts int8\n"
                 "  --kv q4_0            4-bit K/V after a Hadamard rotation (PR #21): half of int8's memory,\n"
                 "                       slightly lower precision\n"
                 "  --kv k8v4            hybrid: INT8 K (exact attention scores) + rotated Q4_0 V, 816 B/cell\n"
                 "                       (vs int8's 1,056); not with --kv-resident\n"
                 "  --kv-resident N      KV streaming: keep N cells of each QSA layer in VRAM (min 20480) and the\n"
                 "                       whole K/V in pinned RAM; the freed VRAM goes to expert slots. 0 (default):\n"
                 "                       all of it in VRAM. A context of N cells or fewer is not streamed. With\n"
                 "                       --glm-pack: N positions of each DSA layer's latent cache\n"
                 "                       (SAGUARO_GLM_KV_RESIDENT)\n"
                 "  --kv-streaming on|off  KV streaming as a switch (the setup writes `on` into the config): on keeps\n"
                 "                       --kv-resident's N cells in VRAM, 32768 when it gives none; off keeps all\n"
                 "                       of the KV in VRAM whatever --kv-resident says\n"
                 "  --stream-token       enqueue token work on the session stream (experimental)\n"
                 "  --check-logits       copy and check all logits in the stream-token path\n"
                 "  --gr-fp32-activations  experimental CUDA-oracle GR activation precision\n"
                 "  --gr-native-mmvf      experimental pinned GR norm/projections; implies FP32 activations\n"
                 "  --native-bf16         experimental CUDA-oracle SSM/router/indexer BF16 projections\n"
                 "  --native-bf16-extra   experimental CUDA-oracle PLE/shared gate BF16 projections\n"
                 "  --native-ple-key      experimental native PLE key; requires --native-dense-gguf\n"
                 "  --native-moe-combine  experimental pinned CUDA routed/shared combination\n"
                 "  --native-gdn          experimental pinned CUDA GDN norms/gates/recurrence\n"
                 "  --native-flash-attn-short  diagnostic pinned vector attention; --max-context <=256\n"
                 "  --native-qsa-indexer  experimental pinned indexer key cache and pooling\n"
                 "  --native-qsa          experimental pinned QSA normalization and output gate\n"
                 "  --native-rope         experimental pinned text-only CUDA rotary arithmetic\n"
                 "  --native-ple-postops  experimental pinned PLE postprojection arithmetic\n"
                 "  --native-router       experimental pinned CUDA 512-expert top-10 routing\n"
                 "  --cpu-oracle-q8-0     experimental pinned CPU expert quantization and dot reduction\n"
                 "  --native SHARD1      every full-context native path at once (plan v0.3 P1): stream-token,\n"
                 "                       GR MMVF, BF16, head, dense + PLE key, MoE combine, GDN, router, QSA,\n"
                 "                       indexer, RoPE, PLE postops, and the CPU q8_0 contract unless the\n"
                 "                       expert cache is on. Individual --native-* flags stay for A/B.\n"
                 "  --native-head-gguf PATH  native Q5_K head from model shard 1; requires --stream-token\n"
                 "  --native-dense-gguf PATH native GDN/QSA/shared projections; repeat for each source model shard\n"
                 "  --expert-cache-cpu-order  experimental GPU expert reduction matching CPU order\n"
                 "  --max-new N          tokens to generate (default 16)\n"
                 "  --max-context N      KV/state capacity (default 4096)\n"
                 "  --greedy             argmax (the default)\n"
                 "  --seed S             enable sampling with this Philox seed\n"
                 "  --top-k N --top-p F --temperature F\n"
                 "  --dump-logits PATH   write one line of raw logits per position\n"
                 "  --logits-stride N    store every Nth row plus final input (default 1); N>1 requires --max-new 1\n"
                 "  --dump-residual PATH write the final R (hc x n_embd, f32) for head bisection\n"
                 "  --dump-layers PATH   write R after EVERY layer, per position: the C1 bisection ladder\n"
                 "  --dump-halves PATH   write both halves' block_out and inject per layer: the half bisection\n"
                 "  --dump-routing PATH  write the routed expert ids and weights per layer per position (P0.S8)\n"
                 "  --no-capture         run the layers directly instead of replaying graphs\n"
                 "  --shared-late        A/B: shared expert after the CPU pool (default: overlapped with it)\n"
                 "  --keep-canonical     A/B: also load canonical copies of natively served tensors (more VRAM)\n"
                 "  --vision             --serve takes images too (GENI requests; embeddings from maya-vision)\n"
                 "  --prompt-cache N     --serve: keep N conversation checkpoints between requests (default 6, ~118 MB\n"
                 "                       of RAM each; 0 = read every prompt from the start)\n"
                 "  --prompt-cache-every N  --serve: also checkpoint every N fresh prompt tokens (default 16384, 0 = off)\n"
                 "  --turn-token ID      --serve: the token that opens a chat turn (default 248045, <|im_start|>)\n"
                 "  --short-read N       --serve: read at most N fresh text tokens through the decode windows instead\n"
                 "                       of the batched prompt path (default 64, 0 = off)\n"
                 "  --suffix-draft N     prompt lookup: draft from an earlier repeat of the last N+ tokens of context\n"
                 "                       when it pays (default 3; 0 = MTP only)\n"
                 "  --mtp-max-t M        cap the MTP's windows at M tokens (0 = --spec; longer ones come from suffixes)\n"
                 "  --control-vector-scaled FILE:SCALE[,...]  a control vector GGUF on the residual stream (llama.cpp's\n"
                 "                       format; --control-vector FILE = scale 1).  --serve: requests switch it (cvec=0|1)\n"
                 "  --control-vector-layer-range A B  the layers it follows (inclusive; default 1 .. the last)\n"
                 "  --cvec-mode add|project  h += s v (default) or h -= s (h.v) v with v unit\n"
                 "  --cvec-dir per-layer|single:L  each layer's own direction (default) or layer L's everywhere (project)\n"
                 "  --no-token-graph     A/B: two graphs per layer (the host launches each) instead of one per token\n"
                 "  --no-fused-gr        A/B: the six-kernel hyper-connection read and a separate write (native)\n"
                 "  --prefill CHUNK      batched prompt processing in chunks of CHUNK tokens (needs --native); auto =\n"
                 "                       the largest chunk up to 8192 whose buffers the expert cache can lend\n"
                 "  --no-pool            skip the CPU expert pool (the GPU-only floor)\n"
                 "  --sync-every-layer   debug: synchronise after every layer\n"
                 "  --ple-gguf PATH      the n-gram/PLE shard.  WITHOUT IT LAYER 1's PLE IS SILENTLY SKIPPED,\n"
                 "                       which changes every number downstream - pass it for any real run\n"
                 "  --dump-mixed PATH    write the post-attention residual (n_embd, f32)\n"
                 "  --stage-timing       per-stage KERNEL-COUNT shares.  NOT a time profile: an uncaptured\n"
                 "                       event interval includes host gaps, so run with --gpu-only-full first\n"
                 "  --graph-only         MEASURE: replay the 48 `pre` graphs only.  OMITS the 48 `post` graphs\n"
                 "                       and the LM head, so it is NOT the GPU floor (R0.3, Memory/ERRORS.md A4)\n"
                 "  --gpu-only-full      MEASURE: replay pre+post for all 48 layers plus the LM head, no pool.\n"
                 "                       THE TRUE PER-TOKEN GPU FLOOR.  Quote this one, not --graph-only.\n"
                 "  --stats              print the per-stage breakdown\n"
                 "  --gpu-stages         R0.9: capture the layer as three graphs (mixer / ffn+router / post)\n"
                 "                       and time them from OUTSIDE the capture.  The per-stage table on the\n"
                 "                       real graph that --stage-timing cannot give.  Prints and exits.\n"
                 "  --expert-profile P   R4.2e: pre-load the VRAM tier from a `profile.bin` (see\n"
                 "                       tools/make_profile.py) instead of admitting on first use.\n"
                 "  --expert-profile-save P  --serve, #477 (--glm-pack): save what the expert tiers learned (the\n"
                 "                       experts in VRAM, then the routing counted) as a profile at P, on QUIT\n"
                 "                       and every --expert-profile-save-every MIN minutes (default 10; 0 = on\n"
                 "                       QUIT only) between requests; start from it with --expert-profile P\n"
                 "  --no-hit-poke        R4.2d's A/B arm.  The hit path pokes the driver once right after its\n"
                 "                       launch so the GPU starts while the CPU pool runs; without it the work\n"
                 "                       waits for the next driver entry and does not overlap at all.\n"
                 "  --no-ple-prefetch     A/B arm: read the PLE table's sixteen rows one at a time, instead of\n"
                 "                       issuing them in one PrefetchVirtualMemory call.\n"
                 "  --expert-cache N     R4: keep N expert blobs resident in VRAM and compute their rows on the\n"
                 "                       GPU via `moe_hit_grouped_s2`.  DEFAULT 0.  Measured at 4096 slots\n"
                 "                       with --expert-cache-per-layer: 54.4%% hits, CPU pool drain 19.1 -> 10.3\n"
                 "                       ms/token, -2.7 ms/token end to end.\n"
                 "  --expert-cache-device1 N  pre-fill N experts on CUDA1 (experimental)\n"
                 "  --expert-cache-device2 N  pre-fill N more experts on CUDA2\n"
                 "  --expert-cache-device3 N  pre-fill N more experts on CUDA3\n"
                 "  --expert-cache-remote-placement stripe|layer  distribute expert ranks or whole\n"
                 "                       layers across CUDA1..3 (default: stripe)\n"
                 "  --expert-cache-per-layer  R4.2g: give each layer its OWN slots instead of letting the first\n"
                 "                       position take all of them.  The default policy fills in arrival order\n"
                 "                       from one shared counter, so 256 slots went to ~26 layers of position 0\n"
                 "                       and measured **2.97%%**.  Per-layer, the same routing gives 21.4%% at 8\n"
                 "                       slots/layer and 70.4%% at 64.\n"
                 "  --no-host-worker     R2.2: the A/B arm.  By default the HOST THREAD joins the drain, so the\n"
                 "                       pool is six threads on six cores instead of five plus an idle core;\n"
                 "                       this flag restores the five-worker form for comparison on `pool phases`.\n"
                 "  --pool-workers N     R2.2: CPU expert pool worker count.  Default 0 = every physical core\n"
                 "                       except the one the host loop spins on.  A sweep is how the pool's\n"
                 "                       deviation from `cpu_s2` is attributed.\n"
                 "  --experts-from-gguf auto|on|off  a native pack without experts.bin: the experts straight from the\n"
                 "                       model GGUF - those the GPUs do not hold kept in RAM, hottest first, as far as\n"
                 "                       the free RAM goes (SAGUARO_EXPERT_RAM_GB caps it), the rest read when routed\n"
                 "                       (auto, the default: when the RAM cannot hold every expert)\n"
                 "  --mmap-experts       R2.1: opt OUT of the resident expert arena, back to MapViewOfFile.\n"
                 "                       The A/B arm: the mmap's rate depends on the OS page cache holding\n"
                 "                       34 GB, and measured 71.97 vs 34.78 ms/token cold vs warm.\n"
                 "  --resident-cpu-experts  with mmap and a static profile, keep CPU misses resident in ordinary RAM.\n"
                 "                       Borrowed GPU-cache entries may read from mmap during prompt prefill.\n");
}


bool parse_i64_list(const char* s, std::vector<int64_t>& out, std::string& err) {
    out.clear();
    std::string text(s);
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream input(text);
    std::string token;
    while (input >> token) {
        int32_t id = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || id < 0) {
            err = "invalid token id: expected an integer in [0, 2147483647]";
            out.clear();
            return false;
        }
        out.push_back(id);
    }
    if (out.empty()) { err = "token list was empty"; return false; }
    return true;
}

// ---- issue #31: what the watchdog prints before it stops a stalled engine
struct MemSample {
    unsigned long long faults = 0, rss_mib = 0, avail_mib = 0, commit_mib = 0;   // faults: hard (Linux) / all (Windows)
};
MemSample mem_sample() {
    MemSample m;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) {
        m.faults = pmc.PageFaultCount;
        m.rss_mib = pmc.WorkingSetSize >> 20;
        m.commit_mib = pmc.PagefileUsage >> 20;
    }
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) m.avail_mib = ms.ullAvailPhys >> 20;
#else
    if (std::FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[4096];
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        std::fclose(f);
        const char* s = std::strrchr(buf, ')');   // fields after the command name: 3 state ... 12 majflt
        for (int field = 2; s && field < 12; ++field) s = std::strchr(s + 1, ' ');
        if (s) m.faults = std::strtoull(s + 1, nullptr, 10);
    }
    auto kb = [](const char* path, const char* key) -> unsigned long long {
        unsigned long long v = 0;
        if (std::FILE* f = std::fopen(path, "r")) {
            char line[256];
            const size_t kl = std::strlen(key);
            while (std::fgets(line, sizeof line, f))
                if (std::strncmp(line, key, kl) == 0) { v = std::strtoull(line + kl, nullptr, 10); break; }
            std::fclose(f);
        }
        return v;
    };
    m.rss_mib = kb("/proc/self/status", "VmRSS:") >> 10;
    m.commit_mib = kb("/proc/self/status", "VmSwap:") >> 10;
    m.avail_mib = kb("/proc/meminfo", "MemAvailable:") >> 10;
#endif
    return m;
}

void stall_report(std::FILE* f, uint64_t layers_during) {
    maya::core::Progress& p = maya::core::progress();
    std::fprintf(f, "saguaro serve: stall report (engine %s): stage \"%s %lld\" for %lld s; %llu layers served since the "
                    "last finished step (0 = stopped, more = slow)\n", MAYA_VERSION, p.where.load(),
                 (long long) p.detail.load(), (long long) ((maya::core::progress_now_ms() - p.since_ms.load()) / 1000),
                 (unsigned long long) layers_during);
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::fprintf(f, "  2 s later:\n");
        }
        if (auto fn = maya::core::diag_pool_fn().load()) fn(f);
        if (auto fn = maya::core::diag_verify_fn().load()) fn(f);
        const MemSample m = mem_sample();
        std::fprintf(f, "  memory: %llu MiB resident, %llu MiB %s, %llu MiB RAM available; %llu %s\n", m.rss_mib,
                     m.commit_mib,
#if defined(_WIN32)
                     "committed", m.avail_mib, m.faults, "page faults so far"
#else
                     "in swap", m.avail_mib, m.faults, "major page faults so far"
#endif
        );
        std::fflush(f);
    }
#if defined(_WIN32)
    // every thread's stack, to read against this build's PDB: a few MB beside the engine's working directory
    if (HMODULE dbg = LoadLibraryA("dbghelp.dll")) {
        using Fn = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, int, void*, void*, void*);
        if (auto write = (Fn) GetProcAddress(dbg, "MiniDumpWriteDump")) {
            char path[64];
            std::snprintf(path, sizeof path, "maya-stall-%lu.dmp", (unsigned long) GetCurrentProcessId());
            HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (h != INVALID_HANDLE_VALUE) {
                const int kThreadInfo = 0x1000;   // MiniDumpWithThreadInfo; MiniDumpNormal = 0
                const BOOL ok = write(GetCurrentProcess(), GetCurrentProcessId(), h, kThreadInfo, nullptr, nullptr, nullptr);
                CloseHandle(h);
                char full[MAX_PATH];
                if (!GetFullPathNameA(path, MAX_PATH, full, nullptr)) std::snprintf(full, sizeof full, "%s", path);
                std::fprintf(f, "  %s the thread stacks to %s (attach it to the issue)\n", ok ? "wrote" : "could not write",
                             full);
            }
        }
    }
#endif
}

// issue #29: a request whose heartbeat (tokens, prompt chunks or layers, verify windows) stops for SAGUARO_WATCHDOG_S
// seconds (default_s when unset; 0 = off) is stuck on a flag nobody will raise - end the engine with where it was, so
// the server starts it again instead of the GPU spinning forever.  Issue #31: before it does, it reports what every
// part was doing (stall_report), so one occurrence says where the wait is.  Issue #40: the GLM serve has it too.
void start_serve_watchdog(int default_s) {
    const char* ws = std::getenv("SAGUARO_WATCHDOG_S");
    const int limit = ws ? std::atoi(ws) : default_s;
    if (limit <= 0) return;
    std::thread([limit] {
        maya::core::Progress& p = maya::core::progress();
        uint64_t last = p.beats.load(), ticks_at = p.ticks.load();
        auto since = std::chrono::steady_clock::now();
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            const auto now = std::chrono::steady_clock::now();
            const uint64_t b = p.beats.load();
            if (!p.busy.load() || b != last) { last = b; ticks_at = p.ticks.load(); since = now; continue; }
            if (now - since < std::chrono::seconds(limit)) continue;
            std::fprintf(stderr, "saguaro serve: no progress for %d s during a request (%s %lld) - stopping "
                                 "the engine so the server starts it again (issue #29)\n",
                         limit, p.where.load(), (long long) p.detail.load());
            stall_report(stderr, p.ticks.load() - ticks_at);
            std::fflush(stderr);
            // not abort(): Windows Error Reporting can hold the process open, and a core dump of tens of GB of
            // pinned expert tiers takes minutes - the stall report above already says where it stopped
            std::_Exit(3);
        }
    }).detach();
}


int argmax(const std::vector<float>& v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[best]) best = (int) i;
    return best;
}


// ================================ the glm5-next pack path (M3.1) ================================
//
// A GLM-5.3-Flash pack (tools/iq_pack.py output) runs the VERIFIED runner - Glm5Model, the same
// path glm_pack_test drives (docs/GLM5-FLASH.md §9) - behind this driver's wire contract:
// `T <id>` per generated token on stdout, `ERR <message>` when a request cannot run, the DONE
// summary at the end, and --serve reading `GEN <max_new> <id,id,...>` lines (QUIT ends).
//
// NOT wired here yet, and refused rather than approximated: prompt-prefix reuse and per-request
// state continuity across GEN lines (each GEN starts a fresh sequence), drafts,
// the batched prompt path (the prompt runs one token at a time), images, control vectors and
// layer splits.  Sampling is the engine's own sampler (maya::kernels::sample_tokens, greedy by
// default), so the top-k/top-p/temperature chain behaves exactly as in the qwen4exp path - the
// penalties too (#273): the request's history (its prompt, then what it wrote) rides in
// SamplerParams::host_history, and every pick counts its last penalty_last_n tokens
// (Glm5Model::pick_tokens; a verify window's row t: those, then the drafts before it).
// The stop set is GLM's own, from the GGUF metadata (the ids llama.cpp marks EOG): 154820
// <|endoftext|> (eos), 154827 <|user|> (eot, the assistant yields the floor) and 154829 <|observation|> (a tool
// call waits for its result); --eos-ids
// overrides it.  Serve stops on them by default; non-serve stops only when asked (--stop-eos or
// --eos-ids), like the qwen path.
// a variable of this process's environment (the GLM model reads its switches from there)
static void set_env(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

// A failure in the request's own work - an allocation, a launch - is thrown (maya/prefill/failure.hpp) and
// caught here: it ends this request with an error, not the process.  The engine lives and the next request runs.
template <typename F>
static bool guarded(F&& f, std::string& err) {
    try {
        return f();
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

// SAGUARO_GLM_CONV_EXPERTS=1: one VRAM expert allocation a layer over the live conversations (bench/glm-conv/replay*.py):
// every running / OPEN conversation's top K0 (the floor), the rest by sum_c w_c share_c(e) + recall + eps global share;
// published as level flags (Glm5Model::conv_publish) - the engine's victims go lowest level first, its preload highest.
// A conversation = the prompts that extend one another past the system prompt (contexts change between turns).
namespace {
struct ConvAlloc {
    static constexpr int kL = 128, kE = 512;
    struct Conv {
        int id = 0;
        std::vector<int32_t> tokens;   // its last prompt + answer
        std::vector<float> cnt;        // decode routes, x DECAY a turn
        std::vector<float> turn;       // this turn's
        std::vector<float> recall;     // recalled experts (brought in at its last resume, then hit): their share there
        std::vector<uint8_t> brought;  // this resume's experts that were outside the allocation before it
        std::chrono::steady_clock::time_point last{};
        int running = 0;
        size_t lcp = 0;
        // CONV_UNION: its own decode routes (the route tags), decayed in time; when it last decoded; its tokens
        std::chrono::steady_clock::time_point decayed{}, decoded{};
        double tok = 0;
        std::vector<uint64_t> frozen;   // CONV_SHADOW: its sequence's lru clocks when its last request ended
    };
    bool on = false, open_only = false;
    // SAGUARO_GLM_CONV_UNION=1 (bench/glm-conv/replay-rec.py's winner, alpha 0): the target a layer = the top slots of
    // sum_c w_c (its own decode share + BETA x the global share / (1 + its tokens / TOK0)) + 0.01 x the global share;
    // w_c = 1 while it decodes or reads, then halving every IDLE_HALF_S, 0 after DROP_S; its counts halve every H_S
    bool uni = false;
    bool shadow = false;   // SAGUARO_GLM_CONV_SHADOW=1: the engine computes the target (Glm5Model::shadow_target)
    double u_h = 30.0, u_idle_half = 5.0, u_drop = 35.0, u_beta = 1.0, u_tok0 = 50.0;
    int u_layers = 0;
    std::vector<std::vector<uint32_t>> seq_prev;   // (each sequence's tagged counts at the last attribution)
    double open_s = 5.0, w_open = 0.25, w_closed = 0.0, half_s = 60.0, eps = 0.1, decay = 0.5, tick_s = 2.0;
    int k0 = 16, recall_cap = 0, closed_floor = 0, slack = 4, max_convs = 16, next_id = 1;
    std::vector<Conv> convs;
    std::vector<float> glob;
    double glob_since = 0;
    std::vector<uint32_t> rc_prev;
    std::vector<uint8_t> flags;   // the last published allocation
    std::chrono::steady_clock::time_point last_pub{};
    std::vector<int> slots;
    std::vector<size_t> blob;
    const maya::core::Glm5Model* model = nullptr;

    static double num(const char* k, double d) {
        const char* v = getenv(k);
        return v != nullptr && *v != '\0' ? std::atof(v) : d;
    }
    void init(const maya::core::Glm5Model& m) {
        on = num("SAGUARO_GLM_CONV_EXPERTS", 0) != 0;
        open_only = !on && getenv("SAGUARO_GLM_CONV_OPEN_S") != nullptr;
        if (!on && !open_only) return;
        open_s = num("SAGUARO_GLM_CONV_OPEN_S", 5.0);
        k0 = (int) num("SAGUARO_GLM_CONV_FLOOR", 16);
        w_open = num("SAGUARO_GLM_CONV_W_OPEN", 0.25);
        // past the window: its shares at W_CLOSED halving every HALF_S, and up to RECALL_CAP of its recalled experts a
        // layer kept as floor (the cap halving every HALF_S) - both 0 = closed is closed (bench/glm-conv replay3-5)
        w_closed = num("SAGUARO_GLM_CONV_W_CLOSED", 0.0);
        recall_cap = (int) num("SAGUARO_GLM_CONV_RECALL_CAP", 0);
        // CLOSED_FLOOR: a closed conversation keeps its top n a layer as floor, n halving every HALF_S (at most K0 a layer
        // for all of them, the most recently active first)
        closed_floor = (int) num("SAGUARO_GLM_CONV_CLOSED_FLOOR", 0);
        half_s = num("SAGUARO_GLM_CONV_HALF_S", 60.0);
        eps = num("SAGUARO_GLM_CONV_EPS", 0.1);
        tick_s = num("SAGUARO_GLM_CONV_TICK_S", 2.0);
        slack = (int) num("SAGUARO_GLM_CONV_SLACK", 4);
        uni = (int) num("SAGUARO_GLM_CONV_UNION", 0) != 0 && (int) num("SAGUARO_GLM_CONV_EXPERTS", 0) != 0;   // (as the engine)
        shadow = (int) num("SAGUARO_GLM_CONV_SHADOW", 0) != 0 && (int) num("SAGUARO_GLM_CONV_EXPERTS", 0) != 0;
        if (shadow) {
            uni = false;
            std::fprintf(stderr, "glm conv: shadow on - each conversation's own lru set, the running ones merged by recency rank\n");
        }
        u_h = num("SAGUARO_GLM_CONV_UNION_H_S", 30.0);
        u_idle_half = num("SAGUARO_GLM_CONV_UNION_IDLE_HALF_S", 5.0);
        u_drop = num("SAGUARO_GLM_CONV_UNION_DROP_S", 35.0);
        u_beta = num("SAGUARO_GLM_CONV_UNION_BETA", 1.0);
        u_tok0 = num("SAGUARO_GLM_CONV_UNION_TOK0", 50.0);
        if (uni)
            std::fprintf(stderr, "glm conv: union on - counts halving every %.0f s, weight halving every %.0f s idle, 0 after "
                         "%.0f s, global prior %.2f / (1 + tokens / %.0f)\n", u_h, u_idle_half, u_drop, u_beta, u_tok0);
        model = &m;
        glob.assign((size_t) kL * kE, 0.0f);
        std::fprintf(stderr, "glm conv: %s - open %.1f s, floor %d, w_open %.2f, closed: w %.2f / recall cap %d (half %.0f s), "
                     "eps %.2f, tick %.1f s, slack %d, closed floor %d\n", on ? "allocation on" : "open window only (live rule)",
                     open_s, k0, w_open, w_closed, recall_cap, half_s, eps, tick_s, slack, closed_floor);
    }
    double age(const Conv& c, std::chrono::steady_clock::time_point now) const {
        return std::chrono::duration<double>(now - c.last).count();
    }
    // the conversation this prompt continues (past the system prompt), else a new one
    int admit(const std::vector<int32_t>& prompt, std::chrono::steady_clock::time_point now) {
        const size_t sys = maya::core::system_prompt_end(prompt.data(), prompt.size(), 154827);
        int best = -1;
        size_t bl = 0;
        for (int i = 0; i < (int) convs.size(); ++i) {
            const auto& t = convs[(size_t) i].tokens;
            size_t l = 0;
            const size_t n = std::min(t.size(), prompt.size());
            while (l < n && t[l] == prompt[l]) ++l;
            if (l > bl && l > sys + 16 && 2 * l >= t.size()) {
                bl = l;
                best = i;
            }
        }
        if (best < 0) {
            if ((int) convs.size() >= max_convs) {   // the least recently active one goes
                size_t o = 0;
                for (size_t i = 1; i < convs.size(); ++i)
                    if (convs[i].running == 0 && (convs[o].running > 0 || convs[i].last < convs[o].last)) o = i;
                convs.erase(convs.begin() + (long) o);
            }
            Conv c;
            c.id = next_id++;
            c.cnt.assign((size_t) kL * kE, 0.0f);
            c.recall.assign((size_t) kL * kE, 0.0f);
            convs.push_back(std::move(c));
            best = (int) convs.size() - 1;
        } else if (!uni) {   // (the union decays in time instead)
            for (float& x : convs[(size_t) best].cnt) x *= (float) decay;
        }
        Conv& c = convs[(size_t) best];
        c.tokens = prompt;
        c.turn.assign((size_t) kL * kE, 0.0f);
        c.brought.clear();
        c.lcp = bl;
        c.last = now;
        return c.id;
    }
    // the running conversations (the serve loop's requests, an admitting one included): counted afresh at each publish
    void set_running(const std::vector<int>& ids) {
        for (auto& c : convs) c.running = 0;
        for (int id : ids)
            if (Conv* c = find(id)) ++c->running;
    }
    Conv* find(int id) {
        for (auto& c : convs)
            if (c.id == id) return &c;
        return nullptr;
    }
    // the decode's route counts since the last call, split evenly over the conversations decoding now
    // CONV_UNION: a sequence's tagged counts from now on belong to the conversation admitted into it
    void seq_mark(int ctx) {
        if (!uni || ctx < 0) return;
        if ((int) seq_prev.size() <= ctx) seq_prev.resize((size_t) ctx + 1);
        maya::core::Glm5Model::route_counts_seq(ctx, seq_prev[(size_t) ctx]);
    }
    void attribute(const std::vector<std::pair<int, int>>& decoding_ctx) {
        std::vector<int> decoding;
        for (const auto& d : decoding_ctx) decoding.push_back(d.first);
        if (uni) {   // each decoding conversation's OWN routes (its sequence's tagged counts), decayed in time
            const auto now = std::chrono::steady_clock::now();
            if (u_layers == 0)
                for (int k : slots) u_layers += k > 0;
            std::vector<uint32_t> rs;
            for (const auto& d : decoding_ctx) {
                Conv* c = find(d.first);
                const int ctx = d.second;
                if (c == nullptr || ctx < 0) continue;
                if ((int) seq_prev.size() <= ctx) seq_prev.resize((size_t) ctx + 1);
                maya::core::Glm5Model::route_counts_seq(ctx, rs);
                auto& pv = seq_prev[(size_t) ctx];
                if (pv.size() != rs.size()) pv.assign(rs.size(), 0);
                const double dt = c->decayed.time_since_epoch().count() == 0
                                      ? 0.0 : std::chrono::duration<double>(now - c->decayed).count();
                c->decayed = now;
                if (dt > 0 && u_h > 0) {
                    const float f = (float) std::pow(0.5, dt / u_h);
                    for (float& x : c->cnt) x *= f;
                }
                double sum = 0;
                for (size_t i = 0; i < rs.size() && i < c->cnt.size(); ++i) {
                    const uint32_t dd = rs[i] - pv[i];
                    if (dd == 0) continue;
                    c->cnt[i] += (float) dd;
                    c->turn[i] += (float) dd;
                    sum += dd;
                }
                if (sum > 0) {
                    c->decoded = now;
                    c->tok += sum / (8.0 * std::max(1, u_layers));
                }
                pv.swap(rs);
            }
        }
        std::vector<uint32_t> rc;
        maya::core::Glm5Model::route_counts(rc);
        const size_t n = rc.size() / 2;
        if (rc_prev.size() != rc.size()) {
            rc_prev = rc;
            return;
        }
        std::vector<Conv*> ds;
        for (int id : decoding)
            if (Conv* c = find(id)) ds.push_back(c);
        double added = 0;
        for (size_t i = 0; i < n; ++i) {
            const uint32_t d = rc[n + i] - rc_prev[n + i];
            if (d == 0) continue;
            glob[i] += (float) d;
            added += d;
            if (!uni)
                for (Conv* c : ds) {
                    c->cnt[i] += (float) d / (float) ds.size();
                    c->turn[i] += (float) d / (float) ds.size();
                }
        }
        rc_prev.swap(rc);
        glob_since += added;
        while (glob_since >= 4.0 * 32768) {   // (the engine's aging pace)
            glob_since -= 4.0 * 32768;
            for (float& x : glob) x *= 0.5f;
        }
    }
    // the allocation: floors, then the fill by score; levels 1..64 by score, the floor flagged
    void publish(const char* why, std::chrono::steady_clock::time_point now) {
        const double unix_s = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        int open = 0, running = 0;
        for (const auto& c : convs) {
            running += c.running > 0;
            open += c.running == 0 && age(c, now) <= open_s;
        }
        maya::core::Glm5Model::conv_others_set(open);   // (CONV_OPEN_S alone: =live's "alone")
        last_pub = now;
        if (!on || shadow) return;   // (CONV_SHADOW: the engine's boundaries compute the target)
        if (std::none_of(slots.begin(), slots.end(), [](int k) { return k > 0; })) {   // (the pools as loaded: at the first use)
            slots = model->layer_slots();
            blob = model->layer_blob();
        }
        if (uni) {
            publish_union(why, now, unix_s, running, open);
            return;
        }
        std::vector<uint8_t> nf((size_t) kL * kE, 0);
        std::vector<double> w(convs.size());
        std::vector<int> rk(convs.size(), 0);   // closed: its recalled experts kept as floor, a layer
        std::vector<int> cf(convs.size(), 0);   // closed: its own top kept as floor, a layer (CLOSED_FLOOR)
        std::vector<char> fl(convs.size());
        for (size_t i = 0; i < convs.size(); ++i) {
            const Conv& c = convs[i];
            const double a = age(c, now);
            const double dk = half_s > 0 ? std::pow(0.5, std::max(0.0, a - open_s) / half_s) : 0.0;
            w[i] = c.running > 0 ? 1.0 : a <= open_s ? w_open : w_closed * dk;
            if (w[i] < 1e-3) w[i] = 0.0;
            fl[i] = c.running > 0 || a <= open_s;
            if (!fl[i] && recall_cap > 0) rk[i] = (int) std::lround(recall_cap * dk);
            if (!fl[i] && closed_floor > 0) cf[i] = (int) std::lround(closed_floor * dk);
        }
        std::vector<size_t> cf_order;   // (closed floors: the most recently active conversation first)
        for (size_t i = 0; i < convs.size(); ++i)
            if (cf[i] > 0) cf_order.push_back(i);
        std::sort(cf_order.begin(), cf_order.end(), [&](size_t a, size_t b) { return convs[a].last > convs[b].last; });
        size_t n_floor = 0, n_fill = 0, bytes = 0;
        std::vector<std::pair<double, int>> sc;
        std::vector<double> score(kE);
        for (int il = 0; il < (int) slots.size() && il < kL; ++il) {
            const int K = slots[(size_t) il] - slack;
            if (K <= 0) continue;
            std::fill(score.begin(), score.end(), 0.0);
            std::vector<char> in(kE, 0), fm(kE, 0);
            int taken = 0;
            const auto layer_sum = [&](const std::vector<float>& v) {
                double s = 0;
                for (int e = 0; e < kE; ++e) s += v[(size_t) il * kE + e];
                return s;
            };
            for (size_t i = 0; i < convs.size(); ++i) {
                const Conv& c = convs[i];
                const double s = layer_sum(c.cnt);
                if (s > 0 && w[i] > 0)
                    for (int e = 0; e < kE; ++e) score[e] += w[i] * c.cnt[(size_t) il * kE + e] / s;
                if (rk[i] > 0) {   // (a closed one's recalled experts, best share first)
                    sc.clear();
                    for (int e = 0; e < kE; ++e)
                        if (c.recall[(size_t) il * kE + e] > 0) sc.push_back({c.recall[(size_t) il * kE + e], e});
                    const size_t k = std::min(sc.size(), (size_t) rk[i]);
                    std::partial_sort(sc.begin(), sc.begin() + (long) k, sc.end(), std::greater<>());
                    for (size_t j = 0; j < k && taken < K; ++j)
                        if (!in[sc[j].second]) {
                            in[sc[j].second] = fm[sc[j].second] = 1;
                            ++taken;
                        }
                }
                if (!fl[i] || s <= 0 || k0 <= 0) continue;
                sc.clear();
                for (int e = 0; e < kE; ++e)
                    if (c.cnt[(size_t) il * kE + e] > 0) sc.push_back({c.cnt[(size_t) il * kE + e], e});
                const size_t k = std::min(sc.size(), (size_t) k0);
                std::partial_sort(sc.begin(), sc.begin() + (long) k, sc.end(), std::greater<>());
                for (size_t j = 0; j < k && taken < K; ++j)
                    if (!in[sc[j].second]) {
                        in[sc[j].second] = fm[sc[j].second] = 1;
                        ++taken;
                    }
            }
            int cft = 0;   // (CLOSED_FLOOR: at most K0 a layer in all)
            for (size_t i : cf_order) {
                const Conv& c = convs[i];
                sc.clear();
                for (int e = 0; e < kE; ++e)
                    if (c.cnt[(size_t) il * kE + e] > 0) sc.push_back({c.cnt[(size_t) il * kE + e], e});
                const size_t k = std::min(sc.size(), (size_t) cf[i]);
                std::partial_sort(sc.begin(), sc.begin() + (long) k, sc.end(), std::greater<>());
                for (size_t j = 0; j < k && taken < K && cft < k0; ++j)
                    if (!in[sc[j].second]) {
                        in[sc[j].second] = fm[sc[j].second] = 1;
                        ++taken;
                        ++cft;
                    }
            }
            const double gs = layer_sum(glob);
            if (gs > 0)
                for (int e = 0; e < kE; ++e) score[e] += eps * glob[(size_t) il * kE + e] / gs;
            sc.clear();
            double mx = 0;
            for (int e = 0; e < kE; ++e) {
                if (score[e] > 0 && !in[e]) sc.push_back({score[e], e});
                mx = std::max(mx, score[e]);
            }
            std::sort(sc.begin(), sc.end(), std::greater<>());
            for (size_t j = 0; j < sc.size() && taken < K; ++j) {
                in[sc[j].second] = 1;
                ++taken;
            }
            for (int e = 0; e < kE; ++e) {
                if (!in[e]) continue;
                const int lv = mx > 0 ? std::min(63, (int) std::lround(63.0 * score[e] / mx)) : 0;
                nf[(size_t) il * kE + e] = (uint8_t) (1u | (fm[e] ? 2u : 0u) | ((unsigned) lv << 2));
                (fm[e] ? n_floor : n_fill) += 1;
                bytes += il < (int) blob.size() ? blob[(size_t) il] : 0;
            }
        }
        // a resume: the experts its own share puts in that were not in the last allocation (recall candidates)
        for (auto& c : convs)
            if (c.running > 0 && c.brought.empty() && !flags.empty()) {
                c.brought.assign((size_t) kL * kE, 0);
                for (size_t i = 0; i < nf.size(); ++i) c.brought[i] = (nf[i] & 1u) && !(flags[i] & 1u) && c.cnt[i] > 0;
            }
        flags.swap(nf);
        maya::core::Glm5Model::conv_publish(flags);
        std::string list;
        for (size_t i = 0; i < convs.size(); ++i)
            if (w[i] > 0 || fl[i] || rk[i] > 0 || cf[i] > 0)
                list += " " + std::to_string(convs[i].id) + ":" + (convs[i].running ? "run" : fl[i] ? "open" : "closed");
        std::fprintf(stderr, "glm conv: alloc (%s) - %d running, %d open; floor %zu + fill %zu experts (%.2f GB) |%s t=%.3f\n",
                     why, running, open, n_floor, n_fill, (double) bytes / 1e9, list.c_str(), unix_s);
    }
    // CONV_UNION's target: the top slots of the weighted union of the conversations' own lists + the global prior
    void publish_union(const char* why, std::chrono::steady_clock::time_point now, double unix_s, int running, int open) {
        const std::vector<int> usable = model->layer_usable_slots();   // (the elastic pool may have given slots away)
        std::vector<uint8_t> nf((size_t) kL * kE, 0);
        std::vector<double> w(convs.size()), prior(convs.size());
        std::string list;
        for (size_t i = 0; i < convs.size(); ++i) {
            const Conv& c = convs[i];
            const double idle = c.decoded.time_since_epoch().count() == 0 ? age(c, now)
                                                                         : std::chrono::duration<double>(now - c.decoded).count();
            const double ia = std::min(idle, age(c, now));   // (a conversation admitted again is active from its admission)
            w[i] = c.running > 0 ? 1.0 : ia <= u_drop && u_idle_half > 0 ? std::pow(0.5, ia / u_idle_half) : 0.0;
            if (w[i] < 1e-3) w[i] = 0.0;
            prior[i] = u_beta / (1.0 + c.tok / std::max(1.0, u_tok0));
            if (w[i] > 0) {
                char b[64];
                std::snprintf(b, sizeof b, " %d:%.2f", c.id, w[i]);
                list += b;
            }
        }
        size_t n_in = 0, bytes = 0;
        std::vector<double> score(kE);
        std::vector<std::pair<double, int>> sc;
        for (int il = 0; il < (int) usable.size() && il < kL; ++il) {
            const int K = usable[(size_t) il] - slack;
            if (K <= 0) continue;
            double gs = 0;
            for (int e = 0; e < kE; ++e) gs += glob[(size_t) il * kE + e];
            for (int e = 0; e < kE; ++e) score[e] = gs > 0 ? 0.01 * glob[(size_t) il * kE + e] / gs : 0.0;
            for (size_t i = 0; i < convs.size(); ++i) {
                if (w[i] <= 0) continue;
                const Conv& c = convs[i];
                double s = 0;
                for (int e = 0; e < kE; ++e) s += c.cnt[(size_t) il * kE + e];
                for (int e = 0; e < kE; ++e)
                    score[e] += w[i] * ((s > 0 ? c.cnt[(size_t) il * kE + e] / s : 0.0) +
                                        (gs > 0 ? prior[i] * glob[(size_t) il * kE + e] / gs : 0.0));
            }
            sc.clear();
            double mx = 0;
            for (int e = 0; e < kE; ++e)
                if (score[e] > 0) {
                    sc.push_back({score[e], e});
                    mx = std::max(mx, score[e]);
                }
            const size_t k = std::min(sc.size(), (size_t) K);
            std::partial_sort(sc.begin(), sc.begin() + (long) k, sc.end(), std::greater<>());
            for (size_t j = 0; j < k; ++j) {
                const int lv = mx > 0 ? std::min(63, (int) std::lround(63.0 * sc[j].first / mx)) : 0;
                nf[(size_t) il * kE + sc[j].second] = (uint8_t) (1u | ((unsigned) lv << 2));
                ++n_in;
                bytes += il < (int) blob.size() ? blob[(size_t) il] : 0;
            }
        }
        flags.swap(nf);
        maya::core::Glm5Model::conv_publish(flags);
        std::fprintf(stderr, "glm conv: alloc (%s) - union, %d running, %d open; %zu experts (%.2f GB) | weights%s t=%.3f\n",
                     why, running, open, n_in, (double) bytes / 1e9, list.c_str(), unix_s);
    }
    // a request's end: its answer joins the conversation; a resume's recalled experts (brought in, then hit) are noted
    void end(int id, const std::vector<int32_t>& emitted, std::chrono::steady_clock::time_point now) {
        Conv* c = find(id);
        if (c == nullptr) return;
        c->tokens.insert(c->tokens.end(), emitted.begin(), emitted.end());
        c->last = now;
        if (recall_cap > 0 && !c->brought.empty()) {
            std::fill(c->recall.begin(), c->recall.end(), 0.0f);
            size_t n = 0;
            for (int il = 0; il < kL; ++il) {
                double s = 0;
                for (int e = 0; e < kE; ++e) s += c->turn[(size_t) il * kE + e];
                if (s <= 0) continue;
                for (int e = 0; e < kE; ++e) {
                    const size_t i = (size_t) il * kE + e;
                    if (c->brought[i] && c->turn[i] > 0) {
                        c->recall[i] = (float) (c->turn[i] / s);
                        ++n;
                    }
                }
            }
            std::fprintf(stderr, "glm conv: conversation %d - %zu recalled experts\n", c->id, n);
        }
        c->brought.clear();
    }
};
}  // namespace

static int glm_pack_generate(const Options& o) {
#if !defined(_WIN32)
    // serving: the server ends the engine with QUIT (or by closing its input), and the engine then sets the conversation
    // aside (SAGUARO_GLM_SLOT_KEEP).  A SIGTERM to the whole process group (systemd stopping a service, a supervisor
    // killing the server's group) must not end it mid-write; a server that is already gone must not end it with
    // SIGPIPE when it writes its last lines.
    if (o.serve) {
        std::signal(SIGTERM, SIG_IGN);
        std::signal(SIGINT, SIG_IGN);
        std::signal(SIGPIPE, SIG_IGN);
    }
#else
    // Windows sends a console's Ctrl+C and Ctrl+Break to every process on it, the engine too: it would end at once and
    // lose the conversation it holds.  The server stops on them and ends the engine with QUIT, as above.
    if (o.serve) {
        std::signal(SIGINT, SIG_IGN);
        std::signal(SIGBREAK, SIG_IGN);
    }
#endif
    maya::core::Glm5Model model;
    std::string err;
    // the prompt chunk, set in the config's args like Strata's: --prefill auto | N (a chunk of N tokens) | 0
    // (the engine reads it as SAGUARO_GLM_PREFILL; without either, auto)
    if (!o.prefill_arg.empty()) set_env("SAGUARO_GLM_PREFILL", o.prefill_arg.c_str());
    // the DFlash2 drafter (--dflash, else SAGUARO_GLM_DFLASH): loaded with the model, on the device of its head
    if (!o.dflash.empty()) set_env("SAGUARO_GLM_DFLASH", o.dflash.c_str());
    // --spec-follow / --spec-follow-save (else SAGUARO_GLM_SPEC_FOLLOW / _SAVE): read below, once the vocabulary is known
    if (!o.spec_follow.empty()) set_env("SAGUARO_GLM_SPEC_FOLLOW", o.spec_follow.c_str());
    if (!o.spec_follow_save.empty()) set_env("SAGUARO_GLM_SPEC_FOLLOW_SAVE", o.spec_follow_save.c_str());
    // the expert profile (Strata's profile.bin: setup passes data/expert-profile-glm.bin) and the routing trace
    // (--dump-routing: tools/make_profile.py's input), before the load reads the one and the first prompt the other
    if (!o.expert_profile.empty()) model.set_expert_profile(o.expert_profile);
    if (!o.dump_routing.empty() && !maya::core::Glm5Model::set_routing_dump(o.dump_routing, err)) {
        std::fprintf(stderr, "saguaro generate: --dump-routing: %s\n", err.c_str());
        return 1;
    }
    // the layer split across the visible GPUs: SAGUARO_GLM_SPLIT, else --layer-split (auto | K1,K2,..)
    if (!model.load_pack_env(o.glm_pack, o.max_context, err, o.layer_split)) {
        std::fprintf(stderr, "saguaro generate: %s\n", err.c_str());
        return 1;
    }
    const int64_t n_vocab = model.geometry().n_vocab;
    // the driver samples on the device (sample_token), so the fast path never downloads the logits
    model.set_host_logits(false);
    // the Monitor's `VRAM dev=...` line, from Glm5Model::vram_report: the start prints it and the split moving
    // later gets it re-printed (PLAN-vram-live, below READY), so it is a lambda.
    const auto print_vram = [](const maya::core::Glm5Model::VramReport& r) {
        // PLAN-context-panel, output only: the K/V budget fields the Monitor's Context panel aggregates - what the
        // part can hold for chat state, the engine's pinned host K/V for streamed K/V, and its parked conversations
        // (this engine keeps none); the re-printed lines below carry them the same way.
        std::printf("VRAM dev=%d pci=%s total=%llu free=%llu weights=%llu kv=%llu kv_ram=%llu act=%llu drafter=%llu "
                    "window=%llu pool=%llu tail=%llu borrow=%llu vision=%llu slots=%lld ctx=%lld layers=%d-%d "
                    "kv_room=%llu kv_ram_room=%llu parked=%llu\n",
                    r.dev, r.pci.empty() ? "-" : r.pci.c_str(), (unsigned long long) r.total,
                    (unsigned long long) r.free, (unsigned long long) r.weights, (unsigned long long) r.kv,
                    (unsigned long long) r.kv_ram, (unsigned long long) r.act, (unsigned long long) r.drafter,
                    (unsigned long long) r.window, (unsigned long long) r.pool, (unsigned long long) r.tail,
                    (unsigned long long) r.borrow, (unsigned long long) r.vision, (long long) r.slots,
                    (long long) r.ctx, r.l0, r.l1 - 1,   // (layers first-last inclusive, as every engine reports them)
                    (unsigned long long) r.kv_room, (unsigned long long) r.kv_ram_room, (unsigned long long) r.parked);
    };
    if (model.fast()) {
        // INFO facts for the server's Monitor tab (strata app): the expert tiers this engine runs with
        const auto st = model.fast_stats();
        // (+ the CPU lane setup's calibration reads: its threads and its PCIe share, as the engine started, and the most
        // threads a request's cpu_threads gets - its pool's size)
        // kv_bytes_token: the latent cache's bytes of one position over every DSA layer, this engine's own dtype;
        // kv_ram_bytes (streaming only): the pinned host K/V it holds now - the Monitor's Context card, never a guess
        std::printf("INFO context=%lld kv=%s kv_resident=%lld kv_bytes_token=%llu%s expert_slots=%lld expert_cache_mib=%lld engine_kind=glm-fast "
                    "experts=%d vram_slots=%lld vram_gb=%.1f ram_slots=%lld ram_gb=%.1f mtp=%d dflash=%d vision_lend=%zu "
                    "cpu_threads=%d cpu_threads_max=%d pcie_share=%.2f logit_bias=1 logprobs=%d\n",
                    (long long) o.max_context, model.kv_int8() ? "int8" : "fp16", (long long) model.kv_resident(),
                    (unsigned long long) model.kv_bytes_token(),
                    (model.kv_resident() > 0 ? " kv_ram_bytes=" + std::to_string(model.kv_host_bytes()) : std::string()).c_str(),
                    (long long) st.pool_slots,
                    (long long) (st.pool_gb * 1024.0), model.geometry().n_expert * (model.geometry().n_layers - model.geometry().dense_lead),
                    (long long) st.pool_slots, st.pool_gb, (long long) st.ram_slots, st.ram_gb, model.has_mtp() ? 1 : 0,
                    model.dflash_ready() ? 1 : 0, model.vision_lend_bytes(), model.cpu_lane_threads(),
                    model.cpu_lane_threads_max(), model.pcie_share(), maya::kernels::kTopLogprobs);
        // the Monitor's map of each GPU: what each part holds on its card, in bytes (Glm5Model::VramReport) - the
        // server finds the card NVML numbers by its PCI address and adds what NVML sees besides.
        for (const auto& r : model.vram_report()) print_vram(r);
        // (Windows on an APU: the commit - the page file - kept experts off the GPU; the server's log, Monitor and
        // /v1/status say so, as for DeepSeek's engine)
        if (const auto cc = model.commit_capped(); cc.first > 0) {
            std::printf("NOTE experts: %.0f GB stay off the GPU - Windows' page file is too small to hold them there; "
                        "a page file of %zu GB keeps them on the GPU (System > About > Advanced system settings > "
                        "Performance > Advanced > Virtual memory), then restart Saguaro\n", (double) cc.first / 1e9, cc.second);
            std::printf("INFO gpu_commit_capped_gb=%.0f pagefile_hint_gb=%zu\n", std::ceil((double) cc.first / 1e9),
                        cc.second);
        }
    }

    maya::kernels::SamplerParams sp;
    sp.greedy = o.greedy;
    sp.top_k = o.top_k;
    sp.top_p = o.top_p;
    sp.temperature = o.temperature;
    sp.seed = o.seed;

    const std::vector<int64_t> glm_eos = {154820, 154827, 154829};
    const std::vector<int64_t>& eos = o.eos_given ? o.eos_ids : glm_eos;
    // SAGUARO_GLM_KL_DUMP=<file>,<n> (a test of the K/V cache formats): --tokens-file's tokens teacher-forced - all but
    // the last n through the prompt path, those n one at a time through the decode - and the logits after each of the
    // n into file (n x n_vocab F32: row i predicts token size - n + i + 1)
    if (const char* kd = std::getenv("SAGUARO_GLM_KL_DUMP")) {
        const std::string spec(kd);
        const size_t c = spec.rfind(',');
        const std::string path = spec.substr(0, c);
        const int64_t n = c == std::string::npos ? 64 : std::atoll(spec.c_str() + c + 1);
        const auto& t = o.tokens;
        if (n <= 0 || (int64_t) t.size() <= n) {
            std::fprintf(stderr, "saguaro generate: SAGUARO_GLM_KL_DUMP needs more than %lld tokens\n", (long long) n);
            return 2;
        }
        model.set_host_logits(true);
        std::string err;
        std::vector<float> lg;
        const std::vector<int32_t> head(t.begin(), t.end() - n);
        const bool pok = model.fast() && model.prefill(head, err, (int32_t) t[t.size() - (size_t) n]);
        if (!pok && (!err.empty() || !model.forward(head, lg, err))) {
            std::fprintf(stderr, "saguaro generate: %s\n", err.c_str());
            return 1;
        }
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) {
            std::fprintf(stderr, "saguaro generate: cannot write %s\n", path.c_str());
            return 1;
        }
        for (size_t i = t.size() - (size_t) n; i < t.size(); ++i) {
            if (!model.forward({(int32_t) t[i]}, lg, err) || (int64_t) lg.size() < n_vocab) {
                std::fprintf(stderr, "saguaro generate: %s\n", err.empty() ? "no logits" : err.c_str());
                std::fclose(f);
                return 1;
            }
            std::fwrite(lg.data(), sizeof(float), (size_t) n_vocab, f);
        }
        std::fclose(f);
        std::fprintf(stderr, "saguaro generate: %lld positions' logits in %s\n", (long long) n, path.c_str());
        return 0;
    }

    const bool stop_on_eos = o.eos_given || o.stop_eos || o.serve;

    // ---- stdin on its own thread (the qwen serve's pattern): a STOP line must reach a request that is
    // still running.  read(2), not std::cin - glibc's exit() flushes stdio and waits on stdin's lock,
    // which getline holds while it waits for input.
    std::atomic<bool> stop_req{false}, quit_req{false};
    // WRAP (the server is stopping): a request still thinking closes its reasoning at the next token and answers
    std::atomic<bool> wrap_req{false};
    // PAUSE (the server's fair slices): STOP, and the conversation's state is kept with what the request wrote (the
    // snapshot moves to the last token fed), so the request that continues it reads only its last token
    std::atomic<bool> pause_req{false}, pause_keep{false};
    // several conversations at once (SAGUARO_GLM_SEQS): STOP <rid> / PAUSE <rid> [keep] for one request; the request
    // whose prompt is being read stops at the next chunk
    std::mutex ctl_mu;
    std::map<int, int> ctl;   // rid -> 1 STOP, 2 PAUSE, 3 PAUSE keep
    std::atomic<int> admitting_rid{-1};
    // SAGUARO_GLM_PREFILL_INTERLEAVE (maya/core/prefill_pieces.hpp): the other requests decode between the pieces of
    // a prompt being read, so a STOP / PAUSE for the request being admitted stops its read alone (admit_stop) - not
    // stop_req, which every decoding request reads as its own.  Off: stop_req as before (nothing decodes meanwhile)
    static const maya::core::PrefillInterleave interleave =
        maya::core::prefill_interleave_parse(std::getenv("SAGUARO_GLM_PREFILL_INTERLEAVE"));
    // SAGUARO_GLM_PREFILL_GAPS (maya/core/prefill_gaps.hpp): the other requests decode at the chunk boundaries of a
    // prompt being read (one call, its chunks pipelined as without) - like interleave, a STOP for the request being
    // read stops its read alone.  Over interleave when both are set.
    static const maya::core::PrefillGaps read_gaps =
        maya::core::prefill_gaps_parse(std::getenv("SAGUARO_GLM_PREFILL_GAPS"));
    std::atomic<bool> admit_stop{false};
    const auto read_stopped = [&] { return stop_req.load() || admit_stop.load(); };
    std::mutex in_mu;
    std::condition_variable in_cv;
    std::deque<std::string> in_lines;
    bool in_eof = false;
    if (o.serve) {
        std::thread([&] {
            std::string l, buf;
            char chunk[4096];
            auto getline_fd = [&](std::string& out) -> bool {
                for (;;) {
                    const size_t nlpos = buf.find('\n');
                    if (nlpos != std::string::npos) {
                        out.assign(buf, 0, nlpos);
                        buf.erase(0, nlpos + 1);
                        return true;
                    }
#if defined(_WIN32)
                    const int n = _read(0, chunk, (unsigned) sizeof chunk);
#else
                    const ssize_t n = ::read(0, chunk, sizeof chunk);
                    if (n < 0 && errno == EINTR) continue;
#endif
                    if (n <= 0) {
                        if (buf.empty()) return false;
                        out.swap(buf);
                        buf.clear();
                        return true;
                    }
                    buf.append(chunk, (size_t) n);
                }
            };
            while (getline_fd(l)) {
                maya::platform::idle_poke();   // (the service threads asleep after a long quiet wake for this line)
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop_req.store(true); continue; }
                if (l.rfind("STOP ", 0) == 0 || (l.rfind("PAUSE ", 0) == 0 && l != "PAUSE keep")) {
                    std::istringstream ls(l);
                    std::string c0, kw;
                    int rid = -1;
                    ls >> c0 >> rid;
                    // (no request id there: nothing to stop - a negative one would match admitting_rid's idle -1 and
                    // set the global stop, which every running request reads as its own)
                    if (ls.fail() || rid < 0) continue;
                    ls >> kw;
                    {
                        std::lock_guard<std::mutex> lk(ctl_mu);
                        ctl[rid] = c0 == "STOP" ? 1 : kw == "keep" ? 3 : 2;
                    }
                    if (rid == admitting_rid.load()) admit_stop.store(true);
                    continue;
                }
                if (l == "PAUSE" || l == "PAUSE keep") {   // keep: the prompt's state is set aside first
                    pause_keep.store(l == "PAUSE keep");
                    pause_req.store(true);
                    stop_req.store(true);
                    continue;
                }
                if (l == "WRAP") { wrap_req.store(true); continue; }
                if (l == "QUIT") {                       // cancel an in-flight request AND queue the line:
                    quit_req.store(true);
                    stop_req.store(true);                // (the decode checks stop_req; reset by the next GEN)
                }
                std::lock_guard<std::mutex> lk(in_mu);   // a GEN written before QUIT must still run (pipes)
                in_lines.push_back(l);
                in_cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(in_mu);
            in_eof = true;
            in_cv.notify_one();
        }).detach();
    }
    const auto next_line = [&](std::string& out) -> bool {
        model.lane_drift_check();
        model.lane_retime();
        std::unique_lock<std::mutex> lk(in_mu);
        in_cv.wait(lk, [&] { return !in_lines.empty() || in_eof || quit_req.load(); });
        if (in_lines.empty()) return false;   // EOF or QUIT with nothing queued behind it
        out = std::move(in_lines.front());
        in_lines.pop_front();
        return true;
    };

    std::vector<int32_t> snap_tokens;   // the prompt prefix the model's saved state belongs to (conversation reuse)
    // images (GENI): this request's <|image|> positions and rows, and a hash of the rows the saved state read - the
    // same tokens with another picture must not reuse it
    std::vector<int64_t> cur_img_pos;
    std::vector<float> cur_img_rows;
    // the thinking budget (GEN keys think_budget=<n> think_end=<id>, the server's run config): a reasoning block still
    // open after n tokens is closed - the next token is </think> instead of the sampled one - and the answer follows
    // in the same decode
    int64_t req_think_budget = 0;
    int64_t req_gt = 0;   // GEN gt=<n>: the prompt's last n tokens open the reply (the template's generation prompt)
    // PLAN-turbo-live: this request's live Turbo (GEN keys, serve/turbo_live.py): turbo_drop=N runs the route's k-N
    // heaviest experts (renormalised), turbo_skip=ssd|ram leaves out what the SSD / the RAM tier would serve; and
    // its strata_tune miss bar (miss_skip_w / miss_skip_from).  Absent: off - the request is exact, its routes are
    // today's byte for byte.  Applied to the sequence the request runs in (set_turbo / set_miss_skip), so a Turbo
    // chat changes only its own tokens and the others in the batch stay exact.
    double req_msw = -1.0;
    int req_msf = -1, req_tdrop = 0, req_tskip = 0;   // turbo_skip: 1 = ssd, 2 = ram
    // where a prompt's state is kept for the next request: before the reply's opening (gt), so the request after an
    // answer that was cancelled or dropped from the history - the next user message stands there - still continues it
    const auto snap_pos = [](const std::vector<int32_t>& prompt, int64_t gt) -> size_t {
        return prompt.size() - (size_t) std::max<int64_t>(1, std::min<int64_t>(gt, (int64_t) prompt.size() - 1));
    };
    int32_t req_think_end = -1;
    // the request's logit_bias (GEN key logit_bias=<id>:<bias>,...): added to its logits before every sample
    std::vector<std::pair<int32_t, float>> req_bias;
    uint64_t snap_img_hash = 1469598103934665603ull;
    const auto img_hash = [&](size_t len) {
        uint64_t h = 1469598103934665603ull;   // FNV-1a over the positions and rows inside [0, len)
        const size_t E = (size_t) model.geometry().n_embd;
        for (size_t k = 0; k < cur_img_pos.size() && (size_t) cur_img_pos[k] < len; ++k) {
            const unsigned char* b = (const unsigned char*) (cur_img_rows.data() + k * E);
            for (size_t j = 0; j < E * sizeof(float); ++j) h = (h ^ b[j]) * 1099511628211ull;
            h = (h ^ (uint64_t) cur_img_pos[k]) * 1099511628211ull;
        }
        return h;
    };
    // CONVERSATION SLOTS (the fast path): a request that does not continue the conversation in the model first sets
    // that conversation aside in a file (Glm5Model::slot_save: its snapshot and DSA caches), then takes back the slot
    // it does continue (slot_load) - chats, and an agent's side requests, that interleave no longer re-read each
    // other's prompts.  SAGUARO_GLM_SLOTS=<n> files (default 4, 0 off), for conversations of SAGUARO_GLM_SLOT_MIN tokens
    // or more (1024), in SAGUARO_GLM_SLOT_DIR (default <pack>/slots, emptied at start), at most SAGUARO_GLM_SLOT_GB on
    // disk (16) and never below 8 GB free.  A conversation with pictures is set aside too: its slot keeps img_hash() of
    // them and is taken back only by a prompt with the same pictures at the same places.  Or the slots
    // live in RAM and nothing reaches the disk: SAGUARO_GLM_SLOT_RAM_GB (0, off) caps their bytes there - the states
    // are host buffers that die with the run, so SLOT_DIR, SLOT_GB (1) and the disk-space guard do not apply.
    // (static: the multi path's Ctx, a local struct, uses it as a default member value - MSVC accepts that of a static
    // only, and a local constant left Ctx without a default constructor: "std::construct_at: no matching function")
    static constexpr uint64_t kNoImages = 1469598103934665603ull;   // img_hash() of a prefix without pictures
    struct Slot {
        std::vector<int32_t> tokens;
        std::string path;                       // empty when the slot is a RAM buffer (slot_ram)
        uint64_t bytes = 0, used = 0;
        std::shared_ptr<std::vector<uint8_t>> data;   // the RAM slot's bytes (null: a file slot)
        // a paused request's prompt (PAUSE keep): the next turn of that chat continues it, not the written answer
        // (its history drops the reasoning), so its own continuations do not supersede it until it is taken back
        bool anchor = false;
        // the end of a system prompt (slot_prefix): other chats start with it, so its conversation's later states
        // never supersede it - it goes only as the least recently used (#179)
        bool prefix = false;
        uint64_t img = kNoImages;               // img_hash() of the pictures in `tokens`
    };
    std::vector<Slot> slots;
    uint64_t slot_clock = 0, slot_seq = 0;
    bool snap_in_slot = false;   // the model's snapshot is one of the slots, unchanged since
    const auto env_num = [](const char* k, double d) { const char* v = getenv(k); return v ? std::atof(v) : d; };
    int slot_max = model.fast() && getenv("SAGUARO_GLM_NO_REUSE") == nullptr ? (int) env_num("SAGUARO_GLM_SLOTS", 4) : 0;
    const size_t slot_min = (size_t) std::max(1.0, env_num("SAGUARO_GLM_SLOT_MIN", 1024));
    const uint64_t slot_budget = (uint64_t) (env_num("SAGUARO_GLM_SLOT_GB", 16) * 1073741824.0);
    // slots in RAM: the conversation's state is parked in a host buffer instead of a file.  The cap moves to RAM too
    // (slot_cap below); the states die with the run, so SLOT_KEEP, SLOT_DAYS, SLOT_DIR and the 8 GB-free disk guard
    // are theirs no longer.  Nothing on the disk is touched: in RAM mode the slots folder is neither used nor wiped.
    const bool slot_ram = env_num("SAGUARO_GLM_SLOT_RAM_GB", 0) > 0;
    const uint64_t slot_cap =
        slot_ram ? (uint64_t) (env_num("SAGUARO_GLM_SLOT_RAM_GB", 16) * 1073741824.0) : slot_budget;
    // SAGUARO_GLM_SLOT_KEEP=1: the slots outlive the engine - the folder is not emptied at start (its files are indexed
    // and kept when they belong to this model), the conversation in the model is set aside when the engine ends
    // (QUIT or end of input), and a file not used for SAGUARO_GLM_SLOT_DAYS days (7) is removed.  Every slot then has a
    // .meta beside it: this model's identity, the conversation's tokens and the slot's size; a slot whose .meta is
    // missing, foreign or does not match its file is deleted.  Both files are written under a temporary name first.
    const bool slot_keep = getenv("SAGUARO_GLM_SLOT_KEEP") != nullptr && std::atoi(getenv("SAGUARO_GLM_SLOT_KEEP")) != 0;
    const double slot_days = env_num("SAGUARO_GLM_SLOT_DAYS", 7);
    namespace fs = std::filesystem;
    std::filesystem::path slot_dir;
    // what a kept slot must match: the session format, the engine version, the pack and the model's shards (names and
    // sizes) and whether the draft block is loaded.  The split itself is checked by slot_load (each part's layers).
    const std::string slot_identity = [&] {
        std::ostringstream id;
        // the state's layout, not the build (every version bump dropped every kept conversation): kSlotFormat moves
        // when a slot's bytes change meaning; the geometry and the latent's storage are in it too.
        // SAGUARO_GLM_SLOT_ID_ENGINE=1: the build's version in it as before.
        constexpr int kSlotFormat = 1;
        if (env_num("SAGUARO_GLM_SLOT_ID_ENGINE", 0) != 0) id << "maya-session-2 engine=" << MAYA_VERSION;
        else {
            const auto& gg = model.geometry();
            id << "maya-session-3 fmt=" << kSlotFormat << " kv8=" << (getenv("SAGUARO_GLM_KV_INT8") != nullptr ? 1 : 0)
               << " geo=" << gg.n_embd << "," << gg.n_layers << "," << gg.kv_lora << "," << gg.idx_key << ","
               << gg.idx_kpool << "," << gg.kda_head_dim << "," << gg.d_conv << "," << gg.hc;
        }
        id << " mtp=" << (model.has_mtp() ? 1 : 0);
        std::error_code ec;
        for (const char* f : {"index.txt", "dense.bin", "native_experts.txt"})
            id << " " << f << ":" << (unsigned long long) fs::file_size(fs::path(o.glm_pack) / f, ec);
        std::vector<std::string> shards;
        for (const auto& de : fs::directory_iterator(fs::path(o.glm_pack).parent_path(), ec))
            if (de.path().extension() == ".gguf")
                shards.push_back(de.path().filename().string() + ":" + std::to_string(de.file_size(ec)));
        std::sort(shards.begin(), shards.end());
        for (const auto& s : shards) id << " " << s;
        return id.str();
    }();
    constexpr uint64_t kMetaMagic = 0x3153534559414d47ull;   // "GMAYESS1"
    const auto meta_of = [](const std::string& bin) { return bin.substr(0, bin.size() - 4) + ".meta"; };
    const auto meta_write = [&](const std::string& path, const std::vector<int32_t>& toks, uint64_t bytes,
                                 uint64_t img) -> bool {
        const std::string tmp = path + ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            const uint64_t il = slot_identity.size(), nt = toks.size();
            f.write((const char*) &kMetaMagic, 8);
            f.write((const char*) &il, 8);
            f.write(slot_identity.data(), (std::streamsize) il);
            f.write((const char*) &bytes, 8);
            f.write((const char*) &nt, 8);
            f.write((const char*) toks.data(), (std::streamsize) (nt * sizeof(int32_t)));
            f.write((const char*) &img, 8);
            f.write((const char*) &kMetaMagic, 8);
            f.flush();
            if (!f) return false;
        }
        std::error_code ec;
        fs::rename(tmp, path, ec);
        return !ec;
    };
    // SAGUARO_GLM_SLOT_FAST: slot files are written in the background (their .meta too, by meta_write above): every
    // one queued is written before this scope's locals go
    struct SlotFlush {
        ~SlotFlush() { maya::core::Glm5Model::slot_flush(); }
    } slot_flush_guard;
    // a .meta of this model -> its tokens, the slot's size and its pictures' hash (false: foreign, damaged or not
    // this model's)
    const auto meta_read = [&](const std::string& path, std::vector<int32_t>& toks, uint64_t& bytes,
                               uint64_t& img) -> bool {
        std::ifstream f(path, std::ios::binary);
        uint64_t m0 = 0, il = 0, nt = 0, m1 = 0;
        if (!f.read((char*) &m0, 8) || m0 != kMetaMagic || !f.read((char*) &il, 8) || il != slot_identity.size())
            return false;
        std::string id(il, '\0');
        if (!f.read(id.data(), (std::streamsize) il) || id != slot_identity) return false;
        if (!f.read((char*) &bytes, 8) || !f.read((char*) &nt, 8) || nt == 0 || nt > (uint64_t) o.max_context) return false;
        toks.resize(nt);
        return (bool) f.read((char*) toks.data(), (std::streamsize) (nt * sizeof(int32_t))) && f.read((char*) &img, 8) &&
               f.read((char*) &m1, 8) && m1 == kMetaMagic;
    };
    const auto slot_age_days = [](const std::string& p) {
        std::error_code ec;
        const auto t = fs::last_write_time(p, ec);
        if (ec) return 1e9;
        return std::chrono::duration<double>(fs::file_time_type::clock::now() - t).count() / 86400.0;
    };
    // NOTE lines on stdout: the server shows them in its own log (what happens to the conversations)
    const auto note = [&](const std::string& m) {
        std::fprintf(stderr, "glm slots: %s\n", m.c_str());
        if (o.serve) std::printf("NOTE sessions: %s\n", m.c_str());
    };
    const auto gb = [](uint64_t b) {
        char s[32];
        std::snprintf(s, sizeof s, "%.2f GB", (double) b / 1e9);
        return std::string(s);
    };
    if (slot_max > 0 && slot_ram)
        note("slots parked in RAM (" + gb(slot_cap) + " max) - nothing is written to disk, the states die with the run");
    if (slot_max > 0 && !slot_ram) {
        slot_dir = getenv("SAGUARO_GLM_SLOT_DIR") ? std::filesystem::path(getenv("SAGUARO_GLM_SLOT_DIR"))
                                                 : std::filesystem::path(o.glm_pack) / "slots";
        std::error_code ec;
        std::filesystem::create_directories(slot_dir, ec);
        struct Found { std::string bin; std::vector<int32_t> toks; uint64_t bytes; fs::file_time_type t; uint64_t img; };
        std::vector<Found> found;
        size_t refused = 0;
        for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {   // a previous run's slots
            const std::string fn = de.path().filename().string(), p = de.path().string();
            if (fn.rfind("slot-", 0) != 0) continue;
            const bool is_bin = fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0;
            const bool is_meta = fn.size() > 10 && fn.compare(fn.size() - 5, 5, ".meta") == 0;
            if (!slot_keep || (!is_bin && !is_meta)) {   // not kept, or a .tmp an interrupted write left
                std::filesystem::remove(de.path(), ec);
                continue;
            }
            if (!is_meta) continue;                      // a .bin is judged by its .meta
            const std::string bin = p.substr(0, p.size() - 5) + ".bin";
            Found fd{bin, {}, 0, fs::last_write_time(de.path(), ec), kNoImages};
            const uint64_t have = fs::exists(bin, ec) ? (uint64_t) fs::file_size(bin, ec) : 0;
            if (meta_read(p, fd.toks, fd.bytes, fd.img) && have == fd.bytes && fd.bytes > 0 && slot_age_days(p) <= slot_days) {
                found.push_back(std::move(fd));
            } else {
                ++refused;
                std::filesystem::remove(p, ec);
                std::filesystem::remove(bin, ec);
            }
        }
        if (slot_keep)                                   // a .bin without its .meta was never completed
            for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {
                const std::string p = de.path().string(), fn = de.path().filename().string();
                if (fn.rfind("slot-", 0) == 0 && fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0 &&
                    !fs::exists(meta_of(p), ec))
                    std::filesystem::remove(de.path(), ec);
            }
        std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.t < b.t; });
        uint64_t total = 0;
        for (auto& fd : found) {
            total += fd.bytes;
            slots.push_back({std::move(fd.toks), fd.bin, fd.bytes, ++slot_clock});
            // a system prompt's state (slot_prefix) goes only as the least recently used, after a restart too: the
            // .meta has no flag for it, its tokens tell (no user turn)
            slots.back().prefix = maya::core::is_system_prefix(slots.back().tokens, 154827);
            slots.back().img = fd.img;
        }
        if (slot_keep) {
            std::ostringstream m;
            m << "folder " << slot_dir.string() << ": " << slots.size() << " conversation(s) kept from earlier runs ("
              << gb(total) << ")";
            if (refused) m << ", " << refused << " not this model's or incomplete - removed";
            note(m.str());
        }
        if (!std::filesystem::is_directory(slot_dir, ec)) {
            std::fprintf(stderr, "glm slots: %s is not writable - conversations are not set aside\n",
                         slot_dir.string().c_str());
            slot_max = 0;
        }
    }
    const auto slot_drop = [&](size_t i) {
        if (slots[i].data)                       // a RAM slot: its buffer simply dies
            slots[i].data.reset();
        else {
            std::error_code ec;
            maya::core::Glm5Model::slot_cancel(slots[i].path);   // (a background write still under way: dropped)
            std::filesystem::remove(slots[i].path, ec);
            std::filesystem::remove(meta_of(slots[i].path), ec);
        }
        slots.erase(slots.begin() + (long) i);
    };
    // SAGUARO_GLM_SLOT_FAST: slots whose background write or commit failed leave the list (their files are gone)
    const auto slot_reap = [&] {
        if (slots.empty()) return;
        for (const std::string& f : maya::core::Glm5Model::slot_failed_writes())
            for (size_t i = slots.size(); i-- > 0;)
                if (!slots[i].data && slots[i].path == f) {
                    note("a conversation's background write failed - it is no longer kept");
                    slot_drop(i);
                }
    };
    const auto slot_lru = [&](size_t keep) {   // the least recently used slot other than `keep`
        size_t lru = SIZE_MAX;
        for (size_t i = 0; i < slots.size(); ++i)
            if (i != keep && (lru == SIZE_MAX || slots[i].used < slots[lru].used)) lru = i;
        return lru;
    };
    // the conversation in the model (its snapshot, snap_tokens) into a slot
    const auto slot_put = [&](const char* why) {
        slot_reap();
        if (slot_max <= 0 || snap_in_slot || snap_tokens.size() < slot_min) return;
        // an earlier state of this conversation is superseded (the same pictures: the same tokens with other
        // pictures are another conversation)
        for (size_t i = slots.size(); i-- > 0;)
            if (!slots[i].anchor && !slots[i].prefix && slots[i].img == snap_img_hash &&
                slots[i].tokens.size() <= snap_tokens.size() &&
                std::equal(slots[i].tokens.begin(), slots[i].tokens.end(), snap_tokens.begin()))
                slot_drop(i);
        while (!slots.empty() && slots.size() >= (size_t) slot_max) slot_drop(slot_lru(SIZE_MAX));
        if (slot_ram) {   // the conversation is parked in a host buffer; nothing touches the disk
            const auto t0 = std::chrono::steady_clock::now();
            std::string e;
            auto buf = std::make_shared<std::vector<uint8_t>>();
            buf->reserve(1 << 20);
            const uint64_t bytes = model.slot_save(*buf, e);
            if (bytes == 0) {
                note("setting the conversation aside in RAM failed (" + (e.empty() ? std::string("it did not copy") : e) +
                     ") - it is not parked and will be read from token 0 when it comes back");
                return;
            }
            slots.push_back({snap_tokens, "", bytes, ++slot_clock});
            slots.back().data = buf;
            slots.back().img = snap_img_hash;
            snap_in_slot = true;
            uint64_t total = 0;
            for (const auto& s : slots) total += s.bytes;
            size_t evicted = 0;
            while (slots.size() > 1 && total > slot_cap) {
                const size_t i = slot_lru(slots.size() - 1);
                total -= slots[i].bytes;
                slot_drop(i);
                ++evicted;
            }
            char m[288];
            std::snprintf(m, sizeof m, "saved a %zu-token conversation%s (%s in RAM in %.1f s); %zu kept, %s%s",
                          snap_tokens.size(), why, gb(bytes).c_str(),
                          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), slots.size(),
                          gb(total).c_str(), evicted ? ", least recently used dropped to stay under the RAM cap" : "");
            note(m);
            return;
        }
        std::error_code ec;
        const auto space = std::filesystem::space(slot_dir, ec);
        if (!ec && space.available < (uint64_t) 8 << 30) {
            std::fprintf(stderr, "glm slots: under 8 GB free in %s - the conversation is not set aside\n",
                         slot_dir.string().c_str());
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string path = (slot_dir / ("slot-" + std::to_string(stamp) + "-" + std::to_string(++slot_seq) +
                                              ".bin")).string();
        std::string e;
        // SAGUARO_GLM_SLOT_FAST: the file and its .meta written in the background, renamed in place when whole
        const bool bg = maya::core::Glm5Model::slot_fast();
        uint64_t bytes = 0;
        if (bg) {
            const std::string mp = meta_of(path);
            const std::vector<int32_t> toks = snap_tokens;
            const uint64_t img = snap_img_hash;
            bytes = model.slot_save_bg(path + ".tmp", path, [&meta_write, mp, toks, img](uint64_t b) {
                return meta_write(mp, toks, b, img);
            }, e);
        } else {
            bytes = model.slot_save(path + ".tmp", e);
        }
        if (bytes == 0 || (!bg && ((std::filesystem::rename(path + ".tmp", path, ec), ec) ||
                                   !meta_write(meta_of(path), snap_tokens, bytes, snap_img_hash)))) {
            std::filesystem::remove(path + ".tmp", ec);
            std::filesystem::remove(path, ec);
            std::fprintf(stderr, "glm slots: setting the conversation aside failed (%s)\n",
                         e.empty() ? "writing its files" : e.c_str());
            return;
        }
        slots.push_back({snap_tokens, path, bytes, ++slot_clock});
        slots.back().img = snap_img_hash;
        snap_in_slot = true;
        uint64_t total = 0;
        for (const auto& s : slots) total += s.bytes;
        size_t evicted = 0;
        while (slots.size() > 1 && total > slot_cap) {
            const size_t i = slot_lru(slots.size() - 1);
            total -= slots[i].bytes;
            slot_drop(i);
            ++evicted;
        }
        char m[256];
        std::snprintf(m, sizeof m, "saved a %zu-token conversation%s (%s in %.1f s); %zu kept, %s%s",
                      snap_tokens.size(), why, gb(bytes).c_str(),
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), slots.size(),
                      gb(total).c_str(), evicted ? ", least recently used removed to stay under the size cap" : "");
        note(m);
    };
    // the longest slot this prompt continues, back in the model: the tokens it covers (0: none)
    bool slot_load_failed = false;   // (a slot_load that failed may have written part of the bound context's rows)
    const auto slot_take = [&](const std::vector<int32_t>& prompt) -> size_t {
        if (slot_max <= 0) return 0;
        slot_reap();
      for (;;) {   // the longest slot this prompt continues; one that will not load (saved by another model layout, a
                   // damaged file) is dropped and the next-longest tried - before, the first failure read the whole
                   // prompt from token 0 although a shorter slot or a checkpoint held most of it
        size_t best = SIZE_MAX, len = 0;
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto& t = slots[i].tokens;
            if (t.size() > len && t.size() < prompt.size() && std::equal(t.begin(), t.end(), prompt.begin()) &&
                img_hash(t.size()) == slots[i].img) {
                best = i;
                len = t.size();
            }
        }
        if (best == SIZE_MAX) return 0;
        const auto t0 = std::chrono::steady_clock::now();
        std::string e;
        const bool back = slots[best].data
                              ? model.slot_load(*slots[best].data, (int64_t) len, e)
                              : model.slot_load(slots[best].path, (int64_t) len, e);
        if (!back) {
            note("taking a conversation back failed (" + e + ") - trying the next slot or checkpoint");
            slot_load_failed = true;
            slot_drop(best);
            snap_tokens.clear();
            continue;
        }
        snap_tokens = slots[best].tokens;
        snap_img_hash = slots[best].img;
        snap_in_slot = true;
        slots[best].used = ++slot_clock;
        slots[best].anchor = false;
        if (slot_keep && !slots[best].data) {   // its age counts from its last use (a file slot's only)
            std::error_code ec;
            maya::core::Glm5Model::slot_wait(slots[best].path);   // (its .meta committed: slot_load waited too)
            fs::last_write_time(meta_of(slots[best].path), fs::file_time_type::clock::now(), ec);
        }
        char m[160];
        std::snprintf(m, sizeof m, "took back a %zu-token conversation from %s (%.1f s); %zu new tokens to read", len,
                      slots[best].data ? "RAM" : "disk",
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                      prompt.size() - len);
        note(m);
        return len;
      }
    };
    // a request's DFlash window and draft cut (its GEN line's spec= / spec_min_p=; 0 / -1: as the engine started)
    int req_spec = 0;
    double req_spec_min_p = -1.0;
    int req_spec_n_min = -1;       // ... its drafts kept whatever their confidence (spec_n_min=; -1: the engine's)
    double req_spec_cum = -1.0;    // ... and the cut on the product of the confidences (spec_cum=; -1: the engine's)
    // THE PROMPT of a request (run_request, and the several-conversation scheduler's admission): the conversation
    // reuse (the snapshot, else a slot), the prompt path for all but the last token, the token path for the rest - the
    // state saved before the last token and the first sample's logits on the tail.  0, or 1 after an ERR line.
    // `tag` prefixes the PP / ERR lines ("<rid> " with several conversations at once).
    std::string tag;
    // CONVERSATION CHECKPOINTS (Glm5Model::ckpt_save): besides the snapshot (the last prompt but its last token), up to
    // SAGUARO_GLM_CKPTS (4; 0 = none) states of the conversation in the model at earlier positions, in host RAM (~150 MB
    // each with GLM-5.3-Flash): at the end of the system prompt (the first user turn's marker), at the start of the
    // prompt's last turn (its last user or tool-result marker) and every SAGUARO_GLM_CKPT_EVERY (16384) fresh prompt
    // tokens - only where at least 512 fresh tokens lie before.  A prompt that leaves the conversation before the
    // snapshot's position - an edited or replaced last message, another chat with the same system prompt and tools, a
    // client that rewrote an older turn - restores the longest checkpoint it starts with and reads only the rest,
    // where it read everything from token 0.  Text prompts only; SAGUARO_GLM_NO_REUSE turns it off with the rest of the
    // reuse.  SEVERAL CONVERSATIONS AT ONCE (SAGUARO_GLM_SEQS > 1) keep them only with SAGUARO_GLM_MULTI_CKPTS=1 (#711):
    // each context then has checkpoints of its own conversation.  The model's checkpoint slots are host buffers that
    // take whichever sequence is bound when one is written, and a checkpoint is good only for the sequence whose DSA
    // rows it stood on, so context c owns the slots [c * ckpt_max, (c + 1) * ckpt_max): its list goes in and out with
    // its snapshot (bind_ctx) and ckpt_base is its first slot.  Every context's checkpoints together are at most
    // kMultiCkptSlots (~150 MB of host RAM each), and the periodic ones stay off there unless SAGUARO_GLM_CKPT_EVERY is
    // set: each one ends a prompt-path call, a pipeline fill on a split, while other conversations wait to decode.
    struct Ckpt {
        std::vector<int32_t> tokens;   // the conversation's first tokens the checkpoint holds the state after (empty: free)
        uint64_t used = 0;
    };
    std::vector<Ckpt> ckpts;   // the bound context's (several conversations: each context's in turn, see bind_ctx)
    uint64_t ckpt_clock = 0;
    int ckpt_base = 0;         // the bound context's first checkpoint slot in the model (0 with one conversation)
    const bool multi_ckpts = model.seq_count() > 1 && env_num("SAGUARO_GLM_MULTI_CKPTS", 0) != 0;
    const int ckpt_max = [&] {
        if (!model.fast() || getenv("SAGUARO_GLM_NO_REUSE") != nullptr) return 0;
        const int want = std::max(0, std::min(16, (int) env_num("SAGUARO_GLM_CKPTS", 4)));
        if (model.seq_count() <= 1) return want;
        if (!multi_ckpts) return 0;
        constexpr int kMultiCkptSlots = 32;   // every context's together (~4.8 GB with GLM-5.3-Flash at most)
        return std::min(want, kMultiCkptSlots / model.seq_count());
    }();
    const int64_t ckpt_every =
        model.seq_count() > 1 && getenv("SAGUARO_GLM_CKPT_EVERY") == nullptr
            ? 0
            : std::max<int64_t>(0, (int64_t) env_num("SAGUARO_GLM_CKPT_EVERY", 16384));
    if (multi_ckpts && ckpt_max > 0) {
        const std::string every = ckpt_every > 0 ? " and every " + std::to_string(ckpt_every) + " fresh tokens" : "";
        std::fprintf(stderr, "glm checkpoints: up to %d for each of the %d conversations at once "
                             "(SAGUARO_GLM_MULTI_CKPTS), at the system prompt's end and the last turn's start%s\n",
                     ckpt_max, model.seq_count(), every.c_str());
    } else if (multi_ckpts)
        std::fprintf(stderr, "glm checkpoints: SAGUARO_GLM_MULTI_CKPTS is set but none is kept "
                             "(SAGUARO_GLM_CKPTS=0 or SAGUARO_GLM_NO_REUSE)\n");
    // a prompt writes the state from position `from` on: the checkpoints past it no longer hold this conversation
    const auto ckpt_drop_past = [&](size_t from) {
        for (auto& c : ckpts)
            if (c.tokens.size() > from) c.tokens.clear();
    };
    // the positions in [from + 512, end) a prompt read from `from` keeps checkpoints at (see above), ascending
    const auto ckpt_marks = [&](const std::vector<int32_t>& prompt, size_t from, size_t end) {
        constexpr int32_t kUserTok = 154827, kObsTok = 154829;   // <|user|>, <|observation|>
        std::vector<size_t> at;
        if (ckpt_max <= 0 || !cur_img_pos.empty() || end <= from + 512) return at;
        size_t first = 0, last = 0;
        for (size_t i = 1; i < end; ++i)
            if (prompt[i] == kUserTok || prompt[i] == kObsTok) {
                if (first == 0 && prompt[i] == kUserTok) first = i;
                last = i;
            }
        std::vector<size_t> turns;
        for (const size_t p : {first, last})
            if (p >= from + 512 && p < end) turns.push_back(p);
        at = turns;
        // the periodic ones, but none within 512 tokens of a turn boundary (that one holds the same state, more useful)
        if (ckpt_every > 0)
            for (size_t p = from + (size_t) ckpt_every; p < end; p += (size_t) ckpt_every) {
                bool by_turn = false;   // (not `near`: Windows' headers define it away)
                for (const size_t t : turns) by_turn = by_turn || (p + 512 > t && t + 512 > p);
                if (!by_turn) at.push_back(p);
            }
        std::sort(at.begin(), at.end());
        at.erase(std::unique(at.begin(), at.end()), at.end());
        // at most ckpt_max a prompt, the latest kept (the turn's start, then the ones before it)
        if ((int) at.size() > ckpt_max) at.erase(at.begin(), at.end() - ckpt_max);
        // every prompt-path call between them must reach its minimum (SAGUARO_GLM_PREFILL_MIN, 32: fewer go through the
        // token path, one token at a time): only the piece after the last one may be shorter
        std::vector<size_t> kept;
        for (const size_t p : at)
            if (p >= (kept.empty() ? from : kept.back()) + 64) kept.push_back(p);
        return kept;
    };
    // ... and with the disk slots on, the state at the end of the system prompt (p) also into a slot of its own: the
    // checkpoints live with the model's state, so an unrelated conversation in between drops them, and another chat
    // with this system prompt read it all again (#179); the slot outlives that.  Once per system prompt.
    const auto slot_prefix = [&](const std::vector<int32_t>& prompt, size_t p,
                                 const char* why = " (its system prompt, for other chats that start with it)") {
        if (slot_max <= 0 || p < slot_min || !cur_img_pos.empty()) return;
        for (auto& sl : slots)
            if (sl.prefix && sl.tokens.size() == p && std::equal(sl.tokens.begin(), sl.tokens.end(), prompt.begin())) {
                sl.used = ++slot_clock;
                return;
            }
        if (!model.snapshot_save()) return;
        snap_tokens.assign(prompt.begin(), prompt.begin() + (long) p);
        snap_img_hash = img_hash(p);
        snap_in_slot = false;
        slot_put(why);
        if (snap_in_slot && !slots.empty() && slots.back().tokens.size() == p) slots.back().prefix = true;
    };
    // the idle contexts' conversations besides the bound one (the several-conversation scheduler sets it)
    std::function<void(const std::function<void(const std::vector<int32_t>&)>&)> each_ctx_snap;
    // SHARED SLOTS - the points a prompt read from `from` to `end` stops at: the checkpoints' (one conversation, see
    // above) and, with the disk slots on (several conversations too), the system prompt's end (sys_end) and where the
    // prompt parts from a kept conversation (div: an agent's sessions share their instructions past the system
    // prompt) - each kept as a shared slot (slot_prefix) for the next conversation that starts there
    // SAGUARO_GLM_SHARED_SLOTS=0: as v1.1.48 - the system prompt's slot only without checkpoints, no divergence slots, no gt=
    static const bool shared_on = env_num("SAGUARO_GLM_SHARED_SLOTS", 1) != 0;
    const auto read_marks = [&](const std::vector<int32_t>& prompt, size_t from, size_t end, size_t sys_end, size_t div) {
        std::vector<size_t> at = ckpt_marks(prompt, from, end);
        static const bool prefix_cut_on = env_num("SAGUARO_GLM_PREFIX_CUT", 1) != 0;   // (#387's switch: 0 = no system-prompt cut)
        if (!shared_on) {
            if (ckpt_max <= 0 && slot_max > 0 && prefix_cut_on && cur_img_pos.empty())
                maya::core::add_mark(at, maya::core::prefix_cut(prompt.data(), prompt.size(), from, slot_min, 154827),
                                       from);
            return at;
        }
        const size_t sys_cut = prefix_cut_on ? sys_end : 0;
        if (slot_max > 0 && cur_img_pos.empty())
            for (const size_t p : {sys_cut, div})
                // (also just before the end: a short first question - the few tokens after go through the token path)
                if (p >= std::max<size_t>(from + 64, (size_t) slot_min) && p < end) at.push_back(p);
        std::sort(at.begin(), at.end());
        at.erase(std::unique(at.begin(), at.end()), at.end());
        std::vector<size_t> kept;   // (every prompt-path call between them reaches its minimum)
        for (const size_t p : at)
            if (p >= (kept.empty() ? from : kept.back()) + 64) kept.push_back(p);
        return kept;
    };
    // the state now (after prompt[0, p)) into a free or the least recently used checkpoint
    const auto ckpt_keep = [&](const std::vector<int32_t>& prompt, size_t p, bool from_snapshot = false) {
        if ((int) ckpts.size() < ckpt_max) ckpts.emplace_back();
        size_t c = 0;
        for (size_t i = 0; i < ckpts.size(); ++i) {
            if (ckpts[i].tokens.empty()) { c = i; break; }
            if (ckpts[i].used < ckpts[c].used) c = i;
        }
        std::string e;
        ckpts[c].tokens.clear();
        const int slot = ckpt_base + (int) c;   // (several conversations: this context's own slots)
        if (!(from_snapshot ? model.ckpt_save_snapshot(slot, e) : model.ckpt_save(slot, e))) {
            std::fprintf(stderr, "glm checkpoints: not kept at %zu (%s)\n", p, e.c_str());
            return;
        }
        ckpts[c].tokens.assign(prompt.begin(), prompt.begin() + (long) p);
        ckpts[c].used = ++ckpt_clock;
    };
    // A route that found a non-finite score (or an expert id out of range) means a NaN or an infinity in the state the
    // request ran on: its conversation must not be served again.  The kept snapshot and every slot this prompt
    // extends (one may have been the poisoned state's source) are dropped and the model is reset, so the server's
    // retry reads the prompt from token 0.  The ERR line names it "non-finite" (Glm5Model::route_ok's words).
    const auto drop_if_poisoned = [&](const std::vector<int32_t>& prompt, const std::string& why) {
        if (why.find("non-finite") == std::string::npos) return;
        snap_tokens.clear();
        snap_in_slot = false;
        ckpt_drop_past(0);
        for (size_t i = slots.size(); i-- > 0;) {
            const auto& t = slots[i].tokens;
            if (t.size() <= prompt.size() && std::equal(t.begin(), t.end(), prompt.begin())) slot_drop(i);
        }
        model.reset();
        std::fprintf(stderr, "glm nan guard: %s - this conversation's state is dropped; it is read again from token 0\n",
                     why.c_str());
    };
    // set by the several-conversation scheduler while it admits a request (SAGUARO_GLM_PREFILL_INTERLEAVE): the piece the
    // prompt path reads next (0: the rest of the segment, as before) and what runs between two pieces - 0 go on,
    // 1 stopped (the request's STOP), 2 failed (prompt_between_err says why; every request ended, the state reset)
    std::function<size_t()> prompt_piece;
    std::function<int()> prompt_between;
    std::string prompt_between_err;
    // SAGUARO_GLM_PREFILL_GAPS: true inside a gap (the decode steps at a chunk boundary of a prompt's read, or between
    // two of its calls) - no tier boundary runs there, nothing binds another context (a PAUSE's snapshot waits for the
    // read's end: deferred_pauses), and an engine error leaves the contexts' reset to gap_reset, after the read has
    // ended (the pool's tail is lent in it); gap_failed: a gap inside a prompt-path call failed (prompt_between_err
    // says why); gaps_over: a gap of this read overran <max gap ms>, so the rest of the read has none
    bool in_read_gap = false, gap_failed = false, read_gap_first = false, gaps_over = false;
    // a request PAUSEd inside a gap: its context stays busy (nothing else binds or reads into it) until its snapshot is
    // taken as it would have been at once - after the read's call, once its first token is taken (flush_pauses)
    struct DeferredPause {
        int ctx = 0;
        std::vector<int32_t> prompt, emitted;
        bool keep = false;
        bool text = true;   // the request had no pictures
    };
    std::vector<DeferredPause> deferred_pauses;
    std::function<void()> gap_reset;
    std::chrono::steady_clock::time_point gap_mark;   // the read's start (after any restore), then each gap's end
    // this read's decode gaps (SAGUARO_GLM_PREFILL_GAPS; declared here so the PP lines' clock can subtract them: a
    // read's reported ms is the time it actually ran, not the time the requests beside it spent decoding in it)
    uint64_t gap_count = 0, gap_tok = 0;
    double gap_ms = 0.0;
    const auto read_ms_since = [&](const std::chrono::steady_clock::time_point& t0) {
        // a read's own running time since t0 (its PP/DONE ms): the decode gaps taken inside it are not read time
        return std::max(0.0, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()
                             - gap_ms);
    };
    // the GPUs the prompt-read line asks the free VRAM of: the ones the expert pool lives on (empty: every visible one)
    const auto vram_devs = [&]() {
        std::vector<int> d;
        if (model.fast())
            for (const auto& pv : model.pool_vram()) d.push_back(pv.dev);
        return d;
    };
    // CONVERSATION REUSE (the fast path): the previous prompt's state, saved just before its last token, is restored
    // when this prompt extends it - a chat's next turn reads only what is new; else a checkpoint or a slot that holds
    // it, else a reset.  The tokens the state already holds.
    size_t restore_div = 0;   // prep_restore's: where the prompt parts from a kept conversation (0: nowhere)
    const auto prep_restore = [&](const std::vector<int32_t>& prompt) -> size_t {
        size_t reuse = 0;
        maya::core::progress_at("the request: restoring a conversation");
        size_t lcp = 0, lcp_of = 0;   // the longest prefix this prompt shares with a kept conversation, and its length
        const auto lcp_with = [&](const std::vector<int32_t>& t) {
            const size_t k = (size_t) (std::mismatch(t.begin(), t.begin() + (long) std::min(t.size(), prompt.size()),
                                                     prompt.begin()).first - t.begin());
            if (k > lcp) {
                lcp = k;
                lcp_of = t.size();
            }
        };
        for (const auto& sl : slots) lcp_with(sl.tokens);
        lcp_with(snap_tokens);
        if (each_ctx_snap) each_ctx_snap(lcp_with);
        const size_t div = shared_on && lcp < lcp_of ? lcp : 0;   // where it parts from one that goes on differently
        restore_div = div;
        const bool snap_ok = model.fast() && !snap_tokens.empty() && snap_tokens.size() < prompt.size() &&
                             std::equal(snap_tokens.begin(), snap_tokens.end(), prompt.begin()) &&
                             img_hash(snap_tokens.size()) == snap_img_hash && getenv("SAGUARO_GLM_NO_REUSE") == nullptr;
        // the longest checkpoint this prompt starts with (it holds less than the snapshot when both match)
        int ck = -1;
        if (ckpt_max > 0 && cur_img_pos.empty())
            for (int c = 0; c < (int) ckpts.size(); ++c) {
                const auto& t = ckpts[(size_t) c].tokens;
                if (!t.empty() && t.size() < prompt.size() && (ck < 0 || t.size() > ckpts[(size_t) ck].tokens.size()) &&
                    std::equal(t.begin(), t.end(), prompt.begin()))
                    ck = c;
            }
        const size_t ck_len = ck >= 0 ? ckpts[(size_t) ck].tokens.size() : 0;
        bool from_ckpt = false;
        if (snap_ok && snap_tokens.size() >= ck_len && model.snapshot_restore()) reuse = snap_tokens.size();
        if (reuse == 0 && ck >= 0) {
            slot_put("");              // the conversation in the model goes on differently past the checkpoint: kept
            size_t slot_len = 0;       // a slot that holds more of this prompt is taken instead (below)
            if (slot_max > 0)
                for (const auto& sl : slots)
                    if (sl.tokens.size() > slot_len && sl.tokens.size() < prompt.size() &&
                        std::equal(sl.tokens.begin(), sl.tokens.end(), prompt.begin()))
                        slot_len = sl.tokens.size();
            std::string ce;
            if (slot_len > ck_len) {   // the slot first; when none loads (another layout, a damaged file) the checkpoint
                slot_load_failed = false;    // below is used instead of reading from token 0 - unless a failed load
                reuse = slot_take(prompt);   // wrote over this context's rows: its checkpoints stand on them
                if (reuse > 0 || slot_load_failed) ckpt_drop_past(0);   // (another conversation's state now)
                slot_load_failed = false;
            }
            if (reuse > 0) {
            } else if (model.ckpt_restore(ckpt_base + ck, ce)) {
                reuse = ck_len;
                from_ckpt = true;
                ckpts[(size_t) ck].used = ++ckpt_clock;
                snap_tokens.clear();   // the snapshot's position lies past the checkpoint: overwritten from here on
                std::fprintf(stderr, "glm checkpoints: %zu of the prompt's %zu tokens from a checkpoint\n", reuse,
                             prompt.size());
            } else {
                std::fprintf(stderr, "glm checkpoints: %s\n", ce.c_str());
                ckpts[(size_t) ck].tokens.clear();
            }
        }
        if (reuse == 0 && slot_max > 0) {
            slot_put("");              // the conversation in the model, before this prompt overwrites it
            reuse = slot_take(prompt);
            if (reuse > 0) ckpt_drop_past(0);   // another conversation's state now
        } else if (reuse > 0 && !from_ckpt && slot_max > 0 && getenv("SAGUARO_GLM_SLOT_ROUNDTRIP") != nullptr) {
            // a test: the restored state goes through a slot file and back - the same tokens must follow
            snap_in_slot = false;
            slot_put("");
            const size_t back = slot_take(prompt);
            if (back != reuse) {
                std::fprintf(stderr, "glm slots: ROUNDTRIP took back %zu of %zu tokens\n", back, reuse);
                reuse = back;
            }
        }
        if (reuse == 0) model.reset();
        if (reuse == 0 && prompt.size() >= 1024)
            std::fprintf(stderr, "glm slots: nothing kept continues this %zu-token prompt; the closest (%zu tokens) shares "
                         "its first %zu\n", prompt.size(), lcp_of, lcp);
        ckpt_drop_past(reuse);   // this prompt writes the state from `reuse` on (all of it after a reset)
        // THE TURN'S START: a prompt that goes on from the snapshot and opens its last turn (<|user|> / <|observation|>)
        // within 512 tokens of it gets no checkpoint there from ckpt_marks - so an edit of that turn's message read the
        // whole conversation again (13.5K tokens: 20.8 s).  The snapshot itself (the previous prompt but its last
        // token, a few tokens before the turn) is kept as a checkpoint instead: copied to host RAM on a side thread
        // while this prompt runs, so it costs the prompt nothing.  SAGUARO_GLM_CKPT_TURN=0: off.  After the slot
        // restore and ckpt_drop_past: a conversation just taken back from a slot gets it too (before, the restore
        // dropped the checkpoint kept here, and an edited turn after A -> B -> A read all of A again: #400)
        if (reuse > 0 && !from_ckpt && reuse == snap_tokens.size() && reuse >= 512 && ckpt_max > 0 &&
            cur_img_pos.empty()) {
            static const bool turn_on = [] { const char* v = getenv("SAGUARO_GLM_CKPT_TURN"); return !v || atoi(v) != 0; }();
            constexpr int32_t kUserTok = 154827, kObsTok = 154829;   // <|user|>, <|observation|>
            size_t last = 0;
            for (size_t i = prompt.size() - 1; i-- > reuse;)
                if (prompt[i] == kUserTok || prompt[i] == kObsTok) { last = i; break; }
            bool have = false;
            for (const auto& c : ckpts)
                have = have || (c.tokens.size() == reuse && std::equal(c.tokens.begin(), c.tokens.end(), prompt.begin()));
            if (turn_on && last >= reuse && last < reuse + 512 && !have) ckpt_keep(prompt, reuse, true);
        }
        return reuse;
    };
    // the prompt from token i on through the token path, the state saved before snap_at (an identical or extending
    // prompt can reuse it) - after the batched path, only the tokens from there are left
    const auto prep_tail = [&](const std::vector<int32_t>& prompt, size_t i_first, size_t reuse, const char*& finish,
                               std::chrono::steady_clock::time_point t0, size_t snap_at,
                               size_t* stopped = nullptr) -> int {
        std::vector<float> lg;
        bool snapped = false;
        for (size_t i = i_first; i < prompt.size();) {
            maya::core::progress_at("the prompt: the token path, at token", (int64_t) i);
            maya::core::progress_beat();
            if (model.fast() && !snapped && i == snap_at) {
                if (model.snapshot_save()) {
                    snap_tokens.assign(prompt.begin(), prompt.begin() + (long) snap_at);
                    snap_img_hash = img_hash(snap_tokens.size());
                    snap_in_slot = false;
                } else {
                    snap_tokens.clear();
                }
                snapped = true;
            }
            size_t chunk = std::min<size_t>(8, prompt.size() - i);
            if (model.fast() && !snapped && i < snap_at && i + chunk > snap_at) chunk = snap_at - i;
            std::vector<int32_t> part(prompt.begin() + (long) i, prompt.begin() + (long) (i + chunk));
            const bool ok = model.forward(part, lg, err);
            if (!ok) {
                snap_tokens.clear();
                drop_if_poisoned(prompt, err);
                std::printf("ERR %s%s\n", tag.c_str(), err.c_str());
                return 1;
            }
            i += chunk;
            if (o.serve) {
                const double ms = read_ms_since(t0);   // (the read's own time: decode gaps taken inside it are out)
                std::printf("PP %s%zu %zu %.1f %.1f %zu\n", tag.c_str(), i, prompt.size(), ms,
                            (double) (i - reuse) / (ms / 1000.0 + 1e-9), reuse);
            }
            if (read_stopped()) {
                finish = "cancel";
                if (stopped != nullptr) *stopped = i;
                break;
            }
        }
        return 0;
    };
    const auto prep_prompt = [&](const std::vector<int32_t>& prompt, size_t& reuse, const char*& finish,
                                 std::chrono::steady_clock::time_point& t0) -> int {
        reuse = prep_restore(prompt);
        // (parity 3: the exact read end after a restore - the line before the read says what the state already
        //  holds, so a restored read's fresh tokens and ms are its own: same shape as PP's, rid-tagged under multi)
        if (o.serve) std::printf("RESUME %s%zu\n", tag.c_str(), reuse);
        const maya::core::VramMark vram0 = maya::core::vram_mark(vram_devs());   // (the line at the read's end)
        size_t read_to = prompt.size();   // (where a cancelled read stopped)
        t0 = std::chrono::steady_clock::now();
        gap_mark = t0;   // (SAGUARO_GLM_PREFILL_GAPS: the first gap's stage starts with the read, not the restore)
        gap_count = gap_tok = 0;   // (every read starts its own gap figures: one without gaps must not inherit them)
        gap_ms = 0.0;
        // the prompt one CHUNK at a time (PP reports per chunk: the server shows it and counts it as a heartbeat);
        // the state is saved before the generation prompt (snap_at), so an identical or extending prompt can reuse it
        // THE SNAPSHOT'S POSITION: before the prompt's generation prompt (its trailing <|assistant|>, then <think> or
        // <think></think>), where it used to stand before the last token.  The next prompt of this conversation goes on
        // from the same place whatever came of this answer - the answer itself, nothing (a cancelled answer that the
        // client sends again with a new message after it), or the same prompt again - so it continues the snapshot
        // instead of leaving it at the <|assistant|> it no longer has.  Opt-in (SAGUARO_GLM_SNAP_GEN=1): the prompt's
        // last 2-4 tokens then go through the token path instead of the batched one, so a greedy answer can differ
        // from a read with the snapshot before the last token (the default).
        // The server's gt=<n> (the reply's opening, the template's own count) wins over the scan below.
        const int64_t gt = shared_on ? req_gt : 0;
        size_t snap_at = gt > 0 ? snap_pos(prompt, gt) : prompt.size() - 1;
        if (gt <= 0) {
            static const bool snap_gen = env_num("SAGUARO_GLM_SNAP_GEN", 0) != 0;
            constexpr int32_t kAsstTok = 154828;   // <|assistant|>
            // (at `reuse` itself too: the same prompt again restores the snapshot there and keeps it there)
            for (size_t k = 2; snap_gen && k <= 4 && k <= prompt.size(); ++k) {
                const size_t a = prompt.size() - k;
                if (a < reuse || a == 0) break;
                if (prompt[a] == kAsstTok) { snap_at = a; break; }
            }
        }
        // the batched prompt path (glm_prefill.cu) reads all but the last prompt token in chunks, layer by layer;
        // the last one goes through the token path, which leaves the logits the first sample reads
        const int pchunk = model.fast() ? model.prefill_chunk() : 0;
        size_t i_first = reuse;
        if (pchunk > 0 && snap_at > reuse) {
            // one call for everything but the last token: the two halves of a split pipeline the chunks; PP lines
            // (the server's progress and heartbeat) come from the progress callback, possibly on the tail's thread.
            // A checkpoint position (ckpt_marks) ends a call: the state there is kept, and the next call goes on.
            std::mutex pp_mu;
            const auto pp_line = [&](size_t done) {
                if (!o.serve) return;
                std::lock_guard<std::mutex> lk(pp_mu);
                const double ms = read_ms_since(t0);   // (the read's own time: no decode gaps in it)
                std::printf("PP %s%zu %zu %.1f %.1f %zu\n", tag.c_str(), reuse + done, prompt.size(), ms,
                            (double) done / (ms / 1000.0 + 1e-9), reuse);
                std::fflush(stdout);
            };
            // the first user turn's marker: the end of the system prompt (ckpt_marks' `first`)
            const size_t sys_end = maya::core::system_prompt_end(prompt.data(), prompt.size(), 154827);   // <|user|>
            const std::vector<size_t> marks = read_marks(prompt, reuse, snap_at, sys_end, restore_div);
            size_t at = reuse;   // the next token the prompt path reads
            bool pok = true, between_failed = false;
            // one call to the next checkpoint mark (or the prompt's last token) - with SAGUARO_GLM_PREFILL_INTERLEAVE and
            // requests decoding, a piece of it, and their decode steps between two pieces (prompt_between)
            size_t mi = 0, seg_start = at;
            // SAGUARO_GLM_PREFILL_GAPS: a gap before the read too (a round for the running requests) - back-to-back reads
            // would otherwise stall them for the last stage of one and the first of the next
            if (read_gap_first && prompt_between && at < snap_at) {
                const int r = prompt_between();
                if (r == 1) {
                    pok = false;
                    err = "cancelled";
                } else if (r == 2) {
                    pok = false;
                    between_failed = true;
                }
            }
            int seg_chunk = 0;   // (exact pieces: the chunk the segment's uncut read takes, from its first piece on)
            while (pok && at < snap_at) {
                const size_t seg_end = mi < marks.size() ? marks[mi] : snap_at;
                size_t piece = prompt_piece ? prompt_piece() : 0;
                if (piece > 0 && interleave.exact) {
                    // whole chunks of the uncut read, each call cut in them: the same arithmetic, the same answer
                    if (at == seg_start || seg_chunk == 0) seg_chunk = model.prefill_chunk_for((int64_t) (seg_end - seg_start));
                    piece = maya::core::prefill_aligned_piece(piece, (size_t) seg_chunk);
                }
                const size_t end = maya::core::prefill_piece_end(at, seg_end, piece);
                // (an interleaved piece cut freely (,cut): both GPUs of a split at work - two chunks, or ~4096-token ones
                // for a larger piece: a call fills and drains the split's pipeline, a GPU idle for a chunk each way)
                model.prefill_min_chunks = piece > 0 && !interleave.exact ? std::max<int>(2, (int) (piece / 4096)) : 1;
                model.prefill_force_chunk = seg_chunk > 0 && interleave.exact ? seg_chunk : 0;
                const std::vector<int32_t> part(prompt.begin() + (long) at, prompt.begin() + (long) end);
                const size_t base = at - reuse;
                model.prefill_progress = [&](int64_t done, int64_t total) {
                    (void) total;
                    pp_line(base + (size_t) done);
                    return !read_stopped();
                };
                pok = guarded([&] { return model.prefill(part, err, prompt[end]); }, err);   // (the NextN block's caches read the next token)
                model.prefill_progress = nullptr;
                model.prefill_min_chunks = 1;
                model.prefill_force_chunk = 0;
                if (!pok && gap_failed) between_failed = true;   // (a decode gap inside the call failed)
                if (pok) {
                    at = end;
                    if (end == seg_end && mi < marks.size()) {
                        if (ckpt_max > 0) ckpt_keep(prompt, end);   // (no checkpoints: the mark is a prefix cut's only)
                        if (end == sys_end) slot_prefix(prompt, end);
                        else if (end == restore_div)
                            slot_prefix(prompt, end, " (where it parts from a kept conversation, for the next one that does)");
                        ++mi;
                        seg_start = end;
                        seg_chunk = 0;
                    }
                    // a cancel between two calls: a call that is one chunk (a checkpoint every chunk, or a piece) never
                    // asks the progress callback, so with one conversation the read ran to its end after the client had
                    // gone - it stops here, at the checkpoint just kept (a cancelled read keeps it, below)
                    if (at < snap_at && !prompt_between && read_stopped()) {
                        pok = false;
                        err = "cancelled";
                        break;
                    }
                    // more to read: the decoding requests' turn (none after the last piece - its last token's logits
                    // must still be on the tail when the first sample reads them)
                    if (at < snap_at && prompt_between) {
                        pp_line(at - reuse);
                        const int r = prompt_between();
                        if (r == 1) {
                            pok = false;
                            err = "cancelled";
                        } else if (r == 2) {
                            pok = false;
                            between_failed = true;
                        }
                    }
                }
            }
            if (between_failed) {   // every context was reset (multi_fail): nothing of this prompt is left to keep
                drop_if_poisoned(prompt, prompt_between_err);   // (a non-finite error: its slots too, as below)
                std::printf("ERR %s%s\n", tag.c_str(), prompt_between_err.c_str());
                if (gap_reset) gap_reset();   // (a gap's failure: the reset multi_fail left for after the read)
                return 1;
            }
            // a call the prompt path did not take (fewer tokens than SAGUARO_GLM_PREFILL_MIN: false, no error) leaves the
            // rest to the token path below - from where the prompt path stopped, not from `reuse`
            i_first = at;
            if (pok) {
                i_first = snap_at;
                pp_line(snap_at - reuse);
            } else if (err == "cancelled") {
                // A CANCELLED READ keeps what it read, so the client's retry - an agent that gave up waiting and sends
                // the same prompt again - or the next turn goes on from there.  The prompt path stops every part at the
                // same chunk boundary: that state is kept as is; failing that, the longest checkpoint of this prompt the
                // read has reached (#573).  Nothing read past what was restored: the restored conversation stays.
                // SAGUARO_GLM_KEEP_CANCELLED=0: the state is dropped, as before.
                static const bool keep_cancelled = env_num("SAGUARO_GLM_KEEP_CANCELLED", 1) != 0;
                static const bool keep_exact = env_num("SAGUARO_GLM_KEEP_CANCELLED_EXACT", 1) != 0;   // (0: #573's alone)
                size_t kept = 0;
                const int64_t pos = model.position();
                const bool as_is = keep_cancelled && keep_exact && pos == (int64_t) reuse && reuse > 0;
                if (keep_cancelled && keep_exact && model.fast() && pos > (int64_t) reuse && pos < (int64_t) prompt.size() &&
                    model.snapshot_save()) {
                    kept = (size_t) pos;
                    snap_tokens.assign(prompt.begin(), prompt.begin() + (long) kept);
                    snap_img_hash = img_hash(kept);
                    snap_in_slot = false;
                    ckpt_drop_past(kept);
                    std::fprintf(stderr, "glm slots: a cancelled prompt keeps the %zu tokens it read (%zu new)\n", kept,
                                 kept - reuse);
                } else if (keep_cancelled && !as_is && ckpt_max > 0 && cur_img_pos.empty()) {
                    int best = -1;
                    for (int c = 0; c < (int) ckpts.size(); ++c) {
                        const auto& t = ckpts[(size_t) c].tokens;
                        if (!t.empty() && t.size() <= at && (best < 0 || t.size() > ckpts[(size_t) best].tokens.size()) &&
                            std::equal(t.begin(), t.end(), prompt.begin()))
                            best = c;
                    }
                    std::string ce;
                    if (best >= 0 && model.ckpt_restore(ckpt_base + best, ce) && model.snapshot_save()) {
                        kept = ckpts[(size_t) best].tokens.size();
                        snap_tokens.assign(prompt.begin(), prompt.begin() + (long) kept);
                        snap_img_hash = img_hash(kept);
                        snap_in_slot = false;
                        ckpts[(size_t) best].used = ++ckpt_clock;
                        ckpt_drop_past(kept);
                        std::fprintf(stderr, "glm checkpoints: the cancelled read keeps %zu of the prompt's %zu tokens\n",
                                     kept, prompt.size());
                    } else if (best >= 0) {
                        std::fprintf(stderr, "glm checkpoints: the cancelled read keeps nothing (%s)\n",
                                     ce.empty() ? "no snapshot" : ce.c_str());
                    }
                }
                if (kept == 0 && !as_is) {
                    snap_tokens.clear();
                    model.reset();
                    ckpt_drop_past(0);
                }
                finish = "cancel";
                read_to = at;
                i_first = prompt.size();
            } else if (!err.empty()) {
                snap_tokens.clear();
                drop_if_poisoned(prompt, err);
                std::printf("ERR %s%s\n", tag.c_str(), err.c_str());
                return 1;
            }
        }
        const int rc = prep_tail(prompt, i_first, reuse, finish, t0, snap_at, &read_to);
        {   // one line a prompt: the tokens read, the time, and the free VRAM before and after (issues #648, #438)
            const std::string vt = maya::core::vram_mark_text(vram0, maya::core::vram_mark(vram_devs()));
            const bool cancelled = finish != nullptr && std::strcmp(finish, "cancel") == 0;
            if (!vt.empty())
                std::fprintf(stderr, "glm vram: prompt %zu tokens read in %.1f s%s; %s\n",
                             read_to > reuse ? read_to - reuse : (size_t) 0,
                             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                             cancelled ? (" (cancelled at token " + std::to_string(read_to) + " of " +
                                          std::to_string(prompt.size()) + ")").c_str() : "",
                             vt.c_str());
        }
        return rc;
    };
    // PAUSE: the snapshot moves from the prompt to the last token fed (the decode feeds every written token but
    // the last), so prompt + what was written continues it with one token to read instead of all it wrote.
    // Text only (`text`: the request had no pictures - the multi path's cur_img_* are the last GEN's, not its own),
    // and only when the state stands exactly there (the decode paths leave pos = fed tokens).
    const auto pause_snapshot = [&](const std::vector<int32_t>& prompt, const std::vector<int32_t>& emitted,
                                    bool keep, bool text) {
        if (model.fast() && text && !emitted.empty() && getenv("SAGUARO_GLM_NO_REUSE") == nullptr) {
            const int64_t p = model.position();
            const int64_t full = (int64_t) (prompt.size() + emitted.size());
            if (keep && slot_max > 0 && p > (int64_t) prompt.size() && p < full) {
                // the request's first slice: its prompt's state goes to a slot (unless one holds it already) and is
                // anchored there, for this chat's next turn
                bool held = false;
                for (auto& sl : slots)
                    if (sl.tokens == snap_tokens) sl.anchor = held = true;
                if (!held && !snap_tokens.empty()) {
                    slot_put(" (a paused request's prompt, kept for its next turn)");
                    for (auto& sl : slots)
                        if (sl.tokens == snap_tokens) sl.anchor = true;
                }
            }
            if (p > (int64_t) prompt.size() && p < full && model.snapshot_save()) {
                snap_tokens.assign(prompt.begin(), prompt.end());
                snap_tokens.insert(snap_tokens.end(), emitted.begin(), emitted.begin() + (p - (int64_t) prompt.size()));
                snap_img_hash = kNoImages;
                snap_in_slot = false;
                std::fprintf(stderr, "glm slots: paused at %lld tokens (%lld written): the state is kept there\n",
                             (long long) p, (long long) emitted.size());
            } else if (p > (int64_t) prompt.size()) {
                std::fprintf(stderr, "glm slots: paused, but the state stands at %lld of %lld tokens: not kept\n",
                             (long long) p, (long long) full);
            }
        }
    };
    // SAGUARO_GLM_SPEC_FOLLOW=<file> (--spec-follow): every request decodes this continuation instead of its own picks
    // and ends with it (Glm5Model::set_follow) - a speed A/B on one text.  SAGUARO_GLM_SPEC_FOLLOW_SAVE=<file>
    // (--spec-follow-save): each request's decoded ids are written there, the next run's continuation.  A text made by
    // another model, or by the same model with other settings, favours the drafts that made it: compare a change on
    // the same model and its own text.
    std::vector<int32_t> follow_ids;
    const char* follow_save = getenv("SAGUARO_GLM_SPEC_FOLLOW_SAVE");
    if (follow_save != nullptr && follow_save[0] == '\0') follow_save = nullptr;
    if (const char* fp = getenv("SAGUARO_GLM_SPEC_FOLLOW"); fp != nullptr && fp[0] != '\0') {
        std::ifstream input(fp);
        std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        std::vector<int64_t> v;
        std::string e;
        if (!input && !input.eof()) e = "cannot read it";
        else if (!parse_i64_list(text.c_str(), v, e)) e = "not a list of token ids: " + e;
        for (const int64_t t : v)
            if (e.empty() && t >= n_vocab) e = "token " + std::to_string(t) + " is past the vocabulary";
        if (!e.empty()) {
            std::fprintf(stderr, "saguaro generate: SAGUARO_GLM_SPEC_FOLLOW %s: %s\n", fp, e.c_str());
            return 2;
        }
        follow_ids.assign(v.begin(), v.end());
        std::fprintf(stderr, "glm follow: every request decodes the %zu tokens of %s (SAGUARO_GLM_SPEC_FOLLOW)\n",
                     follow_ids.size(), fp);
    }
    const auto run_request = [&](const std::vector<int64_t>& ids, int64_t max_new,
                                 const maya::kernels::SamplerParams& rq_in) -> int {
        maya::kernels::SamplerParams sp = rq_in;   // per request (the server sends sampling keys per GEN)
        maya::kernels::LpHold lp_hold;   // logprobs=<k>: this request's rows
        if (sp.lp_k >= 0) {
            std::string le;
            lp_hold.reset(sp.lp = maya::kernels::lp_sink_get(sp.lp_k, le));
            if (sp.lp == nullptr) std::fprintf(stderr, "glm: %s - this request's tokens without them\n", le.c_str());
        }
        stop_req.store(false);
        pause_req.store(false);
        pause_keep.store(false);
        model.force_next(-1);
        model.set_follow(follow_ids);   // SAGUARO_GLM_SPEC_FOLLOW: this request's tokens (empty: its own picks)
        if (!follow_ids.empty()) max_new = std::min<int64_t>(max_new, (int64_t) follow_ids.size());
        bool in_think = req_think_budget > 0 && req_think_end >= 0;
        std::vector<int32_t> prompt;
        prompt.reserve(ids.size());
        for (const int64_t t : ids) {
            if (t < 0 || t >= n_vocab) {
                std::printf("ERR token %lld out of range [0, %lld)\n", (long long) t, (long long) n_vocab);
                return 1;
            }
            prompt.push_back((int32_t) t);
        }
        if (prompt.empty() || max_new <= 0) {
            std::printf("ERR empty prompt or max_new\n");
            return 1;
        }
        // the penalties' history (#273): what the state is fed - the prompt, then every token written (on_token)
        std::vector<int32_t> pen_hist;
        if (sp.penalty_last_n > 0) {
            pen_hist = prompt;
            sp.host_history = &pen_hist;
        }
        size_t reuse = 0;
        const char* finish = "length";
        auto t0 = std::chrono::steady_clock::now();
        if (prep_prompt(prompt, reuse, finish, t0) != 0) return 1;
        std::vector<float> lg;
        if (stop_req.load()) finish = "cancel";
        const auto t1 = std::chrono::steady_clock::now();
        const auto st1 = model.fast_stats();
        const uint64_t spec0 = model.spec_steps_, hits0 = model.spec_hits_;   // (the drafts this request so far: STAT, DONE)
        int64_t produced = 0;
        // one emitted token: the T line, the live tier counters every 8 tokens, the stop conditions (false: stop)
        // LOOP GUARD: a heavily quantized model can fall into repeating a short pattern ("0 0 0 ...") until the
        // thinking budget or max_tokens ends it - minutes of nothing.  When the last W tokens are one pattern of at most
        // 32 tokens repeated exactly, thinking is closed at once (as the budget does); in the answer, what the model is in the
        // middle of is closed (a tool call's argument, else the turn), and a second loop ends it.
        // W: SAGUARO_GLM_LOOP_THINK (256) while thinking, SAGUARO_GLM_LOOP_ANSWER (1024) in the answer - long enough
        // for legitimately repetitive output (a zero-filled array); 0 turns that check off.
        static const int loop_w_think = (int) env_num("SAGUARO_GLM_LOOP_THINK", 256);
        static const int loop_w_answer = (int) env_num("SAGUARO_GLM_LOOP_ANSWER", 1024);
        std::vector<int> gen_hist;
        std::vector<int32_t> emitted;   // every token this request wrote (a PAUSE keeps the state they lead to)
        bool in_arg = false;     // inside a tool call's <arg_value> (the answer's position, for the loop guard)
        int guard_fired = 0;     // soft landings in this answer: a second loop ends it
        const auto looping = [&](int w) -> int {   // the pattern's period, 0 when the tail is not one
            const int n = (int) gen_hist.size();
            if (w <= 0 || n < w) return 0;
            for (int per = 1; per <= 32 && per < w / 4; ++per) {
                bool rep = true;
                for (int i = n - w + per; i < n && rep; ++i) rep = gen_hist[(size_t) i] == gen_hist[(size_t) (i - per)];
                if (rep) return per;
            }
            return 0;
        };
        const auto on_token = [&](int tok) -> bool {
            std::printf("T %d\n", tok);
            if (sp.lp != nullptr) {
                const std::string l = maya::kernels::logprob_line(sp.lp, -1, (int64_t) pen_hist.size(), tok);
                if (!l.empty()) std::printf("%s\n", l.c_str());
            }
            ++produced;
            emitted.push_back((int32_t) tok);
            if (sp.host_history != nullptr) pen_hist.push_back((int32_t) tok);   // (fed next: the picks after count it)
            maya::core::progress_at("decode: after token", produced);   // the watchdog's heartbeat (issue #40)
            maya::core::progress_beat();
            gen_hist.push_back(tok);
            if (in_think && tok != req_think_end) {
                if (const int per = looping(loop_w_think)) {
                    model.force_next(req_think_end);   // as the budget does: the next token closes the reasoning
                    in_think = false;
                    char m[200];
                    std::snprintf(m, sizeof m, "thinking was repeating a %d-token pattern (%d tokens in): closed, "
                                  "answering", per, (int) produced);
                    std::fprintf(stderr, "glm loop guard: %s\n", m);
                    if (o.serve) std::printf("NOTE %s\n", m);
                    gen_hist.clear();
                }
            } else if (!in_think) {
                // where the answer stands: inside a tool call's argument value, the place a loop must be closed for
                // an agent to go on (GLM's single-token tags)
                constexpr int kArgValue = 154849, kArgValueEnd = 154850, kUser = 154827;
                if (tok == kArgValue) in_arg = true;
                if (tok == kArgValueEnd) in_arg = false;
                if (const int per = looping(loop_w_answer)) {
                    char m[240];
                    if (guard_fired == 0) {
                        // a soft landing: close what the model is in the middle of instead of cutting the response -
                        // a tool call's argument ends (the call completes with what was written, the agent runs it
                        // and sees the result), plain text ends the turn (finish "stop")
                        model.force_next(in_arg ? kArgValueEnd : kUser);
                        std::snprintf(m, sizeof m, "the answer was repeating a %d-token pattern (%d tokens in): %s",
                                      per, (int) produced, in_arg ? "the tool call's argument was closed there"
                                                                  : "the turn was ended there");
                    } else {
                        std::snprintf(m, sizeof m, "the answer kept repeating a %d-token pattern (%d tokens in): "
                                      "ended", per, (int) produced);
                    }
                    std::fprintf(stderr, "glm loop guard: %s\n", m);
                    if (o.serve) std::printf("NOTE %s\n", m);
                    gen_hist.clear();
                    if (guard_fired++ > 0) {   // it did not stop: the hard way
                        finish = "length";
                        return false;
                    }
                }
            }
            if (in_think && (tok == req_think_end || produced >= req_think_budget || wrap_req.load())) {
                if (tok != req_think_end) model.force_next(req_think_end);   // the budget: the next token closes it
                in_think = false;
                gen_hist.clear();
            }
            // live expert-tier counters for the server's Monitor (serve/server.py parses STAT lines), every 8 tokens
            if (o.serve && model.fast() && produced % 8 == 0) {
                const auto sn = model.fast_stats();
                const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
                const double n = (double) produced;
                const uint64_t h = sn.hits - st1.hits, m = sn.misses - st1.misses;
                std::printf("STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f ram_fetch=%.3f disk=%.3f promo=%.3f "
                            "vram_used=%lld vram_slots=%lld ram_used=%lld ram_slots=%lld ram_gb=%.1f vram_gb=%.1f "
                            "disk_gb=%.1f draft_acc=%llu draft_off=%llu\n",
                            n / std::max(1e-9, el), 1000.0 * el / n, (double) h / (double) std::max<uint64_t>(1, h + m),
                            (double) (sn.ram_hits - st1.ram_hits) / n, (double) (sn.disk_reads - st1.disk_reads) / n,
                            (double) (sn.promotions - st1.promotions) / n, (long long) sn.pool_used,
                            (long long) sn.pool_slots, (long long) sn.ram_used, (long long) sn.ram_slots, sn.ram_gb,
                            sn.pool_gb, std::max(0.0, sn.all_gb - sn.pool_gb - sn.ram_gb),
                            (unsigned long long) (model.spec_hits_ - hits0),
                            (unsigned long long) (model.spec_steps_ - spec0));
            }
            bool eos_hit = false;
            for (const int64_t e : eos)
                if (e == tok) eos_hit = true;
            if (eos_hit && stop_on_eos) {
                finish = "stop";
                return false;
            }
            if (stop_req.load()) {
                finish = "cancel";
                return false;
            }
            return produced < max_new;
        };
        if (std::strcmp(finish, "cancel") != 0 && model.fast() && model.dflash_ready() && req_spec != 1) {
            // the DFlash decode: a block of drafts a round, one verify window checks them (same tokens as the loop
            // below for the same samples); --spec T caps a round's window at T tokens
            int64_t n_spec = 0;
            const int spec = req_spec >= 2 ? req_spec : o.spec;   // this request's window, else --spec
            const int n_max = spec >= 2 ? spec - 1 : -1;            // -1: the engine's (decode_dflash)
            // the drafts stop at a selector confidence below the cut: the request's, else --spec-min-p, else the
            // engine's (-1: decode_dflash picks it for the window it verifies with)
            const float p_min = req_spec_min_p >= 0.0 && req_spec_min_p <= 1.0 ? (float) req_spec_min_p
                                : o.spec_min_p_set                         ? (float) o.spec_min_p
                                                                           : -1.0f;
            if (!model.decode_dflash(sp, max_new, n_max, p_min, on_token, n_spec, err, req_spec_n_min,
                                     req_spec_cum >= 0.0 && req_spec_cum <= 1.0 ? (float) req_spec_cum : -1.0f)) {
                drop_if_poisoned(prompt, err);
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
        } else if (std::strcmp(finish, "cancel") != 0 && model.fast() && model.mtp_ready() && req_spec != 1) {
            // the one-GPU MTP decode (glm_mtp.cu; by default on the Strix / Gorgon Halo APUs): the NextN block drafts,
            // one verify window checks the drafts (same tokens as the loop below for the same samples)
            int64_t n_spec = 0;
            const uint64_t rounds0 = model.mtp_rounds_;
            uint64_t len0[9];
            std::copy(model.mtp_round_n_, model.mtp_round_n_ + 9, len0);
            if (!model.decode_mtp(sp, max_new, on_token, n_spec, err)) {
                drop_if_poisoned(prompt, err);
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            if (model.mtp_rounds_ > rounds0) {
                std::string lens;
                for (int i = 0; i < 9; ++i)
                    if (model.mtp_round_n_[i] > len0[i])
                        lens += " " + std::to_string(i) + ":" + std::to_string(model.mtp_round_n_[i] - len0[i]);
                std::fprintf(stderr, "glm mtp: %llu rounds, %.2f tokens a round | rounds by draft length%s\n",
                             (unsigned long long) (model.mtp_rounds_ - rounds0),
                             (double) n_spec / (double) (model.mtp_rounds_ - rounds0), lens.c_str());
            }
        } else if (std::strcmp(finish, "cancel") != 0 && model.fast() && model.spec_ready()) {
            // the pipelined speculative decode: the two halves of the split work on consecutive tokens, the head
            // running the NextN block's draft (same tokens as the loop below)
            // SAGUARO_GLM_SPEC_WIN=1: the MTP drafts + the pipelined verify window through a split (decode_spec_pipe)
            int64_t n_spec = 0;
            // (unset: on for a split of three or more Volta GPUs - Glm5Model::spec_win_mode)
            const bool piped = model.spec_win_mode() != 0;
            const bool done_ok = piped ? model.decode_spec_pipe(sp, max_new, on_token, n_spec, err)
                                       : model.decode_spec(sp, max_new, on_token, n_spec, err);
            if (!done_ok) {
                drop_if_poisoned(prompt, err);
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
        } else {
            for (;;) {
                if (stop_req.load()) {
                    finish = "cancel";
                    break;
                }
                int tok = model.sample_token(sp, err);
                if (tok < 0) {
                    drop_if_poisoned(prompt, err);
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                tok = model.forced(tok);
                sp.counter += 1;
                if (!on_token(tok)) break;
                if (!model.forward({(int32_t) tok}, lg, err)) {
                    drop_if_poisoned(prompt, err);
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
            }
        }
        const auto t2 = std::chrono::steady_clock::now();
        const double prompt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        // the decode phase's expert-tier counters (the fast path; zeros on the reference path)
        const auto st2 = model.fast_stats();
        const uint64_t d_hits = st2.hits - st1.hits, d_miss = st2.misses - st1.misses;
        std::printf("DONE %lld %lld %.1f %.1f %s %llu %llu %zu %llu %llu\n", (long long) produced,
                    (long long) prompt.size(), prompt_ms, decode_ms, finish,
                    (unsigned long long) (model.spec_hits_ - hits0), (unsigned long long) (model.spec_steps_ - spec0),
                    reuse, (unsigned long long) d_hits, (unsigned long long) (d_hits + d_miss));
        std::fflush(stdout);
        if (pause_req.load() && std::strcmp(finish, "cancel") == 0) pause_snapshot(prompt, emitted, pause_keep.load(), cur_img_pos.empty());
        // this machine's expert usage, for the next start's warm-up (after DONE: the client is not kept waiting)
        if (model.fast()) model.save_usage();
        if (model.fast() && produced > 0) {
            const double n = (double) produced;
            std::fprintf(stderr,
                         "glm stat: decode %.2f ms/tok (%.2f tok/s) | vram hit %.2f%% | ram fetches %.2f/tok | "
                         "disk %.2f reads/tok in %.2f waits/tok, %.2f ms/tok | promotions %.2f/tok | cpu lane %.2f/tok "
                         "in %.2f ms/tok | prompt %.1f ms/tok | vram %lld/%lld ram %lld/%lld\n",
                         decode_ms / n, 1000.0 * n / decode_ms,
                         100.0 * (double) d_hits / (double) std::max<uint64_t>(1, d_hits + d_miss),
                         (double) (st2.ram_hits - st1.ram_hits) / n, (double) (st2.disk_reads - st1.disk_reads) / n,
                         (double) (st2.miss_layers - st1.miss_layers) / n, (st2.disk_ms - st1.disk_ms) / n,
                         (double) (st2.promotions - st1.promotions) / n,
                         (double) (st2.cpu_experts - st1.cpu_experts) / n, (st2.cpu_ms - st1.cpu_ms) / n,
                         prompt_ms / (double) std::max<size_t>(1, prompt.size()), (long long) st2.pool_used,
                         (long long) st2.pool_slots, (long long) st2.ram_used, (long long) st2.ram_slots);
            if (st2.disk_skips > st1.disk_skips)
                std::fprintf(stderr, "glm skip: %.2f disk-only experts/tok left out (SAGUARO_GLM_DISK_SKIP_W / _FROM)\n",
                             (double) (st2.disk_skips - st1.disk_skips) / n);
            if (model.spec_steps_ > spec0)
                std::fprintf(stderr, "glm spec: %llu of %llu drafts accepted (%.1f%%)\n",
                             (unsigned long long) (model.spec_hits_ - hits0),
                             (unsigned long long) (model.spec_steps_ - spec0),
                             100.0 * (double) (model.spec_hits_ - hits0) / (double) (model.spec_steps_ - spec0));
            if (!follow_ids.empty())
                std::fprintf(stderr, "glm follow: %zu tokens of the continuation decoded, %zu of them differ from the "
                                     "model's own pick\n", model.follow_used(), model.follow_differ());
            if (model.kv_resident() > 0) {
                // KV streaming, cumulative over the process: the blocks the decode's selections named, and those
                // read from RAM (every part summed)
                const auto kv = model.kv_stream_stats();
                std::fprintf(stderr, "glm kv: KV streaming: %.2f%% of %llu block reads hit VRAM, %.1f MiB read from RAM%s\n",
                             kv.lookups ? 100.0 * (double) (kv.lookups - kv.misses) / (double) kv.lookups : 100.0,
                             (unsigned long long) kv.lookups,
                             (double) kv.misses * kv.block_bytes / 1048576.0,
                             kv.overflow ? " - OVERFLOW (too few resident positions)" : "");
            }
        }
        model.set_follow({});
        // SAGUARO_GLM_SPEC_FOLLOW_SAVE: this request's decoded ids, the continuation another run can follow
        if (follow_save != nullptr && !emitted.empty()) {
            const std::string tmp = std::string(follow_save) + ".tmp";
            {
                std::ofstream f(tmp, std::ios::trunc);
                for (size_t i = 0; i < emitted.size(); ++i) f << (i ? "," : "") << emitted[i];
                f << "\n";
            }
            std::error_code ec;
            std::filesystem::rename(tmp, follow_save, ec);
            if (ec)
                std::fprintf(stderr, "glm follow: could not write %s: %s\n", follow_save, ec.message().c_str());
            else
                std::fprintf(stderr, "glm follow: %zu decoded ids written to %s (SAGUARO_GLM_SPEC_FOLLOW_SAVE)\n",
                             emitted.size(), follow_save);
        }
        return 0;
    };

    // ---- SEVERAL CONVERSATIONS AT ONCE (serve, SAGUARO_GLM_SEQS=<n> > 1).  GEN lines carry rid=<id>, and every line
    // a request causes carries it after its keyword (T <rid> <id>, PP <rid> ..., DONE <rid> ..., ERR <rid> ...);
    // STOP <rid> and PAUSE <rid> [keep] end one request, STOP alone all of them.  Each request runs in one of the n
    // sequence states, a CONTEXT: an idle context whose snapshot the prompt extends is taken as it stands (nothing
    // read, nothing loaded), else the least recently used idle one, whose conversation goes to a slot first.  A
    // request alone decodes through the speculative path (the fastest for one); as soon as a second one comes, both go
    // through the pipelined decode (Glm5Model::multi_issue), each card of the split on a different conversation.
    const bool multi = o.serve && model.fast() && model.seq_count() > 1;
    if (multi && !follow_ids.empty()) {
        std::fprintf(stderr, "glm follow: not used with SAGUARO_GLM_SEQS=%d - one conversation at a time only\n",
                     model.seq_count());
        follow_ids.clear();
    }
    // SAGUARO_GLM_MULTI_MTP=<drafts>: the running requests' batched steps carry verify windows - each one's last token
    // and the NextN block's drafts after it (Glm5Model::mtpw_*, maya/core/mtp_rows.hpp)
    const bool mtp_rows = multi && model.mtpw_ok();
    if (multi && maya::core::Glm5Model::multi_mtp_setting() > 0 && !mtp_rows)
        std::fprintf(stderr, "glm multi mtp: SAGUARO_GLM_MULTI_MTP is set but not used here - it needs SAGUARO_GLM_SPLIT_ROWS "
                             "on a split (or one GPU), the NextN block loaded (no SAGUARO_GLM_NO_MTP) and no --dflash\n");
    if (mtp_rows)
        std::fprintf(stderr, "glm multi mtp: the running requests verify up to %d drafts each in their batched steps "
                             "(%d rows a group)\n", maya::core::Glm5Model::multi_mtp_setting(),
                     maya::core::Glm5Model::rows_max());
    if (multi && model.dflash_ready())
        std::fprintf(stderr, "glm dflash: not used with SAGUARO_GLM_SEQS=%d - the several-conversation scheduler decodes "
                             "without it (and without the MTP block, which --dflash leaves unloaded)\n",
                     model.seq_count());
    struct Ctx {   // a context's driver-side state while another one is bound (the bound one's is in the globals)
        std::vector<int32_t> snap_tokens;
        uint64_t snap_img_hash = kNoImages;
        bool snap_in_slot = false;
        std::vector<Ckpt> ckpts;   // its checkpoints (SAGUARO_GLM_MULTI_CKPTS): the model's slots from c * ckpt_max on
        uint64_t used = 0;
        bool busy = false;
        std::chrono::steady_clock::time_point idle_since{};   // its last request's end
    };
    std::vector<Ctx> ctxs((size_t) model.seq_count());
    int cur_ctx = 0;
    uint64_t ctx_clock = 0;
    const auto bind_ctx = [&](int c) {
        if (c != cur_ctx) {
            Ctx& was = ctxs[(size_t) cur_ctx];
            was.snap_tokens = std::move(snap_tokens);
            was.snap_img_hash = snap_img_hash;
            was.snap_in_slot = snap_in_slot;
            was.ckpts = std::move(ckpts);
            Ctx& now = ctxs[(size_t) c];
            snap_tokens = std::move(now.snap_tokens);
            now.snap_tokens.clear();
            snap_img_hash = now.snap_img_hash;
            snap_in_slot = now.snap_in_slot;
            ckpts = std::move(now.ckpts);
            now.ckpts.clear();
            cur_ctx = c;
        }
        ckpt_base = c * ckpt_max;   // (one conversation, or none kept: 0)
        model.seq_bind_all(c);   // (the pipelined decode binds the model's sequences itself)
    };
    each_ctx_snap = [&](const std::function<void(const std::vector<int32_t>&)>& f) {
        for (int c = 0; c < (int) ctxs.size(); ++c)
            if (c != cur_ctx) f(ctxs[(size_t) c].snap_tokens);
    };
    struct Gen {
        int rid = -1, ctx = 0;
        maya::kernels::SamplerParams sp;
        int64_t max_new = 0, produced = 0;
        std::vector<int32_t> prompt, emitted;
        std::vector<int32_t> pen;   // the penalties' history (sp.host_history): the prompt, then every token written
        maya::kernels::LpHold lp;   // logprobs=<k>: its rows (sp.lp)
        // SAGUARO_GLM_ROUTE_DUMP: the route counters at admission and at its first token; shared: another request ran beside
        std::vector<uint32_t> rc0, rc1;
        std::vector<uint32_t> rs1, rs_tl;   // ... its own sequence's decode counts at its first token / the last timeline line
        bool rc_shared = false;
        int conv = -1;   // SAGUARO_GLM_CONV_*: its conversation
        uint64_t pf0[4] = {0, 0, 0, 0};   // ... the preload counters at its admission
        std::chrono::steady_clock::time_point ta;
        bool in_think = false;
        int64_t think_budget = 0;
        int32_t think_end = -1;
        std::vector<int> hist;   // the loop guard's window
        bool in_arg = false;
        int guard = 0;
        int32_t force = -1;      // the next token instead of the sampled one (the thinking budget, the loop guard)
        int32_t pending = -1;    // written, not fed yet
        bool ready = false;      // just admitted: the first sample's logits are on the tail
        bool inflight = false, done = false, pause = false, keep = false;
        bool images = false;     // its prompt had pictures
        bool reading = false;    // its prompt is being read beside the decode (the Reader)
        size_t sa = 0, sys_end = 0, div = 0;   // its state's place (before the reply's opening) and its shared slots'
        std::vector<size_t> stops;             // the Reader's segment ends: the marks, then sa
        const char* finish = "length";
        size_t reuse = 0;
        std::chrono::steady_clock::time_point t0, t1;
        double read_gap_ms = 0.0;   // (the decode gaps inside its read: DONE's prompt_ms is the read's own time)
        uint64_t spec_steps0 = 0, spec_hits0 = 0;
        maya::core::Glm5Model::FastStats st1;
        // SAGUARO_GLM_MULTI_MTP: the NextN block's drafts for the positions after `pending` (made at position
        // draft_at after draft_tok: stale once either moved), and this request's verify statistics
        std::vector<int32_t> drafts;
        int64_t draft_at = -1;
        int32_t draft_tok = -1;
        uint64_t mtp_steps = 0, mtp_drafts = 0, mtp_acc = 0;
    };
    std::vector<std::unique_ptr<Gen>> gens;
    ConvAlloc conv;
    conv.init(model);
    int admitting_conv = -1;   // (the request being admitted's conversation)
    int admitting_ctx = -1;    // ... and its sequence (CONV_SHADOW)
    // SAGUARO_GLM_VRAM_EVICT_LOG (any arm) or the conversation switches: the cumulative VRAM expert hits at each admission,
    // first token and end (the per-phase readout's edges; the n2 phases' boundaries come every 64 steps)
    static const bool ev_log = getenv("SAGUARO_GLM_VRAM_EVICT_LOG") != nullptr || conv.on || conv.open_only;
    const auto ev_line = [&](const char* what, int rid) {
        const auto st = model.fast_stats();
        std::fprintf(stderr, "glm vram evict: event - %s r%d (%zu running) vram_hits=%llu lookups=%llu boundary=0 t=%.3f\n",
                     what, rid, gens.size(), (unsigned long long) st.hits, (unsigned long long) (st.hits + st.misses),
                     std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count());
    };
    const auto conv_running = [&](int extra, int skip_rid, int extra_ctx = -1) {   // (the allocation's running ones)
        std::vector<int> ids;
        const int arid = admitting_rid.load();   // (an in-thread read is not in gens: counted whoever calls)
        if (extra < 0 && arid >= 0 && arid != skip_rid &&
            std::none_of(gens.begin(), gens.end(), [arid](const std::unique_ptr<Gen>& p) { return p->rid == arid; })) {
            extra = admitting_conv;
            extra_ctx = admitting_ctx;
        }
        uint32_t mask = 0;   // (CONV_SHADOW: the running sequences, a reading one included)
        for (const auto& p : gens)
            if (!p->done && p->conv >= 0 && p->rid != skip_rid) {
                ids.push_back(p->conv);
                if (p->ctx >= 0 && p->ctx < 32) mask |= 1u << p->ctx;
            }
        if (extra >= 0) ids.push_back(extra);
        if (extra_ctx >= 0 && extra_ctx < 32) mask |= 1u << extra_ctx;
        conv.set_running(ids);
        if (conv.shadow) maya::core::Glm5Model::shadow_running_set(mask);
        if (conv.shadow || conv.uni) {   // (the prompts being read: admitted, no first token yet; an in-thread one is not in gens)
            int rd = extra >= 0 && extra_ctx >= 0 ? 1 : 0;
            for (const auto& p : gens)
                if (!p->done && p->rid != skip_rid && p->t1 < p->ta) ++rd;
            maya::core::Glm5Model::conv_reading_set(rd);
        }
    };
    const auto conv_decoding = [&] {   // (conversation, its sequence)
        std::vector<std::pair<int, int>> ids;
        for (const auto& p : gens)
            if (!p->done && !p->reading && p->t1 >= p->ta && p->conv >= 0) ids.push_back({p->conv, p->ctx});
        return ids;
    };
    std::deque<Gen*> flight;     // issued, oldest first
    std::deque<int> flight_groups;   // split rows: the sizes of the issued groups, oldest first
    // THE PROMPT BESIDE THE DECODE: a request admitted while others decode reads its prompt's batched part on a thread
    // of its own (prefill_begin's background: the prompt's stream, its own copy of the sequence's layout), and the
    // others go on decoding; its last token and the snapshot follow on this thread (reader_finish).  One at a time:
    // a request that arrives meanwhile waits in `pending`.  SAGUARO_GLM_PREFILL_FG=1: the prompt stops the others, as
    // before (and with --kv-streaming, whose host copies follow the bound sequence)
    struct Reader {
        Gen* g = nullptr;
        std::thread th;
        std::atomic<bool> done{false};
        bool ok = false;
        std::string err;
        std::vector<int32_t> part;
        size_t from = 0, to = 0;   // the prompt's tokens [from, to) this read covers
    };
    std::unique_ptr<Reader> reader;
    static const bool prefill_fg = env_num("SAGUARO_GLM_PREFILL_FG", 0) != 0;
    // an error or the end: the prompt stops at its next chunk, the tail goes back; its request is left to the caller
    const auto reader_abort = [&] {
        if (!reader) return;
        admit_stop.store(true);
        reader->th.join();
        model.prefill_progress = nullptr;
        std::string fe;
        model.prefill_finish(fe);
        admit_stop.store(false);
        admitting_rid.store(-1);
        reader->g->reading = false;
        reader.reset();
    };
    // a request no idle context can hold yet (its prompt needs the large first context, which is busy): it waits
    // here with what its GEN line said, and is admitted when one frees
    struct Pending {
        int rid;
        std::vector<int64_t> ids;
        int64_t max_new;
        maya::kernels::SamplerParams rq;
        int64_t think_budget;
        int32_t think_end;
        std::vector<int64_t> img_pos;
        std::vector<float> img_rows;
        std::vector<std::pair<int32_t, float>> bias;
        int64_t gt = 0;
        double msw = -1.0;   // its miss bar and live Turbo, for the context it gets (PLAN-turbo-live)
        int msf = -1, tdrop = 0, tskip = 0;
    };
    std::deque<Pending> pending;
    // the room a context must leave for the answer (at most what the request asked for): a long prompt goes to a
    // context with this much left after it, and its max tokens are cut to that context
    static const int64_t multi_room = (int64_t) env_num("SAGUARO_GLM_MULTI_ROOM", 16384);
    static const int multi_boundary_every = (int) env_num("SAGUARO_GLM_MULTI_BOUNDARY", 32);
    // SAGUARO_GLM_MULTI_BOUNDARY_READ=N: N steps between boundaries while any prompt is read (each boundary waits on its chunk)
    static const int multi_boundary_read = (int) env_num("SAGUARO_GLM_MULTI_BOUNDARY_READ", 0);
    const auto mb_every = [&] {
        if (multi_boundary_read > 0)
            for (const auto& p : gens)
                if (!p->done && (p->reading || p->t1 < p->ta)) {
                    static uint64_t n_read = 0;   // (proof it engaged: logged at 1, 10, 100, ...)
                    if (++n_read == 1 || n_read == 10 || n_read == 100 || n_read == 1000 || n_read == 10000)
                        std::fprintf(stderr, "glm multi: boundary every %d while a prompt reads (check %llu)\n",
                                     multi_boundary_read, (unsigned long long) n_read);
                    return multi_boundary_read;
                }
        return multi_boundary_every;
    };
    static const bool multi_spec = env_num("SAGUARO_GLM_MULTI_SPEC", 1) != 0;
    static const int split_rows = maya::core::Glm5Model::split_rows_setting();
    // SAGUARO_GLM_EVEN_GROUPS=1: the split's groups cut evenly from all the running requests, two at least, instead of
    // k at a time (maya/core/split_groups.hpp): k 4 no longer makes one group - and idle GPUs - at 4 requests or fewer
    static const bool even_groups = env_num("SAGUARO_GLM_EVEN_GROUPS", 0) != 0;
    // Reuse split-row scheduler buffers; allocating vectors for each decode group defeats part of the batching gain.
    std::vector<Gen*> split_group;
    std::vector<int> split_seqs;
    std::vector<int32_t> split_toks;
    std::vector<maya::kernels::SamplerParams> split_sps;
    if (split_rows > 0) {
        const size_t cap = (size_t) split_rows;
        split_group.reserve(cap);
        split_seqs.reserve(cap);
        split_toks.reserve(cap);
        split_sps.reserve(cap);
    }
    int since_boundary = 0;
    uint64_t multi_n = 0;        // tokens since the engine last had nothing to do (the STAT lines' rate)
    std::chrono::steady_clock::time_point multi_t1;
    maya::core::Glm5Model::FastStats multi_st1;
    uint64_t multi_spec0 = 0, multi_hits0 = 0, multi_mtp_acc = 0, multi_mtp_off = 0;   // (the drafts since then, for the STAT lines)
    const auto input_pending = [&]() -> bool {
        std::lock_guard<std::mutex> lk(in_mu);
        return !in_lines.empty() || quit_req.load();
    };
    const auto in_lines_ready = [&]() -> bool {
        std::lock_guard<std::mutex> lk(in_mu);
        return !in_lines.empty();
    };
    const auto try_line = [&](std::string& out) -> bool {
        std::lock_guard<std::mutex> lk(in_mu);
        if (in_lines.empty()) return false;
        out = std::move(in_lines.front());
        in_lines.pop_front();
        return true;
    };
    // one written token of a request: its T line, the loop guard and the thinking budget (as run_request's
    // on_token), the stop conditions.  false: the request ends here (g.finish says why)
    // SAGUARO_GLM_CONV_*: every TICK_S while decoding - each scheduler round and each written token (a solo stint is one
    // round of up to 4096 tokens): the shares, the open windows' expiry, the allocation
    const auto conv_tick = [&] {
        if (!conv.on && !conv.open_only) return;
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - conv.last_pub).count() < conv.tick_s) return;
        if (conv.on) conv.attribute(conv_decoding());
        const int arid = admitting_rid.load();
        int extra = -1;   // (an in-thread read is not in gens yet)
        if (arid >= 0 && std::none_of(gens.begin(), gens.end(), [arid](const std::unique_ptr<Gen>& p) { return p->rid == arid; }))
            extra = admitting_conv;
        conv_running(extra, -1, extra >= 0 ? admitting_ctx : -1);
        conv.publish("tick", now);
    };
    // SAGUARO_GLM_ROUTE_DUMP: <dir>/timeline.txt, every second while decoding, each running request's own decode routes since
    // its last line: "T <unix ms> <rid> <ctx> <tokens so far> <requests running> layer:expert:n ..."
    const auto route_timeline = [&] {
        static std::chrono::steady_clock::time_point last{};
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last).count() < 1.0) return;
        last = now;
        static std::FILE* f = std::fopen((std::string(getenv("SAGUARO_GLM_ROUTE_DUMP")) + "/timeline.txt").c_str(), "a");
        if (f == nullptr) return;
        const long long unix_ms = (long long) std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::system_clock::now().time_since_epoch()).count();
        size_t running = 0;
        for (const auto& p : gens) running += !p->done;
        std::vector<uint32_t> rs;
        for (auto& p : gens) {
            if (p->done || p->t1 < p->ta || p->rs_tl.empty()) continue;
            maya::core::Glm5Model::route_counts_seq(p->ctx, rs);
            std::fprintf(f, "T %lld %d %d %lld %zu", unix_ms, p->rid, p->ctx, (long long) p->produced, running);
            for (size_t i = 0; i < rs.size(); ++i)
                if (rs[i] != p->rs_tl[i]) std::fprintf(f, " %zu:%zu:%u", i / 512, i % 512, rs[i] - p->rs_tl[i]);
            std::fputc('\n', f);
            p->rs_tl.swap(rs);
        }
        std::fflush(f);
    };
    const auto gen_token = [&](Gen& g, int tok) -> bool {
        if (conv.shadow) maya::core::Glm5Model::shadow_token();   // (every token: the stints' callback too)
        conv_tick();
        if (maya::core::Glm5Model::route_counting()) route_timeline();
        std::printf("T %d %d\n", g.rid, tok);
        if (g.sp.lp != nullptr) {
            const std::string l = maya::kernels::logprob_line(g.sp.lp, g.rid, (int64_t) g.pen.size(), tok);
            if (!l.empty()) std::printf("%s\n", l.c_str());
        }
        ++g.produced;
        ++multi_n;
        g.emitted.push_back((int32_t) tok);
        if (g.sp.host_history != nullptr) g.pen.push_back((int32_t) tok);
        maya::core::progress_at("decode: after token", g.produced);
        maya::core::progress_beat();
        g.hist.push_back(tok);
        const auto looping = [&](int w) -> int {
            const int n = (int) g.hist.size();
            if (w <= 0 || n < w) return 0;
            for (int per = 1; per <= 32 && per < w / 4; ++per) {
                bool rep = true;
                for (int i = n - w + per; i < n && rep; ++i) rep = g.hist[(size_t) i] == g.hist[(size_t) (i - per)];
                if (rep) return per;
            }
            return 0;
        };
        static const int loop_w_think = (int) env_num("SAGUARO_GLM_LOOP_THINK", 256);
        static const int loop_w_answer = (int) env_num("SAGUARO_GLM_LOOP_ANSWER", 1024);
        if (g.in_think && tok != g.think_end) {
            if (const int per = looping(loop_w_think)) {
                g.force = g.think_end;
                g.in_think = false;
                char m[200];
                std::snprintf(m, sizeof m, "thinking was repeating a %d-token pattern (%d tokens in): closed, "
                              "answering", per, (int) g.produced);
                std::fprintf(stderr, "glm loop guard: %s\n", m);
                std::printf("NOTE r%d %s\n", g.rid, m);   // (parity 4: a request's own line carries its rid)
                g.hist.clear();
            }
        } else if (!g.in_think) {
            constexpr int kArgValue = 154849, kArgValueEnd = 154850, kUser = 154827;
            if (tok == kArgValue) g.in_arg = true;
            if (tok == kArgValueEnd) g.in_arg = false;
            if (const int per = looping(loop_w_answer)) {
                char m[240];
                if (g.guard == 0) {
                    g.force = g.in_arg ? kArgValueEnd : kUser;
                    std::snprintf(m, sizeof m, "the answer was repeating a %d-token pattern (%d tokens in): %s", per,
                                  (int) g.produced, g.in_arg ? "the tool call's argument was closed there"
                                                             : "the turn was ended there");
                } else {
                    std::snprintf(m, sizeof m, "the answer kept repeating a %d-token pattern (%d tokens in): ended",
                                  per, (int) g.produced);
                }
                std::fprintf(stderr, "glm loop guard: %s\n", m);
                std::printf("NOTE r%d %s\n", g.rid, m);   // (parity 4: a request's own line carries its rid)
                g.hist.clear();
                if (g.guard++ > 0) {
                    g.finish = "length";
                    return false;
                }
            }
        }
        if (g.in_think && (tok == g.think_end || g.produced >= g.think_budget || wrap_req.load())) {
            if (tok != g.think_end) g.force = g.think_end;
            g.in_think = false;
            g.hist.clear();
        }
        if (multi_n % 8 == 0) {   // the Monitor's live counters, over everything decoded since the engine was idle
            const auto sn = model.fast_stats();
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - multi_t1).count();
            const double n = (double) std::max<uint64_t>(1, multi_n);
            const uint64_t h = sn.hits - multi_st1.hits, m = sn.misses - multi_st1.misses;
            std::printf("STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f ram_fetch=%.3f disk=%.3f promo=%.3f "
                        "vram_used=%lld vram_slots=%lld ram_used=%lld ram_slots=%lld ram_gb=%.1f vram_gb=%.1f "
                        "disk_gb=%.1f draft_acc=%llu draft_off=%llu active=%zu\n",
                        n / std::max(1e-9, el), 1000.0 * el / n, (double) h / (double) std::max<uint64_t>(1, h + m),
                        (double) (sn.ram_hits - multi_st1.ram_hits) / n,
                        (double) (sn.disk_reads - multi_st1.disk_reads) / n,
                        (double) (sn.promotions - multi_st1.promotions) / n, (long long) sn.pool_used,
                        (long long) sn.pool_slots, (long long) sn.ram_used, (long long) sn.ram_slots, sn.ram_gb,
                        sn.pool_gb, std::max(0.0, sn.all_gb - sn.pool_gb - sn.ram_gb),
                        (unsigned long long) (model.spec_hits_ - multi_hits0 + multi_mtp_acc),
                        (unsigned long long) (model.spec_steps_ - multi_spec0 + multi_mtp_off), gens.size());
        }
        for (const int64_t e : eos)
            if (e == tok && stop_on_eos) {
                g.finish = "stop";
                return false;
            }
        {
            std::lock_guard<std::mutex> lk(ctl_mu);
            const auto it = ctl.find(g.rid);
            if (it != ctl.end()) {
                g.pause = it->second >= 2;
                g.keep = it->second == 3;
                ctl.erase(it);
                g.finish = "cancel";
                return false;
            }
        }
        if (quit_req.load() || (stop_req.load() && gens.size() <= 1)) {
            g.finish = "cancel";
            return false;
        }
        return g.produced < g.max_new;
    };
    // the PAUSE snapshots a read's gaps kept back (SAGUARO_GLM_PREFILL_GAPS): taken once no prompt is being read and no
    // request waits for its first token (the tail's logits are its own until then), their contexts idle again
    const auto flush_pauses = [&] {
        for (const auto& d : deferred_pauses) {
            bind_ctx(d.ctx);
            pause_snapshot(d.prompt, d.emitted, d.keep, d.text);
            ctxs[(size_t) d.ctx].busy = false;
        }
        deferred_pauses.clear();
    };
    // SAGUARO_GLM_ROUTE_DUMP=<dir>: a request's routes, <dir>/routes-<unix ms>-r<rid>-c<ctx>.txt - a header, then
    // "P layer expert n" (its prompt read) and "D layer expert n" (its decode; mixed with others' when shared=1);
    // <dir>/slots.txt: each layer's VRAM expert slots
    const auto route_file = [&](Gen& g, std::chrono::steady_clock::time_point t2) {
        static const std::string dir = getenv("SAGUARO_GLM_ROUTE_DUMP");
        static bool slots_done = false;
        if (!slots_done) {
            slots_done = true;
            if (std::FILE* f = std::fopen((dir + "/slots.txt").c_str(), "w")) {
                const std::vector<int> sl = model.layer_slots();
                for (size_t i = 0; i < sl.size(); ++i) std::fprintf(f, "%zu %d\n", i, sl[i]);
                std::fclose(f);
            }
        }
        std::vector<uint32_t> rc2;
        maya::core::Glm5Model::route_counts(rc2);
        const size_t n = rc2.size() / 2;
        if (g.rc0.size() != rc2.size()) return;
        if (g.rc1.size() != rc2.size()) g.rc1 = rc2;   // (never reached its first token: no decode)
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        const long long unix_ms = (long long) std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string path = dir + "/routes-" + std::to_string(unix_ms) + "-r" + std::to_string(g.rid) + "-c" +
                                 std::to_string(g.ctx) + ".txt";
        std::FILE* f = std::fopen(path.c_str(), "w");
        if (f == nullptr) return;
        const bool got1 = g.t1 >= g.ta;
        std::fprintf(f, "# rid=%d ctx=%d prompt=%zu reuse=%zu produced=%lld shared=%d finish=%s read_to_first_ms=%.1f "
                     "decode_ms=%.1f unix_ms=%lld kexp=%d\n", g.rid, g.ctx, g.prompt.size(), g.reuse, (long long) g.produced,
                     g.rc_shared ? 1 : 0, g.finish, got1 ? ms(g.ta, g.t1) : -1.0, got1 ? ms(g.t1, t2) : -1.0, unix_ms,
                     (int) model.geometry().n_expert);
        for (size_t i = 0; i < n; ++i)   // the prompt's (one prompt is read at a time)
            if (g.rc1[i] != g.rc0[i]) std::fprintf(f, "P %zu %zu %u\n", i / 512, i % 512, g.rc1[i] - g.rc0[i]);
        std::vector<uint32_t> rs2;   // the decode's: its own sequence's routes (tagged), not the others' beside it
        maya::core::Glm5Model::route_counts_seq(g.ctx, rs2);
        if (g.rs1.size() != rs2.size()) g.rs1 = rs2;
        for (size_t i = 0; i < rs2.size(); ++i)
            if (rs2[i] != g.rs1[i]) std::fprintf(f, "D %zu %zu %u\n", i / 512, i % 512, rs2[i] - g.rs1[i]);
        std::fclose(f);
        // conservation: every decode route tagged with a sequence (the sum over sequences == the global count)
        // "<unix ms> <rid> <tagged> <global> <others running>": with none running, the counts are read until two reads
        // agree (routes of steps still in flight land first), so such a line must be exact
        size_t others = 0;
        for (const auto& p : gens) others += !p->done && p->rid != g.rid;
        uint64_t tagged = 0, global = 0, t0 = 1, g0 = 1;
        maya::core::Glm5Model::route_totals(tagged, global);
        for (int i = 0; others == 0 && i < 50 && (tagged != t0 || global != g0); ++i) {
            t0 = tagged;
            g0 = global;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            maya::core::Glm5Model::route_totals(tagged, global);
        }
        if (std::FILE* fc = std::fopen((dir + "/conservation.txt").c_str(), "a")) {
            std::fprintf(fc, "%lld %d %llu %llu %zu\n", unix_ms, g.rid, (unsigned long long) tagged, (unsigned long long) global,
                         others);
            std::fclose(fc);
        }
    };
    // a request's end: its DONE line (the engine's counters over its lifetime are shared with the others running
    // meanwhile), the paused state kept, its context idle again
    const auto gen_finish = [&](Gen& g) {
        const auto t2 = std::chrono::steady_clock::now();
        if (maya::core::Glm5Model::route_counting()) route_file(g, t2);
        if (ev_log) ev_line("end", g.rid);
        if (g.conv >= 0 && conv.on && maya::core::Glm5Model::conv_preload_on()) {
            uint64_t pf[4];
            maya::core::Glm5Model::conv_prefetch_counts(pf);
            std::fprintf(stderr, "glm conv: r%d end: preload since admission up %llu, down %llu, landed %llu (%.2f GB)\n", g.rid,
                         (unsigned long long) (pf[0] - g.pf0[0]), (unsigned long long) (pf[1] - g.pf0[1]),
                         (unsigned long long) (pf[2] - g.pf0[2]), (double) (pf[3] - g.pf0[3]) / 1e9);
        }
        if (g.conv >= 0) {
            if (conv.on) conv.attribute(conv_decoding());
            if (conv.shadow) {   // (its lru set freezes as it was: the target when it resumes)
                if (auto* cv0 = conv.find(g.conv)) maya::core::Glm5Model::shadow_seq_save(g.ctx, cv0->frozen);
                maya::core::Glm5Model::shadow_event();
            }
            conv.end(g.conv, g.emitted, t2);
            g.conv = -1;
            conv_running(-1, g.rid);
            conv.publish("end", t2);
        }
        // its read's own time: the decode gaps taken inside it (other requests' tokens) are not read time
        const double prompt_ms = std::max(0.0, std::chrono::duration<double, std::milli>(g.t1 - g.t0).count()
                                                         - g.read_gap_ms);
        const double decode_ms = std::chrono::duration<double, std::milli>(t2 - g.t1).count();
        const auto st2 = model.fast_stats();
        const uint64_t d_hits = st2.hits - g.st1.hits, d_miss = st2.misses - g.st1.misses;
        std::printf("DONE %d %lld %lld %.1f %.1f %s %llu %llu %zu %llu %llu\n", g.rid, (long long) g.produced,
                    (long long) g.prompt.size(), prompt_ms, decode_ms, g.finish,
                    (unsigned long long) (model.spec_hits_ - g.spec_hits0 + g.mtp_acc),
                    (unsigned long long) (model.spec_steps_ - g.spec_steps0 + g.mtp_drafts), g.reuse,
                    (unsigned long long) d_hits, (unsigned long long) (d_hits + d_miss));
        if (mtp_rows && g.mtp_steps > 0)
            std::fprintf(stderr, "glm multi mtp: request %d: %llu of %llu drafts accepted (%.1f%%) in %llu verify steps, "
                                 "%.2f tokens a step\n", g.rid, (unsigned long long) g.mtp_acc,
                         (unsigned long long) g.mtp_drafts,
                         100.0 * (double) g.mtp_acc / (double) std::max<uint64_t>(1, g.mtp_drafts),
                         (unsigned long long) g.mtp_steps,
                         (double) (g.mtp_steps + g.mtp_acc) / (double) g.mtp_steps);
        bool deferred = false;
        if (g.pause && std::strcmp(g.finish, "cancel") == 0) {
            if (in_read_gap) {   // (another context's prompt is being read and stays bound: the snapshot after it)
                deferred_pauses.push_back({g.ctx, g.prompt, g.emitted, g.keep, !g.images});
                deferred = true;
            } else {
                bind_ctx(g.ctx);
                pause_snapshot(g.prompt, g.emitted, g.keep, !g.images);
            }
        }
        if (g.produced > 0)
            std::fprintf(stderr, "glm multi: request %d done (%s): %lld tokens in %.1f s (%.1f tok/s), %zu running\n",
                         g.rid, g.finish, (long long) g.produced, decode_ms / 1000.0,
                         1000.0 * (double) g.produced / std::max(1.0, decode_ms), gens.size() - 1);
        {
            std::lock_guard<std::mutex> lk(ctl_mu);
            ctl.erase(g.rid);
        }
        ctxs[(size_t) g.ctx].busy = deferred;   // (a deferred snapshot: busy until flush_pauses)
        ctxs[(size_t) g.ctx].used = ++ctx_clock;
        ctxs[(size_t) g.ctx].idle_since = std::chrono::steady_clock::now();
        model.set_miss_skip(g.ctx, -1.0, -1);   // (PLAN-turbo-live: its bars end with it - the next request sets its own)
        model.set_turbo(g.ctx, 0, 0);
        g.done = true;
    };
    const auto gen_sweep = [&] {
        gens.erase(std::remove_if(gens.begin(), gens.end(), [](const std::unique_ptr<Gen>& p) { return p->done; }),
                   gens.end());
        // elastic: what the contexts' budget holds beyond its base (a released context's surplus, a finished request's
        // room) goes back to the experts now, not only when the engine is idle (a cheap check when there is none)
        if (model.elastic()) model.pool_take_back_all();
        if (gens.empty()) {
            // (a prompt being read between decode steps keeps a STOP for all: its read ends too)
            if (admitting_rid.load() >= 0 && stop_req.load()) admit_stop.store(true);
            stop_req.store(false);
            // this machine's expert usage, for the next start's warm-up (not inside a prompt's read)
            if (model.fast() && !in_read_gap) model.save_usage();
            if (!deferred_pauses.empty() && !in_read_gap && admitting_rid.load() < 0) flush_pauses();
        }
    };
    // the token a request takes after one came back: forced, else as sampled
    const auto gen_took = [&](Gen& g, int tok) {
        if (g.force >= 0) {
            tok = g.force;
            g.force = -1;
        }
        if (!gen_token(g, tok)) gen_finish(g);
        else g.pending = tok;
    };
    // SAGUARO_GLM_MULTI_MTP: the groups of verify windows in flight, oldest first - each past its front (every part but
    // the last), `back` once its last part and head are queued too
    struct WinGroup {
        std::vector<Gen*> g;
        std::vector<int> seqs, counts;
        std::vector<int32_t> toks;
        std::vector<maya::kernels::SamplerParams> sps;
        bool back = false;
    };
    std::deque<WinGroup> wflight;
    const auto win_back = [&](WinGroup& w) -> bool {
        if (w.back) return true;
        w.back = true;
        return model.mtpw_back(w.seqs.data(), w.counts.data(), (int) w.seqs.size(), w.toks.data(), w.sps.data(), err);
    };
    // the oldest group back: its picks; each window's rows written up to the first draft that differs from what was
    // written before it (T lines, the stop conditions, a forced token - gen_took's rules); the commit; then the next
    // windows' drafts - the next group's last part is queued between the drafts and their wait, so the last GPU goes
    // on with it; the requests that ended, ended
    const auto win_take = [&]() -> bool {
        // phase timing of a rows path's round (SAGUARO_GLM_MTPW_TRACE=1): the wait for the picks, the accept + the
        // commit, the next round's drafts, the whole wall
        static const bool trace = getenv("SAGUARO_GLM_MTPW_TRACE") != nullptr &&
                                  std::atoi(getenv("SAGUARO_GLM_MTPW_TRACE")) != 0;
        const auto t_round0 = std::chrono::steady_clock::now();
        WinGroup w = std::move(wflight.front());
        wflight.pop_front();
        const int n = (int) w.seqs.size();
        int rows = 0;
        for (const int c : w.counts) rows += c;
        std::vector<int32_t> picks((size_t) rows);
        if (!win_back(w) || !model.mtpw_picks(w.seqs.data(), w.counts.data(), n, picks.data(), err)) return false;
        const double t_picks = trace ? std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() - t_round0).count() : 0.0;
        std::vector<int> keep((size_t) n, 1), nd((size_t) n, 0);
        std::vector<int32_t> written, last((size_t) n, -1);
        std::vector<char> ended((size_t) n, 0);
        const int D = maya::core::Glm5Model::multi_mtp_setting();
        for (int i = 0, row = 0; i < n; row += w.counts[(size_t) i], ++i) {
            Gen& g = *w.g[(size_t) i];
            g.inflight = false;
            std::vector<int32_t> wr;
            keep[(size_t) i] = maya::core::mtp_accept(
                w.toks.data() + row, picks.data() + row, w.counts[(size_t) i],
                [&](int, int32_t pick, int32_t& out) {
                    int32_t tok = pick;
                    if (g.force >= 0) {   // (gen_took's: the forced token instead of the pick)
                        tok = g.force;
                        g.force = -1;
                    }
                    out = tok;
                    wr.push_back(tok);
                    g.sp.counter += 1;
                    if (!gen_token(g, tok)) {
                        ended[(size_t) i] = 1;
                        return maya::core::MtpTake::stop;
                    }
                    return g.force >= 0 ? maya::core::MtpTake::forced : maya::core::MtpTake::more;
                });
            const int k = keep[(size_t) i];
            g.mtp_steps += 1;
            g.mtp_drafts += (uint64_t) (w.counts[(size_t) i] - 1);
            g.mtp_acc += (uint64_t) (k - 1);
            multi_mtp_off += (uint64_t) (w.counts[(size_t) i] - 1);
            multi_mtp_acc += (uint64_t) (k - 1);
            for (int t = 0; t + 1 < k; ++t) written.push_back(wr[(size_t) t]);   // (the tokens at p0 + 1 .. p0 + k - 1)
            last[(size_t) i] = wr[(size_t) k - 1];
            g.drafts.clear();
            g.draft_at = -1;
            if (!ended[(size_t) i]) {
                g.pending = last[(size_t) i];
                // the next window's drafts: no more than its tokens left after the pending one and its context's
                // room (draft j's NextN cache entry is at the window's start + j - 1; the position is still this
                // window's start + its rows until the commit below)
                nd[(size_t) i] = (int) std::max<int64_t>(
                    0, std::min<int64_t>({(int64_t) D, g.max_new - g.produced - 1,
                                          model.seq_max_ctx(g.ctx) - 2 -
                                              (model.seq_pos(g.ctx) - w.counts[(size_t) i] + k)}));
            }
        }
        if (!model.mtpw_commit(w.seqs.data(), w.counts.data(), keep.data(), n, err)) return false;
        const double t_commit = trace ? std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - t_round0).count() : 0.0;
        if (!model.mtpw_draft_queue(w.seqs.data(), keep.data(), n, written.data(), last.data(), nd.data(), err))
            return false;
        if (!wflight.empty() && !win_back(wflight.front())) return false;
        {
            std::vector<int32_t> out((size_t) n * (size_t) std::max(1, D), -1);
            if (!model.mtpw_draft_take(out.data(), err)) return false;
            for (int i = 0; i < n; ++i) {
                Gen& g = *w.g[(size_t) i];
                if (ended[(size_t) i]) continue;
                for (int j = 0; j < D && out[(size_t) i * D + (size_t) j] >= 0; ++j)
                    g.drafts.push_back(out[(size_t) i * D + (size_t) j]);
                g.draft_at = model.seq_pos(g.ctx);
                g.draft_tok = g.pending;
            }
        }
        for (int i = 0; i < n; ++i)
            if (ended[(size_t) i]) gen_finish(*w.g[(size_t) i]);
        since_boundary += n;
        if (trace) {
            const double now = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_round0).count();
            int got = 0;
            for (const int k : keep) got += k;
            std::fprintf(stderr, "glm mtpw trace: rows %d (groups %d) -> committed %d in %.2f ms | picks+back %.2f | "
                                 "accept+commit %.2f | drafts %.2f\n",
                         rows, n, got, now, t_picks, t_commit - t_picks, now - t_commit);
        }
        return true;
    };
    // every issued token back (before a prompt, a boundary or the speculative path)
    const auto multi_drain = [&]() -> bool {
        while (!wflight.empty())
            if (!win_take()) return false;
        while (!flight.empty()) {
            Gen& g = *flight.front();
            flight.pop_front();
            const int tok = model.multi_take(g.ctx, err);
            g.inflight = false;
            if (tok < 0) return false;
            gen_took(g, tok);
        }
        flight_groups.clear();
        return true;
    };
    // a just-admitted request's first token, while the tail's logits are still its own
    const auto resolve_ready = [&]() -> bool {
        for (auto& gp : gens) {
            Gen& g = *gp;
            if (!g.ready || g.done) continue;
            bind_ctx(g.ctx);
            const int tok = model.sample_token(g.sp, err);
            g.ready = false;
            if (tok < 0) return false;
            g.sp.counter += 1;
            gen_took(g, model.forced(tok));
        }
        return true;
    };
    // an engine error in the middle: every running request gets it, their sequences are reset (an idle context's
    // conversation was not written meanwhile and stays reusable)
    uint64_t multi_fails = 0;      // (a prompt read between decode steps sees one happen: SAGUARO_GLM_PREFILL_INTERLEAVE)
    std::string multi_fail_why;
    const auto multi_fail = [&](const std::string& why) {
        reader_abort();
        ++multi_fails;
        multi_fail_why = why;
        std::vector<int> failed;
        for (auto& gp : gens) {
            if (gp->done) continue;
            if (why.find("non-finite") != std::string::npos)   // the slots it may have come from (drop_if_poisoned)
                for (size_t i = slots.size(); i-- > 0;) {
                    const auto& t = slots[i].tokens;
                    if (t.size() <= gp->prompt.size() && std::equal(t.begin(), t.end(), gp->prompt.begin()))
                        slot_drop(i);
                }
            std::printf("ERR %d %s\n", gp->rid, why.c_str());
            ctxs[(size_t) gp->ctx].busy = false;
            failed.push_back(gp->ctx);
            gp->done = true;
        }
        flight.clear();
        flight_groups.clear();
        wflight.clear();
        if (in_read_gap) {   // (a prompt is being read: its call ends first, then gap_reset resets them all)
            gap_failed = true;
            prompt_between_err = "the engine failed while other requests decoded during this prompt's read: " + why;
            gen_sweep();
            return;
        }
        for (const auto& d : deferred_pauses) {   // (a pause not kept yet: its state goes with the failed ones)
            ctxs[(size_t) d.ctx].busy = false;
            failed.push_back(d.ctx);
        }
        deferred_pauses.clear();
        for (const int c : failed) {
            bind_ctx(c);
            snap_tokens.clear();
            ckpts.clear();   // (the caches they stood on are reset too)
            model.set_miss_skip(c, -1.0, -1);   // (PLAN-turbo-live: with the sequences, their request bars go too)
            model.set_turbo(c, 0, 0);
            model.reset();
        }
        gen_sweep();
    };
    // the contexts' reset a failed gap left (SAGUARO_GLM_PREFILL_GAPS; multi_fail), once the read has ended
    gap_reset = [&] {
        if (!gap_failed) return;
        gap_failed = false;
        for (const auto& d : deferred_pauses) ctxs[(size_t) d.ctx].busy = false;   // (their states go with the rest)
        deferred_pauses.clear();
        for (int c = 0; c < (int) ctxs.size(); ++c) {
            bind_ctx(c);
            snap_tokens.clear();
            ckpts.clear();   // (the caches they stood on are reset too)
            model.set_miss_skip(c, -1.0, -1);   // (PLAN-turbo-live: with the sequences, their request bars go too)
            model.set_turbo(c, 0, 0);
            model.reset();
        }
    };
    // the decode steps between two pieces of a prompt being read in context read_ctx (SAGUARO_GLM_PREFILL_INTERLEAVE;
    // set below, after multi_step): prompt_between's 0 / 1 / 2
    std::function<int(int)> interleave_round;
    // one decode gap of a prompt being read (SAGUARO_GLM_PREFILL_GAPS; set below, after multi_step): 0 / 1 / 2 as
    // prompt_between's; gap_mark / gap_count / gap_tok / gap_ms: the read's own, declared by its PP clock above
    std::function<int()> read_gap;
    // the last gap's decode round (ms; 0: none yet) - a boundary's gap is due once its budget is half of one
    // (prefill_gap_due; SAGUARO_GLM_PREFILL_GAP_CREDIT=0: every boundary, as before)
    double gap_round_ms = 0.0;
    static const bool gap_credit = env_num("SAGUARO_GLM_PREFILL_GAP_CREDIT", 1) != 0;
    // a request whose prompt has been read (it is in gens): its first token, or its end if it was stopped meanwhile
    const auto gen_admitted = [&](Gen& g) {
        // a STOP <rid> during its prompt; a bare STOP stays set and also ends the running requests
        if (admit_stop.exchange(false) || stop_req.load()) g.finish = "cancel";
        g.t1 = std::chrono::steady_clock::now();
        g.read_gap_ms = gap_ms;   // (the read is over: its gaps are this request's, DONE takes them out)
        if (maya::core::Glm5Model::route_counting()) {
            maya::core::Glm5Model::route_counts(g.rc1);
            maya::core::Glm5Model::route_counts_seq(g.ctx, g.rs1);
            g.rs_tl = g.rs1;
        }
        if (ev_log) ev_line("first", g.rid);
        if (conv.shadow || conv.uni) {   // (its read is over)
            int rd = 0;
            for (const auto& p : gens)
                if (!p->done && p.get() != &g && p->t1 < p->ta) ++rd;
            maya::core::Glm5Model::conv_reading_set(rd);
        }
        if (g.conv >= 0) conv.seq_mark(g.ctx);   // (CONV_UNION: its sequence's counts from its first token on are its own)
        if (conv.on && maya::core::Glm5Model::conv_preload_on()) {
            uint64_t pf[4];
            maya::core::Glm5Model::conv_prefetch_counts(pf);
            std::fprintf(stderr, "glm conv: r%d first token after %.2f s: preload since admission up %llu, down %llu, landed "
                         "%llu (%.2f GB)\n", g.rid, std::chrono::duration<double>(g.t1 - g.ta).count(),
                         (unsigned long long) (pf[0] - g.pf0[0]), (unsigned long long) (pf[1] - g.pf0[1]),
                         (unsigned long long) (pf[2] - g.pf0[2]), (double) (pf[3] - g.pf0[3]) / 1e9);
        }
        g.st1 = model.fast_stats();
        g.spec_steps0 = model.spec_steps_;
        g.spec_hits0 = model.spec_hits_;
        if (gens.size() == 1) {
            multi_n = 0;
            multi_t1 = g.t1;
            multi_st1 = g.st1;
            multi_spec0 = g.spec_steps0;
            multi_hits0 = g.spec_hits0;
            multi_mtp_acc = multi_mtp_off = 0;
        }
        if (std::strcmp(g.finish, "cancel") == 0) {
            gen_finish(g);
            gen_sweep();
            return;
        }
        g.ready = true;
        if (gens.size() > 1 && !resolve_ready()) multi_fail(err);   // the others decode: its first token now
        if (!deferred_pauses.empty() && gens.size() > 1) flush_pauses();   // (its first token taken: the tail is free)
        gen_sweep();
    };
    // a segment [from, to) of gp's prompt on the Reader (prefill_begin's background); false: not started (err says why)
    const auto reader_start = [&](Gen* gp, size_t from, size_t to) -> bool {
        std::vector<int32_t> part(gp->prompt.begin() + (long) from, gp->prompt.begin() + (long) to);
        if (!model.prefill_begin((int64_t) part.size(), gp->prompt[to], true, err)) return false;
        reader = std::make_unique<Reader>();
        Reader* R = reader.get();
        R->part = std::move(part);
        R->g = gp;
        R->from = from;
        R->to = to;
        gp->reading = true;
        const std::string rtag = std::to_string(gp->rid) + " ";
        const auto rt0 = gp->t0;
        const size_t rreuse = gp->reuse, rfrom = from, rtotal = gp->prompt.size();
        model.prefill_progress = [&, rtag, rt0, rreuse, rfrom, rtotal](int64_t done, int64_t total) {
            (void) total;
            if (o.serve) {
                const double ms = read_ms_since(rt0);   // (the read's own time: the gaps this read took are not in it)
                std::printf("PP %s%zu %zu %.1f %.1f %zu\n", rtag.c_str(), rfrom + (size_t) done, rtotal, ms,
                            (double) (rfrom - rreuse + (size_t) done) / (ms / 1000.0 + 1e-9), rreuse);
                std::fflush(stdout);
            }
            return !stop_req.load() && !admit_stop.load();
        };
        R->th = std::thread([&model, R] {
            try {
                R->ok = model.prefill_run(R->part, R->err);
            } catch (const std::exception& e) {   // (a failure in the prompt path is the request's error)
                R->err = e.what();
                R->ok = false;
            }
            R->done.store(true, std::memory_order_release);
        });
        return true;
    };
    // the Reader is done: the decode drains (the token path and the snapshot use the shared streams and buffers), the
    // tail goes back; at a mark its shared slot and the next segment, else the prompt's last token and the request's first
    const auto reader_finish = [&] {
        Reader* R = reader.get();
        R->th.join();
        model.prefill_progress = nullptr;
        Gen& g = *R->g;
        const bool drained = multi_drain();
        std::string fe;
        const bool fok = model.prefill_finish(fe);
        admitting_rid.store(-1);
        std::string e = !R->ok ? R->err : !fok ? fe : std::string();
        const size_t r_to = R->to;
        g.reading = false;
        reader.reset();
        if (!drained) {
            multi_fail(err);
            return;
        }
        bind_ctx(g.ctx);
        tag = std::to_string(g.rid) + " ";
        int rc = 0;
        if (e == "cancelled") {   // (it keeps what it read: every part stopped at the same chunk boundary)
            const int64_t kept = model.position();
            if (model.fast() && kept > (int64_t) g.reuse && kept < (int64_t) g.prompt.size() && model.snapshot_save()) {
                snap_tokens.assign(g.prompt.begin(), g.prompt.begin() + kept);
                snap_img_hash = img_hash(snap_tokens.size());
                snap_in_slot = false;
                std::fprintf(stderr, "glm slots: a cancelled prompt keeps the %lld tokens it read (%zu new)\n",
                             (long long) kept, (size_t) kept - g.reuse);
            } else if (!(kept == (int64_t) g.reuse && g.reuse > 0)) {
                snap_tokens.clear();
                model.reset();
            }
            g.finish = "cancel";
            std::fprintf(stderr, "glm multi: request %d's prompt cancelled while it was read\n", g.rid);
        } else if (!e.empty()) {
            snap_tokens.clear();
            drop_if_poisoned(g.prompt, e);
            model.reset();
            std::printf("ERR %s%s\n", tag.c_str(), e.c_str());
            rc = 1;
        } else {
            size_t from = r_to;   // the token path's start, if the read goes no further on the Reader
            if (r_to < g.sa) {    // a mark: its shared slot, then the next segment on the Reader
                if (r_to == g.sys_end) slot_prefix(g.prompt, r_to);
                else if (r_to == g.div)
                    slot_prefix(g.prompt, r_to, " (where it parts from a kept conversation, for the next one that does)");
                size_t next = g.sa;
                for (const size_t st : g.stops)
                    if (st > r_to) { next = st; break; }
                admitting_rid.store(g.rid);
                if (reader_start(&g, r_to, next)) {
                    tag.clear();
                    return;
                }
                admitting_rid.store(-1);
                if (!err.empty()) {
                    snap_tokens.clear();
                    std::printf("ERR %s%s\n", tag.c_str(), err.c_str());
                    rc = 1;
                }
            } else {
                from = g.sa;
                if (o.serve) {
                    const double ms = read_ms_since(g.t0);   // (the read's own time: its gaps are not in it)
                    std::printf("PP %s%zu %zu %.1f %.1f %zu\n", tag.c_str(), g.sa, g.prompt.size(), ms,
                                (double) (g.sa - g.reuse) / (ms / 1000.0 + 1e-9), g.reuse);
                }
            }
            if (rc == 0) rc = prep_tail(g.prompt, from, g.reuse, g.finish, g.t0, g.sa);
        }
        tag.clear();
        if (rc != 0) {
            ctxs[(size_t) g.ctx].busy = false;
            g.done = true;
            if (conv.shadow || conv.uni) conv_running(-1, g.rid);   // (its read is over)
            gen_sweep();
            return;
        }
        gen_admitted(g);
    };
    // ELASTIC CONTEXTS (SAGUARO_GLM_ELASTIC): the contexts share a budget per card for their context-sized caches.
    // When a request needs more than is left, idle contexts give theirs back, least recently used first - the
    // conversation goes to a slot first (when slots are on), so its next turn reloads it instead of reading it again
    const auto release_idle = [&](int keep, const std::function<bool()>& enough, double min_idle_s = 0) -> bool {
        if (!model.elastic()) return enough();
        const auto now = std::chrono::steady_clock::now();
        while (!enough()) {
            int v = -1;
            for (int c = 0; c < (int) ctxs.size(); ++c)
                if (c != keep && !ctxs[(size_t) c].busy && model.seq_holds(c) &&
                    std::chrono::duration<double>(now - ctxs[(size_t) c].idle_since).count() >= min_idle_s &&
                    (v < 0 || ctxs[(size_t) c].used < ctxs[(size_t) v].used))
                    v = c;
            if (v < 0) return false;
            bind_ctx(v);
            slot_put(" (its context's memory went to another conversation)");
            snap_tokens.clear();
            ckpts.clear();   // (MULTI_CKPTS: they stand on the DSA rows the release hands back)
            model.seq_release(v, true);
            std::fprintf(stderr, "glm elastic: context %d released for another conversation\n", v);
        }
        return true;
    };
    // context c to back `tokens` positions: within the budget, else contexts idle for SAGUARO_GLM_ELASTIC_HOLD_S (120)
    // give theirs back, else expert slots give way (the pipeline drained first), else any idle context (a conversation
    // that just ended is likely to come back: to a slot and back costs seconds)
    static const double hold_s = env_num("SAGUARO_GLM_ELASTIC_HOLD_S", 120);
    const auto ctx_reserve = [&](int c, int64_t tokens) -> bool {
        if (!model.elastic()) return true;
        tokens = std::min<int64_t>(tokens, model.seq_max_ctx(c));
        const auto enough = [&] { return model.seq_can_grow(c, tokens); };
        bool ok = release_idle(c, enough, hold_s);
        std::string re;
        if (!ok && multi_drain()) ok = model.seq_reserve_all(c, tokens, re);
        if (!ok) ok = release_idle(c, enough);
        if (!ok) {
            if (!re.empty()) std::fprintf(stderr, "%s\n", re.c_str());
            return false;
        }
        // (a slot no conversation was in: its state backed now, out of the room just found)
        std::string ae;
        if (!model.seq_active(c) && !model.seq_grow_all(c, 1, ae)) {
            std::fprintf(stderr, "%s\n", ae.c_str());
            return false;
        }
        return true;
    };
    // the engine idle: idle contexts beyond the SAGUARO_GLM_ELASTIC_KEEP most recent (1) give their memory back (to a
    // slot first), and what the budget holds beyond its base goes back to the experts - a conversation alone gets the
    // whole pool again
    const auto idle_trim = [&] {
        static const int keep_n = (int) env_num("SAGUARO_GLM_ELASTIC_KEEP", 1);
        std::vector<int> order;
        for (int c = 0; c < (int) ctxs.size(); ++c)
            if (!ctxs[(size_t) c].busy && model.seq_holds(c)) order.push_back(c);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return ctxs[(size_t) a].used > ctxs[(size_t) b].used; });
        for (size_t i = (size_t) std::max(0, keep_n); i < order.size(); ++i) {
            bind_ctx(order[i]);
            slot_put(" (an idle context's memory went back to the experts)");
            snap_tokens.clear();
            ckpts.clear();   // (MULTI_CKPTS: they stand on the DSA rows the release hands back)
            model.seq_release(order[i], true);
        }
        model.pool_take_back_all();
    };
    // elastic: the next position of a running request must fit the budget - idle contexts give way, else it ends here
    const auto elastic_room = [&](Gen& g) -> bool {
        if (!model.elastic() || model.seq_can_grow(g.ctx, model.seq_pos(g.ctx) + 2) ||
            ctx_reserve(g.ctx, model.seq_pos(g.ctx) + 2 + 4096))
            return true;
        std::fprintf(stderr, "glm elastic: request %d ends - no context memory left\n", g.rid);
        gen_finish(g);
        return false;
    };
    const auto gen_admit = [&](int rid, const std::vector<int64_t>& ids, int64_t max_new,
                               const maya::kernels::SamplerParams& rq) {
        if (reader) {   // one prompt read at a time: this one waits for it
            pending.push_back({rid, ids, max_new, rq, req_think_budget, req_think_end, cur_img_pos, cur_img_rows,
                               req_bias, 0, req_msw, req_msf, req_tdrop, req_tskip});
            return;
        }
        if (!multi_drain() || !resolve_ready()) {
            const std::string why = err;
            multi_fail(why);
            std::printf("ERR %d %s\n", rid, why.c_str());   // (this request too: its client would wait forever)
            return;
        }
        if (!deferred_pauses.empty()) flush_pauses();   // (a PAUSE in the last read's gaps: its state, kept first)
        auto g = std::make_unique<Gen>();
        g->rid = rid;
        g->sp = rq;
        g->max_new = max_new;
        g->in_think = req_think_budget > 0 && req_think_end >= 0;
        g->think_budget = req_think_budget;
        g->think_end = req_think_end;
        for (const int64_t t : ids) {
            if (t < 0 || t >= n_vocab) {
                std::printf("ERR %d token %lld out of range [0, %lld)\n", rid, (long long) t, (long long) n_vocab);
                return;
            }
            g->prompt.push_back((int32_t) t);
        }
        if (g->prompt.empty() || max_new <= 0) {
            std::printf("ERR %d empty prompt or max_new\n", rid);
            return;
        }
        if (g->sp.penalty_last_n > 0) {   // its penalties' history (#273; the Gen lives behind a unique_ptr: it stays put)
            g->pen = g->prompt;
            g->sp.host_history = &g->pen;
        }
        if (g->sp.lp_k >= 0) {
            std::string le;
            g->lp.reset(g->sp.lp = maya::kernels::lp_sink_get(g->sp.lp_k, le));
            if (g->sp.lp == nullptr) std::fprintf(stderr, "glm multi: request %d: %s\n", rid, le.c_str());
        }
        // the context: an idle one that holds the prompt and the room after it, and whose snapshot - or one of its
        // conversation checkpoints (SAGUARO_GLM_MULTI_CKPTS: an edited last message, a cancelled answer sent again
        // without it, an older turn rewritten) - this prompt extends (the longest), else the least recently used of
        // those.  prep_prompt then restores the one it found, as with one conversation.
        const int64_t want = (int64_t) g->prompt.size() + std::min<int64_t>(max_new, multi_room) + 8;
        const auto fits = [&](int c) { return model.seq_max_ctx(c) >= want; };
        int best = -1;
        size_t blen = 0;
        for (int c = 0; c < (int) ctxs.size(); ++c) {
            if (ctxs[(size_t) c].busy || !fits(c)) continue;
            const auto& st = c == cur_ctx ? snap_tokens : ctxs[(size_t) c].snap_tokens;
            const uint64_t hh = c == cur_ctx ? snap_img_hash : ctxs[(size_t) c].snap_img_hash;
            size_t len = 0;
            if (!st.empty() && st.size() < g->prompt.size() && std::equal(st.begin(), st.end(), g->prompt.begin()) &&
                img_hash(st.size()) == hh)
                len = st.size();
            if (ckpt_max > 0 && cur_img_pos.empty()) {   // (checkpoints are kept of text prompts only)
                const std::vector<Ckpt>& cl = c == cur_ctx ? ckpts : ctxs[(size_t) c].ckpts;
                for (const Ckpt& k : cl)
                    if (k.tokens.size() > len && k.tokens.size() < g->prompt.size() &&
                        std::equal(k.tokens.begin(), k.tokens.end(), g->prompt.begin()))
                        len = k.tokens.size();
            }
            if (len > blen) {
                best = c;
                blen = len;
            }
        }
        if (best < 0)
            for (int c = 0; c < (int) ctxs.size(); ++c)
                if (!ctxs[(size_t) c].busy && fits(c) &&
                    (best < 0 || ctxs[(size_t) c].used < ctxs[(size_t) best].used))
                    best = c;
        if (best < 0) {
            bool ever = false;
            for (int c = 0; c < (int) ctxs.size(); ++c) ever = ever || fits(c);
            if (!ever) {
                std::printf("ERR %d prompt (%zu tokens) + room for the answer exceeds every context\n", rid,
                            g->prompt.size());
                return;
            }
            pending.push_back({rid, ids, max_new, rq, req_think_budget, req_think_end, cur_img_pos, cur_img_rows,
                               req_bias, req_gt, req_msw, req_msf, req_tdrop, req_tskip});
            std::fprintf(stderr, "glm multi: request %d waits for a context that holds %lld tokens\n", rid,
                         (long long) want);
            return;
        }
        if (model.elastic() && !ctx_reserve(best, want)) {
            if (gens.empty()) {
                std::printf("ERR %d prompt (%zu tokens) + room for the answer exceeds the contexts' memory\n", rid,
                            g->prompt.size());
                return;
            }
            pending.push_back({rid, ids, max_new, rq, req_think_budget, req_think_end, cur_img_pos, cur_img_rows,
                               req_bias, 0, req_msw, req_msf, req_tdrop, req_tskip});
            std::fprintf(stderr, "glm multi: request %d waits for context memory (%lld tokens)\n", rid, (long long) want);
            return;
        }
        g->max_new = std::min<int64_t>(max_new, model.seq_max_ctx(best) - (int64_t) g->prompt.size() - 8);
        {   // its logit_bias: the table of the context it runs in (no earlier request's sample reads it any more)
            std::string be;
            g->sp.logit_bias = model.logit_bias_table(best, req_bias, be);
            if (!be.empty()) {
                std::printf("ERR %d %s\n", rid, be.c_str());
                return;
            }
        }
        if (gens.empty()) stop_req.store(false);
        {
            std::lock_guard<std::mutex> lk(ctl_mu);
            ctl.erase(rid);
        }
        gap_failed = false;
        bind_ctx(best);
        // PLAN-turbo-live: the request's bars belong to the context it got - its prompt read's and its decode
        // tokens' routes take them, and the chats decoding beside it keep their own (off: the engine's)
        model.set_miss_skip(best, req_msw, req_msf);
        model.set_turbo(best, req_tdrop, req_tskip);
        model.set_image_rows(cur_img_pos, cur_img_rows);
        g->images = !cur_img_pos.empty();
        model.force_next(-1);
        ctxs[(size_t) best].busy = true;
        g->ctx = best;
        tag = std::to_string(rid) + " ";
        // SAGUARO_GLM_PREFILL_GAPS: the running requests decode at the read's chunk boundaries and between its calls
        // (the split's batched rows: they never bind a context; text prompts only, as interleave)
        const bool use_gaps = read_gaps.percent > 0 && cur_img_pos.empty() && split_rows > 0 && model.split_rows_ok();
        if (read_gaps.percent > 0 && !use_gaps && !gens.empty()) {
            static bool told = false;
            if (!cur_img_pos.empty()) {
                std::fprintf(stderr, "glm multi: request %d's prompt has images: read whole, no decode gaps "
                                     "(SAGUARO_GLM_PREFILL_GAPS is for text prompts)\n", rid);
            } else if (!told) {
                told = true;
                std::fprintf(stderr, "glm multi: SAGUARO_GLM_PREFILL_GAPS needs a split with SAGUARO_GLM_SPLIT_ROWS: off "
                                     "here%s\n", interleave.tokens > 0 ? " (SAGUARO_GLM_PREFILL_INTERLEAVE instead)" : "");
            }
        }
        gaps_over = false;
        if (use_gaps) {
            model.prefill_gap_wanted = [&] {
                if (gaps_over ||
                    !std::any_of(gens.begin(), gens.end(), [](const std::unique_ptr<Gen>& p) { return !p->done; }))
                    return false;
                if (!gap_credit) return true;
                const double stage_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gap_mark).count();
                return maya::core::prefill_gap_due(read_gaps, stage_ms, gap_round_ms);
            };
            model.prefill_gap = [&] { return read_gap(); };
            prompt_between = [&] { return read_gap(); };
            read_gap_first = true;
            gap_count = gap_tok = 0;
            gap_ms = 0.0;
        }
        // SAGUARO_GLM_PREFILL_INTERLEAVE: while other requests decode, the prompt in pieces with their steps between
        // (text prompts only: a prompt's image rows are the driver's globals while it is read)
        if (interleave.tokens > 0 && read_gaps.percent == 0 && !cur_img_pos.empty() && !gens.empty())
            std::fprintf(stderr, "glm multi: request %d's prompt has images: read whole, not in pieces "
                                 "(SAGUARO_GLM_PREFILL_INTERLEAVE is for text prompts)\n", rid);
        if (!use_gaps && interleave.tokens > 0 && cur_img_pos.empty()) {
            // (SAGUARO_GLM_PREFILL_INTERLEAVE_ALWAYS=1, a test: the pieces with nothing decoding too - to compare with
            // the prompt read whole, the decode alone)
            static const bool always = env_num("SAGUARO_GLM_PREFILL_INTERLEAVE_ALWAYS", 0) != 0;
            prompt_piece = [&]() -> size_t {
                for (const auto& gp : gens)
                    if (!gp->done) return interleave.tokens;
                return always ? interleave.tokens : 0;   // nothing decodes (any more): the rest in one call, as before
            };
            prompt_between = [&, best]() { return interleave_round(best); };
        }
        admit_stop.store(false);
        admitting_rid.store(rid);
        maya::core::Glm5Model::live_seqs_set((int) gens.size() + 1);   // (SAGUARO_GLM_VRAM_EVICT=live: it and those decoding)
        g->ta = std::chrono::steady_clock::now();
        if (ev_log) ev_line("admit", rid);
        if (conv.on || conv.open_only) {   // its conversation, and the allocation before its read (the preload's kick)
            if (conv.on) conv.attribute(conv_decoding());
            g->conv = admitting_conv = conv.admit(g->prompt, g->ta);
            admitting_ctx = g->ctx;
            if (conv.shadow) {   // (its frozen lru set back into its sequence; a new conversation starts empty)
                if (auto* cv0 = conv.find(g->conv)) maya::core::Glm5Model::shadow_seq_load(g->ctx, cv0->frozen);
                maya::core::Glm5Model::shadow_event();
            }
            const auto* cv = conv.find(g->conv);
            std::fprintf(stderr, "glm conv: r%d conversation %d (%s, %zu tokens shared)\n", rid, g->conv,
                         cv != nullptr && cv->lcp > 0 ? "resumed" : "new", cv != nullptr ? cv->lcp : (size_t) 0);
            maya::core::Glm5Model::conv_prefetch_counts(g->pf0);
            conv_running(g->conv, -1, g->ctx);
            conv.publish("admit", g->ta);
        }
        if (maya::core::Glm5Model::route_counting()) {
            maya::core::Glm5Model::route_counts(g->rc0);
            if (!gens.empty()) {
                g->rc_shared = true;
                for (auto& o2 : gens) o2->rc_shared = true;
            }
        }
        int rc = 0;
        // (upstream's in-thread modes, SAGUARO_GLM_PREFILL_GAPS / _INTERLEAVE, read it themselves when set)
        if (!gens.empty() && model.fast() && !prefill_fg && !use_gaps && !prompt_piece && model.prefill_beside_ok()) {
            // others are decoding: the batched part on the Reader, they go on meanwhile - in segments ending at the
            // marks (the system prompt's end, a divergence point: their shared slots), then before the reply's opening
            g->reuse = prep_restore(g->prompt);
            if (o.serve) std::printf("RESUME %s%zu\n", tag.c_str(), g->reuse);   // (its own restored base, as prep_prompt's)
            g->t0 = std::chrono::steady_clock::now();
            gap_count = gap_tok = 0;   // (the Reader's read takes no gaps)
            gap_ms = 0.0;
            g->sa = snap_pos(g->prompt, req_gt);
            g->div = restore_div;
            g->sys_end = maya::core::system_prompt_end(g->prompt.data(), g->prompt.size(), 154827);   // <|user|>
            g->stops = read_marks(g->prompt, g->reuse, g->sa, g->sys_end, g->div);
            g->stops.push_back(g->sa);
            if (model.prefill_chunk() > 0 && g->sa > g->reuse) {
                if (reader_start(g.get(), g->reuse, g->stops.front())) {
                    tag.clear();
                    const size_t n_read = g->sa - g->reuse;
                    gens.push_back(std::move(g));
                    std::fprintf(stderr, "glm multi: request %d reads %zu prompt tokens beside %zu decoding\n", rid,
                                 n_read, gens.size() - 1);
                    return;
                }
                if (!err.empty()) {
                    snap_tokens.clear();
                    std::printf("ERR %s%s\n", tag.c_str(), err.c_str());
                    rc = 1;
                }
            }
            if (rc == 0) rc = prep_tail(g->prompt, g->reuse, g->reuse, g->finish, g->t0, g->sa);
        } else {
            rc = prep_prompt(g->prompt, g->reuse, g->finish, g->t0);
        }
        admitting_rid.store(-1);
        prompt_piece = nullptr;
        prompt_between = nullptr;
        model.prefill_gap = nullptr;
        model.prefill_gap_wanted = nullptr;
        read_gap_first = false;
        if (use_gaps && gap_count > 0)
            std::fprintf(stderr, "glm multi: request %d's prompt (%zu tokens) read with %llu decode gaps: %.1f s of "
                                 "decode, %llu tokens for the others (%.1f tok/s there)\n",
                         rid, g->prompt.size(), (unsigned long long) gap_count, gap_ms / 1000.0,
                         (unsigned long long) gap_tok, 1000.0 * (double) gap_tok / std::max(1.0, gap_ms));
        tag.clear();
        if (rc != 0) {
            model.set_miss_skip(best, -1.0, -1);   // (PLAN-turbo-live: its read failed - the context's bars go back to the engine's)
            model.set_turbo(best, 0, 0);
            ctxs[(size_t) best].busy = false;
            if (conv.shadow || conv.uni) conv_running(-1, rid);   // (its read is over)
            admit_stop.store(false);
            std::lock_guard<std::mutex> lk(ctl_mu);   // (a STOP for it that came during its prompt: nothing to stop now)
            ctl.erase(rid);
            return;
        }
        gens.push_back(std::move(g));
        gen_admitted(*gens.back());
    };
    // the waiting requests, in order, that an idle context can now hold (a STOP for one ends it unread)
    const auto admit_pending = [&] {
        if (reader) return;
        // (each waiting request looked at once a pass: one gen_admit queues again goes to the back)
        size_t left = pending.size();
        for (size_t i = 0; left > 0 && i < pending.size(); --left) {
            const Pending& w = pending[i];
            bool stopped = false;
            {
                std::lock_guard<std::mutex> lk(ctl_mu);
                const auto it = ctl.find(w.rid);
                if (it != ctl.end()) {
                    stopped = true;
                    ctl.erase(it);
                }
            }
            if (stopped) {
                std::printf("DONE %d 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", w.rid, w.ids.size());
                pending.erase(pending.begin() + (long) i);
                continue;
            }
            const int64_t want = (int64_t) w.ids.size() + std::min<int64_t>(w.max_new, multi_room) + 8;
            bool free_fit = false;
            for (int c = 0; c < (int) ctxs.size(); ++c)
                free_fit = free_fit || (!ctxs[(size_t) c].busy && model.seq_max_ctx(c) >= want);
            if (!free_fit) {
                ++i;
                continue;
            }
            Pending w2 = std::move(pending[i]);
            pending.erase(pending.begin() + (long) i);
            req_think_budget = w2.think_budget;
            req_think_end = w2.think_end;
            req_bias = std::move(w2.bias);
            req_gt = w2.gt;
            req_msw = w2.msw;   // (its miss bar and live Turbo go with it, PLAN-turbo-live)
            req_msf = w2.msf;
            req_tdrop = w2.tdrop;
            req_tskip = w2.tskip;
            cur_img_pos = std::move(w2.img_pos);
            cur_img_rows = std::move(w2.img_rows);
            gen_admit(w2.rid, w2.ids, w2.max_new, w2.rq);
        }
    };
    // one step of the scheduler: a stint of the speculative path for a request alone, else one token back from the
    // pipelined decode (and every request with a token to feed issued first).  Errors end the running requests.
    bool step_no_spec = false;   // (between a prompt's pieces: the speculative stint runs until input comes - not there)
    const auto multi_step = [&] {
        std::vector<float> lg;
        // (a PAUSE in the last read's gaps: its snapshot, once no request waits for its first token)
        if (!deferred_pauses.empty() && !in_read_gap && admitting_rid.load() < 0 &&
            std::none_of(gens.begin(), gens.end(), [](const std::unique_ptr<Gen>& p) { return p->ready && !p->done; }))
            flush_pauses();
        if (reader && reader->done.load(std::memory_order_acquire)) {
            reader_finish();
            return;
        }
        {   // (a prompt being admitted counts: on the Reader it is in gens already, read in-thread it is not yet)
            const int arid = admitting_rid.load();
            const bool extra = arid >= 0 && std::none_of(gens.begin(), gens.end(),
                                                         [arid](const std::unique_ptr<Gen>& p) { return p->rid == arid; });
            maya::core::Glm5Model::live_seqs_set((int) gens.size() + (extra ? 1 : 0));
            if (gens.size() + (extra ? 1 : 0) > 1 && maya::core::Glm5Model::route_counting())
                for (auto& o2 : gens) o2->rc_shared = true;
        }
        conv_tick();
        Gen* only = gens.size() == 1 ? gens[0].get() : nullptr;
        static const bool spec_win_env = model.spec_win_mode() != 0;
        if (only != nullptr && !only->inflight && !only->reading && multi_spec && !step_no_spec && model.spec_ready() && spec_win_env) {
            Gen& g = *only;
            bind_ctx(g.ctx);
            if (g.pending >= 0) {   // its last token through the token path: the next one's logits on the tail
                if (!model.forward({g.pending}, lg, err)) {
                    multi_fail(err);
                    return;
                }
                g.pending = -1;
            }
            g.ready = false;
            model.force_next(g.force);
            g.force = -1;
            int32_t last = -1, forced_unused = -1;
            bool interrupted = false;
            int64_t n_spec = 0;
            const bool ok = model.decode_spec_pipe(g.sp, g.max_new - g.produced, [&](int tok) {
                forced_unused = -1;   // a force handed over is consumed by the token that comes next
                last = tok;
                if (!gen_token(g, tok)) return false;
                if (g.force >= 0) {
                    model.force_next(g.force);
                    forced_unused = g.force;
                    g.force = -1;
                }
                if (input_pending()) {
                    interrupted = true;
                    return false;
                }
                return true;
            }, n_spec, err);
            if (forced_unused >= 0) {   // handed over, never taken: it stays the request's
                g.force = forced_unused;
                model.force_next(-1);
            }
            if (!ok) {
                multi_fail(err);
                return;
            }
            // not interrupted by input: the pipelined decode ended the request (a stop, its budget, the context)
            if (interrupted) g.pending = last;
            else gen_finish(g);
            gen_sweep();
            return;
        }
        if (only != nullptr && !only->inflight && !only->reading && multi_spec && !step_no_spec && model.spec_ready()) {
            Gen& g = *only;
            // elastic: a stint of at most 4096 tokens, room for it reserved first (not the whole answer: that pushed out
            // the conversation that had just ended, to a slot and back for its next turn)
            int64_t stint = g.max_new - g.produced;
            if (model.elastic()) {
                stint = std::min<int64_t>(stint, 4096);
                ctx_reserve(g.ctx, model.seq_pos(g.ctx) + stint + 8);
            }
            bind_ctx(g.ctx);
            if (g.pending >= 0) {   // its last token through the token path: the next one's logits on the tail
                if (!model.forward({g.pending}, lg, err)) {
                    multi_fail(err);
                    return;
                }
                g.pending = -1;
            }
            g.ready = false;
            model.force_next(g.force);
            g.force = -1;
            int32_t last = -1, forced_unused = -1;
            bool interrupted = false, ended = false;
            int64_t n_spec = 0, n_stint = 0;
            const bool ok = model.decode_spec(g.sp, stint, [&](int tok) {
                forced_unused = -1;   // a force handed over is consumed by the token that comes next
                last = tok;
                ++n_stint;
                if (!gen_token(g, tok)) {
                    ended = true;
                    return false;
                }
                if (g.force >= 0) {
                    model.force_next(g.force);
                    forced_unused = g.force;
                    g.force = -1;
                }
                if (input_pending()) {
                    interrupted = true;
                    return false;
                }
                return true;
            }, n_spec, err);
            if (forced_unused >= 0) {   // handed over, never taken: it stays the request's
                g.force = forced_unused;
                model.force_next(-1);
            }
            if (!ok) {
                multi_fail(err);
                return;
            }
            // not interrupted by input: decode_spec ended the request (a stop, its budget, the context) - or the stint
            if (interrupted || (!ended && n_stint == stint && g.produced < g.max_new && last >= 0)) g.pending = last;
            else gen_finish(g);
            gen_sweep();
            return;
        }
        // a request admitted alone that the speculative path does not take: its first token, from its prompt's logits
        if (!resolve_ready()) {
            multi_fail(err);
            return;
        }
        // SAGUARO_GLM_MULTI_MTP: the ready requests' windows (their last token and the drafts made for the positions
        // after it) in groups as the split's batched rows take them, each group through every part but the last;
        // then the oldest group back (win_take) - its last part queued now if it was not
        // (SAGUARO_GLM_MULTI_MTP_MAX, 4: the windows only while that many requests or fewer run - with more, every
        // GPU is already busy and a window's rows cost about what the tokens they save cost; 0: always. Switching
        // drains what the other way has in flight)
        static const int mtp_max = (int) env_num("SAGUARO_GLM_MULTI_MTP_MAX", 4);
        int n_run = 0;
        for (auto& gp : gens) n_run += !gp->done;
        const bool use_win = mtp_rows && (mtp_max <= 0 || n_run <= mtp_max);
        if ((use_win && !flight.empty()) || (!use_win && !wflight.empty())) {
            if (!multi_drain()) {
                multi_fail(err);
                return;
            }
            gen_sweep();
            return;
        }
        if (use_win) {
            int ready = 0, active = 0;
            for (auto& gp : gens) {
                if (gp->done) continue;
                ready += !gp->inflight && gp->pending >= 0;
                active += gp->inflight || gp->pending >= 0;
            }
            const int k = std::max(1, split_rows > 0 ? split_rows : model.seq_count());
            const std::vector<int> sizes = maya::core::split_groups(ready, active, k, even_groups);
            std::vector<Gen*> rdy;
            for (auto& gp : gens)
                if (!gp->done && !gp->inflight && gp->pending >= 0) rdy.push_back(gp.get());
            size_t at = 0;
            for (const int gs : sizes) {
                WinGroup w;
                for (int j = 0; j < gs && at < rdy.size(); ++j) w.g.push_back(rdy[at++]);
                for (Gen* gp : w.g) {
                    const int64_t pos = model.seq_pos(gp->ctx);
                    const int d = gp->draft_at == pos && gp->draft_tok == gp->pending ? (int) gp->drafts.size() : 0;
                    w.seqs.push_back(gp->ctx);
                    w.counts.push_back(maya::core::mtp_rows_count(d, gp->max_new - gp->produced,
                                                                    model.seq_max_ctx(gp->ctx) - 1 - pos));
                    w.sps.push_back(gp->sp);
                }
                maya::core::mtp_rows_fit(w.counts, maya::core::Glm5Model::rows_max());
                for (size_t j = 0; j < w.g.size(); ++j) {
                    w.toks.push_back(w.g[j]->pending);
                    for (int t = 1; t < w.counts[j]; ++t) w.toks.push_back(w.g[j]->drafts[(size_t) t - 1]);
                }
                if (w.g.empty()) continue;
                if (!model.mtpw_front(w.seqs.data(), w.counts.data(), (int) w.seqs.size(), w.toks.data(), err)) {
                    multi_fail(err);
                    return;
                }
                for (Gen* gp : w.g) gp->inflight = true;
                wflight.push_back(std::move(w));
            }
            if (wflight.empty()) {
                gen_sweep();
                return;
            }
            if (!win_take()) {
                multi_fail(err);
                return;
            }
            if (multi_boundary_every > 0 &&
                since_boundary >= mb_every() * (int) std::max<size_t>(1, gens.size())) {
                since_boundary = 0;
                if (!multi_drain() || !model.multi_boundary(err)) {
                    multi_fail(err);
                    return;
                }
            }
            gen_sweep();
            return;
        }
        // ONE GPU: every request with a token to feed goes through one step together, each weight read once for all
        if (model.rows_ok()) {
            std::vector<Gen*> rg;
            std::vector<int> seqs, out;
            std::vector<int32_t> toks;
            std::vector<maya::kernels::SamplerParams> sps;
            for (auto& gp : gens)
                if (!gp->done && gp->pending >= 0 && (int) rg.size() < maya::core::Glm5Model::rows_max() &&
                    elastic_room(*gp)) {
                    rg.push_back(gp.get());
                    seqs.push_back(gp->ctx);
                    toks.push_back(gp->pending);
                    sps.push_back(gp->sp);
                }
            if (!rg.empty()) {
                out.resize(rg.size());
                if (!model.multi_rows(seqs.data(), toks.data(), sps.data(), (int) rg.size(), out.data(), err)) {
                    multi_fail(err);
                    return;
                }
                for (size_t i = 0; i < rg.size(); ++i) {   // (the step keeps the tiers' boundary itself)
                    Gen& g = *rg[i];
                    g.sp.counter += 1;
                    g.pending = -1;
                    gen_took(g, out[i]);
                }
            }
            gen_sweep();
            return;
        }
        if (split_rows > 0 && model.split_rows_ok()) {
            split_group.clear();
            const auto issue = [&]() -> bool {          // one batched step of the gathered group
                if (split_group.empty()) return true;
                split_seqs.clear();
                split_toks.clear();
                split_sps.clear();
                for (Gen* gp : split_group) {
                    split_seqs.push_back(gp->ctx);
                    split_toks.push_back(gp->pending);
                    split_sps.push_back(gp->sp);
                }
                if (!model.multi_issue_rows(split_seqs.data(), split_toks.data(), split_sps.data(),
                                           (int) split_group.size(), err))
                    return false;
                for (Gen* gp : split_group) {
                    gp->sp.counter += 1;
                    gp->pending = -1;
                    gp->inflight = true;
                    flight.push_back(gp);
                }
                flight_groups.push_back((int) split_group.size());
                split_group.clear();
                return true;
            };
            int ready = 0, active = 0;
            for (auto& gp : gens) {
                if (gp->done) continue;
                ready += !gp->inflight && gp->pending >= 0;
                active += gp->inflight || gp->pending >= 0;
            }
            const std::vector<int> sizes = maya::core::split_groups(ready, active, split_rows, even_groups);
            size_t gi = 0;
            for (auto& gp : gens) {
                if (gp->done || gp->inflight || gp->pending < 0 || !elastic_room(*gp)) continue;
                split_group.push_back(gp.get());
                if (gi < sizes.size() && (int) split_group.size() == sizes[gi]) {
                    ++gi;
                    if (!issue()) { multi_fail(err); return; }
                }
            }
            if (!issue()) { multi_fail(err); return; }
            if (flight.empty()) {
                if (reader) std::this_thread::sleep_for(std::chrono::milliseconds(2));   // only the prompt runs
                gen_sweep();
                return;
            }
            const int k = flight_groups.front();   // the oldest group, all of it
            flight_groups.pop_front();
            for (int j = 0; j < k; ++j) {
                Gen& g = *flight.front();
                flight.pop_front();
                const int tok = model.multi_take(g.ctx, err);
                g.inflight = false;
                if (tok < 0) { multi_fail(err); return; }
                gen_took(g, tok);
            }
            since_boundary += k;
            if (multi_boundary_every > 0 && !in_read_gap &&
                since_boundary >= mb_every() * (int) std::max<size_t>(1, gens.size())) {
                since_boundary = 0;
                if (!multi_drain() || !model.multi_boundary(err)) { multi_fail(err); return; }
            }
            gen_sweep();
            return;
        }
        for (auto& gp : gens) {
            Gen& g = *gp;
            if (g.done || g.inflight || g.pending < 0 || !elastic_room(g)) continue;
            if (!model.multi_issue(g.ctx, g.pending, g.sp, err)) {
                multi_fail(err);
                return;
            }
            g.sp.counter += 1;
            g.pending = -1;
            g.inflight = true;
            flight.push_back(&g);
        }
        if (flight.empty()) {
            if (reader) std::this_thread::sleep_for(std::chrono::milliseconds(2));   // only the prompt runs
            gen_sweep();
            return;
        }
        Gen& g = *flight.front();
        flight.pop_front();
        const int tok = model.multi_take(g.ctx, err);
        g.inflight = false;
        if (tok < 0) {
            multi_fail(err);
            return;
        }
        gen_took(g, tok);
        if (multi_boundary_every > 0 && ++since_boundary >= mb_every() * (int) std::max<size_t>(1, gens.size())) {
            since_boundary = 0;
            if (!multi_drain() || !model.multi_boundary(err)) {
                multi_fail(err);
                return;
            }
        }
        gen_sweep();
    };

    // Between two pieces of a prompt being read (SAGUARO_GLM_PREFILL_INTERLEAVE): `rounds` rounds of the scheduler's own
    // steps - each until every request running at its start has a token more (or ended) - then every issued token back
    // (the prompt path needs the GPUs to itself) and the reading context bound again.  The steps are multi_step's, so
    // a request that ends meanwhile ends as it would (DONE, its pause snapshot), and a STOP for one ends it; an engine
    // error fails them all (multi_fail resets every context, the one being read too).  The new prompt's own STOP is
    // admit_stop.  0: go on, 1: the read was stopped, 2: failed (prompt_between_err).
    interleave_round = [&](int read_ctx) -> int {
        if (read_stopped()) return 1;
        if (gens.empty()) return 0;   // (nothing decodes: no sweep either - an empty one clears stop_req)
        const uint64_t fails0 = multi_fails;
        const auto failed = [&] {
            prompt_between_err = "the engine failed while other requests decoded between this prompt's pieces: " +
                                 multi_fail_why;
            return 2;
        };
        step_no_spec = true;
        for (int r = 0; r < interleave.rounds && !gens.empty(); ++r) {
            std::map<int, int64_t> want;   // rid -> the produced count that ends this round for it
            for (const auto& gp : gens)
                if (!gp->done) want[gp->rid] = gp->produced + 1;
            // (bounded: a step takes one token back at least while any request runs)
            for (int it = 0; it < 4 * (int) want.size() + 8 && !want.empty(); ++it) {
                multi_step();
                if (multi_fails != fails0) {
                    step_no_spec = false;
                    return failed();
                }
                for (auto w = want.begin(); w != want.end();) {
                    const Gen* g = nullptr;
                    for (const auto& gp : gens)
                        if (gp->rid == w->first) g = gp.get();
                    w = g == nullptr || g->done || g->produced >= w->second ? want.erase(w) : std::next(w);
                }
            }
        }
        step_no_spec = false;
        if (!multi_drain()) {
            multi_fail(err);
            return failed();
        }
        if (std::any_of(gens.begin(), gens.end(), [](const std::unique_ptr<Gen>& p) { return p->done; })) gen_sweep();
        bind_ctx(read_ctx);
        return read_stopped() ? 1 : 0;
    };

    // One decode gap of a prompt's read (SAGUARO_GLM_PREFILL_GAPS): at a chunk boundary inside a prompt-path call (every
    // part idle, the pool's tail lent; Glm5Model::prefill_gap) or between two of its calls.  The running requests' own
    // steps (multi_step's: the split's batched rows, which never bind a context) in rounds - each until every request
    // running at its start has a token more or ended - for percent / (100 - percent) of the time the read ran since
    // the last gap (one round at least), then every issued token back: the read goes on with the GPUs to itself.  No
    // tier boundary runs in it (in_read_gap), a request that ends meanwhile ends as it would (a PAUSE's snapshot after
    // the read: deferred_pauses), a STOP for one ends it; an engine error fails them all, the contexts' reset waits
    // for the read's end (gap_reset).  A gap that overruns <max gap ms> (its one round can, where experts stream) ends
    // the gaps of this read (gaps_over).  0: go on, 1: the read was stopped, 2: failed (prompt_between_err).
    read_gap = [&]() -> int {
        if (read_stopped()) return 1;
        const auto t_in = std::chrono::steady_clock::now();
        const double stage_ms = std::chrono::duration<double, std::milli>(t_in - gap_mark).count();
        if (gens.empty() || gaps_over) {
            gap_mark = t_in;
            return 0;
        }
        const double budget = maya::core::prefill_gap_budget_ms(read_gaps, stage_ms);
        const uint64_t fails0 = multi_fails, n0 = multi_n;
        in_read_gap = true;
        step_no_spec = true;
        int rounds = 0;
        for (bool first = true; !gens.empty() && multi_fails == fails0; first = false) {
            if (!first && std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_in).count() >=
                              budget)
                break;
            ++rounds;
            std::map<int, int64_t> want;   // rid -> the produced count that ends this round for it
            for (const auto& gp : gens)
                if (!gp->done) want[gp->rid] = gp->produced + 1;
            if (want.empty()) break;   // (every request ended: nothing to decode)
            for (int it = 0; it < 4 * (int) want.size() + 8 && !want.empty() && multi_fails == fails0; ++it) {
                multi_step();
                for (auto w = want.begin(); w != want.end();) {
                    const Gen* g = nullptr;
                    for (const auto& gp : gens)
                        if (gp->rid == w->first) g = gp.get();
                    w = g == nullptr || g->done || g->produced >= w->second ? want.erase(w) : std::next(w);
                }
            }
        }
        if (multi_fails == fails0 && !multi_drain()) multi_fail(err);
        step_no_spec = false;
        in_read_gap = false;
        gap_mark = std::chrono::steady_clock::now();
        ++gap_count;
        gap_tok += multi_n - n0;
        const double this_ms = std::chrono::duration<double, std::milli>(gap_mark - t_in).count();
        gap_ms += this_ms;
        if (rounds > 0) gap_round_ms = this_ms / rounds;   // (the drain's share too: a gap costs it once)
        if (this_ms > read_gaps.max_ms && !gaps_over) {
            // (a round is one token for every running request, whatever the budget: where their experts stream from
            // RAM or disk inside the loan it can take far longer than the stage - the rest of this read has no gaps)
            gaps_over = true;
            std::fprintf(stderr, "glm multi: a decode gap took %.0f ms (more than %.0f): no more gaps in this read\n",
                         this_ms, read_gaps.max_ms);
        }
        if (multi_fails != fails0) return 2;
        if (std::any_of(gens.begin(), gens.end(), [](const std::unique_ptr<Gen>& p) { return p->done; })) gen_sweep();
        return read_stopped() ? 1 : 0;
    };

    // SAGUARO_GLM_MULTI_BENCH=<n>,<prompt>,<gen>[,<boundary rounds>] (needs SAGUARO_GLM_SEQS >= n): n synthetic
    // prompts of <prompt> tokens, then <gen> greedy tokens each - sequence 0 alone through the token path first (the
    // reference), then all n through the selected together, split-rows or pipelined path; prints both rates and whether
    // sequence 0 wrote the same tokens both ways - and every other sequence against its own run alone.
    if (const char* mb = getenv("SAGUARO_GLM_MULTI_BENCH")) {
        int n = 2, L = 512, G = 256, K = 32, same_prompt = 0;
        std::sscanf(mb, "%d,%d,%d,%d,%d", &n, &L, &G, &K, &same_prompt);
        if (n > model.seq_count() || n < 1 || L < 2 || G < 2) {
            std::fprintf(stderr, "multi bench: %d sequences asked, %d states (SAGUARO_GLM_SEQS)\n", n, model.seq_count());
            return 2;
        }
        model.set_host_logits(false);
        maya::kernels::SamplerParams g0 = sp;
        g0.greedy = true;
        std::vector<std::vector<int32_t>> prompts((size_t) n);
        uint64_t rs = 0x9e3779b97f4a7c15ull;
        for (int s = 0; s < n; ++s) {
            if (same_prompt) rs = 0x9e3779b97f4a7c15ull;
            for (int i = 0; i < L; ++i) {
                if (!o.tokens.empty()) {   // --tokens: a real prompt (repeated / cut to L), the same for every sequence
                    prompts[(size_t) s].push_back((int32_t) o.tokens[(size_t) i % o.tokens.size()]);
                    continue;
                }
                rs = rs * 6364136223846793005ull + 1442695040888963407ull;
                prompts[(size_t) s].push_back((int32_t) (1000 + (rs >> 33) % 120000));
            }
        }
        std::vector<float> lg;
        const auto now = [] { return std::chrono::steady_clock::now(); };
        const auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
        for (int s = 0; s < n; ++s) {   // every prompt read once, its state saved before its last token
            model.seq_bind_all(s);
            model.reset();
            const std::vector<int32_t> head(prompts[(size_t) s].begin(), prompts[(size_t) s].end() - 1);
            if (!model.prefill(head, err, prompts[(size_t) s].back()) && !err.empty()) {
                std::fprintf(stderr, "multi bench: prefill: %s\n", err.c_str());
                return 1;
            }
            if (!model.snapshot_save()) {
                std::fprintf(stderr, "multi bench: snapshot\n");
                return 1;
            }
        }
        // the reference: sequence 0 alone, the classic token path - once untimed first (the tiers warm up on it, as
        // they would before any timed run; and is the token path itself repeatable here?), then timed
        std::vector<int> solo, solo2;
        model.seq_bind_all(0);
        double solo_s = 0;
        maya::core::Glm5Model::FastStats st_a, st_b;   // the expert tiers over each timed phase
        const auto tiers = [](const char* what, const maya::core::Glm5Model::FastStats& a,
                              const maya::core::Glm5Model::FastStats& b, int tokens) {
            const double h = (double) (b.hits - a.hits), m = (double) (b.misses - a.misses), t = std::max(1, tokens);
            std::printf("  %s: VRAM hit %.1f%%, a token: %.1f RAM fetches, %.2f disk reads, %.1f promotions, %.1f CPU "
                        "experts\n", what, 100.0 * h / std::max(1.0, h + m), (double) (b.ram_hits - a.ram_hits) / t,
                        (double) (b.disk_reads - a.disk_reads) / t, (double) (b.promotions - a.promotions) / t,
                        (double) (b.cpu_experts - a.cpu_experts) / t);
        };
        for (std::vector<int>* run : {&solo2, &solo}) {
            model.snapshot_restore();
            st_a = model.fast_stats();
            if (!model.forward({prompts[0].back()}, lg, err)) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
            const auto t0 = now();
            run->push_back(model.last_tok_);
            while ((int) run->size() < G) {
                if (!model.forward({(int32_t) run->back()}, lg, err)) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
                run->push_back(model.last_tok_);
            }
            solo_s = secs(t0, now());
            st_b = model.fast_stats();
        }
        const maya::core::Glm5Model::FastStats solo_a = st_a, solo_b = st_b;
        // every other sequence alone too: its own reference
        std::vector<std::vector<int>> solos((size_t) n);
        solos[0] = solo;
        for (int s = 1; s < n; ++s) {
            model.seq_bind_all(s);
            model.snapshot_restore();
            if (!model.forward({prompts[(size_t) s].back()}, lg, err)) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
            solos[(size_t) s].push_back(model.last_tok_);
            while ((int) solos[(size_t) s].size() < G) {
                if (!model.forward({(int32_t) solos[(size_t) s].back()}, lg, err)) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
                solos[(size_t) s].push_back(model.last_tok_);
            }
        }
        // all n through the pipelined decode
        std::vector<std::vector<int>> out((size_t) n);
        for (int s = 0; s < n; ++s) {
            model.seq_bind_all(s);
            model.snapshot_restore();
            if (!model.forward({prompts[(size_t) s].back()}, lg, err)) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
            out[(size_t) s].push_back(model.last_tok_);
        }
        std::deque<int> q;
        st_a = model.fast_stats();
        const auto t0 = now();
        int since = 0, boundaries = 0;
        // SAGUARO_GLM_MULTI_MTP: every sequence's verify windows through the batched rows (the server's order of calls)
        const bool wrows = model.mtpw_ok() && (model.rows_ok() || (split_rows > 0 && model.split_rows_ok()));
        const bool rows = model.rows_ok() && !wrows;   // one GPU: all n together, one step per token of each
        const bool srows = !rows && !wrows && split_rows > 0 && model.split_rows_ok();
        const char* path = wrows ? "mtp-windows" : rows ? "together" : srows ? "split-rows" : "pipelined";
        uint64_t w_steps = 0, w_drafts = 0, w_acc = 0;
        if (wrows) {
            const int D = maya::core::Glm5Model::multi_mtp_setting();
            std::vector<std::vector<int32_t>> drafts((size_t) n);
            struct BG {
                std::vector<int> seqs, counts;
                std::vector<int32_t> toks;
                bool back = false;
            };
            const auto back = [&](BG& b) -> bool {
                if (b.back) return true;
                b.back = true;
                const std::vector<maya::kernels::SamplerParams> sps(b.seqs.size(), g0);
                return model.mtpw_back(b.seqs.data(), b.counts.data(), (int) b.seqs.size(), b.toks.data(), sps.data(),
                                       err);
            };
            for (;;) {
                std::vector<int> act;
                for (int s = 0; s < n; ++s)
                    if ((int) out[(size_t) s].size() < G) act.push_back(s);
                if (act.empty()) break;
                std::vector<BG> gs;
                size_t first = 0;
                for (const int k : maya::core::split_groups((int) act.size(), (int) act.size(),
                                                              std::max(1, split_rows > 0 ? split_rows : n), even_groups)) {
                    BG b;
                    for (int j = 0; j < k; ++j) {
                        const int s = act[first + (size_t) j];
                        b.seqs.push_back(s);
                        b.counts.push_back(maya::core::mtp_rows_count(
                            (int) drafts[(size_t) s].size(), G - (int64_t) out[(size_t) s].size(),
                            model.seq_max_ctx(s) - 1 - model.seq_pos(s)));
                    }
                    first += (size_t) k;
                    maya::core::mtp_rows_fit(b.counts, maya::core::Glm5Model::rows_max());
                    for (size_t j = 0; j < b.seqs.size(); ++j) {
                        const int s = b.seqs[j];
                        b.toks.push_back((int32_t) out[(size_t) s].back());
                        for (int t = 1; t < b.counts[j]; ++t) b.toks.push_back(drafts[(size_t) s][(size_t) t - 1]);
                    }
                    if (!model.mtpw_front(b.seqs.data(), b.counts.data(), (int) b.seqs.size(), b.toks.data(), err)) {
                        std::fprintf(stderr, "multi bench: %s\n", err.c_str());
                        return 1;
                    }
                    gs.push_back(std::move(b));
                }
                for (size_t gi = 0; gi < gs.size(); ++gi) {
                    BG& b = gs[gi];
                    const int m = (int) b.seqs.size();
                    int rows_n = 0;
                    for (const int c : b.counts) rows_n += c;
                    std::vector<int32_t> picks((size_t) rows_n), written, last((size_t) m);
                    std::vector<int> keep((size_t) m), nd((size_t) m, D);
                    if (!back(b) || !model.mtpw_picks(b.seqs.data(), b.counts.data(), m, picks.data(), err)) {
                        std::fprintf(stderr, "multi bench: %s\n", err.c_str());
                        return 1;
                    }
                    for (int i = 0, row = 0; i < m; row += b.counts[(size_t) i], ++i) {
                        const int s = b.seqs[(size_t) i];
                        std::vector<int32_t> wr;
                        keep[(size_t) i] = maya::core::mtp_accept(
                            b.toks.data() + row, picks.data() + row, b.counts[(size_t) i],
                            [&](int, int32_t pick, int32_t&) {
                                wr.push_back(pick);
                                out[(size_t) s].push_back(pick);
                                return (int) out[(size_t) s].size() < G ? maya::core::MtpTake::more
                                                                        : maya::core::MtpTake::stop;
                            });
                        const int k = keep[(size_t) i];
                        ++w_steps;
                        w_drafts += (uint64_t) (b.counts[(size_t) i] - 1);
                        w_acc += (uint64_t) (k - 1);
                        for (int t = 0; t + 1 < k; ++t) written.push_back(wr[(size_t) t]);
                        last[(size_t) i] = wr[(size_t) k - 1];
                        // (no more drafts than the tokens left after the pending one, nor past the context)
                        nd[(size_t) i] = (int) std::max<int64_t>(
                            0, std::min<int64_t>({(int64_t) D, (int64_t) G - (int64_t) out[(size_t) s].size() - 1,
                                                  model.seq_max_ctx(s) - 2 -
                                                      (model.seq_pos(s) - b.counts[(size_t) i] + k)}));
                    }
                    std::vector<int32_t> dr((size_t) m * (size_t) std::max(1, D), -1);
                    if (!model.mtpw_commit(b.seqs.data(), b.counts.data(), keep.data(), m, err) ||
                        !model.mtpw_draft_queue(b.seqs.data(), keep.data(), m, written.data(), last.data(), nd.data(),
                                                err) ||
                        (gi + 1 < gs.size() && !back(gs[gi + 1])) || !model.mtpw_draft_take(dr.data(), err)) {
                        std::fprintf(stderr, "multi bench: %s\n", err.c_str());
                        return 1;
                    }
                    for (int i = 0; i < m; ++i) {
                        auto& d = drafts[(size_t) b.seqs[(size_t) i]];
                        d.clear();
                        for (int j = 0; j < D && dr[(size_t) i * D + (size_t) j] >= 0; ++j)
                            d.push_back(dr[(size_t) i * D + (size_t) j]);
                    }
                    since += m;
                }
                if (K > 0 && since >= K * n) {
                    if (!model.multi_boundary(err)) {
                        std::fprintf(stderr, "multi bench: %s\n", err.c_str());
                        return 1;
                    }
                    ++boundaries;
                    since = 0;
                }
            }
            std::printf("  mtp windows: %llu of %llu drafts accepted (%.1f%%), %.2f tokens a window\n",
                        (unsigned long long) w_acc, (unsigned long long) w_drafts,
                        100.0 * (double) w_acc / (double) std::max<uint64_t>(1, w_drafts),
                        (double) (w_steps + w_acc) / (double) std::max<uint64_t>(1, w_steps));
        }
        if (rows) {
            std::vector<int> seqs((size_t) n), next((size_t) n);
            std::vector<int32_t> toks((size_t) n);
            const std::vector<maya::kernels::SamplerParams> sps((size_t) n, g0);
            for (int s = 0; s < n; ++s) seqs[(size_t) s] = s;
            while ((int) out[0].size() < G) {
                for (int s = 0; s < n; ++s) toks[(size_t) s] = out[(size_t) s].back();
                if (!model.multi_rows(seqs.data(), toks.data(), sps.data(), n, next.data(), err)) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
                for (int s = 0; s < n; ++s) out[(size_t) s].push_back(next[(size_t) s]);
                ++boundaries;   // (one before every step, in the step itself)
            }
        }
        for (int s = 0; s < n && !rows && !srows && !wrows; ++s) {
            if (!model.multi_issue(s, out[(size_t) s].back(), g0, err)) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
            q.push_back(s);
        }
        if (srows) {
            std::deque<int> qg;
            const auto issue_rows = [&](const std::vector<int>& seqs) -> bool {
                // (SAGUARO_GLM_EVEN_GROUPS: the groups the server makes, from the sequences still decoding)
                int active = 0;
                for (int s = 0; s < n; ++s) active += (int) out[(size_t) s].size() < G;
                size_t first = 0;
                for (const int k : maya::core::split_groups((int) seqs.size(), active, split_rows, even_groups)) {
                    std::vector<int> group_seqs;
                    std::vector<int32_t> toks;
                    std::vector<maya::kernels::SamplerParams> sps((size_t) k, g0);
                    for (int j = 0; j < k; ++j) {
                        const int s = seqs[first + (size_t) j];
                        group_seqs.push_back(s);
                        toks.push_back((int32_t) out[(size_t) s].back());
                    }
                    if (!model.multi_issue_rows(group_seqs.data(), toks.data(), sps.data(), k, err)) {
                        std::fprintf(stderr, "multi bench: %s\n", err.c_str());
                        return false;
                    }
                    for (const int s : group_seqs) q.push_back(s);
                    qg.push_back(k);
                    first += (size_t) k;
                }
                return true;
            };
            std::vector<int> initial((size_t) n);
            for (int s = 0; s < n; ++s) initial[(size_t) s] = s;
            if (!issue_rows(initial)) return 1;
            while (!qg.empty()) {
                std::vector<int> again;
                const auto take_group = [&](int k) -> bool {
                    for (int j = 0; j < k; ++j) {
                        const int s = q.front();
                        q.pop_front();
                        const int tok = model.multi_take(s, err);
                        if (tok < 0) {
                            std::fprintf(stderr, "multi bench: %s\n", err.c_str());
                            return false;
                        }
                        out[(size_t) s].push_back(tok);
                        if ((int) out[(size_t) s].size() < G) again.push_back(s);
                    }
                    return true;
                };
                const int k = qg.front();
                qg.pop_front();
                if (!take_group(k)) return 1;
                since += k;
                if (K > 0 && since >= K * n) {
                    while (!qg.empty()) {
                        const int k2 = qg.front();
                        qg.pop_front();
                        if (!take_group(k2)) return 1;
                    }
                    if (!model.multi_boundary(err)) {
                        std::fprintf(stderr, "multi bench: %s\n", err.c_str());
                        return 1;
                    }
                    ++boundaries;
                    since = 0;
                }
                if (!issue_rows(again)) return 1;
            }
        }
        while (!q.empty()) {
            const int s = q.front();
            q.pop_front();
            const int tok = model.multi_take(s, err);
            if (tok < 0) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
            out[(size_t) s].push_back(tok);
            std::vector<int> again;
            if ((int) out[(size_t) s].size() < G) again.push_back(s);
            if (K > 0 && ++since >= K * n) {   // drain: every token in flight taken, the tiers' boundary, then on
                since = 0;
                ++boundaries;
                while (!q.empty()) {
                    const int s2 = q.front();
                    q.pop_front();
                    const int t2 = model.multi_take(s2, err);
                    if (t2 < 0) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
                    out[(size_t) s2].push_back(t2);
                    if ((int) out[(size_t) s2].size() < G) again.push_back(s2);
                }
                if (!model.multi_boundary(err)) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
            }
            for (const int s2 : again) {
                if (!model.multi_issue(s2, out[(size_t) s2].back(), g0, err)) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
                q.push_back(s2);
            }
        }
        const double multi_s = secs(t0, now());
        st_b = model.fast_stats();
        tiers("solo", solo_a, solo_b, G);
        tiers(path, st_a, st_b, n * G);
        const auto agree = [](const std::vector<int>& a, const std::vector<int>& b) {
            size_t k = 0;
            while (k < a.size() && k < b.size() && a[k] == b[k]) ++k;
            return k;
        };
        const size_t same = agree(solo, out[0]);
        std::printf("  solo vs its untimed warm-up run: %zu of %zu agree\n", agree(solo, solo2), solo.size());
        bool all_same = true;
        for (int s = 1; s < n; ++s) {
            const size_t own = agree(out[(size_t) s], solos[(size_t) s]);
            all_same = all_same && own == solos[(size_t) s].size();
            std::printf("  seq %d vs seq 0: %zu agree, vs solo: %zu, vs its own run alone: %zu of %zu\n", s,
                        agree(out[(size_t) s], out[0]), agree(out[(size_t) s], solo), own, solos[(size_t) s].size());
        }
        std::printf("  every sequence wrote its own run's tokens: %s\n", all_same && same == solo.size() ? "yes" : "NO");
        std::printf("MULTI BENCH %d seqs, prompt %d, %d tokens each: solo %.1f tok/s | %s %.1f tok/s total, "
                    "%.1f per sequence (%d boundaries) | sequence 0 same tokens: %zu of %zu\n", n, L, G,
                    (G - 1) / solo_s, path, (double) n * (G - 1) / multi_s,
                    (G - 1) / multi_s, boundaries, same, solo.size());
        for (int s = 0; s < n; ++s) {
            std::printf("  seq %d:", s);
            for (int i = 0; i < 12 && i < (int) out[(size_t) s].size(); ++i) std::printf(" %d", out[(size_t) s][(size_t) i]);
            std::printf("\n");
        }
        std::printf("  solo :");
        for (int i = 0; i < 12 && i < (int) solo.size(); ++i) std::printf(" %d", solo[(size_t) i]);
        std::printf("\n");
        // SAGUARO_GLM_MULTI_DUMP=<file>: every sequence's tokens, a line each (then the solo run's) - two builds' or
        // settings' runs compared whole, not by their first 12
        if (const char* dump = getenv("SAGUARO_GLM_MULTI_DUMP")) {
            if (std::FILE* f = std::fopen(dump, "w")) {
                for (int s = 0; s <= n; ++s) {
                    const std::vector<int>& v = s < n ? out[(size_t) s] : solo;
                    if (s < n) std::fprintf(f, "seq %d:", s);
                    else std::fputs("solo:", f);
                    for (const int t : v) std::fprintf(f, " %d", t);
                    std::fputc('\n', f);
                }
                std::fclose(f);
            }
        }
        std::fflush(stdout);
        return 0;
    }

    if (!o.serve) {
        if (o.tokens.empty()) {
            std::fprintf(stderr, "saguaro generate: --tokens is required (this build has no tokenizer; see the "
                                 "header of src/program/generate.cpp)\n");
            return 2;
        }
        // SAGUARO_GLM_SCORE=<c>: score --tokens instead of generating.  Every token goes through the one-token decode
        // path (the tiers and the CPU lane as in a decode); the tokens after the first c are scored given the ones
        // before: mean NLL, perplexity, top-1 hits.  SAGUARO_GLM_SCORE_DUMP=<file> writes each scored position's
        // log-softmax (n_vocab floats); SAGUARO_GLM_SCORE_REF=<file> reads such a dump and reports KL(ref || this) and
        // how often both pick the same top token - the quality yardstick for engine changes and for quants.
        // SAGUARO_GLM_SCORE_TOPK=<file> is the compact reference (the FP8 model's, tools/maya_quant): an int32 k, then
        // per scored position k int32 token ids (best first) and their k float log-probs.  KL over those k tokens plus
        // one bucket for the rest of the vocabulary.
        if (const char* sc = getenv("SAGUARO_GLM_SCORE")) {
            const size_t c = (size_t) std::max(0L, std::atol(sc));
            const char* dump = getenv("SAGUARO_GLM_SCORE_DUMP");
            const char* ref = getenv("SAGUARO_GLM_SCORE_REF");
            const char* tkf = getenv("SAGUARO_GLM_SCORE_TOPK");
            std::FILE* df = dump && dump[0] ? std::fopen(dump, "wb") : nullptr;
            std::FILE* rf = ref && ref[0] ? std::fopen(ref, "rb") : nullptr;
            std::FILE* tf = tkf && tkf[0] ? std::fopen(tkf, "rb") : nullptr;
            int32_t tk = 0;
            if ((dump && dump[0] && !df) || (ref && ref[0] && !rf) || (tkf && tkf[0] && !tf) ||
                (tf && (std::fread(&tk, sizeof tk, 1, tf) != 1 || tk <= 0 || tk > n_vocab))) {
                std::fprintf(stderr, "saguaro generate: cannot open the score dump/reference\n");
                return 1;
            }
            std::vector<int32_t> tid((size_t) std::max(tk, 1));
            std::vector<float> tlp((size_t) std::max(tk, 1));
            model.set_host_logits(true);
            model.reset();
            std::vector<float> lg, ls((size_t) n_vocab), rl((size_t) n_vocab);
            double nll = 0.0, kl = 0.0, kl_max = 0.0;
            size_t n = 0, top1 = 0, same = 0, nref = 0;
            const auto ts = std::chrono::steady_clock::now();
            // SAGUARO_GLM_SCORE_PREFILL=1: the unscored context (all but its last token) through the prompt path, so a
            // long context is scored in minutes (the KV-cache formats at 128K); the scored tokens stay token by token
            size_t i0 = 0;
            if (getenv("SAGUARO_GLM_SCORE_PREFILL") != nullptr && model.fast() && model.prefill_chunk() > 0 && c > 2 &&
                c < o.tokens.size()) {
                const std::vector<int32_t> part(o.tokens.begin(), o.tokens.begin() + (long) (c - 1));
                if (!model.prefill(part, err, (int32_t) o.tokens[c - 1])) {
                    std::printf("ERR %s\n", err.empty() ? "prefill" : err.c_str());
                    return 1;
                }
                i0 = c - 1;
            }
            for (size_t i = i0; i + 1 < o.tokens.size(); ++i) {
                if (!model.forward({(int32_t) o.tokens[i]}, lg, err)) {
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                if (i + 1 <= c) continue;
                float mx = lg[0];
                size_t am = 0;
                for (size_t v = 1; v < (size_t) n_vocab; ++v)
                    if (lg[v] > mx) {
                        mx = lg[v];
                        am = v;
                    }
                double se = 0.0;
                for (size_t v = 0; v < (size_t) n_vocab; ++v) se += std::exp((double) lg[v] - mx);
                const double lse = mx + std::log(se);
                for (size_t v = 0; v < (size_t) n_vocab; ++v) ls[v] = (float) (lg[v] - lse);
                const size_t nx = (size_t) o.tokens[i + 1];
                nll -= ls[nx];
                top1 += am == nx;
                ++n;
                if (df) std::fwrite(ls.data(), sizeof(float), ls.size(), df);
                if (rf && std::fread(rl.data(), sizeof(float), rl.size(), rf) == rl.size()) {
                    double k = 0.0;
                    size_t ra = 0;
                    for (size_t v = 0; v < (size_t) n_vocab; ++v) {
                        const double p = std::exp((double) rl[v]);
                        if (p > 0.0) k += p * ((double) rl[v] - (double) ls[v]);
                        if (rl[v] > rl[ra]) ra = v;
                    }
                    kl += k;
                    kl_max = std::max(kl_max, k);
                    same += ra == am;
                    ++nref;
                }
                if (tf && std::fread(tid.data(), sizeof(int32_t), (size_t) tk, tf) == (size_t) tk &&
                    std::fread(tlp.data(), sizeof(float), (size_t) tk, tf) == (size_t) tk) {
                    double k = 0.0, pr = 0.0, pt = 0.0;
                    for (int32_t j = 0; j < tk; ++j) {
                        const double p = std::exp((double) tlp[j]);
                        k += p * ((double) tlp[j] - (double) ls[(size_t) tid[j]]);
                        pr += p;
                        pt += std::exp((double) ls[(size_t) tid[j]]);
                    }
                    const double rr = std::max(1.0 - pr, 1e-12), rt = std::max(1.0 - pt, 1e-12);
                    if (pr < 1.0) k += rr * (std::log(rr) - std::log(rt));
                    kl += k;
                    kl_max = std::max(kl_max, k);
                    same += (size_t) tid[0] == am;
                    ++nref;
                }
            }
            if (df) std::fclose(df);
            if (rf) std::fclose(rf);
            if (tf) std::fclose(tf);
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
            std::printf("SCORE %zu tokens: nll %.5f ppl %.4f top1 %.2f%%", n, nll / (double) std::max<size_t>(1, n),
                        std::exp(nll / (double) std::max<size_t>(1, n)), 100.0 * (double) top1 / (double) std::max<size_t>(1, n));
            if (nref > 0)
                std::printf(" | vs ref: KL %.6f (max %.4f), same top token %.2f%%", kl / (double) nref, kl_max,
                            100.0 * (double) same / (double) nref);
            const auto stc = model.fast_stats();
            std::printf(" | %.1f ms/token | cpu lane %llu experts, vram hit %.2f%%\n",
                        1000.0 * sec / (double) std::max<size_t>(1, o.tokens.size() - 1),
                        (unsigned long long) stc.cpu_experts,
                        100.0 * (double) stc.hits / (double) std::max<uint64_t>(1, stc.hits + stc.misses));
            return 0;
        }
        const int rc = run_request(o.tokens, o.max_new, sp);
        return rc;
    }

    // SAGUARO_GLM_WARM_PROMPT=<n> (default 0 = off; 32 or more): one synthetic prompt through the whole part chain
    // before the server is told READY, so the session's first request does not pay what only a session's first
    // prompt pays (issue #8: it read 95-133 ms/token against 23-44 warm - the first chunk carries the one-time work:
    // every kernel's first launch, the prompt path's own allocations, the tiers' first reads).  It costs about one
    // chunk of the smallest part's size (~15 s on a four-way split, beside the tiers' own 90-151 s warm-up), its ids
    // are spread over the vocabulary so the routes are not one expert's, and the state is zeroed right after it: no
    // conversation can see it (a request with no slot in use resets anyway, before its own prompt).
    if (const char* wv = getenv("SAGUARO_GLM_WARM_PROMPT")) {
        const int wn = std::atoi(wv);
        if (wn > 0 && model.prefill_chunk() > 0) {
            const int wtn = std::max(32, wn);
            std::vector<int32_t> wids((size_t) wtn);
            for (int i = 0; i < wtn; ++i) wids[(size_t) i] = (int32_t) (100 + (i * 131) % 20000);
            std::fprintf(stderr, "glm fast: warming the prompt path with a %d-token prompt before READY "
                                 "(SAGUARO_GLM_WARM_PROMPT)\n", wtn);
            const auto wt0 = std::chrono::steady_clock::now();
            std::string werr;
            const bool wok = model.prefill(wids, werr);
            const double wms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wt0).count();
            model.reset();
            if (wok) {
                std::fprintf(stderr, "glm fast: the prompt path is warm (%d tokens in %.0f ms, %.1f ms/token)\n", wtn,
                             wms, wms / (double) wtn);
            } else {
                std::fprintf(stderr, "glm fast: the prompt warm-up did not run%s%s - the first prompt pays it\n",
                             werr.empty() ? "" : ": ", werr.c_str());
            }
        }
    }
    // issue #40: a prompt that stopped moving (the GPU idle, a host thread spinning on a flag) waited forever - the
    // server's own stall check counts a spinning thread as work.  A prompt layer or a token is the heartbeat; a big
    // chunk's layer on a slow PC takes seconds, so three minutes of none is a hang.
    start_serve_watchdog(180);
    // the VRAM left on every GPU once everything is loaded.  Below ~256 MiB (a split's later GPUs: 512 - their verify
    // windows' graphs are captured on first use) a later allocation fails, and under Windows' WDDM the driver pages GPU
    // memory to system RAM instead: decode falls to a few tok/s and a request can stall for good.  Said, never changed:
    // the line names the setting that leaves room (--vram-reserve-mib is deducted before the expert pool is sized).
    const auto pools = model.fast() ? model.pool_vram() : std::vector<maya::core::Glm5Model::PoolVram>();
    const auto free_mib_on = [](int dev) -> int64_t {
        int cur = 0;
        cudaGetDevice(&cur);
        cudaSetDevice(dev);
        size_t fb = 0, tb = 0;
        const bool ok = cudaMemGetInfo(&fb, &tb) == cudaSuccess;
        if (!ok) cudaGetLastError();
        cudaSetDevice(cur);
        return ok ? (int64_t) (fb >> 20) : -1;
    };
    for (size_t i = 0; i < pools.size(); ++i) {
        const auto& pv = pools[i];
        const int64_t fm = free_mib_on(pv.dev);
        if (fm < 0) continue;
        const int64_t ok_mib = i == 0 ? 256 : 512;
        if (fm >= ok_mib) {
            std::fprintf(stderr, "saguaro serve: CUDA%d: %lld MiB of VRAM free with everything loaded\n", pv.dev, (long long) fm);
        } else if (pv.adapted) {
            std::fprintf(stderr, "saguaro serve: WARNING: CUDA%d: %lld MiB of VRAM free with everything loaded - this card "
                                 "only just fits the model (the VRAM reserve was lowered to %d MiB so the expert pool "
                                 "fits): requests may stall. Close other programs that use the GPU, or lower "
                                 "--max-context\n", pv.dev, (long long) fm, pv.reserve_mib);
        } else {
            std::fprintf(stderr, "saguaro serve: CUDA%d: %lld MiB of VRAM free with everything loaded - LOW: requests may "
                                 "stall; add %s %lld to the config's args (or lower --max-context)\n", pv.dev,
                         (long long) fm, i == 0 ? "--vram-reserve-mib" : "--vram-reserve-later-mib",
                         (long long) (pv.reserve_mib + (i == 0 ? 512 : 768) - fm));
            // a number of the user's own (--expert-cache N) is not checked against the VRAM once written (auto is)
            if (pv.fixed > 0)
                std::fprintf(stderr, "saguaro serve: the expert pool was set with --expert-cache %lld: a smaller one - or "
                                     "--expert-cache auto, which checks the free VRAM once the pool is written - leaves "
                                     "room\n", (long long) pv.fixed);
        }
    }
    // (#1026) a start that came out short of its VRAM reserve: the server starts the engine once more, which keeps
    // what this one measured free beside it
    if (model.fast())
        if (const int short_mib = model.late_check(true); short_mib > 0) std::printf("INFO reserve_short=%d\n", short_mib);
    // the serve wire contract (serve/server.py): READY <max_context> [stop] once, then GEN <max_new>
    // [<key>=<value> ...] <id,id,...> lines - the sampling keys are per request and the ids are the LAST
    // token.  "stop" is NOT advertised: a request runs to its end (no mid-request cancel in M3.1).
    // "pause": PAUSE keeps the written state; "multi=<n>": n conversations at once (GEN rid=<id>, see gen_admit)
    if (multi) std::printf("READY %lld stop pause multi=%d\n", (long long) o.max_context, model.seq_count());
    else std::printf("READY %lld stop pause\n", (long long) o.max_context);
    // PLAN-vram-live: the elastic contexts back their KV as they grow and the expert pool gives way and takes back
    // (glm_multi.cu), a prompt borrows the pool's tail, VLEND lends it to the encoder - the Monitor's map of a GPU
    // follows: a card whose bookkeeping bytes moved re-prints its `VRAM dev=...` line (dirty at most every
    // SAGUARO_VRAM_REPORT_S s, every card every 60 s; maya/program/vram_live.hpp). The fingerprint asks no device.
    if (model.fast() && maya::program::vram_report_s() > 0) {
        static maya::program::VramLive vram_live;   // (the beat thread below owns it for the process)
        maya::program::VramLive::beat([&model, print_vram] {
            std::vector<std::string> fp;
            for (const auto& p : model.vram_fingerprint())
                fp.push_back(std::to_string(p[0]) + ":" + std::to_string(p[1]) + ":" + std::to_string(p[2]) + ":" +
                             std::to_string(p[3]) + ":" + std::to_string(p[4]));
            vram_live.tick(fp, [&](size_t i) {   // (vram_report: the same part order as the fingerprint; the device
                const auto rep = model.vram_report();   //  is asked only when a card prints)
                if (i < rep.size()) print_vram(rep[i]);
            });
        });
    }
    std::fprintf(stderr, "saguaro generate: glm5-next serve (M3.1): GEN <max_new> [keys] <ids>, GENI <max_new> [keys] "
                         "<embeddings> <ids>; QUIT to end\n");
    // GENI: the embeddings file is one or more maya-vision records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd
    // floats) in prompt order; their rows stand in for the prompt's <|image|> tokens, in order (GLM-5.3's image
    // tokens take no positions of their own: the rows only replace the embeddings)
    constexpr int64_t kGlmImageToken = 154854;   // <|image|>
    std::vector<int32_t> lp_ctx;   // LOGP: the context the saved state belongs to (the state before its last token)
    // #477 --expert-profile-save: what the expert tiers learned, every --expert-profile-save-every minutes before the
    // next request (a prompt's lent slots are back by then) and after the last one
    Clock::time_point profile_saved_at = Clock::now();
    const auto save_profile = [&](const char* why) {
        if (o.expert_profile_save.empty() || !model.fast()) return;
        std::string e;
        if (model.save_expert_profile(o.expert_profile_save, e))
            std::fprintf(stderr, "saguaro serve: expert profile saved to %s (%s)\n", o.expert_profile_save.c_str(), why);
        else
            std::fprintf(stderr, "saguaro serve: the expert profile was not saved: %s\n", e.c_str());
        profile_saved_at = Clock::now();
    };
    std::string line;
    bool trimmed = true;
    for (;;) {
        if (multi && !pending.empty()) admit_pending();
        if (multi && !gens.empty()) trimmed = false;
        else if (multi && !trimmed && model.elastic() && !in_lines_ready()) {
            idle_trim();
            trimmed = true;
        }
        if (multi && !gens.empty()) {   // requests running: a line when one is there, else one scheduler step
            if (!try_line(line)) {
                struct StepScope {
                    StepScope() { maya::core::progress().busy.store(true); }
                    ~StepScope() { maya::core::progress().busy.store(false); }
                } step_scope;
                multi_step();
                continue;
            }
        } else if (!next_line(line)) {
            break;
        }
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd == "QUIT") break;
        if (multi && !gens.empty() && (cmd == "VLEND" || cmd == "VRECLAIM" || cmd == "LOGP")) {
            std::printf("ERR %s needs the engine idle (%zu requests running)\n", cmd.c_str(), gens.size());
            continue;
        }
        if (cmd == "STOP") continue;   // a stray one between requests: ignore
        if (o.expert_profile_save_min > 0 &&
            Clock::now() - profile_saved_at >= std::chrono::duration<double>(o.expert_profile_save_min * 60.0))
            save_profile("periodic");
        // the watchdog watches a command from here until this iteration ends, whichever way it ends.  A command that
        // ended is progress: back-to-back short commands (the zero-shot suite's LOGP lines, ~0.4 s each on the token
        // path, which has no beat of its own) leave the watchdog's 1 s look almost never on an idle moment, so
        // without this beat it stopped a healthy engine after SAGUARO_WATCHDOG_S of answers
        struct BusyScope {
            BusyScope() { maya::core::progress().busy.store(true); maya::core::progress_at("the request"); }
            ~BusyScope() {
                maya::core::progress_beat();
                maya::core::progress().busy.store(false);
                maya::core::progress_at("idle");
            }
        } busy_scope;
        // LOGP <n_ctx> <id,id,...>: the summed log-probability of the tokens from n_ctx on given those before
        // ("LP <sum> <n> <1 if every one was the most likely>") - the multiple-choice scoring of the zero-shot suite
        // (tools/maya_quant/zs_tasks.py).  A context identical to the previous request's is restored, not read again
        // (the choices of one question share it).
        // VLEND / VRECLAIM (between requests): the on-demand vision encoder is about to start on the first GPU - the
        // expert pool's tail is emptied and freed for it ("VLENT <bytes>") - and has ended ("VRECLAIMED"; an ERR while
        // its memory is not free yet: try again)
        if (cmd == "VLEND") {
            size_t lent = 0;
            if (model.vision_lend(lent, err)) std::printf("VLENT %zu\n", lent);
            else std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        if (cmd == "VRECLAIM") {
            if (model.vision_reclaim(err)) std::printf("VRECLAIMED\n");
            else std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        if (cmd == "LOGP") {
            int64_t n_ctx = 0;
            std::string ids_s;
            ss >> n_ctx >> ids_s;
            std::vector<int64_t> t64;
            std::string e;
            if (n_ctx < 2 || ids_s.empty() || !parse_i64_list(ids_s.c_str(), t64, e) || (int64_t) t64.size() <= n_ctx) {
                std::printf("ERR malformed LOGP line\n");
                std::fflush(stdout);
                continue;
            }
            std::vector<int32_t> t(t64.begin(), t64.end());
            snap_tokens.clear();                  // the saved state is LOGP's now, not a conversation's
            ckpt_drop_past(0);                    // ... and so are the caches the checkpoints stood on
            model.set_host_logits(true);
            const std::vector<int32_t> ctx(t.begin(), t.begin() + n_ctx);
            bool ok = true;
            if (!(model.fast() && ctx == lp_ctx && model.snapshot_restore())) {
                model.reset();
                std::vector<int32_t> head(t.begin(), t.begin() + n_ctx - 1);
                ok = model.prefill(head, err, t[(size_t) n_ctx - 1]);
                if (!ok && err.empty()) {   // no prompt path: the token path, 8 at a time
                    ok = true;
                    std::vector<float> lg0;
                    for (size_t i = 0; ok && i < head.size(); i += 8) {
                        const std::vector<int32_t> part(head.begin() + (long) i,
                                                        head.begin() + (long) std::min(head.size(), i + 8));
                        maya::core::progress_at("LOGP: the context on the token path, at token", (int64_t) i);
                        ok = model.forward(part, lg0, err);
                        maya::core::progress_beat();
                    }
                }
                lp_ctx.clear();
                if (ok && model.fast() && model.snapshot_save()) lp_ctx = ctx;
            }
            std::vector<float> lgv;
            double lp = 0.0;
            bool greedy = true;
            for (int64_t j = n_ctx; ok && j < (int64_t) t.size(); ++j) {
                maya::core::progress_at("LOGP: the continuation, at token", j);
                ok = model.forward({t[(size_t) j - 1]}, lgv, err);
                if (!ok) break;
                maya::core::progress_beat();
                float mx = lgv[0];
                size_t am = 0;
                for (size_t v = 1; v < lgv.size(); ++v)
                    if (lgv[v] > mx) { mx = lgv[v]; am = v; }
                double se = 0.0;
                for (const float x : lgv) se += std::exp((double) x - mx);
                lp += (double) lgv[(size_t) t[(size_t) j]] - mx - std::log(se);
                greedy = greedy && am == (size_t) t[(size_t) j];
            }
            model.set_host_logits(false);
            if (!ok) {
                lp_ctx.clear();
                std::printf("ERR %s\n", err.c_str());
            } else {
                std::printf("LP %.6f %lld %d\n", lp, (long long) (t.size() - (size_t) n_ctx), greedy ? 1 : 0);
            }
            std::fflush(stdout);
            continue;
        }
        if (cmd == "GEN" || cmd == "GENI") {
            if (model.fast()) (void) model.late_check(false);   // (#1026: what the requests before allocated)
            int64_t max_new = 0;
            ss >> max_new;
            maya::kernels::SamplerParams rq = sp;
            bool any_key = false, greedy_req = false, seed_req = false;
            req_think_budget = 0;
            req_think_end = -1;
            req_gt = 0;
            double req_pcie = -1.0;   // setup's calibration: this request's CPU lane (as the engine started: -1 / 0)
            int req_threads = 0;
            int req_chunks = 0;   // ... and its disk reads' pieces (0: as the engine started)
            req_msw = -1.0;   // ... its light-miss bar / first layer (SAGUARO_GLM_MISS_SKIP_W / _FROM; < 0: the engine's)
            req_msf = -1;     // ... and its live Turbo (PLAN-turbo-live; 0/off: the request is exact)
            req_tdrop = 0;
            req_tskip = 0;
            int req_rid = -1;
            req_spec = 0;
            req_spec_min_p = -1.0;
            req_spec_n_min = -1;
            req_spec_cum = -1.0;
            req_bias.clear();
            std::string bias_err;
            int pen_n = 0;
            float pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
            std::string tok2, ids, emb_path;
            while (ss >> tok2) {
                const size_t eq = tok2.find('=');
                if (eq == std::string::npos) {
                    if (cmd == "GENI" && emb_path.empty()) emb_path = tok2;   // GENI: the file, then the ids
                    else ids = tok2;   // the ids list is the last token on the line
                    continue;
                }
                const std::string k = tok2.substr(0, eq), v = tok2.substr(eq + 1);
                if (k == "temperature") {   // temperature=0 (or below): greedy, whatever else the line says
                    rq.temperature = (float) std::atof(v.c_str());
                    if (rq.temperature <= 0.0f) greedy_req = true;
                    else any_key = true;
                }
                else if (k == "top_p") { rq.top_p = (float) std::atof(v.c_str()); any_key = true; }
                else if (k == "top_k") { rq.top_k = std::atoi(v.c_str()); any_key = true; }
                else if (k == "min_p") { rq.min_p = (float) std::atof(v.c_str()); any_key = true; }
                else if (k == "seed") { rq.seed = (uint64_t) std::atoll(v.c_str()); seed_req = rq.seed != 0; }
                else if (k == "think_budget") { req_think_budget = std::atoll(v.c_str()); }
                else if (k == "think_end") { req_think_end = std::atoi(v.c_str()); }
                else if (k == "gt") { req_gt = std::atoll(v.c_str()); }
                else if (k == "pcie_frac") { req_pcie = std::atof(v.c_str()); }
                else if (k == "cpu_threads") { req_threads = std::atoi(v.c_str()); }
                else if (k == "read_chunks") { req_chunks = std::atoi(v.c_str()); }
                else if (k == "miss_skip_w") { req_msw = std::atof(v.c_str()); }
                else if (k == "miss_skip_from") { req_msf = std::atoi(v.c_str()); }
                else if (k == "turbo_drop") { req_tdrop = std::atoi(v.c_str()); }   // PLAN-turbo-live (serve/turbo_live.py)
                else if (k == "turbo_skip") { req_tskip = v == "ram" ? 2 : v == "ssd" ? 1 : 0; }
                else if (k == "rid") { req_rid = std::atoi(v.c_str()); }
                else if (k == "spec") { req_spec = std::atoi(v.c_str()); }
                else if (k == "spec_min_p") { req_spec_min_p = std::atof(v.c_str()); }
                else if (k == "spec_n_min") { req_spec_n_min = std::atoi(v.c_str()); }
                else if (k == "spec_cum") { req_spec_cum = std::atof(v.c_str()); }
                else if (k == "logit_bias") {   // <id>:<bias>,... - -100 or below bans the token
                    for (size_t at = 0; at < v.size() && bias_err.empty();) {
                        size_t end = v.find(',', at);
                        if (end == std::string::npos) end = v.size();
                        const std::string e1 = v.substr(at, end - at);
                        const size_t colon = e1.find(':');
                        char* p1 = nullptr;
                        char* p2 = nullptr;
                        const long id = colon == std::string::npos ? -1 : std::strtol(e1.c_str(), &p1, 10);
                        const float b = colon == std::string::npos ? 0.0f : std::strtof(e1.c_str() + colon + 1, &p2);
                        if (id < 0 || id >= n_vocab || p1 != e1.c_str() + colon || p2 == nullptr || *p2 != '\0' ||
                            !std::isfinite(b))
                            bias_err = "malformed logit_bias entry '" + e1 + "'";
                        else
                            req_bias.emplace_back((int32_t) id, b <= -100.0f ? -INFINITY : b);
                        at = end + 1;
                    }
                }
                // the penalties (#273), llama.cpp's: over the last penalty_last_n tokens the request's state was fed
                // (its prompt, then what it wrote).  They do not choose between greedy and sampled
                else if (k == "penalty_last_n") { pen_n = std::atoi(v.c_str()); }
                else if (k == "penalty_repeat") { pen_repeat = std::strtof(v.c_str(), nullptr); }
                else if (k == "penalty_freq") { pen_freq = std::strtof(v.c_str(), nullptr); }
                else if (k == "penalty_present") { pen_present = std::strtof(v.c_str(), nullptr); }
                else if (k == "logprobs") {   // <k>: an LP line after each T line (logprobs.hpp); no sampler key
                    const int n = std::atoi(v.c_str());
                    rq.lp_k = n < 0 ? -1 : std::min(n, maya::kernels::kTopLogprobs);
                }
            }
            if (any_key) rq.greedy = false;   // any sampler key switches the request to the sampled path
            if (greedy_req) rq.greedy = true;
            // a sampled request without a seed draws a new one (#845: seed 0 for every such request gave the same answer
            // each time); a seed in the request, or --seed when the engine started, stays exact
            if (!rq.greedy && !seed_req && o.seed == 0)
                rq.seed = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
            // a repeat penalty of 0 or below (some clients' "off"; the logits would be divided by it) or a value that is
            // not a number counts as neutral
            if (!std::isfinite(pen_repeat) || pen_repeat <= 0.0f) pen_repeat = 1.0f;
            if (!std::isfinite(pen_freq)) pen_freq = 0.0f;
            if (!std::isfinite(pen_present)) pen_present = 0.0f;
            if (pen_n < 0) pen_n = INT_MAX;   // -1, llama.cpp's "the whole context": every token the window keeps (4096)
            // as llama.cpp: no window, or every penalty neutral, is no penalty - the request's picks are exactly as
            // without the keys (no history kept, nothing uploaded)
            const bool pen_on = pen_n > 0 && (pen_repeat != 1.0f || pen_freq != 0.0f || pen_present != 0.0f);
            if (pen_on) {
                rq.penalty_last_n = pen_n;
                rq.penalty_repeat = pen_repeat;
                rq.penalty_freq = pen_freq;
                rq.penalty_present = pen_present;
            }
            else if (rq.lp_k >= 0) {
                // logprobs: neutral penalties, so every shortcut leaves the picks to pick_tokens (the same tokens) and
                // the request keeps its history (the rows' positions)
                rq.penalty_last_n = 1;
                rq.penalty_repeat = 1.0f;
                rq.penalty_freq = 0.0f;
                rq.penalty_present = 0.0f;
            }
            if (!bias_err.empty()) {
                if (multi && req_rid >= 0) std::printf("ERR %d %s\n", req_rid, bias_err.c_str());
                else std::printf("ERR %s\n", bias_err.c_str());
                std::fflush(stdout);
                continue;
            }
            if (!req_bias.empty() && (rq.greedy || rq.temperature <= 0.0f)) {
                // greedy with a bias: the sampled path's top_k = 1 is the same argmax (ties to the lower id) over the
                // biased logits - the greedy shortcuts (the argmax at the end of a forward, the windows' argmax) read
                // the unbiased ones.  (Greedy with penalties stays greedy: every shortcut leaves a request with
                // penalties to the sampler, whose greedy kernel counts them - the same pick, ~75 us less a token)
                rq.greedy = false;
                rq.temperature = 1.0f;
                rq.top_k = 1;
                rq.top_p = 1.0f;
                rq.min_p = 0.0f;
            }
            // the CPU lane for this request: what it names, else what the engine started with
            model.set_pcie_share(req_pcie >= 0.0 && req_pcie <= 1.0 ? req_pcie : -1.0);
            model.set_cpu_lane_threads(req_threads > 0 ? req_threads : 0);
            model.set_read_chunks(req_chunks > 0 ? req_chunks : 0);
            std::vector<int64_t> toks;
            std::string e;
            if (max_new <= 0 || ids.empty() || !parse_i64_list(ids.c_str(), toks, e)) {
                if (multi && req_rid >= 0) std::printf("ERR %d malformed GEN line\n", req_rid);
                else std::printf("ERR malformed GEN line\n");
                continue;
            }
            lp_ctx.clear();   // a generation saves its own state over LOGP's
            cur_img_pos.clear();
            cur_img_rows.clear();
            if (cmd == "GENI") {
                const int64_t E = model.geometry().n_embd;
                std::FILE* f = std::fopen(emb_path.c_str(), "rb");
                std::string ge = f ? "" : "cannot open " + emb_path;
                while (f && ge.empty()) {
                    int32_t hdr[5];
                    // the end of the records only between two of them: a header cut short is a broken file, and a
                    // record cannot hold more rows than the prompt has tokens (not a multi-GB allocation off one)
                    const size_t got = std::fread(hdr, 1, sizeof hdr, f);
                    if (got == 0 && std::feof(f)) break;
                    if (got != sizeof hdr || hdr[0] != 0x31455653 || hdr[1] <= 0 || hdr[4] != E ||
                        (size_t) hdr[1] > toks.size() - cur_img_rows.size() / (size_t) E) {
                        ge = "not a maya-vision record for this model";
                        break;
                    }
                    const size_t at = cur_img_rows.size();
                    cur_img_rows.resize(at + (size_t) hdr[1] * (size_t) E);
                    if (std::fread(cur_img_rows.data() + at, sizeof(float) * (size_t) E, (size_t) hdr[1], f) !=
                        (size_t) hdr[1])
                        ge = "short embeddings file";
                }
                if (f) std::fclose(f);
                for (size_t i = 0; ge.empty() && i < toks.size(); ++i)
                    if (toks[i] == kGlmImageToken) cur_img_pos.push_back((int64_t) i);
                if (ge.empty() && (int64_t) cur_img_pos.size() * E != (int64_t) cur_img_rows.size())
                    ge = std::to_string(cur_img_pos.size()) + " image tokens, " +
                         std::to_string(cur_img_rows.size() / (size_t) E) + " image rows";
                if (!ge.empty()) {
                    cur_img_pos.clear();
                    cur_img_rows.clear();
                    if (multi && req_rid >= 0) std::printf("ERR %d %s\n", req_rid, ge.c_str());
                    else std::printf("ERR %s\n", ge.c_str());
                    std::fflush(stdout);
                    continue;
                }
            }
            if (multi) {
                if (req_rid < 0) std::printf("ERR this engine runs several conversations: GEN needs rid=<id>\n");
                else gen_admit(req_rid, toks, max_new, rq);
                continue;
            }
            {
                std::string be;
                rq.logit_bias = model.logit_bias_table(0, req_bias, be);
                if (!be.empty()) {
                    std::printf("ERR %s\n", be.c_str());
                    std::fflush(stdout);
                    continue;
                }
            }
            // PLAN-turbo-live: one conversation (multi branched off above): the request's bars go on the engine's
            // own sequence HERE, past every error path of the line (a bad embeddings file, a bad logit_bias) - a
            // request that dies on one of those never leaves its drop/skip behind for the next command to route
            // with (the clear below runs whenever a run began).  Several conversations: the bars travel to the
            // context gen_admit picks (and through `pending`), set and cleared there.
            model.set_miss_skip(model.seq_index(), req_msw, req_msf);
            model.set_turbo(model.seq_index(), req_tdrop, req_tskip);
            model.set_image_rows(cur_img_pos, cur_img_rows);
            run_request(toks, max_new, rq);
            model.set_image_rows({}, {});
            model.set_miss_skip(model.seq_index(), -1.0, -1);   // (PLAN-turbo-live: the next request sets its own; LOGP runs exact)
            model.set_turbo(model.seq_index(), 0, 0);
            continue;
        }
        if (cmd.empty()) continue;
        std::printf("ERR unsupported command '%s' (GEN, GENI, LOGP, VLEND, VRECLAIM, STOP and QUIT)\n", cmd.c_str());
    }
    reader_abort();
    // the engine ends (QUIT, or the server gone): the conversation in the model is kept for the next start
    if (multi) {   // the requests still running end here (their DONE lines), then every context's conversation
        for (const Pending& w : pending) std::printf("DONE %d 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", w.rid, w.ids.size());
        pending.clear();
        multi_drain();
        for (auto& gp : gens)
            if (!gp->done) {
                gp->finish = "cancel";
                gen_finish(*gp);
            }
        gen_sweep();
        if (slot_keep && !slot_ram)
            for (int c = 0; c < (int) ctxs.size(); ++c) {
                bind_ctx(c);
                slot_put(" on shutdown");
            }
        return 0;
    }
    if (slot_keep && !slot_ram) slot_put(" on shutdown");
    save_profile("quit");   // #477
    maya::core::Glm5Model::set_routing_dump("", err);   // --dump-routing: the trace closed whole
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    {
        // the GLM tuning settings taken from the environment, said once at start (#6), with the value used where the
        // engine clamps one (the same ranges as where each is read)
        namespace gf = maya::kernels::glmf;
        static const maya::platform::EnvRange kRanges[] = {
            {"GLM_PREFETCH_N", 0, gf::kSpares - 1, "at most kSpares-1"},          // glm_fast_path.cu
            {"GLM_AHEAD", 0, gf::kAhead, "at most kAhead"},                       // glm_fast_path.cu
            {"GLM_DISK_SKIP_FROM", 0, 8, "0-8"},                                  // glm_fast_path.cu
            {"GLM_PREFILL_SUB", 16, 8192, "16-8192"}};                            // glm_prefill.cu
#if defined(_WIN32)
        char** env = _environ;
#else
        char** env = environ;
#endif
        const std::string line = maya::platform::env_report(env, "GLM_", "glm settings from the environment",
                                                              kRanges, sizeof(kRanges) / sizeof(kRanges[0]));
        if (!line.empty()) std::fprintf(stderr, "%s\n", line.c_str());
    }
    // **UNBUFFERED, BECAUSE THE INTERESTING OUTPUT IS THE OUTPUT BEFORE A CRASH.**  `stdout` redirected to a
    // pipe or a file is block-buffered, so a program that dies loses every line it had already printed - which
    // turns "it crashed at step 7" into "it crashed somewhere", and the difference is a debugging session.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // Load every CUDA kernel when the context is created, before the expert cache takes the free VRAM.  With the
    // default lazy loading, a kernel first used mid-prompt (MMQ for IQ3_XXS at 64K+ on a 12 GB card) found no VRAM
    // left for its code and the engine ended ("out of memory: cudaFuncSetAttribute").  Costs ~30 MB of VRAM.
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
#if defined(_WIN32)
        _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);
#endif
    }
#if defined(_WIN32)
    // Windows' power throttling (EcoQoS) runs a process whose windows are all in the background on a hybrid CPU's
    // efficiency cores at low clocks, and the CPU lane crawled there - an i7-13700: decode 9-12 tok/s with Maya's
    // window focused, ~2 tok/s behind other windows, the same either way once `powercfg /powerthrottling disable`
    // named the engine (#114).  The engine opts out itself.  Resolved at runtime as PrefetchVirtualMemory is
    // (expert_source.cpp): the struct is PROCESS_POWER_THROTTLING_STATE (version 1; EXECUTION_SPEED managed, and off)
    // and 4 is ProcessPowerThrottling - a Windows without them refuses the call and nothing changes
    {
        struct Throttle { ULONG version, control, state; } off{1u, 0x1u, 0u};
        using SetInfoFn = BOOL (WINAPI*)(HANDLE, int, LPVOID, DWORD);
        const HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
        const SetInfoFn set_info = k32 ? reinterpret_cast<SetInfoFn>(GetProcAddress(k32, "SetProcessInformation")) : nullptr;
        if (set_info != nullptr) set_info(GetCurrentProcess(), 4, &off, (DWORD) sizeof off);
    }
#endif
    Options o;
    bool have_tokens = false;
    bool have_logits_stride = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        bool handled = true;
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (a == "--pack") o.pack = next("--pack");
        else if (a == "--glm-pack") o.glm_pack = next("--glm-pack");
        else if (a == "--glm-ram-resident") {
#if defined(_WIN32)
            _putenv_s("SAGUARO_GLM_RAM_RESIDENT", "1");
#else
            setenv("SAGUARO_GLM_RAM_RESIDENT", "1", 1);
#endif
        }
        else if (a == "--parallel") {   // the sequence states (SAGUARO_GLM_SEQS): n conversations at once
            const int n = std::atoi(next("--parallel"));
            if (n < 1 || n > 8) { std::fprintf(stderr, "--parallel takes 1..8\n"); return 2; }
            set_env("SAGUARO_GLM_SEQS", std::to_string(n).c_str());
        }
        else if (a == "--glm-ram-slack") {
            const char* v = next("--glm-ram-slack");
#if defined(_WIN32)
            _putenv_s("SAGUARO_GLM_RAM_SLACK", v);
#else
            setenv("SAGUARO_GLM_RAM_SLACK", v, 1);
#endif
        }
        else if (a == "--tokens") {
            if (have_tokens) { std::fprintf(stderr, "supply one token input only\n"); return 2; }
            std::string e;
            if (!parse_i64_list(next("--tokens"), o.tokens, e)) { std::fprintf(stderr, "%s\n", e.c_str()); return 2; }
            have_tokens = true;
        }
        else if (a == "--tokens-file") {
            if (have_tokens) { std::fprintf(stderr, "supply one token input only\n"); return 2; }
            std::ifstream input(next("--tokens-file"));
            if (!input) { std::fprintf(stderr, "cannot open token file\n"); return 2; }
            std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            if (input.bad()) { std::fprintf(stderr, "cannot read token file\n"); return 2; }
            std::string error;
            if (text.find('\0') != std::string::npos || !parse_i64_list(text.c_str(), o.tokens, error)) {
                std::fprintf(stderr, "malformed token file: %s\n", error.c_str()); return 2;
            }
            have_tokens = true;
        }
        else if (a == "--max-new") o.max_new = std::atoll(next("--max-new"));
        else if (a == "--max-context") o.max_context = std::atoll(next("--max-context"));
        else if (a == "--greedy") o.greedy = true;
        else if (a == "--seed") { o.seed = (uint64_t) std::atoll(next("--seed")); o.greedy = false; }
        else if (a == "--top-k") o.top_k = std::atoi(next("--top-k"));
        else if (a == "--top-p") o.top_p = (float) std::atof(next("--top-p"));
        else if (a == "--temperature") o.temperature = (float) std::atof(next("--temperature"));
        else if (a == "--dump-logits") o.dump_logits = next("--dump-logits");
        else if (a == "--logits-stride") {
            if (have_logits_stride) { std::fprintf(stderr, "--logits-stride must be supplied only once\n"); return 2; }
            if (!maya::program::logits_selection::parse_stride(next("--logits-stride"), o.logits_stride)) {
                std::fprintf(stderr, "--logits-stride requires a positive decimal int64\n"); return 2;
            }
            have_logits_stride = true;
        }
        else if (a == "--dump-residual") o.dump_residual = next("--dump-residual");
        else if (a == "--dump-mixed") o.dump_mixed = next("--dump-mixed");
        else if (a == "--dump-layers") o.dump_layers = next("--dump-layers");
        else if (a == "--dump-halves") o.dump_halves = next("--dump-halves");
        else if (a == "--dump-routing") o.dump_routing = next("--dump-routing");
        else if (a == "--ple-gguf") o.ple_gguf = next("--ple-gguf");
        else if (a == "--no-ple") o.no_ple = true;
        else if (a == "--ple-io") o.ple_io = next("--ple-io");
        else if (a == "--ple-row-cache") o.ple_row_cache = std::atoll(next("--ple-row-cache"));
        else if (a == "--ple-inflight") o.ple_inflight = std::atoi(next("--ple-inflight"));
        else if (a == "--ple-delay-us") o.ple_delay_us = std::atof(next("--ple-delay-us"));
        else if (a == "--ple-sync-submit") o.ple_sync_submit = true;
        else if (a == "--kv") o.kv = next("--kv");
        else if (a == "--kv-resident") o.kv_resident = std::atoll(next("--kv-resident"));
        else if (a == "--kv-streaming") o.kv_streaming = next("--kv-streaming");
        else if (a == "--stream-token") o.stream_token = true;
        else if (a == "--check-logits") o.check_logits = true;
        else if (a == "--gr-fp32-activations") o.gr_fp32_activations = true;
        else if (a == "--gr-native-mmvf") o.gr_native_mmvf = true;
        else if (a == "--native-bf16") o.native_bf16 = true;
        else if (a == "--native-bf16-extra") o.native_bf16_extra = true;
        else if (a == "--native-ple-key") o.native_ple_key = true;
        else if (a == "--native-moe-combine") o.native_moe_combine = true;
        else if (a == "--native-gdn") o.native_gdn = true;
        else if (a == "--native-flash-attn-short") o.native_flash_attn_short = true;
        else if (a == "--native-qsa-indexer") o.native_qsa_indexer = true;
        else if (a == "--native-qsa") o.native_qsa = true;
        else if (a == "--native-rope") o.native_rope = true;
        else if (a == "--native-ple-postops") o.native_ple_postops = true;
        else if (a == "--native-router") o.native_router = true;
        else if (a == "--cpu-oracle-q8-0") o.cpu_oracle_q8_0 = true;
        else if (a == "--native") o.native_preset = next("--native");
        else if (a == "--native-head-gguf") o.native_head_gguf = next("--native-head-gguf");
        else if (a == "--native-dense-gguf") o.native_dense_gguf.push_back(next("--native-dense-gguf"));
        else if (a == "--no-capture") o.no_capture = true;
        else if (a == "--no-pool") o.no_pool = true;
        else if (a == "--sync-every-layer") o.sync_every_layer = true;
        else if (a == "--stage-timing") o.stage_timing = true;
        else if (a == "--graph-only") o.graph_only = true;
        else if (a == "--gpu-only-full") o.gpu_only_full = true;
        else if (a == "--pool-workers") o.pool_workers = std::atoi(next("--pool-workers"));
        else if (a == "--no-host-worker") o.no_host_worker = true;
        else if (a == "--no-ple-prefetch") o.no_ple_prefetch = true;
        else if (a == "--expert-cache") {
            const std::string v = next("--expert-cache");
            o.expert_cache = (v == "auto") ? -1 : std::atoi(v.c_str());
        }
        else handled = false;
        if (!handled) {   // (a second chain: MSVC stops an if-else chain this long, C1061 "blocks nested too deeply")
        if (a == "--expert-cache-device1") o.expert_cache_remote[0] = std::atoi(next("--expert-cache-device1"));
        else if (a == "--expert-cache-device2") o.expert_cache_remote[1] = std::atoi(next("--expert-cache-device2"));
        else if (a == "--expert-cache-device3") o.expert_cache_remote[2] = std::atoi(next("--expert-cache-device3"));
        else if (a == "--expert-cache-remote-placement")
            o.expert_cache_remote_placement = next("--expert-cache-remote-placement");
        else if (a == "--vram-reserve-mib") {
            o.vram_reserve_mib = std::atoi(next("--vram-reserve-mib"));
            o.vram_reserve_given = true;
        }
        else if (a == "--vram-reserve-later-mib") o.vram_reserve_later_mib = std::atoi(next("--vram-reserve-later-mib"));
        else if (a == "--prefill") {
            const std::string v = next("--prefill");
            o.prefill_arg = v;
            o.prefill_auto = v == "auto";
            o.prefill_chunk = o.prefill_auto ? 8192 : std::atoll(v.c_str());
        }
        else if (a == "--no-split-rows") o.no_split_rows = true;
        else if (a == "--no-prefill-borrow") o.no_prefill_borrow = true;
        else if (a == "--prefill-until") o.prefill_until = std::atoll(next("--prefill-until"));
        else if (a == "--dump-final-r") o.dump_final_r = next("--dump-final-r");
        else if (a == "--spec") o.spec = std::atoi(next("--spec"));
        else if (a == "--spec-oracle") o.spec_oracle = next("--spec-oracle");
        else if (a == "--spec-follow") o.spec_follow = next("--spec-follow");
        else if (a == "--spec-follow-save") o.spec_follow_save = next("--spec-follow-save");
        else if (a == "--spec-corrupt") o.spec_corrupt = std::atoi(next("--spec-corrupt"));
        else if (a == "--mtp") o.mtp = next("--mtp");
        else if (a == "--mtp-window") o.mtp_window = std::atoll(next("--mtp-window"));
        else if (a == "--pcie-frac") o.pcie_frac = std::atof(next("--pcie-frac"));
        else if (a == "--adapt-every") o.adapt_every = std::atoi(next("--adapt-every"));
        else if (a == "--spec-min-p") {
            o.spec_min_p = std::atof(next("--spec-min-p"));
            o.spec_min_p_set = true;
        }
        else if (a == "--stop-eos") { o.stop_eos = true; o.eos_given = true; }
        else if (a == "--spec-split") o.spec_split = true;
        else if (a == "--layer-split") o.layer_split = next("--layer-split");
        else if (a == "--dflash") o.dflash = next("--dflash");
        else if (a == "--split-device") o.split_device = next("--split-device");
        else if (a == "--pcie-mode") o.pcie_mode = next("--pcie-mode");
        else if (a == "--serve") o.serve = true;
        else if (a == "--vision") o.vision = true;
        else if (a == "--prompt-cache") o.prompt_cache = std::max(0, std::atoi(next("--prompt-cache")));
        else if (a == "--prompt-cache-every") o.prompt_cache_every = std::max(0LL, std::atoll(next("--prompt-cache-every")));
        else if (a == "--prompt-cache-root") o.prompt_cache_root = std::max(0LL, std::atoll(next("--prompt-cache-root")));
        else if (a == "--turn-token") o.turn_token = std::atoll(next("--turn-token"));
        else if (a == "--short-read") o.short_read = std::max(0LL, std::atoll(next("--short-read")));
        else if (a == "--suffix-draft") o.suffix_draft = std::max(0, std::atoi(next("--suffix-draft")));
        else if (a == "--mtp-max-t") o.mtp_max_t = std::max(0, std::atoi(next("--mtp-max-t")));
        else if (a == "--control-vector") o.cvec_files.push_back({next("--control-vector"), 1.0f});
        else if (a == "--control-vector-scaled") {
            // FILE:SCALE, comma-separated; the LAST colon splits, so a Windows path (C:\...) keeps its drive
            std::stringstream list(next("--control-vector-scaled"));
            std::string item;
            while (std::getline(list, item, ',')) {
                const size_t colon = item.rfind(':');
                char* end = nullptr;
                const float sc = colon == std::string::npos ? 0.0f : std::strtof(item.c_str() + colon + 1, &end);
                if (colon == std::string::npos || colon == 0 || end == item.c_str() + colon + 1 || *end != '\0') {
                    std::fprintf(stderr, "--control-vector-scaled: expected FILE:SCALE, got '%s'\n", item.c_str());
                    return 2;
                }
                o.cvec_files.push_back({item.substr(0, colon), sc});
            }
        }
        else if (a == "--control-vector-layer-range") {
            o.cvec_first = std::atoi(next("--control-vector-layer-range"));
            o.cvec_last = std::atoi(next("--control-vector-layer-range"));
        }
        else if (a == "--cvec-mode") {
            const std::string m = next("--cvec-mode");
            if (m == "project") o.cvec_mode = 0;
            else if (m == "add") o.cvec_mode = 1;
            else { std::fprintf(stderr, "--cvec-mode: add or project, got '%s'\n", m.c_str()); return 2; }
        }
        else if (a == "--cvec-dir") {
            const std::string d = next("--cvec-dir");
            if (d == "per-layer") o.cvec_single = -1;
            else if (d.rfind("single:", 0) == 0) o.cvec_single = std::atoi(d.c_str() + 7);
            else { std::fprintf(stderr, "--cvec-dir: per-layer or single:L, got '%s'\n", d.c_str()); return 2; }
        }
        else if (a == "--no-spec-split") o.spec_split = false;
        else if (a == "--eos-ids") {
            std::string e;
            if (!parse_i64_list(next("--eos-ids"), o.eos_ids, e)) { std::fprintf(stderr, "--eos-ids: %s\n", e.c_str()); return 2; }
            o.stop_eos = true;
            o.eos_given = true;
        }
        else if (a == "--adapt-swaps") o.adapt_swaps = std::atoi(next("--adapt-swaps"));
        else if (a == "--expert-cache-cpu-order") o.expert_cache_cpu_order = true;
        else if (a == "--expert-cache-per-layer") o.expert_cache_per_layer = true;
        else if (a == "--no-hit-poke") o.no_hit_poke = true;
        else if (a == "--expert-profile") o.expert_profile = next("--expert-profile");
        else if (a == "--expert-profile-save") o.expert_profile_save = next("--expert-profile-save");
        else if (a == "--expert-profile-save-every")
            o.expert_profile_save_min = std::atof(next("--expert-profile-save-every"));
        else if (a == "--gpu-stages") o.gpu_stages = true;
        else if (a == "--mmap-experts") o.mmap_experts = true;
        else if (a == "--experts-from-gguf") {
            o.experts_from_gguf = next("--experts-from-gguf");
            if (o.experts_from_gguf != "auto" && o.experts_from_gguf != "on" && o.experts_from_gguf != "off") {
                std::fprintf(stderr, "saguaro generate: --experts-from-gguf takes auto, on or off\n");
                return 2;
            }
        }
        else if (a == "--resident-cpu-experts") o.resident_cpu_experts = true;
        else if (a == "--stats") o.stats = true;
        else if (a == "--shared-late") o.shared_late = true;
        else if (a == "--keep-canonical") o.keep_canonical = true;
        else if (a == "--no-token-graph") o.no_token_graph = true;
        else if (a == "--no-fused-gr") o.no_fused_gr = true;
        else if (a == "--no-fast-attn") o.no_fast_attn = true;
        else if (a == "--no-publish-kernel") o.no_publish_kernel = true;
        else if (a == "--no-fused-gdn") o.no_fused_gdn = true;
        else if (a == "--no-fast-select") o.no_fast_select = true;
        else {
            // An unknown flag is an ERROR and not a warning: a typo'd `--max-neww` that silently generated 16
            // tokens would look like a working run.
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage();
            return 2;
        }
        }
    }
    // ---- the glm5-next pack path (M3.1): BEFORE every qwen4exp-shaped check below (this driver's
    // session/graph machinery is qwen4exp's; a GLM pack runs the verified Glm5Model runner instead,
    // see glm_pack_generate).  --glm-pack with an explicit non-default --pack is almost certainly a
    // mistake worth saying out loud.
    // --kv-streaming on|off: on streams with --kv-resident's window (32768 when it gives none), off keeps every cell in
    // VRAM whatever --kv-resident says
    if (!o.kv_streaming.empty()) {
        if (o.kv_streaming == "on") {
            if (o.kv_resident <= 0) o.kv_resident = 32768;
        } else if (o.kv_streaming == "off") {
            o.kv_resident = 0;
        } else {
            std::fprintf(stderr, "saguaro generate: --kv-streaming takes on or off (not %s)\n", o.kv_streaming.c_str());
            return 2;
        }
    }
    if (!o.glm_pack.empty()) {
        // --kv int8 on the GLM path: the INT8 latent cache (Glm5Model::lat_q8_, read at load)
        if (o.kv == "int8" && std::getenv("SAGUARO_GLM_KV_INT8") == nullptr) set_env("SAGUARO_GLM_KV_INT8", "1");
        // --kv-resident N: KV streaming of the DSA latent cache (Glm5Model::kv_resident_cells, read at load)
        if (o.kv_resident < 0) {
            std::fprintf(stderr, "saguaro generate: --kv-resident must be 0 or more\n");
            return 2;
        }
        if (o.kv_resident > 0 && std::getenv("SAGUARO_GLM_KV_RESIDENT") == nullptr)
            set_env("SAGUARO_GLM_KV_RESIDENT", std::to_string(o.kv_resident).c_str());
        if (o.kv_streaming == "off") set_env("SAGUARO_GLM_KV_RESIDENT", "0");   // (off whatever the environment says)
        // the expert pool's size, set in the config's args (the engine reads them as SAGUARO_GLM_EXPERT_CACHE,
        // SAGUARO_GLM_RESERVE_MB and SAGUARO_GLM_RESERVE_LATER_MB; given here, they win over the environment):
        // --expert-cache auto | N (0 = auto), --vram-reserve-mib N, --vram-reserve-later-mib N
        if (o.expert_cache < -1) {
            std::fprintf(stderr, "saguaro generate: --expert-cache takes auto or a number of experts\n");
            return 2;
        }
        if ((o.vram_reserve_given && o.vram_reserve_mib < 0) || o.vram_reserve_later_mib < -1) {
            std::fprintf(stderr, "saguaro generate: --vram-reserve-mib and --vram-reserve-later-mib take 0 or more MiB\n");
            return 2;
        }
        if (o.expert_cache != 0)
            set_env("SAGUARO_GLM_EXPERT_CACHE", o.expert_cache < 0 ? "auto" : std::to_string(o.expert_cache).c_str());
        if (o.vram_reserve_given) set_env("SAGUARO_GLM_RESERVE_MB", std::to_string(o.vram_reserve_mib).c_str());
        if (o.vram_reserve_later_mib >= 0)
            set_env("SAGUARO_GLM_RESERVE_LATER_MB", std::to_string(o.vram_reserve_later_mib).c_str());
        if (o.pack != "pack/full") {
            std::fprintf(stderr, "saguaro generate: note: --glm-pack supersedes --pack (%s ignored)\n", o.pack.c_str());
        }
        return glm_pack_generate(o);
    }
    // **THE QWEN DRIVER IS GONE.**  What followed the glm-pack branch above - the qwen4exp session/graph
    // machinery, the serve loop, the verifier, the drafter, PLE, QSA and the KV streaming - was removed
    // (T8b): Qwen3.8-Flash-Next runs on saguaro-qfn, not on this engine.
    std::fprintf(stderr, "saguaro generate: this model runs on saguaro-qfn\n");
    return 2;
}
