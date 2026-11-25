#pragma once
#include <cstdint>

namespace strata::kernels {
// Configure before session capture. Existing graphs retain their selected kernels.
// This experiment is off by default and does not modify the legacy router.
void native_router_set_enabled(bool enabled);
bool native_router_enabled();

// Pinned CUDA topk-moe contract for ONE token, 512 experts, top 10, softmax,
// no selection bias, lower normalization clamp 2^-14, and scale 1.
// Reads 512 finite F32 logits; writes 10 I32 IDs and 10 F32 weights. Equal
// computed probabilities select the lower expert index. All spans must be
// four-byte aligned and outputs disjoint from each other and the input.
// Requires a nonnull ordered CUDA stream. No allocation or synchronization.
// PLAN-turbo-live's per-request lossy options, one bundle per (layer, row group).  All pointers are device, all
// nullable, and every one of them is read per row from the window's mapped upload: a row with drop 0 and skip 0
// routes with today's bits, bit for bit.
//   drop[row]: the row keeps its 10-N heaviest picks; a dropped pick is written as the row's top-1 id at weight 0.
//   skip[row]: 1 = a pick not resident in VRAM (`res`) is left out the same way, 2 = a pick in NEITHER VRAM nor
//              the RAM tier (`res` + `ram`); the row's top-1 always stays (fetched if it must be), the kept
//              weights renormalise as the drop's.  Without `res` nothing can be left out and the bar is inert.
//   stats[tag*3 + 0..2]: per counter bucket the picks routed, dropped and skipped (the engine's per-request
//              line); counted only for rows with drop or skip on.
//   skip_usage[layer * n_expert + expert]: the skipped picks' usage, so the adaptive tier still learns what a
//              skipping request asked for (counted for the picks skip left out, not the ones drop dropped).
struct TurboRoute {
    const int32_t* drop = nullptr;
    const int32_t* skip = nullptr;
    const int32_t* res = nullptr;        ///< this layer's residency row: slot or -1
    const int32_t* ram = nullptr;        ///< this layer's RAM-tier row: >= 0 = the RAM copy holds it (skip 2)
    const int32_t* tag = nullptr;
    unsigned long long* stats = nullptr;
    float* skip_usage = nullptr;
    int32_t layer = -1;
    int32_t n_expert = 0;
};
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream,
                         const TurboRoute* tr = nullptr);
/// n_tok rows at once (logits [n,512], ids/weights [n,10]); each row exactly as the single call.
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream,
                               const TurboRoute* tr = nullptr);
// The same drop and skip after any other top-k (ranks in descending weight order): the last drop[row] picks of
// each row, and every pick the row's skip bar leaves out, become the row's top-1 at weight 0, and the kept
// weights are renormalised to their sum (clamped as the router's).  A row nothing is left out of is untouched -
// the router's own bits.
void turbo_drop_apply(int32_t* ids, float* weights, int n_tok, int k, const TurboRoute& tr, void* stream);
/// The same contract for a model family's (experts, top-k): 512 / 10 (Flash-Next: the two calls above, unchanged) and
/// 256 / 8 (Qwen3.6-35B-A3B).  logits [n_tok, n_expert], ids / weights [n_tok, k].  CUDA / HIP.
bool native_router_topk_supported(int n_expert, int k);
void native_router_topk(const float* logits, int32_t* ids, float* weights, int n_expert, int k, int n_tok, void* stream);
/// SAGUARO_ROUTE_RESIDENT: swap non-resident tail-rank experts for resident ones within `margin` logits (see the .cu).
/// `stats` is nullptr or 4 device uint64 counters: tail entries seen, swaps, non-resident before, non-resident after.
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok, float margin,
                           int rank_lo, int rank_hi, unsigned long long* stats, void* stream);
}
