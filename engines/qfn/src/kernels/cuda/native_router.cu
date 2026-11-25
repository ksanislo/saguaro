// Adapted from topk-moe.cu/common.cuh in llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d; finite F32, 512-expert/10-output path.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/kernels/native_router.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}
__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}
// PLAN-turbo-live: the picks a row keeps (10 - its request's drop; no table: all 10)
__device__ __forceinline__ int turbo_keep(const int32_t* drop, int row) {
    if (drop == nullptr) return 10;
    const int d = drop[row];
    return d <= 0 ? 10 : d >= 9 ? 1 : 10 - d;
}
// PLAN-turbo-live skip: is `e` served without leaving this card?  bar 1: resident in VRAM; bar 2: resident or
// held by the RAM tier.  No table: nothing can be left out (the engine would have run the request exact).
__device__ __forceinline__ bool tier_resident(const TurboRoute& tr, int bar, int e) {
    if (tr.res == nullptr) return true;
    if (tr.res[e] >= 0) return true;
    return bar >= 2 && tr.ram != nullptr && tr.ram[e] >= 0;
}
// the per-request counters of one token: [bucket*3] picks, +1 dropped, +2 skipped - only for rows with a turbo on
__device__ __forceinline__ void turbo_count(const TurboRoute& tr, int row, int n_pick, int n_drop, int n_skip) {
    if (tr.stats == nullptr) return;
    if ((tr.drop != nullptr && tr.drop[row] != 0) || (tr.skip != nullptr && tr.skip[row] != 0)) {
        const int bucket = tr.tag != nullptr ? tr.tag[row] : 0;
        if (bucket < 0 || bucket >= 16) return;   // the engine hands out 0 (solo) and 1+slot; nothing else
        atomicAdd(tr.stats + (size_t) bucket * 3 + 0, (unsigned long long) n_pick);
        atomicAdd(tr.stats + (size_t) bucket * 3 + 1, (unsigned long long) n_drop);
        atomicAdd(tr.stats + (size_t) bucket * 3 + 2, (unsigned long long) n_skip);
    }
}
__launch_bounds__(256, 1)
__global__ void route(const float* __restrict__ logits, int32_t* __restrict__ ids,
                      float* __restrict__ weights, const TurboRoute tr) {
    // Preserve the pinned 32x8 block geometry; only row zero is active here.
    // blockIdx.x = the token (a multi-token launch; 0 for the single one)
    logits += (size_t) blockIdx.x * 512; ids += (size_t) blockIdx.x * 10; weights += (size_t) blockIdx.x * 10;
    if (threadIdx.y != 0) return;
    const int lane = threadIdx.x;
    const int keep = turbo_keep(tr.drop, (int) blockIdx.x);
    const int bar = tr.skip != nullptr ? tr.skip[blockIdx.x] : 0;
    int n_drop = 0, n_skip = 0;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncthreads();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    bool alive_lane = true;
    int top1 = 0;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if (rank == 0) top1 = expert;
        // Turbo drop / skip: a pick the row does not run aliases the top-1 at weight 0.  The top-1 itself always
        // stays (fetched if it must be); bar 0 keeps today's drop-only arithmetic, bit for bit.
        bool alive = rank < keep;
        if (alive && rank > 0 && bar != 0 && !tier_resident(tr, bar, expert)) alive = false;
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = alive ? expert : top1;
            // Deliberately accumulate by WINNING EXPERT lane, not output rank.
            // Multiple selected experts in one lane add in selection order.
            if (alive) selected_sum += best;
        }
        if (rank == lane) { selected = best; alive_lane = alive; }
        if (!alive && rank > 0) {
            if (rank >= keep) ++n_drop;
            else {
                ++n_skip;
                if (lane == 0 && tr.skip_usage != nullptr && tr.layer >= 0)
                    atomicAdd(&tr.skip_usage[(size_t) tr.layer * tr.n_expert + expert], 1.0f);
            }
        }
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane == 0) turbo_count(tr, (int) blockIdx.x, 10, n_drop, n_skip);
    if (lane < 10) weights[lane] = (lane < keep && alive_lane) ? selected * inverse_selected_sum : 0.0f;
}
__launch_bounds__(256, 1)
__global__ void route_multi(const float* __restrict__ logits, int32_t* __restrict__ ids,
                            float* __restrict__ weights, int n_tok, const TurboRoute tr) {
    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;
    if (tk >= n_tok) return;
    logits += (size_t) tk * 512; ids += (size_t) tk * 10; weights += (size_t) tk * 10;
    const int lane = threadIdx.x;
    const int keep = turbo_keep(tr.drop, tk);
    const int bar = tr.skip != nullptr ? tr.skip[tk] : 0;
    int n_drop = 0, n_skip = 0;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncwarp();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    bool alive_lane = true;
    int top1 = 0;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if (rank == 0) top1 = expert;
        bool alive = rank < keep;
        if (alive && rank > 0 && bar != 0 && !tier_resident(tr, bar, expert)) alive = false;
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = alive ? expert : top1;
            if (alive) selected_sum += best;
        }
        if (rank == lane) { selected = best; alive_lane = alive; }
        if (!alive && rank > 0) {
            if (rank >= keep) ++n_drop;
            else {
                ++n_skip;
                if (lane == 0 && tr.skip_usage != nullptr && tr.layer >= 0)
                    atomicAdd(&tr.skip_usage[(size_t) tr.layer * tr.n_expert + expert], 1.0f);
            }
        }
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane == 0) turbo_count(tr, tk, 10, n_drop, n_skip);
    if (lane < 10) weights[lane] = (lane < keep && alive_lane) ? selected * inverse_selected_sum : 0.0f;
}
// The same contract for other (experts, top-k) shapes - Qwen3.6-35B-A3B's 256 / 8 - one warp per token, n_tok tokens
// (8 per block).  The 512 / 10 path keeps its own kernels above (the Flash-Next bits), this template is only used
// for the other shapes.
template <int NE, int K>
__launch_bounds__(256, 1)
__global__ void route_topk(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                           int n_tok) {
    static_assert(NE % 32 == 0 && K <= 32, "experts in whole warps, at most 32 picks");
    constexpr int V = NE / 32;
    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;
    if (tk >= n_tok) return;
    logits += (size_t) tk * NE; ids += (size_t) tk * K; weights += (size_t) tk * K;
    const int lane = threadIdx.x;
    float values[V];
#pragma unroll
    for (int i = 0; i < V; ++i) values[i] = logits[lane + i * 32];
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < V; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < V; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < K; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < V; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < K) weights[lane] = selected * inverse_selected_sum;
}
bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}
void native_router_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_router_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream, const TurboRoute* tr) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4)
        || overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4)
        || overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    route<<<1, dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights, tr != nullptr ? *tr : TurboRoute{});
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream,
                               const TurboRoute* tr) {
    if (!stream || n_tok < 1 || !valid(logits, (size_t) n_tok * 512 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi) requires a stream and aligned [n,512]/[n,10] buffers");
    route_multi<<<(unsigned) ((n_tok + 7) / 8), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(
        logits, ids, weights, n_tok, tr != nullptr ? *tr : TurboRoute{});
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
namespace {
// the drop and the skip after any other top-k.  A pick the row does not run (dropped, or left out by its skip
// bar) aliases the row's top-1 at weight 0; the kept weights renormalise to their sum as the router's do.  A row
// with nothing left out keeps the router's bits: no renormalisation of its own (sum(w) != 1 exactly).
__global__ void turbo_drop_k(int32_t* __restrict__ ids, float* __restrict__ w, int n_tok, int k, const TurboRoute tr) {
    const int t = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= n_tok) return;
    const int d = tr.drop != nullptr ? tr.drop[t] : 0;
    const int bar = tr.skip != nullptr ? tr.skip[t] : 0;
    if (d <= 0 && bar <= 0) return;
    const int keep = d >= k ? 1 : (d <= 0 ? k : k - d);
    ids += (size_t) t * k; w += (size_t) t * k;
    float s = 0.0f;
    int n_drop = 0, n_skip = 0;
    bool any = false;
    for (int i = 0; i < k; ++i) {
        bool alive = i < keep && (i == 0 || bar <= 0 || tier_resident(tr, bar, ids[i]));
        if (alive) { s += w[i]; continue; }
        any = true;
        if (i >= keep) ++n_drop;
        else {
            ++n_skip;
            if (tr.skip_usage != nullptr && tr.layer >= 0)
                atomicAdd(&tr.skip_usage[(size_t) tr.layer * tr.n_expert + ids[i]], 1.0f);
        }
    }
    turbo_count(tr, t, k, n_drop, n_skip);
    if (!any) return;
    const float inv = 1.0f / fmaxf(s, 6.103515625e-5f);
    for (int i = 0; i < k; ++i) {
        const bool alive = i < keep && (i == 0 || bar <= 0 || tier_resident(tr, bar, ids[i]));
        if (!alive) { ids[i] = ids[0]; w[i] = 0.0f; }
        else w[i] *= inv;
    }
}
}
void turbo_drop_apply(int32_t* ids, float* weights, int n_tok, int k, const TurboRoute& tr, void* stream) {
    if (n_tok < 1 || (tr.drop == nullptr && tr.skip == nullptr)) return;
    turbo_drop_k<<<(unsigned) ((n_tok + 63) / 64), 64, 0, static_cast<cudaStream_t>(stream)>>>(ids, weights, n_tok, k, tr);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
bool native_router_topk_supported(int n_expert, int k) { return (n_expert == 512 && k == 10) || (n_expert == 256 && k == 8); }
void native_router_topk(const float* logits, int32_t* ids, float* weights, int n_expert, int k, int n_tok, void* stream) {
    if (n_expert == 512 && k == 10) {   // Flash-Next: the pinned kernels above, unchanged
        if (n_tok == 1) native_router_top10(logits, ids, weights, stream);
        else native_router_top10_multi(logits, ids, weights, n_tok, stream);
        return;
    }
    if (!stream || n_tok < 1 || !native_router_topk_supported(n_expert, k) ||
        !valid(logits, (size_t) n_tok * n_expert * 4) || !valid(ids, (size_t) n_tok * k * 4) ||
        !valid(weights, (size_t) n_tok * k * 4))
        throw std::invalid_argument("native_router_topk: a stream, a supported (experts, k) and aligned buffers");
    route_topk<256, 8><<<(unsigned) ((n_tok + 7) / 8), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(
        logits, ids, weights, n_tok);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// SAGUARO_ROUTE_RESIDENT (EXPERIMENTAL, opt-in, changes the output): residency-biased routing for 512 x 10.
// After the top-10: for the ranks lo..hi a non-resident expert is replaced by the best resident expert not already
// picked when its logit is within `margin` of the replaced one; the weights are then the softmax of the selected
// logits (the router's renormalisation).  Tokens without a swap keep the router's exact bits.
// stats (device, 4 x uint64): [0] tail entries seen, [1] swaps, [2] non-resident entries before, [3] after.
//
// One warp per token.  This used to be one THREAD per token scanning all 512 experts (4 ranks x 512 x 12 scalar
// ops on up to 6 threads): 45 us per layer, 2.2 ms of a 19.4 ms window, which is exactly what the option saved in
// PCIe experts (~2.4 ms) - the swap cost as much as the misses it removed.  The ranks are still visited in order
// (a swap changes the set already picked), but the scan is coalesced (lane j reads f = j, j + 32, ...) and reduced
// within the warp.  The bits are unchanged: a strictly greater logit wins, the smallest expert index breaks ties,
// and the softmax is the same expression.
__global__ void route_resident_k(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ weights,
                                 const int32_t* __restrict__ res, int n_tok, float margin, int lo, int hi,
                                 unsigned long long* __restrict__ stats) {
    const int t = blockIdx.x;
    if (t >= n_tok) return;
    const int lane = (int) threadIdx.x;
    const float* l = logits + (size_t) t * 512;
    int32_t* id = ids + (size_t) t * 10;
    int32_t my[10];
    for (int r = 0; r < 10; ++r) my[r] = id[r];
    int swaps = 0, tail = 0, before = 0;
    if (lane == 0)
        for (int r = 0; r < 10; ++r) before += res[my[r]] < 0;
    for (int r = lo; r <= hi && r < 10; ++r) {
        const int e = my[r];
        if (res[e] >= 0) continue;                      // already resident: uniform across the warp
        if (lane == 0) ++tail;
        float bv = -INFINITY;
        int bi = -1;
        for (int f = lane; f < 512; f += 32) {
            if (res[f] < 0) continue;
            bool used = false;
            for (int q = 0; q < 10; ++q) used |= my[q] == f;
            if (used) continue;
            if (l[f] > bv) { bv = l[f]; bi = f; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) { bv = ov; bi = oi; }
        }
        if (lane == 0 && bi >= 0 && l[e] - bv <= margin) { my[r] = bi; ++swaps; }
        my[r] = __shfl_sync(0xffffffffu, my[r], 0);     // all lanes keep the same picked set
    }
    int after = 0;
    if (lane == 0)
        for (int r = 0; r < 10; ++r) after += res[my[r]] < 0;
    const int sw = __shfl_sync(0xffffffffu, swaps, 0);
    if (sw) {
        float m = -INFINITY;
        for (int r = 0; r < 10; ++r) m = fmaxf(m, l[my[r]]);
        float ex[10], sum = 0.0f;
        for (int r = 0; r < 10; ++r) { ex[r] = expf(l[my[r]] - m); sum += ex[r]; }
        for (int r = 0; r < 10; ++r) weights[(size_t) t * 10 + r] = ex[r] / sum;
    }
    if (lane == 0) {
        for (int r = 0; r < 10; ++r) id[r] = my[r];
        if (stats) {
            atomicAdd(stats + 0, (unsigned long long) tail);
            atomicAdd(stats + 1, (unsigned long long) sw);
            atomicAdd(stats + 2, (unsigned long long) before);
            atomicAdd(stats + 3, (unsigned long long) after);
        }
    }
}
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok, float margin,
                           int rank_lo, int rank_hi, unsigned long long* stats, void* stream) {
    if (!stream || n_tok < 1 || !res_layer) throw std::invalid_argument("native_route_resident: bad arguments");
    route_resident_k<<<(unsigned) n_tok, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        logits, ids, weights, res_layer, n_tok, margin, rank_lo, rank_hi, stats);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
