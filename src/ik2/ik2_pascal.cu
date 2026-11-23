// src/ik2/ik2_pascal.cu - saguaro-ik2's Pascal kernels (the interface: ik2_pascal.cuh).
//
// The token path's mat-vecs on a P40 (GP102: 347 GB/s, FP32 FMA, FP16 math at 1/64): the generic kernels read their
// rows a warp a row over K in steps of 32 sub-blocks, which leaves lanes idle where K / 16 is not a multiple of 32
// (2304: 4.5 steps, 1152: 2.25) and keeps one load a lane in flight.  Here a warp takes a group of rows whose sub-blocks
// are a multiple of 32 (gate + up of one row: 288; four down rows: 288), each lane loads all of its sub-blocks first
// (9 in flight) and then sums them; the input stays in shared memory (one copy a block of many rows).  The arithmetic
// is the generic kernels' (a code's FP16 bits to F32, F32 FMAs); only the order of the sums differs.
#include "ik2_pascal.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

// (the kernels' device code for sm_6x only: a newer GPU's pass compiles them empty - the engine never launches them there)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define IK2P_STUB 1
#else
#define IK2P_STUB 0
#endif

namespace ik2::k::pascal {

namespace {

bool env_on(const char* name) {
    const char* e = std::getenv(name);
    return !(e && e[0] == '0' && !e[1]);
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ik2::k::pascal::%s: launch failed: %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}

__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }
__device__ __forceinline__ float2 u32_to_f2(uint32_t u) {
    return __half22float2(__halves2half2(__ushort_as_half((uint16_t) (u & 0xFFFFu)), __ushort_as_half((uint16_t) (u >> 16))));
}
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// an E4M3 scale byte as ggml's NVFP4 reads it (half its value; 0x7F: 0) - ik2_exact.h's ue4m3
__device__ __forceinline__ float ue4m3(uint32_t v) {
    const uint32_t e = (v >> 3) & 0xFu, m = v & 7u;
    const float f = e == 0u ? (float) m * 0x1.0p-10f : __uint_as_float(((e + 119u) << 23) | (m << 20));
    return v == 0x7Fu ? 0.0f : f;
}
// E2M1 pairs (twice their value) through FP16 bits - the generic kernels' decode
__device__ __forceinline__ float2 e2m1_pair_lo(uint32_t s) {
    const uint32_t sel = (s << 4) & 0x7070u;
    const uint32_t sgn = ((s & 8u) << 12) | ((s & 0x800u) << 20);
    return u32_to_f2(__byte_perm(0x42403c00u, 0x4a484644u, sel) | sgn);
}
__device__ __forceinline__ float2 e2m1_pair_hi(uint32_t s) {
    const uint32_t sel = s & 0x7070u;
    const uint32_t sgn = ((s & 0x80u) << 8) | ((s & 0x8000u) << 16);
    return u32_to_f2(__byte_perm(0x42403c00u, 0x4a484644u, sel) | sgn);
}
__device__ __forceinline__ float nvfp4_sub_dot(uint32_t q0, uint32_t q1, float4 x0, float4 x1, float4 x2, float4 x3) {
    const float2 w01 = e2m1_pair_lo(q0), w23 = e2m1_pair_lo(q0 >> 16);
    const float2 w45 = e2m1_pair_lo(q1), w67 = e2m1_pair_lo(q1 >> 16);
    const float2 w89 = e2m1_pair_hi(q0), wAB = e2m1_pair_hi(q0 >> 16);
    const float2 wCD = e2m1_pair_hi(q1), wEF = e2m1_pair_hi(q1 >> 16);
    return (w01.x * x0.x + w01.y * x0.y + w23.x * x0.z + w23.y * x0.w) +
           (w45.x * x1.x + w45.y * x1.y + w67.x * x1.z + w67.y * x1.w) +
           (w89.x * x2.x + w89.y * x2.y + wAB.x * x2.z + wAB.y * x2.w) +
           (wCD.x * x3.x + wCD.y * x3.y + wEF.x * x3.z + wEF.y * x3.w);
}

// the input in shared memory padded as the generic kernels' xp: value i at (i >> 5) * 36 + (i & 31) (a sub-block's 16
// values float4-aligned, no bank conflicts)
__host__ __device__ constexpr int xp_floats(int K) { return (K >> 5) * 36; }
__device__ __forceinline__ void fill_f32(const float* x, int K, float* xs) {
    const float4* x4 = (const float4*) x;
    for (int i4 = threadIdx.x; i4 < (K >> 2); i4 += blockDim.x) {
        const int i = i4 << 2;
        *(float4*) (xs + (i >> 5) * 36 + (i & 31)) = __ldg(x4 + i4);
    }
}
__device__ __forceinline__ void fill_f16(const uint16_t* h, int K, float* xs) {
    const uint2* h4 = (const uint2*) h;
    for (int i4 = threadIdx.x; i4 < (K >> 2); i4 += blockDim.x) {
        const int i = i4 << 2;
        const uint2 u = __ldg(h4 + i4);
        const float2 f0 = u32_to_f2(u.x), f1 = u32_to_f2(u.y);
        *(float4*) (xs + (i >> 5) * 36 + (i & 31)) = make_float4(f0.x, f0.y, f1.x, f1.y);
    }
}

// a warp's RG rows of NS sub-blocks each (rows[k], NVFP4): lane l takes units l, l + 32, ... (unit u: row u / NS,
// sub-block u % NS), all its loads first
template <int NS, int RG>
__device__ __forceinline__ void rows_dot(const uint8_t* const (&rows)[RG], const float* xs, float (&acc)[RG]) {
    constexpr int NI = RG * NS / 32;
    static_assert(RG * NS % 32 == 0, "a warp's units: a multiple of 32");
    const int lane = threadIdx.x & 31;
    uint32_t sc[NI], q0[NI], q1[NI];
#pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int u = lane + 32 * i, rg = u / NS, sub = u - rg * NS;
        const uint8_t* b = rows[0];
#pragma unroll
        for (int k = 1; k < RG; ++k)
            if (rg == k) b = rows[k];
        b += (size_t) (sub >> 2) * 36;
        sc[i] = __ldg((const uint32_t*) b);
        q0[i] = __ldg((const uint32_t*) (b + 4 + (sub & 3) * 8));
        q1[i] = __ldg((const uint32_t*) (b + 8 + (sub & 3) * 8));
    }
#pragma unroll
    for (int k = 0; k < RG; ++k) acc[k] = 0.0f;
#pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int u = lane + 32 * i, rg = u / NS, sub = u - rg * NS;
        const float d = ue4m3((sc[i] >> (8 * (sub & 3))) & 0xFFu);
        const float* x = xs + (sub >> 1) * 36 + (sub & 1) * 16;
        const float4 x0 = *(const float4*) (x + 0), x1 = *(const float4*) (x + 4);
        const float4 x2 = *(const float4*) (x + 8), x3 = *(const float4*) (x + 12);
        const float v = d * nvfp4_sub_dot(q0[i], q1[i], x0, x1, x2, x3);
#pragma unroll
        for (int k = 0; k < RG; ++k)
            if (rg == k) acc[k] += v;
    }
#pragma unroll
    for (int k = 0; k < RG; ++k) acc[k] = wsum(acc[k]);
}

__device__ __forceinline__ float q8_dot4(uint32_t w, float4 x) {
    return (float) (int8_t) (w & 0xFFu) * x.x + (float) (int8_t) ((w >> 8) & 0xFFu) * x.y +
           (float) (int8_t) ((w >> 16) & 0xFFu) * x.z + (float) (int8_t) (w >> 24) * x.w;
}

// rows_dot for Q8_0 blocks (34 bytes: FP16 d, 32 int8; a row's blocks 2-byte aligned): RG rows of NB blocks, lane l
// the blocks l, l + 32, ... - each read as the 9 aligned words around it (all a lane's first), then taken apart
template <int NB, int RG>
__device__ __forceinline__ void rows_dot_q8(const uint8_t* const (&rows)[RG], const float* xs, float (&acc)[RG]) {
    constexpr int NI = RG * NB / 32;
    static_assert(RG * NB % 32 == 0, "a warp's units: a multiple of 32");
    const int lane = threadIdx.x & 31;
    uint32_t w[NI][9];
#pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int u = lane + 32 * i, rg = u / NB, b = u - rg * NB;
        const uint8_t* p = rows[0];
#pragma unroll
        for (int k = 1; k < RG; ++k)
            if (rg == k) p = rows[k];
        p += (size_t) b * 34;
        const uint32_t* a = (const uint32_t*) ((uintptr_t) p & ~(uintptr_t) 3);
#pragma unroll
        for (int j = 0; j < 9; ++j) w[i][j] = __ldg(a + j);
    }
#pragma unroll
    for (int k = 0; k < RG; ++k) acc[k] = 0.0f;
#pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int u = lane + 32 * i, rg = u / NB, b = u - rg * NB;
        const uint8_t* p = rows[0];
#pragma unroll
        for (int k = 1; k < RG; ++k)
            if (rg == k) p = rows[k];
        const uint32_t sh = (uint32_t) ((uintptr_t) (p + (size_t) b * 34) & 3) * 8;   // (0 or 16)
        const float dv = h2f((uint16_t) ((w[i][0] >> sh) & 0xFFFFu));
        const float* x = xs + b * 36;
        float a = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            a += q8_dot4(__funnelshift_rc(w[i][j], w[i][j + 1], 16 + sh), *(const float4*) (x + 4 * j));
        const float v = dv * a;
#pragma unroll
        for (int k = 0; k < RG; ++k)
            if (rg == k) acc[k] += v;
    }
#pragma unroll
    for (int k = 0; k < RG; ++k) acc[k] = wsum(acc[k]);
}

// Q8_0 experts: gate | up of two rows a warp (r, r + 1: 4 rows of 72 blocks), down 8 rows of 36 blocks
template <int NB, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
gate_up_swiglu_q8_kernel(const uint8_t* const* blob, const int* tok, size_t eb0, const float* x, int F, const float* sg,
                         const float* su, float gcap, float ucap, uint16_t* h, const int* orow, int h_ld,
                         const int* cls, int want) {
    if (IK2P_STUB) return;
    constexpr int C = NB * 32;
    constexpr size_t RB = (size_t) NB * 34;
    extern __shared__ float4 xs4[];
    float* xs = (float*) xs4;
    const int s = blockIdx.y;
    if ((uintptr_t) blob[s] <= 1u) return;
    if (cls && cls[s] != want) return;
    fill_f32(x + (size_t) tok[s] * C, C, xs);
    __syncthreads();
    const int r = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * 2;
    if (r >= F) return;
    const int r1 = min(r + 1, F - 1);
    const uint8_t* bg = blob[s];
    const uint8_t* const rows[4] = {bg + (size_t) r * RB, bg + eb0 + (size_t) r * RB, bg + (size_t) r1 * RB,
                                    bg + eb0 + (size_t) r1 * RB};
    float a[4];
    rows_dot_q8<NB, 4>(rows, xs, a);
    const int lane = threadIdx.x & 31;
    if (lane < 2 && r + lane < F) {
        const float sgs = sg ? sg[s] : 1.0f, sus = su ? su[s] : 1.0f;
        const float g = (lane ? a[2] : a[0]) * sgs, u = (lane ? a[3] : a[1]) * sus;
        const float v = gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap);
        h[(size_t) (orow ? orow[s] : s) * h_ld + r + lane] = f2h(v);
    }
}
template <int NB, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
down_q8_kernel(const uint8_t* const* blob, size_t off_down, const uint16_t* h, int C, float* y, const int* cls,
               int want) {
    if (IK2P_STUB) return;
    constexpr int F = NB * 32, RG = 8;
    constexpr size_t RB = (size_t) NB * 34;
    extern __shared__ float4 xs4[];
    float* xs = (float*) xs4;
    const int s = blockIdx.y;
    if ((uintptr_t) blob[s] <= 1u) return;
    if (cls && cls[s] != want) return;
    fill_f16(h + (size_t) s * F, F, xs);
    __syncthreads();
    const int r0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * RG;
    if (r0 >= C) return;
    const uint8_t* w = blob[s] + off_down;
    const uint8_t* rows[RG];
#pragma unroll
    for (int k = 0; k < RG; ++k) rows[k] = w + (size_t) min(r0 + k, C - 1) * RB;
    float a[RG];
    rows_dot_q8<NB, RG>(rows, xs, a);
    const int lane = threadIdx.x & 31;
    float v = a[0];
#pragma unroll
    for (int k = 1; k < RG; ++k)
        if (lane == k) v = a[k];
    if (lane < RG && r0 + lane < C) y[(size_t) s * C + r0 + lane] = v;
}

template <int NS, int WARPS, int PAIRS>
__global__ void __launch_bounds__(WARPS * 32)
gate_up_swiglu_kernel(const uint8_t* const* blob, const int* tok, size_t eb0, const float* x, int F, const float* sg,
                      const float* su, float gcap, float ucap, uint16_t* h, const int* orow, int h_ld, const int* cls,
                      int want) {
    if (IK2P_STUB) return;
    constexpr int C = NS * 16;
    constexpr size_t RB = (size_t) NS / 4 * 36;
    extern __shared__ float4 xs4[];
    float* xs = (float*) xs4;
    const int s = blockIdx.y;
    if ((uintptr_t) blob[s] <= 1u) return;   // (two GPUs: a pick the other GPU runs)
    if (cls && cls[s] != want) return;
    fill_f32(x + (size_t) tok[s] * C, C, xs);
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const uint8_t* bg = blob[s];
    const float sgs = sg ? sg[s] : 1.0f, sus = su ? su[s] : 1.0f;
#pragma unroll 1
    for (int p = 0; p < PAIRS; ++p) {
        const int r = (blockIdx.x * WARPS + warp) * PAIRS + p;
        if (r >= F) break;
        const uint8_t* const rows[2] = {bg + (size_t) r * RB, bg + eb0 + (size_t) r * RB};
        float a[2];
        rows_dot<NS, 2>(rows, xs, a);
        if ((threadIdx.x & 31) == 0) {
            const float g = a[0] * sgs, u = a[1] * sus;
            const float v = gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap);
            h[(size_t) (orow ? orow[s] : s) * h_ld + r] = f2h(v);
        }
    }
}

template <int NS, int WARPS, int GROUPS>
__global__ void __launch_bounds__(WARPS * 32)
down_kernel(const uint8_t* const* blob, size_t off_down, const uint16_t* h, int C, float* y, const int* cls, int want) {
    if (IK2P_STUB) return;
    constexpr int F = NS * 16, RG = 4;
    constexpr size_t RB = (size_t) NS / 4 * 36;
    extern __shared__ float4 xs4[];
    float* xs = (float*) xs4;
    const int s = blockIdx.y;
    if ((uintptr_t) blob[s] <= 1u) return;
    if (cls && cls[s] != want) return;
    fill_f16(h + (size_t) s * F, F, xs);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint8_t* w = blob[s] + off_down;
#pragma unroll 1
    for (int g = 0; g < GROUPS; ++g) {
        const int r0 = ((blockIdx.x * WARPS + warp) * GROUPS + g) * RG;
        if (r0 >= C) break;
        const uint8_t* const rows[RG] = {w + (size_t) r0 * RB, w + (size_t) (r0 + 1) * RB, w + (size_t) (r0 + 2) * RB,
                                         w + (size_t) (r0 + 3) * RB};
        float a[RG];
        rows_dot<NS, RG>(rows, xs, a);
        if (lane < RG) {
            const float v = lane == 0 ? a[0] : lane == 1 ? a[1] : lane == 2 ? a[2] : a[3];
            y[(size_t) s * C + r0 + lane] = v;
        }
    }
}

// a product's row through its short conv (ik2_kernels.cu's mv_conv_row): false if the row has none
__device__ __forceinline__ bool conv_row(const MvConv& cv, int row, float x_new) {
    const int c = row - cv.r0;
    if (c < 0 || c >= cv.n) return false;
    const float h0 = cv.hist[c], h1 = cv.hist[(size_t) cv.n + c], h2 = cv.hist[(size_t) 2 * cv.n + c];
    const float* wc = cv.w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    float a = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x_new;
    if (cv.dyn) {
        const float* dc = cv.dyn + (size_t) (c / (cv.n / cv.heads)) * 4;
        a += dc[0] * x_new + dc[1] * h2 + dc[2] * h1 + dc[3] * h0;
    }
    if (cv.res) a += cv.res[c];
    cv.y[c] = a;
    cv.hist[c] = h1;
    cv.hist[(size_t) cv.n + c] = h2;
    cv.hist[(size_t) 2 * cv.n + c] = x_new;
    return true;
}


// y = W . x for a planar Q8_0 matrix (mv_q8p_kernel's): a warp a row, 8 values a lane a step (a warp reads 256 bytes
// an instruction), all NI steps' loads first
template <int NI, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
q8p_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ d, int N, const float* __restrict__ x, float* y,
           MvConv c0, MvConv c1, MvAlt a0, MvAlt a1) {
    if (IK2P_STUB) return;
    constexpr int K = NI * 256;
    const int row = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint2* qr = (const uint2*) (q + (size_t) row * K);
    const uint16_t* dr = d + (size_t) row * (K >> 5);
    const float4* x4 = (const float4*) ((unsigned) (row - a0.r0) < (unsigned) a0.n   ? a0.x
                                        : (unsigned) (row - a1.r0) < (unsigned) a1.n ? a1.x
                                                                                     : x);
    uint2 w[NI];
    uint16_t sc[NI];
#pragma unroll
    for (int it = 0; it < NI; ++it) w[it] = __ldg(qr + it * 32 + lane), sc[it] = __ldg(dr + it * 8 + (lane >> 2));
    float acc = 0.0f;
#pragma unroll
    for (int it = 0; it < NI; ++it) {
        const float4 xa = __ldg(x4 + (it * 32 + lane) * 2), xb = __ldg(x4 + (it * 32 + lane) * 2 + 1);
        acc += h2f(sc[it]) * (q8_dot4(w[it].x, xa) + q8_dot4(w[it].y, xb));
    }
    acc = wsum(acc);
    if (lane == 0 && !conv_row(c0, row, acc) && !conv_row(c1, row, acc)) y[row] = acc;
}

template <int NS, int WARPS, int PAIRS>
void launch_gu(const uint8_t* const* blob, const int* tok, int S, size_t eb0, const float* x, int F, const float* sg,
               const float* su, float gcap, float ucap, uint16_t* h, const int* orow, int h_ld, const int* cls,
               int want, cudaStream_t s) {
    const dim3 g((unsigned) ((F + WARPS * PAIRS - 1) / (WARPS * PAIRS)), (unsigned) S);
    gate_up_swiglu_kernel<NS, WARPS, PAIRS><<<g, WARPS * 32, xp_floats(NS * 16) * 4, s>>>(
        blob, tok, eb0, x, F, sg, su, gcap, ucap, h, orow, h_ld, cls, want);
}
template <int NS, int WARPS, int GROUPS>
void launch_down(const uint8_t* const* blob, int S, size_t off_down, const uint16_t* h, int C, float* y,
                 const int* cls, int want, cudaStream_t s) {
    const dim3 g((unsigned) ((C + WARPS * GROUPS * 4 - 1) / (WARPS * GROUPS * 4)), (unsigned) S);
    down_kernel<NS, WARPS, GROUPS><<<g, WARPS * 32, xp_floats(NS * 16) * 4, s>>>(blob, off_down, h, C, y, cls, want);
}

// ... a row split over SPLIT warps of a block (NI / SPLIT steps each), their sums added in shared memory in a fixed
// order: the small matrices (2304 rows: 1.2 waves of a warp a row on 30 SMs) in several waves
template <int NI, int SPLIT>
__global__ void __launch_bounds__(SPLIT * 64)
q8p_split_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ d, int N, const float* __restrict__ x,
                 float* y, MvConv c0, MvConv c1, MvAlt a0, MvAlt a1) {
    if (IK2P_STUB) return;
    constexpr int K = NI * 256, NS = NI / SPLIT;
    static_assert(NI % SPLIT == 0, "a row's steps split evenly");
    __shared__ float part[2][SPLIT];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rr = warp / SPLIT, sp = warp - rr * SPLIT;
    const int row = blockIdx.x * 2 + rr;
    if (row < N) {
        const uint2* qr = (const uint2*) (q + (size_t) row * K) + sp * NS * 32;
        const uint16_t* dr = d + (size_t) row * (K >> 5) + sp * NS * 8;
        const float4* x4 = (const float4*) ((unsigned) (row - a0.r0) < (unsigned) a0.n   ? a0.x
                                            : (unsigned) (row - a1.r0) < (unsigned) a1.n ? a1.x
                                                                                         : x) + sp * NS * 64;
        uint2 w[NS];
        uint16_t sc[NS];
#pragma unroll
        for (int it = 0; it < NS; ++it) w[it] = __ldg(qr + it * 32 + lane), sc[it] = __ldg(dr + it * 8 + (lane >> 2));
        float acc = 0.0f;
#pragma unroll
        for (int it = 0; it < NS; ++it) {
            const float4 xa = __ldg(x4 + (it * 32 + lane) * 2), xb = __ldg(x4 + (it * 32 + lane) * 2 + 1);
            acc += h2f(sc[it]) * (q8_dot4(w[it].x, xa) + q8_dot4(w[it].y, xb));
        }
        acc = wsum(acc);
        if (lane == 0) part[rr][sp] = acc;
    }
    __syncthreads();
    if (sp != 0 || lane != 0 || row >= N) return;
    float acc = part[rr][0];
#pragma unroll
    for (int i = 1; i < SPLIT; ++i) acc += part[rr][i];
    if (!conv_row(c0, row, acc) && !conv_row(c1, row, acc)) y[row] = acc;
}

// One token's attention over an FP16 cache, a split of the keys a block (attn_split_kernel's partials: the same
// [H][P][130] for attn_combine_kernel), but a block a KV head and its G query heads - each key's K and V read and
// converted once for the G of them, not once a query head; the query's dot and its relative bias summed in one
// reduction.  The position from the device (a CUDA graph's tokens).
template <int G, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
attn_gqa_kernel(const float* q, const float* r, const uint16_t* kc, const uint16_t* vc, int kv_cap, int Hkv,
                float qk_scale, const float* proj, int d_rel, int extent, int window, float* part, const int* pos_d) {
    if (IK2P_STUB) return;
    __shared__ float sm_m[WARPS][G], sm_s[WARPS][G];
    __shared__ __align__(16) float sm_o[WARPS][G][128];
    const int qpos = *pos_d, k0 = window < (1 << 30) ? max(0, qpos - window + 1) : 0, nk = qpos + 1 - k0;
    const int g = blockIdx.x, p = blockIdx.y, P = gridDim.y, H = Hkv * G, KD = Hkv * 128;
    const int len = (nk + P - 1) / P, j0 = p * len, j1 = min(nk, j0 + len);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, o = lane * 4;
    float4 q4[G];
    float rl[G], m[G], sum[G];
    float4 acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) {
        const int hh = g * G + h;
        q4[h] = __ldg((const float4*) (q + (size_t) hh * 128 + o));
        rl[h] = lane < d_rel ? __ldg(r + (size_t) hh * d_rel + lane) : 0.0f;
        m[h] = -INFINITY, sum[h] = 0.0f, acc[h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (int j = j0 + warp; j < j1; j += WARPS) {
        const int pos = k0 + j, dist = qpos - pos;
        if (dist < 0 || dist >= window) continue;
        const size_t off = (size_t) (kv_cap > 0 ? pos % kv_cap : pos) * KD + (size_t) g * 128 + o;
        const uint2 ku = __ldg((const uint2*) (kc + off)), vu = __ldg((const uint2*) (vc + off));
        const float pr = dist < extent && lane < d_rel ? __ldg(proj + (size_t) dist * d_rel + lane) : 0.0f;
        const float2 k01 = u32_to_f2(ku.x), k23 = u32_to_f2(ku.y);
        float a[G];
#pragma unroll
        for (int h = 0; h < G; ++h)
            a[h] = (q4[h].x * k01.x + q4[h].y * k01.y + q4[h].z * k23.x + q4[h].w * k23.y) * qk_scale + rl[h] * pr;
#pragma unroll
        for (int s = 16; s > 0; s >>= 1)
#pragma unroll
            for (int h = 0; h < G; ++h) a[h] += __shfl_xor_sync(0xffffffffu, a[h], s);
        const float2 v01 = u32_to_f2(vu.x), v23 = u32_to_f2(vu.y);
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float mn = fmaxf(m[h], a[h]);
            const float al = m[h] == -INFINITY ? 0.0f : expf(m[h] - mn), be = expf(a[h] - mn);
            m[h] = mn, sum[h] = sum[h] * al + be;
            acc[h].x = acc[h].x * al + v01.x * be, acc[h].y = acc[h].y * al + v01.y * be;
            acc[h].z = acc[h].z * al + v23.x * be, acc[h].w = acc[h].w * al + v23.y * be;
        }
    }
#pragma unroll
    for (int h = 0; h < G; ++h) {
        if (lane == 0) sm_m[warp][h] = m[h], sm_s[warp][h] = sum[h];
        *(float4*) &sm_o[warp][h][o] = acc[h];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < G * 128; i += WARPS * 32) {
        const int h = i >> 7, d = i & 127;
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) M = fmaxf(M, sm_m[w][h]);
        float sm = 0.0f, ov = 0.0f;
        if (M > -INFINITY) {
#pragma unroll
            for (int w = 0; w < WARPS; ++w) {
                if (sm_m[w][h] == -INFINITY) continue;
                const float f = expf(sm_m[w][h] - M);
                sm += sm_s[w][h] * f, ov += sm_o[w][h][d] * f;
            }
        }
        float* pp = part + ((size_t) (g * G + h) * P + p) * 130;
        if (d == 0) pp[0] = M, pp[1] = sm;
        pp[2 + d] = ov;
    }
    (void) H;
}

}  // namespace

static bool split_off() {   // (SAGUARO_IK2_PASCAL_Q8_SPLIT=0: a warp a row everywhere - the bench's comparison)
    static const bool off = !env_on("SAGUARO_IK2_PASCAL_Q8_SPLIT");
    return off;
}

static bool q8_experts_on() {   // (SAGUARO_IK2_PASCAL_Q8_EXPERTS=0: Q8_0 experts on the generic kernels)
    static const bool on = env_on("SAGUARO_IK2_PASCAL_Q8_EXPERTS");
    return on;
}

bool attn_on() {
    static const bool on = arch_class() == 60 && env_on("SAGUARO_IK2_PASCAL_ATTN");
    return on;
}

bool experts_on() {
    static const bool on = arch_class() == 60 && env_on("SAGUARO_IK2_PASCAL_EXPERTS");
    return on;
}
bool q8_on() {
    static const bool on = arch_class() == 60 && env_on("SAGUARO_IK2_PASCAL_Q8");
    return on;
}
bool gate_up_swiglu(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up, size_t eb0,
                    const float* x, int C, int F, const float* sg, const float* su, float gcap, float ucap, uint16_t* h,
                    const int* orow, int h_ld, const int* cls, int want, cudaStream_t s) {
    if (ty_gate != ty_up || (ty_gate != kNVFP4 && ty_gate != kQ8_0) || C != 2304 || ((uintptr_t) x & 15)) return false;
    if (S <= 0) return true;
    if (h_ld <= 0) h_ld = F;
    if (ty_gate == kQ8_0) {
        if (!q8_experts_on()) return false;
        const dim3 g((unsigned) ((F + 7) / 8), (unsigned) S);   // (4 warps, 2 rows each)
        gate_up_swiglu_q8_kernel<72, 4><<<g, 128, xp_floats(C) * 4, s>>>(blob, tok, eb0, x, F, sg, su, gcap, ucap, h,
                                                                         orow, h_ld, cls, want);
        check("gate_up_swiglu_q8");
        return true;
    }
    launch_gu<144, 4, 1>(blob, tok, S, eb0, x, F, sg, su, gcap, ucap, h, orow, h_ld, cls, want, s);
    check("gate_up_swiglu");
    return true;
}

bool down(const uint8_t* const* blob, int S, int ty_down, size_t off_down, const uint16_t* h, int C, int F, float* y,
          const int* cls, int want, cudaStream_t s) {
    if ((ty_down != kNVFP4 && ty_down != kQ8_0) || F != 1152 || C % 4 || ((uintptr_t) h & 7)) return false;
    if (S <= 0) return true;
    if (ty_down == kQ8_0) {
        if (!q8_experts_on()) return false;
        const dim3 g((unsigned) ((C + 31) / 32), (unsigned) S);   // (4 warps, 8 rows each)
        down_q8_kernel<36, 4><<<g, 128, xp_floats(F) * 4, s>>>(blob, off_down, h, C, y, cls, want);
        check("down_q8");
        return true;
    }
    launch_down<72, 4, 1>(blob, S, off_down, h, C, y, cls, want, s);
    check("down");
    return true;
}

bool mv_q8p_conv(const int8_t* q, const uint16_t* d, int N, int K, const float* x, float* y, const MvConv& c0,
                 const MvConv& c1, const MvAlt& a0, const MvAlt& a1, cudaStream_t s) {
    if (((uintptr_t) q & 7) || (((uintptr_t) x | (uintptr_t) a0.x | (uintptr_t) a1.x) & 15)) return false;
    if (N <= 0) return true;
    // (a warp a row while the rows fill several waves - the head; else a row over 3 warps)
    const unsigned g = (unsigned) ((N + 1) / 2);
    const bool split = N < 16384 && !split_off();
    switch (K) {
    case 2304:   // (the hidden size)
        if (split) q8p_split_kernel<9, 3><<<g, 192, 0, s>>>(q, d, N, x, y, c0, c1, a0, a1);
        else q8p_kernel<9, 2><<<g, 64, 0, s>>>(q, d, N, x, y, c0, c1, a0, a1);
        break;
    case 3072:   // (the heads' output)
        if (split) q8p_split_kernel<12, 3><<<g, 192, 0, s>>>(q, d, N, x, y, c0, c1, a0, a1);
        else q8p_kernel<12, 2><<<g, 64, 0, s>>>(q, d, N, x, y, c0, c1, a0, a1);
        break;
    default: return false;
    }
    check("mv_q8p_conv");
    return true;
}

bool attn_split_f16(const float* q, const float* r, const uint16_t* kc, const uint16_t* vc, int kv_cap, int H, int Hkv,
                    const int* pos, int P, float qk_scale, const float* proj, int d_rel, int window, int extent,
                    float* part, cudaStream_t s) {
    if (Hkv <= 0 || H != 3 * Hkv || d_rel > 32 || (((uintptr_t) q | (uintptr_t) kc | (uintptr_t) vc) & 15)) return false;
    // (the sliding layers' few splits: 8 warps a block, the keys' loads more in flight)
    if (P <= 32) attn_gqa_kernel<3, 8><<<dim3((unsigned) Hkv, (unsigned) P), 256, 0, s>>>(
        q, r, kc, vc, kv_cap, Hkv, qk_scale, proj, d_rel, extent, window, part, pos);
    else attn_gqa_kernel<3, 4><<<dim3((unsigned) Hkv, (unsigned) P), 128, 0, s>>>(
        q, r, kc, vc, kv_cap, Hkv, qk_scale, proj, d_rel, extent, window, part, pos);
    check("attn_split_f16");
    return true;
}

}  // namespace ik2::k::pascal
