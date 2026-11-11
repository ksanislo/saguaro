// src/prefill/gdn_chunk_sm75.cuh - STRATA_GDN_CHUNKED_SM75=1: the chunked (WY form) prompt recurrence of
// gdn_rec_chunked on Turing (sm_75: 64 KB of shared memory an SM, no cp.async, no TF32/BF16 tensor cores).
// Included by src/prefill/kernels.cu and src/prefill/gdn_rec_parity.cu (the test runs this very code), after
// their S, HK, HV, C, VPK, GCH, GDV, GTP, kChunkPrepSmem and gdn_chunk_prep_kernel's math.
//
// The same math as gdn_rec_chunked (chunks of GCH = 32 tokens, T = (I + A)^-1, D = T Y, o = e^g Q S0 + P D,
// S = e^gL S0 + K^T (decayed D)), but the products on FP16 tensor cores (mma.m16n8k8, FP32 accumulate) with every
// operand split in two halves, x = hi + lo (hi = fp16(x), lo = fp16(x - hi)): hi.hi + hi.lo + lo.hi, the lo.lo term
// (2^-22 of the product) dropped - FP32-level, not the same bits.  Before a split each operand is scaled by a power of
// two (exact) that puts its largest |value| in [2^12, 2^13): no fp16 overflow, and lo stays normal down to 2^-11 of
// the largest value.  q and k (L2-normalized) are scaled by 2^8.
//
// A warp per (value head, 16 value columns): 384 warps.  The warp keeps its state slice transposed (S^T, 16 columns x
// 128 rows) in mma accumulator registers, which are also the A operand of the next chunk's S^T K^T / S^T Q^T (no
// shared-memory copy of the state).  A block is a key head's three value heads x GT_NSL column slices; its shared
// memory is the chunk's k and q rows as hi/lo halves (4 x 32 x 128 x 2 B = 32 KB, 16-byte units XOR-swizzled for
// conflict-free ldmatrix), so two blocks share an SM and all 64 blocks are resident on a 40-SM T4.
// gdn_chunk_prep_hl_kernel = gdn_chunk_prep_kernel with T and P written as scaled hi/lo halves (+ their scales).

#if !defined(__HIPCC__)
constexpr int GT_NSL = 2;                                     // 16-column slices per scan block
constexpr int GT_WARPS = VPK * GT_NSL;                        // warps per scan block
constexpr int GSB_T = 1024;                                   // tokens per super-block (scratch 12.8 MB)
constexpr size_t kChunkScanTSmem = 4 * GCH * S * sizeof(__half);   // k hi, k lo, q hi, q lo
constexpr float kGdnQkScale = 256.0f;                         // q / k scale before the split (|q|, |k| <= 1)
static_assert(GCH == 32 && S == 128 && GDV == 16, "gdn_chunk_scan_t_kernel: fragment layout");

__device__ __forceinline__ float gdn_warp_max(float m) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    return m;
}
// a power of two putting m (>= 0) in [2^12, 2^13), and its inverse (both exact); 1 for 0 (or a subnormal m), at most
// 2^+-64 (a scaled state stays far from FP32 overflow)
__device__ __forceinline__ float gdn_p2_scale(float m, float& inv) {
    const int e = (int) ((__float_as_uint(m) >> 23) & 0xff) - 127;
    const int s = e == -127 ? 0 : max(-64, min(64, 12 - e));
    inv = __uint_as_float((uint32_t) (127 - s) << 23);
    return __uint_as_float((uint32_t) (127 + s) << 23);
}
__device__ __forceinline__ void gdn_split2(float x0, float x1, uint32_t& hi, uint32_t& lo) {
    const __half2 h = __floats2half2_rn(x0, x1);
    const float2 b = __half22float2(h);
    const __half2 l = __floats2half2_rn(x0 - b.x, x1 - b.y);
    hi = *reinterpret_cast<const uint32_t*>(&h);
    lo = *reinterpret_cast<const uint32_t*>(&l);
}
__device__ __forceinline__ void gdn_mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(b));
#endif
}
// a read-only global load kept in program order with the mma (ptxas would hoist every T / P fragment load of a chunk
// to its start and spill)
__device__ __forceinline__ uint32_t gdn_ldg_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ float gdn_ldg_f32(const float* p) {
    float v;
    asm volatile("ld.global.nc.f32 %0, [%1];\n" : "=f"(v) : "l"(p));
    return v;
}
// d += (ahi + alo)(bhi + blo) without alo.blo
__device__ __forceinline__ void gdn_mma3(float (&d)[4], const uint32_t (&ah)[2], const uint32_t (&al)[2], uint32_t bh,
                                         uint32_t bl) {
    gdn_mma(d, ah[0], ah[1], bh);
    gdn_mma(d, ah[0], ah[1], bl);
    gdn_mma(d, al[0], al[1], bh);
}
__device__ __forceinline__ void gdn_ldsm4(uint32_t (&r)[4], const __half* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p))
                 : "memory");
#endif
}
__device__ __forceinline__ void gdn_ldsm4t(uint32_t (&r)[4], const __half* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p))
                 : "memory");
#endif
}
// element (token t, row r) of a staged [GCH][S] half array: 16-byte unit r / 8 XOR (t % 8)
__device__ __forceinline__ int gdn_swz(int t, int r) { return t * S + ((((r >> 3) ^ (t & 7))) << 3) + (r & 7); }

// gdn_chunk_prep_kernel's math; the output: per (chunk, value head) T and P as scaled hi/lo halves (Thi[t][i] then
// Tlo[t][i] where tm's FP32 T was, the same for P) and their inverse scales in sc[2 (chunk * HV + vh) + {0, 1}]
__global__ void __launch_bounds__(256) gdn_chunk_prep_hl_kernel(const float* __restrict__ h, const float* __restrict__ gate,
                                                                const float* __restrict__ beta, __half* __restrict__ tm,
                                                                __half* __restrict__ pm, float* __restrict__ gm,
                                                                float* __restrict__ sc, int64_t t0, int64_t T) {
    extern __shared__ __align__(16) float gsm[];
    float* sk = gsm;                       // [GCH][S + 1]
    float* sq = sk + GCH * (S + 1);        // [GCH][S + 1]
    float* skk = sq + GCH * (S + 1);       // [GCH][GTP]  k_t.k_i
    float* sqk = skk + GCH * GTP;          // [GCH][GTP]  q_t.k_i
    float* sa = gsm;                       // [VPK][GCH][GTP]  A, then T (over k and q, read by then)
    const int qh = blockIdx.y, ch = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int64_t tb = t0 + (int64_t) ch * GCH;
    const int n = (int) min((int64_t) GCH, T - tb);
    {
        constexpr int NP = GCH * S / 4 / 256;
        float4 kv[NP], qv[NP];
#pragma unroll
        for (int m = 0; m < NP; ++m) {
            const int p = tid + 256 * m, t = p / (S / 4), r = (p % (S / 4)) * 4;
            kv[m] = t < n ? *reinterpret_cast<const float4*>(h + (tb + t) * C + HK * S + qh * S + r) : make_float4(0.f, 0.f, 0.f, 0.f);
            qv[m] = t < n ? *reinterpret_cast<const float4*>(h + (tb + t) * C + qh * S + r) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int m = 0; m < NP; ++m) {
            const int p = tid + 256 * m, t = p / (S / 4), r = (p % (S / 4)) * 4;
            float* dk = sk + t * (S + 1) + r;
            float* dq = sq + t * (S + 1) + r;
            dk[0] = kv[m].x; dk[1] = kv[m].y; dk[2] = kv[m].z; dk[3] = kv[m].w;
            dq[0] = qv[m].x; dq[1] = qv[m].y; dq[2] = qv[m].z; dq[3] = qv[m].w;
        }
    }
    __syncthreads();
    {
        constexpr int RW = GCH / 8;
        float kk[RW] = {}, qk[RW] = {};
        for (int r = 0; r < S; ++r) {
            const float ki = sk[lane * (S + 1) + r];
#pragma unroll
            for (int m = 0; m < RW; ++m) {
                kk[m] = fmaf(sk[(warp + 8 * m) * (S + 1) + r], ki, kk[m]);
                qk[m] = fmaf(sq[(warp + 8 * m) * (S + 1) + r], ki, qk[m]);
            }
        }
#pragma unroll
        for (int m = 0; m < RW; ++m) {
            const int t = warp + 8 * m;
            skk[t * GTP + lane] = lane <= t ? kk[m] : 0.0f;
            sqk[t * GTP + lane] = lane <= t ? qk[m] : 0.0f;
        }
    }
    __syncthreads();
    if (warp >= VPK) return;
    const int vh = qh + warp * HK, i = lane;
    float* a = sa + warp * GCH * GTP;
    const float b = i < n ? beta[(tb + i) * HV + vh] : 0.0f;
    float gam = i < n ? gate[(tb + i) * HV + vh] : 0.0f;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float u = __shfl_up_sync(0xffffffffu, gam, o);
        if (lane >= o) gam += u;
    }
    for (int t = 0; t < GCH; ++t) {
        const float gt = __shfl_sync(0xffffffffu, gam, t), bt = __shfl_sync(0xffffffffu, b, t);
        a[t * GTP + i] = i < t ? bt * expf(gt - gam) * skk[t * GTP + i] : (i == t ? 1.0f : 0.0f);
    }
    __syncwarp();
    for (int t = 1; t < GCH; ++t) {
        float v = 0.0f;
#pragma unroll 8
        for (int k = 0; k < t; ++k) v = fmaf(-a[t * GTP + k], a[k * GTP + i], v);
        __syncwarp();
        if (i < t) a[t * GTP + i] = v;
        __syncwarp();
    }
    float mt = 0.0f, mp = 0.0f;
    for (int t = 0; t < GCH; ++t) {
        const float gt = __shfl_sync(0xffffffffu, gam, t);
        mt = fmaxf(mt, fabsf(a[t * GTP + i]));
        mp = fmaxf(mp, i <= t ? fabsf(expf(gt - gam) * sqk[t * GTP + i]) : 0.0f);
    }
    float it, ip;
    const float st = gdn_p2_scale(gdn_warp_max(mt), it), sp = gdn_p2_scale(gdn_warp_max(mp), ip);
    __half* T_ = tm + ((int64_t) ch * HV + vh) * 2 * GCH * GCH;
    __half* P_ = pm + ((int64_t) ch * HV + vh) * 2 * GCH * GCH;
    for (int t = 0; t < GCH; ++t) {
        const float gt = __shfl_sync(0xffffffffu, gam, t);
        const float tv = st * a[t * GTP + i], pv = i <= t ? sp * (expf(gt - gam) * sqk[t * GTP + i]) : 0.0f;
        const __half th = __float2half_rn(tv), ph = __float2half_rn(pv);
        T_[t * GCH + i] = th;
        T_[GCH * GCH + t * GCH + i] = __float2half_rn(tv - __half2float(th));
        P_[t * GCH + i] = ph;
        P_[GCH * GCH + t * GCH + i] = __float2half_rn(pv - __half2float(ph));
    }
    gm[((int64_t) ch * HV + vh) * GCH + i] = gam;
    if (i == 0) {
        sc[2 * ((int64_t) ch * HV + vh)] = it;
        sc[2 * ((int64_t) ch * HV + vh) + 1] = ip;
    }
}

// The scan: a warp per (value head vh, columns c0 .. c0 + 15), chunk after chunk.  g = lane / 4, q4 = lane % 4 (the
// mma fragment coordinates).  st[j] is the accumulator tile of S^T for state rows 8j .. 8j + 7: (column g, rows
// 8j + 2q4, + 1) and (column g + 8, the same rows) - as A operand of k-step j, {st[j][0], st[j][1]} and
// {st[j][2], st[j][3]}.  Per chunk: X^T = S^T K^T and QS^T = S^T Q^T (B: ldmatrix of the staged rows), Y^T, D^T = Y^T T^T
// (B: T from the prep), o^T = e^g QS^T + D^T P^T, S^T = e^gL S^T + (decayed D)^T K (B: ldmatrix.trans of the k rows).
__global__ void __launch_bounds__(32 * GT_WARPS, 2) gdn_chunk_scan_t_kernel(
    float* __restrict__ state, const float* __restrict__ h, const float* __restrict__ beta, const __half* __restrict__ tm,
    const __half* __restrict__ pm, const float* __restrict__ gm, const float* __restrict__ sc, float* __restrict__ oc_out,
    int64_t t0, int nch, int64_t T) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    extern __shared__ __align__(16) __half gsh[];
    __half* skh = gsh;                     // [GCH][S] swizzled: 256 k
    __half* skl = skh + GCH * S;
    __half* sqh = skl + GCH * S;           // 256 q
    __half* sql = sqh + GCH * S;
    constexpr int NT = 32 * GT_WARPS, NF4 = 2 * GCH * S / 4, NPF = (NF4 + NT - 1) / NT;
    const int qh = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    const int j = warp % VPK, vh = qh + j * HK, c0 = (blockIdx.y * GT_NSL + warp / VPK) * GDV;
    const int g = lane >> 2, q4 = lane & 3;
    const size_t rs = (size_t) HV * S;
    const float out_scale = rsqrtf((float) S);

    float st[S / 8][4];
#pragma unroll
    for (int jt = 0; jt < S / 8; ++jt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            st[jt][e] = state[(size_t) (8 * jt + 2 * q4 + (e & 1)) * rs + (size_t) vh * S + c0 + g + 8 * (e >> 1)];

    // the k and q rows of the chunk at c_t0 (n valid), scaled and split, into shared memory: rounds of PFR float4 a
    // thread (all of them at once would push the state slice out of registers)
    auto stage = [&](int64_t c_t0, int n) {
        constexpr int PFR = 4;
#pragma unroll
        for (int m0 = 0; m0 < NPF; m0 += PFR) {
            float4 pf[PFR];
#pragma unroll
            for (int m = 0; m < PFR; ++m) {
                const int p = tid + NT * (m0 + m), isq = p >= NF4 / 2, pp = p % (NF4 / 2), t = pp / (S / 4), r = (pp % (S / 4)) * 4;
                pf[m] = m0 + m < NPF && p < NF4 && t < n
                            ? __ldg(reinterpret_cast<const float4*>(h + (c_t0 + t) * C + (isq ? 0 : HK * S) + qh * S + r))
                            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
#pragma unroll
            for (int m = 0; m < PFR; ++m) {
                const int p = tid + NT * (m0 + m), isq = p >= NF4 / 2, pp = p % (NF4 / 2), t = pp / (S / 4), r = (pp % (S / 4)) * 4;
                if (m0 + m < NPF && p < NF4) {
                    uint2 hi, lo;
                    gdn_split2(kGdnQkScale * pf[m].x, kGdnQkScale * pf[m].y, hi.x, lo.x);
                    gdn_split2(kGdnQkScale * pf[m].z, kGdnQkScale * pf[m].w, hi.y, lo.y);
                    const int o = gdn_swz(t, r);
                    *reinterpret_cast<uint2*>((isq ? sqh : skh) + o) = hi;
                    *reinterpret_cast<uint2*>((isq ? sql : skl) + o) = lo;
                }
            }
        }
    };
    stage(t0, (int) min((int64_t) GCH, T - t0));
    __syncthreads();

    for (int ch = 0; ch < nch; ++ch) {
        const int64_t c_t0 = t0 + (int64_t) ch * GCH;
        const int n = (int) min((int64_t) GCH, T - c_t0);
        const int64_t hv = (int64_t) ch * HV + vh;
        if (ch + 1 < nch) {   // the next chunk's inputs into L2: k / q rows (block), this warp's v, T, P
            const int64_t n_t0 = c_t0 + GCH;
            for (int p = tid; p < GCH * 8; p += NT) {
                const int t = p >> 3, l = p & 7;
                if (n_t0 + t < T) gdn_pf_l2(h + (n_t0 + t) * C + (l < 4 ? HK * S : 0) + qh * S + 32 * (l & 3));
            }
            if (n_t0 + lane < T) gdn_pf_l2(h + (n_t0 + lane) * C + 2 * HK * S + vh * S + c0);
            gdn_pf_l2(reinterpret_cast<const float*>(tm + (hv + HV) * 2 * GCH * GCH + lane * 64));
            gdn_pf_l2(reinterpret_cast<const float*>(pm + (hv + HV) * 2 * GCH * GCH + lane * 64));
        }
        // X^T = S^T K^T and QS^T = S^T Q^T (16 columns x 32 tokens): S^T scaled to [2^12, 2^13), q / k by 2^8
        float ss, iss;
        {
            float m = 0.0f;
#pragma unroll
            for (int jt = 0; jt < S / 8; ++jt)
#pragma unroll
                for (int e = 0; e < 4; ++e) m = fmaxf(m, fabsf(st[jt][e]));
            ss = gdn_p2_scale(gdn_warp_max(m), iss);
        }
        float ax[4][4] = {}, aq[4][4] = {};
#pragma unroll
        for (int kk = 0; kk < S / 8; ++kk) {
            uint32_t ah[2], al[2];
            gdn_split2(ss * st[kk][0], ss * st[kk][1], ah[0], al[0]);
            gdn_split2(ss * st[kk][2], ss * st[kk][3], ah[1], al[1]);
            const int o = gdn_swz(lane, 8 * kk);   // matrix nt = lane / 8: tokens 8nt .. 8nt + 7, rows 8kk ..
            uint32_t bkh[4], bkl[4], bqh[4], bql[4];
            gdn_ldsm4(bkh, skh + o);
            gdn_ldsm4(bkl, skl + o);
            gdn_ldsm4(bqh, sqh + o);
            gdn_ldsm4(bql, sql + o);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                gdn_mma3(ax[nt], ah, al, bkh[nt], bkl[nt]);
                gdn_mma3(aq[nt], ah, al, bqh[nt], bql[nt]);
            }
        }
        // this thread's tokens: 8nt + 2q4 + (e & 1), columns c0 + g + 8 (e >> 1)
        float (&y)[4][4] = ax;   // Y^T over X^T, in place
        const float* gmc = gm + hv * GCH;
        const float glast = gdn_ldg_f32(gmc + GCH - 1);
        {
            const float ix = iss * (1.0f / kGdnQkScale);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int tok = 8 * nt + 2 * q4 + e;
                    const float ex = expf(gdn_ldg_f32(gmc + tok));
                    const float bt = tok < n ? gdn_ldg_f32(beta + (c_t0 + tok) * HV + vh) : 0.0f;
                    const float* vr = h + (c_t0 + tok) * C + 2 * HK * S + vh * S + c0 + g;
                    const float v0 = tok < n ? gdn_ldg_f32(vr) : 0.0f, v1 = tok < n ? gdn_ldg_f32(vr + 8) : 0.0f;
                    y[nt][e] = bt * (v0 - ex * (ax[nt][e] * ix));
                    y[nt][e + 2] = bt * (v1 - ex * (ax[nt][e + 2] * ix));
                    aq[nt][e] *= ex * ix;   // e^g QS^T, true scale
                    aq[nt][e + 2] *= ex * ix;
                }
        }
        // D^T = Y^T T^T: A = Y^T (k-step ki = tokens 8ki ..), B[i][t] = T[t][i] = {T[8nt + g][8ki + 2q4], + 1}
        float d[4][4] = {};
        {
            float m = 0.0f;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) m = fmaxf(m, fabsf(y[nt][e]));
            float iy;
            const float sy = gdn_p2_scale(gdn_warp_max(m), iy);
            const uint32_t* th = reinterpret_cast<const uint32_t*>(tm + hv * 2 * GCH * GCH);
            const uint32_t* tl = th + GCH * GCH / 2;
#pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
                uint32_t ah[2], al[2];
                gdn_split2(sy * y[ki][0], sy * y[ki][1], ah[0], al[0]);
                gdn_split2(sy * y[ki][2], sy * y[ki][3], ah[1], al[1]);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int w = (8 * nt + g) * (GCH / 2) + 4 * ki + q4;
                    gdn_mma3(d[nt], ah, al, gdn_ldg_u32(th + w), gdn_ldg_u32(tl + w));
                }
            }
            const float f = iy * __ldg(sc + 2 * hv);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) d[nt][e] *= f;
        }
        // D^T and the decayed Dd^T[col][t] = e^(gL - g_t) D^T[col][t] as split A operands (D itself is not needed after)
        uint32_t dh[4][2], dl[4][2], eh[4][2], el[4][2];
        float id, idd;
        {
            float dec[4][2];   // e^(gL - g_t) of this thread's tokens
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) dec[nt][e] = expf(glast - gdn_ldg_f32(gmc + 8 * nt + 2 * q4 + e));
            float m = 0.0f, mm = 0.0f;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    m = fmaxf(m, fabsf(d[nt][e]));
                    mm = fmaxf(mm, fabsf(d[nt][e] * dec[nt][e & 1]));
                }
            const float sd = gdn_p2_scale(gdn_warp_max(m), id), sdd = gdn_p2_scale(gdn_warp_max(mm), idd);
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                gdn_split2(sd * d[kt][0], sd * d[kt][1], dh[kt][0], dl[kt][0]);
                gdn_split2(sd * d[kt][2], sd * d[kt][3], dh[kt][1], dl[kt][1]);
                const float f0 = sdd * dec[kt][0], f1 = sdd * dec[kt][1];
                gdn_split2(f0 * d[kt][0], f1 * d[kt][1], eh[kt][0], el[kt][0]);
                gdn_split2(f0 * d[kt][2], f1 * d[kt][3], eh[kt][1], el[kt][1]);
            }
        }
        // o^T = e^g QS^T + D^T P^T: the accumulator starts at e^g QS^T over the products' scale (exact powers of two)
        {
            const float ip = __ldg(sc + 2 * hv + 1), f = id * ip, f_inv = 1.0f / f;
            const uint32_t* ph = reinterpret_cast<const uint32_t*>(pm + hv * 2 * GCH * GCH);
            const uint32_t* pl = ph + GCH * GCH / 2;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) aq[nt][e] *= f_inv;
#pragma unroll
            for (int ki = 0; ki < 4; ++ki)
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int w = (8 * nt + g) * (GCH / 2) + 4 * ki + q4;
                    gdn_mma3(aq[nt], dh[ki], dl[ki], gdn_ldg_u32(ph + w), gdn_ldg_u32(pl + w));
                }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int tok = 8 * nt + 2 * q4 + (e & 1), col = c0 + g + 8 * (e >> 1);
                    if (tok < n) oc_out[(c_t0 + tok) * HV * S + vh * S + col] = aq[nt][e] * f * out_scale;
                }
        }
        const bool more = ch + 1 < nch;
        // S^T = e^gL S^T + Dd^T K, accumulated at the products' scale sdd * 2^8
        {
            const float up = expf(glast) * (kGdnQkScale / idd), down = idd * (1.0f / kGdnQkScale);
#pragma unroll
            for (int nj = 0; nj < S / 8; ++nj) {
                const int o = gdn_swz(lane, 8 * nj);   // matrix kt = lane / 8: tokens 8kt .. 8kt + 7, rows 8nj ..
                uint32_t bh[4], bl[4];
                gdn_ldsm4t(bh, skh + o);
                gdn_ldsm4t(bl, skl + o);
#pragma unroll
                for (int e = 0; e < 4; ++e) st[nj][e] *= up;
#pragma unroll
                for (int kt = 0; kt < 4; ++kt) gdn_mma3(st[nj], eh[kt], el[kt], bh[kt], bl[kt]);
#pragma unroll
                for (int e = 0; e < 4; ++e) st[nj][e] *= down;
            }
        }
        __syncthreads();   // every read of this chunk's k / q done
        if (more) {
            stage(c_t0 + GCH, (int) min((int64_t) GCH, T - c_t0 - GCH));
            __syncthreads();
        }
    }
#pragma unroll
    for (int jt = 0; jt < S / 8; ++jt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            state[(size_t) (8 * jt + 2 * q4 + (e & 1)) * rs + (size_t) vh * S + c0 + g + 8 * (e >> 1)] = st[jt][e];
#endif
}

// The scratch, one a device, allocated on first use and kept: GSB_T / GCH chunks x HV of T and P (hi/lo halves), gamma
// and the two scales: 12.8 MB.
float* gdn_chunk_t_scratch() {
    static float* bufs[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return nullptr; }
    if (bufs[dev] == nullptr) {
        const size_t n = (size_t) (GSB_T / GCH) * HV * (2 * GCH * GCH + GCH + 2);
        if (cudaMalloc((void**) &bufs[dev], n * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            bufs[dev] = nullptr;
        }
    }
    return bufs[dev];
}

// STRATA_GDN_CHUNKED_SM75=1's recurrence of T tokens (any T > 0): super-blocks of GSB_T tokens, a prep and a scan
// launch each.  An error return means nothing was launched (the caller takes another kernel): before sm_75, or no
// scratch.  First use on a device logs the scan's occupancy.
cudaError_t gdn_rec_chunked_sm75(float* state, const float* h, const float* gate, const float* beta, float* y, int64_t T,
                                 cudaStream_t s) {
    if (T <= 0) return cudaSuccess;
    static int known[64] = {};   // per device: 0 not asked yet, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return cudaErrorNotSupported; }
    if (known[dev] == 0) {
        int major = 0, minor = 0, sms = 0, per_sm = 0;
        bool yes = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                   cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess &&
                   cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess &&
                   major * 10 + minor >= 75;
        if (yes) {
            cudaFuncSetAttribute(gdn_chunk_scan_t_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                 (int) cudaSharedmemCarveoutMaxShared);
            yes = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gdn_chunk_scan_t_kernel, 32 * GT_WARPS,
                                                                kChunkScanTSmem) == cudaSuccess &&
                  per_sm > 0;
        }
        if (!yes) cudaGetLastError();
        known[dev] = yes ? 1 : 2;
        std::fprintf(stderr, "strata: CUDA%d GDN prompt recurrence chunked on tensor cores (STRATA_GDN_CHUNKED_SM75=1): %s "
                             "(scan %d blocks/SM x %d SMs for %d blocks, %zu B shared)\n", dev, yes ? "on" : "off", per_sm,
                     sms, HK * (S / GDV / GT_NSL), kChunkScanTSmem);
    }
    if (known[dev] != 1) return cudaErrorNotSupported;
    float* scratch = gdn_chunk_t_scratch();
    if (scratch == nullptr) return cudaErrorMemoryAllocation;
    const int nb = (int) ((std::min<int64_t>(GSB_T, T) + GCH - 1) / GCH);
    const size_t tsz = (size_t) nb * HV * GCH * GCH;   // floats: GCH x GCH hi + lo halves a (chunk, head)
    __half* tm = reinterpret_cast<__half*>(scratch);
    __half* pm = reinterpret_cast<__half*>(scratch + tsz);
    float* gm = scratch + 2 * tsz;
    float* sc = gm + (size_t) nb * HV * GCH;
    for (int64_t t0 = 0; t0 < T; t0 += GSB_T) {
        const int nch = (int) ((std::min<int64_t>(GSB_T, T - t0) + GCH - 1) / GCH);
        gdn_chunk_prep_hl_kernel<<<dim3((unsigned) nch, HK), 256, kChunkPrepSmem, s>>>(h, gate, beta, tm, pm, gm, sc, t0, T);
        gdn_chunk_scan_t_kernel<<<dim3(HK, S / GDV / GT_NSL), 32 * GT_WARPS, kChunkScanTSmem, s>>>(state, h, beta, tm, pm, gm,
                                                                                                 sc, y, t0, nch, T);
    }
    return cudaSuccess;
}
#endif
