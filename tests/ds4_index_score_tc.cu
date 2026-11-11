// tests/ds4_index_score_tc.cu - the indexer's scores on the tensor cores (index_score's tc kernel) against the CUDA
// cores' kernel: packed FP4 keys (kKvFp4p: E2M1 codes, a power-of-two scale a 32) and FP4-quantized queries, a decode window's
// rows and a prompt slab's, the shared rows and the candidates' (each query its own blocks of positions), per-query
// limits - the same -inf everywhere, the finite scores within FP32's rounding of each other (the products are exact,
// their sums run in another order), and how many of each row's top 512 positions the two agree on.
// Also the largest relative difference of a score (those of at least 1% of the row's largest), and per 1000 queries
// the queries whose top 512 differ and the picks that differ.
//   ninja -C build ds4_index_score_tc   (a CMake target, EXCLUDE_FROM_ALL: the project's CPU flags - no -mavx2/-mfma,
//   which an AVX1 host (Ivy Bridge) cannot run)
//   DS4_INDEX_TC=1 build/ds4_index_score_tc   (the engine's default is the CUDA cores' kernel; on compute capability
//   7.5 DS4_INDEX_TC_SM75=2 for every case; =1 takes the prompt slabs' cases alone (T > 16) - the run prints "ds4:
//   indexer scores on the int8 tensor cores" when it is taken)
#include "ds4_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::ds4;

static float fp4h(unsigned c) {   // (fp4v on the host)
    static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (c & 8u) ? -mag[c & 7u] : mag[c & 7u];
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {   // (no GPU: no result, not a pass)
        std::printf("no CUDA device - FAIL\n");
        return 2;
    }
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> code(0, 15), sexp(-6, 3), cbyte(121, 130);
    std::normal_distribution<float> nd;
    const int HD = 128, H = 32, k = 512;
    const size_t RB = kv_row_bytes(kKvFp4p, HD);   // (the cache's rows: codes, scales, padded to 16)
    int fails = 0;
    struct Case { int n, T; bool cand; };
    for (Case cs : {Case{1000, 1, false}, Case{65536, 3, false}, Case{131072, 6, false}, Case{20000, 64, false},
                    Case{65536, 3, true}, Case{40000, 16, true}, Case{25000, 512, false}, Case{4000, 2048, false}}) {
        const int n = cs.n, T = cs.T, blk = 8, nb = (n + blk - 1) / blk;
        std::vector<uint8_t> keys((size_t) n * RB);
        for (int p = 0; p < n; ++p) {
            uint8_t* r = keys.data() + (size_t) p * RB;
            for (int b = 0; b < HD / 2; ++b) r[b] = (uint8_t) (code(rng) | (code(rng) << 4));
            for (int b = 0; b < HD / 32; ++b) r[HD / 2 + b] = (uint8_t) cbyte(rng);
        }
        std::vector<float> q((size_t) T * H * HD), w((size_t) T * 64);
        for (size_t i = 0; i < q.size(); i += 32) {   // (a 32's codes times its power of two: quant 3's grid)
            const float sc = std::ldexp(1.0f, sexp(rng));
            for (int j = 0; j < 32; ++j) q[i + j] = fp4h((unsigned) code(rng)) * sc;
        }
        for (float& x : w) x = nd(rng) * 0.1f;
        Rows lim;
        std::vector<int> cand;
        int cmax = 0;
        if (T <= 16)
            for (int t = 0; t < T; ++t) {   // (per-query limits: the later queries see more)
                lim.n[t] = n - (T - 1 - t) * 37;
                lim.k[t] = std::min(k, lim.n[t]);
            }
        else {   // (a prompt slab's rows: generated, query t at position n - T + t)
            lim.gen = 1;
            lim.p0 = n - T;
            lim.R = 1;
            lim.kcap = k;
        }
        Rows cl = lim;
        if (cs.cand) {   // each query 2048 random blocks (its own), the first k of them counted
            cmax = 2048;
            cand.resize((size_t) T * cmax);
            std::uniform_int_distribution<int> bd(0, nb - 1);
            for (int t = 0; t < T; ++t) {
                for (int j = 0; j < cmax; ++j) cand[(size_t) t * cmax + j] = bd(rng);
                cl.k[t] = cmax - t * 3;
            }
        }
        const int ncols = cs.cand ? cmax * blk : n;
        uint8_t* dk;
        float *dq, *dw, *o1, *o2;
        int* dc = nullptr;
        cudaMalloc(&dk, keys.size());
        cudaMalloc(&dq, q.size() * 4);
        cudaMalloc(&dw, w.size() * 4);
        cudaMalloc(&o1, (size_t) T * ncols * 4);
        cudaMalloc(&o2, (size_t) T * ncols * 4);
        cudaMemcpy(dk, keys.data(), keys.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
        if (cs.cand) {
            cudaMalloc(&dc, cand.size() * 4);
            cudaMemcpy(dc, cand.data(), cand.size() * 4, cudaMemcpyHostToDevice);
        }
        const Rows& r = cs.cand ? cl : lim;
        index_score(dq, dw, dk, H, HD, T, ncols, dc, cmax, cs.cand ? blk : 1, r, o1, ncols, nullptr, kKvFp4p, false);
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        float ms[2] = {};
        for (int tc = 0; tc < 2; ++tc) {
            cudaEventRecord(a);
            for (int it = 0; it < 20; ++it)
                index_score(dq, dw, dk, H, HD, T, ncols, dc, cmax, cs.cand ? blk : 1, r, tc ? o2 : o1, ncols, nullptr,
                            kKvFp4p, tc == 1);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            cudaEventElapsedTime(&ms[tc], a, b);
        }
        std::vector<float> h1((size_t) T * ncols), h2(h1.size());
        cudaMemcpy(h1.data(), o1, h1.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h2.data(), o2, h2.size() * 4, cudaMemcpyDeviceToHost);
        bool masks = true;
        double worst = 0.0, agree = 0.0, worst_rel = 0.0;
        long long rows_differ = 0, picks_differ = 0;
        for (int t = 0; t < T; ++t) {
            const float* x = h1.data() + (size_t) t * ncols;
            const float* y = h2.data() + (size_t) t * ncols;
            float mx = 0.0f;
            for (int i = 0; i < ncols; ++i)
                if (std::isfinite(x[i])) mx = std::max(mx, std::fabs(x[i]));
            for (int i = 0; i < ncols; ++i) {
                if (std::isfinite(x[i]) != std::isfinite(y[i])) masks = false;
                else if (std::isfinite(x[i])) {
                    worst = std::max(worst, (double) std::fabs(x[i] - y[i]) / std::max(mx, 1e-30f));
                    if (std::fabs(x[i]) >= 0.01f * mx && x[i] != 0.0f)
                        worst_rel = std::max(worst_rel, (double) std::fabs(x[i] - y[i]) / std::fabs(x[i]));
                }
            }
            std::vector<int> ia(ncols), ib(ncols);
            for (int i = 0; i < ncols; ++i) ia[i] = ib[i] = i;
            const int kk = std::min(k, ncols);
            std::partial_sort(ia.begin(), ia.begin() + kk, ia.end(), [&](int u, int v) { return x[u] > x[v] || (x[u] == x[v] && u < v); });
            std::partial_sort(ib.begin(), ib.begin() + kk, ib.end(), [&](int u, int v) { return y[u] > y[v] || (y[u] == y[v] && u < v); });
            std::sort(ia.begin(), ia.begin() + kk);
            std::sort(ib.begin(), ib.begin() + kk);
            std::vector<int> both;
            std::set_intersection(ia.begin(), ia.begin() + kk, ib.begin(), ib.begin() + kk, std::back_inserter(both));
            agree += (double) both.size() / kk / T;
            rows_differ += (int) both.size() != kk;
            picks_differ += kk - (long long) both.size();
        }
        const bool ok = masks && worst < 2e-5;
        std::printf("n %6d, %4d rows%s: masks %s, worst difference %.2e of the row's largest (%.2e of a score), top-%d "
                    "agreement %.4f, per 1000 queries %.1f differ by %.1f picks; %.1f -> %.1f us a call%s\n", n, T,
                    cs.cand ? ", candidates" : "", masks ? "the same" : "DIFFER", worst, worst_rel, k, agree,
                    1000.0 * rows_differ / T, 1000.0 * picks_differ / T, 1000.0 * ms[0] / 20, 1000.0 * ms[1] / 20,
                    worst == 0.0 ? " (identical: the tensor cores not taken?)" : "");
        fails += !ok;
        cudaFree(dk);
        cudaFree(dq);
        cudaFree(dw);
        cudaFree(o1);
        cudaFree(o2);
        if (dc) cudaFree(dc);
    }
    if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        std::printf("CUDA error\n");
        ++fails;
    }
    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
