// src/kernels/turbo_skip_parity.cpp - PLAN-turbo-live's Turbo skip against its contract (the drop's rule rides
// the same tables): a row with skip on leaves out every pick its tier bar rejects - bar 1 = not resident in VRAM
// (res < 0), bar 2 = in neither VRAM nor the RAM tier (res < 0 and ram < 0) - and writes it as the row's top-1
// at weight 0, the kept weights renormalising on their own sum as the router's does (clamped at 2^-14); the
// row's top-1 pick always stays.  Checked four ways, on the fused 512x10 kernels and on the generic top-k plus
// turbo_drop_apply (512x10 through router_top10, the family's 256x8 through native_router_topk):
//   * skip off / a zero table: bitwise-identical to today's launch (ids and weights by memcmp);
//   * every pick resident: bitwise-identical to off;
//   * no pick resident: the top-1 alone (ids[1..] = ids[0], weights 0, w[0] = 1);
//   * random residency (and drop on top of it): against a host replay of the contract - ids exact, weights
//     within 2e-6 - and the per-request counters plus the skipped-pick usage table against the same replay.
// Run: turbo_skip_parity   (ctest: turbo_skip_parity; GPU, synthetic, no model)
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/router_top10.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
template <typename T>
T* up(const std::vector<T>& h) {
    void* p = nullptr;
    check(cudaMalloc(&p, h.size() * sizeof(T) + 64), "malloc");
    check(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
    return (T*) p;
}
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    return h;
}

// the router's top-k picks of one logits row: softmax over all, stable descending selection, ties by index
std::vector<std::pair<int, float>> topk(const float* lg, int ne, int k) {
    std::vector<float> p(ne);
    float m = -INFINITY;
    for (int i = 0; i < ne; ++i) m = std::max(m, lg[i]);
    float sum = 0.0f;
    for (int i = 0; i < ne; ++i) { p[i] = std::exp(lg[i] - m); sum += p[i]; }
    for (int i = 0; i < ne; ++i) p[i] /= sum;
    std::vector<std::pair<int, float>> out;
    for (int r = 0; r < k; ++r) {
        int bi = 0;
        float bv = -INFINITY;
        for (int i = 0; i < ne; ++i)
            if (p[i] > bv) { bv = p[i]; bi = i; }
        out.push_back({bi, bv});
        p[bi] = -INFINITY;
    }
    return out;
}

// the contract's row: keep = the picks the drop leaves (10-N or 8-N), alive = kept AND resident under the bar,
// the alive weights renormalised on their sum (clamped), everything else the row's top-1 at weight 0.  `w0`,
// when given, are the router's already-renormalised weights (the apply kernel's input); else the raw probs.
void expect_row(const std::vector<std::pair<int, float>>& picks, int k, int keep, int bar,
                const std::vector<int32_t>& res, const std::vector<int32_t>& ram, const float* w0,
                std::vector<int32_t>& ids, std::vector<float>& w) {
    const int top1 = picks[0].first;
    std::vector<bool> alive((size_t) k, false);
    float s = 0.0f;
    for (int r = 0; r < k; ++r) {
        const int e = picks[r].first;
        bool a = r < keep;
        if (a && r > 0 && bar != 0) a = res[e] >= 0 || (bar >= 2 && !ram.empty() && ram[e] >= 0);
        alive[r] = a;
        if (a) s += w0 != nullptr ? w0[r] : picks[r].second;
    }
    const float inv = 1.0f / (s < 6.103515625e-5f ? 6.103515625e-5f : s);
    ids.assign((size_t) k, top1);
    w.assign((size_t) k, 0.0f);
    for (int r = 0; r < k; ++r)
        if (alive[r]) { ids[r] = picks[r].first; w[r] = (w0 != nullptr ? w0[r] : picks[r].second) * inv; }
}

int g_bad = 0;
void expect(bool ok, const char* what) {
    if (!ok) { std::printf("    *** %s ***\n", what); ++g_bad; }
}
bool close(const float* a, const float* b, size_t n, float tol) {
    for (size_t i = 0; i < n; ++i)
        if (!(std::fabs(a[i] - b[i]) <= tol)) return false;
    return true;
}

}  // namespace

int main() {
    using namespace strata::kernels;
    std::mt19937 rng(20261123);
    std::normal_distribution<float> gauss(0.0f, 2.0f);
    cudaStream_t st = nullptr;
    check(cudaStreamCreate(&st), "stream");

    // ================================ the fused 512x10 kernels ================================
    {
        const int n = 20, k = 10;
        std::vector<float> lg((size_t) n * 512);
        for (auto& v : lg) v = gauss(rng);
        for (int t = 0; t < n; t += 5) {   // exact ties for the top, so the tie rule is exercised under a skip
            lg[(size_t) t * 512 + 7] = 6.0f;
            lg[(size_t) t * 512 + 300] = 6.0f;
        }
        std::vector<int32_t> drop((size_t) n, 0), skip((size_t) n, 0);
        std::vector<int32_t> res_all((size_t) n * 512, 0), res_none((size_t) n * 512, -1),
            res_rand((size_t) n * 512, 0), ram((size_t) n * 512, 0);
        for (auto& v : res_rand) v = (int32_t) (rng() % 2);
        for (auto& v : ram) v = (int32_t) (rng() % 2) == 0 ? 0 : -1;
        for (int t = 0; t < n; ++t) {
            if (t % 3 != 0) skip[t] = t % 2 ? 1 : 2;   // rows off stay off (t % 3 == 0)
            if (rng() % 3 == 0) drop[t] = 2;
        }
        skip[0] = 0;

        float* d_l = up(lg);
        int32_t* d_i = nullptr;
        float* d_w = nullptr;
        check(cudaMalloc(&d_i, (size_t) n * k * 4), "i");
        check(cudaMalloc(&d_w, (size_t) n * k * 4), "w");
        native_router_top10_multi(d_l, d_i, d_w, n, st);   // today's call: no table at all
        check(cudaDeviceSynchronize(), "off");
        const auto off_i = down(d_i, (size_t) n * k);
        const auto off_w = down(d_w, (size_t) n * k);

        int32_t* d_drop = up(drop);
        int32_t* d_skip = up(skip);
        int32_t* d_res_all = up(res_all);
        int32_t* d_res_none = up(res_none);
        int32_t* d_res_rand = up(res_rand);
        int32_t* d_ram = up(ram);
        const int32_t* res_d[4] = {d_res_all, d_res_none, d_res_rand, d_res_rand};
        const std::vector<int32_t>* res_h[4] = {&res_all, &res_none, &res_rand, &res_rand};
        int bad0 = g_bad;

        {   // a zero drop/skip table: bitwise the table-free launch ("only branch on the mapped per-row value")
            std::vector<int32_t> z((size_t) n, 0);
            TurboRoute tr;
            tr.drop = up(z);
            tr.skip = up(z);
            native_router_top10_multi(d_l, d_i, d_w, n, st, &tr);
            check(cudaDeviceSynchronize(), "zero");
            const auto i2 = down(d_i, (size_t) n * k);
            const auto w2 = down(d_w, (size_t) n * k);
            expect(std::memcmp(i2.data(), off_i.data(), i2.size() * 4) == 0, "fused: a zero table changed ids");
            expect(std::memcmp(w2.data(), off_w.data(), w2.size() * 4) == 0, "fused: a zero table changed weights");
            cudaFree((void*) tr.drop);
            cudaFree((void*) tr.skip);
        }
        for (int c = 0; c < 4; ++c) {   // all resident / none resident / random, with and without the bar-2 ram row
            TurboRoute tr;
            tr.drop = d_drop;
            tr.skip = d_skip;
            tr.res = res_d[c];
            tr.ram = c == 3 ? d_ram : nullptr;
            native_router_top10_multi(d_l, d_i, d_w, n, st, &tr);
            check(cudaDeviceSynchronize(), "tier");
            const auto gi = down(d_i, (size_t) n * k);
            const auto gw = down(d_w, (size_t) n * k);
            int wb = 0;
            for (int t = 0; t < n; ++t) {
                if (skip[t] == 0 && drop[t] == 0) {   // an off row of a tiered launch: bitwise
                    if (std::memcmp(gi.data() + (size_t) t * k, off_i.data() + (size_t) t * k, (size_t) k * 4) != 0 ||
                        !close(gw.data() + (size_t) t * k, off_w.data() + (size_t) t * k, (size_t) k, 0.0f))
                        { std::printf("    *** fused row %d: an off row moved ***\n", t); ++wb; }
                    continue;
                }
                const int d = drop[t];
                const int keep = d <= 0 ? 10 : d >= 9 ? 1 : 10 - d;
                if (c == 0 && d == 0) {   // every pick resident and no drop: bitwise the table-free router's
                    if (std::memcmp(gi.data() + (size_t) t * k, off_i.data() + (size_t) t * k, (size_t) k * 4) != 0 ||
                        !close(gw.data() + (size_t) t * k, off_w.data() + (size_t) t * k, (size_t) k, 0.0f))
                        { std::printf("    *** fused row %d: an all-resident skip row moved off the exact bits ***\n", t); ++wb; }
                    continue;
                }
                std::vector<int32_t> ids;
                std::vector<float> w;
                expect_row(topk(lg.data() + (size_t) t * 512, 512, 10), 10, keep, skip[t], *res_h[c],
                           c == 3 ? ram : std::vector<int32_t>{}, nullptr, ids, w);
                if (std::memcmp(ids.data(), gi.data() + (size_t) t * k, (size_t) k * 4) != 0) {
                    std::printf("    *** fused row %d (case %d): ids differ from the contract ***\n", t, c);
                    ++wb;
                }
                if (!close(w.data(), gw.data() + (size_t) t * k, (size_t) k, 2e-6f)) {
                    std::printf("    *** fused row %d (case %d): weights differ from the contract ***\n", t, c);
                    ++wb;
                }
                if (c == 1 && skip[t] != 0) {   // nothing resident: the top-1 alone
                    for (int r = 1; r < k; ++r)
                        if (gi[(size_t) t * k + r] != gi[(size_t) t * k] || gw[(size_t) t * k + r] != 0.0f) {
                            std::printf("    *** fused row %d: a none-resident row kept a second pick ***\n", t);
                            ++wb;
                        }
                    if (!(gw[(size_t) t * k] > 0.999999f && gw[(size_t) t * k] < 1.000001f)) {
                        std::printf("    *** fused row %d: the kept top-1 is not renormalised to 1 ***\n", t);
                        ++wb;
                    }
                }
            }
            expect(wb == 0, "fused: skip rows differ from the contract");
        }
        std::printf("  %-52s %s\n", "fused 512x10 skip (off bitwise / all / none / random x2)",
                    g_bad != bad0 ? "*** NO ***" : "matches the contract");
        cudaFree(d_l); cudaFree(d_i); cudaFree(d_w); cudaFree(d_drop); cudaFree(d_skip);
        cudaFree(d_res_all); cudaFree(d_res_none); cudaFree(d_res_rand); cudaFree(d_ram);
    }

    // ================================ the generic top-k + turbo_drop_apply ================================
    for (int shape = 0; shape < 2; ++shape) {
        const int ne = shape == 0 ? 512 : 256, k = shape == 0 ? 10 : 8, n = 9;
        const char* name = shape == 0 ? "router_top10 512x10 + apply" : "native_router_topk 256x8 + apply";
        std::vector<float> lg((size_t) n * ne);
        for (auto& v : lg) v = gauss(rng);
        std::vector<int32_t> drop((size_t) n, 0), skip((size_t) n, 1);
        drop[3] = 2;
        drop[7] = k - 1;   // cut to the top-1 by the drop alone
        skip[0] = 0;       // an off row: its bits must not move at all
        std::vector<int32_t> res_all((size_t) ne, 0), res_none((size_t) ne, -1), res_rand((size_t) ne, 0);
        for (auto& v : res_rand) v = (int32_t) (rng() % 2);
        float* d_l = up(lg);
        int32_t* d_i = nullptr;
        float* d_w = nullptr;
        check(cudaMalloc(&d_i, (size_t) n * k * 4), "i");
        check(cudaMalloc(&d_w, (size_t) n * k * 4), "w");
        int32_t* d_drop = up(drop);
        int32_t* d_skip = up(skip);
        const int32_t* res_d[3] = {up(res_all), up(res_none), up(res_rand)};
        const std::vector<int32_t>* res_h[3] = {&res_all, &res_none, &res_rand};
        int bad1 = g_bad;
        for (int c = 0; c < 3; ++c) {
            if (shape == 0) router_top10(d_l, n, ne, k, (int*) d_i, d_w, st);
            else native_router_topk(d_l, d_i, d_w, ne, k, n, st);
            check(cudaGetLastError(), "router");
            check(cudaDeviceSynchronize(), "router sync");
            const auto ri = down(d_i, (size_t) n * k);
            const auto rw = down(d_w, (size_t) n * k);
            TurboRoute tr;
            tr.drop = d_drop;
            tr.skip = d_skip;
            tr.res = res_d[c];
            turbo_drop_apply(d_i, d_w, n, k, tr, st);
            check(cudaDeviceSynchronize(), "apply");
            const auto gi = down(d_i, (size_t) n * k);
            const auto gw = down(d_w, (size_t) n * k);
            for (int t = 0; t < n; ++t) {
                if (skip[t] == 0 && drop[t] == 0) {   // the off row: bitwise the router's
                    if (std::memcmp(gi.data() + (size_t) t * k, ri.data() + (size_t) t * k, (size_t) k * 4) != 0 ||
                        std::memcmp(gw.data() + (size_t) t * k, rw.data() + (size_t) t * k, (size_t) k * 4) != 0) {
                        std::printf("    *** %s: an off row moved ***\n", name);
                        ++g_bad;
                    }
                    continue;
                }
                const int d = drop[t];
                const int keep = d >= k ? 1 : (d <= 0 ? k : k - d);
                std::vector<std::pair<int, float>> picks;
                for (int r = 0; r < k; ++r) picks.push_back({ri[(size_t) t * k + r], rw[(size_t) t * k + r]});
                std::vector<int32_t> ids;
                std::vector<float> w;
                expect_row(picks, k, keep, skip[t], *res_h[c], std::vector<int32_t>{}, rw.data() + (size_t) t * k,
                           ids, w);
                if (std::memcmp(ids.data(), gi.data() + (size_t) t * k, (size_t) k * 4) != 0 ||
                    !close(w.data(), gw.data() + (size_t) t * k, (size_t) k, 1e-6f)) {
                    std::printf("    *** %s row %d (case %d) differs from the contract ***\n", name, t, c);
                    ++g_bad;
                }
                if (c == 1 && skip[t] != 0 && d <= 0) {   // nothing resident: top-1 only
                    for (int r = 1; r < k; ++r)
                        if (gi[(size_t) t * k + r] != gi[(size_t) t * k] || gw[(size_t) t * k + r] != 0.0f) {
                            std::printf("    *** %s row %d: a none-resident row kept a second pick ***\n", name, t);
                            ++g_bad;
                        }
                }
            }
            if (c == 0) {   // every pick resident: the apply leaves the router's own bits (no renormalisation)
                int m = 0;
                for (int t = 0; t < n; ++t)
                    if (std::memcmp(gi.data() + (size_t) t * k, ri.data() + (size_t) t * k, (size_t) k * 4) != 0 ||
                        std::memcmp(gw.data() + (size_t) t * k, rw.data() + (size_t) t * k, (size_t) k * 4) != 0)
                        ++m;
                expect(m == (drop[3] != 0) + (drop[7] != 0),   // only the drop rows may move
                       "apply: all-resident rows other than the drop's moved");
            }
        }
        std::printf("  %-52s %s\n", name, g_bad != bad1 ? "*** NO ***" : "matches the contract");
        cudaFree(d_l); cudaFree(d_i); cudaFree(d_w); cudaFree(d_drop); cudaFree(d_skip);
        cudaFree((void*) res_d[0]); cudaFree((void*) res_d[1]); cudaFree((void*) res_d[2]);
    }

    // ================================ the counters and the skipped-pick usage ================================
    {
        const int n = 4, k = 10;
        std::vector<float> lg((size_t) n * 512);
        for (auto& v : lg) v = gauss(rng);
        std::vector<int32_t> drop = {0, 2, 0, 0}, skip = {0, 0, 1, 2}, tag = {0, 3, 3, 1};
        std::vector<int32_t> res(512), ram(512, -1);
        for (auto& v : res) v = (int32_t) (rng() % 2);
        for (auto& v : ram) if (rng() % 4 == 0) v = 0;
        float* d_l = up(lg);
        int32_t* d_i = nullptr;
        float* d_w = nullptr;
        check(cudaMalloc(&d_i, (size_t) n * k * 4), "i");
        check(cudaMalloc(&d_w, (size_t) n * k * 4), "w");
        int32_t* d_drop = up(drop);
        int32_t* d_skip = up(skip);
        int32_t* d_tag = up(tag);
        int32_t* d_res = up(res);
        int32_t* d_ram = up(ram);
        unsigned long long* d_stats = nullptr;
        check(cudaMalloc(&d_stats, 16 * 3 * sizeof(unsigned long long)), "stats");
        check(cudaMemset(d_stats, 0, 16 * 3 * sizeof(unsigned long long)), "stats0");
        float* d_usage = nullptr;
        check(cudaMalloc(&d_usage, 512 * sizeof(float)), "usage");
        check(cudaMemset(d_usage, 0, 512 * sizeof(float)), "usage0");
        TurboRoute tr;
        tr.drop = d_drop; tr.skip = d_skip; tr.tag = d_tag; tr.res = d_res; tr.ram = d_ram;
        tr.stats = d_stats; tr.skip_usage = d_usage; tr.layer = 0; tr.n_expert = 512;
        native_router_top10_multi(d_l, d_i, d_w, n, st, &tr);
        check(cudaDeviceSynchronize(), "count");
        const auto sv = down(d_stats, 16 * 3);
        const auto uv = down(d_usage, 512);
        unsigned long long want[16][3] = {};
        std::vector<float> wusage(512, 0.0f);
        for (int t = 0; t < n; ++t) {
            if (drop[t] == 0 && skip[t] == 0) continue;   // an off row is not counted at all
            const auto picks = topk(lg.data() + (size_t) t * 512, 512, 10);
            const int d = drop[t];
            const int keep = d <= 0 ? 10 : d >= 9 ? 1 : 10 - d;
            const int b = tag[t];
            want[b][0] += k;
            for (int r = 1; r < k; ++r) {
                const int e = picks[r].first;
                if (r >= keep) { ++want[b][1]; continue; }
                if (skip[t] != 0 && !(res[e] >= 0 || (skip[t] >= 2 && ram[e] >= 0))) {
                    ++want[b][2];
                    wusage[e] += 1.0f;
                }
            }
        }
        int cb = 0;
        for (int b = 0; b < 16; ++b)
            for (int i = 0; i < 3; ++i)
                if (sv[(size_t) b * 3 + i] != want[b][i]) ++cb;
        float uwant = 0.0f, ugot = 0.0f;
        for (int e = 0; e < 512; ++e) {
            if (std::fabs(uv[e] - wusage[e]) > 1e-3f) ++cb;
            uwant += wusage[e];
            ugot += uv[e];
        }
        (void) uwant; (void) ugot;
        expect(cb == 0, "counters / usage do not match the rows");
        std::printf("  %-52s %s\n", "counters (P/D/S per bucket) and the skipped-pick usage",
                    cb == 0 ? "match" : "*** NO ***");
        cudaFree(d_l); cudaFree(d_i); cudaFree(d_w); cudaFree(d_drop); cudaFree(d_skip); cudaFree(d_tag);
        cudaFree(d_res); cudaFree(d_ram); cudaFree(d_stats); cudaFree(d_usage);
    }

    std::printf("\nturbo_skip_parity: %d failures\n", g_bad);
    if (g_bad) return 1;
    std::printf("turbo_skip_parity OK\n");
    return 0;
}
