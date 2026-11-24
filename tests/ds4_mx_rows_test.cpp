// tests/ds4_mx_rows_test.cpp - the CPU lane's int8 MXFP4 rows (gg::mx_rows_tier): every vector tier this CPU runs
// against the scalar tier, bit for bit (no GPU).  1-16 tokens, 256 / 2304 / 5120 wide; random blocks and activations
// (mx_quant's), every E8M0 scale with activation scales from denormal to inf, the int16 worst case (codes +-12 against
// activations -128 / 127), and sums where fma and a double-rounded mul + add differ (the AVX1 tier's slow lanes).
// NaN against NaN counts as equal (the payload of a NaN is not an answer).
//   ds4_mx_rows_test [--bench]     --bench: ns a row and token, every tier, 1 / 2 / 4 / 8 tokens, one thread, in L2
#include "../src/ds4/ds4_gguf_cpu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace gg = maya::ds4::gguf;

namespace {

constexpr int kBytes = 17;
std::mt19937_64 g_rng(0x5eed);
long g_cmp = 0, g_diff = 0, g_nan = 0;

int act_lane(int b) { return 8 * (b >> 3) + ((b & 7) & 1) * 4 + ((b & 7) >> 1); }

// block b of an activation (k % 256 == 0) written raw: its 32 values, its scale, its -12 x sum
void set_act(uint8_t* act, int k, int b, const int8_t* v, float sc) {
    int8_t* q = (int8_t*) act + (b >> 2) * 128 + 16 * (b & 3);
    int s = 0;
    for (int j = 0; j < 16; ++j) q[j] = v[j], q[64 + j] = v[16 + j];
    for (int j = 0; j < 32; ++j) s += v[j];
    ((float*) (act + k))[act_lane(b)] = sc;
    ((int32_t*) (act + k + k / 8))[act_lane(b)] = -12 * s;
}
// a weight block: E8M0 e, value j's code c[j] (0..15)
void set_w(uint8_t* p, int e, const int* c) {
    p[0] = (uint8_t) e;
    for (int j = 0; j < 16; ++j) p[1 + j] = (uint8_t) (c[j] | (c[16 + j] << 4));
}

struct Case {
    int k = 0, R = 0;
    std::vector<uint8_t> w, act;   // R rows; 16 activations
    size_t rb() const { return (size_t) k / 32 * kBytes; }
    size_t ab() const { return gg::mx_act_bytes(k); }
    void init(int k_, int R_) {
        k = k_, R = R_;
        w.assign(rb() * R, 0);
        act.assign(ab() * 16, 0);
    }
    uint8_t* a(int i) { return act.data() + ab() * i; }
};

bool same(float a, float b) {
    if (std::isnan(a) && std::isnan(b)) {
        ++g_nan;
        return true;
    }
    return std::memcmp(&a, &b, 4) == 0;
}

// every tier against the scalar one at 1..16 tokens, a window of rows
void check(const char* what, Case& c) {
    const int r0 = c.R > 8 ? 3 : 0, r1 = c.R;
    const void* ap[16];
    for (int i = 0; i < 16; ++i) ap[i] = c.a(i);
    long cmp = 0, diff = 0;
    std::string tiers;
    for (int t = 1; t < gg::kMxTiers; ++t) {
        if (!gg::mx_tier_ok(t)) continue;
        tiers += std::string(" ") + gg::mx_tier_name(t);
        for (int n = 1; n <= 16; ++n) {
            std::vector<float> ys((size_t) n * (r1 - r0)), yt(ys.size(), -7.0f);
            std::vector<float*> ps(n), pt(n);
            for (int i = 0; i < n; ++i) ps[i] = ys.data() + (size_t) i * (r1 - r0), pt[i] = yt.data() + (size_t) i * (r1 - r0);
            gg::mx_rows_tier(gg::kMxScalar, c.w.data(), c.rb(), c.k, r0, r1, ap, n, ps.data());
            gg::mx_rows_tier(t, c.w.data(), c.rb(), c.k, r0, r1, ap, n, pt.data(), t == gg::kMxAvx1 ? 0 : 256);
            for (size_t i = 0; i < ys.size(); ++i) {
                ++cmp;
                if (!same(ys[i], yt[i])) {
                    if (diff < 3)
                        std::printf("  DIFF %s %s n=%d #%zu: scalar %a tier %a\n", what, gg::mx_tier_name(t), n, i,
                                    ys[i], yt[i]);
                    ++diff;
                }
            }
        }
    }
    g_cmp += cmp, g_diff += diff;
    std::printf("%-46s k %4d rows %3d, tiers%s: %ld values, %ld differ\n", what, c.k, r1 - r0,
                tiers.empty() ? " (none but scalar)" : tiers.c_str(), cmp, diff);
}

void rand_codes(int* cd) {
    for (int j = 0; j < 32; ++j) cd[j] = (int) (g_rng() & 15);
}

// random blocks (realistic or any scale), activations quantized from random floats (a zero block, a tiny one)
void case_random(int k, bool any_scale) {
    Case c;
    c.init(k, 67);
    for (int r = 0; r < c.R; ++r)
        for (int b = 0; b < k / 32; ++b) {
            int cd[32];
            rand_codes(cd);
            set_w(c.w.data() + c.rb() * r + (size_t) b * kBytes, any_scale ? (int) (g_rng() % 256) : 112 + (int) (g_rng() % 12), cd);
        }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(k);
    for (int i = 0; i < 16; ++i) {
        const float amp = std::ldexp(1.0f, (int) (g_rng() % 20) - 10);
        for (int j = 0; j < k; ++j) x[j] = amp * nd(g_rng);
        for (int j = 0; j < 32; ++j) x[32 + j] = 0.0f;   // (scale 0)
        for (int j = 0; j < 32; ++j) x[96 + j] = 1e-38f * nd(g_rng);
        gg::mx_quant(x.data(), k, c.a(i));
    }
    check(any_scale ? "random codes, every E8M0, mx_quant acts" : "random codes, weight-like scales, mx_quant acts", c);
}

// raw activations of any int8 (-128 too) with scales across the whole float range (denormal .. huge: inf, NaN)
void case_extreme(int k) {
    Case c;
    c.init(k, 40);
    for (int r = 0; r < c.R; ++r)
        for (int b = 0; b < k / 32; ++b) {
            int cd[32];
            const int mode = (int) (g_rng() % 4);
            for (int j = 0; j < 32; ++j) cd[j] = mode == 0 ? 7 : mode == 1 ? 15 : mode == 2 ? (j & 1 ? 7 : 15) : (int) (g_rng() & 15);
            set_w(c.w.data() + c.rb() * r + (size_t) b * kBytes, (int) (g_rng() % 256), cd);
        }
    for (int i = 0; i < 16; ++i)
        for (int b = 0; b < k / 32; ++b) {
            int8_t v[32];
            for (int j = 0; j < 32; ++j) v[j] = (int8_t) (g_rng() & 255);
            const int e = (int) (g_rng() % 300) - 150;
            const float sc = (g_rng() % 16 == 0) ? 0.0f : std::ldexp(1.0f + (float) (g_rng() % 1000) / 1000.0f, e);
            set_act(c.a(i), k, b, v, sc);
        }
    check("extreme: any int8, act scales 2^-150..2^150", c);
}

// the int16 worst case: every code +12 (or -12) against activations all 127 / all -128, largest scales that stay finite
void case_int16(int k) {
    Case c;
    c.init(k, 16);
    for (int r = 0; r < c.R; ++r)
        for (int b = 0; b < k / 32; ++b) {
            int cd[32];
            for (int j = 0; j < 32; ++j) cd[j] = (r & 1) ? 15 : 7;
            set_w(c.w.data() + c.rb() * r + (size_t) b * kBytes, r < 8 ? 254 : 128, cd);
        }
    for (int i = 0; i < 16; ++i)
        for (int b = 0; b < k / 32; ++b) {
            int8_t v[32];
            for (int j = 0; j < 32; ++j) v[j] = (int8_t) ((i + b) & 1 ? 127 : -128);
            set_act(c.a(i), k, b, v, i < 8 ? 1e-30f : 1.0f);
        }
    check("int16 worst case: codes +-12 x 127 / -128", c);
}

// fma(total, s, acc) where mul + add in double then to float rounds twice: acc 1 (group 0), then total 4097 x s =
// 2^-24 (1 + 2^-36) (4097 x 16773121 = 2^36 + 1): the double sum is 1 + 2^-24, a float tie (-> 1), fma's 1 + 2^-23.
// Plus acc far above the products (the double sum drops them), each lane of a group
void case_double_round() {
    const int k = 512;
    Case c;
    c.init(k, 16);
    const float dx = std::ldexp(16773121.0f, -40);   // x dw 2^-20 = 16773121 x 2^-60
    int8_t one[32] = {1}, v4097[32];
    for (int j = 0; j < 31; ++j) v4097[j] = 11;   // 31 x 12 x 11 + 1 x 5 = 4097
    v4097[31] = 5;
    int cz[32] = {};
    for (int r = 0; r < c.R; ++r) {
        const int l = r & 7;   // the group's block (its accumulator lane)
        const bool neg = r >= 8;
        uint8_t* row = c.w.data() + c.rb() * r;
        for (int b = 0; b < 16; ++b) set_w(row + b * kBytes, 128, cz);
        int c0[32] = {}, c1[32];
        c0[0] = neg ? 9 : 1;   // group 0: total +-1 (x 1), dw 1
        set_w(row + l * kBytes, 128, c0);
        for (int j = 0; j < 32; ++j) c1[j] = neg ? 15 : 7;   // group 1: total +-4097, dw 2^-20
        c1[31] = neg ? 9 : 1;
        set_w(row + (8 + l) * kBytes, 108, c1);
    }
    for (int i = 0; i < 16; ++i)
        for (int b = 0; b < 16; ++b) {
            if (i < 8) {
                set_act(c.a(i), k, b, b < 8 ? one : v4097, b < 8 ? 1.0f : dx);
            } else {   // acc 2^40 x .., then products 2^-40 x ..: lost in the double sum
                int8_t v[32];
                for (int j = 0; j < 32; ++j) v[j] = (int8_t) (g_rng() & 255);
                set_act(c.a(i), k, b, b < 8 ? one : v, b < 8 ? std::ldexp(1.0f, 40) : std::ldexp(1.3f, -40));
            }
        }
    // the scalar tier's answer for the tie rows: 1 + 2^-23 (fma), not 1
    const void* ap[1] = {c.a(0)};
    float y[16];
    float* yp[1] = {y};
    gg::mx_rows_tier(gg::kMxScalar, c.w.data(), c.rb(), k, 0, 16, ap, 1, yp);
    const bool ok = y[0] == 1.0f + std::ldexp(1.0f, -23) && y[8] == -(1.0f + std::ldexp(1.0f, -23));
    std::printf("double-rounding row: scalar %a / %a (want 0x1.000002p+0 / -0x1.000002p+0) %s\n", y[0], y[8], ok ? "ok" : "FAIL");
    if (!ok) ++g_diff;
    check("fma vs double-rounded mul + add, acc >> product", c);
}

void bench() {
    const int k = 5120, R = 64;
    Case c;
    c.init(k, R);
    for (int r = 0; r < R; ++r)
        for (int b = 0; b < k / 32; ++b) {
            int cd[32];
            rand_codes(cd);
            set_w(c.w.data() + c.rb() * r + (size_t) b * kBytes, 114 + (int) (g_rng() % 9), cd);
        }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(k);
    for (int i = 0; i < 16; ++i) {
        for (auto& v : x) v = nd(g_rng);
        gg::mx_quant(x.data(), k, c.a(i));
    }
    const void* ap[16];
    for (int i = 0; i < 16; ++i) ap[i] = c.a(i);
    std::vector<float> y(16 * R);
    float* yp[16];
    for (int i = 0; i < 16; ++i) yp[i] = y.data() + i * R;
    std::printf("bench: %d rows x %d (MXFP4, in L2), one thread, ns a row and token (best of 5)\n", R, k);
    for (int t = 0; t < gg::kMxTiers; ++t) {
        if (!gg::mx_tier_ok(t)) continue;
        std::printf("  %-10s", gg::mx_tier_name(t));
        for (int n : {1, 2, 4, 8}) {
            double best = 1e30;
            const int reps = t == gg::kMxScalar ? 2 : 20;
            for (int round = 0; round < 5; ++round) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int rep = 0; rep < reps; ++rep) gg::mx_rows_tier(t, c.w.data(), c.rb(), k, 0, R, ap, n, yp);
                best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / reps);
            }
            std::printf("  %d tok %8.1f", n, best / R / n);
        }
        std::printf("\n");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        bench();
        return 0;
    }
    std::printf("this CPU's MXFP4 tier: %s (%s)\n", gg::mx_tier_name(gg::mx_tier()), gg::mx_tier_report());
    for (int k : {256, 2304, 5120}) {
        case_random(k, false);
        case_random(k, true);
        case_extreme(k);
        case_int16(k);
    }
    case_double_round();
    std::printf("total: %ld values compared, %ld differ (%ld NaN = NaN)\n", g_cmp, g_diff, g_nan);
    return g_diff == 0 ? 0 : 1;
}
