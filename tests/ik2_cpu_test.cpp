// tests/ik2_cpu_test.cpp - saguaro-ik2's CPU lane (src/ik2/ik2_cpu.cpp), no GPU: the NVFP4 rows of this CPU's
// kernel (AVX-512, AVX or scalar) and the scalar ones against the format's definition in double, and a whole expert
// through the pool against the same.
#include "../src/ik2/ik2_cpu.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

double code(int c) {
    static const double v[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    return (c & 8) ? -v[c & 7] : v[c & 7];
}
double ue4m3(int x) {
    if (x == 0 || x == 0x7F) return 0.0;
    const int e = (x >> 3) & 0xF, m = x & 7;
    return e == 0 ? std::ldexp((double) m, -9) : std::ldexp(1.0 + m / 8.0, e - 7);
}
// rows of an NVFP4 matrix in double: a 36-byte block of 64 values, four scales then 8 bytes a 16-value sub-block
void rows_ref(const uint8_t* W, int K, int r0, int r1, const std::vector<double>& x, std::vector<double>& out) {
    const int nb = K / 64;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* b = W + (size_t) r * nb * 36;
        double acc = 0;
        for (int i = 0; i < nb; ++i, b += 36)
            for (int sb = 0; sb < 4; ++sb)
                for (int j = 0; j < 8; ++j) {
                    const uint8_t by = b[4 + sb * 8 + j];
                    acc += ue4m3(b[sb]) * (code(by & 0xF) * x[i * 64 + sb * 16 + j] + code(by >> 4) * x[i * 64 + sb * 16 + j + 8]);
                }
        out[r - r0] = acc;
    }
}
int g_fail = 0;
void expect(const char* what, const std::vector<float>& got, const std::vector<double>& want, double rtol, double atol) {
    double worst = 0;
    size_t bad = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double e = std::fabs(got[i] - want[i]), lim = atol + rtol * std::fabs(want[i]);
        worst = std::max(worst, e / lim);
        if (!(e <= lim)) ++bad;
    }
    std::printf("%s %-40s %zu values, worst %.3f of the tolerance\n", bad ? "FAIL" : "ok  ", what, want.size(), worst);
    if (bad) ++g_fail;
}

}  // namespace

int main() {
    std::mt19937 rng(20261121);
    std::printf("ik2_cpu_test: this CPU's rows are %s\n", ik2::cpu::isa());
    const int C = 2304, F = 1152;
    auto blocks = [&](int rows, int K) {
        std::vector<uint8_t> w((size_t) rows * K / 64 * 36);
        for (auto& b : w) b = (uint8_t) rng();
        for (size_t i = 0; i < w.size(); i += 36)   // (scales as a file has them: finite, mid-range)
            for (int s = 0; s < 4; ++s) w[i + s] = (uint8_t) (0x20 + rng() % 0x30);
        return w;
    };
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int K : {64, 1152, 2304}) {
        const int rows = 37;
        const auto w = blocks(rows, K);
        std::vector<float> x(K);
        for (auto& v : x) v = nd(rng);
        std::vector<double> xd(x.begin(), x.end()), want(rows);
        rows_ref(w.data(), K, 0, rows, xd, want);
        std::vector<float> got(rows), plain(rows);
        ik2::cpu::nvfp4_rows(w.data(), K, 0, rows, x.data(), got.data());
        ik2::cpu::nvfp4_rows_plain(w.data(), K, 0, rows, x.data(), plain.data());
        char n[64];
        std::snprintf(n, sizeof n, "nvfp4 rows, K %d (%s)", K, ik2::cpu::isa());
        expect(n, got, want, 2e-5, 1e-4);
        std::snprintf(n, sizeof n, "nvfp4 rows, K %d (scalar)", K);
        expect(n, plain, want, 2e-5, 1e-4);
        std::vector<float> part(rows - 5);   // (a chunk of the rows, as the pool's threads take them)
        ik2::cpu::nvfp4_rows(w.data(), K, 5, rows, x.data(), part.data());
        bool same = true;
        for (int r = 5; r < rows; ++r) same = same && part[r - 5] == got[r];
        std::printf("%s nvfp4 rows, K %d: a chunk is the whole's rows\n", same ? "ok  " : "FAIL", K);
        if (!same) ++g_fail;
    }
    {   // one expert through the pool: y = down . (gcap tanh(g / gcap) sigmoid(g) ucap tanh(u / ucap))
        const float gcap = 4.0f, ucap = 25.0f, sg = 0.011f, su = 0.007f;
        const auto gate = blocks(F, C), up = blocks(F, C), down = blocks(C, F);
        std::vector<float> x(C), y(C);
        for (auto& v : x) v = nd(rng);
        std::vector<double> xd(x.begin(), x.end()), g(F), u(F), h(F), want(C);
        rows_ref(gate.data(), C, 0, F, xd, g);
        rows_ref(up.data(), C, 0, F, xd, u);
        for (int i = 0; i < F; ++i) {
            const double gv = g[i] * sg, uv = u[i] * su;
            h[i] = gcap * std::tanh(gv / gcap) / (1.0 + std::exp(-gv)) * ucap * std::tanh(uv / ucap);
        }
        rows_ref(down.data(), F, 0, C, h, want);
        ik2::cpu::Pool pool(7);
        ik2::cpu::ExpertJob job;
        job.gate = gate.data(), job.up = up.data(), job.down = down.data(), job.x = x.data(), job.sg = sg, job.su = su;
        job.y = y.data();
        ik2::cpu::post_expert(pool, &job, C, F, gcap, ucap);
        pool.wait();
        double sc = 0;
        for (double v : want) sc = std::max(sc, std::fabs(v));
        expect("an expert through the pool (7 threads)", y, want, 1e-3, 1e-4 * sc);
    }
    std::printf(g_fail ? "FAILED: %d check(s)\n" : "all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
