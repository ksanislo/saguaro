// prefill_mmq_scatter_test - the prompt path's MoE input quantized per token and scattered to its k rows
// (mmq::quantize_scatter) against the per-row gather (mmq::quantize over the expert-order source rows): the same
// q8_1 bytes, the whole buffer, for every q8_1 layout MMQ uses (D4: Q2_0 Q8_0 IQ2_XS, DS4: Q4_1, D2S6: Q2_K), on a
// routing built the way Prefill builds it (10 of 512 experts a token, rows grouped by expert), with an all-zero
// token and token counts that are not multiples of anything.  --bench: both at 2K, 8K and 32K tokens.
// Also STRATA_MMQ_SWIGLU_Q8's fused kernel (mmq::swiglu_quantize) against swiglu / swiglu_scaled + quantize: the same
// q8_1 bytes for the down product's activations (n_ff 640, split / interleaved / scaled gate-up rows), every layout.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

constexpr int kExperts = 512, kUsed = 10;
constexpr int64_t kCols = 2560;

// Prefill's maps: slot[t * k + j] = the token's row in expert order, src[row] = the token
void routing(int64_t T, std::mt19937& rng, std::vector<int32_t>& slot, std::vector<int32_t>& src) {
    std::vector<int32_t> ids((size_t) (T * kUsed));
    std::vector<int32_t> pick(kExperts);
    for (int e = 0; e < kExperts; ++e) pick[(size_t) e] = e;
    for (int64_t t = 0; t < T; ++t) {
        for (int j = 0; j < kUsed; ++j) std::swap(pick[(size_t) j], pick[(size_t) (j + rng() % (kExperts - j))]);
        for (int j = 0; j < kUsed; ++j) ids[(size_t) (t * kUsed + j)] = pick[(size_t) j];
    }
    std::vector<int32_t> cnt(kExperts, 0), fill(kExperts, 0);
    for (int32_t e : ids) ++cnt[(size_t) e];
    for (int e = 1; e < kExperts; ++e) fill[(size_t) e] = fill[(size_t) e - 1] + cnt[(size_t) e - 1];
    slot.assign(ids.size(), 0);
    src.assign(ids.size(), 0);
    for (size_t i = 0; i < ids.size(); ++i) {
        const int32_t p = fill[(size_t) ids[i]]++;
        slot[i] = p;
        src[(size_t) p] = (int32_t) (i / kUsed);
    }
}

int check(ggml_type t, int64_t T, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<int32_t> slot, src;
    routing(T, rng, slot, src);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) (T * kCols));
    for (float& v : x) v = nd(rng) * 0.7f;
    if (T > 3) std::fill(x.begin() + 2 * kCols, x.begin() + 3 * kCols, 0.0f);   // an all-zero token
    const int64_t rows = T * kUsed;
    const size_t qb = mmq::q8_bytes(rows, kCols);
    Dev dx(x.size() * 4), dslot(slot.size() * 4), dsrc(src.size() * 4), qa(qb), qs(qb);
    ck(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(dslot.p, slot.data(), slot.size() * 4, cudaMemcpyHostToDevice), "slot");
    ck(cudaMemcpy(dsrc.p, src.data(), src.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaMemset(qa.p, 0xA5, qb), "memset");
    ck(cudaMemset(qs.p, 0xA5, qb), "memset");
    mmq::quantize((const float*) dx.p, (const int32_t*) dsrc.p, qa.p, (int) t, kCols, kCols, rows, nullptr);
    mmq::quantize_scatter((const float*) dx.p, (const int32_t*) dslot.p, (const int32_t*) dsrc.p, qs.p, (int) t, kCols,
                          kCols, T, kUsed, nullptr);
    ck(cudaDeviceSynchronize(), "quantize");
    std::vector<uint8_t> a(qb), b(qb);
    ck(cudaMemcpy(a.data(), qa.p, qb, cudaMemcpyDeviceToHost), "back");
    ck(cudaMemcpy(b.data(), qs.p, qb, cudaMemcpyDeviceToHost), "back");
    size_t diff = 0;
    for (size_t i = 0; i < qb; ++i) diff += a[i] != b[i];
    std::printf("%s %-7s T=%-5lld %zu of %zu bytes differ\n", diff ? "FAIL" : "ok  ", ggml_type_name(t), (long long) T,
                diff, qb);
    return diff ? 1 : 0;
}

// mode 0 split, 1 interleaved, 2 split with per-expert gate/up scales (bounds of 3 experts)
int check_swiglu(ggml_type t, int64_t rows, int mode, uint32_t seed) {
    constexpr int64_t n_ff = 640;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> gu((size_t) (rows * 2 * n_ff));
    for (float& v : gu) v = nd(rng) * 3.0f;
    if (rows > 3) std::fill(gu.begin() + 2 * 2 * n_ff, gu.begin() + 3 * 2 * n_ff, 0.0f);   // an all-zero row
    if (rows > 5) for (int64_t k = 0; k < n_ff; ++k) gu[(size_t) (4 * 2 * n_ff + k)] = -90.0f;   // silu underflows
    const int32_t b[4] = {0, (int32_t) (rows / 3), (int32_t) (rows / 2), (int32_t) rows};
    const float tails[12] = {0.5f, 1.75f, 1.0f, 0.0f, 2.0f, 0.03125f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f};
    const size_t qb = mmq::q8_bytes(rows, n_ff);
    Dev dgu(gu.size() * 4), dh((size_t) (rows * n_ff * 4)), qa(qb), qf(qb), db(sizeof b), dt(sizeof tails);
    ck(cudaMemcpy(dgu.p, gu.data(), gu.size() * 4, cudaMemcpyHostToDevice), "gu");
    ck(cudaMemcpy(db.p, b, sizeof b, cudaMemcpyHostToDevice), "bounds");
    ck(cudaMemcpy(dt.p, tails, sizeof tails, cudaMemcpyHostToDevice), "tails");
    ck(cudaMemset(qa.p, 0xA5, qb), "memset");
    ck(cudaMemset(qf.p, 0xA5, qb), "memset");   // same fill: q8_bytes' 128-block slack is written by neither
    if (mode == 2) mmq::swiglu_scaled((const float*) dgu.p, (float*) dh.p, rows, n_ff, (const int32_t*) db.p, 3,
                                      (const float*) dt.p, 0, nullptr);
    else mmq::swiglu((const float*) dgu.p, (float*) dh.p, rows, n_ff, mode == 1, nullptr);
    mmq::quantize((const float*) dh.p, nullptr, qa.p, (int) t, n_ff, n_ff, rows, nullptr);
    const bool ok = mmq::swiglu_quantize((const float*) dgu.p, qf.p, (int) t, rows, n_ff, mode == 1,
                                         mode == 2 ? (const int32_t*) db.p : nullptr, 3,
                                         mode == 2 ? (const float*) dt.p : nullptr, 0, nullptr);
    ck(cudaDeviceSynchronize(), "swiglu");
    if (!ok) {
        std::printf("FAIL swiglu_quantize refused %s mode %d\n", ggml_type_name(t), mode);
        return 1;
    }
    std::vector<uint8_t> a(qb), f(qb);
    ck(cudaMemcpy(a.data(), qa.p, qb, cudaMemcpyDeviceToHost), "back");
    ck(cudaMemcpy(f.data(), qf.p, qb, cudaMemcpyDeviceToHost), "back");
    size_t diff = 0, first = qb;
    for (size_t i = 0; i < qb; ++i)
        if (a[i] != f[i]) { ++diff; first = std::min(first, i); }
    if (diff) std::printf("  first differing byte %zu: %02x vs %02x\n", first, a[first], f[first]);
    std::printf("%s swiglu+q8_1 %-7s mode %d rows=%-5lld %zu of %zu bytes differ\n", diff ? "FAIL" : "ok  ",
                ggml_type_name(t), mode, (long long) rows, diff, qb);
    return diff ? 1 : 0;
}

void bench(ggml_type t, int64_t T) {
    std::mt19937 rng(7);
    std::vector<int32_t> slot, src;
    routing(T, rng, slot, src);
    std::vector<float> x((size_t) (T * kCols), 0.25f);
    const int64_t rows = T * kUsed;
    const size_t qb = mmq::q8_bytes(rows, kCols);
    Dev dx(x.size() * 4), dslot(slot.size() * 4), dsrc(src.size() * 4), q(qb);
    ck(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(dslot.p, slot.data(), slot.size() * 4, cudaMemcpyHostToDevice), "slot");
    ck(cudaMemcpy(dsrc.p, src.data(), src.size() * 4, cudaMemcpyHostToDevice), "src");
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    std::vector<float> tg, ts;
    for (int i = 0; i < 23; ++i)
        for (int v = 0; v < 2; ++v) {
            cudaEventRecord(e0);
            if (v == 0) mmq::quantize((const float*) dx.p, (const int32_t*) dsrc.p, q.p, (int) t, kCols, kCols, rows, nullptr);
            else mmq::quantize_scatter((const float*) dx.p, (const int32_t*) dslot.p, (const int32_t*) dsrc.p, q.p, (int) t,
                                       kCols, kCols, T, kUsed, nullptr);
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "bench");
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (i >= 3) (v == 0 ? tg : ts).push_back(ms);
        }
    std::sort(tg.begin(), tg.end());
    std::sort(ts.begin(), ts.end());
    const float g = tg[tg.size() / 2], s = ts[ts.size() / 2];
    std::printf("  %-7s T=%lld x %d rows: gather %.3f ms, scatter %.3f ms (%.2fx)\n", ggml_type_name(t), (long long) T,
                kUsed, g, s, g / s);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}
}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    int fails = 0;
    try {
        const ggml_type types[] = {GGML_TYPE_Q2_0, GGML_TYPE_Q8_0, GGML_TYPE_IQ2_XS, GGML_TYPE_Q4_1, GGML_TYPE_Q2_K};
        uint32_t seed = 11;
        for (ggml_type t : types)
            for (int64_t T : {1, 2, 7, 600, 2099}) fails += check(t, T, seed++);
        // the down product's types (Q5_1: DS4, Q8_0 / Q5_0 / IQ4_NL: D4) and D2S6 (Q2_K)
        for (ggml_type t : {GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_1, GGML_TYPE_Q2_K})
            for (int mode = 0; mode < 3; ++mode)
                for (int64_t rows : {1, 7, 600, 20990}) fails += check_swiglu(t, rows, mode, seed++);
        if (do_bench) {
            std::printf("--bench (median of 20):\n");
            for (ggml_type t : {GGML_TYPE_IQ2_XS, GGML_TYPE_Q2_0}) {
                bench(t, 2048);
                bench(t, 8192);
                bench(t, 32768);
            }
        }
    } catch (const std::exception& e) {
        std::printf("FAIL %s\n", e.what());
        return 1;
    }
    std::printf("prefill_mmq_scatter_test: %d failures\n", fails);
    return fails ? 1 : 0;
}
