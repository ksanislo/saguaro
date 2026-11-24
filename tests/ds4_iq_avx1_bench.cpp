// tests/ds4_iq_avx1_bench.cpp - the AVX1 expert rows (iq_avx1_kern.inc) against ggml-cpu's dot (built with
// -DAVX1_BASE_KERN=<a copy of the .inc in namespace base_avx1>: also against the rows before a change), at
// DeepSeek-V4.1-Flash's expert shapes (gate / up 5120 wide x 2304 rows, down 2304 x 5120), 1 / 2 / 4 / 8 tokens a call,
// one thread and N pinned threads (rows split as the lane does), the experts rotated through more than the L3 so the
// weights come from RAM.  Variants interleaved in rotating order each round; median / min / max of the rounds.  Every
// variant is checked bit for bit against ggml's dot first.  No GPU.
//
//   ds4_iq_avx1_bench [threads=1,20] [rounds=9] [cpus=0-19]
//   env: BENCH_TYPES=16,22 (ggml type ids), BENCH_NT=1,2,4, BENCH_AA=1 (ours twice: the A/A spread), BENCH_NO_GGML=1
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "maya/kernels/cpu/iq_avx1_kern.inc"

#ifdef AVX1_BASE_KERN   // a copy of the rows before a change, in namespace base_avx1: the "base" variant
#include AVX1_BASE_KERN
#endif

namespace {

using RowsFn = std::function<void(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int)>;

void ggml_rows(int type, const uint8_t* w, size_t rb, int n, const void* const* act, int nt, float* const* out, int r0,
               int r1) {
    const auto vd = ggml_get_type_traits_cpu((ggml_type) type)->vec_dot;
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t) vd(n, &out[t][r - r0], 0, w + (size_t) r * rb, 0, act[t], 0, 1);
}

struct Variant {
    std::string name;
    RowsFn fn;
    std::function<void()> on;   // (a global switch, e.g. the PR's one-table IQ2_XXS form)
};

void pin(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

struct Pool {   // N pinned threads, one job a time (spin barrier), rows split in N contiguous ranges
    int n;
    std::vector<std::thread> th;
    std::atomic<int> gen{0}, done{0};
    std::atomic<bool> quit{false};
    std::function<void(int)> job;
    explicit Pool(const std::vector<int>& cpus) : n((int) cpus.size()) {
        for (int i = 1; i < n; ++i)
            th.emplace_back([this, i, c = cpus[(size_t) i]] {
                pin(c);
                int seen = 0;
                for (;;) {
                    while (gen.load(std::memory_order_acquire) == seen)
                        if (quit.load(std::memory_order_relaxed)) return;
                        else _mm_pause();
                    seen = gen.load(std::memory_order_acquire);
                    job(i);
                    done.fetch_add(1, std::memory_order_acq_rel);
                }
            });
        pin(cpus[0]);
    }
    void run(const std::function<void(int)>& j) {
        job = j;
        done.store(0, std::memory_order_relaxed);
        gen.fetch_add(1, std::memory_order_acq_rel);
        job(0);
        while (done.load(std::memory_order_acquire) != n - 1) _mm_pause();
    }
    ~Pool() {
        quit = true;
        for (auto& t : th) t.join();
    }
};

std::vector<uint8_t> rand_rows(int type, int n, int rows, std::mt19937& rng) {
    const size_t rb = ggml_row_size((ggml_type) type, n), bb = ggml_type_size((ggml_type) type);
    std::vector<uint8_t> w(rb * (size_t) rows);
    for (auto& b : w) b = (uint8_t) rng();
    std::uniform_real_distribution<float> sc(0.0005f, 0.02f);
    for (size_t o = 0; o < w.size(); o += bb) {   // (Q2_K: d, dmin after the codes; Q5_K: d, dmin first)
        const uint16_t d = ggml_fp32_to_fp16(sc(rng)), m = ggml_fp32_to_fp16(sc(rng));
        const size_t at = type == GGML_TYPE_Q2_K ? 80 : 0;
        std::memcpy(&w[o + at], &d, 2);
        if (type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q5_K) std::memcpy(&w[o + at + 2], &m, 2);
    }
    return w;
}

std::vector<uint8_t> rand_acts(int n, int nt, std::mt19937& rng) {
    const size_t ab = ggml_row_size(GGML_TYPE_Q8_K, n);
    std::vector<uint8_t> a(ab * (size_t) nt);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) n);
    for (int t = 0; t < nt; ++t) {
        for (auto& v : x) v = nd(rng) * (rng() % 64 == 0 ? 20.0f : 1.0f);
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), a.data() + ab * (size_t) t, n);
    }
    return a;
}

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<int> parse_list(const char* s) {   // "1,20" or "0-19"
    std::vector<int> v;
    std::string str(s);
    size_t p = 0;
    while (p < str.size()) {
        size_t q = str.find(',', p);
        if (q == std::string::npos) q = str.size();
        const std::string tok = str.substr(p, q - p);
        const size_t d = tok.find('-');
        if (d != std::string::npos)
            for (int i = std::atoi(tok.c_str()); i <= std::atoi(tok.c_str() + d + 1); ++i) v.push_back(i);
        else v.push_back(std::atoi(tok.c_str()));
        p = q + 1;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    ggml_cpu_init();
    const std::vector<int> threads = parse_list(argc > 1 ? argv[1] : "1,20");
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 9;
    const std::vector<int> cpus = parse_list(argc > 3 ? argv[3] : "0-19");
    const char* only = std::getenv("BENCH_TYPES");   // e.g. "16,22"
    const char* nts_env = std::getenv("BENCH_NT");
    const std::vector<int> nts = parse_list(nts_env != nullptr ? nts_env : "1,2,4,8");

    struct Shape { const char* name; int n, rows; };
    const Shape shapes[] = {{"gate", 5120, 2304}, {"down", 2304, 5120}};
    const int types[] = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
                         GGML_TYPE_IQ1_S, GGML_TYPE_Q2_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_XS};
    std::printf("type    shape  thr nt  variant      us/call median [min max]   vs ours\n");
    for (int type : types) {
        if (only != nullptr && std::string(",") .append(only).append(",").find("," + std::to_string(type) + ",") == std::string::npos)
            continue;
        std::vector<Variant> vars;
        vars.push_back({"ours", [](int ty, const uint8_t* w, size_t rb, int n, const void* const* a, int nt, float* const* o,
                                   int r0, int r1) { avx1::rows(ty, w, rb, n, a, nt, o, r0, r1); }, [] {}});
#ifdef AVX1_BASE_KERN
        vars.push_back({"base", [](int ty, const uint8_t* w, size_t rb, int n, const void* const* a, int nt, float* const* o,
                                   int r0, int r1) { base_avx1::rows(ty, w, rb, n, a, nt, o, r0, r1); }, [] {}});
#endif
        if (std::getenv("BENCH_AA") != nullptr)   // (an A/A control: ours twice - the run-to-run spread)
            vars.push_back({"ours-again", [](int ty, const uint8_t* w, size_t rb, int n, const void* const* a, int nt,
                                             float* const* o, int r0, int r1) { avx1::rows(ty, w, rb, n, a, nt, o, r0, r1); }, [] {}});
        if (std::getenv("BENCH_NO_GGML") == nullptr) vars.push_back({"ggml", ggml_rows, [] {}});
        for (const Shape& sh : shapes) {
            const size_t rb = ggml_row_size((ggml_type) type, sh.n);
            const size_t mat = rb * (size_t) sh.rows;
            const int E = std::max(4, (int) (160e6 / (double) mat));   // (~160 MB of experts: well past both L3s)
            std::mt19937 rng(11u + (unsigned) type);
            std::vector<std::vector<uint8_t>> W;
            for (int e = 0; e < E; ++e) W.push_back(rand_rows(type, sh.n, sh.rows, rng));
            for (int thr : threads) {
                std::vector<int> cp(cpus.begin(), cpus.begin() + std::min<size_t>((size_t) thr, cpus.size()));
                Pool pool(cp);
                for (int nt : nts) {
                    const auto a = rand_acts(sh.n, nt, rng);
                    const size_t ab = ggml_row_size(GGML_TYPE_Q8_K, sh.n);
                    std::vector<const void*> ap((size_t) nt);
                    for (int t = 0; t < nt; ++t) ap[(size_t) t] = a.data() + ab * (size_t) t;
                    std::vector<float> outv((size_t) nt * sh.rows), ref((size_t) nt * sh.rows);
                    auto call = [&](const Variant& v, const uint8_t* w, float* ob) {
                        pool.run([&](int i) {
                            const int per = (sh.rows + pool.n - 1) / pool.n;
                            const int r0 = std::min(sh.rows, i * per), r1 = std::min(sh.rows, r0 + per);
                            if (r0 >= r1) return;
                            float* o[64];
                            for (int t = 0; t < nt; ++t) o[t] = ob + (size_t) t * sh.rows + r0;
                            v.fn(type, w, rb, sh.n, ap.data(), nt, o, r0, r1);
                        });
                    };
                    // bit check (expert 0) against ggml
                    call(Variant{"ggml", ggml_rows, [] {}}, W[0].data(), ref.data());
                    for (auto& v : vars) {
                        v.on();
                        std::fill(outv.begin(), outv.end(), -7.f);
                        call(v, W[0].data(), outv.data());
                        if (std::memcmp(outv.data(), ref.data(), outv.size() * 4) != 0)
                            std::printf("MISMATCH %s %s %s nt=%d\n", ggml_type_name((ggml_type) type), sh.name, v.name.c_str(), nt);
                    }
                    std::vector<std::vector<double>> tm(vars.size());
                    for (int r = 0; r < rounds; ++r)
                        for (size_t k = 0; k < vars.size(); ++k) {
                            const auto& v = vars[(k + (size_t) r) % vars.size()];
                            v.on();
                            const double t0 = now_us();
                            for (int e = 0; e < E; ++e) call(v, W[(size_t) e].data(), outv.data());
                            tm[(k + (size_t) r) % vars.size()].push_back((now_us() - t0) / E);
                        }
                    double ours_med = 0;
                    for (size_t k = 0; k < vars.size(); ++k) {
                        auto s = tm[k];
                        std::sort(s.begin(), s.end());
                        const double med = s[s.size() / 2];
                        if (k == 0) ours_med = med;
                        std::printf("%-7s %-5s %4d %2d  %-12s %9.1f [%7.1f %7.1f]  %+6.1f%%\n", ggml_type_name((ggml_type) type),
                                    sh.name, thr, nt, vars[k].name.c_str(), med, s.front(), s.back(),
                                    100.0 * (med / ours_med - 1.0));
                    }
                    std::fflush(stdout);
                }
            }
        }
    }
    return 0;
}
