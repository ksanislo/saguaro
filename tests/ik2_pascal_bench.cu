// tests/ik2_pascal_bench.cu - the Pascal token-path kernels (src/ik2/ik2_pascal.cu) against the generic ones at
// Inkling-2-Mini's sizes: the same answers (max relative difference) and GB/s.  Not a test of the build: run by hand.
#include "ik2_kernels.cuh"
#include "ik2_pascal.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace ik2;

#define CU(x)                                                                                                         \
    do {                                                                                                              \
        cudaError_t e_ = (x);                                                                                         \
        if (e_ != cudaSuccess) {                                                                                      \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));                         \
            std::exit(1);                                                                                             \
        }                                                                                                             \
    } while (0)

static float hf(uint16_t h) {
    __half x;
    std::memcpy(&x, &h, 2);
    return __half2float(x);
}

// the card's read bandwidth: uint4 / uint2 / u32 loads, a grid's worth of loads in flight
template <class V>
__global__ void read_kernel(const V* p, size_t n, unsigned* out) {
    unsigned a = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) blockDim.x * gridDim.x) {
        const V v = __ldg(p + i);
        a ^= reinterpret_cast<const unsigned*>(&v)[0] ^ reinterpret_cast<const unsigned*>(&v)[sizeof(V) / 4 - 1];
    }
    if (a == 0x12345678u) *out = a;
}
// the NVFP4 experts' loads alone (gate_up's addresses: 3 u32 a sub-block, lanes over 288 sub-blocks a row pair)
__global__ void nvfp4_loads_kernel(const uint8_t* const* blob, size_t eb0, int F, unsigned* out) {
    const int s = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 16 + warp * 2;
    unsigned a = 0;
    for (int p = 0; p < 2; ++p) {
        const uint8_t* rows[2] = {blob[s] + (size_t) (r + p) * 1296, blob[s] + eb0 + (size_t) (r + p) * 1296};
        uint32_t v[27];
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            const int u = lane + 32 * i, rg = u / 144, sub = u - rg * 144;
            const uint8_t* b = (rg ? rows[1] : rows[0]) + (sub >> 2) * 36;
            v[3 * i] = __ldg((const uint32_t*) b), v[3 * i + 1] = __ldg((const uint32_t*) (b + 4 + (sub & 3) * 8));
            v[3 * i + 2] = __ldg((const uint32_t*) (b + 8 + (sub & 3) * 8));
        }
#pragma unroll
        for (int i = 0; i < 27; ++i) a ^= v[i];
    }
    if (a == 0x12345678u) *out = a;
}

int main() {
    setenv("SAGUARO_IK2_PASCAL_EXPERTS", "0", 1);   // (k::'s entries: the generic kernels; k::pascal:: called directly)
    setenv("SAGUARO_IK2_PASCAL_Q8", "0", 1);
    const int C = 2304, F = 1152, E = 128, S = 8, NSET = 64, IT = 200;
    const size_t rbg = C / 64 * 36, rbd = F / 64 * 36, eb0 = F * rbg, eb1 = F * rbg, ebd = (size_t) C * rbd;
    const size_t eb = eb0 + eb1 + ebd;
    std::mt19937 rng(7);
    std::vector<uint8_t> hb(eb * E);
    for (auto& b : hb) b = (uint8_t) rng();
    for (size_t o = 0; o < hb.size(); o += 36)   // (each block's four E4M3 scales in a sane range)
        for (int j = 0; j < 4; ++j) hb[o + j] = (uint8_t) (0x30 + rng() % 16);
    uint8_t* dblob;
    CU(cudaMalloc(&dblob, hb.size()));
    CU(cudaMemcpy(dblob, hb.data(), hb.size(), cudaMemcpyHostToDevice));
    std::vector<const uint8_t*> ptrs((size_t) NSET * S);
    for (auto& p : ptrs) p = dblob + (size_t) (rng() % E) * eb;
    const uint8_t** dptrs;
    CU(cudaMalloc(&dptrs, ptrs.size() * sizeof(void*)));
    CU(cudaMemcpy(dptrs, ptrs.data(), ptrs.size() * sizeof(void*), cudaMemcpyHostToDevice));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> hx(C), hsg(S), hsu(S);
    for (auto& v : hx) v = nd(rng);
    for (auto& v : hsg) v = 0.02f;
    for (auto& v : hsu) v = 0.02f;
    std::vector<int> htok(S, 0);
    float *dx, *dsg, *dsu, *dy0, *dy1;
    int* dtok;
    uint16_t *dh0, *dh1;
    CU(cudaMalloc(&dx, C * 4));
    CU(cudaMalloc(&dsg, S * 4));
    CU(cudaMalloc(&dsu, S * 4));
    CU(cudaMalloc(&dtok, S * 4));
    CU(cudaMalloc(&dh0, S * F * 2));
    CU(cudaMalloc(&dh1, S * F * 2));
    CU(cudaMalloc(&dy0, S * C * 4));
    CU(cudaMalloc(&dy1, S * C * 4));
    CU(cudaMemcpy(dx, hx.data(), C * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dsg, hsg.data(), S * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dsu, hsu.data(), S * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dtok, htok.data(), S * 4, cudaMemcpyHostToDevice));

    // Q8 planar matrices: the fused attention input (N 5120, K 2304), the heads' output (2304 x 3072), the head
    struct Q8 { int N, K; int8_t* q; uint16_t* d; float* x; float *y0, *y1; };
    std::vector<Q8> q8s = {{6144, 2304}, {2304, 3072}, {201024, 2304}};
    for (auto& m : q8s) {
        const size_t n = (size_t) m.N * m.K;
        std::vector<int8_t> q(n);
        std::vector<uint16_t> d(n / 32);
        for (auto& v : q) v = (int8_t) rng();
        for (auto& v : d) {
            const __half h = __float2half(0.001f + 0.01f * (rng() % 1000) / 1000.0f);
            std::memcpy(&v, &h, 2);
        }
        std::vector<float> x(m.K);
        for (auto& v : x) v = nd(rng);
        CU(cudaMalloc(&m.q, n));
        CU(cudaMalloc(&m.d, n / 16));
        CU(cudaMalloc(&m.x, m.K * 4));
        CU(cudaMalloc(&m.y0, m.N * 4));
        CU(cudaMalloc(&m.y1, m.N * 4));
        CU(cudaMemcpy(m.q, q.data(), n, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(m.d, d.data(), n / 16, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(m.x, x.data(), m.K * 4, cudaMemcpyHostToDevice));
    }

    cudaEvent_t a, b;
    CU(cudaEventCreate(&a));
    CU(cudaEventCreate(&b));
    auto timed = [&](auto fn) {
        fn(0);
        CU(cudaDeviceSynchronize());
        CU(cudaEventRecord(a));
        for (int i = 0; i < IT; ++i) fn(i % NSET);
        CU(cudaEventRecord(b));
        CU(cudaEventSynchronize(b));
        float ms = 0;
        CU(cudaEventElapsedTime(&ms, a, b));
        return ms / IT;
    };
    auto maxrel_h = [&](const uint16_t* p, const uint16_t* q, size_t n) {
        std::vector<uint16_t> u(n), v(n);
        CU(cudaMemcpy(u.data(), p, n * 2, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(v.data(), q, n * 2, cudaMemcpyDeviceToHost));
        double m = 0, sc = 0;
        for (size_t i = 0; i < n; ++i) sc = std::max(sc, (double) std::fabs(hf(u[i])));
        for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs((double) hf(u[i]) - hf(v[i])) / sc);
        return m;
    };
    auto maxrel_f = [&](const float* p, const float* q, size_t n) {
        std::vector<float> u(n), v(n);
        CU(cudaMemcpy(u.data(), p, n * 4, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(v.data(), q, n * 4, cudaMemcpyDeviceToHost));
        double m = 0, sc = 0;
        for (size_t i = 0; i < n; ++i) sc = std::max(sc, (double) std::fabs(u[i]));
        for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs((double) u[i] - v[i]) / sc);
        return m;
    };

    const double gu_bytes = (double) S * (eb0 + eb1), dn_bytes = (double) S * ebd;
    {
        unsigned* dout;
        CU(cudaMalloc(&dout, 4));
        const size_t nb = hb.size() & ~(size_t) 15;
        int nsm = 0;
        CU(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
        for (int mul : {4, 8, 16}) {
            const float t4 = timed([&](int) { read_kernel<uint4><<<nsm * mul, 256>>>((const uint4*) dblob, nb / 16, dout); });
            const float t2 = timed([&](int) { read_kernel<uint2><<<nsm * mul, 256>>>((const uint2*) dblob, nb / 8, dout); });
            const float t1 = timed([&](int) { read_kernel<unsigned><<<nsm * mul, 256>>>((const unsigned*) dblob, nb / 4, dout); });
            std::printf("read %zu MB, %d blocks/SM: uint4 %.0f GB/s, uint2 %.0f, u32 %.0f\n", nb >> 20, mul * 8 / 8,
                        nb / t4 / 1e6, nb / t2 / 1e6, nb / t1 / 1e6);
        }
        const float tl = timed([&](int i) {
            nvfp4_loads_kernel<<<dim3(F / 16, S), 256>>>(dptrs + i * S, eb0, F, dout);
        });
        std::printf("nvfp4 gate_up loads alone: %.1f us %.0f GB/s\n", tl * 1e3, gu_bytes / tl / 1e6);
    }
    const float t_gu0 = timed([&](int i) {
        k::experts_gate_up_swiglu(dptrs + i * S, dtok, S, k::kNVFP4, k::kNVFP4, eb0, dx, C, F, dsg, dsu, 4.0f, 25.0f, dh0,
                                  0);
    });
    const float t_dn0 = timed([&](int i) { k::experts_down(dptrs + i * S, S, k::kNVFP4, eb0 + eb1, dh0, C, F, dy0, 0); });
    std::printf("generic  gate_up %.1f us %.0f GB/s | down %.1f us %.0f GB/s\n", t_gu0 * 1e3, gu_bytes / t_gu0 / 1e6,
                t_dn0 * 1e3, dn_bytes / t_dn0 / 1e6);
    std::vector<float> t_q0;
    for (auto& m : q8s) {
        const float t = timed([&](int) { k::mv_q8p(m.q, m.d, m.N, m.K, m.x, m.y0, 0); });
        t_q0.push_back(t);
        std::printf("generic  q8 %dx%d %.1f us %.0f GB/s\n", m.N, m.K, t * 1e3, (double) m.N * m.K * 1.0625 / t / 1e6);
    }
    if (getenv("BENCH_GEMM")) {   // the prompt's expert GEMMs: Y [T][M] = X [T][K] . W [M][K]^T (cuBLAS's picks and algos)
        cublasHandle_t hb;
        cublasCreate(&hb);
        const int Ms[] = {2304, 2304}, Ks[] = {2304, 1152};
        float *W32, *X32, *Y;
        uint16_t *W16, *X16;
        CU(cudaMalloc(&W32, (size_t) 2304 * 2304 * 4));
        CU(cudaMalloc(&X32, (size_t) 2048 * 2304 * 4));
        CU(cudaMalloc(&Y, (size_t) 2048 * 2304 * 4));
        CU(cudaMalloc(&W16, (size_t) 2304 * 2304 * 2));
        CU(cudaMalloc(&X16, (size_t) 2048 * 2304 * 2));
        CU(cudaMemset(W32, 0, (size_t) 2304 * 2304 * 4));
        CU(cudaMemset(X32, 0, (size_t) 2048 * 2304 * 4));
        CU(cudaMemset(W16, 0, (size_t) 2304 * 2304 * 2));
        CU(cudaMemset(X16, 0, (size_t) 2048 * 2304 * 2));
        const float one = 1.0f, zero = 0.0f;
        for (int si = 0; si < 2; ++si)
            for (int T : {64, 128, 256, 512, 2048}) {
                const int M = Ms[si], K = Ks[si];
                const double fl = 2.0 * M * K * T;
                auto tf = [&](float ms) { return fl / ms / 1e9; };
                const float t16 = timed([&](int) {
                    cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K, &one, W16, CUDA_R_16F, K, X16, CUDA_R_16F, K, &zero,
                                 Y, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                });
                const float t32 = timed([&](int) {
                    cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K, &one, W32, K, X32, K, &zero, Y, M);
                });
                float best = 1e9;
                int ba = -1;
                for (int a = 0; a <= 23; ++a) {
                    if (cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K, &one, W32, CUDA_R_32F, K, X32, CUDA_R_32F, K,
                                     &zero, Y, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, (cublasGemmAlgo_t) a) != CUBLAS_STATUS_SUCCESS)
                        continue;
                    const float t = timed([&](int) {
                        cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K, &one, W32, CUDA_R_32F, K, X32, CUDA_R_32F, K,
                                     &zero, Y, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, (cublasGemmAlgo_t) a);
                    });
                    if (t < best) best = t, ba = a;
                }
                // (W^T X^T the other way round: Y^T [M][T] = W . X^T - cuBLAS's "nt")
                const float tsw = timed([&](int) {
                    cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, T, M, K, &one, X32, K, W32, K, &zero, Y, T);
                });
                std::printf("gemm M %d K %d T %4d: fp16 %.2f TFLOPS, sgemm %.2f, sgemm swapped %.2f, best F32 algo %d %.2f\n", M, K,
                            T, tf(t16), tf(t32), tf(tsw), ba, tf(best));
            }
    }
    {
        const int c = 0;
        // (the same first set for the answers)
        k::experts_gate_up_swiglu(dptrs, dtok, S, k::kNVFP4, k::kNVFP4, eb0, dx, C, F, dsg, dsu, 4.0f, 25.0f, dh0, 0);
        k::pascal::gate_up_swiglu(dptrs, dtok, S, k::kNVFP4, k::kNVFP4, eb0, dx, C, F, dsg, dsu, 4.0f, 25.0f, dh1,
                                  nullptr, 0, nullptr, 0, 0);
        const double e_gu = maxrel_h(dh0, dh1, (size_t) S * F);
        k::experts_down(dptrs, S, k::kNVFP4, eb0 + eb1, dh0, C, F, dy0, 0);
        k::pascal::down(dptrs, S, k::kNVFP4, eb0 + eb1, dh0, C, F, dy1, nullptr, 0, 0);
        const double e_dn = maxrel_f(dy0, dy1, (size_t) S * C);
        const float t_gu = timed([&](int i) {
            k::pascal::gate_up_swiglu(dptrs + i * S, dtok, S, k::kNVFP4, k::kNVFP4, eb0, dx, C, F, dsg, dsu, 4.0f, 25.0f,
                                      dh1, nullptr, 0, nullptr, 0, 0);
        });
        const float t_dn = timed([&](int i) {
            k::pascal::down(dptrs + i * S, S, k::kNVFP4, eb0 + eb1, dh0, C, F, dy1, nullptr, 0, 0);
        });
        std::printf("pascal %d gate_up %.1f us %.0f GB/s (x%.2f, err %.1e) | down %.1f us %.0f GB/s (x%.2f, err %.1e)\n",
                    c, t_gu * 1e3, gu_bytes / t_gu / 1e6, t_gu0 / t_gu, e_gu, t_dn * 1e3, dn_bytes / t_dn / 1e6,
                    t_dn0 / t_dn, e_dn);
        for (size_t j = 0; j < q8s.size(); ++j) {
            auto& m = q8s[j];
            k::mv_q8p(m.q, m.d, m.N, m.K, m.x, m.y0, 0);
            if (!k::pascal::mv_q8p_conv(m.q, m.d, m.N, m.K, m.x, m.y1, k::MvConv{}, k::MvConv{}, k::MvAlt{}, k::MvAlt{},
                                        0)) {
                std::printf("cfg %03d  q8 %dx%d: not taken\n", c, m.N, m.K);
                continue;
            }
            const double e = maxrel_f(m.y0, m.y1, m.N);
            const float t = timed([&](int) {
                k::pascal::mv_q8p_conv(m.q, m.d, m.N, m.K, m.x, m.y1, k::MvConv{}, k::MvConv{}, k::MvAlt{}, k::MvAlt{}, 0);
            });
            std::printf("pascal %s q8 %dx%d %.1f us %.0f GB/s (x%.2f, err %.1e)\n", getenv("SAGUARO_IK2_PASCAL_Q8_SPLIT") ? "nosplit" : "split  ", m.N, m.K, t * 1e3,
                        (double) m.N * m.K * 1.0625 / t / 1e6, t_q0[j] / t, e);
        }
    }
    return 0;
}
