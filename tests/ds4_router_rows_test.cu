// tests/ds4_router_rows_test.cu - the prompt's router rows (gemv_f32's rows kernel) with more tokens a block and more
// rows a warp (DS4_ROUTER_T / DS4_ROUTER_RW) against the 8-token kernel: every logit bit for bit (F32 and BF16
// weights), the one-token kernel's bits on a few tokens too, and each variant's time at 4096 and 8192 tokens
// (DeepSeek-V4.1's router: 384 rows of 5120; the arms interleaved, their order rotated).
//   ninja -C build ds4_router_rows_test   (a CMake target, EXCLUDE_FROM_ALL: the project's CPU flags - no -mavx2/-mfma,
//   which an AVX1 host (Ivy Bridge) cannot run)
//   CUDA_VISIBLE_DEVICES=0 build/ds4_router_rows_test [n] [k] [reps]
#include "ds4_kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::ds4;

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 384, k = argc > 2 ? std::atoi(argv[2]) : 5120;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 20, ldy = 1024;
    const int Ts[] = {4096, 8192, 4093};
    struct V { int tt, rw; };
    const V vs[] = {{8, 1}, {16, 1}, {32, 1}, {64, 1}, {16, 2}, {32, 2}};
    constexpr int NV = sizeof(vs) / sizeof(vs[0]);
    const int Tmax = 8192;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    std::vector<float> x((size_t) Tmax * k), wf((size_t) n * k);
    std::vector<uint16_t> wh((size_t) n * k);
    for (float& v : x) v = nd(rng);
    for (size_t i = 0; i < wf.size(); ++i) {   // (BF16 values, so the F32 and BF16 copies hold the same numbers)
        uint32_t u;
        const float f = nd(rng) * 0.02f;
        std::memcpy(&u, &f, 4);
        wh[i] = (uint16_t) (u >> 16);
        u &= 0xffff0000u;
        std::memcpy(&wf[i], &u, 4);
    }
    float *dx, *dwf, *y0, *y1;
    uint16_t* dwh;
    cudaMalloc(&dx, x.size() * 4);
    cudaMalloc(&dwf, wf.size() * 4);
    cudaMalloc(&dwh, wh.size() * 2);
    cudaMalloc(&y0, (size_t) Tmax * ldy * 4);
    cudaMalloc(&y1, (size_t) Tmax * ldy * 4);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwf, wf.data(), wf.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dwh, wh.data(), wh.size() * 2, cudaMemcpyHostToDevice);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    int fails = 0;
    for (int wt = 0; wt < 2; ++wt) {
        GateW W;
        if (wt) W.h = dwh;
        else W.f = dwf;
        for (int T : Ts) {
            std::vector<float> ref((size_t) T * ldy), got(ref.size());
            cudaMemset(y0, 0, (size_t) T * ldy * 4);
            gemv_f32_rows_with(W, dx, n, k, T, y0, ldy, nullptr, 8, 1);
            cudaMemcpy(ref.data(), y0, ref.size() * 4, cudaMemcpyDeviceToHost);
            // (the one-token kernel on tokens 0, 1, T - 1: the rows kernel's bits)
            bool one_ok = true;
            for (int t : {0, 1, T - 1}) {
                gemv_f32(W, dx + (size_t) t * k, n, k, 1, y1, ldy, nullptr);
                std::vector<float> o(n);
                cudaMemcpy(o.data(), y1, (size_t) n * 4, cudaMemcpyDeviceToHost);
                one_ok = one_ok && std::memcmp(o.data(), ref.data() + (size_t) t * ldy, (size_t) n * 4) == 0;
            }
            fails += !one_ok;
            double ms[NV] = {};
            bool same[NV];
            for (int v = 0; v < NV; ++v) {
                cudaMemset(y1, 0xff, (size_t) T * ldy * 4);
                bool cov = false;
                gemv_f32_rows_with(W, dx, n, k, T, y1, ldy, nullptr, vs[v].tt, vs[v].rw, &cov);
                cudaMemcpy(got.data(), y1, got.size() * 4, cudaMemcpyDeviceToHost);
                same[v] = cov;
                for (int t = 0; t < T && same[v]; ++t)
                    same[v] = std::memcmp(got.data() + (size_t) t * ldy, ref.data() + (size_t) t * ldy, (size_t) n * 4) == 0;
                fails += !same[v];
            }
            for (int r = 0; r < reps; ++r)   // (interleaved, the order rotated each round)
                for (int j = 0; j < NV; ++j) {
                    const int v = (j + r) % NV;
                    float m = 0.0f;
                    cudaEventRecord(a);
                    gemv_f32_rows_with(W, dx, n, k, T, y1, ldy, nullptr, vs[v].tt, vs[v].rw);
                    cudaEventRecord(b);
                    cudaEventSynchronize(b);
                    cudaEventElapsedTime(&m, a, b);
                    ms[v] += m;
                }
            std::printf("%s router %d x %d, %d tokens: one-token kernel %s\n", wt ? "BF16" : "F32", n, k, T,
                        one_ok ? "the same bits" : "DIFFERS");
            for (int v = 0; v < NV; ++v)
                std::printf("  T %2d RW %d: %s, %8.1f us (%.2fx)\n", vs[v].tt, vs[v].rw,
                            same[v] ? "bitwise the same" : "DIFFERS", 1000.0 * ms[v] / reps, ms[0] / ms[v]);
        }
    }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(e));
        fails++;
    }
    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
