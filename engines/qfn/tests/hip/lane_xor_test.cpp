// The register warp sum (include/strata/kernels/lane_xor.hpp) against the shuffle loop it replaces: the same bits in every
// lane, over random warps of mixed magnitudes and signs (a sum's order of operations is what keeps its bits).
#include <hip/hip_runtime.h>
#include "intrinsics.hpp"   // (Saguaro's shared include/maya/hip_compat, on the HIP include path)
#include "strata/kernels/lane_xor.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

__global__ void sums(const float* in, float* dpp, float* shfl) {
    const int i = blockIdx.x * 32 + threadIdx.x;
    float a = in[i], b = in[i];
    a = strata_warp_sum_xor(a);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) b += __shfl_xor_sync(0xffffffffu, b, o);
    dpp[i] = a;
    shfl[i] = b;
}

int main() {
    constexpr int warps = 1 << 14, n = warps * 32;
    std::vector<float> h(n), a(n), b(n);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f), e(-20.0f, 20.0f);
    for (float& x : h) x = u(rng) * std::exp2(e(rng));
    float *din = nullptr, *da = nullptr, *db = nullptr;
    if (hipMalloc(&din, n * 4) || hipMalloc(&da, n * 4) || hipMalloc(&db, n * 4)) return 2;
    if (hipMemcpy(din, h.data(), n * 4, hipMemcpyHostToDevice)) return 2;
    hipLaunchKernelGGL(sums, dim3(warps), dim3(32), 0, 0, din, da, db);
    if (hipDeviceSynchronize() || hipMemcpy(a.data(), da, n * 4, hipMemcpyDeviceToHost) ||
        hipMemcpy(b.data(), db, n * 4, hipMemcpyDeviceToHost)) return 2;
    if (std::memcmp(a.data(), b.data(), n * 4) != 0) {
        for (int i = 0; i < n; ++i)
            if (std::memcmp(&a[i], &b[i], 4) != 0) { std::printf("lane_xor_test: differs at %d: %a vs %a\n", i, a[i], b[i]); break; }
        return 1;
    }
    std::printf("lane_xor_test: %d warp sums, register exchange == shuffle bit for bit\n", warps);
    return 0;
}
