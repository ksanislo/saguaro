// src/prefill/mmq_kq_tab_lib.cu - strata_kq_tab: the pointer-table MMQ (moe_mmq_tab.cuh) for the K-quant experts
// (and any type CMake's STRATA_MMQ_TAB_KQ_TYPES adds), so SAGUARO_MMQ_DIRECT=1 reads resident UD-Q4/Q5/Q6_K_XL experts
// in their cache slots.  A library of its own, like strata_kq: the engine loads every kernel of a module at start
// (CUDA_MODULE_LOADING=EAGER), and these load only when the switch is on.  Built from the same mmq.cuh, flags and
// architectures as strata_kq, whose mul_mat_q_case runs the gathered products: same tiles, same launch, same bits.
#include "moe_mmq_tab.hpp"

#include "strata/prefill/mmq_kq_abi.hpp"

#include <mutex>

#if defined(_WIN32)
#define STRATA_KQ_EXPORT extern "C" __declspec(dllexport)
#else
#define STRATA_KQ_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
namespace sm = strata::prefill::mmq;
std::mutex g_mu;
ggml_backend_cuda_context* g_ctx[GGML_CUDA_MAX_DEVICES] = {};

ggml_backend_cuda_context* ctx_for(int dev) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_ctx[dev] == nullptr) g_ctx[dev] = new ggml_backend_cuda_context(dev);
    return g_ctx[dev];
}

// -1: no instance; else whether it launched (false: rows % 128 != 0); ctx == nullptr only asks
int dispatch(ggml_backend_cuda_context* ctx, ggml_type t, const mmq_args* a, const sm::mmq_x_tab* tab, cudaStream_t s) {
#define STRATA_KQ_TAB_CASE(T) case T: return ctx == nullptr ? 1 : (int) sm::mul_mat_q_tab_case<T>(*ctx, *a, *tab, s)
    switch (t) {
#if defined(STRATA_KQ_TAB_HAS_Q4_K)
        STRATA_KQ_TAB_CASE(GGML_TYPE_Q4_K);
#endif
#if defined(STRATA_KQ_TAB_HAS_Q5_K)
        STRATA_KQ_TAB_CASE(GGML_TYPE_Q5_K);
#endif
#if defined(STRATA_KQ_TAB_HAS_Q5_1)
        STRATA_KQ_TAB_CASE(GGML_TYPE_Q5_1);
#endif
#if defined(STRATA_KQ_TAB_HAS_Q6_K)
        STRATA_KQ_TAB_CASE(GGML_TYPE_Q6_K);
#endif
#if defined(STRATA_KQ_TAB_HAS_Q8_0)
        STRATA_KQ_TAB_CASE(GGML_TYPE_Q8_0);
#endif
        default: return -1;
    }
#undef STRATA_KQ_TAB_CASE
}
}  // namespace

STRATA_KQ_EXPORT int strata_kq_tab_abi() { return STRATA_KQ_ABI_VERSION * 100000 + (int) sizeof(mmq_args); }

STRATA_KQ_EXPORT int strata_kq_tab_init(int device) {
    cudaError_t e = cudaSetDevice(device);
    if (e == cudaSuccess) e = cudaFree(nullptr);
    return (int) e;
}

STRATA_KQ_EXPORT int strata_kq_tab_has(int type) { return dispatch(nullptr, (ggml_type) type, nullptr, nullptr, nullptr) > 0; }

STRATA_KQ_EXPORT int strata_kq_tab_run(int device, int type, const void* args, const void* const* ptrs, void* stream) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return -1;
    if (const cudaError_t e = cudaSetDevice(device); e != cudaSuccess) return (int) e;
    sm::mmq_x_tab tab{};
    for (int z = 0; z < 16; ++z) tab.p[z] = (const char*) ptrs[z];
    const int r = dispatch(ctx_for(device), (ggml_type) type, (const mmq_args*) args, &tab, (cudaStream_t) stream);
    if (r < 0) return -1;
    if (r == 0) return -2;   // not covered: never 1, which is cudaErrorInvalidValue
    return (int) cudaGetLastError();
}
