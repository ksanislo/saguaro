#pragma once
// engines/qfn's HIP build, force-included after Saguaro's shared hip_compat/cuda_runtime.h (cmake/hip_backend.cmake).
#if defined(_WIN32)
// Windows: hipMemGetInfo counts this process's own allocations only, so the desktop's and other programs' share of the
// card is invisible to it; the free figure is lowered by what Windows' video memory budget for this process withholds
// (src/core/device.cu; SAGUARO_WDDM_BUDGET=0: hipMemGetInfo as it is).
namespace strata::hip_compat {
hipError_t mem_get_info(size_t* free_bytes, size_t* total_bytes);
}
#undef cudaMemGetInfo
#define cudaMemGetInfo ::strata::hip_compat::mem_get_info
#endif
