#pragma once
// engines/qfn's HIP build, force-included before Saguaro's shared hip_compat/cuda_runtime.h (cmake/hip_backend.cmake):
// hip_runtime.h pulls in <windows.h> on Windows - keep its min/max macros and the rarely used APIs out of the engine.
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
