// src/ik2/ik2_pascal.cuh - saguaro-ik2's Pascal (sm_6x, k::arch_class 60) kernels: no tensor cores and FP16 math at
// 1/64 rate there, so the token path's mat-vecs get kernels of their own in FP32 (the generic scalar kernels reach
// ~55-65 % of what a P40 reads with ECC on, ~255 GB/s).  Every entry returns false where it does not take the case
// (the caller then runs the generic kernel).  NVIDIA only; each path has a switch (SAGUARO_IK2_PASCAL_EXPERTS / _Q8
// / _ATTN = 0: off).
#pragma once

#include "ik2_kernels.cuh"

#include <cstddef>
#include <cstdint>

namespace ik2::k::pascal {

/// the Pascal paths on this GPU (arch_class 60) and their switches (read once)
bool experts_on();
bool q8_on();
bool attn_on();

/// experts_gate_up_swiglu's job (same arguments, same h): NVFP4 or Q8_0 gate and up at C 2304 (Inkling-2-Mini's
/// width); false otherwise.
bool gate_up_swiglu(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up, size_t eb0,
                    const float* x, int C, int F, const float* sg, const float* su, float gcap, float ucap, uint16_t* h,
                    const int* orow, int h_ld, const int* cls, int want, cudaStream_t s);
/// experts_down's job: NVFP4 or Q8_0 down at F 1152; false otherwise.
bool down(const uint8_t* const* blob, int S, int ty_down, size_t off_down, const uint16_t* h, int C, int F, float* y,
          const int* cls, int want, cudaStream_t s);
/// mv_q8p_conv's job (same epilogues): K a multiple of 256 in the unrolled set; false otherwise.
bool mv_q8p_conv(const int8_t* q, const uint16_t* d, int N, int K, const float* x, float* y, const MvConv& c0,
                 const MvConv& c1, const MvAlt& a0, const MvAlt& a1, cudaStream_t s);


/// attn_decode_f16_at's split launch (the same partials for attn_combine): G = 3 query heads a KV head (Inkling's 24
/// on 8); false otherwise.
bool attn_split_f16(const float* q, const float* r, const uint16_t* kc, const uint16_t* vc, int kv_cap, int H, int Hkv,
                    const int* pos, int P, float qk_scale, const float* proj, int d_rel, int window, int extent,
                    float* part, cudaStream_t s);

}  // namespace ik2::k::pascal
