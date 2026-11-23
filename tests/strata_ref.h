// tests/strata_ref.h - Strata's own CPU decode (engine/src/quant/sk_codecs.cpp scalar paths, ink_cpu_backend.cpp),
// the reference the strata_fmt tests hold our decode to. MIT License - Copyright (c) 2026 Strata contributors.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace ref {

inline float f16(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16, exp = (h >> 10) & 0x1Fu, mant = h & 0x03FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else {
            uint32_t e = 127 - 14, m = mant;
            while ((m & 0x0400u) == 0) m <<= 1, --e;
            m &= 0x03FFu;
            f = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) f = sign | 0x7F800000u | (mant << 13);
    else f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    float o;
    std::memcpy(&o, &f, 4);
    return o;
}
inline void unpack(const uint8_t* pl, int bits, uint8_t q[256]) {
    if (bits <= 3) {
        for (int i = 0; i < 64; ++i)
            for (int k = 0; k < 4; ++k) q[4 * i + k] = (pl[i] >> (2 * k)) & 3;
        if (bits == 3)
            for (int i = 0; i < 32; ++i)
                for (int b = 0; b < 8; ++b) q[8 * i + b] |= ((pl[64 + i] >> b) & 1) << 2;
    } else {
        for (int i = 0; i < 128; ++i) q[2 * i] = pl[i] & 15, q[2 * i + 1] = pl[i] >> 4;
        if (bits == 5)
            for (int i = 0; i < 32; ++i)
                for (int b = 0; b < 8; ++b) q[8 * i + b] |= ((pl[128 + i] >> b) & 1) << 4;
        if (bits == 6)
            for (int i = 0; i < 64; ++i)
                for (int k = 0; k < 4; ++k) q[4 * i + k] |= ((pl[128 + i] >> (2 * k)) & 3) << 4;
    }
}
inline void sk(const uint8_t* raw, int bits, float* out, size_t n) {   // dequant_sk_row
    const size_t sbb = 6 + 32 * (size_t) bits;
    const float half = (float) (1 << (bits - 1)) - 0.5f;
    for (size_t s = 0; s < n / 256; ++s) {
        const uint8_t* blk = raw + s * sbb;
        uint16_t d;
        std::memcpy(&d, blk, 2);
        const float D = f16(d);
        uint8_t q[256];
        unpack(blk + 6, bits, q);
        for (int g = 0; g < 4; ++g) {
            volatile float eff = D * ((float) blk[2 + g] / 255.0f);
            for (int i = 0; i < 64; ++i) {
                volatile float v = ((float) q[g * 64 + i] - half) * eff;
                out[s * 256 + g * 64 + i] = v;
            }
        }
    }
}
inline float e4m3(uint8_t c) {   // sk_codecs.cpp E4M3Table
    const int e = (c & 0x7F) >> 3, m = c & 7;
    float v;
    if ((c & 0x7F) == 0x7F) v = 0;
    else if (e == 0) v = (float) m * (1.0f / 512.0f);
    else v = (1.0f + (float) m * 0.125f) * std::ldexp(1.0f, e - 7);
    return (c & 0x80) ? -v : v;
}
inline float ue8m0(uint8_t e) {   // sk_codecs.cpp ue8m0_to_f32
    if (e == 0) return 5.877471754111438e-39f;
    if (e == 255) return 0.0f;
    uint32_t bits = (uint32_t) e << 23;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
constexpr float e2m1[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6};
}  // namespace ref

// ggml's NVFP4 decode without the outer scale: kvalues (2 x e2m1) x ue4m3 (e4m3 / 2), exact in F32
namespace ref {
inline float nvfp4_code_scale(const uint8_t* blocks, size_t v) {
    static const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const uint8_t* b = blocks + (v / 64) * 36;
    const int i = (int) (v % 64), sb = i / 16, j = i % 16;
    const uint8_t byte = b[4 + sb * 8 + (j & 7)];
    const uint8_t x = b[sb];
    const uint32_t e = (x >> 3) & 0xF, m = x & 7;
    float ue;
    if (e == 0) ue = (float) m * 0x1.0p-10f;
    else {
        const uint32_t bits = ((e + 119u) << 23) | (m << 20);
        std::memcpy(&ue, &bits, 4);
    }
    return (float) kv[j < 8 ? (byte & 0xF) : (byte >> 4)] * ue;
}

}  // namespace ref
