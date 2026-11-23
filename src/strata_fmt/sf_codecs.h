// src/strata_fmt/sf_codecs.h - Strata pack codecs as block types our engines read (host, CUDA and HIP).
// Decode arithmetic from Strata engine/src/quant/sk_codecs.cpp (MIT License - Copyright (c) 2026 Strata contributors).
//
// Every type here is a LOSSLESS re-layout of a Strata codec (the same fields, the same float operations):
//   kSH2..kSH6  half S_K super-block: 128 weights = FP16 D, u8 u[2], the S_K codes of those 128 (Strata's bit planes,
//               halved). w = (q - (2^(b-1) - 0.5)) * (D * (u / 255)). A Strata super-block (256) is two of them; the
//               half makes rows of 1152 (Inkling's expert down) whole blocks.
//   kQ8F        Q8_ROW re-laid: 32 int8 + their row's F32 scale (w = q * s).
//   kBF16       BF16 (ggml's id) as blocks of 32.
//   kMX8B       MXFP8 re-laid: a UE8M0 scale byte, 3 pad bytes, 32 F8_E4M3 weights (w = e4m3 x 2^(e - 127)).
//   kMIX        an expert layer whose experts differ in codec: every role part of an expert starts with a MixHdr.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define SF_HD __host__ __device__ __forceinline__
#else
#define SF_HD inline
#endif

namespace strata_fmt {

constexpr int kBF16 = 30, kNVFP4 = 40, kQ8F = 200, kSH2 = 202, kSH3 = 203, kSH4 = 204, kSH5 = 205, kSH6 = 206,
              kMX8B = 208, kMIX = 210;

SF_HD bool is_sh(int t) { return t >= kSH2 && t <= kSH6; }
SF_HD int sh_bits(int t) { return t - 200; }
SF_HD int block_values(int t) { return is_sh(t) ? 128 : (t == kQ8F || t == kBF16 || t == kMX8B) ? 32 : 0; }
SF_HD size_t block_bytes(int t) {
    return is_sh(t) ? 4 + 16 * (size_t) sh_bits(t) : (t == kQ8F || t == kMX8B) ? 36 : t == kBF16 ? 64 : 0;
}

// a kMIX role part: this header, the rows (rb bytes each, blocks of `type`), then aux (the up part: the expert's MACaQ
// scales s[F] as FP16 - the activation h[j] is divided by s[j] before the down product; aux_off 0: none)
struct MixHdr {
    uint8_t type;
    uint8_t pad[3];
    uint32_t rb;
    uint32_t aux_off;
    uint32_t pad2;
};
static_assert(sizeof(MixHdr) == 16, "MixHdr");

SF_HD float f16_bits(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = sign;
        } else {   // subnormal: m * 2^-24, exact
            float v = (float) m * 5.9604644775390625e-08f;
            uint32_t u;
            std::memcpy(&u, &v, 4);
            f = sign | u;
        }
    } else if (e == 31) {
        f = sign | 0x7F800000u | (m << 13);
    } else {
        f = sign | ((e + 112u) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

SF_HD float bf16_bits(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// IEEE round-to-nearest division / product (no fast-math reciprocal, no FMA contraction): Strata's CPU arithmetic
SF_HD float div_rn(float a, float b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}
SF_HD float mul_rn(float a, float b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __fmul_rn(a, b);
#else
    volatile float r = a * b;
    return r;
#endif
}

// the effective scale of sub-group g (0, 1) of a half block: D * (u / 255)
SF_HD float sh_eff(const uint8_t* b, int g) {
    return mul_rn(f16_bits((uint16_t) (b[0] | (b[1] << 8))), div_rn((float) b[2 + g], 255.0f));
}
SF_HD float sh_offset(int bits) { return (float) (1 << (bits - 1)) - 0.5f; }

// code i (0..127) of a half block of `bits`
SF_HD int sh_code(const uint8_t* b, int bits, int i) {
    const uint8_t* pl = b + 4;
    if (bits <= 3) {
        int q = (pl[i >> 2] >> (2 * (i & 3))) & 3;
        if (bits == 3) q |= ((pl[32 + (i >> 3)] >> (i & 7)) & 1) << 2;
        return q;
    }
    int q = (pl[i >> 1] >> (4 * (i & 1))) & 15;
    if (bits == 5) q |= ((pl[64 + (i >> 3)] >> (i & 7)) & 1) << 4;
    if (bits == 6) q |= ((pl[64 + (i >> 2)] >> (2 * (i & 3))) & 3) << 4;
    return q;
}

// F8_E4M3FN as Strata decodes it (sk_codecs.cpp E4M3Table: 0x7F / 0xFF -> 0), exact
SF_HD float e4m3(uint8_t c) {
    const uint32_t e = (c >> 3) & 15u, m = c & 7u;
    float v;
    if ((c & 0x7F) == 0x7F) {
        v = 0.0f;
    } else if (e == 0) {
        v = (float) m * 0.001953125f;   // m / 512
    } else {
        const uint32_t bits = ((e + 120u) << 23) | (m << 20);
        std::memcpy(&v, &bits, 4);
    }
    return (c & 0x80) ? -v : v;
}
// UE8M0 as Strata decodes it (0 -> 2^-127, 255 -> 0)
SF_HD float ue8m0(uint8_t e) {
    const uint32_t bits = e == 0 ? 0x00400000u : e == 255 ? 0u : (uint32_t) e << 23;
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

// value v of a run of blocks of type t (one of this file's; not kMIX)
SF_HD float value(int t, const uint8_t* blocks, size_t v) {
    if (is_sh(t)) {
        const int bits = sh_bits(t);
        const uint8_t* b = blocks + (v >> 7) * block_bytes(t);
        const int i = (int) (v & 127);
        return mul_rn((float) sh_code(b, bits, i) - sh_offset(bits), sh_eff(b, i >> 6));
    }
    if (t == kQ8F) {
        const uint8_t* b = blocks + (v >> 5) * 36;
        float d;
        std::memcpy(&d, b, 4);
        return mul_rn((float) (int8_t) b[4 + (v & 31)], d);
    }
    if (t == kMX8B) {
        const uint8_t* b = blocks + (v >> 5) * 36;
        return mul_rn(e4m3(b[4 + (v & 31)]), ue8m0(b[0]));
    }
    if (t == kBF16) {
        const uint8_t* b = blocks + v * 2;
        return bf16_bits((uint16_t) (b[0] | (b[1] << 8)));
    }
    return 0.0f;
}

// ---- host re-layouts (Strata bytes -> the types above), all byte moves

// one Strata S_K super-block (sb_bytes = 6 + 32 bits) -> two half blocks (4 + 16 bits each)
inline void sk_to_sh(const uint8_t* sb, int bits, uint8_t* out) {
    const size_t hb = 4 + 16 * (size_t) bits;
    const uint8_t* pl = sb + 6;
    for (int h = 0; h < 2; ++h) {
        uint8_t* o = out + h * hb;
        o[0] = sb[0], o[1] = sb[1], o[2] = sb[2 + 2 * h], o[3] = sb[3 + 2 * h];
        uint8_t* q = o + 4;
        if (bits <= 3) {
            std::memcpy(q, pl + 32 * h, 32);                                  // 2-bit plane (64 B)
            if (bits == 3) std::memcpy(q + 32, pl + 64 + 16 * h, 16);         // 1-bit plane (32 B)
        } else {
            std::memcpy(q, pl + 64 * h, 64);                                  // nibble plane (128 B)
            if (bits == 5) std::memcpy(q + 64, pl + 128 + 16 * h, 16);        // 1-bit plane (32 B)
            if (bits == 6) std::memcpy(q + 64, pl + 128 + 32 * h, 32);        // 2-bit plane (64 B)
        }
    }
}
inline size_t sk_sb_bytes(int bits) { return 6 + 32 * (size_t) bits; }

// one SF4 row (cols values: E2M1 nibbles [cols / 2], low = even column, then E4M3 scales [cols / 16]) -> NVFP4 blocks
// (ggml block_nvfp4: 4 UE4M3 scales, then 4 x 8 bytes, byte j of a sub-block = value j | value j + 8 << 4)
inline void sf4_row_to_nvfp4(const uint8_t* nib, const uint8_t* sc, int cols, uint8_t* out) {
    for (int b = 0; b < cols / 64; ++b) {
        uint8_t* o = out + (size_t) b * 36;
        for (int s = 0; s < 4; ++s) {
            const int g = b * 4 + s;
            o[s] = sc[g];
            uint8_t v[16];
            for (int j = 0; j < 8; ++j) v[2 * j] = nib[g * 8 + j] & 15, v[2 * j + 1] = nib[g * 8 + j] >> 4;
            for (int j = 0; j < 8; ++j) o[4 + s * 8 + j] = (uint8_t) (v[j] | (v[j + 8] << 4));
        }
    }
}

// one MX8 row (E4M3 weights [cols], then elsewhere UE8M0 scales [cols / 32]) -> kMX8 blocks
inline void mx8_row_to_blocks(const uint8_t* w, const uint8_t* sc, int cols, uint8_t* out) {
    for (int b = 0; b < cols / 32; ++b) {
        uint8_t* o = out + (size_t) b * 36;
        o[0] = sc[b], o[1] = o[2] = o[3] = 0;
        std::memcpy(o + 4, w + b * 32, 32);
    }
}

// one Q8_ROW row (int8 [cols], its F32 scale) -> kQ8F blocks
inline void q8row_to_q8f(const int8_t* q, float s, int cols, uint8_t* out) {
    for (int b = 0; b < cols / 32; ++b) {
        std::memcpy(out + (size_t) b * 36, &s, 4);
        std::memcpy(out + (size_t) b * 36 + 4, q + b * 32, 32);
    }
}

}  // namespace strata_fmt
