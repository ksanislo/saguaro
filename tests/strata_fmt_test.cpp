// tests/strata_fmt_test.cpp - a Strata pack through src/strata_fmt against Strata's own CPU decode, value by value.
//   strata_fmt_test PACK.strata [--experts N]   (N: experts a layer to check, default all)
// Reference decode: Strata engine/src/quant/sk_codecs.cpp's scalar paths (dequant_sk_row, dequant_sf4_row) and
// engine/src/inkling2/ink_cpu_backend.cpp (Q8_ROW) - MIT License, Copyright (c) 2026 Strata contributors.
// Pass = every trunk value bit-equal; every S_K expert value bit-equal; every SF4 expert's code x block scale
// bit-equal (exact in F32 both ways) and its outer scale equal - the same real weights.
#include "sf_codecs.h"
#include "strata_inkling.hpp"
#include "strata_ref.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata_fmt;



static size_t g_bad = 0, g_checked = 0, g_negzero = 0;
// bit-equal; for SF4 the code 8 (E2M1 -0) is +0 in ggml's NVFP4 table: the same weight, counted apart
static void expect_eq(const char* what, const std::string& name, size_t i, float a, float b) {
    ++g_checked;
    if (std::memcmp(&a, &b, 4) == 0) return;
    if (a == 0.0f && b == 0.0f && what[1] == 'f') {
        ++g_negzero;
        return;
    }
    if (g_bad++ < 10) std::printf("MISMATCH %s %s [%zu]: ours %.9g strata %.9g\n", what, name.c_str(), i, a, b);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: strata_fmt_test PACK.strata [--experts N]\n");
        return 2;
    }
    int n_exp = -1;
    for (int i = 2; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], "--experts")) n_exp = std::atoi(argv[i + 1]);
    const auto t0 = std::chrono::steady_clock::now();
    InklingView V;
    std::string err;
    if (!V.build(argv[1], err)) {
        std::printf("FAIL: %s\n", err.c_str());
        return 1;
    }
    std::printf("%s\n", V.summary().c_str());
    const Pack& P = V.pack();
    const JVal& tc = P.manifest().at("config").at("text_config");
    const int C = (int) tc.at("hidden_size").i(), F = (int) tc.at("intermediate_size").i();
    const int E = (int) tc.at("n_routed_experts").i(), L = (int) tc.at("num_hidden_layers").i();

    // trunk: every view tensor against its pack tensor decoded Strata's way (reordered as the map says)
    size_t n_trunk = 0;
    for (const auto& t : V.tensors()) {
        std::string pn;
        int tr = 0;
        if (!V.source(t.name, pn, tr)) continue;
        const PackTensor* pt = P.find(pn);
        const size_t n = (size_t) pt->numel();
        std::vector<float> r(n), o(n);
        const uint8_t* src = P.tensor(*pt);
        if (pt->fmt == "F32") std::memcpy(r.data(), src, n * 4);
        else if (pt->fmt == "BF16")
            for (size_t i = 0; i < n; ++i) {
                uint32_t u = (uint32_t) ((const uint16_t*) src)[i] << 16;
                std::memcpy(&r[i], &u, 4);
            }
        else if (pt->fmt == "Q8_ROW") {
            const size_t rows = (size_t) pt->shape[0], cols = n / rows;
            const float* sc = (const float*) src;
            const int8_t* q = (const int8_t*) (src + rows * 4);
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j) r[i * cols + j] = (float) q[i * cols + j] * sc[i];
        } else {
            int bits = 0;
            for (int c = 2; c <= 6; ++c)
                if (pt->fmt == code_name(c)) bits = c;
            if (!bits) {
                std::printf("FAIL: %s format %s\n", pt->name.c_str(), pt->fmt.c_str());
                return 1;
            }
            ref::sk(src, bits, r.data(), n);
        }
        if (tr == 1) {   // [d_rel][ext] -> [ext][d_rel]
            const size_t a = (size_t) tc.at("d_rel").i(), b = n / a;
            std::vector<float> x(n);
            for (size_t i = 0; i < a; ++i)
                for (size_t j = 0; j < b; ++j) x[j * a + i] = r[i * b + j];
            r.swap(x);
        } else if (tr == 2) {
            const size_t k = (size_t) tc.at("sconv_kernel_size").i(), kv = n / k;
            std::vector<float> x(n);
            for (size_t lag = 0; lag < k; ++lag)
                for (size_t ch = 0; ch < kv; ++ch) x[ch * k + (k - 1 - lag)] = r[lag * kv + ch];
            r.swap(x);
        }
        const uint8_t* ours = V.data(t.name);
        for (size_t i = 0; i < n; ++i) {
            if (t.type == 0) std::memcpy(&o[i], ours + 4 * i, 4);
            else o[i] = value((int) t.type, ours, i);
            expect_eq("trunk", t.name, i, o[i], r[i]);
        }
        ++n_trunk;
    }
    std::printf("trunk: %zu tensors (%zu of %zu pack tensors), %zu values checked, %zu mismatches\n", n_trunk, n_trunk,
                P.trunk().size(), g_checked, g_bad);
    if (n_trunk != P.trunk().size()) {
        std::printf("FAIL: pack tensors not mapped\n");
        return 1;
    }

    // experts
    const size_t trunk_checked = g_checked;
    size_t n_done = 0, n_sf4 = 0, n_sk = 0, n_mx8 = 0, scale_bad = 0, macaq_bad = 0;
    std::vector<uint8_t> gu, dn;
    std::vector<float> r(2 * (size_t) F * C), o;
    for (int l = 0; l < L; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        const std::string ngu = b + "ffn_gate_up_exps.weight", ndn = b + "ffn_down_exps.weight";
        const size_t egu = V.bytes(ngu) / E, edn = V.bytes(ndn) / E;
        gu.resize(egu), dn.resize(edn);
        const float* sgu = (const float*) V.data(b + "ffn_gate_up_exps.scale");
        const float* sdn = (const float*) V.data(b + "ffn_down_exps.scale");
        const bool mix = V.layer_type(l) == kMIX;
        const int ne = n_exp > 0 ? std::min(n_exp, E) : E;
        for (int e = 0; e < ne; ++e) {
            const PackExpert& pe = V.expert(l, e);
            const uint8_t* body = P.body(pe);
            V.read_experts(ngu, e, 1, egu, gu.data());
            V.read_experts(ndn, e, 1, edn, dn.data());
            // [gate | up] parts and down: rows pointers, types
            const uint8_t* part[3];
            int ty[3];
            if (mix) {
                const size_t pg = egu / 2;
                for (int h = 0; h < 2; ++h) {
                    MixHdr hd;
                    std::memcpy(&hd, gu.data() + h * pg, 16);
                    ty[h] = hd.type, part[h] = gu.data() + h * pg + 16;
                    if (h == 1) {
                        const uint16_t* mq = V.macaq(l, e);
                        if (mq && (hd.aux_off == 0 || std::memcmp(gu.data() + pg + hd.aux_off, mq, 2 * (size_t) F)))
                            ++macaq_bad;
                        if (!mq && hd.aux_off) ++macaq_bad;
                    }
                }
                MixHdr hd;
                std::memcpy(&hd, dn.data(), 16);
                ty[2] = hd.type, part[2] = dn.data() + 16;
            } else {
                ty[0] = ty[1] = ty[2] = V.layer_type(l);
                part[0] = gu.data(), part[1] = gu.data() + egu / 2, part[2] = dn.data();
            }
            if (pe.gu_code == kMX8) {   // dequant_mx8_row: E4M3 x UE8M0
                ++n_mx8;
                if (sgu[e] != 1.0f || sdn[e] != 1.0f) ++scale_bad;
                const uint8_t* sc = body + (size_t) 2 * F * C;
                for (int rr = 0; rr < 2 * F; ++rr) {
                    const uint8_t* rowp = part[rr & 1] + (size_t) (rr >> 1) * (C / 32 * 36);
                    for (int c = 0; c < C; ++c) {
                        volatile float want = ref::e4m3(body[(size_t) rr * C + c]) * ref::ue8m0(sc[(size_t) rr * (C / 32) + c / 32]);
                        expect_eq("mx8 gate_up", b + std::to_string(e), (size_t) rr * C + c, value(ty[rr & 1], rowp, c), want);
                    }
                }
                const uint8_t* dw = body + pe.dn_off;
                const uint8_t* dsc = dw + (size_t) C * F;
                for (int rr = 0; rr < C; ++rr) {
                    const uint8_t* rowp = part[2] + (size_t) rr * (F / 32 * 36);
                    for (int c = 0; c < F; ++c) {
                        volatile float want = ref::e4m3(dw[(size_t) rr * F + c]) * ref::ue8m0(dsc[(size_t) rr * (F / 32) + c / 32]);
                        expect_eq("mx8 down", b + std::to_string(e), (size_t) rr * F + c, value(ty[2], rowp, c), want);
                    }
                }
            } else if (pe.gu_code == kSF4) {
                ++n_sf4;
                float h[4];
                std::memcpy(h, body, 16);
                const float og = h[0] / h[1], od = h[2] / h[3];
                if (std::memcmp(&og, &sgu[e], 4) || std::memcmp(&od, &sdn[e], 4)) ++scale_bad;
                // code x block scale, Strata's fields: e2m1[nib] * e4m3[scale] (exact)
                const uint8_t* nib = body + 16;
                const uint8_t* sc = nib + (size_t) 2 * F * (C / 2);
                for (int rr = 0; rr < 2 * F; ++rr) {
                    const uint8_t* rowp = part[rr & 1] + (size_t) (rr >> 1) * (C / 64 * 36);
                    for (int c = 0; c < C; ++c) {
                        const uint8_t by = nib[(size_t) rr * (C / 2) + c / 2];
                        const float want = ref::e2m1[(c & 1) ? by >> 4 : by & 15] * ref::e4m3(sc[(size_t) rr * (C / 16) + c / 16]);
                        expect_eq("sf4 gate_up", b + std::to_string(e), (size_t) rr * C + c, ref::nvfp4_code_scale(rowp, c), want);
                    }
                }
                const uint8_t* dnib = body + pe.dn_off;
                const uint8_t* dsc = dnib + (size_t) C * (F / 2);
                for (int rr = 0; rr < C; ++rr) {
                    const uint8_t* rowp = part[2] + (size_t) rr * (F / 64 * 36);
                    for (int c = 0; c < F; ++c) {
                        const uint8_t by = dnib[(size_t) rr * (F / 2) + c / 2];
                        const float want = ref::e2m1[(c & 1) ? by >> 4 : by & 15] * ref::e4m3(dsc[(size_t) rr * (F / 16) + c / 16]);
                        expect_eq("sf4 down", b + std::to_string(e), (size_t) rr * F + c, ref::nvfp4_code_scale(rowp, c), want);
                    }
                }
            } else {
                ++n_sk;
                if (sgu[e] != 1.0f || sdn[e] != 1.0f) ++scale_bad;
                // gate_up: Strata's [2F][C] interleaved rows
                ref::sk(body + 2 * (size_t) F, pe.gu_code, r.data(), 2 * (size_t) F * C);
                const size_t rb = (size_t) C / 128 * (4 + 16 * (size_t) pe.gu_code);
                for (int rr = 0; rr < 2 * F; ++rr) {
                    const uint8_t* rowp = part[rr & 1] + (size_t) (rr >> 1) * rb;
                    for (int c = 0; c < C; ++c)
                        expect_eq("sk gate_up", b + std::to_string(e), (size_t) rr * C + c, value(ty[rr & 1], rowp, c), r[(size_t) rr * C + c]);
                }
                ref::sk(body + pe.dn_off, pe.dn_code, r.data(), (size_t) C * F);
                const size_t rbd = (size_t) F / 128 * (4 + 16 * (size_t) pe.dn_code);
                for (int rr = 0; rr < C; ++rr) {
                    const uint8_t* rowp = part[2] + (size_t) rr * rbd;
                    for (int c = 0; c < F; ++c)
                        expect_eq("sk down", b + std::to_string(e), (size_t) rr * F + c, value(ty[2], rowp, c), r[(size_t) rr * F + c]);
                }
                if (ty[0] != 200 + pe.gu_code || ty[2] != 200 + pe.dn_code) ++scale_bad;
            }
            ++n_done;
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("experts: %zu (%zu SF4, %zu S_K, %zu MX8), %zu values checked, scale/type mismatches %zu, MACaQ mismatches %zu\n",
                n_done, n_sf4, n_sk, n_mx8, g_checked - trunk_checked, scale_bad, macaq_bad);
    const bool ok = g_bad == 0 && scale_bad == 0 && macaq_bad == 0;
    std::printf("%s: %zu values, %zu mismatches (%zu SF4 -0 codes read as +0) (%.0f s)\n", ok ? "PASS" : "FAIL", g_checked,
                g_bad, g_negzero, secs);
    return ok ? 0 : 1;
}
