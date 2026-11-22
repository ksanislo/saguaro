// tests/strata_fmt_gpu_test.cu - a Strata pack's weights as saguaro-ik2's GPU kernels decode them, against Strata's CPU
// decode (tests/strata_ref.h), and one token through the expert kernels of a mixed layer against a double reference.
//   strata_fmt_gpu_test PACK.strata [--experts N]     (N experts a layer, default 6, codes mixed)
// Pass: every decoded value bit-equal (SF4: code x block scale, the outer scale apart; S_K: the value; trunk: the value),
// FP16 decodes = the F32 value rounded to nearest, and the token's y within 2e-3 of the reference (relative, max |y|).
#include "ik2_kernels.cuh"
#include "sf_codecs.h"
#include "strata_inkling.hpp"
#include "strata_ref.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata_fmt;
namespace k = ik2::k;

#define CK(x)                                                                                                \
    do {                                                                                                     \
        cudaError_t e_ = (x);                                                                                \
        if (e_ != cudaSuccess) {                                                                             \
            std::fprintf(stderr, "%s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);                \
            std::exit(1);                                                                                    \
        }                                                                                                    \
    } while (0)

static size_t g_bad = 0, g_n = 0, g_negz = 0;
static void same(const char* what, size_t i, float a, float b) {
    ++g_n;
    if (std::memcmp(&a, &b, 4) == 0) return;
    if (a == 0.0f && b == 0.0f) {
        ++g_negz;
        return;
    }
    if (g_bad++ < 10) std::printf("MISMATCH %s [%zu]: gpu %.9g ref %.9g\n", what, i, a, b);
}

template <class T>
static T* up(const void* h, size_t n) {
    T* d = nullptr;
    CK(cudaMalloc(&d, n * sizeof(T)));
    CK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// decode n values of `type` at device d (F32 and FP16) and hold them to want[]
static void check_decode(const char* what, int type, const uint8_t* dev, size_t n, const float* want) {
    float* o = nullptr;
    uint16_t* o16 = nullptr;
    CK(cudaMalloc(&o, n * 4));
    CK(cudaMalloc(&o16, n * 2));
    k::decode_f32(type, dev, n, o, 0);
    k::decode_f16(type, dev, n, o16, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> h(n);
    std::vector<uint16_t> h16(n);
    CK(cudaMemcpy(h.data(), o, n * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h16.data(), o16, n * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        same(what, i, h[i], want[i]);
        const __half r = __float2half_rn(h[i]);
        uint16_t rb;
        std::memcpy(&rb, &r, 2);
        if (rb != h16[i] && !(((rb | h16[i]) & 0x7FFF) == 0) && g_bad++ < 10)
            std::printf("MISMATCH %s f16 [%zu]: %04x vs %04x\n", what, i, h16[i], rb);
    }
    CK(cudaFree(o));
    CK(cudaFree(o16));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: strata_fmt_gpu_test PACK.strata [--experts N]\n");
        return 2;
    }
    int n_exp = 6;
    for (int i = 2; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], "--experts")) n_exp = std::atoi(argv[i + 1]);
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
    const float gcap = (float) tc.at("mlp_gate_softcap").f(), ucap = (float) tc.at("mlp_up_softcap").f();

    // 1. the trunk's non-F32 tensors (Q8F, BF16, SH*), decoded on the GPU
    size_t nt = 0;
    for (const auto& t : V.tensors()) {
        std::string pn;
        int tr = 0;
        if (!V.source(t.name, pn, tr) || t.type == 0) continue;
        const size_t n = (size_t) t.elements();
        std::vector<float> want(n);
        k::decode_host((int) t.type, V.data(t.name), n, want.data());   // (= Strata's decode: strata_fmt_test)
        const PackTensor* pt = P.find(pn);
        if (pt->fmt != "BF16" && pt->fmt != "Q8_ROW") {   // S_K: also straight against Strata's decode
            std::vector<float> r(n);
            int bits = 0;
            for (int c = 2; c <= 6; ++c)
                if (pt->fmt == code_name(c)) bits = c;
            ref::sk(P.tensor(*pt), bits, r.data(), n);
            for (size_t i = 0; i < n; ++i) same("host S_K", i, want[i], r[i]);
        }
        uint8_t* d = up<uint8_t>(V.data(t.name), V.bytes(t.name));
        check_decode(t.name.c_str(), (int) t.type, d, n, want.data());
        if (t.shape.size() == 2 && t.shape[0] % 32 == 0 && nt % 7 == 0) {   // (a sample: the token path's row dots)
            const int K = (int) t.shape[0], N = (int) t.shape[1];
            std::vector<float> x(K), y(N);
            std::mt19937 r2(nt);
            std::normal_distribution<float> n2(0.0f, 1.0f);
            for (auto& v : x) v = n2(r2);
            float* dx = up<float>(x.data(), K);
            float* dy = nullptr;
            CK(cudaMalloc(&dy, (size_t) N * 4));
            k::mv_blocks((int) t.type, d, N, K, dx, dy, 0);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(y.data(), dy, (size_t) N * 4, cudaMemcpyDeviceToHost));
            double mx = 0, md = 0;
            for (int rr = 0; rr < N; ++rr) {
                double acc = 0;
                for (int c = 0; c < K; ++c) acc += (double) want[(size_t) rr * K + c] * x[c];
                mx = std::max(mx, std::fabs(acc)), md = std::max(md, std::fabs(acc - y[rr]));
            }
            if (md > 1e-4 * mx && g_bad++ < 20) std::printf("MV %s: max |dy| %.3g of %.3g\n", t.name.c_str(), md, mx);
            cudaFree(dx), cudaFree(dy);
        }
        CK(cudaFree(d));
        ++nt;
    }
    std::printf("trunk: %zu tensors on the GPU, %zu values, %zu mismatches\n", nt, g_n, g_bad);

    // 2. experts: each role part (kMIX or NVFP4) decoded on the GPU against Strata's fields
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    double worst = 0;
    int n_tok = 0;
    for (int l = 0; l < L; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        const std::string ngu = b + "ffn_gate_up_exps.weight", ndn = b + "ffn_down_exps.weight";
        const size_t egu = V.bytes(ngu) / E, edn = V.bytes(ndn) / E;
        const int lt = V.layer_type(l);
        const float* sgu = (const float*) V.data(b + "ffn_gate_up_exps.scale");
        const float* sdn = (const float*) V.data(b + "ffn_down_exps.scale");
        // the layer's first SF4 and S_K experts, then the rest in order
        std::vector<int> pick;
        for (int want_sf4 : {1, 0})
            for (int e = 0; e < E && (int) pick.size() < (want_sf4 ? 1 : 2); ++e)
                if ((V.expert(l, e).gu_code == kSF4) == (bool) want_sf4) pick.push_back(e);
        for (int e = 0; e < E && (int) pick.size() < n_exp; ++e)
            if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
        const int S = (int) pick.size();
        const size_t slot = egu + edn;
        std::vector<uint8_t> host(slot * S);
        for (int i = 0; i < S; ++i) {
            V.read_experts(ngu, pick[i], 1, egu, host.data() + i * slot);
            V.read_experts(ndn, pick[i], 1, edn, host.data() + i * slot + egu);
        }
        uint8_t* dev = up<uint8_t>(host.data(), host.size());
        // reference weights (Strata's decode, outer scale and MACaQ applied: the effective weights)
        std::vector<std::vector<double>> Wg(S), Wu(S), Wd(S);
        for (int i = 0; i < S; ++i) {
            const PackExpert& pe = V.expert(l, pick[i]);
            const uint8_t* body = P.body(pe);
            Wg[i].resize((size_t) F * C), Wu[i].resize((size_t) F * C), Wd[i].resize((size_t) C * F);
            std::vector<float> code_scale_gu(2 * (size_t) F * C), code_scale_dn((size_t) C * F);
            double og = 1, od = 1;
            std::vector<double> inv_s(F, 1.0);
            if (pe.gu_code == kSF4) {
                float h4[4];
                std::memcpy(h4, body, 16);
                og = h4[0] / h4[1], od = h4[2] / h4[3];
                const uint8_t* nib = body + 16;
                const uint8_t* sc = nib + (size_t) 2 * F * (C / 2);
                for (size_t r = 0; r < 2 * (size_t) F; ++r)
                    for (int c = 0; c < C; ++c) {
                        const uint8_t by = nib[r * (C / 2) + c / 2];
                        code_scale_gu[r * C + c] = ref::e2m1[(c & 1) ? by >> 4 : by & 15] * ref::e4m3(sc[r * (C / 16) + c / 16]);
                    }
                const uint8_t* dn = body + pe.dn_off;
                const uint8_t* dsc = dn + (size_t) C * (F / 2);
                for (size_t r = 0; r < (size_t) C; ++r)
                    for (int c = 0; c < F; ++c) {
                        const uint8_t by = dn[r * (F / 2) + c / 2];
                        code_scale_dn[r * F + c] = ref::e2m1[(c & 1) ? by >> 4 : by & 15] * ref::e4m3(dsc[r * (F / 16) + c / 16]);
                    }
            } else {
                ref::sk(body + 2 * (size_t) F, pe.gu_code, code_scale_gu.data(), code_scale_gu.size());
                ref::sk(body + pe.dn_off, pe.dn_code, code_scale_dn.data(), code_scale_dn.size());
                for (int j = 0; j < F; ++j) inv_s[j] = 1.0 / ref::f16(((const uint16_t*) body)[j]);
            }
            // GPU decode of each part (rows of gate = Strata's even rows, up = odd rows)
            const uint8_t* ds = dev + i * slot;
            std::vector<float> wg((size_t) F * C), wu((size_t) F * C);
            for (size_t r = 0; r < (size_t) F; ++r)
                for (int c = 0; c < C; ++c) {
                    wg[r * C + c] = code_scale_gu[(2 * r) * C + c];
                    wu[r * C + c] = code_scale_gu[(2 * r + 1) * C + c];
                }
            const int tg = lt == kMIX ? kMIX : kNVFP4;
            check_decode("expert gate", tg, ds, (size_t) F * C, wg.data());
            check_decode("expert up", tg, ds + egu / 2, (size_t) F * C, wu.data());
            check_decode("expert down", tg, ds + egu, (size_t) C * F, code_scale_dn.data());
            for (size_t x = 0; x < wg.size(); ++x) Wg[i][x] = wg[x] * og, Wu[i][x] = wu[x] * og;
            for (size_t r = 0; r < (size_t) C; ++r)
                for (int c = 0; c < F; ++c) Wd[i][r * F + c] = code_scale_dn[r * F + c] * od * inv_s[c];
        }
        // one token through experts_gate_up_swiglu + experts_down (the token path's generic kernels)
        if (l % 4 == 0) {
            std::vector<float> x(C);
            for (auto& v : x) v = nd(rng) * 0.5f;
            std::vector<const uint8_t*> blob(S);
            std::vector<int> tok(S, 0);
            std::vector<float> sg(S), su(S);
            for (int i = 0; i < S; ++i) blob[i] = dev + i * slot, sg[i] = sgu[pick[i]], su[i] = sgu[pick[i]];
            const uint8_t** dblob = up<const uint8_t*>(blob.data(), S);
            int* dtok = up<int>(tok.data(), S);
            float* dx = up<float>(x.data(), C);
            float* dsg = up<float>(sg.data(), S);
            float* dsu = up<float>(su.data(), S);
            uint16_t* dh = nullptr;
            float* dy = nullptr;
            CK(cudaMalloc(&dh, (size_t) S * F * 2));
            CK(cudaMalloc(&dy, (size_t) S * C * 4));
            k::experts_gate_up_swiglu(dblob, dtok, S, lt, lt, egu / 2, dx, C, F, dsg, dsu, gcap, ucap, dh, 0);
            k::experts_down(dblob, S, lt, egu, dh, C, F, dy, 0);
            CK(cudaDeviceSynchronize());
            std::vector<float> y((size_t) S * C);
            CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
            for (int i = 0; i < S; ++i) {
                std::vector<double> h(F);
                for (int j = 0; j < F; ++j) {
                    double g = 0, u = 0;
                    for (int c = 0; c < C; ++c) g += Wg[i][(size_t) j * C + c] * x[c], u += Wu[i][(size_t) j * C + c] * x[c];
                    h[j] = gcap * std::tanh(g / gcap) / (1 + std::exp(-g)) * ucap * std::tanh(u / ucap);
                }
                double mx = 0, md = 0;
                for (int r = 0; r < C; ++r) {
                    double a = 0;
                    for (int j = 0; j < F; ++j) a += Wd[i][(size_t) r * F + j] * h[j];
                    mx = std::max(mx, std::fabs(a)), md = std::max(md, std::fabs(a - (double) y[(size_t) i * C + r] * sdn[pick[i]]));   // (down's scale: the routing weight's, in the engine)
                }
                const double rel = md / std::max(mx, 1e-30);
                worst = std::max(worst, rel);
                if (rel > 2e-3 && g_bad++ < 20)
                    std::printf("TOKEN layer %d expert %d (%s): max |dy| %.3g of max |y| %.3g\n", l, pick[i],
                                code_name(V.expert(l, pick[i]).gu_code), md, mx);
            }
            ++n_tok;
            cudaFree(dblob), cudaFree(dtok), cudaFree(dx), cudaFree(dsg), cudaFree(dsu), cudaFree(dh), cudaFree(dy);
        }
        CK(cudaFree(dev));
    }
    std::printf("experts: %zu values decoded on the GPU in all, %zu SF4 -0 codes read as +0; token checks %d layers, "
                "worst relative error %.2e\n", g_n, g_negz, n_tok, worst);
    std::printf("%s: %zu mismatches\n", g_bad ? "FAIL" : "PASS", g_bad);
    return g_bad ? 1 : 0;
}
