// src/strata_fmt/strata_inkling.cpp - see strata_inkling.hpp. Name map: the inverse of Strata's
// engine/src/inkling2/ink_gguf_loader.h for llama.cpp master's layout (MIT License - Copyright (c) 2026 Strata
// contributors); checked bit-equal against unsloth's UD-Q4_K_XL by tools/strata (tests/strata_fmt_test.cpp).
#include "strata_inkling.hpp"

#include "sf_codecs.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>

namespace strata_fmt {

namespace {

strata::MetaValue mv_u32(uint64_t v) {
    strata::MetaValue m;
    m.type = strata::MetaType::U32;
    m.u = v;
    return m;
}
strata::MetaValue mv_f32(double v) {
    strata::MetaValue m;
    m.type = strata::MetaType::F32;
    m.f = (double) (float) v;
    return m;
}
strata::MetaValue mv_str(const std::string& s) {
    strata::MetaValue m;
    m.type = strata::MetaType::STRING;
    m.s = s;
    return m;
}
strata::MetaValue mv_arr(const std::vector<strata::MetaValue>& items, strata::MetaType elem) {
    strata::MetaValue m;
    m.type = strata::MetaType::ARRAY;
    m.elem = elem;
    m.count = items.size();
    m.items = items;
    return m;
}

size_t row_bytes(int code, int K) {
    if (code == kSF4) return (size_t) K / 64 * 36;
    if (code == kMX8) return (size_t) K / 32 * 36;
    if (code >= kS2_K && code <= kS6_K) return (size_t) K / 128 * (4 + 16 * (size_t) code);
    return 0;
}
size_t align16(size_t x) { return (x + 15) & ~(size_t) 15; }

// S_K super-blocks (n values, flat) -> half blocks
void sk_run_to_sh(const uint8_t* src, int code, size_t n, uint8_t* dst) {
    const size_t sbb = sk_sb_bytes(code), hb = 4 + 16 * (size_t) code;
    for (size_t s = 0; s < n / 256; ++s) sk_to_sh(src + s * sbb, code, dst + s * 2 * hb);
}

}  // namespace

void InklingView::add(const std::string& name, uint32_t type, std::vector<uint64_t> ne, std::vector<uint8_t> buf,
                      const uint8_t* ptr, size_t nbytes) {
    strata::TensorInfo t;
    t.name = name;
    t.type = type;
    t.shape = std::move(ne);
    Owned o;
    o.idx = tensors_.size();
    o.buf = std::move(buf);
    o.ptr = ptr ? ptr : o.buf.data();
    o.nbytes = ptr ? nbytes : o.buf.size();
    by_name_[name] = own_.size();
    tensors_.push_back(std::move(t));
    own_.push_back(std::move(o));
}

bool InklingView::add_trunk(const std::string& gguf, const std::string& sname, int transform, std::string& err) {
    const PackTensor* t = pack_.find(sname);
    if (!t) return err = "the pack has no " + sname + " (for " + gguf + ")", false;
    const uint8_t* src = pack_.tensor(*t);
    const int64_t n = t->numel();
    const int64_t cols = t->shape.empty() ? 1 : t->shape.back();
    const int64_t rows = n / cols;
    std::vector<uint64_t> ne;
    for (auto it = t->shape.rbegin(); it != t->shape.rend(); ++it) ne.push_back((uint64_t) *it);
    if (t->fmt == "F32") {
        if (transform == 0) {
            add(gguf, 0, ne, {}, src, (size_t) n * 4);
            own_.back().src = sname;
            return true;
        }
        const float* f = (const float*) src;
        std::vector<uint8_t> out((size_t) n * 4);
        float* o = (float*) out.data();
        if (transform == 1) {   // [d_rel][ext] -> [ext][d_rel]
            const int64_t a = d_rel_, b = n / d_rel_;
            for (int64_t i = 0; i < a; ++i)
                for (int64_t j = 0; j < b; ++j) o[j * a + i] = f[i * b + j];
            ne = {(uint64_t) a, (uint64_t) b};
        } else {                // static_w [k][KV] lag 0..k-1 -> [KV][k] lag k-1..0
            const int64_t k = sconv_, kv = n / sconv_;
            for (int64_t lag = 0; lag < k; ++lag)
                for (int64_t ch = 0; ch < kv; ++ch) o[ch * k + (k - 1 - lag)] = f[lag * kv + ch];
            ne = {(uint64_t) k, (uint64_t) kv};
        }
        add(gguf, 0, ne, std::move(out));
        own_.back().src = sname, own_.back().transform = transform;
        return true;
    }
    if (transform) return err = sname + ": a " + t->fmt + " tensor to reorder", false;
    if (t->fmt == "BF16") {
        add(gguf, kBF16, ne, {}, src, (size_t) n * 2);
        own_.back().src = sname;
        return true;
    }
    if (t->fmt == "Q8_ROW") {
        if (cols % 32) return err = sname + ": Q8_ROW rows not whole blocks of 32", false;
        std::vector<uint8_t> out((size_t) rows * (cols / 32) * 36);
        const float* sc = (const float*) src;
        const int8_t* q = (const int8_t*) (src + rows * 4);
        for (int64_t r = 0; r < rows; ++r)
            q8row_to_q8f(q + r * cols, sc[r], (int) cols, out.data() + (size_t) r * (cols / 32) * 36);
        add(gguf, kQ8F, ne, std::move(out));
        own_.back().src = sname;
        return true;
    }
    for (int code = kS2_K; code <= kS6_K; ++code)
        if (t->fmt == code_name(code)) {
            if (cols % 128 || n % 256) return err = sname + ": S_K rows not whole half blocks", false;
            std::vector<uint8_t> out((size_t) n / 128 * (4 + 16 * (size_t) code));
            sk_run_to_sh(src, code, (size_t) n, out.data());
            add(gguf, 200 + code, ne, std::move(out));
            own_.back().src = sname;
            return true;
        }
    return err = sname + ": format " + t->fmt + " not read here", false;
}

bool InklingView::build(const std::string& path, std::string& err) {
    if (!pack_.open(path, err)) return false;
    const JVal& m = pack_.manifest();
    try {
        if (!m.has("arch") || m.at("arch").str() != "inkling2_loop")
            return err = path + ": not an Inkling 2 pack (arch " + (m.has("arch") ? m.at("arch").str() : "?") + ")", false;
        const JVal& tc = m.at("config").at("text_config");
        auto gi = [&](const char* k) { return tc.at(k).i(); };
        auto gf = [&](const char* k) { return tc.at(k).f(); };
        const int n_layer = (int) gi("num_hidden_layers");
        C_ = (int) gi("hidden_size"), F_ = (int) gi("intermediate_size"), E_ = (int) gi("n_routed_experts");
        KV_ = (int) (gi("num_key_value_heads") * gi("head_dim")), d_rel_ = (int) gi("d_rel");
        sconv_ = (int) gi("sconv_kernel_size");
        int core0 = 0, core1 = 0, passes = 1, rank = 0;
        if (tc.has("loop_config")) {
            const JVal& lc = tc.at("loop_config");
            core0 = (int) lc.at("core_start").i(), core1 = (int) lc.at("core_end").i();
            passes = (int) lc.at("num_passes").i(), rank = (int) lc.at("gate_rank").i();
        }
        std::set<int> local;
        for (const auto& v : tc.at("local_layer_ids").a) local.insert((int) v.i());
        const int n_visit = n_layer + (passes - 1) * (core1 - core0);

        // metadata: llama.cpp master's keys
        const std::string a = "inkling2.";
        meta_["general.architecture"] = mv_str("inkling2");
        meta_["general.name"] = mv_str(std::string("Inkling-2-Mini-Strata-") + m.at("build").str());
        meta_[a + "block_count"] = mv_u32(n_visit);
        meta_[a + "context_length"] = mv_u32(gi("model_max_length"));
        meta_[a + "embedding_length"] = mv_u32(C_);
        meta_[a + "feed_forward_length"] = mv_u32(F_);
        meta_[a + "expert_feed_forward_length"] = mv_u32(F_);
        meta_[a + "attention.head_count"] = mv_u32(gi("num_attention_heads"));
        meta_[a + "attention.head_count_kv"] = mv_u32(gi("num_key_value_heads"));
        meta_[a + "attention.key_length"] = mv_u32(gi("head_dim"));
        meta_[a + "attention.value_length"] = mv_u32(gi("head_dim"));
        meta_[a + "expert_count"] = mv_u32(E_);
        meta_[a + "expert_used_count"] = mv_u32(gi("num_experts_per_tok"));
        meta_[a + "attention.sliding_window"] = mv_u32(gi("sliding_window_size"));
        meta_[a + "attention.relative_length"] = mv_u32(d_rel_);
        meta_[a + "attention.relative_extent"] = mv_u32(gi("rel_extent"));
        meta_[a + "attn_res.block_size"] = mv_u32(gi("n_layers_per_attn_res_block"));
        meta_[a + "vocab_size"] = mv_u32(gi("vocab_size"));
        meta_[a + "vocab_size_unpadded"] = mv_u32(gi("unpadded_vocab_size"));
        meta_[a + "attention.layer_norm_rms_epsilon"] = mv_f32(gf("rms_norm_eps"));
        meta_[a + "expert_weights_scale"] = mv_f32(gf("route_scale"));
        meta_[a + "attention.qk_clip"] = mv_f32(gf("qk_l2_clip_threshold"));
        meta_[a + "attention.relative_clip"] = mv_f32(gf("rel_l2_clip_threshold"));
        meta_[a + "attention.log_scale_alpha"] = mv_f32(gf("log_scaling_alpha"));
        meta_[a + "attention.log_scale_floor"] = mv_f32(gf("log_scaling_n_floor"));
        meta_[a + "activation.situ_beta"] = mv_f32(gf("mlp_gate_softcap"));
        meta_[a + "activation.situ_linear_beta"] = mv_f32(gf("mlp_up_softcap"));
        meta_[a + "final_logit_softcapping"] = mv_f32(gf("final_logit_softcapping"));
        meta_[a + "shortconv.l_cache"] = mv_u32(sconv_);
        if (passes > 1) {
            meta_[a + "num_loops"] = mv_u32(passes);
            meta_[a + "loop.core_start"] = mv_u32(core0);
            meta_[a + "loop.core_end"] = mv_u32(core1);
            meta_[a + "loop.gate_rank"] = mv_u32(rank);
        }
        {
            std::vector<strata::MetaValue> pat;
            const int nc = core1 - core0;
            for (int v = 0; v < n_visit; ++v) {
                const int l = (passes == 1 || v < core0) ? v
                            : v < core0 + passes * nc ? core0 + (v - core0) % nc : v - (passes - 1) * nc;
                strata::MetaValue b;
                b.type = strata::MetaType::BOOL;
                b.u = local.count(l) ? 1 : 0;
                pat.push_back(b);
            }
            meta_[a + "attention.sliding_window_pattern"] = mv_arr(pat, strata::MetaType::BOOL);
        }
        meta_["tokenizer.ggml.eos_token_id"] = mv_u32(m.at("config").at("eos_token_id").i());
        if (m.has("chat_template")) meta_["tokenizer.chat_template"] = mv_str(m.at("chat_template").str());

        // trunk
        struct M { std::string g, s; int tr; };
        std::vector<M> map = {
            {"token_embd.weight", "model.llm.embed.weight", 0},
            {"token_embd_norm.weight", "model.llm.embed_norm.weight", 0},
            {"output.weight", "model.llm.unembed.weight", 0},
            {"output_norm.weight", "model.llm.norm.weight", 0},
            {"output_res_query.weight", "model.llm.attn_res.output_query", 0},
        };
        if (passes > 1) {
            map.push_back({"loop_res_query.weight", "model.llm.middle_cycle_cross_pass_queries", 0});
            map.push_back({"loop_kv_res_query.weight", "model.llm.middle_cycle_cross_pass_kv_queries", 0});
        }
        for (int l = 0; l < n_layer; ++l) {
            const std::string b = "blk." + std::to_string(l) + ".", p = "model.llm.layers." + std::to_string(l) + ".";
            const char* pairs[][2] = {{"attn_q", "attn.wq_du"}, {"attn_k", "attn.wk_dv"}, {"attn_v", "attn.wv_dv"},
                                      {"attn_rel", "attn.wr_du"}, {"attn_sink_proj", "attn.attn_sink_proj"},
                                      {"attn_gate", "attn.gate_attn"}, {"attn_output", "attn.wo_ud"},
                                      {"ffn_gate_inp", "mlp.gate"}, {"ffn_norm", "mlp_norm"},
                                      {"attn_post_conv", "attn_sconv"}, {"ffn_post_conv", "mlp_sconv"}};
            for (const auto& pr : pairs) map.push_back({b + pr[0] + ".weight", p + pr[1] + ".weight", 0});
            map.push_back({b + "exp_probs_b.bias", p + "mlp.gate.bias", 0});
            map.push_back({b + "attn_q_clip.weight", p + "attn.q_l2_clip.scale", 0});
            map.push_back({b + "attn_rel_clip.weight", p + "attn.rel_l2_clip.scale", 0});
            map.push_back({b + "attn_rel_bias.weight", p + "attn.rel_logits_proj.proj", 1});
            if (local.count(l)) {
                map.push_back({b + "attn_norm.weight", p + "attn_norm.weight", 0});
                map.push_back({b + "attn_k_conv.weight", p + "attn.k_sconv.weight", 0});
                map.push_back({b + "attn_v_conv.weight", p + "attn.v_sconv.weight", 0});
            } else {
                for (const char* x : {"q", "k", "v"})
                    map.push_back({b + "attn_" + x + "_inp_norm.weight", p + x + "_input_norm.weight", 0});
                for (const char* x : {"k", "v"}) {
                    map.push_back({b + "attn_" + x + "_conv.weight", p + "attn." + x + "_sconv.static_w", 2});
                    map.push_back({b + "attn_" + x + "_conv_proj.weight", p + "attn." + x + "_sconv.weight_proj", 0});
                }
            }
            if (l > 0) map.push_back({b + "attn_res_query.weight", "model.llm.attn_res.attn_queries." + std::to_string(l - 1), 0});
            map.push_back({b + "ffn_res_query.weight", "model.llm.attn_res.mlp_queries." + std::to_string(l), 0});
            if (l >= core0 && l < core1)
                for (const char* md : {"attn", "k", "v", "mlp"})
                    for (const char* g : {"down", "gain", "forget"})
                        map.push_back({b + "loop_" + md + "_" + g + ".weight",
                                       p + "controllers." + md + "." + g + "_weight", 0});
        }
        for (const auto& x : map)
            if (!add_trunk(x.g, x.s, x.tr, err)) return false;

        // experts
        const auto& dir = pack_.experts();
        if ((int) dir.size() != n_layer * E_) return err = "expert directory: " + std::to_string(dir.size()) + " entries", false;
        layers_.resize(n_layer);
        for (int l = 0; l < n_layer; ++l) {
            ExpLayer& L = layers_[l];
            L.dir.assign(E_, -1);
            bool all_sf4 = true, all_mx8 = true;
            for (int r = 0; r < E_; ++r) {
                const PackExpert& e = dir[(size_t) l * E_ + r];
                if (e.expert_id >= E_ || L.dir[e.expert_id] >= 0) return err = "expert directory: bad expert id", false;
                L.dir[e.expert_id] = l * E_ + r;
                for (int code : {(int) e.gu_code, (int) e.dn_code})
                    if (code != kSF4 && code != kMX8 && !(code >= kS2_K && code <= kS6_K))
                        return err = "layer " + std::to_string(l) + ": " + code_name(code) + " experts are not read yet", false;
                if ((e.gu_code == kSF4 || e.gu_code == kMX8) != (e.dn_code == kSF4 || e.dn_code == kMX8))
                    return err = "an expert half SF4 / MX8, half S_K: not read", false;
                all_sf4 = all_sf4 && e.gu_code == kSF4 && e.dn_code == kSF4;
                all_mx8 = all_mx8 && e.gu_code == kMX8 && e.dn_code == kMX8;
                L.rb_gu = std::max(L.rb_gu, row_bytes(e.gu_code, C_));
                L.rb_dn = std::max(L.rb_dn, row_bytes(e.dn_code, F_));
            }
            if (all_sf4) {
                L.type = kNVFP4;
                L.gu_bytes = 2 * (size_t) F_ * row_bytes(kSF4, C_);
                L.dn_bytes = (size_t) C_ * row_bytes(kSF4, F_);
            } else if (all_mx8) {
                L.type = kMX8B;
                L.gu_bytes = 2 * (size_t) F_ * row_bytes(kMX8, C_);
                L.dn_bytes = (size_t) C_ * row_bytes(kMX8, F_);
            } else {
                L.type = kMIX;
                L.part_gu = align16(sizeof(MixHdr) + (size_t) F_ * L.rb_gu + 2 * (size_t) F_);
                L.gu_bytes = 2 * L.part_gu;
                L.dn_bytes = align16(sizeof(MixHdr) + (size_t) C_ * L.rb_dn);
            }
            const std::string b = "blk." + std::to_string(l) + ".";
            std::vector<uint8_t> sgu(E_ * 4), sdn(E_ * 4);
            for (int e = 0; e < E_; ++e) {
                const PackExpert& pe = dir[L.dir[e]];
                float g = 1.0f, d = 1.0f;
                if (pe.gu_code == kSF4) {
                    float h[4];
                    std::memcpy(h, pack_.body(pe), 16);
                    g = h[0] / h[1], d = h[2] / h[3];
                }
                std::memcpy(&sgu[e * 4], &g, 4);
                std::memcpy(&sdn[e * 4], &d, 4);
            }
            for (int role = 0; role < 2; ++role) {
                const std::string n = b + (role ? "ffn_down_exps.weight" : "ffn_gate_up_exps.weight");
                std::vector<uint64_t> ne = role ? std::vector<uint64_t>{(uint64_t) F_, (uint64_t) C_, (uint64_t) E_}
                                                : std::vector<uint64_t>{(uint64_t) C_, 2 * (uint64_t) F_, (uint64_t) E_};
                add(n, (uint32_t) L.type, ne, {});
                own_.back().ptr = nullptr;
                own_.back().nbytes = (size_t) E_ * (role ? L.dn_bytes : L.gu_bytes);
                own_.back().exp_layer = l, own_.back().exp_role = role;
            }
            add(b + "ffn_gate_up_exps.scale", 0, {(uint64_t) E_}, std::move(sgu));
            add(b + "ffn_down_exps.scale", 0, {(uint64_t) E_}, std::move(sdn));
        }
    } catch (const std::exception& e) {
        return err = path + ": " + e.what(), false;
    }
    return true;
}

const strata::TensorInfo* InklingView::info(const std::string& name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &tensors_[own_[it->second].idx];
}
const uint8_t* InklingView::data(const std::string& name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : own_[it->second].ptr;
}
size_t InklingView::bytes(const std::string& name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? 0 : own_[it->second].nbytes;
}
bool InklingView::is_experts(const std::string& name) const {
    auto it = by_name_.find(name);
    return it != by_name_.end() && own_[it->second].exp_layer >= 0;
}

bool InklingView::source(const std::string& name, std::string& pack_name, int& transform) const {
    auto it = by_name_.find(name);
    if (it == by_name_.end() || own_[it->second].src.empty()) return false;
    pack_name = own_[it->second].src, transform = own_[it->second].transform;
    return true;
}

const uint16_t* InklingView::macaq(int l, int e) const {
    const PackExpert& pe = pack_.experts()[layers_[l].dir[e]];
    return (pe.dn_code >= kS2_K && pe.dn_code <= kS6_K) ? (const uint16_t*) pack_.body(pe) : nullptr;
}

// expert e's role part (0 gate_up, 1 down) as the engine holds it
void InklingView::fill_expert(int l, int role, int e, uint8_t* dst) const {
    const ExpLayer& L = layers_[l];
    const PackExpert& pe = pack_.experts()[L.dir[e]];
    const uint8_t* body = pack_.body(pe);
    const int C = C_, F = F_;
    if (role == 0) {
        std::memset(dst, 0, L.gu_bytes);
        const int code = pe.gu_code;
        // Strata's gate_up rows interleave (2 j gate, 2 j + 1 up); ours: gate's F rows, then up's
        const size_t rb = row_bytes(code, C);
        uint8_t* rows[2];
        if (L.type == kMIX) {
            for (int h = 0; h < 2; ++h) {
                MixHdr hd{};
                hd.type = (uint8_t) (code == kSF4 ? kNVFP4 : code == kMX8 ? kMX8B : 200 + code);
                hd.rb = (uint32_t) rb;
                if (h == 1 && pe.dn_code >= kS2_K && pe.dn_code <= kS6_K) {
                    hd.aux_off = (uint32_t) (sizeof(MixHdr) + (size_t) F * L.rb_gu);
                    std::memcpy(dst + L.part_gu + hd.aux_off, body, 2 * (size_t) F);   // MACaQ s[F] FP16
                }
                std::memcpy(dst + h * L.part_gu, &hd, sizeof hd);
                rows[h] = dst + h * L.part_gu + sizeof(MixHdr);
            }
        } else {
            rows[0] = dst, rows[1] = dst + (size_t) F * rb;
        }
        if (code == kSF4) {
            const uint8_t* nib = body + 16;
            const uint8_t* sc = nib + (size_t) 2 * F * (C / 2);
            for (int r = 0; r < 2 * F; ++r)
                sf4_row_to_nvfp4(nib + (size_t) r * (C / 2), sc + (size_t) r * (C / 16), C, rows[r & 1] + (size_t) (r >> 1) * rb);
        } else if (code == kMX8) {
            const uint8_t* sc = body + (size_t) 2 * F * C;
            for (int r = 0; r < 2 * F; ++r)
                mx8_row_to_blocks(body + (size_t) r * C, sc + (size_t) r * (C / 32), C, rows[r & 1] + (size_t) (r >> 1) * rb);
        } else {
            const uint8_t* sk = body + 2 * (size_t) F;
            const size_t srow = (size_t) C / 256 * sk_sb_bytes(code);
            for (int r = 0; r < 2 * F; ++r) sk_run_to_sh(sk + (size_t) r * srow, code, (size_t) C, rows[r & 1] + (size_t) (r >> 1) * rb);
        }
    } else {
        std::memset(dst, 0, L.dn_bytes);
        const int code = pe.dn_code;
        const size_t rb = row_bytes(code, F);
        uint8_t* rows = dst;
        if (L.type == kMIX) {
            MixHdr hd{};
            hd.type = (uint8_t) (code == kSF4 ? kNVFP4 : code == kMX8 ? kMX8B : 200 + code);
            hd.rb = (uint32_t) rb;
            std::memcpy(dst, &hd, sizeof hd);
            rows = dst + sizeof(MixHdr);
        }
        const uint8_t* src = body + pe.dn_off;
        if (code == kSF4) {
            const uint8_t* sc = src + (size_t) C * (F / 2);
            for (int r = 0; r < C; ++r) sf4_row_to_nvfp4(src + (size_t) r * (F / 2), sc + (size_t) r * (F / 16), F, rows + (size_t) r * rb);
        } else if (code == kMX8) {
            const uint8_t* sc = src + (size_t) C * F;
            for (int r = 0; r < C; ++r) mx8_row_to_blocks(src + (size_t) r * F, sc + (size_t) r * (F / 32), F, rows + (size_t) r * rb);
        } else {
            sk_run_to_sh(src, code, (size_t) C * F, rows);   // (Strata's paired rows = the flat [C][F] run)
        }
    }
}

void InklingView::read_experts(const std::string& name, int e0, int n, size_t dst_stride, uint8_t* dst) const {
    auto it = by_name_.find(name);
    if (it == by_name_.end() || own_[it->second].exp_layer < 0) throw std::runtime_error(name + ": not an expert tensor");
    const Owned& o = own_[it->second];
    for (int i = 0; i < n; ++i) fill_expert(o.exp_layer, o.exp_role, e0 + i, dst + (size_t) i * dst_stride);
}

std::string InklingView::summary() const {
    std::map<std::string, int> codes;
    for (const auto& e : pack_.experts()) codes[std::string(code_name(e.gu_code)) + "+" + code_name(e.dn_code)]++;
    int mix = 0;
    for (const auto& L : layers_) mix += L.type == kMIX;
    std::string s = pack_.manifest().has("build") ? pack_.manifest().at("build").str() : "?";
    s += ": experts";
    for (const auto& kv : codes) s += " " + kv.first + " x" + std::to_string(kv.second);
    s += "; " + std::to_string(mix) + " of " + std::to_string(layers_.size()) + " layers mixed";
    return s;
}

}  // namespace strata_fmt
