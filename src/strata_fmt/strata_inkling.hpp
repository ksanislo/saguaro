// src/strata_fmt/strata_inkling.hpp - an Inkling 2 Mini Strata pack seen as the llama.cpp-master GGUF saguaro-ik2 reads:
// the same tensor names and metadata keys, the values unchanged (lossless re-layouts into sf_codecs.h's types).
//   trunk: F32 / BF16 as they are, Q8_ROW -> kQ8F, S_K -> kSH*; Strata's HF names mapped back as in Strata's
//          ink_gguf_loader.h (permutations only).
//   experts: a layer whose experts are all SF4 -> NVFP4 + .scale (the bytes of unsloth's NVFP4 GGUFs); any other mix
//          -> kMIX (each role part: MixHdr, rows of the expert's own type, the MACaQ scales after the up part).
#pragma once

#include "maya/artifact/gguf_reader.hpp"
#include "strata_pack.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata_fmt {

class InklingView {
public:
    bool build(const std::string& path, std::string& err);

    const Pack& pack() const { return pack_; }
    const maya::MetaValue* get(const std::string& key) const {
        auto it = meta_.find(key);
        return it == meta_.end() ? nullptr : &it->second;
    }
    const std::vector<maya::TensorInfo>& tensors() const { return tensors_; }
    const maya::TensorInfo* info(const std::string& name) const;
    // the host bytes of a trunk tensor (null for an expert tensor: read_experts)
    const uint8_t* data(const std::string& name) const;
    size_t bytes(const std::string& name) const;   // any tensor's (an expert tensor: n_experts x its expert's bytes)
    bool is_experts(const std::string& name) const;
    // experts e0 .. e0 + n - 1 of an expert tensor (blk.L.ffn_{gate_up,down}_exps.weight), dst_stride apart
    void read_experts(const std::string& name, int e0, int n, size_t dst_stride, uint8_t* dst) const;
    std::string summary() const;

    // for tests: the MACaQ scales s[F] of expert e of layer l (null: none); a trunk tensor's pack name and reorder
    // (0 none, 1 [d_rel][ext] transposed, 2 conv static_w [k][KV] lag 0.. -> [KV][k] lag k-1..)
    const uint16_t* macaq(int l, int e) const;
    const PackExpert& expert(int l, int e) const { return pack_.experts()[layers_[l].dir[e]]; }
    int layer_type(int l) const { return layers_[l].type; }
    // expert e of layer l as the engine holds it: its bytes (a kMIX expert packed: under its layer's slot), and a kMIX
    // expert's up / down part offsets
    size_t expert_bytes(int l, int e) const;
    void mix_layout(int l, int e, size_t& up, size_t& dn, size_t& bytes) const;
    size_t compact_bytes() const {
        size_t t = 0;
        for (size_t l = 0; l < layers_.size(); ++l)
            for (int e = 0; e < E_; ++e) t += expert_bytes((int) l, e);
        return t;
    }
    bool source(const std::string& name, std::string& pack_name, int& transform) const;

private:
    struct ExpLayer {
        int type = 0;            // kNVFP4 or kMIX
        size_t gu_bytes = 0;     // an expert's gate_up part (both halves) / down part
        size_t dn_bytes = 0;
        size_t rb_gu = 0, rb_dn = 0, part_gu = 0;   // kMIX: the widest row of gate/up and of down; a gate (= up) part
        std::vector<int> dir;    // directory index of expert e
    };
    struct Owned {
        size_t idx;
        std::vector<uint8_t> buf;
        const uint8_t* ptr = nullptr;
        size_t nbytes = 0;
        int exp_layer = -1, exp_role = -1;   // an expert tensor: its layer, 0 gate_up / 1 down
        std::string src;
        int transform = 0;
    };
    void add(const std::string& name, uint32_t type, std::vector<uint64_t> ne, std::vector<uint8_t> buf,
             const uint8_t* ptr = nullptr, size_t nbytes = 0);
    bool add_trunk(const std::string& gguf, const std::string& strata, int transform, std::string& err);
    void fill_expert(int l, int role, int e, uint8_t* dst) const;

    Pack pack_;
    std::map<std::string, maya::MetaValue> meta_;
    std::vector<maya::TensorInfo> tensors_;
    std::vector<Owned> own_;
    std::map<std::string, size_t> by_name_;
    std::vector<ExpLayer> layers_;
    int C_ = 0, F_ = 0, E_ = 0, KV_ = 0, d_rel_ = 0, sconv_ = 4;
};

}  // namespace strata_fmt
