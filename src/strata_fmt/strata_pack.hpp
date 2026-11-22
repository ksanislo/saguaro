// src/strata_fmt/strata_pack.hpp - Strata .strata v1.x pack reader (mmap; model-independent: Inkling, Hy4 SQ packs).
// Container layout from Strata engine/src/artifact/strata_reader.cpp (MIT License - Copyright (c) 2026 Strata
// contributors): a 4 KiB header (magic, flags, profile, 6 x 64 B section entries), S0 zstd JSON manifest, S1 trunk
// arena, S2 engram seed cache, S3 expert directory (+ v1.1 u16 VRAM load order), S4 expert bodies, S5 self-check.
#pragma once

#include "sf_json.h"

#include <cstdint>
#include <string>
#include <vector>

namespace strata_fmt {

struct PackTensor {
    std::string name, fmt;       // fmt: F32 BF16 F16 Q8_ROW S2_K..S8_K
    std::vector<int64_t> shape;  // row-major, slowest first
    uint64_t off = 0, bytes = 0; // within S1
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

#pragma pack(push, 1)
struct PackExpert {
    uint16_t expert_id;
    uint8_t gu_code, dn_code;
    uint16_t cluster_id, flags;
    uint64_t body_off;   // within S4
    uint32_t body_bytes, dn_off;
    uint32_t aux[2];
};
#pragma pack(pop)
static_assert(sizeof(PackExpert) == 32, "PackExpert");

// Strata's codec ids (sk_codecs.hpp SkCode)
enum PackCode : int { kS2_K = 2, kS3_K = 3, kS4_K = 4, kS5_K = 5, kS6_K = 6, kS8_K = 8, kSF4 = 15, kMX8 = 16 };
const char* code_name(int code);

class Pack {
public:
    Pack() = default;
    ~Pack();
    Pack(const Pack&) = delete;
    Pack& operator=(const Pack&) = delete;

    bool open(const std::string& path, std::string& err);
    static bool sniff(const std::string& path);   // the file starts with the .strata magic

    const std::string& path() const { return path_; }
    uint32_t flags() const { return flags_; }
    const JVal& manifest() const { return manifest_; }
    const std::vector<PackTensor>& trunk() const { return trunk_; }
    const std::vector<PackTensor>& dspark() const { return dspark_; }
    const std::vector<PackExpert>& experts() const { return experts_; }
    const PackTensor* find(const std::string& name) const;
    const uint8_t* s1() const { return base_ + sec_off_[1]; }
    const uint8_t* tensor(const PackTensor& t) const { return s1() + t.off; }
    const uint8_t* body(const PackExpert& e) const { return base_ + sec_off_[4] + e.body_off; }
    uint64_t file_size() const { return size_; }

private:
    std::string path_;
    const uint8_t* base_ = nullptr;
    uint64_t size_ = 0;
    uint32_t flags_ = 0;
    uint64_t sec_off_[6] = {}, sec_len_[6] = {};
    JVal manifest_;
    std::vector<PackTensor> trunk_, dspark_;
    std::vector<PackExpert> experts_;
#ifdef _WIN32
    void* fh_ = nullptr;
    void* map_ = nullptr;
#endif
};

}  // namespace strata_fmt
