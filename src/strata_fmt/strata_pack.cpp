// src/strata_fmt/strata_pack.cpp - see strata_pack.hpp
#include "strata_pack.hpp"

#include <cstdio>
#include <cstring>

#ifdef SF_HAVE_ZSTD
#include <zstd.h>
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata_fmt {

namespace {
constexpr uint8_t kMagic[8] = {'S', 'T', 'R', 'A', 'T', 'A', 0x00, 0x01};
constexpr uint64_t kPage = 4096;

bool read_tensors(const JVal* arr, std::vector<PackTensor>& out, std::string& err) {
    if (!arr) return true;
    for (size_t i = 0; i < arr->a.size(); ++i) {
        const JVal& t = arr->a[i];
        PackTensor p;
        p.name = t.at("name").str();
        p.fmt = t.at("format").str();
        for (const auto& d : t.at("shape").a) p.shape.push_back(d.i());
        p.off = (uint64_t) t.at("offset").i();
        p.bytes = (uint64_t) t.at("bytes").i();
        out.push_back(std::move(p));
    }
    (void) err;
    return true;
}
}  // namespace

const char* code_name(int c) {
    switch (c) {
    case kS2_K: return "S2_K";
    case kS3_K: return "S3_K";
    case kS4_K: return "S4_K";
    case kS5_K: return "S5_K";
    case kS6_K: return "S6_K";
    case kS8_K: return "S8_K";
    case kSF4: return "SF4";
    case kMX8: return "MX8";
    default: return "?";
    }
}

bool Pack::sniff(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint8_t m[8] = {};
    const bool ok = std::fread(m, 1, 8, f) == 8 && std::memcmp(m, kMagic, 8) == 0;
    std::fclose(f);
    return ok;
}

Pack::~Pack() {
#ifdef _WIN32
    if (base_) UnmapViewOfFile(base_);
    if (map_) CloseHandle((HANDLE) map_);
    if (fh_) CloseHandle((HANDLE) fh_);
#else
    if (base_) munmap((void*) base_, size_);
#endif
}

bool Pack::open(const std::string& path, std::string& err) {
    path_ = path;
#ifdef _WIN32
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) return err = "cannot open " + path, false;
    fh_ = fh;
    LARGE_INTEGER sz;
    GetFileSizeEx(fh, &sz);
    size_ = (uint64_t) sz.QuadPart;
    HANDLE mp = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mp) return err = "cannot map " + path, false;
    map_ = mp;
    base_ = (const uint8_t*) MapViewOfFile(mp, FILE_MAP_READ, 0, 0, 0);
    if (!base_) return err = "cannot map " + path, false;
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return err = "cannot open " + path, false;
    struct stat st{};
    fstat(fd, &st);
    size_ = (uint64_t) st.st_size;
    void* p = size_ >= kPage ? mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
    ::close(fd);
    if (p == MAP_FAILED) return err = "cannot map " + path, false;
    base_ = (const uint8_t*) p;
#endif
    if (size_ < kPage || std::memcmp(base_, kMagic, 8) != 0) return err = path + ": not a .strata pack", false;
    std::memcpy(&flags_, base_ + 8, 4);
    uint16_t nsec = 0;
    std::memcpy(&nsec, base_ + 14, 2);
    if (nsec != 6) return err = path + ": " + std::to_string(nsec) + " sections (6 expected)", false;
    uint32_t s3_flags = 0;
    for (int i = 0; i < 6; ++i) {
        const uint8_t* e = base_ + 0x10 + 64 * i;
        std::memcpy(&sec_off_[i], e, 8);
        std::memcpy(&sec_len_[i], e + 8, 8);
        if (i == 3) std::memcpy(&s3_flags, e + 52, 4);
        if (sec_len_[i] && (sec_off_[i] % kPage || sec_off_[i] + sec_len_[i] > size_))
            return err = path + ": section " + std::to_string(i) + " out of place (a cut download?)", false;
    }
    // S0: the manifest
    std::string js;
    {
        const uint8_t* z = base_ + sec_off_[0];
        const size_t zn = (size_t) sec_len_[0];
#ifdef SF_HAVE_ZSTD
        const unsigned long long n = ZSTD_getFrameContentSize(z, zn);
        if (n == ZSTD_CONTENTSIZE_ERROR || n == ZSTD_CONTENTSIZE_UNKNOWN) return err = path + ": manifest not zstd", false;
        js.resize((size_t) n);
        const size_t r = ZSTD_decompress(&js[0], js.size(), z, zn);
        if (ZSTD_isError(r)) return err = path + ": manifest: " + ZSTD_getErrorName(r), false;
        js.resize(r);
#else
        (void) z, (void) zn;
        return err = "this build reads .strata packs without zstd: rebuild with libzstd", false;
#endif
    }
    try {
        manifest_ = json_parse(js);
        read_tensors(manifest_.get("trunk_tensors"), trunk_, err);
        read_tensors(manifest_.get("dspark_tensors"), dspark_, err);
    } catch (const std::exception& e) {
        return err = path + ": manifest: " + e.what(), false;
    }
    for (const auto& t : trunk_)
        if (t.off + t.bytes > sec_len_[1]) return err = t.name + " outside the trunk section", false;
    // S3: the directory (v1.1: then a u16 VRAM order we do not use)
    const size_t ent = (s3_flags & 1) ? 34 : 32;
    const size_t n = (size_t) (sec_len_[3] / ent);
    experts_.resize(n);
    if (n) std::memcpy(experts_.data(), base_ + sec_off_[3], n * 32);
    for (const auto& e : experts_)
        if (e.body_off + e.body_bytes > sec_len_[4]) return err = "an expert body outside its section", false;
    return true;
}

const PackTensor* Pack::find(const std::string& name) const {
    for (const auto& t : trunk_)
        if (t.name == name) return &t;
    return nullptr;
}

}  // namespace strata_fmt
