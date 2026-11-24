// src/ds4/ds4_elastic.hpp - DS4_ELASTIC_CONV=1 (--parallel): conversations 1..n-1's compressed KV and index keys are
// virtual reservations of their whole context, backed from their start a granule at a time as the conversation grows,
// out of a pool of granules made at load (before the experts take the rest of VRAM).  Conversation 0 keeps its whole
// context in plain memory.  The driver's virtual memory calls come from the runtime (no link to libcuda), as GLM's.
#pragma once

#include <cuda_runtime.h>
#if !defined(MAYA_USE_HIP)
#include <cuda.h>
#endif

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace maya::ds4::elastic {

#if !defined(MAYA_USE_HIP)
struct Vmm {
    decltype(&::cuMemAddressReserve) reserve = nullptr;
    decltype(&::cuMemAddressFree) addr_free = nullptr;
    decltype(&::cuMemCreate) create = nullptr;
    decltype(&::cuMemRelease) release = nullptr;
    decltype(&::cuMemMap) map = nullptr;
    decltype(&::cuMemUnmap) unmap = nullptr;
    decltype(&::cuMemSetAccess) access = nullptr;
    decltype(&::cuMemGetAllocationGranularity) granularity = nullptr;
    decltype(&::cuDeviceGetAttribute) attribute = nullptr;
    bool ok = false;
};
inline const Vmm& vmm() {
    static const Vmm v = [] {
        Vmm r;
        const auto get = [](const char* name, void** fn) {
            cudaDriverEntryPointQueryResult q{};
#if CUDART_VERSION >= 12050
            return cudaGetDriverEntryPointByVersion(name, fn, 12000, cudaEnableDefault, &q) == cudaSuccess &&
                   q == cudaDriverEntryPointSuccess && *fn != nullptr;
#else
            return cudaGetDriverEntryPoint(name, fn, cudaEnableDefault, &q) == cudaSuccess &&
                   q == cudaDriverEntryPointSuccess && *fn != nullptr;
#endif
        };
        r.ok = get("cuMemAddressReserve", (void**) &r.reserve) && get("cuMemAddressFree", (void**) &r.addr_free) &&
               get("cuMemCreate", (void**) &r.create) && get("cuMemRelease", (void**) &r.release) &&
               get("cuMemMap", (void**) &r.map) && get("cuMemUnmap", (void**) &r.unmap) &&
               get("cuMemSetAccess", (void**) &r.access) &&
               get("cuMemGetAllocationGranularity", (void**) &r.granularity) &&
               get("cuDeviceGetAttribute", (void**) &r.attribute);
        return r;
    }();
    return v;
}
inline CUmemAllocationProp prop(int dev) {
    CUmemAllocationProp p{};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = dev;
    return p;
}
inline size_t granule(int dev) {
    const Vmm& v = vmm();
    int sup = 0;
    if (!v.ok || v.attribute(&sup, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev) != CUDA_SUCCESS || !sup)
        return 0;
    const CUmemAllocationProp p = prop(dev);
    size_t g = 0;
    return v.granularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS ? g : 0;
}

struct Pool {
    struct Buf {   // one conversation's rows of one source on one device: rows [0, rows(tokens)) backed
        int di = 0;
        CUdeviceptr va = 0;
        size_t cap = 0, rb = 0;   // its reservation's bytes; a row's
        int64_t ratio = 1, max_rows = 0;
        std::vector<CUmemGenericAllocationHandle> h;   // granule i -> its handle
        size_t bytes_for(int64_t tokens) const {
            const int64_t rows = std::min<int64_t>(max_rows, tokens / ratio + 1);
            return (size_t) rows * rb;
        }
    };
    struct Seq {
        std::vector<Buf> b;
        int64_t tokens = 0;   // what every buffer backs
    };
    std::vector<int> ids;                                        // [device index] the CUDA device
    std::vector<std::vector<CUmemGenericAllocationHandle>> idle;  // [device index] granules no buffer holds
    std::vector<size_t> made;                                    // [device index] granules made at load
    std::vector<size_t> taken_in;                                // [device index] tail granules given to it (DS4_ELASTIC_TAKE)
    std::vector<Seq> seqs;                                       // [conversation] (0: none - its own memory)
    size_t gran = 0;

    // the devices' common granule (0: no virtual memory on one of them - the caller allocates as before)
    bool init(const std::vector<int>& dev_ids, int n_seqs) {
        ids = dev_ids;
        for (int id : ids) {
            const size_t g = granule(id);
            if (g == 0) return false;
            gran = std::max(gran, g);
        }
        idle.assign(ids.size(), {});
        made.assign(ids.size(), 0);
        taken_in.assign(ids.size(), 0);
        seqs.assign((size_t) n_seqs, Seq{});
        return gran > 0;
    }
    size_t up(size_t b) const { return (b + gran - 1) / gran * gran; }
    // conversation s's buffer for `max_rows` rows of rb bytes (one row every `ratio` positions) on device di: its
    // address, nothing backed yet (0: no address space)
    uint8_t* add(int s, int di, size_t rb, int64_t ratio, int64_t max_rows) {
        Buf b;
        b.di = di;
        b.rb = rb;
        b.ratio = std::max<int64_t>(1, ratio);
        b.max_rows = max_rows;
        b.cap = up((size_t) max_rows * rb);
        if (vmm().reserve(&b.va, b.cap, gran, 0, 0) != CUDA_SUCCESS) return nullptr;
        seqs[(size_t) s].b.push_back(std::move(b));
        return (uint8_t*) seqs[(size_t) s].b.back().va;
    }
    // device di's pool: `bytes` more, in granules (false: the device has no more memory)
    bool fill(int di, size_t bytes) {
        const CUmemAllocationProp p = prop(ids[(size_t) di]);
        for (size_t n = up(bytes) / gran; n > 0; --n) {
            CUmemGenericAllocationHandle h = 0;
            if (vmm().create(&h, gran, &p, 0) != CUDA_SUCCESS) return false;
            idle[(size_t) di].push_back(h);
            ++made[(size_t) di];
        }
        return true;
    }
    // granules device di still lacks for conversation s to back `tokens`
    size_t lack(int s, int di, int64_t tokens) const {
        size_t n = 0;
        for (const Buf& b : seqs[(size_t) s].b)
            if (b.di == di) {
                const size_t want = std::min(b.cap, up(b.bytes_for(tokens))) / gran;
                if (want > b.h.size()) n += want - b.h.size();
            }
        return n;
    }
    bool can(int s, int64_t tokens) const {
        for (size_t di = 0; di < ids.size(); ++di)
            if (lack(s, (int) di, tokens) > idle[di].size()) return false;
        return true;
    }
    // conversation s backed to `tokens` positions, its new granules zeroed (false: a device's pool is short or a call
    // failed - what was mapped stays, a later grow or release takes it as it is)
    bool grow(int s, int64_t tokens, size_t& added) {
        added = 0;
        Seq& q = seqs[(size_t) s];
        if (tokens <= q.tokens) return true;
        if (!can(s, tokens)) return false;
        const Vmm& v = vmm();
        for (Buf& b : q.b) {
            const size_t want = std::min(b.cap, up(b.bytes_for(tokens))) / gran;
            const CUmemAllocationProp p = prop(ids[(size_t) b.di]);
            CUmemAccessDesc ad{};
            ad.location = p.location;
            ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            if (b.h.size() < want && cudaSetDevice(ids[(size_t) b.di]) != cudaSuccess) return false;
            while (b.h.size() < want) {
                auto& pool = idle[(size_t) b.di];
                const CUmemGenericAllocationHandle h = pool.back();
                const CUdeviceptr at = b.va + b.h.size() * gran;
                if (v.map(at, gran, 0, h, 0) != CUDA_SUCCESS) return false;
                if (v.access(at, gran, &ad, 1) != CUDA_SUCCESS ||
                    cudaMemsetAsync((void*) at, 0, gran, 0) != cudaSuccess) {
                    v.unmap(at, gran);
                    return false;
                }
                pool.pop_back();
                b.h.push_back(h);
                added += gran;
            }
            if (cudaStreamSynchronize(0) != cudaSuccess) return false;
        }
        q.tokens = tokens;
        return true;
    }
    // every granule of conversation s back to its pool (the caller has nothing in flight that reads it)
    size_t release(int s) {
        Seq& q = seqs[(size_t) s];
        size_t freed = 0;
        for (Buf& b : q.b) {
            while (!b.h.empty()) {
                vmm().unmap(b.va + (b.h.size() - 1) * gran, gran);
                idle[(size_t) b.di].push_back(b.h.back());
                b.h.pop_back();
                freed += gran;
            }
        }
        q.tokens = 0;
        return freed;
    }
    size_t held(int s) const {
        size_t n = 0;
        for (const Buf& b : seqs[(size_t) s].b) n += b.h.size() * gran;
        return n;
    }
    size_t idle_bytes() const {
        size_t n = 0;
        for (const auto& p : idle) n += p.size() * gran;
        return n;
    }
    size_t made_bytes() const {
        size_t n = 0;
        for (size_t m : made) n += m * gran;
        return n;
    }
    // what the pool owns now: its own granules and the tails' given to it; what the conversations hold of it
    size_t total_bytes() const {
        size_t n = made_bytes();
        for (size_t m : taken_in) n += m * gran;
        return n;
    }
    size_t held_dev(int di) const {
        size_t n = 0;
        for (const Seq& q : seqs)
            for (const Buf& b : q.b)
                if (b.di == di) n += b.h.size();
        return n;
    }
    // the books against the granules (empty: they agree): per device, owned = idle + held by the conversations; and
    // every handle in one place only - a tail's mapped granules, a conversation's, the free list
    std::string audit() const {
        std::string err;
        std::vector<CUmemGenericAllocationHandle> all;
        for (size_t di = 0; di < ids.size(); ++di) {
            const size_t own = made[di] + taken_in[di], held = held_dev((int) di);
            if (idle[di].size() + held != own)
                err += " CUDA" + std::to_string(ids[di]) + ": owns " + std::to_string(own) + " granules, " +
                       std::to_string(idle[di].size()) + " free + " + std::to_string(held) + " held;";
            all.insert(all.end(), idle[di].begin(), idle[di].end());
        }
        for (const Seq& q : seqs)
            for (const Buf& b : q.b) all.insert(all.end(), b.h.begin(), b.h.end());
        for (const Tail& t : tails)
            for (CUmemGenericAllocationHandle h : t.h)
                if (h != 0) all.push_back(h);
        std::sort(all.begin(), all.end());
        for (size_t i = 1; i < all.size(); ++i)
            if (all[i] == all[i - 1]) {
                err += " a granule in two places;";
                break;
            }
        return err;
    }
    // DS4_ELASTIC_TAKE: a card's expert tail in granules of its own (Tail::h[i]: 0 while given to the conversations)
    struct Tail {
        CUdeviceptr va = 0;
        size_t bytes = 0;
        std::vector<CUmemGenericAllocationHandle> h;
    };
    std::vector<Tail> tails;   // [device index] (va 0: a plain allocation, nothing taken)
    // device di's tail: `bytes` of address space, every granule backed and zeroed (null: no memory / no address space)
    uint8_t* tail_alloc(int di, size_t bytes) {
        if (tails.size() < ids.size()) tails.resize(ids.size());
        Tail& t = tails[(size_t) di];
        const Vmm& v = vmm();
        t.bytes = up(bytes);
        if (v.reserve(&t.va, t.bytes, gran, 0, 0) != CUDA_SUCCESS) {
            t.va = 0;
            return nullptr;
        }
        t.h.assign(t.bytes / gran, 0);
        const CUmemAllocationProp p = prop(ids[(size_t) di]);
        for (size_t i = 0; i < t.h.size(); ++i) {
            CUmemGenericAllocationHandle h = 0;
            if (v.create(&h, gran, &p, 0) != CUDA_SUCCESS || !map_at(di, t.va + i * gran, h)) {
                if (h != 0) v.release(h);
                return nullptr;   // (the destructor frees what was made)
            }
            t.h[i] = h;
        }
        if (cudaSetDevice(ids[(size_t) di]) != cudaSuccess || cudaMemset((void*) t.va, 0, t.bytes) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess)
            return nullptr;
        return (uint8_t*) t.va;
    }
    bool map_at(int di, CUdeviceptr at, CUmemGenericAllocationHandle h) const {
        const Vmm& v = vmm();
        const CUmemAllocationProp p = prop(ids[(size_t) di]);
        CUmemAccessDesc ad{};
        ad.location = p.location;
        ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (v.map(at, gran, 0, h, 0) != CUDA_SUCCESS) return false;
        if (v.access(at, gran, &ad, 1) != CUDA_SUCCESS) {
            v.unmap(at, gran);
            return false;
        }
        return true;
    }
    // tail granule i of device di to the conversations' pool (the caller: nothing in flight reads it)
    bool tail_give(int di, size_t i) {
        Tail& t = tails[(size_t) di];
        if (t.h[i] == 0 || vmm().unmap(t.va + i * gran, gran) != CUDA_SUCCESS) return false;
        idle[(size_t) di].push_back(t.h[i]);
        t.h[i] = 0;
        ++taken_in[(size_t) di];
        return true;
    }
    // a free granule of the pool back into tail granule i (its bytes as they were left: the caller copies them in)
    bool tail_take_back(int di, size_t i) {
        Tail& t = tails[(size_t) di];
        auto& pool = idle[(size_t) di];
        if (t.h[i] != 0 || pool.empty() || !map_at(di, t.va + i * gran, pool.back())) return false;
        t.h[i] = pool.back();
        pool.pop_back();
        --taken_in[(size_t) di];
        return true;
    }
    ~Pool() {
        const Vmm& v = vmm();
        if (!v.ok) return;
        for (Tail& t : tails) {
            for (size_t i = 0; i < t.h.size(); ++i)
                if (t.h[i] != 0) {
                    v.unmap(t.va + i * gran, gran);
                    v.release(t.h[i]);
                }
            if (t.va != 0) v.addr_free(t.va, t.bytes);
        }
        for (size_t s = 0; s < seqs.size(); ++s) release((int) s);
        for (Seq& q : seqs)
            for (Buf& b : q.b)
                if (b.va != 0) v.addr_free(b.va, b.cap);
        for (auto& p : idle)
            for (CUmemGenericAllocationHandle h : p) v.release(h);
    }
};
#else
struct Pool {   // (HIP: no virtual memory calls here - init refuses, every conversation's KV allocated whole)
    // (the members the CUDA pool's callers name, empty: init refuses, so none is ever used on HIP)
    struct Buf {
        int di = 0;
        uintptr_t va = 0;
        size_t cap = 0;
        std::vector<uint64_t> h;
        size_t rb = 0;
        int64_t ratio = 1, max_rows = 0;
        size_t bytes_for(int64_t) const { return 0; }
    };
    struct Seq {
        std::vector<Buf> b;
        int64_t tokens = 0;
    };
    std::vector<size_t> made, taken_in;
    std::vector<Seq> seqs;
    size_t gran = 0;
    bool init(const std::vector<int>&, int) { return false; }
    size_t total_bytes() const { return 0; }
    std::string audit() const { return ""; }
    size_t up(size_t b) const { return b; }
    uint8_t* add(int, int, size_t, int64_t, int64_t) { return nullptr; }
    bool fill(int, size_t) { return false; }
    bool can(int, int64_t) const { return false; }
    bool grow(int, int64_t, size_t& added) { return (added = 0) != 0; }
    size_t release(int) { return 0; }
    size_t held(int) const { return 0; }
    size_t held_dev(int) const { return 0; }
    size_t lack(int, int, int64_t) const { return 0; }
    size_t idle_bytes() const { return 0; }
    size_t made_bytes() const { return 0; }
    struct Tail {
        uintptr_t va = 0;
        size_t bytes = 0;
        std::vector<uint64_t> h;
    };
    std::vector<Tail> tails;
    std::vector<std::vector<int>> idle;
    uint8_t* tail_alloc(int, size_t) { return nullptr; }
    bool tail_give(int, size_t) { return false; }
    bool tail_take_back(int, size_t) { return false; }
};
#endif

}  // namespace maya::ds4::elastic
