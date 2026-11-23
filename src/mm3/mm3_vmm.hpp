// src/mm3/mm3_vmm.hpp - an address range on one GPU backed by granule-sized pieces of VRAM mapped from its start
// (CUDA's virtual memory API through the runtime's driver entry points): grow / shrink at its end, the address fixed.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if !defined(MAYA_USE_HIP)
#include <cuda.h>
#endif

namespace mm3::vmm {

#if !defined(MAYA_USE_HIP)
struct Api {
    CUresult (*dev_get)(CUdevice*, int) = nullptr;
    CUresult (*attr)(int*, CUdevice_attribute, CUdevice) = nullptr;
    CUresult (*gran)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) = nullptr;
    CUresult (*reserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
    CUresult (*addr_free)(CUdeviceptr, size_t) = nullptr;
    CUresult (*create)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) = nullptr;
    CUresult (*release)(CUmemGenericAllocationHandle) = nullptr;
    CUresult (*map)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) = nullptr;
    CUresult (*unmap)(CUdeviceptr, size_t) = nullptr;
    CUresult (*access)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) = nullptr;
    bool ok = false;
};

template <class F>
inline bool entry(const char* name, F& f) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
#if CUDART_VERSION >= 12050
    if (cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess)
        p = nullptr;
#else
    if (cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q) != cudaSuccess || q != cudaDriverEntryPointSuccess)
        p = nullptr;
#endif
    f = reinterpret_cast<F>(p);
    return p != nullptr;
}

inline const Api& api() {
    static const Api v = [] {
        Api m;
        m.ok = entry("cuDeviceGet", m.dev_get) && entry("cuDeviceGetAttribute", m.attr) &&
               entry("cuMemGetAllocationGranularity", m.gran) && entry("cuMemAddressReserve", m.reserve) &&
               entry("cuMemAddressFree", m.addr_free) && entry("cuMemCreate", m.create) &&
               entry("cuMemRelease", m.release) && entry("cuMemMap", m.map) && entry("cuMemUnmap", m.unmap) &&
               entry("cuMemSetAccess", m.access);
        cudaGetLastError();
        return m;
    }();
    return v;
}

inline CUmemAllocationProp prop_of(CUdevice cd) {
    CUmemAllocationProp p{};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = cd;
    return p;
}

// the GPU's granule (0: no virtual memory API here)
inline size_t granule(int dev) {
    const Api& v = api();
    CUdevice cd = 0;
    int vm = 0;
    if (!v.ok || v.dev_get(&cd, dev) != CUDA_SUCCESS ||
        v.attr(&vm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cd) != CUDA_SUCCESS || vm == 0)
        return 0;
    const CUmemAllocationProp p = prop_of(cd);
    size_t g = 0;
    if (v.gran(&g, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS) return 0;
    return g;
}

// this GPU and every GPU that can reach it (as cudaMalloc's memory with peer access on)
inline std::vector<CUmemAccessDesc> access_list(int dev) {
    std::vector<CUmemAccessDesc> d;
    int n = 0;
    cudaGetDeviceCount(&n);
    for (int i = 0; i < n; ++i) {
        int can = i == dev ? 1 : 0;
        if (i != dev && cudaDeviceCanAccessPeer(&can, i, dev) != cudaSuccess) can = 0;
        if (!can) continue;
        CUdevice cd = 0;
        if (api().dev_get(&cd, i) != CUDA_SUCCESS) continue;
        CUmemAccessDesc a{};
        a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        a.location.id = cd;
        a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        d.push_back(a);
    }
    cudaGetLastError();
    return d;
}

struct Range {
    int dev = -1;
    size_t gran = 0, cap = 0, mapped = 0;   // bytes: a granule, reserved, mapped from the start
    CUdeviceptr base = 0;
    std::vector<CUmemAccessDesc> acc;

    uint8_t* ptr() const { return (uint8_t*) (uintptr_t) base; }
    size_t round(size_t b) const { return (b + gran - 1) / gran * gran; }

    bool open(int d, size_t bytes, std::string& err) {
        dev = d;
        gran = granule(d);
        if (gran == 0) {
            err = "CUDA" + std::to_string(d) + ": no virtual memory management";
            return false;
        }
        cap = round(std::max<size_t>(bytes, 1));
        if (api().reserve(&base, cap, 0, 0, 0) != CUDA_SUCCESS) {
            err = "CUDA" + std::to_string(d) + ": an address range of " + std::to_string(cap >> 20) + " MB not reserved";
            base = 0;
            return false;
        }
        acc = access_list(d);
        return true;
    }
    // map up to `bytes` (rounded to granules, at most cap); false: VRAM short (what fit stays mapped)
    bool grow(size_t bytes, std::string& err) {
        const Api& v = api();
        bytes = std::min(round(bytes), cap);
        CUdevice cd = 0;
        if (mapped >= bytes) return true;
        if (v.dev_get(&cd, dev) != CUDA_SUCCESS) {
            err = "cuDeviceGet failed";
            return false;
        }
        const CUmemAllocationProp p = prop_of(cd);
        const size_t from = mapped;
        bool ok = true;
        while (mapped < bytes) {
            CUmemGenericAllocationHandle h{};
            if (v.create(&h, gran, &p, 0) != CUDA_SUCCESS) {
                err = "CUDA" + std::to_string(dev) + ": out of device memory";
                ok = false;
                break;
            }
            const bool m = v.map(base + mapped, gran, 0, h, 0) == CUDA_SUCCESS;
            v.release(h);   // (the mapping holds the memory: unmapping gives it back)
            if (!m) {
                err = "cuMemMap failed";
                ok = false;
                break;
            }
            mapped += gran;
        }
        if (mapped > from && v.access(base + from, mapped - from, acc.data(), acc.size()) != CUDA_SUCCESS) {
            for (size_t o = from; o < mapped; o += gran) v.unmap(base + o, gran);
            mapped = from;
            err = "cuMemSetAccess failed";
            ok = false;
        }
        cudaGetLastError();
        return ok;
    }
    // unmap past `bytes` (rounded up to granules) - the GPU must not use that part any more
    void shrink(size_t bytes) {
        bytes = std::min(round(bytes), cap);
        while (mapped > bytes) {
            mapped -= gran;
            api().unmap(base + mapped, gran);
        }
    }
    void close() {
        if (base == 0) return;
        shrink(0);
        api().addr_free(base, cap);
        base = 0;
        cap = 0;
    }
};
#else
inline size_t granule(int) { return 0; }
struct Range {
    int dev = -1;
    size_t gran = 0, cap = 0, mapped = 0;
    uintptr_t base = 0;
    uint8_t* ptr() const { return nullptr; }
    size_t round(size_t b) const { return b; }
    bool open(int, size_t, std::string& err) {
        err = "a HIP build: no virtual memory ranges";
        return false;
    }
    bool grow(size_t, std::string& err) {
        err = "a HIP build";
        return false;
    }
    void shrink(size_t) {}
    void close() {}
};
#endif

}  // namespace mm3::vmm
