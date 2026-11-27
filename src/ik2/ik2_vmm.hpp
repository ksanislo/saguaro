// src/ik2/ik2_vmm.hpp - the elastic K / V of the global layers as an address range with physical VRAM behind a
// growing prefix of it (P125o a2 / a3, src/ik2/ik2_plan.hpp): the whole --max-context range is reserved once at the
// load, granule-sized pieces of VRAM stand behind the committed positions, and a request commits more of the range
// (in position order, between its passes - never inside a kernel) before it writes past what is committed.  The
// base pointer never moves, so the kernels' row addressing and the recorded CUDA graphs stay valid.
// CUDA: the driver's virtual memory API through the runtime's entry points (src/mm3/mm3_vmm.hpp's pattern, the
// fresh physical pages map zeroed - the DEPTH test's semantics).  AMD is first-class too: hipMemAddressReserve /
// hipMemCreate / hipMemMap / hipMemSetAccess where this ROCm ships them (SAGUARO_IK2_HIP_VMM, checked at configure
// time); where the HIP build has no piece of it, open() says so and the engine keeps the plain allocation it makes
// today (one stderr line at the load).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if !defined(MAYA_USE_HIP)
#include <cuda.h>
#endif

namespace ik2::vmm {

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

inline const char* api_name() { return "CUDA virtual memory"; }

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
    size_t gran = 0, cap = 0, mapped = 0;   // bytes: a granule, the reserved range, what holds VRAM from its start
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
            err = "CUDA" + std::to_string(d) + ": an address range of " + std::to_string(cap >> 20) + " MiB not reserved";
            base = 0;
            return false;
        }
        acc = access_list(d);
        return true;
    }
    // put VRAM behind [mapped, round(bytes)): granule by granule, one access grant for the span; a failure gives
    // back what this call mapped (the earlier commit stands) and says why
    bool commit(size_t bytes, std::string& err) {
        const Api& v = api();
        if (bytes >= cap) bytes = cap;
        else bytes = round(bytes);
        if (mapped >= bytes) return true;
        CUdevice cd = 0;
        if (v.dev_get(&cd, dev) != CUDA_SUCCESS) {
            err = "cuDeviceGet failed";
            return false;
        }
        const CUmemAllocationProp p = prop_of(cd);
        const size_t from = mapped;
        bool ok = true;
        while (mapped < bytes) {
            CUmemGenericAllocationHandle h{};
            const CUresult r = v.create(&h, gran, &p, 0);
            if (r != CUDA_SUCCESS) {
                err = r == CUDA_ERROR_OUT_OF_MEMORY ? "CUDA" + std::to_string(dev) + ": out of device memory"
                                                    : "cuMemCreate failed (" + std::to_string((int) r) + ")";
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
            err = "cuMemSetAccess failed";
            ok = false;
        }
        if (!ok) {
            for (size_t o = from; o < mapped; o += gran) v.unmap(base + o, gran);
            mapped = from;
        }
        cudaGetLastError();
        return ok;
    }
    // give back the VRAM past round(bytes) (a4: sessions ending, chunks going back); the GPU must not use it then
    void shrink(size_t bytes) {
        if (bytes >= cap) return;
        bytes = round(bytes);
        while (mapped > bytes) {
            mapped -= gran;
            api().unmap(base + mapped, gran);
        }
        cudaGetLastError();
    }
    void close() {
        if (base == 0) return;
        shrink(0);
        api().addr_free(base, cap);
        base = 0, cap = 0, mapped = 0;
    }
};

#elif defined(SAGUARO_IK2_HIP_VMM)   // (ROCm's own virtual memory API: hipMemAddressReserve / hipMemCreate / hipMemMap)

inline const char* api_name() { return "HIP virtual memory"; }

inline hipMemAllocationProp prop_of(int dev) {
    hipMemAllocationProp p{};
    p.type = hipMemAllocationTypePinned;
    p.location.type = hipMemLocationTypeDevice;
    p.location.id = dev;
    return p;
}

inline size_t granule(int dev) {
    const hipMemAllocationProp p = prop_of(dev);
    int vm = 0;
    if (hipDeviceGetAttribute(&vm, hipDeviceAttributeVirtualMemoryManagementSupported, dev) != hipSuccess || vm == 0)
        return 0;
    size_t g = 0;
    if (hipMemGetAllocationGranularity(&g, &p, hipMemAllocationGranularityMinimum) != hipSuccess) return 0;
    return g;
}

struct Range {
    int dev = -1;
    size_t gran = 0, cap = 0, mapped = 0;
    uintptr_t base = 0;

    uint8_t* ptr() const { return (uint8_t*) base; }
    size_t round(size_t b) const { return (b + gran - 1) / gran * gran; }

    bool open(int d, size_t bytes, std::string& err) {
        dev = d;
        gran = granule(d);
        if (gran == 0) {
            err = "HIP" + std::to_string(d) + ": no virtual memory management";
            return false;
        }
        cap = round(std::max<size_t>(bytes, 1));
        hipDeviceptr_t p = 0;
        if (hipMemAddressReserve(&p, cap, gran, 0, 0) != hipSuccess) {
            err = "HIP" + std::to_string(d) + ": an address range of " + std::to_string(cap >> 20) + " MiB not reserved";
            base = 0;
            return false;
        }
        base = (uintptr_t) p;
        return true;
    }
    // put VRAM behind [mapped, round(bytes)): granule by granule, one access grant for the span; a failure gives
    // back what this call mapped (the earlier commit stands) and says why.  The elastic K / V is one GPU's
    // (--parallel, --tp and two GPUs keep their own layouts), so the grant is that device's alone.
    bool commit(size_t bytes, std::string& err) {
        if (bytes >= cap) bytes = cap;
        else bytes = round(bytes);
        if (mapped >= bytes) return true;
        const hipMemAllocationProp p = prop_of(dev);
        hipMemAccessDesc a{};
        a.location.type = hipMemLocationTypeDevice;
        a.location.id = dev;
        a.flags = hipMemAccessFlagsProtReadWrite;
        const size_t from = mapped;
        bool ok = true;
        while (mapped < bytes) {
            hipMemGenericAllocationHandle_t h = 0;
            if (hipMemCreate(&h, gran, &p, 0) != hipSuccess) {
                err = "HIP" + std::to_string(dev) + ": out of device memory (hipMemCreate)";
                ok = false;
                break;
            }
            const bool m = hipMemMap((hipDeviceptr_t) (base + mapped), gran, 0, h, 0) == hipSuccess;
            hipMemRelease(h);   // (the mapping holds the memory: unmapping gives it back)
            if (!m) {
                err = "hipMemMap failed";
                ok = false;
                break;
            }
            mapped += gran;
        }
        if (mapped > from && hipMemSetAccess((hipDeviceptr_t) (base + from), mapped - from, &a, 1) != hipSuccess) {
            err = "hipMemSetAccess failed";
            ok = false;
        }
        if (!ok) {
            for (size_t o = from; o < mapped; o += gran) hipMemUnmap((hipDeviceptr_t) (base + o), gran);
            mapped = from;
        }
        hipGetLastError();
        return ok;
    }
    void shrink(size_t bytes) {
        if (bytes >= cap) return;
        bytes = round(bytes);
        while (mapped > bytes) {
            mapped -= gran;
            hipMemUnmap((hipDeviceptr_t) (base + mapped), gran);
        }
        hipGetLastError();
    }
    void close() {
        if (base == 0) return;
        shrink(0);
        hipMemAddressFree((hipDeviceptr_t) base, cap);
        base = 0, cap = 0, mapped = 0;
    }
};

#else   // (a HIP build whose ROCm has no hipMemCreate / hipMemMap: the engine's fallback commits its K / V as one
        //  plain allocation and says so once at the load)

inline const char* api_name() { return "virtual memory"; }
inline size_t granule(int) { return 0; }

struct Range {
    int dev = -1;
    size_t gran = 0, cap = 0, mapped = 0;
    uint8_t* ptr() const { return nullptr; }
    bool open(int, size_t, std::string& err) {
        err = "a HIP build without the virtual memory API";
        return false;
    }
    bool commit(size_t, std::string& err) {
        err = "a HIP build without the virtual memory API";
        return false;
    }
    void shrink(size_t) {}
    void close() {}
};

#endif

}  // namespace ik2::vmm
