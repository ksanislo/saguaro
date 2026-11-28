#pragma once
// macOS readings of what Linux takes from /proc (meminfo, self/status, self/stat, self/io): the host's RAM and swap,
// and this process's footprint and I/O.  Header-only; every function answers false / 0 off macOS.
#include <cstdint>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#endif

namespace strata::platform {

struct MacHostMemory {
    uint64_t total = 0;      ///< hw.memsize
    uint64_t free = 0;       ///< free + speculative pages (MemFree's sense: handed out without reclaiming anything)
    uint64_t available = 0;  ///< kern.memorystatus_level of the RAM (MemAvailable's sense)
};

inline bool mac_host_memory(MacHostMemory& m) {
#if defined(__APPLE__)
    uint64_t total = 0;
    size_t len = sizeof total;
    if (sysctlbyname("hw.memsize", &total, &len, nullptr, 0) != 0 || total == 0) return false;
    vm_statistics64_data_t vs{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page = 0;
    if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS ||
        host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &count) != KERN_SUCCESS) return false;
    m.total = total;
    m.free = ((uint64_t) vs.free_count + vs.speculative_count) * (uint64_t) page;
    // the kernel's own "memory free percentage" (kern.memorystatus_level, what memory_pressure prints and what the
    // pressure levels and jetsam go by): free pages plus the file cache and what the compressor can still take -
    // MemAvailable's sense on macOS.  The page counts (free + inactive + purgeable) where it cannot be read.
    uint32_t level = 0;
    size_t llen = sizeof level;
    if (sysctlbyname("kern.memorystatus_level", &level, &llen, nullptr, 0) == 0 && level > 0 && level <= 100)
        m.available = total / 100 * level;
    else
        m.available = ((uint64_t) vs.free_count + vs.speculative_count + vs.inactive_count + vs.purgeable_count) *
                      (uint64_t) page;
    if (m.available > total) m.available = total;
    return m.available > 0;
#else
    (void) m;
    return false;
#endif
}

/// The swap in use (vm.swapusage); false when unknown.
inline bool mac_swap_used(uint64_t& bytes) {
#if defined(__APPLE__)
    xsw_usage sw{};
    size_t len = sizeof sw;
    if (sysctlbyname("vm.swapusage", &sw, &len, nullptr, 0) != 0) return false;
    bytes = sw.xsu_used;
    return true;
#else
    (void) bytes;
    return false;
#endif
}

/// The kernel's memory pressure level (kern.memorystatus_vm_pressure_level): 1 normal, 2 warning, 4 critical;
/// 0 when unknown.
inline int mac_memory_pressure() {
#if defined(__APPLE__)
    int level = 0;
    size_t len = sizeof level;
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &len, nullptr, 0) != 0) return 0;
    return level;
#else
    return 0;
#endif
}

struct MacProcessMemory {
    uint64_t resident = 0;    ///< resident set
    uint64_t footprint = 0;   ///< phys_footprint: what Activity Monitor calls Memory (resident + compressed + swapped)
    uint64_t compressed = 0;  ///< this process's pages in the compressor or swap
    uint64_t major_faults = 0;
    uint64_t reads = 0;       ///< block input operations (ru_inblock)
};

inline bool mac_process_memory(MacProcessMemory& p) {
#if defined(__APPLE__)
    task_vm_info_data_t vi{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &vi, &count) != KERN_SUCCESS) return false;
    p.resident = vi.resident_size;
    p.footprint = vi.phys_footprint;
    p.compressed = vi.compressed;
    rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        p.major_faults = (uint64_t) ru.ru_majflt;
        p.reads = (uint64_t) ru.ru_inblock;
    }
    return true;
#else
    (void) p;
    return false;
#endif
}

}  // namespace strata::platform
