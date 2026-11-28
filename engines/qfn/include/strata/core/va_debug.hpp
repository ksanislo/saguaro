// SAGUARO_DEBUG_VA (P124e round 12, manager card B1): one stderr line for EVERY address-space operation -
// every VMM reserve / map / unmap / release (cuMemAddressReserve/hipMemAddressReserve, cuMemMap/hipMemMap
// and their unmap/free) and every pinned host register / unregister - naming the VA range [base, base +
// size), the owner and the op.  The 546K read died on a GPU page fault at one address (dmesg, mortar
// 2026-11-27): with this on, that address falls in exactly one logged range and the log says who reserved
// it, who mapped what of it, and who - if anyone - unmapped it while a kernel could still read it.  The
// owners named: the K/V host copies (whole, pieces, prefixes), the K/V host address ranges, the expert
// cache's VMM arena, the slot/follow/scores pools and their budget chunks, the RoPE and YaRN table ranges.
// Off by default: with the env unset every hook is one static-bool test and nothing is printed.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::core {

inline bool va_debug_on() {
    static const bool on = [] {
        const char* v = std::getenv("SAGUARO_DEBUG_VA");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

/// The line, formatted (pure, unit-testable): "saguaro va: <op> dev <d> [<base>, +<size>) <what>".
/// `dev` -1: a host mapping (no device).  `size` 0: a count of chunks, not a range (budget handouts).
inline void va_debug_format(char* buf, size_t n, const char* op, int dev, uint64_t base, uint64_t size,
                            const char* what) {
    std::snprintf(buf, n, "saguaro va: %-8s dev %d [%#llx, +%llu) %s\n", op, dev,
                  (unsigned long long) base, (unsigned long long) size, what != nullptr ? what : "-");
}

/// One op line (free, gated).  Called from vmm.cpp's reserve/map/unmap/release, the K/V host copies'
/// register/unregister, and the slot budget's chunk handouts.
inline void va_log(const char* op, int dev, uint64_t base, uint64_t size, const char* what) {
    if (!va_debug_on()) return;
    char buf[256];
    va_debug_format(buf, sizeof buf, op, dev, base, size, what);
    std::fputs(buf, stderr);
}

}  // namespace strata::core
