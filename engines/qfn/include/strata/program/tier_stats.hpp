#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::program {

/// P125s: the RAM and SSD fields of the STAT line ("Where the experts live"), from where the experts are.
/// The VRAM caches hold `vram_used` experts / `vram_bytes`; the resident RAM mode's compact copy holds
/// `comp_used` / `comp_bytes` (its whole range with --resident-experts, `--resident-budget-gib` of the
/// hottest ones, the experts no GPU-cache slot keeps); the file tier is the rest - the SSD the panel must
/// say something lives on.  With no copy the tier outside the VRAM caches IS the RAM tier: the mmap's page
/// cache in front of the mapped files (assumed to hold the file tier as far as the machine lets it), or the
/// resident arena, which holds every expert and reads no SSD tier at all.  Counts and bytes clamp at 0 -
/// the helper caches can report bytes over a pack the geometry does not have.
struct StatTiers {
    int64_t ram_used = 0, disk_used = 0;
    double ram_gb = 0.0, disk_gb = 0.0;
};

inline StatTiers stat_tiers(int64_t all_experts, double all_expert_bytes, int64_t vram_used, double vram_bytes,
                            int64_t comp_used, double comp_bytes, bool mmap_mode) {
    constexpr double GIB = 1073741824.0;
    StatTiers t;
    if (comp_used > 0) {   // the resident RAM mode: the copy is the RAM tier, the file tier the rest
        t.ram_used = comp_used;
        t.ram_gb = comp_bytes / GIB;
        t.disk_used = std::max<int64_t>(0, all_experts - vram_used - comp_used);
        t.disk_gb = std::max(0.0, (all_expert_bytes - vram_bytes - comp_bytes) / GIB);
    } else if (mmap_mode) {   // no copy: the page cache in front of the mapped files is what holds the RAM tier
        t.ram_used = std::max<int64_t>(0, all_experts - vram_used);
        t.ram_gb = std::max(0.0, (all_expert_bytes - vram_bytes) / GIB);
    } else {   // the resident arena: every expert is in RAM, and the SSD tier reads nothing
        t.ram_used = std::max<int64_t>(0, all_experts - vram_used);
        t.ram_gb = all_expert_bytes / GIB;
    }
    return t;
}

}  // namespace strata::program
