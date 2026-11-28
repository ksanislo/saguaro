#pragma once
// P124e (round 5, manager card test 11-26): the load-time VRAM accounting.  A named device allocation of the startup
// records the card's free memory before and after it, so the load can print - before the expert cache sizes itself -
// WHAT every big VRAM user is and how much the card actually CHARGED for it: an address reservation can cost 0
// while its backing costs GiB, and only the free-memory delta of the allocation itself can tell the two apart.
// Names aggregate: the same name sums its bytes, its charge and its count.

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// One accounted allocation: constructed just before it, destroyed just after; `bytes` is what was asked for.
struct DeviceNote {
    DeviceNote(const char* what, uint64_t bytes);
    ~DeviceNote();
    DeviceNote(const DeviceNote&) = delete;
    DeviceNote& operator=(const DeviceNote&) = delete;

    const char* what;
    uint64_t bytes;
    uint64_t free0;   // the card's free memory at construction (0: the query failed, the delta is not recorded)
};

/// An accounted name, merged across its allocations: the bytes asked, the VRAM the card charged (the summed
/// free-memory delta; 0 means "asked for and the card stayed free" - an address space that costs nothing), and how
/// many allocations carried the name.
struct DeviceNoteRec {
    std::string what;
    uint64_t bytes = 0;
    int64_t used = 0;
    int n = 0;
};

/// The accounted names so far, moved out (the load prints them once, before the expert cache sizes itself).
std::vector<DeviceNoteRec> device_notes_taken();

}  // namespace strata::core
