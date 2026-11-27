#pragma once
// vmm_grow.hpp - P124e (round 6, manager card; round 7: every growth allocation): the ORDER and the WORDING of a
// device-state grow (layer.cpp's follow pools, block-score scratch, RoPE table and YaRN rows - the arrays a
// streamed conversation keeps on the card and grows as its conversation grows).
//
// The order: the chunks a grow needs are TAKEN from the driver and HELD before any mapping is tried.  Probing
// one chunk at a time (make, free, make, ...) proved only that ONE chunk fits at a time: with the card nearly
// full it promised GiBs it did not have, never asked the card's relief (the expert cache giving way, coldest
// first) for room, and the map then failed mid-run.  Now the first refusal asks the relief ONCE for the whole
// remaining shortfall - so the cache has given way BEFORE the mapping is tried - and the map only starts with
// every chunk in hand.  A conversation that truly cannot fit fails before the map, with its room said out.
//
// The wording: a driver-API refusal leaves cudaGetLastError() clean, so the old failure line read "no error"
// and the caller answered a VRAM shortage with the RAM message.  vmm_grow_failure_text() says what the driver
// returned (vmm_last_error()), which call, how much was asked, how much VRAM is free, and whether the expert
// cache was asked to give way and what it gave.  Pure - no GPU, no allocation - so the test can read it.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::core {

/// A physical chunk handle (strata::core::VmmChunk, vmm.hpp); 0 = none.  Re-declared here so this header needs
/// no CUDA/HIP include: the grow helpers are device-API agnostic.
using GrowChunk = unsigned long long;

/// What a grow's room-taking did, for the failure line: the chunks held, what the relief (the expert cache)
/// was asked to hand over and what it handed back.
struct VmmRoomStat {
    int64_t made = 0;          // chunks taken from the driver and held
    uint64_t relief_ask = 0;   // bytes the relief was asked for (0: never asked - the driver's free memory sufficed)
    uint64_t relief_got = 0;   // bytes the relief handed back (its given chunks), 0 when it gave nothing
};

/// Take `need` chunks of `gran` bytes BEFORE any mapping, holding every one in `got`.  The first refusal asks
/// the relief once (when `may_relief`) for the whole remaining shortfall - `give_way(bytes)` answers the bytes
/// it handed back - and retries; a second refusal does not pester it.  false: the room never arrived; `got`
/// holds the chunks made anyway (the caller frees them) and `st` says what was asked and given, for
/// vmm_grow_failure_text().  `make()` must return 0 when the driver has nothing (that is the signal).
template <class Make, class GiveWay>
inline bool vmm_grow_room(int64_t need, uint64_t gran, Make make, GiveWay give_way, bool may_relief,
                          std::vector<GrowChunk>& got, VmmRoomStat& st) {
    for (int64_t i = 0; i < need; ++i) {
        GrowChunk h = make();
        if (h == 0 && may_relief && st.relief_ask == 0) {   // (once: ask for EVERYTHING still missing, not a chunk)
            st.relief_ask = (uint64_t) (need - (int64_t) got.size()) * gran;
            st.relief_got = give_way(st.relief_ask);
            h = make();
        }
        if (h == 0) return false;
        got.push_back(h);
        st.made = (int64_t) got.size();
    }
    return true;
}

/// The same room for a grow that first helps itself to the card's slot budget (P124e round 7, manager card: the
/// RoPE table and the YaRN rows - the two context arrays the budget had been the ONLY source of, so a card whose
/// budget ran dry failed the grow without ever asking the expert cache to give way).  `budget(want, got)` appends
/// what the pool holds (up to `want`) and returns how many it took; the SHORTFALL then runs exactly
/// vmm_grow_room's order: hold every chunk, the first refusal asks the relief once for the whole shortfall, and
/// the map starts after the cache has given way.  false: the room never arrived - `got` holds the pool's chunks
/// at its front (`*from_pool` says how many, the caller gives those back to the pool and frees the rest).
template <class Budget, class Make, class GiveWay>
inline bool vmm_grow_room_pooled(int64_t need, uint64_t gran, Budget budget, Make make, GiveWay give_way,
                                 bool may_relief, std::vector<GrowChunk>& got, VmmRoomStat& st,
                                 int64_t* from_pool) {
    const int64_t from = need > 0 ? budget(need, got) : 0;
    if (from_pool != nullptr) *from_pool = from;
    return need - from <= 0 || vmm_grow_room(need - from, gran, make, give_way, may_relief, got, st);
}

/// The grow's failure line (round 6's list, and never "no error", never a RAM message for a VRAM failure).
/// `what`: "mapped" (a map that refused) or "grown" (the room never arrived); `driver_err`: the driver's own
/// words (vmm_last_error(), "" when the call recorded nothing).  `chunks` x `gran` were asked; the card had
/// `free_bytes` of `total_bytes`.  The relief clause says whether the expert cache was asked and what it gave.
inline std::string vmm_grow_failure_text(int dev, const char* what, const char* driver_err, int64_t chunks,
                                         uint64_t gran, uint64_t free_bytes, uint64_t total_bytes,
                                         uint64_t relief_ask, uint64_t relief_got) {
    const double mib = 1048576.0, gib = 1073741824.0;
    const uint64_t asked = (uint64_t) std::max<int64_t>(chunks, 0) * gran;
    char buf[512];
    const int n = std::snprintf(buf, sizeof buf,
                                "the context's device state could not be %s on CUDA%d: %s; %.0f MiB asked "
                                "(%lld chunks of %.0f MiB), %.2f of %.2f GiB free on the card; the expert cache %s",
                                what, dev,
                                driver_err != nullptr && driver_err[0] != '\0'
                                    ? driver_err
                                    : "the driver call gave no code (the CUDA error state stayed clean)",
                                (double) asked / mib, (long long) chunks, (double) gran / mib,
                                (double) free_bytes / gib, (double) total_bytes / gib,
                                relief_ask != 0 ? "was asked to give way"
                                                : "was never asked (the driver's free memory was not short)");
    const size_t at = n > 0 ? (size_t) n : 0;
    if (relief_ask != 0 && at + 1 < sizeof buf)
        std::snprintf(buf + at, sizeof buf - at, " of %.0f MiB and gave %.0f MiB", (double) relief_ask / mib,
                      (double) relief_got / mib);
    return std::string(buf);
}

}  // namespace strata::core
