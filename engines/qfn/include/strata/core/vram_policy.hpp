// The card's own VRAM floor (PLAN-p124x-mortar-memory).  Pure policy: the engine reads the cards, the numbers
// here decide what the expert cache must leave free (generate.cpp, "expert cache auto").
//
// WHY A FLOOR: the engine already carries one - kVramSafeMib, the line #2065's stall report prints "LOW:"
// under ("a card this full can page GPU memory to system RAM; decode falls to a few tok/s, then a window never
// finishes") - but it was only ever a label: the stall report and the request-start note print it, nothing
// sizes against it.  The mortar box (RX 9070 XT, Linux/ROCm, `--expert-cache auto --vram-reserve-mib 700`)
// ended every stalled request at 416-432 MiB free: the verify-window captures, the per-slot MTP heads and
// ROCm's own slack came out of the 700 MiB reserve after the cache had taken everything else, and the driver
// paged GPU memory into a host RAM the expert tier had already filled - the exact #2065 shape, 11 aborts in
// three days.  Windows/HIP learned this under #1376 (kWddmAutoReserveMib keeps 2560 MiB on a 15 GiB+ card);
// the same driver behaviour on Linux had no floor because that fix is `#if defined(_WIN32)`.
#pragma once

#include <cstdint>

namespace strata::core::vram {

/// The free VRAM a card should keep so the driver does not page GPU memory to system RAM (#2065).  WDDM pages
/// instead of failing (docs/DETAILS.md, the sysmem fallback), so Windows wants a wider line than the 1.0 line
/// measured on Linux (the same figure generate.cpp's kVramSafeMib and the serve check use).
constexpr int64_t safe_floor_mib() {
#ifdef _WIN32
    return 1024;
#else
    return 512;
#endif
}

/// The reserve (MiB) the expert cache's auto sizing should keep on this card, given the one asked for.
/// Linux/HIP only: a ROCm card of 15 GiB and up (the #1709 threshold - the small cards have their own #496
/// path, which deliberately ends below the floor and says so) never starts with a reserve smaller than the
/// engine's own paging floor plus the slack that came out of it after the cache there: the 9070 XT's 700 MiB
/// reserve had ~272 MiB spent at the first stalled request, so 512 (the floor) + 512 (rounded measured slack)
/// = 1024 MiB is what leaves the card at or above the floor when the windows capture.  Windows has its own
/// floor (#1376), CUDA's free figure was measured honest with the same reserve (#1376's A/B arm), and a
/// reserve at or above the floor is kept as it is.  `integrated` APUs are left alone (review N3): their
/// "total VRAM" is a window into system RAM (BIOS carve-out + GTT, so an APU "reports" tens of GiB of a
/// card it does not have), the 15 GiB threshold was measured on discrete cards, and the UMA sizing above
/// (uma_cache.hpp) already guards the APU's real failure shape - paging the SHARED RAM, not the card.
constexpr int hip_auto_reserve_mib(bool hip_backend, bool windows, bool integrated,
                                   uint64_t total_vram_bytes, int given_mib) {
    if (!hip_backend || windows || integrated) return given_mib;
    if (total_vram_bytes < (15ull << 30)) return given_mib;   // small cards: the #496 shape, said, not floored
    const uint64_t floor_mib = static_cast<uint64_t>(safe_floor_mib()) + 512;
    return given_mib < static_cast<int>(floor_mib) ? static_cast<int>(floor_mib) : given_mib;
}

}  // namespace strata::core::vram
