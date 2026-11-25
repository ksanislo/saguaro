// include/strata/program/turbo_skip.hpp - the Turbo skip's request-line -> kernel mapping, checkable with no GPU.
//
// Two numeric conventions meet at the request line and only these functions translate between them (the review
// of PLAN-s1-qfn-turbo-skip finding 1: an untranslated pass-through routes `ssd` on the ram bar and back):
//   the wire value `turbo_skip=` parses to (the Turbo struct, GLM's convention): 0 off, 1 = ssd, 2 = ram;
//   the kernel bar `SessionState::turbo_skip` carries (native_router.hpp's TurboRoute::skip): 0 off,
//   1 = leave out what is not resident in VRAM (the wire's ram), 2 = ...nor in the RAM tier (the wire's ssd).
// The translation happens once, in turbo_skip_of (src/program/generate.cpp), on its way to the session.
#pragma once

#include <cstring>

namespace strata::program::turbo_skip {

/// The value the request line's `turbo_skip=` parses to: 0 off, 1 ssd, 2 ram (the wire's convention).
inline int wire_of(const char* val) {
    return std::strcmp(val, "ram") == 0 ? 2 : std::strcmp(val, "ssd") == 0 ? 1 : 0;
}

/// The bar the reply's rows route with for a wire skip: wire ram (2) = bar 1 (the VRAM table), wire ssd (1) =
/// bar 2 (the VRAM table plus the RAM-tier mirror).  Anything else is off.
inline int bar_of(int wire) { return wire == 2 ? 1 : wire == 1 ? 2 : 0; }

/// The name the request's line prints for the bar it actually routed with.
inline const char* bar_label(int bar) { return bar == 1 ? "ram" : bar == 2 ? "ssd" : "off"; }

/// Why a request's skip runs exact instead, from the layout's facts: 0 = it runs, 1 = a SAGUARO_ROUTE_PRIOR /
/// SAGUARO_ROUTE_RESIDENT re-pick would undo the cut, 2 = the device residency table is missing (ram needs it),
/// 3 = ssd on a layout with no reliable RAM-tier fact or with the whole model in RAM (ssd_layout: g_turbo_ssd,
/// which is 2 only where the RAM tier's mirror is on the device).  Checked in this order, as turbo_skip_of says them.
inline int refusal(int wire, bool repick, bool res_table, int ssd_layout) {
    if (wire == 0) return 0;
    if (repick) return 1;
    if (!res_table) return 2;
    if (wire == 1 && ssd_layout != 2) return 3;
    return 0;
}

/// The bar to route with: bar_of(wire), or 0 when refusal() says the request runs exact.
inline int bar(int wire, bool repick, bool res_table, int ssd_layout) {
    return refusal(wire, repick, res_table, ssd_layout) != 0 ? 0 : bar_of(wire);
}

}  // namespace strata::program::turbo_skip
