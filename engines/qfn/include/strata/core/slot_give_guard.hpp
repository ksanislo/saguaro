#pragma once
// P124e (round 10, manager card): SAGUARO_DEBUG_SLOT_GUARD=1 - the give-way crossing guard.
// TWO give-ways hand an expert cache's slots out, and both start from the SAME line at setup: the elastic K/V
// (kvg: kvg_start(lend_first), kvg.lo counting DOWN from kvg.top) and the elastic slots (gv: v.top = the parts'
// first lend slot, v.lo counting DOWN from it).  Each counts down on its own, so on a rig that runs both (KV
// streaming + --batch) the same chunks can be handed out twice - once to the K/V pools, once to the slot
// budget / the driver - and the loan's own relayout can re-lay its buffers over slots a give-way gave.  A slot
// two owners mapped is the crash the card chased (the read's gather walking off a workspace an unmap took).
//
// The guard answers one question per give, in every path that touches a cache slot or chunk (kvg's give, move
// and take-back, gv's give, move, take-back and the relief/free-to-driver that run it, and the loan's relayout):
// does this op reach a slot of an ACTIVE prompt loan [p.first_now, slots) of a part on that cache, or a
// slot/chunk of the region the OTHER give-way already handed out?  On a crossing it prints ONE line naming the
// path, the slots/chunks and who owns them, and aborts, so a core dump carries that line as the last word of
// the run - the 40K-token repro at --max-context 1048576 then names the culprit in one run.  Off by default:
// with the env unset or 0 every hook is one predictable test of a static bool and nothing else happens.
//
// Pure (no CUDA/HIP/GPU includes; the caller computes the slot and chunk ranges it passes) so it is unit-testable
// beside park_policy / kv_host_growth / vmm_grow / read_backing.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core {

/// One participant's ACTIVE prompt loan on one cache: slots [first, slots) hold its read's buffers.
/// `first` < 0: the part lends nothing (its own buffers).
struct GiveLoan {
    const void* cache = nullptr;
    const char* part = "";     // who it is (the line names it)
    int64_t first = -1;
    int64_t slots = 0;         // the cache's slot count (the loan runs to it)
};

/// The band of slots one give-way has handed out on one cache: slots [slot_lo, slot_hi), the chunks wholly
/// inside them [chunk_lo, chunk_hi) (-1 ranges: unknown).  `path` names the give-way; an op whose own path is
/// the same string is its owner and never crosses its own region.
struct GiveRegion {
    const void* cache = nullptr;
    const char* path = "";
    int64_t slot_lo = -1, slot_hi = -1;
    int64_t chunk_lo = -1, chunk_hi = -1;
};

/// The give about to happen.  Slot/chunk ranges -1 when the path cannot name them (a release back to the
/// driver knows chunks, not the slots they came from).
struct GiveOp {
    const char* path = "";     // the give-way doing it ("kvg CUDA0", "gv CUDA1", "loan relayout", ...)
    const char* what = "";     // what it is about to do (appears in the line)
    int64_t slot_lo = -1, slot_hi = -1;
    int64_t chunk_lo = -1, chunk_hi = -1;
};

/// SAGUARO_DEBUG_SLOT_GUARD=1, read once.
inline bool slot_give_guard_on() {
    static const bool on = [] {
        const char* v = std::getenv("SAGUARO_DEBUG_SLOT_GUARD");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

namespace slot_give_guard_detail {
inline bool hit(int64_t a_lo, int64_t a_hi, int64_t b_lo, int64_t b_hi, int64_t& at) {
    if (a_lo < 0 || a_hi < 0 || b_lo < 0 || b_hi < 0) return false;
    at = std::max(a_lo, b_lo);
    return at < std::min(a_hi, b_hi);
}
}  // namespace slot_give_guard_detail

/// "" when the op crosses nothing; otherwise the ONE line naming the path, the slots/chunks and who owns them.
inline std::string slot_give_guard_line(const void* cache, const GiveOp& op,
                                        const std::vector<GiveLoan>& loans,
                                        const std::vector<GiveRegion>& regions) {
    using slot_give_guard_detail::hit;
    // (a) the active loans of this cache: a loan's slots are the read's buffers - no give may touch them.
    for (const GiveLoan& L : loans) {
        if (L.cache != cache || L.first < 0) continue;
        int64_t at = -1;
        if (!hit(op.slot_lo, op.slot_hi, L.first, L.slots, at)) continue;
        std::string w = std::string("saguaro SLOT GUARD: ") + op.path + ": " + op.what +
                        ": slot " + std::to_string(at) + " is inside the ACTIVE LOAN of '" + L.part +
                        "' (its buffers are laid out from slot " + std::to_string(L.first) + " to " +
                        std::to_string(L.slots - 1) + ")";
        if (op.chunk_lo >= 0 && op.chunk_hi >= 0)
            w += ", chunks [" + std::to_string(op.chunk_lo) + ", " + std::to_string(op.chunk_hi) + ")";
        return w + " - aborting (SAGUARO_DEBUG_SLOT_GUARD)";
    }
    // (b) what the OTHER give-way already handed out on this cache: a slot/chunk two give-ways own is the bug.
    for (const GiveRegion& R : regions) {
        if (R.cache != cache || std::strcmp(R.path, op.path) == 0) continue;
        int64_t s = -1, c = -1;
        const bool s_hit = hit(op.slot_lo, op.slot_hi, R.slot_lo, R.slot_hi, s);
        const bool c_hit = hit(op.chunk_lo, op.chunk_hi, R.chunk_lo, R.chunk_hi, c);
        if (!s_hit && !c_hit) continue;
        std::string w = std::string("saguaro SLOT GUARD: ") + op.path + ": " + op.what + ": ";
        if (s_hit) w += "slot " + std::to_string(s);
        if (s_hit && c_hit) w += " / ";
        if (c_hit) w += "chunk " + std::to_string(c);
        w += " was ALREADY handed out by '" + std::string(R.path) +
             "' (slots [" + std::to_string(R.slot_lo) + ", " + std::to_string(R.slot_hi) + "), chunks [" +
             std::to_string(R.chunk_lo) + ", " + std::to_string(R.chunk_hi) + "))";
        return w + " - aborting (SAGUARO_DEBUG_SLOT_GUARD)";
    }
    return "";
}

/// Writes the crossing line to `out` and returns true; prints nothing and returns false when the op is clean.
inline bool slot_give_guard_report(FILE* out, const void* cache, const GiveOp& op,
                                   const std::vector<GiveLoan>& loans, const std::vector<GiveRegion>& regions) {
    const std::string w = slot_give_guard_line(cache, op, loans, regions);
    if (w.empty()) return false;
    std::fprintf(out, "%s\n", w.c_str());
    std::fflush(out);
    return true;
}

/// Off: nothing (one test of a static bool).  On: prints the crossing line to stderr and aborts - the core
/// dump then carries the culprit as the last line the engine ever wrote.
inline void slot_give_guard(const void* cache, const GiveOp& op,
                            const std::vector<GiveLoan>& loans, const std::vector<GiveRegion>& regions) {
    if (!slot_give_guard_on()) return;
    if (slot_give_guard_report(stderr, cache, op, loans, regions)) std::abort();
}

}  // namespace strata::core
