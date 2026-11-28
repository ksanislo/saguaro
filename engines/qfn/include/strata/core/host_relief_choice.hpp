#pragma once
// host_relief_choice.hpp - P125p: the DECISION host_make_room (layer.cpp) makes on every K/V host-copy grow step
// whose pin would cut into SAGUARO_HOST_GROW_RESERVE_MIB.  1.2.4-pre on Paul's P40: the line "the host copy grows by
// 0.16 GiB with 7.27 GiB of RAM available: the expert RAM tier gave 0.00 GiB back (2.89 asked)" repeated ~20 times a
// long read - his expert tier is not a pinned resident complement, so relieve_ram answers 0 on every step and the
// line reported a give-back that never happened and never could.  The fix is the question the line assumed: ASK the
// tier only when it CAN give (ram_relievable_bytes() nonzero - 0 when the tier is not a resident complement or is
// registered whole); when it cannot, say the room is short ONCE per load and why, not once per step.  The real
// give-back line (what relieve_ram freed, when it was really asked) is kept untouched.  Pure - no GPU, no RAM query -
// so the test can read it.
#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::core {

/// host_make_room's per-load latch and branch: `relievable` is the tier's answer to ram_relievable_bytes().
/// `ask`: the tier can give - ask it for the shortfall and report what it gave, as before.  `say`: the tier cannot
/// give, and the short-room line belongs to this call: true the FIRST time only (per load - the engine loads once,
/// and host_make_room calls this under g_host_mu), which is what turns twenty repeats into one sentence.
struct HostRoomDecision {
    struct Answer {
        bool ask;
        bool say;
    };
    bool noted = false;   // the short room has been said since the load
    Answer answer(uint64_t relievable) {
        if (relievable > 0) return {true, false};
        const Answer a{false, !noted};
        noted = true;
        return a;
    }
};

/// The one line: the room is short, and the expert RAM tier has nothing it can give back.  `grow_bytes` is what the
/// step pins, `avail_bytes` the RAM the OS has, `ask_bytes` the reserve shortfall the ask would have been.
inline std::string host_room_short_text(uint64_t grow_bytes, uint64_t avail_bytes, uint64_t ask_bytes) {
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "saguaro: KV streaming: the host copy grows by %.2f GiB with %.2f GiB of RAM available and %.2f GiB "
                  "short of the reserve: the expert RAM tier has nothing it can give back (no resident complement, or "
                  "it is pinned whole) - said once, not once per grow step\n",
                  (double) grow_bytes / 1073741824.0, (double) avail_bytes / 1073741824.0,
                  (double) ask_bytes / 1073741824.0);
    return std::string(buf);
}

}  // namespace strata::core
