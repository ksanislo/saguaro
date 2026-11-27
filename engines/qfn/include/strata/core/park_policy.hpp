// What the parked conversations give back, and when (PLAN-qfn-park-budget).  Pure policy: the serve loop samples the
// memory, lists the parked entries and carries out the actions (generate.cpp, park_tick).
//
// THE ORDER OF CLAIMS ON RAM (Ken 11-21):  live context  >  model weights held in RAM  >  parked conversations.
//   * Parked conversations share ONE pool with the live K/V: max_context x batch slots positions, RAM the engine must
//     be able to pin for live context anyway.  Their budget is what the live K/V does not hold right now, so a slot
//     that grows or a new conversation shrinks it and the oldest parked entries leave.
//   * They never take room from the weights or from live context: when the host's available RAM falls under the
//     floor (or the kernel reports memory stalls), they are released until it is back above it.  The engine's own
//     pinned weights and K/V pool are credited back when judging that (Memory::own_pinned) - down to the hard
//     floor: a resident-RAM engine is meant to hold most of the host, and dropping parks cannot change that, but
//     below the hard floor the raw reading decides and the releases fire anyway (PLAN-p124a).
//   * Otherwise they stay: nothing expires by age unless SAGUARO_PARK_IDLE_S says so.
// An entry whose session file is written leaves at once.  One that is not yet on disk has its write started now (no
// write delay) and its RAM goes back when the write ends; it is dropped unwritten only when the available RAM is
// under the hard floor (--conversation-cache-min-free-mib), where pinned live context would be at risk.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <istream>
#include <optional>
#include <string>
#include <vector>

namespace strata::core::park {

struct Limits {
    uint64_t pool_bytes = 0;                 // the shared pool (0: the cache keeps a fixed budget of its own)
    uint64_t free_bytes = 16384ull << 20;    // SAGUARO_PARK_FREE_MIB: available RAM under this is pressure (0: off)
    uint64_t severe_bytes = 2560ull << 20;   // --conversation-cache-min-free-mib: under this, unsaved entries go too
    double psi = 10.0;                       // SAGUARO_PARK_PSI: memory "some avg10" above this is pressure (0: off)
    double idle_s = 0.0;                     // SAGUARO_PARK_IDLE_S: an entry idle longer than this leaves (0: never)
};

struct Memory {
    std::optional<uint64_t> available;       // MemAvailable under the cgroup limits (conversation_memory.hpp)
    std::optional<double> psi_some10;        // /proc/pressure/memory, when the kernel has it
    // RAM the engine itself holds pinned by design and no release gives back: the resident expert tier and the live
    // K/V pool (PLAN-p124a).  `available` counts them as used - on a resident-RAM box that keeps it under the floor
    // forever, and every park is dropped seconds after it lands while the host is wide open.  Pressure asks for what
    // is really available for one more parked conversation: the engine's own tier is credited back for the release
    // decision (weights outrank parked, and they are not leaving).  The credit stops at the hard floor: under
    // `severe_bytes` the RAW reading is what is short - a host genuinely full is not the resident mode working, and
    // its parks must go (F1).  The admission checks keep the raw reading too, so a full host still stops a park.
    uint64_t own_pinned = 0;
};

struct Entry {
    uint64_t serial = 0, bytes = 0, tokens = 0;
    double idle_s = 0.0;
    bool on_disk = false;   // its session file (or a newer one of the same conversation) is written
    bool keep = false;      // holds a pinned shared prefix, or is being borrowed from: never released
};

enum class Why { budget, pressure, severe, idle };
enum class Do { drop, write_drop, drop_unsaved };
struct Action {
    uint64_t serial = 0;
    Do what = Do::drop;
    Why why = Why::budget;
};

inline const char* why_name(Why w) {
    return w == Why::budget ? "budget" : w == Why::pressure ? "pressure" : w == Why::severe ? "unsaved" : "idle";
}

/// What the parked conversations may hold while the live K/V holds `live_bytes` of the pool.
inline uint64_t budget(const Limits& l, uint64_t live_bytes) {
    return l.pool_bytes > live_bytes ? l.pool_bytes - live_bytes : 0;
}

/// `some avg10=` of /proc/pressure/memory; nullopt when the line is not there.
inline std::optional<double> psi_some10(std::istream& in) {
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("some", 0) != 0) continue;
        const size_t at = line.find("avg10=");
        if (at == std::string::npos) return std::nullopt;
        double v = 0.0;
        if (std::sscanf(line.c_str() + at + 6, "%lf", &v) != 1) return std::nullopt;
        return v;
    }
    return std::nullopt;
}

inline bool severe(const Memory& m, const Limits& l) { return m.available && *m.available < l.severe_bytes; }

/// The bytes pressure asks back: up to an eighth above the floor, so one release does not leave it at the edge.
/// The engine's own pinned tier and live K/V pool (`own_pinned`: RAM no release gives back) are credited first -
/// the floor guards against a shortage no parked conversation caused, not against the resident RAM mode working.
/// The credit stops at the hard floor: under `severe_bytes` the raw reading is asked for back, so the floor still
/// fires on a host genuinely full however much the engine itself pins (PLAN-p124a review F1 - with the credit
/// unbounded, `pressed` could never go true on a resident-RAM box and no release, unsaved ones included, ran).
inline uint64_t pressure_need(const Memory& m, const Limits& l) {
    if (l.free_bytes == 0 || !m.available) return 0;
    const uint64_t avail = severe(m, l)
        ? *m.available
        : (*m.available > UINT64_MAX - m.own_pinned ? UINT64_MAX : *m.available + m.own_pinned);
    if (avail >= l.free_bytes) return 0;
    return l.free_bytes - avail + l.free_bytes / 8;
}
inline bool psi_high(const Memory& m, const Limits& l) { return l.psi > 0.0 && m.psi_some10 && *m.psi_some10 > l.psi; }

/// The releases due now.  `entries`: least recently active first.  `parked_bytes` / `budget_bytes`: what the cache
/// holds and may hold.  `pending_bytes`: RAM of entries already released whose session file is still being written
/// (it comes back by itself).  `can_write`: there is a session folder.  `psi_due`: a release for memory stalls alone
/// is allowed now (the caller spaces them out: avg10 falls slowly).  Unknown telemetry releases nothing.
inline std::vector<Action> plan(const std::vector<Entry>& entries, uint64_t parked_bytes, uint64_t budget_bytes,
                                uint64_t pending_bytes, const Memory& mem, const Limits& lim, bool can_write,
                                bool psi_due) {
    std::vector<Action> out;
    std::vector<uint8_t> taken(entries.size(), 0);
    const bool hard = severe(mem, lim);
    const uint64_t over = parked_bytes > budget_bytes ? parked_bytes - budget_bytes : 0;
    uint64_t press = pressure_need(mem, lim);
    press = press > pending_bytes ? press - pending_bytes : 0;
    // memory stalls with RAM still above the floor: one entry at a time
    const bool stalls = press == 0 && pending_bytes == 0 && psi_due && psi_high(mem, lim);
    const uint64_t need = std::max(over, press);
    const Why why = press >= over && press > 0 ? Why::pressure : Why::budget;
    uint64_t freed = 0;
    auto more = [&] { return freed < need || (stalls && freed == 0); };
    // on disk already: gone at once, oldest first
    for (size_t i = 0; i < entries.size() && more(); ++i) {
        const Entry& e = entries[i];
        if (e.keep || !e.on_disk) continue;
        out.push_back({e.serial, Do::drop, stalls && need == 0 ? Why::pressure : why});
        taken[i] = 1;
        freed += e.bytes;
    }
    // not on disk: written now and freed when the write ends; unwritten only under the hard floor, or when there
    // is no folder to write to and the budget itself is exceeded (the cache's own limit, as before)
    for (size_t i = 0; i < entries.size() && more(); ++i) {
        const Entry& e = entries[i];
        if (e.keep || e.on_disk) continue;
        Action a{e.serial, Do::write_drop, stalls && need == 0 ? Why::pressure : why};
        if (hard && (why == Why::pressure || !can_write)) a = {e.serial, Do::drop_unsaved, Why::severe};
        else if (!can_write && over > freed) a = {e.serial, Do::drop_unsaved, Why::budget};
        else if (!can_write) continue;
        out.push_back(a);
        taken[i] = 1;
        freed += e.bytes;
    }
    if (lim.idle_s > 0.0)
        for (size_t i = 0; i < entries.size(); ++i) {
            const Entry& e = entries[i];
            if (taken[i] || e.keep || e.idle_s <= lim.idle_s || (!e.on_disk && !can_write)) continue;
            out.push_back({e.serial, e.on_disk ? Do::drop : Do::write_drop, Why::idle});
        }
    return out;
}

/// PLAN-p124a2: a park that follows its own conversation's turn save does not have to capture a second
/// time - the capture already holds a host copy of everything the park would copy. Figures only:
/// `captured`  - a WHOLE (not delta) capture of this same place is still held (the turn hold keeps a strong
///               reference to it until this park, a continuing request or a rewrite clears it - the card run
///               of 2026-11-26 showed the parks that follow a FINISHED file write are the slow ones, so the
///               hold outlives the writer);
/// `single_stage` - no layer split: one session's snapshot is what the capture covered;
/// `nothing_ran`  - no request began since the capture, so the session still holds exactly the captured bytes;
/// `same_tokens`  - the parked conversation has the token count, ids and steering mode the capture took;
/// `no_images`    - neither the capture nor the park carries image keys (an image list is not compared).
/// These five say the capture's RUNNING STATE (its `meta.live`, a whole copy whatever the K/V capture held)
/// is byte-identical to what this park would copy off the device - the host copy the turn save just made for
/// the same token count. It holds even with the cache's retained K/V in play, which concerns only the K/V.
inline bool reuse_turn_state(bool captured, bool single_stage, bool nothing_ran, bool same_tokens,
                             bool no_images) {
    return captured && single_stage && nothing_ran && same_tokens && no_images;
}

/// The whole-image adoption (conversation_snapshot_from_capture): the five terms above plus
/// `no_retained` - the cache's retained K/V is the source already (with it, the park copies no prefix, and
/// the adoption's aliased K/V has nothing to reuse). All six must hold for it; anything else captures the
/// K/V normally - and where only `no_retained` fails, the park still skips the running state's device copy
/// (reuse_turn_state). The result is byte-identical either way: adoption aliases the same bytes, it does
/// not invent them.
inline bool adopt_turn_capture(bool captured, bool single_stage, bool no_retained, bool nothing_ran,
                               bool same_tokens, bool no_images) {
    return no_retained && reuse_turn_state(captured, single_stage, nothing_ran, same_tokens, no_images);
}

}  // namespace strata::core::park
