// What the parked conversations give back, and when (PLAN-qfn-park-budget).  Pure policy: the serve loop samples the
// memory, lists the parked entries and carries out the actions (generate.cpp, park_tick).
//
// THE ORDER OF CLAIMS ON RAM (Ken 11-21):  live context  >  model weights held in RAM  >  parked conversations.
//   * Parked conversations share ONE pool with the live K/V: max_context x batch slots positions, RAM the engine must
//     be able to pin for live context anyway.  Their budget is what the live K/V does not hold right now, so a slot
//     that grows or a new conversation shrinks it and the oldest parked entries leave.
//   * They never take room from the weights or from live context: when the host's available RAM falls under the
//     floor (or the kernel reports memory stalls), they are released until it is back above it.
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

/// The bytes pressure asks back: up to an eighth above the floor, so one release does not leave it at the edge.
inline uint64_t pressure_need(const Memory& m, const Limits& l) {
    if (l.free_bytes == 0 || !m.available || *m.available >= l.free_bytes) return 0;
    return l.free_bytes - *m.available + l.free_bytes / 8;
}
inline bool psi_high(const Memory& m, const Limits& l) { return l.psi > 0.0 && m.psi_some10 && *m.psi_some10 > l.psi; }
inline bool severe(const Memory& m, const Limits& l) { return m.available && *m.available < l.severe_bytes; }

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

}  // namespace strata::core::park
