// The parked conversations' release policy (include/strata/core/park_policy.hpp): the order entries leave in under
// a shrinking budget, memory pressure and the idle limit, and what happens to one that is not on disk yet.
#include "strata/core/park_policy.hpp"

#include <cstdio>
#include <cstdlib>
#include <sstream>

using namespace strata::core::park;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
constexpr uint64_t G = 1ull << 30;
Entry entry(uint64_t serial, uint64_t gib, bool on_disk, double idle = 100.0, bool keep = false) {
    return {serial, gib * G, gib * 80000, idle, on_disk, keep};
}
std::string said(const std::vector<Action>& a) {
    std::string s;
    for (const Action& x : a)
        s += std::to_string(x.serial) + (x.what == Do::drop ? "d" : x.what == Do::write_drop ? "w" : "u") + ":" +
             why_name(x.why) + " ";
    return s;
}
}  // namespace

int main() {
    Limits lim;   // the defaults: 16 GiB free floor, 2.5 GiB hard floor, PSI 10, no idle limit
    lim.pool_bytes = 100 * G;
    const Memory calm{64 * G, 0.0};
    // oldest first: 1 (4 GiB, on disk), 2 (12 GiB, not), 3 (5 GiB, on disk), 4 (3 GiB, not)
    const std::vector<Entry> four{entry(1, 4, true), entry(2, 12, false), entry(3, 5, true), entry(4, 3, false)};
    const uint64_t parked = 24 * G;

    check(budget(lim, 10 * G) == 90 * G && budget(lim, 100 * G) == 0 && budget(lim, 130 * G) == 0,
          "the budget is the pool less the live K/V, never negative");
    check(plan(four, parked, 90 * G, 0, calm, lim, true, true).empty(), "room and calm memory: everything stays");
    {
        Limits old = lim;
        old.idle_s = 0;
        std::vector<Entry> aged = four;
        for (auto& e : aged) e.idle_s = 86400.0 * 30;
        check(plan(aged, parked, 90 * G, 0, calm, old, true, true).empty(), "no idle limit: a month idle still stays");
    }
    // live demand: the live K/V grew, the budget is 3 GiB under what is parked
    check(said(plan(four, parked, 21 * G, 0, calm, lim, true, true)) == "1d:budget ", "live demand takes the oldest entry on disk");
    check(said(plan(four, parked, 18 * G, 0, calm, lim, true, true)) == "1d:budget 3d:budget ", "then the next one on disk, not the older unwritten one");
    check(said(plan(four, parked, 14 * G, 0, calm, lim, true, true)) == "1d:budget 3d:budget 2w:budget ",
          "past those: the oldest unwritten one is written now and freed when its write ends");
    check(said(plan(four, parked, 0, 0, calm, lim, true, true)) == "1d:budget 3d:budget 2w:budget 4w:budget ",
          "the live K/V fills the pool: every entry leaves, none unwritten");
    check(said(plan(four, parked, 14 * G, 0, calm, lim, false, true)) == "1d:budget 3d:budget 2u:budget ",
          "without a session folder the budget still holds (dropped, as the cache always did)");

    // memory pressure: 9 GiB available under the 16 GiB floor asks for 7 + 2 back
    const Memory low{9 * G, 0.0};
    check(pressure_need(low, lim) == 9 * G && pressure_need(calm, lim) == 0, "pressure asks back the shortfall and an eighth");
    check(said(plan(four, parked, 90 * G, 0, low, lim, true, true)) == "1d:pressure 3d:pressure ", "pressure: entries on disk go first, oldest first");
    const Memory lower{4 * G, 0.0};   // asks 14 GiB: the two on disk (9) are not enough
    check(said(plan(four, parked, 90 * G, 0, lower, lim, true, true)) == "1d:pressure 3d:pressure 2w:pressure ",
          "then unwritten ones are written at once, not dropped");
    check(said(plan(four, parked, 90 * G, 12 * G, lower, lim, true, true)) == "1d:pressure ",
          "RAM already on its way back (writes running) counts: no second round for the same shortfall");
    check(plan(four, parked, 90 * G, 20 * G, lower, lim, true, true).empty(), "nothing more while enough is on its way");
    const Memory dire{1 * G, 0.0};    // under the hard floor
    check(severe(dire, lim) && !severe(lower, lim), "the hard floor is --conversation-cache-min-free-mib");
    check(said(plan(four, parked, 90 * G, 0, dire, lim, true, true)) == "1d:pressure 3d:pressure 2u:unsaved ",
          "under the hard floor an unwritten entry is dropped unsaved (17 GiB asked: 9 on disk, then the oldest unwritten)");
    check(said(plan({entry(7, 2, false, 5, true), entry(8, 2, true, 5, true)}, 4 * G, 0, 0, dire, lim, true, true)).empty(),
          "a pinned shared prefix is never released");
    check(plan(four, parked, 90 * G, 0, Memory{}, lim, true, true).empty(), "unknown memory telemetry releases nothing");
    {
        Limits off = lim;
        off.free_bytes = 0;
        check(plan(four, parked, 90 * G, 0, low, off, true, true).empty(), "SAGUARO_PARK_FREE_MIB=0 turns the floor off");
    }
    // memory stalls with RAM above the floor: one entry a round, only when the caller says a round is due
    const Memory stalled{40 * G, 23.5};
    check(said(plan(four, parked, 90 * G, 0, stalled, lim, true, true)) == "1d:pressure ", "memory stalls: the oldest entry on disk");
    check(plan(four, parked, 90 * G, 0, stalled, lim, true, false).empty(), "not again before the next round is due");
    check(said(plan({entry(2, 12, false)}, 12 * G, 90 * G, 0, stalled, lim, true, true)) == "2w:pressure ", "stalls and nothing on disk: written, then freed");
    {
        Limits off = lim;
        off.psi = 0;
        check(plan(four, parked, 90 * G, 0, stalled, off, true, true).empty(), "SAGUARO_PARK_PSI=0 ignores stalls");
        check(plan(four, parked, 90 * G, 0, Memory{40 * G, std::nullopt}, lim, true, true).empty(), "a kernel without PSI: no stalls");
    }
    // the idle limit (off unless set): only what its limit names, written first when not on disk
    {
        Limits idle = lim;
        idle.idle_s = 1800;
        const std::vector<Entry> e{entry(1, 4, true, 1830), entry(2, 12, false, 2000), entry(3, 5, true, 60), entry(4, 3, false, 60)};
        check(said(plan(e, parked, 90 * G, 0, calm, idle, true, true)) == "1d:idle 2w:idle ", "idle past the limit: dropped if on disk, else written first");
        check(said(plan(e, parked, 90 * G, 0, calm, idle, false, true)) == "1d:idle ", "never the only copy: without a folder an unwritten idle entry stays");
        check(said(plan(e, parked, 21 * G, 0, calm, idle, true, true)) == "1d:budget 2w:idle ", "one action an entry");
    }
    {
        std::istringstream psi("some avg10=12.34 avg60=3.00 avg300=1.00 total=123\nfull avg10=1.00 avg60=0.5 avg300=0.1 total=9\n");
        const auto v = psi_some10(psi);
        check(v && *v > 12.33 && *v < 12.35, "PSI: some avg10");
        std::istringstream none("full avg10=1.00\n"), bad("some total=5\n");
        check(!psi_some10(none) && !psi_some10(bad), "PSI: a file without the figure is unknown");
    }
    std::printf("park_policy_test: %d checks passed\n", checks);
    return 0;
}
