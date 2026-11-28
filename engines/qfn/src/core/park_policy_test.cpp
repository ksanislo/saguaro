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
    // PLAN-p124a (mortar, 2026-11-25): the resident RAM mode pins 68 GiB of experts + the K/V pool and the host is
    // wide open - the reading counts our own tier as used, and the old rule dropped every park seconds after it
    // landed while `free` showed 81 GB.  The engine's own pinned RAM is credited for the release decision.
    Memory mortar{8 * G, 0.0};           // what the sampler reads with the 68 GiB charged against the limit
    mortar.own_pinned = 68 * G;          // the tier + pool: ours, and no release gives them back
    check(pressure_need(mortar, lim) == 0, "the engine's own pinned RAM is credited: 8 GiB free of a 96 GB host is not pressure");
    check(plan(four, parked, 90 * G, 0, mortar, lim, true, true).empty(), "mortar's figures: the parks stay, nothing is released");
    {
        Memory raw = mortar;
        raw.own_pinned = 0;   // the old reading, or an engine pinning nothing of its own
        check(pressure_need(raw, lim) == 10 * G && !plan(four, parked, 90 * G, 0, raw, lim, true, true).empty(),
              "without the credit the same 8 GiB under the 16 GiB floor is still pressure (8 + an eighth asked)");
    }
    {
        // the credit stops at the hard floor (review F1): the old check only asserted `severe` and never called
        // plan(), while the credited pressure_need kept `pressed` false forever and no release ever ran
        Memory deep = mortar;
        deep.available = 1 * G;   // the host genuinely full: under the hard floor the raw reading decides
        check(pressure_need(deep, lim) == 17 * G && severe(deep, lim),
              "a raw 1 GiB under the hard floor asks the raw shortfall back (16 - 1 + an eighth): the credit does not silence it");
        check(said(plan(four, parked, 90 * G, 0, deep, lim, true, true)) == "1d:pressure 3d:pressure 2u:unsaved ",
              "and plan() releases at raw 1 GiB with 68 GiB pinned: on disk first, the unwritten one dropped unsaved");
        Memory above = mortar;
        above.available = 3 * G;   // back above the hard floor: the credit works again
        check(pressure_need(above, lim) == 0 && !severe(above, lim) &&
              plan(four, parked, 90 * G, 0, above, lim, true, true).empty(),
              "1 GiB over the hard floor is the resident mode working again, not pressure");
    }
    // PLAN-p124x: swap in use above the bar voids the credit - the mortar box ran 2-5 GiB of swap behind a
    // credited-calm reading while the releases did nothing (report 1.4), so "the kernel already moved working
    // pages to disk" is what the parks must answer to, however much the engine itself pins.
    {
        check(lim.swap_bytes == 256ull << 20,
              "the swap bar is 256 MiB: above a calm box's residual dust, under every mortar storm (392 MiB+)");
        check(!swap_high(mortar, lim), "no swap reading: only the floor rules decide, as before");
        Memory dust = mortar;
        dust.swap_used = 64ull << 20;   // residual swap under the bar: idle daemons paged out once
        check(!swap_high(dust, lim) && pressure_need(dust, lim) == 0 &&
              plan(four, parked, 90 * G, 0, dust, lim, true, true).empty(),
              "64 MiB of residual swap is dust: the credit holds and the parks stay");
        Memory edge = dust;
        edge.swap_used = lim.swap_bytes;
        check(!swap_high(edge, lim), "swap exactly at the bar is not above it");
        Memory storm = mortar;
        storm.swap_used = 2 * G;   // the mortar shape
        check(swap_high(storm, lim), "swap above the bar: the host is out of RAM, the pinned tier gives no calm");
        check(pressure_need(storm, lim) == 10 * G,
              "the raw reading decides again: the same 8 GiB under the 16 GiB floor asks 8 + an eighth back");
        check(said(plan(four, parked, 90 * G, 0, storm, lim, true, true)) == "1d:pressure 3d:pressure 2w:pressure ",
              "68 GiB pinned and 2 GiB in swap: on disk first, then written - the parks give back before the storm");
        Limits off = lim;
        off.swap_bytes = 0;   // SAGUARO_PARK_SWAP_MIB=0
        check(!swap_high(storm, off) && pressure_need(storm, off) == 0 &&
              plan(four, parked, 90 * G, 0, storm, off, true, true).empty(),
              "the swap rule is off when its env says 0: exactly the P124a behaviour");
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
    // PLAN-p124a2: adopting the turn save's host capture instead of copying the K/V a second time. Only the
    // conjunction adopts: a live whole capture, one stage, nothing retained over it, nothing run since, the same
    // tokens, no images. The mortar case: every one of the six true, and every single one of them false refuses.
    check(adopt_turn_capture(true, true, true, true, true, true),
          "a whole capture the park follows: the park adopts it, no second copy");
    check(!adopt_turn_capture(false, true, true, true, true, true), "no capture alive (its file write ended): capture normally");
    check(!adopt_turn_capture(true, false, true, true, true, true), "a layer split parks per stage: capture normally");
    check(!adopt_turn_capture(true, true, false, true, true, true), "the retained K/V is the source already: it keeps its prefix");
    check(!adopt_turn_capture(true, true, true, false, true, true), "a request ran since the capture: the session is no longer those bytes");
    check(!adopt_turn_capture(true, true, true, true, false, true), "another conversation, or more tokens: the ids say no");
    check(!adopt_turn_capture(true, true, true, true, true, false), "images on either side: the capture is not compared against them");
    // PLAN-p124a2 (manager review 2026-11-26): the RUNNING STATE is reused on these five terms alone - the retained
    // K/V is the sixth adoption term only. Where adoption refuses for retention (or later, for a part that will not
    // alias), the park still takes the capture's host copy of the running state instead of the device copy.
    check(reuse_turn_state(true, true, true, true, true), "the capture's host running state stands in, retained K/V or not");
    check(!reuse_turn_state(false, true, true, true, true), "no capture held: the park copies the running state itself");
    check(!reuse_turn_state(true, false, true, true, true), "a layer split: each stage's running state is its own");
    check(!reuse_turn_state(true, true, false, true, true), "a request ran since the capture: its bytes are stale");
    check(!reuse_turn_state(true, true, true, false, true), "other tokens: the host copy is not this conversation's");
    check(!reuse_turn_state(true, true, true, true, false), "images: the capture is not compared against them");
    check(!adopt_turn_capture(true, true, false, true, true, true) && reuse_turn_state(true, true, true, true, true),
          "retained K/V: no whole adoption, but the running state reuse still holds (the slow park's shape)");
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
