// P124e (round 10, manager card): the SAGUARO_DEBUG_SLOT_GUARD crossing check (include/strata/core/
// slot_give_guard.hpp) - a give of an expert-cache slot or chunk against (a) the ACTIVE prompt loan
// [p.first_now, slots) of every part on that cache and (b) the region the OTHER give-way already handed out.
// The card's two scenarios, run against the pure check: the loan re-sized after setup (a give that reaches
// into it), and the two give-ways counting DOWN from the same line (the second one's give crosses the first's
// region).  The line must name the path, the slots/chunks and who owns them - the card run reads the culprit
// off one line of the core's log.  No GPU, no abort in-process: the crossing itself is tested through the
// report (which prints and returns true); the abort is the caller's std::abort of that same line.
#include "strata/core/slot_give_guard.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

int cache0, cache1;   // stand-ins for two ExpertCache objects (identity is all the guard uses)

bool has(const std::string& w, const char* needle) { return w.find(needle) != std::string::npos; }

// report() into a throwaway buffer: true + exactly the printed line on a crossing, false + nothing when clean.
std::string reported(const void* cache, const GiveOp& op, const std::vector<GiveLoan>& loans,
                     const std::vector<GiveRegion>& regions) {
    FILE* t = tmpfile();
    const bool crossed = slot_give_guard_report(t, cache, op, loans, regions);
    std::string w;
    if (crossed) {
        std::fflush(t);
        char buf[4096];
        std::rewind(t);
        size_t n = 0;
        while ((n = std::fread(buf, 1, sizeof buf - 1, t)) > 0) { buf[n] = 0; w += buf; }
        check(std::count(w.begin(), w.end(), '\n') == 1, "a crossing prints ONE line");
    }
    std::fclose(t);
    return w;
}
}  // namespace

int main() {
    // clean: nothing loaned, nothing given.
    {
        check(slot_give_guard_line(&cache0, {"gv CUDA0", "gives a slot", 100, 2048, 70, 140}, {}, {}).empty(),
              "an op with nothing to cross is clean");
    }
    // (a) THE ACTIVE LOAN, including re-sized AFTER setup: the kvg line was fixed at slot 1800 at setup, the
    // request re-laid its loan to 1500 - a give from 1600 now reaches into the read's buffers.
    {
        const std::vector<GiveLoan> loans = {{&cache0, "CUDA0 part 0", 1500, 2048}};
        const std::string w = slot_give_guard_line(&cache0, {"kvg CUDA0", "gives a slot to the K/V", 1600, 1601,
                                                             -1, -1}, loans, {});
        check(!w.empty() && has(w, "ACTIVE LOAN") && has(w, "CUDA0 part 0"), "a give into the loan crosses");
        check(has(w, "kvg CUDA0") && has(w, "slot 1600") && has(w, "from slot 1500"),
              "the line names the path, the slot and the loan's start");
        check(has(w, "SAGUARO_DEBUG_SLOT_GUARD") && has(w, "aborting"), "and what the guard will do");
        check(slot_give_guard_line(&cache0, {"kvg CUDA0", "gives a slot", 1499, 1500, -1, -1}, loans, {}).empty(),
              "the slot just below the loan does not touch it");
        check(slot_give_guard_line(&cache0, {"kvg CUDA0", "gives a slot", 1700, 1701, -1, -1},
                                   {{&cache1, "CUDA1 part 0", 1500, 2048}}, {}).empty(),
              "another cache's loan is none of this give's business");
        check(slot_give_guard_line(&cache0, {"gv CUDA0", "gives slots", -1, -1, 100, 120}, loans, {}).empty(),
              "an op that cannot name its slots is not accused of reaching into a loan");
    }
    // (b) THE OTHER GIVE-WAY, counting down from the SAME line: gv took [2000, 2048) first; kvg's give of
    // slot 2030 hands the same chunks out a second time.
    {
        const std::vector<GiveRegion> regions = {{&cache0, "gv CUDA0", 2000, 2048, 1430, 1465}};
        const std::string w = slot_give_guard_line(&cache0, {"kvg CUDA0", "unmaps chunks for the K/V",
                                                             2030, 2048, 1452, 1465}, {}, regions);
        check(!w.empty() && has(w, "ALREADY handed out by 'gv CUDA0'"), "the second give-way crosses");
        check(has(w, "kvg CUDA0") && has(w, "slot 2030") && has(w, "chunk 1452"),
              "the line names the path, the slot and the chunk");
        check(has(w, "slots [2000, 2048), chunks [1430, 1465)"), "and the region it belongs to");
        check(slot_give_guard_line(&cache0, {"gv CUDA0", "gives a slot", 2010, 2011, 1437, 1438}, {}, regions)
                  .empty(),
              "a give-way never crosses its OWN region");
        check(slot_give_guard_line(&cache0, {"kvg CUDA0", "gives a slot", 1999, 2000, -1, -1}, {}, regions)
                  .empty(),
              "the slot just below the other give-way's region is still the cache's");
        // chunk-only crossing (a release to the driver names chunks, not the slots they came from):
        const std::string wc = slot_give_guard_line(&cache0, {"gv CUDA0", "frees budget chunks to the driver",
                                                              -1, -1, 1460, 1470}, {},
                                                    {{&cache0, "kvg CUDA0", 2000, 2048, 1430, 1465}});
        check(!wc.empty() && has(wc, "chunk 1460") && has(wc, "'kvg CUDA0'"), "a chunk-only op crosses on its chunks");
        check(slot_give_guard_line(&cache0, {"kvg CUDA0", "gives a slot", 2030, 2048, 1452, 1465}, {},
                                   {{&cache1, "gv CUDA1", 2000, 2048, 1430, 1465}})
                  .empty(),
              "another cache's given region is not this one's");
    }
    // THE LOAN'S OWN RELAYOUT: the new region re-lays buffers over slots the K/V give-way gave - (a) skips the
    // part whose loan it is (the caller omits it), (b) catches the crossed region.
    {
        const std::vector<GiveRegion> regions = {{&cache0, "kvg CUDA0", 1600, 1800, 1148, 1291}};
        const std::string w = slot_give_guard_line(&cache0, {"loan relayout", "CUDA0 relays the prompt buffers",
                                                             1500, 2048, 1076, 1465}, {}, regions);
        check(!w.empty() && has(w, "ALREADY handed out by 'kvg CUDA0'") && has(w, "slot 1600"),
              "a relayout over given slots crosses");
        const std::string w2 = slot_give_guard_line(&cache0, {"loan relayout", "CUDA0 relays the prompt buffers",
                                                              1800, 2048, 1291, 1465}, {}, regions);
        check(w2.empty(), "a relayout that stays above the given region is clean");
    }
    // (a) outranks (b): a give into BOTH the loan and the other give-way's region says the loan first
    // (it is the active read's memory).
    {
        const std::vector<GiveLoan> loans = {{&cache0, "CUDA0 part 0", 1500, 2048}};
        const std::vector<GiveRegion> regions = {{&cache0, "gv CUDA0", 1400, 2048, 1000, 1465}};
        const std::string w = slot_give_guard_line(&cache0, {"kvg CUDA0", "gives a slot", 1600, 1601, -1, -1},
                                                   loans, regions);
        check(has(w, "ACTIVE LOAN") && !has(w, "ALREADY handed out"), "the loan is named first");
    }
    // move-into: the expert copy's DESTINATION slot inside a loan (the loan region is not the cache's to
    // shuffle experts through, mapped or not).
    {
        const std::vector<GiveLoan> loans = {{&cache0, "CUDA0 part 0", 1900, 2048}};
        const std::string w = slot_give_guard_line(&cache0, {"gv CUDA0", "moves a hotter expert into a slot",
                                                             1950, 1951, -1, -1}, loans, {});
        check(!w.empty() && has(w, "ACTIVE LOAN") && has(w, "slot 1950"), "a move into the loan crosses");
        check(slot_give_guard_line(&cache0, {"gv CUDA0", "moves an expert", 1899, 1900, -1, -1}, loans, {})
                  .empty(),
              "and the slot below it does not");
    }
    // the printed line: report() writes it once; a clean op writes nothing.
    {
        const std::vector<GiveLoan> loans = {{&cache0, "CUDA0 part 0", 1500, 2048}};
        const std::string w = reported(&cache0, {"kvg CUDA0", "gives a slot", 1600, 1601, 1148, 1150}, loans, {});
        check(has(w, "saguaro SLOT GUARD") && has(w, "chunks [1148, 1150)"),
              "the printed line names the chunk range too when the op has one");
        check(reported(&cache0, {"kvg CUDA0", "gives a slot", 1200, 1201, -1, -1}, loans, {}).empty(),
              "a clean op prints nothing");
    }
    // OFF is free: with SAGUARO_DEBUG_SLOT_GUARD unset (the default everywhere but the card) even a crossing
    // op returns without a sound.  (When the env IS set - the manager's debug run - this is exactly what must
    // NOT happen in a unit test, so it is skipped; the card is the abort's test.)
    {
        if (!slot_give_guard_on()) {
            slot_give_guard(&cache0, {"kvg CUDA0", "gives a slot", 1600, 1601, -1, -1},
                            {{&cache0, "CUDA0 part 0", 1500, 2048}}, {});
            check(true, "off: a crossing op does nothing (no line, no abort)");
        } else {
            std::fprintf(stderr, "note: SAGUARO_DEBUG_SLOT_GUARD is set in this environment: "
                                 "skipping the off-behavior check (the abort itself is the card's test)\n");
        }
    }

    std::printf("slot_give_guard_test: %d checks OK\n", checks);
    return 0;
}
