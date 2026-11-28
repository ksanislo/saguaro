// How much of the host the resident expert tier may claim (include/strata/core/resident_policy.hpp,
// PLAN-p124x): the margin, the runtime beside the tier, and the cap generate.cpp cuts an over-big
// --resident-budget-gib to.  Header-only rule, no GPU.
#include "strata/core/resident_policy.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::core::resident;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
constexpr uint64_t G = 1ull << 30;
}  // namespace

int main() {
    // the host margin: 1/8 of the host, never less than the 8 GiB a small host can still afford
    check(host_margin_bytes(96 * G) == 12 * G, "a 96 GiB host keeps its eighth (12 GiB) outside the tier");
    check(host_margin_bytes(32 * G) == 8 * G, "a 32 GiB host: the 8 GiB floor, not the 4 GiB eighth");
    check(host_margin_bytes(16 * G) == 8 * G, "a 16 GiB host: the floor is half of it (the cap then says no)");
    check(host_margin_bytes(0) == 8 * G, "unknown total: the floor stands; the cap for it is 0 below");

    // what the engine commits beside the tier: the shared K/V pool and the turn-save captures the pool bounds
    // (mortar measured RSS less tier = 18-21 GiB for the 10.2 GiB pool: two pool-sized passes)
    check(runtime_bytes(10 * G) == 20 * G, "runtime is twice the K/V pool");
    check(runtime_bytes(0) == 0, "no shared pool, no runtime beside the tier");

    // the cap: total - runtime - margin; 0 when nothing is left; unknown inputs derive no cap
    check(budget_cap(0, 10 * G) == 0, "an unknown host derives no cap (0: keep the asked budget's own rules)");
    check(budget_cap(96 * G, 0) == 96 * G - 12 * G, "no pool: only the margin is kept beside the tier");
    const uint64_t mortar_pool = 51 * G / 5;   // the mortar K/V pool: ~10.2 GiB (892 MiB per 45,845 tokens)
    const uint64_t cap = budget_cap(96 * G, mortar_pool);
    check(cap > 63 * G && cap < 64 * G, "the mortar host: 96 - 2x10.2 - 12 = 63.6 GiB of tier fits");
    check(cap < 68 * G, "and the asked 68 GiB is over that cap - the 89 GiB-RSS, swapped, aborting shape");
    check(budget_cap(24 * G, 10 * G) == 0, "a 24 GiB host cannot carry a 20 GiB runtime plus its margin (0)");
    check(budget_cap(40 * G, 16 * G) == 0, "exactly full still means nothing is left for a tier (0)");
    check(budget_cap(40 * G + 1, 16 * G) == 1, "one byte over the tight fit leaves exactly that byte");

    // P124e round 12 (manager card A): the margin counts the K/V the host copies COMMIT - backed + one
    // growth step of lead - never the --max-context ceiling.  The mortar shape of 2026-11-27: 91.74 GiB
    // host, 12700 B per position across the QSA layers, --batch 2, the elastic start and step 32768 cells,
    // 1.30 GiB pinned for K/V at load whatever --max-context says, the asked tier 68 GiB.
    const uint64_t host = (uint64_t) (91.74 * (double) G);
    const uint64_t asked = 68 * G;
    const uint64_t cell = 12700;
    const uint64_t ceil262 = cell * 262144ull * 2;    // what --max-context 262144 x 2 slots would take whole
    const uint64_t ceil1m = cell * 1048576ull * 2;    // ... and 1048576: the SAME 32768-cell start backed
    const uint64_t before262 = budget_cap(host, ceil262);
    const uint64_t before1m = budget_cap(host, ceil1m);
    check(before262 > (uint64_t) (67.5 * (double) G) && before262 < asked,
          "before the fix, 262144: the asked 68 GiB cut to ~67.9 by the ceiling-sized pool (mortar's 67.90)");
    check(before1m < before262 / 2, "before the fix, 1048576: the same host kept only ~30.8 GiB of tier - the regression");
    const uint64_t backed = (uint64_t) (1.30 * (double) G);
    const uint64_t lead = (uint64_t) (1.30 * (double) G);   // the start IS one step: the lead matches it
    check(committed_kv_bytes(backed, lead) == backed + lead, "committed = backed + the step it runs ahead");
    const uint64_t after262 = budget_cap(host, committed_kv_bytes(backed, lead));
    const uint64_t after1m = budget_cap(host, committed_kv_bytes(backed, lead));
    check(after262 == after1m, "after the fix the tier is sized the same at 262144 and 1048576: memory follows use");
    check(after1m >= asked, "and the asked 68 GiB stands at both contexts (mortar's pre-merge 68.00/68.00)");
    // a pinned-whole layout commits its whole context at load: counted as backed, it still sizes the margin
    const uint64_t whole1m = budget_cap(host, committed_kv_bytes(ceil1m, 0));
    check(whole1m == before1m, "pinned whole (CUDA0-on-HIP fallback, no elastic copies): the whole context is COMMITTED");
    std::printf("resident_policy_test: %d checks passed\n", checks);
    return 0;
}
