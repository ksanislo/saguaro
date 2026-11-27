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
    std::printf("resident_policy_test: %d checks passed\n", checks);
    return 0;
}
