// The card's own VRAM floor (include/strata/core/vram_policy.hpp, PLAN-p124x): the #2065 paging line the
// stall report prints, and the reserve the expert cache's auto sizing keeps on a Linux/ROCm discrete card
// (the Linux sibling of #1376's Windows-only kWddmAutoReserveMib).  Header-only rule, no GPU.
#include "strata/core/vram_policy.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::core::vram;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
constexpr uint64_t G = 1ull << 30;
}  // namespace

int main() {
#ifdef _WIN32
    check(safe_floor_mib() == 1024, "WDDM pages instead of failing: the wider 1024 MiB line");
#else
    check(safe_floor_mib() == 512, "the Linux paging floor the LOW: line prints (#2065): 512 MiB");
#endif
    const uint64_t xt = 159ull * G / 10;   // the RX 9070 XT reports 15.9 GiB (the #1709 near-miss)
    check(hip_auto_reserve_mib(false, false, false, xt, 700) == 700, "a CUDA build: never raised (#1376's A/B arm)");
    check(hip_auto_reserve_mib(true, true, false, xt, 700) == 700, "Windows keeps its own kWddmAutoReserveMib path");
    check(hip_auto_reserve_mib(true, false, true, 64 * G, 700) == 700,
          "an integrated APU is left alone: its total is GTT RAM, not a card (review N3)");
    check(hip_auto_reserve_mib(true, false, false, 12 * G, 200) == 200, "a small card keeps its #496 lowering");
    check(hip_auto_reserve_mib(true, false, false, 0, 700) == 700, "a total that cannot be read: nothing is floored");
    check(hip_auto_reserve_mib(true, false, false, xt, 700) == 1024,
          "the 9070 XT shape: 700 asked, floor + measured slack kept (512 + 512: down to 416-432 was the abort)");
    check(hip_auto_reserve_mib(true, false, false, 15 * G, 700) == 1024, "exactly 15 GiB is on the #1709 threshold");
    check(hip_auto_reserve_mib(true, false, false, xt, 1023) == 1024, "one under the floor is raised to it");
    check(hip_auto_reserve_mib(true, false, false, xt, 1024) == 1024, "at the floor: kept as it is");
    check(hip_auto_reserve_mib(true, false, false, xt, 2600) == 2600, "a bigger reserve is never lowered");
    check(hip_auto_reserve_mib(true, false, false, xt, 0) == 1024,
          "a reserve GIVEN below the floor is floored too: this floor does not step back (#496 lowers, this raises)");
    std::printf("vram_policy_test: %d checks passed\n", checks);
    return 0;
}
