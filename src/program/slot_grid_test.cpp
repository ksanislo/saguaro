#include "strata/program/slot_grid.hpp"

#include <cstdio>
#include <vector>

// Expected values computed with an independent Python model of the same rule (bucket by distance, oldest + newest).
int main() {
    using namespace strata::program;
    int checks = 0, bad = 0;
    auto eq = [&](const std::vector<int64_t>& a, const std::vector<int64_t>& b, const char* what) {
        ++checks;
        if (a != b) {
            ++bad;
            std::printf("FAIL %s:", what);
            for (int64_t x : a) std::printf(" %lld", (long long) x);
            std::printf("\n");
        }
    };
    constexpr int64_t E = 16384;
    {   // buckets: last `every` tokens = -1, then one per doubling, the last one takes everything further
        const int64_t d[] = {0, 16383, 16384, 32767, 32768, 65536, 600000};
        const int want[] = {-1, -1, 0, 0, 1, 2, 4};
        for (int i = 0; i < 7; ++i) {
            ++checks;
            if (slot_grid_bucket(d[i], E, 6) != want[i]) { ++bad; std::printf("FAIL bucket %lld\n", (long long) d[i]); }
        }
    }
    // a fresh read of a 424K conversation saves 10 points, not one every 16K
    const std::vector<int64_t> fresh = slot_grid_new(0, 424000, E, 6);
    eq(fresh, {16384, 147456, 163840, 278528, 294912, 344064, 360448, 376832, 393216, 409600}, "fresh 424K");
    // the next turn (+23K) reads [424000, 447000): two new points beside the held ones
    const std::vector<int64_t> inc = slot_grid_new(424000, 447000, E, 6, fresh);
    eq(inc, {425984, 442368}, "incremental turn");
    // and at 447000 the slot keeps these (360448 has aged into a bucket whose oldest and newest are others)
    std::vector<int64_t> held = fresh;
    held.insert(held.end(), inc.begin(), inc.end());
    std::vector<int64_t> kept;
    for (size_t i : slot_grid_keep(held, 447000, E, 6)) kept.push_back(held[i]);
    eq(kept, {16384, 163840, 278528, 294912, 344064, 376832, 393216, 409600, 425984, 442368}, "kept at 447K");
    eq(slot_grid_new(0, 16000, E, 6), {}, "shorter than one step: none");
    eq(slot_grid_new(0, 40000, E, 6), {16384, 32768}, "40K");
    // never at or past the end, never non-positive; buckets < 2 or every <= 0: none
    ++checks;
    if (!slot_grid_keep({500000, -1, 0}, 424000, E, 6).empty()) { ++bad; std::printf("FAIL past end\n"); }
    ++checks;
    if (!slot_grid_keep({1, 2}, 100, E, 1).empty() || !slot_grid_keep({1, 2}, 100, 0, 6).empty()) {
        ++bad;
        std::printf("FAIL disabled\n");
    }
    // equal lengths in one bucket keep one of them
    ++checks;
    if (slot_grid_keep({163840, 163840}, 424000, E, 6).size() != 1) { ++bad; std::printf("FAIL duplicates\n"); }
    if (bad) return 1;
    std::printf("slot_grid_test OK (%d checks)\n", checks);
}
