// tests/ik2_plan_test.cpp - the elastic start step's sizing decision (src/ik2/ik2_plan.hpp), no GPU: the committed
// K / V of the global layers, the engine's fits arithmetic at the ceiling and at the commit, and when the elastic
// start is taken (the mortar rows of ref/REPORT-p125b-ik2-vram.md: 32768 resident, 262144 with 0 of 3072 experts in
// VRAM - and what a1 makes of the second row).
#include "../src/ik2/ik2_plan.hpp"

#include <cstdio>

namespace {

const size_t GiB = (size_t) 1 << 30;
int fails = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        ++fails;
        std::printf("FAIL %s\n", what);
    }
}

}  // namespace

int main() {
    using namespace ik2plan;

    // the committed start step: the conversation's, never past the ceiling; 0 (or past it) is today's layout
    expect(kv_committed(262144, kKvStartDefault) == 32768, "commit: 262144 ceiling starts at 32768");
    expect(kv_committed(32768, kKvStartDefault) == 32768, "commit: a ceiling at the start step is itself");
    expect(kv_committed(8192, kKvStartDefault) == 8192, "commit: a smaller ceiling stays smaller");
    expect(kv_committed(262144, 0) == 262144, "commit: SAGUARO_IK2_KV_START=0 keeps the whole ceiling");
    expect(kv_committed(262144, -1) == 262144, "commit: a negative start is off, not negative");
    expect(kv_committed(65536, 131072) == 65536, "commit: a start past the ceiling is the ceiling");

    // the fits arithmetic: the whole cache of the ceiling vs the committed one, on the mortar card (the one-T4
    // quant's 11.45 GiB of experts, 1.8 GiB of dense blocks (the embedding is not counted), 1.5 GiB reserve,
    // 15.9 GiB free, f16 K / V of 32768 B a position across the global visits: 8 GiB at 262144, 1 GiB at 32768)
    const size_t exp_b = (size_t) (11.45 * GiB), dense_b = (size_t) (1.8 * GiB), res = (size_t) (1.5 * GiB);
    const size_t fr0 = (size_t) (15.9 * GiB);
    const size_t kv_262k = (size_t) 262144 * 32768, kv_32k = (size_t) 32768 * 32768;
    expect(experts_fit(exp_b, dense_b, kv_32k, res, fr0, 24.0), "32768: every expert fits beside the cache");
    expect(!experts_fit(exp_b, dense_b, kv_262k, res, fr0, 24.0), "262144: not beside the cache's ceiling");
    expect(experts_fit(exp_b, dense_b, kv_262k, res, (size_t) 24.0 * GiB, 24.0), "a bigger card: the ceiling fits");
    expect(!experts_fit((size_t) 20.0 * GiB, dense_b, kv_32k, res, fr0, 16.0), "cache-gb caps the experts too");

    // the elastic start: only the second mortar row takes it - and only where nothing else streams or splits
    const bool f_start = experts_fit(exp_b, dense_b, kv_32k, res, fr0, 24.0);
    expect(f_start, "the committed layout fits on the mortar card (the experts stay)");
    expect(elastic_start(false, true, false, 262144, kKvStartDefault, false, false, false, false),
           "the 262144 row plans the elastic start");
    expect(!elastic_start(true, true, false, 262144, kKvStartDefault, false, false, false, false),
           "where the ceiling fits, nothing changes");
    expect(!elastic_start(false, false, false, 262144, kKvStartDefault, false, false, false, false),
           "where not even the commit fits, the card is simply too small");
    expect(!elastic_start(false, true, false, 32768, kKvStartDefault, false, false, false, false),
           "a ceiling at the commit is not elastic (the commit is the ceiling)");
    expect(!elastic_start(false, true, false, 262144, 0, false, false, false, false),
           "SAGUARO_IK2_KV_START=0 keeps today's layout");
    expect(!elastic_start(false, true, true, 262144, kKvStartDefault, false, false, false, false),
           "KV streaming holds the whole ceiling in host RAM: no elastic start");
    expect(!elastic_start(false, true, false, 262144, kKvStartDefault, true, false, false, false), "not two GPUs");
    expect(!elastic_start(false, true, false, 262144, kKvStartDefault, false, true, false, false), "not q4 rows");
    expect(!elastic_start(false, true, false, 262144, kKvStartDefault, false, false, true, false), "not --parallel");
    expect(!elastic_start(false, true, false, 262144, kKvStartDefault, false, false, false, true), "not --tp");

    std::printf("ik2_plan_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
