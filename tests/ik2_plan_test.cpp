// tests/ik2_plan_test.cpp - the elastic start step's sizing decision (src/ik2/ik2_plan.hpp), no GPU: the committed
// K / V of the global layers, the engine's fits arithmetic at the ceiling and at the commit, and when the elastic
// start is taken (the mortar rows of ref/REPORT-p125b-ik2-vram.md: 32768 resident, 262144 with 0 of 3072 experts in
// VRAM - and what a1 makes of the second row).
#include "../src/ik2/ik2_plan.hpp"

#include <cstdio>
#include <vector>

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
    expect(!elastic_start(false, true, false, 262144, kKvStartDefault, false, false, false, true), "not --tp");

    // P125o (a2 / a3): the chunked, growable commit.  Inkling's global visit: 4096 B a position in FP16 (2048 in
    // FP8), the usual 2 MiB granule, 4096 positions a chunk: 16 MiB - a whole number of granules already.
    expect(kKvChunkPos == 4096, "a chunk aims at 4096 positions");
    expect(kv_rows_per_chunk(kKvChunkPos, 4096, 2u << 20) == 4096, "the FP16 row: the granule leaves the chunk whole");
    expect(kv_rows_per_chunk(kKvChunkPos, 2048, 2u << 20) == 4096, "the FP8 row too (8 MiB of granules)");
    expect(kv_rows_per_chunk(kKvChunkPos, 2048, 16u << 20) == 8192, "a 16 MiB granule makes the chunk bigger, never smaller");
    expect(kv_rows_per_chunk(kKvChunkPos, 1024, 3u << 20) == 6144, "a 3 MiB granule: 6144 rows of 1 KiB");
    expect(kv_rows_per_chunk(kKvChunkPos, 4096, 0) == 4096, "no VMM: the exact rows");
    expect(kv_rows_per_chunk(0, 4096, 2u << 20) == 0, "no rows wanted: no chunk");
    expect(kv_rows_per_chunk(kKvChunkPos, 0, 2u << 20) == 0, "no row bytes: no chunk");
    // the chunk holding a position, and the chunks covering a request's positions: whole chunks, no gaps
    expect(kv_chunk_of(0, 4096) == 0, "position 0 sits in chunk 0");
    expect(kv_chunk_of(4095, 4096) == 0, "the last position of chunk 0");
    expect(kv_chunk_of(4096, 4096) == 1, "the first position of chunk 1");
    expect(kv_chunk_of(262143, 4096) == 63, "the last position of a 262144 ceiling sits in chunk 63");
    expect(kv_chunks_for(0, 4096) == 0, "no positions: no chunks");
    expect(kv_chunks_for(1, 4096) == 1, "one position commits its whole chunk");
    expect(kv_chunks_for(4096, 4096) == 1, "a chunk of positions is one chunk");
    expect(kv_chunks_for(4097, 4096) == 2, "one position past it takes the next chunk");
    expect(kv_chunks_for(60000, 4096) == 15, "a 60000-token read covers 15 chunks");
    expect(kv_chunks_for(262144, 4096) == 64, "the whole ceiling is 64 chunks");
    // the commit order: the load commits the start step, a request commits up to its last position - ascending,
    // each chunk once, nothing skipped (a position is only written after its chunk holds pages)
    {
        const int rpc = kv_rows_per_chunk(kKvChunkPos, 4096, 2u << 20);
        const int load_chunks = kv_chunks_for(kv_committed(262144, kKvStartDefault), rpc);
        expect(load_chunks == 8, "the load commits the start step: 32768 of 4096-row chunks");
        // the memory at the load does not follow the ceiling: the committed chunks, not the reserved range
        expect((size_t) load_chunks * (size_t) rpc * 4096u == (size_t) 32768 * 4096u,
               "the load commits 128 MiB a visit at 32768, whichever ceiling it reserves under");
        int committed = load_chunks;
        std::vector<int> order;
        const int want = kv_chunks_for(20000, rpc);   // a 20K read: already inside the commit - nothing to do
        for (int c = committed; c < want; ++c) order.push_back(c);
        expect(order.empty(), "a read inside the commit commits nothing new");
        const int grow = kv_chunks_for(60000, rpc);   // a 60K read past it: chunks 8 .. 14, in order
        for (int c = committed; c < grow; ++c) order.push_back(c);
        expect(order.size() == 7 && order.front() == 8 && order.back() == 14, "the 60K read commits chunks 8 .. 14");
        bool ascending = true;
        for (size_t i = 1; i < order.size(); ++i) ascending = ascending && order[i] == order[i - 1] + 1;
        expect(ascending, "the commit order is the position order, one chunk at a time");
        committed = grow;
        expect(committed == 15, "the commit advances to where the conversation reached");
    }

    std::printf("ik2_plan_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
