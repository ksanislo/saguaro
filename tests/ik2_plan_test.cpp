// tests/ik2_plan_test.cpp - the elastic start step's sizing decision (src/ik2/ik2_plan.hpp), no GPU: the committed
// K / V of the global layers, the engine's fits arithmetic at the ceiling and at the commit, and when the elastic
// start is taken (the mortar rows of ref/REPORT-p125b-ik2-vram.md: 32768 resident, 262144 with 0 of 3072 experts in
// VRAM - and what a1 makes of the second row); the chunk arithmetic of the growable commit (P125o a2 / a3); and the
// give-back's high-water arithmetic and free-chunk list (P125u a4).
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

    // P125u (a4): the give-back.  The live high-water across the slots: the largest position a live conversation
    // can still read (attention reads [0, pos); an ended one's rows live in host RAM, and so does a snapshot's
    // rewound state) - everything above it goes back.
    expect(kv_high_water({}) == 0, "no live slot: the high-water is 0");
    expect(kv_high_water({0}) == 0, "an empty conversation reads nothing");
    expect(kv_high_water({-1, -1}) == 0, "no snapshot anywhere does not pull it below 0");
    expect(kv_high_water({4096, 123}) == 4096, "the slot that reached furthest sets it");
    expect(kv_high_water({70000, -1, 8000, 65536}) == 70000, "a slot past an empty snapshot");
    expect(kv_high_water({512}) == 512, "one cache conversation, no snapshots");
    // the give-back lands on whole chunks: keep the ones covering the high-water, return the rest
    {
        const int rpc = kv_rows_per_chunk(kKvChunkPos, 4096, 2u << 20);
        expect(kv_chunks_for(kv_high_water({60000, 40000}), rpc) == 15, "a 60K high-water keeps chunks 0 .. 14");
        expect(kv_chunks_for(kv_high_water({4097, -1}), rpc) == 2, "one position into chunk 1 keeps it whole");
        expect(kv_chunks_for(kv_high_water({}), rpc) == 0, "an ended conversation keeps no chunk of its own");
    }
    // the free-chunk list: the ceiling's chunks split into the committed prefix and the free rest (every buffer
    // commits in lockstep); the 262144 ceiling of 4096-position chunks, the 32768 start step
    {
        KvFreeChunks fl;
        fl.n_total = kv_chunks_for(262144, 4096);
        expect(fl.n_total == 64 && fl.free() == 64, "the range: 64 chunks, all free before the load commits");
        expect(fl.commit(kv_chunks_for(32768, 4096)), "the load takes the start step's chunks from the list");
        expect(fl.used == 8 && fl.free() == 56, "8 committed, 56 free at the 32768 start step");
        expect(!fl.commit(71), "past the ceiling's range: nothing is taken");
        expect(fl.used == 8, "a refused commit leaves the list where it was");
        expect(fl.commit(15), "a 60K read takes chunks 8 .. 14");
        expect(fl.used == 15 && fl.free() == 49, "the grown conversation holds 15 chunks");
        expect(fl.commit(15), "committing what is already committed takes nothing and succeeds");
        expect(fl.used == 15, "a commit never unmaps");
        expect(fl.give_back_to(15) == 0, "a conversation ending where the commit stands gives nothing back");
        expect(fl.give_back_to(20) == 0, "nothing above the committed prefix needs returning");
        expect(fl.used == 15, "and the list stands");
        expect(fl.give_back_to(10) == 5, "a rewind to 40960 gives chunks 10 .. 14 back");
        expect(fl.used == 10 && fl.free() == 54, "the free rest grows by what went back");
        expect(fl.commit(11), "the next request takes a freed chunk again");
        expect(fl.used == 11 && fl.free() == 53, "commit is not the high-water: it re-maps from the list");
        expect(fl.give_back_to(0) == 11, "a session end gives every chunk back");
        expect(fl.used == 0 && fl.free() == 64, "the whole range stands free");
        expect(fl.give_back_to(-1) == 0, "a negative target on an empty list gives nothing");
        expect(fl.commit(64), "the whole ceiling can be committed");
        expect(fl.free() == 0, "nothing free at the ceiling");
        expect(fl.give_back_to(-1) == 64, "a negative target is 0: everything goes");
    }
    // the serve loop's arithmetic in one walk: the 262144 row, a request that commits far past where its answer
    // stops (a big max_tokens) and a new conversation after it
    {
        KvFreeChunks fl;
        fl.n_total = kv_chunks_for(262144, 4096);
        fl.commit(kv_chunks_for(32768, 4096));
        const int asked = kv_chunks_for(std::min(60000 + 230000, 262144), 4096);   // (the guard clamps to the ceiling)
        expect(fl.commit(asked) && fl.used == 64, "the 60K ask with a huge max_tokens commits the whole range");
        const int answered = 60000 + 40;   // (the answer stopped early - the state the request settles into)
        expect(fl.give_back_to(kv_chunks_for(answered, 4096)) == 49,
               "at its end only the chunks past the answer go back, the max_tokens slack with them");
        expect(fl.used == 15 && fl.free() == 49, "the card keeps what the conversation still reads");
        expect(fl.give_back_to(kv_chunks_for(kv_high_water({0, -1, -1}), 4096)) == 15,
               "the next request finds a new conversation: reset leaves nothing live, all 15 go back");
        expect(fl.free() == 64, "and the grown VRAM is the driver's again");
    }

    std::printf("ik2_plan_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
