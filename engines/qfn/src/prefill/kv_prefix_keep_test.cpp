// kv_prefix_keep_test: the read-prefix keep's bookkeeping (include/strata/prefill/kv_prefix_keep.hpp).
// Pure: the view a layer pass copies, what the commit records, and every way the record is stale. No CUDA.
#include "strata/prefill/kv_prefix_keep.hpp"

#include <cstdio>

using strata::prefill::KvKeepRec;
using strata::prefill::KvKeepView;
using strata::prefill::kv_keep_commit;
using strata::prefill::kv_keep_invalidate;
using strata::prefill::kv_keep_view;
using strata::prefill::kv_keep_wanted;

static int g_checks = 0, g_fail = 0;
static void expect(bool ok, const char* what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

int main() {
    constexpr int64_t ps = 4, region = 1000;

    // the env policy: off unless SAGUARO_KV_PREFIX_KEEP asks for it
    expect(!kv_keep_wanted(nullptr), "unset: off (the default path stays 0.2.3's)");
    expect(kv_keep_wanted("1"), "SAGUARO_KV_PREFIX_KEEP=1: on");
    expect(!kv_keep_wanted("0"), "SAGUARO_KV_PREFIX_KEEP=0: off");
    expect(!kv_keep_wanted(""), "an empty value is off");

    // a region that never staged anything stages the whole prefix (today's pass), and the commit records the
    // chunk's full blocks - the first pass of the first read of a fresh conversation
    {
        KvKeepRec rec;
        KvKeepView v = kv_keep_view(true, false, rec.staged_blocks, region, 0, 256, ps);
        expect(v.from_block == 0 && v.to_block == 0 && !v.checked, "fresh conversation at p0 0: nothing to stage");
        expect(v.staged_after == 64, "a 256-cell chunk commits 64 whole blocks");
        kv_keep_commit(rec, (const void*) 1, (const void*) 10, 1, v);
        expect(rec.staged_blocks == 64 && rec.fp_valid, "commit: staged and a watermark to check");
        // the next read of the same conversation: continuation, prefix still equals the host copy
        v = kv_keep_view(true, true, rec.staged_blocks, region, 256, 256, ps);
        expect(v.from_block == 64 && v.to_block == 64, "continuation: the prefix stays, the aligned chunk copies nothing");
        expect(v.skipped_blocks == 64 && v.checked, "64 blocks skipped, by a fingerprint check");
        expect(v.staged_after == 128, "and 64 more blocks committed");
        kv_keep_commit(rec, (const void*) 1, (const void*) 10, 1, v);
        expect(rec.staged_blocks == 128, "staged grows");
    }

    // an unaligned chunk: the partial page is staged again whole (as the full staging always did) before the
    // append completes it - blocks staged_after counts whole blocks only
    {
        KvKeepRec rec;
        rec.staged_blocks = 64;
        KvKeepView v = kv_keep_view(true, true, rec.staged_blocks, region, 258, 300, ps);   // p0 258: mid-page
        expect(v.from_block == 64, "the full prefix up to block 64 is kept");
        expect(v.to_block == 65, "the page holding cell 258 is staged again (partial page, as before)");
        expect(v.staged_after == (258 + 300) / 4, "commit: whole blocks of the chunk's end");
    }

    // every way the record says nothing: identity, fingerprint, rewind, coverage
    {
        const int64_t staged = 64;
        expect(kv_keep_view(false, true, staged, region, 1000, 4, ps).from_block == 0,
               "another conversation (session/host/format): stage it all");
        expect(kv_keep_view(true, false, staged, region, 1000, 4, ps).from_block == 0,
               "the watermark's bytes changed (restore, rejected tokens): stage it all");
        expect(kv_keep_view(true, true, staged, region, staged * ps - 1, 4, ps).from_block == 0,
               "a read starting inside the staged prefix (rewind, branch): stage it all");
        expect(kv_keep_view(true, true, region + 5, region, 100000, 4, ps).from_block == 0,
               "a record past the region's coverage: stage it all");
        // exactly at the watermark's cell is a continuation, not a rewind
        expect(kv_keep_view(true, true, staged, region, staged * ps, 4, ps).from_block == staged,
               "a read starting at the watermark's first cell: the prefix is kept");
    }

    // p0 one cell inside the last staged page: the record must not skip what the conversation no longer holds
    {
        KvKeepRec rec;
        rec.staged_blocks = 10;
        KvKeepView v = kv_keep_view(true, true, rec.staged_blocks, region, 10 * 4, 4, ps);
        expect(v.from_block == 10, "p0 == 40 with staged 10 pages: exact boundary, kept");
        v = kv_keep_view(true, true, rec.staged_blocks, region, 10 * 4 - 1, 4, ps);
        expect(v.from_block == 0, "p0 one cell lower: the last page's bytes are not trusted");
    }

    // an append that ends mid-page leaves that page out of the record (the next pass re-stages it)
    {
        KvKeepView v = kv_keep_view(true, true, 0, region, 0, 255, ps);
        expect(v.staged_after == 63, "255 cells commit 63 whole blocks, not 64");
    }

    // invalidation (the serve path refills the loan's slots with experts) clears the record but keeps the shape
    {
        KvKeepRec rec;
        KvKeepView v = kv_keep_view(true, false, 0, region, 0, 1024, ps);
        kv_keep_commit(rec, (const void*) 7, (const void*) 9, 2, v);
        expect(rec.staged_blocks == 256 && rec.ss == (const void*) 7, "committed");
        kv_keep_invalidate(rec);
        expect(rec.staged_blocks == 0 && !rec.fp_valid, "invalidated: nothing staged, no watermark");
        expect(kv_keep_view(true, true, rec.staged_blocks, region, 4096, 4, ps).from_block == 0,
               "and the next pass stages the whole prefix again");
    }

    // the region's coverage caps the record: a commit at the region's end does not claim past it
    {
        KvKeepView v = kv_keep_view(true, true, 0, region, 0, region * ps + 100, ps);
        expect(v.staged_after == region, "staged_after never exceeds the region's blocks");
    }

    // a long conversation read in chunks: every chunk copies only the page the last one left partial
    {
        KvKeepRec rec;
        int64_t pos = 0, copied = 0;
        const int64_t chunk = 999;   // a chunk that is not a whole number of pages, 3 times in 4
        for (int c = 0; c < 10; ++c) {   // the first stages the (empty) prefix; the rest copy at most one page
            KvKeepView v = kv_keep_view(true, c > 0, rec.staged_blocks, 100000, pos, chunk, ps);
            copied += v.to_block - v.from_block;
            kv_keep_commit(rec, (const void*) 1, (const void*) 2, 0, v);
            pos += chunk;
        }
        expect(copied == 7, "10 chunks of one read copy the pages the chunk boundaries cut, nothing else");
        expect(rec.staged_blocks == 10 * chunk / ps, "the whole 9,990-cell conversation stands staged");
    }

    if (g_fail) {
        std::fprintf(stderr, "kv_prefix_keep_test: %d of %d checks FAILED\n", g_fail, g_checks);
        return 1;
    }
    std::printf("kv_prefix_keep_test: OK (%d checks)\n", g_checks);
    return 0;
}
