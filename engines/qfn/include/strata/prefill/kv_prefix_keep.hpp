// include/strata/prefill/kv_prefix_keep.hpp - the read-prefix keep's bookkeeping (PLAN-prefix-redma).
//
// A streamed session's prompt path used to DMA the conversation's WHOLE K/V prefix host->VRAM once per QSA layer
// per chunk into the one staging pool (`kv_stage_from_host`, prefill.cpp): at 300K tokens that is ~10 s of idle
// GPUs per read (REPORT-qfn-read-stall.md section 2). The keep's fix: the staging loan is laid out as ONE region
// per QSA layer of the stage, a region keeps its layer's prefix on the card across the chunks of a read (and
// across reads of the same conversation while its staging loan holds), and every layer pass copies only the
// blocks that are new. The same K/V bytes reach the same kernels: a region is the same identity-layout pool,
// staged the same way, just starting further along.
//
// When is a region's [0, staged_blocks) trusted to equal the host copy?
//   * the same conversation: the same session pointer, the same host-copy address, the same K/V format;
//   * a continuation, not a rewind: the chunk starts at or beyond the whole blocks staged (p0 >= staged*page).
//     A read that starts lower - a rejection, a branch, a checkpoint restored under the conversation - rewrote
//     staged cells, and cells below p0 of a rewind can differ, so the region is staged again in full;
//   * and (staged_blocks > 0) a FINGERPRINT of the watermark block agrees. The watermark is the last fully
//     staged block, host block `staged_blocks - 1`: its bytes are hashed from the host copy when the keep is
//     consulted and compared with the hash of the region's copy taken when it was committed (both by one kernel
//     reading through the device - `kv_host_block_match` / `kv_pool_block_hash`). A conversation restored over
//     the host copy while the region still holds another one changes the watermark's bytes, and the mismatch
//     sends that layer back to a full stage. Cells staged from an append were written by kernels that ran after
//     the commit's hash was enqueued, so a chunk boundary's stream sync has to have landed them; where it has
//     not, the hash reads torn or stale bytes and mismatches - which only costs the re-stage, never a wrong read.
//
// This header is the pure decision (no CUDA, no device): `kv_keep_view` is what one layer pass copies, and
// `kv_keep_commit` is what the record looks like after it. The unit test is src/prefill/kv_prefix_keep_test.cpp.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace strata::prefill {

/// SAGUARO_KV_PREFIX_KEEP=1 (opt-in, default off): lay the staging loan out as one region per QSA layer and
/// keep a conversation's staged prefix on the card. Off, every layer of every chunk stages the whole prefix
/// again into the one shared pool, exactly as before - and no loan is ever sized for the regions.
inline bool kv_keep_wanted(const char* env) { return env != nullptr && std::atoi(env) != 0; }

/// What one QSA layer's region is known to hold. `fp` is not here: the committed watermark hash lives in the
/// pinned device-written slot the prefill keeps per layer (it is a kernel's output, and the record outlives it
/// only through the region's life - which is the slot's life too).
struct KvKeepRec {
    int64_t staged_blocks = 0;   ///< leading whole blocks the region equals the host copy for
    const void* ss = nullptr;    ///< the session whose conversation the region holds
    const void* host0 = nullptr; ///< the host copy's first array it was staged from (its address, identity)
    int fmt = -1;                ///< the K/V format it was staged as (kKvF16/kKvInt8/kKvQ4/kKvHybrid)
    bool fp_valid = false;       ///< a watermark hash of `staged_blocks - 1` was committed (staged_blocks > 0)
};

/// One layer pass of one chunk: copy host blocks [from_block, to_block) into the region, then append the chunk.
struct KvKeepView {
    int64_t from_block = 0;
    int64_t to_block = 0;
    int64_t staged_after = 0;   ///< commit: whole blocks the region will hold for the next pass
    int64_t skipped_blocks = 0; ///< blocks NOT copied this pass (the measured win)
    bool checked = false;       ///< a stored prefix was confirmed by its watermark fingerprint
};

/// `rec_ident`: the record's session, host copy and format are this pass's. `fp_ok`: the host watermark block
/// hashes to the committed hash (only asked when the record says blocks are staged).
inline KvKeepView kv_keep_view(bool rec_ident, bool fp_ok, int64_t staged_blocks, int64_t region_blocks,
                               int64_t p0, int64_t chunk, int64_t page_size) {
    KvKeepView v;
    v.to_block = (p0 + page_size - 1) / page_size;
    int64_t after = (p0 + chunk) / page_size;
    if (after > region_blocks) after = region_blocks;   // the region never claims more than it covers
    v.staged_after = after;
    const bool usable = rec_ident && staged_blocks <= p0 / page_size && staged_blocks <= region_blocks &&
                        (staged_blocks == 0 || fp_ok);
    v.from_block = usable ? staged_blocks : 0;
    v.checked = usable && staged_blocks > 0;
    v.skipped_blocks = v.from_block;
    return v;
}

/// The append completed by this pass ends at `p0 + chunk`; its fully written blocks are valid for the next pass.
inline void kv_keep_commit(KvKeepRec& rec, const void* ss, const void* host0, int fmt, const KvKeepView& v) {
    rec.ss = ss;
    rec.host0 = host0;
    rec.fmt = fmt;
    rec.staged_blocks = v.staged_after;
    rec.fp_valid = v.staged_after > 0;   // the caller enqueues the pool hash of the new watermark block
}

/// The region's contents are gone (its memory was re-laid-out, or the loan it lives in was refilled with experts).
inline void kv_keep_invalidate(KvKeepRec& rec) {
    rec.staged_blocks = 0;
    rec.fp_valid = false;
}

}  // namespace strata::prefill
