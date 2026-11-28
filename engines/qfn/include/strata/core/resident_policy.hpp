// How much of the host the resident expert tier may claim (PLAN-p124x-mortar-memory).  Pure policy: the engine
// samples the host and the model, the numbers here decide what is left over (generate.cpp, before the tier pins).
//
// THE MORTAR SHAPE (2026-11-24): a 96 GB host, `--resident-budget-gib 68`, a 262144-token x 2-slot K/V pool of
// ~10.2 GiB beside it, turn-save captures of up to 2.9 GiB held while the writer runs, prompt staging of 2.4 GiB,
// and a file tier that needs ~10 GiB of page cache to stop re-reading.  The engine's RSS reached 89 GiB, the host
// had 0.2-2 GiB available, 2-5 GiB in swap and 100k+ major page faults, and every abort followed.  A budget is
// only safe with a real margin beside it - for the engine's own RAM above the tier AND for everything outside
// the engine (OS, harness, the page cache).
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::core::resident {

/// RAM that stays outside the engine on a host of `total_bytes`: the OS, whatever else runs, and a page cache
/// the file tier can read through.  1/8 of the host, never less than 8 GiB - the mortar host's file tier wanted
/// ~10 GiB held and reached 41 GiB when it could; 8 GiB is the floor a small host can still afford.
inline uint64_t host_margin_bytes(uint64_t total_bytes) {
    return std::max<uint64_t>(8ull << 30, total_bytes / 8);
}

/// What the engine commits in RAM beside the expert tier while it serves: the shared K/V pool (the live
/// conversations' streamed K/V and the parked ones share exactly this budget - park_policy.hpp) and the
/// turn-save captures, each a fresh host copy of one slot's conversation (bounded by the pool), with the prompt
/// staging riding along.  Measured on the mortar box: RSS less the tier = 18-21 GiB = 2 x the 10.2 GiB pool.
inline uint64_t runtime_bytes(uint64_t kv_pool_bytes) {
    return 2 * kv_pool_bytes;
}

/// The pool figure the margin counts (P124e, manager card round 12, finding A): what the K/V host copies
/// COMMIT - the pages pinned now (the elastic copies' backed pages plus any copy pinned whole,
/// qsa_kv_host_bytes) plus the one growth step the elastic copies run ahead of their conversations
/// (qsa_kv_host_step_bytes).  Never --max-context x slots: that is only the ceiling the elastic copies'
/// ADDRESSES could grow to, and counting it at load made a 1048576 setting cost 37 GiB of expert tier
/// beside a 32768-cell start (the merge regression on mortar, 2026-11-27: 67.90 GiB of experts at 262144,
/// 30.78 at 1048576).  A pinned-whole layout is counted by the same rule and comes out the same as before:
/// it has committed the whole context at load, so `backed_bytes` already IS that ceiling.  Growth past the
/// start pays the way every grow pays: room is made for it when it arrives (park releases, ram_reclaim).
inline uint64_t committed_kv_bytes(uint64_t backed_bytes, uint64_t step_lead_bytes) {
    return backed_bytes + step_lead_bytes;
}

/// The largest resident tier the host can carry: `total - runtime - margin`, 0 when nothing is left.  Unknown
/// inputs (0) mean no cap is derived, never a cap of the whole host.
inline uint64_t budget_cap(uint64_t total_bytes, uint64_t kv_pool_bytes) {
    if (total_bytes == 0) return 0;
    const uint64_t need = runtime_bytes(kv_pool_bytes) + host_margin_bytes(total_bytes);
    return total_bytes > need ? total_bytes - need : 0;
}

}  // namespace strata::core::resident
