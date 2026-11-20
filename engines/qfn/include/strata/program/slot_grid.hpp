#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

// SAGUARO_SLOT_GRID: periodic checkpoints for a batch slot's conversation.  A slot keeps its turn (and message)
// checkpoint for a continuation; a request that keeps only a PREFIX of the conversation - an agent's compaction
// summary over its oldest messages, a client that rewrote an older turn - needs a checkpoint well before the end.
// The solo path keeps one every --prompt-cache-every tokens; a slot keeps a few, by distance from the end: bucket -1
// is the last `every` tokens, bucket k >= 0 is [every * 2^k, every * 2^(k+1)) back, the last bucket takes everything
// further.  Each bucket keeps its oldest and its newest checkpoint, so a checkpoint ages outward from bucket to bucket
// as the conversation grows instead of being replaced by the next one (a bucket that kept only its newest would never
// let one reach the far buckets).  Simulated over growth steps of 2K-150K up to 1M with cuts 40K-250K back: with 6
// buckets at most 12 can be held (10 in the simulation) and the re-read below a cut is at most 0.71x (average ~0.22x) of the cut's
// distance from the end.  Each checkpoint holds the GDN state (~118 MB with Qwen3.8-Flash-Next).  Positions are
// multiples of `every`, so the points a slot keeps recur turn after turn.
namespace strata::program {

inline int slot_grid_bucket(int64_t distance, int64_t every, int buckets) {
    if (distance < every) return -1;
    int b = 0;
    for (int64_t q = distance / every; q > 1; q >>= 1) ++b;
    return std::min(b, buckets - 2);
}

// Which grid checkpoints (their lengths, any order) a conversation ending at `end` keeps: indexes, ascending.
// Lengths at or past `end`, or not positive, are never kept.  buckets < 2: none.
inline std::vector<size_t> slot_grid_keep(const std::vector<int64_t>& held, int64_t end, int64_t every, int buckets) {
    std::vector<size_t> out;
    if (every <= 0 || buckets < 2) return out;
    std::map<int, std::pair<size_t, size_t>> by;   // bucket -> (oldest, newest) index
    for (size_t i = 0; i < held.size(); ++i) {
        if (held[i] <= 0 || held[i] >= end) continue;
        const int b = slot_grid_bucket(end - held[i], every, buckets);
        auto it = by.find(b);
        if (it == by.end()) { by[b] = {i, i}; continue; }
        if (held[i] < held[it->second.first]) it->second.first = i;
        if (held[i] > held[it->second.second]) it->second.second = i;
    }
    for (const auto& [b, oe] : by) {
        out.push_back(oe.first);
        if (oe.second != oe.first && held[oe.second] != held[oe.first]) out.push_back(oe.second);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// The grid positions a read of [from, end) should save, given the lengths the slot already holds: the multiples of
// `every` inside the read that slot_grid_keep keeps at `end` beside them (the rest would be dropped at once).
inline std::vector<int64_t> slot_grid_new(int64_t from, int64_t end, int64_t every, int buckets,
                                          const std::vector<int64_t>& held = {}) {
    std::vector<int64_t> out;
    if (every <= 0 || buckets < 2 || end <= 0) return out;
    std::vector<int64_t> all = held;
    const size_t first_new = all.size();
    for (int64_t p = (std::max<int64_t>(from, 0) / every + 1) * every; p < end; p += every)
        if (std::find(held.begin(), held.end(), p) == held.end()) all.push_back(p);
    for (const size_t i : slot_grid_keep(all, end, every, buckets))
        if (i >= first_new) out.push_back(all[i]);
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace strata::program
