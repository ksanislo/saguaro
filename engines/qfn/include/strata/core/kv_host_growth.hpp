#pragma once
// kv_host_growth.hpp - P124e: the pure arithmetic of the streamed host K/V's elastic growth
// (SAGUARO_KV_HOST_ELASTIC, layer.cpp).  The host copy of a streamed state reserves address
// space for the whole --max-context and holds RAM only for the steps its conversation has grown
// through; these helpers answer "how many pages, how many pieces, how many bytes" for one such
// copy without a GPU, so a conversation's cost follows its use and NOT --max-context.
//
// A copy's arrays (K codes, V codes, scales - `kv_per_slot`'s order) each cover `pages` logical
// pages of `per[i]` bytes; growth and pin steps are `unit` pages (SAGUARO_KV_HOST_STEP cells);
// in piece mode every array is pinned in `piece`-byte pieces (the bulk copies split at the same
// boundaries), in prefix mode (P124e: a card that cannot use a registered page's host address)
// every array is ONE registered prefix and growing re-registers it - the page count answers
// below are the same arithmetic for both.
#include <algorithm>
#include <cstdint>

namespace strata::core {

/// Logical pages the first `cells` cells need, at least one (`page_size` cells per page).
inline int64_t kv_host_pages_for(int64_t cells, int64_t page_size) {
    return std::max<int64_t>((cells + page_size - 1) / page_size, 1);
}

/// Growth steps of `unit` pages that cover `pages` pages, capped at `max_pages` pages.
inline int64_t kv_host_step_up(int64_t pages, int64_t unit, int64_t max_pages) {
    const int64_t p = std::max<int64_t>(pages, 1);
    const int64_t u = std::max<int64_t>(unit, 1);
    return std::min(max_pages, (p + u - 1) / u * u);
}

/// Pieces of `piece` bytes an array needs pinned for its first `pages` pages (a whole number of
/// growth steps, capped at `max_pages`, so the pieces end on a step boundary).
inline int64_t kv_host_pieces(int64_t pages, int64_t unit, int64_t max_pages, uint64_t bytes_per_page,
                              uint64_t piece_bytes) {
    const uint64_t pg = (uint64_t) kv_host_step_up(pages, unit, max_pages);
    return (int64_t) ((pg * bytes_per_page + piece_bytes - 1) / piece_bytes);
}

/// A grow asked for `need` pages backs at least that many, and doubles the pages it holds when it
/// has to act at all - so a conversation crossing many steps pays O(log) registrations, not one
/// per step, and a conversation that stays put pays nothing.  Always a whole number of steps and
/// at most `max_pages`; a state arriving with nothing backed asks for one (a prompt read is backed
/// whole, ahead of need) and lands on one step - the conversation's start, about 32768 cells.
inline int64_t kv_host_grow_pages(int64_t backed_pages, int64_t need_pages, int64_t max_pages, int64_t unit) {
    const int64_t u = std::max<int64_t>(unit, 1);
    // the first grow of a state pins the whole ask (a prompt read is backed ahead of need, never token by token)
    // but a conversation arriving with nothing asked pins one step: the start --max-context must not buy.
    if (backed_pages <= 0) return kv_host_step_up(std::max(need_pages, u), u, max_pages);
    if (need_pages <= backed_pages) return backed_pages;   // already backed: nothing to do
    return kv_host_step_up(std::max(need_pages, 2 * backed_pages), u, max_pages);
}

/// A trim keeps the steps `need` pages ask for, never the last one (`min_pages`, the start), and
/// only gives back a whole step's worth - the pages between two steps stay (no churn a step later).
inline int64_t kv_host_trim_pages(int64_t backed_pages, int64_t need_pages, int64_t min_pages, int64_t unit) {
    const int64_t keep = std::max(kv_host_step_up(need_pages, unit, backed_pages), std::min(min_pages, backed_pages));
    return std::min(backed_pages, keep);
}

/// Bytes one array's registered prefix / pinned run covers for `pages` pages (page-aligned for a
/// registration; `piece_bytes` rounding is left to the caller's pieces).
inline uint64_t kv_host_prefix_bytes(int64_t pages, uint64_t bytes_per_page, uint64_t page_align) {
    const uint64_t b = (uint64_t) std::max<int64_t>(pages, 0) * bytes_per_page;
    return (b + page_align - 1) / page_align * page_align;
}

}  // namespace strata::core
