// Carrying out the release policy (park_policy.hpp) on the conversation cache and the session folder.  Host code,
// called by the serve loop on its own thread (the cache has no lock; the folder has its own).
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/park_policy.hpp"
#include "strata/core/session_dir.hpp"

#include <vector>

namespace strata::core::park {

/// The parked entries as the policy sees them, least recently active first.  An entry is on disk when it came from
/// its file, or the folder holds its file (or a newer one of the same conversation).
inline std::vector<Entry> entries_of(const ConversationCache& cache, const SessionDir* dir, double now) {
    std::vector<Entry> out;
    for (const auto& i : cache.list()) {
        const bool on_disk = i.on_disk ||
                             (dir != nullptr && dir->saved(cache.shared_serial(i.serial)) == SessionDir::Saved::yes);
        out.push_back({i.serial, i.bytes, i.tokens, now - i.parked_at, on_disk, i.keep});
    }
    return out;
}

struct Released {
    Entry entry;
    Action action;
};

/// Each action in turn; returns the ones that released an entry.
///   drop          the entry leaves the cache (its file is in the folder).
///   write_drop    its session write is queued if it is not, the entry leaves the cache, and the write is moved up:
///                 the image lives on in the writer's queue (a request can still take it back from there) and its
///                 RAM is freed when the write ends.
///   drop_unsaved  a queued or running write of it is stopped and the entry leaves: its RAM is free now.
inline std::vector<Released> release(ConversationCache& cache, SessionDir* dir, const std::vector<Entry>& entries,
                                     const std::vector<Action>& actions) {
    std::vector<Released> out;
    bool wrote = false;
    for (const Action& a : actions) {
        const Entry* e = nullptr;
        for (const Entry& x : entries)
            if (x.serial == a.serial) e = &x;
        auto image = cache.shared_serial(a.serial);
        if (e == nullptr || !image || e->keep) continue;
        if (a.what == Do::write_drop) {
            if (dir == nullptr) continue;
            if (dir->saved(image) == SessionDir::Saved::no) dir->write(image);   // (never queued, or its write failed)
            wrote = true;
        } else if (a.what == Do::drop_unsaved && dir != nullptr) {
            dir->claim(image.get());
        }
        if (!cache.drop(a.serial)) continue;
        out.push_back({*e, a});
    }
    if (wrote) dir->expedite();   // (the images only the queue holds now)
    return out;
}

}  // namespace strata::core::park
