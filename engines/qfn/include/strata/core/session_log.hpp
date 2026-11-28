// A session file that grows with its conversation (format v3, "*.sessionlog"): the same content as a v1/v2 session
// file (conversation_file.hpp), stored as fixed-size chunks of each array (the token ids, the running state, every
// K/V buffer of every stage) behind a commit record that names each chunk's place and hash.  K/V is positional and
// append-only, so the next save of the same conversation finds almost every chunk already in the file (same hash)
// and writes only the ones that changed or are new, then a new commit.  Nothing a commit names is ever overwritten:
// a save that dies half-way leaves the previous commit readable, and a rewound conversation simply stops naming the
// chunks past its divergence.  The bytes no commit names any more are dropped by rewriting the file whole once they
// pass a limit.  Pure host code: no CUDA.  Layout in session_log.cpp.
#pragma once

#include "strata/core/conversation_file.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>

namespace strata::core {

constexpr const char* kSessionLogExt = ".sessionlog";

// false where this format is not built (Windows): the caller writes v1/v2 files there.
bool session_log_supported();

struct SessionLogOptions {
    uint64_t chunk_bytes = 256u << 10;    // a new file's chunk size (an existing file keeps its own)
    // an append that would leave more unnamed bytes than max(garbage_min, the named ones) rewrites the file
    // (a turn rewrites the running state, ~240 MB: a tighter limit rewrote a 40K-token file every second turn)
    uint64_t garbage_min = 512u << 20;
    int threads = 0;                      // hashing / reading threads (0: up to 8)
};

struct SessionLogStats {
    uint64_t file_bytes = 0;      // the file's size after the save
    uint64_t written_bytes = 0;   // what this save wrote (changed chunks, the commit)
    uint64_t reused_bytes = 0;    // chunk bytes found in the file and left in place
    uint64_t garbage_bytes = 0;   // bytes in the file that its newest commit does not name
    bool appended = false;        // an existing file was extended (false: written whole, through a temporary file)
    bool compacted = false;       // an existing file was rewritten whole to drop its unnamed bytes
    uint64_t commit = 0;          // the new commit's identity (what a later turn save continues from)
    uint64_t hashed_bytes = 0;    // bytes this save had to hash (a turn save: only what it captured)
    uint64_t verify_bad = 0;      // SessionLogTurn claims: chunks claimed unchanged whose bytes differ from the file's
};

// The disk a big capture spills to instead of a second host copy (PLAN-p125r): a capture's parts past the writer's
// RAM cap are copied from their source through a buffer of at most that cap into this file and read back by the save
// in bounded pieces.  It is opened in the conversation's folder and unlinked at once: its blocks are disk the folder
// already accounts, what survives a crash is nothing, and its fd is the only way in.  Linux only, like the format.
struct SessionLogSpool {
    ~SessionLogSpool();
    bool open(const std::string& dir, size_t cap, std::string& error);   // the caller holds exactly one shared_ptr once this is true
    bool write_at(uint64_t at, const uint8_t* p, size_t n) const;
    bool pread(uint8_t* p, size_t n, uint64_t at) const;
    int fd = -1;
    size_t cap_bytes = 0;   // the capture's RAM budget it was opened under (its log line names it)
};

// A turn's state captured from where it lives, for a save that runs later on another thread (session_log_capture).
// Each K/V part holds only its bytes from `from` on (a multiple of the chunk size): the chunks before it are the
// ones the file's commit `base` already names, which the capturing side knows were not rewritten since.  from == 0
// everywhere is a whole capture and continues from nothing.  `claim` (verification only, with from == 0): what a
// delta would have left out; the save counts the chunks under it that differ from the file's.  A part the capture
// streamed instead of holding in RAM (see SessionLogSpool) has an empty `bytes` and names its [from, size) at
// `spool_at` in `spool`.
struct SessionLogKvTail {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    std::array<uint64_t, 5> size{}, from{}, claim{};
    std::array<ConversationBuffer, 5> bytes;
    std::array<std::shared_ptr<const SessionLogSpool>, 5> spool;
    std::array<uint64_t, 5> spool_at{};
};
struct SessionLogTurn {
    SavedConversation meta;                            // everything but the K/V (kv empty, in the stage images too)
    std::vector<std::vector<SessionLogKvTail>> kv;     // [0]: the first stage's layers, then each later stage's
    uint64_t chunk_bytes = 256u << 10;
    std::shared_ptr<SessionLogSpool> spool;            // the capture streamed its big parts through this (null: none)
    size_t spool_bytes = 0;                            // (and how many)
    bool delta() const {
        for (const auto& im : kv) for (const auto& l : im) for (uint64_t f : l.from) if (f) return true;
        return false;
    }
    size_t bytes() const {   // the captured K/V bytes: the held ones and the spooled ones (spooled: not RAM)
        size_t n = meta.bytes() + spool_bytes;
        for (const auto& im : kv) for (const auto& l : im) for (const auto& b : l.bytes) n += b.bytes();
        return n;
    }
};
// Reads each part's bytes from floor(unchanged / chunk) * chunk on out of its source (`unchanged`: per image, per
// layer, per part, bytes known not to have changed since the commit the turn will continue; null: a whole capture).
// With `verify` everything is read and `unchanged` becomes the claim.  Parts read while the capture holds less than
// `spool_cap_bytes` land in `bytes` as ever; past the cap they stream through a buffer of that size into the turn's
// spool file, opened in `spool_dir` (cap 0 or empty dir: no spool, as before P125r).  The save reads both kinds back.
bool session_log_capture(SessionLogTurn& out, SavedConversation&& meta, const std::vector<SessionKvSource>& kv,
                         const std::vector<std::vector<SessionKvSource>>& stage_kv,
                         const std::vector<std::vector<std::array<uint64_t, 5>>>* unchanged, bool verify,
                         std::string& error, const std::string& spool_dir = {}, size_t spool_cap_bytes = 0);
// Saves a captured turn.  A delta is written only onto the file whose current commit is `base` (else it fails with
// SessionError::invalid and nothing is changed: the caller captures whole next time).
bool session_log_save_turn(const std::string& path, const SessionLogTurn& turn, uint64_t base,
                           const SessionFileIdentity& id, SessionLogStats& stats, std::string& error,
                           const SessionWriteOptions& options = {}, SessionStatus* status = nullptr,
                           const SessionLogOptions& log = {});

// Saves `image` at `path`.  A readable file of this identity there is extended in place (see above); anything else
// (no file, another identity, a damaged one) is replaced by a whole write through a temporary file and a rename.
// SessionWriteOptions as for session_file_write; `fault` steps: "write", "file_flush", "commit" (after the data is
// flushed, before the commit is made current: what a crash there leaves), "rename".
bool session_log_save(const std::string& path, const SavedConversation& image, const SessionFileIdentity& id,
                      SessionLogStats& stats, std::string& error, const SessionWriteOptions& options = {},
                      SessionStatus* status = nullptr, const SessionLogOptions& log = {});

// Full validation (file kind, header, identity, the newest readable commit, bounds, every chunk's hash) before
// `image` is replaced; all buffers are owned copies.  `limits` as for session_file_read.
bool session_log_read(const std::string& path, const SessionFileIdentity& id, SavedConversation& image, size_t& bytes,
                      std::string& error, const SessionReadLimits& limits = {}, SessionStatus* status = nullptr,
                      const SessionLogOptions& log = {});

// The folder index's view (as session_file_peek): the live token ids and images and the deepest checkpoint's.
bool session_log_peek(const std::string& path, const SessionFileIdentity& id, SessionPeek& out, std::string& error);

} // namespace strata::core
