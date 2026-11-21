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

#include <cstdint>
#include <string>

namespace strata::core {

constexpr const char* kSessionLogExt = ".sessionlog";

// false where this format is not built (Windows): the caller writes v1/v2 files there.
bool session_log_supported();

struct SessionLogOptions {
    uint64_t chunk_bytes = 256u << 10;    // a new file's chunk size (an existing file keeps its own)
    // an append that would leave more unnamed bytes than max(garbage_min, half the named ones) rewrites the file
    uint64_t garbage_min = 256u << 20;
    int threads = 0;                      // hashing / reading threads (0: up to 8)
};

struct SessionLogStats {
    uint64_t file_bytes = 0;      // the file's size after the save
    uint64_t written_bytes = 0;   // what this save wrote (changed chunks, the commit)
    uint64_t reused_bytes = 0;    // chunk bytes found in the file and left in place
    uint64_t garbage_bytes = 0;   // bytes in the file that its newest commit does not name
    bool appended = false;        // an existing file was extended (false: written whole, through a temporary file)
    bool compacted = false;       // an existing file was rewritten whole to drop its unnamed bytes
};

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
