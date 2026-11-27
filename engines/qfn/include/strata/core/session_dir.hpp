// A folder of session files, one per conversation (--session-dir): the conversations the engine parks are written
// to it behind the engine's back (a writer thread, after a short delay so a conversation taken back at once is never
// written), and its files come back into the conversation cache - in the background after the start, newest first,
// or at once when a request continues one.  Every file is a session file (conversation_file.hpp) bound to this
// runtime's identity; a file this runtime refuses is deleted.  Pure host code: no CUDA.
#pragma once

#include "strata/core/conversation_file.hpp"
#include "strata/core/session_log.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

// The file lineage of one place a live conversation sits in (the main session, a batch slot): what its last turn
// save left, so the next one may write only what changed.  Written by the folder's writer, read by the serve loop.
struct SessionTurnTrack {
    std::atomic<uint64_t> commit{0}, written_seq{0};
    std::atomic<bool> broken{false};        // a turn save did not land (or its file wants a rewrite): capture whole next
    std::atomic<int64_t> saved_tokens{-1};
    std::string path;                       // (the writer's)
};
struct SessionTurnJob {
    std::shared_ptr<SessionLogTurn> turn;
    std::shared_ptr<SessionTurnTrack> track;
    uint64_t seq = 0, base_seq = 0;         // this capture, and the one a delta continues
    double capture_ms = 0.0;                // what the capture cost the serve thread (for the log line)
    std::string label;                      // "main", "slot 3"
};

class SessionDir {
public:
    struct Options {
        std::string dir;
        SessionFileIdentity id;
        // SAGUARO_YARN_ELASTIC: the identity of a conversation at the YaRN level (config 0: none); its files read as such
        SessionFileIdentity id_yarn;
        SessionReadLimits limits;              // this runtime's bounds (its stages included); admit = RAM preflight
        // the least recently used files go first when the folder is over max_bytes (or, a safety net, max_files);
        // a file not written or resumed for max_age goes in any case
        size_t max_files = 64;
        uint64_t max_bytes = 64ull << 30;
        std::chrono::hours max_age{24 * 7};
        uint64_t preload_bytes = 0;            // the background preload stops before this many bytes (0: no preload)
        size_t preload_files = 0;
        std::chrono::milliseconds write_delay{30000};   // a parked image is written this long after it was parked
        uint64_t min_free_bytes = 4ull << 30;  // a write is refused when the disk would keep less than this free
        // background writes go to appending files (session_log.hpp): a conversation's next save extends its own file
        // by what changed instead of writing it whole again.  Off (or unsupported): v1/v2 files, as before.
        bool append = true;
        SessionLogOptions log_format;
        std::function<void(const std::string&)> log;
        // machine-readable notes for the server, one line each (no newline): an event
        // ("sess_event=<what> sess_tokens=N sess_bytes=B sess_ms=T sess_why=<why>") and, after every change, the
        // folder's state as JSON ("sess_json={...}", no spaces).  Called with the folder's lock held.
        std::function<void(const std::string&)> event;
        std::function<void()> write_block;     // tests: called on the writer thread after each block of a background write
    };
    explicit SessionDir(Options o);
    ~SessionDir();                             // stops the preload, writes every queued image, then stops
    SessionDir(const SessionDir&) = delete;
    SessionDir& operator=(const SessionDir&) = delete;

    // Index the folder (peek): the files this runtime reads; any other *.session file is deleted.  Starts the writer.
    void open();
    // The background preload: newest file first, until preload_bytes / preload_files.
    void start_preload();
    // A parked image, written after write_delay (or at flush).  Files it supersedes (the same conversation, older)
    // are deleted once it is on disk.
    void write(std::shared_ptr<const SavedConversation> image);
    // Low RAM: write a conversation NOW from streamed sources (no host copy of its K/V: conversation_snapshot_sources;
    // one list per stage image in stage_kv), on the caller's thread; indexed and superseding like a parked image.
    // `chain`: the conversation's whole checkpoint chain (token ids and images are used), for deciding which older
    // files it supersedes - the file itself holds only the deepest checkpoint (meta.checkpoints).
    bool write_now(const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                   const std::vector<std::vector<SessionKvSource>>& stage_kv, size_t& bytes, std::string& error,
                   const std::vector<ConversationCheckpoint>* chain = nullptr);
    // Before the caller moves `image` out (take): a queued write of it is dropped; a running one is cancelled - it stops
    // at its next block and its temporary file is removed (the conversation is in RAM again, its file is not needed) -
    // and waited for until it has stopped, with a watchdog beat each time it moves.  A write already in its flush or
    // publish is waited for to the end (it still reads the image to index the file).
    void claim(const SavedConversation* image);
    // A turn of a conversation that stays where it is, captured by the caller: written at once, in order, to that
    // conversation's appending file.  A delta lands only on the file its track names at the commit it continues.
    void write_turn(SessionTurnJob job);
    // Whether `image` is safe on disk: written (its file, or a newer file of the same conversation, is in the folder),
    // or still on its way there.
    enum class Saved { no, queued, writing, yes };
    Saved saved(const std::shared_ptr<const SavedConversation>& image) const;
    // RAM is wanted: `image`'s queued write starts now instead of after write_delay (nullptr: every queued write of an
    // image only the queue still holds - what the cache already let go of is freed when its write ends).
    // Returns how many writes were moved up.
    size_t expedite(const SavedConversation* image = nullptr);
    // The RAM of images only the writer still holds (queued or being written): it comes back as their writes end.
    uint64_t orphan_bytes() const;
    // The images the preload has read, for the caller's cache (each file's image once).
    std::vector<SavedConversation> take_preloaded();
    // The on-disk conversation that continues `ids` furthest when it beats `beat` tokens, read now (or waited for
    // while the preload reads it).  nullopt when none beats it, or the read failed (logged; the file is deleted).
    // A conversation the caller's cache let go of before its file was written (queued behind write_delay, or being
    // written) is not lost: it lives on in the writer's queue, and is taken from there when it beats the files - a
    // queued write is dropped, a running one cancelled (claim) - and returned whole (every checkpoint, no read).
    // *in_ram says which: true for such an image (it has no file: a caller that cannot use it should keep it).
    std::optional<SavedConversation> fetch(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs,
                                           bool cvec, int64_t beat, int64_t& tokens, bool* in_ram = nullptr);
    // How far the folder - its files, and the writer's queued and running images - continues `ids`, without reading
    // anything (the index and the images' tokens only): for a caller deciding where a request goes (a batch slot's
    // admission takes the path that fetches only when this beats what the slots hold).
    int64_t best_tokens(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs, bool cvec) const;
    // The engine is quitting: writes from here on are labelled as the shutdown's.
    void begin_exit();
    // Write every queued image now and wait until they are on disk.
    void flush();
    // The same, but waited for only until `deadline`, logged at start and at its end or timeout (`what` names the
    // phase: "exit").  false: writes were still queued or running when the deadline passed - drop_queued() gives up
    // on them so the exit is never blocked by them.
    bool flush_until(std::chrono::steady_clock::time_point deadline, const char* what);
    // The save ran out of time: forget the queued writes and cancel the running one (it stops at its next block).
    // What is on disk stays what the last turn left; the next turn save of a dropped one is written whole.
    void drop_queued();
    size_t files() const;
    uint64_t bytes() const;

    // Exposed for tests: does `newer` supersede `older` (the same conversation, further on)?
    static bool supersedes(const ConversationCheckpoint& newer_live, const std::vector<ConversationCheckpoint>& newer_chain,
                           const SessionPeek& older);

private:
    const SessionFileIdentity& id_of(const SavedConversation& s) const {
        return o_.id_yarn.config != 0 && s.live.rope_level == 1 ? o_.id_yarn : o_.id;
    }

    enum class State { idle, loading, preloaded, writing };   // writing: the writer is extending this file
    struct Entry {
        std::string path;
        SessionPeek peek;
        bool log = false;                      // an appending file (session_log.hpp)
        std::weak_ptr<const SavedConversation> image;   // the parked image it was last written from (saved())
        uint64_t seq = 0;                      // recency: higher is newer
        std::chrono::system_clock::time_point used;   // last written or resumed (the file's mtime across restarts)
        State state = State::idle;
    };
    struct Job {
        std::shared_ptr<const SavedConversation> image;
        std::chrono::steady_clock::time_point due;
    };
    struct Ready {
        std::string path;
        SavedConversation image;
    };

    void say(const std::string& m) const;
    void add_written_locked(const std::string& path, const SavedConversation& image, size_t written, double s,
                            const char* what, const std::vector<ConversationCheckpoint>* chain = nullptr,
                            const std::shared_ptr<const SavedConversation>& from = nullptr,
                            const SessionLogStats* log = nullptr);
    void writer_loop();
    void preload_loop();
    bool read_file(const std::string& path, SavedConversation& image, std::string& error) const;
    void drop_entry_locked(size_t i, const char* why);
    void touch_locked(Entry& e);
    void note_locked(const char* what, uint64_t tokens, uint64_t bytes, double ms, const char* why);
    void state_locked();
    void enforce_caps_locked();
    int64_t match_image(const SavedConversation& s, const std::vector<int32_t>& ids,
                        const std::vector<ConversationImageKey>& imgs, bool cvec) const;
    void wait_write_locked(std::unique_lock<std::mutex>& lk, const SavedConversation* image, const char* what);
    int64_t match(const Entry& e, const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs,
                  bool cvec) const;

    Options o_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<uint64_t> write_moved_{0};      // the background write's blocks and blocking steps so far
    std::atomic<uint64_t> write_blocking_{0};   // the bytes of the blocking step it is in, 0: streaming
    // claim: the image whose running write is to stop at its next block (its file is not needed: it was taken back)
    std::atomic<const SavedConversation*> write_cancel_{nullptr};
    std::vector<Entry> entries_;
    std::deque<Job> jobs_;
    std::deque<SessionTurnJob> turn_jobs_;
    bool turn_busy_ = false;
    void turn_write(std::unique_lock<std::mutex>& lk);
    std::vector<Ready> ready_;
    const SavedConversation* writing_ = nullptr;
    std::shared_ptr<const SavedConversation> writing_img_;   // the running write's image (fetch can take it back)
    uint64_t seq_ = 0, preloaded_bytes_ = 0;
    size_t preloaded_files_ = 0;
    bool stop_ = false, flushing_ = false, fetching_ = false, exiting_ = false;
    std::thread writer_, preloader_;
};

} // namespace strata::core
