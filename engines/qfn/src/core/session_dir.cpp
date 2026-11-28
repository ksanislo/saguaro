// --session-dir: a folder of session files, one per conversation (session_dir.hpp).
#include "strata/core/session_dir.hpp"
#include "strata/core/progress.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;

namespace strata::core {
namespace {

bool same_head(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs;
}

// a's tokens and images are the start of b's
bool head_prefix(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    if (a.ids.empty() || a.ids.size() > b.ids.size() || a.imgs.size() > b.imgs.size()) return false;
    return std::equal(a.ids.begin(), a.ids.end(), b.ids.begin()) &&
           std::equal(a.imgs.begin(), a.imgs.end(), b.imgs.begin());
}

// what a file written from `image` (deepest checkpoint only) holds, for the index - without reading it back
SessionPeek peek_of(const SavedConversation& image, uint64_t bytes) {
    SessionPeek p;
    p.live.ids = image.live.ids;
    p.live.imgs = image.live.imgs;
    p.cvec = image.cvec;
    p.bytes = bytes;
    p.version = image.stage_images.empty() ? 1 : 2;
    bool aligned = !image.checkpoints.empty();
    for (const auto& st : image.stage_images) aligned = aligned && st.checkpoints.size() == image.checkpoints.size();
    if (aligned) {
        const ConversationCheckpoint* deepest = session_deepest_checkpoint(image.checkpoints);   // the file's one
        p.deepest.ids = deepest->ids;
        p.deepest.imgs = deepest->imgs;
    }
    return p;
}

std::string random_name(const char* ext = ".session") {
    std::random_device rd;
    const uint64_t v = (uint64_t) rd() << 32 ^ rd();
    char b[40];
    std::snprintf(b, sizeof b, "%016" PRIx64, v);
    return std::string(b) + ext;
}

} // namespace

SessionDir::SessionDir(Options o) : o_(std::move(o)) {}

SessionDir::~SessionDir() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
        flushing_ = true;   // the writer writes what is queued before it stops
    }
    cv_.notify_all();
    if (preloader_.joinable()) preloader_.join();
    if (writer_.joinable()) writer_.join();
}

void SessionDir::say(const std::string& m) const {
    if (o_.log) o_.log(m);
}

bool SessionDir::supersedes(const ConversationCheckpoint& newer_live, const std::vector<ConversationCheckpoint>& newer_chain,
                            const SessionPeek& older) {
    // the same conversation further on: its live tokens continue the older file's
    if (head_prefix(older.live, newer_live)) return true;
    // or the older file's deepest checkpoint is held by the newer chain (#342's rule: a client that rewrote the tail)
    if (older.deepest.ids.empty()) return false;
    if (same_head(older.deepest, newer_live)) return true;
    for (const auto& c : newer_chain)
        if (same_head(older.deepest, c)) return true;
    return false;
}

void SessionDir::open() {
    std::error_code ec;
    fs::create_directories(o_.dir, ec);
    std::vector<std::pair<fs::file_time_type, Entry>> found;
    for (const auto& de : fs::directory_iterator(o_.dir, ec)) {
        const bool log = de.path().extension() == kSessionLogExt;
        if (!de.is_regular_file() || (de.path().extension() != ".session" && !log)) continue;
        if (log && !session_log_supported()) continue;   // (not this build's to judge)
        Entry e;
        e.path = de.path().string();
        e.log = log;
        std::string err;
        auto peek = [&](const SessionFileIdentity& id) {
            return log ? session_log_peek(e.path, id, e.peek, err) : session_file_peek(e.path, id, e.peek, err);
        };
        if (!peek(o_.id) && (o_.id_yarn.config == 0 || !peek(o_.id_yarn))) {
            say("drop " + e.path + " (" + err + ")");
            const uint64_t sz = (uint64_t) fs::file_size(de.path(), ec);
            fs::remove(de.path(), ec);
            std::lock_guard<std::mutex> lk(mu_);
            note_locked("deleted", 0, ec ? 0 : sz, 0, "foreign");
            continue;
        }
        const auto ft = de.last_write_time(ec);
        e.used = std::chrono::time_point_cast<std::chrono::system_clock::duration>(std::chrono::file_clock::to_sys(ft));
        found.emplace_back(ft, std::move(e));
    }
    std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& f : found) {
        f.second.seq = ++seq_;
        entries_.push_back(std::move(f.second));
    }
    enforce_caps_locked();
    uint64_t total = 0;
    for (const auto& e : entries_) total += e.peek.bytes;
    say(std::to_string(entries_.size()) + " conversation file(s), " + std::to_string(total >> 20) + " MiB in " + o_.dir);
    state_locked();   // first: the server's "opened" line names the count from it
    note_locked("opened", 0, total, 0, "start");
    writer_ = std::thread([this] { writer_loop(); });
}

void SessionDir::start_preload() {
    if (o_.preload_bytes == 0 || o_.preload_files == 0) return;
    preloader_ = std::thread([this] { preload_loop(); });
}

void SessionDir::write(std::shared_ptr<const SavedConversation> image) {
    if (!image || image->live.ids.empty()) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        // a queued image of the same conversation, older: this one replaces it (never written twice)
        jobs_.erase(std::remove_if(jobs_.begin(), jobs_.end(), [&](const Job& j) {
            return SessionDir::supersedes(image->live, image->checkpoints, peek_of(*j.image, 0));
        }), jobs_.end());
        jobs_.push_back({std::move(image), std::chrono::steady_clock::now() + o_.write_delay});
    }
    cv_.notify_all();
}

void SessionDir::claim(const SavedConversation* image) {
    std::unique_lock<std::mutex> lk(mu_);
    jobs_.erase(std::remove_if(jobs_.begin(), jobs_.end(), [&](const Job& j) { return j.image.get() == image; }),
                jobs_.end());
    if (writing_ != image) return;
    // The running write is cancelled: it stops before its next block (or before its flush) and removes its temporary
    // file.  It used to be waited for to the end - 81 s for a 611K conversation (9 GB) - while the request that took
    // the conversation back sat here.  What is left to wait for is one block, or a flush or publish already started.
    // It beats only when the write moved, so a write that hangs still ends in the watchdog; a step that blocks in one
    // call (the file's flush) gets its allowance, as in the SAVE command.
    wait_write_locked(lk, image, "request: waiting for its session file write to stop");
}

// claim / fetch: the running write of `image` cancelled and waited for until it has stopped (see claim)
void SessionDir::wait_write_locked(std::unique_lock<std::mutex>& lk, const SavedConversation* image, const char* what) {
    write_cancel_.store(image, std::memory_order_relaxed);
    progress_at(what);
    uint64_t seen = UINT64_MAX;
    while (writing_ == image) {
        const uint64_t moved = write_moved_.load(std::memory_order_relaxed);
        if (moved != seen) {
            seen = moved;
            const uint64_t blocking = write_blocking_.load(std::memory_order_relaxed);
            progress_allow(blocking != 0 ? session_phase_limit_s(blocking) : 0);
        }
        cv_.wait_for(lk, std::chrono::seconds(1));
    }
    write_cancel_.store(nullptr, std::memory_order_relaxed);
    progress_allow(0);
    progress_at("request");
}

SessionDir::Saved SessionDir::saved(const std::shared_ptr<const SavedConversation>& image) const {
    if (!image) return Saved::no;
    std::lock_guard<std::mutex> lk(mu_);
    if (writing_ == image.get()) return Saved::writing;
    for (const Job& j : jobs_)
        if (j.image == image) return Saved::queued;
    const ConversationCheckpoint* deepest = session_deepest_checkpoint(image->checkpoints);
    for (const Entry& e : entries_) {
        if (e.state == State::writing || e.peek.cvec != image->cvec) continue;
        // its own file, or a file of this conversation further on (supersedes()'s rules, the file being the newer)
        if (e.image.lock() == image || head_prefix(image->live, e.peek.live) ||
            (deepest != nullptr && !deepest->ids.empty() &&
             (same_head(*deepest, e.peek.live) || same_head(*deepest, e.peek.deepest))))
            return Saved::yes;
    }
    return Saved::no;
}

size_t SessionDir::expedite(const SavedConversation* image) {
    size_t n = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        const auto now = std::chrono::steady_clock::now();
        for (Job& j : jobs_)
            if ((image != nullptr ? j.image.get() == image : j.image.use_count() == 1) && j.due > now) {
                j.due = now;
                ++n;
            }
        // the writer takes the front job when it is due: the ones moved up go first, in their order
        std::stable_sort(jobs_.begin(), jobs_.end(), [](const Job& a, const Job& b) { return a.due < b.due; });
    }
    if (n) cv_.notify_all();
    return n;
}

uint64_t SessionDir::orphan_bytes() const {
    std::lock_guard<std::mutex> lk(mu_);
    uint64_t n = 0;
    for (const Job& j : jobs_)
        if (j.image.use_count() == 1) n += j.image->bytes();
    // (the running write: the writer's job and writing_img_)
    if (writing_img_ && writing_img_.use_count() == 2) n += writing_img_->bytes();
    return n;
}

void SessionDir::flush() {
    std::unique_lock<std::mutex> lk(mu_);
    flushing_ = true;
    cv_.notify_all();
    cv_.wait(lk, [&] { return jobs_.empty() && writing_ == nullptr && turn_jobs_.empty() && !turn_busy_; });
    flushing_ = stop_;
}

bool SessionDir::flush_until(std::chrono::steady_clock::time_point deadline, const char* what) {
    std::unique_lock<std::mutex> lk(mu_);
    const auto queued = [&] { return jobs_.size() + turn_jobs_.size() + (writing_ != nullptr ? 1 : 0); };
    if (what != nullptr)
        say(std::string("flushing ") + std::to_string(queued()) + " session write(s) at " + what);
    const auto t0 = std::chrono::steady_clock::now();
    flushing_ = true;
    cv_.notify_all();
    const auto done = [&] { return jobs_.empty() && writing_ == nullptr && turn_jobs_.empty() && !turn_busy_; };
    bool ok = done();
    if (!ok) {
        const auto left = deadline - std::chrono::steady_clock::now();
        if (left > std::chrono::steady_clock::duration::zero()) ok = cv_.wait_for(lk, left, done);
    }
    flushing_ = stop_;
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (what != nullptr)
        say(ok ? "flush done after " + std::to_string((long long) (s * 1000.0)) + " ms"
               : "flush TIMED OUT after " + std::to_string((long long) s) + " s: " + std::to_string(queued()) +
                     " write(s) still queued");
    return ok;
}

void SessionDir::drop_queued() {
    std::lock_guard<std::mutex> lk(mu_);
    const size_t n = jobs_.size() + turn_jobs_.size();
    jobs_.clear();
    for (SessionTurnJob& j : turn_jobs_) j.track->broken = true;   // its turn did not land: the next one goes whole
    turn_jobs_.clear();
    if (writing_ != nullptr) write_cancel_.store(writing_, std::memory_order_relaxed);   // stops at its next block
    cv_.notify_all();
    say("dropped " + std::to_string(n) + " queued write(s) at exit: what is on disk is what the last turn left");
}

size_t SessionDir::files() const {
    std::lock_guard<std::mutex> lk(mu_);
    return entries_.size();
}

uint64_t SessionDir::bytes() const {
    std::lock_guard<std::mutex> lk(mu_);
    uint64_t n = 0;
    for (const auto& e : entries_) n += e.peek.bytes;
    return n;
}

void SessionDir::writer_loop() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        if (!turn_jobs_.empty()) {   // a turn's save never waits out the delay
            turn_write(lk);
            continue;
        }
        if (jobs_.empty()) {
            if (stop_) return;
            cv_.wait(lk);
            continue;
        }
        const auto now = std::chrono::steady_clock::now();
        if (!flushing_ && jobs_.front().due > now) {
            cv_.wait_until(lk, jobs_.front().due);   // (a turn job's notify wakes it too)
            continue;
        }
        Job job = std::move(jobs_.front());
        jobs_.pop_front();
        writing_ = job.image.get();
        writing_img_ = job.image;

        // an appending file of this conversation is extended (the newest one, when a rewritten tail left several)
        const bool use_log = o_.append && session_log_supported();
        std::string path;
        bool extend = false;
        if (use_log) {
            size_t best = entries_.size();
            for (size_t i = 0; i < entries_.size(); ++i)
                if (entries_[i].log && entries_[i].state == State::idle && entries_[i].peek.cvec == job.image->cvec &&
                    supersedes(job.image->live, job.image->checkpoints, entries_[i].peek) &&
                    (best == entries_.size() || entries_[i].seq > entries_[best].seq))
                    best = i;
            if (best != entries_.size()) {
                entries_[best].state = State::writing;
                path = entries_[best].path;
                extend = true;
            }
        }
        if (path.empty()) path = (fs::path(o_.dir) / random_name(use_log ? kSessionLogExt : ".session")).string();
        lk.unlock();

        SessionWriteOptions wo;
        wo.deepest_checkpoint_only = true;
        wo.min_free_bytes = o_.min_free_bytes;
        wo.progress = [this](uint64_t, uint64_t) {
            write_blocking_.store(0, std::memory_order_relaxed);
            write_moved_.fetch_add(1, std::memory_order_relaxed);
            if (o_.write_block) o_.write_block();
        };
        wo.phase = [this](const char*, uint64_t bytes) {
            write_blocking_.store(bytes, std::memory_order_relaxed);
            write_moved_.fetch_add(1, std::memory_order_relaxed);
        };
        const SavedConversation* const img = job.image.get();
        wo.cancelled = [this, img] { return write_cancel_.load(std::memory_order_relaxed) == img; };
        size_t written = 0;
        std::string err;
        SessionStatus st;
        const auto t0 = std::chrono::steady_clock::now();
        SessionLogStats ls;
        const bool ok = use_log ? session_log_save(path, *job.image, id_of(*job.image), ls, err, wo, &st, o_.log_format)
                                : session_file_write(path, *job.image, id_of(*job.image), written, err, wo, &st);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        lk.lock();
        writing_ = nullptr;
        writing_img_.reset();
        if (extend && !ok)   // (its file is as its last commit left it)
            for (Entry& e : entries_)
                if (e.path == path && e.state == State::writing) e.state = State::idle;
        if (ok) {
            add_written_locked(path, *job.image, use_log ? (size_t) ls.file_bytes : written, s, "wrote", nullptr,
                               job.image, use_log ? &ls : nullptr);
        } else if (write_cancel_.load(std::memory_order_relaxed) == job.image.get()) {
            say("write of " + std::to_string(job.image->live.ids.size()) + " tokens stopped after " +
                std::to_string((long long) (s * 1000.0)) + " ms: the conversation was taken back");
            note_locked("write_cancelled", job.image->live.ids.size(), 0, s * 1000.0, "claim");
        } else {
            say("write failed: " + err);
            note_locked("write_failed", job.image->live.ids.size(), 0, s * 1000.0, "error");
        }
        job.image.reset();   // an image evicted from the cache meanwhile is freed here, outside the cache's budget
        cv_.notify_all();
    }
}

void SessionDir::write_turn(SessionTurnJob job) {
    if (!job.turn || !job.track || job.turn->meta.live.ids.empty()) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        turn_jobs_.push_back(std::move(job));
    }
    cv_.notify_all();
}

void SessionDir::turn_write(std::unique_lock<std::mutex>& lk) {
    SessionTurnJob j = std::move(turn_jobs_.front());
    turn_jobs_.pop_front();
    const SavedConversation& meta = j.turn->meta;
    const bool delta = j.turn->delta();
    std::string path, why;
    auto take = [&](size_t i) { entries_[i].state = State::writing; path = entries_[i].path; };
    if (delta) {
        if (j.track->broken.load() || j.track->written_seq.load() != j.base_seq || j.track->path.empty())
            why = "the turn save before it did not land";
        else {
            for (size_t i = 0; i < entries_.size() && path.empty(); ++i)
                if (entries_[i].path == j.track->path && entries_[i].log && entries_[i].state == State::idle) take(i);
            if (path.empty()) why = "its file is gone or in use";
        }
    } else {
        // its own file first (the same place's conversation, perhaps rewound far back), else one it supersedes
        size_t best = entries_.size();
        for (size_t i = 0; i < entries_.size() && !j.track->path.empty(); ++i)
            if (entries_[i].path == j.track->path && entries_[i].log && entries_[i].state == State::idle) best = i;
        for (size_t i = 0; i < entries_.size() && j.track->path.empty(); ++i)
            if (entries_[i].log && entries_[i].state == State::idle && entries_[i].peek.cvec == meta.cvec &&
                // (also a file whose deepest checkpoint starts this conversation: the same one after a rewind, when the
                // chain's copy of that checkpoint is gone - a turn save names only the newest, a restore keeps what it reused)
                (supersedes(meta.live, meta.checkpoints, entries_[i].peek) || head_prefix(entries_[i].peek.deepest, meta.live)) &&
                (best == entries_.size() || entries_[i].seq > entries_[best].seq))
                best = i;
        if (best != entries_.size()) take(best);
    }
    if (!why.empty()) {
        j.track->broken = true;
        say("turn save (" + j.label + ") skipped: " + why + "; the next one is written whole");
        note_locked("turn_skipped", meta.live.ids.size(), 0, 0, j.label.c_str());
        cv_.notify_all();
        return;
    }
    const bool extend = !path.empty();
    if (!extend) path = (fs::path(o_.dir) / random_name(kSessionLogExt)).string();
    turn_busy_ = true;
    lk.unlock();
    SessionWriteOptions wo;
    wo.deepest_checkpoint_only = true;
    wo.min_free_bytes = o_.min_free_bytes;
    wo.progress = [this](uint64_t, uint64_t) { if (o_.write_block) o_.write_block(); };
    SessionLogStats ls;
    std::string err;
    SessionStatus st;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = session_log_save_turn(path, *j.turn, j.track->commit.load(), id_of(meta), ls, err, wo, &st, o_.log_format);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    lk.lock();
    turn_busy_ = false;
    if (ok) {
        add_written_locked(path, meta, (size_t) ls.file_bytes, ms / 1000.0, nullptr, nullptr, nullptr, &ls);
        j.track->path = path;
        j.track->commit = ls.commit;
        j.track->written_seq = j.seq;
        j.track->saved_tokens = (int64_t) meta.live.ids.size();
        j.track->broken = false;
        char b[360];
        std::snprintf(b, sizeof b, "turn save (%s): %zu tokens, %llu MiB written, %llu MiB already in its file, %llu MiB "
                      "hashed, %.0f ms in the background; the capture took the serve thread %.0f ms%s", j.label.c_str(),
                      meta.live.ids.size(), (unsigned long long) (ls.written_bytes >> 20),
                      (unsigned long long) (ls.reused_bytes >> 20), (unsigned long long) (ls.hashed_bytes >> 20), ms,
                      j.capture_ms, delta ? "" : " (whole capture)");
        say(b);
        if (ls.verify_bad)
            say("turn save VERIFY: " + std::to_string(ls.verify_bad) + " chunk(s) a delta would have left out differ from the file's");
        note_locked("turn_saved", meta.live.ids.size(), ls.written_bytes, ms, j.label.c_str());
        state_locked();
    } else {
        for (Entry& e : entries_)
            if (e.path == path && e.state == State::writing) e.state = State::idle;
        j.track->broken = true;
        say("turn save (" + j.label + ") failed: " + err + "; the next one is written whole");
        note_locked("write_failed", meta.live.ids.size(), 0, ms, "turn");
    }
    cv_.notify_all();
}

void SessionDir::add_written_locked(const std::string& path, const SavedConversation& image, size_t written, double s,
                                    const char* what, const std::vector<ConversationCheckpoint>* chain,
                                    const std::shared_ptr<const SavedConversation>& from, const SessionLogStats* log) {
    Entry e;
    e.path = path;
    e.peek = peek_of(image, written);
    e.seq = ++seq_;
    e.used = std::chrono::system_clock::now();
    e.log = log != nullptr;
    e.image = from;
    if (log != nullptr) e.peek.version = 3;
    // (a file that was extended in place: its old index entry goes, the file stays)
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [&](const Entry& x) { return x.path == path; }),
                   entries_.end());
    size_t dropped = 0;
    for (size_t i = 0; i < entries_.size();) {
        // (a file the writer is extending stays: its own write decides what it holds next)
        if (entries_[i].state != State::loading && entries_[i].state != State::writing &&
            supersedes(image.live, chain ? *chain : image.checkpoints, entries_[i].peek)) {
            drop_entry_locked(i, "superseded");
            ++dropped;
        } else {
            ++i;
        }
    }
    if (what == nullptr) {   // (a turn save says its own line)
        entries_.push_back(std::move(e));
        enforce_caps_locked();
        return;
    }
    char b[280];
    if (log != nullptr && log->appended)
        std::snprintf(b, sizeof b, "appended %zu tokens: %llu MiB written, %llu MiB already in its file (%zu MiB) in "
                      "%.1f s (%zu older file(s) of it dropped)", image.live.ids.size(),
                      (unsigned long long) (log->written_bytes >> 20), (unsigned long long) (log->reused_bytes >> 20),
                      written >> 20, s, dropped);
    else
        std::snprintf(b, sizeof b, "%s %zu tokens, %zu MiB in %.1f s (%zu older file(s) of it dropped)%s", what,
                      image.live.ids.size(), written >> 20, s, dropped,
                      log != nullptr && log->compacted ? "; its file was rewritten whole to drop replaced chunks" : "");
    say(b);
    // (sess_bytes: what this save put on the disk - an append's is its new chunks only)
    note_locked("wrote", image.live.ids.size(), log != nullptr ? (uint64_t) log->written_bytes : written, s * 1000.0,
                exiting_ ? "exit" : log != nullptr && log->appended ? "appended"
                         : std::strcmp(what, "wrote") == 0 ? "background" : "direct");
    entries_.push_back(std::move(e));
    enforce_caps_locked();
    state_locked();
}

bool SessionDir::write_now(const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                           const std::vector<std::vector<SessionKvSource>>& stage_kv, size_t& bytes, std::string& error,
                           const std::vector<ConversationCheckpoint>* chain) {
    const std::string path = (fs::path(o_.dir) / random_name()).string();
    SessionWriteOptions wo;
    wo.min_free_bytes = o_.min_free_bytes;
    SessionStatus st;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = meta.stage_images.empty() ? session_file_write(path, meta, kv, id_of(meta), bytes, error, wo, &st)
                                              : session_file_write(path, meta, kv, stage_kv, id_of(meta), bytes, error, wo, &st);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> lk(mu_);
    if (!ok) {
        say("write failed: " + error);
        note_locked("write_failed", meta.live.ids.size(), 0, s * 1000.0, "error");
        return false;
    }
    add_written_locked(path, meta, bytes, s, "wrote (directly, low RAM)", chain);
    return true;
}

void SessionDir::preload_loop() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        cv_.wait(lk, [&] { return stop_ || !fetching_; });   // a request's own read goes first
        if (stop_) return;
        Entry* pick = nullptr;
        for (auto& e : entries_)
            if (e.state == State::idle && preloaded_files_ < o_.preload_files &&
                preloaded_bytes_ + e.peek.bytes <= o_.preload_bytes && (!pick || e.seq > pick->seq))
                pick = &e;
        if (!pick) return;
        pick->state = State::loading;
        const std::string path = pick->path;
        lk.unlock();
        SavedConversation image;
        std::string err;
        const auto t0 = std::chrono::steady_clock::now();
        bool transient = false;
        const bool ok = read_file(path, image, err, &transient);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        lk.lock();
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].path != path) continue;
            if (ok) {
                entries_[i].state = State::preloaded;
                preloaded_bytes_ += entries_[i].peek.bytes;
                ++preloaded_files_;
                char b[200];
                std::snprintf(b, sizeof b, "preloaded %zu tokens in %.1f s", image.live.ids.size(), s);
                say(b);
                ready_.push_back({path, std::move(image)});
                state_locked();
            } else if (transient) {   // kept: no RAM to preload it now (and no retrying in a loop: preloading stops)
                entries_[i].state = State::idle;
                say("preload stopped: " + err + " (the file is kept)");
                cv_.notify_all();
                return;
            } else {
                say("preload failed: " + err);
                drop_entry_locked(i, "unreadable");
            }
            break;
        }
        cv_.notify_all();
    }
}

std::vector<SavedConversation> SessionDir::take_preloaded() {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<SavedConversation> out;
    for (auto& r : ready_) {
        for (auto& e : entries_)
            if (e.path == r.path) e.state = State::idle;   // the caller's cache holds it now; the file stays
        out.push_back(std::move(r.image));
    }
    ready_.clear();
    return out;
}

int64_t SessionDir::match(const Entry& e, const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs,
                          bool cvec) const {
    if (e.peek.cvec != cvec) return 0;
    const int64_t t = conversation_prefix(e.peek.live, ids, imgs);
    return t > 0 ? t : conversation_prefix(e.peek.deepest, ids, imgs);
}

int64_t SessionDir::match_image(const SavedConversation& s, const std::vector<int32_t>& ids,
                                const std::vector<ConversationImageKey>& imgs, bool cvec) const {
    if (s.cvec != cvec) return 0;
    int64_t t = conversation_prefix(s.live, ids, imgs);
    for (const auto& c : s.checkpoints) t = std::max(t, conversation_prefix(c, ids, imgs));
    return t;
}

int64_t SessionDir::best_tokens(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& imgs,
                                bool cvec) const {
    std::lock_guard<std::mutex> lk(mu_);
    int64_t t = 0;
    for (const auto& e : entries_) t = std::max(t, match(e, ids, imgs, cvec));
    for (const Job& j : jobs_) t = std::max(t, match_image(*j.image, ids, imgs, cvec));
    if (writing_img_) t = std::max(t, match_image(*writing_img_, ids, imgs, cvec));
    return t;
}

std::optional<SavedConversation> SessionDir::fetch(const std::vector<int32_t>& ids,
                                                   const std::vector<ConversationImageKey>& imgs, bool cvec,
                                                   int64_t beat, int64_t& tokens, bool* in_ram) {
    if (in_ram) *in_ram = false;
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        size_t best = entries_.size();
        int64_t best_t = beat;
        for (size_t i = 0; i < entries_.size(); ++i) {
            const int64_t t = match(entries_[i], ids, imgs, cvec);
            if (t > best_t) { best = i; best_t = t; }
        }
        // A conversation the cache evicted while its write was still queued (write_delay) or running: only the
        // writer holds it (measured 11-08: parked, evicted by the cache's count cap 24 s later, asked for 5 s after
        // that, its file done 11 s later still - the request read 425K tokens from 0 in 397 s).  Taken only when
        // nothing else holds it (the job's reference, writing_img_'s, this one): an image still in the cache is the
        // cache's to give, and its write goes on.
        std::shared_ptr<const SavedConversation> mem;
        int64_t mem_t = best_t;
        bool mem_running = false;
        for (const Job& j : jobs_) {
            const int64_t t = match_image(*j.image, ids, imgs, cvec);
            if (t > mem_t && j.image.use_count() == 1) { mem = j.image; mem_t = t; mem_running = false; }
        }
        if (writing_img_ && writing_img_.get() == writing_) {
            const int64_t t = match_image(*writing_img_, ids, imgs, cvec);
            // (the writer's job + writing_img_)
            if (t > mem_t && writing_img_.use_count() == 2) { mem = writing_img_; mem_t = t; mem_running = true; }
        }
        if (mem) {
            const auto t0 = std::chrono::steady_clock::now();
            if (mem_running) {
                wait_write_locked(lk, mem.get(), "request: stopping the write of the conversation it takes back");
            } else {
                jobs_.erase(std::remove_if(jobs_.begin(), jobs_.end(), [&](const Job& j) { return j.image == mem; }),
                            jobs_.end());
            }
            // (the writer let go of it under this lock: the only reference left is `mem`)
            if (mem.use_count() == 1) {
                SavedConversation out = std::move(const_cast<SavedConversation&>(*mem));
                mem.reset();
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                char b[200];
                std::snprintf(b, sizeof b, "took %zu tokens back from the %s (it had left the cache before its file was "
                              "written; %.0f ms)", out.live.ids.size(), mem_running ? "running write" : "write queue", ms);
                say(b);
                note_locked("taken_back", out.live.ids.size(), 0, ms, mem_running ? "running" : "queued");
                tokens = mem_t;
                if (in_ram) *in_ram = true;
                return out;
            }
            continue;   // (someone took a reference meanwhile: look again)
        }
        if (best == entries_.size()) return std::nullopt;
        Entry& e = entries_[best];
        const std::string path = e.path;
        if (e.state == State::loading || e.state == State::writing) {   // being read by the preload, or extended
            cv_.wait(lk, [&] {
                for (const auto& x : entries_)
                    if (x.path == path) return x.state != State::loading && x.state != State::writing;
                return true;
            });
            continue;
        }
        if (e.state == State::preloaded) {
            for (size_t r = 0; r < ready_.size(); ++r) {
                if (ready_[r].path != path) continue;
                SavedConversation out = std::move(ready_[r].image);
                ready_.erase(ready_.begin() + (std::ptrdiff_t) r);
                e.state = State::idle;
                touch_locked(e);
                tokens = best_t;
                return out;
            }
            e.state = State::idle;
            continue;
        }
        e.state = State::loading;
        fetching_ = true;
        lk.unlock();
        SavedConversation image;
        std::string err;
        const auto t0 = std::chrono::steady_clock::now();
        bool transient = false;
        const bool ok = read_file(path, image, err, &transient);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        lk.lock();
        fetching_ = false;
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].path != path) continue;
            if (ok) {
                entries_[i].state = State::idle;
                touch_locked(entries_[i]);
            } else if (transient) {   // kept for a later turn: the RAM was short, not the file
                entries_[i].state = State::idle;
                say("read deferred: " + err + " (the file is kept)");
                cv_.notify_all();
                // the request must NOT loop back up: the kept entry would re-match unchanged, be taken again,
                // and be refused again forever.  nullopt (tokens left at 0) sends this request down the plain
                // prompt path; a later turn with more RAM re-reads the kept file.
                return std::nullopt;
            } else {
                say("read failed: " + err);
                drop_entry_locked(i, "unreadable");
            }
            break;
        }
        cv_.notify_all();
        if (ok) {
            char b[160];
            std::snprintf(b, sizeof b, "read %zu tokens for this request in %.1f s", image.live.ids.size(), s);
            say(b);
            {   // (the folder's lock is held again here)
                uint64_t sz = 0;
                for (const auto& x : entries_) if (x.path == path) sz = x.peek.bytes;
                note_locked("read", image.live.ids.size(), sz, s * 1000.0, "request");
                state_locked();
            }
            tokens = best_t;
            return image;
        }
    }
}

bool SessionDir::read_file(const std::string& path, SavedConversation& image, std::string& error, bool* transient) const {
    size_t bytes = 0;
    SessionStatus st;
    if (transient) *transient = false;
    const bool log = fs::path(path).extension() == kSessionLogExt;
    auto read = [&](const SessionFileIdentity& id) {
        // (v1/v2: K/V as views into the mapping, no RAM copy; an appending file's chunks are read into owned buffers)
        return log ? session_log_read(path, id, image, bytes, error, o_.limits, &st, o_.log_format)
                   : session_file_map(path, id, image, bytes, error, o_.limits, &st);
    };
    if (read(o_.id)) return true;
    if (st.error == SessionError::memory) {   // the RAM admission refused it: the file is fine, try again later
        if (transient) *transient = true;
        return false;
    }
    if (o_.id_yarn.config == 0 || !read(o_.id_yarn)) {
        if (transient && st.error == SessionError::memory) *transient = true;
        return false;
    }
    // a YaRN conversation's file: the level rides on every state it restores
    image.live.rope_level = 1;
    for (auto& p : image.live.stage_parts) p.rope_level = 1;
    for (auto& c : image.checkpoints) {
        c.rope_level = 1;
        for (auto& p : c.stage_parts) p.rope_level = 1;
    }
    for (auto& si : image.stage_images) {
        si.live.rope_level = 1;
        for (auto& c : si.checkpoints) c.rope_level = 1;
    }
    return true;
}

void SessionDir::drop_entry_locked(size_t i, const char* why) {
    std::error_code ec;
    fs::remove(entries_[i].path, ec);
    const char* key = std::strstr(why, "superseded") ? "superseded" : std::strstr(why, "age") ? "age"
                    : std::strstr(why, "least recently") ? "lru" : std::strstr(why, "unreadable") ? "unreadable" : "other";
    note_locked("deleted", entries_[i].peek.live.ids.size(), entries_[i].peek.bytes, 0, key);
    say(std::string("deleted ") + fs::path(entries_[i].path).filename().string() + " (" + why + ")");
    entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
}

void SessionDir::note_locked(const char* what, uint64_t tokens, uint64_t bytes, double ms, const char* why) {
    if (!o_.event) return;
    char b[256];
    std::snprintf(b, sizeof b, "sess_event=%s sess_tokens=%llu sess_bytes=%llu sess_ms=%.0f sess_why=%s", what,
                  (unsigned long long) tokens, (unsigned long long) bytes, ms, why);
    o_.event(b);
}

void SessionDir::state_locked() {
    if (!o_.event) return;
    std::vector<const Entry*> order;
    for (const auto& e : entries_) order.push_back(&e);
    std::sort(order.begin(), order.end(), [](const Entry* a, const Entry* b) { return a->seq > b->seq; });
    uint64_t total = 0;
    for (const auto& e : entries_) total += e.peek.bytes;
    std::string j = "sess_json={\"files\":" + std::to_string(entries_.size()) + ",\"bytes\":" + std::to_string(total) +
                    ",\"max_bytes\":" + std::to_string(o_.max_bytes) + ",\"max_files\":" + std::to_string(o_.max_files) +
                    ",\"max_age_days\":" + std::to_string(o_.max_age.count() / 24) + ",\"conversations\":[";
    for (size_t i = 0; i < order.size(); ++i) {
        const Entry& e = *order[i];
        const long long used = (long long) std::chrono::duration_cast<std::chrono::seconds>(e.used.time_since_epoch()).count();
        j += (i ? "," : "");
        j += "{\"tokens\":" + std::to_string(e.peek.live.ids.size()) + ",\"bytes\":" + std::to_string(e.peek.bytes) +
             ",\"last_used\":" + std::to_string(used) + ",\"state\":\"" +
             (e.state == State::loading ? "reading" : e.state == State::preloaded ? "in_ram" : "on_disk") + "\"}";
    }
    j += "]}";
    o_.event(j);
}

void SessionDir::begin_exit() {
    std::lock_guard<std::mutex> lk(mu_);
    exiting_ = true;
}

void SessionDir::touch_locked(Entry& e) {
    e.seq = ++seq_;
    e.used = std::chrono::system_clock::now();
    std::error_code ec;
    fs::last_write_time(e.path, fs::file_time_type::clock::now(), ec);   // recency survives a restart
}

void SessionDir::enforce_caps_locked() {
    const auto cutoff = std::chrono::system_clock::now() - o_.max_age;
    for (size_t i = 0; i < entries_.size();) {
        if (entries_[i].state == State::idle && entries_[i].used < cutoff) drop_entry_locked(i, "not used for the folder's age limit");
        else ++i;
    }
    for (;;) {
        uint64_t total = 0;
        for (const auto& e : entries_) total += e.peek.bytes;
        if (entries_.size() <= o_.max_files && total <= o_.max_bytes) return;
        size_t oldest = entries_.size();
        for (size_t i = 0; i < entries_.size(); ++i)
            if (entries_[i].state == State::idle && (oldest == entries_.size() || entries_[i].seq < entries_[oldest].seq))
                oldest = i;
        if (oldest == entries_.size()) return;   // everything left is being read: next time
        drop_entry_locked(oldest, "least recently used, over the folder's cap");
    }
}

} // namespace strata::core
