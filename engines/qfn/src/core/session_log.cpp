// Session files that grow by appending (session_log.hpp), format v3.
//
//   header  64 B  magic[8] version:u32 header_size:u32 model:u64 config:u64 chunk_bytes:u64 reserved:u64[2] hash:u64
//   slot A  64 B  \ the two commit pointers, written in turn (seq odd: A, even: B): magic:u64 seq:u64 commit_off:u64
//   slot B  64 B  / commit_len:u64 commit_hash:u64 file_end:u64 named_bytes:u64 hash:u64
//   data          chunks (raw bytes, back to back) and commit records, in the order they were written
//
// A commit record is the image without its arrays (geometry, layer range, pictures, K/V headers) and a table of every
// array: key, size and each chunk's offset and hash.  A save writes its new chunks and its commit past the current
// commit's end, flushes, then writes the OTHER slot and flushes again; a reader takes the valid slot with the highest
// seq.  So a save that stops anywhere leaves the previous commit and everything it names intact.
// Little-endian, like v1.  The hashes detect corruption; they do not authenticate a file.
#include "strata/core/session_log.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

namespace strata::core {

#ifdef _WIN32
bool session_log_supported() { return false; }
bool session_log_save(const std::string&, const SavedConversation&, const SessionFileIdentity&, SessionLogStats&,
                      std::string& error, const SessionWriteOptions&, SessionStatus* st, const SessionLogOptions&) {
    if (st) st->error = SessionError::invalid;
    error = "session log: not built on this platform";
    return false;
}
bool session_log_read(const std::string&, const SessionFileIdentity&, SavedConversation&, size_t&, std::string& error,
                      const SessionReadLimits&, SessionStatus* st, const SessionLogOptions&) {
    if (st) st->error = SessionError::invalid;
    error = "session log: not built on this platform";
    return false;
}
bool session_log_peek(const std::string&, const SessionFileIdentity&, SessionPeek&, std::string& error) {
    error = "session log: not built on this platform";
    return false;
}
#else

namespace {
constexpr char kMagic[8] = {'S', 'A', 'G', 'S', 'L', 'O', 'G', '\x01'};
constexpr uint32_t kVersion = 3;
constexpr uint64_t kHeader = 64, kSlot = 64, kData = kHeader + 2 * kSlot;
constexpr uint64_t kSlotMagic = 0x0154'4f4c'5347'4153ull;   // "SAGSLOT\1"
constexpr uint64_t kCommitSeed = 0x636f'6d6d'6974ull;
constexpr uint64_t kMaxMeta = 1ull << 30;
constexpr size_t kBatch = 16u << 20;

SessionError errno_kind(int e) {
    return e == ENOSPC || e == EDQUOT || e == EFBIG ? SessionError::storage
         : e == ENOMEM ? SessionError::memory : SessionError::io;
}

uint64_t get64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
void put64(uint8_t* p, uint64_t v) { std::memcpy(p, &v, 8); }

struct W {
    std::vector<uint8_t> b;
    void u64(uint64_t v) { const size_t n = b.size(); b.resize(n + 8); put64(b.data() + n, v); }
    void i64(int64_t v) { u64((uint64_t) v); }
};
struct R {
    const uint8_t* p;
    const uint8_t* e;
    bool ok = true;
    uint64_t left() const { return (uint64_t) (e - p); }
    uint64_t u64() {
        if (!ok || left() < 8) { ok = false; return 0; }
        const uint64_t v = get64(p);
        p += 8;
        return v;
    }
    int64_t i64() { return (int64_t) u64(); }
    // a count whose elements take at least `elem` bytes each of what is left
    uint64_t count(uint64_t elem) {
        const uint64_t n = u64();
        if (ok && n > left() / elem) ok = false;
        return ok ? n : 0;
    }
};

// ---- the arrays of an image, in one fixed order
enum Group : uint64_t { kLiveState = 0, kCkState = 1, kKv = 2, kLiveIds = 3, kCkIds = 4 };
uint64_t key_of(uint64_t image, uint64_t group, uint64_t layer, uint64_t part) {
    return image << 48 | group << 40 | layer << 8 | part;
}
struct Ref {
    uint64_t key = 0, size = 0;
    const uint8_t* flat = nullptr;
    const ConversationBuffer* buf = nullptr;
    uint64_t from = 0, claim = 0;   // a captured turn: `buf` holds [from, size); see SessionLogKvTail
    template<class Fn> bool visit(uint64_t at, uint64_t n, Fn fn) const {
        if (n == 0) return true;
        if (buf == nullptr) return fn(flat + at, (size_t) n);
        if (at < from) return false;
        return buf->visit((size_t) (at - from), (size_t) n, [&](const uint8_t* q, size_t c, size_t) { return fn(q, c); });
    }
};
struct Chunk { uint64_t off = UINT64_MAX, hash = 0; };
struct Array {
    uint64_t key = 0, size = 0;
    std::vector<Chunk> chunks;
};
uint64_t chunks_of(uint64_t size, uint64_t chunk) { return (size + chunk - 1) / chunk; }
uint64_t chunk_len(uint64_t size, uint64_t chunk, uint64_t i) { return std::min(chunk, size - i * chunk); }

template<class T> Ref flat_ref(uint64_t key, const std::vector<T>& v) {
    return {key, (uint64_t) v.size() * sizeof(T), reinterpret_cast<const uint8_t*>(v.data()), nullptr};
}
void state_refs(std::vector<Ref>& out, uint64_t image, uint64_t group, uint64_t layer, const ConversationCheckpoint& c) {
    const std::vector<uint8_t>* parts[5] = {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos};
    for (uint64_t p = 0; p < 5; ++p) out.push_back(flat_ref(key_of(image, group, layer, p), *parts[p]));
}
bool same_ids(const ConversationCheckpoint& a, const ConversationCheckpoint& b) { return a.ids == b.ids; }

// which checkpoints of each image are written (conversation_file.cpp's deepest_pick, or all of them)
std::vector<size_t> picks_of(const SavedConversation& s, bool deepest_only) {
    std::vector<size_t> all(s.checkpoints.size());
    for (size_t i = 0; i < all.size(); ++i) all[i] = i;
    if (!deepest_only) return all;
    if (s.checkpoints.empty()) return {};
    for (const auto& st : s.stage_images)
        if (st.checkpoints.size() != s.checkpoints.size()) return {};
    bool any_pinned = false;
    for (const auto& c : s.checkpoints) any_pinned = any_pinned || c.pinned;
    size_t best = s.checkpoints.size();
    for (size_t i = 0; i < s.checkpoints.size(); ++i)
        if ((!any_pinned || s.checkpoints[i].pinned) &&
            (best == s.checkpoints.size() || s.checkpoints[i].ids.size() > s.checkpoints[best].ids.size())) best = i;
    for (const auto& st : s.stage_images)
        if (st.checkpoints[best].ids != s.checkpoints[best].ids) return {};
    return {best};
}

struct ImagePlan {
    const SavedConversation* s = nullptr;
    const std::vector<SessionLogKvTail>* tails = nullptr;   // a captured turn's K/V (s->kv is empty then)
    std::vector<size_t> pick;
    bool live_from_first = false;
    std::vector<uint8_t> ck_from_first;
};
// a later stage's token ids are the first stage's: written once
std::vector<ImagePlan> plan_images(const SavedConversation& image, bool deepest_only) {
    std::vector<ImagePlan> out;
    const std::vector<size_t> pick0 = picks_of(image, deepest_only);
    auto add = [&](const SavedConversation& s, bool first) {
        ImagePlan p;
        p.s = &s;
        if (deepest_only) p.pick = pick0;
        else for (size_t i = 0; i < s.checkpoints.size(); ++i) p.pick.push_back(i);
        p.live_from_first = !first && same_ids(s.live, image.live);
        for (size_t j = 0; j < p.pick.size(); ++j)
            p.ck_from_first.push_back(!first && j < pick0.size() && p.pick[j] < s.checkpoints.size() &&
                                      same_ids(s.checkpoints[p.pick[j]], image.checkpoints[pick0[j]]) ? 1 : 0);
        out.push_back(std::move(p));
    };
    add(image, true);
    for (const auto& st : image.stage_images) add(st, false);
    return out;
}
std::vector<Ref> refs_of(const std::vector<ImagePlan>& plan) {
    std::vector<Ref> out;
    for (uint64_t i = 0; i < plan.size(); ++i) {
        const ImagePlan& p = plan[i];
        const SavedConversation& s = *p.s;
        if (!p.live_from_first) out.push_back(flat_ref(key_of(i, kLiveIds, 0, 0), s.live.ids));
        state_refs(out, i, kLiveState, 0, s.live);
        for (uint64_t j = 0; j < p.pick.size(); ++j) {
            const ConversationCheckpoint& c = s.checkpoints[p.pick[j]];
            if (!p.ck_from_first[j]) out.push_back(flat_ref(key_of(i, kCkIds, j, 0), c.ids));
            state_refs(out, i, kCkState, j, c);
        }
        for (uint64_t l = 0; p.tails != nullptr && l < p.tails->size(); ++l) {
            const SessionLogKvTail& k = (*p.tails)[l];
            for (uint64_t part = 0; part < 5; ++part)
                out.push_back({key_of(i, kKv, l, part), k.size[part], nullptr, &k.bytes[part], k.from[part], k.claim[part]});
        }
        for (uint64_t l = 0; p.tails == nullptr && l < s.kv.size(); ++l) {
            const ConversationKv& k = s.kv[l];
            const ConversationBuffer* parts[5] = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
            for (uint64_t part = 0; part < 5; ++part)
                out.push_back({key_of(i, kKv, l, part), (uint64_t) parts[part]->size(), nullptr, parts[part]});
        }
    }
    return out;
}

void put_ck_meta(W& w, const ConversationCheckpoint& c, bool from_first) {
    w.u64(c.imgs.size());
    for (const auto& k : c.imgs) { w.i64(k.start); w.u64(k.hash); }
    w.u64(c.used);
    w.u64(from_first ? 1 : 0);
}
std::vector<uint8_t> meta_of(const std::vector<ImagePlan>& plan, const std::vector<Array>& arrays) {
    W w;
    w.u64(plan.size());
    for (const ImagePlan& p : plan) {
        const SavedConversation& s = *p.s;
        for (int64_t g : s.geometry) w.i64(g);
        w.i64(s.layer_lo); w.i64(s.layer_hi); w.u64(s.cvec ? 1 : 0);
        put_ck_meta(w, s.live, p.live_from_first);
        w.u64(p.pick.size());
        for (size_t j = 0; j < p.pick.size(); ++j) put_ck_meta(w, s.checkpoints[p.pick[j]], p.ck_from_first[j] != 0);
        if (p.tails != nullptr) {
            w.u64(p.tails->size());
            for (const auto& k : *p.tails)
                for (int64_t v : {int64_t(k.format), k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim}) w.i64(v);
            continue;
        }
        w.u64(s.kv.size());
        for (const auto& k : s.kv)
            for (int64_t v : {int64_t(k.format), k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim}) w.i64(v);
    }
    w.u64(arrays.size());
    for (const Array& a : arrays) {
        w.u64(a.key); w.u64(a.size); w.u64(a.chunks.size());
        for (const Chunk& c : a.chunks) { w.u64(c.off); w.u64(c.hash); }
    }
    return std::move(w.b);
}

// ---- a parsed commit: the images without their arrays, and the table
struct Skeleton {
    struct Ck { ConversationCheckpoint c; bool from_first = false; };
    struct Image {
        SavedConversation s;   // geometry, range, cvec, K/V headers; live / checkpoints filled from `live`, `cks`
        Ck live;
        std::vector<Ck> cks;
    };
    std::vector<Image> images;
    std::vector<Array> arrays;
};
bool get_ck_meta(R& r, Skeleton::Ck& k) {
    const uint64_t n = r.count(16);
    k.c.imgs.resize((size_t) n);
    for (auto& im : k.c.imgs) { im.start = r.i64(); im.hash = r.u64(); }
    k.c.used = r.u64();
    const uint64_t ff = r.u64();
    if (ff > 1) r.ok = false;
    k.from_first = ff == 1;
    return r.ok;
}
bool parse_meta(const std::vector<uint8_t>& meta, uint64_t chunk, uint64_t commit_off, Skeleton& sk, std::string& error) {
    R r{meta.data(), meta.data() + meta.size()};
    auto fail = [&](const char* m) { error = std::string("session log: ") + m; return false; };
    const uint64_t n_img = r.count(18 * 8 + 3 * 8 + 4 * 8 + 16);
    if (!r.ok || n_img == 0 || n_img > 64) return fail("invalid image count");
    sk.images.resize((size_t) n_img);
    for (auto& im : sk.images) {
        for (auto& g : im.s.geometry) g = r.i64();
        im.s.layer_lo = r.i64(); im.s.layer_hi = r.i64();
        const uint64_t cvec = r.u64();
        if (!r.ok || cvec > 1) return fail("invalid image header");
        im.s.cvec = cvec == 1;
        if (!get_ck_meta(r, im.live)) return fail("invalid live state header");
        const uint64_t n_ck = r.count(24);
        if (!r.ok) return fail("invalid checkpoint count");
        im.cks.resize((size_t) n_ck);
        for (auto& k : im.cks) if (!get_ck_meta(r, k)) return fail("invalid checkpoint header");
        const uint64_t n_kv = r.count(56);
        if (!r.ok) return fail("invalid K/V layer count");
        im.s.kv.resize((size_t) n_kv);
        for (auto& k : im.s.kv) {
            const int64_t format = r.i64();
            k.cells = r.i64(); k.heads = r.i64(); k.head_dim = r.i64(); k.page_size = r.i64();
            k.pooled_rows = r.i64(); k.idx_dim = r.i64();
            if (!r.ok || format < INT32_MIN || format > INT32_MAX) return fail("invalid K/V header");
            k.format = (int) format;
        }
    }
    if (sk.images[0].live.from_first) return fail("the first image names no token ids");
    for (const auto& k : sk.images[0].cks) if (k.from_first) return fail("the first image names no token ids");
    const uint64_t n_arr = r.count(24);
    if (!r.ok) return fail("invalid array count");
    sk.arrays.resize((size_t) n_arr);
    for (auto& a : sk.arrays) {
        a.key = r.u64(); a.size = r.u64();
        const uint64_t n = r.count(16);
        if (!r.ok || a.size > (uint64_t) 1 << 46 || n != chunks_of(a.size, chunk)) return fail("invalid array entry");
        a.chunks.resize((size_t) n);
        for (uint64_t i = 0; i < n; ++i) {
            a.chunks[(size_t) i].off = r.u64();
            a.chunks[(size_t) i].hash = r.u64();
            const uint64_t off = a.chunks[(size_t) i].off, len = chunk_len(a.size, chunk, i);
            if (!r.ok || off < kData || off > commit_off || len > commit_off - off) return fail("a chunk lies outside the data");
        }
    }
    if (!r.ok || r.left() != 0) return fail("invalid commit record");
    // the table must be exactly the arrays the images call for, in order
    size_t at = 0;
    auto want = [&](uint64_t key) { return at < sk.arrays.size() && sk.arrays[at++].key == key; };
    for (uint64_t i = 0; i < sk.images.size(); ++i) {
        const auto& im = sk.images[(size_t) i];
        bool ok = im.live.from_first || want(key_of(i, kLiveIds, 0, 0));
        for (uint64_t p = 0; ok && p < 5; ++p) ok = want(key_of(i, kLiveState, 0, p));
        for (uint64_t j = 0; ok && j < im.cks.size(); ++j) {
            ok = im.cks[(size_t) j].from_first || want(key_of(i, kCkIds, j, 0));
            for (uint64_t p = 0; ok && p < 5; ++p) ok = want(key_of(i, kCkState, j, p));
        }
        for (uint64_t l = 0; ok && l < im.s.kv.size(); ++l)
            for (uint64_t p = 0; ok && p < 5; ++p) ok = want(key_of(i, kKv, l, p));
        if (!ok) return fail("the array table does not match the images");
    }
    if (at != sk.arrays.size()) return fail("the array table does not match the images");
    return true;
}

uint64_t chunk_hash(uint64_t key, uint64_t index, const Ref& ref, uint64_t chunk) {
    SessionHasher h(key * 0x9E3779B97F4A7C15ull + index);
    ref.visit(index * chunk, chunk_len(ref.size, chunk, index), [&](const uint8_t* p, size_t n) {
        h.update(p, n);
        return true;
    });
    return h.digest();
}

// fn(i) for i in [0, n) on a few threads; `poll` runs on the calling thread meanwhile (true: stop).  False when a
// call failed or it was stopped (`stopped` says which).
bool parallel(size_t n, int threads, const std::function<bool(size_t)>& fn, const std::function<bool()>& poll,
              bool& stopped) {
    stopped = false;
    if (threads <= 0) threads = (int) std::min<unsigned>(8, std::max<unsigned>(1, std::thread::hardware_concurrency() / 2));
    threads = (int) std::min<size_t>((size_t) threads, std::max<size_t>(1, n / 4));
    std::atomic<size_t> next{0};
    std::atomic<bool> stop{false}, failed{false};
    std::atomic<int> running{threads};
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t)
        th.emplace_back([&] {
            for (size_t i; !stop.load(std::memory_order_relaxed) && (i = next.fetch_add(1)) < n;)
                if (!fn(i)) { failed = true; stop = true; }
            { std::lock_guard<std::mutex> g(mu); --running; }
            cv.notify_all();
        });
    {
        std::unique_lock<std::mutex> lk(mu);
        while (running.load() > 0) {
            cv.wait_for(lk, std::chrono::milliseconds(20));
            if (running.load() > 0 && poll && !stop.load()) {
                lk.unlock();
                if (poll()) { stopped = true; stop = true; }
                lk.lock();
            }
        }
    }
    for (auto& t : th) t.join();
    return !failed.load() && !stopped;
}

bool pread_all(int fd, uint8_t* p, size_t n, uint64_t off) {
    while (n) {
        const ssize_t r = ::pread(fd, p, n, (off_t) off);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) { if (r == 0) errno = EIO; return false; }
        p += r; n -= (size_t) r; off += (uint64_t) r;
    }
    return true;
}
bool pwrite_all(int fd, const uint8_t* p, size_t n, uint64_t off) {
    while (n) {
        const ssize_t w = ::pwrite(fd, p, n, (off_t) off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) { if (w == 0) errno = EIO; return false; }
        p += w; n -= (size_t) w; off += (uint64_t) w;
    }
    return true;
}

struct Fd {
    int fd = -1;
    ~Fd() { if (fd >= 0) ::close(fd); }
};

void header_bytes(uint8_t* h, const SessionFileIdentity& id, uint64_t chunk) {
    std::memset(h, 0, kHeader);
    std::memcpy(h, kMagic, 8);
    const uint32_t size = (uint32_t) kData;
    std::memcpy(h + 8, &kVersion, 4);
    std::memcpy(h + 12, &size, 4);
    put64(h + 16, id.model); put64(h + 24, id.config); put64(h + 32, chunk);
    put64(h + 56, session_hash64(h, 56, 0));
}
struct Slot { uint64_t seq = 0, off = 0, len = 0, hash = 0, end = 0, named = 0; };
void slot_bytes(uint8_t* b, const Slot& s) {
    put64(b, kSlotMagic); put64(b + 8, s.seq); put64(b + 16, s.off); put64(b + 24, s.len);
    put64(b + 32, s.hash); put64(b + 40, s.end); put64(b + 48, s.named);
    put64(b + 56, session_hash64(b, 56, 1));
}

// A file opened and its newest readable commit found.
struct Opened {
    Fd f;
    uint64_t size = 0, chunk = 0;
    Slot slot;
    std::vector<uint8_t> meta;
};
bool open_log(const std::string& path, const SessionFileIdentity& id, bool rw, Opened& o, std::string& error,
              SessionError& kind) {
    kind = SessionError::invalid;
    auto fail = [&](const std::string& m) { error = "session log: " + m; return false; };
    o.f.fd = ::open(path.c_str(), (rw ? O_RDWR : O_RDONLY) | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
    if (o.f.fd < 0) {
        if (errno != ELOOP) kind = errno_kind(errno);
        return fail(errno == ELOOP ? std::string("a symbolic link, not a file") : std::string(std::strerror(errno)));
    }
    struct stat st {};
    if (::fstat(o.f.fd, &st) != 0) { kind = SessionError::io; return fail(std::strerror(errno)); }
    if (!S_ISREG(st.st_mode)) return fail("not a regular file");
    if (st.st_nlink != 1) return fail("the file has more than one name (hard link)");
    const int fl = ::fcntl(o.f.fd, F_GETFL);
    if (fl >= 0) ::fcntl(o.f.fd, F_SETFL, fl & ~O_NONBLOCK);
    o.size = (uint64_t) st.st_size;
    if (o.size < kData) return fail("too small to be a session log");
    uint8_t h[kData];
    if (!pread_all(o.f.fd, h, kData, 0)) { kind = SessionError::io; return fail(std::string("header read error: ") + std::strerror(errno)); }
    if (std::memcmp(h, kMagic, 8) != 0) return fail("not a session log (magic)");
    if (get64(h + 56) != session_hash64(h, 56, 0)) return fail("header checksum mismatch");
    uint32_t version = 0, hsize = 0;
    std::memcpy(&version, h + 8, 4);
    std::memcpy(&hsize, h + 12, 4);
    if (version != kVersion) return fail("unsupported version " + std::to_string(version));
    if (hsize != kData) return fail("invalid header fields");
    if (get64(h + 16) != id.model || get64(h + 24) != id.config)
        return fail("saved with another model or configuration than this runtime's");
    o.chunk = get64(h + 32);
    if (o.chunk < 4096 || o.chunk > (64u << 20)) return fail("invalid chunk size");
    Slot cand[2];
    int n = 0;
    for (int k = 0; k < 2; ++k) {
        const uint8_t* b = h + kHeader + (size_t) k * kSlot;
        if (get64(b) != kSlotMagic || get64(b + 56) != session_hash64(b, 56, 1)) continue;
        Slot s{get64(b + 8), get64(b + 16), get64(b + 24), get64(b + 32), get64(b + 40), get64(b + 48)};
        if (s.off < kData || s.len == 0 || s.len > kMaxMeta || s.off > o.size || s.len > o.size - s.off ||
            s.end != s.off + s.len) continue;   // (the next save writes from `end`: it must be this commit's)
        cand[n++] = s;
    }
    if (n == 2 && cand[0].seq < cand[1].seq) std::swap(cand[0], cand[1]);
    for (int k = 0; k < n; ++k) {
        std::vector<uint8_t> meta((size_t) cand[k].len);
        if (!pread_all(o.f.fd, meta.data(), meta.size(), cand[k].off)) continue;
        if (session_hash64(meta.data(), meta.size(), kCommitSeed) != cand[k].hash) continue;
        o.slot = cand[k];
        o.meta = std::move(meta);
        return true;
    }
    return fail("no readable commit");
}

std::string random_hex() {
    std::random_device rd;
    char b[24];
    std::snprintf(b, sizeof b, "%08x%08x", (unsigned) rd(), (unsigned) rd());
    return b;
}
bool free_bytes(const std::string& path, uint64_t& out) {
    const size_t slash = path.find_last_of('/');
    const std::string dir = slash == std::string::npos ? "." : slash == 0 ? "/" : path.substr(0, slash);
    struct statvfs s {};
    if (::statvfs(dir.c_str(), &s) != 0) return false;
    out = (uint64_t) s.f_bavail * (uint64_t) s.f_frsize;
    return true;
}
} // namespace

bool session_log_supported() { return true; }

namespace {
bool save_core(const std::string& path, const SavedConversation& image, const std::vector<ImagePlan>& plan,
               uint64_t turn_chunk, uint64_t base, const SessionFileIdentity& id, SessionLogStats& stats,
               std::string& error, const SessionWriteOptions& options, SessionStatus* status, const SessionLogOptions& log);
}
bool session_log_save(const std::string& path, const SavedConversation& image, const SessionFileIdentity& id,
                      SessionLogStats& stats, std::string& error, const SessionWriteOptions& options,
                      SessionStatus* status, const SessionLogOptions& log) {
    return save_core(path, image, plan_images(image, options.deepest_checkpoint_only), 0, 0, id, stats, error, options,
                     status, log);
}
bool session_log_save_turn(const std::string& path, const SessionLogTurn& turn, uint64_t base,
                           const SessionFileIdentity& id, SessionLogStats& stats, std::string& error,
                           const SessionWriteOptions& options, SessionStatus* status, const SessionLogOptions& log) {
    std::vector<ImagePlan> plan = plan_images(turn.meta, options.deepest_checkpoint_only);
    if (plan.size() != turn.kv.size()) {
        if (status) status->error = SessionError::invalid;
        error = "session log: the captured turn's stages do not line up";
        return false;
    }
    for (size_t i = 0; i < plan.size(); ++i) plan[i].tails = &turn.kv[i];
    return save_core(path, turn.meta, plan, turn.chunk_bytes, base, id, stats, error, options, status, log);
}
bool session_log_capture(SessionLogTurn& out, SavedConversation&& meta, const std::vector<SessionKvSource>& kv,
                         const std::vector<std::vector<SessionKvSource>>& stage_kv,
                         const std::vector<std::vector<std::array<uint64_t, 5>>>* unchanged, bool verify,
                         std::string& error) {
    out.meta = std::move(meta);
    out.kv.assign(1 + stage_kv.size(), {});
    for (size_t i = 0; i < out.kv.size(); ++i) {
        const std::vector<SessionKvSource>& src = i == 0 ? kv : stage_kv[i - 1];
        for (size_t l = 0; l < src.size(); ++l) {
            const SessionKvSource& s = src[l];
            SessionLogKvTail t;
            t.format = s.format; t.cells = s.cells; t.heads = s.heads; t.head_dim = s.head_dim;
            t.page_size = s.page_size; t.pooled_rows = s.pooled_rows; t.idx_dim = s.idx_dim;
            for (size_t p = 0; p < 5; ++p) {
                t.size[p] = s.sizes[p];
                uint64_t keep = unchanged != nullptr && i < unchanged->size() && l < (*unchanged)[i].size() ? (*unchanged)[i][l][p] : 0;
                keep = std::min<uint64_t>(keep, t.size[p]) / out.chunk_bytes * out.chunk_bytes;
                if (verify) t.claim[p] = keep; else t.from[p] = keep;
                const uint64_t from = t.from[p], n = t.size[p] - from;
                t.bytes[p].resize_for_overwrite((size_t) n);
                if (n && (!s.read || !t.bytes[p].visit(0, (size_t) n, [&](uint8_t* q, size_t c, size_t at) {
                        return s.read(p, (size_t) from + at, q, c);
                    }))) {
                    error = "session log: reading the K/V failed" + (s.error ? ": " + s.error() : std::string());
                    return false;
                }
            }
            out.kv[i].push_back(std::move(t));
        }
    }
    return true;
}

namespace {
bool save_core(const std::string& path, const SavedConversation& image, const std::vector<ImagePlan>& plan,
               uint64_t turn_chunk, uint64_t base, const SessionFileIdentity& id, SessionLogStats& stats,
               std::string& error, const SessionWriteOptions& options, SessionStatus* status, const SessionLogOptions& log) {
    SessionStatus local;
    SessionStatus& st = status ? *status : local;
    st = {};
    stats = {};
    auto fail = [&](SessionError kind, const std::string& m) { st.error = kind; error = "session log: " + m; return false; };
    auto fail_errno = [&](const char* what, int e) { return fail(errno_kind(e), std::string(what) + ": " + std::strerror(e)); };
    auto fault = [&](const char* step) { return options.fault ? options.fault(step) : 0; };
    if (image.live.ids.empty()) return fail(SessionError::invalid, "nothing to save");
    const std::vector<Ref> refs = refs_of(plan);
    bool delta = false;
    for (const Ref& r : refs) delta = delta || r.from > 0;

    // the file to extend, when there is a readable one of this identity
    Opened old;
    Skeleton old_sk;
    std::string ignore;
    SessionError ignore_kind;
    bool have = open_log(path, id, true, old, ignore, ignore_kind) &&
                parse_meta(old.meta, old.chunk, old.slot.off, old_sk, ignore);
    const uint64_t chunk = have ? old.chunk : turn_chunk ? turn_chunk : std::clamp<uint64_t>(log.chunk_bytes, 4096, 64u << 20);
    const bool on_base = have && base != 0 && old.slot.hash == base;
    // a delta names chunks it does not hold: only the file at the commit it continues has them
    if (delta && (!on_base || chunk != turn_chunk))
        return fail(SessionError::invalid, "the file is not at the commit this turn continues");

    // every chunk's hash
    std::vector<Array> arrays(refs.size());
    struct Task { uint32_t a; uint32_t i; };
    std::vector<Task> tasks;
    uint64_t total = 0;
    for (size_t a = 0; a < refs.size(); ++a) {
        arrays[a].key = refs[a].key;
        arrays[a].size = refs[a].size;
        arrays[a].chunks.resize((size_t) chunks_of(refs[a].size, chunk));
        for (size_t i = 0; i < arrays[a].chunks.size(); ++i)
            if ((i + 1) * chunk > refs[a].from) {   // (the chunks before `from` keep the file's hash)
                if (i * chunk < refs[a].from) return fail(SessionError::invalid, "a captured part does not start on a chunk");
                tasks.push_back({(uint32_t) a, (uint32_t) i});
                stats.hashed_bytes += chunk_len(refs[a].size, chunk, i);
            }
        total += refs[a].size;
    }
    bool stopped = false;
    if (options.progress) options.progress(0, total);
    if (!parallel(tasks.size(), log.threads, [&](size_t t) {
            const Task k = tasks[t];
            arrays[k.a].chunks[k.i].hash = chunk_hash(refs[k.a].key, k.i, refs[k.a], chunk);
            return true;
        }, [&] {
            if (options.progress) options.progress(0, total);
            return options.cancelled && options.cancelled();
        }, stopped))
        return fail(SessionError::io, stopped ? std::strerror(ECANCELED) : "hashing failed");

    // which chunks the file already holds
    std::unordered_map<uint64_t, const Array*> old_by_key;
    if (have)
        for (const Array& a : old_sk.arrays) old_by_key.emplace(a.key, &a);
    bool base_short = false;
    // a whole rewrite of a delta: the chunks it did not capture are copied from the old file (their offsets there)
    std::vector<std::vector<uint64_t>> copy_from(arrays.size());
    auto place = [&](bool reuse, uint64_t start, uint64_t& reused) {
        uint64_t cursor = start;
        reused = 0;
        for (size_t ai = 0; ai < arrays.size(); ++ai) {
            Array& a = arrays[ai];
            const Ref& ref = refs[ai];
            const Array* o = nullptr;
            if (const auto it = old_by_key.find(a.key); it != old_by_key.end()) o = it->second;
            copy_from[ai].assign(a.chunks.size(), UINT64_MAX);
            for (size_t i = 0; i < a.chunks.size(); ++i) {
                const uint64_t len = chunk_len(a.size, chunk, i);
                const bool in_old = o != nullptr && i < o->chunks.size() && chunk_len(o->size, chunk, i) == len;
                if ((i + 1) * chunk <= ref.from) {   // not captured: the file's own chunk, or the delta is no good
                    if (!in_old) { base_short = true; continue; }
                    a.chunks[i].hash = o->chunks[i].hash;
                    if (reuse) {
                        a.chunks[i].off = o->chunks[i].off;
                        reused += len;
                    } else {
                        copy_from[ai][i] = o->chunks[i].off;
                        a.chunks[i].off = cursor;
                        cursor += len;
                    }
                    continue;
                }
                if (reuse && on_base && (i + 1) * chunk <= ref.claim && !(in_old && o->chunks[i].hash == a.chunks[i].hash))
                    ++stats.verify_bad;
                if (reuse && in_old && o->chunks[i].hash == a.chunks[i].hash) {
                    a.chunks[i].off = o->chunks[i].off;
                    reused += len;
                } else {
                    a.chunks[i].off = cursor;
                    cursor += len;
                }
            }
        }
        return cursor;
    };
    uint64_t reused = 0;
    uint64_t start = have ? old.slot.end : kData;
    uint64_t data_end = place(have, start, reused);
    if (base_short) return fail(SessionError::invalid, "the file lacks a chunk this turn left out");
    std::vector<uint8_t> meta = meta_of(plan, arrays);
    if (have) {
        const uint64_t named = total + meta.size();
        const uint64_t garbage = data_end + meta.size() - kData - named;
        if (garbage > std::max<uint64_t>(log.garbage_min, named)) {
            have = false;
            stats.compacted = true;
            start = kData;
            data_end = place(false, start, reused);
            meta = meta_of(plan, arrays);
        }
    }
    if (meta.size() > kMaxMeta) return fail(SessionError::invalid, "the conversation has too many chunks");
    const uint64_t commit_off = (data_end + 7) / 8 * 8;
    const uint64_t file_end = commit_off + meta.size();
    const uint64_t to_write = file_end - start;
    {
        uint64_t avail = 0;
        if (free_bytes(path, avail) && (avail < options.min_free_bytes || to_write > avail - options.min_free_bytes))
            return fail(SessionError::storage, "not enough free disk space (" + std::to_string(to_write >> 20) +
                                               " MiB to write, " + std::to_string(avail >> 20) + " MiB free)");
    }

    // the file written into: the existing one past its commit, or a new temporary one
    Fd fresh;
    std::string temp;
    struct Unlink { const std::string& p; bool on = false; ~Unlink() { if (on) ::unlink(p.c_str()); } } temp_guard{temp};
    int fd = old.f.fd;
    if (!have) {
        const size_t slash = path.find_last_of('/');
        temp = (slash == std::string::npos ? std::string() : path.substr(0, slash + 1)) + "." +
               (slash == std::string::npos ? path : path.substr(slash + 1)) + "." + random_hex() + ".tmp";
        fresh.fd = ::open(temp.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (fresh.fd < 0) return fail_errno("cannot create the file", errno);
        temp_guard.on = true;
        fd = fresh.fd;
        uint8_t h[kData];
        std::memset(h, 0, sizeof h);
        header_bytes(h, id, chunk);
        if (!pwrite_all(fd, h, kData, 0)) return fail_errno("write error", errno);
    } else if (::ftruncate(fd, (off_t) start) != 0) {   // (what a save that died left past the commit)
        return fail_errno("truncate error", errno);
    }
    std::unique_ptr<uint8_t[]> batch(new (std::nothrow) uint8_t[kBatch]);
    if (!batch) return fail(SessionError::memory, "out of memory for the write buffer");
    size_t used = 0;
    uint64_t batch_off = start, done = 0;
    bool first = true;
    auto flush = [&]() -> bool {
        if (used == 0) return true;
        if (options.cancelled && options.cancelled()) { errno = ECANCELED; return false; }
        if (const int e = first ? fault("write") : 0) { errno = e; return false; }
        first = false;
        if (!pwrite_all(fd, batch.get(), used, batch_off)) return false;
        batch_off += used;
        done += used;
        used = 0;
        if (options.progress) options.progress(done, to_write);
        return true;
    };
    for (size_t a = 0; a < refs.size(); ++a)
        for (size_t i = 0; i < arrays[a].chunks.size(); ++i) {
            const uint64_t off = arrays[a].chunks[i].off;
            if (off < start) continue;   // already in the file
            if (copy_from[a][i] != UINT64_MAX) {   // from the old file into the new one
                uint64_t at = copy_from[a][i], n = chunk_len(refs[a].size, chunk, i);
                while (n) {
                    const size_t c = (size_t) std::min<uint64_t>(n, kBatch - used);
                    if (!pread_all(old.f.fd, batch.get() + used, c, at)) return fail_errno("read error", errno);
                    used += c; at += c; n -= c;
                    if (used == kBatch && !flush()) return fail_errno("write error", errno);
                }
                continue;
            }
            const bool ok = refs[a].visit(i * chunk, chunk_len(refs[a].size, chunk, i), [&](const uint8_t* p, size_t n) {
                while (n) {
                    const size_t c = std::min(n, kBatch - used);
                    std::memcpy(batch.get() + used, p, c);
                    used += c; p += c; n -= c;
                    if (used == kBatch && !flush()) return false;
                }
                return true;
            });
            if (!ok) return fail_errno("write error", errno);
        }
    {   // the commit record, on an 8-byte boundary
        const size_t pad = (size_t) (commit_off - data_end);
        if (used + pad > kBatch && !flush()) return fail_errno("write error", errno);
        std::memset(batch.get() + used, 0, pad);
        used += pad;
        if (!flush()) return fail_errno("write error", errno);
        if (!pwrite_all(fd, meta.data(), meta.size(), commit_off)) return fail_errno("write error", errno);
    }
    if (options.phase) options.phase("flush", to_write);
    if (options.cancelled && options.cancelled()) return fail_errno("write stopped", ECANCELED);
    if (const int e = fault("file_flush")) return fail_errno("flush error", e);
    if (options.durable && ::fsync(fd) != 0) return fail_errno("flush error", errno);
    if (const int e = fault("commit")) return fail_errno("commit error", e);
    Slot slot;
    slot.seq = have ? old.slot.seq + 1 : 1;
    slot.off = commit_off;
    slot.len = meta.size();
    slot.hash = session_hash64(meta.data(), meta.size(), kCommitSeed);
    slot.end = file_end;
    slot.named = total + meta.size();
    uint8_t sb[kSlot];
    slot_bytes(sb, slot);
    if (!pwrite_all(fd, sb, kSlot, kHeader + (slot.seq & 1 ? 0 : kSlot))) return fail_errno("commit error", errno);
    if (options.durable && ::fsync(fd) != 0) return fail_errno("commit error", errno);
    if (!have) {
        if (options.phase) options.phase("publish", 0);
        int e = fault("rename");
        if (!e && ::rename(temp.c_str(), path.c_str()) != 0) e = errno;
        if (e) return fail_errno("rename error", e);
        temp_guard.on = false;
        st.published = true;
        if (options.durable) {
            const size_t slash = path.find_last_of('/');
            Fd dir;
            dir.fd = ::open(slash == std::string::npos ? "." : slash == 0 ? "/" : path.substr(0, slash).c_str(),
                            O_RDONLY | O_DIRECTORY | O_CLOEXEC);
            if (dir.fd >= 0 && ::fsync(dir.fd) != 0 && errno == EINVAL) st.dir_flush_unsupported = true;
        }
    }
    stats.file_bytes = file_end;
    stats.written_bytes = to_write;
    stats.reused_bytes = reused;
    stats.garbage_bytes = file_end - kData - slot.named - (commit_off - data_end);   // (less this commit's padding)
    stats.appended = have;
    stats.commit = slot.hash;
    return true;
}
} // namespace

namespace {
struct Dest {
    std::vector<uint8_t>* bytes = nullptr;
    std::vector<int32_t>* ids = nullptr;
    ConversationBuffer* buf = nullptr;
};
// the image's arrays in the table's order (the same walk as refs_of)
std::vector<Dest> dests_of(Skeleton& sk) {
    std::vector<Dest> out;
    auto state = [&](ConversationCheckpoint& c) {
        for (std::vector<uint8_t>* v : {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos}) out.push_back({v, nullptr, nullptr});
    };
    for (auto& im : sk.images) {
        if (!im.live.from_first) out.push_back({nullptr, &im.live.c.ids, nullptr});
        state(im.live.c);
        for (auto& k : im.cks) {
            if (!k.from_first) out.push_back({nullptr, &k.c.ids, nullptr});
            state(k.c);
        }
        for (auto& k : im.s.kv)
            for (ConversationBuffer* b : {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled}) out.push_back({nullptr, nullptr, b});
    }
    return out;
}
bool read_chunk(int fd, const Array& a, const Dest& d, uint64_t chunk, uint64_t i) {
    const uint64_t at = i * chunk, len = chunk_len(a.size, chunk, i), off = a.chunks[(size_t) i].off;
    SessionHasher h(a.key * 0x9E3779B97F4A7C15ull + i);
    if (d.buf != nullptr) {
        if (!d.buf->visit((size_t) at, (size_t) len, [&](uint8_t* p, size_t n, size_t abs) {
                if (!pread_all(fd, p, n, off + (abs - at))) return false;
                h.update(p, n);
                return true;
            })) return false;
    } else {
        uint8_t* p = (d.ids != nullptr ? reinterpret_cast<uint8_t*>(d.ids->data()) : d.bytes->data()) + at;
        if (!pread_all(fd, p, (size_t) len, off)) return false;
        h.update(p, (size_t) len);
    }
    if (h.digest() != a.chunks[(size_t) i].hash) { errno = EILSEQ; return false; }
    return true;
}
// the token ids a later stage shares with the first, and the skeleton's parts into the image
bool assemble(Skeleton& sk, SavedConversation& out) {
    for (size_t i = 0; i < sk.images.size(); ++i) {
        auto& im = sk.images[i];
        if (im.live.from_first) im.live.c.ids = sk.images[0].live.c.ids;
        for (size_t j = 0; j < im.cks.size(); ++j)
            if (im.cks[j].from_first) {
                if (j >= sk.images[0].cks.size()) return false;
                im.cks[j].c.ids = sk.images[0].cks[j].c.ids;
            }
    }
    for (auto& im : sk.images) {
        im.s.live = std::move(im.live.c);
        for (auto& k : im.cks) im.s.checkpoints.push_back(std::move(k.c));
    }
    out = std::move(sk.images[0].s);
    for (size_t i = 1; i < sk.images.size(); ++i) out.stage_images.push_back(std::move(sk.images[i].s));
    return true;
}
} // namespace

bool session_log_read(const std::string& path, const SessionFileIdentity& id, SavedConversation& image, size_t& bytes,
                      std::string& error, const SessionReadLimits& limits, SessionStatus* status,
                      const SessionLogOptions& log) {
    SessionStatus local;
    SessionStatus& st = status ? *status : local;
    st = {};
    auto fail = [&](SessionError kind, const std::string& m) { st.error = kind; error = "session log: " + m; return false; };
    Opened o;
    Skeleton sk;
    SessionError kind = SessionError::invalid;
    if (!open_log(path, id, false, o, error, kind)) { st.error = kind; return false; }
    if (!parse_meta(o.meta, o.chunk, o.slot.off, sk, error)) { st.error = SessionError::invalid; return false; }
    // the file's kind and every bound against the runtime's, before anything is allocated
    if (sk.images.size() == 1 && !limits.stages.empty())
        return fail(SessionError::invalid, "saved on one GPU, but this runtime has a layer split");
    if (sk.images.size() != limits.stages.size() + 1)
        return fail(SessionError::invalid, "saved with " + std::to_string(sk.images.size()) +
                                           " layer-split stages, this runtime has " + std::to_string(limits.stages.size() + 1));
    std::vector<Dest> dests = dests_of(sk);
    uint64_t total = 0;
    {
        size_t at = 0;
        for (size_t i = 0; i < sk.images.size(); ++i) {
            const SessionReadLimits& L = i == 0 ? limits : limits.stages[i - 1];
            const auto& im = sk.images[i];
            if (L.geometry && *L.geometry != im.s.geometry)
                return fail(SessionError::invalid, "saved with another model geometry than this runtime's");
            if (L.layer_range && (L.layer_range->first != im.s.layer_lo || L.layer_range->second != im.s.layer_hi))
                return fail(SessionError::invalid, "saved with another layer range than this runtime's");
            if (im.cks.size() > L.max_checkpoints || im.s.kv.size() > L.max_kv_layers)
                return fail(SessionError::invalid, "more checkpoints or K/V layers than this engine's limit");
            auto ids_ok = [&](const Skeleton::Ck& k) {
                if (k.c.imgs.size() > L.max_tokens) return false;
                if (k.from_first) return true;
                const Array& a = sk.arrays[at++];
                return a.size % 4 == 0 && a.size / 4 <= L.max_tokens;
            };
            auto state_ok = [&] {
                for (size_t p = 0; p < 5; ++p)
                    if (sk.arrays[at++].size > L.max_state_bytes[p]) return false;
                return true;
            };
            bool ok = ids_ok(im.live) && state_ok();
            for (size_t j = 0; ok && j < im.cks.size(); ++j) ok = ids_ok(im.cks[j]) && state_ok();
            if (!ok) return fail(SessionError::invalid, "a state array exceeds this engine's limit");
            for (size_t l = 0; l < im.s.kv.size(); ++l)
                for (size_t p = 0; p < 5; ++p) {
                    const uint64_t n = sk.arrays[at++].size;
                    if (l < L.max_kv_bytes.size() && n > L.max_kv_bytes[l][p])
                        return fail(SessionError::invalid, "a K/V buffer exceeds this engine's limit");
                }
        }
        for (const Array& a : sk.arrays) {
            if (a.size > UINT64_MAX - total) return fail(SessionError::invalid, "size overflow");
            total += a.size;
        }
    }
    if (total + o.meta.size() + kData > session_read_max_file_bytes(limits))
        return fail(SessionError::invalid, "larger than this engine's limit");
    if (limits.admit) {
        std::string why;
        if (!limits.admit(session_read_peak_bytes(total + o.meta.size()), why))
            return fail(SessionError::memory, why.empty() ? "not enough RAM to read it" : why);
    }
    struct Task { uint32_t a; uint32_t i; };
    std::vector<Task> tasks;
    try {
        for (size_t a = 0; a < sk.arrays.size(); ++a) {
            const Array& arr = sk.arrays[a];
            Dest& d = dests[a];
            if (d.buf != nullptr) d.buf->resize_for_overwrite((size_t) arr.size);
            else if (d.ids != nullptr) d.ids->resize((size_t) (arr.size / 4));
            else d.bytes->resize((size_t) arr.size);
            for (size_t i = 0; i < arr.chunks.size(); ++i) tasks.push_back({(uint32_t) a, (uint32_t) i});
        }
    } catch (const std::bad_alloc&) {
        return fail(SessionError::memory, "out of memory for the conversation");
    }
    std::atomic<uint64_t> done{0};
    std::atomic<int> err{0};
    bool stopped = false;
    const int fd = o.f.fd;
    if (!parallel(tasks.size(), log.threads, [&](size_t t) {
            const Task k = tasks[t];
            if (!read_chunk(fd, sk.arrays[k.a], dests[k.a], o.chunk, k.i)) { err = errno ? errno : EIO; return false; }
            done.fetch_add(chunk_len(sk.arrays[k.a].size, o.chunk, k.i), std::memory_order_relaxed);
            return true;
        }, [&] {
            if (limits.progress) limits.progress(done.load(std::memory_order_relaxed), total);
            return false;
        }, stopped)) {
        const int e = err.load();
        return fail(e == EILSEQ ? SessionError::invalid : SessionError::io,
                    e == EILSEQ ? std::string("a chunk's checksum does not match") : std::string("read error: ") + std::strerror(e));
    }
    if (limits.progress) limits.progress(total, total);
    SavedConversation out;
    if (!assemble(sk, out)) return fail(SessionError::invalid, "a stage names a checkpoint the first image lacks");
    image = std::move(out);
    bytes = (size_t) o.size;
    return true;
}

bool session_log_peek(const std::string& path, const SessionFileIdentity& id, SessionPeek& out, std::string& error) {
    Opened o;
    Skeleton sk;
    SessionError kind = SessionError::invalid;
    if (!open_log(path, id, false, o, error, kind) || !parse_meta(o.meta, o.chunk, o.slot.off, sk, error)) return false;
    std::vector<Dest> dests = dests_of(sk);
    // the first image's live ids are the table's first array; its deepest checkpoint's ids by size
    auto read_ids = [&](size_t a) {
        const Array& arr = sk.arrays[a];
        if (arr.size % 4 != 0) return false;
        try { dests[a].ids->resize((size_t) (arr.size / 4)); } catch (const std::bad_alloc&) { return false; }
        for (uint64_t i = 0; i < arr.chunks.size(); ++i)
            if (!read_chunk(o.f.fd, arr, dests[a], o.chunk, i)) return false;
        return true;
    };
    auto fail = [&] { error = "session log: the token ids could not be read"; return false; };
    if (!read_ids(0)) return fail();
    size_t deepest = SIZE_MAX, deepest_a = 0;
    for (size_t j = 0, a = 6; j < sk.images[0].cks.size(); ++j, a += 6)   // (ids + five state arrays each)
        if (deepest == SIZE_MAX || sk.arrays[a].size > sk.arrays[deepest_a].size) { deepest = j; deepest_a = a; }
    if (deepest != SIZE_MAX && !read_ids(deepest_a)) return fail();
    out = {};
    out.live.ids = std::move(sk.images[0].live.c.ids);
    out.live.imgs = std::move(sk.images[0].live.c.imgs);
    if (deepest != SIZE_MAX) {
        out.deepest.ids = std::move(sk.images[0].cks[deepest].c.ids);
        out.deepest.imgs = std::move(sk.images[0].cks[deepest].c.imgs);
    }
    out.cvec = sk.images[0].s.cvec;
    out.bytes = o.size;
    out.version = kVersion;
    return true;
}
#endif

} // namespace strata::core
