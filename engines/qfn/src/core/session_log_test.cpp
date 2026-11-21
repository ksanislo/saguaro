// CPU-only tests for the appending session file (include/strata/core/session_log.hpp): round trips, appends that
// write only what changed, rewinds, a save that died half-way, damaged files, limits, compaction, the old format.
#include "strata/core/session_log.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

// positional bytes: byte i depends on i and on the seed in force at i (`seed2` from `turn` on: a rewritten tail)
uint8_t byte_at(size_t i, uint8_t seed) { return uint8_t((i * 2654435761u >> 7) + seed * 31u + (i >> 11)); }
void fill(ConversationBuffer& b, size_t n, uint8_t seed, size_t turn = SIZE_MAX, uint8_t seed2 = 0) {
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = byte_at(at + i, at + i >= turn ? seed2 : seed);
        return true;
    });
}
std::vector<uint8_t> state(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = byte_at(i, seed);
    return v;
}
ConversationCheckpoint head(const std::vector<int32_t>& ids, uint8_t seed) {
    ConversationCheckpoint c;
    c.ids = ids;
    c.gdn = state(9000, seed); c.ple = state(170, seed + 1); c.tails = state(5000, seed + 2);
    c.dead = state(64, seed + 3); c.block_pos = state(64, seed + 4);
    c.used = seed;
    return c;
}
// a conversation of `n` tokens whose tokens and K/V differ from `turn` on (SIZE_MAX: nowhere); 96 K/V bytes a token
SavedConversation conv(size_t n, uint8_t seed, int stages = 0, size_t turn = SIZE_MAX) {
    std::vector<int32_t> ids(n);
    for (size_t i = 0; i < n; ++i) ids[i] = int32_t(1000 + i * 3 + (i >= turn ? 7 : 0));
    auto one = [&](int64_t lo, int64_t hi, uint8_t s) {
        SavedConversation c;
        for (size_t i = 0; i < c.geometry.size(); ++i) c.geometry[i] = int64_t(7 + i);
        c.layer_lo = lo; c.layer_hi = hi;
        c.cvec = false;
        c.live = head(ids, uint8_t(s + n));
        c.live.imgs = {{3, 0xabcdef}};
        c.checkpoints = {head({ids.begin(), ids.begin() + n / 4}, s + 1), head({ids.begin(), ids.begin() + n / 2}, s + 2)};
        for (int l = 0; l < 3; ++l) {
            ConversationKv kv;
            kv.format = 2; kv.cells = int64_t(n); kv.heads = 1; kv.head_dim = 8; kv.page_size = 4; kv.idx_dim = 4;
            const uint8_t ls = uint8_t(s + 10 * l);
            fill(kv.k, n * 48, ls, turn * 48, ls + 100); fill(kv.v, n * 32, ls + 1, turn * 32, ls + 101);
            fill(kv.k_scale, n * 8, ls + 2, turn * 8, ls + 102); fill(kv.v_scale, n * 8, ls + 3, turn * 8, ls + 103);
            fill(kv.pooled, n / 16 * 16, ls + 4, turn / 16 * 16, ls + 104);
            c.kv.push_back(std::move(kv));
        }
        return c;
    };
    SavedConversation c = one(0, stages ? 10 : 48, seed);
    for (int k = 0; k < stages; ++k) c.stage_images.push_back(one(10 + 12 * k, 22 + 12 * k, uint8_t(seed + 40 * (k + 1))));
    return c;
}
bool same_ck(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used;
}
// what a file written with deepest_checkpoint_only holds of `a`, against what was read back
bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !same_ck(a.live, b.live) || b.checkpoints.size() != 1 || !same_ck(a.checkpoints.back(), b.checkpoints[0]) ||
        a.kv.size() != b.kv.size() || a.stage_images.size() != b.stage_images.size()) return false;
    for (size_t l = 0; l < a.kv.size(); ++l) {
        const auto &x = a.kv[l], &y = b.kv[l];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled)) return false;
    }
    for (size_t k = 0; k < a.stage_images.size(); ++k)
        if (!same(a.stage_images[k], b.stage_images[k])) return false;
    return true;
}
std::vector<uint8_t> slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
void spit(const fs::path& p, const std::vector<uint8_t>& b) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write((const char*) b.data(), (std::streamsize) b.size());
}
} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("saguaro-sessionlog-test-" + std::to_string((long) ::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    const SessionFileIdentity id{0x1111, 0x2222};
    SessionWriteOptions wo;
    wo.deepest_checkpoint_only = true;
    SessionLogOptions lo;
    lo.chunk_bytes = 4096;
    lo.garbage_min = 64u << 20;
    SessionLogStats st;
    std::string err;
    size_t bytes = 0;
    auto path = [&](const char* n) { return (dir / n).string(); };
    auto reads_as = [&](const std::string& p, const SavedConversation& want, const SessionReadLimits& lim = {}) {
        SavedConversation got;
        std::string e;
        size_t n = 0;
        const bool ok = session_log_read(p, id, got, n, e, lim, nullptr, lo);
        if (!ok) std::fprintf(stderr, "  read: %s\n", e.c_str());
        return ok && same(want, got);
    };
    check(session_log_supported(), "built here");

    {   // a whole write and its read back; a layer split too
        const SavedConversation a = conv(4000, 1);
        check(session_log_save(path("a"), a, id, st, err, wo, nullptr, lo), "a new file is written");
        check(!st.appended && !st.compacted && st.reused_bytes == 0 && st.garbage_bytes == 0, "written whole");
        check(st.file_bytes == fs::file_size(path("a")) && st.written_bytes + 192 == st.file_bytes, "its size is what was written");
        check(reads_as(path("a"), a), "it reads back as saved");
        SessionReadLimits split;
        split.stages.resize(2);
        const SavedConversation s = conv(3000, 5, 2);
        check(session_log_save(path("s"), s, id, st, err, wo, nullptr, lo), "a layer split is written");
        check(reads_as(path("s"), s, split), "every stage reads back");
        check(!reads_as(path("s"), s) && !reads_as(path("a"), a, split), "a file of another stage count is refused");
        SessionPeek pk;
        check(session_log_peek(path("s"), id, pk, err) && pk.live.ids == s.live.ids && pk.live.imgs == s.live.imgs &&
                  pk.deepest.ids == s.checkpoints.back().ids && pk.version == 3 && !pk.cvec &&
                  pk.bytes == fs::file_size(path("s")), "the index's view: live and deepest ids");
    }
    {   // the next turns append only what is new
        const SavedConversation a = conv(4000, 1), b = conv(4400, 1), c = conv(4500, 1);
        const uint64_t whole = fs::file_size(path("a"));
        check(session_log_save(path("a"), b, id, st, err, wo, nullptr, lo), "the conversation, 400 tokens on, is saved");
        check(st.appended && !st.compacted, "appended to its file");
        // 400 tokens are ~38 KB a layer; the state arrays (2 x ~14 KB) and each array's last chunk come on top
        check(st.written_bytes < whole / 3 && st.reused_bytes > whole / 2, "most of the file is left in place");
        check(reads_as(path("a"), b), "it reads back as the longer conversation");
        const uint64_t after_b = st.file_bytes;
        check(session_log_save(path("a"), c, id, st, err, wo, nullptr, lo) && st.appended && reads_as(path("a"), c),
              "and again");
        check(st.file_bytes > after_b && st.garbage_bytes > 0, "the replaced chunks are the file's unnamed bytes");
        check(session_log_save(path("a"), c, id, st, err, wo, nullptr, lo) && st.appended &&
                  st.written_bytes < 16384 && reads_as(path("a"), c), "an unchanged conversation writes its commit only");
    }
    {   // a rewind: the tail past the divergence is replaced, the start stays
        const SavedConversation b = conv(4400, 2), r = conv(4300, 2, 0, 3900);
        check(session_log_save(path("r"), b, id, st, err, wo, nullptr, lo), "written");
        const uint64_t whole = st.file_bytes;
        check(session_log_save(path("r"), r, id, st, err, wo, nullptr, lo) && st.appended, "the rewound conversation is appended");
        check(st.reused_bytes > whole / 2 && st.written_bytes < whole / 2, "only the rewritten tail is written");
        check(reads_as(path("r"), r) && !reads_as(path("r"), b), "it reads back as the rewound conversation, not the old tail");
        const SavedConversation shorter = conv(2000, 2);
        check(session_log_save(path("r"), shorter, id, st, err, wo, nullptr, lo) && reads_as(path("r"), shorter),
              "a much shorter state of it reads back whole");
    }
    {   // a save that died: the last commit stays readable, and the next save goes on from it
        const SavedConversation a = conv(3000, 3), b = conv(3600, 3), c = conv(3700, 3);
        check(session_log_save(path("t"), a, id, st, err, wo, nullptr, lo), "written");
        const std::vector<uint8_t> good = slurp(path("t"));
        SessionWriteOptions die = wo;
        die.fault = [](const char* step) { return std::string(step) == "commit" ? EIO : 0; };
        check(!session_log_save(path("t"), b, id, st, err, die, nullptr, lo), "a save that stops before its commit fails");
        check(fs::file_size(path("t")) > good.size() && reads_as(path("t"), a), "the torn tail is ignored: the last commit reads");
        die.fault = [](const char* step) { return std::string(step) == "write" ? ENOSPC : 0; };
        SessionStatus ss;
        check(!session_log_save(path("t"), b, id, st, err, die, &ss, lo) && ss.error == SessionError::storage &&
                  reads_as(path("t"), a), "a full disk mid-append: reported, the last commit reads");
        check(session_log_save(path("t"), b, id, st, err, wo, nullptr, lo) && st.appended && reads_as(path("t"), b),
              "the next save appends past the last commit");
        // the file cut in the middle of that append's data: its commit lies past the end
        const std::vector<uint8_t> full = slurp(path("t"));
        std::vector<uint8_t> cut(full.begin(), full.begin() + (std::ptrdiff_t) (good.size() + (full.size() - good.size()) / 2));
        spit(path("t"), cut);
        check(reads_as(path("t"), a), "a file cut inside the last append reads as the commit before it");
        check(session_log_save(path("t"), c, id, st, err, wo, nullptr, lo) && st.appended && reads_as(path("t"), c),
              "and is appended to again");
        // the newest commit pointer damaged (a torn 64-byte write): the other one is used
        std::vector<uint8_t> torn = slurp(path("t"));
        SavedConversation got;
        check(session_log_save(path("t"), conv(3800, 3), id, st, err, wo, nullptr, lo), "one more turn");
        torn = slurp(path("t"));
        const std::vector<uint8_t> newest = torn;
        bool fell_back = false;
        for (size_t slot = 64; slot <= 128 && !fell_back; slot += 64) {
            torn = newest;
            torn[slot + 20] ^= 0x40;
            spit(path("t"), torn);
            fell_back = reads_as(path("t"), c);
        }
        check(fell_back, "a damaged newest commit pointer falls back to the commit before it");
        std::vector<uint8_t> tail = newest;
        tail.insert(tail.end(), 5000, 0x5a);
        spit(path("t"), tail);
        check(reads_as(path("t"), conv(3800, 3)), "bytes after the newest commit do not matter");
        std::vector<uint8_t> flip = newest;
        flip[192 + 70000] ^= 1;
        spit(path("t"), flip);
        std::string e;
        SessionStatus rs;
        check(!session_log_read(path("t"), id, got, bytes, e, {}, &rs, lo) && rs.error == SessionError::invalid &&
                  e.find("checksum") != std::string::npos, "a flipped data byte is refused by its chunk's checksum");
        spit(path("t"), std::vector<uint8_t>(newest.begin(), newest.begin() + 100));
        check(!session_log_read(path("t"), id, got, bytes, e, {}, &rs, lo), "a stub is refused");
        // a damaged file is replaced whole by the next save
        check(session_log_save(path("t"), c, id, st, err, wo, nullptr, lo) && !st.appended && reads_as(path("t"), c),
              "the next save replaces a file it cannot extend");
    }
    {   // a cancelled save leaves the file as it was; a new file is not left behind
        const SavedConversation a = conv(3000, 4), b = conv(5000, 4);
        check(session_log_save(path("c"), a, id, st, err, wo, nullptr, lo), "written");
        SessionWriteOptions stop = wo;
        stop.cancelled = [] { return true; };
        check(!session_log_save(path("c"), b, id, st, err, stop, nullptr, lo) && reads_as(path("c"), a), "cancelled: unchanged");
        check(!session_log_save(path("c2"), b, id, st, err, stop, nullptr, lo) && !fs::exists(path("c2")), "cancelled: no new file");
        size_t left = 0;
        for (const auto& e : fs::directory_iterator(dir)) left += e.path().extension() == ".tmp";
        check(left == 0, "no temporary file is left");
    }
    {   // identity and limits
        SavedConversation got;
        std::string e;
        SessionStatus rs;
        const SavedConversation a = conv(3000, 4);
        check(!session_log_read(path("c"), SessionFileIdentity{0x1111, 0x3333}, got, bytes, e, {}, &rs, lo) &&
                  rs.error == SessionError::invalid, "another configuration's file is refused");
        SessionPeek pk;
        check(!session_log_peek(path("c"), SessionFileIdentity{0x9, 0x2222}, pk, e), "and not indexed");
        SessionReadLimits lim;
        lim.max_tokens = 2999;
        check(!session_log_read(path("c"), id, got, bytes, e, lim, &rs, lo) && rs.error == SessionError::invalid, "over the token limit");
        lim = {};
        lim.geometry = a.geometry;
        lim.layer_range = {0, 48};
        lim.max_kv_bytes = {{3000 * 48, 3000 * 32, 3000 * 8, 3000 * 8, 3000}};
        check(reads_as(path("c"), a, lim), "within the runtime's bounds: read");
        lim.max_kv_bytes[0][0] -= 1;
        check(!session_log_read(path("c"), id, got, bytes, e, lim, &rs, lo), "a K/V buffer over its bound is refused");
        lim = {};
        lim.layer_range = {0, 47};
        check(!session_log_read(path("c"), id, got, bytes, e, lim, &rs, lo), "another layer range is refused");
        lim = {};
        lim.admit = [](uint64_t, std::string& w) { w = "no RAM"; return false; };
        check(!session_log_read(path("c"), id, got, bytes, e, lim, &rs, lo) && rs.error == SessionError::memory, "the RAM preflight refuses");
        // another conversation saved over a file: nothing matches, it is written through and reads back
        const SavedConversation other = conv(2500, 77);
        check(session_log_save(path("c"), other, id, st, err, wo, nullptr, lo) && reads_as(path("c"), other), "another conversation at the same path");
    }
    {   // the unnamed bytes are dropped by a whole rewrite once they pass the limit
        SessionLogOptions small = lo;
        small.garbage_min = 100000;
        check(session_log_save(path("g"), conv(4000, 6), id, st, err, wo, nullptr, small), "written");
        const uint64_t whole = st.file_bytes;
        bool compacted = false;
        uint64_t biggest = 0;
        for (int turn = 0; turn < 12 && !compacted; ++turn) {
            const SavedConversation r = conv(4000, 6, 0, 3600 + 10 * (size_t) turn);   // rewinds: the tails replaced
            check(session_log_save(path("g"), r, id, st, err, wo, nullptr, small) && reads_as(path("g"), r), "a rewind saved");
            compacted = st.compacted;
            if (!compacted) biggest = std::max(biggest, st.file_bytes);
        }
        check(biggest > whole + 200000, "the appends grew the file by what they replaced");
        check(compacted && !st.appended && st.garbage_bytes == 0, "the file was rewritten whole");
        check(st.file_bytes < biggest && st.file_bytes < whole + 4096, "and is its named size again");
    }
    {   // an old-format session file is not a session log, and stays readable as what it is
        const SavedConversation a = conv(1000, 9);
        size_t n = 0;
        check(session_file_write(path("old.session"), a, id, n, err, wo), "a v1 file");
        SavedConversation got;
        std::string e;
        check(!session_log_read(path("old.session"), id, got, bytes, e, {}, nullptr, lo) && e.find("magic") != std::string::npos,
              "is refused as a session log");
        check(session_file_read(path("old.session"), id, got, n, e) && same(a, got), "and read by the v1 reader");
        check(!session_file_read(path("a"), id, got, n, e), "the v1 reader refuses a session log");
    }
    fs::remove_all(dir);
    std::printf("session_log_test: %d checks passed\n", checks);
    return 0;
}
