// P125r: a turn capture past the writer's RAM cap (SAGUARO_TURN_SAVE_SPOOL_MIB) streams its K/V into the turn's
// spool file instead of building a second host copy, and the folder's writer saves it from there in bounded pieces.
// The test runs the whole path over a FAKE HOST COPY: the conversation's K/V as plain host buffers read through
// SessionKvSource - exactly what the capture sees for a streamed session (conversation_snapshot.cpp pools()).
// Parts held in RAM and parts spooled must save, read back, extend by a delta, and freeze at capture time alike.
// CPU only.
#include "strata/core/session_dir.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
uint8_t byte_at(size_t i, uint8_t seed) { return uint8_t((i * 2654435761u >> 9) + seed * 37u); }

constexpr uint64_t kTokK = 64, kTokV = 32;      // K/V bytes per token
constexpr size_t kTokens = 81920;               // 5 MiB of K and 2.5 MiB of V per layer
constexpr uint64_t kKBytes = kTokens * kTokK, kVBytes = kTokens * kTokV;

// The fake host copy: tokens and two K/V layers sitting in host buffers, rewritten from a token on by each request.
struct Host {
    SavedConversation s;
    void turn(size_t resume, size_t n, uint8_t seed) {
        s.live.ids.resize(n);
        for (size_t i = resume; i < n; ++i) s.live.ids[i] = int32_t(100 + i + seed);
        while (s.kv.size() < 2) {
            ConversationKv& k = s.kv.emplace_back();
            k.format = 2; k.cells = int64_t(kTokens); k.heads = 1; k.head_dim = 8; k.page_size = 4;
            k.k.resize_for_overwrite((size_t) kKBytes); k.v.resize_for_overwrite((size_t) kVBytes);
        }
        auto fill = [&](ConversationBuffer& b, uint64_t per, uint8_t sd) {
            b.visit(resume * per, (n - resume) * per, [&](uint8_t* p, size_t c, size_t at) {
                for (size_t i = 0; i < c; ++i) p[i] = byte_at(at + i, sd);
                return true;
            });
        };
        fill(s.kv[0].k, kTokK, uint8_t(seed)); fill(s.kv[1].k, kTokK, uint8_t(seed + 1));   // both layers move together, as a real turn does
        fill(s.kv[0].v, kTokV, uint8_t(seed + 50)); fill(s.kv[1].v, kTokV, uint8_t(seed + 51));
        s.live.gdn.assign(4096, seed); s.live.tails.assign(1024, uint8_t(seed + 1));
        s.checkpoints.assign(1, s.live);
        s.checkpoints[0].ids.resize(n - 2);
        for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(3 + i);
        s.layer_hi = 48;
        s.cvec = false;
    }
    std::vector<SessionKvSource> sources() const {
        std::vector<SessionKvSource> out;
        for (const ConversationKv& k : s.kv) {
            SessionKvSource src;
            src.format = k.format; src.cells = k.cells; src.heads = k.heads; src.head_dim = k.head_dim;
            src.page_size = k.page_size;
            src.sizes = {k.k.size(), k.v.size(), 0, 0, 0};
            const ConversationKv* kp = &k;
            src.read = [kp](size_t part, size_t at, void* dst, size_t n) { return (part == 0 ? kp->k : kp->v).read(dst, at, n); };
            out.push_back(std::move(src));
        }
        return out;
    }
    // as the engine captures a turn; `spool_cap` bytes of RAM budget, spooling into `dir` past it
    std::shared_ptr<SessionLogTurn> capture(const std::string& dir, size_t spool_cap,
                                            const std::vector<std::vector<std::array<uint64_t, 5>>>* keep = nullptr,
                                            bool verify = false) const {
        SavedConversation meta = s;
        meta.kv.clear();
        auto t = std::make_shared<SessionLogTurn>();
        t->chunk_bytes = 4096;
        std::string e;
        check(session_log_capture(*t, std::move(meta), sources(), {}, keep, verify, e, dir, spool_cap), "captured");
        return t;
    }
};
bool same(const SavedConversation& a, const SavedConversation& b) {
    if (!(a.live.ids == b.live.ids && a.live.gdn == b.live.gdn && a.live.tails == b.live.tails && b.checkpoints.size() == 1 &&
          a.checkpoints[0].ids == b.checkpoints[0].ids && b.kv.size() == 2 && a.kv[0].k == b.kv[0].k &&
          a.kv[0].v == b.kv[0].v && a.kv[1].k == b.kv[1].k && a.kv[1].v == b.kv[1].v))
        return false;
    return true;
}
}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("saguaro-spool-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0x1234, 0x5678};
    std::vector<std::string> said;
    std::mutex said_mu;
    auto heard = [&](const char* w) {
        std::lock_guard<std::mutex> g(said_mu);
        for (const auto& m : said) if (m.find(w) != std::string::npos) return true;
        return false;
    };
    std::atomic<bool> stall{false};
    std::atomic<int> stalled{0};
    SessionDir::Options so;
    so.dir = dir.string();
    so.id = id;
    so.min_free_bytes = 0;
    so.write_delay = std::chrono::milliseconds(600000);
    so.log_format.chunk_bytes = 4096;
    so.log = [&](const std::string& m) { std::lock_guard<std::mutex> g(said_mu); said.push_back(m); };
    so.write_block = [&] { while (stall.load()) { stalled = 1; std::this_thread::sleep_for(std::chrono::milliseconds(2)); } };
    auto file_is = [&](const std::string& path, const SavedConversation& want) {
        SavedConversation got;
        size_t n = 0;
        std::string e;
        SessionLogOptions lo;
        return session_log_read(path, id, got, n, e, {}, nullptr, lo) && same(want, got);
    };
    Host live;
    live.turn(0, 2000, 1);
    const SavedConversation whole = live.s;

    // 1. the budget splits the capture: parts up to the cap stay in RAM, the rest spool
    {
        auto t = live.capture(dir.string(), 8u << 20);   // 5 + 2.5 + 5 + 2.5 MiB: held k0 and v0, spooled k1 and v1
        check(t->kv[0][0].bytes[0].size() == kKBytes && !t->kv[0][0].spool[0], "the held parts are RAM as ever");
        check(t->kv[0][0].bytes[1].size() == kVBytes && !t->kv[0][0].spool[1], "the held prefix covers the whole budget's start");
        check(t->kv[0][1].bytes[0].size() == 0 && !!t->kv[0][1].spool[0] && t->kv[0][1].spool_at[0] == 0,
              "a part past the budget spools with no RAM copy");
        check(t->kv[0][1].bytes[1].size() == 0 && !!t->kv[0][1].spool[1] && t->kv[0][1].spool_at[1] == kKBytes,
              "and the next one lands behind it in the same spool");
        check(t->spool && t->spool_bytes == kKBytes + kVBytes && t->spool->cap_bytes == (8u << 20), "the spool counts what it holds");
        // 2. the save runs over held and spooled bytes alike and the file is the captured state
        SessionWriteOptions wo;
        wo.deepest_checkpoint_only = true;
        SessionLogStats ls;
        std::string e;
        check(session_log_save_turn((dir / "spooled.sessionlog").string(), *t, 0, id, ls, e, wo), "the mixed capture saves");
        check(file_is((dir / "spooled.sessionlog").string(), whole), "held and spooled parts read back alike");
        // cap 0: the whole capture in RAM, exactly as before P125r
        auto t0 = live.capture(dir.string(), 0);
        check(!t0->spool && t0->spool_bytes == 0 && t0->kv[0][1].bytes[0].size() == kKBytes, "no budget: the old behaviour");
        // no dir either: same
        auto t1 = live.capture("", 8u << 20);
        check(!t1->spool, "nowhere to spool: RAM");
    }

    // 3. through the folder: a spooled whole capture lands, says so in its line, and a delta continues it
    auto track = std::make_shared<SessionTurnTrack>();
    uint64_t seq = 0;
    auto job = [&](std::shared_ptr<SessionLogTurn> t, const char* label = "main") {
        SessionTurnJob j;
        j.turn = std::move(t); j.track = track; j.seq = ++seq; j.base_seq = seq - 1; j.capture_ms = 1.5; j.label = label;
        return j;
    };
    {
        SessionDir sd(so);
        sd.open();
        stall = true;
        sd.write_turn(job(live.capture(dir.string(), 1u << 20)));   // 1 MiB: every part spools, none is held
        for (int i = 0; i < 3000 && !stalled.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        check(stalled.load() == 1, "the spooled save runs in the background");
        live.turn(1900, 2400, 2);   // the request that arrived mid-save: the host copy it reads from moves under it
        stall = false;
        sd.flush();
        check(sd.files() == 1 && !track->broken.load() && file_is(track->path, whole),
              "the file holds what was spooled at capture, not what the next request wrote meanwhile");
        check(heard("streamed to its folder through a 1 MiB writer buffer"), "the line names the cap and the spooled size");
        check(heard("(whole capture)"), "still one line, still says whole capture");
        // the delta after a spooled whole: the request rewrote from token 1900 on, so that is what may not be kept
        const SavedConversation turn2 = [&] { Host h = live; h.turn(2400, 2600, 3); return h.s; }();
        live.turn(2400, 2600, 3);
        const std::vector<std::vector<std::array<uint64_t, 5>>> keep{{{1900 * kTokK, 1900 * kTokV, 0, 0, 0},
                                                                      {1900 * kTokK, 1900 * kTokV, 0, 0, 0}}};
        auto t2 = live.capture(dir.string(), 1u << 20, &keep);
        check(t2->delta(), "a delta after a spooled whole is a delta");
        sd.write_turn(job(t2));
        sd.flush();
        check(sd.files() == 1 && file_is(track->path, turn2), "the delta lands on the same file and it reads back whole");
    }
    {   // a new folder instance finds the spooled turn's state on disk
        SessionDir sd(so);
        sd.open();
        auto prompt = live.s.live.ids;   // this conversation, continuing
        prompt.push_back(7);
        int64_t t = 0;
        auto got = sd.fetch(prompt, {}, false, 0, t);
        check(got && same(live.s, *got) && t == (int64_t) live.s.live.ids.size(), "after a restart the last (spooled-then-delta) turn resumes from disk");
    }
    {   // nothing of a spool survives its turn: the folder holds only session files
        std::error_code ec;
        bool stray = false;
        for (const auto& de : fs::directory_iterator(dir, ec))
            if (std::string(de.path().filename()).find(".session-spool") != std::string::npos) stray = true;
        check(!stray, "the spool was unlinked when it was opened: the folder only has its session files");
    }
    fs::remove_all(dir);
    std::printf("session_log_spool_test: %d checks passed\n", checks);
    return 0;
}
