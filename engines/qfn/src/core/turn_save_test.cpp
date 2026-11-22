// Per-turn saves of a conversation that stays where it is (SessionDir::write_turn, session_log_capture): the state
// is captured from its sources, written in the background, and later turns write only what changed.  CPU only.
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
constexpr size_t kPerToken = 64;   // K bytes a token; V half of it
uint8_t byte_at(size_t i, uint8_t seed) { return uint8_t((i * 2654435761u >> 9) + seed * 37u); }

// The memory a live conversation sits in: tokens and one K/V layer, rewritten from a token on by each "request".
struct Live {
    SavedConversation s;
    void turn(size_t resume, size_t n, uint8_t seed) {   // tokens [resume, n) are (re)written
        s.live.ids.resize(n);
        for (size_t i = resume; i < n; ++i) s.live.ids[i] = int32_t(100 + i + seed);
        if (s.kv.empty()) s.kv.emplace_back();
        ConversationKv& k = s.kv[0];
        k.format = 2; k.cells = int64_t(n); k.heads = 1; k.head_dim = 8; k.page_size = 4;
        auto fill = [&](ConversationBuffer& b, size_t per) {
            b.resize(n * per);
            b.visit(resume * per, (n - resume) * per, [&](uint8_t* p, size_t c, size_t at) {
                for (size_t i = 0; i < c; ++i) p[i] = byte_at(at + i, seed);
                return true;
            });
        };
        fill(k.k, kPerToken); fill(k.v, kPerToken / 2);
        s.live.gdn.assign(20000, seed); s.live.tails.assign(3000, uint8_t(seed + 1));
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
            src.format = k.format; src.cells = k.cells; src.heads = k.heads; src.head_dim = k.head_dim; src.page_size = k.page_size;
            src.sizes = {k.k.size(), k.v.size(), 0, 0, 0};
            const ConversationKv* kp = &k;
            src.read = [kp](size_t part, size_t at, void* dst, size_t n) { return (part == 0 ? kp->k : kp->v).read(dst, at, n); };
            out.push_back(std::move(src));
        }
        return out;
    }
    // as the engine captures a turn: `unchanged` tokens are known not rewritten since the last capture
    std::shared_ptr<SessionLogTurn> capture(size_t unchanged, bool verify = false) const {
        SavedConversation meta = s;
        meta.kv.clear();
        const std::vector<std::vector<std::array<uint64_t, 5>>> keep{{{unchanged * kPerToken, unchanged * kPerToken / 2, 0, 0, 0}}};
        auto t = std::make_shared<SessionLogTurn>();
        t->chunk_bytes = 4096;
        std::string e;
        check(session_log_capture(*t, std::move(meta), sources(), {}, &keep, verify, e), "captured");
        return t;
    }
};
bool same(const SavedConversation& a, const SavedConversation& b) {
    return a.live.ids == b.live.ids && a.live.gdn == b.live.gdn && a.live.tails == b.live.tails && b.checkpoints.size() == 1 &&
           a.checkpoints[0].ids == b.checkpoints[0].ids && b.kv.size() == 1 && a.kv[0].k == b.kv[0].k && a.kv[0].v == b.kv[0].v &&
           a.kv[0].cells == b.kv[0].cells;
}
}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("saguaro-turnsave-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0x77, 0x88};
    std::atomic<bool> stall{false};
    std::atomic<int> stalled{0};
    std::vector<std::string> said;
    std::mutex said_mu;
    auto heard = [&](const char* w) {
        std::lock_guard<std::mutex> g(said_mu);
        for (const auto& m : said) if (m.find(w) != std::string::npos) return true;
        return false;
    };
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
    Live live;
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
        // turn 1: a whole capture; the next request rewrites the memory while the save is still running
        live.turn(0, 3000, 1);
        const SavedConversation turn1 = live.s;
        stall = true;
        sd.write_turn(job(live.capture(0)));
        for (int i = 0; i < 2000 && !stalled.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        check(stalled.load() == 1, "the save runs in the background, without the write delay");
        live.turn(2900, 3400, 2);   // (the request that arrived mid-save: reply re-rendered from 2900, then 400 more)
        const SavedConversation turn2 = live.s;
        stall = false;
        sd.flush();
        check(sd.files() == 1 && track->saved_tokens.load() == 3000 && !track->broken.load(), "turn 1 is on disk");
        check(file_is(track->path, turn1), "the file holds the state as captured, not what the next request wrote meanwhile");
        check(heard("turn save (main): 3000 tokens") && heard("(whole capture)"), "said in one line");
        // turn 2: only what changed since the capture (tokens from 2900 on) is captured and hashed
        auto t2 = live.capture(2900);
        check(t2->delta() && t2->bytes() < turn2.bytes() / 2, "a delta holds the tail only");
        sd.write_turn(job(t2));
        sd.flush();
        check(sd.files() == 1 && track->saved_tokens.load() == 3400 && file_is(track->path, turn2), "turn 2 lands on the same file, whole when read back");
        check(heard("turn save (main): 3400 tokens"), "its line");
        // turn 3 is a rewind far back, then a turn whose save is skipped: the chain is broken, the next is whole
        live.turn(1000, 2000, 3);
        sd.write_turn(job(live.capture(1000)));
        sd.flush();
        check(file_is(track->path, live.s), "a rewind inside the conversation: the file is the new state");
        live.turn(1990, 2600, 4);
        ++seq;   // (a capture that never reached the folder)
        sd.write_turn(job(live.capture(1990)));
        sd.flush();
        check(track->broken.load() && heard("skipped") && track->saved_tokens.load() == 2000, "a delta whose base did not land is refused, the file untouched");
        const SavedConversation before = live.s;
        sd.write_turn(job(live.capture(0)));
        sd.flush();
        check(!track->broken.load() && sd.files() == 1 && file_is(track->path, before), "the whole capture after it lands on the same file");
        // SAGUARO_TURN_SAVE_VERIFY: a claim that is wrong shows up
        live.turn(2590, 2700, 5);
        sd.write_turn(job(live.capture(2590, true)));
        sd.flush();
        check(!heard("VERIFY"), "verify: a right claim is silent");
        live.s.kv[0].k.visit(100, 1, [](uint8_t* p, size_t, size_t) { *p ^= 1; return true; });   // a cell the tracking missed
        live.turn(2690, 2800, 6);
        sd.write_turn(job(live.capture(2690, true)));
        sd.flush();
        check(heard("VERIFY") && file_is(track->path, live.s), "verify: a stale claim is reported (and the whole capture fixed the file)");
        // a second conversation in a slot has its own file
        Live other;
        other.turn(0, 1500, 9);
        auto strack = std::make_shared<SessionTurnTrack>();
        SessionTurnJob j;
        j.turn = other.capture(0); j.track = strack; j.seq = 1; j.label = "slot3";
        sd.write_turn(std::move(j));
        sd.flush();
        check(sd.files() == 2 && strack->path != track->path && file_is(strack->path, other.s) && heard("turn save (slot3)"), "a slot's conversation is saved to its own file");
    }
    {   // killed between turns: a new engine finds the last committed turn
        SessionDir sd(so);
        sd.open();
        auto prompt = live.s.live.ids; prompt.push_back(7);
        int64_t t = 0;
        auto got = sd.fetch(prompt, {}, false, 0, t);
        check(sd.files() == 2 && got && same(live.s, *got) && t == (int64_t) live.s.live.ids.size(), "after a restart the last turn is resumed from disk");
    }
    {   // killed mid-save: the commit before it is what is read
        const std::string path = track->path;
        const SavedConversation last = live.s;
        live.turn(2790, 3300, 7);
        SessionWriteOptions die;
        die.deepest_checkpoint_only = true;
        die.fault = [](const char* step) { return std::string(step) == "commit" ? 5 : 0; };
        SessionLogStats st;
        std::string e;
        check(!session_log_save_turn(path, *live.capture(2790), track->commit.load(), id, st, e, die), "a save that dies before its commit");
        check(file_is(path, last), "leaves the last committed turn readable");
        SessionWriteOptions wo;
        wo.deepest_checkpoint_only = true;
        SessionStatus ss;
        check(!session_log_save_turn(path, *live.capture(2790), 0x1234, id, st, e, wo, &ss) && ss.error == SessionError::invalid &&
                  file_is(path, last), "a delta for another commit is refused and changes nothing");
        check(session_log_save_turn(path, *live.capture(2790), track->commit.load(), id, st, e, wo) && file_is(path, live.s) &&
                  st.hashed_bytes < st.reused_bytes, "the same delta on its own commit lands; most of the file was never read or hashed");
    }
    {   // a delta that pushes the file past its limit rewrites it, copying what it did not capture from the old file
        const std::string path = track->path;
        SessionLogOptions small;
        small.garbage_min = 1;
        SessionWriteOptions wo;
        wo.deepest_checkpoint_only = true;
        SessionLogStats st;
        std::string e;
        bool compacted = false;
        uint64_t base = 0;
        {   // (a whole save to know the commit)
            check(session_log_save_turn(path, *live.capture(0), 0, id, st, e, wo, nullptr, small), "whole");
            base = st.commit;
        }
        for (int i = 0; i < 40 && !compacted; ++i) {
            const size_t n = live.s.live.ids.size();
            live.turn(n - 10, n + 50, uint8_t(20 + i));
            auto t = live.capture(n - 10, i % 2 == 1);   // (every other one as a verify claim)
            check(session_log_save_turn(path, *t, base, id, st, e, wo, nullptr, small) && file_is(path, live.s), "a turn lands");
            check(st.verify_bad == 0, "a right claim stays silent, through a rewrite too");
            base = st.commit;
            compacted = st.compacted && t->delta();
        }
        check(compacted && st.garbage_bytes == 0, "a delta rewrote the file whole and it reads back");
    }
    {   // a reply with a thinking continuation (pm-v2): the slot is saved between the continuation's requests.  Its
        // file must hold a checkpoint at or before the reply's start: the client's next turn renders that reply again
        // without its thinking.  The engine now keeps the turn's checkpoint through a fast-lane continuation read and
        // makes none inside the reply (cont=1); before, the file held no checkpoint (the lane dropped the slot's) or
        // one inside the reply (the continuation's own turn cut, at the server's note turn).
        constexpr int32_t kTurn = 7;   // <|im_start|>
        constexpr size_t kReply = 1990, kThink = 2300, kLive = 2400;
        Live c;
        c.turn(0, kLive, 11);
        for (size_t i = 0; i < kLive; ++i) c.s.live.ids[i] = int32_t(i < kReply ? 1000 + i : 50000 + i);
        c.s.live.ids[kReply] = kTurn;
        c.s.live.ids[kThink] = kTurn;   // (the server's note turn inside the reply)
        auto ck = [&](size_t n) { ConversationCheckpoint k = c.s.live; k.ids.resize(n); k.gdn.assign(20000, uint8_t(n)); return k; };
        // the next turn's prompt: the history, the reply without its thinking, a new user turn
        std::vector<int32_t> next(c.s.live.ids.begin(), c.s.live.ids.begin() + kReply + 1);
        for (int i = 0; i < 40; ++i) next.push_back(70000 + i);
        next.push_back(kTurn);
        for (int i = 0; i < 30; ++i) next.push_back(80000 + i);
        auto save_with = [&](const fs::path& d, const ConversationCheckpoint* pick, int64_t& resumed, size_t& files) {
            fs::create_directories(d);
            SessionDir::Options o2 = so;
            o2.dir = d.string();
            {
                SessionDir sd(o2);
                sd.open();
                auto tr = std::make_shared<SessionTurnTrack>();
                for (size_t live_n : {kThink - 50, kThink + 10, kLive}) {   // the continuation's saves, the last at the end
                    Live at = c;
                    at.s.live.ids.resize(live_n);
                    at.s.checkpoints.clear();
                    if (pick) at.s.checkpoints.assign(1, *pick);
                    SessionTurnJob j;
                    j.turn = at.capture(0); j.track = tr; j.seq = ++seq; j.base_seq = j.seq - 1; j.label = "slot2";
                    sd.write_turn(std::move(j));
                    sd.flush();
                }
            }
            SessionDir sd(o2);   // killed and started again
            sd.open();
            resumed = 0;
            auto got = sd.fetch(next, {}, false, 0, resumed);
            if (!got) resumed = 0;
            files = sd.files();
        };
        int64_t resumed = 0;
        size_t files = 0;
        save_with(dir / "none", nullptr, resumed, files);
        check(resumed == 0, "(before: the lane left no checkpoint - the next turn reads it all)");
        const ConversationCheckpoint inside = ck(kThink);
        save_with(dir / "inside", &inside, resumed, files);
        check(resumed == 0, "(before: a checkpoint inside the reply - the same)");
        const ConversationCheckpoint at_reply = ck(kReply);
        save_with(dir / "turn", &at_reply, resumed, files);
        check(files == 1 && resumed == (int64_t) kReply, "the turn's checkpoint resumes the next turn at the reply's start, one file");
        {   // the turn after that resume (a new place's track) lands on the same file
            SessionDir::Options o2 = so;
            o2.dir = (dir / "turn").string();
            SessionDir sd(o2);
            sd.open();
            Live after = c;
            after.s.live.ids = next;
            for (int i = 0; i < 60; ++i) after.s.live.ids.push_back(90000 + i);
            ConversationCheckpoint k = c.s.live;
            k.ids = next;
            k.ids.resize(next.size() - 31);
            after.s.checkpoints.assign(1, k);
            auto tr = std::make_shared<SessionTurnTrack>();
            SessionTurnJob j;
            j.turn = after.capture(0); j.track = tr; j.seq = 1; j.label = "main";
            sd.write_turn(std::move(j));
            sd.flush();
            check(sd.files() == 1, "the resumed conversation's next turn save extends its file (no second file)");
        }
    }
    fs::remove_all(dir);
    std::printf("turn_save_test: %d checks passed\n", checks);
    return 0;
}
