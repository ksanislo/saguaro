// The release policy carried out on a real conversation cache and session folder (park_release.hpp): entries on disk
// leave at once, one that is not is written first and stays reachable until its file is there, and only the hard
// floor drops one unsaved.  CPU only.
#include "strata/core/park_release.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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
std::vector<int32_t> tokens(int32_t base, size_t n) {
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = base + int32_t(i);
    return v;
}
SavedConversation conv(const std::vector<int32_t>& ids) {
    SavedConversation c;
    for (size_t i = 0; i < c.geometry.size(); ++i) c.geometry[i] = int64_t(7 + i);
    c.layer_hi = 48;
    c.cvec = false;
    c.live.ids = ids;
    c.live.gdn.assign(4096, uint8_t(ids[0]));
    ConversationKv kv;
    kv.format = 2; kv.cells = int64_t(ids.size()); kv.heads = 1; kv.head_dim = 8; kv.page_size = 4;
    kv.k.resize(ids.size() * 32, uint8_t(ids[0] >> 3));
    c.kv.push_back(std::move(kv));
    return c;
}
bool wait_for(const std::function<bool()>& f, int ms = 5000) {
    for (int i = 0; i < ms / 5; ++i) {
        if (f()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return f();
}
std::string order(const std::vector<park::Released>& r, const std::vector<uint64_t>& serials) {
    std::string s;
    for (const auto& x : r)
        for (size_t i = 0; i < serials.size(); ++i)
            if (serials[i] == x.entry.serial)
                s += std::string(1, char('A' + i)) + (x.action.what == park::Do::drop ? "d" : x.action.what == park::Do::write_drop ? "w" : "u") + " ";
    return s;
}
}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("saguaro-parkrelease-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0x5151, 0x7272};
    std::atomic<bool> stall{false};
    std::atomic<int> stalled{0};
    SessionDir::Options so;
    so.dir = dir.string();
    so.id = id;
    so.min_free_bytes = 0;
    so.write_delay = std::chrono::milliseconds(600000);   // nothing is written unless it is moved up
    so.log_format.chunk_bytes = 4096;
    so.write_block = [&] {
        while (stall.load()) { stalled = 1; std::this_thread::sleep_for(std::chrono::milliseconds(2)); }
    };
    SessionDir sd(so);
    sd.open();

    const auto a = tokens(1000, 3000), b = tokens(20000, 3000), c = tokens(40000, 3000), d = tokens(60000, 3000),
               f = tokens(80000, 3000);
    ConversationCache cache(64u << 20, 8);
    // A: read back from its file earlier; B: parked and written; C: parked, its write waiting out the delay;
    // D: parked, never queued (a conversation that came back from the writer's queue)
    check(cache.put(conv(a), 0, 10.0, true), "A parked");
    check(cache.put(conv(b), 0, 20.0), "B parked");
    sd.write(cache.newest());
    check(sd.expedite(cache.newest().get()) == 1, "B's write moved up");
    check(wait_for([&] { return sd.saved(cache.newest()) == SessionDir::Saved::yes; }), "B is on disk");
    check(cache.put(conv(c), 0, 30.0), "C parked");
    sd.write(cache.newest());
    check(cache.put(conv(d), 0, 40.0), "D parked");
    std::vector<uint64_t> serials;
    for (const auto& i : cache.list()) serials.push_back(i.serial);
    const size_t one = cache.list()[0].bytes;

    auto entries = park::entries_of(cache, &sd, 100.0);
    check(entries.size() == 4 && entries[0].on_disk && entries[1].on_disk && !entries[2].on_disk && !entries[3].on_disk &&
              entries[0].idle_s == 90.0 && entries[3].idle_s == 60.0, "the policy's view: A and B on disk, C and D not");

    park::Limits lim;
    lim.pool_bytes = 100u << 20;
    const park::Memory calm{64ull << 30, 0.0};
    // live demand: the budget is one entry short, then three
    auto acts = park::plan(entries, cache.parked_bytes_live(), cache.parked_bytes_live() - one, 0, calm, lim, true, true);
    auto rel = park::release(cache, &sd, entries, acts);
    check(order(rel, serials) == "Ad " && cache.size() == 3 && sd.files() == 1, "one entry short: the oldest one on disk leaves, nothing is written");

    stall = true;   // the writer stops inside the next write
    entries = park::entries_of(cache, &sd, 100.0);
    acts = park::plan(entries, cache.parked_bytes_live(), one, 0, calm, lim, true, true);
    rel = park::release(cache, &sd, entries, acts);
    check(order(rel, serials) == "Bd Cw " && cache.size() == 1, "two short: B (on disk) leaves, C is released to its write");
    check(wait_for([&] { return stalled.load() == 1; }), "C's write started without the delay");
    check(sd.orphan_bytes() >= one / 2, "its RAM is counted as on its way back while the write runs");
    {   // the request that wants C before its file is there takes it back from the writer, whole
        auto prompt = c; prompt.push_back(9);
        int64_t t = 0;
        bool in_ram = false;
        std::thread unstall([&] { std::this_thread::sleep_for(std::chrono::milliseconds(50)); stall = false; });
        auto got = sd.fetch(prompt, {}, false, 0, t, &in_ram);
        unstall.join();
        check(got && got->live.ids == c && in_ram, "a conversation released to its write is still reachable (taken back from the writer)");
        check(sd.orphan_bytes() == 0, "and no longer pending");
        check(cache.put(std::move(*got), 0, 50.0), "it is parked again");
        serials.push_back(cache.list().back().serial);   // (E: C again)
    }
    // C again (E), not queued this time: released to a write that has to be queued first
    entries = park::entries_of(cache, &sd, 100.0);
    acts = park::plan(entries, cache.parked_bytes_live(), 0, 0, calm, lim, true, true);
    rel = park::release(cache, &sd, entries, acts);
    check(order(rel, serials) == "Dw Ew " && cache.size() == 0, "the pool is full of live K/V: both leave, both are written");
    check(wait_for([&] { return sd.files() == 3 && sd.orphan_bytes() == 0; }), "their files appear and their RAM is back");
    {
        auto prompt = d; prompt.push_back(9);
        int64_t t = 0;
        bool in_ram = true;
        auto got = sd.fetch(prompt, {}, false, 0, t, &in_ram);
        check(got && got->live.ids == d && !in_ram, "a conversation dropped after its write comes back from disk");
    }

    // the hard floor: one that is not on disk is dropped unsaved, a queued write of it stopped
    check(cache.put(conv(f), 0, 60.0), "F parked");
    sd.write(cache.newest());
    serials.push_back(cache.list().back().serial);   // (F)
    const park::Memory dire{1ull << 30, 0.0};
    entries = park::entries_of(cache, &sd, 100.0);
    acts = park::plan(entries, cache.parked_bytes_live(), 64u << 20, 0, dire, lim, true, true);
    rel = park::release(cache, &sd, entries, acts);
    check(order(rel, serials) == "Fu " && cache.size() == 0 && sd.orphan_bytes() == 0, "under the hard floor: dropped unsaved, its write removed");
    sd.flush();
    {
        auto prompt = f; prompt.push_back(9);
        int64_t t = 0;
        check(sd.files() == 3 && !sd.fetch(prompt, {}, false, 0, t), "it is gone: its next turn reads the prompt again");
    }
    // an action for an entry that is kept, or gone, releases nothing
    check(cache.put(conv(a), 0, 70.0, true), "A parked again");
    entries = park::entries_of(cache, &sd, 100.0);
    cache.guard(entries[0].serial);
    rel = park::release(cache, &sd, entries, {{entries[0].serial, park::Do::drop, park::Why::pressure}, {12345, park::Do::drop, park::Why::pressure}});
    check(rel.empty() && cache.size() == 1, "the guarded donor and unknown serials are left alone");
    cache.unguard();
    // without a session folder: a release for the budget drops, pressure above the hard floor keeps unsaved entries
    ConversationCache bare(64u << 20, 8);
    check(bare.put(conv(a), 0, 1.0) && bare.put(conv(b), 0, 2.0), "two parked without a folder");
    entries = park::entries_of(bare, nullptr, 10.0);
    const park::Memory low{9ull << 30, 0.0};
    check(park::release(bare, nullptr, entries, park::plan(entries, bare.parked_bytes_live(), 64u << 20, 0, low, lim, false, true)).empty() &&
              bare.size() == 2, "no folder, pressure above the hard floor: the only copies stay");
    rel = park::release(bare, nullptr, entries, park::plan(entries, bare.parked_bytes_live(), one, 0, calm, lim, false, true));
    check(rel.size() == 1 && bare.size() == 1, "no folder, over the budget: the oldest is dropped, as the cache always did");
    fs::remove_all(dir);
    std::printf("park_release_test: %d checks passed\n", checks);
    return 0;
}
