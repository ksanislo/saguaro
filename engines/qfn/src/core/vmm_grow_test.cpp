// P124e (round 6, manager card): the ORDER of a device-state grow (include/strata/core/vmm_grow.hpp) - the chunks
// are taken and HELD before the mapping is tried, the first refusal asks the card's relief (the expert cache
// giving way) once for the WHOLE remaining shortfall, and the map never starts without the room; and the WORDING
// - the failure line names the driver's error, the bytes asked, the card's free memory and what the cache gave,
// never "no error" and never a RAM message for a VRAM failure.  A fake allocator - no GPU, no allocation.
#include "strata/core/vmm_grow.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

constexpr uint64_t G = 2ull << 20;   // the VMM chunk (2 MiB)

// The card: `free_` chunks the driver will hand out, plus what the relief (the expert cache) can free.
// Every call is recorded, so the test can read the ORDER the grow ran in.
struct Fake {
    int64_t free_ = 0, relief_ = 0;
    bool may_relief = true;
    std::vector<std::string> events;
    GrowChunk make() {
        if (free_ > 0) { --free_; events.push_back("make"); return (GrowChunk) (0x1000ull + events.size()); }
        return 0;
    }
    uint64_t give_way(uint64_t bytes) {   // the relief hands over whole chunks of what its cache can free
        events.push_back("give:" + std::to_string(bytes));
        const uint64_t gave = std::min<uint64_t>((uint64_t) relief_ * G, bytes / G * G + (bytes % G != 0 ? G : 0));
        relief_ -= (int64_t) (gave / G);
        free_ += (int64_t) (gave / G);
        return gave;
    }
    void map(const std::vector<GrowChunk>& got) { events.push_back("map:" + std::to_string(got.size())); }
    bool ran(const std::string& e) const {
        for (const std::string& x : events) if (x == e) return true;
        return false;
    }
    int count(const std::string& pfx) const {
        int n = 0;
        for (const std::string& x : events) if (x.rfind(pfx, 0) == 0) ++n;
        return n;
    }
    int index(const std::string& pfx) const {   // first event starting with pfx, -1: none
        for (size_t i = 0; i < events.size(); ++i)
            if (events[i].rfind(pfx, 0) == 0) return (int) i;
        return -1;
    }
};

// The grow as layer.cpp runs it: room first, the map only if the room arrived (what it never took is freed).
bool grow(Fake& f, int64_t need, std::vector<GrowChunk>& got, VmmRoomStat& st) {
    const bool ok = vmm_grow_room(need, G, [&] { return f.make(); }, [&](uint64_t b) { return f.give_way(b); },
                                  f.may_relief, got, st);
    if (ok) f.map(got);
    return ok;
}
}  // namespace

int main() {
    // THE ORDER: the driver is short, so the cache gives way - and it gives way BEFORE the map is tried.
    {
        Fake f;
        f.free_ = 3;
        f.relief_ = 2;
        std::vector<GrowChunk> got;
        VmmRoomStat st;
        check(grow(f, 5, got, st), "the shortfall the cache covers makes the grow fit");
        check(got.size() == 5 && st.made == 5, "every chunk is held when the map starts");
        check(f.index("give") < f.index("map"), "the cache gives way BEFORE the map is tried");
        check(f.count("give") == 1, "the relief is asked ONCE, not once per chunk");
        check(f.ran("give:" + std::to_string(2 * G)), "it is asked for the WHOLE shortfall (need - made)");
        check(st.relief_ask == 2 * G && st.relief_got == 2 * G, "the room line's ask and give");
    }
    // a card that has it all: the relief is never consulted (the experts keep their room until context needs it).
    {
        Fake f;
        f.free_ = 5;
        std::vector<GrowChunk> got;
        VmmRoomStat st;
        check(grow(f, 5, got, st) && got.size() == 5, "a card with the room maps without asking");
        check(f.count("give") == 0 && st.relief_ask == 0, "unused context costs the experts nothing");
        check(f.count("make") == 5, "the chunks come one by one, from the driver");
    }
    // TRULY no room (the cache gave what it could): the grow fails BEFORE the map - no half-mapped state.
    {
        Fake f;
        f.free_ = 1;
        f.relief_ = 2;
        std::vector<GrowChunk> got;
        VmmRoomStat st;
        check(!grow(f, 5, got, st), "a conversation that cannot fit fails");
        check(!f.ran("map:5") && f.index("map") < 0, "the map is NEVER tried without the room");
        check(f.count("give") == 1 && st.relief_ask == 4 * G && st.relief_got == 2 * G,
              "asked once for the whole shortfall, got what the cache had");
        check((int64_t) got.size() == st.made, "the chunks made anyway are the caller's to free");
    }
    // may_relief = false (the load's start step: no relief installed yet): a refusal fails without asking.
    {
        Fake f;
        f.free_ = 1;
        f.may_relief = false;
        f.relief_ = 9;
        std::vector<GrowChunk> got;
        VmmRoomStat st;
        check(!grow(f, 3, got, st), "no relief, no room: it fails");
        check(f.count("give") == 0 && st.relief_ask == 0, "the load never asks a relief that is not installed");
    }
    // zero need: nothing taken, nothing asked, the map of nothing would be a no-op anyway.
    {
        Fake f;
        std::vector<GrowChunk> got;
        VmmRoomStat st;
        check(vmm_grow_room(0, G, [&] { return f.make(); }, [&](uint64_t b) { return f.give_way(b); }, true, got, st),
              "a grow that needs nothing needs nothing");
        check(got.empty() && st.made == 0 && st.relief_ask == 0, "and takes nothing");
    }

    // THE WORDING: the failure line says the driver's error, the bytes asked, the free memory and the cache -
    // never "no error" (the old cudaGetLastError() line), never RAM for a VRAM failure.
    {
        const std::string t = vmm_grow_failure_text(
            0, "mapped", "cuMemCreate (a VMM chunk) failed: out of memory (2097152 bytes)", 388, G,
            2ull << 30, 15ull << 30, 776ull << 20, 512ull << 20);
        check(t.find("cuMemCreate") != std::string::npos && t.find("out of memory") != std::string::npos,
              "the line names the driver call and its error code");
        check(t.find("776 MiB asked") != std::string::npos && t.find("388 chunks of 2 MiB") != std::string::npos,
              "the line says which allocation and how much was asked");
        check(t.find("2.00 of 15.00 GiB free on the card") != std::string::npos, "and how much VRAM was free");
        check(t.find("was asked to give way") != std::string::npos && t.find("of 776 MiB and gave 512 MiB") !=
                                                                  std::string::npos,
              "and whether the expert cache was asked to give way and what it gave");
        check(t.find("no error") == std::string::npos, "never 'no error'");
        check(t.find("RAM") == std::string::npos, "never a RAM message for a VRAM failure");
        check(t.find("mapped") != std::string::npos, "it names the step that failed");
    }
    // the driver call that recorded nothing (a stub build): the line says THAT, and still never "no error".
    {
        const std::string t = vmm_grow_failure_text(0, "grown", "", 4, G, 0, 0, 0, 0);
        check(t.find("no error") == std::string::npos, "a bare failure never reads 'no error'");
        check(t.find("the driver call gave no code") != std::string::npos, "it says the call recorded nothing");
        check(t.find("was never asked") != std::string::npos, "and that the cache was never asked");
        check(t.find("RAM") == std::string::npos, "still no RAM wording");
    }

    std::printf("vmm_grow_test: %d checks OK\n", checks);
    return 0;
}
