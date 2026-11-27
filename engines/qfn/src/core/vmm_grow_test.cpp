// P124e (round 6, manager card): the ORDER of a device-state grow (include/strata/core/vmm_grow.hpp) - the chunks
// are taken and HELD before the mapping is tried, the first refusal asks the card's relief (the expert cache
// giving way) once for the WHOLE remaining shortfall, and the map never starts without the room; and the WORDING
// - the failure line names the driver's error, the bytes asked, the card's free memory and what the cache gave,
// never "no error" and never a RAM message for a VRAM failure.  A fake allocator - no GPU, no allocation.
//
// P124e (round 7, manager card): the give-way order of EVERY growth allocation.  The context arrays that grow on
// use take their room through one of the two helpers tested here, and both run the same order:
//   * the follow pools (page table, pooled indexer rows - qsa_follow_grow_at -> follow_room -> vmm_grow_room);
//   * the block-score scratch (qsa_scores_grow -> follow_room -> vmm_grow_room);
//   * the RoPE table past the trained 262144 (qsa_rope_grow -> vmm_grow_room_pooled);
//   * the YaRN RoPE rows (qsa_rope_yarn_fit -> vmm_grow_room_pooled).
// The pooled path first helps itself to the card's slot budget (chunks that cost the experts nothing), and only
// the SHORTFALL runs the take-hold-ask-then-map order - so a grow the budget cannot cover asks the cache before
// it maps, which is exactly what round 7's card found missing at the YaRN switch.  What the slots' own ring
// needs arrives the same way: qsa_slots_grow only remaps chunks the budget holds, and the budget is filled by
// the same give() that serves the relief.
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

// The card: `free_` chunks the driver will hand out, plus what the relief (the expert cache) can free, plus the
// slot budget's `budget_` pooled chunks (the rope grow's first source).  Every call is recorded, so the test can
// read the ORDER the grow ran in.
struct Fake {
    int64_t free_ = 0, relief_ = 0, budget_ = 0;
    bool may_relief = true;
    std::vector<std::string> events;
    GrowChunk make() {
        if (free_ > 0) { --free_; events.push_back("make"); return (GrowChunk) (0x1000ull + events.size()); }
        return 0;
    }
    int64_t budget(int64_t want, std::vector<GrowChunk>& out) {   // qsa_slot_budget_take's shape
        int64_t k = 0;
        for (; k < want && budget_ > 0; ++k) { --budget_; out.push_back((GrowChunk) (0x2000ull + events.size())); }
        events.push_back("budget:" + std::to_string(k));
        return k;
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
// The ROPE grow as layer.cpp runs it (qsa_rope_grow / qsa_rope_yarn_fit): the budget first, the shortfall from
// the driver with the same order, the map only with the room in hand; a refusal hands the budget's chunks back.
bool grow_pooled(Fake& f, int64_t need, std::vector<GrowChunk>& got, VmmRoomStat& st, int64_t& from_pool) {
    const bool ok = vmm_grow_room_pooled(need, G, [&](int64_t w, std::vector<GrowChunk>& out) { return f.budget(w, out); },
                                         [&] { return f.make(); }, [&](uint64_t b) { return f.give_way(b); },
                                         f.may_relief, got, st, &from_pool);
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

    // ROUND 7 - THE POOLED GROW (the RoPE table and the YaRN rows): the budget first, then the driver, and the
    // SHORTFALL asks the cache before the map.  THE BUDGET ALONE: the experts keep their room (round 7's card
    // asked for the order per growth allocation; the follow pools and the scratch run the plain one above).
    {
        Fake f;
        f.budget_ = 5;
        std::vector<GrowChunk> got;
        VmmRoomStat st;
        int64_t from_pool = -1;
        check(grow_pooled(f, 5, got, st, from_pool) && got.size() == 5 && from_pool == 5,
              "the rope grow takes the budget's chunks whole");
        check(f.count("make") == 0 && f.count("give") == 0, "the budget's chunks cost the driver and the cache nothing");
        check(f.ran("budget:5") && f.index("budget") < f.index("map"), "the budget is emptied before anything maps");
    }
    // budget + driver + relief: the cache is asked for the shortfall AFTER the budget, once, before the map.
    {
        Fake f;
        f.budget_ = 2;
        f.free_ = 1;
        f.relief_ = 4;
        std::vector<GrowChunk> got;
        VmmRoomStat st;
        int64_t from_pool = -1;
        check(grow_pooled(f, 5, got, st, from_pool) && got.size() == 5 && from_pool == 2,
              "the budget's 2, the driver's 1, the cache's 2: the rope grow fits");
        check(f.ran("give:" + std::to_string(2 * G)), "the relief is asked for the SHORTFALL after the budget (5-2-1)");
        check(f.count("give") == 1, "and once, for the whole of it");
        check(f.index("budget") < f.index("give") && f.index("give") < f.index("map"),
              "budget, then the cache gives way, BEFORE the map is tried (the YaRN switch's order)");
        check(st.relief_ask == 2 * G && st.relief_got == 2 * G, "the room line's ask and give");
    }
    // TRULY no room: the rope grow fails before the map, and the caller learns how many budget chunks to give back.
    {
        Fake f;
        f.budget_ = 2;
        f.free_ = 1;
        std::vector<GrowChunk> got;
        VmmRoomStat st;
        int64_t from_pool = -1;
        check(!grow_pooled(f, 5, got, st, from_pool), "a YaRN grow the card cannot cover fails");
        check(f.index("map") < 0, "the map is NEVER tried without the room");
        check(from_pool == 2 && (int64_t) got.size() == 3,
              "the budget's chunks sit at the front of got: the caller gives those back to the pool");
        check(f.count("give") == 1 && st.relief_ask == 2 * G && st.relief_got == 0,
              "the cache was asked, gave nothing (it is at its floor), and the grow says so");
    }
    // no relief asked where none may come (may_relief=false, the same rule the load's start step keeps):
    {
        Fake f;
        f.budget_ = 1;
        f.free_ = 1;
        f.may_relief = false;
        f.relief_ = 9;
        std::vector<GrowChunk> got;
        VmmRoomStat st;
        int64_t from_pool = -1;
        check(!grow_pooled(f, 3, got, st, from_pool), "short budget, short card, no relief: the grow fails");
        check(f.count("give") == 0 && st.relief_ask == 0, "the pool path never asks a relief that is not allowed");
    }
    // zero need: the budget is not even reached (nothing to grow, nothing to ask).
    {
        Fake f;
        f.budget_ = 5;
        std::vector<GrowChunk> got;
        VmmRoomStat st;
        int64_t from_pool = -1;
        check(grow_pooled(f, 0, got, st, from_pool) && got.empty() && from_pool == 0,
              "a rope grow that needs nothing touches nothing");
        check(f.count("budget") == 0 && f.count("make") == 0 && f.count("give") == 0,
              "no budget, no driver, no relief for a grow of nothing");
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
