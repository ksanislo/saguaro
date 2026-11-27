// The lazy pinned host arena (include/maya/core/granule_arena.hpp): what a byte range costs in pinned granules,
// giving it all back and taking it again, and what a driver refusal leaves behind.  The registration is a recording
// fake - the test is host code, no GPU: the pinning the GLM engine passes is cudaHostRegister (hipHostRegister).
#include "maya/core/granule_arena.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace maya::core;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

struct FakePinner : Pinner {
    std::vector<std::pair<uint64_t, uint64_t>> pins, unpins;   // (offset in the arena, bytes)
    uint8_t* base = nullptr;
    int fail_at = -1;   // refuse this pin (0-based), -1: never
    bool pin(void* at, uint64_t bytes) override {
        if ((int) pins.size() == fail_at) return false;
        pins.push_back({(uint8_t*) at - base, bytes});
        return true;
    }
    void unpin(void* at, uint64_t bytes) override { unpins.push_back({(uint8_t*) at - base, bytes}); }
};

constexpr uint64_t M = (uint64_t) 1 << 20;
}  // namespace

int main() {
    FakePinner p;
    std::string err;

    // open: an address range only - nothing pinned, and no second open of one that is open
    GranuleArena a;
    check(a.empty() && a.base() == nullptr, "a fresh arena is empty");
    check(a.open(512 * M, 128 * M, err) && !a.empty() && a.base() != nullptr, "open reserves the range");
    p.base = a.base();
    check(a.reserved_bytes() == 512 * M && a.granule_bytes() == 128 * M && a.pinned_bytes() == 0,
          "the start is pure address space: 0 pinned");
    check(!a.open(64 * M, 16 * M, err), "a second open is refused, not a second mapping");
    check(a.pinned_bytes() == 0, "the refused open pinned nothing");

    // ensure: one granule per small range, once; a straddle takes both; the tail granule counts what it is
    check(a.ensure(40 * M, 1024, p, err), "a first write pins its granule");
    check(p.pins.size() == 1 && p.pins[0].first == 0 && p.pins[0].second == 128 * M,
          "the pin names exactly one granule");
    check(a.pinned_bytes() == 128 * M && a.pinned_at(0) && a.pinned_at(127 * M), "pinned bytes: one granule");
    check(a.pinned_at(128 * M) == false, "the next granule is not");
    check(a.ensure(100 * M, 20 * M, p, err) && p.pins.size() == 1, "a range inside the pinned granule costs nothing");
    check(a.ensure(120 * M, 16 * M, p, err), "a range over the boundary pins the next granule too");
    check(p.pins.size() == 2 && p.pins[1].first == 128 * M, "and only the granule the tail falls in");
    check(a.pinned_bytes() == 256 * M && a.pinned_at(128 * M), "two granules pinned");
    check(a.ensure(500 * M, 12 * M, p, err), "the arena's tail");
    check(p.pins.size() == 3 && p.pins[2].first == 384 * M && p.pins[2].second == 128 * M,
          "the tail of the last whole granule");
    check(a.ensure(384 * M, 128 * M, p, err) && p.pins.size() == 3, "the last granule is already pinned: no re-pin");

    // the edges: past the end names itself, empty ranges cost nothing
    check(!a.ensure(512 * M - 1024, 2048, p, err) && !err.empty(), "a range past the arena is refused");
    check(a.pinned_bytes() == 384 * M, "the refusal pinned nothing");
    check(a.ensure(700 * M, 1024, p, err) == false, "a start past the arena is refused too");
    check(a.ensure(200 * M, 0, p, err) && p.pins.size() == 3, "an empty range is not a thing to pin");

    // zero-filled, as the one cudaHostAlloc was: fresh granules read zero, and so do re-taken ones
    a.base()[40 * M] = 0xAB;
    check(a.ensure(256 * M, 1024, p, err), "the third untouched granule");
    check(a.base()[256 * M] == 0 && a.base()[256 * M + 4095] == 0, "a newly pinned granule is zero-filled");

    // a driver refusal (the WSL cap of ~1 GB pinned in all): the granules before it stay, the request's is not
    {
        GranuleArena b;
        FakePinner q;
        check(b.open(256 * M, 64 * M, err), "open for the refusal test");
        q.base = b.base();
        q.fail_at = 1;   // the second granule this call asks for
        check(!b.ensure(10 * M, 192 * M, q, err) && !err.empty(), "the refusal fails the ensure");
        check(q.pins.size() == 1 && b.pinned_bytes() == 64 * M, "what pinned before the refusal stays pinned");
        check(b.pinned_at(0) && !b.pinned_at(64 * M), "the refused granule is not pinned, the first one is");
        q.fail_at = -1;
        check(b.ensure(10 * M, 192 * M, q, err) && b.pinned_bytes() == 256 * M,
              "a later ensure takes the refused granules too");
        b.close();
    }

    // release: every pinned granule back in the driver's order, the addresses kept, re-ensure re-pins
    const size_t pins_before = p.pins.size();
    p.unpins.clear();
    a.release(p);
    check(p.unpins.size() == 4, "release unpins every pinned granule");
    check(p.unpins[0].first == 0 && p.unpins[3].first == 384 * M, "over the whole arena, granule by granule");
    check(a.pinned_bytes() == 0 && !a.pinned_at(0) && a.base() != nullptr, "pinned: 0, the addresses stay");
    check(a.ensure(0, 1024, p, err), "the same arena serves the next conversation");
    check(p.pins.size() == pins_before + 1, "the granule is pinned again, once");
    check(a.base()[40 * M] == 0, "and what the last conversation wrote is gone (zero again)");

    // granule sizes: page-rounded at the small end, and the last short granule is only its own bytes
    {
        GranuleArena c;
        FakePinner q;
        check(c.open(99 * 4096, 5000, err) && c.granule_bytes() == 8192, "a granule rounds up to whole pages");
        q.base = c.base();
        check(c.ensure(98 * 4096, 4096, q, err), "a write into the last, half-granule tail");
        check(q.pins.size() == 1 && q.pins[0].second == 4096, "the short tail granule pins its own 4 KB only");
        check(c.pinned_bytes() == 4096, "and its bytes are what the pinned figure says");
        c.release(q);
        check(c.pinned_bytes() == 0 && q.unpins.size() == 1 && q.unpins[0].second == 4096, "release takes it back");
        c.close();
    }
    {
        GranuleArena d;
        std::string e;
        check(d.open(64 * M, 0, e) && d.granule_bytes() == 4096, "granule 0 means the smallest one");
        d.close();
    }

    // close: the mapping goes, and the arena says so
    const uint8_t* was = a.base();
    a.close();
    check(a.empty() && a.base() == nullptr && was != nullptr, "close gives the range back");
    check(!a.ensure(0, 1024, p, err), "an ensure on a closed arena is refused");

    std::printf("granule_arena_test: %d checks passed\n", checks);
    return 0;
}
