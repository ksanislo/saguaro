// P124y: the VMM grow paths (qsa_follow_grow, qsa_follow_grow_at, the block-score scratch and the rope
// tables) gather fresh physical chunks into a `got` vector, map them into a VmmRange through a take()
// lambda, and then run "release what the map did not take" cleanup loops over `got`. The tree's
// ownership contract (qsa_slot_grow, qsa_slot_trim, VmmRange::release) is that a chunk which enters a
// range keeps its own handle reference inside the range until the range gives the chunk back. A take()
// that leaves the chunk visible in `got` therefore lets those cleanup loops release a handle the range
// still holds - a double cuMemRelease, and the driver may already have recycled the handle value for
// another owner's live allocation (the libamdhip64 crash loop in the 1.2.4-pre reports). This test
// pins strata::core::take_grown_chunk, the fixed protocol: every handle is released exactly once on
// both outcomes, and it pins that the pre-P124y protocol (take without the handover) is what doubled.
#include "strata/core/vmm.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::core::VmmChunk;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
    ++checks;
}

// The grow call modelled with handle accounting: `freed[h]` counts releases of handle value h.
struct Grow {
    std::vector<VmmChunk> got;
    std::vector<int> freed;
    std::vector<VmmChunk> held;   // chunks the VmmRange holds (their handles live in h_)

    explicit Grow(size_t n) : freed(n + 1, 0) { for (size_t i = 0; i < n; ++i) got.push_back(VmmChunk(i + 1)); }

    void free_chunk(VmmChunk h) { if (h != 0) ++freed[h]; }

    // map_range(): consumes takes until one hands out 0; the chunk whose map fails is released by the
    // map itself, earlier takes stay landed (in the range), later ones never taken.
    template <class Take>
    bool map_all(Take take, int fail_at) {
        for (int i = 0;; ++i) {
            const VmmChunk h = take();
            if (h == 0) return fail_at < 0;
            if (i == fail_at) { free_chunk(h); return false; }   // map_range released the failed chunk itself
            held.push_back(h);
        }
    }

    // follow_map_locked(): when the map failed, the fresh chunks it had landed are unmapped and their
    // handles given back (they are gone from the range now).
    void clean_failed_map() { for (const VmmChunk h : held) free_chunk(h); held.clear(); }

    // the grow call's release loop over `got` (qsa_follow_grow, qsa_scores_make, rope_map failure).
    void release_not_taken() { for (const VmmChunk h : got) free_chunk(h); }

    // range reclaim (qsa_follow_trim / VmmRange::release): vmm_chunk_free(unmap(i)) per chunk.
    void trim() { for (const VmmChunk h : held) free_chunk(h); held.clear(); }

    bool each_once() const { for (size_t h = 1; h < freed.size(); ++h) if (freed[h] != 1) return false; return true; }
};

// the pre-P124y protocol: hand out got[i], leave it in place
VmmChunk take_legacy(std::vector<VmmChunk>& got, size_t& i) {
    return i < got.size() ? got[i++] : VmmChunk(0);
}
// the P124y protocol: hand it out and take it out of `got` (the range owns it from here)
VmmChunk take_fixed(std::vector<VmmChunk>& got, size_t& i) { return strata::core::take_grown_chunk(got, i); }
}  // namespace

int main() {
    // --- the fixed protocol: take() hands the chunk over and removes it from got -------------------
    {   // grow succeeded: every chunk landed; the release loop must give back only untaken entries
        Grow g(6);
        size_t i = 0;
        check(g.map_all([&] { return take_fixed(g.got, i); }, -1), "fixed: grow maps everything");
        g.release_not_taken();   // the qsa_follow_grow / qsa_scores_make release loop over got
        g.trim();                // the range later gives its chunks back
        check(g.each_once(), "fixed: grown chunks are released exactly once");
    }
    {   // the map failed on the 3rd chunk: the map released it, the cleanup loop gives the landed
        // fresh chunks back, the grow call releases what was never taken - nobody releases twice
        Grow g(6);
        size_t i = 0;
        check(!g.map_all([&] { return take_fixed(g.got, i); }, 2), "fixed: failed map reported");
        g.clean_failed_map();
        g.release_not_taken();
        g.trim();
        check(g.each_once(), "fixed: failed grow still releases each chunk exactly once");
    }
    {   // exhaustion: further takes hand out 0 (the map's stop condition) and never touch got
        Grow g(2);
        size_t i = 0;
        check(take_fixed(g.got, i) == VmmChunk(1) && take_fixed(g.got, i) == VmmChunk(2), "fixed: takes run in order");
        check(take_fixed(g.got, i) == 0 && take_fixed(g.got, i) == 0, "fixed: exhausted take hands out 0");
        check(g.got[0] == 0 && g.got[1] == 0, "fixed: taken chunks are removed from got");
    }
    // --- the pre-P124y protocol, pinned to show what this fix removed ------------------------------
    {   // same successful grow: the release loop sees the taken chunks again and frees the landed
        // handles before the range has given them back - twice per chunk
        Grow g(6);
        size_t i = 0;
        check(g.map_all([&] { return take_legacy(g.got, i); }, -1), "legacy: grow maps everything");
        g.release_not_taken();
        g.trim();
        bool doubled = false;
        for (size_t h = 1; h < g.freed.size(); ++h) doubled = doubled || g.freed[h] > 1;
        check(doubled, "legacy protocol double-releases a range-held handle (the P124y bug)");
    }
    {   // same failed grow: the cleanup loop and the release loop both free the landed fresh chunks
        Grow g(6);
        size_t i = 0;
        check(!g.map_all([&] { return take_legacy(g.got, i); }, 2), "legacy: failed map reported");
        g.clean_failed_map();
        g.release_not_taken();
        g.trim();
        bool doubled = false;
        for (size_t h = 1; h < g.freed.size(); ++h) doubled = doubled || g.freed[h] > 1;
        check(doubled, "legacy protocol double-releases on the failed grow too");
    }
    std::printf("vmm_grow_ownership_test: %d checks passed\n", checks);
    return 0;
}
