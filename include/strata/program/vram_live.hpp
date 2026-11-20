#pragma once
// PLAN-vram-live: the Monitor's map of a GPU follows the cards.  An engine prints its `VRAM dev=...` lines once at the
// start, so everything that moves VRAM at runtime (the elastic slots giving and taking back, VLEND, a conversation's KV
// backed as it grows, the expert pool's tail given to it) stayed invisible and landed in "act".  The serve loop feeds
// this a per-card fingerprint of its own bookkeeping - cheap host reads, no device calls - and a card whose fingerprint
// moved is dirty: `print(card)` re-renders that card's line (its `cudaMemGetInfo` is allowed only there) - one printf
// and fflush a card.  Cadence: a dirty card prints at most once an SAGUARO_VRAM_REPORT_S stretch (default 5 s, 0: the
// start-up lines only), and every card every 60 s even when nothing is marked (the heartbeat catches the unmarked).
// The beat is a detached 1 s thread - outside the per-token path, like the engine's watchdog thread.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace strata::program {

/// SAGUARO_VRAM_REPORT_S (default 5; 0: print only at the start).
inline long long vram_report_s() {
    static const long long s = [] {
        const char* v = std::getenv("SAGUARO_VRAM_REPORT_S");
        if (v == nullptr) return 5LL;
        char* e = nullptr;
        const long long n = std::strtoll(v, &e, 10);
        if (e == v) return 5LL;
        return n > 0 ? n : 0LL;
    }();
    return s;
}

class VramLive {
public:
    /// `fp`: one cheap fingerprint a card (its bookkeeping bytes - never the card's total/free).  A card whose
    /// fingerprint moved since its last print is dirty: `print(i)` renders card i's line and prints it.  Call it from
    /// any point outside the token path (the beat thread below, or a settle in the serve loop); it rate-limits itself.
    void tick(const std::vector<std::string>& fp, const std::function<void(size_t)>& print) {
        const long long rep = vram_report_s();
        if (rep <= 0) return;
        const auto t = std::chrono::steady_clock::now();
        if (fp.size() != last_.size()) {   // first look: what the start-up lines printed is the base, not a change
            last_ = fp;
            dirty_.assign(fp.size(), 0);
            next_ = t + std::chrono::seconds(rep);
            next_all_ = t + std::chrono::seconds(60);
            return;
        }
        for (size_t i = 0; i < fp.size(); ++i)
            if (fp[i] != last_[i]) dirty_[i] = 1;   // (the dirty mark: the card's split moved)
        const bool all = t >= next_all_;            // (the heartbeat: every card, marked or not)
        if (!all && t < next_) return;              // (coalesced: one print round an SAGUARO_VRAM_REPORT_S stretch)
        for (size_t i = 0; i < fp.size(); ++i)
            if (all || dirty_[i]) print(i);
        last_ = fp;
        dirty_.assign(fp.size(), 0);
        next_ = t + std::chrono::seconds(rep);
        if (all) next_all_ = t + std::chrono::seconds(60);
    }

    /// The detached once-a-second beat until the process ends (the engine's watchdog thread's pattern): `fn` runs once
    /// a second, outside every request's path.
    template <class Fn>
    static void beat(Fn fn) {
        std::thread([fn] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                fn();
            }
        }).detach();
    }

private:
    std::vector<std::string> last_;
    std::vector<char> dirty_;
    std::chrono::steady_clock::time_point next_{}, next_all_{};
};

}  // namespace strata::program
