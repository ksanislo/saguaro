// P125p: host_make_room's ask-or-note decision (include/strata/core/host_relief_choice.hpp): the expert RAM tier is
// asked for a grow step's reserve shortfall only when it CAN give (ram_relievable_bytes() nonzero); when it cannot,
// the short room is said ONCE per load - Paul's P40 repeated "the expert RAM tier gave 0.00 GiB back (2.89 asked)"
// ~20 times a long read from a tier that has nothing to give.  Pure arithmetic - no GPU, no RAM query.
#include "strata/core/host_relief_choice.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
constexpr uint64_t GiB = 1073741824ull;
}  // namespace

int main() {
    // a tier that CAN give: every short step asks it, and the give-back line (the caller's) is what is printed -
    // the decision never silences a tier that really is asked, and never "notes" beside a real ask
    {
        HostRoomDecision d;
        const HostRoomDecision::Answer a = d.answer(6ull * GiB);
        check(a.ask && !a.say, "a tier with relievable bytes is asked, not noted");
        const HostRoomDecision::Answer b = d.answer(GiB);
        check(b.ask && !b.say, "the answer keeps asking while bytes stay relievable");
    }

    // a tier that CANNOT give (0: not a resident complement, cudaHostAlloc, or registered whole): never asked,
    // and the short room is said exactly once - the P40 card's ~20 steps per long read draw ONE line
    {
        HostRoomDecision d;
        int said = 0, asked = 0;
        for (int step = 0; step < 20; ++step) {
            const HostRoomDecision::Answer a = d.answer(0);
            if (a.ask) ++asked;
            if (a.say) ++said;
        }
        check(asked == 0, "a tier with nothing to give is never asked");
        check(said == 1, "the short room is said once, not once per grow step");
        check(!d.answer(0).say, "and never again after that one line");
    }

    // the latch is per load: the FIRST answer that finds the tier short says, whatever came before it - a tier that
    // could give and later became registered whole (or a handler that disappeared) notes once, freshly, at that point
    {
        HostRoomDecision d;
        check(d.answer(2ull * GiB).ask, "a giving tier: asked");
        check(!d.answer(2ull * GiB).say, "no note beside a real ask");
        HostRoomDecision::Answer a = d.answer(0);   // the give-way died (registered whole / install changed the backing)
        check(!a.ask && a.say, "the first short step after giving steps notes the short room once");
        check(!d.answer(0).say && !d.answer(0).ask, "and the repeat steps say nothing");
    }

    // the line itself: room short, both sizes in GiB, the reason, and the once-promise - and it ends in a newline
    // like every other saguaro line (host_make_room prints it verbatim)
    {
        const std::string s = host_room_short_text(171798692ull, (uint64_t) (7.27 * GiB), (uint64_t) (2.89 * GiB));
        check(std::strstr(s.c_str(), "saguaro: KV streaming") != nullptr, "the line carries the log prefix");
        check(std::strstr(s.c_str(), "grows by 0.16 GiB") != nullptr, "it names the grow in GiB");
        check(std::strstr(s.c_str(), "7.27 GiB of RAM available") != nullptr, "and the RAM available");
        check(std::strstr(s.c_str(), "2.89 GiB short of the reserve") != nullptr, "and the reserve shortfall");
        check(std::strstr(s.c_str(), "nothing it can give back") != nullptr, "and WHY: the tier cannot give");
        check(std::strstr(s.c_str(), "said once, not once per grow step") != nullptr, "and that it stays one line");
        check(!s.empty() && s.back() == '\n', "the line ends whole");
    }

    std::printf("host_relief_choice_test: %d checks OK\n", checks);
    return 0;
}
