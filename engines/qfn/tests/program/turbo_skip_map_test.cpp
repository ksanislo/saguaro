// The Turbo skip's request-line -> kernel-bar mapping (REVIEW-RESULT finding 1: the two conventions are swapped
// relative to each other, and the pass-through routed `ssd` on the ram bar).  Header-only rule, no GPU, no model.
#include <cstdio>
#include <string_view>

#include "strata/program/turbo_skip.hpp"

using namespace strata::program;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the request line: "ram" is the wire's 2, "ssd" the wire's 1 (GLM's convention); anything else is off
    CHECK(turbo_skip::wire_of("ram") == 2);
    CHECK(turbo_skip::wire_of("ssd") == 1);
    CHECK(turbo_skip::wire_of("off") == 0);
    CHECK(turbo_skip::wire_of("") == 0);
    CHECK(turbo_skip::wire_of("RAM") == 0);   // exact words only
    // the translation into the kernels' bar (native_router.hpp's TurboRoute::skip): wire ram -> bar 1 (the VRAM
    // table alone), wire ssd -> bar 2 (VRAM + the RAM-tier mirror); the pass-through inverted exactly these two
    CHECK(turbo_skip::bar_of(turbo_skip::wire_of("ram")) == 1);
    CHECK(turbo_skip::bar_of(turbo_skip::wire_of("ssd")) == 2);
    CHECK(turbo_skip::bar_of(0) == 0);
    CHECK(turbo_skip::bar_of(3) == 0);
    // what the bar means on the kernel side: bar 1 = leave out what is not in VRAM = the wire's ram tier
    CHECK(turbo_skip::bar_label(1) == std::string_view("ram"));
    CHECK(turbo_skip::bar_label(2) == std::string_view("ssd"));
    CHECK(turbo_skip::bar_label(0) == std::string_view("off"));
    // a working request's line labels the tier it asked for: label(bar(wire(s))) == s
    for (const char* s : {"ram", "ssd"})
        CHECK(turbo_skip::bar_label(turbo_skip::bar(turbo_skip::wire_of(s), false, true, 2)) == std::string_view(s));

    // the gates (turbo_skip_of's refusals, in its order): a re-pick, then the residency table, then the ssd layout
    // ram (wire 2) needs only the VRAM table: every ssd_layout must let it through, unrefused, on bar 1
    for (const int layout : {0, 1, 2}) {
        CHECK(turbo_skip::refusal(2, false, true, layout) == 0);
        CHECK(turbo_skip::bar(2, false, true, layout) == 1);
    }
    // ssd (wire 1) runs only where the layout mirrors the RAM tier (g_turbo_ssd == 2) - and then on bar 2
    CHECK(turbo_skip::refusal(1, false, true, 2) == 0);
    CHECK(turbo_skip::bar(1, false, true, 2) == 2);
    // plain mmap / page cache (0): said once, runs exact; arena / whole model in RAM (1): a no-op, runs exact
    CHECK(turbo_skip::refusal(1, false, true, 0) == 3);
    CHECK(turbo_skip::bar(1, false, true, 0) == 0);
    CHECK(turbo_skip::refusal(1, false, true, 1) == 3);
    CHECK(turbo_skip::bar(1, false, true, 1) == 0);
    // no residency table: both tiers run exact (the table refusal first, as the message order says)
    CHECK(turbo_skip::refusal(2, false, false, 2) == 2);
    CHECK(turbo_skip::bar(2, false, false, 2) == 0);
    CHECK(turbo_skip::refusal(1, false, false, 0) == 2);
    // SAGUARO_ROUTE_PRIOR / SAGUARO_ROUTE_RESIDENT re-picks would undo the cut: off, said once
    CHECK(turbo_skip::refusal(2, true, true, 2) == 1);
    CHECK(turbo_skip::refusal(1, true, true, 2) == 1);
    CHECK(turbo_skip::bar(2, true, true, 2) == 0);
    // off asks for nothing: never refused, never routed
    CHECK(turbo_skip::refusal(0, true, false, 0) == 0);
    CHECK(turbo_skip::bar(0, false, true, 2) == 0);

    if (failed != 0) {
        std::printf("turbo_skip_map_test: %d failures\n", failed);
        return 1;
    }
    std::printf("turbo_skip_map_test OK\n");
    return 0;
}
