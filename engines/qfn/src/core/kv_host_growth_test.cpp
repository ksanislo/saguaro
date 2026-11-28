// P124e: the sizing of the streamed host K/V's elastic growth (include/strata/core/kv_host_growth.hpp):
// the start one conversation pays at load (~32768 cells whatever --max-context says), the doubling
// steps a growing conversation pays for the positions it uses, and the trim that gives RAM back to
// the expert tier when it is parked.  Pure arithmetic - no GPU, no allocation.
#include "strata/core/kv_host_growth.hpp"
#include "strata/core/va_debug.hpp"   // P124e round 12 (card B1): the SAGUARO_DEBUG_VA line

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
constexpr int64_t PAGE = 4;                          // cells per logical page (the QSA default)
constexpr int64_t UNIT = 32768 / PAGE;               // the step: SAGUARO_KV_HOST_STEP cells
constexpr uint64_t PER = 1056;                       // bytes per page of one array (K codes, say)
constexpr uint64_t PIECE = 2ull << 20;               // kHostPiece
int64_t step_up(int64_t p) { return (p + UNIT - 1) / UNIT * UNIT; }
}  // namespace

int main() {
    // the pages a request needs, and the step it rounds to
    check(kv_host_pages_for(1, PAGE) == 1 && kv_host_pages_for(4, PAGE) == 1 && kv_host_pages_for(5, PAGE) == 2 &&
              kv_host_pages_for(0, PAGE) == 1,
          "pages for `cells`: at least one");
    check(kv_host_step_up(1, UNIT, 262144 / PAGE) == UNIT, "one cell asks for a whole start step");
    check(kv_host_step_up(30000 / PAGE, UNIT, 262144 / PAGE) == step_up(30000 / PAGE),
          "a 30K prompt rounds up to the step");
    check(kv_host_step_up(1048576 / PAGE + 7, UNIT, 1048576 / PAGE) == 1048576 / PAGE,
          "the step never passes the context");

    // the pieces of one array: a whole number of steps, so the copies' split boundaries and the
    // registrations end on the same page
    const int64_t max262 = 262144 / PAGE;
    const int64_t p1 = kv_host_pieces(1, UNIT, max262, PER, PIECE);
    check((uint64_t) p1 * PIECE >= UNIT * PER && (uint64_t) (p1 - 1) * PIECE < UNIT * PER,
          "one step's pages pin a whole number of pieces");
    check(kv_host_pieces(0, UNIT, max262, PER, PIECE) == p1, "zero pages still keep one step");
    check((uint64_t) kv_host_pieces(max262, UNIT, max262, PER, PIECE) >= (max262 * PER + PIECE - 1) / PIECE,
          "a full context pins all of it");

    // a grow at load: the start step, and it costs the SAME at 262144 and at 1048576 (Ken's acceptance:
    // just HAVING 1M set must not hurt)
    const int64_t max1m = 1048576 / PAGE;
    const int64_t a = kv_host_grow_pages(0, 1, max262, UNIT);
    const int64_t b = kv_host_grow_pages(0, 1, max1m, UNIT);
    check(a == UNIT && b == UNIT, "the first grow backs one start step, whatever --max-context");

    // a growing conversation: nothing below what is backed, a doubled step past it, capped at the context
    check(kv_host_grow_pages(UNIT, UNIT, max1m, UNIT) == UNIT, "a grow inside what is backed does nothing");
    check(kv_host_grow_pages(UNIT, UNIT + 1, max1m, UNIT) == 2 * UNIT,
          "crossing the backed end doubles the backing (O(log) registrations)");
    check(kv_host_grow_pages(2 * UNIT, 3 * UNIT + 1, max1m, UNIT) == 4 * UNIT,
          "the doubling wins when it is bigger than the ask");
    int64_t backed = 0, steps = 0;
    while (backed < max1m) { backed = kv_host_grow_pages(backed, backed + 1, max1m, UNIT); ++steps; }
    check(backed == max1m && steps < 20, "1M cells fill in fewer than twenty growth steps");
    check(kv_host_grow_pages(0, max1m, max1m, UNIT) == max1m,
          "a prompt arriving with nothing backed is backed whole, ahead of need");
    check(kv_host_grow_pages(0, 30000 / PAGE, max1m, UNIT) == step_up(30000 / PAGE),
          "a 30K first request backs its whole prompt, rounded to the step");
    check(kv_host_grow_pages(max1m, max1m, max1m, UNIT) == max1m, "a full context grows no further");

    // the trim a parked conversation runs: down to the step it needs, never under the start, never
    // up again; a whole step is what it gives back (no churn the next turn)
    check(kv_host_trim_pages(max1m, 1, UNIT, UNIT) == UNIT, "a park gives back everything but the start");
    check(kv_host_trim_pages(max1m, max1m, UNIT, UNIT) == max1m, "a conversation that grew keeps it all");
    check(kv_host_trim_pages(3 * UNIT, 2 * UNIT + 1, UNIT, UNIT) == 3 * UNIT,
          "half a page short of a step keeps the step (no churn)");
    check(kv_host_trim_pages(3 * UNIT, UNIT, UNIT, UNIT) == UNIT, "back to one step");
    check(kv_host_trim_pages(0, 1, UNIT, UNIT) == 0, "nothing backed stays nothing");

    // the prefix mode's registration size: the pages' bytes, page-aligned
    check(kv_host_prefix_bytes(0, PER, 4096) == 0 && kv_host_prefix_bytes(1, PER, 4096) == 4096,
          "a prefix registers whole pages");
    check(kv_host_prefix_bytes(UNIT, PER, 4096) == (UNIT * PER + 4095) / 4096 * 4096,
          "the start step's prefix bytes");

    // P124e round 12 (manager card B1): the SAGUARO_DEBUG_VA line - the fault address must fall in exactly
    // one logged range, so the format names the op, the device, the range and the owner on one line.
    char buf[256];
    va_debug_format(buf, sizeof buf, "map", 0, 0x7f9ebba00000ull, 2ull << 20, "yarn cos (YaRN-level table)");
    check(std::strstr(buf, "map") != nullptr && std::strstr(buf, "0x7f9ebba00000") != nullptr &&
              std::strstr(buf, "2097152") != nullptr && std::strstr(buf, "yarn cos") != nullptr,
          "the va line names the op, the range [base, +size) and the owner");
    va_debug_format(buf, sizeof buf, "unpin", -1, 0, 4096, nullptr);
    check(std::strstr(buf, "unpin") != nullptr && std::strstr(buf, "dev -1") != nullptr &&
              std::strstr(buf, "-") != nullptr,
          "a host op says dev -1 and a null owner prints -");
    // (the gate itself: env unset here, so va_debug_on() is false and the hooks print nothing - the card's
    //  run sets SAGUARO_DEBUG_VA=1 and that line is the manager's test of the on side)
    check(!va_debug_on(), "SAGUARO_DEBUG_VA off by default: every hook is one static-bool test");

    std::printf("kv_host_growth_test: %d checks OK\n", checks);
    return 0;
}
