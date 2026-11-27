// The batch window's row layout (include/strata/core/batch_rows.hpp): plain one-row slots, a slot's contiguous group of
// proposed rows (--batch-mtp), the hand-off capacity of a layer split's stages, and (P124w) the bound of a window's rows
// against each elastic slot session's backed cells.  Header-only, CPU.
#include "strata/core/batch_rows.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
const char* run(const int* rows, const long long* p, int S, int n_slots, int max_t, int hbase, bool handoff) {
    int64_t pos[8];
    for (int t = 0; t < S; ++t) pos[t] = p[t];
    return strata::core::batch_rows_error(rows, S, pos, n_slots, max_t, hbase, handoff, 8);
}
}  // namespace

int main() {
    using strata::core::batch_rows_error;
    // plain batch: one row per slot, any positions
    {
        const int rows[] = {0, 1, 2};
        const long long pos[] = {10, 500, 3};
        check(run(rows, pos, 3, 3, 8, 0, false) == nullptr, "one row per slot");
        check(run(rows, pos, 3, 3, 8, 0, true) == nullptr, "one row per slot through the hand-off");
    }
    // --batch-mtp: a slot's two rows at consecutive positions; also on a layer split (S > slots)
    {
        const int rows[] = {1, 1, 0, 0};
        const long long pos[] = {40, 41, 7, 8};
        check(run(rows, pos, 4, 2, 8, 0, false) == nullptr, "two rows per slot, one stage");
        check(run(rows, pos, 4, 2, 8, 0, true) == nullptr, "two rows per slot across a layer split (S > slots)");
    }
    // eight rows = four slots of two: the whole hand-off
    {
        const int rows[] = {0, 0, 1, 1, 2, 2, 3, 3};
        const long long pos[] = {1, 2, 5, 6, 9, 10, 20, 21};
        check(run(rows, pos, 8, 4, 8, 0, true) == nullptr, "eight rows fill the hand-off");
        check(run(rows, pos, 8, 4, 8, 1, true) != nullptr, "a group past the hand-off's end");
        check(run(rows, pos, 8, 4, 8, 1, false) == nullptr, "no hand-off: the base does not bound the rows");
    }
    // refused layouts
    {
        const int rows[] = {0, 1, 0};
        const long long pos[] = {5, 9, 6};
        check(run(rows, pos, 3, 2, 8, 0, false) != nullptr, "a slot's rows must be contiguous");
    }
    {
        const int rows[] = {0, 0};
        const long long pos[] = {5, 7};
        check(run(rows, pos, 2, 1, 8, 0, false) != nullptr, "a slot's rows need consecutive positions");
    }
    {
        const int rows[] = {0, 2};
        const long long pos[] = {5, 7};
        check(run(rows, pos, 2, 2, 8, 0, false) != nullptr, "a slot past the slots held");
        const int neg[] = {-1};
        const long long p1[] = {1};
        check(run(neg, p1, 1, 2, 8, 0, false) != nullptr, "a negative slot");
    }
    {
        const int rows[] = {0, 0, 0};
        const long long pos[] = {1, 2, 3};
        check(run(rows, pos, 3, 1, 2, 0, false) != nullptr, "more rows than the window holds");
        check(run(rows, pos, 0, 1, 8, 0, false) != nullptr, "an empty window");
        check(run(rows, pos, 3, 1, 8, -1, false) != nullptr, "a negative hand-off base");
        check(run(rows, pos, 3, 1, 8, 0, false) == nullptr, "three rows of one slot are a group");
    }
    // P124w: the backed-cells bound of an elastic --batch slot (SessionState::slot_cells; -1: backed whole)
    using strata::core::first_row_past_backed;
    {
        const int64_t pos[] = {10, 11, 500};
        const int64_t backed[] = {12, 12, 501};
        check(first_row_past_backed(3, pos, backed) == -1, "rows inside their slots' backed cells");
        const int64_t edge[] = {10, 11, 500};
        const int64_t edge_backed[] = {11, 12, 1000};
        check(first_row_past_backed(3, edge, edge_backed) == -1, "the last backed cell is still readable");
        const int64_t past[] = {10, 12, 1000};
        check(first_row_past_backed(3, edge, past) == 0, "a row at the first unbacked cell");
        const int64_t late[] = {10, 11, 500};
        const int64_t late_backed[] = {12, 12, 500};
        check(first_row_past_backed(3, late, late_backed) == 2, "the first offending row, not a later one");
        const int64_t elastic[] = {-1, -1, -1};
        check(first_row_past_backed(3, late, elastic) == -1, "a session backed whole is never past");
        const int64_t zero[] = {0};
        const int64_t none[] = {0};
        check(first_row_past_backed(1, zero, none) == 0, "an unfitted elastic slot holds nothing");
    }
    if (failures != 0) return EXIT_FAILURE;
    std::printf("batch_rows_test: ok\n");
    return EXIT_SUCCESS;
}
