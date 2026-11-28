# REPORT - P125p: the expert-RAM give-back logs only what happened (Saguaro 1.2.5)

## What changed

The card: on Paul's P40, 1.2.4-pre printed on EVERY K/V host-copy growth step "the host copy grows by 0.16 GiB
with 7.27 GiB of RAM available: the expert RAM tier gave 0.00 GiB back (2.89 asked)" - ~20 times a long read -
from a tier that is not a pinned resident complement and has nothing it can give
(`FileExpertSource::relieve_ram` answers 0 up front). The line reported a give-back that never happened and
never could. Now the give-back line is printed only when the tier was really asked, and a tier that cannot
give says the room is short ONCE per load.

- **engines/qfn/include/strata/core/expert_source.hpp** - new ExpertSource query
  `virtual uint64_t ram_relievable_bytes()` (default 0) and the FileExpertSource override.
- **engines/qfn/src/core/expert_source.cpp** - `FileExpertSource::ram_relievable_bytes()`: relieve_ram's early
  ends as a number - 0 for no resident complement / null arena / active exchange storage / a cudaHostAlloc
  arena / the unaligned pageable fallback / Windows, and `complement_bytes_ - complement_pin_limit_` when
  registered (0 when registered WHOLE), `complement_bytes_` otherwise. Scalars only: no offset scan, no lock -
  a racing install/close can at worst make one step ask a tier that then answers 0 (the old behaviour).
- **engines/qfn/include/strata/core/host_relief_choice.hpp** (new) - the decision, pure (no GPU, no RAM query):
  `HostRoomDecision::answer(relievable)` -> {ask, say}: relievable > 0 asks every short step (the real
  give-back lines kept); 0 never asks and `say` is true the FIRST time only, per load.
  `host_room_short_text()` builds the one line: grow, RAM available, reserve shortfall, why ("no resident
  complement, or it is pinned whole"), and the once-promise.
- **engines/qfn/include/strata/core/layer.hpp / src/core/layer.cpp** - the handler pair
  `set_host_relief_query` / `host_relief_query` (0 with no handler); `host_make_room` consults it: cannot-give
  -> one `host_room_short_text` line per load (static latch; it runs under g_host_mu and the engine loads
  once, so per-load = per-process), can-give -> asks and prints the give-back line exactly as before.
- **engines/qfn/src/program/generate.cpp** - the query handler is set beside the relief handler from the same
  `srcp` and cleared in the same `HostReliefScope`.
- **engines/qfn/src/core/host_relief_choice_test.cpp** (new) + **engines/qfn/CMakeLists.txt** - the decision
  test, registered beside kv_host_growth_test (STRATA_BUILD_CONVERSATION_TESTS, header-only, CPU).

serve/ untouched (nothing in it parses the line); serve/qfn_engine.py not touched, extract_qfn_engine.py had
nothing to run. No GPU runs.

## Tests run (one at a time)

- host_relief_choice_test: standalone (g++ -std=c++17 -Wall -Wextra -Iengines/qfn/include) -> 16 checks OK;
  built via ninja in the QFN build dir and run again -> 16 checks OK. Covers: a relievable tier is asked every
  short step and never noted; a 0 tier is never asked and draws ONE short-room line for twenty grow steps; the
  latch says freshly at the first short step after giving ones; the line's wording.

## Build

- `QFN_GGML_DIR=../strata-main/third_party/llama.cpp tools/qfn/build.sh scratch/build-p125p-r1` ->
  QFN BUILD OK 3dc8769e3 (strata + strata_kq + strata_kq_tab, sm_75, AVX floor). The build log shows only
  pre-existing warnings (layer.cpp indentation/unused at 583/3108/3179/3465/3615, expert_source.hpp #611-D,
  generate.cpp unused 'w'); none on the touched lines.

## Commits

1. step 1: the query + host_make_room + generate.cpp wiring + the decision header.
2. step 2: host_relief_choice_test + the CMake entry.
3. this report.

## Open questions

- "One line per request (or per load)": per-load implemented (the engine process loads once; the latch is
  process-wide, as vmm_grow's one-shot notes are elsewhere). A per-request reset would need a serve-level hook
  the plan did not name.
- `ram_relievable_bytes()` reads the backing's scalars without complement_mu_ (documented in expert_source.cpp):
  the worst race is one stale answer for one grow step. If a stricter answer is wanted, take complement_mu_ -
  relief's r3 finding 1 says why the lock is NOT held across queries by preference.
- The note stays silent forever after its first line even if the tier later CHANGES its mind about the reserve
  (a tier that turns relievable mid-load goes back to real give-back lines - the latch never re-notes).
