# FIX-REPORT: REVIEW-RESULT.md (FIX-FIRST) -> what was changed

Branch `saguaro/q1-s1-qfn-turbo-skip`, after the review's base range. One commit per finding, in the review's
numbering. `serve/qfn_engine.py` was not touched (nothing extracted changed, so `tools/saguaro/extract_qfn_engine.py`
had nothing to regenerate); no other engine, no `engines/qfn/serve/server.py` change (the wire keys already flow).

## 1. FIX-FIRST - skip tier inverted between the request line and the kernels -> fixed, 2519feb13

Translates once, at the boundary, exactly as the finding prescribed:

- New header `engines/qfn/include/strata/program/turbo_skip.hpp` (header-only, no GPU - the `split_rules_test`
  pattern) holds the rule: `wire_of` (`ram` -> 2, `ssd` -> 1, else 0), `bar_of` (wire ram 2 -> bar 1, wire ssd 1 ->
  bar 2 - the kernels' convention of `native_router.hpp` / `session.hpp`), `bar_label` (the log name of a bar), and
  `refusal`/`bar` with the layout gates in the messages' order (re-pick, residency table, ssd layout).
- `turbo_skip_of` (generate.cpp) now returns `bar_of(t.skip)` - the kernel bar reaches `SessionState::turbo_skip`
  and the mapped upload, never the wire value; the ssd refusal is gated on `refusal(...)==3`, i.e. the wire ssd
  (`t.skip == 1`) with `g_turbo_ssd != 2`. So: `turbo_skip=ram` gets bar 1 wherever the VRAM table exists (never
  refused with the ssd message again - plan step 3's "ram needs only the VRAM table" holds); `turbo_skip=ssd` runs
  bar 2 only where the RAM mirror is mirrored, and is said-once-exact on plain mmap and a no-op said-once in the
  arena (plan step 3's messages now fire where they were written to fire).
- The per-request lines print `bar_label` of the executed bar (`turbo_skip_of`'s value), so the line labels the
  behaviour it performed; with a working request the line's tier name round-trips from the request string
  (tested).
- Host-side no-GPU test: `tests/program/turbo_skip_map_test.cpp`, registered in ctest as `turbo_skip_map_test` -
  parse ("ram"/"ssd"/garbage/case), the two inversions the bug swapped, the bar -> label round trip, the full gate
  truth table (ram with every ssd-layout value, ssd only at layout 2, no-table, re-pick, off). Run here: OK.
- `g++ -fsyntax-only` (C++20, CUDA headers) on `generate.cpp` after the change: clean.

## 2. Medium - the "skip is off" per-request message was not removed -> fixed, c7967c42c

The solo request parse still printed "this engine applies only drop; skip is off" for every request carrying a
skip key (the line 96b818b84 and the REPORT claimed gone). Deleted; the now-unused `Turbo::skip_name` went with
it. The legitimate said-once refusals in `turbo_skip_of` (re-pick on / no residency table / ssd without a
RAM-tier fact) stay - they are the only skip messages left.

## 3. Low - REPORT's "no GPU on this box" is false -> fixed, 7d9cb2f54

`REPORT-s1-qfn-turbo-skip.md`'s test section header and bullet corrected: the box has 4 GPUs and CUDA 13.1; not
running the GPU modules was the session's rule, not the box's limit. The section now records what the review ran
on hardware (`turbo_skip_parity`, `native_multi_parity`, `router_top10_parity`, `route_resident_parity` - all OK)
and the new host-side `turbo_skip_map_test`.

## 4. Low - a failing RAM-mirror copy could fail an unrelated non-turbo request -> fixed, 443af911f

`res_sync` (every window's pre-launch hook) no longer errors the request when `ram_upload`'s copy fails, and
`res_upload` reports only the residency table's own verdict (the mirror error was folded into it, so the
`res_dirty` fallback could fail a window for the mirror alone). The said-once note moved inside `ram_upload`
itself (every call point benefits), and the version is now banked only when a copy lands - previously a failed
copy still advanced `ram_ver`, so the mirror would never retry; now it stays stale (the one-round lag the
pipelined paths already accept) and the next upload point retries. Blast radius is inside turbo.

## 5. Low - per-adapt-round host work with turbo entirely off -> fixed, f467a5848

`fold_skip_usage` is gated on `g_turbo_skip_live` - a count of requests currently routing with a skip bar - and
`g_turbo_skip_dirty` - set when one starts, cleared by the fold that drains the table once none is live. The
live count is set idempotently per batch slot inside `slot_turbo` (its on/off calls already bracket every
request: prompt read and the next request's start both pass 0) and by the solo request beside its session
writes, released at its DONE. With turbo off the ~100 KB synchronising D2H (per stage, on the legacy default
stream, behind pipelined windows) and its memsets no longer run at the adapt/async/RAM rounds - the plan's
Don't ("no new work with skip off") restored on the tier path. A lost release can only keep the old behaviour
(fold anyway), never skip a fold while rows route with a bar; counts that land from an in-flight window after
the last fold are the pre-existing finding-6 under-count class, unchanged.

## 6. Info (accepted, documented) - staleness and counter races -> no change

The finding records the design as accepted (notes.md path 3, REPORT risks): table lag under-skip is old
behaviour, over-skip only experts the host already marked gone, counters may under-count by an in-flight
window, HIP's `atomicAdd` on `unsigned long long` is CAS-emulated on gfx906 (performance only). Nothing to
fix; the fixes above do not widen any of these (fix 4 states the mirror's staleness honestly instead of
failing the request; fix 5's gate cannot drop a needed fold).

## Verification after all fixes

- `turbo_skip_map_test` compiled with g++ into ./scratch and run: OK (host-side, no GPU - the plan allows GPU-free
  unit tests).
- `g++ -fsyntax-only -std=c++20` of `generate.cpp` with the CUDA headers: clean after each commit (the one
  `-Wextra` warning at generate.cpp:15481 predates this branch's diff).
- No kernel, `verify.*`, `session.hpp` or generated-file file changed after the review's GPU run of the four
  plan-named modules, so their results stand; nothing here touches the kernels' host replay either.
