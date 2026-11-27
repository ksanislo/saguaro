# FIX-REPORT - P125j layer pause, re-review round 2 (findings of REVIEW-RESULT.md at 3208d412c)

Every finding of the review is fixed; one commit per finding, no re-review finding left open.
The previous round's ten findings were re-verified fixed BY the review itself ("Previous round's
findings - fixes re-verified here at HEAD") and are untouched here.

## MAJOR 1 - the live path dropped an armed pause without the parked path's trust clamp - FIXED (ec0f6bd0b)

`engines/qfn/src/program/generate.cpp`, the go_on block (was 20321-20335).

- A dropped pause with `resume > read_pause.pos` now caps the trust exactly the way the parked
  adopt does (20124-20148): `resume = pause_pos`, `live.resize(pause_pos)`,
  `live_imgs = imgs_below(live_imgs, pause_pos)`, checkpoints deeper than `pause_pos` erased, and
  one stderr line naming the pause's first token and the shared depth - so the paused chunk is
  re-read over ALL its layers before anything deeper is attended over or parked. The trigger the
  review verified reachable with no env vars (image follow-up: `go_on` fails on `req_imgs.empty()`
  at `resume == live.size() == P+T`) now reads from P, and the request can no longer end as a
  normal `live_ok` turn with cells no layer >= the pause ever wrote.
- The whole pause decision MOVED above `mtp.kv_restore(resume)` (it sat below it) so the drafter's
  ring is truncated to the CLAMPED depth - at the old place a clamped `resume` would have left the
  ring restored to the never-written depth. Nothing between the old and new spot reads the pause or
  writes `resume` (only the drafter restore, `tr`, `req_times`, `set_prompt_len`), and the
  checkpoint-mount above it keeps using the `resume` it mounts at; the RESUME line, `read_from`,
  `turn_main.begin`, `limit_reuse` and `pp_next_check` all see the clamped figure as intended.
- The mid-chunk prompt-edit trigger stays unreachable (full-prefix `resume`, checkpoint/borrow
  limits) - the clamp is the missing invariant guard, as the review said, not a new behaviour.

## MAJOR 2 - the card script's leg-sum could not pass - FIXED engine-side (bb0c65a76)

`engines/qfn/src/program/generate.cpp`, the pause keep branch.

- Took the review's first option ("bump `pp_reached = lp_pos + lp_tokens` in the pause keep"):
  the keep now sets `pp_reached` to the paused chunk's end, so the DONE `read` figure
  (`clamp(pp_reached - resume, 0, fresh)` at 23083+, and the engine's
  "prompt N tokens = R reused + K read" line the script sums) counts the paused chunk on the leg
  that paused it - which is what REPORT:166 already claims. The script is unchanged and its sums
  now close by construction: cut leg K = P+T - resume_leg, resumed leg K = (where it stopped) -
  (P+T), so the legs' reads sum to the prompt within READ_TOL, and the last leg's
  `LAST_R + LAST_K == LAST_N` stays exact (a completed leg reads `fresh = n - resume`; the paused
  chunk is in its R, not its K). The M > 0 reused check is unaffected (RESUME prints the
  un-clamped `resume` = P+T).
- Only reachable-after-the-keep uses of `pp_reached` are the #1620 keep (an `else` of the pause
  keep, so a bumped leg never reaches it), the P124b landing (`!live_ok`, false once the keep ran)
  and the cancelled `read` figure - which is the point. A leg whose pause keep is REFUSED (MAJOR 3
  with the head-alias flag, or no prompt_cache) keeps the old figure: its chunk was not kept and
  is genuinely re-read, so counting it as unread stays right.
- The grace-default flip's card proof is intact: `bash -n` clean; not run (no card on this box,
  the plan excludes GPU runs).

## MAJOR 3 - `--prefill-head-alias` desynced the pause record from its rows - FIXED (df890dd14)

`engines/qfn/src/program/generate.cpp`, three spots.

- Both `sp.reset()` sites (the next request's top rebuild ~19512 and the after-read head restore
  ~21720) now call `drop_read_pause()` beside the reset: the record goes with the rows it was
  pointing at, like everywhere else does.
- The pause keep additionally refuses when `sp.pause_rows_bytes() == 0` - the direct guard for the
  reset that runs BEFORE the keep branch (21720): with the rows gone the keep records nothing
  (and its "0 MiB" line can no longer print), the leg degrades to the #1620 keep at `pp_reached` -
  a completed chunk end, fully written - and the existing pair-check at 22966 drops record+rows
  together on the mismatch. With the pair together, the next request either goes on with a real
  pair or reads on from a clamped trust (MAJOR 1), and a park either has rows to take or no record
  to park by: the half-written-session-dir write (11089-11108) cannot be reached this way again.
- Flag off (the default, no repo config and no server code passes it): untouched - with the flag
  off nothing resets `sp` between the run and the keep, and the keep's new condition was always
  implicitly true there (the pause armed rows).

## NIT 4 - `paused_chunk_index()` doc said 0, restored pauses report -1 - FIXED (6a2ea08e4)

`engines/qfn/include/strata/prefill/prefill.hpp`: the doc now says -1 for a restore-handed-back
pause (give_pause never sets `SavedPause::chunk`; its default -1 stands - the paused run's
run-relative label cannot survive the park) and notes that the engine's DONE/keep line carries the
label captured in the run. Chose "say -1" over "store 0": a restored pause genuinely has no
run-relative index, and the label-only reader never sees it. Doc-only.

## NIT 5 - `state_bytes` wrapped `per_row` before its guard - FIXED (dc4dbea8a)

`engines/qfn/include/strata/program/layer_pause.hpp`: clamp `row_floats > UINT64_MAX / 4` first
(such a row's own byte count is >= 2^64: saturate at once), so `per_row` can no longer wrap - and
the guard below can no longer divide by a wrapped zero. `engines/qfn/src/program/layer_pause_test.cpp`:
two checks added - a 2^62-float row saturates, the widest countable row (2^62-1 floats) stays
exact. Built and run here: "ok: 43 checks, 0 bad", `ctest -R layer_pause_test` passes.

## NIT 6 - a resumed chunk above max(LB,1) still gathered and landed ~84 MB of PLE rows - FIXED (fc4e5f5d8)

`engines/qfn/src/prefill/prefill.cpp`: `ple_reads = l_from <= max(LB, 1)` gates BOTH the chunk-top
gather and `ple_pending` (hence the post-loop land), matching where `m.ple_emb`'s only readers sit
(the l == max(LB,1) blocks, which `l_from` skips). A resumed chunk past layer 1 now gathers
nothing and lands nothing. Where the chunk before such a resume had already read its rows ahead,
the skip waits the gather out and clears `ple_next` at the chunk's end, so the next chunk's top
sees a broken pipeline and restarts the gather at its own chunk - the cold-start path the first
chunk already uses - instead of landing the stale rows as its own. (`ple_on` means LB <= 1, so
`ple_reads` is `l_from <= 1`; a continuation resumes at its own c0 = 0, where `ple_next` starts
empty - the consume is the guard for any layout that changes that.) `strata_prefill` builds exit 0.

## NIT 7 - SETTINGS.md contradicted itself on the grace vs the pause switch - FIXED (79cebc30e)

`docs/SETTINGS.md`, both the `SAGUARO_PRIORITY_IDLE_S` row and the Request-priority paragraph:
with `SAGUARO_LAYER_PAUSE=0` the read stops between chunks again and the grace is what keeps a
follow-up's wait short - set it back to 5. Docs only; the 0 default itself ships with the pause,
as the plan's step 3 and the manager's gate require.

## What I ran (plan-named modules, one at a time, this tree)

- `cmake --build scratch/build-p125j-cuda75 --target strata` exit 0 (links generate.cpp +
  prefill.cpp as committed) and `--target strata_prefill` exit 0.
- `cmake --build ... --target layer_pause_test` + `ctest -R layer_pause_test`: Passed, binary
  prints "ok: 43 checks, 0 bad".
- `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_priority`: Ran 68, OK.
- `tools/saguaro/extract_qfn_engine.py`: regenerated `serve/qfn_engine.py` byte-identical
  (`git status` clean after) - no server-side change was needed for any finding.
- `bash -n tools/card_tests/layer_pause.sh`: clean; `py_compile` on serve/server.py,
  serve/qfn_engine.py, engines/qfn/serve/server.py, serve/test_priority.py: OK.
- Not run: `tools/card_tests/layer_pause.sh` itself - no card on this box and the plan excludes
  GPU runs; with MAJOR 2 fixed engine-side its checks are the ones that must hold on the card.
