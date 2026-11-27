# REVIEW-RESULT - P125j layer pause (re-review round 3 of d789228f0..HEAD; HEAD = the round-2 fix report)

MERGE

All seven open items of the round-2 review (3 MAJORs + 4 NITs) are fixed and re-verified here from the
sources, each fix's blast radius traced; everything else the brief asks for re-checked and holds.
`git diff 3208d412c..HEAD` shows the fixes touched ONLY the seven finding sites (generate.cpp: the two
head-alias drops, the hoisted pause decision + live clamp, the pause keep's `pause_rows_bytes() > 0` and
`pp_reached` line; prefill.cpp: the PLE skip; the two headers; the test; SETTINGS.md) - the ten previously
verified fixes are byte-identical to what the round-2 review re-verified. Every plan-named test module was
run here and passes. No new finding rises above a note. The card script remains the one thing no box here
can run (no card, the plan excludes GPU runs); with MAJOR 2 fixed engine-side its sums now close by
construction, so it is ready to earn its keep on the card.

## Round 2's findings - fix verified for each (file:line of this tree)

1. MAJOR 1 (live path trusted deeper than a dropped pause) - FIXED (ec0f6bd0b). The dropped-pause branch
   (generate.cpp:20334-20351) now caps the trust at `read_pause.pos` exactly like the parked adopt
   (20136-20149): `resume = pause_pos`, `live.resize`, `live_imgs = imgs_below(...)`, checkpoints deeper
   than the pause erased - so the image follow-up that shares only the live prefix re-reads the paused
   chunk over ALL its layers before anything deeper is attended over, parked or spilled. Verified the
   orderings around the hoist: the whole pause decision now sits above `mtp.kv_restore(resume)`
   (20353-20356), so the drafter's ring restores to the clamped depth; nothing between the old and new
   spot reads the pause or `resume` (the parked-path mirror, the `off_path` erasure at 20195-20203 which
   drops checkpoints deeper than the *clamped* resume too, and `limit_reuse` / `turn_main.begin` /
   `pp_next_check` / the RESUME line at 20360-20380 all see the clamped figure). `resume <= pause_pos`
   needs no clamp - the read from there rewrites the chunk over every layer before going deeper - and the
   clamp cannot fire on the exact continuation (the `go_on` branch is exclusive, 20333).
2. MAJOR 2 (leg-sum could not pass) - FIXED engine-side (bb0c65a76). The pause keep sets
   `pp_reached = lp_pos + lp_tokens` (generate.cpp:22924) before both the cancelled `read` figure
   (`clamp(pp_reached - resume, 0, fresh)`, 23115) and the engine's "prompt N tokens = R reused + K read"
   line the script sums (23354, regex/awk fields verified). The legs then telescope exactly: cut leg
   K = (P+T) - resume, resumed leg R = P+T (the RESUME line prints the un-clamped `resume`) with
   K counted from there, last leg R + K = N exact (a completed leg reads `fresh = n - resume`). Re-checked
   the neighbours: the #1620 keep is the pause keep's `else` and the P124b landing needs `!live_ok`, so
   the bumped figure reaches nothing else; a leg that merely CARRIED the pair (cancel before the chunk)
   has `resume == pp_reached == P+T` and keeps reporting `read = 0` for that leg; a refused keep keeps the
   old figure, which is right - its chunk was not kept and is genuinely re-read.
3. MAJOR 3 (`--prefill-head-alias` desynced the record from its rows) - FIXED (df890dd14). Both
   head-alias `sp.reset()` sites now pair the drop: generate.cpp:19513 and 21746 call `drop_read_pause()`
   (the lambda drops record AND rows, 10453). The pause keep additionally refuses on
   `sp.pause_rows_bytes() == 0` (22907), so the after-read reset (21746, which runs BEFORE the keep)
   degrades the leg to the #1620 keep at `pp_reached` - a completed chunk end - and the mismatch
   pair-check (22998-22999) plus the rowless-clear (23000) take the leftovers down together. With that,
   an armed record always owns rows (armed only at the keep 22924-22928 or a `give_pause` that returned
   true, 20128-20135; cleared with the rows everywhere: take_pause_rows clears `pause_` itself,
   prefill.cpp:1959-1962), so the rowless park - and with it the half-written session-dir write - is
   unreachable (the park takes rows exactly when the record is armed, 11089-11104). Checked the other
   `Prefill::reset()` sites: 9704/9707 are the startup init retry, before any request can arm a pause.
   Default off: the flag's sites are guarded by `o.prefill_head_alias`, nothing changes without it.
4. NIT 4 (doc said 0, restored pauses report -1) - FIXED (6a2ea08e4): prefill.hpp:359-363 now says -1 for
   a restore-handed-back pause and says why (`give_pause` never sets `SavedPause::chunk`,
   prefill.cpp:1966-1974; the label's only reader carries the in-run capture). Doc-only.
5. NIT 5 (`state_bytes` wrapped `per_row` before its guard) - FIXED (dc4dbea8a):
   layer_pause.hpp:69-71 clamps `row_floats > UINT64_MAX/4` first, so `per_row` can never wrap and the
   guard below never divides by a wrapped zero. Two new checks (a 2^62-float row saturates; the widest
   countable row stays exact) at layer_pause_test.cpp:141-144; 43 checks, 0 bad.
6. NIT 6 (resumed chunk above max(LB,1) gathered + landed ~84 MB of PLE rows nobody read) - FIXED
   (fc4e5f5d8): `ple_reads = l_from <= max(LB, 1)` (prefill.cpp:3308) gates the chunk-top gather (3309)
   and `ple_pending` (3314, hence both lands), matching the only readers of `m.ple_emb` - the
   `l == max(LB,1)` land at 3617 and the `l == 1` blocks at 3622/3663, which `l_from` skips. A stale
   ahead-read gather is waited out at the chunk's end and clears `ple_next` (5549-5554), so the NEXT
   chunk's top restarts the pipeline at its own gather instead of mistaking the stale one for its own -
   the first chunk's cold-start shape, and a continuation always resumes at its c0 = 0 where `ple_next`
   starts fresh. Buffer bookkeeping verified: the skip never flips `ple_buf`, the wait completes the old
   gather before the same buffer is refilled, and the `ple_copied` event pairs stay correct. NON-RESUME
   runs are untouched: `l_from == LB` makes `ple_reads` true, so all three sites reduce to the old
   expressions term for term.
7. NIT 7 (SETTINGS.md contradicted the pause switch) - FIXED (79cebc30e): both the
   `SAGUARO_PRIORITY_IDLE_S` row (docs/SETTINGS.md:238) and the Request-priority paragraph (250-253) now
   say that with `SAGUARO_LAYER_PAUSE=0` the read stops between CHUNKS again and the grace is what keeps
   a follow-up's wait short - set it back to 5.

## Re-checked per the brief (my own pass, no regression found)

- Feature OFF: `lp_on` is false for every run with SAGUARO_LAYER_PAUSE=0, a job, a batch, a split, a hand
  or picture rows (prefill.cpp:2978-2979), and every new block needs `lp_on`, `lp_resume` (never true
  without it) or an armed `pause_` (never armed): the embedding wrapper (3227/3300), `gr_broadcast`
  (3301), the loop start `l_from = LB` (3611), the pause block (5516-5546), the PLE trio (3308-3314,
  5549), run()'s read-past clear (6204) - all no-ops; the generate.cpp capture (20667), keep branch
  (22907), clamp (20323) and DONE `%s` likewise; the server pause line without the keys is byte-identical
  (serve/server.py:5877-5887) and `SavedConversation::bytes()` adds only capacities. MAJOR 1's move of
  `kv_restore`/`tr`/`set_prompt_len` below the pause decision sits above a dead block with OFF - behaviour
  unchanged. The grace default 0 is the plan's intended change, gated on this work shipping (step 3).
- Locks / threads / stdout: no new mutex anywhere in the diff (checked every `+` line); the pause reads
  the stdin thread's atomic `stop_req` from the run thread (generate.cpp:12690-12694, the same flag as
  `should_stop`); one `fprintf(stderr)+fflush` per pause (prefill.cpp:5537-5542) and the keep line
  (22932-22936), the DONE keys ride inside the single existing DONE printf (23139-23147) - no line
  interleaving; the server takes the keys only from the fresh DONE after `gen.close()`
  (serve/server.py:5819-5822, 5882) and `carry_leg`/`add_carry` never touch `paused_*`.
- Generated file: `tools/saguaro/extract_qfn_engine.py` re-run here produced a byte-identical
  serve/qfn_engine.py (tree clean after); the only hunk in both engines/qfn/serve/server.py:1636-1638 and
  serve/qfn_engine.py:1193-1195 is the DONE-key tuple.
- Tests really exercise the change: `pause_point`/`resumes`/`state_bytes` are the exact functions
  prefill.cpp:5517-5518/3230/5541 calls, and the 43 hand-computed checks walk the pause/resume/size rules
  the engine relies on; IdleGraceDefault asserts the new 0.0 default. The card script checks the plan's
  four promises end to end (port 8092, own dirs, api_key dropped, engine stderr kept in its out folder).
- No secrets and no path outside the repo in the diff; no Co-Authored-By / Claude-Session / Assisted-by
  trailer anywhere in `git log d789228f0..HEAD`.

## What I ran (plan-named modules, one at a time)

- `cmake --build scratch/build-p125j-cuda75 --target layer_pause_test` +
  `ctest -R layer_pause_test`: Passed; binary: "ok: 43 checks, 0 bad".
- `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_priority`: Ran 68, OK.
- `cmake --build ... --target strata_prefill` / `--target strata`: exit 0 each - the committed tree
  compiles and links.
- `tools/saguaro/extract_qfn_engine.py`: byte-identical regeneration; `git status` clean.
- `bash -n tools/card_tests/layer_pause.sh` clean; `py_compile` on serve/server.py, serve/qfn_engine.py,
  engines/qfn/serve/server.py, serve/test_priority.py: OK.
- Not run: the card script itself - no card on this box and the plan excludes GPU runs.

## Notes (no action)

- When the MAJOR-1 clamp fires, the INFO `ctx_source=`/`ctx_reused=` line (generate.cpp:20208, printed
  before the clamp) can disagree with the RESUME line's clamped depth; the value feeds only the
  "new conversation, N tokens to read" log for `source=none` (serve/qfn_engine.py:1026-1029), which that
  request never takes. Cosmetic.
- A restored pause re-kept without a re-capture reports `paused_chunk=-1` on the DONE (the run-relative
  label cannot survive a park - NIT 4's doc); the server line and the card regex tolerate it
  (`inside chunk [0-9-]+ at layer [0-9]+`).

## Manager's four

- Resume at the exact layer: implemented (`l_from` = the saved layer, prefill.cpp:3227-3239, 3611) and
  helper-tested; the silent-wrong-read holes (image follow-up, head-alias) are closed, so the resume
  either goes on at the saved layer or re-reads the chunk over all its layers. The card proof is the
  script's, for the card.
- Nothing read twice: the reused figure covers the paused chunk, the resumed read restarts at its first
  token, rows read past a pause are cleared (prefill.cpp:6204), and the engine's own leg figures now sum
  to the prompt exactly (MAJOR 2 above) - `layer_pause.sh` demonstrates it rather than failing on it.
- Parks with the conversation: record + rows ride park (generate.cpp:11089-11104) and restore
  (20125-20151), are counted in the budget (10978, `bytes()`), refuse disk writes while live (10738,
  10845, 11108), and the record can no longer outlive its rows (MAJOR 3) - evicting a parked pause drops
  it and the prompt re-reads, as before P125j.
- Idle grace -> 0 only with this: 0.0 ships with the working pause, tested (test_priority 68) and
  documented, including the switch-off combination that needs it back at 5 (NIT 7).
