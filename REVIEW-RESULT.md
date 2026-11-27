# REVIEW-RESULT - P125j layer pause (re-review of d789228f0..HEAD after the FIX-REPORT round)

FIX-FIRST

Reviewed the committed diff `d789228f..HEAD` (16 files, +1338/-18) against PLAN-p125j-layer-pause.md,
REPORT-p125j-layer-pause.md and FIX-REPORT.md. Every finding of the previous round is genuinely fixed
(verified below, including a fresh build of the committed tree). Two new MAJORs: the LIVE conversation
path lacks the trust clamp the parked path got, and the promised card test's own figures check cannot
pass against the engine's real stats. All plan-named test modules were run and pass.

## MAJOR

1. Dropping an armed pause on the live path reads on from `resume` over cells no layer >= pause_layer
   ever wrote - the parked path's clamp was not copied to the live path.
   - The keep branch makes the LIVE conversation run THROUGH the paused chunk
     (`live.assign(cur.begin(), cur.begin() + lp_pos + lp_tokens)`, generate.cpp:22892) - for layers
     < pause_layer only. The invariant "every cell below live.size() is written for every layer" held
     before P125j; the pause is the first thing that breaks it, and the parked path knows this: adopting
     a paused image landing inside the paused chunk clamps `incoming_tokens = incoming->pause_pos` and
     caps live/imgs/checkpoints with exactly that reasoning ("the conversation is trusted only as deep
     as every layer wrote", generate.cpp:20124-20148).
   - The LIVE path has no such clamp. go_on (generate.cpp:20321-20333) accepts only the EXACT
     continuation; every rejection is `else drop_read_pause()` (generate.cpp:20333) and the read then
     starts at `read_from = resume` (generate.cpp:20335) - which can be strictly past
     `read_pause.pos`. Reachable sequences, no env vars needed:
     a) image follow-up: low read pauses at chunk P; the same conversation's next request carries an
        image (new last message, prefix [0, P+T) unchanged -> resume == P+T == live.size()); go_on fails
        only on `req_imgs.empty()` (generate.cpp:20323) -> drop -> read_from = P+T. Layers >=
        pause_layer never read [P, P+T) - nothing ever will - and decode attends over their
        never-written (zero or previous-turn) K/V cells: silently wrong answer.
     b) prompt edited inside the paused chunk: resume lands in (P, P+T) -> read_from = resume, and
        [P, resume) is never rewritten for the layers above the pause.
   - Fix: mirror the parked-path clamp on the live path - when an armed pause is dropped and
     `resume > read_pause.pos`, cap the read (and the trust: live/imgs/checkpoints) at
     `read_pause.pos`, so the chunk is re-read over all its layers before anything deeper is trusted.
     (go_on true is fine: read_from = pos, resume-or-reread covers the chunk whole; `reread_to > 0`
     reads from 0.)
2. The card script's "nothing read twice" leg-sum check cannot pass: the REPORT's stats claim (REPORT
   :166, "The PP progress lines count the paused chunk as read on the leg that paused it") is not
   implemented, and tools/card_tests/layer_pause.sh is written against the claim, not the engine.
   - pp_reached advances only in on_chunk, at a COMPLETED chunk's end (generate.cpp:11514-11515); a
     paused chunk never reaches on_chunk, and nothing bumps pp_reached on the pause keep (the keep
     branch generate.cpp:22881-22905 does not touch it). The cancelled leg's read figure is
     `read_n = pp_reached - resume` (generate.cpp:23083) - the paused chunk is in NEITHER leg's `read`
     and only in the next leg's `reused`.
   - The script sums the per-leg `K read` figures from the engine's own lines and fails above
     READ_TOL=1024 (tools/card_tests/layer_pause.sh:47, :232-237). The shortfall is exactly one paused
     chunk per cut - 8-16K per the plan header - so on the card a correct feature FAILS the script's
     condition 3 with "something was read twice" (it is actually read once LESS). Condition 4 (last
     leg `reused + read = prompt`) does hold; so does the engine's own accounting - only the sum is
     wrong. The last-leg check at :238 and the `M > 0` check at :224-226 are correct.
   - Fix either side: bump `pp_reached = lp_pos + lp_tokens` in the pause keep (which also makes the
     RDONE/DONE `read` figures match what the REPORT says and the server's PP-based position honest),
     or account for one chunk per cut in the script. As written, the plan's step-4 card proof wastes a
     full card run (two model loads + a 60K control read) before failing at :237.

## NIT

3. `Prefill::paused_chunk_index()` doc says "0 when the pause was not taken in a run"
   (prefill.hpp:457-459) but `give_pause` leaves `SavedPause::chunk = -1` (prefill.cpp:1964-1973), so
   a restore-then-pause-label read reports -1, not 0. Label-only; say -1 or store 0.
4. `state_bytes` computes `per_row = row_floats * 4` BEFORE the saturation guard
   (layer_pause.hpp:66-71): a `row_floats >= 2^62` wraps per_row first and the guard then divides by a
   wrapped value. Unreachable with the real D = 10240 and untested; clamp `row_floats` first.
5. A resumed chunk whose saved layer > max(LB,1) still launches its PLE gather (prefill.cpp:3305-3309)
   and uploads T x N floats into `m.ple_emb` at the post-loop `ple_land` - but the only readers of
   `m.ple_emb` are the l == 1 blocks (prefill.cpp:3640, 3668), which `l_from` skips. Correct, but a
   wasted SSD gather + ~84 MB H2D per resume; skip the gather/land when `lp_resume && l_from > max(LB,1)`.
6. SETTINGS.md now defaults SAGUARO_PRIORITY_IDLE_S to 0 "since P125j: a low prompt read steps aside
   at the next LAYER" (docs/SETTINGS.md:238, :250-252) - but with SAGUARO_LAYER_PAUSE=0 (whose new row,
   docs/SETTINGS.md:2103, promises "exactly as before") the grace that protected a follow-up after an
   answer is gone and the between-chunk wait is back. Worth one clause: with the pause off, set the
   grace back to 5.

## Previous round's findings - all fixed, re-verified here

- BLOCKER 1 (used-before-declared `lp_resume`, tree did not compile): the resume block is hoisted to
  the chunk top (prefill.cpp:3227-3240) ahead of its uses, and the H2D now goes on `m.cs` +
  stream-sync. Reproduced GREEN on this box: `cmake --build scratch/build-p125j-cuda75 --target
  strata_prefill` and `--target strata` both up to date, exit 0 (scratch/review_build_strata.log).
- MAJOR 2: layer_pause_test.cpp exists with 41 hand-computed checks + CMake entry
  (engines/qfn/CMakeLists.txt:1713-1718); ran: passes ("ok: 41 checks, 0 bad").
  tools/card_tests/layer_pause.sh exists, executable, `bash -n` clean - but see MAJOR 2 above.
- MAJOR 3: grace default 0.0 (serve/server.py:416), pause line names chunk/layer only when the DONE
  carries the keys (serve/server.py:5878-5885 - byte-identical otherwise, test_lost_read.py:56's
  un-anchored regex still matches), docs updated, IdleGraceDefault asserts 0.0. Generated file: I ran
  tools/saguaro/extract_qfn_engine.py - serve/qfn_engine.py came out byte-identical (clean `git
  status`): it IS in sync, and the only hunk is the DONE-key tuple.
- MAJOR 4: deeper-than-pause borrows refused (generate.cpp:19715-19725), whole-adopt keeps the pause
  only landing at the chunk end and otherwise clamps incoming_tokens/live/imgs/checkpoints to
  pause_pos (generate.cpp:20124-20148); disk writes refuse a paused conversation
  (turn_capture :10738, spill_current :10845, park file :11108).
- MEDIUM 5: drop_read_pause at LOAD (:19188), slot copy-in (:19944), borrow (:20014), incoming
  (:20067), cvec switch (:20189), #879 flush (:19546) and every keep branch (:22880, :22913, :22950).
- MEDIUM 6: `resumes()` no longer tautological (layer_pause.hpp:54-61; caller passes the saved
  point's own fields, prefill.cpp:3230); doc matches behaviour; the cross-run continuation is covered
  by the unit test.
- NIT 7-9: take_pause_rows clears the record (prefill.cpp:1957-1960), give_pause returns whether it
  armed and the restore mirrors only on true (generate.cpp:20124-20135), the chunk label uses the
  run's own C (`c0 / C` stored, prefill.cpp:5521; read via paused_chunk_index(),
  generate.cpp:20646).

## Checked and OK

- Feature OFF: `lp_on` (prefill.cpp:2978-2979) is constant false with SAGUARO_LAYER_PAUSE=0 or no
  callback, and every new block needs it or `pause_`/`lp_resume` (never armed) or `lp_pos >= 0`;
  `if (!lp_resume && hand_in_ == nullptr)` and the wrapped embedding block degenerate to the old
  code; run()'s read-past clear (prefill.cpp:6194) is a no-op with `pause_.pos0 == -1`; the DONE
  `%s` prints empty; SavedConversation::bytes() adds only capacities (0 when unused) and the disk
  format is untouched; the server line without keys is byte-identical. Grace 0 -> NIT 6 aside, the
  pause itself restores base behaviour with the switch.
- Pause/resume mechanics: check sits last in a full layer pass (prefill.cpp:5502-5541), never after
  the stage's last layer; sync on `m.cs` before the DtoH; the return unwinds the existing RAII
  (IssuerJoin/StagerDone) and the PLE future's blocking dtor (prefill.cpp:3124); `normed` is per
  chunk (prefill.cpp:3592) so the resumed half recomputes from m.R; PLE history advanced iff the
  pause is at layer > 1, and the resume at l_from >= 2 skips that block - no double advance; the
  drafter is untouched (on_chunk only at completed chunk ends); go_on's `std::equal` is bounded by
  the `pos + tokens <= n` check evaluated first; `mtp.kv_restore(resume)` over [P, P+T) stays stale
  but is rewritten by the resumed chunk's on_chunk before any window reads it (drafts only propose).
- Locks/threads/stdout: no new mutex (so no nested acquire); the stop flag is the stdin thread's
  `std::atomic` (generate.cpp:12690-12694) read from the run thread; one fprintf+fflush per pause,
  one DONE printf; `fresh`/`last0` guard in serve/server.py:5882 reads the DONE written by
  gen.close() at :5816-5821; `self.last` is a fresh dict per DONE (serve/qfn_engine.py:1180) and
  carry_leg/add_carry (serve/qfn_engine.py:3869-3886) never touch paused_* keys.
- The leg figures the script reads: the awk fields map correctly onto both the `K read` and the
  cancelled `K of F read` line shapes (generate.cpp:23316-23323) - the sum is the problem, not the
  parsing; `maxpos`'s regex matches the server's "reading the prompt: X of Y tokens" line
  (serve/server.py:5436-5437).
- No secrets, no paths outside the repo (card script: relative ROOT_DIR, own --out folder, api_key
  dropped from the test copy, 127.0.0.1 only; PYTHON default is the install venv). Scratch build dir
  is repo-internal. Commits: per-finding, selective, no Co-Authored-By lines.

## What I ran (plan-named modules, one at a time)

- `cmake --build scratch/build-p125j-cuda75 --target layer_pause_test` +
  `ctest --test-dir scratch/build-p125j-cuda75 -R layer_pause_test`: Passed; binary: "ok: 41 checks,
  0 bad".
- `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_priority`: 68 passed (incl.
  IdleGraceDefault 0.0).
- `cmake --build scratch/build-p125j-cuda75 --target strata_prefill` and `--target strata`: exit 0
  (up to date - the committed tree compiles; BLOCKER 1 verified fixed).
- `tools/saguaro/extract_qfn_engine.py` (regeneration check): output byte-identical to the committed
  serve/qfn_engine.py; tree still clean afterwards.
- `bash -n tools/card_tests/layer_pause.sh`: clean; `py_compile` on serve/server.py,
  serve/qfn_engine.py, engines/qfn/serve/server.py: OK. The card script itself was NOT run - the plan
  excludes GPU runs and there is no card on this box (and see MAJOR 2 before it earns its keep).

## Manager's four

- Resume at the exact layer: implemented (l_from = saved layer, prefill.cpp:3227-3238, 3607), compiles,
  helper-tested; card proof pending on layer_pause.sh (which must first fix MAJOR 2).
- Nothing read twice: holds in the engine's own figures on the exact-continuation path (reused covers
  the paused chunk, resume skips layers below the pause, read-past rows are cleared, prefill.cpp:6194);
  the script's cross-leg sum as written cannot demonstrate it (MAJOR 2).
- Parks with the conversation: record + rows ride park/restore, counted in the budget
  (generate.cpp:10978, 11086-11108), disk writes refused; borrow/adopt holes closed. Open trust hole is
  the LIVE drop path, not the park (MAJOR 1).
- Idle grace -> 0 only with this: 0.0 shipped with the working pause, tested and documented (NIT 6 for
  the switch-off combination).
