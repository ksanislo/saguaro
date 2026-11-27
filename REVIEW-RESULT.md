# REVIEW-RESULT - P125j layer pause (re-review of d789228f0..HEAD; HEAD = the previous round's review commit)

FIX-FIRST

Reviewed the committed diff `d789228f..HEAD` (16 files, +1355/-18) against PLAN-p125j-layer-pause.md,
REPORT-p125j-layer-pause.md and FIX-REPORT.md. No code has landed since the previous re-review
(HEAD 7271dc31b touches only REVIEW-RESULT.md), so its two open MAJORs and four NITs are all still in
the tree - each one re-verified here from the code, not carried over. One NEW MAJOR: `--prefill-head-alias`
resets the prompt path after every read and desyncs the pause record from its rows, which re-opens the
same silent-wrong-read hole on EVERY follow-up of a paused conversation (not just image ones) and lets a
parked paused conversation reach the session dir as if complete. Everything else the brief asks for was
re-checked and holds; all plan-named test modules were run and pass.

## MAJOR

1. (carried, re-verified) Dropping an armed pause on the LIVE path reads on from `resume` over cells no
   layer >= pause_layer ever wrote - the parked path's trust clamp was never copied to the live path.
   - The keep branch runs the LIVE conversation THROUGH the paused chunk (`live.assign(cur.begin(),
     cur.begin() + lp_pos + lp_tokens)`, generate.cpp:22889) - for layers < pause_layer only - and that
     is the one place the pre-P125j invariant "every cell below live.size() is written for every layer"
     is broken. The parked path knows this and clamps (generate.cpp:20124-20148: adopt of a paused image
     landing inside the chunk caps incoming_tokens/live/imgs/checkpoints to pause_pos, "the conversation
     is trusted only as deep as every layer wrote").
   - The live path has no clamp: go_on (generate.cpp:20321-20331) accepts only the EXACT continuation;
     every rejection is `else drop_read_pause()` (generate.cpp:20333) and the read then starts at
     `read_from = reread_to > 0 ? 0 : resume` (generate.cpp:20335), which can be strictly past
     `read_pause.pos`. Reachable with no env vars: a low read pauses at chunk P; the same conversation's
     next request carries an image - `starts_with(live, live_imgs)` matches the whole live prefix
     (generate.cpp:19588, the image sits past it) so resume == P+T == live.size(); go_on fails only on
     `req_imgs.empty()` (generate.cpp:20323) -> drop -> read_from = P+T. Layers >= pause_layer never read
     [P, P+T) - nothing ever will - decode attends over never-written (previous-turn) K/V: silently wrong,
     and the request ends as a normal successful turn with live_ok (generate.cpp:22877-22880), so the
     corruption parks/spills as a complete conversation.
   - The previous round's second trigger (prompt edited inside the paused chunk, resume in (P, P+T)) is
     NOT reachable at this HEAD: `resume` is only ever a full prefix length - live.size() or a checkpoint
     /borrow length (generate.cpp:19588-19613) - checkpoints are only saved at COMPLETED part ends
     (generate.cpp:21687-21695), deeper-than-pause borrows are refused (generate.cpp:19715-19725) and the
     slot copy-in drops the pause (generate.cpp:19944). But the missing clamp leaves the invariant
     unprotected; the same fix is still right: when an armed pause is dropped and `resume >
     read_pause.pos`, cap the read and the trust (live/imgs/checkpoints) at `read_pause.pos` so the chunk
     is re-read over all its layers before anything deeper is trusted.
2. (carried, re-verified) The card script's "nothing read twice" leg-sum check cannot pass - it is
   written against the REPORT's stats claim, which is not implemented.
   - "The PP progress lines count the paused chunk as read on the leg that paused it" (REPORT:166): not
     true. `pp_reached` is set to `read_from` per request (generate.cpp:20340) and advances ONLY in
     on_chunk, at a COMPLETED chunk's end (generate.cpp:11514-11515); a paused chunk never reaches
     on_chunk and the pause keep (generate.cpp:22881-22904) does not touch it. The cancelled leg's read
     figure is `read_n = clamp(pp_reached - resume, 0, fresh)` (generate.cpp:23083), so the paused chunk
     is in NEITHER leg's `read` (the cut leg's pp_reached stops at its start; the resumed leg's
     `pp_reached - resume` starts counting above it) and only in the next leg's `reused`.
   - The script sums the per-leg `K read` figures from the engine's own
     "prompt N tokens = R reused + K read" lines (tools/card_tests/layer_pause.sh:228-237) against
     READ_TOL=1024 (line 47): the shortfall is exactly one paused chunk per LAYER-pause cut - 8-16K per
     the plan header - so on the card a CORRECT feature FAILS line 237 ("something was read twice" - it
     is actually counted one chunk LESS) after spending two model loads and a 60K control read. A
     chunk-end (non-layer) stop does sum exactly (its leg's last chunk completes). Condition 4,
     `LAST_R + LAST_K == LAST_N` (line 238), does hold, as do the parsing/awk fields and the `M > 0`
     check (lines 224-226).
   - Fix either side: bump `pp_reached = lp_pos + lp_tokens` in the pause keep (which also makes the
     DONE `read` figures and the PP/server progress line match the REPORT's wording), or account one
     chunk per cut in the script. Do not ship the grace-default flip's card proof broken.
3. (NEW) `--prefill-head-alias` breaks the record/rows pairing invariant the whole feature rests on.
   With the flag, the request loop resets the prompt path AFTER the read loop - including on a cancelled
   /paused leg (`if (o.prefill_head_alias) { sp.reset(); ... }`, generate.cpp:21719-21720) - and again
   at the next request's top (generate.cpp:19511-19512). `Prefill::reset()` clears `pause_` with the
   buffers (prefill.cpp:1184-1187) but NOT the engine's `read_pause` record:
   - the keep branch still arms it - its `lp_pos` was captured from `sp.paused_chunk()` during the run
     (generate.cpp:20642-20646), before the reset - so generate.cpp:22893-22896 records a pause whose
     rows are gone (the keep line itself prints "0 MiB", generate.cpp:22904);
   - the next request's go_on fails on `sp.paused_tokens() == read_pause.tokens` (generate.cpp:20330)
     for EVERY continuation - even the exact one - falls into the MAJOR 1 hole at generate.cpp:20333-20335
     and reads from P+T over the never-written cells;
   - and a later park finds nothing to take (generate.cpp:11089-11098): `pause_rows` empty ->
     `parked_paused` stays false -> `sdir->write(conversations.newest())` runs (generate.cpp:11108) and
     the half-written session goes to the session dir looking complete - exactly what the P125j refuse
     rules were added to prevent.
   Not reachable with the flag off (default false, generate.cpp:860; no repo config and no server code
   passes it) - which is why this is MAJOR and not a BLOCKER - but the fix is small: keep the pair
   together like everywhere else does, e.g. call `drop_read_pause()` around the two sp.reset() sites, or
   refuse the pause keep when `sp.pause_rows_bytes() == 0`.

## NIT (all carried, all still present at HEAD)

4. `paused_chunk_index()` doc: "0 when the pause was not taken in a run" (prefill.hpp:359-361) - a
   restore-armed pause has `SavedPause::chunk = -1` (give_pause never sets it, prefill.cpp:1966-1974),
   so a restored pause's label reports -1, not 0. Label-only; say -1 or store 0.
5. `state_bytes` computes `per_row = (uint64_t) row_floats * 4` BEFORE the saturation guard
   (layer_pause.hpp:69-70): `row_floats >= 2^62` wraps per_row first and the guard then divides by the
   wrapped value (per_row == 0 divides by zero). Unreachable with the real D = 10240 and untested; clamp
   `row_floats` first.
6. A resumed chunk whose saved layer > max(LB,1) still starts its PLE gather (prefill.cpp:3305-3309) and
   lands ~84 MB of rows into `m.ple_emb` at the post-loop `ple_land` (prefill.cpp:5544) - but the only
   readers of `m.ple_emb` are the l == 1 blocks (prefill.cpp:3640, 3668), which `l_from` skips
   (prefill.cpp:3613 gates the in-loop land). Correct, one wasted SSD gather + H2D per resume.
7. SETTINGS.md now defaults SAGUARO_PRIORITY_IDLE_S to 0 "since P125j: a low prompt read steps aside at
   the next LAYER" (docs/SETTINGS.md:238, 250-252) while the new SAGUARO_LAYER_PAUSE row promises "0: ...
   exactly as before" (docs/SETTINGS.md:2103) - with the pause off, the grace that protected a follow-up
   is gone and the between-chunk wait is back. One clause: with the pause off, set the grace back to 5.

## Previous round's findings - fixes re-verified here at HEAD

- BLOCKER 1 (used-before-declared `lp_resume`): the resume decision is hoisted to the chunk top
  (prefill.cpp:3227-3238) ahead of both uses (:3241, :3301); the H2D goes up on `m.cs` and waits.
  Reproduced green on this box: `cmake --build scratch/build-p125j-cuda75 --target strata_prefill` and
  `--target strata` both exit 0.
- MAJOR 2: engines/qfn/src/program/layer_pause_test.cpp exists (CMake entry
  engines/qfn/CMakeLists.txt:1713-1718), ran here: "ok: 41 checks, 0 bad"; tools/card_tests/layer_pause.sh
  exists, executable, `bash -n` clean - but see MAJOR 2 above.
- MAJOR 3: grace 0.0 (serve/server.py:416), pause line names chunk/layer only when the DONE carries the
  keys (serve/server.py:5878-5885), docs updated, IdleGraceDefault asserts 0.0 - serve/test_priority.py
  passes here (68). Generated file: `tools/saguaro/extract_qfn_engine.py` re-run here produced a
  byte-identical serve/qfn_engine.py (tree clean afterwards) - in sync, generator-made, the only hunk the
  DONE-key tuple in BOTH engines/qfn/serve/server.py:1636-1638 and serve/qfn_engine.py:1194-1196.
- MAJOR 4: deeper-than-pause borrows refused (generate.cpp:19715-19725); whole-adopt keeps a pause only
  landing at the chunk end, otherwise clamps incoming_tokens/live/imgs/checkpoints to pause_pos
  (generate.cpp:20124-20148); disk writes refuse a paused conversation (turn_capture
  generate.cpp:10738, spill_current :10845, park-file write :11108).
- MEDIUM 5: drop_read_pause at LOAD (generate.cpp:19188), #879 flush (:19546), slot copy-in (:19944),
  borrow (:20014), incoming (:20067), cvec switch (:20189), keep branches (:22880, :22913, :22950) and
  the cancelled-request cleanup pair-check (:22963-22968).
- MEDIUM 6: `resumes()` is the whole-chunk check at the run's own c0 (layer_pause.hpp:56-62); the
  caller passes the saved point's own fields (prefill.cpp:3230); doc matches; the cross-run continuation
  is covered by the unit test.
- NIT 7-9: take_pause_rows clears the record (prefill.cpp:1959-1962); give_pause returns whether it
  armed and the restore mirrors only on true (generate.cpp:20126-20135); the chunk label uses the run's
  own C (prefill.cpp:5524, read via paused_chunk_index at generate.cpp:20646).

## Checked and OK (my own pass)

- Feature OFF: `lp_on` is false for every run with SAGUARO_LAYER_PAUSE=0, no callback, a job, a batch, a
  split or picture rows (prefill.cpp:2978-2979), and every new block needs it, `lp_resume` (never true)
  or `pause_` (never armed): the embedding block degenerates to the old code (prefill.cpp:3241-3301),
  the gr_broadcast condition to `hand_in_ == nullptr` (:3301), the loop to `l = LB` (:3607); run()'s
  read-past clear (prefill.cpp:6194) no-ops with `pause_.pos0 == -1`; the DONE `%s` prints empty so the
  line is byte-identical; the server line without the keys is byte-identical (serve/server.py:5875-5887);
  SavedConversation::bytes() adds only capacities (0 when never used) and the disk format is untouched;
  `layer_pause_wanted()` is a cached env read, `stop_between_layers` is wired but never CALLED with the
  switch off (prefill.cpp:5512-5515).
- Pause/resume mechanics: the check sits at the FULL layer-pass end, after both HC halves and the U5
  heartbeat, never after the stage's last layer or in a one-layer stage (prefill.cpp:5505-5542,
  layer_pause.hpp:34-45); stream-sync then DtoH of m.R holds the post-layer-l rows = layer l+1's input;
  the return unwinds the real RAII (IssuerJoin :3536-3539, StagerDone :3496-3498, KvDrain :3009-3011,
  the blocking future dtor :3124); `normed` is per chunk (prefill.cpp:3592) so the resumed half recomputes
  from m.R; PLE history advances only in the l == 1 blocks (prefill.cpp:3618-3674), so a pause at layer
  >= 2 and a resume at l_from >= 2 neither skip nor double it; `m.tok_dev`/`m.emb` have no readers outside
  the skipped embedding block; the drafter is untouched (on_chunk only at completed chunk ends,
  generate.cpp:11506-11524); go_on's `std::equal` is bounded by the `pos + tokens <= n` term evaluated
  first; run()'s clear (prefill.cpp:6194) covers a successful run that re-read the chunk whole.
- Pause keep fallbacks: if the keep is refused, the #1620 branch lands the session at `pp_reached` - a
  completed-chunk end, fully written - and the cleanup pairs record+rows down together
  (generate.cpp:22905-22915, 22963-22968); a leg that merely carried an armed pause (cancelled before the
  chunk) re-keeps the identical, still-valid pair.
- Locks/threads/stdout: no new mutex anywhere (no nested acquire possible); the stop flag is the stdin
  thread's atomic read from the run thread (generate.cpp:12688-12694, same flag as should_stop); one
  fprintf+fflush per pause (stderr) and one DONE printf (stdout) - no interleaving; the server reads the
  pause keys only from the fresh DONE written by `gen.close()` (serve/server.py:5816-5821, 5882),
  `self.last` is a fresh dict per DONE (serve/qfn_engine.py:1180) and carry_leg/add_carry
  (serve/qfn_engine.py:3869-3886) never touch paused_* keys.
- Card script: port 8092, relative ROOT_DIR, own --out folder and session dir, api_key dropped from the
  test config copy, 127.0.0.1 only, engine stderr kept in the out folder; the `prompt N tokens = R
  reused + K read` awk fields map onto both the `K` and cancelled `K of F` shapes (the regex trims at
  "prompt", so $4/$7 land right); maxpos's regex matches the server's progress line
  (serve/server.py:5436-5437). NOT run - no card on this box and the plan excludes GPU runs (and MAJOR 2
  must land first or it wastes the run).
- No secrets, no paths outside the repo in the diff; commits are per-finding and selective, no
  Co-Authored-By/Claude-Session/Assisted-by trailers anywhere in `git log d789228f..HEAD`.

## What I ran (plan-named modules, one at a time)

- `cmake --build scratch/build-p125j-cuda75 --target layer_pause_test` +
  `ctest -R layer_pause_test`: Passed; binary: "ok: 41 checks, 0 bad".
- `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_priority`: Ran 68, OK (IdleGraceDefault 0.0).
- `cmake --build scratch/build-p125j-cuda75 --target strata_prefill` / `--target strata`: exit 0 each -
  the committed tree compiles.
- `tools/saguaro/extract_qfn_engine.py` (regeneration check): byte-identical to the committed
  serve/qfn_engine.py; `git status` clean afterwards.
- `bash -n tools/card_tests/layer_pause.sh`: clean; `py_compile` on serve/server.py, serve/qfn_engine.py,
  engines/qfn/serve/server.py, serve/test_priority.py: OK.

## Manager's four

- Resume at the exact layer: implemented and helper-tested (l_from = saved layer, prefill.cpp:3229-3237,
  3607); the card proof is blocked by MAJOR 2, and the resume silently degrades to the MAJOR-1 hole on
  image follow-ups (MAJOR 1) and, with --prefill-head-alias, on every follow-up (MAJOR 3).
- Nothing read twice: holds in the engine's own figures on the exact-continuation path (reused covers the
  paused chunk, the read restarts at its first token, read-past rows are cleared, prefill.cpp:6194); the
  script's cross-leg sum as written cannot demonstrate it (MAJOR 2).
- Parks with the conversation: record + rows ride park/restore and the budget (generate.cpp:10978,
  11088-11108), disk writes refuse while paused, borrow/adopt holes closed - but the head-alias reset
  desyncs the pair and re-opens the disk-write hole (MAJOR 3).
- Idle grace -> 0 only with this: 0.0 ships with the working pause, tested and documented (NIT 7 for the
  switch-off combination).
