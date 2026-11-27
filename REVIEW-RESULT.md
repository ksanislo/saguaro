# REVIEW-RESULT - P125j layer pause (re-review round 4 of d789228f0..HEAD; HEAD = bcfc5c03f, the round-3 MERGE; superseded by the manager's card FAIL)

FIX-FIRST

The round-3 MERGE holds for the code it read - all seven of round 2's findings are still fixed in the
tree and both plan-named unit modules pass here - but the feature never reaches the shape the SHIPPED
QFN config runs its prompt reads in (`--batch`, the concurrent reader), and the `SAGUARO_PRIORITY_IDLE_S`
5 -> 0 flip ships into exactly that config. The manager's card run on 4xT4 (live Q4 `--batch` config,
scratch/card-r1-test.txt, scratch/card-r1-engine.txt) failed exactly where this review locates it:
cut 1's normal request waited 12.2 s (the plan promises 1 s), no pause/priority/STOP line anywhere,
the low read ran 37632 tokens straight through "on a thread of its own".

## Findings

1. BLOCKER - the layer pause (and every read preemption) is inert under `--batch`; the card run fails.
   - serve/server.py:5744-5747: the read preemption (the cut that would send STOP) fires only for
     `keeps_stopped_read AND not engine.batch`. QfnEngine sets `keeps_stopped_read = True`
     (serve/server.py:9501) and its docstring (serve/server.py:9497-9507) says outright: "the preemption
     never cuts a batch read". `can_pause = False` (serve/server.py:9500) also kills the fair-read pause
     (serve/server.py:5727-5729). Under `--batch` the server therefore never sends a STOP for a prompt
     read at any granularity - neither chunk-end nor layer. That is why the card log has no STOP line
     at all.
   - Engine side: under `--batch` every long prompt read runs on the piped concurrent reader thread
     (generate.cpp:20766 gates `piped && (pr_conc || slot_admit)`; the reader thread runs `sp.run` at
     generate.cpp:20911-20928; the card engine log's line is its own print at generate.cpp:21010-21009
     region, scratch/card-r1-engine.txt:35). The P125j capture (`lp_pos/lp_layer/lp_tokens/lp_chunk`)
     exists ONLY in the inline solo `run` lambda (generate.cpp:20667-20670); the reader-thread run sets
     none of them, so the pause keep (generate.cpp:22907-22936) can never arm on that leg, and a run
     that ended `err "layer paused"` on the reader thread falls into `if (!read_ok)`
     (generate.cpp:21010-21013) and aborts the read as a failure - the saved rows are then dropped by
     generate.cpp:23000 (`sp.clear_pause()`). Note `lp_on` IS true for the reader-thread run
     (prefill.cpp:2978: no job, `bat` false, hand_in_/next_/embd_rows null, stop_between_layers wired at
     generate.cpp:12694), so the layer pause itself would fire if the flag were ever set - what is
     missing is the server cutting batch reads, the capture/keep on that leg, and the give-way/slot
     carrying of the layer+rows (the batch machinery stops at chunk heads only:
     `sp.stop_at_chunk` at generate.cpp:20906-20909, BYIELD at 20974-20975, give_way parking at
     20700-20718).
   - Consequence for the plan: step 4's promise (normal request's first token within 1 s, on the card,
     with the live config) is not met; `tools/card_tests/layer_pause.sh` correctly FAILs on any
     `--batch` CFG at "no 'prompt read paused' line" (scratch/card-r1-test.txt, cut 1 at position
     16128, TTFT 12.22 s).
2. MAJOR - the grace default 5 -> 0 ships without the protection it was bought with (MANAGER: "idle
   grace -> 0 only with this"). serve/server.py:416-417 defaults `SAGUARO_PRIORITY_IDLE_S` to 0.0,
   docs/SETTINGS.md:238 and 250-253 justify 0 by "a low prompt read steps aside at its next LAYER", and
   serve/test_priority.py's IdleGraceDefault asserts 0.0 - but finding 1 means that sentence is false
   for the batch config the package ships in: in 1.2.5 the 5 s grace was the only shield for a normal
   follow-up arriving while no low read has started yet (the old SETTINGS wording: "long enough for an
   agent's tool-call gap, since a low prompt read steps aside only at the end of a chunk"); with 0 the
   low read starts the moment the box is quiet and the follow-up queues behind the whole concurrent
   read (the card run's 12.2 s class of wait). NIT 7's fix (79cebc30e) documents only the
   `SAGUARO_LAYER_PAUSE=0` combination, not the `--batch` one. Either fix finding 1 (then 0 is earned)
   or keep the default at 5 when the engine runs batch slots and say so in the docs and the test.
3. NIT - REPORT-p125j-layer-pause.md:283-285 is accurate ("serve/server.py only ever cuts a solo QFN
   read, keeps_stopped_read and not batch") but then calls that "the shape the priority cut actually
   uses". For a live `--batch` QFN box it is not - that is the shape the cut CANNOT use. The report's
   honest-scope note should name `--batch` as the config where the whole feature is inert, and step 4's
   card claim should say its checks can only pass where a pause can be taken at all.
4. NIT - when finding 1 is fixed through the batch machinery, the card script's expectations must move
   with it: the script greps the solo mechanics (server line "inside chunk c at layer l", engine line
   "the read paused at chunk" from prefill.cpp:5537, the solo RESUME/RDONE "prompt N tokens = R reused +
   K read" figures summed against READ_TOL). A batch cut that gives way through BYIELD/give_way parks a
   partial read in a SLOT (generate.cpp:20700-20718) and reports slot/BGEN figures instead - either the
   script covers both, or it pins a solo config for these exact lines.

## Round 2/3's items - re-verified in this tree, nothing regressed

All seven of round 2's fixes sit where round 3 said (spot-checked at HEAD): the dropped-pause trust
clamp above `mtp.kv_restore` (generate.cpp:20323-20351); the pause keep's `pp_reached = lp_pos +
lp_tokens` (generate.cpp:22924); both head-alias pair-drops (generate.cpp:19513, 21746) plus the keep's
`sp.pause_rows_bytes() > 0` refusal (22907); the `-1` doc in prefill.hpp; the `state_bytes` row clamp
with the two extra checks (43 green); `ple_reads` gating the gather/pending/land plus the stale
`ple_next` wait (prefill.cpp:3308-3314, 5549-5554); the SETTINGS.md grace-back-to-5 clause.

## Re-checked per the brief (this pass)

- Feature OFF (`SAGUARO_LAYER_PAUSE=0`): `lp_on` false at prefill.cpp:2978-2979; `lp_resume` never true;
  `l_from == LB` (prefill.cpp:3228, 3611); the embedding/gr_broadcast guards (3241/3300/3301), the pause
  block (5516-5554), the run-past clear (6198-6201) and every generate.cpp block sit behind `lp_on`,
  `lp_resume` or an armed `pause_` that only a taken pause can arm; DONE's new `%s` carries the empty
  `lp_note`; the server line is byte-identical without the DONE keys (serve/server.py:5877-5887); the
  extra `sp.clear_pause()` at 23000 is a no-op on an empty record. The grace default is the only
  intended OFF-visible change - finding 2.
- Locks / threads / stdout: no new mutex/lock_guard/unique_lock on any `+` line of the diff; the pause
  reads the stdin thread's atomic `stop_req` (generate.cpp:12694) from the run thread - the same
  pattern as `should_stop`; one `fprintf(stderr)+fflush` per pause (prefill.cpp:5537-5542) and one keep
  line (generate.cpp:22932-22936); the DONE keys ride the single existing DONE printf (23139-23147) - no
  line interleaving; nothing acquires a lock the pause path could already hold.
- Generated file: serve/qfn_engine.py's only hunk (serve/qfn_engine.py:1193-1195) mirrors the source
  hunk (engines/qfn/serve/server.py:1636-1638) key for key; no other drift in either file (regenerating
  would edit the tree, so this pass verified hunk-for-hunk; round 3 verified byte-identical output).
- Tests really exercise the change: layer_pause_test walks the exact `pause_point` / `resumes` /
  `state_bytes` the engine calls (prefill.cpp:5517, 3230, 5541); test_priority pins the new default.
  But nothing automated covers the server gate, the reader-thread leg, or a resume end-to-end - those
  rest solely on the card script, and it failed (findings 1/4).
- No secrets, no paths outside the repo in the diff (the script reads CFG only, writes under its
  --out folder, drops `api_key`); no forbidden commit trailers in `git log d789228f0..HEAD`.

## What ran here (plan-named modules, one at a time)

- `cmake --build scratch/build-p125j-cuda75 --target layer_pause_test` + `ctest -R layer_pause_test`:
  Passed ("ok: 43 checks, 0 bad" from the binary's own rule).
- `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_priority`: Ran 68, OK.
- `bash -n tools/card_tests/layer_pause.sh`: clean.
- `tools/card_tests/layer_pause.sh` NOT run here (no card); the manager's card run on the live Q4
  `--batch` config FAILED as quoted in finding 1 - that is the plan's step-4 gate.

## Manager's four

- Resume at the exact layer: implemented and helper-tested on the SOLO GEN path
  (prefill.cpp:3229-3239/3611 + capture 20667 + keep 22907) - but never reached on a `--batch` read
  (finding 1).
- Nothing read twice: the leg figures telescope after bb0c65a76 (22924) wherever a pause can arm; the
  card run exercised no leg at all (no cut), so the sums stay unproven end to end.
- Parks with the conversation: record+rows ride the park (generate.cpp:11089-11104), come back at the
  restore (20125-20135), count in the budget (10978), and refuse disk writes while live (10738, 10845,
  11108-refuse) - holds, for solo reads.
- Idle grace -> 0 only with this: NOT met - 0 ships into the batch config where the pause is inert
  (finding 2).

## Fix (the manager's steps, with the sites this review verified)

1. FIX-REPORT: trace a low read under `--batch` from the server's step-aside (serve/server.py:601
   `step_aside`, gated out at 5744-5747, QfnEngine docstring 9497-9507) into the engine's reader
   (generate.cpp:20766, 20892-21033): the stop is checked at chunk heads only (`sp.stop_at_chunk`
   20906-20909 wired to BYIELD 20974-20975), and `stop_between_layers` (12694) is armed but its result
   has no capture/keep on that leg (20667 is inline-only; 21010-21013 aborts; 23000 drops the rows).
2. Make the concurrent reader honour the layer-boundary stop and the server's step-aside: open the cut
   for batch (or route it through the BYIELD lane), capture the pause figures on the reader-thread leg
   and let the keep / give_way carry them (a slot's partial read would need the layer+rows too -
   20700-20718 parks ids only), keep the legs' reused+read = prompt closed either way, and log
   'prompt read paused' with chunk/layer as the solo path does.
3. A unit/fake test for the concurrent-reader path; rebuild QFN sm_75 in a NEW dir under scratch/; run
   tools/card_tests/layer_pause.sh on the live Q4 config until it PASSes (its expectations may need the
   batch wording - finding 4).
4. The grace default stays 0 only once step 2 works end to end on the card; if the batch cut does not
   ship, the default goes back to 5 (serve/server.py:416, test_priority, SETTINGS.md in one step).
