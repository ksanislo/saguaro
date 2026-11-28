# P125j re-review round 6 (MERGE-WITH-NITS) - layer pause incl. the P125m batch/concurrent-reader lane

Reviewed `git diff c1f93671..HEAD` at 4a75d5fd3 against PLAN-p125j-layer-pause.md; the plan's task report
was dropped by cand3, FIX-REPORT.md is the live document and matches the sources at every site it cites.
The manager's round-5 card FAIL (no pause on the `--batch` concurrent reader, first token after 11.2 s) is
fixed end-to-end: the engine's layer-split HEAD now takes the pause (prefill.cpp:2986-2987 `lp_on` dropped
`next_ == nullptr`, block at :5524-5553), the reader leg captures it after the join
(generate.cpp:21240-21248) and reports where it got to (:21219-21221), the INFO batch line says
`layer_pause=1` while the feature is on (:13039-13042), and the server cuts a batch read only with that
fact (serve/server.py:6054-6058, QfnEngine.pauses_batch_read :10038-10044); the guard test pins the old
rule for engines without it (serve/test_priority.py:1027-1038).

MANAGER's four points, each verified from the sources:
- RESUME AT THE EXACT LAYER - the saved rows go up into m.R and the layer loop starts at the saved layer
  (prefill.cpp:3235-3246, `l_from = pause_.layer`); only the exact continuation goes on (go_on at
  generate.cpp:20540-20550) and a dropped pause caps trust at the pause token (:20552-20567, the parked
  adopt clamps the same way :20342-20368); PLE gather/land skip exactly when the resumed range skips its
  only reader (:3316, :5557-5561).
- NOTHING READ TWICE - the pausing leg counts the paused chunk through its end (`pp_reached = lp_pos +
  lp_tokens`, generate.cpp:23164, read figure :23355), the resumed leg reports it as reused (RESUME/DONE
  reuse = live end), so the legs' `prompt N = R reused + K read` figures sum to the prompt - the script's
  sum + last-leg checks (tools/card_tests/layer_pause.sh:236-246) close by construction.
- PARKS WITH THE CONVERSATION - rows move into the parked image (generate.cpp:11208-11224), the session-dir
  write and the disk spill are skipped while a pause is armed (:11228, :10965), and the restore arms record
  + rows only when the rows really landed (:20342-20352).
- IDLE GRACE -> 0 ONLY WITH THIS - default 0.0 (serve/server.py:416-420, IdleGraceDefault
  serve/test_priority.py:2122-2127), the batch cut gated on the engine's own `layer_pause` fact, and
  docs/SETTINGS.md tells operators to set the grace back to 5 with SAGUARO_LAYER_PAUSE=0 or an older
  batch engine.

Checks the brief asked for: OFF path - SAGUARO_LAYER_PAUSE=0 makes lp_on false for every run
(prefill.cpp:1946-1952, :2986-2987), the pause/resume blocks unreachable, DONE byte-identical (lp_note
empty), the INFO carries no layer_pause, and the server gate falls back to the pre-P125m rule
(serve/server.py:6054-6058) - the solo read behaves exactly as before. Locks/threads - no new lock
anywhere in the diff (the stop flag is the existing atomic, generate.cpp:12814, :12853); the pause state
is read only after `reader.join()` (:21207/:21240); the hand-off slots drain before any keep
(Prefill::run drains at prefill.cpp:6199-6208). stdout/stderr - the pause line is one fprintf+fflush to
stderr (prefill.cpp:5545-5550), protocol lines are single-line printf+fflush as elsewhere; the reader
thread already printed PP lines on this leg before the change. Generated file -
tools/saguaro/extract_qfn_engine.py re-run here: serve/qfn_engine.py byte-identical, engines/qfn/serve/server.py
untouched by the fix. Paths/secrets - nothing outside the repo, no keys (the card script drops api_key,
port 8092, own dirs under scratch/).

What ran (one at a time): ctest -R layer_pause_test in scratch/build-p125j-r5 (43 checks, 0 bad);
`python -m unittest serve.test_priority` 82 OK; `cmake --build scratch/build-p125j-r5 --target
saguaro-qfn` exit 0 (CUDA arch 75, binary newer than every source); the extractor; `bash -n` the card
script; py_compile the Python touched. NOT run: tools/card_tests/layer_pause.sh itself - no card in this
session and the plan excludes GPU runs; the manager's rerun on the live Q4 --batch config is where the
batch lane earns its keep.

Nits (none blocking):
1. engines/qfn/include/strata/prefill/prefill.hpp:345-352 - stop_between_layers' header doc still says the
   check runs for "a SOLO run (no queued job, no batched chunk, no layer split, no picture rows)"; since
   P125m the layer split's HEAD qualifies (prefill.cpp:2974-2987 has the corrected rule) - the header now
   contradicts the shipped behaviour for exactly the shape the card runs.
2. tools/card_tests/layer_pause.sh:80 - `rm -rf "$DIR"` runs on whatever --out names; the guard only
   rejects "", "." and ".." (:43), so a mistyped shared folder is deleted outright. Refusing a non-empty
   folder costs one line.
3. generate.cpp:21219-21221 - `lp_stop` asks `sp.paused_chunk() >= 0`, which a pause ARMED BY A RESTORE
   satisfies too: a reader-leg run failing for a non-stop reason while such a pause sat armed logs the
   read "to" the pause end instead of b0. Error routing is decided by stop_req (:21934-21946), so this is
   log-only on a path that ends in ERR anyway - worth a `read_err == layer paused` ask at most.
4. INFO - the batch-lane engine change has no automated coverage on this box beyond the build (the
   pure-helper test covers the arithmetic, test_priority the server side, the card script the leg);
   the plan accepts that split, but the manager's card rerun is the only thing that exercises
   lp_on-on-a-split-head for real.

Verdict: MERGE-WITH-NITS - the manager's four criteria hold and the round-5 FAIL is closed with a
RED/GREEN-pinned test; the three nits are comment/robustness only and can ride along or follow.
