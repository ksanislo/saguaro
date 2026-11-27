# FIX-REPORT — P124k keepread: answers to REVIEW-RESULT round 3 (diff 856af24ab..f0629fa17)

One finding at a time: 3da343739 (finding 1, engine half), 1a1444ac9 (finding 1, server half + tests), a7132a2d8
(round 2's two doc nits, the ones finding 4 carried over as standing); this file is the fourth. The round-1 answers
stand in history (`git show d2a96ae46:FIX-REPORT.md`); finding 4 says the round-2 review text - its nits and infos -
survives unchanged at `git show 3637f9901:REVIEW-RESULT.md`, and the two cheap doc nits from that list are fixed
here. `serve/qfn_engine.py` and `engines/qfn/serve/server.py` were not touched (the QFN driver already sets its
`reused`), so no extractor run was owed. No GPU run.

## Finding 1 — HIGH (FIX-FIRST) — FIXED (3da343739 + 1a1444ac9)

The review's first option taken, its second not: the engine now **says `RESUME <n>`** for every read and the
StrataEngine parses it into the attribute the continue line reads, reset per call. No fallback to the last DONE —
that option needs the `keepread` gate's scoping to stay honest (the gate is P124u on the candidate, not this file),
and a looser scope would figure GLM's and ik2's dropped reads exactly as the review warns.

* Engine (`src/ds4/ds4_main.cpp:1660-1665`): `prompt_begin` prints `RESUME%s %zu` with the request's `from` and the
  same `tag` its PP/DONE lines carry — `RESUME 32768` solo, `RESUME 7 32768` with `--parallel` — before the read,
  the convention `generate.cpp:2118` (`prep_prompt`) already keeps for GLM. It is the read's own figure by
  construction: a request resuming a kept read arrives as a continuation, and the path that computes `from`
  (`:1540-1543`, `:1570-1591`) is exactly where the keep (a live state at the kept position, or the restored
  checkpoint) lands — the card's kept 32,768 now rides the line the way it already rode the DONE's f[8] (:1883).
* Server (`serve/server.py`): a `reused` property beside `last` (:1848-1865) — thread-local with `multi`, instance
  solo, **None until a RESUME arrives**; the solo loop parses `RESUME <n>` (:2131-2133) and the multi loop parses
  the `RESUME <rid> <n>` `_route` brings to its own request (:2368-2370); `generate()` (:2054) and
  `_generate_multi()` reset it per call, and a figure that does not parse is treated as none. `Service._sliced` is
  untouched — its `getattr(self.engine, "reused", None)` (:5732, the :5704 the review quoted under the old line
  numbers) simply starts seeing DS4's number, and the line
  prints `the low prompt read continued: 32,768 of 83,932 tokens reused` for the card's shape.
* Honesty (P124b, serve/test_lost_read.py:65-79) holds: an engine that says nothing over the pipe keeps the
  numberless line (ik2, maya-metal, the `UnreportingRead` fake — nothing parses now that did not before, and the
  fakes are not `StrataEngine`s); a lost read keeps logging its own 0. Engines' lines only reach the figure their
  own read reused: there is no cross-call carry (reset) and no cross-request carry (per-thread in multi).
* New tests (`serve/test_priority.py`, `TheResumeLine`): the **real `StrataEngine`** against the fake parity engines —
  solo takes `RESUME 4` and then `RESUME 1` on the next call (per-call, not last-call's), multi takes its routed
  `RESUME <rid> 0`. The review's "Ds4Read fake ... is the one fix 1 must update" lives on the candidate line (its
  `serve/test_priority.py` carries `Ds4Read`, this branch's carries none — `grep -c Ds4Read` 0 vs 3); there is
  nothing here to update, and the new class carries the same intent for this line.
* This worktree's `serve/` was the base file; touching it is the review's own instruction ("Server change -> run
  the affected modules"), and the gate itself (READY's `keepread` word -> `keeps_stopped_read`) stays P124u — no
  gate line was added here.

## Finding 2 — HIGH (FIX-FIRST) — NOT FIXABLE IN THIS WORKTREE; lands on the candidate line

`tools/card_tests/lost_read.sh` is **not in this branch's tree**: the path exists only in the P124r range (the blob
at `a6f1c2ef2`), which is not an ancestor of HEAD (`git merge-base --is-ancestor e5f683c24 HEAD` fails), and
`git ls-files` carries no `tools/` — exactly as the review itself records ("the file is not in this worktree at
all ... fix 2 lands on that line too"). Creating the file here would fork the P124r script the candidate owns, and
rewriting the candidate's branch is outside this worktree. Nothing in this branch greps the wrong wording: this
branch's own check is `scratch/card-test-ds4-keep-read.sh`, whose PASS greps are engine-stderr only (:152-175) and
which the review says would have PASSED on the card's evidence; `serve/test_lost_read.py`, the tracked module of
this wording, is green.

What fix 2 must do where the file lives (candidate, `:187-198`): count the **figured** line
(`prompt read continued: M of N tokens reused`) — which after finding 1 DS4 now emits and QFN always had — and its
per-resume `M > 0` checks parse M from it; accept the numberless `continues from where the engine kept it` only as
a fallback that **skips the M > 0 checks with a NOTE**; fail fast (~60 s, not the full 600 s) when neither line
comes. Recorded here so the candidate round has it with the fix it depends on.

## Finding 3 — INFO — no change due

The correction (both priority lines DO print; only the figure was missing; the keep's numbers were right) is what
finding 1's fix answers — wiring, not new lines; nothing here contradicts it. P124k's own card script stands as
reviewed.

## Finding 4 — INFO — no code change; its "serve/ carries no change" note is superseded by design

The finding 1 fix adds the `serve/` wiring the review asked for; the diff is otherwise byte-identical to what
round 2 verified, except the one engine `printf` and this report round.

## Round-2 nits finding 4 carried over — the two cheap ones FIXED (a7132a2d8), the two snapshot ones left

* REPORT §4 now names the **stderr request-summary line** the script actually greps (`ds4_main.cpp:2076`/`:2404`)
  as the PASS criterion, with the DONE named only as the stdout carrier of the same `from` (nit 3).
* REPORT §4's ignore claim corrected: `.gitignore` has no `REPORT-*` pattern — the force-add was the card script's
  under ignored `scratch/`, and `REVIEW-RESULT.md` is what `.git/info/exclude:42` hides (nit 4; verified with
  `grep -n` on `.gitignore` and the exclude file).
* The two snapshot nits stay untouched, as round 2 itself rated them ("bounded ... no correctness impact", "not
  worth API churn", "noted only"): they are one redundant `snap_into` on the stop path, and neither call site is
  observable from the plan's module (`ds4_marks_test` covers `read_stop_landing`, not the `prompt_piece` calls).

## Files and verification

* Touched: `src/ds4/ds4_main.cpp` (3da343739), `serve/server.py` + `serve/test_priority.py` (1a1444ac9),
  `REPORT-p124k-ds4-keep-read.md` (a7132a2d8), this file.
* Run, one module at a time, only the ones the plan/the review's fix constraint name:
  `g++ -std=c++17 -Wall -Wextra -O1 tests/ds4_marks_test.cpp` (out `scratch/ds4_marks_test_r4`) → `ds4_marks_test:
  ok`, no warnings; `ninja -C scratch/build-p124k maya-ds4` → EXIT 0 relink after the engine edit (nvcc 13.1, arch
  75, no GPU run) and `strings` shows the new `RESUME%s %zu` in `saguaro-ds4`;
  `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_lost_read` → 3 ok;
  `... -m unittest serve.test_priority` → 58 ok, including the two new `TheResumeLine` tests against the real
  `StrataEngine`.
* The RESUME parses are leaf branches: no yield, no queue, no routing change — the readers' line flow is what it
  was (verified by reading the loops; the modules above cover the drivers).
