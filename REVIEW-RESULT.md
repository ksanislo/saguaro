# REVIEW-RESULT — P124k DS4 keepread, round 3 (card-test follow-up), diff 856af24ab..3637f9901

**FIX-FIRST (small)** — the engine's keep WORKS on the card (cut at 24,576 kept live at 32,768, resumed read
51,164, no second whole read). What fails is the SERVER's continue line and the shared card script: DS4's resumed
read gets the numberless line because the engine's own figure never reaches the attribute the line reads, and
`tools/card_tests/lost_read.sh` greps a wording it can never see and waits its full 600 s before saying so.
The P124k diff itself is unchanged since the round-2 review (MERGE-WITH-NITS) and needs nothing; both fixes sit
in the P124u/candidate territory, not in this worktree's diff.

## Findings by severity

### 1. HIGH — the resumed read's continue line carries no figure for DS4: the engine's number is parsed but
never wired to the line. `Service._sliced` asks `getattr(self.engine, "reused", None)` (serve/server.py:5704; the
identical block at saguaro/rc124-cand:serve/server.py:5813) — but that attribute exists only on the QFN driver
(serve/qfn_engine.py:684, set at :2504 from the engine's `RESUME` line, reset per call at :1782/:2430/:2743).
The `StrataEngine` driver that runs DS4 never sets it (no `self.reused` anywhere in serve/server.py): its DONE's
reused field lands in `self.last["reused"]` only (serve/server.py:1852; cand :1866), and RESUME lines are merely
routed for the multi PP base, never parsed into the attribute (serve/server.py:1694-1705). The engine already
reports the truth over the pipe — the stopped branch's DONE carries the kept position as f[8]
(src/ds4/ds4_main.cpp:1877) — so the line falls to the numberless branch (serve/server.py:5709-5710): the card
run printed `the low prompt read continues from where the engine kept it` (manager's extract
scratch/card-r1-engine.txt:15, 11:16:42) while the same run's engine lines say `kept live at 32768 tokens` (:3)
and the resumed request read only the rest: `prompt 83932 tokens = 32768 reused + 51164 read` (:55). This is
exactly the gap the P124u report left open on the candidate (commit d6dab3fd0, open question "the DONE's reused
figure never reaching the continue line"). Fix constraints (P124b's honesty rule, serve/test_lost_read.py:65-79 —
a lost read must log 0, an engine that reports nothing keeps the numberless line, serve/test_priority.py:715):
the figure must be that read's own — either the DS4 engine emits `RESUME <n>` (the convention the pipe already
has: src/program/generate.cpp:2118, :3900 are its only emitters today; src/ds4 emits none) and the StrataEngine
parses it into `.reused` with a per-call reset, or a fallback to the last DONE is scoped strictly to engines whose
STOPPED-read DONE carries the keep — GLM's and ik2's cancelled DONE carries the base, not a kept read, and must
keep the numberless line. Server change → run the affected modules: `serve.test_lost_read`, `serve.test_priority`.

### 2. HIGH — `tools/card_tests/lost_read.sh` cannot see DS4's line and burns its full timeout when it is
missing. It counts the literal `'prompt read continued'` (saguaro/rc124-cand:tools/card_tests/lost_read.sh:188,
:191), extracts M only from the figured wording (`prompt read continued: M of N tokens reused`, :196-198), and
polls 600 s before failing (:187-191) — the manager's run matches: last real work at 11:18:18
(scratch/card-r1-engine.txt:57), the script's SIGTERM at 11:26:33 (:58), ~8 min of idle cards. The fix has a trap
the manager's wording alone walks into: greping only `continues from where the engine kept it` (the today-wording)
leaves the script's per-resume `M > 0` assertions (:196-205) with no figure to parse — accept the figured line
(which after fix 1 DS4 gets, and QFN already had), accept the numberless line as a fallback that SKIPS the M>0
check with a NOTE, and fail fast (~60 s) when neither comes. Note the file is not in this worktree at all: it
arrives from P124r (candidate commits e5f683c24..a6f1c2ef2); this HEAD tracks only serve/test_lost_read.py, so
fix 2 lands on that line too.

### 3. INFO — the round-3 brief overstates the server gap, and that changes the fix. The DS4 path does log both
priority lines: the pause line printed verbatim (scratch/card-r1-engine.txt:4, printed at serve/server.py:5871-5873;
cand :5983) and the continue line printed too, in its numberless wording (:15). What is missing is only the figure
on the continue line (finding 1) — the fix is wiring, not new lines. And the keep's own numbers are right: the cut
was seen at 24,576 = 3 × the 8192 default step and landed live at 32,768 = 4 × step (a between-runs stop, the
`read_stop_landing` live case, src/ds4/ds4_marks.hpp:128-144), the total engine read was one pass, not two (:3,
:55). P124k's own card check would have PASSED on this very evidence — its greps are engine-stderr only, `stopped
while reading the prompt` / `kept live at N tokens` / `prompt N tokens = K reused` with 0 < K < N
(scratch/card-test-ds4-keep-read.sh:152-175); it was the shared P124r script's server-log expectation that failed.

### 4. INFO — the P124k diff needs nothing for this. `git show --stat 3637f9901` touches only REVIEW-RESULT.md, so
the code at HEAD is byte-identical to what round 2 verified (landing correctness, OFF-path, multi, locks/stdout,
file discipline — that text survives at `git show 3637f9901:REVIEW-RESULT.md`, its nits 1-4 and infos unchanged by
the card run). This worktree's serve/server.py is the base file — the whole range `git diff --name-only
856af24ab..HEAD` carries no `serve/`, and here `keeps_stopped_read` still defaults False (serve/server.py:1464)
with a READY parse that does not know `keepread` (:1516-1523); the gate itself is P124u on the candidate
(cand serve/server.py:1529-1530, the word-not-the-name rule). Fixes 1+2 therefore go to that line/branch; the
Ds4Read fake P124u step 3 added (candidate test_priority, "keeps the read ... no figure over the pipe") is the one
fix 1 must update.

## Commands run for this round (one at a time, the plan's module only)

- `g++ -std=c++17 -Wall -Wextra -O1 tests/ds4_marks_test.cpp -o scratch/ds4_marks_test_r3` + run →
  `ds4_marks_test: ok`, no warnings; the round-2 build-dir binary (`scratch/build-p124k/ds4_marks_test`, newer
  than every source) also runs ok. No CUDA rebuild needed: the sources have not moved since it was built. No GPU,
  no serve test run (the plan names only the C++ module; the affected serve modules belong to the fix, finding 1).
- Evidence reads at the candidate: `tools/card_tests/lost_read.sh` and `serve/server.py` from `saguaro/rc124-cand`
  (HEAD is an ancestor of that branch, so the manager's card exe is this code plus the gate) — copies left in
  `scratch/review_lost_read_cand.sh`, `scratch/review_server_cand.py` (scratch/ is ignored). Manager's extract:
  `scratch/card-r1-engine.txt`.
- Hygiene re-check: the diff's added lines carry no paths outside the repo and no secrets (the only hits are the
  script dropping `api_key` and the sanctioned `/opt/llm/saguaro/.venv/bin/python`); `serve/qfn_engine.py` and
  `engines/qfn/serve/server.py` untouched in the range → no extractor owed; the ten code/report commits and this
  review carry no Co-Authored-By/Claude-Session/Assisted-by lines.

Only REVIEW-RESULT.md was written; nothing else edited or committed.
