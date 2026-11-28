# REVIEW-RESULT - P124t2 card-test fail-fast (diff c1f936711..0e35e711b, 4 commits)

MERGE-WITH-NITS

Scope verified: `git diff c1f936711..HEAD` touches exactly the 7 files the report names (4 card-test
scripts + new health.sh + new test + this-plan report). `serve/`, `engines/`, `saguaro.py` are untouched
(empty diff stat), so serve/qfn_engine.py (generated) is untouched and no engine behaviour changes —
feature-OFF paths byte-identical. Commits 258b53cbd / 6a64e80ad / c79896018 / 0e35e711b, clean tree,
no Co-Authored-By / Claude-Session / Assisted-by trailers. No secrets, no new paths outside the repo
(health.sh and the test use only 127.0.0.1 and `__file__`-relative paths). Locks/threads/stdout: shell-only
change, one captured stdout word per probe call — nothing interleaves; no engine locks are in play.

## What checks out

- Server contract as the report quotes it: failed load -> `LoadState.fail` (serve/server.py:8928) called at
  serve/server.py:10046, launcher prints "[saguaro] the model failed to load:" (serve/server.py:10047),
  direct start stands at the page (serve/server.py:8992); the starting handler answers `/health` 503 with
  `LoadState.health()` (serve/server.py:9117-9121, 8949-8963) whose first key is `status`; the running
  Service answers `/health` 200 "status": "ok|updating|reloading" (serve/server.py:8058-8073); during the
  load `/metrics` gets the same plain 503 (`_loading`, serve/server.py:9062-9063), so park.sh's old
  `/metrics` poll could not tell loading from failed — both flip at `take_over` (serve/server.py:9157),
  the "same moment" claim holds.
- tools/card_tests/health.sh:23 deliberately uses `curl` WITHOUT `-f` (the obvious way to break this:
  `-f` withholds the 503 body, and the failed-detection needs it) and splits `-w '\n%{http_code}'` with
  `${out##*$'\n'}` / `${out%$'\n'*}` (health.sh:24-25) — safe on bodies with newlines; any curl failure
  (closed port) is `loading` (health.sh:23), so a dead server is still the loop's `kill -0`, never an
  invented /health state. Always exit 0 — `set -eu` (park.sh:25, glm_pinned_start.sh:35) and
  `set -u -o pipefail` (lost_read.sh:24, mortar_pressure.sh:34) safe; the test proves it (probe() asserts
  returncode 0 under `set -eu`, tools/test_card_tools.py:44-49).
- The four readiness loops stop at once on `failed` (park.sh:111, lost_read.sh:107,
  mortar_pressure.sh:119, glm_pinned_start.sh:133) and the `ready` post-loop checks keep the original
  timeout messages verbatim (park.sh:116, lost_read.sh:112, mortar_pressure.sh:124,
  glm_pinned_start.sh:138); the pre-existing `kill -0` server-died checks stay (park.sh:113,
  lost_read.sh:109, mortar_pressure.sh:121, glm_pinned_start.sh:135). park.sh prints the server log's
  last 20 lines as the plan asks (park.sh:111); the other three print their own existing fail tail
  (40 lines of serve.log, which carries the last 20, + the engine log) — their own messages, per the
  manager rule.
- lost_read.sh log-line waits 180 s -> 90 s and 600 s -> 90 s (lost_read.sh:186, 191), the plan's "~90 s,
  not 600 s".
- Tests really exercise the change: tools/test_card_tools.py answers health_status with the exact 503
  bodies serve/server.py's `_json` produces (default separators -> `"status": "failed"` with the space,
  and the regex at health.sh:28 is whitespace-tolerant, also matching the compact form), covers
  failed / loading (incl. a loading body whose engine `line` merely mentions "failed") / closed port /
  200, plus a bash -n gate over every tools/card_tests script.

## Findings (by severity)

1. NIT — false-failed possible from the `line` field: health.sh:28 greps the WHOLE body for
   `"status":"failed"`, not the status field. LoadState.health() puts `status` first and the engine's last
   log line in `line` (serve/server.py:8955-8956), so a loading answer whose engine line literally
   contains `"status": "failed"` (e.g. an engine that logs a /health reply) would stop a healthy run.
   Anchoring (e.g. `^{"status"[[:space:]]*:[[:space:]]*"failed"`, valid because health() always writes
   status first) removes the shape. Probability low, and a false positive fails safe (aborts, prints the
   log) — not a blocker.
2. NIT — scope vs plan: tools/card_tests/layer_pause.sh and ik2_kv_start.sh (named by the plan) do not
   exist on this branch (verified absent); they ride the p125j/p125e merges. The coder's choice not to
   import feature-branch code into a tools-only change is right and disclosed (report open question 1),
   but the follow-up — same health_status block + ~90 s caps when those merges land — must be tracked or
   the plan's "in every tool" stays unmet there.
3. NIT — interpretation, disclosed: the plan names ~90 s for "a phase that waits for a log line (e.g.
   'prompt read paused')" and 600 s explicitly; the coder also capped lost_read.sh's 180 s paused-wait at
   90 s (lost_read.sh:186). Defensible reading (report open question 3), but on a slow card a legitimate
   pause line slower than 90 s now fails a run that previously passed; one-line revert if the manager
   meant the 600 s loop alone.
4. NIT — message drift, plan-driven: park.sh's server-died line changed from "the server died; see $SRV"
   (c1f9367) to "the server died while loading; the last 20 lines of $SRV:" (park.sh:113) to satisfy the
   plan's print-the-tail ask. Every other tool's own messages are byte-preserved. Flagged only against
   the manager's "keep each tool's own messages" line.
5. INFO — the test gates the probe function and bash -n, not the loop wiring (the four case blocks' exit
   path or the printed tails). That matches the plan's "tiny test ... runs the readiness function against
   a fake /health", and the wiring is a 4-line identical block per script; a fake-HTTP end-to-end run of
   one script's loop would close it.
6. INFO — poll bound drift: the probe uses `-m 5` (health.sh:23) where the old curls used `-m 2`
   (lost_read/mortar/glm), so the worst-case 240-poll window is ~40 min instead of ~28 min with a wedged
   (accepting, silent) server. Cosmetic.

## What I ran (one at a time, plan's test only)

- `bash -n` on all five tools/card_tests scripts: clean.
- `/opt/llm/saguaro/.venv/bin/python tools/test_card_tools.py`: **7 tests, OK** (2.1 s). Consistent with
  the report; nothing else in the plan's test list exists, no card test executed (no card here).
- Git hygiene only otherwise (log/diff/ls-files/status); serve/, engines/ diff verified empty.
