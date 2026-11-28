# REPORT-p124t2-card-fail-fast - Saguaro 1.2.4, tools only

Plan: PLAN-p124t2-card-fail-fast.md - the card-test tools wait ~20 min for readiness even when the engine
dies at load; stop at once when /health reports the failed load (or the server process exits), and cap the
log-line waits at ~90 s. Tools only; no engine, no serve/qfn_engine.py touched.

## What changed

The failed-load state, read from serve/server.py: the "starting" service answers on the port from the first
second of every start; a failed load calls LoadState.fail (serve/server.py:10046) and GET /health then
answers 503 with LoadState.health()'s fields - `{"status": "failed", "model": ..., "elapsed_s": ...,
"line": ..., "supervisor": ..., "error": "the engine exited before it was ready ..."}` (serve/server.py:8928,
8949, 9117). While the load still runs the same endpoint answers 503 `{"status": "loading", ...}`; only the
real Service (take_over) answers 200. With a direct start (no supervisor, no --exit-on-fail) the process
stands at the error page (ends_on_failed_load, serve/server.py:8992) - so the old `curl -f /health` polls
saw a failing 503 exactly like a loading one, and waited the whole 240 x 5 s.

- **tools/card_tests/health.sh** (new): the shared readiness probe. `health_status <port>` prints one of
  `ready` (200), `failed` (503 body with "status": "failed"), `loading` (anything else, including no answer
  on the port - a dead server is caught by the loop's own kill -0, never invented here). Never fails under
  `set -eu`; sourced by every card-test script.
- **tools/card_tests/park.sh, lost_read.sh, mortar_pressure.sh, glm_pinned_start.sh**: the readiness loops
  now poll health_status. `failed` -> exit 1 at once printing the server log's last 20 lines (park.sh) or
  the script's existing fail tail (the other three: 40 lines of serve.log + the engine log, which carries
  the 20). A `ready` check replaces the old post-loop curl; the existing kill -0 server-died checks stay
  (park.sh now prints the log's last 20 lines there too, per the plan). park.sh's readiness signal moves
  from /metrics to /health - the same moment (the Service takes the port over); both answered the loading
  and failed states with the same 503, which is exactly what hid the failure. Everything else identical.
- **tools/card_tests/lost_read.sh** (log-line phases): the two phases that wait for a log line -
  'prompt read paused' (was 180 s) and 'prompt read continued' (was 600 s) - now give up at ~90 s.
- **tools/test_card_tools.py** (new): the plan's tiny test - health_status, run under bash with `set -eu`
  (as park.sh sources it), against a fake /health on 127.0.0.1 answering the failed-load body (seen at
  once), a loading body (still loading - even one whose "line" merely mentions a failed something), a
  closed port (loading, not failed), and a 200 (ready); plus a bash -n gate over health.sh and every
  tools/card_tests script.

## Files

- tools/card_tests/health.sh (new)
- tools/card_tests/park.sh, tools/card_tests/lost_read.sh, tools/card_tests/mortar_pressure.sh,
  tools/card_tests/glm_pinned_start.sh (modified)
- tools/test_card_tools.py (new)

Commits: 258b53cbd (health.sh), 6a64e80ad (the four scripts: fail-fast readiness + ~90 s log-line waits),
c79896018 (tools/test_card_tools.py), plus this report.

## Tests run

- `bash -n` on all five scripts (health.sh + the four card tests): clean.
- The plan's test, run only it: `/opt/llm/saguaro/.venv/bin/python tools/test_card_tools.py` ->
  **7 tests, OK** (2.1 s). No GPU, no card, no model, nothing destructive.
- No card test was executed end to end: this box has no card and no model weights, as in the plans that
  added these scripts.

## Open questions

1. **Two scripts the plan names are not on this branch.** tools/card_tests/layer_pause.sh and
   tools/card_tests/ik2_kv_start.sh were added on the p125j / p125e candidates (64ee73f9e, 6be4583c2) and
   live in saguaro/rc124-cand2, but this branch forked from c1f936711 before those merges - the files do
   not exist here, and I did not import feature-branch code into a tools-only change. When those merges
   land, they need the same treatment: source tools/card_tests/health.sh, swap the curl readiness loop for
   the health_status case block (as in the four scripts here), and cap their log-line waits
   ('prompt read paused' and friends) at ~90 s. The plan's 600 s example for 'prompt read paused' reads
   like it came from layer_pause.sh; on this branch the 600 s wait was lost_read.sh's 'prompt read
   continued' (its 180 s 'prompt read paused' loop was brought to 90 s as well).
2. The plan quotes the launcher line as "[saguaro] the model failed to load: ..."; this branch's
   serve/server.py prints exactly that ([saguaro], serve/server.py:10047), while engines/qfn's own
   launcher lines are still "[strata]". Nothing to change here - the readiness signal is the /health JSON,
   not the prefix.
3. Whether the 180 -> 90 s cap also belongs to the paused-wait here is my reading of "a phase that waits
   for a log line ... gives up after ~90 s"; only 600 s is explicitly named. Easy to revert if the intent
   was the 600 s loop alone.
4. The card tests' verdict on a failed load now lands in seconds; they still exit non-zero through their
   normal fail()/exit paths, so whatever wrapped them sees the same failure shape (exit 1), just earlier.
