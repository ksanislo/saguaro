# FIX-REPORT: P2 review findings (REVIEW-RESULT.md) - finding by finding

All fixes are page-only JS plus serve/test_dashboard.py; serve/server.py and serve/qfn_engine.py untouched (nothing was owed
from extract_qfn_engine.py). The two named modules run one at a time after every step: serve.test_dashboard (now 68 OK, the
five new regressions in; node present, PageLogic not skipped) and serve.test_new_engines_ui (35 OK). The review's own repros
pass: scratch/gate_hidden_open.mjs now shows `health, metrics` and `supervisor: llama-swap` after the tab is shown, and
scratch/health_retry_lost.mjs 2 tries / loaded true (its stub needed the reviewer's listener-capture trick to fire
visibilitychange - the stubs' `addEventListener: noop` gap that finding 9 names; scratch/ is gitignored, so only the fix
report itself notes that edit).

1. High - a page first opened in a hidden tab never loads /health. FIXED (serve/web/server.js, commit 8afb9b06).
   loadHealth's hidden branch now sets `healthWait = true` before returning, as the review prescribes; `unloaded()` stays a
   plain return right after it, so a resume into a closed gate fetches nothing (poll's swap-back detection re-arms health).
   The visibilitychange resume (already calling loadHealth when healthWait is set) needs no change. Regression test
   `test_a_tab_opened_hidden_loads_health_when_it_is_seen`: hidden at open asks nothing; shown, health goes first, the
   supervisor becomes known and metrics resume.

2. Medium - a health retry whose 2 s timer fires while hidden kills the chain. FIXED by the same change (the fired timer
   now lands in the parking branch instead of a silent return). Regression test
   `test_a_health_retry_landing_in_a_hidden_tab_is_made_up_when_seen`: the dark asks nothing (1 try), seen again the missed
   retry runs (2 tries, facts loaded).

3. Low - a transient /running error un-gates for one cycle and re-loads the model. FIXED (serve/web/server.js, commit
   b9c72083): swapRunning(swap, was) keeps `was` on anything but the plan's own namesakes - 404 and a refused key (401, and
   403 as its sibling) still answer `null` = poll as today; any other status, a network throw or unparseable JSON is a blip
   and keeps the last known state (a not-yet-known gate stays unknown, i.e. polls as today - the old behaviour, nothing
   stricter). Regression test `test_a_blip_on_running_keeps_the_gate_as_it_was` covers both halves.

4. Low - the answer-reconnect loop ignores the gate and visibility. FIXED (serve/web/chats.js, commit 5a95efed):
   sleepOrVisible no longer resolves on its plain timeout while hidden - a wait landing in the dark waits for the tab to be
   seen again, so the loop's repeating request asks nothing while nobody looks; its 10 min cap applies from the resume.
   The first attempt at a break stays immediate (that is the screen-sleep case the resume exists for, and behind llama-swap
   a swapped-out model's 404 ends the loop at that attempt, as finding 4 itself notes). Not covered by a PageLogic test:
   sleepOrVisible/follow are closures inside answer(), not exported, and driving answer() under the stubs is not the small
   test this is; the module-load smoke (scratch/chats_import.mjs) proves chats.js still parses and loads.

5. Low - watchReload/watchUpdate loop while hidden and while gated. NOT CHANGED, as agreed: the review accepts it
   ("self-declared in the report, user-initiated, transient, and llama-swap-managed servers refuse those POSTs outright").
   It stays exactly the deviation REPORT-p2-llamaswap-polling.md open question 3 describes.

6. Low - the loadMcp "servers starting" retry chain dropped by a hidden tab. FIXED (serve/web/server.js, commit
   55be4154): same parking pattern as health - the hidden branch sets `mcpWait`, the visibilitychange resume makes the
   chain up (and a boot on a hidden tab now gets its mcp when first seen, instead of waiting for a Monitor visit + 10 s
   tick). Regression test `test_a_mcp_retry_landing_in_a_hidden_tab_is_made_up_when_seen`.

7. Low - api/tokens bypasses the gate; browsing chats loads the model. FIXED (serve/web/chats.js, commit 1d2b5719):
   updateCtxMeter while unloaded() keeps the local estimateTokens figure (the older-server fallback) and asks nothing; a
   pending 400 ms debounce re-checks the gate when it fires; once /running says ready (health re-loads on swap-back, and
   app.js's health hook re-arms the meter) the real count is asked once. A chat already counted before the swap keeps its
   last real figure without churn or re-ask. Regression test
   `test_browsing_a_chat_behind_llama_swap_asks_no_token_count` (the first PageLogic case to import chats.js).

8. Nit - the re-indented "key"/"Waiting" continuation lines in monitor.js. FIXED (commit 4c9f5c164): the two pre-existing
   lines are back at column 41; `git diff c3844c07d -- serve/web/monitor.js` is now the single added unloaded branch.

9. Nit - the visibilitychange handler was unreachable in the harness. FIXED in its main part (commit 8afb9b06): the
   PAGE_STUBS document now records listeners (with removeEventListener) and exposes `fire(k)`, so the handler findings 1-2
   live in is executed by the three new hidden/retry tests. Not added: the 5 s gate cadence and the mcpTick scheduling -
   both need real-timer sleeps of 5-10 s per case for a cadence a code read already pins; left uncovered.

Also re-checked after all fixes: feature-OFF paths untouched (the 404/401 fallback, key/down handling and the 1 s cadence
tests still pass); sending still loads the model (nothing gated on S.conn in the send path); no Python, no generated-file
work owed. Commits, oldest first: 8afb9b06b, b9c720834, 55be41541, 5a95efed3, 1d2b57199, 4c9f5c164, this report.
