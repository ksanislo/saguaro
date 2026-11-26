# REVIEW - P124p API prices from OpenRouter, with a shipped fallback (round 2)

Verdict: **MERGE-WITH-NITS**

Reviewed `git diff eeae50d8f67052ff739ce79e963eff444b7b7489..HEAD` (11 commits, HEAD b1d1ab406) against
`PLAN-p124p-live-prices.md` (incl. Ken's 11-26 OFF-by-default change) and `REPORT-p124p-live-prices.md`.
This round re-checks the diff at HEAD and verifies the round-1 fixes; no code edited, nothing committed
but this file. Tests named by the plan/report, run one at a time, no network:
`serve.test_savings` -> **34 OK**; `serve.test_server` -> **179 OK**.

## The manager's card, checked at HEAD

- **Nothing leaves the box by default**: `live_enabled` defaults False (serve/savings.py:264-268), the server
  wires it at serve/server.py:10029, and the only `urlopen` calls are `PriceBook._fetch` (serve/savings.py:352,
  started only while `live`) and the manual release tool (tools/refresh_prices.py:35).
  `test_off_by_default_nothing_is_ever_fetched` (serve/test_savings.py:423) asserts urlopen is never touched.
- **OFF is inert**: a default start reads no live cache, logs nothing (serve/savings.py:301-302;
  test at serve/test_savings.py:556); the old GitHub fetch is fully gone (`url`/`beats_disk` removed, no
  callers left - grep clean).
- **Never blocks**: the fetch is a daemon thread (serve/savings.py:330), started from `__init__`/`refresh`/
  `set_live`, never joined by a request; 10 s timeout (:288); at most a day, an hour after a failure (:286-287).
- **Conversion exact**: `per_million` is Decimal-on-string (serve/savings.py:164-170). Verified against
  `ref/openrouter-models-sample.json`: 0.0000001405 -> 0.1405, MiniMax 0.075/0.0075/0.3, Hy4 0.035/0.0015/0.1405,
  Qwen 0.15/0.016/0.47, Inkling 0.075/0.015/0.3 - the shipped `data/api-prices.json` figures and the plan's six
  `openrouter` ids match the sample exactly; DeepSeek's peak 2.0 is the sample's own 100-400/600-1000 overrides.
- **Fallback shipped and prefilled at release**: data/api-prices.json (as_of 2026-11, ids per family) +
  tools/refresh_prices.py + its RELEASE-CHECKS.md row.
- **Easy to find**: the switch is the first row of Settings > Savings, under the source note
  (serve/web/settings.js:352-354); POST /api/savings/live is own-page-only, JSON-only, refused under a
  supervisor (serve/server.py:8112, 8177-8194).

## Round-1 findings, verified fixed

- **FIX 1 (peak on the live built-in-name path)**: `check_price` keeps a carried peak (serve/savings.py:94-101);
  `parse_openrouter` carries the family's factor (:212-213). Tested both ways:
  `test_the_sample_parses_and_maps_every_family` (peak 2.0 / GLM none) and
  `test_deepseeks_peak_survives_a_shipped_file_that_cannot_be_read` (real path: live on, shipped file unreadable,
  peak-hour request counted twice).
- **FIX 4 (env beats the switch at runtime)**: `env_live` (serve/savings.py:253-261), refusal in
  `Ledger.set_live` (:499-504) surfacing as the endpoint's 400 (serve/server.py:8190-8192). Tested at
  serve/test_savings.py:270-277 and :576-597 (refused request writes nothing and dials nothing).
- **Nit 2** read-once locals: serve/savings.py:377, 383, 394. **Nit 3** off start touches no cache: :301-302,
  :337-345. **Nit 6** docs sentence: docs/SETTINGS.md (the line keeps saying built-in until the first answer).
  **Nit 5** (no tools/test_refresh_prices.py) declined with a stated reason the plan supports - accepted;
  the tool's logic all lives in tested `savings` functions.

## Findings

**1 (Low). `PriceBook.start` lost the old claim-inside-the-lock, so two fetches can double.**
serve/savings.py:321-331: `stale()` is tested outside the lock and the re-check inside is only
`thread.is_alive()`. Two concurrent `report()`/`set_live` callers can both pass the outer check; if the first
fetch thread finishes (a fast failure - offline, no route) before the second takes the lock, the second spawns
a fresh fetch seconds later. Consequence: a duplicate GET and a duplicate "did not come" line while the user
has opted in; the day budget itself holds. The pre-diff code claimed inside the lock (`if now < self.next_try:
return`); re-checking `time.time() < self.next_try` inside the `with self.lock` restores it.

**2 (Low). A malformed list price can traceback instead of the one promised log line.**
serve/savings.py:218-221: `parse_openrouter` calls `file_prices` unguarded, and `check_price` raises
ValueError for a per-1M figure outside 0-1000 - a listed family priced, say, per-token "0.002" (or negative)
makes `_fetch` (serve/savings.py:357, outside its try) die with a thread traceback instead of the
"OpenRouter prices did not come" line. State stays safe (cache/prices below count, retry timer set by `start`),
so this is noise on a pathological answer only - wrap the build loop and return `(None, why)`.

**3 (Nit). README anchor is one dash short.** README.md:303 links
`docs/SETTINGS.md#live-api-prices-live_prices-saguaro_live_prices--off-by-default`, but GitHub slugifies the
heading (docs/SETTINGS.md, "…`) - off by default") to `…saguaro_live_prices---off-by-default` (the `) - off`
leaves a dash between two spaces, each becoming a dash) - as computed with the slugger rules. The link lands at
the page top.

**4 (Nit). The Page tests trust the ambient environment.** serve/test_savings.py:192-202 (`Page.setUp`) does
not neutralize `SAGUARO_LIVE_PRICES` the way `OpenRouterPrices.setUp` does (:344): with
`SAGUARO_LIVE_PRICES=0 python -m unittest serve.test_savings`, `test_the_live_switch_from_the_page` (:254)
fails - the switch it exercises is refused by fix 4.

**5 (Info). A corrupt live cache with a fresh mtime delays its refetch up to a day.**
serve/savings.py:308-313: `next_try` comes from the file's mtime even when `load_live_file` refused the file.
Harmless (the shipped table counts, matching the plan's fallback), but resetting `next_try` on a refused load
would refetch at the next page ask (still rate-limited by the claim).

**6 (Info). `Ledger.set_live` writes the run config outside `Ledger.lock`** (serve/savings.py:505-514): two
concurrent toggles share `<config>.tmp`; both writes are complete JSON and `os.replace` is atomic, so the
result is last-writer-wins - cosmetic only, and consistent with the file's other writers.

**7 (Info). Open question 1 stands as reported**: `SAGUARO_LIVE_PRICES` is not in `SERVER_ENV`
(serve/server.py:8705-8711), so `--env SAGUARO_LIVE_PRICES=1` in the config's `"env"` block does nothing while
the top-level `live_prices` key and the shell spelling work; docs/SETTINGS.md documents exactly the working
spellings. Ken's design call.

**8 (Info). `SAGUARO_SAVINGS_PATH=off` with live on**: no folder -> no cache file -> `start` never fetches
(serve/savings.py:324), and the page shows the switch on with the line still saying built-in prices.

## Checked and clean

- Locks: `PriceBook.lock` never nested (network outside it, :347-372); `Ledger.lock` never held while any
  `PriceBook` lock is taken (`refresh` runs before `report` takes it, :582-583); no path acquires a
  non-reentrant lock it already holds.
- Threads/stdout: one daemon thread per book, guarded by the `thread` slot; every log line one `print(flush=True)`;
  the fetch thread swaps `live_doc` wholesale and readers bind locals once (:377, :383, :394).
- Feature OFF, beyond the intentional changes (GitHub fetch gone, figures/ids, the new report fields the
  shipped settings.js reads): no unrelated code path changed; /metrics, /api/savings/prices, reset and the
  ledger file format are untouched by the diff.
- Generated file: `serve/qfn_engine.py` is not in the diff, nothing under `engines/` is, and it contains no
  `savings` reference - no regeneration owed, per the extract-tool rule.
- Tests really exercise the change: urlopen mocked everywhere, the test client bound to `REAL_OPEN`
  (serve/test_savings.py:35, 215), real background threads joined, cache/config files asserted on disk, the
  endpoint driven over real HTTP (403 from a foreign Origin, 400 on junk).
- No paths outside the repo (writes are beside the config/ledger or `--out`), no secrets; commit history clean
  of Co-Authored-By / Claude-Session / Assisted-by lines.
