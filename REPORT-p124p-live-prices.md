# REPORT - P124p: API prices from OpenRouter, with a shipped fallback (Saguaro 1.2.4)

Plan: PLAN-p124p-live-prices.md, including Ken's 11-26 change: **live_prices is OFF by default** ("nothing leaves
this box"), easy to find in Settings, SAGUARO_LIVE_PRICES=1 turns it on. Each numbered step committed separately
(787044d1f step 2, 7e3c77a9d step 3, ad205aee5 step 4, 4e6670e5d step 5; step 1 read-only, step 6's tests were
written with the code they test and needed no separate commit).

## Step 1 - how prices matched and the Savings line was built (before)

- `serve/savings.py` `reference()` scanned the built-in `REFERENCES` (per-1M input/cached/output + DeepSeek's peak
  factor); `PriceBook` layered on top: a GitHub-fetched `api-prices.json` (once a day, synchronously, when
  `/api/savings` was asked, gated by `SAGUARO_UPDATE_CHECK=0`) > shipped `data/api-prices.json` (each entry checked
  by `check_price`, matched by its `match` regex) > built-ins. `Ledger.price()` put the page's own prices above all;
  `cost()` priced the token bands; the sidebar line came from `GET /metrics` (`summary()`), the Settings page from
  `GET /api/savings` (`report()`, with `prices_source`/`prices_as_of`).
- Only deepseek/glm/minimax/hy4 matched anything: Qwen3.8-Flash-Next (the box's own model, every quant) and
  Inkling-2-Mini had no price at all.

## What changed

**Step 2 - the shipped fallback (commit 787044d1f)**
- `data/api-prices.json`: `as_of` 2026-11; every family carries an `"openrouter"` id; figures refreshed to the
  sample's, converted exactly from per-token strings (MiniMax 0.075/0.0075/0.3, Hy4 0.035/0.0015/0.1405); new
  entries `qwen3.8-flash` (0.15/0.016/0.47, matches Qwen3.8-Flash-Next of every quant) and `inkling-2-mini`
  (0.075/0.015/0.3).
- `serve/savings.py`: the built-in `REFERENCES` carry the same ids and figures (same six families), and
  `check_price`/entries now carry `openrouter` through validation.

**Step 3 - the live fetch (commit 7e3c77a9d)**
- `serve/savings.py`: the GitHub price-file fetch is gone. New: `per_million()` (exact Decimal conversion of the
  per-token strings), `price_families()` (file entries override built-ins, keyed by OpenRouter id),
  `parse_openrouter()` (families' ids -> per-1M prices, DeepSeek's peak factor kept), `load_live_file()` (cache
  with its fetch time), `live_enabled()` (env beats config, off by default), `release_name()`.
- `PriceBook`: fetches `https://openrouter.ai/api/v1/models` (no key, 10 s timeout) on a **background thread**, at
  most once a day and at a start whose cache (beside the ledger, `openrouter-prices.json`, with `fetched`) is a day
  old; `refresh()` from `report()` never blocks. Any failure keeps the cache or the shipped table, logs one line,
  retries in an hour. `source()` is now `{"from": "openrouter", "date": ...}` or `{"from": "built-in", "as_of": ...,
  "release": <VERSION>}`; `reference()` consults the live doc > shipped file > built-ins (live off ignores the
  cache entirely).
- `serve/server.py`: the Ledger is created with `config=<run config>` and `live=savings.live_enabled(cfg.get
  ("live_prices"))`; new `POST /api/savings/live {"live": bool}` (own page only, refused under a supervisor) saves
  the switch in the run config and starts/stops the daily path; `report()` carries `"live"`.

**Step 4 - the setting (commit ad205aee5)**
- `serve/web/settings.js`: a switch right at the top of Settings > Savings - "Update API prices once a day from
  OpenRouter", its sub-line says it sends no data about you and that off means nothing leaves the box; the source
  note reads "Prices: OpenRouter, <date>." or "Prices: built-in prices, Saguaro <release> ...".
- `docs/SETTINGS.md`: new section "Live API prices (`live_prices`, `SAGUARO_LIVE_PRICES`) - off by default": one
  paragraph, what is fetched (the public model list only, nothing about the user), how often (a day, and at a start
  with an older cache), how to turn it off (switch off = default; `SAGUARO_LIVE_PRICES=0` beats the config).
- README's Saved bullet updated (Qwen's and Thinking Machines' APIs, prices ride offline, the opt-in refresh).

**Step 5 - the release step (commit 4e6670e5d)**
- `tools/refresh_prices.py`: GETs the public list once (10 s), rewrites input/cached/output of every family the
  shipped file names by its `openrouter` id (match/api/peak/ids stay), moves `as_of` to the fetch's month;
  `--check`, `--in FILE`, `--out FILE`, `--timeout`; a dropped family keeps its figures and says so, a list holding
  none of ours exits 1 with the file untouched. `docs/RELEASE-CHECKS.md`: one row in the tools table.

**Step 6 - tests (committed with the steps above)**
- `serve/test_savings.py` rewritten around the new flow, fed only by `ref/openrouter-models-sample.json`
  (urllib mocked throughout; the test client binds the real opener so patched `urlopen` cannot intercept it):
  per-token parsing exact (`0.0000001405` -> 0.1405, bad strings refused), the six-family mapping end to end (ids,
  figures, DeepSeek peak kept, file names vs built-in names), cache age (fresh cache: no fetch at the start; day
  old: fetched), failure fallback (network error keeps the cache, or the shipped table with no cache; junk answer
  logged once and not cached), the setting (off by default -> nothing ever fetched; `live_enabled` precedence; the
  switch saves `live_prices` in the run config; off ignores a fresh cache), the shipped-file layer (precedence,
  bad file one log line), and the `POST /api/savings/live` endpoint through the mock-engine server (403 from other
  origins, 400 on junk, the sample answer faked).

## Files changed

- `data/api-prices.json` - fallback refreshed to OpenRouter's 2026-11 figures + `openrouter` ids + two families
- `serve/savings.py` - OpenRouter parsing/families/cache/background fetch/source; `Ledger` gains `config`/`live`/
  `set_live`; `report()` gains `live`; the GitHub fetch is gone
- `serve/server.py` - Ledger wiring; `POST /api/savings/live` (+ supervisor guard)
- `serve/web/settings.js` - the switch, the two source phrasings
- `docs/SETTINGS.md`, `docs/RELEASE-CHECKS.md`, `README.md` - the paragraph, the table line, the Saved bullet
- `tools/refresh_prices.py` - new (release step)
- `serve/test_savings.py` - the suites above

## Tests run (only the modules touched; one at a time; no network)

- `python -m unittest serve.test_savings` -> **30 tests OK** (run after each step; final verbose run after step 6)
- `python -m unittest serve.test_server` -> **179 tests OK** (after step 4, for the `serve/server.py` changes)
- `node --input-type=module --check serve/web/settings.js` -> syntax OK
- `tools/refresh_prices.py` verified offline against the sample: unchanged file reports "no changes"; a file with
  the October figures gets exactly `0.15/0.015/0.6 -> 0.075/0.0075/0.3` and `0.07/0.003/0.281 ->
  0.035/0.0015/0.1405` and `as_of 2026-10 -> 2026-11`, and rewriting from the sample yields the shipped file's
  figures again; a bad `--in` exits 1.

## Open questions

1. **`live_prices` in the config's `"env"` block does nothing** - the server process is spawned with the shell's
   environment, not the config's `"env"` (that goes to the engine), so the documented spellings are the Settings
   switch (writes `"live_prices": true` at the config's top level), `SAGUARO_LIVE_PRICES` in the shell before the
   start, or hand-editing that top-level key. If `--env SAGUARO_LIVE_PRICES=1` should work too, `saguaro.py`'s
   server spawn would have to pass selected config-env keys through.
2. **Free and vanished families**: an OpenRouter entry priced "free" or `-` (unparseable per-token strings) is
   treated like a family that is not listed at all - it keeps the price below it rather than becoming $0. Ken may
   want an explicit listing to zero a price instead.
3. **`"peak"` (DeepSeek's 2x hours) stays hand-knowledge** - the list carries hourly `overrides`, but the plan's
   mapping asked for the three per-token strings only; the override table is not parsed anywhere (the server's own
   peak-hour rule still decides which band tokens go to).
4. **No `tools/test_refresh_prices.py`** was added (the step-6 test list did not name the tool, and its logic lives
   in `savings.parse_openrouter`, which is tested); the tool itself was exercised offline as above. The repo's
   habit of a `tools/test_*.py` per tool would be a cheap addition if Ken wants one.
5. **`as_of` vs fetch date in the cache** - a hand-edited cache could claim any `as_of`; the date shown is taken
   from the fetch time when absent or malformed. Harmless (it only words the Savings line).
