# U2 - savings prices fetched from the web, the built-ins as the offline fallback

## What changed

- **Step 1 - the price file** (`dd53a02c0`): new `data/api-prices.json` -
  `{"as_of": "2026-10", "prices": {<model name>: {api, match, input, cached, output, (peak)}}}`, the four built-in
  entries moved over **without touching a number** (verified by hand: every entry equals its `savings.REFERENCES`
  row and `PriceBook.reference` equals `savings.reference` for the known names and answers None for the unknown
  ones).  In `serve/savings.py`, `parse_price_file` validates the text (`as_of` a `YYYY-MM` month, every entry
  through `check_price`, plus the optional `match` regex that says which model names an entry fits - default: the
  key itself), `load_price_file` reads a path - missing is quiet, broken is **ignored with one log line** - and the
  new `PriceBook` holds them: `Ledger.price` / `price_list` / `report` now take a model's default from the book,
  which falls back to the built-in `REFERENCES` for every model the file does not name.  The built-in dict stays
  untouched as the last-resort fallback.
- **Step 2 - the daily fetch** (`37e45a33b`), in the `PriceBook`: when the Savings page asks (`report()` calls
  `book.refresh()`, nothing else does) it fetches at most **once a day** the newest copy of that file from
  `https://raw.githubusercontent.com/<owner>/<name>/main/data/api-prices.json` - `owner/name` from
  `serve/update.py`'s `origin_repo()` - with a **8 s timeout**, `urllib` only, no new dependencies.  The copy is
  cached as `api-prices.json` in the ledger's own folder (its mtime is the clock across starts: an existing cache
  is read at startup, unwritable cache still counts this run).  It is used only **valid and newer `as_of` than the
  folder's own** file; the precedence is *page (Settings > Savings) > fetched > `data/api-prices.json` > built-ins*.
  A fetch that does not parse is ignored with one log line and does **not** overwrite a good cache; a network error
  is remembered in `book.error` and retried after an hour (`RETRY_AFTER`), a success waits a day (`FETCH_EVERY`).
  `SAGUARO_UPDATE_CHECK=0` turns the fetch off entirely (no outgoing request), as does a ledger kept with
  `SAGUARO_SAVINGS_PATH=off` (no folder to cache in).  `report()` now also answers `prices_source`:
  `{"from": "github" | "file" | "built-in", "as_of": "YYYY-MM"}`, and `prices_as_of` is that month, not the
  constant.
- **Step 3 - the page's line** (`d57454fa7`), `serve/web/settings.js`: the Savings view got its one line -
  *"Prices: the newest api-prices.json from GitHub, the APIs' own as of Nov 2026."* (or "the api-prices.json in
  this Saguaro" / "Saguaro's built-in prices").  The price cards now say where their default came from
  (`DeepSeek's prices: the newest api-prices.json from GitHub (as of Nov 2026)`) instead of the hardcoded
  "built in (Oct 2026)", and the fallback button reads "Use <where they come from>".  `sidebar.js` needed nothing:
  its Saved popover names the APIs, not a month.
- **Step 4 - tests** (`8a2c5a306`), `serve/test_savings.py`: a `PriceFiles` class with mocked `urllib` (nothing
  touches the network): `test_precedence_page_file_builtins`, `test_the_fetch_once_a_day_and_only_a_newer_file_wins`
  (URL shape, cache written, throttle - one call for two reports), `test_a_fetched_copy_that_is_not_newer_does_not_move_the_prices`,
  `test_yesterdays_cached_copy_is_used_at_the_start` (and an older cached copy is not),
  `test_the_fetch_is_off_when_the_user_said_no_requests`, `test_a_network_error_falls_back_to_the_file_and_the_builtins`
  (OSError and URLError, with and without a disk file), `test_a_bad_price_file_is_ignored_with_one_log_line`
  (not JSON, no as_of, an entry that is not a price, a broken match regex), `test_fetched_junk_is_ignored_with_one_log_line`.
  The existing `Counting` and `Page` classes run with `SAGUARO_UPDATE_CHECK=0` patched in (their reports would
  otherwise ask GitHub once a day).

## Files touched

- `data/api-prices.json` - new: the prices, as of 2026-10 (step 1).
- `serve/savings.py` - the file loader, `PriceBook`, the fetch/cache/precedence, `prices_source` (steps 1-2).
- `serve/web/settings.js` - the source/as_of line and the price cards' wording (step 3).
- `serve/test_savings.py` - the eight new tests and the no-network guard (step 4).
- `REPORT-u2-savings-prices.md` - this file.
- Not touched, per plan: no price number, `serve/qfn_engine.py` (and so no `extract_qfn_engine.py` run),
  the engine, `server.py` (its `Ledger(...)` call keeps its shape; the book comes with it).

## Tests

- `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_savings` - 22 tests, OK (14 before this branch,
  +8 from step 4).  No network: every GitHub answer is a mock, every file lives in a temp dir; the `url()` git
  call is local.
- `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_dashboard` - 55 tests, OK.
- `node --check serve/web/settings.js` - clean.
- Pre-existing noise, not from this change: `serve/telemetry.py` ResourceWarnings ("/proc/stat", "/proc/cpuinfo")
  in the `Page` tests.

## Open questions

- A file entry carries an optional `match` regex (default: the model-name key itself, case-insensitive substring)
  so fetched prices keep matching decorated names ("Hy4-Flash-Maya-GGUF" -> `hy4-flash`) exactly the way the
  built-in table does; the plan's sketch showed only `{input, cached, output, peak}`.  Publishing a new model
  without a `match` works for names that contain the key.
- `check_price` still decides "may this model have a peak factor" from the **built-in** table, so a brand-new
  peak-hour model added only to the fetched file cannot get its factor - deliberate (the plan's "only where they
  are its own"), but a future fetched peak model will need the built-in table to learn it too.
- On a non-github origin (or no `.git`) the fetch falls back to `ksanislo/saguaro`'s main, exactly like
  `update.py`'s release check does.
