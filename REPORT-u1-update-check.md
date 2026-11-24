# U1 - the About > Updates check on a repository with no releases

## What changed

- **Step 1 - 404 is a normal state** (`4bdc98f93`): in `serve/update.py`, `_fetch_latest` now catches
  `urllib.error.HTTPError` before the general `OSError` handler. A **404** from
  `releases/latest` (the repository has no releases yet - GitHub *was* reached) sets
  `no_releases=True`, clears any earlier error, leaves `latest` None, and schedules the next check
  after the **usual six hours** (`CHECK_EVERY`), so no 10-minute retry storm. `info()` carries a new
  `no_releases` field. Real failures keep the old wording: offline/proxy (`OSError`), the rate-limit
  **403** and **5xx** still say "GitHub could not be reached (...)" with the 10-minute retry
  (`RETRY_AFTER`); a non-version tag still raises through to the same error. In
  `serve/web/settings.js` the `UpdateCard` gets a branch before the error one: *"No releases yet -
  No Saguaro releases have been published yet; <version> is the newest code you pulled."* with a
  Check-again button; the old "Couldn't check for updates" card only appears on a real failure now.
- **Step 2 - the version the page names** (`40eb2d848`): the banner's version function
  (`saguaro.py:8870 saguaro_version`, which used `S.out` + `HERE`) moved **verbatim in logic** to a
  new root-level `saguaro_version.py` that both sides import (nothing is duplicated): a git clone is
  named by `git log -1 --format=%h %cs` - "a1b2c3d (2026-11-24)"; a folder without git by `v<VERSION>`;
  10 s timeout and a missing/failing git fall back to the VERSION text. `saguaro.py` imports the name
  (its banner prints exactly what it printed before), and `Updater.__init__` computes `self.label`
  the same way against the updater's own root; `info()` carries it as `current_label`.
  `settings.js` shows `u.current_label` where the card said `v${u.current}`: the running-version line
  of the error card, the no-releases line and "Up to date" now read e.g. *"Saguaro 4bdc98f9
  (2026-11-24) is running"* instead of the upstream Maya version "v1.1.50". The banner does not name
  a branch, so neither does the page (the plan's "and the branch if the banner does").
  `server.py` needs no change: it constructs `Updater(SAGUARO_VERSION)` and the updater takes the
  label from the shared helper itself; `current` stays the VERSION string, which `newer()` compares.
- **Step 3 - tests** (`fc769a785`): `serve/test_update.py`, all with mocked `urllib` (no network):
  - `test_404_means_no_releases_yet_not_an_error` - `HTTPError 404` → `error` None, `no_releases`
    True, `latest` None, not `newer`, and a second `check()` asks GitHub again nothing (six hours).
  - `test_rate_limit_and_server_errors_keep_the_error_wording` - 403 and 500 → the error wording,
    `no_releases` False.
  - `test_a_newer_tag_offers_the_update` - a mocked release `v1.0.18` on a `1.0.17` install →
    `newer` True, the version/tag/summary read off the answer.
  - `test_the_page_names_the_clone_by_its_commit` (in the throwaway-clone class) and
    `test_the_version_named_without_git_is_the_version_line` - `current_label` is
    "hash (yyyy-mm-dd)" for a clone, "v1.0.17" without git.
  - the existing unreachable-GitHub test also asserts `no_releases` stays False.

## Files touched

- `serve/update.py` - the 404 branch, `no_releases`, `current_label` (steps 1, 2).
- `serve/web/settings.js` - the no-releases card and the three lines that name the running version.
- `saguaro_version.py` - new: the shared version-text helper (step 2).
- `saguaro.py` - `saguaro_version` deleted, the helper imported; banner output unchanged.
- `serve/test_update.py` - the five new/extended tests (step 3).
- `REPORT-u1-update-check.md` - this file.
- Not touched, per plan: the update (git fast-forward) path in `update.py` (`start`/`_update`),
  `serve/qfn_engine.py`, the engine's INFO/health version fields.

## Tests

- `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_update` - 15 tests, OK (10 before this
  branch, +5 from step 3). No network: every GitHub answer in them is a mock; the git ones run
  against a throwaway clone in a temp dir.
- `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_dashboard` - 55 tests, OK (run after
  step 2's import change and again at the end).
- Smoke-checked by hand: `import saguaro` still works, its banner text equals the new
  `Updater(...).label` for this checkout ("4bdc98f93 (2026-11-24)"), and a root without `.git` gives
  "v1.1.50".

## Open questions

- The About page's "Version" fact row (settings.js `facts()`, from the engine's INFO
  `saguaro_version`/health `version`, still the VERSION line) was left as it is: the plan's version
  item read as the Updates card's running-version text, and changing what the engine's INFO reports
  looked like a wider change. Say the word if that row should name the commit too.
- The no-releases card's title/words are mine within the plan's "(or similar plain words)":
  *"No releases yet - No Saguaro releases have been published yet; <version> is the newest code you
  pulled."*
- `saguaro_version.py` lives at the repo root (both `saguaro.py` and `serve/server.py` put the root
  on `sys.path`; `tools/` is not on it for every importer of `serve.update`). A `tools/` home would
  need a path dance in `serve/update.py`.
