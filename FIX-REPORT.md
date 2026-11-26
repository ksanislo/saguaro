# FIX-REPORT - P124p review findings (REVIEW-RESULT.md), one commit each

Default OFF = no network, unchanged throughout: `serve.test_savings` 34 OK and `serve.test_server` 179 OK after the
last commit, no network, run one module at a time; `test_off_by_default_nothing_is_ever_fetched` still asserts a
default install dials nothing, ever. `serve/qfn_engine.py` untouched and not needed (nothing in these diffs is in
`engines/qfn/serve/server.py`).

## FIX-FIRST

**1. DeepSeek's peak factor dropped on the live built-in-name path.** FIXED - `59060122c`.
`check_price` (serve/savings.py:93-101) keeps the peak factor of any entry that already carries one (validated 1-10
as before); the base table's factor is still the default only for names its regex fits, so the live doc built by
`parse_openrouter` no longer loses `peak: 2.0` when a family is keyed by the built-in api name `"DeepSeek"` (the
shipped file unreadable). The price counted is now the one the shipped table gives. One consequence, deliberate:
the page's prices endpoint accepts a `peak` for a model the built-ins do not flag as peaked - the review's suggested
fix, and the same validated field. Tests: `test_the_sample_parses_and_maps_every_family` now asserts
`doc["prices"]["DeepSeek"]["peak"] == 2.0` (and GLM's stays none), and the new
`test_deepseeks_peak_survives_a_shipped_file_that_cannot_be_read` runs the real path (live ON, the shipped file
unreadable) and counts a peak-hour DeepSeek request at twice. Verified against the old code: the built-in-name doc
carries `peak: None` before, `2.0` after.

**4. `SAGUARO_LIVE_PRICES=0` must win at runtime too.** FIXED - `328bfa454`.
New `env_live()` (serve/savings.py) reads the environment's say (1/on/true/yes, 0/off/false/no, or nothing);
`live_enabled` uses it, so the start is unchanged; `Ledger.set_live` refuses a toggle that contradicts an explicit
env value - the endpoint answers 400 with the reason (the page's error toast shows it), the run config is not
written, and no fetch starts. Symmetric on purpose: `=1` cannot be switched off from the page either; both are
"said, not attempted". docs/SETTINGS.md says it. Tests: `test_the_environment_beats_the_switch_while_running` (env
0 beats an in-page yes, env 1 beats an in-page no, a refused request writes nothing and dials nothing, an agreeing
spell works), the `env_live` precedence assertions in `test_the_setting_and_its_priority`, and
`test_the_live_switch_refused_while_the_environment_says_no` through the real endpoint (400, reason, still off).

## Nits

**2. Double read of `self.live` / `self.live_doc`.** FIXED - `f46fc1a56`. `source()`, `as_of()` and `reference()`
bind both to locals once before the checks.

**3. OFF path touched the live cache at startup.** FIXED - `e00ee1db2`. `PriceBook.read_live_cache()` (the cache
read, its log lines and its age) runs only while live is on - at the start, and when `set_live` turns it on, so a
cache from an earlier on spell counts again at once and its age decides whether a refetch is due. A corrupt
leftover no longer prints on a default start. Test: `test_the_off_start_touches_nothing_of_the_live_cache`.

**4. also a docs ask.** Covered by fix 4's sentence in docs/SETTINGS.md (the env beats the switch at runtime, the
refusal is said, nothing saved).

**5. No `tools/test_refresh_prices.py`.** NOT DONE. `main()`'s arg handling, the diff and the output format are the
thin glue the review itself scores as the whole risk, and every logic path it takes (`parse_openrouter`,
`per_million`, `price_families`, `parse_price_file`) is tested through `serve/test_savings.py`; a new module this
branch must not run (the instruction is to run only the plan's modules) would be shipped unexercised, which is
worse than none. It belongs in a release-steps branch that runs it.

**6. Switch can momentarily disagree with the source note.** FIXED (docs) - `4915844ef`. The behavior is by design
(report never waits for the network, and waiting would break the step-3 promise), so docs/SETTINGS.md now says the
line keeps saying "built-in prices" until the first fetch's answer lands.

## Notes (no action, as the review itself says)

The run-config `"env"` block does not carry `SAGUARO_LIVE_PRICES` to the spawned server, and the free-vs-`-` mapping
plus the unparsed hourly `overrides` stay as mapped - all three are deliberate choices, unchanged.
