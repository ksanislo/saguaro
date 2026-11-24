# T6b report: QFN's stats on the Monitor and /metrics (server + web; no engine C++)

Task 6b of the QFN-first project. The engine-side lines (T6a: `mtp=` in INFO, `promo=` in STAT) were already
there; this task traced what the shared server (`serve/server.py`) and the Monitor page (`serve/web/monitor.js`)
do with QFN's stats, closed the gaps, put the session folder's state into `/metrics`, and tested it. No engine C++
(`engines/qfn/src`), no GLM/DS4/MM3 behaviour, no new tiles. `serve/qfn_engine.py` was only ever changed by
re-running `tools/saguaro/extract_qfn_engine.py` after editing `engines/qfn/serve/server.py`.

## Step 1 - what the Monitor shows today (answer in ./scratch/t6b-gaps.txt; one line each here)

| item | today | why |
|---|---|---|
| Speculation mode (MTP) | SHOWN | INFO `mtp=` (T6a) -> driver's generic INFO parse -> `speculation_modes()` reads `mtp` (serve/server.py:2648) -> `m.engine.speculation` -> the tile (speculationView) |
| RAM tier tiles (qfn set, incl. promo) | SHOWN | INFO `engine_kind=qfn` -> `tierSet()` qfn tiles; STAT parsed generically into `engine.stat`/`stat_history`, `metrics()` publishes `m.tiers` with `promo` among the history keys (serve/server.py:4398-4404) |
| Context RAM (`info["context_ram"]`, HMEM) | MISSING from the page | the driver parses it and `metrics()` spreads `info` into `m.engine`, but no `monitor.js` code read `context_ram` (only QFN's own fork UI did) |
| Slot memory (`info["slot_memory"]`, KMEM) | MISSING from the page | same chain, same missing reader |
| Last turn saves (`engine.turn_saves`) | MISSING from the page | serve/server.py:4329 already puts it in `m.context.turn_saves`; the Monitor never read it |
| Parked conversations | PARTLY SHOWN | the parked BYTES (VRAM lines' `parked=`) render as "Parked in RAM"; the COUNT (`parked_n` -> `m.context.parked.conversations`, serve/server.py:4294,4325) was never shown |

`scratch/` is gitignored, so the step-1 file itself is not committed (no files changed in that step).

## Steps 2 + 3 - what changed

- **`serve/web/monitor.js`** (step 2, commits `ca031e212` + `3be5403db`): the Context card (the existing tile - no
  new tile) gained
  - a pure exported `contextFacts(eng, C)` (the `chatsModel`/`tierSet` precedent: "a page-logic test can run it")
    rendering the three missing facts from the fields the server already passes through:
    `X GB of RAM pinned for the context's K/V (Y GB if every slot were full)` (HMEM `context_ram`),
    `context memory: U of L GB budgeted on its cards` (KMEM `slot_memory`, used = total-free, limit = total+givable -
    the same sums QFN's own fork UI makes), and `last turn save on disk: main 12k tokens, 34 MB 42 s ago` per place
    (`m.context.turn_saves`);
  - that note line under the engine's Context bar when there is anything to say (nothing shows for engines that
    report none, so GLM/DS4/MM3 pages are unchanged);
  - the parked segment's legend now appends `N conversations` when the engine's VRAM lines carry `parked_n`
    (`m.context.parked.conversations`).
  No server change was needed for these: `metrics()` spreads `engine.info` and `_engine_context()` already passes
  `turn_saves`/`parked_n` through, so QFN's values reach the page in the shape the page now reads.
- **`engines/qfn/serve/server.py` + regenerated `serve/qfn_engine.py`** (step 3, commit `40cbc04c7`): `/metrics` -
  no other engine has a session folder, so per the plan the state goes under the engine's info with a one-line
  comment: `StrataEngine._session_note` now mirrors the engine's `INFO sess_json=` note into
  `self.info["sessions"] = {"files": …, "bytes": …, "max_bytes": …}` - the saved conversations count and bytes -
  which flows into `m.engine.sessions` automatically (the per-conversation list is left out of `/metrics` to keep
  the payload small; the full note stays in `engine.sessions`).

## Step 4 - tests (commit `259a145e1`)

- **`serve/test_dashboard.py`** - `QfnStatsToMonitor` (3 tests). Feeds a sample line set - the engine's own formats,
  guarded by asserting the printf strings still live in `engines/qfn/src/program/generate.cpp` (the
  `test_the_engines_lines_are_the_ones_read_here` pattern): the INFO main line (`mtp=1 lookup=64`), the
  `engine_kind=qfn` line, the solo-decode STAT with `promo=`, HMEM, KMEM, `sess_json=`, `sess_event=turn_saved` -
  through the real `QfnEngine._pump` (fake process) and a real `Service.metrics()`:
  `speculation_modes` = ["MTP", "Lookup"]; `info["context_ram"]` / `info["slot_memory"]` / `info["sessions"]` /
  `turn_saves["main"]`; `stat.promo` = 0.031 with a `stat_history` entry; and `/metrics` carrying
  `engine.speculation`, `engine.context_ram/slot_memory/sessions`, `tiers.now.promo`, `tiers.history.promo`,
  `context.parked.conversations` = 3, `context.parked.budget_bytes`, `context.turn_saves.main.tokens`.
- **`serve/test_new_engines_ui.py`** - `Page.test_monitor_qfn_stats_and_context_facts` (Node, like the hy4/ik2 page
  tests): the qfn tier tiles from the STAT line (`[["promo",0.031],["ram_fetch",0.487],["disk",0.0]]`),
  `tierShares` (4096 in VRAM / 9728 in RAM of 13824), `speculationView` = kept · MTP · 64% (37 of 58), and
  `contextFacts` producing the three Context-card lines; `contextFacts({}, {})` = [] (other engines unaffected).

## Tests run (repo root, one at a time, /opt/llm/saguaro/.venv/bin/python)

| module | before | after |
|---|---|---|
| `serve.test_dashboard` | 55 tests, OK | **58 tests, OK** (+3 new) |
| `serve.test_models` | 60 (file untouched by this task) | **60 tests, OK** |
| `serve.test_new_engines_ui` | 33 tests, OK | **34 tests, OK** (+1 new) |
| `serve/test_*qfn*` | - | no such module exists (`ls serve/test_*qfn*`: nothing; the QFN cases live in the two modules above, which `grep qfn serve/test_*.py` finds) |

## Files changed

- `serve/web/monitor.js` - `contextFacts` + Context-card note line + parked conversations count.
- `engines/qfn/serve/server.py` - `_session_note` passes the session folder's state into `info["sessions"]`.
- `serve/qfn_engine.py` - regenerated by `tools/saguaro/extract_qfn_engine.py` (was byte-identical before the run).
- `serve/test_dashboard.py`, `serve/test_new_engines_ui.py` - the tests above.
- `scratch/t6b-gaps.txt` - step 1's answer (gitignored, not committed).

Commits: `ca031e212` (step 2), `3be5403db` (step 2 follow-up: one decimal on the budget), `40cbc04c7` (step 3),
`259a145e1` (step 4), plus this report.

## Open questions / notes

- **Sessions shape in `/metrics`.** `m.engine.sessions` carries `files`/`bytes`/`max_bytes` only, not the engine's
  per-conversation list (`sess_json`'s `conversations[]`), to keep every `/metrics` poll small. If the page should
  one day list the saved conversations individually, the full note is still on the driver (`engine.sessions`).
- **HMEM/KMEM are configuration-dependent.** The engine prints them with elastic slots / KV-host-elastic only
  (`SAGUARO_ELASTIC_SLOTS_GIVE`, `SAGUARO_KV_HOST_ELASTIC`); a QFN run without them shows no context-RAM /
  slot-memory lines - correct absence, and the per-card pinned KV ("GB pinned in RAM" on the GPU card) comes from
  the VRAM lines regardless.
- **Turn saves render inside the engine-bar branch** of the Context card (the branch that needs the engine's VRAM
  lines, which QFN always prints); a hypothetical VRAM-line-less engine keeps the old fallback card, as before.
- **`mtp=` requires a T6a engine.** An older `saguaro-qfn` prints no `mtp=`, so Speculation keeps saying "off" -
  the server passes through what the engine says, as designed.
- **Parked count wording.** The legend reads `Parked in RAM 12.3k tokens · 152.6 GB · 3 conversations`; only the
  parked bytes (no `kv_bytes_token` engines) show without token counts - the count appears whenever the engine
  lines carry `parked_n=`, which QFN's always do.
