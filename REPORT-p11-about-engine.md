# P11 report: no engine version strings in the page (1.2.1 item 11)

Branch `saguaro/q1-p11-about-engine`, one change commit (`bdf480106`) plus this report.

## What changed

The plan said the About row and other labels show the engine's own version (QFN's frozen
Strata string `0.1.49.2`, stale ones elsewhere). Inventory first, as the plan asked
("find them all first: grep for the engine's version fields"). The engine's version fields
in the payloads the page reads are the metrics `engine` blob's `version`, `engine`
(QFN's INFO `engine=0.1.49.2`) and the old metrics key `maya_version`. `/health`'s
`version` is Saguaro's release in this checkout's whole history (verified against the old
revisions), but it is a bare "version of what is running" field - the header label prints
whatever a foreign or older server answers there. Labels:

- `serve/web/chat-view.js` - the chat header's model-menu card ("`<model>` on this machine
  · v`<health.version>`"): the `· v` field is gone; the card names the model, no version.
  `ModelMenu` is exported for the page test (the `TurboMenu` precedent).
- `serve/web/settings.js` - About's Version row: the fallback chain loses the engine-blob
  field `eng.maya_version` (`eng.saguaro_version || S.health.version` remain, both Saguaro's
  release fields; an older server's `health.version` says the same thing as its `maya_version`
  did). `About` is exported for the page test (the `Tools` precedent).
- `serve/web/server.js` - `fetchReport()` (Copy report / Show engine log, used by About and
  the Monitor) strips the INFO `version=` / `engine=` pairs from the report text client-side
  (`\b(?:engine|version)=\S+ ?`); `engine_kind=`, the engine's name, and every other engine
  fact stay. This is how the report keeps the version out without touching Python: the
  server's own `report()` still contains the full INFO pairs.
- Untouched, as the plan required: `serve/server.py` / `serve/qfn_engine.py` / `engines/` /
  `saguaro.py`. `/health`, `/metrics` and `/v1/status` keep sending the fields (other
  clients), and the saved-conversation keying (the QFN session folder named after the
  engine version) is untouched.

MCP servers' own version in Settings > Tools (`s.info.version`) is not the engine's - left.
The update card, the nav dot and the "Updated to v…" toast name Saguaro's release - left.

## Tests

- New: `serve.test_dashboard.PageLogic.test_the_page_shows_no_engine_version_string` -
  node, fake state whose engine-version fields (`health.version`, `metrics.engine.version/
  engine/maya_version`) all say `9.9.9-old`: the rendered Settings > About, the chat
  header's card and `fetchReport()` contain no `9.9.9`; About still says `Saguaro v1.2.1`
  and the header still says "on this machine"; the report keeps `Engine: arena_mib=70000
  engine_kind=qfn experts=13824`. (About's own row passes before the change too - its
  Version row already led with `saguaro_version` since the earlier About fix; the header
  and the report are the two sites that leaked `9.9.9-old` pre-change: the old header
  template rendered `· v9.9.9-old` for a `health.version` of that value (checked against
  the `HEAD~1` source), and the old `fetchReport()` returned the report verbatim - run
  against a scratch copy of the `HEAD~1` web files, it said `reportHas999: true` where the
  fixed one says `false`.)
- Run list, one module at a time, `/opt/llm/saguaro/.venv/bin/python -m unittest`:
  - `serve.test_dashboard`: 79 tests OK (78 before the new one; both node-skipped-none,
    node v20 present)
  - `serve.test_new_engines_ui`: 38 tests OK (before and after)
  - `serve.test_turbo_live`: 13 tests OK (before and after)

## Open questions

- The About Version row keeps the `S.health.version` fallback: on this server that field is
  `SAGUARO_VERSION` (in this checkout's whole history), but it is a bare "version of what
  is running" field and a foreign server answering it with its engine's version would print
  as `Saguaro v<engine>` - the page cannot tell the two apart from `/health` alone. Dropping
  the fallback would leave the Version row blank when `/metrics` never loaded (model
  unloaded), so it stays.
- The report scrub is name-based: an engine INFO pair literally keyed `engine=` loses its
  value in the copied report even when it names the engine (`engine=hy4`); `engine_kind=`
  carries the same name right beside it, so nothing is really lost.
- `engines/qfn/serve/server.py` and `serve/qfn_engine.py` were not touched, so
  `tools/saguaro/extract_qfn_engine.py` was not run (no driver change to extract).
