# U5: the read heartbeat - prompt reads report progress every second

REPORT for `PLAN-u5-read-heartbeat.md`, branch `saguaro/q1-u5-read-heartbeat`, commits `8477e8189..15a78ef4b`.

## What was done

A prompt read that takes ten seconds (Ken's live log: "143633 tokens = 142491 reused + 1142 read in 10877 ms")
showed nothing on the Monitor until its `PP`/`BRP` lines landed: no rate, no progress, nothing - the eleven
seconds of staging the reused K/V from RAM are invisible. Now the engine reports the read in flight about
once a second, and every live figure refreshes with it.

**1. Engine (8477e8189, host code only - no kernel, launch or ordering change).** New line, at most one per
~1 s per reading request, printed from host points between layer passes / pipe pieces / KV staging passes:

    RHB rid=<rid|0> stage=<stage|read> done=<new tokens done> new=<new tokens total> reused=<n> pass=<i>/<n> ms=<read ms>

with `slot=<b>` appended on the batch path. `stage` while the reused prefix is copied RAM→VRAM, `read` while
new tokens are read; `ms` is the request's own read clock, `pass` the current layer pass (or pipe piece) over
its count. The mechanism is a host-only callback `on_layer_pass(pass, passes, staged, pos0)` on `PromptJob`
and `Prefill` (fired at the end of each layer iteration, the staged flag set where the prefix copy lands),
wired by `generate.cpp`: the solo read's `sp.on_layer_pass` + its pipe-piece boundaries, the batch admission's
`adm_read` layer callback + its `make_piece` jobs, each throttled to one line per ~1 s per reader (a new
reader's first beat prints immediately). Every existing line is unchanged; the fast-lane read path (its own
per-window `BRP` progress) keeps the old behaviour.

**2. QFN driver (a07c6ed53).** The pump parses RHB into `read_live = {"solo": entry, "slots": {slot: entry}}`
(the live prefill rate = the tokens between two heartbeats over the time between them, standing while nothing
moves; first beat = the read's own mean). A slot's RHB lines ride its queue beside its `BRP` ones; `RDONE`
clears the solo entry, `BRA` the slot's. `_control` / `generate` / `_admission_wait` wake on each line
(yield None), so a waiting client sees a keep-alive per second instead of per chunk. `serve/qfn_engine.py`
re-extracted with `tools/saguaro/extract_qfn_engine.py` (never hand-edited).

**3. /metrics + Monitor (2982523b4).** Each heartbeat marks the chat's `RequestStat.read_hb` (stage, done/new,
reused, rate; three seconds of silence = stale) and steps its counted tokens into the engine-total window, so
the live Prefill figure moves per second instead of per chunk-end. `/metrics`: every reading chat's row carries
`read_stage` / `read_done` / `read_new` / `read_reused`, and `live.read_states` collects them. monitor.js: with
most of a prompt reused the Prefill card says **"restoring 142,491 cached · reading 1,142 new"** (plus
"staging" while the KV copy runs) instead of a frozen rate, and the conversation row shows the same with its
`done / new`. An engine without the lines shows exactly what it did before.

**4. Tests (15a78ef4b).** `serve/test_read_heartbeat.py`, 13 tests, all offline: the real driver's `_pump`
over fake RHB lines (solo + slot routing, the between-beat rate, `pass=` parsing, RDONE/BRA ending, an engine
without lines never even laying the state out), and the shared Service behind a fake engine feeding RHB
through the real `_admission_wait`: the one-chunk read with eleven seconds of staging (row + `read_states`
each second, no rate to freeze at, then the read steps counting per beat, BRA exacting without double
counting), a multi-chunk read (each beat's rate, the window over the whole read), two chats reading at once
(each row its own figures, the engine-total window summing them), staleness, and a contract check that
generate.cpp prints exactly the format the driver parses.

## Files

    engines/qfn/include/strata/prefill/prefill.hpp |  10 +   on_layer_pass on PromptJob + Prefill
    engines/qfn/src/prefill/prefill.cpp            |  13 +-  the callback, the staged flag (host, after each pass)
    engines/qfn/src/program/generate.cpp           |  84 ++   the RHB emitters (solo + slot), wiring, protocol note
    engines/qfn/serve/server.py                    |  73 ++   _rhb, read_live, pump routing, RHB keep-alives
    serve/qfn_engine.py                            |  73 ++   (extracted)
    serve/server.py                                |  55 ++   read_hb/_note_hb, tls.on_hb, row fields, read_states
    serve/web/monitor.js                           |  19 +-  "restoring N cached · reading M new", done/new rows
    serve/test_read_heartbeat.py                   | 319 ++   new: 13 tests
    8 files changed, 642 insertions(+), 4 deletions(-)

## Tests and results

- `tools/qfn/build.sh ./build-u5` with `QFN_GGML_DIR=$(realpath ../strata-main/third_party/llama.cpp)`
  (fresh dir, own failed artifacts removed first): **`QFN BUILD OK 15a78ef4b -> ./build-u5`**, 179/179, exit 0.
- `/opt/llm/saguaro/.venv/bin/python -m unittest serve.test_read_heartbeat` - **13 tests OK**
- `... -m unittest serve.test_dashboard` - **58 tests OK**
- `... -m unittest serve.test_models` - **62 tests OK**
- `node --check` over monitor.js (scratch copy): clean.

## Open questions

- The throttle is a plain per-reader timestamp (950 ms), not a global rate limit: eight slots reading at once
  print at most one line per second *each* - a few lines per second in total, which the pump parses cheaply.
  If that ever reads as chatter, a shared interval is a one-line change.
- The solo read's `pass=` for the layer-pass beats counts the current chunk's passes (staging shows 1/n of the
  staging passes); its piece-boundary beats show piece i/cuts. The Monitor does not show `pass` yet (only
  stage/done/new/reused ride /metrics) - a progress bar is a possible U6 garnish.
- The fast-lane read path was left as it is (it already answers `BRP` per window); it prints no RHB.
