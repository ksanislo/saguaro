# REPORT-p125f-turbo-cleanup - the rest of the Turbo cleanup (Saguaro 1.2.5)

## Step 1: every Turbo setting, env var, sidecar name and page control, with file:line

Turbo grew in layers (7092f3936 button, d79062341 live per-request, d636d87d9 the button's live half,
c3844c07d "the button drops no experts", P5/S1 plain words + live skip, P124s "the panel reaches the app's own
chats only").  What each layer left behind, marked LIVE / DEAD (nothing reads it) / DUP (a twin of another one):

### The live per-request Turbo (serve/turbo_live.py, PLAN-turbo-live)

- `serve/turbo_live.py:15,16` SKIP_LEVELS / DROP_MAX - LIVE (normalize_turbo).
- `serve/turbo_live.py:19` normalize_turbo - LIVE: server.py:1225 (the request field), 3474 (POST options),
  10083 (the sidecar at start).
- `serve/turbo_live.py:42` turbo_on / `:47` turbo_of - LIVE both (3475,3477,10083 / 6045); DUP of each other
  (same test on a different wrapper), kept.
- `serve/turbo_live.py:53` turbo_keys - LIVE: the GEN line of both drivers: server.py:1937,1956 (GLM/DS4/MM3),
  serve/qfn_engine.py:1399 (from engines/qfn/serve/server.py:1840).  Engine side: src/program/generate.cpp:5223-5224,
  engines/qfn/src/program/generate.cpp:1221-1222.
- `serve/turbo_live.py:67` turbo_badge - **DEAD**: nothing calls it.  The Monitor's badge text is its JS twin
  serve/web/turbo-core.js:81 (monitor.js:28,173, chat-view.js:366); the server sends the raw dict on the
  monitor rows (server.py:3186,4649,6555).
- request field `turbo`: server.py:1224 (the 400), serve/responses.py:109,270 - LIVE.
- `NO_LIVE_TURBO` server.py:84,3392,3475,8012 (Inkling 2, Hy4-Flash) - LIVE (tested test_new_engines_ui.py:152).
- the model's live default `turbo_default`: server.py:3228 (init), 3386-3393 (check_request: the web app's own
  chats only, PLAN-p124s), 3469-3486 (set + sidecar write), 8004-8012 (GET /api/turbo/options),
  8381-8400 (POST) - LIVE.
- skip-support gate `turbo_skip_supported`: server.py:3488-3495, 8010; the engine says so itself
  (`INFO turbo_skip=1`: engines/qfn/src/program/generate.cpp:12707) - LIVE gate, but its docstring (3489-3491
  "QFN's applies drop only") is STALE since S1; page half S.turboSkip turbo.js:139,145 - LIVE.

### The sidecar

- `<config stem minus .json>.turbo.json`: path server.py:10080, written/unlinked 3478-3485, read at start
  10079-10089; left out of the model list serve/models.py:831 + tools/config_names.py:26 (SIDECAR_SUFFIXES);
  moved with the C122 rename saguaro.py:838-859, tested tools/test_config_names.py:50-269.  **One name only -
  git log -S shows no older sidecar name ever existed** (d79062341 introduced it as-is), so the plan's
  "read an old name once, then rename" clause has nothing to apply to.
- GAP found: serve/models.py `delete()` (1475-1480) sweeps `.prev` and `.shared-settings.json` but **not**
  `.turbo.json` - a deleted model's live Turbo file survives and re-enables Turbo for a later model set up
  under the same config name (the page's own chats, never an API request).

### The Turbo button (the engine-settings half)

- serve/models.py: TURBO_CONTEXT 1061, TURBO_CONTEXT_FAMILIES 1062, TURBO_ENV 1079, TURBO_DRAFT 1083,
  TURBO_DSPARK_CONF 1090, TURBO_DSPARK_FAMILIES 1091, TURBO_NO_TOPK 1094 (+ turbo_smallest_quant 1098),
  TURBO_NONE 1095, turbo_patch 1107, turbo_held 1132, turbo_one_at_a_time 1137, turbo_on 1142, turbo_info 1146,
  turbo_enable 1163, turbo_disable 1191 - LIVE.
- `TURBO_LIVE_DEFAULT` models.py:1088 = `{}` - **DEAD since c3844c07d** ("the button drops no experts"): its
  only reader is server.py:4052 and it always misses.
- server.py set_turbo: the no-config `turbo_mem` path 4044-4046 - LIVE; the live half 4052-4059 (`live is
  not None` with an empty dict) - **DEAD** (the violating path P124s step 1 named, disabled by the empty dict
  instead of removed); `turbo_live_before` 3230,4055,4058-4059 - **DEAD**; the `live_before` param and the
  `turbo.live_before` note key models.py:1163,1186-1187 (written from server.py:4050) - **DEAD write**: the
  note is only ever read back at the dead 4058.
- turbo note keys `since`/`before`/`args`/`env`/`parallel`/`applied`/`one_at_a_time` (models.py:1179-1185,
  read 1137-1217) - LIVE (`since` is informational: no reader, but it is config-file text for people, kept).
- `/api/turbo` GET/POST: server.py:8000-8003, 8160-8161, 8366-8379; `fifo.set_one` 444,4051,10081 - LIVE.

### The env vars Turbo writes into the config (TURBO_ENV) - all LIVE, all engine-read, all in docs/SETTINGS.md

- `SAGUARO_GLM_MISS_SKIP_W=0.16`, `SAGUARO_GLM_MISS_SKIP_FROM=23` - src/core/glm_fast_path.cu:2381,2382; docs 154.
- `SAGUARO_GLM_RESIDENT_TOPK=4` - src/kernels/cuda/glm_fast.cu:6653; docs 155 ("Turbo uses _TOPK=4").
- `SAGUARO_DS4_RAM_SKIP_W=0.14` - src/ds4/ds4_model.cu:2063; docs 948.
- `SAGUARO_DS4_DSPARK_CUM=0.3` - src/ds4/ds4_main.cpp:279; docs 874.
- `SAGUARO_DS4_RESIDENT_TOPK=3` - src/ds4/ds4_kernels.cu:3957; docs 949 ("Turbo uses _TOPK=3").
- Named as deliberately unused in the models.py:1069-1072 comment (SAGUARO_GLM_TRUNK_KEEP(_MAP),
  SAGUARO_*_DISK_SKIP_*, SAGUARO_GLM_DFLASH_MIN/_CUM, SAGUARO_*_RESIDENT_SKIP_W): engine knobs Turbo does not
  touch - the comment is true, kept.  No `SAGUARO_*TURBO*` env var exists anywhere.

### The page controls

- serve/web/turbo.js: S.turbo 16, loadTurbo 18 (app.js:125), WHAT/engineLines 28-35 (the restart dialog),
  enterPrefs/leavePrefs 38-55 (turbo-core snapshot, localStorage `turbo_<key>` 27), followRestart 58-76,
  setTurbo 78-113, TurboButton 116-130 (settings.js:159) - LIVE.
- serve/web/turbo.js: S.turboLive 138, loadTurboLive 141, setTurboLive 151, TurboLive panel 168-187
  (settings.js:160) - LIVE.  The panel is the one settings page place for the app's own chats' Turbo; the
  button beside the heading is the engine-settings half; the chat's own is the chip - one place per half.
- serve/web/turbo-core.js: fastestThinking 6, turboEnter 11, turboLeave 22, CODING_IDEAS 29 (chat-view.js:498),
  TURBO_DROPS 46 (1/2/3 while the server allows 0..9 - the page's own choice, fine), TURBO_DROP_HINT 47,
  TURBO_SKIPS 48, turboSkipWarn 56, turboOfChat 59 (chats.js:45), turboToggle 68, turboChoose 74,
  turboBadge 81 - all LIVE.  turboBadge is the surviving half of the turbo_badge DUP (the Python twin dies).
- serve/web/chat-view.js: TurboMenu 346-363, TurboChip 364-376, composer wiring 425 - LIVE.  Class hooks
  `turbo-menu` (349) and `turbo-chip` (368) have no rule in any css - harmless hooks, left.
- serve/web/turbo.css: the button's styles - LIVE, except `.turbo__state` (18, 27) - **DEAD**: no element
  carries that class since e0b9d5485 put the label in a plain `<span>` (turbo.js:128).
- serve/web/monitor.css:102 `.now__chat--turbo` - LIVE (monitor.js:173).
- monitor.js TurboBadge 28,207,754,789; chats.js setChatTurbo 46, body.turbo 436-437 - LIVE.
- Stale page text: the button tooltip (turbo.js:120-121) lists "(greedy decoding, no thinking, expert
  skipping, aggressive drafting)" - the fixed words of 7092f3936; what the button really changes per model is
  the plan rows it already holds (t.plan / t.env, engineLines 31-35), and since c3844c07d it never touches the
  live options.  One place to say what is on now = the button's own data, not a hand-copied list.

### tools/ and docs/

- tools/config_names.py:26 SIDECAR_SUFFIXES, tools/ik2/serve_check.py:440 (asks refused unloaded) - LIVE.
- docs/SETTINGS.md: **no Turbo section exists at all** - only the env rows (154,155,874,948,949); the models
  -py:1064-1078 header cites "docs/SETTINGS.md" as the Turbo reference, so the section must be written
  (step 2) to match the code.
- docs/DIVERGENCE.md:27-31 (GLM drop - true), :307 (MM3 keys - true), **:521-524 (QFN entry 15 "drop is
  applied, skip is not", "logged as 'this engine applies only drop'") - STALE since S1/caffb6ed2: QFN applies
  skip (strata/program/turbo_skip.hpp) and reports INFO turbo_skip=1**.
- docs/AMD_SAGUARO.md:277, docs/HY4-FLASH.md:112, docs/INKLING2-MINI.md:171 - true (TURBO_NONE, TURBO_DRAFT).
- CHANGELOG entries: history, untouched.

### Docstrings/comments that describe the removed behavior (fixed in step 2)

- server.py:3488-3491 (turbo_skip_supported "QFN's applies drop only"), 4038-4043 (set_turbo "puts its live
  Turbo to its default, switching off restores what it was"), models.py:1084-1088 (the TURBO_LIVE_DEFAULT
  block), test_models.py:526-528 (QfnTurbo's class doc) and :582 (turbo_live_before),
  test_new_engines_ui.py:242 (assertNotIn("qfn", M.TURBO_LIVE_DEFAULT)).

## Step 2: what went, what was said instead (commit 0bb4163e4)

- `serve/turbo_live.py`: `turbo_badge` gone (the badge has always been `turbo-core.js` turboBadge's own words;
  the server sends the dict raw on the monitor rows).  `serve/web/turbo-core.js`: the twin's comment follows.
- `serve/server.py`: the set_turbo live half (the `live is not None` block, `turbo_live_before`, the `was` read
  feeding it) and its `Service.turbo_live_before` init gone; `set_turbo` and `turbo_skip_supported` docstrings
  say what is true (the button never touches the live half - PLAN-p124s; skip applies everywhere, the QFN
  `INFO turbo_skip=1` fact now only hides the choice for a pre-s1 QFN build).
- `serve/models.py`: `TURBO_LIVE_DEFAULT` and the `live_before` param/write in `turbo_enable` gone (the comment
  keeps the why: c3844c07d, PLAN-p124s); `delete()` sweeps `<stem>.turbo.json` beside `.prev` and
  `.shared-settings.json`, for the config and its maya- shadow (left, a deleted model's panel choice would
  re-switch Turbo on for a later model under the same name).
- `serve/web/turbo.js`: the button tooltip no longer recites 7092f3936's fixed feature list - the button says
  its own on/off state (the one place for it), the dialog says what this model changes from the server's plan
  rows, and the TurboLive panel below stays the one place for the app's own chats' live options.
  `serve/web/turbo.css`: `.turbo__state` (both rules) gone - no element ever carried it.
- Old sidecar name: none ever existed (git log -S `turbo.json`: `<stem>.turbo.json` since d79062341), so the
  read-once-then-rename clause had nothing to apply to; one name is kept.
- `docs/SETTINGS.md`: new `## Turbo` section where the models.py header points - the two halves, the exact
  `TURBO_ENV` / `TURBO_DRAFT` / context values, the Maya-XS cap exception, the `"turbo"` note, the sidecar and
  its delete, and "an API request gets Turbo only from its own field"; phrased so test_turbo.py's
  `count("changes the answers - Turbo") == 2` still reads true.
  `docs/DIVERGENCE.md` QFN entry 15: what S1 made true (skip applies via `turbo_skip_of` /
  `strata/program/turbo_skip.hpp`, `INFO turbo_skip=1` at generate.cpp:12707, said-once refusals for what
  still runs exact; no "this engine applies only drop" line left).
- Tests follow in the same commit: test_models.py (QfnTurbo doc, the live_before line, the panel's sidecar
  asserted untouched across a button flip), test_new_engines_ui.py (no `TURBO_LIVE_DEFAULT` attribute).

## Step 3: tests (commit a162404a0)

- `serve/test_turbo_live.py` PageSide `test_the_page_turbo_state_round_trips_to_the_wire` (node): bolt on ->
  drop 2 + skip RAM -> the dict the body carries every turn -> read back whole next turn -> both picked off ->
  nothing sent; and the server's half reads the same dict (`normalize_turbo` as-is, `turbo_keys` =
  ` turbo_drop=2 turbo_skip=ram`, badge `-2 experts · VRAM only`, off = the exact line).
- `serve/test_turbo_live.py` `ButtonAndLiveApart`: a glm run config and a real sidecar through HTTP -
  `set_turbo(True)` writes the config note (no `live_before`, no `TURBO_LIVE_DEFAULT` to write one) and nothing
  else: the panel's `{"drop": 1, "skip": "ssd"}` and its sidecar byte-identical across the switch on and off;
  **nothing an API request sees changes** - `/v1/chat/completions` without its own field reaches the engine
  with `turbo` None and an empty `turbo_keys` line with the button on and with it off, while the app's own
  `strata_resume` request still takes the panel's.  (The P124s guards -
  `test_the_api_never_borrows_the_pages_turbo`, the EngineLine exactness - still pass untouched.)
- `serve/test_models.py` `test_off_the_list_only`: the deleted model now leaves no `<stem>.turbo.json`.

## Files touched

serve/turbo_live.py, serve/models.py, serve/server.py, serve/web/turbo.js, serve/web/turbo-core.js,
serve/web/turbo.css, serve/test_models.py, serve/test_new_engines_ui.py, serve/test_turbo_live.py,
docs/SETTINGS.md, docs/DIVERGENCE.md.  serve/qfn_engine.py untouched by hand and needed no re-extract (the QFN
driver sources engines/qfn/serve/server.py, which references none of the removed names - grep says so).

## Tests run (each module one at a time, /opt/llm/saguaro/.venv/bin/python 3.13.5, node 22)

| module | before | after |
| --- | --- | --- |
| serve.test_turbo | 24 OK | 24 OK |
| serve.test_turbo_live | 14 OK | 16 OK (+2 new) |
| serve.test_models | 64 OK | 64 OK (three tests reworded/extended) |
| serve.test_new_engines_ui | 38 OK | 38 OK |
| serve.test_dashboard | - | 85 OK (renders the Turbo rows/button under node) |
| serve.test_server | - | 179 OK |
| serve.test_priority | - | 67 OK (the set_one Turbo test) |
| serve.test_model_api | - | 28 OK (models.delete through the page's API) |

## Open questions

1. The chat's bolt shows only the chat's own Turbo; a chat that merely inherits the panel's choice looks
   exact until its first answer badge appears. Showing the inherited state in the chip would need
   `/api/turbo/options` folded into the chip - a feature, left alone here.
2. `turbo_on` / `turbo_of` in serve/turbo_live.py stay a near-twin pair (dict vs sampling wrappers, both
   read); consolidating one into the other would only shorten one line - left.
3. `serve/web/chat-view.js` sets classes `turbo-chip` / `turbo-menu` no stylesheet defines (hooks from P5).
   Harmless; a design pass should either style them or drop the hooks.
4. The `/api/turbo` info fields `since` and `one_at_a_time` have no page reader (`since` lives in the
   config's turbo note for people, `one_at_a_time` describes what `plan: {}` means); left as diagnostics.
5. A model deleted while its server still runs keeps its in-memory `turbo_default` until the next start (the
   sidecar file now goes with the config).  Out of the cleanup's line; say if the page's Delete should also
   reset the live panel.
