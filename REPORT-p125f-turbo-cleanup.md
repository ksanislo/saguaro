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
