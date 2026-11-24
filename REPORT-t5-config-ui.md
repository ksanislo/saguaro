# REPORT-t5-config-ui: QFN configuration in the server + web UI (QFN-first project, task T5)

Branch `saguaro/q1-t5-config-ui`, on top of T1 (family "qfn") and T6 (engine stats).
One commit per plan step, plus two small fixups the step-5 test run found (noted below).

## What changed

### Step 1 - QFN's own field set (`cce69fb36`)

`serve/models.py`:
- New `QFN_KEYS` (models.py:77), separate from `QWEN_KEYS` (which stays untouched for CYBER-FROST):
  `context, images, gpus, reserve_mib, env, parallel, draft, expert_cache, prefill, kv, kv_resident, ram_gib`.
- `settings_of` / `apply` / `choices` got a dedicated `qfn` branch each (they used to share the Qwen path's):
  - **RAM tier** = the engine argument `--resident-budget-gib` (key `ram_gib`, empty = flag dropped, the engine
    sizes it itself). The Qwen path's `ram_gb` / `SAGUARO_EXPERT_RAM_GB` is now *refused* for QFN
    ("not a setting of this model: ram_gb") - QFN ignores that variable.
  - **K/V kept in VRAM** = `--kv-resident` (read as it is, 1024..1048576, empty drops the flag).
  - **Answers at once** = the top-level config key `"parallel"` (the QFN driver's `parallel_args` turns it into
    `--batch`; 1 or absent = one at a time). `apply` sets/pops the key and never writes a `--parallel` flag,
    which saguaro-qfn does not parse. `_commit`'s "needs a restart" tuple gained `"parallel"`.
  - **Drafting** on/off = the config's `--mtp` args vs `--mtp-max-t 1` (`--mtp` kept loaded either way, like the
    Qwen path). QFN's drafter is the setup's Lamina download the config names - not CYBER-FROST's `<pack>/mtp`
    with its `dense.txt` marker - so `choices` for qfn uses a new `qfn_mtp()` (models.py:345: the `--mtp`
    folder, if it is on this PC) instead of `qwen_mtp()`.
  - **Images**: see the note under "Choices and open questions" - the toggle now takes QFN's own pieces
    (`vision`/`vision_off` entry + the engine's bare `--vision` flag; a text-only config gets QFN's message
    instead of the Qwen path's "the setup adds it").
  - Kept for QFN, as before: expert cache (`--expert-cache`), prefill chunk (`--prefill`), VRAM reserve
    (`--vram-reserve-mib`), KV format (`--kv`, same four formats - checked against generate.cpp:3582, the
    engine parses exactly fp16/int8/k8v4/q4_0), GPUs and the raw `env` rows. Unknown keys are still refused
    and everything else in the config stays as it is.
- `describe()` prunes a stored `installer.as_set_up` down to QFN_KEYS for a qfn config (models.py:970): the
  AMD setup-written one (./ref/) was recorded while QFN shared the CYBER-FROST keys, and Reset must not send
  `ram_gb` back to a page that now (rightly) refuses it.
- Module docstring lists the QFN family's settings now.

`serve/web/models.js`:
- New `qfnFields()` (models.js:561), dispatched on `m.family === "qfn"` (models.js:296); `qwenFields` stays for
  CYBER-FROST. Tabs: Model (Context size with the `--kv-resident` window in the cost line, Answers at once,
  Drafting, Images), GPUs and memory (GPUs, VRAM kept free, **K/V kept in VRAM** `--kv-resident`,
  **RAM for the experts** `--resident-budget-gib`), Prompts and cache (Expert cache, Prompt chunk, KV cache).
- `ram_gib` added to `NUMS` so the draft serializes it.

### Step 2 - the live context change from Settings > Model (`6889a7835`)

- `engines/qfn/serve/server.py`: `StrataEngine.set_context(n)` (server.py:1507) rewrites `--max-context` in
  `self.spawn` (the 7-field tuple the driver keeps; same shape as the GLM engine's `set_context`,
  serve/server.py:1271). `restart()` re-`__init__`s from `spawn`, so the new process runs with the new context
  and `Service._restart_with` (~3342) now works end to end: stop (QFN's is the known no-op), set_context,
  restart; a start that fails puts the old size back, a good one is saved by `_save_context`.
- Regenerated `serve/qfn_engine.py` with
  `/opt/llm/saguaro/.venv/bin/python tools/saguaro/extract_qfn_engine.py` (never hand-edited; the re-extract
  changed nothing but the new 11-line method - the extraction is deterministic).
- `serve/server.py`: removed `QfnEngine`'s `set_context` override that raised
  "this engine's context is set in its config (--max-context)" - which made every reload from the dashboard
  fail - leaving the inherited driver method in place (comment kept where it was).

### Step 3 - the Turbo button for QFN (`d636d87d9`, fixup `198289424`)

Choice, as the plan asks for the reasoning: **QFN's config-side Turbo turns drafting on (MTP) and caps
Answers at once at 1; its server-side Turbo additionally puts the model's live Turbo (PLAN-turbo-live, the
`<config>.turbo.json` sidecar) to its default `{drop: 1, skip: "off"}` while Turbo is on, and restores what
it was when Turbo goes off.**
- `turbo_patch` needed no QFN-specific rules: it already does `TURBO_DRAFT` ("qfn": "mtp") and parallel->1,
  but until step 1 the draft half was *unreachable* for QFN - `choices` went through `qwen_mtp`, which demands
  `<pack>/mtp/dense.txt` (CYBER-FROST's own layout), so QFN configs always got `draft: ["none"]` and the patch
  was empty (the no-op the plan names). With `qfn_mtp` the patch is real: `{"parallel": 1, "draft": "mtp"}`.
- No `TURBO_ENV` knobs for qfn: the QFN engine has no `SAGUARO_QFN_*_MISS_SKIP`/`RAM_SKIP`/`RESIDENT_TOPK`
  equivalents in this repo, and inventing environment values for an engine whose docs (the T9a SETTINGS.md
  section) live on another branch would be guessing. What QFN *does* have is the per-request lossy path: the
  driver already appends `turbo_drop=` / `turbo_skip=` to the batch GEN line (serve/qfn_engine.py:1304), so
  the live Turbo is the meaningful half - `drop: 1` is the same smallest, safest default the TurboLive panel's
  own switch uses (serve/web/turbo.js).
- `serve/models.py`: `TURBO_LIVE_DEFAULT = {"qfn": {"drop": 1, "skip": "off"}}`; the Turbo note now also keeps
  the config's `parallel` value so switching Turbo off gives QFN's "8 at once" back (test-found fixup).
- `serve/server.py` `Service.set_turbo`: when the config's family has a `TURBO_LIVE_DEFAULT` entry, the button
  saves the current live Turbo, sets the default on Turbo-on and restores it on Turbo-off (after the config
  write, so a refused enable/disable leaves it untouched - the fixup commit).

### Step 4 - small UI fixes (`226ddc148`)

- a. 401: `serve/web/server.js` `loadCtxInfo` marks `S.conn = "key"` when `/api/context` answers 401;
  `serve/web/settings.js` (Model section, ~line 93) then says "API key needed: paste it under
  Settings > Connect your tools." instead of "This server does not say its context range". On the Models page
  `loadModels` maps a 401 `/api/models` to the same wording (it already reached `MS.err`, but as the generic
  lib message).
- b. `serve/models.py` `config_paths` excludes `*.turbo.json` beside `.shared-settings.json`, so a model's live
  Turbo sidecar (`<stem>.turbo.json`, serve/server.py:8691) is no longer listed as a model.
- c. `serve/web/turbo.js` ~166: the bare "On" label became "Live Turbo", and the label/description now stack
  with the `set-row__text` / `set-row__label` classes like every other settings row (they were plain `<span>`s,
  which is why they ran together as "Ondrop the weakest experts"); checked the Drop experts and Skip rows and
  gave them the same classes.

### Step 5 - tests (`c374dd90d` + `198289424`)

- `serve/test_models.py`: new `qfn_cfg` fixture (the AMD setup-written shape from ./ref/, in small; `mtp` /
  `vision` knobs); `Settings.test_qfn_read / test_qfn_written_back / test_qfn_drafting / test_qfn_images /
  test_qfn_reset_of_an_old_as_set_up / test_the_turbo_sidecar_is_not_a_model`; new `QfnTurbo` class
  (patch contents, on/off returns args + env + the parallel key, and the Service-level live-Turbo sidecar
  on/off).
- `serve/test_dashboard.py`: `QfnContext` - the real `QfnEngine` (`serve.server._qfn_engine_class()`, no
  process): `set_context` rewrites `--max-context` in the spawn command in place, appends it when absent, and
  no longer raises on the reload flow's stop/set_context/restart shape.
- `serve/test_new_engines_ui.py`: `QfnSettings` - every argument the page writes is a flag saguaro-qfn's own
  parser reads (`QFN_FLAGS`, scraped from `engines/qfn/src/program/generate.cpp`), the other engines' flags are
  never written, the `parallel` key round-trips, CYBER-FROST/GLM settings are refused, Turbo supported;
  a Node `Page` test: `loadCtxInfo` against a 401 sets `S.conn === "key"` (and no context data).

## Files touched

serve/models.py, serve/web/models.js, serve/web/settings.js, serve/web/server.js, serve/web/turbo.js,
engines/qfn/serve/server.py, serve/qfn_engine.py (generated), serve/server.py,
serve/test_models.py, serve/test_dashboard.py, serve/test_new_engines_ui.py.
Not touched: the engine C++ sources, saguaro.py / setup.py, `QWEN_KEYS` / `qwenFields` (CYBER-FROST).

## Tests (repo root, one module at a time, /opt/llm/saguaro/.venv/bin/python -m unittest ...)

| module | before | after |
|---|---|---|
| serve.test_models | 51 OK | **60 OK** |
| serve.test_dashboard | 52 OK | **55 OK** |
| serve.test_new_engines_ui | 27 OK | **33 OK** (Node present, so the page tests ran) |
| serve/test_qfn* | - | **no such module exists** (see open questions) |

No test was removed or weakened; all failures seen along the way were in new code and are fixed.
`node --check` passes on every edited web JS file.

## Choices and open questions

1. **Images, "allowed for setup-written QFN configs"**: I made the QFN Images switch work with QFN's own pieces -
   on = the `vision`/`vision_off` entry returns and the bare `--vision` engine flag goes into the command,
   off = the reverse - and `choices.images` also counts a lone `--vision` in the args as support (both ./ref
   live configs have section + flag, so they toggle today already). For a config with neither (the AMD
   setup-written one, `as_set_up.images: false`), enabling it would need a `vision-qfn` section whose mmproj
   path nothing here knows: the switch stays disabled there, with QFN's own wording ("its image encoder
   (vision-qfn with the model's mmproj file) is not set up here") replacing CYBER-FROST's "the setup adds it".
   If the plan's intent was that the page should *synthesize* a vision entry (vision-qfn beside the engine, the
   mmproj beside the model's files, guarded by their existence), say so and I'll add it.
2. **QFN serve test module**: the plan's run list ends with "the QFN serve test module (ls serve/test_qfn*)" -
   there is no `serve/test_qfn*.py`. QFN's own serve tests live in `engines/qfn/serve/test_*.py`; I ran none of
   them (the plan restricts the run list), and put the `QfnEngine.set_context` test in `serve.test_dashboard`,
   next to the other engines' set_context test. Worth confirming that is where it was wanted.
3. **Turbo live default persistence**: the pre-Turbo live Turbo setting is kept in memory only
   (`Service.turbo_live_before`); a Turbo-on -> restart -> Turbo-off sequence restores "off" rather than the
   old sidecar value. Persisting it would mean a second key in the sidecar file itself.
4. **Settings names**: the RAM tier is `ram_gib` (GiB, matching `--resident-budget-gib`; the Qwen path's GB
   key `ram_gb` stays for it) and "K/V kept in VRAM" reuses the key name `kv_resident` (GLM/MiniMax call the
   same engine flag "Positions kept in VRAM" behind a streaming switch QFN does not have - the QFN row is a
   plain number, no switch).
5. **`kv_resident` bound** 1024..1048576 mirrors the GLM/MiniMax checks; the engine's own floor for QFN was not
   re-derived from this branch's sources (the T9a SETTINGS.md section with QFN's flag table lives on another
   branch).
6. **describe-side `as_set_up` pruning** leaves the stored key untouched (it stays as an older record); only
   what the page gets for Reset is filtered. `_save_context` and `models.save` keep writing `as_set_up` with
   today's `settings_of`, so new records already hold QFN's own keys.
