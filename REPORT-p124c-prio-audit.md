# P124c — request priority on every engine: the audit (REPORT ONLY; no code changed)

Audit of how far each engine is from QFN's request-priority behaviour (1.2.2, R122), with file:line evidence.
The five engines: GLM-5.3-Flash (saguaro-glm), DeepSeek-V4.1-Flash (saguaro-ds4), MiniMax-M3.1-Flash (saguaro-mm3),
Inkling 2 Mini (saguaro-ik2), Hy4-Flash (saguaro-hy4). Docs: `docs/SETTINGS.md:240-280` (Request priority).

## 0. The baseline the engines are measured against (what the server does, and QFN)

The line engines all run through one Python driver, `serve/server.py`'s `StrataEngine` (:1453); its capabilities come
from the engine's READY line: `can_stop`, `can_pause`, `multi` are parsed at :1518-1520. QFN runs its own driver
(`serve/qfn_engine.py`, batch BGEN/BDONE/BSTOP, mounted as `QfnEngine` at `serve/server.py:9430`); the FIFO's places
are the engine's `multi` (`serve/server.py:9917`, :3531-3533).

The preemption itself (`Service._sliced`, `serve/server.py:5647-5928`):

* Who steps aside: `FairLock.step_aside` (`serve/server.py:602-627`) — low always for any normal/high running or
  waiting, normal only for a waiting high that does not fit (last-started normal first, `_last_normal` :593), high never.
* Mid-answer, on every token's tick (:5770-5781): **PAUSE if `engine.can_pause`, else a cut** — `cut=True`,
  `gen.close()` sends STOP and drains to the DONE (:5797-5798), then `yield_place` + queue again (:5848), and the
  resume re-sends `prompt + what it wrote` (:5911). Whether that re-sends the whole prompt or reuses the engine's
  conversation state is what differs per engine below.
* Mid-read, on the engine's PP/BRP tick (:5726-5740), gated to engines that **keep a stopped read**:
  `(can_pause and multi)` — GLM with several conversations — or `(keeps_stopped_read and not batch)` — QFN solo
  (`keeps_stopped_read = True` at :9438, `#1620`: engines/qfn/src/program/generate.cpp:11116, :22598-22604
  "cancelled request keeps its conversation: N tokens live"; a QFN **batch slot's** cancelled read is dropped,
  :9442-9446). Every other engine's read runs to its end (:5725). A `PAUSE` during a read stops at the next chunk
  and keeps what it read; the driver never sends PAUSE to an engine that did not offer it (:5711, :2389-2402).
* Nothing is dropped for waiting: never cut for an equal/lower priority, paused requests wait forever with
  keep-alives (`docs/SETTINGS.md:266-271`).

QFN's "level" in Ken's terms: nothing dropped (a cut answer keeps its conversation live at the token it reached,
a solo cut read too), a high request answers within about a second of stepping in, slots park conversations
(`serve/qfn_engine.py:826-871` batch slots, `slot_keep` :851, `BSTOP` frees a slot at its BDONE :1552-1560).

## 1. GLM-5.3-Flash (saguaro-glm) — the line driver's reference; PAUSE exists

1. **Mid-answer.** STOP honoured at every token (`src/program/generate.cpp:2536-2539`; STOP/QUIT set `stop_req`
   :1302, :1326-1328). PAUSE exists: READY carries `pause` (`src/program/generate.cpp:5007-5008`), and the Metal
   variant too (`src/metal/serve.cpp:560`). PAUSE = STOP + snapshot moved to the last fed token
   (:1248-1250, :1319-1324; `pause_snapshot` :2328-2361), so "prompt + what was written" continues with **one** token
   to read; logged `glm slots: paused at N tokens (M written): the state is kept there` (:2354) — or, when the state
   does not stand exactly there (drafting edge, pictures), `paused, but the state stands at P of F tokens: not kept`
   (:2357) and the resume re-reads the written tokens from the prompt-end state. `PAUSE keep` (first slice) sets the
   prompt's state aside for the chat's next turn (:2337-2347). Today a priority pause **is** this PAUSE
   (`serve/server.py:5776-5778`) — GLM is the engine the feature was shaped on.
2. **Mid-read.** STOP between the prompt's chunks keeps what it read: `SAGUARO_GLM_KEEP_CANCELLED` (on by default)
   keeps the exact half-read state, else the longest checkpoint (:2256-2304; log "a cancelled prompt keeps the N
   tokens it read"). The server still only *preempts* reads when `can_pause and multi` (:5726-5729) — on a solo
   (`multi=0`) GLM a PAUSE during a read may be ignored or lose the read (:5720-5722), so a solo read runs on today
   even though the engine would in fact keep a *cancelled* read.
3. **Park/unpark.** Conversation slots: `SAGUARO_GLM_SLOTS` (4, 0 off), ≥`SAGUARO_GLM_SLOT_MIN` (1024) tokens,
   files under `SAGUARO_GLM_SLOT_DIR` or in RAM with `SAGUARO_GLM_SLOT_RAM_GB` (:1386-1430); the parked-conversations
   room is an INFO fact (`parked=` :1146-1157). Cost is printed per park: `saved a N-token conversation (GB in RAM in
   S s)` (:1644-1648) / disk form (:1698-1701). Checkpoints (`SAGUARO_GLM_MULTI_CKPTS`, off by default) add
   mid-prompt return points per context (:1763-1805).
4. **Multi.** `READY ... multi=<n>` with `SAGUARO_GLM_SEQS`; per-request `STOP <rid>` / `PAUSE <rid> [keep]`
   (:1251-1254, :1303-1317), and the request being admitted has its own stop flag so its read alone ends
   (`admitting_rid`/`admit_stop` :1255-1266, with `SAGUARO_GLM_PREFILL_INTERLEAVE/_GAPS` letting the others decode
   between the pieces). A pause landing inside a read gap is deferred until the read call ends (`deferred_pauses`,
   :3085-3093, :3196-3202, :3232). Pause works per conversation in multi.
5. **Python 3.12.** No engine-specific wrapper generator: `StrataEngine.generate` only `yield from`s
   `_generate_multi` (`serve/server.py:2027`) — `yield from` propagates `close()`, safe.

## 2. DeepSeek-V4.1-Flash (saguaro-ds4) — STOP yes, PAUSE no; a stopped read is DISCARDED

1. **Mid-answer.** STOP honoured at every token, solo (`src/ds4/ds4_main.cpp:1929-1933`) and per rid in `--parallel`
   (:2704-2707; `stop_rids` :172-178, :215-218). READY offers `stop` only, never `pause` (:750-753) — a priority
   pause today is a **cut**: STOP + drain, resume with prompt+written; the seq's checkpoint stands at the prompt's
   end ("a stop within accepted drafts leaves the state past it ... goes back to the checkpoint at this prompt's
   end", :1989-1990), so the answer's written tokens **are re-read** (cheap — read speed ~30x decode — but not
   "nothing re-generated").
2. **Mid-read.** A STOP between pieces stops the read (:1717-1721), but the partly-read prompt is then **thrown
   away**: the conversation is restored to the base `from` before the prompt, else reset, and `held` cleared
   (:1805-1818, DONE with `reused=from`). No PAUSE offered. The server therefore never cuts a DS4 read (:5725-5729) —
   today a long DS4 read cannot be preempted at all; a waiting high request sits behind it (the drain comment at
   `serve/server.py:2129-2134`: STOP is seen between chunks, a deep 32K chunk runs a minute).
   Restore points (`marks=`/`shared=`, :530-531, :1723-1744) already keep states *at chosen positions* — the
   mechanism a read-keep would build on.
3. **Park/unpark.** `SAGUARO_DS4_SLOT_KEEP=1`: conversations parked as slots (state, compressed KV) in RAM with a
   folder overflow (`SAGUARO_DS4_SLOT_DIR/_CAP 4/_MIN 1024/_GB 16`), :814-860; park prints its cost:
   `saved a N-token conversation (GB in RAM in S s)` (:1085-1089). LRU park when another conversation takes a seq
   (:2036-2037).
4. **Multi.** `--parallel n` → `READY ... multi=n` (:750-751); n conversations via `seq_bind`, reads take turns a
   piece each, a bare STOP stops every request while `STOP <rid>` stops one (:215-218, :2032-2048). A pause (cut)
   touches only the cut request; its seq keeps the prompt-end checkpoint.
5. **Python 3.12.** Same shared driver as GLM — nothing DS4-specific; the shared API-layer wrappers are in §6.

## 3. MiniMax-M3.1-Flash (saguaro-mm3) — STOP yes, PAUSE no; a stopped read keeps (unused)

1. **Mid-answer.** STOP per token solo (:849-850, :965-966) and per rid multi (:1158-1159; `stopped(rid)`
   :210-219, :252-256). READY: `stop` only (:601, :1042). Priority pause = **cut**; the sequence's cache holds the
   prompt (fed tokens pushed as they run, :1197), so a resume re-reads the written tokens only.
2. **Mid-read.** A STOP between chunks ends the read (:740-743 solo, :1449-1451 multi) and — unlike DS4 — the
   cancel paths keep what was fed: `state.insert` per chunk (:737, :1442) and **no `state.clear()`** on the cancel
   paths (:758-762 solo, :1473-1478 multi); only an engine error clears (:754, :1467). So a stopped read already
   keeps the conversation up to the chunk boundary, at the DONE's reported `reused=from` figure; but the server
   never exercises it (`can_pause` false, `keeps_stopped_read` unset → read runs to its end, :5725-5729). The one
   caveat: a STOP during the *last* chunk is not taken (:740 `&& !last`) — the read finishes, the cut lands in decode.
3. **Park/unpark.** Folder slots (`SAGUARO_MM3_SLOT_DIR`, KV streaming only, `Model::state_ok`): the conversation the
   next request would drop is saved, the best continued one read back (:389-586, :671-676 solo, :1379-1392 multi).
   Cost printed on both sides: "saved a conversation of N tokens (G GB) in S s" (:542-543) / "read back ... in S s"
   (:573-574).
4. **Multi.** `--parallel` → `READY ... multi=n` (:1042); one KV cache per sequence, bound to the idle one the
   prompt extends most, else LRU (:1026-1027); reads run in chunks with a decode step of the others between
   (:1027-1029, :1453-1457); no drafter in multi (:1031, :1077). `STOP <rid>` ends one; a bare STOP all.
5. **Python 3.12.** Nothing MM3-specific; shared wrappers in §6.

## 4. Inkling 2 Mini (saguaro-ik2) — STOP yes, PAUSE explicitly not; the worst read behaviour

1. **Mid-answer.** STOP honoured per token/prompt-pass solo (`g_cancel` :10181-10182, decode checks :10968,
   :11029) and per rid multi (:10406-10419, :10530-10534). READY: `stop` only (:10165-10166); PAUSE/WRAP are
   received-but-ignored by design ("Maya's PAUSE / WRAP and the like: not offered in READY, so not sent", :10612;
   the server also never sends WRAP to ik2, `serve/server.py:2235`). Priority pause = **cut**; hist keeps the
   written fed tokens (:11015-11018, :11057 solo; :10560 multi) and a prompt-end snapshot exists (:10893-10901
   solo, `prompt_done` :10340-10353 multi), so a resume re-reads the written tokens from the snapshot.
2. **Mid-read.** Solo: a STOP between passes goes **back to the latest snapshot**, not to the token reached —
   "a prompt half run: back to a snapshot it kept" (:10817, :10838; handling :10870-10892: restore the best snapshot
   with `0 < pos < ids.size()`, else `reset()`); the pass's partial rows are discarded. Multi is worse: "a reader
   stopped: nothing of its prompt is kept" → `slot_reset` (:10494-10499). No PAUSE. So today: reads run to their
   end, and even a user-Stop would lose the partial read — the opposite of QFN's #1620 solo keep.
3. **Park/unpark.** Snapshots: `SAGUARO_IK2_SNAPS` (8) per live conversation, host buffers of the conv histories +
   windowed KV rings (:9591-9673). Kept conversations: `SAGUARO_IK2_KEEP` (4) in host RAM (`KEEP_GB` 8), overflowing
   to `SAGUARO_IK2_SLOT_DIR` (`SLOT_MIN` 1024, `SLOT_GB` 64, 7-day files) — **solo only** (`conv_ok = g_par <= 1 &&
   !kv_streaming && !kv_q4`, :9674-9711). Park prints its cost (:9949-9973): "a conversation of N tokens set aside
   (M MiB, T ms)"; unpark copies the rows + snapshot back (:9976-10009) — nothing re-read.
4. **Multi.** `--parallel N` only on the single-GPU resident path (every expert in VRAM, ≥4096 positions each, else
   it says so and runs one at a time, :10117-10126); slot state costs "GiB more for their rings and convs"
   (:10127-10132); drafting off in multi (:10253-10258); no kept-conversation park (`conv_ok` false, above). A pause
   (cut) ends one rid; the slot keeps prompt+written in `hist` (:10338, :10560) but the snapshot machinery is the
   only return point.
5. **Python 3.12.** Nothing IK2-specific: it rides `StrataEngine`; the ik2 `gen_keys` filter (:1869-1881) and the
   stats plumbing (:2046, :2118-2120) are not generator wrappers.

## 5. Hy4-Flash (saguaro-hy4) — solo only; STOP keeps reads like MM3, but nothing is exercised

1. **Mid-answer.** STOP per token (:1161-1163; stdin thread :430). READY: `READY <ctx> stop` only (:958) — no
   pause, no `multi=` ever (solo protocol, :58-62). Priority pause = **cut**; `state` keeps the fed tokens
   (:1086 read, :1187/:1202 decode), so the resume re-reads the written tokens against the cached conversation.
2. **Mid-read.** STOP between chunks ends the read (:1092-1095) and the cancel path keeps what was fed (state
   inserted per chunk, **no clear** at :1106-1110; only an error clears, :1101). As with MM3 the server never cuts
   an HY4 read (:5725-5729), so this keep is unexercised. A STOP during the last chunk is not taken (:1092 `&& !last`).
   WRAP exists (:61, :430) — unlike IK2.
3. **Park/unpark.** Folder slots only (`HY4_SLOT_DIR`, not with KV streaming or a drafter loaded — then conversations
   are not saved): save "saved a conversation of N tokens (G GB) in S s" (:886-890), restore "read back ... in S s"
   (:920-922); identity + LRU cap (:736-745, :950-953). One conversation lives in the caches; nothing else is kept
   in RAM.
4. **Multi.** None. The FIFO keeps one place (`multi=0` → capacity 1, `serve/server.py:9917` stays 1), so a paused
   request and every other request share the single conversation slot; a parked conversation is the folder, not a
   live slot.
5. **Python 3.12.** Nothing HY4-specific.

## 6. Q5 across the drivers: generator wrappers that rely on the inner `finally` (the 1.2.3 bug class)

Fixed with `contextlib.closing` in 1.2.3 (commit 88809c4ea, comment "3.12 ran the inner STOP + drain only when the
outer generator was freed"):

* `serve/server.py:6635` — `run_with_mcp` over `svc.run` (all engines).
* `serve/server.py:6995` — `anthropic_events` over `svc.run` (all engines).
* `serve/server.py:9558` — `QfnEngine.generate` over `qfn_engine.StrataEngine.generate` (QFN only).

Correct by other means (safe today):

* `Service._sliced` closes the engine generator explicitly in `finally` (`serve/server.py:5797-5798`).
* `Service.run` closes its engine generator in `finally` (:6451-6454).
* `StrataEngine.generate` → `yield from _generate_multi` (:2027); `qfn_engine.StrataEngine.generate` →
  `yield from generate_batched` (`serve/qfn_engine.py:2428`) — `yield from` propagates `close()`.

**Still relying on the outer being closed + frame GC, not `contextlib.closing`** — same shape as the 1.2.3 bug, on
every engine (these wrap `Service.run`, engine-independent):

* `serve/server.py:6772` — `openai_chunks` iterates `svc.run` (non-MCP chat/completions and the ollama routes) with
  a bare `for`; only the outer `chunks.close()` (:7644, :8535) is guaranteed, the inner relies on refcount drop.
* `serve/responses.py:393` — `response_events` iterates the passed `run` (created at `serve/server.py:8463`) with a
  bare `for`; callers close the outer `events` (:7763, :8440, :8471, :8626) only.

## 7. Park / unpark costs seen in the logs (`../logs`, beside this worktree)

* IK2: `saguaro ik2 sessions: a conversation of 100 tokens set aside (35 MiB, 37 ms)`; 80 tokens `(40 MiB, 28-29 ms)`
  — `../logs/maya/Inkling-2-Mini-UD-Q4_K_XL-gpu01.log` (park is the snapshot/rows copy, ~30 ms at these sizes; the
  bytes scale with context length, not token count alone).
* MM3: `saved a conversation of 9818 tokens (0.77 GB) in 0.5 s`, `read back a saved conversation of 9805 tokens in
  0.3-0.4 s` — `../logs/maya/MiniMax-M3.1-Flash-Maya-S.log`. (HY4's SlotStore prints the same lines, same order.)
* DS4: `ds4 slots: saved a 26272-token conversation ... (0.04 GB in RAM in 0.0 s); 1 in RAM ...` —
  `../logs/maya/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL.log` (RAM park ~free; the folder round-trip measured in PR #932:
  import 69,455 tokens in 1.05 s, first token from the restored slot in 0.27 s vs 21-26 s cold —
  `../logs/opencode-dev/plans/q1/r2-upstream-prs/ref/prs/932.json`).
* GLM: keeps proven live in `../logs/maya/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS.log`: `glm slots: paused at 52600 tokens
  (3282 written): the state is kept there`; `glm slots: a cancelled prompt keeps the 152512 tokens it read
  (152512 new)`.

## 8. Work to reach QFN's level, per engine (smallest first)

"QFN's level": a priority pause never regenerates anything already written; a half-read prompt is kept and resumed;
a high request's first token about a second after it steps in. Estimates are engineering hours incl. tests on the
named engine (mock-engine tests where possible), smallest first:

1. **GLM — ~1-2 h.** Already the reference. Remaining: open the read preemption to the **solo** engine (the engine
   keeps cancelled reads, `generate.cpp:2256-2304`, but `serve/server.py:5726-5729` gates on `multi`) — one gate +
   a verification test that a solo PAUSE inside a read does not lose the read (the reason it was gated, :5720-5722).
2. **HY4 — ~3-4 h.** Reads already survive a cut between chunks (:1106-1110). Work: set `keeps_stopped_read` for
   `engine_kind == "hy4"` in the driver gate (:5726-5729), confirm the DONE `reused` parity for a resumed cut read,
   test the cut→queue→resume round trip. (The last-chunk STOP gap, `&& !last`, costs ≤ one chunk of the ~1 s goal;
   note it, don't fix.) No multi to worry about.
3. **MM3 — ~4-6 h.** Same keep as HY4 in both solo and multi (:758-762, :1473-1478): driver gate + per-rid cut in
   multi tested (the drain is bounded there, `serve/server.py:2357-2365`), plus one fix in the engine so a STOP for
   a request also ends when it arrives during the **last** chunk path (:740, :1449 — today it falls through to
   decode). Park/unpark already measured (§7); no engine-side keep needed.
4. **DS4 — ~6-9 h.** The cut **discards** the half-read prompt (`ds4_main.cpp:1805-1818`). Work: on a stopped read,
   keep the read's own last checkpoint (the `marks`/`feed` machinery already stores states at cut points,
   :1723-1744) or at least the whole-piece end, and report it; driver: `keeps_stopped_read` for ds4 when the engine
   declares a READY capability (new token, e.g. `keepread`), so the server keeps gating on truth. Solo + `--parallel`
   share `prompt_piece`, so one engine change covers both. Drain per chunk fits the ~1 s goal.
5. **IK2 — ~10-14 h.** The deepest gap. Solo a cut read rolls back to the last snapshot (lose ≤ one pass back),
   multi **drops everything** (:10494-10499). Work: per-slot snapshot ring so a stopped read restores at the chunk
   it stopped (extend `Snap`/`snap_copy` :9591-9673 to per-slot checkpointing), stop the discard in multi
   (`slot_reset` → restore-to-snapshot), a READY capability + driver gate, and tests solo + `--parallel`; keep the
   resident-path constraint (:10117-10126) in the test matrix. (Conversation *parking* between chats is fine solo;
   missing with `--parallel` — `conv_ok` at :9706 — but that is park/unpark, not priority, and only second-order
   for the pause path.)

Plus, engine-independent (counts once, ~1-2 h): wrap `openai_chunks`'s inner and `response_events`'s `run` in
`contextlib.closing` (§6) so the STOP+drain of every engine's cut is prompt on Python 3.12 — the "first token in
about a second" of a high request depends on the cut's drain completing before the next grant.

## 9. Tests run

None: the plan is report-only ("REPORT ONLY, no code changes") and names no test modules. Nothing in `serve/` or
`src/` was touched.

## 10. Open questions

1. Solo GLM read preemption (:5720-5722 says a solo PAUSE "may be ignored or lose the read") — is that still true
   of the current engine, or can the gate simply drop `multi` for GLM? Needs a live GLM run, not visible from code.
2. MM3/HY4 keep a cut read silently, but their cancelled-read DONE reports `reused=from` (the *old* base,
   `mm3_main.cpp:759`, `hy4_main.cpp:1107`), not the partial end — is the partial state trusted for *resumed
   sampling* (penalty windows, drafts) as well as for KV? A parity test should decide before flipping
   `keeps_stopped_read`.
3. Should a `PAUSE`-less engine get a real PAUSE (like GLM's snapshot move) rather than a cut? It removes the
   written-tokens re-read on every resume — worth it on DS4/MM3 where the checkpoint machinery exists, questionable
   on IK2/HY4 solo.
4. HY4 has no `--parallel` and no in-RAM conversation keep — a paused HY4 request that loses its cache slot to the
   next request re-reads from the folder (or cold). Acceptable for the 1.2.4 goal, or park-on-pause first?
5. The §6 unfixed wrappers were not flagged in 1.2.3's review — deliberate (chat/responses streams close the outer
   from the socket layer) or an oversight the same test (`StopReachesTheEngine`, `serve/test_priority.py`) would
   catch?
