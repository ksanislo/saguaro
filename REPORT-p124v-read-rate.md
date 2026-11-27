# P124v — the Monitor's prompt-read tok/s is now a live, steady, TRUE readout (Saguaro; Ken 11-26)

## What was wrong (traced end to end, with file:line at HEAD)

The read the Monitor shows ("Prefill", `live.prefill_tok_s`) is produced at every layer, and each layer had a
way to read a real 100-800 tok/s read as a single digit or a sawtooth. The number's full path:

1. **Engine — the lines it prints** (`engines/qfn/src/program/generate.cpp`):
   - `PP <pos> <total> <ms> <tok/s> new_read=<n> read_ms=<ms> rid=<id>` at every prompt-chunk end
     (generate.cpp:11424 the batched callback, :20316 and :20387 the two verify-window readers, :21474 the
     merge-first end). `new_read` counts from where THIS read started — a reused/restored/parked prefix never
     counts (PLAN-qfn-per-request-read-stats) — but `read_ms`/`pp_running_ms()` is the *whole read clock*, which
     also covers the **KV staging passes** (copying the reused prefix RAM→VRAM) that run inside `sp.run`.
   - `RDONE rid= new= ms=` once at the read's end (:21529, and the early/cancel close :22871) — same clock.
   - `RHB rid=… stage=<stage|read> done= new= reused= pass=i/n ms= …` — the U5 heartbeat (:11129 solo, :16462
     batch `rid=0 … slot=`). It printed **at most one line per ~950 ms** and its `ms` was the same staging-
     inclusive read clock; `done` is the read's shared U5b progress figure.
   - Batch slot lines `BRP <slot> <done> <total> <ms> <tok/s> new_read= read_ms=` (:16801, :17537) and
     `BRA <slot> 1 prompt= reused= read_ms=` (:17058, :17796) carry the same read clock.

   **Where the single digits came from:** a read that first stages ~50K restored tokens spends most of its
   read clock staging. When its first `read`-stage beats finally land, `done` is small but `ms` (and the PP
   `read_ms`) already include all that staging time, so `done/ms` — the figure every consumer divides — is a
   fraction of the real 100-800 tok/s. On the P40 and RX 9070 XT boxes that showed as **1-9 tok/s**.
   **Where the sawtooth came from:** on the server the engine-total figure was a 5 s window
   (`RATE_WINDOW_S`) of read steps divided by the window's real span. With the ~1 s heartbeat cadence and chunk
   -end-only progress (4× T4: a chunk every many seconds), `n` grew in lumps while the denominator grew every
   poll → the figure decayed then jumped, saw-ing around the true rate.

2. **Driver — parse only** (`engines/qfn/serve/server.py`, the source of the generated `serve/qfn_engine.py`):
   `StrataEngine._rhb` (:1686) keeps the latest heartbeat per reader in `read_live` (`{"solo":…, "slots":…}`,
   :1406/:1414 route RHB/BRA) and computes `entry["rate"]` as the tokens between two beats over the time between
   them (the beat `ms`, staging included). `_pp_new` (:1649) and `_rdone` (:1671) lift `new_read`/`read_ms` and
   RDONE's `new`/`ms` for the shared server to take.

3. **Server adapter** (`serve/server.py`, class `QfnEngine`): solo `generate` (:9716) forwards each solo RHB via
   `on_hb` and each PP/RDONE via `on_read` (`cb(new, rate, exact, prov, rms)`); the batch `_admission_wait`
   (:9767) forwards the slot's BRP (`new_read=`/`read_ms=`) and BRA (`prompt-reused`/`read_ms`) plus the slot's
   RHB beats. Before this change those clocks still carried the staging time.

4. **Service** (`serve/server.py`): `_note_hb` (:4571) books the beat's `done` and its `ms`; `_note_read`
   (:4588) pairs tokens with the engine's `read_ms` into `read_eng_n`/`read_eng_ms`; `_own_read_rate` (:4502) is
   one chat's rate; `_prefill_tok_s` (:4383) is the engine total. `/metrics` exposes `live.prefill_tok_s`
    (:4980) and each chat row's `prefill_tok_s` (:4753).

5. **Monitor** (`serve/web/monitor.js`, `Throughput`): reads `live.prefill_tok_s` and the per-chat
   `c.prefill_tok_s` straight from the server (no client-side averaging), with the peaks from
   `totals.peak_decode`/`peak_prefill`.

**Reproduced before fixing** with recorded RHB/BRP line sequences in `serve/test_read_rate_true.py` (step 1,
committed red): 2.4 tok/s on a real 100 tok/s read after 40 s of staging, a stalled read frozen at 400,
sparse chunk-end steps saw-ing 0 → 3276 → 0 around a true 410, and two reads totalling 640 not 800.

*Note:* the two `ref/mangonel-*-30k.txt` timelines the plan named are **empty / absent** in this worktree
(`ref/mangonel-monitor-30k.txt` is 0 lines; `ref/mangonel-engine-30k.txt` does not exist), so they could not be
explained value-by-value. The reproduction uses the mechanism-faithful recorded sequences above instead; a
re-recorded timeline on the box is an open item below.

## What changed (step by step)

### Step 1 — trace + reproduction (`9e1b4bcd8`)
- `serve/test_read_rate_true.py`: 5 recorded-sequence tests driving the real QFN adapter + real `Service`
  (a P40-like slow 100 tok/s read after 50K restore, an 800 tok/s fast read, sparse chunk-end steps, two reads
  at once, a stalled read). Committed red (all 5 failing), documenting each wrong value.

### Step 2 — the engine reports a steady per-read figure with staging split out (`ec6def7f4`)
`engines/qfn/src/program/generate.cpp` (output only, old positional fields untouched):
- RHB cadence **950 ms → 230 ms** — a steady ~4 beats/second (solo :11126, batch :16460).
- **New `stage_ms=`** on the RHB (:11129, :16462), PP (:11424, :20316, :20387, :21474), RDONE (:21529, :22871),
  BRP (:16801, :17537) and BRA (:17058, :17796) lines: the part of the read's own clock spent **staging** the
  reused prefix (host-point attributed, 2 s cap on a chain broken by a hold). `ms - stage_ms` is the read's own
  elapsed **READ** time.
- The batch RHB `ms=` is now the read's own clock over its legs (matches BRP `read_ms`).
- `engines/qfn/serve/server.py` (`_rhb` :1686, `_pp_new` :1649, `_rdone` :1671) parses `stage_ms` and computes
  the beat rate / PP / RDONE over `ms - stage_ms` (floored at 0); `serve/qfn_engine.py` re-extracted with
  `tools/saguaro/extract_qfn_engine.py` (never hand-edited).
- `serve/test_read_heartbeat.py` engine-side contract updated for the new format + a staging-clock driver test.

### Step 3 — the server's rate = (finished + in-flight share) / the read's own elapsed READ time (`268bea420`)
`serve/server.py`:
- `_note_hb` (:4571) pairs the beat's `done` with `rt = ms - stage_ms`, so `read_eng_n / read_eng_ms` is the
  read's own rate with no staging / restore / queue / decode time; the beat's interval rate no longer feeds a
  peak.
- `_own_read_rate` (:4502): a read the engine beats for that goes quiet **falls** with the silence
  (`READ_BEAT_STALE_S` grace, :294) and drains to nothing — never a frozen last-beat figure; a read that ended
  ages out. The fallback for engines before the fields drains once the window is empty.
- `_prefill_tok_s` (:4383): once the engine reports its own figures this is the **sum over the reads running
  now** of each read's true rate — the engine's ~4/s cadence is the only smoothing, no client-side window that
  hides a stall. The old 5 s window spread stays the fallback for engines before the fields.
- The QFN adapter's BRP/BRA `read_ms` now subtract `stage_ms` before use.

### Step 4 — tests + CUDA sm_75 build (`77cf4334b`)
- `serve/test_read_rate_true.py` (5) + the step-1 scenarios run green.
- QFN built for **CUDA sm_75** in a new dir `scratch/build-p124v-sm75` (`saguaro-qfn` linked, 0 errors, no new
  warnings; no GPU runs). Fix in the committed tree: the batched-pieces BRP reads `stage_us` from the job.

### Also — the PEAKS (`dbbcad439`)
Ken (11-26): the Monitor showed `peak_decode 94.7 tok/s (4668 tok)` and `peak_prefill 171,860.8 tok/s
("859,304 tok")`; the engine's own per-request log lines on that box (`ref/mangonel-peaks-log.txt`) support a
best single answer of **64.3 tok/s** and a best read of **~1,300-2,128 tok/s**.
- **What the OLD peaks were computed from** (no silent reset — a running server keeps its old
  `totals.peak_decode`/`peak_prefill` until it restarts; line numbers are the pre-P124v file at step 1's parent):
  `peak_decode` was the **summed** live Decode over all requests writing at once (`serve/server.py:4338`
  `_tok_s` `rate = sum(...)`, gated only by `now - first_token >= PEAK_MIN_SPAN_S` = 2 s), so two ~64 tok/s chats
  became one 128 "answer", and the 2 s gate read a young burst as a rate — the 94.7/4668 figure. `peak_prefill`
  was the **5 s window** `n/span` (`serve/server.py:4368` `_prefill_tok_s`, peaking at the old `:4424`
  `_note_peak_read(total, int(round(n)), now)`) plus the per-request engine means recorded directly in
  `_note_read` (old `:4563`/`:4590`); the window summed every read in it and, on engines before the per-request
  fields, the RESUME / position-delta paths counted the restored prefix as read — so "859,304 tok at 171,860"
  was that aggregate divided by its ~5 s span, not one read's speed, counting what the reads restored.
- **New rule** (`serve/server.py`): `PEAK_MIN_SPAN_S 2 → 5 s` (:296) and `_note_peak_read` (:4676) takes the
  read's own `span` and refuses windows shorter than `PEAK_MIN_SPAN_S` or fewer than `PEAK_MIN_READ_TOKENS`;
  a read's peak counts only `read_eng_n` (tokens actually read, staging left out — never reused/restored).
- **Per answer and engine total kept separate and labelled**: `peak_decode`/`peak_prefill` are now only ever
  ONE answer's / ONE read's own readout; the summed-over-chats figures move to new `peak_decode_engine` /
  `peak_prefill_engine` (`_tok_s` :4338 records each answer's own window rate on the per-answer key and the sum
  on the engine key; `_prefill_tok_s` records the summed window as engine-total). The Monitor
  (`serve/web/monitor.js` Throughput) shows e.g. `Peak 64.3 tok/s · all chats together 128.7` with the 5 s /
  actually-read tooltip; Prometheus gains `maya:decode_engine_peak_tokens_seconds` and
  `maya:prefill_engine_peak_tokens_seconds` beside the per-answer gauges (`serve/server.py:5113`).
- **Tests** with the recorded log figures (`serve/test_read_rate_true.py::PeaksTellTheTruth`,
  `serve/test_read_rate.py::PeakPrefill/PrometheusPeak`): two 64 tok/s answers never become a 94.7 per-answer
  peak (64.3 per answer, 128.7 engine); a restored read peaks from the 8.2K it actually read, not the 58K it
  touched; a 2128-class read stays at its own true rate; the summed figure only ever lands on the labelled
  engine key.

## Files changed
- `engines/qfn/src/program/generate.cpp` — RHB cadence, `stage_ms=` on the read lines, batch RHB clock (step 2, 4).
- `engines/qfn/serve/server.py` — driver parses `stage_ms`, rate over `ms - stage_ms` (step 2).
- `serve/qfn_engine.py` — re-extracted from the above (step 2; GENERATED, not hand-edited).
- `serve/server.py` — per-read true readout, stall-falls, engine-total sum, split + gated peaks, constants
  (`READ_BEAT_STALE_S`, `PEAK_MIN_SPAN_S=5`, `peak_decode_engine`/`peak_prefill_engine`), Prometheus gauges,
  QFN adapter `stage_ms` (steps 3, PEAKS).
- `serve/web/monitor.js` — per-answer vs engine-total peaks labelled; the live read is unchanged (server-fed).
- `serve/test_read_rate_true.py` — new: 5 reproduction/readout tests + `PeaksTellTheTruth` (steps 1, 4, PEAKS).
- `serve/test_read_heartbeat.py` — engine-side contract + staging-clock driver test (step 2).
- `serve/test_read_rate.py` — 3 window-semantics assertions and the peak tests rewritten to the true readout
  (steps 3, PEAKS).

## Tests run and results (each named serve module, one at a time)
- `serve.test_read_rate_true` — **9 OK** (5 readout + 4 peak).
- `serve.test_read_rate` — **33 OK**.
- `serve.test_read_heartbeat` — **18 OK**.
- `serve.test_priority` — 68 OK. `serve.test_multi` + `serve.test_lost_read` — 18 OK.
- `serve.test_server` + `serve.test_right_now` + `serve.test_conversations` — 200 OK. `serve.test_dashboard` — 85 OK.
- `node --check serve/web/monitor.js` — clean. CUDA sm_75 `strata` target — builds clean, `saguaro-qfn` links.
- Pre-existing, unrelated to this work: the engine tree's `engines/qfn/serve/test_perreq_stats.py` fails to
  import under this venv (`cannot import name 'turbo_live' from 'serve'`), and that same failure reproduces with
  the worktree stashed — the CI (`.github/workflows/tests.yml`) only discovers `serve/` and `tools/`, not
  `engines/qfn/serve`, so it does not gate here.

## Open questions
1. **A re-recorded timeline is the real acceptance test.** The plan's `ref/mangonel-monitor-30k.txt` is empty and
   `ref/mangonel-engine-30k.txt` is absent here, so the fix is validated against mechanism-faithful recorded
   sequences, not the box's own live timeline. Ken should re-record one fresh ~30K read on mangonel and the P40
   box with the rebuilt sm_75 engine: the readout must be steady and match the engine's real rate, and neither
   peak may exceed what the per-request log lines support. The RX 9070 XT / P40 single-digit captures the plan
   said the manager would add had not landed in `ref/` when this ran.
2. **`stage_ms` host-point attribution across a multi-card split.** It is exact on a single card (P40, RX 9070
   XT) and an approximation across a layer split's pipeline (a held pass completing after a wait is capped at
   2 s and floored at 0 server-side). Confirm on mangonel that a split read's restored-read rate reads ~true, not
   still diluted, once real staging is measured rather than folded into `read_ms`.
3. **Running servers keep their old peaks** (a silent reset would hide history). The corrected per-answer vs
   engine-total split and the 5 s / actually-read gate apply from the next start. If Ken wants the current
   `peak_prefill=171860` / `peak_decode=94.7` cleared rather than kept, that is a one-line reset in `/metrics`
   totals or a restart — say which.
