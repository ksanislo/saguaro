# REVIEW-RESULT — P124v2 (read b7ed679..b750d48e6, round 2; PLAN-p124v2-read-rate-true.md / REPORT-p124v2-read-rate-true.md / FIX-REPORT.md)

## Verdict: FIX-FIRST — one correctness regression (finding 1); everything else holds.

Manager's gate: the committed replay `MangonelTrueReadout` (serve/test_read_rate_true.py:276-355) is green as
run here (10/10, 81 s): never 0 after the read's first RHB (:334-337), time-average |avg − 1,567.8| < 5 %
(:342-346, measured 1,560.3, −0.5 %), peak ≤ 1.25× (:347-352), closes at exactly 27,254 (:355). The recorded
file (ref/mangonel-engine-after-p124v.txt, one summary line — verified) holds no RHB beats, so the replay shape
is derived from that line; disclosed in REPORT:86-91 and FIX-REPORT §5, which is the right call.

## 1. MUST-FIX — the share booking is not belted; on a layer split it can step the read counter backwards
   and fire the "new read" branch (serve/server.py:4606-4610, :4677-4687)

`_note_hb` books the absolute count `n = int(done + min(chunk, new−done) × pass_i/pass_n)` straight into
`_note_read`. The `done` field alone is belted (engines/qfn/serve/server.py:1762, mirrored to
serve/qfn_engine.py:1318); `done + share` is not. Each card's beat carries its own `pass=<i>/<n>` of its own
layer range (generate.cpp:16990-16993, :17762-17764 -> rhb_slot :16657), and ReadProgress withholds
part-credit for the chunk a card is working on (read_progress.hpp:88-93) — so while the shared `done` stands
exactly at the unit start, the guard at serve/server.py:4608 holds for every card reporting that unit, and the
lead card's beat books `done + chunk×77/78` while the next beat — the lagging card's lower `pass=`, or either
card's next beat once `done` has moved off the boundary and the guard fails (`pos0 − reused < done`) — books
the shared figure only. A booked value below `stat.read_pos` is the engine-counter-went-backwards signal, so
`_note_read` closes the interval and restarts the read (serve/server.py:4677-4687): the closed interval keeps
the inflated count and `_read_counted` sums closed + open (serve/server.py:4552-4554), `engine_t0`
re-baselines mid-read, `read_leg_ms` zeroes — the exact span-collapse and token inflation this plan set out
to kill.

Demonstrated through the real `_rhb` + `_note_hb` + `_note_read` pipeline (scratch/probe_split_share.py, two
cards beating the same in-flight unit, done=4608 pos0=4608, pass 40/78 then 10/78):

    after pass-40 beat:  read_pos=6971  reads=1  counted=6971   engine_t0=1000.0
    after pass-10 beat:  read_pos=5198  reads=2  counted=12169  engine_t0=1000.46   <- restart branch fired
    row rate: 1921 tok/s against an engine pair reading ~1,568

Reachability: `p0=` is the current chunk's start (prefill.cpp:3133, :5436-5440), so the same-unit window is
the chunk-handoff overlap between stages (read_progress.hpp:9-17 documents cards beating at the same unit at
handoffs) — narrow (~one layer pass) but recurring every chunk on split boxes (the U5b 4x T4 configs).
Single-card reads (the recorded mangonel) cannot regress: `done == pos0 − reused` holds there only while the
engine's own part-credit is zero, and `pass=` rises monotonically within the unit — which is why the replay
stays monotone and green.

Fix: belt the booking like the engine belted `done` — book the beat as `max(n, stat.read_pos)` (or only add
the share when it does not step back). Real restarts keep working: `_rhb` resets the belt on a new `new`/`rid`
(engines/qfn/serve/server.py:1759-1762), and a same-`new` retry's beats can never legitimately land below
the belt anyway; the counter-decrease path stays for PP/BRP/slice lines, which are not belted.

## 2. NIT — the protocol comments call pos0=/chunk= optional; the prints emit them unconditionally
   (generate.cpp:8395-8396 vs rhb_print :11328-11331 / rhb_slot :16669-16672; the piece line prints the
   defined `pos0=-1 chunk=0` at :11350-11351). Readers tolerate absence (engines/qfn/serve/server.py:1772-1779),
   so this is doc drift only.

## 3. Observation — the denominator rewrite is not field-gated, only the share is
   `serve/server.py:4520-4546` changes the readout for any engine carrying the new/read_ms pair, not only for
   engines with pos0/chunk: the wall-span floor (`_read_span_s`, :4495-4506) and the silence curve (:4523-4530,
   replacing P124v's `span += gap − READ_BEAT_STALE_S`) also apply to pair-only engines. That is the plan's
   intended fix (the old denominator was the bug), and the untouched P124v suite pins it — 34/34 green. No
   heartbeat line at all: the engine pair is never set for RHB, `_note_hb` never runs, the fallback branch
   (:4541-4546) is byte-identical; the only shared changes are the no-op `int()` on `_read_counted` (:4554)
   and the new `RequestStat.read_leg_ms` field (:3159). Feature OFF = satisfied with that nuance.

## Checks that passed
- Tests really exercise the change: MangonelTrueReadout feeds pos0/chunk lines through the real adapter parse
  (test_read_rate_true.py:62-65 -> `_rhb`) and the real Service; committed red at 2b5e1daaa. The heartbeat
  contract pins the new two-part printf for solo and slot (serve/test_read_heartbeat.py:386-392). Modules run
  one at a time at this tip: 10 + 34 + 17 green; py_compile clean on all five touched .py files.
- Generated file: `tools/saguaro/extract_qfn_engine.py` re-run here reproduces serve/qfn_engine.py
  byte-identically (working tree clean afterwards); the _rhb change went source -> generator -> output.
- C++ sites: all `on_layer_pass` declarations/assignments/uses are the 6-arg type
  (prefill.hpp:70/:341, prefill.cpp:5439/:5440, generate.cpp:11704/:16942/:16991/:17762); the FIX-REPORT's
  compile probes covered the vector moves. `T`/`p0` at prefill.cpp:5440 are the current chunk's length and
  absolute start (prefill.cpp:3133); batch `sg.n` is the job's slice of the current chunk (prefill.hpp:78-82),
  matching what `done=` counts. No nvcc here — the card build stays owed (REPORT open Q1).
- Locks/threads/stdout: no new locks; the RHB line stays one `printf` under `rhb_mu`
  (generate.cpp:11323-11331, :16666-16675 — two-part literals concatenate at compile time, no new
  interleaving); `ReadProgress::mu_` is not held across the print; `_note_hb` takes no lock and books through
  `_note_read`'s single status_lock acquisition (:4640); `_read_span_s`/`_own_read_rate` run under the
  status_lock held by `_prefill_tok_s` (:4390) and only read. No nested acquire of the non-reentrant lock.
- Paths/secrets: no /data, /tmp or /opt paths and no secrets in any added line (the two /opt hits are the
  sanctioned interpreter quoted in report prose); scratch stayed in ./scratch/ (gitignored).
