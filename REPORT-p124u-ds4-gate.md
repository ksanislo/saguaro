# P124u — request priority: the server cuts DeepSeek's prompt reads when the engine says it keeps them (SERVER PART ONLY; Saguaro 1.2.4)

ref/REPORT-p124k-ds4-keep-read.md §5's open question 1, as P124i (Hy4) and P124j (MM3) were for their engines.
Server side only: `serve/` touched, nothing under `src/` or `engines/`; `serve/qfn_engine.py` untouched (nothing in
`engines/qfn/serve/server.py` changed, so the extractor was not owed and was not run).

## 1. Step 1 — the driver reads the engine's word (serve/server.py)

The line driver's READY parsing (`StrataEngine.__init__`, where the words set `can_stop`/`can_pause`/`multi`,
server.py:1515-1527) gained one test, after the mm3 line (P124j):

* server.py:1525-1526: `if "keepread" in f[2:]: self.keeps_stopped_read = True` — any engine whose READY carries the
  word gets the flag; the word is the truth, not the engine name (the READY parser already ignored unknown words,
  so the new token itself was always safe to say, P124k §1). DS4 says `READY <ctx> stop [multi=n] keepread` only
  with its keep on (`SAGUARO_DS4_KEEP_READ`, P124k §2), so with the keep off it still says no word and nothing is
  cut — the gate keys on truth, which was P124h's open question of the same shape: **answered, the word**.

The read-preemption gate itself (`Service._sliced`, server.py:5753-5767) is unchanged: DS4 has no `pause`, so it
takes the second clause — `keeps_stopped_read and not batch` — and the `cut = True; break` branch (5765-5767), a
STOP rather than a PAUSE, which is what the P124k keep is built for (the stopped branch lands the conversation at
the deepest stored point and DONE says it as its `reused`).

## 2. Step 2 — the multi cut ends ONE request (verified, no file change)

`saguaro-ds4 --parallel` says `READY ... stop multi=n keepread` (P124k §2): `self.multi` is parsed from the same
READY line (server.py:1520), so its requests ride `_generate_multi` (dispatch at server.py:2030-2035) and the cut's
`gen.close()` (the `_sliced` finally at server.py:5825) lands in that generator's `finally`, which writes
`STOP {rid}` — this request's rid alone (server.py:2358), drains that rid's route to its DONE (2360-2376) and never
an engine-wide STOP that would end every conversation. This is the exact close P124j proved for mm3 in
`TheMultiCutNamesOneRequest` (serve/test_priority.py, which drives the real `StrataEngine._generate_multi` with a
fake stdin and asserts the last line sent is `STOP 1`); DS4 shares the path, so no code and no new mechanism, only
the test's docstring now names the engine that follows.

## 3. Step 3 — tests (serve/test_priority.py)

* `INFO_KIND_ENGINE` (the driver's fake) now says its READY words: `sys.argv[3]` after the kind, default `stop`, so
  the existing hy4/mm3/ik2 cases send the identical `READY 4096 stop`; `TheReadKeepDriver.start(kind, ready)` passes
  them through.
* `test_the_keepread_word_gives_it_to_any_engine`: READY `stop keepread` → `keeps_stopped_read` (solo: no PAUSE,
  multi 0); READY `stop multi=4 keepread` → multi 4 AND the flag (the per-request `STOP <rid>` cut stays the class
  below's); READY `stop keepread` under a name the driver never heard → the flag, from the word alone.
* `test_an_engine_without_the_word_is_not_given_it`: kind `ds4` but READY without the word
  (`SAGUARO_DS4_KEEP_READ=0`, P124k) → not given it: the name decides nothing on its own.
* `Ds4Read` (the line driver's shape: keeps the cut read, no pause, reports no figure over the pipe) in
  `ReadsPreempt`: `test_a_ds4_solo_read_is_cut_for_a_normal_and_continues_kept` and
  `test_a_ds4_multi_read_is_cut_for_a_normal_and_continues_kept` (multi=2) — a LOW read is stopped for a waiting
  NORMAL one (STOP, not PAUSE), the NORMAL answers first, the LOW one's second call is the same prompt and reads on
  from where the cut left it (ticks == CHUNKS: nothing kept was read again), both log lines naming the pause and
  the numberless continue.

## 4. Files

* `serve/server.py` — the two-line READY gate (step 1). Nothing else.
* `serve/test_priority.py` — the fake's READY words, the three driver cases, `Ds4Read` and the solo/multi Service
  races, the multi-cut class's docstring, the module docstring.
* Not touched: `serve/qfn_engine.py` (no `engines/qfn` change → no extractor run), `src/ds4/` (the engine is
  P124k's, unmerged — nothing here depends on its code, only on its READY word).

## 5. Tests run and results

* `serve.test_priority`: `python -m unittest serve.test_priority` → **Ran 67 tests, OK** (the 4 new cases plus the
  whole R122 suite; run again per class: the 7 `TheReadKeepDriver`+Ds4 cases OK).
* `serve.test_fair`: `python -m unittest serve.test_fair` → **Ran 14 tests, OK**.
* `serve/server.py` parses (`ast.parse`); the venv's Python 3.13.5. The plan named these two modules only; nothing
  else was run.
* Commits: step 1 (server.py), step 3 (tests) — step 2 changed no files (verification); this REPORT
  (force-added: `REPORT-*.md` sits in the checkout's `info/exclude`, as P124k's did).

## 6. Open questions

1. **The continue line never shows DS4's figure**: the engine's DONE carries the keep as its `reused` field
   (P124k), and `StrataEngine._parse_done` puts it in `engine.last["reused"]` (server.py:1854-1855), but the resume
   log reads `engine.reused` (server.py:5717), which only QFN's RESUME path sets — so a resumed DS4 read logs the
   numberless "continues from where the engine kept it". Honest (P124b's rule: no figure the driver did not read
   as its own), but the figure is on the wire; a follow-up could have the line driver's `_sliced` prefer
   `last["reused"]` for a DONE with the conversation fields.
2. **A cut before the engine's first keep point keeps nothing** (P124k §5.2, now live): the server cuts any read
   past its first PP tick regardless of position, so a LOW read stopped inside the first `SAGUARO_DS4_KEEP_READ_STEP`
   re-reads from its base — the engine's `reused=0` says so honestly; the card run of P124k's script will show where
   the cuts land.
3. **`batch` and DS4**: the gate exempts `batch` engines (a QFN slot drops a cancelled read); DS4 is a pure line
   engine with no `batch` attribute, so it takes the cut. Correct today; if a DS4 batching path ever appears, its
   class needs `batch` set (the test `test_a_batch_read_runs_to_its_end_even_for_a_high_request` is the pattern).
4. **Live parity untested**: the whole chain (a real DS4 STOP landing on a stored point, the resumed GEN sharing it)
   is P124k's card script's job with the gate now on — it has not been run here (no GPU, no card model), and its
   PASS criteria assume this gate exists, which it now does.
