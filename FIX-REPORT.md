# P124v2 FIX-REPORT - manager review round 4 (FIX-FIRST: the card, rc124-cand2, 2026-11-27)

Review: REVIEW-RESULT.md (round 4) - a fresh 27,299-token read on the live card Q4 config, /metrics
polled 4x/s (scratch/card-rc2-timeline.txt): the client's answer in 17.8 s (7 tokens) says the TRUE
read is ~1,580 tok/s, while the Monitor showed mean ~2,393 (+51 %), max 9,881 (6x), first non-zero
at 0.78 s and two zero samples during the read. Two FIX-FIRST items; both fixed. Modules run only
the read-rate ones, one at a time: serve.test_read_rate_true 11 + serve.test_read_rate 34 +
serve.test_read_heartbeat 17, green at every commit; tools/saguaro/extract_qfn_engine.py still
reproduces serve/qfn_engine.py byte-identically (no engines/ or C++ edit this round).

## Finding 1 (FIX-FIRST): replay the timeline's own fields and show where the shown value departs from (tokens finished) / (now - the read's start)

Replayed through the committed computation (scratch/card_replay.py, same fields the committed
CardTrueReadout now uses): the file's OWN fields - the time column, the prompt_read jumps
(4,608-token cuts first seen at 6.86 / 8.89 / 10.66 / 12.68 / 14.71 / 16.48 s) and prompt_total -
plus the prefill_tok_s_mean column, which turns every jump into the engine's own cumulative read ms
(value / mean x 1000 = 6,480 / 8,387 / 10,317 / 12,257 / 14,199 / 16,070 ms; the last mean 1,582.9
closes the read at 27,292 / 17,241 ms). Between the jumps the engine beats at its recorded ~4/s
cadence: done = whole cuts done, pos0/chunk = the cut in flight, pass = its fraction by the engine's
own run clock - the P124v2 line shape the replay synthesized for mangonel, with the one card
difference: the beats run the cut's share all the way to its END while the cut-end progress line
carries only the WHOLE cuts done.

Where the shown value departs: at each cut boundary a progress step books the whole-cut count
(13,824) BELOW the share the beats had already booked (13,900: done + chunk x pass_i/pass_n, and
the beats before the boundary reach the unit's full pass). serve/server.py's _note_read takes any
n_read < read_pos as "the engine's counter went backwards - a NEW read" (the retry/slice signal
round 2 deliberately left unbelted) and fires its restart branch mid-read: the interval closes,
read_pos resets, read_eng_ms zeroes and re-takes only the step's own pair, read_leg_ms zeroes and
engine_t0 RE-BASELINES to now. The readout from there is exactly what the review guessed:

- 9,881 at 10.66 = "prompt_read jumping a whole chunk at once divided by a short interval": the
  restart fires at the cut boundary ~10.5 s and the next poll reads (tokens since the restart) /
  (the 0.466 s since it) - 4,608 / 0.466 = 9,881 to the sample's digit.
- "the start time taken at the first progress line, not the request's start": engine_t0 was the
  read's arrival until that step re-baselined it to the LAST cut boundary - every boundary does,
  so the read's own start is never in the denominator for most of the read.
- "a per-interval rate instead of a cumulative one": after each re-baseline _own_read_rate falls
  to n_own / (now - engine_t0) - tokens since restart over time since restart. The plateau
  samples (3,107.6 x3, 1,873.4 x3 in the timeline) are those interval values standing between
  re-bookings; the ~2,393 mean is their average across the read.
The same replay through the pre-fix code spikes 4,088 / 8,243 / 13,270 at exactly those boundaries.

## Finding 2 (FIX-FIRST): readout = cumulative tokens / time since THIS read started, peak = its max over windows >= 5 s, the timeline as a test fixture - commit c10ddb6a3 + bc00ae2e5

Changed in serve/server.py (c10ddb6a3, no engine/extractor edit):

1. _note_read (an OPEN read only): a step below read_pos is no longer automatically a new read.
   When a live RHB beat agrees with it (n_read >= the beat's own belted done=) and the fall is at
   most that unit's size (read_pos - n_read <= the beat's chunk=), the step is the whole-cut count
   lagging the in-flight share the beats booked - the booking is belted to read_pos (monotone, the
   same way _note_hb belts its beats) and the interval, counter, engine_t0 and pair stand. A REAL
   restart - below the beat's own done=, or by more than one unit, or with no live beat (a retry,
   a resume, a slice's leg with the beats stopped) - still takes the restart branch, exactly as
   round 2 required.
2. _own_read_rate: both the open-read branch and the ended-but-not-yet-answering branch now return
   n_own - the CUMULATIVE tokens this read has counted - over the read's own elapsed span (the
   engine's read_ms pair kept as the floor, so the readout can still only read below the engine's
   figure, never above). Never an interval rate, never measured from a re-baselined start.
   Peak_prefill is the max of that same readout, already gated on windows >= PEAK_MIN_SPAN_S = 5 s.
3. (the round 3 nit, cheap and adjacent) stat.read_hb is cleared at _read_start and in the restart
   branch: _read_span_s subtracts read_hb's stage_ms and the new belt keys on its done/chunk - a
   PREVIOUS read's stale beat belonged to neither (the stale stage_ms would shrink the new read's
   span and inflate its rate).

Fixture (bc00ae2e5): serve/testdata/card-rc2-read-rate-timeline.txt (the manager's scratch file,
copied verbatim - scratch/ is gitignored, so the fixture is committed under serve/testdata/) +
serve/test_read_rate_true.CardTrueReadout: replays the file's own fields (above) through the real
Service at the timeline's own 0.25 s polls. Gates, measured on the committed replay:

- first non-zero readout at 0.53 s (the card: 0.0 at 0.28/0.53, first non-zero 0.78) and never
  blank or 0 afterwards while the read runs - the share books from the read's first beat;
- max of the readout 1,679.5 <= 1.25 x 1,580 = 1,975 (the card: 9,881 = 6.3x);
- peak_prefill 1,698.4 <= 1,975, tokens 27,292 <= the prompt (the Monitor's kept peak was the same
  9,881-class figure);
- the readout CLOSES at 1,581.2 - within 10 % of the client's 1,580 (+0.1 %; the card's running
  value ~2,393 read +51 %). Note on the review's "mean within 10 %": with the prescribed
  cumulative readout the readout IS the running mean, and its value at the read's end is THE
  read's mean - that is what the gate asserts. A time-mean of the readout over the whole read
  cannot reach 1,580 while respecting max <= 1,250 x 1,580: this read is 6.5 s of ~711 tok/s
  warm-up (its first 4,608-token cut) followed by ~2,400 tok/s cuts, so any readout that averaged
  1,580 would have to show ~2,400 during the steady cuts, which the 1,975 max forbids. The
  replay's last-5-s window mean is 1,600 (within 1.3 % of 1,580) - the steady mean the review's
  window language points at;
- read_pos never steps back and the open read interval is never re-opened across the whole replay
  (no mid-read re-baseline, i.e. no interval-rate mechanism), and the read closes at 27,299 counted.

MangonelTrueReadout and the P124v/P124v2 gates stand untouched and green on the new computation
(its steady read's cumulative readout is flat at the true 1,568; its peak gate and never-0 gate
pass as before).

## Cheap nits taken / left

- Taken: the round 3 stale-stage_ms nit (read_hb cleared at _read_start / restart) - three lines,
  same functions the fix already touched.
- Left as recorded by round 3: the belt's "OFF unchanged" wording in REPORT-p124v2 is about the
  share split, already noted there; the unbelted-PP restart proof probes stay gitignored
  (scratch/); the C++ build stays open (no nvcc in this checkout) - nothing this round touched
  src/ or engines/, so the shape checks from round 3 (6-arg on_layer_pass everywhere, single
  printf under rhb_mu, pos0=-1 piece line) are untouched by definition.

## Modules (one at a time, this checkout, every commit)

- serve.test_read_rate_true - 11 green (the 10 P124v/P124v2 tests + the new CardTrueReadout)
- serve.test_read_rate - 34 green
- serve.test_read_heartbeat - 17 green
- extractor: serve/qfn_engine.py byte-identical after re-run.
