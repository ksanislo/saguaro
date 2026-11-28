"""serve/test_read_rate_true.py - P124v: the Monitor's prompt-read tok/s must be a live, steady, TRUE readout.

Recorded engine-line sequences (the shapes mangonel / the P40 box / the RX 9070 XT box produced, see
PLAN-p124v-read-rate.md and ref/mangonel-peaks-*.txt) driven through the real QFN adapter and the real
Service: the live figures (the chat row's prefill_tok_s and /metrics' live.prefill_tok_s) must show the
READ's own true rate - (tokens finished + the in-flight chunk's passed share) over the read's own elapsed
READ time (no queue / restore / KV-staging / decode time) - at the engine's beat cadence, never a single
digit during a read that really runs at 100-800 tok/s, never sawtoothing, never counting a restored prefix,
and two reads at once must sum.  A stalled read's rate must FALL, not freeze at the last beat's value.

    python -m unittest serve.test_read_rate_true -v
"""
from __future__ import annotations

import bisect
import queue
import re
import sys
import threading
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, RequestStat, Service, _qfn_engine_class  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096


def make_svc():
    tok = ByteTokenizer()
    return Service(MockEngine(tok, "x", max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))


def reader(stat):
    stat.started = 1.0
    stat.at_engine = True
    return stat


class ReadingEngine:
    """A fake QFN batch read behind the real adapter: the pump's parse-then-route happens by hand and the
    request waits in _admission_wait exactly as generate_batched leaves it.  run() installs the same
    thread-local hooks on the engine it does for a real one (serve/test_read_heartbeat's harness)."""

    def __init__(self, svc, slots=(7,)):
        self.svc = svc
        self.eng = _qfn_engine_class().__new__(_qfn_engine_class())
        self.eng._tls = threading.local()
        self.eng.slot_q = [queue.Queue() for _ in range(max(slots) + 1)]
        self.eng.read_live = {"solo": None, "slots": {}}
        self.hooked, self.active = {}, slots[0]

    def watch(self, slot, stat):
        self.hooked[slot] = stat
        tls = self.eng._tls
        tls.on_read_start = lambda: self.svc._read_start(self.hooked[self.active])
        tls.on_read = lambda n, ts=None, ex=False, pv=False, rms=None: \
            self.svc._note_read(self.hooked[self.active], n, ts, ex, pv, rms)
        tls.on_hb = lambda d: self.svc._note_hb_safe(self.hooked.get(d["slot"], self.hooked[self.active]), d)

    def feed(self, slot, line):
        if line.startswith("RHB "):
            self.eng._rhb(line)                          # (the pump parses before it routes)
        self.eng.slot_q[slot].put(line)

    def wait(self, slot):
        return self.eng._admission_wait(slot, threading.Event(),
                                         {"state": "reading", "prompt_read": 0, "prompt_tokens": 0})

    def step(self, it, slot=7):
        self.active = slot
        try:
            next(it)
            return None
        except StopIteration as e:
            return e.value


class Base(unittest.TestCase):
    """Mocked clock + a busy server; every read runs on fake slots with recorded engine lines."""

    def setUp(self):
        self.svc = make_svc()
        self.t = [1000.0]
        self.clock = unittest.mock.patch("serve.server.time.time", lambda: self.t[0])
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.svc.status.update(busy=True)

    def tick(self, s=1.0):
        self.t[0] += s

    def reading(self, stat=None):
        stat = reader(stat or RequestStat())
        self.svc.running.add(stat)
        return stat

    def row(self, reused=None):
        rows = [c for c in self.svc.metrics()["live"]["chats"] if c["state"] == "reading"]
        if reused is not None:
            rows = [c for c in rows if c.get("read_reused") == reused]
        self.assertEqual(len(rows), 1, rows)
        return rows[0]

    def live(self):
        return self.svc.metrics()["live"]["prefill_tok_s"]


class SlowReadAfterRestore(Base):
    """The P40 / RX 9070 XT shape: a read that first stages a ~50K-token reused prefix out of RAM (the RHB
    `stage` beats), then reads ~8K new tokens at ~100 tok/s.  During the read the row and the engine-total
    figure must show ~100 tok/s - within 2 s of the read's first beat and NEVER a single digit (live: the
    rows showed 3-9 tok/s because the restored prefix's staging time was mixed into the read's clock)."""

    SEQ = [("stage", 40, 0, 40000),        # 40 beats of staging 50K tokens' KV in from RAM
           ("read", 82, 100.0, 40000)]     # then the read: 100 tok/s for 8.2 s, in 1-s beats

    def beats(self):
        # recorded RHB lines, the engine's own clock in ms (stage_ms = the part of it spent staging the
        # reused prefix; ms includes it - the read's own READ time is ms minus stage_ms)
        for k in range(1, 41):
            yield f"RHB rid=0 stage=stage done=0 new=8192 reused=50000 pass={k % 8 or 8}/8 " \
                  f"ms={k * 1000} stage_ms={k * 1000} slot=7"
        for k in range(1, 83):
            yield f"RHB rid=0 stage=read done={100 * k} new=8192 reused=50000 pass={k % 8 or 8}/8 " \
                  f"ms={40000 + k * 1000} stage_ms=40000 slot=7"

    def test_readout_hits_the_true_rate_and_never_shows_single_digits(self):
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        for i, line in enumerate(self.beats()):
            eng.feed(7, line)
            self.assertIsNone(eng.step(it))
            self.tick()
            row = self.row(reused=50000)
            v = row["prefill_tok_s"]
            if i < 40:                                  # staging: nothing read - the row shows the stage, no rate
                self.assertIsNone(v)
                continue
            # the read itself: never a single digit, within 2 s up to the true ~100 tok/s
            self.assertIsNotNone(v, f"beat {i}: no rate while reading")
            self.assertGreaterEqual(v, 50, f"beat {i}: row rate {v} tok/s during a read that runs at 100")
            if i >= 42:                                 # 2 s into the read: at the true rate
                self.assertLess(v, 150, f"beat {i}: row rate {v} tok/s - well above 100")
                self.assertGreater(v, 70, f"beat {i}: row rate {v} tok/s - under 100")
            if i % 7 == 0:
                live = self.live()
                self.assertIsNotNone(live)
                self.assertGreater(live, 50, f"beat {i}: engine total {live} tok/s")
                self.assertLess(live, 150, f"beat {i}: engine total {live} tok/s")
        # the restored 50K never counted as read: only the 8.2K new tokens were
        self.assertLess(self.svc._read_counted(stat), 9000)
        self.assertEqual(self.svc._read_counted(stat), 8200)


class FastRead(Base):
    """The same shape on a fast card: 800 tok/s (live: the rows read 400-900 erratically, then fell to the
    restored-prefix mean).  The readout must sit at the true rate within 2 s and hold there."""

    def test_800_tok_s_read_shows_800(self):
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        for k in range(1, 21):                           # 5 s of staging, then 20 x 0.5 s beats at 800 tok/s
            eng.feed(7, f"RHB rid=0 stage=stage done=0 new=6000 reused=300000 pass={k}/12 "
                        f"ms={k * 500} stage_ms={k * 500} slot=7")
            self.assertIsNone(eng.step(it))
            self.tick(0.5)
        for k in range(1, 41):
            eng.feed(7, f"RHB rid=0 stage=read done={400 * k} new=6000 reused=300000 pass={k % 12 or 12}/12 "
                        f"ms={10000 + k * 500} stage_ms=10000 slot=7")
            self.assertIsNone(eng.step(it))
            self.tick(0.5)
            row = self.row(reused=300000)
            v = row["prefill_tok_s"]
            self.assertIsNotNone(v)
            self.assertGreater(v, 400, f"beat {k}: row rate {v} tok/s during a read that runs at 800")
            if k >= 4:
                self.assertLess(v, 1000, f"beat {k}: row rate {v} tok/s - well above 800")
                self.assertGreater(v, 600, f"beat {k}: row rate {v} tok/s - under 800")
            live = self.live()
            self.assertIsNotNone(live)
            if k >= 4:
                self.assertGreater(live, 600, f"beat {k}: engine total {live} tok/s")
                self.assertLess(live, 1000, f"beat {k}: engine total {live} tok/s")


class SteadyDuringSparseSteps(Base):
    """The mangonel shape: with progress lines landing only at chunk ends (a BRP every ~20 s on 4x T4), the
    window figure saw 0 - 1638 - 0 - 1638 around the true ~410 tok/s.  The engine's own per-read figures
    (new_read over read_ms) must give a steady readout between the chunk steps instead."""

    def test_chunk_end_steps_give_a_steady_true_rate(self):
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        for c in range(1, 4):                            # three 8K chunks: 8192 tokens, 20 s of read each
            eng.feed(7, f"BRP 7 {8192 * c} 108192 20000.0 409.6 new_read={8192 * c} read_ms={20000 * c}.0")
            self.assertIsNone(eng.step(it))
            for _ in range(8):                           # sample through the quiet stretch after the step
                self.tick(2.5)
                v = self.row()
                rate = v["prefill_tok_s"]
                self.assertIsNotNone(rate, f"chunk {c}: no rate mid-read")
                self.assertGreater(rate, 300, f"chunk {c}: {rate} tok/s - the window drained toward 0")
                self.assertLess(rate, 520, f"chunk {c}: {rate} tok/s - the window's burst value")
                live = self.live()
                self.assertIsNotNone(live, f"chunk {c}: engine total blank mid-read")
                self.assertGreater(live, 300, f"chunk {c}: engine total {live} tok/s")
                self.assertLess(live, 520, f"chunk {c}: engine total {live} tok/s")


class TwoReadsSum(Base):
    """Two reads at once: the engine total is the SUM of the two reads' own rates (300 + 500 = ~800),
    while each chat's row keeps its own - never one shared clock."""

    def test_three_hundred_plus_five_hundred(self):
        eng = ReadingEngine(self.svc, slots=(7, 8))
        a, b = self.reading(), self.reading()
        eng.watch(7, a)
        eng.watch(8, b)
        it7, it8 = eng.wait(7), eng.wait(8)
        for k in range(1, 11):                           # slot 7 at 300 tok/s, slot 8 at 500, 1-s beats
            eng.feed(7, f"RHB rid=0 stage=read done={300 * k} new=3000 reused=1000 pass={k % 4 or 4}/4 "
                        f"ms={k * 1000} stage_ms=0 slot=7")
            eng.feed(8, f"RHB rid=0 stage=read done={500 * k} new=5000 reused=2000 pass={k % 4 or 4}/4 "
                        f"ms={k * 1000} stage_ms=0 slot=8")
            self.assertIsNone(eng.step(it7, 7))
            self.assertIsNone(eng.step(it8, 8))
            self.tick()
            ra = self.row(reused=1000)["prefill_tok_s"]
            rb = self.row(reused=2000)["prefill_tok_s"]
            live = self.live()
            if k >= 3:
                self.assertLess(abs(ra - 300), 60, f"beat {k}: chat a shows {ra}")
                self.assertLess(abs(rb - 500), 100, f"beat {k}: chat b shows {rb}")
                self.assertLess(abs(live - 800), 160, f"beat {k}: engine total {live}, expected ~800")
        eng.feed(7, "BRA 7 1 prompt=4000 reused=1000 read_ms=10000.0 stage_ms=0 wait_ms=1.0")
        eng.feed(8, "BRA 8 1 prompt=7000 reused=2000 read_ms=10000.0 stage_ms=0 wait_ms=1.0")
        self.assertTrue(eng.step(it7, 7)["start"])
        self.assertTrue(eng.step(it8, 8)["start"])
        self.assertEqual((self.svc._read_counted(a), self.svc._read_counted(b)), (3000, 5000))


class StalledReadFalls(Base):
    """A read that stops moving (a stall, a GPU wedge): with the engine's beats coming at a steady cadence,
    a quiet engine is information - the readout must FALL with the time since the last beat and go away,
    never freeze at the last beat's figure."""

    def test_rate_falls_after_the_beats_stop(self):
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        for k in range(1, 9):                            # 8 beats, 400 tok/s
            eng.feed(7, f"RHB rid=0 stage=read done={400 * k} new=40000 reused=0 pass=1/40 "
                        f"ms={k * 1000} stage_ms=0 slot=7")
            self.assertIsNone(eng.step(it))
            self.tick()
        first = self.row()["prefill_tok_s"]
        self.assertGreater(first, 350)
        self.tick(3.0)                                   # no beats for 3 s (past the beat-cadence grace): fallen
        later = self.row()["prefill_tok_s"]
        self.assertIsNotNone(later)
        self.assertLess(later, first * 0.95, f"frozen at {later}")
        self.tick(10.0)                                  # and it is gone, not a stale headline
        self.assertEqual(self.row()["prefill_tok_s"], None)
        self.assertEqual(self.live(), 0.0)


class MangonelTrueReadout(Base):
    """P124v2 (PLAN-p124v2-read-rate-true): replay of the recorded mangonel read
    (ref/mangonel-engine-after-p124v.txt: "prompt 27254 tokens = 0 reused + 27254 read in 17383 ms
    (1567.8 tok/s)") - a fresh 27,254-token read in 17,383 ms, read in 4,608-token chunks at a steady
    ~4 lines a second (ref/mangonel-timeline-after-p124v.txt: 0.0 for the first 1.8 s, then 834 rising
    to 2,000-2,450, peak 2,577: the shape is right, the VALUE reads 30-60% high and says 0 while the
    first chunk is read).  The true readout is (tokens finished + the chunk in flight's tokens x passes
    done / passes) over the read's OWN elapsed time: its integral over the read is the tokens read, so
    the readout must average to the engine's own 1,567.8 over this read, be never 0 after the read's
    first RHB, and never peak over 1.25 x it for a read this steady.  `chunk=` is the size of the chunk
    the pass counts (P124v2: the line's own contribution - without it the server cannot share a chunk
    it has not seen end)."""

    NEW = 27254
    CHUNK = 4608                # five 4,608 chunks + a 4,214 tail: 27,254 (the recorded PP steps: 4608, 9216, ...)
    MS = 17383                  # the read's own time, the recorded summary
    RATE = NEW / (MS / 1000.0)  # 1,567.8 tok/s - the engine's own figure, the truth to average to
    BEAT_MS = 230               # the engine's recorded cadence (~4 a second)
    PASSES = 78                 # layer passes per chunk (the RHB pass=i/n field; any fixed count works)

    def lines(self):
        """The read's RHB beats, its chunk-end BRP steps at the chunk's own end time, and the closing
        BRA, all in engine-clock order: done = whole chunks done (what the recorded engine's U5b figure
        gave - the chunk in flight is NOT in it), pass= counts its layer passes, pos0/chunk= name the
        chunk in flight."""
        beats = int(self.MS / self.BEAT_MS)
        sizes = [min(self.CHUNK, self.NEW - self.CHUNK * c) for c in range((self.NEW + self.CHUNK - 1) // self.CHUNK)]
        ends, cum = [], 0
        for s in sizes:
            cum += s
            ends.append(cum / self.RATE * 1000.0)
        nxt = 0
        for k in range(1, beats + 1):
            ms = k * self.BEAT_MS
            while nxt < len(ends) and ends[nxt] < ms:      # (the chunk ended before this beat: its step first)
                pos_e = self.CHUNK * (nxt + 1) if nxt + 1 < len(sizes) else self.NEW
                yield (f"BRP 7 {int(pos_e)} {self.NEW} {ends[nxt]:.0f} {self.RATE:.1f} new_read={int(pos_e)} "
                       f"read_ms={ends[nxt]:.0f} stage_ms=0")
                nxt += 1
            pos = self.RATE * ms / 1000.0
            c = min(int(pos // self.CHUNK), len(sizes) - 1)
            frac = (pos - self.CHUNK * c) / sizes[c] if pos >= self.CHUNK * c else 0.0
            p = max(1, min(self.PASSES, int(frac * self.PASSES)))
            yield (f"RHB rid=0 stage=read done={self.CHUNK * c} new={self.NEW} reused=0 "
                   f"pass={p}/{self.PASSES} ms={ms} stage_ms=0 pos0={self.CHUNK * c} "
                   f"chunk={sizes[c]} slot=7")
        yield f"BRA 7 1 prompt={self.NEW} reused=0 read_ms={self.MS}.0 stage_ms=0"

    def test_the_read_readout_averages_the_engine_s_own_figure(self):
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        shown = []
        for line in self.lines():
            eng.feed(7, line)
            if not line.startswith("BRA"):
                self.assertIsNone(eng.step(it))
            if line.startswith("RHB "):                   # (a): never blank or zero once the read beats
                v = self.row()["prefill_tok_s"]
                self.assertIsNotNone(v, f"{line[:24]}: no rate after the read's first RHB")
                self.assertGreater(v, 0, f"{line[:24]}: 0 while the read runs")
            if line.startswith("RHB "):                   # (the chunk-end steps land in the same window)
                self.tick(self.BEAT_MS / 1000.0)
            if line.startswith("RHB ") and not eng.hooked[7].read_engine:
                shown.append(self.live())
        # (b): the readout's time-average over the read = its integral / the read's time = the tokens read /
        # the read's time - the engine's own 1,567.8, within 5%.
        self.assertTrue(shown)
        avg = sum(shown) / len(shown)
        self.assertLess(abs(avg - self.RATE), 0.05 * self.RATE, f"readout averages {avg:.0f}, not {self.RATE:.0f}")
        # (c): a peak over a window >= 5 s of the same readout: a steady read never peaks 25% over it
        # (the recorded Monitor peaked 2,577 - 1.64x the truth).
        self.assertLessEqual(max(shown), 1.25 * self.RATE, f"readout peaks {max(shown):.0f}")
        p = self.svc.totals.get("peak_prefill")
        self.assertIsNotNone(p, "the readout never made a peak")
        self.assertLessEqual(p["tok_s"], 1.25 * self.RATE, f"peak_prefill {p['tok_s']} tok/s")
        # and the read closes on the engine's exact figures: 27,254 counted, nothing invented
        self.assertTrue(eng.step(it)["start"])
        self.assertEqual(self.svc._read_counted(stat), self.NEW)


class CardTrueReadout(Base):
    """P124v2 review round 4 (the manager's card run, saguaro/rc124-cand2): replay of the recorded
    /metrics timeline serve/fixtures/card-rc2-read-rate-timeline.txt - a fresh 27,299-token read on
    the card's Q4 QFN config, polled 4x a second: the client got its answer in 17.8 s (7 tokens)
    and the engine's own summary (the last prefill_tok_s_mean, 1,582.9 over 27,292 read) says the
    TRUE read is ~1,580 tok/s; the Monitor at the time showed mean ~2,393 (+51 %), MAX 9,881 (6x),
    first non-zero at 0.78 s and two zero samples during the read.

    The replay uses the file's own fields, as the review asked: the time column, the prompt_read
    jumps (whole 4,608-token cuts at 6.86, 8.89, 10.66, 12.68, 14.71, 16.48 s) and the
    prefill_tok_s_mean column, which gives each cut's engine ms (value / mean: 6,480, 8,387,
    10,317, 12,257, 14,199, 16,070 ms) - synthesized into RHB beats at the engine's ~4/s cadence
    (done = whole cuts done, pos0/chunk = the cut in flight, pass = its fraction by the engine's
    own run clock) and cut-end BRP steps whose new_read lags the share the beats already booked
    (the cut's whole count 13,824 lands below the booked 13,900), closing with the BRA that carries
    the read's last mean.  That fall below the booked share used to fire _note_read's restart
    branch (a retry/slice signal): read_pos, the interval, engine_t0 and the pair all reset
    mid-read and the readout became (tokens since the false restart) / (the ~0.47 s since it) -
    4,608 / 0.466 = 9,881 at 10.66 s.  The fixed readout is cumulative tokens read / time since
    THIS read started: never blank or 0 after the read's first RHB, never over 1.25 x 1,580 on
    this read, closing within 10 % of 1,580 (the client's figure), the read never re-based
    (read_pos never steps down, the open interval is never re-opened), and the peak it feeds
    (the same readout, windows >= 5 s) stays under 1.25 x 1,580."""

    FIXTURE = ROOT / "serve" / "fixtures" / "card-rc2-read-rate-timeline.txt"
    TOTAL = 27299                 # prompt_total on the timeline
    TARGET = 1580.0               # the client's figure (27,299 read in 17.8 s; engine 1,582.9)
    BEAT_MS = 230                 # the engine's recorded RHB cadence
    PASSES = 60                   # layer passes per cut (the pass=i/n field; any fixed count works)

    def rows(self):
        out = []
        for ln in self.FIXTURE.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*([\d.]+)\s+(\S+)\s+(\S+)\s+\{", ln)
            if not m:
                continue                                  # (the header and the client's usage line)
            rest = ln[ln.index("{"):]                     # (the /metrics JSON is cut at line end: search, parse)
            pr = re.search(r'"prompt_read": (\d+|null)', rest)
            pm = re.search(r'"prefill_tok_s_mean": ([0-9.]+|null)', rest)
            out.append((float(m.group(1)), m.group(3),
                        pr.group(1) if pr else "null", pm.group(1) if pm else "null"))
        return out

    def events(self, rows):
        """(engine ms, line) in order: the read's RHB beats and the prompt_read jumps as cut-end
        BRP steps (a jump's engine ms = value / the prefill_tok_s_mean on the sample that first saw
        it), closing with a BRA carrying the read's last mean."""
        arrival = next(t for t, reading, p, m in rows if reading == "1")   # the first reading sample: the request at the engine
        cuts, prev, mean_last = [], None, None
        for t, reading, p, m in rows:
            if m != "null":
                mean_last = float(m)
            if p != "null" and reading == "1" and int(p) != prev and m != "null":
                prev = int(p)
                cuts.append((1000.0 * int(p) / float(m), int(p)))         # (engine ms at the cut end, its whole count)
        self.assertTrue(cuts, "no prompt_read jumps in the timeline")
        self.assertIsNotNone(mean_last, "no prefill_tok_s_mean in the timeline")
        rd_ms = 1000.0 * cuts[-1][1] / mean_last                          # the read's own end: 27,292 / 1,582.9
        evs = []
        lo_ms, lo_val = 0.0, 0
        for ms_e, v in cuts:
            k = int(lo_ms // self.BEAT_MS) + 1
            while k * self.BEAT_MS <= ms_e:
                ms = k * self.BEAT_MS
                p = max(1, min(self.PASSES, int((ms - lo_ms) / (ms_e - lo_ms) * self.PASSES)))
                evs.append((ms, f"RHB rid=0 stage=read done={lo_val} new={self.TOTAL} reused=0 "
                                f"pass={p}/{self.PASSES} ms={ms} stage_ms=0 pos0={lo_val} "
                                f"chunk={v - lo_val} slot=7"))
                k += 1
            evs.append((ms_e, f"BRP 7 {v} {self.TOTAL} {ms_e:.0f} {v / (ms_e / 1000.0):.1f} "
                              f"new_read={v} read_ms={ms_e:.0f} stage_ms=0"))
            lo_ms, lo_val = ms_e, v
        evs.append((rd_ms, f"BRA 7 1 prompt={self.TOTAL} reused=0 read_ms={rd_ms:.1f} stage_ms=0"))
        return arrival, evs

    def test_the_card_read_readout_is_cumulative_never_the_interval_spike(self):
        rows = self.rows()
        self.assertTrue(rows, "the card timeline fixture is unreadable")
        arrival, evs = self.events(rows)
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        self.tick(arrival)                                    # the request arrives at the engine: the read's own start
        it = eng.wait(7)
        base_t = self.t[0]
        shown, shown_t, ei, prev_pos, open_id = [], [], 0, 0, None
        for t_row, reading, p, m in rows:
            if reading != "1":
                continue
            due = (t_row - arrival) * 1000.0                  # engine ms visible by this poll
            while ei < len(evs) and evs[ei][0] <= due:
                self.t[0] = base_t + evs[ei][0] / 1000.0
                eng.feed(7, evs[ei][1])
                if not evs[ei][1].startswith("BRA"):
                    self.assertIsNone(eng.step(it))
                self.assertGreaterEqual(stat.read_pos, prev_pos,
                                        f"t={t_row}: the read counter went back ({prev_pos} -> {stat.read_pos})")
                prev_pos = stat.read_pos
                if open_id is None:
                    open_id = id(stat.read_cur)
                else:
                    self.assertEqual(id(stat.read_cur), open_id,
                                     f"t={t_row}: the open read interval was re-opened - the readout would "
                                     f"be an interval rate from the re-baselined start")
                ei += 1
            self.t[0] = base_t + t_row
            shown.append(self.live())                         # (the Monitor polls /metrics 4x a second)
            shown_t.append(t_row)
        self.assertTrue(shown)
        first = next((i for i, v in enumerate(shown) if v), None)
        self.assertIsNotNone(first, "the readout never showed a rate during the read")
        self.assertLessEqual(shown_t[first], 0.78,            # the card: first non-zero at 0.78 s, two zeros
                             f"first non-zero at {shown_t[first]} s")
        for i, v in enumerate(shown):                         # ... and never blank or 0 after it
            if shown_t[i] >= shown_t[first]:
                self.assertTrue(v, f"t={shown_t[i]}: blank/zero while the read runs")
        vals = [v for v in shown if v]
        self.assertLessEqual(max(vals), 1.25 * self.TARGET,  # the card: 9,881 = 6.3x the true read
                             f"the readout peaks at {max(vals)} tok/s")
        self.assertTrue(shown[-1], "the readout went blank at the read's end")
        self.assertLess(abs(shown[-1] - self.TARGET), 0.10 * self.TARGET,
                        f"the read closes at {shown[-1]} tok/s, not within 10 % of {self.TARGET}")
        pk = self.svc.totals.get("peak_prefill")              # the Monitor's peak: the same readout, windows >= 5 s
        self.assertIsNotNone(pk, "the readout never made a peak")
        self.assertLessEqual(pk["tok_s"], 1.25 * self.TARGET, f"peak_prefill {pk['tok_s']} tok/s")
        self.assertLessEqual(pk["tokens"], self.TOTAL)
        while ei < len(evs):                                  # close the read on the engine's BRA
            eng.feed(7, evs[ei][1])
            ei += 1
        self.assertTrue(eng.step(it)["start"])
        self.assertEqual(self.svc._read_counted(stat), self.TOTAL)


class CounterFallsRestartOrCorrectDown(Base):
    """P124v2 review round 5 (P1): a progress step BELOW the booked read_pos is round 2's "the engine's
    counter went back" restart signal, and round 4's tolerance for a lagging whole-cut step could
    swallow a REAL restart (an early read whose beats booked only the share of a still-open first unit:
    its retry's live beat sits at done=0, so "agrees with the beat, fall <= one unit" held and the
    restart was belted away).  The tolerance now needs the engine's own counter to have a WHOLE unit
    closed (n_read >= the live beat's chunk=) on top of the beat's agreement; a genuine restart -
    nothing whole closed yet, or a step below the live beat's own done= - closes the interval,
    re-baselines engine_t0 and re-opens (what these tests pin).  A step that merely lags a booked share
    corrects read_pos DOWN to the engine's figure instead (round 4 belted it up, which kept a split
    read's leading card above the engine's cross-card figure until BRA - the review's P2); the replay
    shapes below are ones the engine really prints: the unit-1 alignment of a split read, where both
    cards' beats hold the share guard at done=4,608 and the next cross-card step lands below the
    leading card's booking."""

    def _aligned_unit(self, eng, stat, it, passes=(10, 40, 70)):
        """Beats on the unit [4,608, 9,216) of a 9,216-token read, the engine's whole-cut done= at 4,608
        (the split read's unit-1 alignment: both cards report pos0=4,608 with pass fractions)."""
        for p in passes:
            eng.feed(7, f"RHB rid=0 stage=read done=4608 new=9216 reused=0 pass={p}/78 "
                        f"ms={3000 + p * 10} stage_ms=0 pos0=4608 chunk=4608 slot=7")
            self.assertIsNone(eng.step(it))
            self.tick(0.23)

    def test_a_restart_before_any_whole_unit_closed_still_reopens(self):
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        eng.feed(7, "RHB rid=0 stage=read done=0 new=9216 reused=0 pass=40/78 ms=1000 "
                    "stage_ms=0 pos0=0 chunk=4608 slot=7")
        self.assertIsNone(eng.step(it))
        self.assertGreater(stat.read_pos, 0)            # (the open first unit's share booked)
        leg0, t0 = id(stat.read_cur), stat.engine_t0
        self.tick(0.23)
        eng.feed(7, "BRP 7 0 9216 1300 0.0 new_read=0 read_ms=300.0 stage_ms=0")   # the retry's first step
        self.assertIsNone(eng.step(it))
        self.assertNotEqual(id(stat.read_cur), leg0, "the interval was not re-opened")
        self.assertEqual(stat.read_pos, 0)
        self.assertEqual(stat.engine_t0, self.t[0])     # re-baselined (nit 3), and NOT the old start
        self.assertNotEqual(stat.engine_t0, t0)
        self.assertEqual(len([r for r in self.svc.reads if r[0] is stat]), 2)   # closed, and a fresh one

    def test_a_restart_below_the_live_beat_still_reopens(self):
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        self._aligned_unit(eng, stat, it)               # share booked above the beat's done=4,608
        self.assertGreater(stat.read_pos, 4608)
        leg0, t0 = id(stat.read_cur), stat.engine_t0
        eng.feed(7, "BRP 7 4000 9216 4000 1146.2 new_read=4000 read_ms=4000.0 stage_ms=0")   # below the
        self.assertIsNone(eng.step(it))                 #  beat's own done= -> a real restart
        self.assertNotEqual(id(stat.read_cur), leg0, "the interval was not re-opened")
        self.assertEqual(stat.engine_t0, self.t[0])
        self.assertNotEqual(stat.engine_t0, t0)
        self.assertEqual(stat.read_pos, 4000)           # counts from the new read's step

    def test_a_step_lagging_a_booked_share_correct_down_instead(self):
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        self._aligned_unit(eng, stat, it, passes=(70,))
        booked = stat.read_pos                          # (4,608 + 4,608 x 70/78)
        self.assertGreater(booked, 4608)
        eng.feed(7, "RHB rid=0 stage=read done=4608 new=9216 reused=0 pass=3/78 "
                    "ms=3830 stage_ms=0 pos0=4608 chunk=4608 slot=7")     # the second card's beat
        self.assertIsNone(eng.step(it))
        leg0, t0 = id(stat.read_cur), stat.engine_t0
        eng.feed(7, "BRP 7 7007 9216 4000 1751.8 new_read=7007 read_ms=4000.0 stage_ms=0")
        self.assertIsNone(eng.step(it))                 # the cross-card figure lags the leading booking
        self.assertEqual(id(stat.read_cur), leg0, "the interval was re-opened by a lagging step")
        self.assertEqual(stat.engine_t0, t0)            # nothing re-baselined
        self.assertEqual(len([r for r in self.svc.reads if r[0] is stat]), 1)
        self.assertLess(stat.read_pos, booked)          # corrected DOWN, not belted up
        self.assertGreaterEqual(stat.read_pos, 4608)    # ... to the engine's own figure
        eng.feed(7, "RHB rid=0 stage=read done=7007 new=9216 reused=0 pass=1/78 "
                    "ms=4300 stage_ms=0 pos0=9216 chunk=0 slot=7")       # the engine's figure carries on
        self.assertIsNone(eng.step(it))
        self.assertEqual(stat.read_pos, 7007)
        eng.feed(7, "BRA 7 1 prompt=9216 reused=0 read_ms=4500.0 stage_ms=0")
        self.assertTrue(eng.step(it)["start"])
        self.assertEqual(self.svc._read_counted(stat), 9216)              # BRA's exact tops it up, right

    def test_a_restart_rebaselines_the_rate_numerator_with_the_span(self):
        """P124v2 review round 6 (P1, the blocker): the restart branch zeroed the SPAN (read_leg_ms,
        engine_t0) while _read_counted kept the CLOSED intervals - so after a mid-read restart the
        readout divided the abandoned attempt's tokens by the NEW read's span: a 24 s / 400 tok/s read
        (9,600 tokens, its beats' own ms said 400 the whole way) retried at 300 tok/s showed 6,800
        tok/s 1.5 s after the restart (the no-pair path) and 3,500..1,500 on the retry's own steps -
        the engine's own figure all along: 300.  The numerator now re-baselines with the span
        (stat.read_count_reset): the retry's readout is the retry's tokens over the retry's span;
        _read_counted keeps the history figure the tests read back."""
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        for k in range(1, 25):                           # 24 s of RHB beats at 400 tok/s: 9,600 tokens
            eng.feed(7, f"RHB rid=0 stage=read done={400 * k} new=40000 reused=0 pass=1/40 "
                        f"ms={k * 1000} stage_ms=0 slot=7")
            self.assertIsNone(eng.step(it))
            self.tick()
        eng.feed(7, "BRP 7 600 40000 25200 300.0 new_read=600 read_ms=2000.0 stage_ms=0")   # the retry:
        self.assertIsNone(eng.step(it))                  #  600 under the live beat's done=9,600: restart
        self.assertEqual(self.svc._read_counted(stat), 10200)      # (history keeps both attempts: 9,600 + 600)
        self.tick(1.5)                                   # the poll between the restart and the retry's next
        rate = self.svc._own_read_rate(stat, self.t[0], self.svc._read_counted(stat))   #  step (the no-pair
        self.assertLess(rate, 1000.0, f"1.5 s after the restart: readout {rate} - the old attempt's tokens "
                                      "still divide the new read's span")   #  path: pre-fix 10,200 / 1.5 s)
        self.assertGreater(rate, 100.0, f"1.5 s after the restart: the readout went blank ({rate})")
        for k in range(1, 7):                            # the retry runs on at 300 tok/s (BRP steps)
            eng.feed(7, f"BRP 7 {600 + 300 * k} 40000 {25200 + k * 1000} 300.0 "
                        f"new_read={600 + 300 * k} read_ms={2000 + k * 1000}.0 stage_ms=0")
            self.assertIsNone(eng.step(it))
            self.tick()
            r = self.svc._own_read_rate(stat, self.t[0], self.svc._read_counted(stat))
            self.assertLess(r, 1000.0, f"retry second {k}: readout {r} - old tokens still in the numerator")
        self.assertAlmostEqual(r, 300.0, delta=50.0)     # (the engine's own: 2,400 tokens / 8.0 s = 300)
        self.assertLess(self.row()["prefill_tok_s"], 1000)        # the chat row shows the fixed readout
        pk = self.svc.totals.get("peak_prefill")                   # (row()'s metrics ran the live readout:
        self.assertTrue(pk is None or pk["tok_s"] < 1000, f"peak_prefill {pk}")   #  a peak from it too)

    def test_the_restart_exclusion_survives_the_next_admission_hook(self):
        """P124v2 review round 8 (P1, the blocker, re-deriving round 7): round 6's exclusion release in
        _read_start rests on the premise that the hook does NOT zero read_leg_ms - true only while the
        span was not JUST zeroed by the restart branch.  _admission_wait fires _read_start on EVERY
        admission (serve/server.py:9907-9909, the hook installed at :6326), so the next admission of a
        retried read - its slice's next leg, or a retried read's fresh admission - released the
        exclusion while the span still stood zeroed: the abandoned attempt's 9,600 tokens went back
        into the numerator without their time (the gate test's exact shape, this hook fired between the
        restart and the retry's steps: 3,500..1,500 tok/s where the engine says 300, pre-fix), and
        _prefill_tok_s feeds that to peak_prefill (read_eng_n >= PEAK_MIN_READ_TOKENS over the retry's
        pair).  The exclusion now survives the hook: it and the closed intervals it counts against
        both outlive _read_start, and min(excl, closed) keeps it coherent as they prune - without a
        restart it is 0 anyway, so a plain slice leg is unchanged."""
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        for k in range(1, 25):                           # 24 s of RHB beats at 400 tok/s: 9,600 tokens
            eng.feed(7, f"RHB rid=0 stage=read done={400 * k} new=40000 reused=0 pass=1/40 "
                        f"ms={k * 1000} stage_ms=0 slot=7")
            self.assertIsNone(eng.step(it))
            self.tick()
        eng.feed(7, "BRP 7 600 40000 25200 300.0 new_read=600 read_ms=2000.0 stage_ms=0")   # the retry:
        self.assertIsNone(eng.step(it))                  #  600 under the live beat's done=9,600: restart
        self.assertEqual(self.svc._read_counted(stat), 10200)      # (history keeps both attempts: 9,600 + 600)
        self.svc._read_start(stat)                       # the NEXT admission of this request (exactly
        self.assertGreater(stat.read_count_reset, 0,     #  what tls.on_read_start calls for a slice leg
                           "the admission released the restart exclusion")
        for k in range(1, 7):                            # the retry runs on at 300 tok/s (BRP steps)
            eng.feed(7, f"BRP 7 {600 + 300 * k} 40000 {25200 + k * 1000} 300.0 "
                        f"new_read={600 + 300 * k} read_ms={2000 + k * 1000}.0 stage_ms=0")
            self.assertIsNone(eng.step(it))
            self.tick()
            r = self.svc._own_read_rate(stat, self.t[0], self.svc._read_counted(stat))
            self.assertLess(r, 1000.0, f"retry step {k} after restart + admission: readout {r} - the "
                                       "released exclusion put the abandoned attempt back in the numerator")
        self.assertAlmostEqual(r, 300.0, delta=50.0)     # (the engine's own: 2,400 tokens / 8.0 s = 300)
        self.assertLess(self.row()["prefill_tok_s"], 1000)        # the chat row shows the fixed readout
        pk = self.svc.totals.get("peak_prefill")
        self.assertTrue(pk is None or pk["tok_s"] < 1000, f"peak_prefill {pk}")   # (a peak from it too)


class SplitReadFollowsEngineFigure(Base):
    """P124v2 review round 5 (P2): a two-card layer split (each card half the layers, 78 passes each), one
    fresh 13,824-token read in three 4,608-token units, replayed as the engine really prints it: RHB lines
    every 230 ms from EITHER card's host points, done= the read's ONE shared figure - ReadProgress
    (engines/qfn/include/strata/program/read_progress.hpp) gives no part credit while no boundary above a
    card's unit position has been revealed, so between boundaries done == pos0 - reused for BOTH cards and
    their pass fractions differ - and BRP steps carrying that figure at the chunk ends.  Before this round
    every such beat added its own share and the belt kept the MAX: the counter followed the leading card,
    up to a whole unit of tokens above the engine's own cross-card figure, and the figure's next step below
    it was the fall P1 talks about.  A pass fraction coming back DOWN on the same unit (the lagging card
    reports) flags the read: from there no share is added and the counter follows the engine's figure,
    never above it.  Gates: once the engine has printed a cumulative figure, the counted tokens never
    exceed it at any sample; read_pos never steps back, the open interval is never re-opened, engine_t0
    never moves; and the read closes at exactly 13,824."""

    NEW = 13824
    U = 4608
    PASSES = 78

    def seq(self):
        """(engine_cum, line) in clock order; the RHB done= fields come from a faithful replica of
        ReadProgress::refresh_ (two cards x 78 layers of 156): the same host points in the same order."""
        cards, bounds, full, best, evs = {}, [], 0, 0, []

        def m():                                        # the engine's own clock: 230 ms per printed line
            return 1000 + 230 * len(evs)

        def report(card, pos0, pi):
            nonlocal best
            cards[card] = (pos0, pi)
            if pos0 not in bounds:
                bisect.insort(bounds, pos0)
            s = 0.0
            for p0, k in cards.values():
                i = bisect.bisect_right(bounds, p0)
                b = bounds[i] if i < len(bounds) else p0
                s += p0 * self.PASSES + (b - p0) * k
            best = max(best, full, int(s / (2 * self.PASSES) + 0.5))
            return best

        def beat(card, pos0, pi):
            d = report(card, pos0, pi)
            evs.append((d, f"RHB rid=0 stage=read done={d} new={self.NEW} reused=0 pass={pi}/{self.PASSES} "
                           f"ms={m()} stage_ms=0 pos0={pos0} chunk={self.U if pos0 < self.NEW else 0} slot=7"))

        # unit 0: card A first (share booked: one card is indistinguishable from two until B's lower
        # fraction lands), then card B's lagging beat - the flag
        for pi in (10, 20, 30, 40):
            beat("A", 0, pi)
        beat("B", 0, 5)                                   # fraction DOWN on the same unit: split flagged
        beat("A", self.U, 1)                              # A crosses: B gets part credit, done moves above 0
        beat("B", 0, 20)
        beat("A", self.U, 50)
        for pi in (40, 60):
            beat("B", 0, pi)
        beat("A", self.U, 70)
        full = self.U
        evs.append((full, f"BRP 7 {full} {self.NEW} {m()} {full / (m() / 1000.0):.1f} "
                          f"new_read={full} read_ms={m()}.0 stage_ms=0"))
        # unit 1: both cards aligned on it - the guard holds for every beat, and with the flag none of
        # them books a share (unflagged, A's 40/78 would have booked 6,971 above the engine's 4,608)
        beat("B", self.U, 1)
        beat("A", self.U, 40)
        beat("B", self.U, 10)
        beat("A", 2 * self.U, 1)
        beat("B", self.U, 40)
        full = 2 * self.U
        evs.append((full, f"BRP 7 {full} {self.NEW} {m()} {full / (m() / 1000.0):.1f} "
                          f"new_read={full} read_ms={m()}.0 stage_ms=0"))
        beat("B", 2 * self.U, 1)
        beat("A", 2 * self.U, 40)
        full = self.NEW
        evs.append((full, f"BRP 7 {full} {self.NEW} {m()} {full / (m() / 1000.0):.1f} "
                          f"new_read={full} read_ms={m()}.0 stage_ms=0"))
        evs.append((full, f"BRA 7 1 prompt={self.NEW} reused=0 read_ms={m()}.0 stage_ms=0"))
        return evs

    def test_the_split_counter_never_exceeds_the_engine_s_figure(self):
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        prev_pos, cum, leg0, t0 = 0, 0, None, None
        for cum_line, line in self.seq():
            cum = max(cum, cum_line)
            eng.feed(7, line)
            if not line.startswith("BRA"):
                self.assertIsNone(eng.step(it))
            if leg0 is None:                            # (the first step opens the read: _admission_wait
                leg0, t0 = id(stat.read_cur), stat.engine_t0   #  only runs its body - the start hook - there)
                self.assertIsNotNone(leg0)
            else:
                self.assertEqual(id(stat.read_cur), leg0, f"{line[:18]}: the open interval was re-opened")
                self.assertEqual(stat.engine_t0, t0, f"{line[:18]}: engine_t0 moved - a restart")
            self.assertGreaterEqual(stat.read_pos, prev_pos,
                                    f"{line[:18]}: read_pos went back ({prev_pos} -> {stat.read_pos})")
            prev_pos = stat.read_pos
            if cum > 0:                                   # (the engine's own cumulative figure, at every
                self.assertLessEqual(svc_count := self.svc._read_counted(stat), cum,   #  sample, once it has
                                     f"{line[:18]}: counted {svc_count} over the engine's {cum}")   # one)
            self.tick(0.23)
        self.assertEqual(self.svc._read_counted(stat), self.NEW)      # closed exact on the BRA
        self.assertEqual(stat.read_pos, self.NEW)


class LateStepAfterClose(Base):
    """P124v2 review round 5 (P3 nit): a read step that arrives AFTER the read's interval closed (a late
    BRP after the BRA, or after the answer's first token landed before the closer ran) re-opens a fresh
    leg - and _read_span_s measured the open leg from engine_t0 while read_leg_ms ALREADY held the
    closed leg: the closed leg's wall time was counted twice, the span inflated and the rate read low.
    The leg is now measured from read_cur's own start."""

    def test_a_step_after_the_close_does_not_double_the_span(self):
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        for k in range(1, 9):                            # 8 s of beats at 400 tok/s
            eng.feed(7, f"RHB rid=0 stage=read done={400 * k} new=4000 reused=0 pass=1/40 "
                        f"ms={k * 1000} stage_ms=0 slot=7")
            self.assertIsNone(eng.step(it))
            self.tick()
        stat.first_token = self.t[0]                     # the answer started: the next poll closes the leg
        self.svc._prefill_tok_s()
        self.assertEqual(self.svc._read_span_s(stat, self.t[0]), 8.0)      # the closed leg stands
        self.tick(5.0)
        eng.feed(7, "BRP 7 3200 4000 13000 400.0 new_read=3200 read_ms=8000.0 stage_ms=0")   # late step
        self.assertIsNone(eng.step(it))                  # re-opens the leg, no start hook: engine_t0 old
        span = self.svc._read_span_s(stat, self.t[0])
        self.assertLess(span, 9.0, f"span {span}: the closed leg was counted twice")   # 8 s, not 8 + 13
        rate = self.svc._own_read_rate(stat, self.t[0], self.svc._read_counted(stat))
        self.assertGreater(rate, 300.0, f"the late step drags the readout to {rate}")  # ~400, not ~150


class PeaksTellTheTruth(Base):
    """The PEAKS half of PLAN-p124v-read-rate (ref/mangonel-peaks-shown.txt vs ref/mangonel-peaks-log.txt):
    the Monitor showed peak_decode 94.7 tok/s ("4668 tokens") and peak_prefill 171,860.8 tok/s
    ("859,304 tokens"), while the engine's own per-request lines on that box support 64.3 tok/s for any single
    answer and ~2,128.9 tok/s for any single read.  Neither per-answer peak may exceed what one line supports;
    the summed figures become their own engine-total peaks, labelled."""

    def test_two_answers_at_64_tok_s_never_become_a_94_7_answer_peak(self):
        now = self.t[0]
        a, b = reader(RequestStat()), reader(RequestStat())
        for st in (a, b):                               # each its own window: 320 tokens over its last 5 s
            st.first_token = now - 10.0
            st.base = (now - 5.0, 0)
            st.samples.append((now, 320))
        with self.svc.status_lock:
            self.svc.current_requests.update({a, b})
        m = self.svc.metrics()                          # (metrics() runs _tok_s: the live figures feed the peaks)
        self.assertAlmostEqual(m["live"]["tok_s"], 128.0, delta=2)     # the engine total, summed, stays live
        self.assertAlmostEqual(self.svc.totals["peak_decode"]["tok_s"], 64.0, delta=1.0)   # one answer's own
        self.assertEqual(self.svc.totals["peak_decode"]["tokens"], 320)    # its own tokens, never the 4668 sum
        self.assertAlmostEqual(self.svc.totals["peak_decode_engine"]["tok_s"], 128.0, delta=2)  # summed: its own

    def _run_read(self, lines, rate_true, reused, n_new):
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        for line in lines:
            eng.feed(7, line)
            self.assertIsNone(eng.step(it))
            self.tick(1.0)
            self.live()                                  # (the Monitor polls ~4 a second: the readout, and the
            #  peaks it feeds, are computed inside _prefill_tok_s - a read nobody watches records no peak)
        p = self.svc.totals["peak_prefill"]
        self.assertIsNotNone(p, "the readout never made a peak")
        self.assertLessEqual(p["tok_s"], rate_true * 1.25)   # the peak never exceeds the read's true speed...
        self.assertGreaterEqual(p["tok_s"], rate_true * 0.75)
        self.assertLessEqual(p["tokens"], n_new)             # ...and never counts more than the tokens READ
        self.assertGreaterEqual(p["tokens"], 512)
        return p

    def test_a_restored_read_peaks_from_only_what_it_read(self):
        lines = [f"RHB rid=0 stage=stage done=0 new=8192 reused=50000 pass=1/8 ms={k * 1000} stage_ms={k * 1000} slot=7"
                 for k in range(1, 41)]
        lines += [f"RHB rid=0 stage=read done={100 * k} new=8192 reused=50000 pass=1/8 ms={40000 + k * 1000} "
                  f"stage_ms=40000 slot=7" for k in range(1, 83)]
        # 40 s of staging 50K restored tokens then a 100 tok/s read: the peak is ~100 tok/s over tokens read
        # only - never the restored 50K, and never diluted toward the single digits by the staging time
        self._run_read(lines, 100.0, 50000, 8192)

    def test_a_fast_read_peaks_at_its_true_rate(self):
        lines = [f"RHB rid=0 stage=read done={800 * k} new=64000 reused=0 pass=1/40 ms={k * 1000} stage_ms=0 slot=7"
                 for k in range(1, 81)]
        self._run_read(lines, 800.0, 0, 64000)

    def test_two_reads_at_once_sum_into_their_own_engine_peak(self):
        eng = ReadingEngine(self.svc, slots=(7, 8))
        a, b = self.reading(), self.reading()
        eng.watch(7, a)
        eng.watch(8, b)
        it7, it8 = eng.wait(7), eng.wait(8)
        for k in range(1, 11):                           # 300 + 500 tok/s, 10 s: engine total 800, best read 500
            eng.feed(7, f"RHB rid=0 stage=read done={300 * k} new=3000 reused=1000 pass=1/4 ms={k * 1000} "
                        f"stage_ms=0 slot=7")
            eng.feed(8, f"RHB rid=0 stage=read done={500 * k} new=5000 reused=2000 pass=1/4 ms={k * 1000} "
                        f"stage_ms=0 slot=8")
            self.assertIsNone(eng.step(it7, 7))
            self.assertIsNone(eng.step(it8, 8))
            self.tick()
            self.live()                                  # (the Monitor poll computes the readout and its peaks)
        per, tot = self.svc.totals["peak_prefill"], self.svc.totals["peak_prefill_engine"]
        self.assertIsNotNone(per)
        self.assertLessEqual(per["tok_s"], 560)          # one read's own peak: never the 800 sum
        self.assertIsNotNone(tot)
        self.assertLess(tot["tok_s"], 900)               # the summed figure is real (~800) - and it is the
        self.assertEqual(tot["tokens"], 1500 + 2500)     #  engine total's own, labelled, peak (recorded as the
        #  reads first pass the 5 s window, k=5: the equal-rate beats after it never top their own sum)


if __name__ == "__main__":
    unittest.main()
