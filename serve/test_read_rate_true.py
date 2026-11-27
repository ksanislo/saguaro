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

import queue
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
        self.tick(2.0)                                   # no beats for 2 s: the rate has fallen
        later = self.row()["prefill_tok_s"]
        self.assertLess(later, first * 0.8, f"frozen at {later}")
        self.tick(10.0)                                  # and it is gone, not a stale headline
        self.assertEqual(self.row()["prefill_tok_s"], None)
        self.assertEqual(self.live(), 0.0)


if __name__ == "__main__":
    unittest.main()
