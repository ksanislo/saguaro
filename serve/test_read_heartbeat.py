"""serve/test_read_heartbeat.py - the U5 read heartbeat: prompt reads report their progress at a steady cadence.

The engine prints `RHB rid=<rid|0> stage=<stage|read> done=<new tokens done> new=<new tokens total>
reused=<n> pass=<i>/<n> ms=<read ms> stage_ms=<staging part, P124v>` (the batch path's with an appended
`slot=<b>`) at most once per ~0.23 s
from host points between layer passes / pipe pieces / KV staging passes.  Behind a fake engine speaking
those lines: the driver parses them into read_live (the solo read + per slot, the live rate the tokens
between two heartbeats over the time between them), and the shared server's live figures refresh with every
beat - the chat's row carries stage / done/new / reused, /metrics' live carries read_states, and the counted
tokens step into the engine-total window - so a one-chunk read mostly busy staging its K/V from RAM no
longer shows nothing until its PP line lands.  An engine before the lines keeps working untouched.

    python -m unittest serve.test_read_heartbeat -v
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
from serve.server import (ByteTokenizer, MockEngine, RequestStat, Service,  # noqa: E402
                          _qfn_engine_class)

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096


def make_svc():
    tok = ByteTokenizer()
    return Service(MockEngine(tok, "x", max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))


def reader(stat):
    stat.started = 1.0
    stat.at_engine = True
    return stat


class FakeProc:
    def __init__(self, lines):
        self.stdout = iter(lines)

    def kill(self):
        pass


def pump_engine(lines, slots=()):
    """The real QFN driver, its real _pump, a fake engine's stdout: the lines run through it for real."""
    eng = _qfn_engine_class().__new__(_qfn_engine_class())
    eng.info, eng.vram, eng.vlen_lines = {}, [], None
    eng.lines, eng.log = queue.Queue(), None
    eng.slot_q = [queue.Queue() for _ in range(max(slots, default=-1) + 1)]     # (the driver's: one per slot)
    eng.proc = FakeProc(lines)
    eng._pump()
    return eng


class DriverReadLive(unittest.TestCase):
    """The driver parses RHB into read_live: the latest per reader, the interval rate, the routing."""

    def test_the_solo_line_lands_on_the_control_lines(self):
        eng = pump_engine([
            "RHB rid=1 stage=stage done=0 new=1142 reused=142491 pass=3/12 ms=3000",
            "RHB rid=1 stage=read done=600 new=1142 reused=142491 pass=0/1 ms=11000",
        ])
        solo = eng.read_live["solo"]
        self.assertEqual((solo["stage"], solo["done"], solo["new"], solo["reused"]), ("read", 600, 1142, 142491))
        self.assertEqual(solo["pass"], [0, 1])
        # the interval between the two heartbeats: 600 new tokens done over the 8000 ms between their read clocks
        self.assertAlmostEqual(solo["rate"], 600 / 8.0, places=1)
        # the line passes through to the control lines too: the read's own loop wakes on every beat
        ctl = [x for x in (eng.lines.get_nowait() for _ in range(eng.lines.qsize())) if x]
        self.assertEqual([x[:4] for x in ctl], ["RHB ", "RHB "])

    def test_rdone_ends_the_solo_heartbeat(self):
        eng = pump_engine(["RHB rid=1 stage=read done=600 new=1142 reused=142491 pass=0/1 ms=11000",
                           "RDONE rid=1 new=1142 ms=12000.0"])
        self.assertIsNotNone(eng.read_live["solo"])
        line = next(x for x in (eng.lines.get_nowait() for _ in range(eng.lines.qsize())) if x.startswith("RDONE"))
        eng._rdone(line)                                 # (what the read's own loop does when the line reaches it)
        self.assertIsNone(eng.read_live["solo"])         # RDONE carries the final figures: the beat is over

    def test_the_slot_line_goes_to_its_queue_and_bra_ends_it(self):
        eng = pump_engine([
            "RHB rid=0 stage=stage done=0 new=1142 reused=142491 pass=6/12 ms=6000 slot=7",
            "RHB rid=0 stage=read done=1142 new=1142 reused=142491 pass=0/1 ms=11000 slot=7",
            "BRP 7 143633 143633 11000.0 104.7 new_read=1142 read_ms=11000.0",
            "BRA 7 1 prompt=143633 reused=142491 read_ms=11000.0 wait_ms=5.0",
        ], slots=(7,))
        self.assertNotIn(7, eng.read_live["slots"])      # BRA: its read ended, the heartbeat is stale
        # the reader's queue saw every line (the RHB beats among them) - and the parse kept up with the routing:
        # BRA pops the entry, so its state is read back from the lines the queue carries
        got = [x for x in (eng.slot_q[7].get_nowait() for _ in range(eng.slot_q[7].qsize())) if x]
        self.assertEqual([x.split()[0] for x in got], ["RHB", "RHB", "BRP", "BRA"])
        self.assertEqual([x for x in (eng.lines.get_nowait() for _ in range(eng.lines.qsize())) if x],
                         [])                             # (a slot's lines never reach the control lines)

    def test_the_slot_state_between_its_lines(self):
        eng = pump_engine(["RHB rid=0 stage=stage done=0 new=1142 reused=142491 pass=6/12 ms=6000 slot=7",
                           "RHB rid=0 stage=read done=1142 new=1142 reused=142491 pass=0/1 ms=11000 slot=7"],
                          slots=(7,))
        entry = eng.read_live["slots"][7]
        self.assertEqual((entry["stage"], entry["done"], entry["new"], entry["reused"]), ("read", 1142, 1142, 142491))
        self.assertEqual((entry["rate"], entry["slot"]), (1142 / 5.0, 7))   # 1142 over the 5 s between the beats

    def test_an_engine_before_the_lines_is_untouched(self):
        eng = pump_engine(["RDONE rid=1 new=500 ms=158.0"])
        self.assertFalse(hasattr(eng, "read_live"))      # (no RHB line ever came: the state is never laid out)

    def test_the_staging_time_leaves_the_read_s_clock(self):
        # P124v (PLAN-p124v-read-rate): ms is the read's clock and stage_ms the part of it spent staging the
        # reused prefix in from RAM.  While the read only stages (stage_ms == ms) its READ time is 0 and there
        # is no rate; the first read beat's rate is its tokens over the READ time, undiluted by the staging.
        eng = pump_engine([
            "RHB rid=1 stage=stage done=0 new=800 reused=50000 pass=4/64 ms=1500 stage_ms=1500",
            "RHB rid=1 stage=read done=120 new=800 reused=50000 pass=1/64 ms=2700 stage_ms=1500",
        ])
        s = eng.read_live["solo"]
        self.assertEqual((s["stage"], s["done"], s["rt"], s["stage_ms"]), ("read", 120, 1200.0, 1500.0))
        self.assertAlmostEqual(s["rate"], 100.0)         # 120 tokens over the 1.2 s it spent READING, not 2.7
        eng = pump_engine(["RHB rid=1 stage=stage done=0 new=800 reused=50000 pass=4/64 ms=1500 stage_ms=1500"])
        s = eng.read_live["solo"]
        self.assertEqual(s["rt"], 0.0)                   # nothing has been read yet
        self.assertIsNone(s["rate"])                     # (and there is no rate to freeze at)
        # RDONE's finished figures carry it too: the exact read time is the clock minus the staging
        eng = pump_engine([])
        eng._rdone("RDONE rid=1 new=800 ms=5500.0 stage_ms=4000.0")
        self.assertEqual(eng.read_done, (800, 1500.0))

    def test_a_split_read_solo_beats_are_a_belt(self):
        # U5b (PLAN-u5b-split-progress), a 4 x T4 layer split (ref/hb-4xT4.txt): every card's host point printed
        # its own piece count, so a line printed 14592, 19456, 14592; 34048, 29184 - backwards beats whose next
        # forward step looked like a rate spike.  The driver's belt: a recorded beat never goes back within the
        # read, and a backwards line changes neither done nor rate; a new read (other new/rid) starts over.
        eng = pump_engine([])
        beats = (0, 4864, 0, 4864, 4864, 14592, 19456, 14592, 19456, 34048, 29184, 34048, 38903)
        seen, prev_done, prev_rate = [], None, None
        for i, raw in enumerate(beats):
            eng._rhb(f"RHB rid=1 stage=read done={raw} new=38910 reused=0 pass=0/1 ms={1000 + i * 1000}")
            s = eng.read_live["solo"]
            seen.append(s["done"])
            if prev_done is not None:
                self.assertGreaterEqual(s["done"], prev_done)        # the beats never go back
                if raw < prev_done:                                  # a backwards line (a lagging card won
                    self.assertEqual(s["done"], prev_done)           #   the beat): the belt holds the figure
                    self.assertEqual(s["rate"], prev_rate)           #   and the rate is untouched: no spike
            prev_done, prev_rate = s["done"], s["rate"]
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(seen[-1], 38903)
        eng._rhb("RHB rid=2 stage=read done=100 new=5000 reused=0 pass=0/1 ms=500")   # another read (rid): over
        self.assertEqual(eng.read_live["solo"]["done"], 100)
        eng._rhb("RHB rid=2 stage=read done=50 new=9000 reused=0 pass=0/1 ms=1000")    # another read (new): over
        self.assertEqual(eng.read_live["solo"]["done"], 50)

    def test_the_p124v2_fields_parse_without_the_float_round_trip(self):
        # P124v2 review round 5: the pos0=/chunk= parse paths the modules had never walked - a line
        # without them (the P124v engine: pos0 None, chunk 0, the old behaviour), and a malformed
        # field: int(float(...)) used to turn chunk=1e400 into int(inf) - an OverflowError that
        # ESCAPED _rhb on the pump thread.  The fields are plain %lld integers now: a bad value
        # reads as absent, and the defined piece-boundary line (pos0=-1 chunk=0) keeps its values.
        eng = pump_engine(["RHB rid=1 stage=read done=600 new=1142 reused=0 pass=0/1 ms=11000"])
        s = eng.read_live["solo"]
        self.assertIsNone(s["pos0"])
        self.assertEqual(s["chunk"], 0)
        eng._rhb("RHB rid=2 stage=read done=1 new=2 reused=0 pass=0/1 ms=1 chunk=1e400 pos0=12345")
        s = eng.read_live["solo"]
        self.assertEqual((s["chunk"], s["pos0"]), (0, 12345))     # (no exception escapes _rhb)
        eng._rhb("RHB rid=3 stage=read done=1 new=2 reused=0 pass=0/1 ms=1 chunk=abc pos0=nan")
        s = eng.read_live["solo"]
        self.assertEqual((s["chunk"], s["pos0"]), (0, None))
        eng._rhb("RHB rid=0 stage=read done=9216 new=27254 reused=0 pass=0/1 ms=5000 pos0=-1 chunk=0 slot=7")
        s = eng.read_live["slots"][7]
        self.assertEqual((s["pos0"], s["chunk"]), (-1, 0))

    def test_a_split_read_slot_beats_are_a_belt(self):
        # the same belt on a batch slot's read (rid=0): its slot entry only ever moves forward, the rate follows
        # the forward part - and the BRA that ends the read still clears the entry
        eng = pump_engine([], slots=(7,))
        prev_done, prev_rate = None, None
        for i, raw in enumerate((4864, 0, 4864, 14592, 9728, 19456)):
            eng._rhb(f"RHB rid=0 stage=read done={raw} new=38910 reused=0 pass=0/1 ms={1000 + i * 1000} slot=7")
            s = eng.read_live["slots"][7]
            if prev_done is not None:
                self.assertGreaterEqual(s["done"], prev_done)
                if raw < prev_done:
                    self.assertEqual(s["done"], prev_done)
                    self.assertEqual(s["rate"], prev_rate)
            prev_done, prev_rate = s["done"], s["rate"]
        self.assertEqual(prev_done, 19456)
        self.assertEqual(prev_rate, 4864.0)     # 19456 over the belt's 14592 in the second between: no 9728 spike
        # (the BRA that ends a read clears the entry: test_the_slot_line_goes_to_its_queue_and_bra_ends_it)


class ReadingEngine:
    """A fake QFN batch read behind the real adapter: what the pump does (parse, then route) happens here by
    hand, and the request waits in _admission_wait exactly as generate_batched leaves it.  run() installs the
    same thread-local hooks on the engine it does for a real one."""

    def __init__(self, svc, slots=(7,)):
        self.svc = svc
        self.eng = _qfn_engine_class().__new__(_qfn_engine_class())
        self.eng._tls = threading.local()
        self.eng.slot_q = [queue.Queue() for _ in range(max(slots) + 1)]
        self.eng.read_live = {"solo": None, "slots": {}}
        self.hooked, self.active = {}, slots[0]

    def watch(self, slot, stat):
        """The hooks run() installs on the engine's thread-local: a step of this slot's read is booked to stat.
        (One thread-local serves the whole thread, so with several admissions in flight the step names its slot
        and the hooks route by it - the heartbeat already carries its slot.)"""
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
        self.active = slot                               # (which admission this step belongs to)
        try:
            next(it)
            return None
        except StopIteration as e:
            return e.value


class LiveEverySecond(unittest.TestCase):
    """The fake engine's RHB lines -> the /metrics live fields, refreshed with every beat."""

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

    def read_states(self):
        return self.svc.metrics()["live"]["read_states"]

    def row(self):
        return next(c for c in self.svc.metrics()["live"]["chats"] if c["state"] == "reading")

    def test_one_chunk_read_long_staging(self):
        # Ken's live log: "143633 tokens = 142491 reused + 1142 read in 10877 ms" - eleven seconds of staging
        # the reused KV from RAM before a single new token is read; every second the row tells it, and there is
        # no rate to freeze at: nothing has been READ yet
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        for i in range(1, 12):
            eng.feed(7, f"RHB rid=0 stage=stage done=0 new=1142 reused=142491 pass={i}/12 ms={i * 1000} slot=7")
            self.assertIsNone(eng.step(it))
            self.tick()
            self.assertEqual(self.read_states(), [{"stage": "stage", "done": 0, "new": 1142, "reused": 142491}])
            row = self.row()
            self.assertEqual((row["read_stage"], row["read_done"], row["read_new"], row["read_reused"]),
                             ("stage", 0, 1142, 142491))
            self.assertIsNone(row["prefill_tok_s"])      # (its own: nothing read - the row shows the stage instead)
        self.assertEqual(self.svc._prefill_tok_s(), 0.0)
        self.assertEqual(self.svc.metrics()["live"]["prefill_tok_s"], 0.0)
        # the read of the new tokens: the heartbeats' steps count into the engine-total window as they land
        eng.feed(7, "RHB rid=0 stage=read done=600 new=1142 reused=142491 pass=0/1 ms=11900 slot=7")
        self.assertIsNone(eng.step(it))
        rs = self.read_states()[0]
        self.assertEqual((rs["stage"], rs["done"]), ("read", 600))
        self.assertEqual(self.svc._read_counted(stat), 600)
        self.tick()
        eng.feed(7, "RHB rid=0 stage=read done=1142 new=1142 reused=142491 pass=0/1 ms=12700 slot=7")
        self.assertIsNone(eng.step(it))
        self.tick()
        self.assertEqual(self.svc._read_counted(stat), 1142)
        self.assertGreater(self.svc.metrics()["live"]["prefill_tok_s"], 0)   # the live figure moves per second
        eng.feed(7, "BRA 7 1 prompt=143633 reused=142491 read_ms=12800.0 wait_ms=5.0")
        res = eng.step(it)
        self.assertTrue(res["start"])
        self.assertEqual(res["figures"]["prompt_read"], 1142)
        self.assertEqual(self.svc._read_counted(stat), 1142)                 # BRA exacts it: no double count

    def test_multi_chunk_read(self):
        # a long prompt read in one long stretch: each heartbeat's rate is the tokens since the last beat over
        # the time between them, and the live state moves from beat to beat every second
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        for done in (5000, 10000, 15000):
            eng.feed(7, f"RHB rid=0 stage=read done={done} new=20000 reused=30000 pass=0/1 ms={done / 5.0:.0f} slot=7")
            self.assertIsNone(eng.step(it))
            self.tick()
            entry = eng.eng.read_live["slots"][7]
            self.assertAlmostEqual(entry["rate"], 5000.0, places=1)     # 5000 tokens over the 1 s between beats
            self.assertEqual(self.read_states(),
                             [{"stage": "read", "done": done, "new": 20000, "reused": 30000}])
        self.assertEqual(self.svc._read_counted(stat), 15000)
        self.assertAlmostEqual(self.svc._prefill_tok_s(), 5000.0, places=0)    # 15000 over the 3 s since it began

    def test_two_chats_reading_at_once(self):
        # two admissions reading beside each other: each chat's row carries ITS read, the live read_states carry
        # both, and the engine-total window sums them (neither chat's figures alone say what the engine reads)
        eng = ReadingEngine(self.svc, slots=(7, 8))
        a, b = self.reading(), self.reading()
        eng.watch(7, a)
        eng.watch(8, b)
        it7, it8 = eng.wait(7), eng.wait(8)
        for i, (d7, d8) in enumerate(((400, 900), (800, 1800)), start=1):
            eng.feed(7, f"RHB rid=0 stage=read done={d7} new=5000 reused=1000 pass=0/1 ms={i * 1000} slot=7")
            eng.feed(8, f"RHB rid=0 stage=read done={d8} new=9000 reused=2000 pass=0/1 ms={i * 1000} slot=8")
            self.assertIsNone(eng.step(it7, 7))
            self.assertIsNone(eng.step(it8, 8))
            self.tick()
            rs = sorted(self.read_states(), key=lambda r: r["reused"])
            self.assertEqual([(r["done"], r["new"], r["reused"]) for r in rs], [(d7, 5000, 1000), (d8, 9000, 2000)])
        self.assertEqual(self.svc._read_counted(a), 800)
        self.assertEqual(self.svc._read_counted(b), 1800)
        # both reads land inside the window together: one engine, both reads, over the 2 s between
        self.assertAlmostEqual(self.svc._prefill_tok_s(), (800 + 1800) / 2.0, places=0)
        eng.feed(7, "BRA 7 1 prompt=6000 reused=1000 read_ms=2000.0 wait_ms=1.0")
        eng.feed(8, "BRA 8 1 prompt=11000 reused=2000 read_ms=2000.0 wait_ms=1.0")
        self.assertTrue(eng.step(it7, 7)["start"])
        self.assertTrue(eng.step(it8, 8)["start"])
        self.assertEqual((self.svc._read_counted(a), self.svc._read_counted(b)), (5000, 9000))

    def test_no_heartbeat_lines_no_live_states(self):
        # an engine from before the heartbeat: no read_states, no read fields on the row - and the old lines
        # still do all they did (the BRA's exact figures count the read)
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        eng.feed(7, "BRP 7 196236 196374 9500.0 44.2 new_read=300 read_ms=9500.0")
        eng.feed(7, "BRA 7 1 prompt=196374 reused=195936 read_ms=9897 wait_ms=75")
        res = eng.step(it)                               # (a BRP without a beat is no keep-alive: the BRA answers)
        self.assertTrue(res["start"])
        self.assertEqual(self.svc.metrics()["live"]["read_states"], [])
        self.assertNotIn("read_stage", self.row())
        self.assertEqual(self.svc._read_counted(stat), 438)

    def test_a_split_read_row_never_goes_back(self):
        # U5b (PLAN-u5b-split-progress), the 4 x T4 split read (ref/hb-4xT4.txt) behind the real adapter: a line
        # printed the reading card's own piece count, so the row's figure jumped back (14592 after 19456) and the
        # next forward step read as a spike.  With the shared figure and the driver's belt the row's read_done only
        # moves forward, the counted tokens are the forward ones, and the beat's rate stays the true second's step.
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        prev = 0
        for i, raw in enumerate((4864, 0, 4864, 14592, 9728, 19456)):
            eng.feed(7, f"RHB rid=0 stage=read done={raw} new=38910 reused=0 pass=0/1 ms={1000 + i * 1000} slot=7")
            self.assertIsNone(eng.step(it))
            self.tick()
            got = self.row()["read_done"]
            self.assertGreaterEqual(got, prev)                           # the row's beats never go back
            prev = got
        self.assertEqual(prev, 19456)
        self.assertEqual(self.svc._read_counted(stat), 19456)            # only the forward tokens were booked
        self.assertAlmostEqual(eng.eng.read_live["slots"][7]["rate"], 4864.0, places=1)

    def test_a_piece_boundary_line_books_only_its_done(self):
        # P124v2 review round 5: the piece-boundary RHB line (pos0=-1 chunk=0, rhb_solo_piece) behind
        # the real adapter - no unit in flight, so the server's share guard (chunk > 0) takes nothing
        # from it: the whole done= stands, booked straight, no share and no fall signal.
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        eng.feed(7, "RHB rid=0 stage=read done=4608 new=13824 reused=0 pass=10/78 ms=6000 stage_ms=0 "
                    "pos0=4608 chunk=4608 slot=7")
        self.assertIsNone(eng.step(it))
        self.assertEqual(self.svc._read_counted(stat), 4608 + 590)      # (the unit's 10/78 share)
        self.tick(0.3)                                                  # (the mocked clock must move:
        eng.feed(7, "RHB rid=0 stage=read done=9216 new=13824 reused=0 pass=1/78 ms=6300 stage_ms=0 "   #  on_hb fires
                    "pos0=-1 chunk=0 slot=7")                           #  once per beat timestamp)
        self.assertIsNone(eng.step(it))
        self.assertEqual(self.svc._read_counted(stat), 9216)            # (done only: the piece line)
        self.assertEqual(len([r for r in self.svc.reads if r[0] is stat]), 1)   # (no restart signal)

    def test_a_stale_heartbeat_is_not_shown(self):
        # a read whose last beat is older than three seconds (the engine went quiet, a BRA was lost): the row
        # stops showing it - a frozen figure on the Monitor is worse than none
        eng = ReadingEngine(self.svc)
        stat = self.reading()
        eng.watch(7, stat)
        it = eng.wait(7)
        eng.feed(7, "RHB rid=0 stage=stage done=0 new=500 reused=90000 pass=1/12 ms=1000 slot=7")
        self.assertIsNone(eng.step(it))
        self.tick(2.0)
        self.assertEqual(len(self.read_states()), 1)
        self.tick(2.0)                                     # 4 s after the last beat: stale
        self.assertEqual(self.read_states(), [])


class EngineSideContract(unittest.TestCase):
    """The engine's side of the protocol: generate.cpp prints exactly the format the driver parses (the two
    trees checked against each other, as test_dashboard does for the other QFN lines)."""

    def test_the_engine_prints_the_documented_fields(self):
        cpp = (ROOT / "engines" / "qfn" / "src" / "program" / "generate.cpp").read_text(encoding="utf-8")
        # P124v: the beat carries the read's own staging time (stage_ms=) beside its read clock (ms=), and the
        # gate is a steady ~4 lines a second (230 ms), not the old ~1.
        # P124v2: and the unit in flight's position and length (pos0=, chunk=), so the server can count its
        # pass=<i>/<n> share of tokens that done= (whole units only) leaves out.
        self.assertIn('"RHB rid=%lld stage=%s done=%lld new=%lld reused=%lld pass=%lld/%lld '
                      'ms=%.0f stage_ms=%.0f"', cpp)
        self.assertIn('" pos0=%lld chunk=%lld\\n"', cpp)
        self.assertIn('"RHB rid=0 stage=%s done=%lld new=%lld reused=%lld pass=%lld/%lld '
                      'ms=%.0f stage_ms=%.0f"', cpp)
        self.assertIn('" pos0=%lld chunk=%lld slot=%d\\n"', cpp)
        self.assertIn("at most one line per ~0.23 s", cpp)
        self.assertIn("stage_ms=%.1f", cpp)          # PP / RDONE / BRP / BRA carry it too (the finished figures)

    def test_the_driver_keeps_the_heartbeat_per_reader(self):
        for rel in ("engines/qfn/serve/server.py", "serve/qfn_engine.py"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            self.assertIn("def _rhb(self, line):", text)
            self.assertIn('self.read_live = {"solo": None, "slots": {}}', text)

    def test_the_server_marks_the_heartbeat_fields_on_the_live_view(self):
        text = (ROOT / "serve" / "server.py").read_text(encoding="utf-8")
        for field in ("read_stage", "read_done", "read_new", "read_reused", "read_states"):
            self.assertIn(f'"{field}"', text)
        js = (ROOT / "serve" / "web" / "monitor.js").read_text(encoding="utf-8")
        self.assertIn("read_states", js)
        self.assertIn("cached · reading", js)


if __name__ == "__main__":
    unittest.main()
