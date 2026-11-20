"""serve/test_read_rate.py - the engine-total prompt READ rate (the Monitor's live Prefill), its peak, and the
totals' wall-time denominators.  The bug: live.prefill_tok_s_mean was ONE request's engine-reported mean, picked by
whichever request last wrote the field - under several chats it was another chat's number or 0, and mixed with
decode time; the totals divided read tokens by the SUMMED prompt_ms of overlapping requests, which is not wall time.

    cd serve && python -m unittest -q test_read_rate
"""
from __future__ import annotations

import io
import itertools
import json
import queue
import sys
import threading
import time
import unittest
import unittest.mock
import urllib.request
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (ByteTokenizer, MockEngine, PEAK_MIN_READ_TOKENS, RATE_WINDOW_S,  # noqa: E402
                          RequestStat, Service, StrataEngine, _qfn_engine_class, serve)

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096


def make_svc():
    tok = ByteTokenizer()
    return Service(MockEngine(tok, "x", max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))


def reader(stat):
    stat.started = 1.0
    stat.at_engine = True
    return stat


class ReadWindowSums(unittest.TestCase):
    """_prefill_tok_s(): the tokens REALLY read in one window (RATE_WINDOW_S) over that window's real span -
    every chat's read steps add into the one count - and 0.0 while nothing reads."""

    def setUp(self):
        self.svc = make_svc()
        self.t = [1000.0]
        self.clock = unittest.mock.patch("serve.server.time.time", lambda: self.t[0])
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def start_marker(self, *stats):
        # what the engine's start hook does: the read's interval (and the window's count) starts where the engine
        # took it - a wait before that is no read time
        for st in stats:
            self.svc._read_start(st)

    def test_two_overlapping_reads_sum(self):
        # what the two chats really read adds up in the one window: 600 tokens by one and 400 by the other over
        # the same 2 s read as one engine at 500 - neither chat's own figures alone say so
        svc = self.svc
        a, b = reader(RequestStat()), reader(RequestStat())
        svc.running.update({a, b})
        svc.status.update(busy=True)
        self.start_marker(a, b)
        svc._note_read(a, 600)
        svc._note_read(b, 400)
        self.t[0] += 2.0
        self.assertEqual(svc._prefill_tok_s(), 500.0)
        live = svc.metrics()["live"]
        self.assertEqual((live["state"], live["reading_n"], live["prefill_tok_s"]), ("reading", 2, 500.0))

    def test_reads_sum_without_any_engine_rate(self):
        # QFN's slot admits (BRP lines) carry no usable rate and none is asked for: the live figure is the
        # window's own count over the time it took - 400 read by one and 400 by the other in the same second
        # read as one engine at 800
        svc = self.svc
        a, b = reader(RequestStat()), reader(RequestStat())
        svc.running.update({a, b})
        svc.status.update(busy=True)
        self.start_marker(a, b)
        svc._note_read(a, 400)
        svc._note_read(b, 400)
        self.t[0] += 1.0
        self.assertEqual(svc._prefill_tok_s(), 800.0)
        self.assertTrue(a.read_shared and b.read_shared)  # both read beside the other: their rows get the shared label

    def test_zero_while_nothing_reads(self):
        svc = self.svc
        a = reader(RequestStat())
        svc.running.add(a)
        svc.status.update(busy=True)
        svc._note_read(a, 500)
        self.t[0] += 1.0
        svc._note_read(a, 900)                            # 400 read in the last second: a live rate exists
        self.assertGreater(svc._prefill_tok_s(), 0.0)
        a.first_token = self.t[0]                         # the answer started: no one reads now
        self.assertEqual(svc._prefill_tok_s(), 0.0)
        svc.status["busy"] = False                        # and the whole server idle
        self.assertEqual(svc._prefill_tok_s(), 0.0)

    def test_a_new_read_of_the_same_request_recounts(self):
        svc = self.svc
        a = reader(RequestStat())
        svc.running.add(a)
        svc.status.update(busy=True)
        svc._note_read(a, 800)
        self.t[0] += 1.0
        svc._note_read(a, 200)                    # below the last position: the engine reads the prompt again
        self.assertEqual(a.read_pos, 200)         # (a retry, a fair slice's next leg: counted from there)
        self.assertEqual(svc._read_counted(a), 1000)  # (the old read closed with its 800, the retry adds its 200)

    def test_a_read_stalled_past_the_window_gives_no_rate(self):
        # a read that pauses (for another chat, slow media) and lands no token inside RATE_WINDOW_S: the window
        # drains empty - no rate, and the empty window must not raise (its every-second poll is /metrics itself)
        svc = self.svc
        a = reader(RequestStat())
        svc.running.add(a)
        svc.status.update(busy=True)
        self.start_marker(a)
        svc._note_read(a, 500)
        self.t[0] += RATE_WINDOW_S + 1.0
        self.assertEqual(svc._prefill_tok_s(), 0.0)
        self.assertEqual(svc.metrics()["live"]["prefill_tok_s"], 0.0)


class PeakPrefill(unittest.TestCase):
    """totals.peak_prefill is kept like peak_decode - and ignores tiny reads (their rate says little)."""

    def test_kept_and_smaller_rates_do_not_replace_it(self):
        svc = make_svc()
        svc._note_peak_read(800.0, PEAK_MIN_READ_TOKENS, 100.0)
        first = svc.totals["peak_prefill"]
        self.assertEqual(first, {"tok_s": 800.0, "tokens": PEAK_MIN_READ_TOKENS, "time": 100.0})
        svc._note_peak_read(900.0, PEAK_MIN_READ_TOKENS - 1, 200.0)   # under the minimum: no record
        svc._note_peak_read(500.0, 4096, 300.0)                       # a slower read: the peak stands
        self.assertEqual(svc.totals["peak_prefill"], first)
        svc._note_peak_read(1200.0, 4096, 400.0)                      # a faster one beats it
        self.assertEqual(svc.totals["peak_prefill"]["tok_s"], 1200.0)

    def test_live_window_records_it_only_over_min_tokens(self):
        # a live peak needs the window's REAL span past PEAK_MIN_SPAN_S and enough tokens really read in it:
        # 200 over 2 s say little (no peak), 1000 over 4 s become one, and a window whose mean falls keeps it
        svc = make_svc()
        svc.status.update(busy=True)
        t = [1000.0]
        with unittest.mock.patch("serve.server.time.time", lambda: t[0]):
            a = reader(RequestStat())
            svc.running.add(a)
            svc._read_start(a)                        # (the read's interval starts where the engine took it)
            svc._note_read(a, 100)
            t[0] += 2.0
            svc._note_read(a, 200)                           # 200 tokens over the 2 s span: under the 512 minimum
            self.assertAlmostEqual(svc._prefill_tok_s(), 100.0)
            self.assertIsNone(svc.totals["peak_prefill"])
            svc._note_read(a, 1000)                          # the read goes on: 1000 tokens counted by now
            t[0] += 2.0
            self.assertAlmostEqual(svc._prefill_tok_s(), 250.0)   # 1000 over the window's 4 s span
            peak = svc.totals["peak_prefill"]
            self.assertEqual((peak["tok_s"], peak["tokens"]), (250.0, 1000))
            t[0] += 2.0                                      # nothing read since: the window's mean falls
            self.assertAlmostEqual(svc._prefill_tok_s(), 900 / 5.0)   # the 900 counted since the window's start
            self.assertEqual(svc.totals["peak_prefill"]["tok_s"], 250.0)   # the peak keeps the higher

    def test_a_finished_read_records_it(self):
        # the done path records the rate the row shows: 400 tokens under the minimum leaves no peak,
        # 1000 over it does (the row's own figures go through _note_peak_read there)
        svc = make_svc()
        svc._note_peak_read(2000.0, 400, 100.0)
        self.assertIsNone(svc.totals["peak_prefill"])
        svc._note_peak_read(2000.0, 1000, 100.0)
        self.assertEqual(svc.totals["peak_prefill"]["tok_s"], 2000.0)


class WallTimeTotals(unittest.TestCase):
    """totals read_s / decode_s: the wall seconds with at least one prompt reading / one answer writing -
    two chats overlapping add the same second ONCE, not twice (summed prompt_ms/decode_ms did)."""

    def test_overlap_counts_wall_time_once(self):
        svc = make_svc()
        a, b = reader(RequestStat()), reader(RequestStat())
        svc.running.update({a, b})
        svc.status.update(busy=True)
        svc._wall_tick()                          # two chats read: one shared reading second runs
        time.sleep(0.06)
        read_s = svc.metrics()["totals"]["read_s"]
        self.assertGreater(read_s, 0.0)
        self.assertLess(read_s, 0.4)              # wall time, not 2 x it (no sleeps in the copy path)
        a.first_token = b.first_token = time.time()
        svc._wall_tick()                          # both answered: the read clock closes, the write clock opens
        closed = svc.metrics()["totals"]["read_s"]
        time.sleep(0.03)
        m = svc.metrics()["totals"]
        self.assertLessEqual(m["read_s"], closed + 1e-6)     # nothing reads anymore
        self.assertGreater(m["decode_s"], 0.0)
        for r in (a, b):
            r.at_engine = False                   # both gave the engine back: both clocks close
            svc._wall_tick()
        self.assertLess(svc.metrics()["totals"]["decode_s"], 0.4)

    def test_engine_ms_stay_as_they_were(self):
        # the totals keep summing the engines' own prompt_ms/decode_ms (Prometheus' llamacpp: counters keep that
        # meaning); the wall clocks are separate fields the Monitor's totals line divides by
        svc = make_svc()
        self.assertEqual((svc.totals["prompt_ms"], svc.totals["decode_ms"]), (0.0, 0.0))
        self.assertNotIn("read_s", svc.totals)    # until a metrics() call adds the computed one
        t = svc.metrics()["totals"]
        self.assertIn("read_s", t)
        self.assertIn("decode_s", t)


class StrataEngineReadHook(unittest.TestCase):
    """The maya line driver counts its PP lines into the service: the hook installed by run() sees each step with
    the reused prefix taken out, and the engine's own mean beside it."""

    def engine(self):
        eng = StrataEngine.__new__(StrataEngine)
        eng.proc = SimpleNamespace(stdin=io.StringIO())
        eng.lines = queue.Queue()
        eng.can_stop = False
        eng.max_context = 262144
        eng._tls = threading.local()
        return eng

    def run_lines(self, eng, lines):
        for ln in lines:
            eng.lines.put(ln)
        gen = eng.generate([1], 1, {}, threading.Event())
        out = []
        try:
            while True:
                out.append(next(gen))
        except StopIteration:
            pass
        return out

    def test_pp_steps_exclude_the_reused_prefix(self):
        eng = self.engine()
        got = []
        eng._tls.on_read = lambda n, ts=None: got.append((n, ts))
        #            PP <read> <total> <ms> <mean> <reuse>: 3000 of 8000, 2000 the cache held -> 1000 read
        self.run_lines(eng, ["PP 3000 8000 1000.0 600.0 2000", "PP 5000 8000 2000.0 700.0 2000",
                             "DONE 1 8000 100 10 stop 0 0 2000"])
        self.assertEqual(got, [(1000, 600.0), (3000, 700.0)])

    def test_a_service_counts_the_steps_end_to_end(self):
        # the full path of one real read: PP lines -> the hook run() installs -> the window -> the live figure:
        # the tokens really read, over the span the window spans - the engine's means ride along as peak candidates
        svc = make_svc()
        eng = self.engine()
        svc.engine = eng
        t = [1000.0]
        with unittest.mock.patch("serve.server.time.time", lambda: t[0]):
            stat = reader(RequestStat())
            svc.running.add(stat)
            svc.status.update(busy=True)
            eng._tls.on_read = lambda n, ts=None: svc._note_read(stat, n, ts)   # (no start hook: the first PP
            # step opens the read's interval - the window spreads this exact read's count over the time it took)
            self.run_lines(eng, ["PP 4000 6000 2000.0 2000.0 2000", "PP 6000 6000 2500.0 2400.0 2000",
                                 "DONE 1 6000 100 10 stop 0 0 2000"])
            self.assertEqual(stat.read_pos, 4000)            # 6000 read, 2000 reused
            self.assertIsNotNone(stat.read_t0)
            t[0] += 2.0
            self.assertEqual(svc._prefill_tok_s(), 4000 / 2.0)   # 4000 really read over the window's 2 s span
            # the engine's own means rode in on the PP lines: the peak saw the faster of the two (2000, then 2400)
            self.assertEqual(svc.totals["peak_prefill"]["tok_s"], 2400.0)


    def test_multi_pp_line_counts_too(self):
        # the batch driver's PP line carries the rid: `PP <rid> <read> <total> <ms> <mean> <reuse>` - same count
        eng = self.engine()
        eng.multi = 2
        eng._rids = itertools.count(1)
        eng._routes, eng._routes_lock, eng._sent = {}, threading.Lock(), {}
        got = []
        eng._tls.on_read = lambda n, ts=None: got.append((n, ts))
        gen = eng._generate_multi(list(range(5)), 1, {}, threading.Event())

        def feed():
            q = None
            while q is None:
                with eng._routes_lock:
                    q = eng._routes.get(1)
                time.sleep(0.005)
            q.put("PP 1 3000 8000 1000.0 600.0 2000")
            q.put("PP 1 5000 8000 2000.0 700.0 2000")
            q.put("DONE 1 2 8000 3.0 2.0 length 0 0 2000")

        th = threading.Thread(target=feed, daemon=True)
        th.start()
        list(gen)
        self.assertEqual(got, [(1000, 600.0), (3000, 700.0)])


class QfnAdmissionReads(unittest.TestCase):
    """The real agents case (live log 11-19): almost every read is a QFN batch ADMISSION read beside the decoding
    slots - `slot 7 admitted beside the decoding slots: 196374 prompt tokens (195936 reused) read in 9897 ms` - and
    the Monitor showed no Prefill figure for it.  The adapter's slot-queue view counts the BRP steps' position
    DELTAS since the read's first one (real reads - their reported rates are never used), and the BRA line's
    figures make the exact count (prompt - reused, the total topping up what the steps counted); that stands for
    the 2..431 token "one fast chunk" follow-up reads too."""

    def engine(self, svc=None, stat=None, slot=7):
        cls = _qfn_engine_class()
        eng = cls.__new__(cls)
        eng._tls = threading.local()
        eng.slot_q = {slot: queue.Queue()}
        got = []
        if svc is not None:
            eng._tls.on_read = lambda n, ts=None, ex=False, pv=False: svc._note_read(stat, n, ts, ex, pv)
        else:
            eng._tls.on_read = lambda n, ts=None, ex=False, pv=False: got.append((n, ts, ex, pv))
        return eng, got

    @staticmethod
    def drain(eng, live=None, slot=7):
        it = eng._admission_wait(slot, threading.Event(), live if live is not None else
                                 {"state": "reading", "prompt_read": 0})
        res = None
        try:
            while True:
                next(it)
        except StopIteration as e:
            res = e.value
        return res

    def reading(self, svc, stat):
        svc.status.update(busy=True)
        svc.running.add(stat)
        return stat

    @staticmethod
    def start_marker(svc, stat, t):
        svc._read_start(stat)                         # (the engine's own start hook: its read time, and the
                                                      #  window's count for it, start there)

    def test_big_reuse_admission_counts_exactly_the_new_tokens(self):
        # the steps count positions since the read's first one (0, then 200) and the BRA line tops the count up
        # to prompt - reused, exactly; the window's figure is that count over its real span, not a reported rate
        svc = make_svc()
        t = [1000.0]
        with unittest.mock.patch("serve.server.time.time", lambda: t[0]):
            stat = self.reading(svc, reader(RequestStat()))
            self.start_marker(svc, stat, t)
            eng, got = self.engine(svc, stat)
            for ln in ("BRP 7 100 438 2000.0 44.2", "BRP 7 300 438 6000.0 44.2",
                       "BRA 7 1 prompt=196374 reused=195936 read_ms=9897 wait_ms=75"):
                eng.slot_q[7].put(ln)
            res = self.drain(eng)
            self.assertTrue(res["start"])
            self.assertEqual(stat.read_pos, 438)              # prompt - reused, exactly
            self.assertEqual(svc._read_counted(stat), 438)    # the engine-total window saw the same
            t[0] += 2.0
            self.assertEqual(svc._prefill_tok_s(), 438 / 2.0)  # the real count over the window's 2 s span
            self.assertIsNone(svc.totals["peak_prefill"])      # 438 < PEAK_MIN_READ_TOKENS: no peak from a short read

    def test_provisional_steps_never_count_as_reads_they_may_be_positions(self):
        # a BRP line is the read's start (its position may count from the prompt's start): a delta of 0 leaves
        # nothing in the sum until the steps move, and its reported rate is never used
        svc = make_svc()
        stat = self.reading(svc, reader(RequestStat()))
        eng, got = self.engine(svc, stat)
        eng.slot_q[7].put("BRP 7 96000 196374 5000.0 8000.0")       # the read's start: counts nothing
        self.assertEqual(svc._read_counted(stat), 0)                # nothing counted yet
        eng.slot_q[7].put("BRA 7 1 prompt=196374 reused=195936 read_ms=9897 wait_ms=75")
        self.drain(eng)
        self.assertEqual((stat.read_pos, svc._read_counted(stat)), (438, 438))
        self.assertIsNone(svc.totals["peak_prefill"])               # the 8000 tok/s on the line made no peak either

    def test_one_fast_chunk_2_to_431_tokens(self):
        # a follow-up read that finishes before its first BRP line: the BRA figures alone say it - and they must show
        svc = make_svc()
        t = [1000.0]
        with unittest.mock.patch("serve.server.time.time", lambda: t[0]):
            stat = self.reading(svc, reader(RequestStat()))
            self.start_marker(svc, stat, t)
            eng, got = self.engine(svc, stat)
            eng.slot_q[7].put("BRA 7 1 prompt=1220 reused=789 read_ms=33 wait_ms=5")
            self.drain(eng)
            self.assertEqual((stat.read_pos, svc._read_counted(stat)), (431, 431))
            t[0] += 1.0
            self.assertEqual(svc._prefill_tok_s(), 431.0)           # the real count over the window's 1 s span

    def test_legs_of_one_request_add_up(self):
        # after a fair slice the same request is admitted again (the held prefix reused): the legs' reads add up
        svc = make_svc()
        stat = self.reading(svc, reader(RequestStat()))
        eng, got = self.engine(svc, stat)
        eng.slot_q[7].put("BRA 7 1 prompt=5000 reused=0 read_ms=100 wait_ms=0")
        self.drain(eng)
        eng.slot_q[7].put("BRA 7 1 prompt=5020 reused=5000 read_ms=40 wait_ms=0")
        self.drain(eng)
        self.assertEqual((stat.read_pos, svc._read_counted(stat)), (5020, 5020))
        self.assertEqual(svc.totals["peak_prefill"]["tok_s"], round(5000 / 0.1, 1))   # from the first leg (>= 512)

    def test_lines_and_return_value_pass_through(self):
        eng, got = self.engine()
        for ln in ("BRP 7 400 5000 700.0 900.0", "BRP 7 1200 5000 1400.0 900.0",
                   "BRA 7 1 prompt=5000 reused=2000 read_ms=1400 wait_ms=5"):
            eng.slot_q[7].put(ln)
        live = {"state": "reading", "prompt_read": 0, "prompt_tokens": 5000}
        res = self.drain(eng, live)
        self.assertEqual(got, [(0, None, False, False), (800, None, False, False),     # position deltas since the
                               (3000, 3000 / 1.4, True, False)])                       # first BRP; the BRA total
        self.assertTrue(res["start"])
        self.assertEqual(res["figures"]["prompt_read"], 3000)       # the driver's own figures, untouched
        self.assertEqual(live["prompt_read"], 1200)                 # written through by the driver itself
        self.assertIs(type(eng.slot_q[7]), queue.Queue)             # the queue restored after the wait

    def test_a_refused_admission_counts_nothing(self):
        # the read's first BRP is its start (a delta of 0 counts nothing) and the refusal line (BRA ... 0)
        # brings no total: the engine-total window stays empty
        svc = make_svc()
        stat = self.reading(svc, reader(RequestStat()))
        eng, got = self.engine()
        eng._tls.on_read = lambda n, ts=None, ex=False, pv=False: (got.append(n), svc._note_read(stat, n, ts, ex, pv))
        eng.slot_q[7].put("BRP 7 200 5000 400.0 500.0")
        eng.slot_q[7].put("BRA 7 0 no-room")
        res = self.drain(eng)
        self.assertFalse(res["start"])
        self.assertEqual(got, [0])                                  # only the start: never a count
        self.assertEqual((stat.read_pos, svc._read_counted(stat)), (0, 0))   # a refused admission counts nothing

    def test_no_hook_passes_through(self):
        cls = _qfn_engine_class()
        eng = cls.__new__(cls)
        eng._tls = threading.local()
        q0 = queue.Queue()
        eng.slot_q = {0: q0}
        q0.put("BRA 0 0 no-room")
        res = self.drain(eng, slot=0)
        self.assertFalse(res["start"])
        self.assertIs(eng.slot_q[0], q0)


class ReadsSpreadOverTheirTime(unittest.TestCase):
    """Live 08:58:34 (11-20): three QFN admissions ended within 1 s - 10624 new tokens in 26.9 s, 10608 in 27.1 s,
    524 in 21.5 s: about 810 tok/s of real reading together.  BRP progress was sparse, so most tokens arrived in
    the BRA top-ups at the END, and the 5 s window read them all in one second: 11518 tokens -> peak 1536.7, twice
    too high.  Each read is now an interval whose exact count the window spreads over its own span."""

    def test_three_reads_ending_in_one_second_read_at_their_true_speed(self):
        svc = make_svc()
        t = [1000.0]
        with unittest.mock.patch("serve.server.time.time", lambda: t[0]):
            stats = []
            for t0, t_end, n_read in ((0.0, 26.9, 10624), (0.2, 27.1, 10608), (5.6, 21.5, 524)):
                st = reader(RequestStat())
                svc.running.add(st)
                t[0] = 1000.0 + t0
                svc._read_start(st)                       # (the engine took it: the read's interval opens)
                t[0] = 1000.0 + t0 + 1.0                  # (a sparse BRP step mid-read...)
                svc._note_read(st, 400)
                t[0] = 1000.0 + t_end                     # (...and the BRA top-up at the real end - all three of
                svc._note_read(st, n_read, None, True)    #  those land within 0.2 s of each other)
                stats.append(st)
            svc.status.update(busy=True)
            t[0] = 1027.4                                 # (the window ends 0.3 s after the last top-up)
            rate = svc._prefill_tok_s()
            self.assertGreater(rate, 500.0)               # (their true shared ~810...)
            self.assertLess(rate, 1100.0)                 # (...never the 11518 -> 1536.7 of before)
            peak = svc.totals["peak_prefill"]
            self.assertIsNotNone(peak)
            self.assertLess(peak["tok_s"], 1536.0)
            self.assertGreater(peak["tok_s"], 500.0)
            self.assertEqual(svc._read_counted(stats[0]), 10624)   # (the exact counts stand, per read)


class PrometheusPeak(unittest.TestCase):
    def test_peak_prefill_gauge(self):
        svc = make_svc()
        svc._note_peak_read(900.0, 1000, 100.0)
        prom = svc.prometheus()
        self.assertIn("maya:prefill_peak_tokens_seconds 900.0", prom)


class ReadEmittingEngine(MockEngine):
    """A mock engine that reports prompt-read steps the way the maya line's PP lines do: through the on_read hook
    run() installs on the engine's thread-local, in pieces, with the engine's mean beside them (None: this mock
    reports no rate of its own, so the server times the intervals itself)."""

    def __init__(self, tok, script, step=64, **kw):
        super().__init__(tok, script, **kw)
        self._tls = threading.local()
        self.step = step

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        cb = getattr(self._tls, "on_read", None)
        if cb:
            done = 0
            while done < len(ids):
                done = min(len(ids), done + self.step)
                cb(done, None)
                time.sleep(0.02)                      # (the reading takes wall time, like a chunk does)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class ThroughRun(unittest.TestCase):
    """The full wiring over real HTTP: run() installs the hook, the read steps count, overlapping chats get the
    summed rate and the shared label, and the totals' read_s is wall time."""

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = ReadEmittingEngine(self.tok, "ok", max_context=CTX, delay_s=0.02)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.fifo.set_capacity(2)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, text):
        body = json.dumps({"model": "m", "max_tokens": 4, "temperature": 0,
                           "messages": [{"role": "user", "content": text}]}).encode()
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=30).read())

    def metrics(self):
        with urllib.request.urlopen(self.base + "/metrics", timeout=10) as r:
            return json.loads(r.read())

    def test_two_overlapping_reads_over_http(self):
        outs = {}

        def one(k, text):
            outs[k] = self.post(text)

        # two chats of ~300 prompt bytes, read in 64-token pieces: their reads overlap (the FIFO has 2 places)
        ths = [threading.Thread(target=one, args=(k, "abcdefgh" * 40)) for k in (1, 2)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(30)
        self.assertEqual(len(outs), 2)
        m = self.metrics()
        rows = m["requests"]
        self.assertEqual(len(rows), 2)
        for row in rows:                              # each read ~320 tokens, both beside the other
            self.assertGreater(row["prompt_tokens"], 250)
            self.assertTrue(row["prefill_shared"], row)
            self.assertIsNotNone(row["prefill_tok_s"])
            self.assertGreater(row["prefill_tok_s"], 0.0)
        t = m["totals"]
        self.assertGreater(t["read_s"], 0.0)          # wall seconds with a prompt reading
        self.assertGreater(t["decode_s"], 0.0)        # ... and with an answer writing
        self.assertEqual(t["prompt_tokens"] - t["reused"], sum(r["prompt_tokens"] for r in rows))


if __name__ == "__main__":
    unittest.main()


class ReadCountedQueueStaysAQueue(unittest.TestCase):
    """Live 11-19: the QFN pump puts into slot_q[slot] while a request waits through the counting view - it must take put."""

    def test_the_wrapped_slot_queue_takes_puts(self):
        import inspect
        from serve import server as S
        src = inspect.getsource(S)
        self.assertIn("def __getattr__(self, name):", src[src.index("class _ReadCounted"):][:600])


class OnReadTakesTheProvisionalFlag(unittest.TestCase):
    """Live 11-19: the QFN view calls on_read(n, rate, False, True); the 4th argument must be `prov`, not the stat."""

    def test_on_read_lambda_signature(self):
        import inspect
        from serve import server as S
        src = inspect.getsource(S)
        line = next(x for x in src.splitlines() if "tls.on_read = lambda" in x)
        self.assertIn("prov=False, st=stat", line)
        self.assertIn("self._note_read_safe(st, n, ts, ex, prov)", line)
