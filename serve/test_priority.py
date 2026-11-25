"""serve/test_priority.py - R122: request priority. The FIFO's grant order (highest first, FIFO inside one
priority, the plain order for equals), a lower-priority answer or prompt read giving its place up at once for a
higher-priority request and continuing from exactly where it was, the request's fields (the body's "priority",
the X-Saguaro-Priority header, the config's default, a bad value a 400) and the /metrics fields - against
scripted engines that continue an answer or a half-read prompt from wherever the prompt says it got to, as the
real ones do: QFN keeps a stopped read's conversation live at the token it reached (generate.cpp #1620), GLM
pauses by its PAUSE.

    python -m unittest serve.test_priority -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import random
import string
import sys
import threading
import time
import unittest
import unittest.mock
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (IM_END, PRIORITY_NAMES, ByteTokenizer, FairLock, MockEngine, Service, serve,  # noqa: E402
                          PRIORITY_BACKGROUND, PRIORITY_HIGH, PRIORITY_NORMAL)
from serve.test_fair import ROOT, ResumingEngine, PausingEngine  # noqa: E402

LONG = "".join(random.Random(600).choices(string.ascii_letters, k=400))     # (no loop: the guard stays out)
SHORT = "s" * 20


def ask(svc, user: str, max_new: int, out: dict, prio: int | None = None, cancel=None, stop_after=None):
    """Run one request (at `prio`, normal when None) and collect its text; stop_after: the client going away."""
    ids, thinking, max_new = svc.prepare([{"role": "user", "content": user}], None,
                                         {"enable_thinking": False}, max_new)
    sampling = {} if prio is None else {"_priority": prio}
    text, n = "", 0
    gen = svc.run(ids, thinking, None, max_new, sampling, cancel or threading.Event())
    for kind, val in gen:
        if kind == "event" and val.kind == "content":
            text += val.text
            n += len(val.text)
            if stop_after and n >= stop_after:
                gen.close()                          # the client goes away mid-answer
                break
    out[user] = (text, time.monotonic())


class Env(unittest.TestCase):
    NAMES = ("SAGUARO_FAIR_SLICE_S", "SAGUARO_FAIR_READ_S", "SAGUARO_PRIORITY_AGE_S")

    def setUp(self):
        self.env = {k: os.environ.get(k) for k in self.NAMES}
        for k in self.NAMES:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class ReadEngine(ResumingEngine):
    """The prompt is read first, in chunks: one None heartbeat (a PP line) per chunk, with this request's read
    progress in `_tls.own_pp` like the drivers report it.  A consumer that goes away mid-read leaves the engine
    holding what it read: the conversation stays live at that token and the next call reads on from there - QFN's
    stopped-read behaviour (#1620, solo and batch alike).  Chunks are counted per base: a read that CONTINUED,
    not one re-read from 0, finishes its base in exactly CHUNKS of them."""
    keeps_stopped_read = True
    CHUNKS = 6

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._tls = threading.local()
        self.read_pos, self.ticks = {}, {}
        self.rlock = threading.Lock()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        with self.lock:
            self.calls.append((list(ids), max_new, dict(sampling or {})))
            base = next((b for b in self.bases if ids[:len(b)] == b), None)
            if base is None:
                base = list(ids)
                self.bases.append(base)
        key, total = id(base), len(base)
        step = max(1, -(-total // self.CHUNKS))
        pos = self.read_pos.get(key, 0)
        self._tls.own_pp = None
        while pos < total:
            pos = min(total, pos + step)
            if cancel.is_set():
                return
            self._tls.own_pp = ((pos, total), None)
            with self.rlock:
                self.read_pos[key] = pos
                self.ticks[key] = self.ticks.get(key, 0) + 1
            time.sleep(self.delay)
            yield None                                  # a PP line: a keep-alive, and a read tick
        self._tls.own_pp = None
        text = self.tok.decode(base)
        answer = next(v for k, v in self.answers.items() if k in text)
        script = self.tok.encode(answer) + self.end
        for t in script[len(ids) - len(base):][:max_new]:
            if cancel.is_set():
                return
            time.sleep(self.delay)
            yield t


class PausingRead(ReadEngine):
    """GLM's way: can_pause, and PAUSE ends the read a couple of ticks later with its DONE saying "cancel" -
    what it read stays in the engine, the rest is read when the request's turn comes back."""
    keeps_stopped_read = False
    can_pause = True

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.paused, self.pauses = False, 0

    def pause(self, keep=False):
        self.paused = True
        self.pauses += 1

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.paused = False
        late = None
        for t in super().generate(ids, max_new, sampling, cancel, embeddings):
            if self.paused and t is None:
                late = 2 if late is None else late - 1
                if late == 0:
                    self.last = {"finish": "cancel", "generated": 0, "prompt_ms": 0.0, "decode_ms": 0.0}
                    return
            yield t
        self.last = {"finish": "stop", "generated": 0, "prompt_ms": 0.0, "decode_ms": 0.0}


class FairLockPriority(Env):
    def waiting_order(self, prios):
        """One thread per waiter, enqueued in the order listed; -> the order they got the engine."""
        lock, order = FairLock(), []
        lock.acquire()
        threads = []
        for i, p in enumerate(prios):
            th = threading.Thread(target=lambda i=i, p=p: (lock.acquire(priority=p), order.append(i),
                                                           lock.release()))
            th.start()
            threads.append(th)
            while lock.waiting() < i + 1:               # the next one asks only once this one is in line
                time.sleep(0.001)
        lock.release()
        for th in threads:
            th.join(5)
        return lock, order

    def test_equal_priorities_keep_the_plain_fifo_order(self):
        lock, order = self.waiting_order([None] * 6)
        self.assertEqual(order, list(range(6)))         # unchanged: arrival order, as ever
        self.assertFalse(lock.locked())

    def test_the_highest_priority_is_granted_first_fifo_inside_one(self):
        _lock, order = self.waiting_order([PRIORITY_NORMAL, PRIORITY_BACKGROUND, PRIORITY_HIGH, PRIORITY_NORMAL])
        self.assertEqual(order, [2, 0, 3, 1])           # high first, the normals by arrival, the background last

    def test_a_waiting_background_does_not_hold_back_a_new_normal_request(self):
        lock = FairLock()
        lock.acquire()
        bg = lock.enqueue(priority=PRIORITY_BACKGROUND)
        lock.release()
        self.assertTrue(lock.acquire(timeout=1))        # the fresh normal steps over the queued background
        self.assertEqual(lock.waiting(), 1)             # the background keeps its place, behind it now
        lock.release()
        self.assertTrue(lock.wait_turn(bg, timeout=2))
        lock.release()

    def test_the_aging_setting_is_off_by_default_and_on_above_zero(self):
        lock = FairLock()
        bg = lock.enqueue(units=1, priority=PRIORITY_BACKGROUND)
        time.sleep(0.05)                               # no promotion by waiting at all: the default
        self.assertEqual(lock._eff(bg), PRIORITY_BACKGROUND)
        self.assertEqual(lock.waiting(above=PRIORITY_BACKGROUND), 0)      # it never cuts a running background
        os.environ["SAGUARO_PRIORITY_AGE_S"] = "0.01"                     # set: that long of a wait counts normal
        self.assertEqual(lock._eff(bg), PRIORITY_NORMAL)
        self.assertEqual(lock.waiting(above=PRIORITY_BACKGROUND), 1)
        lock.leave(bg)

    def test_a_waiting_normal_holds_back_a_new_one_of_its_own(self):
        lock = FairLock()
        lock.acquire()
        one = lock.enqueue()                            # a normal waiter (nobody wait_turn-ing it here)
        lock.release()
        self.assertFalse(lock.acquire(timeout=0.05))    # no fast path past an equal: the FIFO order, unchanged
        self.assertEqual(lock.waiting(), 1)             # (the timed-out one left the line)
        lock.leave(one)

    def test_an_all_places_waiter_holds_back_its_own_not_a_higher_one(self):
        lock = FairLock(2)
        lock.acquire(units=2)
        allp = lock.enqueue(units=2)                    # an image encode, a context reload
        low = lock.enqueue(units=1)                     # behind it, its own priority: held back, as before
        self.assertTrue(lock._first(allp))
        self.assertFalse(lock._first(low))
        lock.release(units=2)                           # both places free now
        self.assertFalse(lock.acquire(timeout=0.05, units=1))    # the all-places waiter still comes first
        hi = lock.enqueue(units=1, priority=PRIORITY_HIGH)       # a high one does not wait behind it
        self.assertTrue(lock.wait_turn(hi, timeout=2))
        lock.release(units=1)
        self.assertTrue(lock.wait_turn(allp, timeout=2))         # then the all-places waiter, before its equal
        lock.leave(low)

    def test_waiting_counts_only_the_ones_that_step_ahead(self):
        lock = FairLock()
        lock.enqueue(priority=PRIORITY_BACKGROUND)
        lock.enqueue()
        lock.enqueue(priority=PRIORITY_HIGH)
        self.assertEqual(lock.waiting(), 3)
        self.assertEqual(lock.waiting(above=PRIORITY_NORMAL), 1)         # only the high one cuts a normal runner
        self.assertEqual(lock.waiting(above=PRIORITY_BACKGROUND), 2)     # normal and high both cut a background
        self.assertEqual(lock.highest_waiting(), PRIORITY_HIGH)
    def test_a_queued_background_ages_into_normal(self):
        os.environ["SAGUARO_PRIORITY_AGE_S"] = "0.05"
        lock = FairLock()
        lock.acquire()
        bg = lock.enqueue(priority=PRIORITY_BACKGROUND)
        time.sleep(0.08)
        nm = lock.enqueue()                             # a fresh normal, behind the aged background
        self.assertTrue(lock._first(bg))                # aged into normal, it keeps its earlier arrival place
        self.assertFalse(lock._first(nm))
        lock.release()
        self.assertTrue(lock.wait_turn(bg, timeout=2))
        lock.release()
        self.assertTrue(lock.wait_turn(nm, timeout=2))
        lock.release()

    def test_yield_place_keeps_its_priority(self):
        lock = FairLock()
        lock.acquire()
        again = lock.yield_place(priority=PRIORITY_HIGH)         # the high answer that gave its place up
        hi = lock.enqueue(priority=PRIORITY_HIGH)                # waits behind it: a resumed request keeps its claim
        self.assertTrue(lock._first(again))
        self.assertFalse(lock._first(hi))
        lock.leave(again)
        lock.leave(hi)

    def test_only_the_waiters_above_cut_and_only_as_many_as_needed(self):
        lock = FairLock(2)
        lock.acquire(units=1)
        lock.acquire(units=1)
        self.assertFalse(lock.claim_yield(0.0, above=PRIORITY_BACKGROUND))   # nobody above it at all yet
        lock.enqueue(units=1, priority=PRIORITY_BACKGROUND)                  # an equal is never a reason
        self.assertFalse(lock.claim_yield(0.0, above=PRIORITY_BACKGROUND))
        self.assertEqual(lock.waiting(above=PRIORITY_BACKGROUND), 0)
        lock.enqueue(units=1)                          # one normal waiter: cuts a background runner, one of them
        self.assertFalse(lock.claim_yield(0.0, above=PRIORITY_NORMAL))       # not a normal one (equal)
        self.assertTrue(lock.claim_yield(0.0, above=PRIORITY_BACKGROUND))    # one gives its place up
        self.assertFalse(lock.claim_yield(0.0, above=PRIORITY_BACKGROUND))   # the other stays: it is covered
        lock.unclaim_yield()

    def test_a_free_place_nobody_yields(self):
        lock = FairLock(2)
        lock.acquire(units=1)
        lock.enqueue(units=1, priority=PRIORITY_HIGH)   # a place is free: the waiter takes it by itself
        self.assertFalse(lock.claim_yield(0.0, above=PRIORITY_BACKGROUND))


class ServicePreempts(Env):
    def setUp(self):
        super().setUp()
        self.tok = ByteTokenizer()
        self.engine = ResumingEngine(self.tok, {"long": LONG, "short": SHORT})
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def race(self, low_prio=PRIORITY_BACKGROUND, high_prio=PRIORITY_HIGH):
        out = {}
        a = threading.Thread(target=ask, args=(self.svc, "long one", 4096, out, low_prio))
        a.start()
        while self.svc.latest_request not in self.svc.current_requests:
            time.sleep(0.001)
        time.sleep(0.05)                                # the answer is past its first tokens
        b = threading.Thread(target=ask, args=(self.svc, "short one", 4096, out, high_prio))
        b.start()
        a.join(30)
        b.join(30)
        return out

    def long_calls(self):
        base = self.engine.bases[0]
        return [c for c in self.engine.calls if c[0][:len(base)] == base]

    def test_a_high_request_cuts_a_background_answer_at_once(self):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            out = self.race()
        self.assertEqual(out["short one"][0], SHORT)
        self.assertEqual(out["long one"][0], LONG)      # nothing lost or repeated at the seam
        self.assertLess(out["short one"][1], out["long one"][1])
        calls = self.long_calls()
        self.assertGreaterEqual(len(calls), 2)          # it was cut, and continued from where it got to
        written = len(calls[1][0]) - len(calls[0][0])
        self.assertGreater(written, 0)                  # prompt + the tokens it had written
        self.assertEqual(calls[1][1], calls[0][1] - written)            # the budget left, not the whole again
        text = log.getvalue()
        self.assertIn("[saguaro] priority: a background answer paused at", text)
        self.assertIn("for a high-priority request", text)
        self.assertIn("[saguaro] priority: the background answer continues", text)
        self.assertNotIn("fair slice", text)            # the priority, not the SAGUARO_FAIR_SLICE_S timer
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual(self.svc.fifo.waiting(), 0)
        self.assertEqual(self.svc.fifo._yields, 0)
        self.assertEqual(self.svc.status["queued"], 0)

    def test_the_pause_engine_keeps_the_state_and_the_stream(self):
        self.engine = PausingEngine(self.tok, {"long": LONG, "short": SHORT})
        self.svc.engine = self.engine
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            out = self.race()
        self.assertEqual(out["long one"][0], LONG)      # the same streamed text as without the pause
        self.assertEqual(out["short one"][0], SHORT)
        self.assertLess(out["short one"][1], out["long one"][1])
        self.assertGreaterEqual(self.engine.pauses, 1)  # PAUSE where the engine can, not a cut
        self.assertFalse(self.svc.fifo.locked())

    def test_equals_never_preempt(self):
        for prio in (PRIORITY_NORMAL, PRIORITY_BACKGROUND):
            with self.subTest(prio=prio):
                self.engine.calls.clear()
                self.engine.bases.clear()
                log = io.StringIO()
                with contextlib.redirect_stdout(log):
                    out = self.race(low_prio=prio, high_prio=prio)
                self.assertEqual(out["long one"][0], LONG)
                self.assertEqual(out["short one"][0], SHORT)
                self.assertLess(out["long one"][1], out["short one"][1])   # the second waits, exactly as before
                self.assertEqual(len(self.long_calls()), 1)                # never cut
                self.assertNotIn("priority:", log.getvalue())

    def test_a_queued_background_ages_and_the_answer_steps_aside(self):
        os.environ["SAGUARO_PRIORITY_AGE_S"] = "0.02"  # a waiting background counts as normal once it aged
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            out = self.race(low_prio=PRIORITY_BACKGROUND, high_prio=PRIORITY_BACKGROUND)
        self.assertLess(out["short one"][1], out["long one"][1])   # the aged waiter cuts the running background
        self.assertEqual(out["long one"][0], LONG)
        self.assertIn("priority: a background answer paused", log.getvalue())

    def test_a_cancel_while_paused_leaves_the_line(self):
        out, log = {}, io.StringIO()
        cancel = threading.Event()
        with contextlib.redirect_stdout(log):
            a = threading.Thread(target=ask, args=(self.svc, "long one", 4096, out, PRIORITY_BACKGROUND),
                                 kwargs={"cancel": cancel})
            a.start()
            while self.svc.latest_request not in self.svc.current_requests:
                time.sleep(0.001)
            b = threading.Thread(target=ask, args=(self.svc, "short one", 4096, out, PRIORITY_HIGH))
            b.start()
            while not any(getattr(s, "paused_for_priority", False) for s in list(self.svc.running)):
                time.sleep(0.005)                       # it is cut, and waits in the line with keep-alives
            self.assertIn(self.svc.fifo.waiting(), (1, 2))         # (1 once the high one took the free place)
            self.assertTrue(self.svc.fifo.locked())
            cancel.set()                                # its client goes away while it waits
            a.join(10)
            b.join(30)
        self.assertEqual(out["short one"][0], SHORT)
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual(self.svc.fifo.waiting(), 0)    # (the line it left)
        self.assertEqual(self.svc.status["queued"], 0)

    def test_a_long_pause_neither_ends_it_nor_promotes_it(self):
        other = "".join(random.Random(900).choices(string.ascii_letters, k=400))
        self.engine = ResumingEngine(self.tok, {"long": LONG, "other": other, "short": SHORT}, delay_s=0.003)
        self.svc.engine = self.engine
        out, log = {}, io.StringIO()
        with contextlib.redirect_stdout(log):
            a = threading.Thread(target=ask, args=(self.svc, "long one", 4096, out, PRIORITY_BACKGROUND))
            a.start()
            while self.svc.latest_request not in self.svc.current_requests:
                time.sleep(0.001)
            b = threading.Thread(target=ask, args=(self.svc, "other one", 4096, out, PRIORITY_HIGH))
            b.start()
            while not any(getattr(s, "paused_for_priority", False) for s in list(self.svc.running)):
                time.sleep(0.005)
            self.assertEqual([s.priority for s in self.svc.running if getattr(s, "paused_for_priority", False)],
                             [PRIORITY_NAMES[PRIORITY_BACKGROUND]])   # it waits: still a background request
            clock = [time.monotonic()]

            def fake():                                               # a minute of waiting per call: the ten-
                clock[0] += 60                                        #  minute note comes without the ten minutes
                return clock[0]

            with unittest.mock.patch("time.monotonic", new=fake):
                stop = time.time() + 30                               # (the wall clock stays real for the wait)
                while "has been paused for" not in log.getvalue() and time.time() < stop:
                    time.sleep(0.02)
            b.join(60)
            a.join(60)
        self.assertIn("priority: a background answer has been paused for", log.getvalue())   # it says so, once
        self.assertEqual(log.getvalue().count("has been paused for"), 1)
        self.assertEqual(out["long one"][0], LONG)                    # and it went on to its end: waiting ends
        self.assertEqual(out["other one"][0], other)                  #   nothing, and promotes nothing

    def test_parallel_two_cuts_only_as_many_as_needed(self):
        longs = {f"long {c}": "".join(random.Random(700 + i).choices(string.ascii_letters, k=400))
                 for i, c in enumerate("AB")}
        self.engine = ResumingEngine(self.tok, {**longs, "short": SHORT}, delay_s=0.003)
        self.svc.engine = self.engine
        self.svc.fifo.set_capacity(2)                   # two conversations at once, both places taken
        out, threads = {}, []
        for user in longs:
            th = threading.Thread(target=ask, args=(self.svc, user, 4096, out, PRIORITY_BACKGROUND))
            th.start()
            threads.append(th)
        while len(self.svc.current_requests) < 2:
            time.sleep(0.001)
        time.sleep(0.05)                                # both answers are under way
        hi = threading.Thread(target=ask, args=(self.svc, "short one", 4096, out, PRIORITY_HIGH))
        hi.start()
        for th in threads + [hi]:
            th.join(60)
        self.assertEqual(out["short one"][0], SHORT)
        for user, text in longs.items():
            self.assertEqual(out[user][0], text)        # both continue intact
        resumed = [b for b in self.engine.bases if sum(1 for c in self.engine.calls
                     if c[0][:len(b)] == b and len(c[0]) > len(b)) > 0]
        self.assertEqual(len(resumed), 1)               # exactly one answer gave its place up for the one waiter
        self.assertLess(out["short one"][1], out["long A"][1])
        self.assertLess(out["short one"][1], out["long B"][1])
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual(self.svc.fifo._yields, 0)


class ReadsPreempt(Env):
    def setUp(self):
        super().setUp()
        self.tok = ByteTokenizer()

    def race_read(self, engine):
        svc = Service(engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        out, log = {}, io.StringIO()
        with contextlib.redirect_stdout(log):
            a = threading.Thread(target=ask, args=(svc, "long one", 4096, out, PRIORITY_BACKGROUND))
            a.start()
            while sum(engine.ticks.values()) < 2:
                time.sleep(0.001)                       # it is in the middle of reading its prompt
            b = threading.Thread(target=ask, args=(svc, "short one", 4096, out, PRIORITY_HIGH))
            b.start()
            a.join(30)
            b.join(30)
        return out, log.getvalue()

    def test_a_qfn_read_is_cut_and_continues_from_where_it_was(self):
        engine = ReadEngine(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        out, log = self.race_read(engine)
        self.assertEqual(out["short one"][0], SHORT)
        self.assertLess(out["short one"][1], out["long one"][1])
        self.assertEqual(out["long one"][0], LONG)
        self.assertEqual(engine.ticks[id(engine.bases[0])], ReadEngine.CHUNKS)   # every prompt chunk read exactly
        #                                                    # once: what the engine still held was not read again
        long_calls = [c for c in engine.calls if c[0][:len(engine.bases[0])] == engine.bases[0]]
        self.assertEqual(len(long_calls), 2)
        self.assertEqual(long_calls[1][0], long_calls[0][0])          # the same prompt: nothing written yet
        self.assertIn("priority: a background prompt read paused", log)
        self.assertIn("prompt tokens for a high-priority request", log)
        self.assertIn("priority: the background prompt read continues", log)

    def test_a_glm_read_pauses_by_its_pause(self):
        engine = PausingRead(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        out, log = self.race_read(engine)
        self.assertEqual(out["short one"][0], SHORT)
        self.assertLess(out["short one"][1], out["long one"][1])
        self.assertEqual(out["long one"][0], LONG)
        self.assertGreaterEqual(engine.pauses, 1)       # PAUSE, not a STOP: the state with what it read stays
        self.assertEqual(engine.ticks[id(engine.bases[0])], ReadEngine.CHUNKS)

    def test_an_engine_that_would_lose_a_stopped_read_finishes_it(self):
        engine = ReadEngine(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        engine.keeps_stopped_read = False               # no PAUSE either: a cut would lose the read - no cutting
        out, log = self.race_read(engine)
        self.assertEqual(out["long one"][0], LONG)
        self.assertEqual(out["short one"][0], SHORT)
        self.assertNotIn("prompt read paused", log)
        self.assertEqual(engine.ticks[id(engine.bases[0])], ReadEngine.CHUNKS)

    def test_the_qfn_engine_reports_where_the_property_comes_from(self):
        from serve.server import _qfn_engine_class     # the property: a stopped request keeps its conversation
        cls = _qfn_engine_class()                      # live at the token it reached - generate.cpp #1620, solo
        self.assertTrue(cls.keeps_stopped_read)        # GEN as well as the batch slots - so a read may be cut
        self.assertFalse(cls.can_pause)                # by a STOP (it has no PAUSE line)


class AcrossThePause(Env):
    def setUp(self):
        super().setUp()

        class Tok(ByteTokenizer):
            SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>"]

        self.tok = Tok()
        self.end = self.tok.encode(IM_END, parse_special=True)
        self.think_end = self.tok.encode("</think>", parse_special=True)

    def test_the_thinking_budget_counts_from_where_it_got_to(self):
        class ThinkEngine:
            """Reasoning first (letters: no sentence or line for the soft close), then the end of the thinking,
            then the answer; records the prompt, budget and sampling of every call."""
            def __init__(self, tok, end, think_end, delay=0.002):
                self.tok, self.end, self.think_end = tok, end, think_end
                self.delay, self.max_context, self.can_pause = delay, 32768, False
                self.think = "".join(random.Random(11).choices(string.ascii_lowercase, k=120))
                self.bases, self.calls, self.lock = [], [], threading.Lock()
                self.last = {}

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                with self.lock:
                    self.calls.append((list(ids), max_new, dict(sampling or {})))
                    base = next((b for b in self.bases if ids[:len(b)] == b), None)
                    if base is None:
                        base = list(ids)
                        self.bases.append(base)
                script = self.tok.encode(self.think) + self.think_end + self.tok.encode("8") + self.end
                for t in script[len(ids) - len(base):][:max_new]:
                    if cancel.is_set():
                        return
                    time.sleep(self.delay)
                    yield t

        engine = ThinkEngine(self.tok, self.end, self.think_end)
        svc = Service(engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.think_budget = 40
        ids, thinking, max_new = svc.prepare([{"role": "user", "content": "think long one"}], None, {}, 2048)
        self.assertTrue(thinking)                      # (the budget rides a thinking request)
        out = {}

        def low():
            text = ""
            for kind, val in svc.run(ids, True, None, max_new, {"_priority": PRIORITY_BACKGROUND},
                                     threading.Event()):
                if kind == "event" and val.kind == "content":
                    text += val.text
            out["long one"] = text

        a = threading.Thread(target=low)
        a.start()
        while not svc.current_requests:
            time.sleep(0.001)
        hi = threading.Thread(target=ask, args=(svc, "short one", 4096, out, PRIORITY_HIGH))
        hi.start()                                     # the person asks while the answer is still thinking
        a.join(30)
        hi.join(30)
        self.assertEqual(out["long one"], "8")
        calls = [c for c in engine.calls if c[0][:len(ids)] == ids]
        self.assertGreaterEqual(len(calls), 2)
        written = len(calls[1][0]) - len(calls[0][0])
        self.assertGreater(written, 0)
        self.assertEqual(calls[1][0][:len(calls[0][0])], calls[0][0])   # prompt + what it had written
        self.assertIn("_think_budget", calls[0][2])
        self.assertLess(written, calls[0][2]["_think_budget"])          # cut inside the thinking, not past it
        self.assertEqual(calls[1][2]["_think_budget"], max(1, calls[0][2]["_think_budget"] - written))
        self.assertEqual(calls[1][1], calls[0][1] - written)            # the output budget follows too

    def test_the_loop_guard_survives_a_pause(self):
        block = "".join(random.Random(40).choices(string.ascii_lowercase, k=40))
        script = "".join(random.Random(41).choices(string.ascii_letters, k=100)) + block * 8
        answers = {"long": script + IM_END, "short": SHORT}

        def run_both(interrupted):
            engine = ResumingEngine(self.tok, answers)
            svc = Service(engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
            out, log = {}, io.StringIO()
            with contextlib.redirect_stdout(log):
                if interrupted:
                    a = threading.Thread(target=ask, args=(svc, "long one", 4096, out, PRIORITY_BACKGROUND))
                    a.start()
                    while svc.latest_request not in svc.current_requests:
                        time.sleep(0.001)
                    time.sleep(0.05)
                    b = threading.Thread(target=ask, args=(svc, "short one", 4096, out, PRIORITY_HIGH))
                    b.start()
                    a.join(30)
                    b.join(30)
                else:
                    ask(svc, "long one", 4096, out, PRIORITY_BACKGROUND)
            return out["long one"][0], log.getvalue()

        plain, plog = run_both(False)
        self.assertIn("loop guard", plog)              # the guard acted: this answer really loops
        cut, clog = run_both(True)
        self.assertIn("priority: a background answer paused", clog)
        self.assertIn("loop guard", clog)
        self.assertEqual(cut, plain)                   # the same answer, and the same guard actions, across it


class WaitKeepAlive(Env):
    """R122 step 3b: a request that is queued or paused for a higher priority must not let its client's idle
    timeout kill it. The keep-alive comments stay; once SAGUARO_WAIT_EVENT_S goes by without a real event the
    streams add one EMPTY event (chat.completions a chunk whose delta is {}, /v1/messages an Anthropic ping,
    /v1/responses another response.in_progress; the ollama streams nothing), never text - the same request when
    it runs undisturbed streams exactly the same answer text; non-streaming gets its 200 and headers early and a
    space before the JSON. Parsed back with the client-side rules."""
    NAMES = Env.NAMES + ("SAGUARO_QUEUE_PING_S", "SAGUARO_WAIT_EVENT_S", "SAGUARO_WAIT_SPACE_S")
    WAIT = "".join(random.Random(701).choices(string.ascii_lowercase, k=30))     # short: runs inside the space limit

    def setUp(self):
        super().setUp()
        self.tok = ByteTokenizer()
        os.environ["SAGUARO_QUEUE_PING_S"] = "0.1"
        os.environ["SAGUARO_WAIT_EVENT_S"] = "0.3"
        os.environ["SAGUARO_WAIT_SPACE_S"] = "0.3"
        self.engine = ResumingEngine(self.tok, {"wait": WaitKeepAlive.WAIT, "block": LONG}, delay_s=0.004)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        super().tearDown()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def stream(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return [raw.decode().rstrip("\r\n") for raw in r]

    def blocked(self, make):
        """Run `make` while the other conversation holds the engine; returns its result."""
        a = threading.Thread(target=self.post,
                             args=("/v1/chat/completions", {"model": "m", "stream": False,
                                                            "messages": [{"role": "user", "content": "block one"}]}))
        a.start()
        while not self.svc.current_requests:
            time.sleep(0.001)
        time.sleep(0.1)
        try:
            return make()
        finally:
            a.join(60)

    @staticmethod
    def sse(lines):
        """The client's own SSE rules: comments (`:`) dropped, `data:` payloads parsed, per `event:` name."""
        events, comments, name = [], 0, "message"
        for line in lines:
            if line.startswith(":"):
                comments += 1
            elif line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                if line[6:] == "[DONE]":
                    break
                events.append((name, json.loads(line[6:])))
        return events, comments

    def test_a_queued_chat_stream_gets_empty_chunks_and_no_text(self):
        body = lambda: {"model": "m", "stream": True, "messages": [{"role": "user", "content": "wait one"}]}
        plain, _ = self.sse(self.stream("/v1/chat/completions", body()))
        events, comments = self.blocked(lambda: self.sse(self.stream("/v1/chat/completions", body())))
        text = lambda evs: "".join(p["choices"][0]["delta"].get("content", "") for _n, p in evs)
        self.assertEqual(text(events), text(plain))   # the wait changed the answer text in no way
        empty = [p for _n, p in events if p["choices"][0]["delta"] == {} and p["choices"][0]["finish_reason"] is None]
        self.assertGreaterEqual(len(empty), 1)        # the empty event: an event, not a comment only
        self.assertTrue(all(p["id"] == events[0][1]["id"] and p["object"] == "chat.completion.chunk"
                            for p in empty))          # it belongs to this stream
        self.assertGreater(comments, 0)               # the keep-alive comments stayed too

    def test_a_queued_anthropic_stream_pings(self):
        body = lambda: {"model": "m", "max_tokens": 256, "stream": True,
                        "messages": [{"role": "user", "content": "wait one"}]}
        text = lambda evs: "".join(p["delta"]["text"] for n, p in evs
                                   if n == "content_block_delta" and p["delta"].get("type") == "text_delta")
        plain, _ = self.sse(self.stream("/v1/messages", body()))
        events, comments = self.blocked(lambda: self.sse(self.stream("/v1/messages", body())))
        self.assertIn("ping", [n for n, _p in events])    # Anthropic's own ping event, while it waited
        self.assertEqual(text(events), text(plain))       # nothing of the ping in the answer
        self.assertGreater(comments, 0)

    def test_a_queued_responses_stream_repeats_in_progress(self):
        body = lambda: {"model": "m", "input": "wait one", "stream": True}
        text = lambda evs: "".join(p["delta"] for n, p in evs if n == "response.output_text.delta")
        plain, _ = self.sse(self.stream("/v1/responses", body()))
        events, comments = self.blocked(lambda: self.sse(self.stream("/v1/responses", body())))
        names = [n for n, _p in events]
        self.assertEqual(names[0], "response.created")
        self.assertGreaterEqual(names.count("response.in_progress"), 2)   # the keep-alive's, not the start's alone
        self.assertEqual(text(events), text(plain))
        self.assertGreater(comments, 0)

    def test_the_ollama_stream_gets_nothing_extra(self):
        self.svc.model_management = True                 # (/api/chat is gated on model management)
        body = lambda: {"model": "m", "messages": [{"role": "user", "content": "wait one"}], "stream": True}
        plain = self.stream("/api/chat", body())
        waited = self.blocked(lambda: self.stream("/api/chat", body()))
        norm = lambda lines: [sorted(json.loads(l)) for l in lines if l.strip()]
        self.assertEqual(norm(waited), norm(plain))    # nothing extra while queued: an empty line breaks ndjson

    def test_a_queued_nonstream_answer_gets_its_headers_early(self):
        body = {"model": "m", "stream": False, "messages": [{"role": "user", "content": "wait one"}]}
        plain = json.loads(self.post("/v1/chat/completions", body)[1])
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            s, out = self.blocked(lambda: self.post("/v1/chat/completions", body))
        self.assertEqual(s, 200)
        self.assertTrue(out.startswith(b" "))            # spaces while it waited, valid before the JSON
        self.assertEqual(json.loads(out)["choices"][0]["message"]["content"],
                         plain["choices"][0]["message"]["content"])
        self.assertIn("the status and headers go out now", log.getvalue())
        self.assertIn("an error after this cannot change the status", log.getvalue())

    def test_a_paused_nonstream_answer_gets_its_headers_at_once(self):
        os.environ["SAGUARO_WAIT_SPACE_S"] = "30"        # only the pause starts it, not the waiting
        body = {"model": "m", "stream": False, "priority": "background",
                "messages": [{"role": "user", "content": "block one"}]}
        plain = json.loads(self.post("/v1/chat/completions", body)[1])
        log = io.StringIO()
        box = {}
        with contextlib.redirect_stdout(log):
            a = threading.Thread(target=lambda: box.update(long=self.post("/v1/chat/completions", body)))
            a.start()
            while not self.svc.current_requests:
                time.sleep(0.001)
            hi = self.post("/v1/chat/completions", {"model": "m", "stream": False, "priority": "high",
                                                    "messages": [{"role": "user", "content": "wait one"}]})
            a.join(60)
        self.assertTrue(box["long"][1].startswith(b" "))          # the answer was cut: the status went out at once
        self.assertEqual(json.loads(box["long"][1])["choices"][0]["message"]["content"],
                         plain["choices"][0]["message"]["content"])   # same answer: paused, continued, nothing lost
        self.assertFalse(hi[1].startswith(b" "))                  # the one that ran needs none of it
        self.assertIn("the status and headers go out now", log.getvalue())


class RequestFields(Env):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = MockEngine(cls.tok, ["hello there"])
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def post(self, path, body, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_every_api_takes_the_field(self):
        chat = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        for path, req in (("/v1/chat/completions", {**chat, "stream": False, "priority": "background"}),
                          ("/v1/messages", {**chat, "max_tokens": 64, "priority": "high"}),
                          ("/v1/responses", {"model": "m", "input": "hi", "stream": False,
                                             "priority": "background"})):
            s, b = self.post(path, req)
            self.assertEqual(s, 200, f"{path}: {b[:200]!r}")
        s, b = self.post("/v1/chat/completions", {**chat, "stream": False}, {"X-Saguaro-Priority": "high"})
        self.assertEqual(s, 200, b[:200])              # and the header on its own

    def test_a_bad_value_is_a_400_that_names_the_field(self):
        chat = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False}
        s, b = self.post("/v1/chat/completions", {**chat, "priority": "urgent"})
        self.assertEqual(s, 400)
        self.assertIn('priority: "high", "normal" or "background"', json.loads(b)["error"]["message"])
        s, b = self.post("/v1/chat/completions", chat, {"X-Saguaro-Priority": "5"})
        self.assertEqual(s, 400)
        self.assertIn("X-Saguaro-Priority", json.loads(b)["error"]["message"])

    def test_the_body_wins_the_header_loses_and_the_config_default_applies(self):
        svc = self.svc
        req = {"messages": [], "priority": "normal", "_priority_header": "background"}
        svc.check_request(req)
        self.assertEqual(req["_priority"], PRIORITY_NORMAL)        # the body wins the header
        req = {"messages": [], "_priority_header": "background"}
        svc.check_request(req)
        self.assertEqual(req["_priority"], PRIORITY_BACKGROUND)
        req = {"messages": []}
        svc.check_request(req)
        self.assertEqual(req["_priority"], PRIORITY_NORMAL)        # nothing said anywhere: normal
        svc.priority_default = PRIORITY_BACKGROUND                 # the run config's "priority"
        try:
            req = {"messages": []}
            svc.check_request(req)
            self.assertEqual(req["_priority"], PRIORITY_BACKGROUND)
            req = {"messages": [], "priority": "high"}
            svc.check_request(req)
            self.assertEqual(req["_priority"], PRIORITY_HIGH)      # a request still overrides the config
        finally:
            svc.priority_default = None
        for bad in ("urgent", 5, True, {"p": 1}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                svc.check_request({"messages": [], "priority": bad})

    def test_the_metrics_rows_and_the_live_state_carry_it(self):
        answer = "".join(random.Random(9).choices(string.ascii_lowercase, k=60))     # (no loop)
        self.svc.engine = MockEngine(self.tok, [answer], tok_s=25)                   # slow: it is still running
        try:
            a = threading.Thread(target=self.post,
                                 args=("/v1/chat/completions",
                                       {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                        "stream": False, "priority": "background"}))
            a.start()
            time.sleep(0.5)
            with urllib.request.urlopen(self.base + "/metrics", timeout=15) as r:
                m = json.loads(r.read())
            a.join(30)
            rows = m["live"]["chats"]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["priority"], "background")     # the "Right now" row's tag
            self.assertFalse(rows[0]["paused_for_priority"])
            self.assertEqual(m["live"]["priority"], "background")   # and the live state's fields
            self.assertFalse(m["live"]["paused_for_priority"])
        finally:
            self.svc.engine = self.engine
        with urllib.request.urlopen(self.base + "/metrics", timeout=15) as r:
            m = json.loads(r.read())
        self.assertEqual(m["requests"][0]["priority"], "background")   # the finished request's row says it too


if __name__ == "__main__":
    unittest.main()
