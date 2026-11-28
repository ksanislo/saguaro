"""serve/test_priority.py - R122: request priority. The FIFO's grant order (highest first, FIFO inside one
priority, the plain order for equals), a lower-priority answer or prompt read giving its place up at once for a
higher-priority request and continuing from exactly where it was, the request's fields (the body's "priority",
the X-Saguaro-Priority header, the config's default, a bad value a 400) and the /metrics fields - against
scripted engines that continue an answer or a half-read prompt from wherever the prompt says it got to, as the
real ones do: QFN keeps a stopped read's conversation live at the token it reached - solo (generate.cpp #1620; a
batch slot drops a cancelled read, so a batch read is never cut) - and GLM pauses by its PAUSE. saguaro-hy4 keeps a
stopped read solo too (hy4_main.cpp:1086, :1106-1110; P124i: the driver reports keeps_stopped_read for its INFO
engine_kind=hy4, and a LOW read steps aside for a waiting NORMAL one as well; saguaro-mm3 the same by its INFO line,
P124j - and from P124u any line engine can claim it itself: READY carrying `keepread`, what saguaro-ds4 says when it
keeps a stopped read live, keeps the read at the point it stored); P125l gives it to saguaro-ik2 SOLO by its INFO engine_kind=ik2 line - its
STOP lands between the prompt's chunk passes and (after the lost_read.sh fix) the engine keeps the read AT that
boundary, snapshot and rows (ik2_main.cu:10821-10846, :10888-10900), while its --parallel line drops a cut read
(:10494-10499), so only the solo driver is given it. Who steps aside
(Ken's 11-25 block): a LOW answer gives its place up whenever a normal or high request runs or waits - even when
a place is free, all running lows at once; a NORMAL one only for a waiting high one that does not fit; a HIGH one
never.

    python -m unittest serve.test_priority -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import random
import re
import shutil
import socket
import string
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import model_api  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (IM_END, PRIORITY_NAMES, ByteTokenizer, FairLock, MockEngine, Service, StrataEngine,  # noqa: E402
                          serve, PRIORITY_HIGH, PRIORITY_LOW, PRIORITY_NORMAL)
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
    NAMES = ("SAGUARO_FAIR_SLICE_S", "SAGUARO_FAIR_READ_S", "SAGUARO_PRIORITY_AGE_S", "SAGUARO_PRIORITY_IDLE_S")

    def setUp(self):
        self.env = {k: os.environ.get(k) for k in self.NAMES}
        for k in self.NAMES:
            os.environ.pop(k, None)
        os.environ["SAGUARO_PRIORITY_IDLE_S"] = "0"      # the preemption races below cut and continue at once:
        #  the idle rule's grace has its own class (IdleRule) - nowhere else does anything wait for it

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
    stopped-read behaviour (#1620, the SOLO path: a batch slot drops a cancelled read - "a read that failed keeps
    none" - so the server must not cut one, see the batch test below).  Chunks are counted per base: a read that
    CONTINUED, not one re-read from 0, finishes its base in exactly CHUNKS of them."""
    keeps_stopped_read = True
    CHUNKS = 6
    reports_reused = True           # it says RESUME (the engine's own reused figure) like QFN's solo engine does

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
        if self.reports_reused:                         # (RESUME: what THIS call reads on from - P124b's figure)
            self.reused = pos
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
    """    GLM's way: can_pause, and PAUSE ends the read a couple of ticks later with its DONE saying "cancel" -
    what it read stays in the engine, the rest is read when the request's turn comes back.  Its reads are
    preempted on a multi engine by can_pause, and on a solo one only when it keeps a stopped read
    (keeps_stopped_read: the GLM line engine does, generate.cpp #573 - maya-metal drops one, so it does not)."""
    keeps_stopped_read = False
    can_pause = True
    multi = 1
    reports_reused = False          # the line-protocol driver has no per-call reused figure over the pipe

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


class LostRead(ReadEngine):
    """The mortar bug's engine (P124b): it lets a read be cut - the server believes the conversation stays live at
    the token reached (#1620) - but the read stopped INSIDE a batched part and its state was dropped: the request
    coming back reads the base again from 0, and its own RESUME figure is 0.  The continue line must carry that 0 -
    never the reuse the old wording claimed."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        with self.rlock:
            for key, pos in list(self.read_pos.items()):    # (a call that continues a base re-reads it from 0)
                if pos:
                    self.read_pos[key] = 0
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class UnreportingRead(ReadEngine):
    """Keeps a cut read, but reports no reused figure over the pipe (no RESUME line, as the line-protocol drivers
    have none): the continue line keeps its old numberless wording - it never invents a figure the engine did not
    tell it."""
    reports_reused = False


class Ds4Read(UnreportingRead):
    """saguaro-ds4 as the line driver sets it up (P124u): READY ... stop keepread [multi=n] (P124k) - no PAUSE; a
    STOP'd prompt read stays live at the deepest point it stored (ds4_main.cpp's stopped branch), so the READY word
    gives the driver keeps_stopped_read and a LOW read steps aside for a waiting NORMAL one."""
    can_pause = False


class Hy4Read(ReadEngine):
    """saguaro-hy4 as the line driver sets it up: SOLO (its READY says stop only - no PAUSE, never a multi=), with
    keeps_stopped_read - serve/server.py gives it to the engine when its INFO line says engine_kind=hy4, for its
    read keeping every chunk in the conversation (hy4_main.cpp:1086; the cancel path :1106-1110 clears nothing)."""
    keeps_stopped_read = True
    can_pause = False
    multi = 0
    info = {"engine_kind": "hy4"}


class Ik2Read(ReadEngine):
    """saguaro-ik2 solo as the line driver sets it up (P125l: READY ... stop, no PAUSE, INFO engine_kind=ik2):
    a STOP is seen only BETWEEN the prompt's chunk passes (ik2_main.cu:10821-10846 - a chunk pass is never
    split), and the read is KEPT exactly there (the fixed engine: its conversation stays at the boundary with a
    snapshot taken at the stop, :10888-10900; the shape before kept only its last snapshot - a first prompt has
    none before the last message's start, which for a card-shaped prompt is 0 - and reset the read away, the
    lost_read.sh finding). The driver knows the figure the resume starts from: the engine's PP line carries a
    reuse field, and the fixed driver keeps it as `progress_reuse` (serve/server.py) - None for a line engine
    whose PP says no number, a real int here, 0 included."""
    keeps_stopped_read = True
    can_pause = False
    multi = 0
    reports_reused = False
    info = {"engine_kind": "ik2"}
    keeps_cut = True                          # False: the pre-fix engine, rolled back to its last snapshot

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.snaps, self.resumed_from, self.chunk_reads = {}, [], {}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        with self.lock:
            self.calls.append((list(ids), max_new, dict(sampling or {})))
            base = next((b for b in self.bases if ids[:len(b)] == b), None)
            if base is None:
                base = list(ids)
                self.bases.append(base)
        key, total = id(base), len(base)
        step = max(1, -(-total // self.CHUNKS))
        pos = start = self.read_pos.get(key, 0)
        if 0 < start < total:
            self.resumed_from.append(start)   # (this call continues a cut read - from where the engine kept it)
        self._tls.own_pp = None
        self.progress_reuse = start           # (the call's PP lines will report it, fixed driver's parse)
        try:
            while pos < total:
                pos = min(total, pos + step)
                if cancel.is_set():
                    return
                self._tls.own_pp = ((pos, total), None)
                self.progress = (pos, total, start, 0.0)
                with self.rlock:
                    self.read_pos[key] = pos
                    self.ticks[key] = self.ticks.get(key, 0) + 1
                    reads = self.chunk_reads.setdefault(key, [0] * self.CHUNKS)
                    reads[min((pos - 1) // step, self.CHUNKS - 1)] += 1
                time.sleep(self.delay)
                yield None                      # a PP line: a keep-alive, a read tick, and the reuse figure
        except GeneratorExit:                   # the STOP, seen between the chunk passes
            if pos < total:
                if not self.keeps_cut:          # the pre-fix engine: back to the latest snapshot it kept - a
                    self.read_pos[key] = self.snaps.get(key, 0)   #  card-shaped first prompt has none: 0
                self.last = {"finish": "cancel", "generated": 0, "prompt_ms": 0.0, "decode_ms": 0.0}
            raise
        self._tls.own_pp = None
        text = self.tok.decode(base)
        answer = next(v for k, v in self.answers.items() if k in text)
        script = self.tok.encode(answer) + self.end
        for t in script[len(ids) - len(base):][:max_new]:
            if cancel.is_set():
                return
            time.sleep(self.delay)
            yield t


class Ik2LostRead(Ik2Read):
    """The card test's engine, as measured by the card host's lost_read run (the run behind FIX-REPORT.md's
    fix 1): a cut read rolls back to the latest snapshot the conversation kept, and a FIRST prompt has none
    before its last user message's start - which for a card-shaped prompt sits below the engine's arming bar
    (ik2_main.cu:10807) - so the stopped read resets to 0 and the resume reads the prompt again from 0. Its PP
    line still reports its reuse: 0. (The keep that fixes the engine is Ik2Read's shape.)"""
    keeps_cut = False


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
        _lock, order = self.waiting_order([PRIORITY_NORMAL, PRIORITY_LOW, PRIORITY_HIGH, PRIORITY_NORMAL])
        self.assertEqual(order, [2, 0, 3, 1])           # high first, the normals by arrival, the low one last

    def test_a_waiting_low_request_does_not_hold_back_a_new_normal_request(self):
        lock = FairLock()
        lock.acquire()
        bg = lock.enqueue(priority=PRIORITY_LOW)
        lock.release()
        self.assertTrue(lock.acquire(timeout=1))        # the fresh normal steps over the queued low one
        self.assertEqual(lock.waiting(), 1)             # the low one keeps its place, behind it now
        lock.release()
        self.assertTrue(lock.wait_turn(bg, timeout=2))
        lock.release()

    def test_the_aging_setting_is_off_by_default_and_on_above_zero(self):
        lock = FairLock()
        bg = lock.enqueue(units=1, priority=PRIORITY_LOW)
        time.sleep(0.05)                               # no promotion by waiting at all: the default
        self.assertEqual(lock._eff(bg), PRIORITY_LOW)
        self.assertEqual(lock.waiting(above=PRIORITY_LOW), 0)      # it never cuts a running low answer
        os.environ["SAGUARO_PRIORITY_AGE_S"] = "0.01"                     # set: that long of a wait counts normal
        self.assertEqual(lock._eff(bg), PRIORITY_NORMAL)
        self.assertEqual(lock.waiting(above=PRIORITY_LOW), 1)
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
        lock.enqueue(priority=PRIORITY_LOW)
        lock.enqueue()
        lock.enqueue(priority=PRIORITY_HIGH)
        self.assertEqual(lock.waiting(), 3)
        self.assertEqual(lock.waiting(above=PRIORITY_NORMAL), 1)         # only the high one cuts a normal runner
        self.assertEqual(lock.waiting(above=PRIORITY_LOW), 2)     # normal and high both cut a low answer
        self.assertEqual(lock.highest_waiting(), PRIORITY_HIGH)
    def test_a_queued_low_ages_into_normal(self):
        os.environ["SAGUARO_PRIORITY_AGE_S"] = "0.05"
        lock = FairLock()
        lock.acquire()
        bg = lock.enqueue(priority=PRIORITY_LOW)
        time.sleep(0.08)
        nm = lock.enqueue()                             # a fresh normal, behind the aged low waiter
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
        self.assertFalse(lock.claim_yield(0.0, above=PRIORITY_LOW))   # nobody above it at all yet
        lock.enqueue(units=1, priority=PRIORITY_LOW)                  # an equal is never a reason
        self.assertFalse(lock.claim_yield(0.0, above=PRIORITY_LOW))
        self.assertEqual(lock.waiting(above=PRIORITY_LOW), 0)
        lock.enqueue(units=1)                          # one normal waiter: cuts a low runner, one of them
        self.assertFalse(lock.claim_yield(0.0, above=PRIORITY_NORMAL))       # not a normal one (equal)
        self.assertTrue(lock.claim_yield(0.0, above=PRIORITY_LOW))    # one gives its place up
        self.assertFalse(lock.claim_yield(0.0, above=PRIORITY_LOW))   # the other stays: it is covered
        lock.unclaim_yield()

    def test_a_free_place_makes_the_low_one_step_aside_anyway(self):
        lock = FairLock(2)                              # Ken's "who steps aside": a FREE place is no shield
        lock.acquire(units=1, priority=PRIORITY_LOW)    # a low answer has one of the two places
        hi = lock.enqueue(units=1, priority=PRIORITY_HIGH)     # the waiter would take the free one by itself
        self.assertFalse(lock.step_aside(PRIORITY_NORMAL))     # normal: only a waiter that does not fit cuts one
        self.assertTrue(lock.step_aside(PRIORITY_LOW))         # low: gives its place up even here: only idle
        lock.unclaim_yield()                            # (the claim step_aside's True came with)
        lock.release(priority=PRIORITY_LOW)
        lock.leave(hi)


class IdleRule(Env):
    """low = "only while idle" (Ken's 11-25): a low request starts, or resumes after a pause, only when nothing
    normal or high runs or waits AND none finished in the last SAGUARO_PRIORITY_IDLE_S (0 = no grace).  The
    releases pass the level, the way _turn does: a finished answer is what starts the grace."""

    def test_a_low_request_waits_for_the_running_normal_and_the_grace_after_it(self):
        os.environ["SAGUARO_PRIORITY_IDLE_S"] = "0.3"
        lock = FairLock()
        lock.acquire()                                    # a normal request has the engine
        low = lock.enqueue(priority=PRIORITY_LOW)
        self.assertFalse(lock.wait_turn(low, timeout=0.05))      # it runs: a low one starts only while idle
        lock.release(priority=PRIORITY_NORMAL)            # it finished: the grace starts here
        self.assertFalse(lock.wait_turn(low, timeout=0.05))      # the last answer is only that old: still waits
        self.assertTrue(lock.wait_turn(low, timeout=2))          # the grace is over - time woke it - it starts
        lock.release(priority=PRIORITY_LOW)

    def test_a_normal_arriving_inside_the_grace_keeps_the_low_waiting(self):
        os.environ["SAGUARO_PRIORITY_IDLE_S"] = "0.4"
        lock = FairLock()
        lock.acquire()
        lock.release(priority=PRIORITY_NORMAL)            # an answer ended: the grace runs until +0.4 s
        low = lock.enqueue(priority=PRIORITY_LOW)
        self.assertFalse(lock.wait_turn(low, timeout=0.1))       # mid-grace: the low one waits
        lock.acquire()                                      # a person asks again in the meantime (the fast path)
        lock.release(priority=PRIORITY_NORMAL)              # and finishes: the grace starts over
        self.assertFalse(lock.wait_turn(low, timeout=0.2))       # of the new one only 0.1 s elapsed: still waiting
        lock.leave(low)

    def test_grace_zero_the_low_one_starts_the_moment_it_is_next(self):
        os.environ["SAGUARO_PRIORITY_IDLE_S"] = "0"
        lock = FairLock()
        lock.acquire()
        lock.release(priority=PRIORITY_NORMAL)            # an answer ended a heartbeat ago: grace 0 = no grace
        self.assertTrue(lock.acquire(timeout=0.5, priority=PRIORITY_LOW))   # nothing runs or waits: straight in
        lock.release(priority=PRIORITY_LOW)


class ServicePreempts(Env):
    def setUp(self):
        super().setUp()
        self.tok = ByteTokenizer()
        self.engine = ResumingEngine(self.tok, {"long": LONG, "short": SHORT})
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def race(self, low_prio=PRIORITY_LOW, high_prio=PRIORITY_HIGH):
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

    def test_a_high_request_cuts_a_low_answer_at_once(self):
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
        self.assertIn("[saguaro] priority: a low answer paused at", text)
        self.assertIn("for a high-priority request", text)
        self.assertIn("[saguaro] priority: the low answer continues", text)
        self.assertNotIn("fair slice", text)            # the priority, not the SAGUARO_FAIR_SLICE_S timer
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual(self.svc.fifo.waiting(), 0)
        self.assertEqual(self.svc.fifo._yields, 0)
        self.assertEqual(self.svc.status["queued"], 0)

    def test_a_low_ik2_answer_steps_aside_and_resumes_from_its_token(self):
        # P125l: saguaro-ik2's answer loop honours the STOP between tokens (solo ik2_main.cu:10968, the
        # multi one's writer check :10531) and keeps the conversation through the prompt it read - the
        # token-tick step-aside was never gated on an engine flag, and this is what the card test mirrors
        self.engine = ResumingEngine(self.tok, {"long": LONG, "short": SHORT})
        self.engine.info = {"engine_kind": "ik2"}
        self.svc.engine = self.engine
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            out = self.race()
        self.assertEqual(out["short one"][0], SHORT)
        self.assertEqual(out["long one"][0], LONG)                 # resumed intact at the token it reached
        self.assertLess(out["short one"][1], out["long one"][1])   # the high one answered first
        calls = self.long_calls()
        self.assertGreaterEqual(len(calls), 2)                     # cut at a token, and continued
        written = len(calls[1][0]) - len(calls[0][0])
        self.assertGreater(written, 0)                             # prompt + the tokens it had written
        self.assertIn("[saguaro] priority: a low answer paused at", log.getvalue())
        self.assertIn("[saguaro] priority: the low answer continues", log.getvalue())

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
        for prio in (PRIORITY_NORMAL, PRIORITY_LOW):
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

    def test_a_queued_low_ages_and_the_answer_steps_aside(self):
        os.environ["SAGUARO_PRIORITY_AGE_S"] = "0.02"  # a waiting low request counts as normal once it aged
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            out = self.race(low_prio=PRIORITY_LOW, high_prio=PRIORITY_LOW)
        self.assertLess(out["short one"][1], out["long one"][1])   # the aged waiter cuts the running low answer
        self.assertEqual(out["long one"][0], LONG)
        self.assertIn("priority: a low answer paused", log.getvalue())

    def test_turbo_one_at_a_time_respects_the_priorities(self):
        # (the review's nit 7): a Turbo engine keeps one answer at a time whatever the places (set_one):
        # a free second place is no shield in either direction - a running low gives its place up for a
        # waiting high, and a fresh normal does not take it beside a running low either, it steps the low
        # aside and runs first (one_at_a_time is the FIFO's _fits, the priorities ride on top of it)
        self.svc.fifo.set_capacity(2)
        self.svc.fifo.set_one(True)
        for above in (PRIORITY_HIGH, PRIORITY_NORMAL):
            with self.subTest(above=above):
                self.engine.calls.clear()
                self.engine.bases.clear()
                log = io.StringIO()
                with contextlib.redirect_stdout(log):
                    out = self.race(low_prio=PRIORITY_LOW, high_prio=above)
                self.assertEqual(out["long one"][0], LONG)            # resumed intact either way
                self.assertEqual(out["short one"][0], SHORT)
                self.assertLess(out["short one"][1], out["long one"][1])
                self.assertGreaterEqual(len(self.long_calls()), 2)    # cut: a free place is no shield
                self.assertIn("[saguaro] priority: a low answer paused", log.getvalue())
                self.assertEqual(self.svc.fifo._yields, 0)
                self.assertFalse(self.svc.fifo.locked())
                self.assertEqual(self.svc.fifo.waiting(), 0)

    def test_a_cancel_while_paused_leaves_the_line(self):
        out, log = {}, io.StringIO()
        cancel = threading.Event()
        with contextlib.redirect_stdout(log):
            a = threading.Thread(target=ask, args=(self.svc, "long one", 4096, out, PRIORITY_LOW),
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
            a = threading.Thread(target=ask, args=(self.svc, "long one", 4096, out, PRIORITY_LOW))
            a.start()
            while self.svc.latest_request not in self.svc.current_requests:
                time.sleep(0.001)
            b = threading.Thread(target=ask, args=(self.svc, "other one", 4096, out, PRIORITY_HIGH))
            b.start()
            while not any(getattr(s, "paused_for_priority", False) for s in list(self.svc.running)):
                time.sleep(0.005)
            self.assertEqual([s.priority for s in self.svc.running if getattr(s, "paused_for_priority", False)],
                             [PRIORITY_NAMES[PRIORITY_LOW]])   # it waits: still a low request
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
        self.assertIn("priority: a low answer has been paused for", log.getvalue())   # it says so, once
        self.assertEqual(log.getvalue().count("has been paused for"), 1)
        self.assertEqual(out["long one"][0], LONG)                    # and it went on to its end: waiting ends
        self.assertEqual(out["other one"][0], other)                  #   nothing, and promotes nothing

    def test_parallel_two_two_lows_and_a_high_both_lows_pause(self):
        # Ken's 11-25 block, scenario 4: two lows and a high on parallel 2 -> BOTH lows pause while the
        # high is around: low is "only while idle", so free places and claim counting cut none of them
        # down to "one per waiter" - every low answer gives its place up.
        longs = {f"long {c}": "".join(random.Random(700 + i).choices(string.ascii_letters, k=400))
                 for i, c in enumerate("AB")}
        self.engine = ResumingEngine(self.tok, {**longs, "short": SHORT}, delay_s=0.003)
        self.svc.engine = self.engine
        self.svc.fifo.set_capacity(2)                   # two conversations at once, both places taken
        out, threads, log = {}, [], io.StringIO()
        for user in longs:
            th = threading.Thread(target=ask, args=(self.svc, user, 4096, out, PRIORITY_LOW))
            th.start()
            threads.append(th)
        while len(self.svc.current_requests) < 2:
            time.sleep(0.001)
        time.sleep(0.05)                                # both answers are under way
        with contextlib.redirect_stdout(log):
            hi = threading.Thread(target=ask, args=(self.svc, "short one", 4096, out, PRIORITY_HIGH))
            hi.start()
            for th in threads + [hi]:
                th.join(60)
        self.assertEqual(out["short one"][0], SHORT)
        for user, text in longs.items():
            self.assertEqual(out[user][0], text)        # both continue intact
        resumed = [b for b in self.engine.bases if sum(1 for c in self.engine.calls
                     if c[0][:len(b)] == b and len(c[0]) > len(b)) > 0]
        self.assertEqual(len(resumed), 2)               # BOTH lows were cut, and both went on from the
        text = log.getvalue()                           #  token they reached
        self.assertEqual(text.count("priority: a low answer paused"), 2)
        self.assertEqual(text.count("for a high-priority request"), 2)
        self.assertEqual(text.count("priority: the low answer continues"), 2)
        self.assertLess(out["short one"][1], out["long A"][1])
        self.assertLess(out["short one"][1], out["long B"][1])
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual(self.svc.fifo._yields, 0)

    def test_a_normal_and_a_high_run_side_by_side_untouched(self):
        # Ken's 11-25 block, scenario 1: parallel 2, one normal + one high running together - high never
        # steps aside, and a normal one pauses only for a WAITING high, so nobody gives a place up here
        self.svc.fifo.set_capacity(2)
        out, log = {}, io.StringIO()
        with contextlib.redirect_stdout(log):
            a = threading.Thread(target=ask, args=(self.svc, "long one", 4096, out, PRIORITY_NORMAL))
            a.start()
            while self.svc.latest_request not in self.svc.current_requests:
                time.sleep(0.001)
            time.sleep(0.05)                            # the normal answer is under way
            ask(self.svc, "short one", 4096, out, PRIORITY_HIGH)    # a place is free: it just takes it
            a.join(30)
        self.assertEqual(out["long one"][0], LONG)
        self.assertEqual(out["short one"][0], SHORT)
        self.assertEqual(len(self.long_calls()), 1)     # the normal answer was never cut
        self.assertNotIn("priority:", log.getvalue())   # and nothing says it should have been
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual(self.svc.fifo.waiting(), 0)

    def test_a_low_answer_steps_aside_for_a_normal_taking_the_free_place(self):
        # Ken's 11-25 block, scenario 2: parallel 2, a low running + a normal arriving while a place is
        # free -> the low pauses: beside a person the card is never the low one's to keep
        self.svc.fifo.set_capacity(2)
        out, log = {}, io.StringIO()
        with contextlib.redirect_stdout(log):
            a = threading.Thread(target=ask, args=(self.svc, "long one", 4096, out, PRIORITY_LOW))
            a.start()
            while self.svc.latest_request not in self.svc.current_requests:
                time.sleep(0.001)
            time.sleep(0.05)                            # the low answer is past its first tokens
            ask(self.svc, "short one", 4096, out, PRIORITY_NORMAL)  # the free place does not shield it
            a.join(30)
        self.assertEqual(out["long one"][0], LONG)      # cut at a token, continued: nothing lost at the seam
        self.assertEqual(out["short one"][0], SHORT)
        self.assertLess(out["short one"][1], out["long one"][1])
        self.assertGreaterEqual(len(self.long_calls()), 2)
        self.assertIn("[saguaro] priority: a low answer paused", log.getvalue())
        self.assertIn("for a normal-priority request", log.getvalue())     # it names the RUNNING normal
        self.assertIn("[saguaro] priority: the low answer continues", log.getvalue())
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual(self.svc.fifo.waiting(), 0)
        self.assertEqual(self.svc.fifo._yields, 0)

    def calls_for(self, key: str):
        """The engine calls made for the conversation whose prompt contains `key` (all of them: its first and
        any resume - more than one means the answer was cut and continued)."""
        base = next(b for b in self.engine.bases if key in self.engine.tok.decode(b))
        return [c for c in self.engine.calls if c[0][:len(base)] == base]

    def test_the_normal_that_started_last_steps_aside_for_a_high_one(self):
        # (finding 4, PLAN line 40): parallel 2, TWO normals running and a high waiting that cannot fit: the
        # normal that STARTED LAST is the one that gives its place up - not whichever runner first ticks on it.
        longs = {f"long {c}": "".join(random.Random(800 + i).choices(string.ascii_letters, k=400))
                 for i, c in enumerate("AB")}
        self.engine = ResumingEngine(self.tok, {**longs, "short": SHORT}, delay_s=0.003)
        self.svc.engine = self.engine
        self.svc.fifo.set_capacity(2)                   # both places taken: the high one cannot fit
        out, log = {}, io.StringIO()
        started = lambda key: any(key in self.engine.tok.decode(b) for b in self.engine.bases)
        a = threading.Thread(target=ask, args=(self.svc, "long A one", 4096, out, PRIORITY_NORMAL))
        a.start()
        while not started("long A"):
            time.sleep(0.001)                           # A has its turn: the FIFO's first normal grant
        b = threading.Thread(target=ask, args=(self.svc, "long B one", 4096, out, PRIORITY_NORMAL))
        b.start()
        while not started("long B"):
            time.sleep(0.001)                           # B started LAST: B is the one meant to step aside
        time.sleep(0.05)                                # both answers are past their first tokens
        with contextlib.redirect_stdout(log):
            hi = threading.Thread(target=ask, args=(self.svc, "short one", 4096, out, PRIORITY_HIGH))
            hi.start()
            for th in (a, b, hi):
                th.join(60)
        self.assertEqual(out["short one"][0], SHORT)
        for user, text in longs.items():
            self.assertEqual(out[user + " one"][0], text)      # both answers end whole
        self.assertEqual(len(self.calls_for("long A")), 1)     # the earlier starter ran on, untouched
        self.assertGreaterEqual(len(self.calls_for("long B")), 2)   # the last-started one was cut ...
        text = log.getvalue()                                   #  ... and continued from its token
        self.assertEqual(text.count("priority: a normal answer paused"), 1)
        self.assertIn("for a high-priority request", text)
        self.assertEqual(text.count("priority: the normal answer continues"), 1)
        self.assertLess(out["short one"][1], out["long B one"][1])
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual(self.svc.fifo.waiting(), 0)
        self.assertEqual(self.svc.fifo._yields, 0)

    def test_a_high_request_cuts_a_running_normal(self):
        # Ken's 11-25 block, scenario 3: parallel 1, a normal running + a high arriving -> the normal
        # pauses (the high waits, all places are taken, and claim_yield's counting cuts this one runner)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            out = self.race(low_prio=PRIORITY_NORMAL, high_prio=PRIORITY_HIGH)
        self.assertEqual(out["long one"][0], LONG)      # resumed intact from the token it reached
        self.assertEqual(out["short one"][0], SHORT)
        self.assertLess(out["short one"][1], out["long one"][1])
        self.assertGreaterEqual(len(self.long_calls()), 2)
        self.assertIn("[saguaro] priority: a normal answer paused at", log.getvalue())
        self.assertIn("for a high-priority request", log.getvalue())
        self.assertIn("[saguaro] priority: the normal answer continues", log.getvalue())
        self.assertNotIn("fair slice", log.getvalue())  # the priority, not the SAGUARO_FAIR_SLICE_S timer
        self.assertEqual(self.svc.fifo._yields, 0)

    def test_a_low_request_waits_the_idle_grace_after_an_answer(self):
        os.environ["SAGUARO_PRIORITY_IDLE_S"] = "0.4"     # the rule through run() and _turn, not just the lock
        out = {}
        ask(self.svc, "short one", 4096, out)             # a normal answer ends: the grace starts
        t0 = time.monotonic()
        ask(self.svc, "short one", 4096, out, PRIORITY_LOW)      # the engine is free - the grace is not over yet
        self.assertGreaterEqual(time.monotonic() - t0, 0.3)      # it waited it out (with keep-alives), then ran
        self.assertEqual(out["short one"][0], SHORT)


class ReadsPreempt(Env):
    def setUp(self):
        super().setUp()
        self.tok = ByteTokenizer()

    def race_read(self, engine, second=PRIORITY_HIGH):
        svc = Service(engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        out, log = {}, io.StringIO()
        with contextlib.redirect_stdout(log):
            a = threading.Thread(target=ask, args=(svc, "long one", 4096, out, PRIORITY_LOW))
            a.start()
            while sum(engine.ticks.values()) < 2:
                time.sleep(0.001)                       # it is in the middle of reading its prompt
            b = threading.Thread(target=ask, args=(svc, "short one", 4096, out, second))
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
        self.assertIn("priority: a low prompt read paused", log)
        self.assertIn("prompt tokens for a high-priority request", log)
        self.assertIn("priority: the low prompt read continued", log)     # ... with the engine's reused figure

    def test_a_hy4_solo_read_is_cut_for_a_normal_request_and_resumed(self):
        # P124i: a solo engine reporting keeps_stopped_read (the hy4 driver's: INFO engine_kind=hy4) - a low
        # read steps aside for a waiting NORMAL one (not only a high), and continues at the chunk it kept
        engine = Hy4Read(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        out, log = self.race_read(engine, second=PRIORITY_NORMAL)
        self.assertEqual(out["short one"][0], SHORT)
        self.assertLess(out["short one"][1], out["long one"][1])   # the normal one went first, the low after
        self.assertEqual(out["long one"][0], LONG)
        self.assertEqual(engine.ticks[id(engine.bases[0])], ReadEngine.CHUNKS)   # read exactly once through:
        #                                                    # what the engine kept was not read again
        long_calls = [c for c in engine.calls if c[0][:len(engine.bases[0])] == engine.bases[0]]
        self.assertEqual(len(long_calls), 2)
        self.assertEqual(long_calls[1][0], long_calls[0][0])       # the same prompt: nothing written yet
        self.assertIn("priority: a low prompt read paused", log)
        self.assertIn("prompt tokens for a normal-priority request", log)
        self.assertIn("priority: the low prompt read continue", log)   # (continued: N reused, or continues - P124b)

    def test_an_ik2_solo_read_steps_aside_for_a_normal_and_continues_at_the_cut(self):
        # P125l (fixed, the lost_read.sh finding): a LOW read stops at the next chunk boundary (never inside a
        # pass) and CONTINUES THERE - the conversation stays at the boundary it stopped at, and the resume's PP
        # reuse figure says so; nothing read before the pause is read again
        engine = Ik2Read(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        out, log = self.race_read(engine, second=PRIORITY_NORMAL)
        self.assertEqual(out["short one"][0], SHORT)
        self.assertLess(out["short one"][1], out["long one"][1])   # the normal one went first, the low after
        self.assertEqual(out["long one"][0], LONG)                 # nothing lost or repeated at the seam
        long_calls = [c for c in engine.calls if c[0][:len(engine.bases[0])] == engine.bases[0]]
        self.assertEqual(len(long_calls), 2)                       # cut once, and continued
        self.assertEqual(long_calls[1][0], long_calls[0][0])       # the same prompt: nothing written yet
        self.assertEqual(len(engine.resumed_from), 1)              # the resume started where the read stopped ...
        total = len(engine.bases[0])
        step = max(1, -(-total // ReadEngine.CHUNKS))
        self.assertTrue(0 < engine.resumed_from[0] < total)        # ... not from 0 (the card run's shape) ...
        self.assertEqual(engine.resumed_from[0] % step, 0)         # ... at a chunk boundary, where it stopped
        reads = engine.chunk_reads[id(engine.bases[0])]
        self.assertEqual(set(reads), {1})                          # every chunk read EXACTLY once: the pause
        #                                                    # threw no read away (the card lost 32,768 tokens)
        self.assertIn("priority: a low prompt read paused", log)
        self.assertIn("prompt tokens for a normal-priority request", log)
        self.assertIn(f"prompt read continued: {engine.resumed_from[0]:,} of {total:,} tokens reused", log)
        self.assertNotIn("prompt read continues from where the engine kept it", log)   # a figure: the claim is not

    def test_a_high_request_goes_first_and_the_ik2_read_continues(self):
        # the high one answers while the low read waits, and only then does the read go on from its cut
        engine = Ik2Read(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        out, log = self.race_read(engine)
        self.assertEqual(out["short one"][0], SHORT)
        self.assertLess(out["short one"][1], out["long one"][1])   # high first: its answer ended long before
        self.assertEqual(out["long one"][0], LONG)
        self.assertEqual(len(engine.resumed_from), 1)
        self.assertGreater(engine.resumed_from[0], 0)              # continued from the cut, not from 0
        self.assertIn(f"prompt read continued: {engine.resumed_from[0]:,} of", log)   # the engine's own figure
        self.assertIn("priority: a low prompt read paused", log)
        self.assertIn("prompt tokens for a high-priority request", log)

    def test_an_ik2_read_cut_keeps_everything_it_read(self):
        # the fixed engine, tested straight (no race): the STOP between the chunk passes keeps the read at the
        # boundary its passes ran to (ik2_main.cu:10888-10900) - never a rollback to 0, never a snapshot short
        # of it: THE test the card run needed - it fails when the resumed read starts at 0 (or anywhere short
        # of the cut), which is exactly what lost 32,768 tokens per cut on the card
        engine = Ik2Read(self.tok, {"long": LONG}, delay_s=0.0)
        ids = self.tok.encode("long one " + LONG)
        total, step = len(ids), max(1, -(-len(ids) // ReadEngine.CHUNKS))
        gen = engine.generate(ids, 4096, {}, threading.Event())
        for _ in range(3):
            self.assertIsNone(next(gen))                   # chunks 1..3 read
        gen.close()                                        # the STOP, seen between the passes
        key = next(iter(engine.read_pos))
        self.assertEqual(engine.read_pos[key], 3 * step)   # kept EXACTLY there: no rollback, no half chunk
        self.assertEqual(sum(engine.chunk_reads[key]), 3)  # nothing read beyond the three passes that ran
        gen2 = engine.generate(ids, 4096, {}, threading.Event())
        self.assertIsNone(next(gen2))                      # (the call starts: at the cut, not at 0)
        self.assertEqual(engine.progress_reuse, 3 * step)  # its PP lines report where the read starts: the cut
        self.assertEqual(engine.resumed_from, [3 * step])
        for _ in gen2:
            pass
        self.assertEqual(engine.read_pos[key], total)
        reads = engine.chunk_reads[key]
        self.assertEqual(set(reads), {1})                  # every chunk, once: the pause threw nothing away

    def test_an_ik2_lost_read_reports_zero_never_a_claim(self):
        # FIX-REPORT fix 3: the server logs "continues" only when the engine reports kept tokens. The card's
        # engine shape (Ik2LostRead): the resume re-reads from 0 and its PP reuse figure is 0 - the line must
        # carry that 0 and never claim the read continues (the numberless wording is for engines that report no
        # figure at all; a reported 0 is a lost read, said as one - the P124b rule)
        engine = Ik2LostRead(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        out, log = self.race_read(engine, second=PRIORITY_NORMAL)
        self.assertEqual(out["long one"][0], LONG)               # the re-read answers in the end anyway
        self.assertEqual(engine.resumed_from, [])                # it came back to 0: no resume, only a restart
        self.assertIn("prompt read continued: 0 of", log)
        self.assertNotIn("prompt read continues from where the engine kept it", log)

    def test_an_ik2_request_that_never_waits_is_unchanged(self):
        # alone on an idle server the read answers exactly as before: one engine call, every chunk read
        # exactly once, no pause line of any kind
        engine = Ik2Read(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        svc = Service(engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        out, log = {}, io.StringIO()
        with contextlib.redirect_stdout(log):
            ask(svc, "long one", 4096, out)                        # normal priority: it never waits
        self.assertEqual(out["long one"][0], LONG)
        self.assertEqual(len(engine.calls), 1)                     # never cut
        self.assertEqual(engine.resumed_from, [])
        self.assertEqual(engine.ticks[id(engine.bases[0])], ReadEngine.CHUNKS)
        self.assertEqual(set(engine.chunk_reads[id(engine.bases[0])]), {1})   # every chunk, exactly once
        self.assertNotIn("priority:", log.getvalue())

    def test_a_glm_read_pauses_by_its_pause(self):
        engine = PausingRead(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        out, log = self.race_read(engine)
        self.assertEqual(out["short one"][0], SHORT)
        self.assertLess(out["short one"][1], out["long one"][1])
        self.assertEqual(out["long one"][0], LONG)
        self.assertGreaterEqual(engine.pauses, 1)       # PAUSE, not a STOP: the state with what it read stays
        self.assertEqual(engine.ticks[id(engine.bases[0])], ReadEngine.CHUNKS)

    def test_a_solo_glm_read_pauses_by_its_pause_and_keeps_it(self):
        engine = PausingRead(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        engine.multi = 0                                # solo GLM (READY ... stop pause, no multi=): the line
        engine.keeps_stopped_read = True                #  engine keeps a cancelled read (#573), so it steps aside
        out, log = self.race_read(engine, PRIORITY_NORMAL)   # a low read steps aside for a waiting normal too
        self.assertEqual(out["short one"][0], SHORT)
        self.assertLess(out["short one"][1], out["long one"][1])   # the normal one answered first
        self.assertEqual(out["long one"][0], LONG)
        self.assertGreaterEqual(engine.pauses, 1)       # PAUSE, not a cut
        self.assertIn("priority: a low prompt read paused", log)
        self.assertIn("prompt read continues from where the engine kept it", log)
        long_calls = [c for c in engine.calls if c[0][:len(engine.bases[0])] == engine.bases[0]]
        self.assertEqual(len(long_calls), 2)
        self.assertEqual(long_calls[1][0], long_calls[0][0])   # the same prompt: the kept read is not re-read
        self.assertEqual(engine.ticks[id(engine.bases[0])], ReadEngine.CHUNKS)

    def test_an_mm3_solo_read_is_cut_for_a_normal_and_continues_kept(self):
        # saguaro-mm3 solo (READY ... stop, no pause): the driver sets keeps_stopped_read for engine_kind mm3
        # (P124j) - a cancelled read keeps what it fed (mm3_main.cpp's cancel paths) - so a STOP cuts it here.
        engine = UnreportingRead(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        engine.multi = 0
        out, log = self.race_read(engine, PRIORITY_NORMAL)     # a low read steps aside for a waiting normal too
        self.assertEqual(out["short one"][0], SHORT)
        self.assertLess(out["short one"][1], out["long one"][1])   # the normal one answered first
        self.assertEqual(out["long one"][0], LONG)
        self.assertEqual(engine.ticks[id(engine.bases[0])], ReadEngine.CHUNKS)   # the kept read was not read again
        long_calls = [c for c in engine.calls if c[0][:len(engine.bases[0])] == engine.bases[0]]
        self.assertEqual(len(long_calls), 2)
        self.assertEqual(long_calls[1][0], long_calls[0][0])   # the same prompt: nothing written yet
        self.assertIn("priority: a low prompt read paused", log)
        self.assertIn("prompt read continues from where the engine kept it", log)   # the line driver reports no figure

    def test_an_mm3_multi_read_is_cut_for_a_normal_and_continues_kept(self):
        engine = UnreportingRead(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        engine.multi = 2                            # saguaro-mm3 --parallel: the cut ends THIS request (STOP <rid>,
        out, log = self.race_read(engine, PRIORITY_NORMAL)    #  the test below), the others' reads and answers run on
        self.assertEqual(out["short one"][0], SHORT)
        self.assertLess(out["short one"][1], out["long one"][1])
        self.assertEqual(out["long one"][0], LONG)
        self.assertEqual(engine.ticks[id(engine.bases[0])], ReadEngine.CHUNKS)
        long_calls = [c for c in engine.calls if c[0][:len(engine.bases[0])] == engine.bases[0]]
        self.assertEqual(len(long_calls), 2)
        self.assertEqual(long_calls[1][0], long_calls[0][0])
        self.assertIn("priority: a low prompt read paused", log)
        self.assertIn("prompt read continues from where the engine kept it", log)

    def test_a_ds4_solo_read_is_cut_for_a_normal_and_continues_kept(self):
        # saguaro-ds4 solo (READY ... stop keepread): P124u gives the driver keeps_stopped_read from the word - a
        # STOP'd read stays live at the point it stored (P124k's keep), so a LOW read steps aside for a NORMAL one
        engine = Ds4Read(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        out, log = self.race_read(engine, PRIORITY_NORMAL)
        self.assertEqual(out["short one"][0], SHORT)
        self.assertLess(out["short one"][1], out["long one"][1])   # the normal one answered first
        self.assertEqual(out["long one"][0], LONG)
        self.assertEqual(engine.ticks[id(engine.bases[0])], ReadEngine.CHUNKS)   # the kept read was not read again
        long_calls = [c for c in engine.calls if c[0][:len(engine.bases[0])] == engine.bases[0]]
        self.assertEqual(len(long_calls), 2)
        self.assertEqual(long_calls[1][0], long_calls[0][0])       # the same prompt: nothing written yet
        self.assertIn("priority: a low prompt read paused", log)
        self.assertIn("prompt read continues from where the engine kept it", log)   # no figure over the pipe

    def test_a_ds4_multi_read_is_cut_for_a_normal_and_continues_kept(self):
        engine = Ds4Read(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        engine.multi = 2                            # saguaro-ds4 --parallel: READY ... stop multi=n keepread - the
        out, log = self.race_read(engine, PRIORITY_NORMAL)    #  cut ends THIS request (STOP <rid>: the test below)
        self.assertEqual(out["short one"][0], SHORT)
        self.assertLess(out["short one"][1], out["long one"][1])
        self.assertEqual(out["long one"][0], LONG)
        self.assertEqual(engine.ticks[id(engine.bases[0])], ReadEngine.CHUNKS)
        long_calls = [c for c in engine.calls if c[0][:len(engine.bases[0])] == engine.bases[0]]
        self.assertEqual(len(long_calls), 2)
        self.assertEqual(long_calls[1][0], long_calls[0][0])
        self.assertIn("priority: a low prompt read paused", log)
        self.assertIn("prompt read continues from where the engine kept it", log)

    def test_a_solo_pause_engine_lets_its_read_finish(self):
        engine = PausingRead(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        engine.multi = 0                            # solo AND an engine that drops a stopped read (no
        engine.keeps_stopped_read = False           #  keeps_stopped_read - maya-metal is one): the read runs on
        out, log = self.race_read(engine)
        self.assertEqual(out["long one"][0], LONG)
        self.assertEqual(out["short one"][0], SHORT)
        self.assertNotIn("prompt read paused", log)
        self.assertEqual(engine.ticks[id(engine.bases[0])], ReadEngine.CHUNKS)

    def test_an_engine_that_would_lose_a_stopped_read_finishes_it(self):
        engine = ReadEngine(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        engine.keeps_stopped_read = False               # no PAUSE either: a cut would lose the read - no cutting
        out, log = self.race_read(engine)
        self.assertEqual(out["long one"][0], LONG)
        self.assertEqual(out["short one"][0], SHORT)
        self.assertNotIn("prompt read paused", log)
        self.assertEqual(engine.ticks[id(engine.bases[0])], ReadEngine.CHUNKS)

    def test_a_batch_read_runs_to_its_end_even_for_a_high_request(self):
        engine = ReadEngine(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        engine.batch = 4                                # the batch engine: its slots DROP a stopped read
        out, log = self.race_read(engine)               # (BRA 0 cancel keeps none of it) - so it is not cut,
        self.assertEqual(out["long one"][0], LONG)      #  it runs to its end and the high one goes after
        self.assertEqual(out["short one"][0], SHORT)
        self.assertNotIn("prompt read paused", log)
        self.assertEqual(engine.ticks[id(engine.bases[0])], ReadEngine.CHUNKS)

    def test_the_qfn_engine_reports_where_the_property_comes_from(self):
        from serve.server import _qfn_engine_class     # the property: SOLO, a stopped request keeps its
        cls = _qfn_engine_class()                      # conversation live at the token it reached - generate.cpp
        self.assertTrue(cls.keeps_stopped_read)        # #1620 (the GEN path) - a batch slot drops a cancelled
        self.assertFalse(cls.can_pause)                # read, so only a solo read may be cut by a STOP


# a line engine that only says who it is (its INFO engine_kind=) and what its READY offers (its words after the
# ctx), answers nothing, and ends on QUIT: the driver's own start-up parsing, for which engines it reports
# keeps_stopped_read
INFO_KIND_ENGINE = r'''#!/usr/bin/env python3
import os, sys, threading, time
kind, ready = sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "stop")
def out(s):
    sys.stdout.write(s + "\n"); sys.stdout.flush()
def reader():
    for line in sys.stdin:
        if line.strip() == "QUIT":
            os._exit(0)
threading.Thread(target=reader, daemon=True).start()
out("INFO engine=%s engine_kind=%s model=Test ctx=4096" % (kind, kind))
out("READY 4096 " + ready)
while True:
    time.sleep(0.05)
'''


class TheReadKeepDriver(Env):
    """serve/server.py starts every line engine with StrataEngine; P124i gave the one whose INFO line says
    engine_kind=hy4 (its own line: hy4_main.cpp:932) the keeps_stopped_read the read preemption gates on -
    its stopped read keeps the conversation live at the chunk reached.  P124j did the same for mm3, and P124u let
    any engine claim it itself: READY carrying `keepread` (saguaro-ds4 says it with its keep on, P124k) - the word
    is the truth, not the engine's name.  P125l gave it to saguaro-ik2 SOLO (its STOP stops at a chunk boundary
    and the engine keeps the read at its last snapshot, ik2_main.cu:10815-10841/:10870-10892); its --parallel
    line slot_resets a stopped reader (:10494-10499), so the multi one is not given it.  An engine neither named
    nor saying the word is not given it."""

    def start(self, kind, ready="stop"):
        path = Path(tempfile.mkdtemp()) / "saguaro-x.py"
        path.write_text(INFO_KIND_ENGINE)
        path.chmod(0o755)
        try:
            return StrataEngine(str(path), [kind, ready])
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)

    def test_the_hy4_driver_reports_keeps_stopped_read(self):
        eng = self.start("hy4")
        try:
            self.assertEqual(eng.info.get("engine_kind"), "hy4")
            self.assertTrue(eng.keeps_stopped_read)    # solo: READY ... stop - no PAUSE, no multi
            self.assertFalse(eng.can_pause)
            self.assertEqual(eng.multi, 0)
        finally:
            eng.close()

    def test_the_mm3_driver_reports_keeps_stopped_read(self):
        eng = self.start("mm3")                        # P124j: its INFO engine_kind=mm3 line gives it the flag
        try:
            self.assertTrue(eng.keeps_stopped_read)
        finally:
            eng.close()

    def test_the_keepread_word_gives_it_to_any_engine(self):
        # P124u: the word is the truth, not the engine's name - READY ... stop keepread (any engine's, DS4's with
        # its keep on per P124k) is what the driver gates on, solo as multi
        eng = self.start("ds4", "stop keepread")
        try:
            self.assertTrue(eng.keeps_stopped_read)    # solo: READY ... stop keepread - no PAUSE, no multi=
            self.assertFalse(eng.can_pause)
            self.assertEqual(eng.multi, 0)
        finally:
            eng.close()
        eng = self.start("ds4", "stop multi=4 keepread")        # --parallel: the same word rides the multi READY
        try:
            self.assertEqual(eng.multi, 4)
            self.assertTrue(eng.keeps_stopped_read)    # the cut there ends ONE request: STOP <rid>, the class below
        finally:
            eng.close()
        eng = self.start("some-new-line-engine", "stop keepread")   # a name the driver never heard: the word holds
        try:
            self.assertTrue(eng.keeps_stopped_read)
        finally:
            eng.close()

    def test_an_engine_without_the_word_is_not_given_it(self):
        eng = self.start("ds4")                        # SAGUARO_DS4_KEEP_READ=0 (P124k): its READY says no keepread,
        try:                                           #   it drops a stopped read - the name alone must not decide
            self.assertFalse(getattr(eng, "keeps_stopped_read", False))
        finally:
            eng.close()

    def test_the_ik2_driver_is_given_it_solo_but_not_parallel(self):
        # P125l: solo saguaro-ik2 keeps a cut read at its last snapshot (the STOP lands between chunk
        # passes, ik2_main.cu:10815-10841) - its --parallel slot_resets a stopped reader (:10494-10499,
        # audit 8.5), so ONLY the solo line gets the flag and its multi reads still run to their end
        eng = self.start("ik2")                        # READY ... stop: solo
        try:
            self.assertEqual(eng.multi, 0)
            self.assertFalse(eng.can_pause)
            self.assertTrue(eng.keeps_stopped_read)
        finally:
            eng.close()
        eng = self.start("ik2", "stop multi=2")        # READY ... stop multi=2: --parallel
        try:
            self.assertEqual(eng.multi, 2)
            self.assertFalse(eng.keeps_stopped_read)
        finally:
            eng.close()

    def test_another_line_engine_is_not_given_it(self):
        eng = self.start("some-new-name")              # an engine neither named nor saying keepread: not given it
        try:
            self.assertFalse(getattr(eng, "keeps_stopped_read", False))
        finally:
            eng.close()


class TheMultiCutNamesOneRequest(unittest.TestCase):
    """P124j: saguaro-mm3 --parallel (READY ... stop multi=n) keeps a cut read too, so a low read is preempted
    there as well - and the cut must end ONE conversation: the line driver's close sends `STOP <rid>`, never a
    bare STOP, which on a multi engine ends every request the engine is running.  saguaro-ds4 --parallel (READY ...
    stop multi=n keepread, its cut gated in by P124u) drives this same close."""

    class Stdin:
        def __init__(self, eng):
            self.eng, self.sent = eng, []

        def write(self, s):
            self.sent.append(s.strip())
            if s.startswith("STOP "):                   # the engine sees its rid, ends that request, says DONE
                rid = int(s.split()[1])
                q = self.eng._routes.get(rid)
                if q is not None:
                    q.put(f"DONE {rid} 0 300 10.0 0.0 cancel 0 0 100 0 0")

        def flush(self):
            pass

    def test_closing_a_multi_request_stops_only_its_rid(self):
        import itertools
        from serve.server import StrataEngine
        eng = StrataEngine.__new__(StrataEngine)
        eng._rids, eng._routes, eng._routes_lock, eng._sent = itertools.count(1), {}, threading.Lock(), {}
        eng._tls, eng.multi, eng.can_stop = threading.local(), 2, True
        eng.progress = eng.prefill_tok_s_mean = None
        eng.proc = unittest.mock.Mock()
        eng.proc.stdin = self.Stdin(eng)
        got, err = [], []

        def drive():
            try:
                gen = eng._generate_multi([1, 2, 3], 50, {}, threading.Event())
                got.append(next(gen))                   # its read: the PP tick, where _sliced's cut would break
                gen.close()                             # STOP + drain to the DONE, still holding the FIFO
            except Exception as e:                      # noqa: BLE001 - reported through `err`
                err.append(e)

        th = threading.Thread(target=drive)
        th.start()
        while not eng._routes and not err:
            time.sleep(0.001)
        eng._routes[1].put("PP 1 100 300 10.0 10000.0 0")
        th.join(10)
        self.assertEqual(err, [])
        self.assertEqual(got, [None])
        self.assertEqual(eng.proc.stdin.sent[-1], "STOP 1")     # its rid alone: never an engine-wide STOP
class TheResumeLine(Env):
    """P124k round 3: the RESUME figure a line-protocol engine says before its read (GLM's prep_prompt, DS4's kept
    prompt read) must reach the attribute the continue line reads - not only the QFN driver set engine.reused, so
    a DS4 read kept at 32,768 logged its numberless line beside the figure its own engine had printed.  The real
    StrataEngine driver, against the fake engines that speak those lines."""

    def test_the_solo_driver_keeps_its_resume_figure(self):
        from serve.server import StrataEngine
        eng = StrataEngine(str(ROOT / "serve/fake_solo_parity_engine.py"), [])
        try:
            self.assertIsNone(eng.reused)                    # (nothing heard yet: the numberless wording stands)
            [t for t in eng.generate(list(range(30)), 3, {}, threading.Event()) if t is not None]
            self.assertEqual(eng.reused, 4)                  # (the fake's RESUME 4: that read's own base)
            [t for t in eng.generate(list(range(2)), 3, {}, threading.Event()) if t is not None]
            self.assertEqual(eng.reused, 1)                  # (per call: this read's figure, not the last call's)
        finally:
            eng.close()

    def test_the_multi_driver_keeps_its_own_resume_figure(self):
        from serve.server import StrataEngine
        eng = StrataEngine(str(ROOT / "serve/fake_multi_engine.py"), ["3"],
                           env=dict(os.environ, FAKE_PARITY_LINES="1"))
        try:
            self.assertIsNone(eng.reused)                    # (no RESUME heard yet for this thread's request)
            [t for t in eng.generate(list(range(20)), 5, {}, threading.Event()) if t is not None]
            self.assertEqual(eng.reused, 0)                  # (RESUME <rid> 0: routed to its request, parsed)
        finally:
            eng.close()


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
            for kind, val in svc.run(ids, True, None, max_new, {"_priority": PRIORITY_LOW},
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
                    a = threading.Thread(target=ask, args=(svc, "long one", 4096, out, PRIORITY_LOW))
                    a.start()
                    while svc.latest_request not in svc.current_requests:
                        time.sleep(0.001)
                    time.sleep(0.05)
                    b = threading.Thread(target=ask, args=(svc, "short one", 4096, out, PRIORITY_HIGH))
                    b.start()
                    a.join(30)
                    b.join(30)
                else:
                    ask(svc, "long one", 4096, out, PRIORITY_LOW)
            return out["long one"][0], log.getvalue()

        plain, plog = run_both(False)
        self.assertIn("loop guard", plog)              # the guard acted: this answer really loops
        cut, clog = run_both(True)
        self.assertIn("priority: a low answer paused", clog)
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

    def test_an_http_1_0_client_is_not_answered_with_chunked_padding(self):
        # an HTTP/1.0 client parses no chunked body: it waits quietly (as before R122) and the answer arrives
        # whole - no Transfer-Encoding, no space before the JSON, the status line its own version
        body = {"model": "m", "stream": False, "messages": [{"role": "user", "content": "wait one"}]}
        plain = json.loads(self.post("/v1/chat/completions", body)[1])
        wire = json.dumps(body)
        port = self.httpd.server_address[1]

        def raw():
            with socket.create_connection(("127.0.0.1", port), timeout=60) as s:
                s.sendall(("POST /v1/chat/completions HTTP/1.0\r\nContent-Type: application/json\r\n"
                           f"Content-Length: {len(wire)}\r\nConnection: close\r\n\r\n{wire}").encode())
                data = b""
                while True:
                    chunk = s.recv(1 << 16)
                    if not chunk:
                        return data
                    data += chunk

        raw_out = self.blocked(raw)
        head, _, out = raw_out.partition(b"\r\n\r\n")
        self.assertIn(b"HTTP/1.0 200", head.split(b"\r\n")[0])
        self.assertNotIn(b"chunked", head.lower())
        self.assertEqual(json.loads(out)["choices"][0]["message"]["content"],
                         plain["choices"][0]["message"]["content"])   # the same answer, no space before it

    def test_an_error_of_the_gpu_busy_kind_rides_the_body_after_the_headers(self):
        class Busy(ResumingEngine):                       # a GpuBusy or QueueWaitExpired is a plain RuntimeError:
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):    #  the worker thread used to
                if "wait" in self.tok.decode(ids):        #  die on it with the box empty, and the main thread's
                    raise RuntimeError("the GPU is busy") #  KeyError reached the client as a "malformed request"
                return super().generate(ids, max_new, sampling, cancel, embeddings)
        self.svc.engine = Busy(self.tok, {"wait": WaitKeepAlive.WAIT, "block": LONG}, delay_s=0.004)
        body = {"model": "m", "stream": False, "messages": [{"role": "user", "content": "wait one"}]}
        with contextlib.redirect_stdout(io.StringIO()):
            s, out = self.blocked(lambda: self.post("/v1/chat/completions", body))   # it ends after the headers
        self.assertEqual(s, 200)                          # went out, so only the real error can follow -
        self.assertIn("the GPU is busy", json.loads(out)["error"]["message"])     #  not a KeyError about a body

    def test_a_paused_nonstream_answer_gets_its_headers_at_once(self):
        os.environ["SAGUARO_WAIT_SPACE_S"] = "30"        # only the pause starts it, not the waiting
        body = {"model": "m", "stream": False, "priority": "low",
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

    def post_h(self, path, body):
        """post() with the response's headers too (a padded answer arrives chunked; a plain one does not)."""
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, dict(r.headers), r.read()

    def test_an_unqueued_slow_answer_keeps_the_plain_framing(self):
        # (finding 1, Ken's decision on the trigger): the padding is for requests that wait for the model. This
        # one answers alone on an idle server, slower than SAGUARO_WAIT_SPACE_S: the gate never opens, so the
        # 200 and headers go with the JSON, not before it, and the wire is the pre-R122 one.
        self.svc.engine = ResumingEngine(self.tok, {"wait": WaitKeepAlive.WAIT, "block": LONG}, delay_s=0.006)
        body = {"model": "m", "stream": False, "messages": [{"role": "user", "content": "block one"}]}
        s, head, out = self.post_h("/v1/chat/completions", body)          # ~2.4 s at this pace: over the 0.3 s
        self.assertEqual(s, 200)
        self.assertFalse(out.startswith(b" "))         # no padding spaces before the JSON ...
        self.assertIsNone(head.get("Transfer-Encoding"))    # ... and no chunked framing: the plain Content-Length
        self.assertEqual(int(head["Content-Length"]), len(out))           # the answer whole, in one framed body
        again = self.post("/v1/chat/completions", body)[1]                # the undisturbed answer itself:
        content = lambda b: json.loads(b)["choices"][0]["message"]["content"]
        self.assertEqual(content(out), content(again))               #  same text, framing changed nothing

    def test_an_unqueued_error_keeps_its_status(self):
        # (finding 1, the same gate's other half): the old trigger started padding at the deadline of ANY slow
        # answer, so a failure after it rode a 200 with the error in the body. A request that never waited has
        # no padding, so its status is still free when the answer fails: the client gets the real 400.
        class SlowFail(ResumingEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                if "wait" in self.tok.decode(ids):
                    time.sleep(0.6)                    # past SAGUARO_WAIT_SPACE_S (0.3 here) - nothing queued
                    raise ValueError("the prompt is too long for the context")
                return super().generate(ids, max_new, sampling, cancel, embeddings)
        self.svc.engine = SlowFail(self.tok, {"wait": WaitKeepAlive.WAIT, "block": LONG}, delay_s=0.004)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            s, out = self.post("/v1/chat/completions",
                               {"model": "m", "stream": False,
                                "messages": [{"role": "user", "content": "wait one"}]})
        self.assertEqual(s, 400)                       # the error's own status, not a 200 carrying it
        self.assertIn("too long", json.loads(out)["error"]["message"])
        self.assertNotIn("the status and headers go out now", log.getvalue())   # nothing ever padded

    def test_an_unqueued_long_read_gets_no_empty_events(self):
        # (finding 1, the streams): a prompt read's PP ticks arrive as the same None items the queue's pings do,
        # so the old gate put an empty event into ANY >SAGUARO_WAIT_EVENT_S read - of a stream that never
        # queued and never paused. With the gate the read keeps its plain keep-alive comments and nothing else,
        # on the three APIs that have an empty event at all.
        class QuickRead(ReadEngine):
            CHUNKS = 3                                 # ~0.45 s of read at this delay, over the 0.3 s gap
        self.svc.engine = QuickRead(self.tok, {"wait": SHORT}, delay_s=0.15)
        events, comments = self.sse(self.stream("/v1/chat/completions", {
            "model": "m", "stream": True, "messages": [{"role": "user", "content": "wait one"}]}))
        empty = [p for _n, p in events
                 if p["choices"][0]["delta"] == {} and p["choices"][0]["finish_reason"] is None]
        self.assertEqual(empty, [])                    # no empty chunk: a read is not a wait ...
        self.assertGreater(comments, 0)                # ... and the plain keep-alive comments still came
        text = "".join(p["choices"][0]["delta"].get("content", "") for _n, p in events)
        plain = json.loads(self.post("/v1/chat/completions", {            # the same request not streamed:
            "model": "m", "stream": False,
            "messages": [{"role": "user", "content": "wait one"}]})[1])   #  the same answer, nothing changed
        self.assertEqual(text, plain["choices"][0]["message"]["content"])
        evs, _ = self.sse(self.stream("/v1/messages", {
            "model": "m", "max_tokens": 256, "stream": True,
            "messages": [{"role": "user", "content": "wait one"}]}))
        self.assertNotIn("ping", [n for n, _p in evs])                       # no Anthropic ping either
        evs, _ = self.sse(self.stream("/v1/responses", {"model": "m", "input": "wait one", "stream": True}))
        self.assertEqual([n for n, _p in evs].count("response.in_progress"), 1)   # the start's, alone


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
        for path, req in (("/v1/chat/completions", {**chat, "stream": False, "priority": "low"}),
                          ("/v1/messages", {**chat, "max_tokens": 64, "priority": "high"}),
                          ("/v1/responses", {"model": "m", "input": "hi", "stream": False,
                                             "priority": "low"})):
            s, b = self.post(path, req)
            self.assertEqual(s, 200, f"{path}: {b[:200]!r}")
        s, b = self.post("/v1/chat/completions", {**chat, "stream": False}, {"X-Saguaro-Priority": "high"})
        self.assertEqual(s, 200, b[:200])              # and the header on its own

    def test_the_ollama_apis_take_the_field_too(self):
        # (the review's nit 7): /api/chat and /api/generate carry the field through the ollama translation
        # (serve/model_api.py) - asserted at the translation AND sent through both endpoints, bad value included.
        self.assertEqual(model_api.chat_to_openai({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                                   "priority": "low"})["priority"], "low")
        self.assertEqual(model_api.generate_to_openai({"model": "m", "prompt": "hi",
                                                       "priority": "high"})["priority"], "high")
        was = getattr(self.svc, "model_management", False)
        self.svc.model_management = True                    # (both ollama endpoints are gated on it, as before)
        try:
            for path, body in (("/api/chat", {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                              "stream": False, "priority": "low"}),
                               ("/api/generate", {"model": "m", "prompt": "hi", "stream": False,
                                                  "priority": "high"})):
                s, b = self.post(path, body)
                self.assertEqual(s, 200, f"{path}: {b[:200]!r}")
            s, b = self.post("/api/chat", {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                                        "stream": False}, {"X-Saguaro-Priority": "high"})
            self.assertEqual(s, 200, b[:200])               # and the header on the ollama body too
            s, b = self.post("/api/generate", {"model": "m", "prompt": "hi", "stream": False,
                                               "priority": "urgent"})
            self.assertEqual(s, 400)                        # a bad value is a 400 here too, naming the three ...
            self.assertIn('priority: "high", "normal" or "low"', json.loads(b)["error"])   # (ollama's own shape:
        finally:                                            #  the error is the message, not an object)
            self.svc.model_management = was

    def test_a_bad_value_is_a_400_that_names_the_field(self):
        chat = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False}
        s, b = self.post("/v1/chat/completions", {**chat, "priority": "urgent"})
        self.assertEqual(s, 400)
        self.assertIn('priority: "high", "normal" or "low"', json.loads(b)["error"]["message"])
        s, b = self.post("/v1/chat/completions", chat, {"X-Saguaro-Priority": "5"})
        self.assertEqual(s, 400)
        self.assertIn("X-Saguaro-Priority", json.loads(b)["error"]["message"])

    def test_the_body_wins_the_header_loses_and_the_config_default_applies(self):
        svc = self.svc
        req = {"messages": [], "priority": "normal", "_priority_header": "low"}
        svc.check_request(req)
        self.assertEqual(req["_priority"], PRIORITY_NORMAL)        # the body wins the header
        req = {"messages": [], "_priority_header": "low"}
        svc.check_request(req)
        self.assertEqual(req["_priority"], PRIORITY_LOW)
        req = {"messages": []}
        svc.check_request(req)
        self.assertEqual(req["_priority"], PRIORITY_NORMAL)        # nothing said anywhere: normal
        svc.priority_default = PRIORITY_LOW                 # the run config's "priority"
        try:
            req = {"messages": []}
            svc.check_request(req)
            self.assertEqual(req["_priority"], PRIORITY_LOW)
            req = {"messages": [], "priority": "high"}
            svc.check_request(req)
            self.assertEqual(req["_priority"], PRIORITY_HIGH)      # a request still overrides the config
        finally:
            svc.priority_default = None
        for bad in ("urgent", "background", "idle", 5, True, {"p": 1}):    # the old names: refused like any
            with self.assertRaises(ValueError, msg=repr(bad)):             #  other wrong value (ONE name)
                svc.check_request({"messages": [], "priority": bad})

    def test_the_metrics_rows_and_the_live_state_carry_it(self):
        answer = "".join(random.Random(9).choices(string.ascii_lowercase, k=60))     # (no loop)
        self.svc.engine = MockEngine(self.tok, [answer], tok_s=25)                   # slow: it is still running
        try:
            a = threading.Thread(target=self.post,
                                 args=("/v1/chat/completions",
                                       {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                        "stream": False, "priority": "low"}))
            a.start()
            time.sleep(0.5)
            with urllib.request.urlopen(self.base + "/metrics", timeout=15) as r:
                m = json.loads(r.read())
            a.join(30)
            rows = m["live"]["chats"]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["priority"], "low")     # the "Right now" row's tag
            self.assertFalse(rows[0]["paused_for_priority"])
            self.assertEqual(m["live"]["priority"], "low")   # and the live state's fields
            self.assertFalse(m["live"]["paused_for_priority"])
        finally:
            self.svc.engine = self.engine
        with urllib.request.urlopen(self.base + "/metrics", timeout=15) as r:
            m = json.loads(r.read())
        self.assertEqual(m["requests"][0]["priority"], "low")   # the finished request's row says it too


if __name__ == "__main__":
    unittest.main()


class PicturesNotStreamed(Env):
    """1.2.2 gate (9070 XT box): a non-streaming answer runs on _collect_wait's side thread, and the request's
    picture embeddings live in a thread-local (Service.embeddings) - the side thread saw none, so the engine got the
    picture's pad tokens with no picture and GLM described one that was not there. Every API, not streamed."""

    class Vision:
        owed = False

        def __init__(self, d):
            self.dir = Path(d)
            self.rows = self.dir / "img.sve"
            self.rows.write_bytes(b"rows")

        def encode(self, source):
            return self.rows, 3

        def active(self):
            return contextlib.nullcontext(self)

    class Engine(MockEngine):
        def generate(self, ids, max_new, sampling, cancel, embeddings=None):
            self.seen.append(embeddings)
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)

    def test_every_api_hands_the_picture_to_the_engine(self):
        import tempfile
        tok = ByteTokenizer()
        red = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        bodies = {
            "/v1/chat/completions": {"model": "x", "max_tokens": 4, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": red}}, {"type": "text", "text": "colour?"}]}]},
            "/v1/messages": {"model": "x", "max_tokens": 4, "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": red.split(",")[1]}},
                {"type": "text", "text": "colour?"}]}]},
            "/v1/responses": {"model": "x", "max_output_tokens": 4, "input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": red}, {"type": "input_text", "text": "colour?"}]}]},
            "/api/chat": {"model": "x", "stream": False, "options": {"num_predict": 4}, "messages": [
                {"role": "user", "content": "colour?", "images": [red.split(",")[1]]}]},
        }
        with tempfile.TemporaryDirectory() as d:
            engine = self.Engine(tok, "ok", max_context=1 << 16)
            engine.seen = []
            svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=self.Vision(d))
            svc.model_management = True                  # (/api/chat is gated on model management)
            httpd = serve(svc, port=0)
            try:
                for path, body in bodies.items():
                    with self.subTest(api=path):
                        engine.seen.clear()
                        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}",
                                                     data=json.dumps(body).encode(),
                                                     headers={"Content-Type": "application/json"})
                        with urllib.request.urlopen(req, timeout=60) as r:
                            self.assertEqual(r.status, 200)
                        self.assertTrue(engine.seen and engine.seen[0], f"{path}: the engine got no picture")
            finally:
                httpd.shutdown()


class StopReachesTheEngine(unittest.TestCase):
    """Paul's box (Python 3.12): closing the QFN engine's generate() left the driver's own generator open - its
    finally (STOP + drain to DONE) ran only when the outer generator was freed - so a cut answer (a priority pause,
    a client gone) never stopped, and the NEXT request read the old answer's tokens. The wrapper closes it now."""

    class Stdin:
        def __init__(self, eng):
            self.eng, self.sent = eng, []

        def write(self, s):
            self.sent.append(s.strip())
            if s.startswith("STOP"):
                self.eng.lines.put("DONE 3 3 1.0 1.0 cancel")

        def flush(self):
            pass

    def test_closing_the_answer_stops_the_engine_before_close_returns(self):
        import queue
        from serve.server import _qfn_engine_class
        cls = _qfn_engine_class()
        eng = cls.__new__(cls)
        eng.lines, eng.batch, eng.can_stop, eng.silence_s = queue.Queue(), 0, True, 0
        eng.proc = unittest.mock.Mock()
        eng.proc.stdin = self.Stdin(eng)
        eng.alive = lambda: True
        eng.info, eng.progress, eng._tls, eng._reader, eng.read_live = {}, None, threading.local(), None, None
        for i in range(1, 200):
            eng.lines.put(f"T {i}")
        with unittest.mock.patch("serve.qfn_engine.bias_api.validate_request", lambda *a, **k: None):
            gen = eng.generate([1, 2, 3], 500, {}, threading.Event())
            got = [next(gen) for _ in range(3)]
            gen.close()                               # (`gen` stays referenced, as in Service._sliced)
            self.assertEqual(got, [1, 2, 3])
            self.assertIn("STOP", eng.proc.stdin.sent, "the engine was not told to stop when the answer closed")


class ClosingTheOtherWrappersClosesTheirRun(unittest.TestCase):
    """The same 3.12 shape in the two wrappers P124c's audit found still bare: openai_chunks (the plain
    chat/completions and ollama routes) closed its run only when its frame was GC'd - a client gone mid-answer left
    the engine's STOP + drain for the NEXT request to trip over. response_events already closed its run in a
    try/finally; its test keeps that true. (On 3.13 both pass either way: the manager runs these on 3.12.)"""

    @staticmethod
    def run_events(log):
        try:
            for i in range(1000):
                yield "ping", None
        finally:
            log.append("closed")

    def test_closing_the_openai_answer_closes_its_run_before_close_returns(self):
        from serve.server import openai_chunks
        svc = unittest.mock.Mock()
        svc.model = "m"
        log = []
        gen = openai_chunks(svc, {}, [1, 2], None, None, 100, threading.Event(), run=self.run_events(log))
        got = [next(gen) for _ in range(4)]
        self.assertEqual((got[0]["object"], got[-1]), ("chat.completion.chunk", None))
        gen.close()                                   # (`gen` stays referenced, as in the routes' finally)
        self.assertEqual(log, ["closed"], "openai_chunks left its run to the frame GC")

    def test_closing_the_response_answer_closes_its_run_before_close_returns(self):
        from serve.responses import response_events
        log = []
        gen = response_events("m", {}, self.run_events(log))
        got = [next(gen) for _ in range(4)]
        self.assertEqual([got[0][0], got[1][0], got[-1]], ["response.created", "response.in_progress", None])
        gen.close()                                   # (`gen` stays referenced, as in _response_work's finally)
        self.assertEqual(log, ["closed"], "response_events left its run to the frame GC")


class IdleGraceDefault(unittest.TestCase):
    def test_the_default_grace_is_zero(self):
        from serve.server import priority_idle_seconds
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SAGUARO_PRIORITY_IDLE_S", None)
            self.assertEqual(priority_idle_seconds(), 0.0)   # (1.2.4: 30 -> 5; P125j: 0 - reads pause between layers)
