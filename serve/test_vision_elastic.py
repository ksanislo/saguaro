"""serve/test_vision_elastic.py - the elastic QFN picture encoder in the shared server (no GPU, no engine): the first
picture makes the expert engine lend elastic_mib of VRAM (VLEND on an engine that says vision_lend=1, its old
vram_reserve otherwise) and only then starts the encoder; after elastic_idle_s without one it stops and the room is
reclaimed (VRECLAIM), while conversations keep decoding when the engine lends.

    cd serve && python -m unittest -q test_vision_elastic
"""
from __future__ import annotations

import contextlib
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096


class ElasticVision(unittest.TestCase):
    """What the service does around the encode turn, with a fake engine and a fake reader (qfn_engine.Vision's shape)."""

    class Engine(MockEngine):
        """QFN's engine here: a vram_reserve that records its numbers (refuse: one without --vram-elastic)."""

        def __init__(self, tok, log, refuse=False):
            super().__init__(tok, "ok", max_context=CTX)
            self.log, self.refuse, self.vram_calls = log, refuse, []

        def vram_reserve(self, reserve_mib):
            self.log.append(("vram", reserve_mib))
            self.vram_calls.append(reserve_mib)
            if self.refuse:
                raise ValueError("VRAM needs an engine started with --vram-elastic (one NVIDIA GPU, --serve)")
            return {"expert_cache_mib": 12288, "expert_cache_full_mib": 13312}

    class LendEngine(MockEngine):
        """A QFN engine that supports VLEND / VRECLAIM (its INFO says vision_lend=1): it lends CUDA0's VRAM while the
        batch slots decode (refuse: one that answers ERR VLEND, its cache at its smallest size)."""

        def __init__(self, tok, log, refuse=False):
            super().__init__(tok, "ok", max_context=CTX)
            self.log, self.refuse = log, refuse
            self.info = {"logit_bias": 1, "vision_lend": 1}
            self.lend_calls, self.reclaim_calls = [], []

        def vision_lend(self, mib, timeout=45.0):
            self.log.append(("lend", mib))
            self.lend_calls.append(mib)
            if self.refuse:
                raise ValueError("VLEND: CUDA0's expert cache is at its smallest size of 128 slots: 900 of 1696 MiB free")
            return {"free_mib": 1700, "cache_mib": 11008}

        def vision_reclaim(self, timeout=45.0):
            self.log.append(("reclaim",))
            self.reclaim_calls.append(True)
            return {"cache_mib": 12544}

    class Reader:
        """The elastic encoder: alive/restart/unload, the elastic numbers, an encode that records; no active()."""

        def __init__(self, d, log, elastic_idle_s=3600.0):
            self.dir = Path(d)
            self.rows = self.dir / "img.sve"
            self.rows.write_bytes(b"rows")
            self.log = log
            self.elastic = True
            self.elastic_mib = 1536
            self.elastic_idle_s = elastic_idle_s
            self.running = False

        def alive(self):
            return self.running

        def restart(self):
            self.log.append(("start",))
            self.running = True

        def unload(self):
            self.log.append(("unload",))
            self.running = False

        def encode(self, source):
            self.log.append(("encode",))
            return self.rows, 3

    class Resident:
        """The shared server's own encoder shape (no "elastic"): active() around the encode, no alive/restart/unload."""

        def __init__(self, d, log):
            self.dir = Path(d)
            self.rows = self.dir / "img.sve"
            self.rows.write_bytes(b"rows")
            self.log = log
            self.owed = False

        def encode(self, source):
            self.log.append(("encode",))
            return self.rows, 3

        def active(self):
            return contextlib.nullcontext(self)

    def make(self, refuse=False, elastic_idle_s=3600.0, resident=False, lend=False):
        tok = ByteTokenizer()
        log, d = [], tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        if resident:
            engine, vision = MockEngine(tok, "ok", max_context=CTX), self.Resident(d.name, log)
        elif lend:
            engine, vision = self.LendEngine(tok, log, refuse), self.Reader(d.name, log, elastic_idle_s)
        else:
            engine, vision = self.Engine(tok, log, refuse), self.Reader(d.name, log, elastic_idle_s)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=vision)
        self.addCleanup(lambda: getattr(svc.embeddings, "path", None)
                        and svc.embeddings.path.unlink(missing_ok=True))
        return svc, log

    def picture(self, svc):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                             {"type": "image", "source": "x.png"}]}]
        return svc.prepare(msgs, None, {})

    def test_no_encoder_until_a_picture(self):
        svc, log = self.make()
        self.assertFalse(svc.vision.running)                    # "elastic": true: the start built nothing
        self.assertEqual(svc._vision_idle_down(force=True), "nothing")
        self.assertEqual(log, [])

    def test_first_picture_keeps_room_then_starts_then_encodes(self):
        svc, log = self.make()
        self.picture(svc)
        self.assertEqual(log, [("vram", 1536), ("start",), ("encode",)])   # the room first, the encoder after it
        self.assertTrue(svc.vision.running)
        self.assertTrue(svc.vision_held)
        self.assertGreater(svc.last_image_at, 0.0)
        self.assertEqual(svc.fifo._used, 0)                     # the turn was given back with the encode

    def test_idle_stops_the_encoder_and_gives_the_vram_back(self):
        svc, log = self.make()
        self.picture(svc)
        self.assertEqual(svc._vision_idle_down(force=True), "down")
        self.assertEqual([e[0] for e in log], ["vram", "start", "encode", "unload", "vram"])
        self.assertEqual(svc.engine.vram_calls, [1536, 0])      # the encoder's room goes back to the cache
        self.assertFalse(svc.vision.running)
        self.assertFalse(svc.vision_held)

    def test_a_picture_that_just_came_keeps_it_up(self):
        svc, log = self.make()                                  # elastic_idle_s 3600: far from idle
        self.picture(svc)
        self.assertEqual(svc._vision_idle_down(), "busy")
        self.assertTrue(svc.vision.running)
        self.assertEqual(svc.engine.vram_calls, [1536])

    def test_a_request_holding_the_engine_keeps_it_up(self):
        svc, log = self.make(elastic_idle_s=0.0)                # idle time passed: the FIFO is what says busy
        self.picture(svc)
        self.assertTrue(svc.fifo.acquire())                     # a conversation has the engine: no VRAM command now
        try:
            self.assertEqual(svc._vision_idle_down(), "busy")
        finally:
            svc.fifo.release()
        self.assertTrue(svc.vision.running)
        self.assertEqual(svc.engine.vram_calls, [1536])

    def test_a_cache_that_cannot_shrink_still_gets_its_encoder(self):
        svc, log = self.make(refuse=True)                       # an engine without --vram-elastic
        self.picture(svc)                                       # the refusal costs a log line, not the request
        self.assertEqual(log, [("vram", 1536), ("start",), ("encode",)])
        self.assertTrue(svc.vision.running)
        self.assertFalse(svc.vision_held)                       # nothing was held, so idle-down gives nothing back
        self.assertEqual(svc._vision_idle_down(force=True), "down")
        self.assertEqual(svc.engine.vram_calls, [1536])

    def test_the_idle_thread_stops_an_idle_encoder(self):
        svc, log = self.make(elastic_idle_s=0.5)                # the thread the server starts, on its own clock
        svc.start_vision_idle()
        self.picture(svc)
        deadline = time.time() + 6.0                            # it looks at least once a second
        while svc.engine.vram_calls != [1536, 0] and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(svc.engine.vram_calls, [1536, 0])      # stopped and given back by the thread, not by us
        self.assertFalse(svc.vision.running)
        self.assertFalse(svc.vision_held)

    def test_a_resident_encoder_is_left_exactly_as_it_was(self):
        svc, log = self.make(resident=True)                     # no "elastic": the GLM/Qwen path, untouched
        self.assertEqual(svc._vision_idle_down(force=True), "nothing")
        self.picture(svc)
        self.assertEqual(log, [("encode",)])                    # active() around it, no engine commands, no clock
        self.assertFalse(svc.vision_held)

    # ---- the VLEND / VRECLAIM path (the engine says vision_lend=1): the encoder lives beside the decoding slots

    def test_lend_first_picture_lends_room_then_starts_then_encodes(self):
        svc, log = self.make(lend=True)
        self.picture(svc)
        self.assertEqual([e[0] for e in log], ["lend", "start", "encode"])   # the room first, the encoder after it
        self.assertEqual(svc.engine.lend_calls, [1536])
        self.assertTrue(svc.vision.running)
        self.assertTrue(svc.vision_held)
        self.assertEqual(svc.fifo._used, 0)                     # the turn was given back with the encode

    def test_lend_encode_takes_only_its_own_fifo_place(self):
        svc, log = self.make(lend=True)
        svc.fifo.set_capacity(8)                                # the --batch 8 engine: seven conversations decode
        for _ in range(7):
            self.assertTrue(svc.fifo.acquire(units=1))
        try:
            th = threading.Thread(target=lambda: self.picture(svc), daemon=True)
            th.start()
            th.join(timeout=10.0)                               # (with units=None it would sit there: seven used)
            self.assertEqual([e[0] for e in log], ["lend", "start", "encode"])
        finally:
            for _ in range(7):
                svc.fifo.release(units=1)

    def test_lend_idle_down_reclaims_while_a_slot_decodes(self):
        svc, log = self.make(lend=True)
        self.picture(svc)
        self.assertTrue(svc.fifo.acquire())                     # a conversation decodes: VRECLAIM works anyway
        try:
            self.assertEqual(svc._vision_idle_down(force=True), "down")
        finally:
            svc.fifo.release()
        self.assertEqual([e[0] for e in log], ["lend", "start", "encode", "unload", "reclaim"])
        self.assertFalse(svc.vision.running)
        self.assertFalse(svc.vision_held)

    def test_lend_running_encode_keeps_the_encoder_up(self):
        svc, log = self.make(lend=True, elastic_idle_s=0.0)     # idle long ago: the encode lock is what says busy
        self.picture(svc)
        self.assertTrue(svc._vision_enc.acquire(blocking=False))  # an encode is running on the encoder
        try:
            self.assertEqual(svc._vision_idle_down(), "busy")
        finally:
            svc._vision_enc.release()
        self.assertTrue(svc.vision.running)
        self.assertEqual(svc.engine.reclaim_calls, [])

    def test_lend_refusal_is_still_followed_by_a_reclaim(self):
        svc, log = self.make(lend=True, refuse=True)            # ERR VLEND: the cache is at its smallest already
        self.picture(svc)                                       # the refusal costs a log line, not the request
        self.assertEqual([e[0] for e in log], ["lend", "start", "encode"])
        # a refused / timed-out VLEND may still leave the engine's floor raised: the idle stop reclaims after any VLEND
        self.assertTrue(svc.vision_held)
        self.assertEqual(svc._vision_idle_down(force=True), "down")
        self.assertEqual([e[0] for e in log], ["lend", "start", "encode", "unload", "reclaim"])
        self.assertEqual(svc.engine.reclaim_calls, [True])
        self.assertFalse(svc.vision.running)
        self.assertFalse(svc.vision_held)

    def test_the_idle_thread_stops_a_lent_encoder(self):
        svc, log = self.make(lend=True, elastic_idle_s=0.5)     # the thread the server starts, on its own clock
        svc.start_vision_idle()
        self.picture(svc)
        deadline = time.time() + 6.0
        while not svc.engine.reclaim_calls and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(svc.engine.reclaim_calls, [True])      # stopped and reclaimed by the thread, not by us
        self.assertFalse(svc.vision.running)
        self.assertFalse(svc.vision_held)


class LendProtocol(unittest.TestCase):
    """The adapter's answer routing (server.py's _VlenLines and QfnEngine._vlen_command): a VLENT that arrives late,
    after the request lines of the batch readers, still reaches vision_lend; the answer to an already-timed-out
    command is dropped before the next one."""

    @staticmethod
    def engine():
        import queue
        from serve.server import _VlenLines, _qfn_engine_class

        base = _qfn_engine_class()

        class Stub(base):                           # (no engine process here: only the line plumbing)
            def __init__(self):
                self.vlen_lines = queue.Queue()
                self.lines = _VlenLines(self.vlen_lines)
                self.info = {"vision_lend": 1}
                self.ended = False
                self.sent = []

            def alive(self):
                return True

            def _send(self, text):
                self.sent.append(text)

        return Stub()

    def test_vlent_answer_arrives_after_other_lines(self):
        eng = self.engine()

        def late():                                 # (what the engine's pump does while the batch requests run)
            eng.lines.put("KMEM 1 0 1103 900 500\n")            # an ordinary line of the batch readers
            time.sleep(0.3)
            eng.lines.put("VLENT free_mib=1700 cache_mib=11008\n")

        threading.Thread(target=late, daemon=True).start()
        out = eng.vision_lend(1536, timeout=10.0)
        self.assertEqual(out, {"free_mib": 1700, "cache_mib": 11008})
        self.assertEqual(eng.sent, ["VLEND 1536"])
        self.assertEqual(eng.lines.get_nowait(), "KMEM 1 0 1103 900 500\n")   # its line stayed with the readers

    def test_stale_answer_is_dropped_before_the_next_command(self):
        eng = self.engine()
        eng.vlen_lines.put("VLENT free_mib=1 cache_mib=1\n")    # the answer to an earlier, timed-out VLEND

        def late():
            time.sleep(0.2)
            eng.lines.put("VRECLAIMED cache_mib=12544\n")

        threading.Thread(target=late, daemon=True).start()
        out = eng.vision_reclaim(timeout=10.0)
        self.assertEqual(out, {"cache_mib": 12544})             # the fresh answer, not the stale one
        self.assertEqual(eng.sent, ["VRECLAIM"])


if __name__ == "__main__":
    unittest.main()


class InfoAfterReady(unittest.TestCase):
    """The live engine prints INFO vision_lend=1 after READY (its batch setup): it must reach engine.info, not a request."""

    def test_info_line_after_ready_joins_the_info(self):
        import queue as _q
        from serve.server import _VlenLines
        info = {"kv_bytes_token": 12672}
        lines = _VlenLines(_q.Queue(), info)
        lines.put("INFO vision_lend=1")
        lines.put("T 0 42")
        self.assertEqual(info.get("vision_lend"), 1)
        self.assertEqual(lines.get_nowait(), "T 0 42")
        self.assertTrue(lines.empty())


class MapWhileLent(unittest.TestCase):
    """While the encoder holds VRAM the engine gave up (VLEND), CUDA0's engine parts shrink to the engine's NVML total:
    the card is never drawn fuller than it is (it showed 16/16 live)."""

    def test_parts_follow_the_engine_total(self):
        from serve import gpu_map
        G = 1 << 20
        rep = [{"dev": 0, "pci": "0000:01:00", "total": 15360 * G, "free": 200 * G, "weights": 4000 * G,
                "kv": 2000 * G, "act": 0, "pool": 9000 * G, "act_own": 1}]
        cards = [{"pci": "0000:01:00", "index": 0, "total": 15360 * G, "used": 15200 * G,
                  "procs": {10: 13634 * G, 11: 1196 * G}}]
        v = gpu_map.view(rep, cards, 10, 11)[0]
        p = v["parts"]
        self.assertEqual(p["weights"] + p["kv"] + p["pool"] + p["act"], 13634 * G)   # the engine as NVML lists it
        self.assertEqual(p["vision"], 1196 * G)
        self.assertLessEqual(sum(p.values()), 15360 * G)
