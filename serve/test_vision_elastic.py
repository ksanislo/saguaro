"""serve/test_vision_elastic.py - the elastic QFN picture encoder in the shared server (no GPU, no engine): the first
picture makes the expert cache keep elastic_mib free and only then starts the encoder; after elastic_idle_s without
one it stops and the reserve goes back to 0.

    cd serve && python -m unittest -q test_vision_elastic
"""
from __future__ import annotations

import contextlib
import sys
import tempfile
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

    def make(self, refuse=False, elastic_idle_s=3600.0, resident=False):
        tok = ByteTokenizer()
        log, d = [], tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        if resident:
            engine, vision = MockEngine(tok, "ok", max_context=CTX), self.Resident(d.name, log)
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


if __name__ == "__main__":
    unittest.main()
