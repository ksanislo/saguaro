"""The elastic image encoder (the vision section's "elastic": true with "vram_elastic"): the first picture makes the
expert cache keep the encoder's VRAM free and then starts it, later pictures use it as it is, an idle encoder stops and
the cache takes its VRAM back, a POST /v1/vram reserve stacks on the encoder's share, and the engine's load never starts
it on its own.  CheapRestart: the cheap-restart state the stop leaves behind (PLAN-vision-fast-start)."""
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import serve.server as server
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, Vision
from serve.test_vram import VramEngine

ROOT = Path(__file__).resolve().parents[1]
IMAGE = {"type": "image", "source": "x.png"}


class ElasticVision:
    """An encoder process stand-in: starts, stops, and turns every picture into three image tokens."""
    def __init__(self, d, elastic=True):
        self.elastic, self.elastic_mib, self.elastic_idle_s = elastic, 2048, 60.0
        self.up, self.starts, self.unloads = False, 0, 0
        self.dir = Path(d)
        self.rows = self.dir / "img.sve"
        self.rows.write_bytes(b"rows")

    def alive(self):
        return self.up

    def restart(self):
        self.up, self.starts = True, self.starts + 1

    def unload(self):
        self.up, self.unloads = False, self.unloads + 1

    def encode(self, source):
        assert self.up, "a picture reached an encoder that was not started"
        return self.rows, 3


class Elastic(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        self.tok = ByteTokenizer()
        self.engine = VramEngine(self.tok)
        self.engine.loaded = True
        self.vision = ElasticVision(d)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=self.vision)
        self.addCleanup(self.drop_embeddings)

    def drop_embeddings(self):
        path = getattr(getattr(self.svc, "embeddings", None), "path", None)
        if path is not None:
            path.unlink(missing_ok=True)

    def picture(self):
        self.svc.prepare([{"role": "user", "content": [{"type": "text", "text": "what is it"}, IMAGE]}], None, {})

    def test_the_first_picture_makes_room_then_starts_the_encoder(self):
        self.picture()
        self.assertEqual(self.engine.calls, [2048])         # the cache keeps the encoder's share free first
        self.assertEqual(self.vision.starts, 1)
        self.assertTrue(self.svc.vision_held)
        self.picture()                                      # the next one uses the running encoder
        self.assertEqual(self.engine.calls, [2048])
        self.assertEqual(self.vision.starts, 1)

    def test_an_idle_encoder_stops_and_the_cache_grows_back_whole(self):
        self.picture()
        self.assertEqual(self.svc._vision_idle_down(), "busy")             # a picture came just now
        self.svc.last_image_at = time.time() - 61
        with self.svc.fifo:                                                # a request holds the turn
            self.assertEqual(self.svc._vision_idle_down(), "busy")
        self.assertEqual(self.svc._vision_idle_down(), "down")
        self.assertFalse(self.vision.up)
        self.assertFalse(self.svc.vision_held)
        self.assertEqual(self.engine.calls, [2048, 0])      # 0, not None: the full size, not the start's reserve
        self.assertEqual(self.svc._vision_idle_down(), "nothing")
        self.picture()                                      # and the next picture starts it again
        self.assertEqual(self.vision.starts, 2)
        self.assertEqual(self.engine.calls, [2048, 0, 2048])

    def test_a_vram_reserve_stacks_on_the_encoders_share(self):
        self.svc.vram(4000)
        self.picture()
        self.assertEqual(self.engine.calls, [4000, 6048])
        self.svc.vram(1000)                                 # asked while the encoder runs
        self.assertEqual(self.engine.calls[-1], 3048)
        self.svc.last_image_at = 0
        self.svc._vision_idle_down()
        self.assertEqual(self.engine.calls[-1], 1000)       # back to the asked reserve, not to 0

    def test_an_engine_that_cannot_shrink_still_gets_its_encoder(self):
        self.engine.elastic = False
        self.picture()
        self.assertEqual(self.vision.starts, 1)
        self.assertFalse(self.svc.vision_held)
        self.svc.last_image_at = 0
        self.assertEqual(self.svc._vision_idle_down(), "down")
        self.assertEqual(self.engine.calls, [2048])         # the refused shrink only: nothing to give back

    def test_the_load_never_starts_an_elastic_encoder(self):
        self.assertFalse(self.svc._vision_down())
        self.svc.vision = ElasticVision(tempfile.mkdtemp(), elastic=False)
        self.assertTrue(self.svc._vision_down())            # a non-elastic encoder that is down is started again

    def test_an_unload_lets_go_of_the_share(self):
        self.picture()
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertFalse(self.svc.vision_held)
        self.assertFalse(self.vision.up)

    def test_the_config_keys(self):
        v = Vision({"exe": "e", "mmproj": "m", "model": "t", "elastic": True, "elastic_mib": 3000,
                    "elastic_idle_s": 30}, lazy=False)          # elastic: nothing starts with the server
        self.assertTrue(v.elastic and v.elastic_mib == 3000 and v.elastic_idle_s == 30 and not v.alive())
        self.assertEqual(v.elastic_keep_s, 600.0)               # the cheap-restart state (hot weights) outlives the stop
        v = Vision({"exe": "e", "mmproj": "m", "model": "t", "elastic": True, "elastic_keep_s": 0}, lazy=True)
        self.assertEqual(v.elastic_keep_s, 0.0)                 # and it is tunable per config (0: off)
        v = Vision({"exe": "e", "mmproj": "m", "model": "t"}, lazy=True)
        self.assertFalse(v.elastic)
        self.assertEqual((v.elastic_mib, v.elastic_idle_s), (2048, 30.0))   # 30 s: a restart is cheap now (PLAN-vision-fast-start)


class CheapRestart(unittest.TestCase):
    """After the idle stop the cheap-restart state is kept: the weight files stay hot in the OS file cache for
    elastic_keep_s (keep_hot), the encoded pictures and the work dir survive the restart (nothing re-reads them),
    and the elastic reader starts with --warmup-lazy so the first picture does not wait on the full-size warm-up -
    a reader that does not know the flag yet is retried without it."""

    class Proc:
        """A reader process: answers its stdin lines from a fixed script (READY first)."""

        def __init__(self, out):
            self.stdout = io.StringIO(out)
            self.stdin = io.StringIO()

        def poll(self):
            return None

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 0

    def reader(self, tmp, keep_s=600.0, extra=None):
        cfg = {"exe": "strata-vision", "mmproj": str(Path(tmp) / "mmproj.gguf"),
               "model": str(Path(tmp) / "model.gguf"), "gpu": True, "elastic": True,
               "elastic_keep_s": keep_s}
        if extra:
            cfg.update(extra)
        return Vision(cfg, lazy=True)

    def start_with(self, v, out="READY 1024\n", rec=None):
        def fake_popen(what, args, **kw):
            if rec is not None:
                rec.append(list(args))
            return CheapRestart.Proc(out)
        with mock.patch.object(server, "popen", fake_popen):
            v._start()

    def test_the_elastic_reader_starts_with_warmup_lazy(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = self.reader(tmp)
            rec = []
            self.start_with(v, rec=rec)
            self.addCleanup(v.shutdown)
            self.assertIn("--warmup-lazy", rec[0])
            self.assertTrue(v.alive())

    def test_a_reader_that_does_not_know_the_flag_is_retried_without_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = self.reader(tmp)
            rec = []
            dead = CheapRestart.Proc("")                # an older reader: "unknown argument", exit 2, says nothing
            dead.poll = lambda: 2
            procs = iter([dead, CheapRestart.Proc("READY 1024\n")])

            def fake_popen(what, args, **kw):
                rec.append(list(args))
                return next(procs)

            with mock.patch.object(server, "popen", fake_popen):
                v._start()
            self.addCleanup(v.shutdown)
            self.assertIn("--warmup-lazy", rec[0])
            self.assertNotIn("--warmup-lazy", rec[1])   # the retry: the flag is gone, the encoder runs
            self.assertTrue(v.alive())

    def test_keep_hot_holds_the_weight_files_until_keep_s(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = self.reader(tmp, keep_s=0.5)
            self.start_with(v)
            self.addCleanup(v.shutdown)
            v.unload()                                  # the idle stop happened: the holder runs while it is down
            touched = []
            with mock.patch.object(Vision, "_touch_weights", staticmethod(lambda p: touched.append(p))):
                v.keep_hot()
                deadline = time.time() + 2.0
                while v._hot.is_alive() and time.time() < deadline:
                    time.sleep(0.02)
            self.assertFalse(v._hot.is_alive())         # the holder ends with elastic_keep_s
            self.assertEqual(sorted(set(touched)), sorted([v.weights[0], v.weights[1]]))   # mmproj and model, hot
            v2 = self.reader(tmp, keep_s=0)
            self.assertIsNone(getattr(v2, "_hot", None))
            v2.keep_hot()                               # elastic_keep_s 0: off, no holder at all
            self.assertFalse(hasattr(v2, "_hot"))

    def test_a_restart_encodes_nothing_twice_and_rereads_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = self.reader(tmp)
            png = b"\x89PNG\r\n\x1a\n" + b"0" * 32      # a pass-through format for normalize()
            self.start_with(v, "READY 1024\nOK 4 2 2 1\n")
            self.addCleanup(v.shutdown)
            first = v.encode(png)
            self.assertEqual(first[1], 4)
            d, kept = v.dir, v.cache
            v.unload()
            self.assertFalse(v.alive())
            self.start_with(v, "READY 1024\n")          # the restart: a new process, the same dir and cache
            self.assertEqual(v.dir, d)
            self.assertIs(v.cache, kept)
            self.assertEqual(v.encode(png), first)      # the picture is still cached: no ENC line, no re-load of it
            self.assertIn("READY 1024\n", v.proc.stdout.getvalue())   # (and this process was told nothing else)


if __name__ == "__main__":
    unittest.main()
