"""serve/test_parity_output.py - the lines PLAN-parity-output added to the maya engines, as the two drivers read them
(serve/server.py StrataEngine): the GLM multi RESUME line (rid-tagged, before a restored read) must be taken - routed
to its request when one is live, dropped otherwise - and never left on the shared queue nothing reads under multi;
the loop-guard NOTE carrying `r<rid>` is still printed engine-wide; DS4's new STAT residency keys parse; and the solo
reader of the new-shape read lines (RESUME, read-only PP ms, the 10-field DONE) parses them as before.

    python -m unittest serve.test_parity_output -v
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import StrataEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class MultiLineRouting(unittest.TestCase):
    def setUp(self):
        self.engine = StrataEngine(str(ROOT / "serve/fake_multi_engine.py"), ["3"],
                                   env=dict(os.environ, FAKE_PARITY_LINES="1"))

    def tearDown(self):
        self.engine.close()

    def test_a_request_survives_the_new_lines(self):
        ids = list(range(20))
        toks = [t for t in self.engine.generate(ids, 5, {}, threading.Event()) if t is not None]
        self.assertEqual(toks, [1000 + i for i in ids][:5])
        time.sleep(0.2)
        self.assertEqual(self.engine.lines.qsize(), 0)   # (the RESUME / NOTE lines were taken, not queued to nobody)

    def test_route_takes_the_resume_line(self):
        self.assertTrue(self.engine._route("RESUME 7 120"))        # no live rid: taken and dropped
        self.assertEqual(self.engine.lines.qsize(), 0)
        q: queue.Queue = queue.Queue()
        with self.engine._routes_lock:
            self.engine._routes[7] = q
        try:
            self.assertTrue(self.engine._route("RESUME 7 120"))    # its own: the read began (a heard line)
            self.assertEqual(q.get_nowait(), "RESUME 7 120")
        finally:
            with self.engine._routes_lock:
                del self.engine._routes[7]

    def test_route_prints_the_rid_tagged_note(self):
        self.assertTrue(self.engine._route(
            "NOTE r3 the answer was repeating a 2-token pattern (40 tokens in): the turn was ended there"))
        self.assertEqual(self.engine.lines.qsize(), 0)

    def test_stat_carries_the_residency_slots(self):   # (deepseek's new STAT keys: parsed like every other key)
        self.assertTrue(self.engine._route(
            "STAT tok_s=9.1 ms_tok=1.1 vram_used=640 vram_slots=896 ram_used=1536 ram_slots=1536 cpu_n=3200 "
            "vram_gb=9.4 ram_gb=21.0 disk_gb=3.2"))
        self.assertEqual(self.engine.stat["vram_slots"], 896)
        self.assertEqual(self.engine.stat["ram_slots"], 1536)


class SoloReader(unittest.TestCase):
    def setUp(self):
        self.engine = StrataEngine(str(ROOT / "serve/fake_solo_parity_engine.py"), [])

    def tearDown(self):
        self.engine.close()

    def test_generate_and_progress_with_the_resume_line(self):
        ids = list(range(30))
        toks = [t for t in self.engine.generate(ids, 3, {}, threading.Event()) if t is not None]
        self.assertEqual(toks, [1000, 1001, 1002])
        self.assertEqual(self.engine.progress[:3], (30, 30, 4))    # (read, total, reuse: the PP line's own fields)
        self.assertEqual(self.engine.last["reused"], 4)
        self.assertEqual(self.engine.last["drafts_accepted"], 2)
        self.assertEqual(self.engine.last["hits"], 10)
        self.assertEqual(self.engine.info["vram_slots"], 48)       # (INFO's residency capacities)
        self.assertEqual(self.engine.info["ram_slots"], 96)


if __name__ == "__main__":
    unittest.main()
