"""PLAN-qfn-park-budget, the server's side: the engine's VRAM lines carry how many conversations are parked and what
they may hold right now (the shared pool less the live K/V), /metrics' context shows both, and the engine's notes
about a released conversation or an appended session file become plain lines.  Fake engines only.

    python -m unittest serve.test_park_budget -v
"""
from __future__ import annotations

import unittest

from serve import gpu_map
from serve import qfn_engine as QE
from serve.test_context_panel import CTX, KBT, FakeEngine, service, vram_line


def park_line(dev=0, parked=0, parked_n=0, park_budget=0):
    return vram_line(dev=dev, kv_ram_room=100 * KBT, parked=parked) + f" parked_n={parked_n} park_budget={park_budget}"


class ParkFields(unittest.TestCase):
    def test_parse_and_merge(self):
        r = gpu_map.parse(park_line(parked=3 * KBT, parked_n=2, park_budget=90 * KBT))
        self.assertEqual((r["parked"], r["parked_n"], r["park_budget"]), (3 * KBT, 2, 90 * KBT))
        # the engine's own figures on every card's line: two parts of a split never add them up
        m = gpu_map.merge([gpu_map.parse(park_line(parked=3 * KBT, parked_n=2, park_budget=90 * KBT)),
                           gpu_map.parse(park_line(parked=3 * KBT, parked_n=2, park_budget=90 * KBT))])[0]
        self.assertEqual((m["parked"], m["parked_n"], m["park_budget"]), (3 * KBT, 2, 90 * KBT))

    def test_metrics_context(self):
        rows = [gpu_map.parse(park_line(dev=d, parked=3 * KBT, parked_n=2, park_budget=90 * KBT)) for d in range(4)]
        svc = service(FakeEngine(batch=8, vram=rows))
        ctx = svc._engine_context([])
        self.assertEqual(ctx["parked"], {"bytes": 3 * KBT, "tokens": 3, "conversations": 2, "budget_bytes": 90 * KBT})
        self.assertEqual(ctx["capacity"]["positional"], 8 * CTX)
        svc.metrics()                                    # (the whole page builds with the new fields)

    def test_an_engine_without_the_fields(self):
        svc = service(FakeEngine(batch=8, vram=[gpu_map.parse(vram_line(parked=5 * KBT))]))
        self.assertEqual(svc._engine_context([])["parked"], {"bytes": 5 * KBT, "tokens": 5})
        svc.metrics()

    def test_nothing_parked(self):
        svc = service(FakeEngine(batch=8, vram=[gpu_map.parse(park_line(park_budget=90 * KBT))]))
        self.assertEqual(svc._engine_context([])["parked"], {"bytes": 0, "conversations": 0, "budget_bytes": 90 * KBT})


class Notes(unittest.TestCase):
    def text(self, **f):
        return QE.session_event_text({k: str(v) for k, v in f.items()}, {"files": 3})

    def test_released(self):
        base = dict(sess_event="dropped", sess_tokens=319032, sess_bytes=5242514720, sess_ms=0)
        self.assertEqual(self.text(**base, sess_why="pressure"),
                         "released a 319,032-token conversation parked in RAM (5.24 GB): the host is short of RAM")
        self.assertIn("the live conversations need its room", self.text(**base, sess_why="budget"))
        self.assertIn("idle past the limit", self.text(**base, sess_why="idle"))
        self.assertIn("NOT on disk yet", self.text(**base, sess_why="unsaved"))

    def test_appended(self):
        t = self.text(sess_event="wrote", sess_tokens=149134, sess_bytes=350_000_000, sess_ms=900, sess_why="appended")
        self.assertEqual(t, "saved a 149,134-token conversation to disk: 350 MB appended to its file (0.9 s)")
        whole = self.text(sess_event="wrote", sess_tokens=149134, sess_bytes=2400 << 20, sess_ms=2700, sess_why="background")
        self.assertIn("in the background", whole)

    def test_unknown_event(self):
        self.assertIsNone(self.text(sess_event="something_new", sess_tokens=1, sess_bytes=1, sess_ms=1, sess_why="x"))


if __name__ == "__main__":
    unittest.main()
