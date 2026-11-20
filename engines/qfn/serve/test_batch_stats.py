"""PLAN-qfn-batch-stats: the figures the batch path tells the Monitor and the request rows.

The engine prints a STAT line while its batch windows run (at most once a second, the maya engines' key=value
format), and every BDONE carries the slot's own "<hits> <lookups> <drafts accepted> <drafts offered>".  The fake
engine below speaks that protocol behind the real StrataEngine: the STAT lines reach the driver's tiers parser
(self.stat / stat_history) and never a request's token stream, and the BDONE tail lands in the request's figures
(the rows' hit_rate and drafts).  The tail is optional: an older engine's five-field line parses as before.
"""
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.server import StrataEngine

# The fake batch engine: INFO/READY, a STAT line at the start, BGEN answered with the admission (T, DONE, BADM 1)
# and three window tokens - a STAT line and a BT each - then BDONE with the slot's stats tail (40 hits / 50 lookups,
# 3 drafts kept of 6).  --old: BDONE without the tail (an engine before the fields).
FAKE_BATCH_STATS = r'''import queue, sys, threading, time
old = "--old" in sys.argv
lines, quit_ = queue.Queue(), threading.Event()
def reader():
    for l in sys.stdin:
        l = l.strip()
        if l == "QUIT":
            quit_.set()
        elif l:
            lines.put(l)
    lines.put(None)
threading.Thread(target=reader, daemon=True).start()
print("INFO engine=0.1.50 batch_slots=2", flush=True)
print("READY 4096 stop", flush=True)
print("STAT tok_s=0.00 ms_tok=0.00 vram_hit=0.5000 draft_acc=0 draft_off=0 active=0", flush=True)
while not quit_.is_set():
    try:
        line = lines.get(timeout=0.05)
    except queue.Empty:
        continue
    f = line.split()
    if f[0] == "BGEN":
        slot, max_new = int(f[1]), int(f[2])
        ids = [int(x) for x in f[-1].split(",")]
        toks = [(i + 1000) % 256 for i in ids[:max_new]]
        print(f"T {toks[0]}", flush=True)
        print(f"DONE 1 {len(ids)} 2.0 1.0 length 1 2 0 5 9", flush=True)
        print(f"BADM {slot} 1", flush=True)
        for k, t in enumerate(toks[1:4]):
            time.sleep(0.05)
            print(f"STAT tok_s=12.00 ms_tok=83.33 vram_hit=0.9100 draft_acc={k + 1} draft_off={2 * (k + 1)} active=1",
                  flush=True)
            print(f"BT {slot} {t}", flush=True)
        tail = "" if old else " 40 50 3 6"
        print(f"BDONE {slot} {1 + min(3, len(toks) - 1)} stop 123.4{tail}", flush=True)
print("bye", flush=True)
'''


def start_engine(old=False):
    """A real StrataEngine over the fake batch script; -> (engine, tmp dir)."""
    import serve.server as server
    tmp = tempfile.TemporaryDirectory()
    script = Path(tmp.name) / "fake_strata.py"
    script.write_text(FAKE_BATCH_STATS, encoding="utf-8")
    real = server.subprocess.Popen
    args = ["--batch", "2"] + (["--old"] if old else [])
    with mock.patch.object(server.subprocess, "Popen",
                           lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
        engine = StrataEngine("strata", args)
    return engine, tmp


def run_bgen(engine, ids, max_new):
    """generate_batched as a request thread would (one slot held busy so the request is admitted to a slot)."""
    with engine.slot_cv:
        engine.slot_busy[1] = True          # not alone: the request goes to a batch slot, not the solo path
    cancel = threading.Event()
    return list(engine.generate_batched(ids, max_new, {}, cancel))


class ApplyBDone(unittest.TestCase):
    def engine(self):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.last = {"finish": "length", "hits": 5, "lookups": 9}
        return e

    def test_the_stats_tail_lands_in_the_request_figures(self):
        e = self.engine()
        e._apply_bdone("BDONE 0 4 stop 123.4 40 50 3 6".split())
        self.assertEqual(e.last["finish"], "stop")
        self.assertEqual(e.last["generated"], 4)
        self.assertEqual(e.last["decode_ms"], 123.4)
        self.assertEqual((e.last["hits"], e.last["lookups"]), (40, 50))
        self.assertEqual((e.last["drafts_accepted"], e.last["drafts_offered"]), (3, 6))

    def test_an_old_five_field_line_keeps_the_admissions_figures(self):
        e = self.engine()
        e._apply_bdone("BDONE 0 4 stop 123.4".split())
        self.assertEqual(e.last["finish"], "stop")
        self.assertEqual((e.last["hits"], e.last["lookups"]), (5, 9))   # the DONE's, untouched
        self.assertNotIn("drafts_offered", e.last)

    def test_a_malformed_tail_is_ignored(self):
        e = self.engine()
        e._apply_bdone("BDONE 0 4 stop 123.4 x y z w".split())
        self.assertEqual(e.last["finish"], "stop")
        self.assertEqual((e.last["hits"], e.last["lookups"]), (5, 9))


class ParseStat(unittest.TestCase):
    def test_stat_and_history(self):
        e = StrataEngine("missing-executable", [], lazy=True)
        self.assertIsNone(e.stat)
        e._parse_stat("STAT tok_s=31.20 ms_tok=32.05 vram_hit=0.9100 draft_acc=4 draft_off=9 mode=x")
        self.assertEqual(e.stat["tok_s"], 31.2)
        self.assertEqual(e.stat["ms_tok"], 32.05)
        self.assertEqual(e.stat["vram_hit"], 0.91)
        self.assertEqual((e.stat["draft_acc"], e.stat["draft_off"]), (4.0, 9.0))
        self.assertEqual(e.stat["mode"], "x")
        self.assertIn("time", e.stat)
        e._parse_stat("STAT vram_hit=0.5")
        self.assertEqual(len(e.stat_history), 2)
        self.assertEqual(e.stat_history[0]["vram_hit"], 0.91)


class FakeBatchEngine(unittest.TestCase):
    def setUp(self):
        self.engine, self.tmp = start_engine()

    def tearDown(self):
        self.engine.unload()
        self.tmp.cleanup()

    def test_the_start_stat_line_reaches_the_tiers_not_a_request(self):
        deadline = time.monotonic() + 10
        while self.engine.stat is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNotNone(self.engine.stat)
        self.assertEqual(self.engine.stat["vram_hit"], 0.5)
        self.assertEqual(list(self.engine.lines.queue), [])     # a STAT is never a line a request reads

    def test_a_batch_request_carries_the_slots_figures(self):
        toks = run_bgen(self.engine, [65, 66, 67, 68, 69], 4)
        self.assertEqual(toks, [41, 42, 43, 44])                # only tokens streamed (no STAT leaked in)
        last = self.engine.last
        self.assertEqual(last["finish"], "stop")
        self.assertEqual(last["generated"], 4)
        self.assertEqual((last["hits"], last["lookups"]), (40, 50))
        self.assertEqual((last["drafts_accepted"], last["drafts_offered"]), (3, 6))
        self.assertEqual(self.engine.stat["vram_hit"], 0.91)    # the last window's STAT line
        self.assertEqual(self.engine.stat["draft_acc"], 3.0)
        self.assertGreaterEqual(len(self.engine.stat_history), 3)


class FakeBatchEngineOld(unittest.TestCase):
    """An engine whose BDONE has no stats tail: everything still works, and the rows keep the admission's figures."""

    def setUp(self):
        self.engine, self.tmp = start_engine(old=True)

    def tearDown(self):
        self.engine.unload()
        self.tmp.cleanup()

    def test_a_batch_request_works_without_the_tail(self):
        toks = run_bgen(self.engine, [65, 66, 67, 68, 69], 4)
        self.assertEqual(toks, [41, 42, 43, 44])
        last = self.engine.last
        self.assertEqual(last["finish"], "stop")
        self.assertEqual(last["generated"], 4)                      # BDONE's count still replaces the DONE's
        self.assertEqual((last["hits"], last["lookups"]), (5, 9))   # the admission DONE's, untouched
        self.assertEqual((last["drafts_accepted"], last["drafts_offered"]), (1, 2))   # the DONE's as well


if __name__ == "__main__":
    unittest.main()
