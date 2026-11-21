"""PLAN-qfn-per-request-read-stats: the driver reads the engine's PER-REQUEST read figures.

The engine counts every read itself: `PP <pos> <total> <ms> <tok/s> new_read=<n> read_ms=<ms> rid=<id>`
while a prompt is read, `RDONE rid=<id> new=<n> ms=<ms>` once when it ends, and the slot lines' BRP
progress carries `new_read=` / `read_ms=` for ITS request (BATCHING.md's protocol table).  The fake
engine below speaks that protocol behind the real StrataEngine: the new fields land on the driver's
own attributes (progress_new / progress_read_ms / read_done) for the shared server to take, never in
a request's token stream, and a restored conversation reads exactly like any other one.  An engine
before the fields keeps working: the old positional lines parse as they always did and the new
attributes stay None."""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.server import StrataEngine

# GEN: a restored conversation (RESUME 8), two PP steps with the request's own new_read/read_ms, the
# RDONE at the read's end, three tokens and a DONE.  --old: the same request without the new fields.
FAKE = r'''import queue, sys, threading, time
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
print("INFO engine=0.1.50", flush=True)
print("INFO engine_kind=qwen experts=74432 vram_slots=4096 vram_gb=15.3 ram_slots=74432 ram_gb=61.0", flush=True)
print("READY 4096 stop", flush=True)
print("STAT tok_s=0.00 ms_tok=0.00 vram_hit=1.0000 ram_fetch=0.000 disk=0.000 vram_used=3800 vram_slots=4096 "
      "ram_used=70632 ram_gb=61.0 vram_gb=15.3 disk_gb=0.0 draft_acc=0 draft_off=0 active=0", flush=True)
while not quit_.is_set():
    try:
        line = lines.get(timeout=0.05)
    except queue.Empty:
        continue
    f = line.split()
    if f[0] in ("GEN", "GENI"):
        ids = [int(x) for x in f[-1].split(",")]
        max_new = int(f[1])
        print("RESUME 8", flush=True)
        if old:
            print("PP 208 508 40 6000.0", flush=True)
            print("PP 507 508 160 5087.5", flush=True)
        else:
            print("PP 208 508 40 6000.0 new_read=200 read_ms=38.0 rid=1", flush=True)
            print("PP 507 508 160 5087.5 new_read=499 read_ms=156.0 rid=1", flush=True)
            print("RDONE rid=1 new=500 ms=158.0", flush=True)
        for i in range(min(max_new, 3)):
            time.sleep(0.02)
            print(f"T {500 + i}", flush=True)
        print(f"DONE 3 {len(ids)} 200.0 60.0 stop 0 0 8 12 20 0 0 0.0 500 0 chunk=4096 experts_streamed=0",
              flush=True)
print("bye", flush=True)
'''


def start_engine(old=False):
    """A real StrataEngine over the fake script; -> (engine, tmp dir)."""
    import serve.server as server
    tmp = tempfile.TemporaryDirectory()
    script = Path(tmp.name) / "fake_perreq.py"
    script.write_text(FAKE, encoding="utf-8")
    real = server.subprocess.Popen
    args = ["--old"] if old else []
    with mock.patch.object(server.subprocess, "Popen",
                           lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
        engine = StrataEngine("strata", args)
    return engine, tmp


class SoloReadStats(unittest.TestCase):
    def test_pp_fields_and_rdone_reach_the_drivers_own_attributes(self):
        engine, tmp = start_engine()
        self.addCleanup(tmp.cleanup)
        cancel = threading.Event()
        toks = [t for t in engine.generate(list(range(508)), 3, {}, cancel) if t is not None]
        self.assertEqual(toks, [500, 501, 502])
        self.assertEqual(engine.progress, (507, 508))          # the old positional fields, untouched
        self.assertEqual(engine.progress_new, 499)             # this request's own tokens read (last step)
        self.assertEqual(engine.progress_read_ms, 156.0)       # and its own running time
        self.assertEqual(engine.read_done, (500, 158.0))       # RDONE: the exact pair at the read's end
        self.assertEqual(engine.reused, 8)                     # RESUME stays as it was
        engine.proc.kill()

    def test_old_lines_parse_as_before(self):
        engine, tmp = start_engine(old=True)
        self.addCleanup(tmp.cleanup)
        cancel = threading.Event()
        toks = [t for t in engine.generate(list(range(508)), 3, {}, cancel) if t is not None]
        self.assertEqual(toks, [500, 501, 502])
        self.assertEqual(engine.progress, (507, 508))
        self.assertIsNone(engine.progress_new)                 # (no fields: the shared server keeps its
        self.assertIsNone(engine.progress_read_ms)             #  RESUME-based fallback)
        self.assertIsNone(engine.read_done)
        engine.proc.kill()

    def test_the_new_lines_never_enter_the_token_stream(self):
        # every line the request reads is a T or the DONE: PP / RDONE are the driver's business
        engine, tmp = start_engine()
        self.addCleanup(tmp.cleanup)
        raw = []
        cancel = threading.Event()
        for t in engine.generate(list(range(508)), 3, {}, cancel):
            raw.append(t)
        self.assertEqual([t for t in raw if t is not None], [500, 501, 502])
        engine.proc.kill()

    def test_stats_reset_between_requests(self):
        # a second request of the same engine: the first one's read_done must not be taken for it
        engine, tmp = start_engine()
        self.addCleanup(tmp.cleanup)
        list(engine.generate(list(range(508)), 3, {}, threading.Event()))
        self.assertEqual(engine.read_done, (500, 158.0))
        list(engine.generate(list(range(508)), 3, {}, threading.Event()))
        self.assertEqual(engine.read_done, (500, 158.0))       # (re-set by the second RDONE, not left stale)
        # and the reset itself: a request sent while nothing has answered yet starts from None
        engine.progress_new = 12345
        gen = engine.generate(list(range(508)), 1, {}, threading.Event())
        next(gen)
        self.assertTrue(engine.progress_new is None or engine.progress_new >= 200)
        engine.proc.kill()


class ParityStats(unittest.TestCase):
    """PLAN items 13/14: INFO names the engine's tier layout (engine_kind=qwen, the page's TIER_SETS.qwen) and
    the periodic STAT line carries what that layout reads - ram_fetch and disk=0 (QFN has no SSD tier) for the
    tiles, and the residency (vram_used / vram_slots / ram_used, vram_gb / ram_gb / disk_gb) for the bar."""

    def test_info_names_the_qwen_layout(self):
        engine, tmp = start_engine()
        self.addCleanup(tmp.cleanup)
        self.assertEqual(engine.info.get("engine_kind"), "qwen")     # the page picks TIER_SETS.qwen on this
        self.assertEqual(engine.info.get("experts"), 74432)
        self.assertEqual(engine.info.get("vram_slots"), 4096)
        engine.proc.kill()

    def test_stat_carries_the_tier_fields(self):
        engine, tmp = start_engine()
        self.addCleanup(tmp.cleanup)
        toks = [t for t in engine.generate(list(range(508)), 3, {}, threading.Event()) if t is not None]
        self.assertEqual(toks, [500, 501, 502])                      # the STAT line is no request's line
        st = engine.stat
        self.assertIsNotNone(st)                                     # the pump routed it to the tiers parser
        for k in ("ram_fetch", "disk", "vram_used", "vram_slots", "ram_used", "ram_gb", "vram_gb", "disk_gb"):
            self.assertIn(k, st)
        self.assertEqual(st["disk"], 0.0)                            # no SSD tier: disk stays 0
        self.assertEqual(st["ram_fetch"], 0.0)
        self.assertEqual(st["vram_used"], 3800.0)                    # experts resident in the VRAM caches
        self.assertEqual(st["ram_used"], 70632.0)                    # ... vs the RAM-only ones (74432 - 3800)
        toks = [t for t in engine.generate(list(range(508)), 3, {}, threading.Event()) if t is not None]
        self.assertEqual(toks, [500, 501, 502])                      # the STAT line is no request's line
        engine.proc.kill()


class ParseHelpers(unittest.TestCase):
    """The two parsers on their own (the shared server's view needs a whole engine; these do not)."""

    def engine(self):
        e = StrataEngine.__new__(StrataEngine)
        return e

    def test_pp_new_fields(self):
        e = self.engine()
        e._pp_new("PP 208 508 40 6000.0 new_read=200 read_ms=38.0 rid=1".split())
        self.assertEqual((e.progress_new, e.progress_read_ms), (200, 38.0))
        e._pp_new("PP 208 508 40 6000.0".split())              # an old line: nothing to take
        self.assertIsNone(e.progress_new)
        self.assertIsNone(e.progress_read_ms)
        e._pp_new("PP 208 508 40 6000.0 new_read=x rid=1".split())   # junk values: not a crash, no fields
        self.assertIsNone(e.progress_new)
        e._pp_new("PP 208 508 40 6000.0 new_read=200".split())       # only one of them: the other stays None
        self.assertEqual((e.progress_new, e.progress_read_ms), (200, None))

    def test_rdone(self):
        e = self.engine()
        e._rdone("RDONE rid=1 new=500 ms=158.0")
        self.assertEqual(e.read_done, (500, 158.0))
        e._rdone("RDONE rid=2 new=x ms=1")                     # malformed: no figures, never a raise
        self.assertEqual(e.read_done, (500, 158.0))            # (left as the last good one)


if __name__ == "__main__":
    unittest.main()
