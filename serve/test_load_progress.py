"""serve/test_load_progress.py - the load's progress comes from the engine's own load lines, not a timer (P125m
phase 2, part 2).  Each engine has a small table of its load milestones (serve.server.LOAD_MILESTONES); a line the
table does not know keeps the last value.  The parse is tested against RECORDED load logs of each engine
(serve/testdata/load-<engine>.log: qfn, glm and ds4 through maya, ik2, hy4, mm3), the starting service is tested
with a log that grows line by line (/health's pct and step follow it, unknown lines hold the bar), and the page's
loading screen is tested on Node (the percent and the step's words from a starting /health).

    python -m unittest serve.test_load_progress -v
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import LOAD_MILESTONES, LoadState, load_progress, serve_loading  # noqa: E402
from serve.test_dashboard import PAGE_STUBS, ROOT  # noqa: E402  (the Node stubs for the page's modules)
from serve.test_starting import req  # noqa: E402

DATA = ROOT / "serve" / "testdata"


def trace(family: str, lines) -> list:
    """(pct, step) after each line, the way the service walks a load: unknown lines keep the last value."""
    out, pct, step = [], None, ""
    for line in lines:
        pct, step = load_progress(family, line, pct, step)
        out.append((pct, step))
    return out


class RecordedLogs(unittest.TestCase):
    """Every engine's table against its recorded load log: the milestones it must know, monotonic percentages,
    and noise that keeps the last value."""

    def load(self, name):
        text = (DATA / name).read_text(encoding="utf-8")
        return [x for x in text.splitlines() if x.strip()]

    def common(self, family, name):
        lines = self.load(name)
        self.assertTrue(LOAD_MILESTONES[family], f"{family}: its table is empty")
        t = trace(family, lines)
        pcts = [p for p, _ in t if p is not None]
        self.assertEqual(pcts, sorted(pcts), f"{name}: a milestone moved the bar back")
        self.assertTrue(any(p is not None for p in pcts), f"{name}: no milestone matched its own log")
        return lines, t

    def test_glm(self):
        lines, t = self.common("glm", "load-glm.log")
        at = {ln: t[i] for i, ln in enumerate(lines)}
        self.assertEqual(at["PLE on: 78 layers, 288 experts, key pack/full/ple.gguf (rows read direct)"],
                         (12, "setting up the expert fast path"))
        self.assertEqual(at["expert arena: 78 layers x 288 blobs x 6291456 B = 1343.719 GiB"],
                         (18, "making room for the experts in RAM"))
        self.assertEqual(at["WARNING - Windows' commit limit caps the RAM tier at 118.2 GiB"],
                         (18, "making room for the experts in RAM"))     # (noise: the bar holds)
        self.assertEqual(at["model loaded 124.62 GiB at 980 MiB/s (128 s)"], (55, "experts read into RAM"))
        self.assertEqual(at["CUDA0 VRAM before the expert pool: 18.94 GiB free, pool 6.00; state/KV (32768 ctx) 0.66"],
                         (70, "cards ready: the context first"))
        self.assertEqual(at["expert cache 1400 slots, 8.42 GiB of VRAM; policy is least-recent-pick"],
                         (82, "filling the GPU's expert cache"))
        self.assertEqual(at["session is up (engine 0.1.10)"], (95, "almost ready"))

    def test_ds4_through_maya(self):
        lines, t = self.common("deepseek", "load-ds4.log")
        at = {ln: t[i] for i, ln in enumerate(lines)}
        self.assertEqual(at["saguaro ds4: RAM margin 8.0 GiB (DS4_RAM_MARGIN_GB)"], (None, ""))
        self.assertEqual(at["saguaro ds4: loading 61440 experts (regions up to 78.4 MB, blobs up to 61.2 MB)"],
                         (5, "reading the pack's expert list"))
        self.assertEqual(at["saguaro ds4: experts 4096 / 61440 (8.4 GB, 2.45 GB/s)"], (11, "copying the experts into RAM"))
        self.assertEqual(at["saguaro ds4: experts 61440 / 61440 (126.2 GB, 2.58 GB/s)"], (65, "copying the experts into RAM"))
        self.assertEqual(at["saguaro ds4: copying experts 15360 / 61440 (31.5 GB, 3.10 GB/s)"],
                         (69, "pinning the experts for the CPU lane"))
        self.assertEqual(at["saguaro ds4: copying experts 61440 / 61440 (126.2 GB, 3.05 GB/s)"],
                         (82, "pinning the experts for the CPU lane"))
        self.assertEqual(t[-2][0], 88)                         # (CUDA1 layers placed: cards ready)
        self.assertEqual(t[-1], (88, "cards ready: layers placed"))    # (the serve help line: the bar holds)

    def test_qfn(self):
        lines, t = self.common("qfn", "load-qfn.log")
        at = {ln: t[i] for i, ln in enumerate(lines)}
        self.assertEqual(t[0], (2, "the engine starts"))
        self.assertEqual(at["saguaro generate: native pack: 512 experts (largest blob 118.32 MB), token embedding F16"],
                         (10, "shards read: the pack's index"))
        self.assertEqual(at["saguaro generate: expert arena: 78 layers x 512 blobs x 67108864 B = 2491.182 GiB"],
                         (18, "making room for the experts in RAM"))
        self.assertEqual(at["saguaro generate: CUDA0 context ready, 1.94 GiB free before expert arena registration"],
                         (24, "cards ready"))
        self.assertIn("PCIe probe", lines[6])
        self.assertEqual(t[6], (24, "cards ready"))            # (its PCIe probe line holds the bar at 24)
        self.assertEqual(at["saguaro generate: expert arena read through the file cache (3 min 12 s)"],
                         (68, "experts read into RAM"))
        self.assertEqual(at["saguaro generate: expert arena checksum 5a17c0de9b8f1a23 (2491.18 GiB)"],
                         (74, "experts pinned in RAM"))
        self.assertEqual(at["saguaro generate: expert cache 1400 slots, 8.42 GiB of VRAM; policy is least-recent-pick"],
                         (84, "filling the GPU's expert cache"))
        self.assertEqual(at["saguaro generate: session is up (engine 0.1.10)"], (95, "almost ready"))

    def test_ik2(self):
        lines, t = self.common("ik2", "load-ik2.log")
        at = {ln: t[i] for i, ln in enumerate(lines)}
        self.assertEqual(at["saguaro ik2: 24 layers (20 sliding), 128 experts (8 a token), width 2304, vocab 200058; 1 shards"],
                         (8, "shards read"))
        self.assertEqual(at["kv cache: FP16 (1.31 GiB across 48 layer visits)"], (30, "allocating the K/V cache"))
        self.assertEqual(at["experts in pinned (mmap+register) memory: 14.7 GiB; allocating 0.0 s, reading 4.9 s (3.01 GiB/s)"],
                         (60, "experts pinned in RAM"))
        self.assertEqual(at["expert cache, GPU 0: 2088 slots across 2 pools (10.0 GiB); load 7.2 s in all"],
                         (85, "cards ready"))
        self.assertEqual(t[-2], (85, "cards ready"))           # (its second card: still 85, then the bar holds)

    def test_hy4(self):
        lines, t = self.common("hy4", "load-hy4.log")
        at = {ln: t[i] for i, ln in enumerate(lines)}
        self.assertEqual(at["hy4: reading the pack pack/full (18.7 GB): 2048 experts, 60 layers"], (None, ""))
        self.assertEqual(at["saguaro hy4: CPU lane: 6144 experts re-coded at 4 bits (11.9 GB); 0 of 6144 expert runs kept on the GPU"],
                         (55, "recoding the CPU lane's experts"))
        self.assertEqual(t[2], (55, "recoding the CPU lane's experts"))    # (its GPU-fetch line: the bar holds)
        self.assertEqual(at["saguaro hy4: measuring this PC once, before the first answer: drafting (dfly2: a load and 24 short prompts)"],
                         (70, "measuring this PC once"))
        self.assertEqual(t[-1], (80, "trying the drafters"))

    def test_mm3(self):
        lines, t = self.common("minimax", "load-mm3.log")
        at = {ln: t[i] for i, ln in enumerate(lines)}
        self.assertEqual(at["saguaro mm3: NOTE: mbind to NUMA node 1 failed: the pool's pages go where they are touched"],
                         (None, ""))
        self.assertEqual(at["saguaro mm3: layer 0's router: reading the model's files for NaN/Inf (layers 0-59) ..."],
                         (45, "checking the model's files"))
        self.assertEqual(at["saguaro mm3: CPU lane vs staging (measured): an expert 2.40 ms on the CPU lane (+0.30 ms each more), staging 1.85 ms"],
                         (68, "measuring the CPU lane"))
        self.assertEqual(t[-1], (68, "measuring the CPU lane"))    # (its kv-check line: the bar holds)


class TheParser(unittest.TestCase):
    """The small contract of load_progress itself."""

    def test_an_unknown_family_never_moves_the_bar(self):
        self.assertEqual(load_progress("mock", "anything at all"), (None, ""))
        self.assertEqual(load_progress("mock", "anything", 40, "waiting"), (40, "waiting"))
        self.assertEqual(load_progress("", "expert arena: 3 GiB", None, ""), (None, ""))

    def test_a_known_line_names_its_step_and_a_partial_counter_sits_between(self):
        self.assertEqual(load_progress("glm", "session is up (engine 1)", None, ""), (95, "almost ready"))
        self.assertEqual(load_progress("deepseek", "saguaro ds4: experts 30720 / 61440 (63.1 GB, 2.60 GB/s)",
                                       None, "")[0], 36)
        self.assertEqual(load_progress("deepseek", "saguaro ds4: experts 61440 / 61440 (126.2 GB, 2.58 GB/s)",
                                       None, "")[0], 65)
        self.assertEqual(load_progress("deepseek", "saguaro ds4: experts 0 / 61440 (0.0 GB, 0.00 GB/s)",
                                       None, "")[0], 8)


class ServiceProgress(unittest.TestCase):
    """The starting service: /health's pct and step follow the log as it grows - this is what the loading screen
    and the switch bar show - while the line field keeps the raw last line (phase 1)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "engine.log"
        self.log.write_text("", encoding="utf-8")
        self.state = LoadState("glm-5.3-flash", log_path=self.log, family="glm")
        self.httpd = serve_loading(self.state, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.shutdown)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.tmp.cleanup)

    def health(self):
        return json.loads(req(self.base, "GET", "/health")[2])

    def append(self, *lines):
        with open(self.log, "a", encoding="utf-8") as f:
            f.write("".join(x + "\n" for x in lines))

    def test_the_bar_follows_the_engines_own_lines(self):
        self.assertEqual((self.health()["pct"], self.health()["step"]), (None, ""))
        self.append("PLE on: 78 layers, 288 experts, key pack/full/ple.gguf (rows read direct)")
        self.assertEqual((self.health()["pct"], self.health()["step"]), (12, "setting up the expert fast path"))
        self.append("expert arena: 78 layers x 288 blobs x 6291456 B = 1343.719 GiB")
        h = self.health()
        self.assertEqual((h["pct"], h["step"]), (18, "making room for the experts in RAM"))
        self.assertEqual(h["line"], "expert arena: 78 layers x 288 blobs x 6291456 B = 1343.719 GiB")
        self.append("WARNING - the OS may swap", "model loaded 124.62 GiB at 980 MiB/s (128 s)")
        self.assertEqual((self.health()["pct"], self.health()["step"]), (55, "experts read into RAM"))
        self.append("expert cache 1400 slots, 8.42 GiB of VRAM; policy is least-recent-pick")
        self.append("session is up (engine 0.1.10)")
        self.assertEqual((self.health()["pct"], self.health()["step"]), (95, "almost ready"))

    def test_an_unknown_line_keeps_the_last_value(self):
        self.append("expert arena: 78 layers x 288 blobs x 6291456 B = 1343.719 GiB")
        for _ in range(3):
            self.append("saguaro: humming to itself")
        h = self.health()
        self.assertEqual((h["pct"], h["step"]), (18, "making room for the experts in RAM"))

    def test_only_this_starts_part_of_the_log_moves_the_bar(self):
        # a start whose load later fails re-reads nothing of the START BEFORE (the log keeps the earlier ones)
        with open(self.log, "w", encoding="utf-8") as f:
            f.write("session is up (engine 0.1.10)\n")          # an earlier start, already ready
        state = LoadState("glm-5.3-flash", log_path=self.log, family="glm")
        self.assertEqual((state.health()["pct"], state.health()["step"]), (None, ""))
        with open(self.log, "a", encoding="utf-8") as f:
            f.write("PLE on: 78 layers\n")
        self.assertEqual(state.health()["pct"], 12)


@unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
class PageProgress(unittest.TestCase):
    def node(self, body):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.mjs"
            p.write_text(PAGE_STUBS + body, encoding="utf-8")
            r = subprocess.run(["node", str(p), (ROOT / "serve" / "web").as_uri()],
                               capture_output=True, encoding="utf-8", timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_the_screen_shows_the_percent_and_the_steps_words(self):
        got = self.node("""
const so = await import(web + "/starting.js");
console.log(JSON.stringify({
  loading: so.startingView({status: "loading", model: "glm-5.3-flash", pct: 55, step: "experts read into RAM",
                            elapsed_s: 95, line: "model loaded 124.62 GiB at 980 MiB/s (128 s)"}),
  round: so.startingView({status: "loading", pct: 43.6, step: "x"}),
  over: so.startingView({status: "loading", pct: 120}),
  none: so.startingView({status: "loading"}),
  failed: so.startingView({status: "failed", pct: 77, error: "boom"}),
}));""")
        self.assertEqual((got["loading"]["pct"], got["loading"]["step"], got["loading"]["elapsed"]),
                         (55, "experts read into RAM", "2 min"))
        self.assertEqual(got["loading"]["line"], "model loaded 124.62 GiB at 980 MiB/s (128 s)")
        self.assertEqual(got["round"]["pct"], 44)
        self.assertEqual(got["over"]["pct"], 100)              # (a bad figure never fills the bar wrong)
        self.assertIsNone(got["none"]["pct"])                  # (before the engine's first known line)
        self.assertEqual(got["failed"]["pct"], None)           # (a failed load has no percent: the error shows)


if __name__ == "__main__":
    unittest.main()
