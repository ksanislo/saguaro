"""serve/test_lost_read.py - P124b: the log line of a resumed low prompt read carries the engine's REAL reused
figure.  The old line claimed "the low prompt read continues from where the engine kept it" whatever the engine
had done with the half-read prompt: mortar's switch to another conversation quietly dropped a read stopped inside
a batched part, and the 16,384 tokens the line said it continued from had been read again from 0 (ref/
mortar-paused-read-20261125.log: "prompt 105603 tokens = 0 reused + 16384 ... (cancelled)", then "0 reused +
105603 read").  The engine reports its own figure on the resumed call (QFN's RESUME line, the same one its DONE
carries, as `engine.reused`), and the line waits for it: `N of M tokens reused` - N being what the engine really
reused, 0 when it lost the read, and no figure at all (the old wording) for an engine that reports none.

    python -m unittest serve.test_lost_read -v
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import ByteTokenizer, PRIORITY_LOW, PRIORITY_HIGH, Service  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.test_fair import ROOT  # noqa: E402
from serve.test_priority import LONG, SHORT, Env, LostRead, ReadEngine, UnreportingRead, ask  # noqa: E402


class ContinueLine(Env):
    def setUp(self):
        super().setUp()
        self.tok = ByteTokenizer()

    def race_read(self, engine):
        """The low one reads its long prompt, a high request cuts it once, it resumes: -> (answers, the log)."""
        svc = Service(engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        out, log = {}, io.StringIO()
        with contextlib.redirect_stdout(log):
            a = threading.Thread(target=ask, args=(svc, "long one", 4096, out, PRIORITY_LOW))
            a.start()
            while sum(engine.ticks.values()) < 2:
                time.sleep(0.001)                       # it is in the middle of reading its prompt
            b = threading.Thread(target=ask, args=(svc, "short one", 4096, out, PRIORITY_HIGH))
            b.start()
            a.join(30)
            b.join(30)
        return out, log.getvalue()


class TheContinueLine(ContinueLine):
    def test_the_line_carries_the_engines_reused_figure(self):
        engine = ReadEngine(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        out, log = self.race_read(engine)
        self.assertEqual(out["long one"][0], LONG)      # the read went on, not over
        paused = re.search(r"a low prompt read paused at ([\d,]+) of ([\d,]+) prompt tokens", log)
        cont = re.search(r"the low prompt read continued: ([\d,]+) of ([\d,]+) tokens reused", log)
        self.assertIsNotNone(paused)
        self.assertIsNotNone(cont)                      # P124b: the engine's own figure, not the tick's
        n = lambda s: int(s.replace(",", ""))           # noqa: E731
        self.assertGreater(n(cont.group(1)), 0)         # what the engine kept really was reused
        self.assertEqual(n(cont.group(1)), n(paused.group(1)))    # the tokens read before the pause, ...
        self.assertEqual(n(cont.group(2)), n(paused.group(2)))    # ... over the same prompt

    def test_a_read_the_engine_lost_logs_zero_reused_never_a_claim(self):
        engine = LostRead(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        out, log = self.race_read(engine)               # the server cuts it (the engine says #1620) ...
        self.assertEqual(out["long one"][0], LONG)      # ... and the prompt is read again from 0
        cont = re.search(r"the low prompt read continued: ([\d,]+) of ([\d,]+) tokens reused", log)
        self.assertIsNotNone(cont)
        self.assertEqual(cont.group(1).replace(",", ""), "0")     # the engine's figure: nothing was reused
        self.assertNotIn("continues from where the engine kept it", log)   # the untested claim is gone

    def test_an_engine_without_a_figure_keeps_the_numberless_wording(self):
        engine = UnreportingRead(self.tok, {"long": LONG, "short": SHORT}, delay_s=0.004)
        out, log = self.race_read(engine)
        self.assertEqual(out["long one"][0], LONG)
        self.assertIn("priority: the low prompt read continues from where the engine kept it", log)
        self.assertNotIn("tokens reused", log)          # nothing over the pipe: nothing printed as a figure


if __name__ == "__main__":
    unittest.main()
