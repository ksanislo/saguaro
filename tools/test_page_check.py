"""tools/test_page_check.py - P124d's page check: the CONSOLE-line parser on fixed chromium logs (both formats it
writes, the new 'CONSOLE:1' and the old 'CONSOLE(34)', plus its own WARNING noise which must not count), the
decision on fixed inputs (the shell, the loading screen, a console error on either, nothing drawn, and the class
names that sit INSIDE .start but are not it), and the real check against a local `python -m serve.server --engine
mock` on a free port - run only when this PC has a chromium (skipped otherwise, the tool's own SKIP case).

    python -m unittest tools.test_page_check -v
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import page_check as PC  # noqa: E402

# what chromium 156 writes for a page's console.error and an uncaught throw (measured against a real page)
LINE_NEW = ('[2030466:2030466:1126/004139.233634:INFO:CONSOLE:1] "manual error here", '
            'source: file:///home/ken/scratch/broken.html (1)')
LINE_UNCAUGHT = ('[2030466:2030466:1126/004139.260018:INFO:CONSOLE:1] "Uncaught TypeError: Cannot read properties '
                 'of null (reading \'x\')", source: file:///home/ken/scratch/broken.html (1)')
LINE_OLD = ('[0516/101716.835173:INFO:CONSOLE(34)] "Uncaught ReferenceError: startingView is not defined", '
            'source: http://127.0.0.1:8095/web/app.js (34)')
CHROMIUM_NOISE = ("[2021428:2021428:1126/004029.592905:WARNING:sandbox/policy/linux/sandbox_linux.cc:405] "
                  "InitializeSandbox() called with multiple threads in process gpu-process.\n"
                  "[2021381:2021381:1126/004030.093675:WARNING:ui/base/idle/idle_linux.cc:111] None of the known "
                  "D-Bus ScreenSaver services could be used.")

DOM_SHELL = '<!doctype html><html><body><div id="app"><div class="shell shell--closed"><header class="chat-head">' \
            '</header><button class="btn btn--restart">again</button></div></div></body></html>'
DOM_START = '<!doctype html><html><body><div id="app"><div class="start"><div class="start__card">' \
            '<div class="start__spin"></div><h1 class="start__title">Loading the model</h1></div></div>' \
            '</div></body></html>'
DOM_NOTHING = '<!doctype html><html lang="en"><head></head><body><div id="app"></div>' \
              '<noscript><p>Saguaro\'s page needs JavaScript.</p></noscript></body></html>'


class Parser(unittest.TestCase):
    def test_console_lines_both_formats(self):
        self.assertEqual(PC.console_errors(LINE_NEW), ["manual error here"])
        self.assertEqual(PC.console_errors(LINE_OLD), ["Uncaught ReferenceError: startingView is not defined"])

    def test_many_lines_in_order(self):
        log = "\n".join([CHROMIUM_NOISE, LINE_UNCAUGHT, LINE_NEW])
        self.assertEqual(PC.console_errors(log),
                         ["Uncaught TypeError: Cannot read properties of null (reading 'x')", "manual error here"])

    def test_chromium_own_lines_are_not_errors(self):
        # the W20 lesson: a dead page logs nothing else, a healthy one logs nothing at all - and chromium's own
        # startup WARNINGs are not console messages
        self.assertEqual(PC.console_errors(CHROMIUM_NOISE), [])
        self.assertEqual(PC.console_errors(""), [])

    def test_escaped_quotes_decode(self):
        line = ('[1:1:1126/000000.000000:INFO:CONSOLE:2] "a \\"quoted\\" thing\\nand a newline", source: '
                'http://127.0.0.1:8095/ (2)')
        self.assertEqual(PC.console_errors(line), ['a "quoted" thing\nand a newline'])


class Decision(unittest.TestCase):
    def test_shell_and_loading_screen_pass(self):
        ok, reason = PC.decide([], DOM_SHELL)
        self.assertTrue(ok)
        self.assertIn("shell", reason)
        ok, reason = PC.decide([], DOM_START)
        self.assertTrue(ok)
        self.assertIn("loading", reason)

    def test_class_names_inside_start_are_not_start(self):
        # start__card is the CARD INSIDE the loading screen; without the screen itself nothing rendered
        self.assertEqual(PC.rendered('<div class="start__card start__logo">'), "")
        self.assertEqual(PC.rendered(DOM_START), "start")
        self.assertEqual(PC.rendered(DOM_SHELL), "shell")          # shell--closed counts as the shell
        self.assertEqual(PC.rendered(DOM_NOTHING), "")

    def test_nothing_drawn_fails(self):
        ok, reason = PC.decide([], DOM_NOTHING)
        self.assertFalse(ok)
        self.assertIn("neither", reason)
        self.assertIn("30 s", reason)

    def test_console_error_fails_whatever_drew(self):
        for dom in (DOM_SHELL, DOM_START, DOM_NOTHING):
            ok, reason = PC.decide(["Uncaught TypeError: x"], dom)
            self.assertFalse(ok)
            self.assertIn("Uncaught TypeError: x", reason)
        ok, reason = PC.decide(["one", "two"], DOM_SHELL)
        self.assertIn("2 console error(s)", reason)


@unittest.skipUnless(PC.find_browser(), "no chromium on this PC (the tool's own SKIP case)")
class RealPage(unittest.TestCase):
    """the tool end to end: the page a local mock engine serves really draws, and the check says PASS -
    the exact first paint that killed 1.2.2 (startingView(null)) would now be a FAIL."""

    proc = None

    @classmethod
    def setUpClass(cls):
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            cls.port = s.getsockname()[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.log = tempfile.NamedTemporaryFile(prefix="page-check-server-", suffix=".log", delete=False)
        cls.proc = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "mock",
                                     "--port", str(cls.port)], cwd=ROOT, stdout=cls.log, stderr=subprocess.STDOUT)
        deadline = time.time() + 30.0
        while time.time() < deadline:
            try:                                  # any 200 from /health: the port answers (loading or loaded)
                with urllib.request.urlopen(cls.base + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.HTTPError, urllib.error.URLError, OSError):
                pass
            if cls.proc.poll() is not None:
                raise RuntimeError("the mock server died: " + Path(cls.log.name).read_text()[-2000:])
            time.sleep(0.3)
        raise RuntimeError("the mock server never answered /health")

    @classmethod
    def tearDownClass(cls):
        if cls.proc and cls.proc.poll() is None:
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
        cls.log.close()
        Path(cls.log.name).unlink(missing_ok=True)

    def test_the_page_load_passes(self):
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "page_check.py"), self.base],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(r.stdout.startswith("PASS"), r.stdout)
        self.assertIn("the page drew", r.stdout)


if __name__ == "__main__":
    unittest.main()
