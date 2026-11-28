"""serve/test_load_crash.py - an engine that dies during a load (P125m phase 2, part 3): the standing page shows
its exit code and the last ~20 lines of the engine's own log, and offers a "load again" (start this config over)
or "the last model that worked" choice; the port stays up.  Tested with a fake engine that really exits before it
says READY (serve/fake_multi_engine.py's dying cousin, run through the real StrataEngine so the exit code and log
tail come from a live process), with the starting service's /health over HTTP, with the crash page's own answers,
and on Node (web/starting.js' crash view).

    python -m unittest serve.test_load_crash -v
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock as mock
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import models as models_mod  # noqa: E402  (SWITCH_FILE: the note the buttons leave)
from serve import server as SV  # noqa: E402  (the module under test: its ROOT and helpers get patched)
from serve.server import (EngineDied, EngineStartFailed, LoadState, StrataEngine, engine_exit_words,  # noqa: E402
                          engine_log_tail, serve_loading)
from serve.test_dashboard import PAGE_STUBS, ROOT  # noqa: E402  (the Node stubs for the page's modules)
from serve.test_starting import req  # noqa: E402

NOTE = ROOT / models_mod.LAST_GOOD
SWITCH = ROOT / models_mod.SWITCH_FILE


def dying_engine(d: Path, code: int, lines) -> Path:
    """An executable that writes `lines` to its log and exits `code` before saying READY (StrataEngine starts it as
    it starts a real engine: this is a genuine process death, not a mock)."""
    script = d / "dying_engine.py"
    body = [f"#!{sys.executable}", "import sys, time", "sys.stderr.reconfigure(line_buffering=True)"]
    for ln in lines:
        body.append(f"sys.stderr.write({ln!r} + '\\n')")
    body += ["time.sleep(0.3)", f"import os as _os; _os._exit({code})"]
    script.write_text("\n".join(body) + "\n", encoding="utf-8")
    script.chmod(0o755)
    return script


class StrataEngineCrash(unittest.TestCase):
    """The line engine's own start failure: EngineStartFailed carries the exit code and the engine's last lines."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.log = self.d / "engine.log"
        self.log.write_text("an earlier start's noise\n", encoding="utf-8")
        self.addCleanup(self.tmp.cleanup)

    def test_its_exit_code_and_last_lines_come_off_a_real_process(self):
        eng = dying_engine(self.d, 43, ["saguaro fake: reading 12 shards",
                                        "saguaro fake: experts pinned, cards ready",
                                        "saguaro fake: FATAL: could not allocate 61 GB"])
        with self.assertRaises(EngineStartFailed) as c:
            StrataEngine(str(eng), [], log=str(self.log))
        self.assertEqual(c.exception.exit_code, 43)
        self.assertIn("exited before it was ready", str(c.exception))
        self.assertEqual(c.exception.lines[-1], "saguaro fake: FATAL: could not allocate 61 GB")
        self.assertNotIn("an earlier start's noise", "\n".join(c.exception.lines))   # (only this start's lines)

    def test_a_signal_death_names_the_signal(self):
        script = self.d / "killed.py"
        script.write_text(f"#!{sys.executable}\nimport os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n", encoding="utf-8")
        script.chmod(0o755)
        with self.assertRaises(EngineStartFailed) as c:
            StrataEngine(str(script), [], log=str(self.log))
        self.assertEqual(c.exception.exit_code, -9)
        self.assertIn("SIGKILL", engine_exit_words(c.exception.exit_code))

    def test_engine_log_tail_reads_the_last_lines(self):
        lines = [f"saguaro fake: line {i}" for i in range(50)]
        self.log.write_text("an earlier start's noise\n" + "\n".join(lines) + "\n\n", encoding="utf-8")
        start = len("an earlier start's noise\n".encode())
        self.assertEqual(engine_log_tail(str(self.log), start, 20), lines[-20:])
        self.assertEqual(len(engine_log_tail(str(self.log), 0, 20)), 20)     # (the window never reads the whole log)
        self.assertEqual(engine_log_tail(None), [])
        self.assertEqual(engine_log_tail(str(self.d / "missing.log")), [])


class ExitWords(unittest.TestCase):
    def test_plain_codes_signals_and_windows_statuses(self):
        self.assertEqual(engine_exit_words(None), "")
        self.assertEqual(engine_exit_words(1), "exit code 1")
        self.assertEqual(engine_exit_words(0), "exit code 0")
        if os.name != "nt":
            self.assertIn("SIGSEGV", engine_exit_words(-11))
            self.assertIn("out-of-memory killer", engine_exit_words(-9))     # the OOM kill, in words
        self.assertIn("not found", engine_exit_words(0xC0000135))            # (a DLL missing: Windows' loader, no log line)


class HealthOfACrash(unittest.TestCase):
    """The starting service, standing on the port after the engine died mid-load."""

    def open(self, **kw):
        self.state = LoadState("glm-5.3-flash", **kw)
        self.httpd = serve_loading(self.state, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.state.port = self.httpd.server_address[1]
        self.addCleanup(self.httpd.shutdown)
        self.addCleanup(self.httpd.server_close)

    def crash(self, code=43, lines=("saguaro fake: experts pinned, cards ready",
                                    "saguaro fake: FATAL: could not allocate 61 GB")):
        self.state.fail(EngineStartFailed("the engine exited before it was ready", exit_code=code, lines=list(lines)))

    def test_the_health_of_a_crash_names_the_code_and_the_engines_lines(self):
        self.open()
        self.crash()
        s, _, b = req(self.base, "GET", "/health")
        h = json.loads(b)
        self.assertEqual((s, h["status"], h["crashed"], h["exit_code"], h["exit"]),
                         (503, "failed", True, 43, "exit code 43"))
        self.assertEqual(h["lines"], ["saguaro fake: experts pinned, cards ready",
                                      "saguaro fake: FATAL: could not allocate 61 GB"])
        self.assertNotIn("again", h)                        # (no config of its own: the button would have nothing to retry)

    def test_the_page_stands_on_the_port_with_the_crash(self):
        self.open()
        self.crash()
        s, hd, b = req(self.base, "GET", "/")
        self.assertEqual((s, "text/html" in hd["Content-Type"], b"<!doctype html>" in b[:64].lower()), (200, True, True))

    def test_a_plain_failed_load_is_not_shown_as_a_crash(self):
        self.open()
        self.state.fail(RuntimeError("the model folder is gone"))
        h = json.loads(req(self.base, "GET", "/health")[2])
        self.assertEqual(h["status"], "failed")
        self.assertNotIn("crashed", h)
        self.assertNotIn("exit_code", h)
        self.assertNotIn("lines", h)

    def test_the_batch_drivers_engine_died_is_the_crash_page_too(self):
        # (start_qfn raises the batch driver's EngineDied with .exit_code and .lines attached; the shared server
        #  reads the same two fields off it, so qfn's crash page matches every other engine's)
        self.open()
        e = EngineDied("the engine exited before it was ready")
        e.exit_code, e.lines = -11, ["saguaro generate: CUDA0 context ready", "Segmentation fault"]
        self.state.fail(e)
        h = json.loads(req(self.base, "GET", "/health")[2])
        self.assertEqual((h["crashed"], h["exit_code"], h["lines"]),
                         (True, -11, ["saguaro generate: CUDA0 context ready", "Segmentation fault"]))
        self.assertIn("SIGSEGV", h["exit"])

    def test_the_offers_the_load_again_button_can_only_show_with_a_real_config(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cfg = Path(self.tmp.name) / "saguaro-glm.json"
        cfg.write_text('{"model_name": "glm-5.3-flash"}', encoding="utf-8")
        with mock.patch.dict(os.environ, {"SAGUARO_RESTART_ON_UPDATE": "1"}):
            self.open(config_path=cfg)
            self.crash()
            h = json.loads(req(self.base, "GET", "/health")[2])
        self.assertEqual(h["again"], cfg.name)              # (the page's button: Load saguaro-glm.json again)
        # (a start that cannot restart itself, or under a supervisor: no "load again" button is offered)
        self.open()
        self.crash()
        self.assertNotIn("again", json.loads(req(self.base, "GET", "/health")[2]))


class LoadAgain(unittest.TestCase):
    """POST /api/load-again, and start_again() itself: the same RESTART_EXIT note the last-good button leaves, for
    this very config; refused where it cannot work, and never under a supervisor."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.cfg = self.d / "saguaro-glm.json"
        self.cfg.write_text('{"model_name": "glm-5.3-flash"}', encoding="utf-8")
        self.state = LoadState("glm-5.3-flash", config_path=self.cfg)
        self.state.port = 8095
        self.httpd = serve_loading(self.state, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.state.port = self.httpd.server_address[1]
        self.addCleanup(self.httpd.shutdown)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.tmp.cleanup)
        self.state.fail(EngineStartFailed("the engine exited before it was ready", exit_code=43,
                                          lines=["saguaro fake: FATAL"]))

    def post_again(self):
        return req(self.base, "POST", "/api/load-again", {})

    def test_while_a_model_is_only_loading_there_is_nothing_to_retry(self):
        state = LoadState("glm-5.3-flash")
        httpd = serve_loading(state, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            s, _, b = req(base, "POST", "/api/load-again", {})
            self.assertEqual((s, json.loads(b)["error"]["message"]), (409, "nothing failed to load again"))
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_the_button_needs_the_start_that_can_restart_itself(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SAGUARO_RESTART_ON_UPDATE", None)
            s, _, b = self.post_again()
            self.assertEqual(s, 409)
            self.assertIn("saguaro.sh", json.loads(b)["error"]["message"])

    def test_the_button_refuses_a_supervisor_and_a_vanished_config(self):
        self.state.supervisor = "llama-swap"
        with mock.patch.dict(os.environ, {"SAGUARO_RESTART_ON_UPDATE": "1"}):
            s, _, b = self.post_again()
            self.assertEqual((s, "llama-swap" in json.loads(b)["error"]["message"]), (409, True))
        self.state.supervisor = None
        gone = Path(self.d) / "saguaro-gone.json"
        self.state.config_path = str(gone)                  # (the note's config was deleted since the start failed)
        with mock.patch.dict(os.environ, {"SAGUARO_RESTART_ON_UPDATE": "1"}):
            s, _, b = self.post_again()
            self.assertIn("gone", json.loads(b)["error"]["message"])
        self.state.config_path = None
        with mock.patch.dict(os.environ, {"SAGUARO_RESTART_ON_UPDATE": "1"}):
            self.assertIn("without a config", json.loads(self.post_again()[2])["error"]["message"])

    def test_the_button_notes_this_config_and_ends_the_start(self):
        called = []
        with mock.patch.dict(os.environ, {"SAGUARO_RESTART_ON_UPDATE": "1"}), \
                mock.patch.object(sys.modules["serve.server"], "ROOT", self.d), \
                mock.patch("serve.server._restart_after_answer", lambda delay=0.5: called.append(delay)):
            s, _, b = self.post_again()
            self.assertEqual((s, json.loads(b)), (200, {"starting": self.cfg.name}))
            note = json.loads((self.d / models_mod.SWITCH_FILE).read_text(encoding="utf-8"))
        self.assertEqual((Path(note["config"]).resolve(), note["why"], note["port"]),
                         (self.cfg.resolve(), "load again", self.state.port))
        for _ in range(100):
            if called:
                break
            time.sleep(0.02)
        self.assertEqual(called, [0.5])

    def test_the_button_is_an_api_call_so_the_key_and_origin_still_apply(self):
        st = LoadState("glm-5.3-flash", api_key="sag-key", config_path=self.cfg)
        st.fail(EngineStartFailed("dead", exit_code=43))
        httpd = serve_loading(st, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            self.assertEqual(req(base, "POST", "/api/load-again", {})[0], 401)
            s, _, b = req(base, "POST", "/api/load-again", {}, {"x-api-key": "sag-key",
                                                                "Origin": "http://elsewhere.example"})
            self.assertEqual(s, 403)                        # (the same-origin rule, like the real handler's posts)
            s, _, b = req(base, "POST", "/api/load-again", {}, {"x-api-key": "sag-key"})
            self.assertEqual(s, 409)                        # (no restart env here: it answers WHY, not "unauthorized")
            self.assertIn("saguaro.sh", json.loads(b)["error"]["message"])
        finally:
            httpd.shutdown()
            httpd.server_close()


class RealCrash(unittest.TestCase):
    """The whole path with serve.server.main() in a child: an engine that dies mid-load - the port stays up with
    the crash page (exit code, the engine's lines, the choices), and "load again" ends the start with RESTART_EXIT
    leaving the note for saguaro.py."""

    WRAPPER = '''"""scratch: serve.server.main() with the engine dying mid-load (P125m phase 3 tests)."""
import os
import sys

sys.path.insert(0, os.environ["W20_ROOT"])
import serve.server as S  # noqa: E402

sys.argv = ["server.py", "--engine", "mock", "--config", os.environ["W20_CFG"], "--port", os.environ["W20_PORT"]]


class Lines:                                   # (stand-in for the engine's own last lines; the message ends it)
    pass


def boom(*a, **k):
    raise S.EngineStartFailed(
        "the engine exited before it was ready (see the log)", exit_code=int(os.environ.get("W20_CODE", "137")),
        lines=["saguaro fake: reading 12 shards", "saguaro fake: experts pinned, cards ready",
               "saguaro fake: FATAL: out of memory"])


S.MockEngine = boom
sys.exit(S.main())
'''

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.wrapper = self.d / "wrapper.py"
        self.wrapper.write_text(self.WRAPPER, encoding="utf-8")
        self.procs = []
        for p in (NOTE, SWITCH):
            p.unlink(missing_ok=True)

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
                p.wait(10)
        for p in (NOTE, SWITCH):
            p.unlink(missing_ok=True)
        self.tmp.cleanup()

    def start(self, env=None):
        self.cfg = self.d / "saguaro-crash.json"
        log = self.d / "run.log"
        log.write_text("saguaro fake: FATAL: out of memory\n", encoding="utf-8")
        self.cfg.write_text(json.dumps({"model_name": "Crash Fake 9B", "log": str(log)}), encoding="utf-8")
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        e = dict(os.environ, W20_ROOT=str(ROOT), W20_CFG=str(self.cfg), W20_PORT=str(port))
        e.pop("SAGUARO_SUPERVISOR", None)
        e.pop("SAGUARO_RESTART_ON_UPDATE", None)
        e.update(env or {})
        p = subprocess.Popen([sys.executable, str(self.wrapper)], env=e, cwd=str(ROOT),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs.append(p)
        return p, f"http://127.0.0.1:{port}"

    def wait_health(self, base, want, timeout=30):
        t0, last = time.time(), None
        while time.time() - t0 < timeout:
            try:
                st, _, b = req(base, "GET", "/health")
                last = (st, json.loads(b))
                if want(*last):
                    return last[1]
            except (urllib.error.URLError, ConnectionError, ValueError, TimeoutError):
                pass
            time.sleep(0.1)
        self.fail(f"/health never said what we waited for: {last}")

    def test_a_direct_start_stands_with_the_crash_page(self):
        p, base = self.start()
        h = self.wait_health(base, lambda c, b: b.get("status") == "failed" and b.get("crashed"))
        self.assertEqual((h["exit_code"], h["exit"], h["model"]), (137, "exit code 137", "Crash Fake 9B"))
        self.assertIn("out of memory", "\n".join(h["lines"]))
        self.assertNotIn("again", h)                                  # (not started by saguaro.sh: cannot restart)
        time.sleep(1.0)
        self.assertIsNone(p.poll())                                   # (a direct start stands: the port stays up)
        self.assertEqual(req(base, "GET", "/")[0], 200)               # (the crash page, on the port)
        p.send_signal(2)
        self.assertIn(p.wait(15), (130, 143, 1, -2, -15))

    def test_load_again_restarts_this_very_config(self):
        p, base = self.start({"SAGUARO_RESTART_ON_UPDATE": "1"})
        h = self.wait_health(base, lambda c, b: b.get("crashed"))
        self.assertEqual(h["again"], self.cfg.name)                   # (a start saguaro.py owns: the button is offered)
        s, _, b = req(base, "POST", "/api/load-again", {})
        self.assertEqual((s, json.loads(b)), (200, {"starting": self.cfg.name}))
        self.assertEqual(p.wait(15), 76)                              # (RESTART_EXIT: saguaro.py reads the note)
        note = json.loads(SWITCH.read_text(encoding="utf-8"))
        self.assertEqual((Path(note["config"]).resolve(), note["why"]), (self.cfg.resolve(), "load again"))


@unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
class PageCrash(unittest.TestCase):
    def node(self, body):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.mjs"
            p.write_text(PAGE_STUBS + body, encoding="utf-8")
            r = subprocess.run(["node", str(p), (ROOT / "serve" / "web").resolve().as_uri()],
                               capture_output=True, encoding="utf-8", timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_the_crash_view_takes_the_code_the_lines_and_the_two_choices(self):
        got = self.node("""
const so = await import(web + "/starting.js");
console.log(JSON.stringify({
  crash: so.startingView({status: "failed", crashed: true, model: "glm-5.3-flash", exit_code: 137,
                          exit: "exit code 137", lines: ["a: shards read", "a: FATAL out of memory"],
                          again: "saguaro-glm.json", last_good: {config: "saguaro-ik2.json", time: 1}}),
  noagain: so.startingView({status: "failed", crashed: true, exit: "killed by signal 11 (SIGSEGV)",
                            lines: ["x"], error: "the engine exited before it was ready"}),
  plain: so.startingView({status: "failed", error: "the model folder is gone"}),
  loading: so.startingView({status: "loading", pct: 30}),
}));""")
        self.assertEqual((got["crash"]["crashed"], got["crash"]["exit"], got["crash"]["exitCode"]),
                         (True, "exit code 137", 137))
        self.assertEqual(got["crash"]["lines"], ["a: shards read", "a: FATAL out of memory"])
        self.assertEqual((got["crash"]["again"], got["crash"]["lastGood"]),
                         ("saguaro-glm.json", {"config": "saguaro-ik2.json", "time": 1}))
        self.assertTrue(got["noagain"]["crashed"])
        self.assertIsNone(got["noagain"]["again"])           # (no config of its own: no "load again" button)
        self.assertEqual(got["noagain"]["exit"], "killed by signal 11 (SIGSEGV)")
        self.assertEqual((got["plain"]["failed"], got["plain"]["crashed"], got["plain"]["lines"]), (True, False, []))
        self.assertFalse(got["loading"]["crashed"])

    def test_the_crash_page_offers_load_again_and_the_last_config(self):
        got = self.node("""
const so = await import(web + "/starting.js");
const calls = [];
globalThis.fetch = async (u) => { calls.push(u); return {ok: true, json: async () => ({starting: "saguaro-glm.json"})}; };
st.S.starting = {status: "failed", crashed: true, exit: "exit code 137", lines: ["x"], again: "saguaro-glm.json",
                 last_good: {config: "saguaro-ik2.json", time: 1}};
await so.loadAgain();
console.log(JSON.stringify({calls, restarting: st.S.starting.restarting}));""")
        self.assertEqual(got["calls"], ["api/load-again"])
        self.assertEqual(got["restarting"], "saguaro-glm.json")


if __name__ == "__main__":
    unittest.main()
