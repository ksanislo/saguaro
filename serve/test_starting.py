"""serve/test_starting.py - the page on the port before the model (issue #20 phase 1): the starting service - the
page's files, /health {"status": "loading"}, the 503 {"error": "the model is loading"} + Retry-After: 5 on every
engine path, the API key still where it was -; the hand-over of the same socket to the real service with no gap;
a load that fails (the standing failed page, the exit under a supervisor or --exit-on-fail, and the page's
"start with the last config that worked"); the start order against a slow engine through the real main(); and the
page's own starting screen (Node, as test_dashboard's page-logic tests).  The slow and the dead engine are the
mock one made slow or dead by a wrapper that runs serve.server.main() in a child process (no GPU, no pack).

    python -m unittest serve.test_starting -v
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock as mock
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import models as models_mod  # noqa: E402  (SWITCH_FILE, LAST_GOOD: the notes on disk)
from serve import server as SV  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, LoadState, MockEngine, Service, serve_loading, take_over  # noqa: E402
from serve.test_dashboard import PAGE_STUBS, ROOT  # noqa: E402  (the Node stubs for the page's modules)

NOTE = ROOT / models_mod.LAST_GOOD      # what loaded last; a real start records it in the repo's state folder
SWITCH = ROOT / models_mod.SWITCH_FILE  # and the failed page's button leaves this note for saguaro.py


def req(base, method, path, body=None, headers=None, timeout=5):
    h = {"Content-Type": "application/json"} if body is not None else {}
    h.update(headers or {})
    r = urllib.request.Request(base + path, method=method, headers=h,
                               data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            with resp:
                return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        with e:
            return e.code, dict(e.headers), e.read()


def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Loading(unittest.TestCase):
    """The starting service itself, on a socket of its own (no engine anywhere near it)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpd = Path(self.tmp.name)
        self.log = self.tmpd / "engine.log"
        self.log.write_text("some earlier start\nw20: filling the experts' slots 1200/2000   \n", encoding="utf-8")
        self.state = LoadState("glm-5.3-flash", log_path=self.log)
        self.httpd = serve_loading(self.state, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.state.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def test_the_page_and_its_starting_screen_answer(self):
        s, h, b = req(self.base, "GET", "/")
        self.assertEqual(s, 200)
        self.assertIn("text/html", h["Content-Type"])
        self.assertIn(b"<!doctype html>", b[:64].lower())
        self.assertIn("web/starting.css", b.decode())          # (the index links the screen's own file)
        for name, kind in (("web/starting.css", "text/css"), ("web/starting.js", "text/javascript")):
            s, h, b = req(self.base, "GET", "/" + name)
            self.assertEqual((s, kind in h["Content-Type"], b.startswith(b"/*") or b.startswith(b"//")),
                             (200, True, True), name)
        self.assertIn(b"start__spin", req(self.base, "GET", "/web/starting.css")[2])
        self.assertEqual(req(self.base, "GET", "/web/nope.js")[0], 404)

    def test_health_while_loading(self):
        s, _, b = req(self.base, "GET", "/health")
        h = json.loads(b)
        self.assertEqual(s, 503)                               # (REQUIRED: a 200 here would tell a supervisor "ready")
        self.assertEqual(h["status"], "loading")
        self.assertEqual(h["model"], "glm-5.3-flash")
        self.assertIsInstance(h["elapsed_s"], int)
        self.assertEqual(h["line"], "w20: filling the experts' slots 1200/2000")   # (the log's last line, trimmed)
        self.assertIsNone(h["supervisor"])
        self.assertNotIn("error", h)

    def test_every_engine_path_says_try_again_in_five(self):
        for method, path in (("POST", "/v1/chat/completions"), ("POST", "/v1/messages"), ("GET", "/api/tags"),
                             ("GET", "/metrics"), ("DELETE", "/v1/responses/x"), ("OPTIONS", "/v1/chat/completions")):
            t0 = time.time()
            s, h, b = req(self.base, method, path, {} if method == "POST" else None)
            self.assertEqual((method, s, h.get("Retry-After"), json.loads(b)),
                             (method, 503, "5", {"error": "the model is loading"}))
            self.assertLess(time.time() - t0, 2, f"{method} {path} waited unbounded")   # (requests may not queue up)

    def test_the_api_key_still_guards_what_it_guarded(self):
        st = LoadState("glm-5.3-flash", api_key="sag-key, other")
        httpd = serve_loading(st, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            self.assertEqual(req(base, "GET", "/health")[0], 503)             # (open, as today: supervisors read it)
            self.assertEqual(req(base, "GET", "/")[0], 200)                   # (open, as today: the page)
            self.assertEqual(req(base, "POST", "/v1/chat/completions", {})[0], 401)
            self.assertEqual(req(base, "GET", "/metrics")[0], 401)
            self.assertEqual(req(base, "GET", "/metrics", headers={"Authorization": "***"})[0], 401)
            s, _, b = req(base, "GET", "/metrics", headers={"x-api-key": "sag-key"})
            self.assertEqual((s, json.loads(b)), (503, {"error": "the model is loading"}))
            self.assertEqual(req(base, "POST", "/api/last-good/start", {})[0], 401)   # (the button is an api call too)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_health_of_a_failed_load(self):
        self.state.fail(RuntimeError("the engine exited before it was ready"))
        s, _, b = req(self.base, "GET", "/health")
        h = json.loads(b)
        self.assertEqual((s, h["status"], h["error"]), (503, "failed", "the engine exited before it was ready"))
        self.assertNotIn("last_good", h)                       # (nothing known to have worked before)
        self.assertTrue(self.state.failed_seen.is_set())       # (the supervisor has now heard it: main() may end)


class LastGood(unittest.TestCase):
    """The note of what loaded last, and the failed page's offer to start it again."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpd = Path(self.tmp.name)
        self.bad = self.tmpd / "maya-bad.json"
        self.good = self.tmpd / "maya-good.json"
        for p in (self.bad, self.good):
            p.write_text('{"model_name": "x"}', encoding="utf-8")
        self.note = self.tmpd / models_mod.LAST_GOOD
        self.state = LoadState("glm-5.3-flash", config_path=self.bad)
        self.httpd = serve_loading(self.state, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.state.port = self.httpd.server_address[1]
        patch = mock.patch.object(SV, "last_good_note_path", lambda: self.note)
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def write_note(self, path, t=None):
        self.note.write_text(json.dumps({"config": str(path), "time": t or time.time(), "model": "x"}), encoding="utf-8")

    def test_write_and_read_round_trip(self):
        self.assertIsNone(SV.read_last_good())                 # (nothing noted yet)
        SV.write_last_good(self.good, "GLM 5.3")
        self.assertEqual(SV.read_last_good(),
                         {"config": self.good.name, "path": str(self.good), "time": mock.ANY})
        self.good.unlink()
        self.assertIsNone(SV.read_last_good())                 # (that config is gone: not offered)
        self.note.write_text("{not json", encoding="utf-8")
        self.assertIsNone(SV.read_last_good())

    def test_the_failed_health_offers_it_but_only_where_it_works(self):
        self.write_note(self.good)
        self.state.fail(RuntimeError("out of memory"))
        h = json.loads(req(self.base, "GET", "/health")[2])
        self.assertEqual(h["last_good"], {"config": self.good.name, "path": str(self.good), "time": mock.ANY})
        self.state.config_path = str(self.good)                # (that is this very config: not "another one")
        self.assertNotIn("last_good", json.loads(req(self.base, "GET", "/health")[2]))
        self.state.config_path = str(self.bad)
        self.state.supervisor = "llama-swap"                   # (a supervisor owns the starts: never an offer)
        self.assertNotIn("last_good", json.loads(req(self.base, "GET", "/health")[2]))

    def test_the_button_is_refused_where_it_cannot_work(self):
        self.state.fail(RuntimeError("boom"))
        gone = req(self.base, "POST", "/api/last-good/start", {})
        self.assertEqual((gone[0], json.loads(gone[2])["error"]["type"]), (409, "invalid_request_error"))
        self.assertIn("no last config that worked", json.loads(gone[2])["error"]["message"])
        self.write_note(self.good)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SAGUARO_RESTART_ON_UPDATE", None)
            s, _, b = req(self.base, "POST", "/api/last-good/start", {})
            self.assertEqual(s, 409)                                          # (nothing started it: it cannot restart)
            self.assertIn("saguaro.sh", json.loads(b)["error"]["message"])
        s, _, _ = req(self.base, "POST", "/api/last-good/start", {}, {"Origin": "http://elsewhere.example"})
        self.assertEqual(s, 403)                                              # (the same-origin rule, as the real one)
        self.state.supervisor = "llama-swap"
        s, _, b = req(self.base, "POST", "/api/last-good/start", {})
        self.assertEqual(s, 409)
        self.assertIn("llama-swap", json.loads(b)["error"]["message"])
        self.state.supervisor = None
        self.state.config_path = str(self.good)                               # (that is what this start already tried)
        s, _, b = req(self.base, "POST", "/api/last-good/start", {})
        self.assertEqual(s, 409)
        self.assertIn("tried and failed", json.loads(b)["error"]["message"])

    def test_the_button_notes_the_switch_and_ends_the_start(self):
        self.state.fail(RuntimeError("boom"))
        self.write_note(self.good)
        called = []
        with mock.patch.dict(os.environ, {"SAGUARO_RESTART_ON_UPDATE": "1"}), \
                mock.patch.object(SV, "ROOT", self.tmpd), \
                mock.patch.object(SV, "_restart_after_answer", lambda delay=0.5: called.append(delay)):
            s, _, b = req(self.base, "POST", "/api/last-good/start", {})
            self.assertEqual((s, json.loads(b)), (200, {"starting": self.good.name}))
            note = json.loads((self.tmpd / models_mod.SWITCH_FILE).read_text(encoding="utf-8"))
        self.assertEqual((note["config"], note["why"], note["port"]), (str(self.good), "switch", self.state.port))
        for _ in range(100):                                                   # (the thread that ends the process)
            if called:
                break
            time.sleep(0.02)
        self.assertEqual(called, [0.5])

    def test_a_failed_load_ends_the_process_for_a_supervisor_and_for_exit_on_fail(self):
        self.assertEqual([(SV.ends_on_failed_load(s, e)) for s, e in
                          ((None, False), ("llama-swap", False), (None, True), ("docker-gen", True))],
                         [False, True, True, True])


class TakeOver(unittest.TestCase):
    """The same listening socket, the loading handler out and the real service in: one client never notices."""

    def setUp(self):
        self.state = LoadState("glm-5.3-flash")
        self.httpd = serve_loading(self.state, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.port = self.httpd.server_address[1]
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nhello", max_context=8192), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_the_hand_over_drops_nothing(self):
        seen, stop = [], threading.Event()

        def client():
            while not stop.is_set():
                try:
                    with urllib.request.urlopen(self.base + "/health", timeout=5) as r:
                        seen.append(r.status)
                except urllib.error.HTTPError as e:            # (503 is an answer too, not a gap)
                    seen.append(e.code)
                    e.close()
                except Exception:  # noqa: BLE001 - a refused connection would be the gap we must not have
                    seen.append("down")
                time.sleep(0.02)

        th = threading.Thread(target=client, daemon=True)
        th.start()
        try:
            for _ in range(100):
                if seen:
                    break
                time.sleep(0.02)
            self.assertEqual(seen[-1], 503)                    # (the starting service, before the engine)
            take_over(self.httpd, self.svc)
            for _ in range(200):
                if seen[-1] == 200:
                    break
                time.sleep(0.02)
        finally:
            stop.set()
            th.join(5)
        self.assertNotIn("down", seen)                         # (no refused connection across the swap)
        self.assertEqual(seen[0], 503)
        self.assertEqual(seen[-1], 200)
        self.assertNotIn(200, seen[:seen.index(200)])          # (one change: loading, then the real service)
        self.assertEqual(self.httpd.server_address[1], self.port)   # (the same socket: no second bind)

    def test_after_the_hand_over_the_real_service_answers(self):
        take_over(self.httpd, self.svc)
        s, _, b = req(self.base, "GET", "/health")
        self.assertEqual((s, json.loads(b)["version"]), (200, SV.SAGUARO_VERSION))
        s, _, b = req(self.base, "POST", "/v1/chat/completions",
                      {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 200)
        self.assertEqual(json.loads(b)["choices"][0]["message"]["content"], "hello")
        self.assertEqual(req(self.base, "GET", "/")[0], 200)   # (the page, from the real handler as ever)


# ----------------------------------------------------------------------------------------- the real start, as a child
WRAPPER = '''"""scratch: the real serve.server.main() with the mock engine slow or dead (issue #20 tests)."""
import os
import sys
import time

sys.path.insert(0, os.environ["W20_ROOT"])
import serve.server as S  # noqa: E402

sys.argv = ["server.py", "--engine", "mock", "--config", os.environ["W20_CFG"], "--port", os.environ["W20_PORT"]]
if os.environ.get("W20_EXIT_ON_FAIL") == "1":
    sys.argv.append("--exit-on-fail")
if os.environ.get("W20_MODE") == "slow":
    Real = S.MockEngine
    delay = float(os.environ.get("W20_SLOW", "10"))

    class Slow(Real):
        def __init__(self, *a, **k):
            time.sleep(delay)          # (the engine not answering yet: the port must answer anyway)
            super().__init__(*a, **k)

    S.MockEngine = Slow
elif os.environ.get("W20_MODE") == "broken":
    def boom(*a, **k):
        raise RuntimeError("fake engine died: out of memory")
    S.MockEngine = boom
sys.exit(S.main())
'''


class RealStart(unittest.TestCase):
    """serve.server.main() as a child process: the start order, the failed load's three lives, the button."""

    @classmethod
    def setUpClass(cls):
        cls.saved = {}
        for p in (NOTE, SWITCH):
            cls.saved[p] = p.read_bytes() if p.is_file() else None

    @classmethod
    def tearDownClass(cls):
        for p, was in cls.saved.items():
            if was is None:
                p.unlink(missing_ok=True)
            else:
                p.write_bytes(was)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpd = Path(self.tmp.name)
        self.wrapper = self.tmpd / "wrapper.py"
        self.wrapper.write_text(WRAPPER, encoding="utf-8")
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
                p.wait(10)
        NOTE.unlink(missing_ok=True)
        SWITCH.unlink(missing_ok=True)
        self.tmp.cleanup()

    def start(self, mode, env=None, exit_on_fail=False):
        self.cfg = self.tmpd / "maya-w20.json"
        log = self.tmpd / "run.log"
        log.write_text("w20: the engine is thinking about its experts\n", encoding="utf-8")
        self.cfg.write_text(json.dumps({"model_name": "W20 Fake 9B", "log": str(log)}), encoding="utf-8")
        port = free_port()
        e = dict(os.environ, W20_ROOT=str(ROOT), W20_CFG=str(self.cfg), W20_PORT=str(port), W20_MODE=mode)
        e.pop("SAGUARO_SUPERVISOR", None)
        e.pop("SAGUARO_RESTART_ON_UPDATE", None)
        if exit_on_fail:
            e["W20_EXIT_ON_FAIL"] = "1"
        e.update(env or {})
        p = subprocess.Popen([sys.executable, str(self.wrapper)], env=e, cwd=str(ROOT),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs.append(p)
        return p, port, log

    def health(self, port):
        return req(f"http://127.0.0.1:{port}", "GET", "/health")

    def wait_health(self, port, want, timeout=45):
        t0, last = time.time(), None
        while time.time() - t0 < timeout:
            try:
                st, _, b = self.health(port)
                last = (st, json.loads(b))
                if want(*last):
                    return last[1]
            except (urllib.error.URLError, ConnectionError, ValueError, TimeoutError):
                pass
            time.sleep(0.15)
        self.fail(f"/health never said what we waited for (port {port}): {last}")

    def test_the_page_answers_while_the_slow_engine_loads(self):
        p, port, _ = self.start("slow", {"W20_SLOW": "10"})
        base = f"http://127.0.0.1:{port}"
        h = self.wait_health(port, lambda c, b: c == 503 and b.get("status") == "loading")
        self.assertLess(h["elapsed_s"], 9)                     # (answered long before the 10 s engine is up)
        self.assertEqual(h["model"], "W20 Fake 9B")
        self.assertEqual(h["line"], "w20: the engine is thinking about its experts")
        self.assertEqual(req(base, "GET", "/")[0], 200)        # (the page on the port, from the first second)
        s, hd, b = req(base, "POST", "/v1/chat/completions", {"model": "m", "messages": []})
        self.assertEqual((s, hd.get("Retry-After"), json.loads(b)), (503, "5", {"error": "the model is loading"}))
        self.wait_health(port, lambda c, b: c == 200)          # (the hand-over: the real service, same port)
        s, _, b = req(base, "POST", "/v1/chat/completions",
                      {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 200)
        self.assertIn("mock engine", json.loads(b)["choices"][0]["message"]["content"])
        note = json.loads(NOTE.read_text(encoding="utf-8"))    # (a load that worked is remembered)
        self.assertEqual((Path(note["config"]).resolve(), note["model"]), (self.cfg.resolve(), "W20 Fake 9B"))
        p.send_signal(15)                                      # (SIGTERM as a supervisor's stop)
        self.assertIn(p.wait(20), (0, 143, -15))

    def test_sigterm_while_loading_stops_at_once(self):
        p, port, _ = self.start("slow", {"W20_SLOW": "30"})
        self.wait_health(port, lambda c, b: c == 503 and b.get("status") == "loading")
        t0 = time.time()
        p.send_signal(15)
        self.assertIn(p.wait(15), (143, -15, 1))
        self.assertLess(time.time() - t0, 12)                  # (not the engine's remaining minutes)
        with self.assertRaises(urllib.error.URLError):
            self.health(port)                                  # (the port is closed with it)

    def test_a_failed_load_stands_with_the_error_and_offers_the_last_config(self):
        good = self.tmpd / "maya-good.json"
        good.write_text('{"model_name": "W20 Good 9B"}', encoding="utf-8")
        NOTE.write_text(json.dumps({"config": str(good), "time": time.time(), "model": "x"}), encoding="utf-8")
        p, port, _ = self.start("broken")
        h = self.wait_health(port, lambda c, b: b.get("status") == "failed")
        self.assertEqual(h["error"], "fake engine died: out of memory")
        self.assertEqual(h["last_good"], {"config": good.name, "path": str(good), "time": mock.ANY})
        time.sleep(1.5)
        self.assertIsNone(p.poll())                            # (a direct start stands: the page says what went wrong)
        self.assertEqual(self.health(port)[0], 503)
        self.assertEqual(req(f"http://127.0.0.1:{port}", "GET", "/")[0], 200)
        p.send_signal(2)                                       # (Ctrl+C closes the standing page)
        self.assertIn(p.wait(15), (130, 143, 1, -2, -15))

    def test_a_failed_load_ends_the_process_with_exit_on_fail(self):
        p, port, _ = self.start("broken", exit_on_fail=True)
        self.assertEqual(p.wait(30), 1)
        out = p.communicate()[0]
        self.assertIn("the model failed to load: fake engine died", out)
        with self.assertRaises(urllib.error.URLError):
            self.health(port)

    def test_a_failed_load_ends_under_llama_swap_once_its_health_saw_it(self):
        good = self.tmpd / "maya-good.json"
        good.write_text('{"model_name": "x"}', encoding="utf-8")
        NOTE.write_text(json.dumps({"config": str(good), "time": time.time(), "model": "x"}), encoding="utf-8")
        p, port, _ = self.start("broken", {"SAGUARO_SUPERVISOR": "llama-swap"})
        t0 = time.time()
        h = self.wait_health(port, lambda c, b: b.get("status") == "failed", timeout=30)
        self.assertNotIn("last_good", h)                       # (llama-swap owns the starts: no button, no offer)
        self.assertEqual(p.wait(15), 1)                        # (it ends once the check has seen it, for llama-swap
        self.assertLess(time.time() - t0, 20)                  #  to retry or give up - never stands forever)

    def test_the_button_starts_the_last_config_that_worked(self):
        good = self.tmpd / "maya-good.json"
        good.write_text('{"model_name": "x"}', encoding="utf-8")
        NOTE.write_text(json.dumps({"config": str(good), "time": time.time(), "model": "x"}), encoding="utf-8")
        p, port, _ = self.start("broken", {"SAGUARO_RESTART_ON_UPDATE": "1"})
        self.wait_health(port, lambda c, b: b.get("status") == "failed")
        s, _, b = req(f"http://127.0.0.1:{port}", "POST", "/api/last-good/start", {})
        self.assertEqual((s, json.loads(b)), (200, {"starting": good.name}))
        self.assertEqual(p.wait(15), SV.RESTART_EXIT)          # (saguaro.py reads the note and starts again)
        note = json.loads(SWITCH.read_text(encoding="utf-8"))
        self.assertEqual((note["config"], note["why"], note["port"]), (str(good), "switch", port))


# ----------------------------------------------------------------------------------- the page's own starting screen
@unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
class PageStart(unittest.TestCase):
    def node(self, body):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.mjs"
            p.write_text(PAGE_STUBS + body, encoding="utf-8")
            r = subprocess.run(["node", str(p), (ROOT / "serve" / "web").resolve().as_uri()],
                               capture_output=True, encoding="utf-8", timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_the_screen_says_the_name_the_time_the_line_or_the_error(self):
        got = self.node("""
const so = await import(web + "/starting.js");
console.log(JSON.stringify({
  loading: so.startingView({status: "loading", model: "glm-5.3-flash", elapsed_s: 95, line: "experts 1200/2000"}),
  failed: so.startingView({status: "failed", model: "glm-5.3-flash", error: "nope",
                           last_good: {config: "maya-good.json", time: 1}}),
  waiting: so.startingView(undefined),
  first: so.startingView(null),
}));""")
        self.assertEqual((got["loading"]["state"], got["loading"]["name"], got["loading"]["elapsed"],
                          got["loading"]["line"]), ("loading", "GLM 5.3 Flash", "2 min", "experts 1200/2000"))
        self.assertTrue(got["loading"]["lastGood"] is None and got["loading"]["error"] == "")
        self.assertEqual((got["failed"]["state"], got["failed"]["failed"], got["failed"]["error"],
                          got["failed"]["lastGood"]),
                         ("failed", True, "nope", {"config": "maya-good.json", "time": 1}))
        self.assertEqual(got["waiting"]["state"], "waiting")   # (/health has not answered to this page yet)
        self.assertEqual(got["first"]["state"], "waiting")     # (the page's first paint: S.starting is null)

    def test_the_screen_waits_one_second_and_becomes_the_page_on_the_first_200(self):
        got = self.node("""
const times = []; let n = 0;
globalThis.fetch = async (u) => { times.push(Date.now()); n++;
  if (u === "health" && n < 3) return {ok: false, status: 503,
    json: async () => ({status: "loading", model: "glm-5.3-flash", elapsed_s: n})};
  return {ok: true, status: 200, json: async () => ({model: "glm-5.3-flash", max_context: 8192})}; };
const states = [];
sv.watchStarting((h) => states.push([h.status, h.elapsed_s]), (h) => { states.push(["ready", h.model]); });
await new Promise((ok) => setTimeout(ok, 3200));
console.log(JSON.stringify({states, health: times.length}));""")
        self.assertEqual(got["states"], [["loading", 1], ["loading", 2], ["ready", "glm-5.3-flash"]])
        self.assertEqual(got["health"], 3)                     # (and the boot's health IS this one: not a fourth)

    def test_a_hidden_tab_asks_nothing_and_a_revealed_one_picks_the_watch_up(self):
        got = self.node("""
const log = []; let down = true;
globalThis.fetch = async (u) => { log.push(u);
  if (down) return {ok: false, status: 503, json: async () => ({status: "loading", model: "x", elapsed_s: 1})};
  return {ok: true, status: 200, json: async () => ({model: "x"})}; };
document.hidden = true;
let asked = 0, ready = 0;
sv.watchStarting(() => asked++, () => ready++);
await new Promise((ok) => setTimeout(ok, 2300));               // (two seconds of the screen's life, in the dark)
const inDark = log.slice();
down = false; document.hidden = false; fire("visibilitychange");
await new Promise((ok) => setTimeout(ok, 200));
console.log(JSON.stringify({inDark, seen: log.slice(), asked, ready}));""")
        self.assertEqual(got["inDark"], [])                    # (asks nothing, as today's rules)
        self.assertEqual(got["seen"], ["health"])              # (seen again: it picks the watch up at once)
        self.assertEqual((got["asked"], got["ready"]), (0, 1))

    def test_behind_llama_swap_off_the_card_only_its_running_is_asked(self):
        got = self.node("""
location.pathname = "/upstream/glm-5.3-flash/";
const log = []; let ready = false;
globalThis.fetch = async (u) => { log.push(u);
  if (u.endsWith("/running")) return {ok: true, status: 200,
    json: async () => ({running: [{model: ready ? "glm-5.3-flash" : "other", state: "ready"}]})};
  return {ok: true, status: 200, json: async () => ({model: "glm-5.3-flash", supervisor: "llama-swap"})}; };
let states = 0, booted = 0, with_ = "unset";
sv.watchStarting(() => states++, (h) => { booted++; with_ = h; });
await new Promise((ok) => setTimeout(ok, 700));
const offCard = log.slice();
sv.stopPoll();
console.log(JSON.stringify({offCard, states, booted, with_, unloaded: sv.unloaded()}));""")
        # another model has the card: the normal page boots in its quiet unloaded state (nothing asked of the model), not
        # a starting screen that never ends
        self.assertTrue(all(u.endswith("/running") for u in got["offCard"]), got["offCard"])   # (/health would load it)
        self.assertEqual((got["states"], got["booted"], got["with_"], got["unloaded"]), (0, 1, None, True))

    def test_behind_llama_swap_while_it_starts_the_model_the_screen_waits(self):
        got = self.node("""
location.pathname = "/upstream/glm-5.3-flash/";
const log = [];
globalThis.fetch = async (u) => { log.push(u);
  if (u.endsWith("/running")) return {ok: true, status: 200,
    json: async () => ({running: [{model: "glm-5.3-flash", state: "starting"}]})};
  return {ok: false, status: 503, json: async () => ({status: "loading"})}; };
let booted = 0;
sv.watchStarting(() => {}, () => booted++);
await new Promise((ok) => setTimeout(ok, 700));
console.log(JSON.stringify({log, booted})); process.exit(0);""")
        self.assertTrue(all(u.endswith("/running") for u in got["log"]), got["log"])           # (not asked: llama-swap starts it)
        self.assertEqual(got["booted"], 0)


if __name__ == "__main__":
    unittest.main()
