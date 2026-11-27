"""serve/test_switch.py - switching models without blanking the app (P125m phase 2, part 1).  A switch ends the
process and Saguaro starts again with the other model; the port keeps answering all through it (issue #20) and:

* the starting service of a switch says /health {"switch": true} with the model coming, keeps /v1/models answering
  200 with that model listed (status "loading"), and answers the chat routes with a 503 that names the model and
  says to retry once it is up (Retry-After: 5, as before); a start that is not a switch keeps every one of those
  exactly as issue #20 phase 1 wrote them;
* against a real serve.server.main() run (the mock engine, slow, SAGUARO_SWITCH=1 - the env saguaro.py sets for
  the second half of a switch): the switch keeps /v1/models and the chat API answering, before the new model is
  ready and after;
* the page itself (Node, as test_starting's page tests): switchingView's words, a booted app noticing the switch
  from a failed poll (probeSwitch / watchSwitch), sending waiting with the bar, and the app returning to normal
  (reload) when /health answers 200 again.

    python -m unittest serve.test_switch -v
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
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import models as models_mod  # noqa: E402  (LAST_GOOD: the note a successful mock start records)
from serve.server import LoadState, serve_loading  # noqa: E402
from serve.test_dashboard import PAGE_STUBS, ROOT  # noqa: E402  (the Node stubs for the page's modules)
from serve.test_starting import WRAPPER, free_port, req  # noqa: E402  (the mock-engine child wrapper)


class StartingSwitch(unittest.TestCase):
    """The starting service when this start is the second half of a switch."""

    def open(self, **kw):
        self.state = LoadState("glm-5.3-flash", **kw)
        self.httpd = serve_loading(self.state, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.shutdown)
        self.addCleanup(self.httpd.server_close)

    def test_health_says_switch_and_which_model_is_coming(self):
        self.open(switching=True)
        s, _, b = req(self.base, "GET", "/health")
        h = json.loads(b)
        self.assertEqual((s, h["status"], h["switch"], h["model"]), (503, "loading", True, "glm-5.3-flash"))

    def test_a_plain_start_says_nothing_about_switching(self):
        self.open()
        self.assertNotIn("switch", json.loads(req(self.base, "GET", "/health")[2]))

    def test_the_model_listing_keeps_answering_through_a_switch(self):
        self.open(switching=True)
        s, _, b = req(self.base, "GET", "/v1/models")
        d = json.loads(b)["data"][0]
        self.assertEqual(s, 200)
        self.assertEqual((d["id"], d["object"], d["owned_by"], d["status"]["value"]),
                         ("glm-5.3-flash", "model", "saguaro", "loading"))
        self.assertEqual(req(self.base, "GET", "/models")[0], 200)     # (the llama.cpp alias too)

    def test_without_a_switch_the_listing_still_says_the_model_is_loading(self):
        self.open()
        s, hd, b = req(self.base, "GET", "/v1/models")
        self.assertEqual((s, hd.get("Retry-After"), json.loads(b)), (503, "5", {"error": "the model is loading"}))

    def test_the_listing_respects_the_api_key_as_the_engine_routes_do(self):
        self.open(switching=True, api_key="sag-key")
        self.assertEqual(req(self.base, "GET", "/v1/models")[0], 401)
        self.assertEqual(req(self.base, "GET", "/v1/models", headers={"x-api-key": "sag-key"})[0], 200)
        self.assertEqual(req(self.base, "GET", "/health")[0], 503)     # (open, as ever: supervisors read it)

    def test_the_chat_api_answers_and_names_the_model_that_is_coming(self):
        self.open(switching=True)
        for path in ("/v1/chat/completions", "/v1/responses", "/v1/messages"):
            t0 = time.time()
            s, hd, b = req(self.base, "POST", path, {"model": "anything", "messages": []})
            j = json.loads(b)
            self.assertEqual((path, s, hd.get("Retry-After"), j["loading"]), (path, 503, "5", "glm-5.3-flash"))
            self.assertIn("loading model glm-5.3-flash", j["error"]["message"])
            self.assertLess(time.time() - t0, 2)                       # (answered at once: requests may not queue)

    def test_a_plain_start_keeps_the_bare_loading_error_on_the_chat_routes(self):
        self.open()
        s, _, b = req(self.base, "POST", "/v1/chat/completions", {"model": "m", "messages": []})
        self.assertEqual((s, json.loads(b)), (503, {"error": "the model is loading"}))

    def test_the_page_and_the_other_routes_are_unchanged(self):
        self.open(switching=True)
        self.assertEqual(req(self.base, "GET", "/")[0], 200)            # (the page, on the port, as ever)
        self.assertEqual(req(self.base, "GET", "/metrics")[0], 503)     # (not a switch-listing route: as before)
        self.assertEqual(json.loads(req(self.base, "GET", "/metrics")[2]), {"error": "the model is loading"})


class SwitchStart(unittest.TestCase):
    """serve.server.main() as a child with the mock engine slow and SAGUARO_SWITCH=1: what the real switch's second
    half answers before the new model is ready - and that the same port then serves the chat."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpd = Path(self.tmp.name)
        self.wrapper = self.tmpd / "wrapper.py"
        self.wrapper.write_text(WRAPPER, encoding="utf-8")
        self.cfg = self.tmpd / "saguaro-switch.json"
        self.log = self.tmpd / "run.log"
        self.cfg.write_text(json.dumps({"model_name": "Switch Fake 9B", "log": str(self.log)}), encoding="utf-8")
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
                p.wait(10)
        (ROOT / models_mod.LAST_GOOD).unlink(missing_ok=True)
        self.tmp.cleanup()

    def start(self):
        port = free_port()
        e = dict(os.environ, W20_ROOT=str(ROOT), W20_CFG=str(self.cfg), W20_PORT=str(port),
                 W20_MODE="slow", W20_SLOW="8", SAGUARO_SWITCH="1")
        e.pop("SAGUARO_SUPERVISOR", None)
        p = subprocess.Popen([sys.executable, str(self.wrapper)], env=e, cwd=str(ROOT),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs.append(p)
        return p, f"http://127.0.0.1:{port}"

    def wait_health(self, base, want, timeout=45):
        t0, last = time.time(), None
        while time.time() - t0 < timeout:
            try:
                st, _, b = req(base, "GET", "/health")
                last = (st, json.loads(b))
                if want(*last):
                    return last[1]
            except (urllib.error.URLError, ConnectionError, ValueError, TimeoutError):
                pass
            time.sleep(0.15)
        self.fail(f"/health never said what we waited for: {last}")

    def test_the_switch_keeps_the_model_listing_and_the_chat_api_answering(self):
        import urllib.error  # noqa: F401 - wait_health reads through it
        p, base = self.start()
        h = self.wait_health(base, lambda c, b: c == 503 and b.get("status") == "loading")
        self.assertTrue(h["switch"])                            # (the second half of a switch, per saguaro.py)
        self.assertEqual(h["model"], "Switch Fake 9B")
        s, _, b = req(base, "GET", "/v1/models")                 # (/v1/models keeps answering)
        self.assertEqual((s, json.loads(b)["data"][0]["id"], json.loads(b)["data"][0]["status"]["value"]),
                         (200, "Switch Fake 9B", "loading"))
        s, hd, b = req(base, "POST", "/v1/chat/completions",
                       {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual((s, hd.get("Retry-After"), json.loads(b)["loading"]),
                         (503, "5", "Switch Fake 9B"))            # (the chat API answers, and says what it waits for)
        self.wait_health(base, lambda c, b: c == 200)             # (the hand-over on the same port)
        s, _, b = req(base, "GET", "/v1/models")
        self.assertEqual((s, json.loads(b)["data"][0]["status"]["value"]), (200, "loaded"))
        s, _, b = req(base, "POST", "/v1/chat/completions",
                      {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 200)
        self.assertIn("mock engine", json.loads(b)["choices"][0]["message"]["content"])
        p.send_signal(15)
        self.assertIn(p.wait(20), (0, 143, -15))


@unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
class PageSwitch(unittest.TestCase):
    def node(self, body):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.mjs"
            p.write_text(PAGE_STUBS + body, encoding="utf-8")
            r = subprocess.run(["node", str(p), (ROOT / "serve" / "web").resolve().as_uri()],
                               capture_output=True, encoding="utf-8", timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_the_bar_names_the_model_the_percent_and_the_step(self):
        got = self.node("""
const so = await import(web + "/starting.js");
console.log(JSON.stringify({
  loading: sv.switchingView({status: "loading", model: "glm-5.3-flash", pct: 43, step: "experts 1200/2000"}),
  round: sv.switchingView({status: "loading", model: "glm-5.3-flash", pct: 43.6}),
  nopct: sv.switchingView({status: "loading", model: "glm-5.3-flash", line: "reading the shards"}),
  asked: sv.switchingView({label: "GLM 5.3 Flash (Maya-S)"}),
  failed: sv.switchingView({status: "failed", model: "glm-5.3-flash", error: "out of memory"}),
  none: sv.switchingView(null),
}));""")
        self.assertEqual((got["loading"]["text"], got["loading"]["pct"], got["loading"]["step"]),
                         ("loading GLM 5.3 Flash 43%", 43, "experts 1200/2000"))
        self.assertEqual(got["round"]["pct"], 44)                    # (a whole percent on the bar)
        self.assertEqual(got["nopct"]["text"], "loading GLM 5.3 Flash")
        self.assertEqual(got["nopct"]["step"], "reading the shards")  # (the engine's line until a step word exists)
        self.assertEqual(got["asked"]["text"], "loading GLM 5.3 Flash (Maya-S)")
        self.assertTrue(got["failed"]["failed"] and got["failed"]["pct"] is None)
        self.assertEqual((got["none"]["text"], got["none"]["pct"]), ("loading the next model", None))

    def test_a_switch_this_page_asked_shows_the_bar_at_once_and_reloads_when_the_model_is_up(self):
        got = self.node("""
let n = 0, reloaded = 0;
location.reload = () => { reloaded++; };
globalThis.fetch = async () => { n++;
  if (n < 3) return {ok: false, status: 503, json: async () =>
    ({status: "loading", model: "glm-5.3-flash", pct: 20 * n, switch: true})};
  return {ok: true, status: 200, json: async () => ({model: "glm-5.3-flash", max_context: 8192})}; };
st.S.switching = {label: "GLM 5.3 Flash"};                          // (models.js set it when the page asked)
sv.watchSwitch();
await new Promise((ok) => setTimeout(ok, 3300));
console.log(JSON.stringify({sw: st.S.switching, reloaded, asked: n})); process.exit(0);""")
        self.assertEqual(got["reloaded"], 1)                          # (back: the app returns to normal)
        self.assertEqual(got["asked"], 3)                             # (/health once a second)
        self.assertEqual(got["sw"]["pct"], 40)                        # (the bar followed the load's own progress)

    def test_a_switch_asked_over_the_api_is_noticed_when_a_poll_fails(self):
        got = self.node("""
let reloaded = 0;
location.reload = () => { reloaded++; };
let down = true;
globalThis.fetch = async (u) => {
  if (u === "health" && down) return {ok: false, status: 503,
    json: async () => ({status: "loading", model: "glm-5.3-flash", pct: 12, switch: true})};
  return {ok: true, status: 200, json: async () => ({model: "glm-5.3-flash", max_context: 8192})}; };
const noticed = await sv.probeSwitch();
const askedWhileSwitching = (st.S.conn, sv.poll(), st.S.switching ? st.S.switching.model : null);
down = false;
await new Promise((ok) => setTimeout(ok, 1600));
console.log(JSON.stringify({noticed, askedWhileSwitching, reloaded})); process.exit(0);""")
        self.assertTrue(got["noticed"])                               # (the failed poll found the starting service)
        self.assertEqual(got["askedWhileSwitching"], "glm-5.3-flash")  # (S.switching set: the bar's data)
        self.assertEqual(got["reloaded"], 1)                          # (the watch ends in the reload)

    def test_poll_asks_nothing_while_the_switch_bar_stands(self):
        got = self.node("""
const log = [];
globalThis.fetch = async (u) => { log.push(u); return {ok: true, status: 200, json: async () => ({})}; };
st.S.switching = {status: "loading", model: "glm-5.3-flash", pct: 30};
sv.stopPoll();
await sv.poll();
console.log(JSON.stringify({log})); process.exit(0);""")
        self.assertEqual(got["log"], [])                              # (the starting service answers nothing here)


if __name__ == "__main__":
    unittest.main()
