"""serve/test_turbo_live.py - the live per-request Turbo (PLAN-turbo-live): the request field, the engine-line keys
(both drivers), the model's live global setting (GET/POST /api/turbo/options), the Monitor's badge fields - and
that a request without Turbo still gets today's exact line.  PLAN-p124s: that global is the web app's own chats'
choice - an API request gets Turbo only from its own fields.

    python -m unittest serve.test_turbo_live -v
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import turbo_live as T  # noqa: E402
from serve.server import ByteTokenizer, Service, StrataEngine, serve  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.test_server import ROOT, RecordingEngine  # noqa: E402


class FieldShape(unittest.TestCase):
    """the wire form: {"drop": N, "skip": "off"|"ssd"|"ram"}; anything else says so (the 400 names the field)."""

    def test_good_forms(self):
        self.assertEqual(T.normalize_turbo({"drop": 2, "skip": "SSD"}), {"drop": 2, "skip": "ssd"})
        self.assertEqual(T.normalize_turbo({"drop": "3"}), {"drop": 3, "skip": "off"})   # (string numbers: read, as elsewhere)
        self.assertEqual(T.normalize_turbo({"skip": "ram"}), {"drop": 0, "skip": "ram"})
        self.assertEqual(T.normalize_turbo({"drop": 0, "skip": "off"}), {"drop": 0, "skip": "off"})   # (an explicit off stays: it overrides the global)
        self.assertIsNone(T.normalize_turbo(None))
        self.assertEqual(T.normalize_turbo({"drop": None}), {"drop": 0, "skip": "off"})   # (null parts read as off)

    def test_bad_forms_are_named(self):
        for bad in (True, "fast", {"drop": 10}, {"drop": -1}, {"drop": 1.5}, {"drop": True}, {"skip": "disk"},
                    {"skip": 2}, [1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                T.normalize_turbo(bad)


class EngineLine(unittest.TestCase):
    """the request-line keys: absent/off is today's line, byte for byte; a part that is on adds its key."""

    def keys(self, sampling):
        return StrataEngine.sampling_keys(sampling)

    def test_absent_is_the_line_without_turbo(self):
        base = {"temperature": 0.6, "top_p": 0.9, "seed": 7, "max_tokens": 64}
        self.assertEqual(self.keys(dict(base)), self.keys({**base, "turbo": None}))
        self.assertEqual(self.keys(dict(base)), self.keys({**base, "turbo": {"drop": 0, "skip": "off"}}))
        self.assertEqual(self.keys(dict(base, temperature=0)), self.keys({**base, "temperature": 0,
                                                                          "turbo": {"drop": 0, "skip": "off"}}))
        self.assertNotIn("turbo", self.keys(base))
        self.assertNotIn("turbo", self.keys({**base, "turbo": {"drop": 0, "skip": "off"}}))

    def test_the_keys_join_the_line(self):
        s = {"temperature": 0.6, "turbo": {"drop": 2, "skip": "ram"}}
        self.assertEqual(self.keys(s), StrataEngine.sampling_keys({"temperature": 0.6}) + " turbo_drop=2 turbo_skip=ram")
        self.assertEqual(self.keys({"temperature": 0, "turbo": {"drop": 0, "skip": "ssd"}}),
                         StrataEngine.sampling_keys({"temperature": 0}) + " turbo_skip=ssd")   # (the greedy line takes it too)
        self.assertEqual(self.keys({"turbo": {"drop": 3, "skip": "off"}}), " turbo_drop=3")

    def test_the_qfn_driver(self):
        from serve.qfn_engine import StrataEngine as Q
        s = {"temperature": 0.6, "turbo": {"drop": 1, "skip": "ssd"}}
        self.assertEqual(Q.sampling_keys(s), Q.sampling_keys({"temperature": 0.6}) + " turbo_drop=1 turbo_skip=ssd")
        self.assertNotIn("turbo", Q.sampling_keys({"temperature": 0.6, "turbo": {"drop": 0, "skip": "off"}}))


class ServerSide(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class TurboTaker(RecordingEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                self.last_sampling = dict(sampling or {})
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        tok = ByteTokenizer()
        cls.engine = TurboTaker(tok, "ok", max_context=32768)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.d = Path(tempfile.mkdtemp())
        cls.svc.turbo_path = str(cls.d / "maya-t.turbo.json")
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        shutil.rmtree(cls.d, ignore_errors=True)

    def get(self, path):
        import urllib.request
        with urllib.request.urlopen(self.base + path) as r:
            return r.status, json.loads(r.read())

    def post(self, path, body):
        import urllib.error
        import urllib.request
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def chat(self, **extra):
        status, b = self.post("/v1/chat/completions",
                              {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, **extra})
        self.assertEqual(status, 200, b)
        return self.engine.last_sampling.get("turbo")

    def page(self, **extra):
        # the web app's own chat request (serve/web/chats.js): strata_resume is what the server marks _api "web"
        status, b = self.post("/v1/chat/completions",
                              {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
                               "strata_resume": True, **extra})
        self.assertEqual(status, 200, b)
        return self.engine.last_sampling.get("turbo")

    def messages_api(self, **extra):
        status, b = self.post("/v1/messages", {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                               "max_tokens": 4, **extra})
        self.assertEqual(status, 200, b)
        return self.engine.last_sampling.get("turbo")

    def responses_api(self, **extra):
        status, b = self.post("/v1/responses", {"model": "m", "input": "hi", "max_output_tokens": 4, **extra})
        self.assertEqual(status, 200, b)
        return self.engine.last_sampling.get("turbo")

    def test_the_turbo_reaches_the_engine(self):
        self.assertIsNone(self.chat())                                              # (absent: nothing reaches the engine line)
        self.assertEqual(self.chat(turbo={"drop": 2, "skip": "ssd"}), {"drop": 2, "skip": "ssd"})
        self.assertEqual(self.chat(turbo={"drop": 0, "skip": "off"}), {"drop": 0, "skip": "off"})   # (canonical, says off)

    def test_a_bad_turbo_is_a_400(self):
        for bad in ({"drop": 12}, {"skip": "disk"}, "yes"):
            with self.subTest(bad=bad):
                status, b = self.post("/v1/chat/completions",
                                      {"model": "m", "messages": [{"role": "user", "content": "hi"}], "turbo": bad})
                self.assertEqual(status, 400, b)
                self.assertIn("turbo", b["error"]["message"])

    def test_the_api_never_borrows_the_pages_turbo(self):
        # PLAN-p124s: with the sidecar ON (POST /api/turbo/options stores it) and the page's last choice ON,
        # a /v1/chat/completions, a /v1/messages and a /v1/responses request without Turbo fields run with
        # Turbo off; with their own field they run with it on; the web app's own request still follows it
        status, out = self.post("/api/turbo/options", {"turbo": {"drop": 2, "skip": "ram"}})
        self.assertEqual((status, out), (200, {"turbo": {"drop": 2, "skip": "ram"}}))
        self.assertTrue(Path(self.svc.turbo_path).exists())          # (the sidecar is on: it survives a restart)
        for call in (self.chat, self.messages_api, self.responses_api):
            self.assertIsNone(call(), call.__name__)                 # (no field of its own: today's exact request)
            self.assertEqual(call(turbo={"drop": 1, "skip": "ssd"}), {"drop": 1, "skip": "ssd"})   # (its own asks: on)
        self.assertEqual(self.page(), {"drop": 2, "skip": "ram"})    # (the app's own chat follows the panel's choice)
        self.assertEqual(self.page(turbo={"drop": 3}), {"drop": 3, "skip": "off"})     # (its chip wins)
        self.assertEqual(self.page(turbo={"drop": 0, "skip": "off"}), {"drop": 0, "skip": "off"})   # (and opts out)
        self.assertEqual(self.post("/api/turbo/options", {"turbo": None})[1], {"turbo": None})
        self.assertIsNone(self.page())                               # (panel off: the chats run exact again)

    def test_the_live_global_setting_applies_and_persists(self):
        self.assertEqual(self.post("/api/turbo/options", {"turbo": None})[1], {"turbo": None})   # (clean start)
        status, out = self.post("/api/turbo/options", {"turbo": {"drop": 1, "skip": "ssd"}})
        self.assertEqual((status, out), (200, {"turbo": {"drop": 1, "skip": "ssd"}}))
        self.assertIsNone(self.chat())                                                          # (PLAN-p124s: an API caller without its own runs exact)
        self.assertEqual(self.page(), {"drop": 1, "skip": "ssd"})                               # (the app's own chats take the panel's)
        self.assertEqual(self.chat(turbo={"drop": 3}), {"drop": 3, "skip": "off"})             # (its own wins)
        self.assertEqual(self.chat(turbo={"drop": 0, "skip": "off"}), {"drop": 0, "skip": "off"})   # (an explicit off opts out)
        self.assertEqual(Path(self.svc.turbo_path).read_text().strip(), json.dumps({"drop": 1, "skip": "ssd"}, indent=1).strip())
        status, out = self.post("/api/turbo/options", {"turbo": {"drop": 0, "skip": "off"}})    # (off = no global)
        self.assertEqual(out, {"turbo": None})
        self.assertFalse(Path(self.svc.turbo_path).exists())
        self.assertIsNone(self.chat())
        self.assertEqual(self.post("/api/turbo/options", {"turbo": {"skip": "nonsense"}})[0], 400)

    def test_the_page_is_told_when_the_engine_leaves_skip_off(self):
        # PLAN-p5: /api/turbo/options only, "turbo_skip" (default true): false only for a QFN whose engine
        # does not report that it applies skip - the page hides the Skip control on that
        s, out = self.get("/api/turbo/options")
        self.assertEqual((s, out["turbo_skip"]), (200, True))          # (every engine but QFN: skip applies)
        self.svc.engine_kind = "qfn"                                   # (QFN's engine: drop only until it says otherwise)
        try:
            self.assertFalse(self.get("/api/turbo/options")[1]["turbo_skip"])
            self.engine.info["turbo_skip"] = 1                         # (the engine reports it: INFO turbo_skip=1)
            self.assertTrue(self.get("/api/turbo/options")[1]["turbo_skip"])
            del self.engine.info["turbo_skip"]
        finally:
            self.svc.engine_kind = None
        self.assertTrue(self.get("/api/turbo/options")[1]["turbo_skip"])   # (and the POST body still says only {"turbo": ...})
        self.assertEqual(self.post("/api/turbo/options", {"turbo": {"drop": 2, "skip": "ssd"}})[1],
                         {"turbo": {"drop": 2, "skip": "ssd"}})
        self.post("/api/turbo/options", {"turbo": None})

    def test_the_monitor_rows_carry_the_options(self):
        self.chat(turbo={"drop": 2, "skip": "ram"})
        m = json.loads(json.dumps(self.svc.metrics()))          # (the /metrics body: the badge's fields)
        row = (m.get("requests") or [{}])[0]
        self.assertEqual(row.get("turbo"), {"drop": 2, "skip": "ram"})


class ResponsesApi(unittest.TestCase):
    def test_turbo_is_an_allowed_responses_parameter(self):
        from serve.responses import normalize_request
        out = normalize_request({"input": "hi", "turbo": {"drop": 2, "skip": "ram"}})
        self.assertEqual(out["turbo"], {"drop": 2, "skip": "ram"})
        self.assertNotIn("turbo", normalize_request({"input": "hi"}))


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class PageSide(unittest.TestCase):
    """the page's per-chat half (serve/web/turbo-core.js, under node): the chat's options as they go on the wire."""

    def run_js(self, code):
        import subprocess
        src = (Path(__file__).resolve().parent / "web" / "turbo-core.js").as_uri()
        out = subprocess.run(["node", "--input-type=module", "-e", f'import * as T from "{src}";\n{code}'],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_the_chat_options(self):
        r = self.run_js('''console.log(JSON.stringify({
  on: T.turboOfChat({turbo: {drop: 2, skip: "ssd"}}),
  off: T.turboOfChat({turbo: {drop: 0, skip: "off"}}),
  none: T.turboOfChat({}),
  skip_only: T.turboChoose(null, "skip", "ram"),
  chosen: T.turboChoose({turbo: {drop: 1, skip: "off"}}, "drop", 3),
  toggle_on: T.turboToggle(null), toggle_off: T.turboToggle({turbo: {drop: 1, skip: "off"}}),
  badge: T.turboBadge({drop: 2, skip: "ram"}), empty: T.turboBadge(null) === ""}));''')
        self.assertEqual(r["on"], {"drop": 2, "skip": "ssd"})
        self.assertIsNone(r["off"])
        self.assertIsNone(r["none"])
        self.assertEqual(r["skip_only"], {"drop": 0, "skip": "ram"})
        self.assertEqual(r["chosen"], {"drop": 3, "skip": "off"})
        self.assertEqual(r["toggle_on"], {"drop": 1, "skip": "off"})
        self.assertIsNone(r["toggle_off"])
        self.assertEqual(r["badge"], "-2 experts · VRAM only")
        self.assertTrue(r["empty"])

    def test_the_options_are_named_plainly(self):
        """PLAN-p5: the words the rows and the menu show; the warning text exists only for RAM."""
        r = self.run_js('''console.log(JSON.stringify({
  drop: T.TURBO_DROP_HINT,
  words: T.TURBO_SKIPS.map((s) => [s.id, s.name, s.hint, s.warn || ""]),
  warns: T.TURBO_SKIPS.map((s) => T.turboSkipWarn(s.id))}));''')
        self.assertEqual(r["drop"], "fewer of the 10 routed experts a token, the weakest dropped")
        self.assertEqual([w[0] for w in r["words"]], ["off", "ssd", "ram"])
        self.assertEqual([w[1] for w in r["words"]], ["off", "SSD", "RAM"])
        self.assertEqual(r["words"][1][2], "an expert that would be read from the SSD is left out")
        self.assertEqual(r["words"][2][2], "an expert that is not on the card is left out")
        warn = "can break answers: only where almost everything is on the card"
        self.assertEqual((r["words"][0][3], r["words"][1][3]), ("", ""))      # (off and SSD carry no warning)
        self.assertEqual(r["words"][2][3], warn)
        self.assertEqual(r["warns"], ["", "", warn])                          # (turboSkipWarn: only RAM answers)


if __name__ == "__main__":
    unittest.main()
