"""serve/test_dashboard.py - what the web app's Settings and Monitor ask the server for: the context size (its range, a
reload, the 503 while it reloads, the run config it is saved into), the engine's measured context cost, and the report.
Against the mock engine (no GPU, no pack).

    python -m unittest serve.test_dashboard -v
"""
from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import gpu_map  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (CONTEXT_MIN, SAGUARO_VERSION, ByteTokenizer, MockEngine, Service, StrataEngine, serve,  # noqa: E402
                          speculation_modes, trained_context_ds4)

ROOT = Path(__file__).resolve().parents[1]


class Dashboard(unittest.TestCase):
    def setUp(self):
        os.environ["SAGUARO_MOCK_RELOAD_S"] = "0.3"
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, "</think>\n\nhello", max_context=32768)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Path(self.tmp.name) / "maya-test.json"
        self.cfg.write_text(json.dumps({"exe": "strata", "args": ["--glm-pack", "/x/pack", "--max-context", "32768"],
                                        "env": {"SAGUARO_GLM_RAM_GB": "20", "MY_API_KEY": "secret"}}, indent=1))
        self.svc.config_path = str(self.cfg)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def req(self, method, path, body=None, headers=None):
        h = {"Content-Type": "application/json"} if body is not None else {}
        h.update(headers or {})
        r = urllib.request.Request(self.base + path, method=method, headers=h,
                                   data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                raw = resp.read()
                return resp.status, resp.headers, raw
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.headers, e.read()

    def wait_reload(self):
        for _ in range(100):
            s, _, b = self.req("GET", "/api/context")
            rl = json.loads(b)["reload"]
            if rl and rl["state"] in ("done", "failed"):
                return json.loads(b)
            time.sleep(0.05)
        self.fail("the reload did not end")

    def test_health_has_the_version(self):
        s, _, b = self.req("GET", "/health")
        self.assertEqual(json.loads(b)["version"], SAGUARO_VERSION)

    def test_index_busts_the_favicon_cache_with_the_version(self):
        s, _, b = self.req("GET", "/")
        html = b.decode()
        self.assertEqual(s, 200)
        self.assertIn(f"web/icon.svg?v={SAGUARO_VERSION}", html)
        self.assertNotIn("?v=__VERSION__", html)

    def test_index_still_serves_without_the_placeholder_or_the_version(self):
        """The replace keeps the page working either way: no VERSION (the version is None) yields ?v=, and an
        index.html without the placeholder is served byte-identical (the replace is a no-op)."""
        from unittest import mock
        from serve import server as SV
        with mock.patch.object(SV, "SAGUARO_VERSION", None):
            s, _, b = self.req("GET", "/")
        self.assertEqual(s, 200)
        self.assertIn(b'href="web/icon.svg?v="', b)                 # (no version after the ?v=)
        page = Path(self.tmp.name) / "serve" / "web"
        page.mkdir(parents=True)
        plain = '<link rel="icon" href="web/icon.svg?v=1.2.1f">'    # (an index.html that carries no placeholder)
        (page / "index.html").write_text(plain, encoding="utf-8")
        with mock.patch.object(SV, "ROOT", Path(self.tmp.name)):
            s, _, b = self.req("GET", "/")
        self.assertEqual((s, b.decode()), (200, plain))             # (served byte-identical)

    def test_context_range(self):
        s, _, b = self.req("GET", "/api/context")
        j = json.loads(b)
        self.assertEqual((s, j["context"], j["min"]), (200, 32768, CONTEXT_MIN))
        self.assertGreaterEqual(j["max"], 32768)
        self.assertEqual(j["tested"], 262144)                   # (the GLM engine's: the dashboard warns past it)

    def test_deepseek_context_range(self):
        """A DeepSeek pack: the context its ds4.txt says the model was trained for is the top (V4.1-Flash: 1M), and its
        engine is tested up to it."""
        pack = Path(self.tmp.name) / "maya-ds4"
        pack.mkdir()
        (pack / "ds4.txt").write_text("hidden_size 4096\nmax_position_embeddings 1048576\nsliding_window 128\n")
        self.assertEqual(trained_context_ds4(pack), 1048576)
        self.assertIsNone(trained_context_ds4(Path(self.tmp.name) / "nowhere"))
        (Path(self.tmp.name) / "bad").mkdir()
        (Path(self.tmp.name) / "bad" / "ds4.txt").write_text("max_position_embeddings many\n")
        self.assertIsNone(trained_context_ds4(Path(self.tmp.name) / "bad"))
        self.svc.trained_context = trained_context_ds4(pack)
        self.engine.info = dict(self.engine.info, engine_kind="ds4")
        s, _, b = self.req("GET", "/api/context")
        j = json.loads(b)
        self.assertEqual((j["max"], j["tested"]), (1048576, 1048576))
        s, _, _ = self.req("POST", "/api/context", {"max_context": 1048576})
        self.assertEqual(s, 202)
        self.assertEqual((self.wait_reload()["context"]), 1048576)
        s, _, _ = self.req("POST", "/api/context", {"max_context": 2097152})   # (past what it was trained for)
        self.assertEqual(s, 400)

    def test_qfn_context_range_is_the_elastic_yarn_reach(self):
        """C18: for a running saguaro-qfn the Context size is the elastic YaRN's reach: the 1M top, 1M tested (it
        runs live on a 4 x T4 box) and the yarn_above mark that says where the trained 256K ends (past it the page
        says "YaRN, extended"). The cost line's data rides along - the engine's own bytes a token and its pinned
        K/V - and a reload applies like any context change; nothing past the reach is accepted."""
        self.engine.info = dict(self.engine.info, engine_kind="qfn", kv_bytes_token=12672, kv_resident=32768)
        self.engine.vram = [{"kv_ram": 262144 * 12672}]           # (QFN's VRAM line's pinned-cache bytes)
        s, _, b = self.req("GET", "/api/context")
        j = json.loads(b)
        self.assertEqual((j["max"], j["tested"], j["yarn_above"]), (1048576, 1048576, 262144))
        self.assertEqual(j["kv_bytes_token"], 12672)               # (the server's cost line: the engine's count)
        self.assertAlmostEqual(j["kv_ram_gb"], 262144 * 12672 / 2**30, places=2)
        s, _, _ = self.req("POST", "/api/context", {"max_context": 1048576})
        self.assertEqual(s, 202)
        self.assertEqual(self.wait_reload()["context"], 1048576)
        s, _, _ = self.req("POST", "/api/context", {"max_context": 2097152})   # (past the YaRN reach)
        self.assertEqual(s, 400)

    def test_a_qfn_reload_saves_the_yarn_factor_with_the_context(self):
        """Review finding 1: a live slider move past 256K saves the factor that reaches the saved context too.
        The running command got the new --rope-scale (the driver's set_context) but the run config only ever
        saw --max-context - and the next start spawns from that file, past the reach of the factor written
        there, silently. A 1M move on the 512K config below saves --rope-scale 4 beside --max-context 1048576,
        and the mirror move back to 512K writes 2 with it (no over-large factor kept)."""
        self.cfg.write_text(json.dumps({"exe": "saguaro-qfn",
                                        "args": ["--pack", "/x/pack", "--max-context", "524288", "--kv", "int8",
                                                 "--rope-scaling", "yarn", "--rope-scale", "2",
                                                 "--yarn-orig-ctx", "262144"]}, indent=1))
        self.engine.info = dict(self.engine.info, engine_kind="qfn")
        self.req("POST", "/api/context", {"max_context": 1048576})
        self.assertEqual(self.wait_reload()["reload"]["state"], "done")
        args = json.loads(self.cfg.read_text())["args"]
        self.assertEqual(args[args.index("--max-context") + 1], "1048576")
        self.assertEqual(args[args.index("--rope-scale") + 1], "4")
        self.req("POST", "/api/context", {"max_context": 524288})
        self.assertEqual(self.wait_reload()["reload"]["state"], "done")
        args = json.loads(self.cfg.read_text())["args"]
        self.assertEqual(args[args.index("--rope-scale") + 1], "2")            # (the factor follows back down)

    def test_context_reports_the_engines_own_kv_bytes(self):
        """/api/context carries the engine's own K/V bytes (its INFO's kv_bytes_token / kv_ram_bytes) to the page: an
        engine that reports none leaves them null, and the page then shows no size (never a guessed one)."""
        self.engine.info = dict(self.engine.info, kv_bytes_token=1056, kv_ram_bytes=4096)
        j = json.loads(self.req("GET", "/api/context")[2])
        self.assertEqual((j["kv_bytes_token"], j["kv_ram_bytes"]), (1056, 4096))
        self.engine.info = {k: v for k, v in self.engine.info.items() if not k.startswith("kv_")}
        j = json.loads(self.req("GET", "/api/context")[2])
        self.assertIsNone(j["kv_bytes_token"])
        self.assertIsNone(j["kv_ram_bytes"])

    def test_reload_changes_the_context_and_saves_it(self):
        s, _, b = self.req("POST", "/api/context", {"max_context": 65536})
        self.assertEqual(s, 202)
        j = self.wait_reload()
        self.assertEqual((j["reload"]["state"], j["context"]), ("done", 65536))
        args = json.loads(self.cfg.read_text())["args"]
        self.assertEqual(args[args.index("--max-context") + 1], "65536")
        self.assertEqual(json.loads(self.cfg.read_text())["env"]["SAGUARO_GLM_RAM_GB"], "20")   # the rest kept
        self.assertGreaterEqual(j["load_s"], 0.3)       # (how long that load took: the page's estimate of the next)
        # the setup's own size is kept for the Models page's "Reset to the setup's values", through later reloads
        self.assertEqual(json.loads(self.cfg.read_text())["installer"]["as_set_up"]["context"], 32768)
        self.req("POST", "/api/context", {"max_context": 16384})
        self.wait_reload()
        self.assertEqual(json.loads(self.cfg.read_text())["installer"]["as_set_up"]["context"], 32768)

    def test_metrics_say_whether_deepseek_drafts(self):
        # DeepSeek's engine does not say it drafts with DSpark: the server does (the Monitor's Speculation), and only
        # for a DeepSeek config
        s, _, b = self.req("GET", "/metrics")
        self.assertNotIn("dspark", json.loads(b)["engine"])
        for on in (True, False):
            self.svc.dspark = on
            s, _, b = self.req("GET", "/metrics")
            self.assertIs(json.loads(b)["engine"]["dspark"], on)

    def test_metrics_list_the_drafters_that_run(self):
        # the Monitor's Speculation: the drafters this engine runs with, per model engine; empty = drafting is off
        from serve.server import request_spec_off, speculation_modes
        cases = [
            ({"engine_kind": "glm-fast", "mtp": 1, "dflash": 0}, {}, ["MTP"]),
            ({"engine_kind": "glm-fast", "mtp": 1, "dflash": 1}, {}, ["DFlash2"]),      # (DFlash decode takes the request first)
            ({"engine_kind": "glm-fast", "mtp": 0, "dflash": 1}, {}, ["DFlash2"]),
            ({"engine_kind": "glm-fast", "mtp": 0, "dflash": 0}, {}, []),
            ({"engine_kind": "qfn", "mtp": 1, "lookup": 3}, {}, ["MTP", "Lookup"]),
            ({"engine_kind": "qfn", "mtp": 0, "lookup": 0}, {}, []),
            ({"engine_kind": "mm3", "dspark": 1}, {}, ["DSpark"]),
            ({"engine_kind": "mm3", "dspark": 0}, {}, []),
            ({"engine_kind": "ds4"}, {"dspark": True, "lookup": True}, ["DSpark", "Lookup"]),
            ({"engine_kind": "ds4"}, {"dspark": False, "lookup": True}, ["Lookup"]),
            ({"engine_kind": "ds4"}, {"dspark": False, "lookup": False}, []),
        ]
        for info, kw, want in cases:
            with self.subTest(info=info, kw=kw):
                self.assertEqual(speculation_modes(info, **kw), want)
        self.assertTrue(request_spec_off({"strata_tune": {"spec": 1}}))
        self.assertFalse(request_spec_off({"strata_tune": {"spec": 4}}))
        self.assertFalse(request_spec_off({}))
        self.assertFalse(request_spec_off(None))
        s, _, b = self.req("GET", "/metrics")
        self.assertEqual(json.loads(b)["engine"]["speculation"], [])               # (the mock engine drafts nothing)
        self.engine.info.update(mtp=1)
        self.assertEqual(json.loads(self.req("GET", "/metrics")[2])["engine"]["speculation"], ["MTP"])

    def test_stat_lines_carry_the_draft_counts_and_the_ssd_bytes(self):
        # the engines' STAT lines (GLM, CYBER-FROST, DeepSeek, MiniMax): live draft_acc / draft_off and disk_gb reach the
        # Monitor's tiers.now as numbers
        class Stub:
            pass
        eng = Stub()
        StrataEngine._parse_stat(eng, "STAT tok_s=31.20 vram_hit=0.9100 vram_used=300 ram_used=500 ram_gb=20.3 "
                                      "vram_gb=12.5 disk_gb=31.4 draft_acc=124 draft_off=200")
        self.assertEqual((eng.stat["draft_acc"], eng.stat["draft_off"], eng.stat["disk_gb"]), (124.0, 200.0, 31.4))
        self.engine.stat = dict(eng.stat)
        now = json.loads(self.req("GET", "/metrics")[2])["tiers"]["now"]
        self.assertEqual((now["draft_acc"], now["draft_off"], now["disk_gb"]), (124.0, 200.0, 31.4))

    def test_requests_carry_the_drafts_offered_and_kept(self):
        # the page's last-answer acceptance: the engine's DONE counts reach the request row, None when it reported none
        self.engine.last = None
        self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        row = json.loads(self.req("GET", "/metrics")[2])["requests"][0]
        self.assertIn("drafts_offered", row)
        self.assertIn("spec_off", row)
        self.assertFalse(row["spec_off"])

    def test_chat_gets_503_while_reloading(self):
        os.environ["SAGUARO_MOCK_RELOAD_S"] = "1.5"
        self.req("POST", "/api/context", {"max_context": 16384})
        s, h, b = self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 503)
        self.assertEqual(h.get("Retry-After"), "30")
        self.assertIn("reloading", json.loads(b)["error"]["message"])
        s, _, b = self.req("GET", "/health")
        self.assertEqual(json.loads(b)["status"], "reloading")
        self.wait_reload()
        s, _, _ = self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 200)

    def test_out_of_range_and_foreign_pages_are_refused(self):
        s, _, _ = self.req("POST", "/api/context", {"max_context": 100})
        self.assertEqual(s, 400)
        s, _, _ = self.req("POST", "/api/context", {"max_context": 65536}, {"Origin": "http://evil.example"})
        self.assertEqual(s, 403)
        r = urllib.request.Request(self.base + "/api/context", method="POST", data=b"max_context=65536",
                                   headers={"Content-Type": "application/x-www-form-urlencoded"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(r, timeout=10)
        self.assertEqual(cm.exception.code, 415)
        cm.exception.close()

    def test_report_leaves_secrets_out(self):
        s, h, b = self.req("GET", "/api/report")
        text = b.decode()
        self.assertEqual(s, 200)
        self.assertIn("Saguaro", text)
        self.assertIn("SAGUARO_GLM_RAM_GB", text)   # (a config's settings under the names users see)
        self.assertNotIn("secret", text)

    def test_token_count_matches_the_prompt(self):
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "Hi! How can I help?"}, {"role": "user", "content": "count me"}]
        s, _, b = self.req("POST", "/api/tokens", {"messages": msgs, "reasoning_effort": "high"})
        j = json.loads(b)
        from serve.frontend import openai_to_messages
        m, tools, kw = openai_to_messages({"messages": msgs, "reasoning_effort": "high"})
        ids, _, _ = self.svc.prepare(m, tools, kw)
        self.assertEqual((s, j["tokens"], j["approximate"]), (200, len(ids), False))
        s, _, b = self.req("POST", "/api/tokens", {"messages": msgs + [{"role": "user", "content": [
            {"type": "text", "text": "and this"}, {"type": "image_url", "image_url": {"url": "data:,"}}]}]})
        self.assertTrue(json.loads(b)["approximate"])

    def test_tokenize_and_detokenize(self):
        """POST /tokenize and /detokenize in llama.cpp's and vLLM's forms: a text's ids (its special tokens' text as
        those tokens unless parse_special is false), a chat request's prompt as prepare() makes it, and the text back."""
        tok = ByteTokenizer()
        s, _, b = self.req("POST", "/tokenize", {"content": "hi<|im_end|>"})
        self.assertEqual((s, json.loads(b)["tokens"]), (200, tok.encode("hi<|im_end|>", parse_special=True)))
        s, _, b = self.req("POST", "/tokenize", {"model": "m", "prompt": "hi<|im_end|>", "parse_special": False})
        j = json.loads(b)
        self.assertEqual((j["tokens"], j["count"]), (tok.encode("hi<|im_end|>"), len("hi<|im_end|>")))
        self.assertEqual(j["max_model_len"], self.svc.engine.max_context)
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "count me"}]
        s, _, b = self.req("POST", "/tokenize", {"messages": msgs, "reasoning_effort": "high"})
        from serve.frontend import openai_to_messages
        m, tools, kw = openai_to_messages({"messages": msgs, "reasoning_effort": "high"})
        ids, _, _ = self.svc.prepare(m, tools, kw)
        self.assertEqual((s, json.loads(b)["tokens"]), (200, ids))
        s, _, b = self.req("POST", "/detokenize", {"tokens": ids})
        j = json.loads(b)
        self.assertEqual((s, j["content"]), (200, tok.decode(ids)))
        self.assertEqual(j["prompt"], j["content"])
        for path, body in (("/tokenize", {"content": 3}), ("/tokenize", {}), ("/detokenize", {"tokens": [1, -1]}),
                           ("/detokenize", {"tokens": "1,2"})):
            s, _, _ = self.req("POST", path, body)
            self.assertEqual(s, 400, (path, body))

    def test_peak_decode_speed(self):
        # the Monitor's peak decode: the fastest answer since the server started, of PEAK_MIN_TOKENS tokens or more
        from serve.server import PEAK_MIN_TOKENS
        s, _, b = self.req("GET", "/metrics")
        self.assertIsNone(json.loads(b)["totals"]["peak_decode"])
        end = self.engine.script[-1:]                                  # (its end-of-turn token)
        self.engine.script = self.engine.scripts[0] = ByteTokenizer().encode("</think>\n\n" + "a long enough answer " * 3) + end
        self.engine.tok_s = 4000
        rows = []
        for _ in range(3):
            self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
            rows.append(json.loads(self.req("GET", "/metrics")[2])["requests"][0])
        peak = json.loads(self.req("GET", "/metrics")[2])["totals"]["peak_decode"]
        top = max(rows, key=lambda r: r["decode_tok_s"])
        self.assertEqual((peak["tok_s"], peak["tokens"], peak["time"]), (top["decode_tok_s"], top["output_tokens"], top["time"]))
        self.assertGreaterEqual(peak["tokens"], PEAK_MIN_TOKENS)
        _, _, b = self.req("GET", "/metrics/prometheus")
        self.assertIn(f"maya:decode_peak_tokens_seconds {peak['tok_s']}", b.decode())
        # a short answer, however fast, is not the peak
        self.engine.tok_s = 400000
        self.req("POST", "/v1/chat/completions", {"model": "m", "max_tokens": 5,
                                                  "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(json.loads(self.req("GET", "/metrics")[2])["totals"]["peak_decode"], peak)

    def test_peak_decode_follows_the_live_figure(self):
        # a live Decode reading above the peak is the peak in the same /metrics answer (the owner's screenshot: 30.0
        # live, peak 27.8), rounded as shown; a reading before PEAK_MIN_TOKENS tokens is not
        from serve.server import PEAK_MIN_TOKENS, RequestStat
        svc, now = self.svc, time.time()
        svc.totals["peak_decode"] = {"tok_s": 27.8, "tokens": 400, "time": now - 60}
        r = RequestStat()
        r.first_token = now - 10
        with svc.status_lock:
            svc.status.update(busy=True, generated=PEAK_MIN_TOKENS - 1)
            svc.current_requests.add(r)
            svc.latest_request = r
        try:
            r.base = (now - 1, 0); r.samples.append((now, PEAK_MIN_TOKENS - 1))   # 31 tok/s, but only 31 tokens
            m = json.loads(self.req("GET", "/metrics")[2])
            self.assertEqual(m["totals"]["peak_decode"]["tok_s"], 27.8)
            for k, n in enumerate((300, 330, 310)):                   # 300, 330, 310 tok/s live, one a reading
                r.samples.clear(); r.base = (time.time() - 1, 1000 * k); r.samples.append((time.time(), 1000 * k + n))
                m = json.loads(self.req("GET", "/metrics")[2])
                self.assertGreaterEqual(m["totals"]["peak_decode"]["tok_s"], m["live"]["tok_s"])
            self.assertAlmostEqual(m["totals"]["peak_decode"]["tok_s"], 330, delta=2)
            self.assertEqual(m["totals"]["peak_decode"]["tokens"], 1330)
            self.assertEqual(round(m["totals"]["peak_decode"]["tok_s"], 1), m["totals"]["peak_decode"]["tok_s"])
        finally:
            with svc.status_lock:
                svc.status.update(busy=False)
                svc.current_requests.discard(r)

    def test_requests_record_their_api(self):
        self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        s, _, b = self.req("GET", "/metrics")
        self.assertEqual(json.loads(b)["requests"][0]["api"], "openai")


class Port(unittest.TestCase):
    """The dashboard's Port: saved into the config and Saguaro's start script for it; a busy, a reserved or the same port is
    refused with nothing saved; only Saguaro's own page may move the server.  (No restart here: saguaro.py did not start this
    server, so the port applies from the next start - the restart itself is saguaro.py's RESTART_EXIT, tested there.)"""

    def setUp(self):
        os.environ.pop("SAGUARO_RESTART_ON_UPDATE", None)
        Dashboard.setUp(self)
        self.script = Path(self.tmp.name) / "run-maya-test.bat"
        self.script.write_bytes(b"@echo off\r\npython saguaro.py --config " + str(self.cfg).encode() + b" --port 8080 %*\r\n")

    tearDown = Dashboard.tearDown
    req = Dashboard.req

    @staticmethod
    def free_port():
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def test_saved_and_reported(self):
        s, _, b = self.req("GET", "/api/port")
        info = json.loads(b)
        self.assertEqual((s, info["port"], info["restarts"]), (200, self.httpd.server_address[1], False))
        n = self.free_port()
        s, _, b = self.req("POST", "/api/port", {"port": n})
        self.assertEqual((s, json.loads(b)), (202, {"port": n, "restarts": False}))
        self.assertEqual(json.loads(self.cfg.read_text())["port"], n)
        self.assertEqual(json.loads(self.cfg.read_text())["args"][-1], "32768")   # (the rest kept)
        self.assertIn(f"--port {n} %*\r\n".encode(), self.script.read_bytes())   # (CR LF kept)

    def test_refused_with_nothing_saved(self):
        import socket
        busy = socket.socket()
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        try:
            for port in (busy.getsockname()[1], 80, self.httpd.server_address[1], "x"):
                s, _, b = self.req("POST", "/api/port", {"port": port})
                self.assertEqual(s, 400, port)
        finally:
            busy.close()
        s, _, _ = self.req("POST", "/api/port", {"port": self.free_port()}, {"Origin": "http://elsewhere.example"})
        self.assertEqual(s, 403)
        self.assertNotIn("port", json.loads(self.cfg.read_text()))
        self.assertIn(b"--port 8080 %*", self.script.read_bytes())


class ContextCost(unittest.TestCase):
    """The GLM engine's start lines give what its context costs in VRAM, summed over its GPUs."""

    def test_parsed_from_the_start_log(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.write_text("old start\nglm fast: CUDA0 VRAM before the expert pool: 4.0 of 31.7 GB in use - dense weights 3.1, "
                           "state/KV (16384 ctx) 0.40, activations 0.1\n"
                           "this start\nglm fast: CUDA0 VRAM before the expert pool: 4.4 of 31.7 GB in use - dense weights 3.1, "
                           "state/KV (32768 ctx) 0.66, activations 0.1\n"
                           "glm fast: CUDA1 VRAM before the expert pool: 4.3 of 31.7 GB in use - dense weights 2.9, "
                           "state/KV (32768 ctx) 0.65, activations 0.1\n")
            eng = StrataEngine.__new__(StrataEngine)
            eng.log_path, eng.info = str(log), {}
            eng._context_cost(log.read_text().index("this start"))
            self.assertEqual((eng.info["kv_ctx"], eng.info["kv_gb"]), (32768, 1.31))

    def test_set_context_rewrites_the_command(self):
        eng = StrataEngine.__new__(StrataEngine)
        eng.spawn = ("strata", ["--glm-pack", "p", "--max-context", "32768"], None, None, None)
        eng.set_context(131072)
        self.assertEqual(eng.spawn[1], ["--glm-pack", "p", "--max-context", "131072"])


class QfnContext(unittest.TestCase):
    """The dashboard's Context size on a running QFN engine: QfnEngine.set_context no longer refuses with
    "this engine's context is set in its config" (which made every reload fail) - the driver rewrites
    --max-context in its spawn command, and the shared _restart_with() starts the engine again with it,
    exactly as the other engines' set_context + restart do."""

    def engine(self, args):
        from serve.server import _qfn_engine_class
        cls = _qfn_engine_class()
        eng = cls.__new__(cls)                     # (the real driver, without starting a process)
        eng.spawn = ("saguaro-qfn", list(args), None, None, None, False, None)
        return eng

    def test_set_context_rewrites_the_command(self):
        eng = self.engine(["--pack", "p", "--max-context", "65536", "--kv", "int8"])
        eng.set_context(131072)
        self.assertEqual(eng.spawn[1], ["--pack", "p", "--max-context", "131072", "--kv", "int8"])
        eng.set_context(262144)                    # (again: replaced in place, never added a second time)
        self.assertEqual(eng.spawn[1].count("--max-context"), 1)

    def test_a_command_without_the_flag_gets_it(self):
        eng = self.engine(["--pack", "p"])
        eng.set_context(32768)
        self.assertEqual(eng.spawn[1], ["--pack", "p", "--max-context", "32768"])

    def test_a_past_256k_line_moves_its_yarn_factor_too(self):
        """Review finding 2: set_context's YaRN branch, run - the cases above never carried the pair, so the
        sync in serve/qfn_engine.py never executed here.  A 512K spawn line (the setup's --rope-scale 2)
        raised to 1M gets 4 with the new --max-context, back down gets 2, and a move to or under the trained
        262144 keeps the written factor (inside it the elastic engine never reads the YaRN table)."""
        eng = self.engine(["--pack", "p", "--max-context", "524288", "--rope-scaling", "yarn",
                           "--rope-scale", "2", "--yarn-orig-ctx", "262144"])
        eng.set_context(1048576)
        a = eng.spawn[1]
        self.assertEqual(a[a.index("--max-context") + 1], "1048576")
        self.assertEqual(a[a.index("--rope-scale") + 1], "4")
        eng.set_context(524288)
        self.assertEqual(eng.spawn[1][eng.spawn[1].index("--rope-scale") + 1], "2")   # (follows back down)
        eng.set_context(262144)
        self.assertEqual(eng.spawn[1][eng.spawn[1].index("--rope-scale") + 1], "2")   # (at it: stays)
        self.assertEqual(eng.spawn[1].count("--rope-scale"), 1)                       # (never a second pair)

    def test_the_reload_flow_reaches_it(self):
        # Service._restart_with: hasattr(engine, "set_context") -> stop, set_context, restart.  QfnEngine.stop is
        # the known no-op (each request stops its own slot); set_context is the driver's, and restart() re-runs
        # the spawn command it just rewrote.
        eng = self.engine(["--max-context", "65536"])
        eng.stop(60.0)
        eng.set_context(131072)                    # (before this change: ValueError, the reload always failed)
        self.assertEqual(eng.spawn[1], ["--max-context", "131072"])


QFN_CPP = (ROOT / "engines" / "qfn" / "src" / "program" / "generate.cpp").read_text(encoding="utf-8")
QFN_DRIVER = (ROOT / "engines" / "qfn" / "serve" / "server.py").read_text(encoding="utf-8")
QFN_GEN = (ROOT / "serve" / "qfn_engine.py").read_text(encoding="utf-8")
QFN_KBT, QFN_CTX = 12672, 1048576         # a QFN Q8 position's bytes (12 QSA layers x 1056 B), its 1M context

# saguaro-qfn's lines as T6a left them printing (the INFO main line, the engine_kind line, the solo-decode STAT, the
# elastic-slot HMEM/KMEM notes, and the --session-dir sess_json / turn_saved INFO notes)
QFN_INFO = ("INFO logit_bias=1 context=1048576 kv=q8_0 kv_resident=65536 kv_bytes_token=12672 expert_slots=16384 "
            "expert_cache_mib=52000 expert_slots_primary=12288 expert_cache_primary_mib=39000 spec=5 "
            "mtp_max=5 mtp=1 lookup=64 vram_free_mib=1200 cvec=1 arena_mib=70000 pool_workers=4 pcie_frac=0.62 "
            "spec_min_p=0.05 conversation_cache_mib=65536 conversation_cache_slots=8 "
            "conversation_cache_min_free_mib=2048 conversation_cache_min_tokens=4096 tail_role_token=0 "
            "vram_elastic=0 prefill_cap=8192 prefill_stages=1 stream_all_min=0 prefill_pipe=0 engine=0.1.49.2")
QFN_KIND = "INFO engine_kind=qfn experts=13824 vram_slots=16384 vram_gb=50.8 ram_slots=0 ram_gb=63.4"
QFN_STAT = ("STAT tok_s=21.40 ms_tok=46.73 vram_hit=0.9512 ram_fetch=0.487 disk=0.000 promo=0.031 vram_used=4096 "
            "vram_slots=5120 ram_used=9728 ram_gb=71.2 vram_gb=2.4 disk_gb=0.0 draft_acc=37 draft_off=58 active=1")
QFN_HMEM = "HMEM 3221225472 6442450944 2 1073741824 2147483648"
QFN_KMEM = "KMEM 1 0 4294967296 1073741824 536870912"
QFN_SESS = ('INFO sess_json={"files":2,"bytes":73400320,"max_bytes":1073741824,"max_files":16,"max_age_days":14,'
            '"conversations":[{"tokens":12345,"bytes":35651584,"last_used":1760000000,"state":"on_disk"}]}')
QFN_SAVE = "INFO sess_event=turn_saved sess_tokens=12345 sess_bytes=35651584 sess_ms=8.2 sess_why=main"
QFN_VRAM = (f"VRAM dev=0 pci=0000:01:00.0 total=25296896000 free=751828992 weights=6678575104 kv=665719603 "
            f"kv_ram={4 * 2**30} act=107374182 drafter=751828992 window=0 pool=15536793108 tail=0 borrow=0 vision=0 "
            f"slots=1231 ctx={QFN_CTX} layers=0-45 kv_room={600_000 * QFN_KBT} kv_ram_room={400_000 * QFN_KBT} "
            f"parked={10_000 * QFN_KBT} parked_n=3 park_budget={64 * 2**20}\n")


class QfnStatsToMonitor(unittest.TestCase):
    """T6b: a sample of saguaro-qfn's lines fed to its driver, then to a Service - every field the Monitor reads for
    a QFN engine (the Speculation name, the tiers' tiles incl. promo, the context RAM and slot memory, the session
    folder's state, the parked conversations' count and the last turn saves).  No GPU, no process."""

    class FakeProc:
        def __init__(self, lines):
            self.stdout = iter(lines)

        def kill(self):
            pass

    def driver(self, lines):
        from serve.server import _qfn_engine_class
        eng = _qfn_engine_class().__new__(_qfn_engine_class())     # (the real driver, its real _pump, a fake engine)
        eng.info, eng.vram, eng.vlen_lines = {}, [], None
        eng.lines, eng.slot_q, eng.log = queue.Queue(), [], None
        eng.proc = self.FakeProc(lines)
        eng._pump()
        return eng

    def test_the_lines_are_the_formats_the_engine_prints(self):
        """The fixtures above are the formats generate.cpp prints (T6a's mtp= and promo= included) and the driver
        still passes its special lines through into engine.info / the session folder's state."""
        self.assertIn("mtp_max=%d mtp=%d lookup=%d vram_free_mib=%lld", QFN_CPP)
        self.assertIn('"STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f ram_fetch=%.3f disk=0.000 promo=%.3f "', QFN_CPP)
        self.assertIn('"INFO engine_kind=qfn experts=%lld vram_slots=%lld vram_gb=%.1f ram_slots=%lld ram_gb=%.1f\\n"',
                      QFN_CPP)
        for src in (QFN_DRIVER, QFN_GEN):
            self.assertIn('self.info["context_ram"] = ram', src)
            self.assertIn('self.info["slot_memory"] = cards', src)
            self.assertIn('self.info["sessions"]', src)

    def test_driver_reads_them_into_the_engines_facts(self):
        eng = self.driver([QFN_INFO, QFN_KIND, QFN_HMEM, QFN_KMEM, QFN_STAT, QFN_SESS, QFN_SAVE])
        self.assertEqual((eng.info["mtp"], eng.info["lookup"], eng.info["engine_kind"]), (1, 64, "qfn"))
        self.assertEqual(speculation_modes(eng.info), ["MTP", "Lookup"])     # (the Speculation tile's name)
        self.assertEqual(eng.info["context_ram"], {"pinned": 3221225472, "whole": 6442450944,
                                                   "slots": [1073741824, 2147483648]})
        self.assertEqual(eng.info["slot_memory"], [{"device": 0, "total": 4294967296, "free": 1073741824,
                                                    "givable": 536870912}])
        self.assertEqual((eng.stat["promo"], eng.stat["ram_fetch"], eng.stat["disk"]), (0.031, 0.487, 0.0))
        self.assertEqual((eng.stat["draft_acc"], eng.stat["draft_off"]), (37.0, 58.0))   # (Speculation's figures)
        self.assertEqual(len(eng.stat_history), 1)
        self.assertEqual(eng.info["sessions"], {"files": 2, "bytes": 73400320, "max_bytes": 1073741824})
        self.assertEqual((eng.turn_saves["main"]["tokens"], eng.turn_saves["main"]["bytes"],
                          eng.turn_saves["main"]["ms"]), (12345, 35651584, 8.2))
        self.assertIn("at", eng.turn_saves["main"])

    def test_metrics_passes_every_field_the_page_reads(self):
        eng = self.driver([QFN_INFO, QFN_KIND, QFN_HMEM, QFN_KMEM, QFN_STAT, QFN_SESS, QFN_SAVE])
        qfn = SimpleNamespace(max_context=QFN_CTX, info=dict(eng.info), vram=[gpu_map.parse(QFN_VRAM)],
                              multi=0, batch=8, proc=SimpleNamespace(pid=4242), progress=None,
                              stat=eng.stat, stat_history=list(eng.stat_history), turn_saves=eng.turn_saves)
        m = Service(qfn, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja")).metrics()
        json.dumps(m)                                        # (every new field serialises into /metrics)
        self.assertEqual(m["engine"]["speculation"], ["MTP", "Lookup"])
        self.assertEqual(m["engine"]["context_ram"]["pinned"], 3221225472)
        self.assertEqual(m["engine"]["slot_memory"][0]["givable"], 536870912)
        self.assertEqual(m["engine"]["sessions"], {"files": 2, "bytes": 73400320, "max_bytes": 1073741824})
        self.assertEqual(m["tiers"]["now"]["promo"], 0.031)  # (the RAM tier's Promoted tile)
        self.assertEqual(m["tiers"]["history"]["promo"], [0.031])
        self.assertEqual(m["context"]["parked"]["conversations"], 3)   # (the parked conversations' count)
        self.assertEqual(m["context"]["parked"]["budget_bytes"], 64 * 2**20)
        self.assertEqual(m["context"]["turn_saves"]["main"]["tokens"], 12345)   # (the Context card's save line)


GB = 2**30
# the 3090 alone with EXL3-4.05bpw and DFlash2, as its start measured it (glm fast: CUDA0 VRAM before the expert pool ...)
LINE = ("VRAM dev=0 pci=0000:01:00.0 total=25296896000 free=751828992 weights=6678575104 kv=665719603 kv_ram=0 "
        "act=107374182 drafter=751828992 window=246415360 pool=15536793108 tail=7945689497 borrow=7945689497 "
        "vision=0 slots=1231 ctx=32768 layers=0-45\n")


class GpuMap(unittest.TestCase):
    """What fills each GPU: the engine's VRAM lines, joined with NVML's cards and processes (serve/gpu_map.py)."""

    def setUp(self):
        from serve import gpu_map
        self.gm = gpu_map
        self.r = gpu_map.parse(LINE)

    def card(self, used, procs, index=0, pci="0000:01:00"):
        return {"index": index, "pci": pci, "name": "NVIDIA GeForce RTX 3090", "used": used, "total": 25296896000,
                "procs": procs}

    def test_parse(self):
        self.assertEqual((self.r["dev"], self.r["pci"], self.r["layers"], self.r["slots"]), (0, "0000:01:00", [0, 45], 1231))
        self.assertEqual(self.r["pool"], 15536793108)
        self.assertIsNone(self.gm.parse("INFO context=32768"))
        self.assertEqual(self.gm.pci_key("00000000:0A:00.0"), "0000:0a:00")

    def test_hy4_tensor_parallel_lines(self):
        # maya-hy4's VRAM lines (Model::vram_lines): the head's GPU (the output head's share, the MTP block and its
        # scratch as the drafter, a helper's slices of the experts) and eight members of one tensor-parallel group, each
        # with slices of every layer's experts - layers [0, 42) - and its group's size; they hold the same slots, so the
        # room a member has beyond the group's least free stays idle
        head = self.gm.parse(f"VRAM dev=0 pci=0000:3b:00.0 total={24 * GB} free={4 * GB} weights={GB // 3} kv=0 kv_ram=0 "
                             f"act=0 drafter={6 * GB} window=0 pool={9 * GB} helper=0 tail=0 borrow=0 vision=0 "
                             f"vision_own=0 prompt={2 * GB} tp=0 tp_n=1 slots=13981 ctx=262144 layers=0-42")
        members = [self.gm.parse(f"VRAM dev={d} pci=0000:{0x40 + d:02x}:00.0 total={16 * GB} free={GB // (2 + d % 2)} "
                                 f"weights={GB} kv={2 * GB} kv_ram=0 act=0 drafter={GB // 50} window=0 pool={9 * GB} "
                                 f"helper=0 tail=0 borrow=0 vision=0 vision_own=0 prompt={GB} tp=0 tp_n=8 slots=14022 "
                                 f"ctx=262144 layers=0-42") for d in range(1, 9)]
        self.assertEqual((head["tp_n"], head["layers"], head["slots"], head["drafter"]), (1, [0, 42], 13981, 6 * GB))
        gs = self.gm.view([head] + members, [], engine_pid=100)
        self.assertEqual(len(gs), 9)
        self.assertEqual({k: v for k, v in gs[0]["parts"].items() if v and k != "rest"},
                         {"weights": GB // 3, "prompt": 2 * GB, "drafter": 6 * GB, "pool": 9 * GB})
        self.assertEqual(gs[1]["parts"]["kv"], 2 * GB)

    def test_helper_pool_is_accounted_separately_from_drafter_and_rest(self):
        helper = 20_000_000_000
        r = self.gm.parse(f"VRAM dev=2 pci=0000:0f:00.0 total={32 * GB} free={5 * GB} "
                          f"weights=0 kv=0 act=0 drafter={4 * GB} pool=0 helper={helper} slots=0 layers=-1--1")
        (g,) = self.gm.view([r], [], engine_pid=100)
        self.assertEqual(g["parts"]["helper"], helper)
        self.assertEqual(g["parts"]["drafter"], 4 * GB)
        self.assertEqual(g["parts"]["rest"], 23 * GB - helper)
        self.assertIsNone(g["layers"])
        self.assertEqual(sum(g["parts"].values()) + g["free"], g["total"])
        merged = self.gm.merge([r, dict(dev=2, helper=1024, free=5 * GB)])
        self.assertEqual(merged[0]["helper"], helper + 1024)

    def test_helper_pool_and_primary_pool_on_one_card_do_not_overlap(self):
        r = dict(self.r, helper=2 * GB)
        engine = sum(r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool", "helper"))
        card = dict(self.card(engine + GB, {100: engine + GB}), total=32 * GB)
        (g,) = self.gm.view([r], [card], engine_pid=100)
        self.assertEqual(g["parts"]["helper"], 2 * GB)
        self.assertEqual(g["parts"]["pool"], r["pool"])
        self.assertEqual(g["parts"]["rest"], GB)
        self.assertEqual(sum(g["parts"].values()) + g["free"], g["total"])

    def test_nvml_tells_the_engine_from_other_programs(self):
        engine = sum(self.r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        own, other = engine + 400 * 2**20, 600 * 2**20
        (g,) = self.gm.view([self.r], [self.card(own + other, {100: own, 200: other})], engine_pid=100)
        self.assertEqual((g["index"], g["measured"]), (0, "nvml"))
        self.assertEqual(g["parts"]["rest"], 400 * 2**20)        # the CUDA context and the small buffers
        self.assertEqual(g["parts"]["other"], other)
        self.assertEqual(g["parts"]["drafter"], self.r["drafter"] + self.r["window"])
        self.assertEqual(g["free"], g["total"] - own - other)
        self.assertEqual(sum(g["parts"].values()) + g["free"], g["total"])

    def test_the_drivers_reserve_is_not_another_program(self):
        # (9 GPUs, the engine alone on each: NVML's used memory ~0.58 GB beyond its process - the driver's reserve)
        engine = sum(self.r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        own, reserve = engine + 500 * 2**20, 590 * 2**20
        (g,) = self.gm.view([self.r], [self.card(own + reserve, {100: own})], engine_pid=100)
        self.assertEqual((g["parts"]["other"], g["parts"]["rest"]), (0, 500 * 2**20 + reserve))

    def test_found_by_pci_not_by_place(self):
        cards = [self.card(1 * GB, {}, index=0, pci="0000:02:00"), self.card(23 * GB, {100: 22 * GB}, index=1)]
        (g,) = self.gm.view([self.r], cards, engine_pid=100)
        self.assertEqual(g["index"], 1)

    def test_without_nvml_the_start_measurement(self):
        (g,) = self.gm.view([self.r], [], engine_pid=100)
        start = self.r["total"] - self.r["free"]
        engine = sum(self.r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        self.assertEqual((g["measured"], g["parts"]["rest"], g["parts"]["other"]), ("start", start - engine, 0))
        self.assertEqual(g["free"], self.r["free"])

    def test_the_vision_encoder_has_the_tail(self):
        r = dict(self.r, vision=self.r["tail"])
        engine = sum(r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool")) - r["tail"]
        own, enc = engine + 400 * 2**20, r["tail"] - 50 * 2**20
        (g,) = self.gm.view([r], [self.card(own + enc, {100: own, 300: enc})], 100, vision_pid=300, vision_lent=True)
        self.assertTrue(g["vision_lent"])
        self.assertEqual((g["parts"]["pool"], g["parts"]["vision"]), (r["pool"] - r["tail"], enc))
        self.assertEqual(g["parts"]["other"], 0)

    def test_a_vision_tower_in_the_engines_process(self):
        # (strata-ds4: the tower on the first GPU inside the engine - vision_own, not a pool tail lent to another process)
        r = dict(self.r, vision_own=318 * 2**20)
        engine = sum(r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        own = engine + r["vision_own"] + 400 * 2**20
        (g,) = self.gm.view([r], [self.card(own, {100: own})], engine_pid=100)
        self.assertEqual((g["parts"]["vision"], g["vision_own"]), (r["vision_own"], r["vision_own"]))
        self.assertEqual((g["parts"]["rest"], g["parts"]["other"]), (400 * 2**20, 0))   # the CUDA context, not vision
        (s,) = self.gm.view([r], [], engine_pid=100)             # (no NVML: the start's measurement)
        self.assertEqual(s["parts"]["vision"], r["vision_own"])
        self.assertEqual(sum(s["parts"].values()) + s["free"], s["total"])

    def test_an_engine_run_through_a_launcher(self):
        # (the engine's command a script that runs `nsys launch` (pid 63), which starts the engine (85) as a grandchild:
        # NVML lists 85 - found among the launcher's tree; the launcher's pid alone found nothing, and the engine's own
        # growth since the start's measurement was drawn as other programs)
        engine = sum(self.r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        own, reserve, other = engine + 500 * 2**20, 128 * 2**20, 600 * 2**20
        tree = {63, 84, 85}
        (g,) = self.gm.view([self.r], [self.card(own + reserve, {85: own})], engine_pid=tree)
        self.assertEqual((g["measured"], g["parts"]["other"], g["parts"]["rest"]), ("nvml", 0, 500 * 2**20 + reserve))
        (g,) = self.gm.view([self.r], [self.card(own + reserve + other, {85: own, 200: other})], engine_pid=tree)
        self.assertEqual((g["parts"]["other"], g["parts"]["rest"]), (other, 500 * 2**20 + reserve))   # a real one
        r = dict(self.r, vision=self.r["tail"])                  # (the on-demand encoder, run through one too)
        enc = r["tail"] - 50 * 2**20
        own = engine - r["tail"] + 400 * 2**20
        (g,) = self.gm.view([r], [self.card(own + enc, {85: own, 301: enc})], tree, vision_pid={300, 301},
                            vision_lent=True)
        self.assertEqual((g["parts"]["vision"], g["parts"]["other"], g["parts"]["rest"]), (enc, 0, 400 * 2**20))

    def test_the_engine_not_listed_the_start_measurement(self):
        # (NVML lists the engine under a pid the server can't name - a pid namespace it does not share: the card is
        # drawn from the start's measurement, as without NVML)
        engine = sum(self.r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        (g,) = self.gm.view([self.r], [self.card(engine + GB, {450462: engine + GB})], engine_pid=65)
        start = self.r["total"] - self.r["free"]
        self.assertEqual((g["measured"], g["parts"]["rest"]), ("start", start - engine))

    def test_never_more_than_the_card(self):
        (g,) = self.gm.view([self.r], [self.card(26 * GB, {100: 25 * GB, 200: 2 * GB})], engine_pid=100)
        self.assertLessEqual(sum(g["parts"].values()), g["total"])
        self.assertEqual(g["free"], 0)

    def test_strata_ds4s_prompt_buffers_and_tensor_parallel(self):
        # (strata-ds4 at --tp 8: a 5060 Ti of the group of 8 - its prompt chunks' buffers and tensor parallel's copies
        # told apart, the GLM engine's lines without them)
        r = self.gm.parse("VRAM dev=3 pci=0000:05:00.0 total=16610951168 free=301989888 weights=268435456 kv=33554432 "
                          "kv_ram=0 act=134217728 drafter=0 window=0 pool=13421772800 tail=0 borrow=0 vision=0 "
                          "vision_own=0 prompt=1610612736 tp=805306368 tp_n=8 slots=3343 ctx=262144 layers=3-39")
        (g,) = self.gm.view([r], [], engine_pid=100)
        self.assertEqual((g["parts"]["prompt"], g["parts"]["tp"], g["tp_n"]), (1610612736, 805306368, 8))
        self.assertEqual(sum(g["parts"].values()) + g["free"], g["total"])
        (h,) = self.gm.view([self.r], [], engine_pid=100)
        self.assertEqual((h["parts"]["prompt"], h["parts"]["tp"], h["tp_n"]), (0, 0, 0))

    def test_two_parts_on_one_card_add_up(self):
        a = dict(self.r, layers=[0, 20], slots=500)
        b = dict(self.r, layers=[20, 45], slots=700, free=self.r["free"] - GB)
        (m,) = self.gm.merge([a, b])
        self.assertEqual((m["layers"], m["slots"], m["pool"], m["free"]), ([0, 45], 1200, 2 * self.r["pool"], b["free"]))

    def test_metrics_carry_it(self):
        from types import SimpleNamespace
        tok = ByteTokenizer()
        eng = MockEngine(tok, "hi", max_context=32768)
        eng.vram, eng.proc = [self.r], SimpleNamespace(pid=100)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        card = self.card(24 * GB, {100: 23 * GB})
        svc.telemetry = SimpleNamespace(snapshot=lambda: {"now": {"gpu_mem_used": 24 * GB, "vram": [card]},
                                                          "history": {}, "static": {}})
        m = svc.metrics()
        self.assertNotIn("vram", m["hardware"])                  # (the processes stay in the server)
        self.assertEqual(m["gpu_map"][0]["slots"], 1231)
        json.dumps(m)
        eng.vram = []
        self.assertIsNone(svc.metrics()["gpu_map"])             # (an engine without VRAM lines: no card)

    def test_later_lines_follow_a_picture(self):
        """maya-ds4 prints its VRAM lines again when a picture's room is lent to its vision tower and when it is given
        back: the map of that GPU follows them, the others stay.  The stamp on the entry says how fresh that card's
        split is (PLAN-vram-live: every engine re-prints a line whose split moved)."""
        from types import SimpleNamespace
        from serve.server import StrataEngine
        other = dict(self.r, dev=1, pci="0000:02:00")
        eng = SimpleNamespace(vram=[self.r, other])
        plain = lambda vs: [{k: v for k, v in p.items() if k != "at"} for p in vs]   # noqa: E731
        lent = LINE.replace("vision=0", "vision=0 vision_own=334495744").replace("pool=15536793108", "pool=15202297364")
        StrataEngine._vram_line(eng, lent)
        self.assertEqual((eng.vram[0]["vision_own"], eng.vram[0]["pool"], eng.vram[1]), (334495744, 15202297364, other))
        StrataEngine._vram_line(eng, LINE)
        self.assertEqual(plain(eng.vram), [self.r, other])
        self.assertIsNotNone(eng.vram[0].get("at"))              # (the followed card carries its stamp, the rest not)
        self.assertNotIn("at", eng.vram[1])
        StrataEngine._vram_line(eng, "VRAM nonsense\n")          # (not a GPU's line: nothing changes)
        self.assertEqual(plain(eng.vram), [self.r, other])

    def test_split_age_says_how_fresh_the_split_is(self):
        """(PLAN-vram-live) the card's title shows the age of its split: the stamp of the line the entry holds, of a
        merged card the older part's; no stamp or no clock: no age."""
        r = dict(self.r, at=1000.0)
        self.assertEqual(self.gm.view([r], [], 100, now=1003.4)[0]["split_age"], 3.4)
        self.assertIsNone(self.gm.view([r], [], 100)[0]["split_age"])              # (the server asked without a clock)
        self.assertIsNone(self.gm.view([self.r], [], 100, now=1003.4)[0]["split_age"])   # (a line never re-printed)
        b = dict(self.r, layers=[46, 47], at=1001.0)
        self.assertEqual(self.gm.merge([r, b])[0]["at"], 1000.0)                    # (as old as its oldest part)
        self.assertEqual(self.gm.view([r, b], [], 100, now=1010.0)[0]["split_age"], 10.0)


class GpuRelease(unittest.TestCase):
    """A reload starts the new engine only once the GPUs have dropped the old one (its memory is freed after it ends)."""

    def test_waits_until_no_gpu_lists_the_engine(self):
        from unittest import mock
        from serve import server as SV, telemetry as T
        seen = [{4242, 7}, {4242, 7}, {7}]
        with mock.patch.object(T, "gpu_pids", side_effect=lambda: seen.pop(0) if len(seen) > 1 else seen[0]), \
                mock.patch.object(SV.time, "sleep") as sleep, \
                mock.patch.object(SV.sys, "platform", "linux"):
            SV.wait_gpu_release(4242, timeout=30)
        self.assertEqual(seen, [{7}])
        self.assertEqual(sleep.call_count, 2)

    def test_without_nvml_a_short_wait(self):
        from unittest import mock
        from serve import server as SV, telemetry as T
        with mock.patch.object(T, "gpu_pids", return_value=None), mock.patch.object(SV.time, "sleep") as sleep, \
             mock.patch.object(SV.sys, "platform", "linux"):
            SV.wait_gpu_release(4242)
        sleep.assert_called_once()

    def test_mac_waits_until_free_memory_stops_rising(self):
        from unittest import mock
        from serve import server as SV
        gb = 1 << 30
        free = [4 * gb, 20 * gb, 40 * gb, 40 * gb, 90 * gb]
        with mock.patch.object(SV, "mac_free_bytes", side_effect=lambda: free.pop(0)), \
                mock.patch.object(SV.time, "sleep") as sleep, mock.patch.object(SV.sys, "platform", "darwin"):
            SV.wait_gpu_release(4242)
        self.assertEqual(sleep.call_count, 3)                   # rising, rising, settled: started
        self.assertEqual(free, [90 * gb])

    def test_mac_free_bytes_reads_vm_stat(self):
        from unittest import mock
        from serve import server as SV
        out = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:      10.\n"
               "Pages active:   99.\nPages inactive:  20.\nPages speculative:  3.\nPages purgeable:  2.\n")
        with mock.patch.object(SV.subprocess, "run", return_value=mock.Mock(stdout=out)):
            self.assertEqual(SV.mac_free_bytes(), 35 * 16384)
        with mock.patch.object(SV.subprocess, "run", return_value=mock.Mock(stdout="")):
            self.assertIsNone(SV.mac_free_bytes())


class NvmlPids(unittest.TestCase):
    """The GPU map finds an engine run through a launcher: NVML lists a process the launcher started, not it."""

    def test_its_own_pid_when_listed_or_nothing_is(self):
        from unittest import mock
        from serve import server as SV
        with mock.patch.dict(sys.modules, {"psutil": None}):     # (nothing is looked up then)
            self.assertEqual(SV.nvml_pids(85, {85, 200}), {85})
            self.assertEqual(SV.nvml_pids(63, set()), {63})
            self.assertEqual(SV.nvml_pids(None, {85}), set())
            self.assertEqual(SV.nvml_pids(63, {85}), {63})        # (no psutil: the pid alone)

    def test_the_processes_a_launcher_started(self):
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil is not installed")
        import subprocess
        from serve import server as SV
        # (a launcher that does not exec: python starting python, which sleeps)
        launcher = subprocess.Popen([sys.executable, "-c", "import subprocess, sys; subprocess.run([sys.executable, "
                                     "'-c', 'import sys, time; print(flush=True); time.sleep(60)'])"],
                                    stdout=subprocess.PIPE, text=True)
        try:
            launcher.stdout.readline()                           # (the grandchild runs)
            pids = SV.nvml_pids(launcher.pid, {1 << 30})
            self.assertIn(launcher.pid, pids)
            self.assertGreater(len(pids), 1)                     # (and the grandchild: a venv's on Windows, more)
        finally:
            for p in psutil.Process(launcher.pid).children(recursive=True):
                p.kill()
            launcher.kill()
            launcher.wait()
            launcher.stdout.close()


class Offline(unittest.TestCase):
    """The page works without the internet: its scripts, styles, fonts and icons are all in serve/web (Preact, htm
    and Lucide vendored with their notices) - nothing is loaded from another host."""
    WEB = ROOT / "serve" / "web"

    def test_nothing_loaded_from_the_network(self):
        import re
        remote = re.compile(r"""(?:\b(?:src|href|action)\s*=\s*|\burl\(\s*|\bfrom\s+|\bimport\s*\(\s*|"""
                            r"""\bfetch\(\s*|@import\s+|new\s+(?:WebSocket|EventSource|Worker)\(\s*)["'`]?(?:https?:)?//""",
                            re.I)
        found = []
        for p in sorted(self.WEB.rglob("*")):
            if p.suffix not in (".js", ".css", ".html", ".svg", ".webmanifest"):
                continue
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if remote.search(line) and "w3.org/" not in line:
                    found.append(f"{p.relative_to(ROOT)}:{n}: {line.strip()[:120]}")
        self.assertEqual(found, [])

    def test_the_vendored_code_keeps_its_notices(self):
        self.assertIn("The MIT License", (self.WEB / "preact.js").read_text(encoding="utf-8")[:3000])
        self.assertIn("Apache License, Version 2.0", (self.WEB / "preact.js").read_text(encoding="utf-8")[:3000])
        self.assertIn("ISC License", (self.WEB / "LICENSE.lucide.txt").read_text(encoding="utf-8"))
        self.assertTrue((self.WEB / "fonts" / "OFL.txt").is_file())
        index = (self.WEB.parent / "web" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("fonts.googleapis", index)

    def test_about_shows_only_saguaros_version(self):
        # the engine's own version is no longer printed to users (About > Version); Saguaro's own stays
        about = (self.WEB / "settings.js").read_text(encoding="utf-8")
        self.assertIn("Saguaro v${ver}", about)
        self.assertNotIn("engine ${eng.version}", about)

    def test_the_index_favicon_version_is_the_placeholder(self):
        # the favicon's cache-bust version comes from VERSION at serve time: the file carries the placeholder, not a hard-coded one
        index = (self.WEB / "index.html").read_text(encoding="utf-8")
        self.assertIn("?v=__VERSION__", index)
        self.assertNotIn("1.2.1f", index)


# the page's own modules in Node (skipped without it), the browser's few globals they touch at load stubbed out
PAGE_STUBS = """
const noop = () => {};
const el = new Proxy({}, {get: (t, k) => (k in t ? t[k] : k === "dataset" || k === "style" ? (t[k] = {})
  : k === "classList" ? (t[k] = {add: noop, remove: noop, toggle: noop}) : noop)});
globalThis.navigator = {platform: "linux", userAgent: "node"};   // (lib.js' MOD: platform or user agent)
globalThis.matchMedia = () => ({matches: false, addEventListener: noop, addListener: noop});
globalThis.localStorage = {getItem: () => null, setItem: noop, removeItem: noop};
const docL = {};                                                   // (the document's listeners: a test can fire them)
globalThis.document = {documentElement: el, body: el, head: el, hidden: false,
                       addEventListener: (k, f) => { (docL[k] = docL[k] || []).push(f); },
                       removeEventListener: (k, f) => { const a = docL[k] || [], i = a.indexOf(f); if (i >= 0) a.splice(i, 1); },
                       querySelector: () => null, createElement: () => el};
globalThis.fire = (k) => { for (const f of (docL[k] || []).slice()) f(); };   // (as the browser fires it on a tab change)
globalThis.window = globalThis; globalThis.addEventListener = noop;
globalThis.location = {href: "http://x/", hash: "", search: "", pathname: "/"};
const web = process.argv[2];
const st = await import(web + "/state.js"), sv = await import(web + "/server.js");
"""


@unittest.skipUnless(__import__("shutil").which("node"), "Node.js is not installed")
class PageLogic(unittest.TestCase):
    def node(self, body):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.mjs"
            p.write_text(PAGE_STUBS + body, encoding="utf-8")
            r = subprocess.run(["node", str(p), (ROOT / "serve" / "web").resolve().as_uri()], capture_output=True,
                               encoding="utf-8", timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_thinking_pill_covers_the_chosen_level_only(self):
        # the owner's phone: Low chosen, the fill spanned Off and Low.  Every model's levels, every level: the pill sits
        # on that level's segment alone
        got = self.node("""
const out = {};
for (const model of ["glm-5.3-flash", "deepseek-v4.1-flash", "minimax-m3.1-flash", "cyber-frost-3.8-ps"]) {
  st.setFamily(model);
  out[model] = st.LEVELS.map((l) => [l.id, st.thinkPill(l.id)]);
}
console.log(JSON.stringify(out));""")
        self.assertEqual([lv for lv, _ in got["glm-5.3-flash"]], ["none", "low", "high", "max"])
        self.assertEqual(len(got["minimax-m3.1-flash"]), 7)
        for model, levels in got.items():
            n = len(levels)
            for i, (lv, pill) in enumerate(levels):
                with self.subTest(model=model, level=lv):
                    self.assertEqual((pill["i"], pill["n"]), (i, n))
                    self.assertAlmostEqual(pill["at"], 100 * i / n)
                    self.assertAlmostEqual(pill["width"], 100 / n)      # (one segment wide: not from 0 up to it)
        self.assertEqual(got["glm-5.3-flash"][1][1]["at"], 25)            # Low: the second quarter only

    def test_one_model_label_names_the_model_and_its_files(self):
        # item 9: the one label every page names a model with - the server's two fields, the quant exactly as named
        # ("Q8_0", never "Q8 0"); the {name, quant} pair where the quant gets a line of its own; a server without
        # the fields keeps today's text, and a page row (Models, Saved) carries model + quant
        got = self.node("""
const lib = await import(web + "/lib.js");
const qfn = {model: "qwen3.8-flash-next", model_name_pretty: "Qwen3.8 Flash Next", model_quant: "Q8_0",
             model_label: "Qwen3.8-Flash-Next · Q8_0 (saguaro-qfn)"};
console.log(JSON.stringify({
  qfn: lib.modelLabel(qfn, {compact: true}),
  two: [lib.modelLabel(qfn).name, lib.modelLabel(qfn).quant],
  ud: lib.modelLabel({model_name_pretty: "Qwen3.8 Flash Next", model_quant: "UD-Q4_K_XL"}, {compact: true}),
  maya: lib.modelLabel({model_name_pretty: "GLM 5.3", model_quant: "Maya-L"}, {compact: true}),
  frost: lib.modelLabel({model_name_pretty: "Qwen3.8 Flash Next", model_quant: "CYBER-FROST-3.8 Q5_K_M"}, {compact: true}),
  byHand: lib.modelLabel({model: "qwen3.8-flash-next", model_name_pretty: "Qwen3.8 Flash Next",
                          model_quant: ""}, {compact: true}),
  row: lib.modelLabel({model: "minimax-m3.1-flash", quant: "Maya-XS"}),
  oldLabel: lib.modelLabel({model: "minimax-m3.1-flash", model_label: "MiniMax M3.1 · Maya-XS"}, {compact: true}),
  oldPlain: lib.modelLabel({model: "qwen3.8-flash-next"}, {compact: true}),
  none: lib.modelLabel({}),
}));""")
        self.assertEqual(got["qfn"], "Qwen3.8 Flash Next (Q8_0)")      # ("Q8_0" as named - not "Q8 0")
        self.assertEqual(got["two"], ["Qwen3.8 Flash Next", "Q8_0"])   # (the Models list: a line of its own)
        self.assertEqual(got["ud"], "Qwen3.8 Flash Next (UD-Q4_K_XL)")
        self.assertEqual(got["maya"], "GLM 5.3 (Maya-L)")
        self.assertEqual(got["frost"], "Qwen3.8 Flash Next (CYBER-FROST-3.8 Q5_K_M)")
        self.assertEqual(got["byHand"], "Qwen3.8 Flash Next")          # (a hand-made config: no quant to say)
        self.assertEqual(got["row"], {"name": "MiniMax M3.1 Flash", "quant": "Maya-XS"})
        self.assertEqual(got["oldLabel"], "MiniMax M3.1 · Maya-XS")    # (an old server: today's text)
        self.assertEqual(got["oldPlain"], "Qwen3.8 Flash Next")        # (... or the id as words, as before)
        self.assertEqual(got["none"], {"name": "", "quant": ""})

    def test_speculation_says_off_waiting_or_the_acceptance(self):
        # the Monitor's Speculation, three states: off (no drafter, or the request asked for none: "off" in the field and
        # in the figure), drafting on with nothing drafted yet ("-", never a fake 0%), drafting on with drafts (accepted of
        # drafted, %) - live from the STAT draft counts, else the last answer's
        got = self.node("""
const now = 1000;
const eng = (modes) => ({speculation: modes});
const live = (o) => ({state: "generating", elapsed_s: 5, ...o});
const stat = (o) => ({now: {time: now - 1, ...o}});
const v = (m) => sv.speculationView(m, now);
const out = {
  none: v({engine: eng([]), live: {state: "idle"}, requests: []}),
  noneLive: v({engine: eng([]), live: live({}), tiers: stat({draft_acc: 0, draft_off: 0}), requests: []}),
  specOne: v({engine: eng(["MTP"]), live: live({spec_off: true}), tiers: stat({draft_acc: 0, draft_off: 0}), requests: []}),
  lastSpecOne: v({engine: eng(["MTP"]), live: {state: "idle"}, requests: [{spec_off: true, drafts_offered: 0, drafts_accepted: 0}]}),
  idleNever: v({engine: eng(["MTP"]), live: {state: "idle"}, requests: []}),
  idleZero: v({engine: eng(["MTP"]), live: {state: "idle"}, requests: [{drafts_offered: 0, drafts_accepted: 0}]}),
  idleNull: v({engine: eng(["DFlash2"]), live: {state: "idle"}, requests: [{drafts_offered: null, drafts_accepted: null}]}),
  liveNone: v({engine: eng(["MTP"]), live: live({}), tiers: stat({draft_acc: 0, draft_off: 0}), requests: []}),
  liveNoKeys: v({engine: eng(["Lookup"]), live: live({}), tiers: stat({vram_hit: 1}), requests: []}),
  liveOld: v({engine: eng(["MTP"]), live: live({elapsed_s: 5}), tiers: {now: {time: now - 100, draft_acc: 9, draft_off: 10}}, requests: []}),
  beforeThis: v({engine: eng(["MTP"]), live: live({elapsed_s: 2}), tiers: {now: {time: now - 20, draft_acc: 9, draft_off: 10}}, requests: []}),
  live: v({engine: eng(["MTP"]), live: live({}), tiers: stat({draft_acc: 124, draft_off: 200}), requests: []}),
  liveAllKept: v({engine: eng(["DSpark", "Lookup"]), live: live({}), tiers: stat({draft_acc: 50, draft_off: 50}), requests: []}),
  liveRatio: v({engine: eng(["DSpark"]), live: live({}), tiers: stat({drafts: 0.5}), requests: []}),
  zeroKept: v({engine: eng(["MTP"]), live: live({}), tiers: stat({draft_acc: 0, draft_off: 40}), requests: []}),
  last: v({engine: eng(["MTP"]), live: {state: "idle"}, requests: [{drafts_offered: 1200, drafts_accepted: 600}]}),
  oldServer: v({engine: {}, live: {state: "idle"}, requests: []}),
};
console.log(JSON.stringify(out));""")
        for k in ("none", "noneLive", "specOne"):                              # state 1: off, in the field and the figure
            self.assertEqual((got[k]["state"], got[k]["name"], got[k]["text"]), ("off", "off", "off"), k)
        self.assertEqual((got["lastSpecOne"]["state"], got["lastSpecOne"]["name"], got["lastSpecOne"]["text"]),
                         ("off", "MTP", "off"))                                # (idle: the engine's drafter, the last answer had none)
        for k in ("idleNever", "idleZero", "idleNull", "liveNone", "liveNoKeys", "liveOld", "beforeThis"):
            self.assertEqual((got[k]["state"], got[k]["text"]), ("waiting", "\u2014"), k)   # state 2: on, nothing drafted yet
        self.assertEqual(got["idleNever"]["name"], "MTP")
        self.assertEqual(got["idleNull"]["name"], "DFlash2")
        self.assertEqual((got["live"]["state"], got["live"]["text"], got["live"]["name"]), ("kept", "62% (124 of 200)", "MTP"))
        self.assertEqual(got["liveAllKept"]["text"], "100% (50 of 50)")
        self.assertEqual(got["liveAllKept"]["name"], "DSpark + Lookup")
        self.assertEqual(got["liveRatio"]["text"], "50%")                       # (an engine that gives the share only)
        self.assertEqual(got["zeroKept"]["text"], "0% (0 of 40)")                  # a real 0%: drafted, none kept
        self.assertEqual(got["last"]["text"], "50% (600 of 1,200)")
        self.assertEqual(got["oldServer"]["state"], "none")

    def test_experts_on_the_ssd_have_their_bytes_like_the_other_tiers(self):
        got = self.node("""
const n = {vram_used: 300, ram_used: 500, vram_gb: 12.5, ram_gb: 20.25, disk_gb: 31.4};
const ds = {vram_used: 40, ram_used: 100, cpu_n: 400, vram_gb: 9, ram_gb: 25, disk_gb: 75.5};
console.log(JSON.stringify({glm: sv.tierShares(n, {experts: 2000}), none: sv.tierShares({vram_used: 5, ram_used: 5}, {experts: 10}),
  ds4: sv.tierShares(ds, {engine_kind: "ds4"}), full: sv.tierShares({vram_used: 10, ram_used: 0, disk_gb: 0}, {experts: 10})}));""")
        self.assertEqual((got["glm"]["vu"], got["glm"]["ru"], got["glm"]["disk"]), (300, 500, 1200))
        self.assertEqual((got["glm"]["vramGb"], got["glm"]["ramGb"], got["glm"]["diskGb"]), (12.5, 20.25, 31.4))
        self.assertEqual((got["ds4"]["total"], got["ds4"]["disk"], got["ds4"]["diskGb"]), (440, 300, 75.5))
        self.assertIsNone(got["none"]["diskGb"])                                   # (an engine that reports no bytes: the count alone)
        self.assertEqual((got["full"]["disk"], got["full"]["diskGb"]), (0, 0))

    def test_context_slider_reaches_each_models_own_top(self):
        # the Models page's Context size: Settings > Model's stops over each model's range (its trained context, up to
        # 1M), the size a config has kept as a stop of its own
        got = self.node("""
const r1m = {min: 4096, max: 1048576}, r256 = {min: 4096, max: 262144};
st.S.ctxInfo = {...r1m, context: 131072};
const kvTok = 12288;   // (the bytes a position, as an engine's INFO would report them: kvBytesToken's source)
console.log(JSON.stringify({m1: sv.ctxStopsIn(r1m, 131072), m256: sv.ctxStopsIn(r256, 131072, 100000),
                            settings: sv.ctxStops(), none: sv.ctxStopsIn(null, 1),
                            win: sv.ctxHint(16384, {now: 131072, win: 32768, whole: true, ram: kvTok}),
                            past: sv.ctxHint(131072, {now: 131072, win: 32768, whole: true, ram: kvTok})}));""")
        self.assertEqual((got["m1"][0], got["m1"][-1]), (4096, 1048576))
        self.assertEqual(got["m1"], got["settings"])                     # (the same stops as Settings > Model's)
        self.assertEqual(got["m256"][-1], 262144)
        self.assertIn(100000, got["m256"])
        self.assertEqual(got["none"], [])
        self.assertEqual(got["win"], "Up to 32K the whole cache stays in VRAM.")
        self.assertTrue(got["past"].startswith("≈ 1.5 GB of RAM for the context"), got["past"])

    def test_the_size_line_takes_the_engines_own_bytes_per_position(self):
        # the slider's line under it: the bytes of a position come from the engine's INFO (the metrics'
        # engine.kv_bytes_token): one that reports them sizes the line, one that does not shows no size at all
        got = self.node("""
st.S.metrics = {engine: {kv_bytes_token: 1056}};
const tok = sv.kvBytesToken();
console.log(JSON.stringify({tok, none: (() => { st.S.metrics = {}; return sv.kvBytesToken(); })(),
                            line: sv.ctxHint(131072, {now: 32768, win: 32768, ram: tok}),
                            noguess: sv.ctxHint(131072, {now: 32768, win: 32768, ram: sv.kvBytesToken()})}));""")
        self.assertEqual(got["tok"], 1056)
        self.assertEqual(got["none"], 0)
        self.assertTrue(got["line"].startswith("\u2248 0.1 GB of RAM for the context"), got["line"])
        self.assertNotIn("GB", got["noguess"])

    def test_the_yarn_part_of_the_qfn_slider_and_the_servers_cost_line(self):
        # C18: QFN's context_range says where the trained 256K ends (yarn_above) and runs to the elastic YaRN's
        # 1M: the slider's stops follow (512K and 1M appear, nothing past 1M), sizes past the trained reach
        # carry the "YaRN, extended" hint, and the cost line under the slider keeps reading the server's own
        # pinned-K/V count (context_limits' kv_ram_gb scaled by the engine's bytes a token): 1M ~ 12.4 GiB,
        # the newest 32K of it on the cards too
        got = self.node("""
const r = {min: 4096, max: 1048576, trained: null, tested: 1048576, yarn_above: 262144};
st.S.ctxInfo = {...r, context: 262144, kv_resident: 32768, kv_ram_gb: (262144 * 12672) / 2 ** 30,
                ram_free_gb: 60};
console.log(JSON.stringify({stops: sv.ctxStops(), past: sv.ctxYarn(393216, r), at: sv.ctxYarn(262144, r),
                            over: sv.ctxYarn(1048576, r),
                            plain: sv.ctxYarn(393216, {min: 4096, max: 262144}),
                            est: sv.ctxEstimate(1048576),
                            line: sv.ctxHint(1048576, {live: true, now: 262144})}));""")
        self.assertIn(524288, got["stops"])
        self.assertEqual(got["stops"][-1], 1048576)
        self.assertEqual(got["past"], "YaRN, extended")
        self.assertEqual(got["at"], "")                             # (the trained reach itself: no hint)
        self.assertEqual(got["over"], "YaRN, extended")             # (1M itself is the extension too)
        self.assertEqual(got["plain"], "")                          # (any other family: unchanged)
        self.assertTrue(got["est"]["ram"])
        self.assertAlmostEqual(got["est"]["gb"], 262144 * 12672 / 2**30 * 4, delta=0.05)
        self.assertTrue(got["line"].startswith("\u2248 12.4 GB of RAM for the context"), got["line"])
        self.assertIn("kept pinned", got["line"])
        self.assertIn("32K positions of it also in VRAM", got["line"])

    def test_behind_llama_swap_the_page_asks_running_not_the_model(self):
        # under llama-swap EVERY request under /upstream/<id>/ loads the model (and evicts another from the card), so the
        # Monitor's per-second /metrics keeps reloading it: while the model is not "ready" in llama-swap's own /running
        # (the origin, it loads nothing) the page asks only that - no /metrics, no /health, no /mcp - and the Monitor
        # shows the quiet unloaded state instead of the model's figures
        got = self.node("""
location.pathname = "/upstream/glm-5.3-flash/";
st.S.health.model = "glm-5.3-flash"; st.S.health.supervisor = "llama-swap";
const log = [];
globalThis.fetch = async (u) => { log.push(u);
  return {ok: true, status: 200, json: async () => u.endsWith("/running")
    ? {running: [{model: "deepseek-v4.1-flash", state: "ready"}]} : {model: "glm-5.3-flash", live: {state: "idle"}}}; };
await sv.poll(); sv.stopPoll();
await sv.loadHealth(); await sv.loadMcp();
console.log(JSON.stringify({log, conn: st.S.conn, metricsNull: st.S.metrics === null}));""")
        self.assertEqual(got["log"], ["/running"])                    # (llama-swap's own, at the origin root)
        self.assertEqual(got["conn"], "unloaded")
        self.assertTrue(got["metricsNull"])

    def test_running_says_ready_the_page_polls_and_coming_back_reloads_health(self):
        # ready in /running: /metrics polled as before; and when the model is swapped out and back in, the page notices
        # (its health, the model's facts, fetched once more as it returns)
        got = self.node("""
location.pathname = "/upstream/glm-5.3-flash/";
st.S.health.model = "glm-5.3-flash"; st.S.health.supervisor = "llama-swap";
const log = [];
let ready = false;
globalThis.fetch = async (u) => { log.push(u);
  return {ok: true, status: 200, json: async () => u.endsWith("/running")
    ? {running: ready ? [{model: "glm-5.3-flash", state: "ready"}] : []}
    : u === "health" ? {model: "glm-5.3-flash", supervisor: "llama-swap"} : {model: "glm-5.3-flash", live: {state: "idle"}}}; };
await sv.poll();                                                    // (off the card: only /running asked)
const off = log.slice(), offConn = st.S.conn;
ready = true;
await sv.poll(); sv.stopPoll();                                     // (back: its health once, then the metrics again)
console.log(JSON.stringify({off, offConn, log, conn: st.S.conn, metrics: !!st.S.metrics}));""")
        self.assertEqual(got["off"], ["/running"])
        self.assertEqual(got["offConn"], "unloaded")
        self.assertEqual(got["log"], ["/running", "/running", "health", "metrics"])
        self.assertEqual(got["conn"], "ok")
        self.assertTrue(got["metrics"])

    def test_a_hidden_tab_asks_nothing_in_every_setup(self):
        # a hidden tab stops all polling - plain, and behind llama-swap; visible again it picks up at once
        got = self.node("""
const log = [];
globalThis.fetch = async (u) => { log.push(u);
  return {ok: true, status: 200, json: async () => u.endsWith("/running")
    ? {running: [{model: "glm-5.3-flash", state: "ready"}]} : {model: "glm-5.3-flash", live: {state: "idle"}}}; };
document.hidden = true;
await sv.poll(); await sv.loadHealth(); await sv.loadMcp();         // (hidden: nothing asked at all)
const hiddenPlain = log.slice();
document.hidden = false;
await sv.poll(); sv.stopPoll();                                     // (seen again: metrics as before)
const shown = log.slice();
location.pathname = "/upstream/glm-5.3-flash/";
st.S.health.model = "glm-5.3-flash"; st.S.health.supervisor = "llama-swap";
document.hidden = true;
await sv.poll();                                                    // (behind llama-swap the same: nothing at all)
console.log(JSON.stringify({hiddenPlain, shown, hiddenSwap: log.slice(), conn: st.S.conn}));""")
        self.assertEqual(got["hiddenPlain"], [])
        self.assertEqual(got["shown"], ["metrics"])
        self.assertEqual(got["hiddenSwap"], ["metrics"])             # (nothing new while hidden, in either setup)
        self.assertEqual(got["conn"], "ok")

    def test_not_behind_llama_swap_polls_exactly_as_before(self):
        # no supervisor (the usual single-server setup), or an /upstream/ path without one (another proxy): /metrics
        # directly, as before - /running is never asked
        got = self.node("""
const log = [];
globalThis.fetch = async (u) => { log.push(u);
  return {ok: true, status: 200, json: async () => ({model: "glm-5.3-flash", live: {state: "idle"}})}; };
await sv.poll(); sv.stopPoll();
const plain = log.slice();
location.pathname = "/upstream/glm-5.3-flash/";                     // (the path alone is not enough: health said no supervisor)
st.S.health.supervisor = null;
await sv.poll(); sv.stopPoll();
console.log(JSON.stringify({plain, pathOnly: log.slice(), conn: st.S.conn}));""")
        self.assertEqual(got["plain"], ["metrics"])
        self.assertEqual(got["pathOnly"], ["metrics", "metrics"])
        self.assertEqual(got["conn"], "ok")

    def test_the_first_poll_of_a_page_load_asks_no_health_of_its_own(self):
        # (the model-names review's nit 2: app.js starts poll() next to loadChats().then(loadHealth), and the first
        #  /metrics usually lands first - it names the real engine while S.health is still state.js' placeholder, and
        #  the identityChanged term fired a second /health on every page load.  The term waits for healthSeen now;
        #  a real switch after health has been read once still re-reads it)
        got = self.node("""
const log = [];
const eng = {model: "glm-5.3-flash", model_label: "GLM-5.3 · Maya-L",
             model_name_pretty: "GLM 5.3", model_quant: "Maya-L"};
globalThis.fetch = async (u) => { log.push(u);
  return {ok: true, status: 200, json: async () => u === "health" ? eng : {model: eng.model, engine: eng}}; };
await sv.poll();                                                    // (the page load: only /metrics, the health on its way)
const load = log.slice();
await sv.loadHealth();                                              // (app.js' health lands: healthSeen is set now)
log.length = 0;
await sv.poll(); sv.stopPoll();                                     // (the engine it names is this one: no re-read)
const same = log.slice();
st.S.health = {model: "qwen3.8-flash-next", model_label: "Qwen3.8-Flash-Next · Q8_0 (saguaro-qfn)"};
log.length = 0;
await sv.poll(); sv.stopPoll();                                     // (another engine named: /health once more)
console.log(JSON.stringify({load, same, moved: log.slice()}));""")
        self.assertEqual(got["load"], ["metrics"])                    # (the placeholder matches no engine: no ask)
        self.assertEqual(got["same"], ["metrics"])
        self.assertEqual(got["moved"], ["metrics", "health"])        # (a switch after healthSeen: one re-read, as item 10)

    def test_a_running_that_does_not_answer_polls_as_today_while_visible(self):
        # an older llama-swap without /running (404) or a key it refuses (401): the page falls back to today's polling -
        # but only while the tab is visible, and it stops the moment the tab is hidden
        body = """
location.pathname = "/upstream/glm-5.3-flash/";
st.S.health.model = "glm-5.3-flash"; st.S.health.supervisor = "llama-swap";
const log = [];
globalThis.fetch = async (u) => { log.push(u);
  return u.endsWith("/running") ? {ok: false, status: __CODE__, json: async () => ({})}
                                : {ok: true, status: 200, json: async () => ({model: "glm-5.3-flash", live: {state: "idle"}})}; };
await sv.poll(); sv.stopPoll();                                     // (__CODE__: metrics as today, after the failed /running)
const shown = log.slice();
document.hidden = true;
await sv.poll();                                                    // (hidden: nothing at all, /running included)
console.log(JSON.stringify({shown, hidden: log.slice(), conn: st.S.conn}));"""
        for code in (404, 401):
            with self.subTest(code=code):
                got = self.node(body.replace("__CODE__", str(code)))
                self.assertEqual(got["shown"], ["/running", "metrics"])
                self.assertEqual(got["hidden"], ["/running", "metrics"])   # (nothing new once hidden)
                self.assertEqual(got["conn"], "ok")

    def test_a_tab_opened_hidden_loads_health_when_it_is_seen(self):
        # (the review's finding 1: a page opened in a background tab - ctrl-click, "open in new tab", a restored
        # session - used to lose the boot chain's first health with nothing to retry it: the gate never learned to
        # close and the model's facts never loaded.  Seen again, the missed health loads and the gate engages)
        got = self.node("""
location.pathname = "/upstream/glm-5.3-flash/";
const log = [];
globalThis.fetch = async (u) => { log.push(u);
  return {ok: true, status: 200, json: async () => u.endsWith("/running")
    ? {running: [{model: "glm-5.3-flash", state: "ready"}]}
    : {model: "glm-5.3-flash", supervisor: "llama-swap", live: {state: "idle"}}}; };
document.hidden = true;
await sv.loadHealth();                                              // (app.js' first health, tab opened hidden)
const atOpen = log.slice();
document.hidden = false; fire("visibilitychange");                  // (the user switches to the tab)
await new Promise((ok) => setTimeout(ok, 50)); sv.stopPoll();
console.log(JSON.stringify({atOpen, afterShow: log.slice(), supervisor: st.S.health.supervisor || null,
                            first: log[0] || null}));""")
        self.assertEqual(got["atOpen"], [])                         # (hidden at open: nothing asked)
        self.assertTrue(got["first"].endswith("/running"))         # (the gate first: /health would load a model)
        self.assertEqual(got["supervisor"], "llama-swap")           # (the facts loaded: the gate can engage now)
        self.assertIn("metrics", got["afterShow"])                  # (and the ready model is polled again)

    def test_a_revealed_tab_whose_model_is_swapped_out_asks_only_running(self):
        got = self.node("""
location.pathname = "/upstream/glm-5.3-flash/";
const log = [];
globalThis.fetch = async (u) => { log.push(u);
  return {ok: true, status: 200, json: async () => ({running: [{model: "other", state: "ready"}]})}; };
document.hidden = true;
await sv.loadHealth();                                              // (tab opened hidden; its health is missed)
document.hidden = false; fire("visibilitychange");
await new Promise((ok) => setTimeout(ok, 50)); sv.stopPoll();
console.log(JSON.stringify({log, conn: st.S.conn}));""")
        self.assertTrue(got["log"] and all(u.endswith("/running") for u in got["log"]), got["log"])   # (nothing loaded)

    def test_a_health_retry_landing_in_a_hidden_tab_is_made_up_when_seen(self):
        # (the review's finding 2: health fails while the server starts, the tab hides before the 2 s retry fires -
        # the chain used to die there: /health was never asked again.  It must retry the moment the tab is seen)
        got = self.node("""
const log = [];
let down = true;
globalThis.fetch = async (u) => { log.push(u);
  if (u === "health" && down) throw new Error("nope");
  return {ok: true, status: 200, json: async () => ({model: "glm-5.3-flash", live: {state: "idle"}})}; };
await sv.loadHealth();                                              // (fails: the 2 s retry is armed)
await new Promise((ok) => setTimeout(ok, 1000));
document.hidden = true; fire("visibilitychange");
await new Promise((ok) => setTimeout(ok, 2500));                    // (the retry fires in the dark: nothing asked)
const whileHidden = log.filter((u) => u === "health").length;
down = false; document.hidden = false; fire("visibilitychange");    // (seen again: the missed retry happens)
await new Promise((ok) => setTimeout(ok, 100)); sv.stopPoll();
console.log(JSON.stringify({whileHidden, total: log.filter((u) => u === "health").length,
                            loaded: st.S.health.model === "glm-5.3-flash"}));""")
        self.assertEqual(got["whileHidden"], 1)                     # (only the failed first try; the dark asked nothing)
        self.assertEqual(got["total"], 2)                           # (seen again: the retry it lost, made up)
        self.assertTrue(got["loaded"])                              # (and the model's facts are here at last)

    def test_a_blip_on_running_keeps_the_gate_as_it_was(self):
        # (the review's finding 3: /running answering a network error or a 500 is a blip, not "older llama-swap" -
        #  answering no-gate then would fetch /metrics and load the model straight back.  A blip keeps the last
        #  answer: a closed gate stays closed; a blip before any answer yet polls as today, visible-only as ever)
        got = self.node("""
location.pathname = "/upstream/glm-5.3-flash/";
st.S.health.model = "glm-5.3-flash"; st.S.health.supervisor = "llama-swap";
const log = [];
let blip = false;
globalThis.fetch = async (u) => { log.push(u);
  if (u.endsWith("/running") && blip) throw new Error("no route");
  return {ok: true, status: 200, json: async () => u.endsWith("/running")
    ? {running: []} : {model: "glm-5.3-flash", live: {state: "idle"}}}; };
await sv.poll();                                                    // (off the card: the gate closes)
blip = true;
await sv.poll();                                                    // (the blip: the closed gate must stay closed)
blip = false;
const blipped = log.slice(), blipConn = st.S.conn;
await sv.poll(); sv.stopPoll();                                     // (and answering again, it behaves as before)
console.log(JSON.stringify({blipped, blipConn, log, conn: st.S.conn}));""")
        self.assertEqual(got["blipped"], ["/running", "/running"])  # (the blip asked /running only: no /metrics)
        self.assertEqual(got["blipConn"], "unloaded")
        self.assertEqual(got["log"], ["/running", "/running", "/running"])
        self.assertEqual(got["conn"], "unloaded")
        got = self.node("""
location.pathname = "/upstream/glm-5.3-flash/";
st.S.health.model = "glm-5.3-flash"; st.S.health.supervisor = "llama-swap";
const log = [];
globalThis.fetch = async (u) => { log.push(u);
  return u.endsWith("/running") ? {ok: false, status: 500, json: async () => ({})}
                                : {ok: true, status: 200, json: async () => ({model: "glm-5.3-flash", live: {state: "idle"}})}; };
await sv.poll(); sv.stopPoll();                                     // (500 before any answer: no gate to keep, poll on)
console.log(JSON.stringify({log, conn: st.S.conn}));""")
        self.assertEqual(got["log"], ["/running", "metrics"])       # (nothing known yet: poll as today)
        self.assertEqual(got["conn"], "ok")

    def test_a_mcp_retry_landing_in_a_hidden_tab_is_made_up_when_seen(self):
        # (the review's finding 6: the "servers starting" retry chain loadMcp arms every 3 s used to die when its
        #  timer landed while the tab was hidden - same hole as the health retry, one floor down in the module)
        got = self.node("""
const log = [];
let starting = true;
globalThis.fetch = async (u) => { log.push(u);
  return {ok: true, status: 200, json: async () => starting ? {servers: [{status: "starting"}], tools: 0}
                                                             : {servers: [{status: "ready"}], tools: 2}}; };
await sv.loadMcp();                                                 // (servers starting: the 3 s retry is armed)
document.hidden = true; fire("visibilitychange");
await new Promise((ok) => setTimeout(ok, 3500));                    // (the retry fires in the dark: parked, no ask)
const whileHidden = log.filter((u) => u === "mcp").length;
starting = false; document.hidden = false; fire("visibilitychange");   // (seen again: the chain it lost, made up)
await new Promise((ok) => setTimeout(ok, 50)); sv.stopPoll();
console.log(JSON.stringify({whileHidden, total: log.filter((u) => u === "mcp").length, tools: st.S.mcp.tools}));""")
        self.assertEqual(got["whileHidden"], 1)                     # (the dark asked nothing new)
        self.assertEqual(got["total"], 2)                           # (seen again: the dropped retry ran at last)
        self.assertEqual(got["tools"], 2)                           # (and the servers' tools are known again)

    def test_browsing_a_chat_behind_llama_swap_asks_no_token_count(self):
        # (the review's finding 7: the context meter POSTs api/tokens under /upstream/ - an upstream request like
        #  /metrics - so *opening or typing in* a chat loaded the model and softened the Monitor's "the next
        #  message loads it".  While unloaded the meter keeps its local estimate; the count asks again once the
        #  model is back on the card)
        got = self.node("""
location.pathname = "/upstream/glm-5.3-flash/";
st.S.health.model = "glm-5.3-flash"; st.S.health.supervisor = "llama-swap";
const ch = await import(web + "/chats.js");
const log = [];
let ready = false;
globalThis.fetch = async (u) => { log.push(u);
  return {ok: true, status: 200, json: async () => u.endsWith("/running")
    ? {running: ready ? [{model: "glm-5.3-flash", state: "ready"}] : []}
    : u === "api/tokens" ? {tokens: 1234}
    : u === "health" ? {model: "glm-5.3-flash", supervisor: "llama-swap"} : {model: "glm-5.3-flash", live: {state: "idle"}}}; };
await sv.poll();                                                    // (the model is off the card)
st.S.current = ch.newChatObj();
st.S.current.messages.push({role: "user", text: "hello there, how are you today"});
ch.updateCtxMeter();                                                // (opening the chat: the meter wakes)
await new Promise((ok) => setTimeout(ok, 500));
const unloaded = {log: log.slice(), tokens: st.S.ctx.tokens, approx: st.S.ctx.approx};
ready = true;
await sv.poll();                                                    // (back on the card: health, metrics)
ch.updateCtxMeter();
await new Promise((ok) => setTimeout(ok, 500)); sv.stopPoll();
console.log(JSON.stringify({unloaded, back: {tokens: st.S.ctx.tokens, approx: st.S.ctx.approx,
                                             asked: log.includes("api/tokens")}}));""")
        self.assertEqual(got["unloaded"]["log"], ["/running"])      # (browsing the chat asked nothing of the model)
        self.assertTrue(got["unloaded"]["tokens"] > 0)              # (the meter still says a word: the local estimate)
        self.assertTrue(got["unloaded"]["approx"])
        self.assertTrue(got["back"]["asked"])                       # (back on the card: the real count, once)
        self.assertEqual(got["back"]["tokens"], 1234)
        self.assertFalse(got["back"]["approx"])
    # every string a rendered tree carries (a vnode's text sits in its props; functions and refs are passed over)
    WALK = """
const strs = (v, out = []) => {
  if (typeof v === "string") out.push(v);
  else if (Array.isArray(v)) v.forEach((x) => strs(x, out));
  else if (v && typeof v === "object") {
    if ("type" in v && "props" in v) strs(v.props, out);
    else for (const k in v) strs(v[k], out);
  }
  return out;
};"""

    def test_opening_settings_while_unloaded_asks_only_running(self):
        # (the P2 review's L3: Settings' six asks - settings, mcp, api/context, api/port, api/thinking-budget,
        # api/update - are upstream requests behind llama-swap too, and the dialog's open effect fired them all:
        # a click that wanted the dialog, not the model. Every loader now refuses while the gate is closed, the
        # server-backed rows say "unloaded - the next message loads it" (the browser's own settings show all
        # along), and a dialog still open loads its rows the moment the model is back on the card.)
        got = self.node(self.WALK + """
location.pathname = "/upstream/glm-5.3-flash/";
st.S.health.model = "glm-5.3-flash"; st.S.health.supervisor = "llama-swap";
document.createElementNS = (ns, t) => el;                        // (enough DOM for preact to render the dialog into)
document.createTextNode = (s) => ({nodeType: 3, nodeValue: s, data: s});
const {html, render} = await import(web + "/preact.js");
const se = await import(web + "/settings.js");
const log = [];
let ready = false;
globalThis.fetch = async (u) => { log.push(u);
  return {ok: true, status: 200, json: async () => u.endsWith("/running")
    ? {running: ready ? [{model: "glm-5.3-flash", state: "ready"}] : []}
    : u === "settings" ? {shared: false, defaults: null}
    : u === "mcp" ? {servers: [{status: "ready", name: "x", transport: "stdio", tools: [{tool: "t"}]}], tools: 1}
    : u === "api/context" ? {context: 32768, min: 4096, max: 262144}
    : u === "api/port" ? {port: 8080, restarts: false, saved: true}
    : u === "api/thinking-budget" ? {budget: 0, supported: true, saved: true}
    : u === "api/update" ? {enabled: false, current: "9.9.9"}
    : u === "health" ? {model: "glm-5.3-flash", supervisor: "llama-swap"} : {model: "glm-5.3-flash", live: {state: "idle"}}}; };
await sv.poll();                                                // (the model is off the card)
st.S.settings = "general";
render(html`<${se.SettingsDialog}/>`, el);                      // (Settings opens over the closed gate)
await new Promise((ok) => setTimeout(ok, 50));
const unloaded = {log: log.slice(), note: strs(se.Tools()).some((s) => s.includes("unloaded - the next message loads it"))};
ready = true;
await sv.poll();                                                // (back on the card: its health - and its rows with it)
await new Promise((ok) => setTimeout(ok, 100)); sv.stopPoll();
console.log(JSON.stringify({unloaded,
  asked: log.filter((u) => !u.endsWith("/running") && u !== "health" && u !== "metrics"),
  loaded: st.S.ctxInfo !== null && st.S.portInfo !== null && st.S.thinkBudget !== null && st.S.update !== null
          && (st.S.mcp.servers || []).length === 1}));""")
        self.assertEqual(got["unloaded"]["log"], ["/running"])   # (the open dialog asked nothing of the model)
        self.assertTrue(got["unloaded"]["note"])                 # (Tools says what stands in for the servers)
        self.assertEqual(sorted(got["asked"]), sorted(["settings", "mcp", "api/context", "api/port",
                                                       "api/thinking-budget", "api/update"]))   # (ready: all six once)
        self.assertTrue(got["loaded"])

    def test_the_meter_re_estimates_when_the_chat_changes_while_unloaded(self):
        # (the P2 review's L4: with the gate closed the meter only re-estimated when it held no figure - so a chat
        # opened after one the server had counted went on showing that count as "This chat". It re-estimates whenever
        # the chat or its messages change, still asking nothing of the model, and the exact count asks again once
        # the model is back on the card.)
        got = self.node("""
location.pathname = "/upstream/glm-5.3-flash/";
st.S.health.model = "glm-5.3-flash"; st.S.health.supervisor = "llama-swap";
const ch = await import(web + "/chats.js");
const log = [];
let ready = true;
globalThis.fetch = async (u) => { log.push(u);
  return {ok: true, status: 200, json: async () => u.endsWith("/running")
    ? {running: ready ? [{model: "glm-5.3-flash", state: "ready"}] : []}
    : u === "api/tokens" ? {tokens: 1234}
    : u === "health" ? {model: "glm-5.3-flash", supervisor: "llama-swap"} : {model: "glm-5.3-flash", live: {state: "idle"}}}; };
const a = ch.newChatObj(), b = ch.newChatObj();
a.messages.push({role: "user", text: "how are the llamas today"});
b.messages.push({role: "user", text: "a completely different question about cacti and the desert"});
st.S.chats = [a, b];
ch.openChat(a);                                                 // (ready: the server's exact count)
await new Promise((ok) => setTimeout(ok, 500));
const exact = {tokens: st.S.ctx.tokens, approx: st.S.ctx.approx};
ready = false;
await sv.poll();                                                // (the model is swapped off the card)
ch.openChat(b);                                                 // (another chat: its own figure, not a's)
await new Promise((ok) => setTimeout(ok, 50));
const switched = {tokens: st.S.ctx.tokens, approx: st.S.ctx.approx, why: st.S.ctx.why, log: log.slice()};
st.S.draft = " and typing a new message"; ch.updateCtxMeter();  // (its messages change: re-estimate again)
await new Promise((ok) => setTimeout(ok, 50));
const typed = {tokens: st.S.ctx.tokens, log: log.slice()};
ready = true;
await sv.poll();                                                // (back on the card)
ch.updateCtxMeter();
await new Promise((ok) => setTimeout(ok, 500)); sv.stopPoll();
console.log(JSON.stringify({exact, switched, typed,
                            back: {tokens: st.S.ctx.tokens, approx: st.S.ctx.approx,
                                   asked: log.filter((u) => u === "api/tokens").length}}));""")
        self.assertEqual((got["exact"]["tokens"], got["exact"]["approx"]), (1234, False))
        self.assertEqual(got["switched"]["log"], ["api/tokens", "/running"])     # (the switch asked nothing)
        self.assertNotEqual(got["switched"]["tokens"], 1234)                     # (not the previous chat's figure)
        self.assertTrue(got["switched"]["approx"])
        self.assertEqual(got["typed"]["log"], ["api/tokens", "/running"])        # (typing asks nothing either)
        self.assertNotEqual(got["typed"]["tokens"], got["switched"]["tokens"])   # (changed messages: a fresh estimate)
        self.assertEqual(got["back"]["tokens"], 1234)                            # (back on the card: the real count)
        self.assertFalse(got["back"]["approx"])
        self.assertEqual(got["back"]["asked"], 2)                                # (once per chat, never while off-card)

    def test_typing_off_card_never_stringifies_the_whole_chat(self):
        # (P12: while the model was off the card the meter keyed its estimate on apiMessages(...) stringified - the
        #  whole chat, arrays and all - on every key press.  The key is now cheap (the chat's id, its instructions,
        #  the edit point, the messages' count and lengths, the draft, the attachments): 200 rapid inputs re-estimate
        #  off at most a couple of whole-chat stringifications, the meter still moves, a chat switch and an edited
        #  message still re-estimate, and the model is still asked for nothing off-card.)
        got = self.node("""
location.pathname = "/upstream/glm-5.3-flash/";
st.S.health.model = "glm-5.3-flash"; st.S.health.supervisor = "llama-swap";
const ch = await import(web + "/chats.js");
document.getElementById = () => null;                          // (enough DOM for the composer focus editMessage wants)
const log = [];
globalThis.fetch = async (u) => { log.push(u);
  return {ok: true, status: 200, json: async () => u.endsWith("/running")
    ? {running: []}
    : u === "api/tokens" ? {tokens: 1234}
    : u === "health" ? {model: "glm-5.3-flash", supervisor: "llama-swap"} : {model: "glm-5.3-flash", live: {state: "idle"}}}; };
await sv.poll();                                                // (the model is off the card)
let whole = 0;                                                  // (a whole-chat estimate: an array stringified)
const realStringify = JSON.stringify;
JSON.stringify = (...a) => { if (Array.isArray(a[0])) whole++; return realStringify(...a); };
const a = ch.newChatObj(), b = ch.newChatObj();
a.messages.push({role: "user", text: "how are the llamas today"});
a.messages.push({role: "assistant", text: "Thirsty, mostly. Bring water."});
b.messages.push({role: "user", text: "a completely different question about cacti and the desert"});
st.S.chats = [a, b];
ch.openChat(a);
const opened = {tokens: st.S.ctx.tokens, approx: st.S.ctx.approx};
for (let i = 0; i < 200; i++) {                                 // (200 rapid key presses into the composer)
  st.S.draft = "typing one more character into the composer " + "x".repeat(i);
  ch.updateCtxMeter();
}
const typed = {whole, tokens: st.S.ctx.tokens, log: log.slice()};
st.S.draft = "";
ch.openChat(b);                                                 // (another chat: its own figure, not a's)
await new Promise((ok) => setTimeout(ok, 50));
const switched = {whole, tokens: st.S.ctx.tokens, approx: st.S.ctx.approx, log: log.slice()};
ch.editMessage(0);                                              // (a message back in the composer to be edited...)
st.S.draft = "a much shorter question"; ch.updateCtxMeter();    // (... and made shorter: the estimate follows)
await new Promise((ok) => setTimeout(ok, 50));
const edited = {whole, tokens: st.S.ctx.tokens, approx: st.S.ctx.approx, log: log.slice()};
JSON.stringify = realStringify;
sv.stopPoll();
console.log(realStringify({opened, typed, switched, edited}));""")
        self.assertTrue(got["opened"]["tokens"] > 0 and got["opened"]["approx"])   # (off-card: the local estimate)
        self.assertLessEqual(got["typed"]["whole"], 2)                             # (200 inputs: at most a couple)
        self.assertNotEqual(got["typed"]["tokens"], got["opened"]["tokens"])       # (... and the meter still moves)
        self.assertEqual(got["typed"]["log"], ["/running"])                        # (typing asked nothing of the model)
        self.assertLessEqual(got["switched"]["whole"], 2)
        self.assertNotEqual(got["switched"]["tokens"], got["typed"]["tokens"])     # (the switch: b's own figure)
        self.assertTrue(got["switched"]["approx"])
        self.assertEqual(got["switched"]["log"], ["/running"])                     # (the switch asked nothing either)
        self.assertLessEqual(got["edited"]["whole"], 2)
        self.assertNotEqual(got["edited"]["tokens"], got["switched"]["tokens"])    # (the edit re-estimated)
        self.assertTrue(got["edited"]["approx"])
        self.assertEqual(got["edited"]["log"], ["/running"])

    def test_the_ram_skip_carries_its_warning_under_the_row_only_for_ram(self):
        # Settings > Turbo, live: the rows name the options plainly, and the RAM warning (callout--warn, the
        # class the page already has for warnings) shows under the Skip row only while RAM is chosen
        got = self.node(self.WALK + """
const tb = await import(web + "/turbo.js");
const at = () => strs(tb.TurboLive());
st.S.turboLive = {drop: 1, skip: "off"}; const off = at();
st.S.turboLive = {drop: 1, skip: "ssd"}; const ssd = at();
st.S.turboLive = {drop: 1, skip: "ram"}; const ram = at();
console.log(JSON.stringify({off, ssd, ram}));""")
        warn = "can break answers: only where almost everything is on the card"
        for name in ("off", "ssd", "ram"):
            tree = got[name]
            for wanted in ("Drop experts", "fewer of the 10 routed experts a token, the weakest dropped",
                           "Skip slow experts", "an expert that would be read from the SSD is left out",
                           "an expert that is not on the card is left out"):
                self.assertIn(wanted, tree, f"{name}: {wanted}")
            self.assertEqual(any("callout--warn" in s for s in tree), name == "ram", name)
            self.assertEqual(warn in tree, name == "ram", name)

    def test_the_skip_control_hides_where_the_engine_leaves_it_off(self):
        # PLAN-p5: the server's /api/turbo/options says turbo_skip false (a QFN whose engine applies drop only);
        # loadTurboLive keeps that in S.turboSkip, and the Skip control disappears - the Settings row and the
        # chat menu's Skip items - while Drop experts, the toggle and the warning stay as they were
        got = self.node(self.WALK + """
const tb = await import(web + "/turbo.js"), cv = await import(web + "/chat-view.js");
st.S.turboLive = {drop: 1, skip: "ram"};
st.S.current = {turbo: {drop: 1, skip: "ram"}};
const menu = () => strs(cv.TurboMenu({open: true, close: () => {}, anchor: {}}));
const shown = {row: strs(tb.TurboLive()), menu: menu()};
st.S.turboSkip = false;
const hidden = {row: strs(tb.TurboLive()), menu: menu()};
console.log(JSON.stringify({shown, hidden}));""")
        warn = "can break answers: only where almost everything is on the card"
        for half in ("row", "menu"):
            self.assertIn("Skip slow experts", got["shown"][half], half)
            self.assertIn(warn, got["shown"][half], half)
            self.assertNotIn("Skip slow experts", got["hidden"][half], half)
            self.assertNotIn(warn, got["hidden"][half], half)
            self.assertIn("Drop experts", got["hidden"][half], half)   # (the rest of Turbo stays: nothing is disabled)
        self.assertIn("skip: SSD", got["shown"]["menu"])               # (the menu's own items go with the row)
        self.assertNotIn("skip: SSD", got["hidden"]["menu"])

    def test_the_page_shows_no_engine_version_string(self):
        # (1.2.1 item 11: the engine's own version - QFN's frozen Strata string in its INFO engine=/version=,
        # stale ones elsewhere - is never a page label.  With every engine-version field of the fake state
        # carrying "9.9.9-old", the rendered Settings > About, the chat header's model card and the Copy report
        # say Saguaro's release version and nothing else.  The server keeps sending those fields (other clients,
        # and the saved conversations' keying depends on the engine string); the page just never renders them.)
        got = self.node(self.WALK + """
const cv = await import(web + "/chat-view.js"), se = await import(web + "/settings.js");
const mo = await import(web + "/models.js");
mo.MS.data = {models: []};                                            // (the open menu would ask for the Models list)
const {html, render} = await import(web + "/preact.js");
document.createElementNS = (ns, t) => el;                             // (the two DOM stubs the other render test adds)
document.createTextNode = (s) => ({nodeType: 3, nodeValue: s, data: s});
const old = "9.9.9-old";
st.S.health = {model: "qwen3.8-flash-next", model_name_pretty: "Qwen3.8 Flash Next", model_quant: "Q8_0",
               max_context: 1048576, images: false, api_key: false, mcp_tools: false, version: old};
st.S.metrics = {engine: {model: "qwen3.8-flash-next", engine_kind: "qfn", max_context: 1048576,
                         version: old, engine: old, maya_version: old, saguaro_version: "1.2.1"}};
let aboutTree, headTree;                                             // (the sections have hooks: render them and walk
const about = () => { aboutTree = se.About(); return null; };         //  the trees they return, not fresh calls)
const head = () => { headTree = cv.ModelMenu({open: true, close: () => {}, anchor: {}}); return null; };
render(html`<${about}/>`, el);
render(html`<${head}/>`, el);
globalThis.fetch = async () => ({ok: true, status: 200, text: async () =>
  "Saguaro 1.2.1 - report of 2026-11-25 06:00\\n" +
  "Model: qwen3.8-flash-next (Qwen3.8-Flash-Next \\u00b7 Q8_0 (saguaro-qfn)), context 1048576\\n" +
  "Engine: arena_mib=70000 engine=" + old + " engine_kind=qfn experts=13824 version=" + old + "\\n"});
const report = await sv.fetchReport();
console.log(JSON.stringify({about: strs(aboutTree), head: strs(headTree), report}));""")
        self.assertNotIn("9.9.9", " ".join(got["about"]))              # (About: no engine string anywhere)
        self.assertIn("Saguaro v1.2.1", " ".join(got["about"]))        # (the release version: the only one shown)
        self.assertNotIn("9.9.9", " ".join(got["head"]))               # (the header's card names the model, not a version)
        self.assertIn("on this machine", " ".join(got["head"]))
        self.assertNotIn("9.9.9", got["report"])                       # (Copy report: the engine's INFO pairs are out)
        self.assertIn("Engine: arena_mib=70000 engine_kind=qfn experts=13824", got["report"])  # (... its facts stay)


if __name__ == "__main__":
    unittest.main()
