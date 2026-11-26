"""serve/test_savings.py - the sidebar's "Saved": each request's tokens counted per model (new prompt, prompt reused
from the cache, output), kept across starts, and priced at the model's own API (DeepSeek's peak hours at twice its
price) - against the mock engine, no GPU.  And the prices themselves: data/api-prices.json, the newer copy fetched
from GitHub once a day when the Savings page asks, and the precedence the page's own prices > fetched > file >
built-ins.  urllib is mocked throughout: nothing here touches the network.

    python -m unittest serve.test_savings -v
"""
from __future__ import annotations

import calendar
import json
import os
import shutil
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import savings as SV  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UTC = lambda *a: calendar.timegm((*a, 0, 0, 0))   # noqa: E731  (year, month, day, hour, minute, second)
TUE_OFF, TUE_PEAK, SAT = UTC(2026, 10, 27, 12, 0, 0), UTC(2026, 10, 27, 2, 30, 0), UTC(2026, 10, 31, 2, 30, 0)


class Prices(unittest.TestCase):
    def test_references(self):
        self.assertEqual(SV.reference("deepseek-v4.1-flash")["api"], "DeepSeek")
        self.assertEqual(SV.reference("glm-5.3-flash")["output"], 0.50)
        mm = SV.reference("minimax-m3.1-flash")                       # (MiniMax's: flat, no peak hours)
        self.assertEqual((mm["api"], mm["input"], mm["cached"], mm["output"], mm.get("peak")),
                         ("MiniMax", 0.075, 0.0075, 0.30, None))
        self.assertEqual(SV.reference("MiniMax-M3.1-Flash")["api"], "MiniMax")
        hy = SV.reference("hy4-flash")                                # (Tencent's: flat, no peak hours)
        self.assertEqual((hy["api"], hy["input"], hy["cached"], hy["output"], hy.get("peak")),
                         ("Tencent", 0.035, 0.0015, 0.1405, None))
        for name in ("Hy4-Flash", "tencent/Hy4-flash", "hy4_flash", "Hy4-Flash-Maya-GGUF"):
            self.assertEqual((SV.reference(name) or {}).get("api"), "Tencent", name)
        for name in ("why4-flash", "hy4", "hy4-pro"):
            self.assertIsNone(SV.reference(name), name)
        q = SV.reference("Qwen3.8-Flash-Next-UD-Q4_K_XL")             # (every quant of the family has the same API)
        self.assertEqual((q["api"], q["input"], q["cached"], q["output"]), ("Qwen", 0.15, 0.016, 0.47))
        ik = SV.reference("Inkling-2-Mini-UD-Q4_K_XL")
        self.assertEqual((ik["api"], ik["input"], ik["cached"], ik["output"]), ("Thinking Machines", 0.075, 0.015, 0.30))
        for name in ("qwen3.9-flash", "inkling-3-mini"):
            self.assertIsNone(SV.reference(name), name)

    def test_openrouter_ids(self):
        """"openrouter" per family: the built-ins and data/api-prices.json name the same ids, and the built-ins
        carry every price the shipped file does (tools/refresh_prices.py and the live fetch map by these)."""
        doc = SV.load_price_file(SV.ROOT / "data" / SV.PRICE_FILE)
        self.assertIsNotNone(doc)
        by_id = {p["openrouter"]: (name, p) for name, p in doc["prices"].items()}
        self.assertEqual(len(by_id), len(doc["prices"]))               # (no family listed twice)
        for ref in SV.REFERENCES:
            name, p = by_id[ref["openrouter"]]
            for k in ("input", "cached", "output", "peak", "api"):
                self.assertEqual(p.get(k), ref.get(k), f"{name}.{k}")  # (the fallback and the file agree)
        self.assertEqual(set(p["openrouter"] for p in SV.REFERENCES),
                         {"deepseek/deepseek-v4.1-flash-1017", "z-ai/glm-5.3-flash", "minimax/minimax-m3.1-flash",
                          "tencent/hy4-flash", "qwen/qwen3.8-flash", "thinkingmachines/inkling-2-mini"})

    def test_peak_hours(self):
        self.assertTrue(SV.deepseek_peak(TUE_PEAK))
        self.assertFalse(SV.deepseek_peak(TUE_OFF))
        self.assertFalse(SV.deepseek_peak(SAT))                       # (weekends: off-peak all day)
        self.assertTrue(SV.deepseek_peak(UTC(2026, 10, 27, 9, 59, 0)))
        self.assertFalse(SV.deepseek_peak(UTC(2026, 10, 27, 10, 0, 0)))

    def test_cost(self):
        # FreeToken's "This launch": 2.8M in + 118K out at V4 Flash's $0.14 / $0.28 = $0.43; here the cached part is
        # priced as an API bills it, and DeepSeek's peak hours at twice
        b = SV.zero()
        b["base"].update(input=1_000_000, cached=10_000_000, output=1_000_000)
        ds = SV.reference("deepseek-v4.1-flash")
        self.assertAlmostEqual(SV.cost(b, ds), 0.15 + 0.03 + 0.60)
        b["peak"].update(input=1_000_000)
        self.assertAlmostEqual(SV.cost(b, ds), 0.15 + 0.03 + 0.60 + 0.30)
        self.assertIsNone(SV.cost(b, None))


class Counting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / SV.FILE
        self.off = mock.patch.dict(os.environ, {"SAGUARO_UPDATE_CHECK": "0"})    # (a report asks for prices: not here)
        self.off.start()

    def tearDown(self):
        self.off.stop()
        self.tmp.cleanup()

    def test_kept_across_starts_and_per_model(self):
        a = SV.Ledger(self.path, "maya-gguf-q8.json", "deepseek-v4.1-flash", "UD-Q8_K_XL")
        a.add(1_000_000, 900_000, 500_000, now=TUE_OFF)
        a.add(1_000_000, 0, 0, now=TUE_PEAK)
        self.assertAlmostEqual(a.summary()["usd"], 0.1 * 0.15 + 0.9 * 0.003 + 0.5 * 0.60 + 0.30, places=6)
        b = SV.Ledger(self.path, "maya-maya-m.json", "glm-5.3-flash", "Maya-M")     # (another model starts)
        self.assertEqual(b.summary()["start_usd"], 0.0)
        b.add(2_000_000, 1_000_000, 100_000, now=TUE_PEAK)                         # (no peak price for GLM)
        glm = 0.15 + 0.03 + 0.05
        r = b.report()
        self.assertAlmostEqual(r["usd"], a.summary()["usd"] + glm, places=6)
        self.assertEqual([m["key"] for m in r["models"]], ["maya-maya-m.json", "maya-gguf-q8.json"])  # (running first)
        self.assertEqual(r["models"][1]["tokens"], {"input": 1_100_000, "cached": 900_000, "output": 500_000, "requests": 2})
        self.assertAlmostEqual(r["start"]["usd"], glm, places=6)
        self.assertEqual({p["model"] for p in r["prices"]}, {"deepseek-v4.1-flash", "glm-5.3-flash"})

    def test_hy4_flash_is_priced(self):
        # (the sidebar shows Saved only for a priced model: Hy4-flash had none, so its line never showed)
        h = SV.Ledger(self.path, "maya-hy4.json", "hy4-flash", "hy4")
        h.add(1_000_000, 400_000, 100_000, now=TUE_PEAK)              # (Tencent has no peak hours: the base price)
        s = h.summary()
        self.assertTrue(s["priced"])
        self.assertAlmostEqual(s["start_usd"], 0.6 * 0.035 + 0.4 * 0.0015 + 0.1 * 0.1405, places=6)
        self.assertEqual(h.report()["prices"][0]["api"], "Tencent")

    def test_reused_never_above_the_prompt_and_unknown_models(self):
        q = SV.Ledger(self.path, "x.json", "no-such-model-9000")
        q.add(100, 500, 10)
        self.assertEqual(q.report()["models"][0]["tokens"]["cached"], 100)
        self.assertEqual(q.summary(), {"usd": 0.0, "start_usd": 0.0, "priced": False})

    def test_reset(self):
        a = SV.Ledger(self.path, "maya-maya-m.json", "glm-5.3-flash")
        a.add(1_000_000, 0, 0)
        a.set_price("glm-5.3-flash", {"input": 1, "cached": 0.1, "output": 2})
        a.reset()
        b = SV.Ledger(self.path, "maya-maya-m.json", "glm-5.3-flash")
        self.assertEqual(b.summary()["usd"], 0.0)
        self.assertEqual(b.ref["input"], 1.0)                          # (the page's prices stay)

    def test_own_prices_count_every_total_again(self):
        a = SV.Ledger(self.path, "maya-gguf-q8.json", "deepseek-v4.1-flash")
        a.add(1_000_000, 0, 1_000_000, now=TUE_OFF)
        self.assertAlmostEqual(a.summary()["usd"], 0.75, places=6)
        a.set_price("deepseek-v4.1-flash", {"input": 0.30, "cached": 0.006, "output": 1.20})
        self.assertAlmostEqual(a.summary()["usd"], 1.50, places=6)
        p = SV.Ledger(self.path, "maya-gguf-q8.json", "deepseek-v4.1-flash").price_list()[0]   # (kept in the file)
        self.assertEqual((p["custom"], p["input"], p["peak"], p["default"]["input"]), (True, 0.30, 2.0, 0.15))
        a.set_price("deepseek-v4.1-flash", None)                        # (back to the built-in ones)
        self.assertAlmostEqual(a.summary()["usd"], 0.75, places=6)
        # a model without a built-in price can get one; nonsense is refused with nothing changed
        q = SV.Ledger(self.path, "maya-q.json", "no-such-model-9000")
        q.add(2_000_000, 0, 0)
        self.assertIsNone(q.price("no-such-model-9000"))               # (unpriced until the page gives it one)
        q.set_price("no-such-model-9000", {"input": 0.5, "cached": 0.05, "output": 1})
        self.assertAlmostEqual(q.report()["start"]["usd"], 1.0, places=6)
        for bad in ({"input": -1, "cached": 0, "output": 0}, {"input": "1", "cached": 0, "output": 0}, {"input": 1},
                    {"input": 1, "cached": 0, "output": 0, "peak": 0.5}):
            with self.assertRaises(ValueError, msg=bad):
                a.set_price("deepseek-v4.1-flash", bad)
        self.assertFalse(a.price_list()[0]["custom"])


class LedgerPath(unittest.TestCase):
    def test_default_is_beside_the_config(self):
        self.assertEqual(SV.ledger_path(Path("/a/b/run.json"), {}), Path("/a/b") / SV.FILE)

    def test_env_names_the_file_and_off_keeps_none(self):
        self.assertEqual(SV.ledger_path(Path("/a/b/run.json"), {"SAGUARO_SAVINGS_PATH": "/w/s.json"}), Path("/w/s.json"))
        for v in ("off", "OFF", "none"):
            self.assertIsNone(SV.ledger_path(Path("/a/b/run.json"), {"SAGUARO_SAVINGS_PATH": v}))

    def test_a_failed_save_is_logged_once(self):
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as d:
            led = SV.Ledger(Path(d) / "no-such-folder" / SV.FILE, "k.json", "m")
            out = io.StringIO()
            with redirect_stdout(out):
                for _ in range(3):
                    led.add(10, 0, 5)
            self.assertEqual(out.getvalue().count("could not save"), 1)


class Page(unittest.TestCase):
    """A request through the server is counted; /metrics carries the sidebar's line, /api/savings the breakdown."""

    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nhello", max_context=32768), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"), model_name="glm-5.3-flash")
        self.tmp = tempfile.TemporaryDirectory()
        self.svc.savings = SV.Ledger(Path(self.tmp.name) / SV.FILE, "maya-maya-m.json", "glm-5.3-flash", "Maya-M")
        self.off = mock.patch.dict(os.environ, {"SAGUARO_UPDATE_CHECK": "0"})    # (the page reads prices: no fetching in tests)
        self.off.start()
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.off.stop()
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
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_a_request_is_counted(self):
        s, b = self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 200)
        s, r = self.req("GET", "/api/savings")
        t = r["models"][0]["tokens"]
        self.assertEqual((t["requests"], t["input"], t["output"]),
                         (1, b["usage"]["prompt_tokens"], b["usage"]["completion_tokens"]))
        self.assertGreater(r["usd"], 0)
        s, m = self.req("GET", "/metrics")
        self.assertEqual((m["savings"]["usd"], m["savings"]["priced"]), (r["usd"], True))
        self.assertIsNone(m["live"]["prompt_reused"])                     # (the Monitor's Context bar: idle now)
        self.assertTrue(json.loads((Path(self.tmp.name) / SV.FILE).read_text())["models"]["maya-maya-m.json"])

    def test_prices_from_the_page(self):
        self.svc.savings.add(1_000_000, 0, 0)
        s, r = self.req("POST", "/api/savings/prices", {"model": "glm-5.3-flash",
                                                        "price": {"input": 1, "cached": 0.1, "output": 2}})
        self.assertEqual((s, r["usd"], r["price_list"][0]["custom"]), (200, 1.0, True))
        s, _ = self.req("POST", "/api/savings/prices", {"model": "glm-5.3-flash", "price": {"input": "x"}})
        self.assertEqual(s, 400)
        s, _ = self.req("POST", "/api/savings/prices", {"model": "glm-5.3-flash", "price": None},
                        {"Origin": "http://elsewhere.example"})
        self.assertEqual(s, 403)
        s, r = self.req("POST", "/api/savings/prices", {"model": "glm-5.3-flash", "price": None})
        self.assertEqual((s, r["usd"]), (200, 0.15))

    def test_reset_only_from_the_page(self):
        self.svc.savings.add(1_000_000, 0, 0)
        s, _ = self.req("POST", "/api/savings/reset", {}, {"Origin": "http://elsewhere.example"})
        self.assertEqual(s, 403)
        s, r = self.req("POST", "/api/savings/reset", {})
        self.assertEqual((s, r["usd"]), (200, 0.0))


class PriceFiles(unittest.TestCase):
    """The default prices: the page's own > a fetched api-prices.json (valid, newer as_of) > data/api-prices.json >
    the built-ins.  A file that does not check out is ignored with one log line; the fetch runs at most once a day,
    only when the Savings page asks, and SAGUARO_UPDATE_CHECK=0 keeps it off."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo_file = self.tmp / "api-prices.json"                # (in place of the checkout's data/api-prices.json)
        self.save = self.tmp / "savings"                             # (the ledger's folder: the cache lives here too)
        self.save.mkdir()
        self.off = mock.patch.dict(os.environ, {"SAGUARO_UPDATE_CHECK": "0"})    # (no test fetches: each one says "1" itself)
        self.off.start()

    def tearDown(self):
        self.off.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def text(as_of="2026-11", inp=0.99):
        return json.dumps({"as_of": as_of, "prices":
                           {"glm-5.3-flash": {"api": "Z.ai", "input": inp, "cached": 0.03, "output": 0.5}}})

    def ledger(self):
        return SV.Ledger(self.save / SV.FILE, "maya-maya-m.json", "glm-5.3-flash",
                         prices=SV.PriceBook(self.save, self.repo_file))

    @staticmethod
    def answered(text):
        r = mock.MagicMock()
        r.__enter__.return_value.read.return_value = text.encode()
        return r

    def test_precedence_page_file_builtins(self):
        led = self.ledger()
        self.assertEqual(led.price("glm-5.3-flash")["input"], 0.15)             # (no file yet: the built-ins count)
        self.repo_file.write_text(self.text())
        led = self.ledger()
        self.assertEqual(led.price("glm-5.3-flash")["input"], 0.99)             # (the file on disk beats the built-ins)
        self.assertEqual(led.report()["prices_as_of"], "2026-11")
        led.set_price("glm-5.3-flash", {"input": 0.5, "cached": 0.05, "output": 1})
        self.assertEqual(led.price("glm-5.3-flash")["input"], 0.5)              # (the page beats the file)
        led.set_price("glm-5.3-flash", None)
        self.assertEqual(led.price("glm-5.3-flash")["input"], 0.99)             # (back to the file's)

    def test_the_fetch_once_a_day_and_only_a_newer_file_wins(self):
        self.repo_file.write_text(self.text("2026-10", 0.10))
        led = self.ledger()
        with mock.patch.dict(os.environ, {"SAGUARO_UPDATE_CHECK": "1"}), \
                mock.patch.object(SV.urllib.request, "urlopen", return_value=self.answered(self.text("2026-11", 0.99))) as net:
            rep = led.report()
            self.assertEqual(led.price("glm-5.3-flash")["input"], 0.99)         # (fetched beats this folder's own)
            self.assertEqual(rep["prices_source"], {"from": "github", "as_of": "2026-11"})
            self.assertEqual(json.loads((self.save / SV.PRICE_FILE).read_text())["as_of"], "2026-11")   # (cached for the next start)
            req = net.call_args[0][0]
            self.assertRegex(req.full_url, r"raw\.githubusercontent\.com/[\w.-]+/[\w.-]+/main/data/api-prices\.json$")
            led.report()                                                        # (once a day, not on every page load)
            net.assert_called_once()

    def test_a_fetched_copy_that_is_not_newer_does_not_move_the_prices(self):
        self.repo_file.write_text(self.text("2026-11", 0.99))
        led = self.ledger()
        with mock.patch.dict(os.environ, {"SAGUARO_UPDATE_CHECK": "1"}), \
                mock.patch.object(SV.urllib.request, "urlopen", return_value=self.answered(self.text("2026-10", 0.10))):
            rep = led.report()
        self.assertEqual(led.price("glm-5.3-flash")["input"], 0.99)
        self.assertEqual(rep["prices_source"], {"from": "file", "as_of": "2026-11"})

    def test_yesterdays_cached_copy_is_used_at_the_start(self):
        self.repo_file.write_text(self.text("2026-10", 0.10))
        (self.save / SV.PRICE_FILE).write_text(self.text("2026-11", 0.99))
        self.assertEqual(self.ledger().price("glm-5.3-flash")["input"], 0.99)   # (fetched yesterday, still the newer)
        (self.save / SV.PRICE_FILE).write_text(self.text("2026-09", 0.07))      # (older than the folder's own: not used)
        self.assertEqual(self.ledger().price("glm-5.3-flash")["input"], 0.10)

    def test_the_fetch_is_off_when_the_user_said_no_requests(self):
        self.repo_file.write_text(self.text())
        led = self.ledger()
        with mock.patch.object(SV.urllib.request, "urlopen") as net:            # (SAGUARO_UPDATE_CHECK=0 from setUp)
            rep = led.report()
        net.assert_not_called()
        self.assertEqual(led.price("glm-5.3-flash")["input"], 0.99)
        self.assertEqual(rep["prices_source"], {"from": "file", "as_of": "2026-11"})

    def test_a_network_error_falls_back_to_the_file_and_the_builtins(self):
        self.repo_file.write_text(self.text("2026-11", 0.99))
        led = self.ledger()
        with mock.patch.dict(os.environ, {"SAGUARO_UPDATE_CHECK": "1"}), \
                mock.patch.object(SV.urllib.request, "urlopen", side_effect=OSError("offline")):
            rep = led.report()
        self.assertEqual(led.price("glm-5.3-flash")["input"], 0.99)             # (the disk file counts meanwhile)
        self.assertEqual(rep["prices_source"], {"from": "file", "as_of": "2026-11"})
        self.assertIn("offline", led.book.error)
        led2 = SV.Ledger(self.save / SV.FILE, "k.json", "glm-5.3-flash",
                         prices=SV.PriceBook(self.save, self.tmp / "no-such-file.json"))   # (and no file at all)
        with mock.patch.dict(os.environ, {"SAGUARO_UPDATE_CHECK": "1"}), \
                mock.patch.object(SV.urllib.request, "urlopen", side_effect=urllib.error.URLError("no route")):
            led2.report()
        self.assertEqual(led2.price("glm-5.3-flash")["input"], 0.15)            # (the built-ins, again)

    def test_a_bad_price_file_is_ignored_with_one_log_line(self):
        import io
        from contextlib import redirect_stdout
        bad = ["{not json",
               json.dumps({"prices": {}}),                                        # (no as_of month)
               json.dumps({"as_of": "2026-11", "prices": {"glm-5.3-flash": {"input": 1, "cached": 0}}}),   # (no output)
               json.dumps({"as_of": "2026-11", "prices":
                          {"glm-5.3-flash": {"api": "Z.ai", "input": 1, "cached": 0, "output": 1, "match": "([("}}})]
        out = io.StringIO()
        for text in bad:
            self.repo_file.write_text(text)
            with redirect_stdout(out):
                led = self.ledger()
            self.assertEqual(led.price("glm-5.3-flash")["input"], 0.15, text)   # (the built-ins count)
        self.assertEqual(out.getvalue().count("ignored"), len(bad))

    def test_fetched_junk_is_ignored_with_one_log_line(self):
        import io
        from contextlib import redirect_stdout
        self.repo_file.write_text(self.text("2026-11", 0.99))
        led = self.ledger()
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"SAGUARO_UPDATE_CHECK": "1"}), \
                mock.patch.object(SV.urllib.request, "urlopen", return_value=self.answered("<html>not a price file</html>")), \
                redirect_stdout(out):
            led.report()
        self.assertEqual(out.getvalue().count("ignored"), 1)
        self.assertEqual(led.price("glm-5.3-flash")["input"], 0.99)             # (the disk file still counts)
        self.assertFalse((self.save / SV.PRICE_FILE).exists())                  # (the good cache is not overwritten)


if __name__ == "__main__":
    unittest.main()
