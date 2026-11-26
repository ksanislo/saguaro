"""serve/test_savings.py - the sidebar's "Saved": each request's tokens counted per model (new prompt, prompt reused
from the cache, output), kept across starts, and priced at the model's own API (DeepSeek's peak hours at twice its
price) - against the mock engine, no GPU.  And the prices themselves: the shipped table of data/api-prices.json, the
OpenRouter list fetched once a day on a background thread when the user turns live_prices on (off by default: a
default install dials nothing, ever), and the precedence the page's own prices > OpenRouter > shipped > built-ins.
urllib is mocked throughout: nothing here touches the network; the OpenRouter answer is
ref/openrouter-models-sample.json, a real response cut to our models.

    python -m unittest serve.test_savings -v
"""
from __future__ import annotations

import calendar
import json
import os
import shutil
import sys
import tempfile
import time
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
SAMPLE_TEXT = (ROOT / "ref" / "openrouter-models-sample.json").read_text(encoding="utf-8")   # a real OpenRouter answer
REAL_OPEN = urllib.request.urlopen    # (the test's own client, kept out of the way of tests that patch urlopen)


def answered(text=None):
    r = mock.MagicMock()
    r.__enter__.return_value.read.return_value = (SAMPLE_TEXT if text is None else text).encode()
    return r


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

    def tearDown(self):
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
            with REAL_OPEN(r, timeout=30) as resp:      # (REAL_OPEN: a test may patch urlopen to fake OpenRouter)
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

    def test_the_live_switch_from_the_page(self):
        """POST /api/savings/live: the Settings switch (its own page only); the report answers with the new state;
        on starts the daily path - a mock answer here, never the network."""
        self.assertFalse(self.req("GET", "/api/savings")[1]["live"])            # (off by default)
        with mock.patch.object(SV.urllib.request, "urlopen", return_value=answered()) as net:
            s, r = self.req("POST", "/api/savings/live", {"live": True})
            self.assertEqual((s, r["live"]), (200, True))
            self.svc.savings.book.thread.join(10)
            net.assert_called_once()
            s, r = self.req("POST", "/api/savings/live", {"live": False})
            self.assertEqual((s, r["live"]), (200, False))
        s, _ = self.req("POST", "/api/savings/live", {"live": "yes"})
        self.assertEqual(s, 400)
        s, _ = self.req("POST", "/api/savings/live", {"live": True}, {"Origin": "http://elsewhere.example"})
        self.assertEqual(s, 403)


class PriceFiles(unittest.TestCase):
    """The shipped table: the page's own prices > data/api-prices.json > the built-ins.  A file that does not check
    out is ignored with one log line; nothing here fetches anything."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo_file = self.tmp / "api-prices.json"                # (in place of the checkout's data/api-prices.json)
        self.save = self.tmp / "savings"                             # (the ledger's folder)
        self.save.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def text(as_of="2026-11", inp=0.99):
        return json.dumps({"as_of": as_of, "prices":
                           {"glm-5.3-flash": {"api": "Z.ai", "input": inp, "cached": 0.03, "output": 0.5}}})

    def ledger(self):
        return SV.Ledger(self.save / SV.FILE, "maya-maya-m.json", "glm-5.3-flash",
                         prices=SV.PriceBook(self.save, self.repo_file))

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


class OpenRouterPrices(unittest.TestCase):
    """The live prices: OpenRouter's public model list (no key; nothing about the user in the request), fetched at
    most once a day on a background thread when live_prices is on - off by default, so a default install dials
    nothing, ever.  The cache lies beside the count, with its fetch time; on any failure the cache or the shipped
    table counts, with one log line.  The answer here is the sample, never the network."""

    SAMPLE = json.loads(SAMPLE_TEXT)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo_file = self.tmp / "api-prices.json"                # (the shipped fallback, with prices of its own)
        self.save = self.tmp / "savings"                             # (the ledger's folder: the cache lives here too)
        self.save.mkdir()
        self.env = mock.patch.dict(os.environ, {"SAGUARO_LIVE_PRICES": ""})     # (nothing ambient: the config decides)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def text(as_of="2026-11", inp=0.99):
        return json.dumps({"as_of": as_of, "prices":
                           {"glm-5.3-flash": {"api": "Z.ai", "input": inp, "cached": 0.03, "output": 0.5}}})

    def ledger(self, live=False, config=None):
        return SV.Ledger(self.save / SV.FILE, "maya-maya-m.json", "glm-5.3-flash",
                         prices=SV.PriceBook(self.save, self.repo_file, live=live), config=config, live=live)

    def write_cache(self, doc, fetched):
        (self.save / SV.LIVE_FILE).write_text(json.dumps({"fetched": fetched, "as_of": doc["as_of"],
                                                          "prices": doc["prices"]}))
        os.utime(self.save / SV.LIVE_FILE, (fetched, fetched))

    def test_per_million_converts_the_per_token_string_exactly(self):
        self.assertEqual(SV.per_million("0.000000075"), 0.075)        # (float(...) * 1e6 would give 0.07500000000000001)
        self.assertEqual(SV.per_million("0.0000001405"), 0.1405)
        self.assertEqual(SV.per_million(" 0.0000000015 "), 0.0015)
        self.assertEqual(SV.per_million("0"), 0.0)
        for bad in ("-", "free", "", None, "0.0000001x"):
            with self.assertRaises(ValueError, msg=str(bad)):
                SV.per_million(bad)

    def test_the_sample_parses_and_maps_every_family(self):
        doc, why = SV.parse_openrouter(json.dumps(self.SAMPLE), SV.price_families(None))   # (the built-in families)
        self.assertIsNone(why)
        self.assertEqual({k: (v["input"], v["cached"], v["output"]) for k, v in doc["prices"].items()},
                         {"DeepSeek": (0.15, 0.003, 0.6), "Z.ai": (0.15, 0.03, 0.5),
                          "MiniMax": (0.075, 0.0075, 0.3), "Tencent": (0.035, 0.0015, 0.1405),
                          "Qwen": (0.15, 0.016, 0.47), "Thinking Machines": (0.075, 0.015, 0.3)})
        # (a family the shipped file does not name is keyed by the built-in's api name, which its match regex does
        # not fit: the peak factor still has to come through - the price shown is the one the shipped table gives)
        self.assertEqual(doc["prices"]["DeepSeek"]["peak"], 2.0)
        self.assertIsNone(doc["prices"]["Z.ai"].get("peak"))
        # with the shipped file read, the families carry the file's names (and its DeepSeek peak factor stays)
        fam = SV.price_families(SV.parse_price_file((ROOT / "data" / SV.PRICE_FILE).read_text(encoding="utf-8"))[0])
        doc, why = SV.parse_openrouter(json.dumps(self.SAMPLE), fam)
        self.assertIsNone(why)
        self.assertEqual({k: (v["input"], v["cached"], v["output"]) for k, v in doc["prices"].items()},
                         {"deepseek-v4.1-flash": (0.15, 0.003, 0.6), "glm-5.3-flash": (0.15, 0.03, 0.5),
                          "minimax-m3.1-flash": (0.075, 0.0075, 0.3), "hy4-flash": (0.035, 0.0015, 0.1405),
                          "qwen3.8-flash": (0.15, 0.016, 0.47), "inkling-2-mini": (0.075, 0.015, 0.3)})
        self.assertEqual(doc["prices"]["deepseek-v4.1-flash"]["peak"], 2.0)   # (peak hours stay ours: the list shows the base)
        self.assertEqual({v["openrouter"] for v in doc["prices"].values()},
                         {r["openrouter"] for r in SV.REFERENCES})            # (the mapping, end to end)
        self.assertEqual(doc["as_of"], time.strftime("%Y-%m-%d", time.gmtime()))

    def test_deepseeks_peak_survives_a_shipped_file_that_cannot_be_read(self):
        """Live on with data/api-prices.json gone: the DeepSeek family is keyed by the built-in's api name, and the
        price counted is still the one the shipped table gives - peak hours and all."""
        import io
        from contextlib import redirect_stdout
        self.repo_file.write_text("{not json")                            # (the shipped file cannot be read)
        out = io.StringIO()
        with mock.patch.object(SV.urllib.request, "urlopen",
                               return_value=answered(json.dumps(self.SAMPLE))), redirect_stdout(out):
            led = SV.Ledger(self.save / SV.FILE, "maya-ds.json", "deepseek-v4.1-flash",
                            prices=SV.PriceBook(self.save, self.repo_file, live=True))
            led.book.thread.join(10)
        self.assertEqual(out.getvalue().count("ignored"), 1)               # (the one line for the broken file)
        ref = led.price("deepseek-v4.1-flash")
        self.assertEqual((ref["input"], ref["output"]), (0.15, 0.60))      # (OpenRouter's, from the sample)
        self.assertEqual(ref["peak"], 2.0)                                 # (the shipped table's peak factor stays)
        led.add(1_000_000, 0, 0, now=TUE_PEAK)
        self.assertAlmostEqual(led.summary()["start_usd"], 2 * 0.15, places=6)   # (peak hours: counted twice)

    def test_a_list_that_holds_none_of_ours_is_refused(self):
        doc, why = SV.parse_openrouter(json.dumps({"data": [{"id": "other/thing", "pricing": {}}]}),
                                       SV.price_families(None))
        self.assertIsNone(doc)
        self.assertIn("none of our models", why)

    def test_off_by_default_nothing_is_ever_fetched(self):
        self.repo_file.write_text(self.text())
        led = self.ledger()                                            # (the default: no live_prices anywhere)
        with mock.patch.object(SV.urllib.request, "urlopen") as net:
            led.add(1_000_000, 0, 0)
            rep = led.report()
            led.report()
        net.assert_not_called()
        self.assertFalse(rep["live"])
        self.assertEqual(rep["prices_source"]["from"], "built-in")     # (the Savings line: built-in prices, <release>)
        self.assertEqual(rep["prices_source"]["release"], (ROOT / "VERSION").read_text().strip())
        self.assertEqual(led.price("glm-5.3-flash")["input"], 0.99)    # (the shipped table counts)

    def test_on_fetches_once_a_day_on_a_thread_and_caches_with_its_time(self):
        self.repo_file.write_text(self.text("2026-10", 0.10))
        with mock.patch.object(SV.urllib.request, "urlopen",
                               return_value=answered(json.dumps(self.SAMPLE))) as net:
            led = self.ledger(live=True)                               # (the start: the cache is older, so fetch)
            self.assertIsNotNone(led.book.thread)                      # ... on a thread: the start did not wait
            led.book.thread.join(10)
            self.assertEqual(led.price("glm-5.3-flash")["input"], 0.15)    # (OpenRouter's beat the shipped table)
            self.assertEqual(led.price("Qwen3.8-Flash-Next-UD-Q4_K_XL")["output"], 0.47)
            self.assertEqual(led.price("Inkling-2-Mini-UD-Q4_K_XL")["input"], 0.075)
            rep = led.report()                                         # (the page asks again: not a second fetch)
        net.assert_called_once()
        self.assertRegex(net.call_args[0][0].full_url, r"^https://openrouter\.ai/api/v1/models$")
        self.assertTrue(rep["live"])
        self.assertEqual(rep["prices_source"]["from"], "openrouter")   # (the Savings line: OpenRouter, <date>)
        today = time.strftime("%Y-%m-%d", time.gmtime())
        self.assertEqual(rep["prices_source"]["date"], today)
        self.assertEqual(rep["prices_as_of"], today)
        cached = json.loads((self.save / SV.LIVE_FILE).read_text())    # (cached beside the count, with its fetch time)
        self.assertAlmostEqual(cached["fetched"], time.time(), delta=30)
        self.assertEqual(cached["as_of"], today)
        led2 = SV.Ledger(self.save / SV.FILE, "maya-maya-m.json", "glm-5.3-flash",
                         prices=SV.PriceBook(self.save, self.repo_file, live=True))
        self.assertIsNone(led2.book.thread)                            # (a start with a fresh cache dials nothing)
        self.assertEqual(led2.price("glm-5.3-flash")["input"], 0.15)   # ... and counts from the cache

    def test_a_day_old_cache_is_refetched_at_the_start(self):
        old = time.time() - 2 * 86400
        doc, _ = SV.parse_openrouter(json.dumps(self.SAMPLE), SV.price_families(None), old)
        self.write_cache(doc, old)
        self.repo_file.write_text(self.text())
        with mock.patch.object(SV.urllib.request, "urlopen",
                               return_value=answered(json.dumps(self.SAMPLE))) as net:
            led = self.ledger(live=True)
            self.assertIsNotNone(led.book.thread)
            led.book.thread.join(10)
        net.assert_called_once()

    def test_a_network_failure_keeps_the_cache_and_logs_one_line(self):
        import io
        from contextlib import redirect_stdout
        doc, _ = SV.parse_openrouter(json.dumps(self.SAMPLE), SV.price_families(None), time.time() - 86400)
        self.write_cache(doc, time.time() - 86400)                     # (yesterday's fetch; a new day: fetch again)
        self.repo_file.write_text(self.text())
        out = io.StringIO()
        with mock.patch.object(SV.urllib.request, "urlopen", side_effect=OSError("offline")), redirect_stdout(out):
            led = self.ledger(live=True)
            led.book.thread.join(10)
        self.assertEqual(out.getvalue().count("the OpenRouter prices did not come"), 1)
        self.assertEqual(led.price("glm-5.3-flash")["input"], 0.15)    # (yesterday's cache goes on counting)
        self.assertEqual(led.report()["prices_source"]["from"], "openrouter")
        self.assertIn("offline", led.book.error)

    def test_a_failure_without_a_cache_keeps_the_shipped_table(self):
        import io
        from contextlib import redirect_stdout
        self.repo_file.write_text(self.text())
        out = io.StringIO()
        with mock.patch.object(SV.urllib.request, "urlopen", side_effect=urllib.error.URLError("no route")), \
                redirect_stdout(out):
            led = self.ledger(live=True)
            led.book.thread.join(10)
        self.assertEqual(out.getvalue().count("the OpenRouter prices did not come"), 1)
        self.assertEqual(led.price("glm-5.3-flash")["input"], 0.99)    # (the shipped table)
        self.assertEqual(led.report()["prices_source"]["from"], "built-in")
        self.assertFalse((self.save / SV.LIVE_FILE).exists())          # (nothing was cached)

    def test_a_junk_answer_is_refused_with_one_log_line_and_not_cached(self):
        import io
        from contextlib import redirect_stdout
        self.repo_file.write_text(self.text())
        out = io.StringIO()
        with mock.patch.object(SV.urllib.request, "urlopen",
                               return_value=answered("<html>not the model list</html>")), redirect_stdout(out):
            led = self.ledger(live=True)
            led.book.thread.join(10)
            rep = led.report()
        self.assertEqual(out.getvalue().count("the OpenRouter prices did not come"), 1)
        self.assertEqual(led.price("glm-5.3-flash")["input"], 0.99)    # (the shipped table still counts)
        self.assertEqual(rep["prices_source"]["from"], "built-in")
        self.assertFalse((self.save / SV.LIVE_FILE).exists())

    def test_turning_it_off_leaves_only_the_shipped_table(self):
        doc, _ = SV.parse_openrouter(json.dumps(self.SAMPLE), SV.price_families(None))
        self.write_cache(doc, time.time())                             # (a fresh cache from an on spell)
        self.repo_file.write_text(self.text())
        with mock.patch.object(SV.urllib.request, "urlopen") as net:
            led = self.ledger(live=True)
            self.assertEqual(led.price("glm-5.3-flash")["input"], 0.15)
            led.set_live(False)
            self.assertEqual(led.price("glm-5.3-flash")["input"], 0.99)    # (shipped table only)
            rep = led.report()
        net.assert_not_called()
        self.assertEqual(rep["prices_source"]["from"], "built-in")

    def test_the_setting_and_its_priority(self):
        self.assertFalse(SV.live_enabled(None, {}))                            # (off by default: nothing leaves the box)
        self.assertFalse(SV.live_enabled(True, {"SAGUARO_LIVE_PRICES": "0"}))  # (the env beats the run config)
        self.assertTrue(SV.live_enabled(False, {"SAGUARO_LIVE_PRICES": "1"}))
        self.assertTrue(SV.live_enabled(True, {"SAGUARO_LIVE_PRICES": ""}))    # (empty: the config decides)
        self.assertTrue(SV.live_enabled(True, {}))
        cfg = self.tmp / "maya-maya-m.json"
        cfg.write_text(json.dumps({"model_name": "glm-5.3-flash"}))
        led = self.ledger(config=cfg)
        self.assertFalse(led.book.live)
        with mock.patch.object(SV.urllib.request, "urlopen",
                               return_value=answered(json.dumps(self.SAMPLE))) as net:
            led.set_live(True)                                                 # (the Settings switch)
            self.assertTrue(led.book.live)
            self.assertEqual(json.loads(cfg.read_text())["live_prices"], True) # (saved: the next start keeps it)
            self.assertIsNotNone(led.book.thread)                              # (and the daily path starts at once)
            led.book.thread.join(10)
            led.set_live(False)
            self.assertEqual(json.loads(cfg.read_text())["live_prices"], False)
            led.book.start()
        net.assert_called_once()                                               # (the off spell dials nothing after)

    def test_a_page_turns_it_on_through_the_server(self):
        """POST /api/savings/live: the Settings switch, its own page only; the report answers with the new state."""
        cfg = self.tmp / "maya-maya-m.json"
        cfg.write_text(json.dumps({"model_name": "glm-5.3-flash"}))
        led = self.ledger(config=cfg)
        self.assertFalse(led.report()["live"])
        with mock.patch.object(SV.urllib.request, "urlopen",
                               return_value=answered(json.dumps(self.SAMPLE))):
            led.set_live(True)
            led.book.thread.join(10)
            rep = led.report()
        self.assertTrue(rep["live"])
        self.assertEqual(rep["prices_source"]["from"], "openrouter")


if __name__ == "__main__":
    unittest.main()
