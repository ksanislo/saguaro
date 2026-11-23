"""tools/smoke_profiles.py without a server: the SSE parsing, the leak check, the api/context check and one whole
check_model() driven by the fixtures under tools/saguaro/fixtures/ through a stub opener.  Nothing here talks to
127.0.0.1 or any other host.

    python -m unittest tools.test_smoke_profiles
"""
from __future__ import annotations

import io
import json
import sys
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "saguaro"))
import smoke_profiles as sp  # noqa: E402

FIX = ROOT / "tools" / "saguaro" / "fixtures"
SIX = ["DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS", "DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL", "GLM-5.3-Flash-Maya-L-IQ3_S",
       "MiniMax-M3.1-Flash-Maya-S", "Qwen3.8-Flash-Next-Q8_0", "Qwen3.8-Flash-Next-UD-Q4_K_XL"]
KEY = "sekret-key-999"


def fixture(name: str) -> bytes:
    return (FIX / name).read_bytes()


class FakeResp:
    def __init__(self, payload: bytes):
        self._b = payload

    def read(self, n: int = -1) -> bytes:
        part, self._b = (self._b[:n], self._b[n:]) if n and n > 0 else (self._b, b"")
        return part

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def stub_opener(chat: bytes, context: bytes, seen: list | None = None):
    def op(req, timeout=None):
        if seen is not None:
            seen.append(req)
        if req.full_url.endswith("/v1/chat/completions"):
            return FakeResp(chat)
        if "/api/context" in req.full_url:
            return FakeResp(context)
        raise AssertionError(f"unexpected url {req.full_url}")
    return op


class Defaults(unittest.TestCase):
    # ref-configs/ is the deploy's (git-ignored): without it the default list is simply empty
    @unittest.skipUnless((ROOT / "ref-configs").is_dir(), "no ref-configs/ here (the deploy's, not the repository's)")
    def test_the_six_ref_config_ids(self):
        self.assertEqual(sorted(sp.default_model_ids()), sorted(SIX))
        self.assertEqual(len(sp.default_model_ids()), 6)

    @unittest.skipUnless((ROOT / "ref-configs").is_dir(), "no ref-configs/ here (the deploy's, not the repository's)")
    def test_main_list_prints_them(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = sp.main(["--list"])
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue().split(), SIX)


class Sse(unittest.TestCase):
    def test_good_stream(self):
        events, done, errs = sp.sse_decode(fixture("stream-good.sse").decode())
        self.assertEqual(errs, [])
        self.assertTrue(done)
        self.assertEqual(len(events), 3)
        text, think, finishes, cerr = sp.answer_from_events(events)
        self.assertEqual((text, think, finishes, cerr), ("Hello there.", "", ["stop"], []))

    def test_broken_stream_reports_lines(self):
        events, done, errs = sp.sse_decode(fixture("stream-broken.sse").decode())
        self.assertTrue(done)
        self.assertEqual(len(events), 1)
        self.assertEqual(len(errs), 2)
        self.assertIn("not an SSE field", errs[0])
        self.assertIn("not JSON", errs[1])

    def test_empty_stream(self):
        events, done, errs = sp.sse_decode("")
        self.assertEqual((events, done, errs), ([], False, []))

    def test_leaks_found(self):
        events, _, _ = sp.sse_decode(fixture("stream-leak.sse").decode())
        text, _, _, _ = sp.answer_from_events(events)
        self.assertEqual(sorted(sp.find_leaks(text)), sorted(name for name, _ in sp.LEAKS))

    def test_clean_text_has_no_leaks(self):
        self.assertEqual(sp.find_leaks("Hello there. One short sentence."), [])

    def test_error_event(self):
        events = [{"error": {"message": "engine died"}}, {"choices": [{"delta": {}, "finish_reason": "error"}]}]
        text, _, finishes, errs = sp.answer_from_events(events)
        self.assertEqual(finishes, ["error"])
        self.assertIn("engine died", errs[0])


class Context(unittest.TestCase):
    def test_good(self):
        ok, why = sp.context_has_kv(json.loads(fixture("context-good.json")))
        self.assertTrue(ok)
        self.assertIsNone(why)

    def test_absent(self):
        ok, why = sp.context_has_kv(json.loads(fixture("context-no-kv.json")))
        self.assertFalse(ok)
        self.assertIn("kv_bytes_token", why)

    def test_null_is_not_a_figure(self):
        ok, why = sp.context_has_kv({"kv_bytes_token": None})
        self.assertFalse(ok)
        self.assertIn("not reported", why)


class CheckModel(unittest.TestCase):
    def test_pass(self):
        ok, reasons = sp.check_model("Qwen3.8-Flash-Next-Q8_0", "http://127.0.0.1:8080", KEY,
                                     opener=stub_opener(fixture("stream-good.sse"), fixture("context-good.json")))
        self.assertTrue(ok, reasons)
        self.assertEqual(reasons, [])

    def test_leak_fails(self):
        ok, reasons = sp.check_model("m", "http://h/", None,
                                     opener=stub_opener(fixture("stream-leak.sse"), fixture("context-good.json")))
        self.assertFalse(ok)
        self.assertIn("protocol leaked", "; ".join(reasons))

    def test_no_kv_fails(self):
        ok, reasons = sp.check_model("m", "http://h", None,
                                     opener=stub_opener(fixture("stream-good.sse"), fixture("context-no-kv.json")))
        self.assertFalse(ok)
        self.assertIn("kv_bytes_token", "; ".join(reasons))

    def test_broken_sse_and_no_finish_fail(self):
        ok, reasons = sp.check_model("m", "http://h", None,
                                     opener=stub_opener(fixture("stream-broken.sse"), fixture("context-good.json")))
        self.assertFalse(ok)
        joined = "; ".join(reasons)
        self.assertIn("unparsable SSE", joined)

    def test_reasoning_only_cut_at_length_passes(self):
        sse = (b'data: {"choices":[{"delta":{"reasoning_content":"Let me think about one short sentence"}}]}\n\n'
               b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n')
        ok, reasons = sp.check_model("m", "http://h", None, opener=stub_opener(sse, fixture("context-good.json")))
        self.assertTrue(ok, reasons)

    def test_reasoning_only_stopped_fails(self):
        sse = (b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n'
               b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
        ok, reasons = sp.check_model("m", "http://h", None, opener=stub_opener(sse, fixture("context-good.json")))
        self.assertFalse(ok)
        self.assertIn("empty answer text", "; ".join(reasons))

    def test_default_budget_fits_thinking(self):
        sent = []

        def opener(req, timeout=None):
            if req.data:
                sent.append(json.loads(req.data))
            return stub_opener(fixture("stream-good.sse"), fixture("context-good.json"))(req, timeout)
        sp.check_model("m", "http://h", None, opener=opener)
        self.assertGreaterEqual(sent[0]["max_tokens"], 400)

    def test_empty_stream_fails(self):
        ok, reasons = sp.check_model("m", "http://h", None, opener=stub_opener(b"", fixture("context-good.json")))
        self.assertFalse(ok)
        self.assertIn("empty stream", "; ".join(reasons))
        self.assertIn("no data: [DONE]", "; ".join(reasons))

    def test_http_error_is_the_reason(self):
        def boom(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", None,
                                         io.BytesIO(b'{"error": "model not loaded"}'))
        ok, reasons = sp.check_model("m", "http://h", None, opener=boom)
        self.assertFalse(ok)
        self.assertIn("chat HTTP 503", reasons[0])
        self.assertIn("model not loaded", reasons[0])

    def test_key_is_sent_but_never_printed(self):
        seen = []
        ok, reasons = sp.check_model("m", "http://h", KEY,
                                     opener=stub_opener(fixture("stream-good.sse"), fixture("context-good.json"), seen))
        self.assertTrue(ok, reasons)
        self.assertEqual(len(seen), 2)                                # (chat + context, both authorized)
        for req in seen:
            self.assertEqual(req.get_header("Authorization"), f"Bearer {KEY}")
        self.assertNotIn("sekret", "; ".join(reasons))

        def boom(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 500, "oops", None, io.BytesIO(b"bad"))
        _, reasons = sp.check_model("m", "http://h", KEY, opener=boom)
        self.assertNotIn("sekret", "; ".join(reasons))

    def test_upstream_url_is_escaped(self):
        seen = []
        sp.check_model("DeepSeek-V4.1-Flash 1017/XS", "http://h/", None,
                       opener=stub_opener(fixture("stream-good.sse"), fixture("context-good.json"), seen))
        self.assertIn("/upstream/DeepSeek-V4.1-Flash%201017%2FXS/api/context", seen[1].full_url)


class Main(unittest.TestCase):
    def test_missing_key_file_exits_2(self):
        err = io.StringIO()
        with redirect_stdout(err):
            rc = sp.main(["--key-file", "scratch/no-such-key", "--list"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
