"""serve/test_conversations.py - the Monitor's Conversations rows: one per conversation, not one per request.  A
client re-sends the whole conversation every turn, so a resumed turn renews its row (turns 1, 2, 3) instead of
adding rows; interleaved conversations keep their own; an unrelated request gets a row of its own; an idle
conversation leaves the list.  Against the mock engine (no GPU, no pack).

    cd serve && python -m unittest -q test_conversations
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (ByteTokenizer, MockEngine, RequestStat, Service, conv_head, conv_head_key,  # noqa: E402
                          serve)

ROOT = Path(__file__).resolve().parents[1]
KBT = 12672   # (the QFN Q8's own report: 12 QSA layers x 1056 B a position)


def make_svc(info=None, max_context=32768):
    tok = ByteTokenizer()
    eng = MockEngine(tok, "x", max_context=max_context)
    eng.info = dict(eng.info, **(info or {}))
    return Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))


NODE_STUBS = """
const noop = () => {};
const el = new Proxy({}, {get: (t, k) => (k in t ? t[k] : k === "dataset" || k === "style" ? (t[k] = {})
  : k === "classList" ? (t[k] = {add: noop, remove: noop, toggle: noop}) : noop)});
globalThis.navigator = {userAgent: "node", platform: "node", maxTouchPoints: 0};
globalThis.matchMedia = () => ({matches: false, addEventListener: noop, addListener: noop});
globalThis.localStorage = {getItem: () => null, setItem: noop, removeItem: noop};
globalThis.document = {documentElement: el, body: el, head: el, addEventListener: noop, querySelector: () => null,
                       createElement: () => el};
globalThis.window = globalThis; globalThis.addEventListener = noop;
globalThis.location = {href: "http://x/", hash: "", search: "", pathname: "/"};
globalThis.fetch = () => new Promise(() => {});
const web = process.argv[2];
"""


class Turns(unittest.TestCase):
    """Three turns of one conversation -> one row with turns=3; interleaved conversations -> one row each."""

    def setUp(self):
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, "x", max_context=32768)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, messages):
        body = json.dumps({"model": "m", "messages": messages}).encode()
        r = urllib.request.Request(self.base + "/v1/chat/completions", data=body,
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=30) as resp:
            resp.read()

    def convs(self):
        with urllib.request.urlopen(self.base + "/metrics", timeout=30) as resp:
            return json.loads(resp.read())["live"]

    def test_three_turns_of_one_conversation_are_one_row(self):
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "count the stars"}]
        self.post(msgs)
        one = self.convs()["conversations"]
        self.assertEqual(len(one), 1)
        self.assertEqual((one[0]["turns"], one[0]["state"]), (1, "idle"))
        msgs += [{"role": "assistant", "content": "Many."}, {"role": "user", "content": "and the moons"}]
        self.post(msgs)
        msgs += [{"role": "assistant", "content": "A few."}, {"role": "user", "content": "and the planets"}]
        self.post(msgs)
        convs = self.convs()["conversations"]
        self.assertEqual(len(convs), 1, f"a resumed turn must renew its row, not add one: {convs}")
        c = convs[0]
        self.assertEqual((c["turns"], c["state"]), (3, "idle"))
        self.assertEqual((c["api"], c["words"]), ("openai", "count the stars"))
        self.assertGreater(c["tokens_written"], 0)
        self.assertGreaterEqual(c["idle_s"], 0)
        self.assertGreater(c["context_used"], 0)
        self.assertEqual(self.convs()["ended"][-1]["finish"], "stop")     # (the per-request list stays for readers)

    def test_two_interleaved_conversations_keep_their_rows(self):
        a = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "talk about alpha"}]
        b = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "write about zebra"}]
        self.post(a)
        self.post(b)
        self.post(a + [{"role": "assistant", "content": "Alphas."}, {"role": "user", "content": "more"}])
        self.post(b + [{"role": "assistant", "content": "Zebras."}, {"role": "user", "content": "more"}])
        convs = self.convs()["conversations"]
        self.assertEqual(len(convs), 2)
        self.assertEqual([c["words"] for c in convs], ["write about zebra", "talk about alpha"])   # newest first
        self.assertEqual([c["turns"] for c in convs], [2, 2])
        self.assertEqual(len({c["id"] for c in convs}), 2)

    def test_an_unrelated_request_gets_its_own_row(self):
        self.post([{"role": "user", "content": "hello there"}])
        self.post([{"role": "user", "content": "quite another question entirely"}])
        convs = self.convs()["conversations"]
        self.assertEqual(len(convs), 2)
        self.assertEqual([c["turns"] for c in convs], [1, 1])

    def test_idle_expiry(self):
        self.post([{"role": "user", "content": "remember me"}])
        self.assertEqual(len(self.convs()["conversations"]), 1)
        self.svc.conv_idle_s = -1.0                     # (what 2 h of quiet does to a row)
        self.assertEqual(self.convs()["conversations"], [])


class Rows(unittest.TestCase):
    """The row's shape: a running conversation's `turn` carries the fields a request row showed today, the cache
    split is the "Right now" rows', live.chats carries the conversation id, and the reuse rule re-homes a turn whose
    opening changed onto the conversation the engine still held."""

    def test_a_running_turn_keeps_its_fields_and_the_row_gains_its_totals(self):
        svc = make_svc({"kv_bytes_token": KBT, "kv_resident": 50}, max_context=4096)
        now = time.time()
        st = RequestStat()
        st.id, st.api = 7, "openai"
        st.started, st.at_engine, st.first_token = now - 5.0, True, now - 4.0
        st.phase, st.prompt_tokens, st.max_tokens, st.generated = "answering", 1000, 100, 40
        st.note(now - 4.0, 0)
        st.note(now - 0.5, 40)
        with svc.status_lock:
            st.conv = svc._conv_open(list(range(400)), "openai", "hello there", st)
            svc.running.add(st)
        live = svc.metrics()["live"]
        (c,) = live["conversations"]
        self.assertEqual(c["state"], "writing")
        self.assertEqual(c["turns"], 1)
        self.assertEqual(c["words"], "hello there")
        for k in ("state", "phase", "prompt_tokens", "generated", "max_tokens", "prompt_read", "prompt_total",
                  "prefill_tok_s", "prefill_shared", "tok_s", "elapsed_s"):
            self.assertIn(k, c["turn"])                 # (what one request's row shows today)
        self.assertEqual(c["turn"]["generated"], 40)
        self.assertGreater(c["turn"]["tok_s"], 0)
        self.assertEqual(c["context_used"], 1040)
        self.assertEqual(c["cache_vram_bytes"], 50 * KBT)     # (the same VRAM / RAM split the chats rows give)
        self.assertEqual(c["cache_ram_bytes"], 990 * KBT)
        self.assertEqual(live["chats"][0]["conv"], c["id"])   # ("Right now" / Context label the chat by it)

    def test_idle_rows_come_after_running_ones(self):
        svc = make_svc()
        now = time.time()
        st = RequestStat()
        st.started, st.first_token = now - 2.0, now - 1.5
        st.prompt_tokens, st.generated = 10, 2
        st.note(now - 1.5, 0)
        st.note(now - 0.1, 2)
        with svc.status_lock:
            past = RequestStat()
            old = svc._conv_open([1, 2, 3], "web", "first", past)
            svc._conv_idle(old, past, [1, 2, 3], 0, 3, 1, reused=0)   # (a conversation that already answered)
            st.conv = svc._conv_open([9, 8, 7], "openai", "second", st)
            svc.running.add(st)
        convs = svc.metrics()["live"]["conversations"]
        self.assertEqual([c["state"] for c in convs], ["writing", "idle"])
        self.assertEqual([c["words"] for c in convs], ["second", "first"])
        self.assertEqual(convs[1]["prompt_last"], 3)
        self.assertGreaterEqual(convs[1]["idle_s"], 0)

    def test_a_changed_opening_continues_the_conversation_the_engine_held(self):
        svc = make_svc()
        a = [1000] * 60 + list(range(100, 150)) + [7] * 500      # two prompts sharing their first 60 tokens,
        b = [1000] * 60 + list(range(200, 260)) + [8] * 500      # diverging inside the opening
        with svc.status_lock:
            sa, sb = RequestStat(), RequestStat()
            ca = svc._conv_open(a, "openai", "one", sa)
            svc._conv_idle(ca, sa, a, 0, len(a), 5, reused=0, read_tok_s=100.0, decode_tok_s=40.0)
            cb = svc._conv_open(b, "openai", "two", sb)
            self.assertNotEqual(ca.key, cb.key)                  # (its opening is new: a row of its own for now)
            self.assertEqual(len(svc.convs), 2)
            svc._conv_idle(cb, sb, b, 0, len(b), 7, reused=60)   # (the engine held the first 60 tokens: its prefix)
        self.assertEqual(len(svc.convs), 1)
        (c,) = svc.convs.values()
        self.assertEqual((c.id, c.turns), (ca.id, 2))
        self.assertEqual(c.key, conv_head_key(conv_head(b, 0)))   # (it lives under this turn's opening now)
        with svc.status_lock:                                     # (the next turn lands right on that row)
            again = svc._conv_open(b, "openai", "two", RequestStat())
        self.assertIs(again, c)
        self.assertEqual(again.turns, 3)

    def test_a_growing_young_conversation_finds_its_row(self):
        svc = make_svc()
        with svc.status_lock:
            s1, s2 = RequestStat(), RequestStat()
            c1 = svc._conv_open([1, 2, 3], "web", "hi", s1, 1)          # (prompt "hi", then the reply's opening)
            svc._conv_idle(c1, s1, [1, 2, 3], 1, 2, 1, reused=0)
            c2 = svc._conv_open([1, 2, 3, 4, 5], "web", "hi", s2, 1)    # (turn 2: the opening grew with the chat)
            self.assertEqual(len(svc.convs), 2)                          # (a new row while its turn runs...)
            svc._conv_idle(c2, s2, [1, 2, 3, 4, 5], 1, 4, 1, reused=0)
        self.assertEqual(len(svc.convs), 1)                              # (...gone once it found its conversation)
        (c,) = svc.convs.values()
        self.assertEqual((c.id, c.turns), (c1.id, 2))


@unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
class PageLoads(unittest.TestCase):
    """monitor.js (the rewritten Conversations card) parses and imports in Node."""

    def test_monitor_module_loads(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.mjs"
            p.write_text(NODE_STUBS + 'await import(web + "/monitor.js");\n'
                         'process.stdout.write(JSON.stringify({ok: true}) + "\\n");', encoding="utf-8")
            r = subprocess.run(["node", str(p), (ROOT / "serve" / "web").resolve().as_uri()],
                               capture_output=True, encoding="utf-8", timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertTrue(json.loads(r.stdout.strip().splitlines()[-1])["ok"])


if __name__ == "__main__":
    unittest.main()
