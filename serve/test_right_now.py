"""serve/test_right_now.py - /metrics' live.chats (the Monitor's "Right now": every chat queued or with the engine,
its own state, progress and rates, and what its cache costs) and the page's chatsModel over it.

    cd serve && python -m unittest -q test_right_now
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, RequestStat, Service  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096
KBT = 12672   # (the QFN Q8's own report: 12 QSA layers x 1056 B a position)


def make_svc(info=None):
    tok = ByteTokenizer()
    eng = MockEngine(tok, "x", max_context=CTX)
    eng.info = dict(info or {})
    return Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))


def reading_stat(i, started):
    st = RequestStat()
    st.id, st.api = i, "web"
    st.started, st.at_engine = started, True
    st.phase, st.prompt_tokens, st.max_tokens = "reading the prompt", 1000, 500
    st.progress, st.read_avg = (400, 1000, 0, 1.0), 900.0
    return st


def writing_stat(i, started, now):
    st = RequestStat()
    st.id, st.api = i, "openai"
    st.started, st.at_engine, st.first_token = started, True, started + 0.2
    st.phase, st.prompt_tokens, st.max_tokens, st.generated = "answering", 100, 100, 15
    st.note(started + 0.2, 0)
    st.note(now - 0.1, 15)
    st.spec_off = False
    return st


def queued_stat(i, at):
    st = RequestStat()
    st.id, st.api, st.queued_at = i, "anthropic", at
    st.prompt_tokens, st.max_tokens = 900, 50
    return st


class LiveChats(unittest.TestCase):
    """/metrics' live.chats: one row per request queued or with the engine - the per-request state, not the
    single shared status - with stable ids, both its rates, and the cache its context costs."""

    def build(self, info):
        svc = make_svc(info)
        now = time.time()
        r, w, q = reading_stat(11, now - 2.0), writing_stat(12, now - 12.0, now), queued_stat(13, now - 30.0)
        svc.running.update({r, w})
        svc.current_requests.add(w)                     # (as run() does from the first answer token on)
        svc.latest_request = w
        svc.waiting.add(q)
        svc.status.update(busy=True, phase="answering", generated=15, prompt_tokens=100, max_tokens=100)
        return svc, r, w, q

    def test_two_chats_plus_a_queued_one(self):
        svc, r, w, q = self.build({"kv_bytes_token": KBT, "kv_resident": 50})
        m = svc.metrics()
        self.assertEqual(m["live"]["state"], "reading")            # aggregate: any chat reading reads as reading
        self.assertIsNotNone(m["live"]["tok_s"])                   # ... and the writing one keeps its live Decode
        self.assertEqual(m["live"]["reading_n"], 1)
        chats = {c["id"]: c for c in m["live"]["chats"]}
        self.assertEqual(sorted(chats), [11, 12, 13])
        self.assertEqual(chats[11]["state"], "reading")
        self.assertEqual((chats[11]["prompt_read"], chats[11]["prompt_total"], chats[11]["prefill_tok_s"]),
                         (400, 1000, 900.0))
        self.assertEqual(chats[12]["state"], "writing")
        self.assertGreater(chats[12]["tok_s"], 0)
        self.assertEqual(chats[12]["generated"], 15)
        self.assertEqual(chats[13]["state"], "queued")
        self.assertGreater(chats[13]["elapsed_s"], 0)
        self.assertIsNone(chats[13]["prompt_read"])
        # ids are stable reads of the same requests
        again = {c["id"] for c in svc.metrics()["live"]["chats"]}
        self.assertEqual(again, {11, 12, 13})

    def test_cache_bytes_per_chat_and_streaming_split(self):
        # kv_resident 50 of 4096: the first 50 positions in VRAM, the rest in RAM - at the engine's own bytes/token
        svc, r, w, q = self.build({"kv_bytes_token": KBT, "kv_resident": 50})
        chats = {c["id"]: c for c in svc.metrics()["live"]["chats"]}
        self.assertEqual(chats[11]["context_used"], 1000)
        self.assertEqual(chats[11]["cache_vram_bytes"], 50 * KBT)
        self.assertEqual(chats[11]["cache_ram_bytes"], 950 * KBT)
        self.assertEqual(chats[12]["cache_vram_bytes"], 50 * KBT)
        self.assertEqual(chats[12]["cache_ram_bytes"], 65 * KBT)
        self.assertNotIn("cache_vram_bytes", chats[13])            # queued: nothing loaded yet

    def test_no_sizes_without_the_report(self):
        svc, r, w, q = self.build({"kv_resident": 50})             # no kv_bytes_token: no size is claimed
        chats = {c["id"]: c for c in svc.metrics()["live"]["chats"]}
        self.assertNotIn("cache_vram_bytes", chats[11])
        self.assertEqual(chats[11]["context_used"], 1000)

    def test_no_ram_split_without_streaming(self):
        svc, r, w, q = self.build({"kv_bytes_token": KBT, "kv_resident": 32768})   # >= context: whole cache in VRAM
        chats = {c["id"]: c for c in svc.metrics()["live"]["chats"]}
        self.assertEqual(chats[11]["cache_vram_bytes"], 1000 * KBT)
        self.assertEqual(chats[11]["cache_ram_bytes"], 0)

    def test_thinking_and_tool_phases(self):
        svc = make_svc({})
        now = time.time()
        t = writing_stat(21, now - 8.0, now)
        t.phase = "thinking"
        u = writing_stat(22, now - 9.0, now)
        u.phase = "writing a tool call: search"
        svc.running.update({t, u})
        svc.status.update(busy=True)
        chats = {c["id"]: c for c in svc.metrics()["live"]["chats"]}
        self.assertEqual(chats[21]["state"], "thinking")
        self.assertEqual(chats[22]["state"], "tool")

    def test_one_chat_one_row(self):
        svc = make_svc({})
        w = writing_stat(31, time.time() - 3.0, time.time())
        svc.running.add(w)
        svc.status.update(busy=True)
        m = svc.metrics()
        self.assertEqual(m["live"]["state"], "generating")         # nothing reading: as today
        self.assertEqual([c["id"] for c in m["live"]["chats"]], [31])
        self.assertEqual(m["live"]["chats"][0]["state"], "writing")


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


@unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
class PageChatsModel(unittest.TestCase):
    """monitor.js's chatsModel (the "Right now" rows, pure): summary counts, the order, and the +N more fold."""

    def node(self, body):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.mjs"
            p.write_text(NODE_STUBS + body, encoding="utf-8")
            r = subprocess.run(["node", str(p), (ROOT / "serve" / "web").resolve().as_uri()],
                               capture_output=True, encoding="utf-8", timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_rows_summary_and_fold(self):
        got = self.node("""
const {chatsModel} = await import(web + "/monitor.js");
const mk = (id, state, elapsed_s) => ({id, label: "web", state, elapsed_s, generated: 3});
const eight = {live: {chats: [mk(1, "reading", 2), mk(2, "writing", 12), mk(3, "queued", 30), mk(4, "thinking", 40),
                         mk(5, "tool", 5), mk(6, "writing", 6), mk(7, "queued", 70), mk(8, "reading", 8)]}};
const r = chatsModel(eight);
const single = chatsModel({live: {chats: [mk(9, "writing", 1)]}});
const none = chatsModel({live: {}});
process.stdout.write(JSON.stringify({summary: r.summary, order: r.rows.map((x) => x.id),
                                     shown: r.shown.length, more: r.more, single, none}) + "\\n");
""")
        self.assertEqual(got["summary"], "4 writing, 2 reading, 2 waiting")  # thinking/tool count as writing
        self.assertEqual(got["order"], [7, 4, 3, 2, 8, 6, 5, 1])             # oldest start first
        self.assertEqual((got["shown"], got["more"]), (6, 2))                 # above six: +N more
        self.assertIsNone(got["single"])                                      # one chat: the card looks like today
        self.assertIsNone(got["none"])


if __name__ == "__main__":
    unittest.main()
