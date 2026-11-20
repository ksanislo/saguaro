"""serve/test_conversations.py - the Monitor's Conversations rows: one per conversation, not one per request.  A
turn continues a conversation when its prompt starts with exactly that row's LAST PROMPT (minus the open reply's
generation tail) - the part every later turn re-sends byte-for-byte, as the engine's own prefix reuse keys on; an
opening is no identity.  A resumed turn renews its row (turns 1, 2, 3), two chats sharing a system prompt part at
the second turn, a fork takes its own row (the branch that ran first advanced the last prompt), an unrelated request
gets a row of its own; an idle conversation leaves the list.  Against the mock engine (no GPU, no pack).

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
from serve.server import (ByteTokenizer, MockEngine, RequestStat, Service, conv_hist_hash,  # noqa: E402
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
    """Three turns of one conversation -> one row with turns=3; interleaved conversations keep their own: a resumed
    prompt starts with the row's last prompt, whatever the engine answered with."""

    def setUp(self):
        self.httpd = None
        self._serve("x")

    def _serve(self, script):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, script, max_context=32768)
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

    def test_two_chats_with_the_same_opening_part_at_the_second_turn(self):
        # the live bug (11-20): agents share one system prompt - and chats can share a first message too.  Until
        # anything differs there is nothing to tell two chats apart (the engine could not either: its cache is the
        # same); the two rows appear at the SECOND turn, where the two prompts no longer continue one last prompt.
        shared = [{"role": "system", "content": "You are opencode."},
                  {"role": "user", "content": "fix the build"}]
        self.post(shared)
        self.post(shared)                                                   # (the same prompt: nothing says two)
        convs = self.convs()["conversations"]
        self.assertEqual((len(convs), convs[0]["turns"]), (1, 2))
        self.post(shared + [{"role": "assistant", "content": "first answer"},
                            {"role": "user", "content": "run the tests"}])
        self.post(shared + [{"role": "assistant", "content": "second answer"},
                            {"role": "user", "content": "deploy it"}])
        convs = self.convs()["conversations"]
        self.assertEqual(len(convs), 2,
                         f"the second turns part: neither continues the other's last prompt: {convs}")
        self.assertEqual(sorted(c["turns"] for c in convs), [1, 3])         # (one row carries the shared opening)

    def test_a_fork_makes_two_rows(self):
        # the first branch advances the row's last prompt; the second (an edited retry) no longer starts with it
        msgs = [{"role": "user", "content": "one question"}]
        self.post(msgs)
        self.post(msgs + [{"role": "assistant", "content": "x."}, {"role": "user", "content": "left branch"}])
        self.post(msgs + [{"role": "assistant", "content": "x."}, {"role": "user", "content": "right branch"}])
        convs = self.convs()["conversations"]
        self.assertEqual(len(convs), 2, f"a fork continues nothing and takes a row of its own: {convs}")
        self.assertEqual(sorted(c["turns"] for c in convs), [1, 2])

    def test_three_interleaved_agents_are_three_rows(self):
        # live 11-20: three opencode agents (the SAME system prompt) all merged into one row - three rows now
        sysp = {"role": "system", "content": "You are opencode, an interactive CLI tool."}
        msgs = {k: [sysp, {"role": "user", "content": f"agent {k} task"}] for k in "abc"}
        for k in "abc":
            self.post(msgs[k])
        for k, r in zip("abc", ("a1", "b1", "c1")):
            msgs[k] += [{"role": "assistant", "content": r}, {"role": "user", "content": "keep going"}]
        for k in "abc":
            self.post(msgs[k])
        convs = self.convs()["conversations"]
        self.assertEqual(len(convs), 3, f"three agents are three conversations, not one: {convs}")
        self.assertEqual([c["turns"] for c in convs], [2, 2, 2])           # newest first: c, b, a
        self.assertEqual([c["words"] for c in convs], ["agent c task", "agent b task", "agent a task"])

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
    split is the "Right now" rows', live.chats carries the conversation id - and a turn is a row's continuation:
    the prompt that starts with the row's last prompt joins it, one that only shares an opening does not."""

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
            svc._conv_idle(old, past, [1, 2, 3], 0, 3, 1, reused=0)      # (answered: its key is now [1, 2, 3])
            st.conv = svc._conv_open([9, 8, 7], "openai", "second", st)   # (starts with nothing of it: a new row)
            svc.running.add(st)
        convs = svc.metrics()["live"]["conversations"]
        self.assertEqual([c["state"] for c in convs], ["writing", "idle"])
        self.assertEqual([c["words"] for c in convs], ["second", "first"])
        self.assertEqual(convs[1]["prompt_last"], 3)
        self.assertGreaterEqual(convs[1]["idle_s"], 0)

    def test_a_resumed_turn_continues_the_row_it_started(self):
        # turn 2's prompt starts with turn 1's prompt (the answer between them may re-render however it likes):
        svc = make_svc()
        with svc.status_lock:
            s1, s2 = RequestStat(), RequestStat()
            c1 = svc._conv_open([1, 2, 3], "web", "hi", s1, 1)          # (prompt "hi", then the reply's opening)
            svc._conv_idle(c1, s1, [1, 2, 3], 1, 2, 1, reused=0)        # (its key: the prompt without that opening)
            self.assertEqual(c1.cont_len, 2)
            c2 = svc._conv_open([1, 2, 4, 5, 6, 7], "web", "hi", s2, 1)  # (the answer's tokens, a new message, ...)
            self.assertIs(c2, c1)                                       # (continued on the spot, not re-homed)
            self.assertEqual(len(svc.convs), 1)
            self.assertEqual(c1.turns, 2)
            svc._conv_idle(c2, s2, [1, 2, 4, 5, 6, 7], 1, 6, 1, reused=0)
        self.assertEqual(len(svc.convs), 1)
        (c,) = svc.convs.values()
        self.assertEqual((c.id, c.turns, c.cont_len), (c1.id, 2, 5))    # (the key grew to this turn's prompt)
        self.assertEqual(c.cont_hash, conv_hist_hash([1, 2, 4, 5, 6]))

    def test_the_same_opening_with_two_continuations_is_two_rows(self):
        # two prompts sharing their whole opening (the shared system prompt: 60 tokens, then everything else
        # differs): neither continues the other's history - the second takes its own row from its first turn
        svc = make_svc()
        a = [1000] * 60 + list(range(100, 150)) + [7] * 500
        b = [1000] * 60 + list(range(200, 260)) + [8] * 500
        with svc.status_lock:
            sa, sb = RequestStat(), RequestStat()
            ca = svc._conv_open(a, "openai", "one", sa)
            svc._conv_idle(ca, sa, a, 0, len(a), 5, reused=0, read_tok_s=100.0, decode_tok_s=40.0)
            cb = svc._conv_open(b, "openai", "two", sb)
            self.assertIsNot(cb, ca)
            self.assertEqual(len(svc.convs), 2)                         # (an opening is no identity)
            svc._conv_idle(cb, sb, b, 0, len(b), 7, reused=0)
            # a turn of chat a continues a's row (the longest last prompt it starts with) - and only a's
            again = svc._conv_open(a + [9] + [5] * 40, "openai", "one", RequestStat())
            self.assertIs(again, ca)
        self.assertEqual(again.turns, 2)
        self.assertEqual(len(svc.convs), 2)

    def test_a_fork_continues_nothing_and_takes_its_own_row(self):
        # the branch that ran first advanced the row's history; the second branch shares the prompt but not the
        # finished turn: it matches no history and opens its own row
        svc = make_svc()
        shared = [1000] * 60 + list(range(100, 150))
        with svc.status_lock:
            s1, s2 = RequestStat(), RequestStat()
            c1 = svc._conv_open(shared + [7] * 500, "openai", "one", s1)
            svc._conv_idle(c1, s1, shared + [7] * 500, 0, 660, 5, reused=0)   # (a's last prompt now reaches further)
            c2 = svc._conv_open(shared + [8] * 500, "openai", "two", s2)  # (the fork: no longer starts with it)
            self.assertIsNot(c2, c1)
            svc._conv_idle(c2, s2, shared + [8] * 500, 0, 660, 7, reused=0)
        self.assertEqual(len(svc.convs), 2)                             # (both branches keep their own row)

    def test_a_resumed_turn_after_a_tool_call_continues_the_row(self):
        # a tool call's round-trip is no new conversation: the client re-sends the prompt, the answer the model
        # wrote (the call) and the tool's result - the prompt still starts with the recorded history
        svc = make_svc()
        with svc.status_lock:
            s1, s2 = RequestStat(), RequestStat()
            c1 = svc._conv_open([10, 11, 12], "openai", "run it", s1, 1)
            svc._conv_idle(c1, s1, [10, 11, 12], 1, 3, 2, reused=0)      # (its key: [10, 11] - the gen tail left out)
            again = svc._conv_open([10, 11, 12, 77, 78, 90, 91, 92], "openai", "run it", s2, 1)
            # (the client re-sends the written tool call and the tool's result; only a re-render of the prompt
            #  itself could break the continuation - and it does not)
            self.assertIs(again, c1)
            self.assertEqual(again.turns, 2)
            self.assertEqual(again.cont_len, 2)                          # (it has not answered yet: still turn 2)


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
