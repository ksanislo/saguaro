"""tools/test_quality_score.py - P125h's task-score runner, tested without a GPU and without the network beyond
a local port: the answer extraction (the GPQA letter - an answer-word mention, a bracketed letter, a lone letter
line, exactly one choice named by its text, and refusals for an answer that never commits), the scoring (letter
and choice-text equivalence; the agentic call's name and every expected argument, 5 and "5" the same value, an
extra argument tolerated), the 95 % Wilson interval, the JSONL set loader, the resume (an id already in --out is
never asked again, and a dead port writes an error line instead of stopping the run), and two real runs against
`python -m serve.server --engine mock` on a free port: the multiple-choice path and the agentic tool-call path.

    python -m unittest tools.test_quality_score -v
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import quality_score as Q  # noqa: E402

NL = chr(10)
END_TC = "</" + "tool_call" + ">"          # (written in pieces: this file's tool-block tags)
GPQA = {"id": "q1", "question": "Which one of these is a mammal?",
        "choices": ["Shark", "Sparrow", "Bat", "Cod"], "answer": "C"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Extraction(unittest.TestCase):
    TABLE = [
        ("...reasoning...</think>" + NL + NL + "Answer: C", "C"),
        ("Final answer: B", "B"),
        ("The answer is (d) bat.", "D"),
        ("(B)", "B"),
        ("I think B is wrong, but the right choice is A.", "A"),
        ("The answer is Bat, the only mammal here.", "C"),          # names the choice's own text
        ("The bat (C) is the mammal.", "C"),
        ("Answer: C. Sharks and cod are fish; sparrows have feathers.", "C"),
        ("Mammals have hair; none of these do.", None),              # never commits
        ("The answer is a cat, not in the list.", None),             # "a" is not a letter
        ("The shark and the cod are fish, the sparrow a bird.", None),   # names three: not scored
    ]

    def test_the_letter_a_reply_picks(self):
        for text, want in self.TABLE:
            self.assertEqual(Q.extract_letter(text, GPQA), want, text[:50])

    def test_no_reply_think_is_read(self):
        """ask() hands the extractor the visible answer; reasoning_content comes separately off the reply, so a
        letter the model rules out inside its think text never reaches the score."""
        reply = {"text": "The only mammal here is the bat: the answer is C.",
                 "thinking": "Maybe it is the B, the sparrow?", "calls": []}
        self.assertEqual(Q.answer_for(GPQA, reply), "C")


class Scoring(unittest.TestCase):
    def test_letter_and_choice_text_equivalence(self):
        self.assertTrue(Q.is_correct(GPQA, "C"))
        self.assertFalse(Q.is_correct(GPQA, "B"))
        self.assertFalse(Q.is_correct(GPQA, ""))
        self.assertEqual(Q.expected_letter({"choices": ["a", "Bat", "c"], "answer": "bat"}), "B")

    def test_the_agentic_call_must_match_name_and_arguments(self):
        q = {"id": "t", "question": "x", "expect": {"name": "calc", "arguments": {"expr": "17*23"}}}
        self.assertTrue(Q.is_correct(q, {"name": "calc", "arguments": {"expr": "17*23"}}))
        self.assertFalse(Q.is_correct(q, {"name": "calculate", "arguments": {"expr": "17*23"}}))
        self.assertFalse(Q.is_correct(q, {"name": "calc", "arguments": {"expr": "23*17"}}))
        self.assertFalse(Q.is_correct(q, {"name": "calc", "arguments": {}}))
        self.assertFalse(Q.is_correct(q, None))                       # called nothing

    def test_arguments_compare_by_value(self):
        q = {"expect": {"name": "search_docs", "arguments": {"limit": 5, "query": "retry policy"}}}
        self.assertTrue(Q.is_correct(q, {"name": "search_docs",
                                         "arguments": {"limit": "5", "query": "  retry policy "}}))
        self.assertTrue(Q.is_correct(q, {"name": "search_docs",       # an argument it added is fine
                                         "arguments": {"limit": 5, "query": "retry policy", "sort": "new"}}))

    def test_extract_call_normalizes_the_servers_pieces(self):
        got = Q.extract_call({"tool_calls": [{"id": "c1", "function": {"name": "calc",
                                                                      "arguments": '{"expr": "17*23"}'}}]})
        self.assertEqual(got, {"name": "calc", "arguments": {"expr": "17*23"}})
        self.assertIsNone(Q.extract_call({"tool_calls": []}))


class Interval(unittest.TestCase):
    def test_the_wilson_interval(self):
        lo, hi = Q.wilson_interval(2, 3)
        self.assertAlmostEqual(lo, 0.2077, places=3)
        self.assertAlmostEqual(hi, 0.9385, places=3)
        lo, hi = Q.wilson_interval(0, 5)
        self.assertEqual(lo, 0.0)                                     # a clean 0/5 does not claim 0 % exactly
        self.assertAlmostEqual(hi, 0.4345, places=3)
        lo, hi = Q.wilson_interval(5, 5)
        self.assertAlmostEqual(lo, 0.5655, places=3)
        self.assertEqual(hi, 1.0)
        self.assertEqual(Q.wilson_interval(0, 0), (0.0, 1.0))         # nothing scored yet: the full range

    def test_always_a_range_inside_0_and_1_that_narrows_with_evidence(self):
        prev = None
        for n in (1, 5, 20, 100):
            lo, hi = Q.wilson_interval(n // 2, n)
            self.assertLessEqual(0.0, lo)
            self.assertLessEqual(hi, 1.0)
            self.assertLessEqual(lo, 0.5)
            self.assertLessEqual(0.5, hi)
            if prev is not None:
                self.assertLess(hi - lo, prev)
            prev = hi - lo


class SetFile(unittest.TestCase):
    def test_jsonl_and_array_and_filled_ids(self):
        with tempfile.TemporaryDirectory() as d:
            jl = Path(d) / "s.jsonl"
            jl.write_text(NL.join([json.dumps({"question": "one", "choices": ["a", "b"], "answer": "a"}),
                                   "",                                              # blank line: skipped
                                   json.dumps({"id": "x", "question": "two", "choices": ["a", "b"],
                                               "answer": "B"})]) + NL, encoding="utf-8")
            qs = Q.load_set(jl)
            self.assertEqual([q["id"] for q in qs], ["q1", "x"])
            arr = Path(d) / "s.json"
            arr.write_text(json.dumps([{"question": "one", "choices": ["a", "b"], "answer": "a"}]), encoding="utf-8")
            self.assertEqual(len(Q.load_set(arr)), 1)

    def test_a_record_without_a_question_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "s.jsonl"
            f.write_text(json.dumps({"id": "n", "choices": ["a", "b"], "answer": "a"}) + NL, encoding="utf-8")
            with self.assertRaises(ValueError):
                Q.load_set(f)


class ServerRun(unittest.TestCase):
    """the runner end to end against `python -m serve.server --engine mock` on a free port."""

    def serve(self, scripts):
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        cmd = [sys.executable, "-m", "serve.server", "--engine", "mock", "--port", str(port)]
        for s in scripts:
            cmd += ["--script", s]
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def stop():
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=20)
        self.addCleanup(stop)
        deadline = time.time() + 45
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        return base
            except (urllib.error.HTTPError, urllib.error.URLError, OSError):
                pass
            if proc.poll() is not None:
                self.fail("the mock server died before answering /health")
            time.sleep(0.3)
        proc.terminate()
        self.fail("the mock server never answered /health")

    def cli(self, base, *args, out):
        return subprocess.run([sys.executable, str(ROOT / "tools" / "quality_score.py"), base,
                               "--out", str(out), "--timeout", "30", *args],
                              cwd=ROOT, capture_output=True, text=True, timeout=180)

    def test_gpqa_run_scores_the_letter_and_the_ci(self):
        base = self.serve(["Let me look it over.</think>" + NL + NL + "The only mammal is the bat: the answer is C."])
        with tempfile.TemporaryDirectory() as d:
            sf, out = Path(d) / "gpqa.jsonl", Path(d) / "out.jsonl"
            sf.write_text(NL.join([json.dumps(GPQA),
                                   json.dumps({"id": "q2", "question": "And which is a bird?",
                                               "choices": ["Shark", "Sparrow", "Bat", "Cod"], "answer": "B"})]) + NL,
                          encoding="utf-8")
            r = self.cli(base, "--set", "gpqa-diamond", "--set-file", str(sf), out=out)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([x["id"] for x in rows], ["q1", "q2"])
            self.assertEqual([x["answer"] for x in rows], ["C", "C"])     # q2 wanted B: right answer, wrong question
            self.assertEqual([x["correct"] for x in rows], [True, False])
            for x in rows:
                self.assertIsInstance(x["tokens"], int)
                self.assertGreater(x["seconds"], 0.0)
            self.assertIn("accuracy: 1/2 = 50.0 %", r.stdout)
            self.assertIn("95 % CI", r.stdout)

    def test_agentic_builtin_five_tasks_end_to_end(self):
        """the built-in set against the scripted tool call: calc-sum wants calc/17*23 (correct); file-stat wants
        a different tool and name (wrong) - so the path scores both ways."""
        script = "I will call it.</think>" + NL + NL + \
            "<tool_call>calc<arg_key>expr</arg_key><arg_value>17*23</arg_value>" + END_TC
        base = self.serve([script])
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.jsonl"
            r = self.cli(base, "--set", "agentic", "--limit", "2", out=out)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([x["id"] for x in rows], ["calc-sum", "file-stat"])
            self.assertEqual(rows[0]["correct"], True)
            self.assertEqual(rows[0]["answer"], {"name": "calc", "arguments": {"expr": "17*23"}})
            self.assertEqual(rows[1]["correct"], False)
            self.assertIn("accuracy: 1/2 = 50.0 %", r.stdout)

    def test_resume_never_asks_an_id_twice(self):
        base = self.serve(["The answer is C."])
        with tempfile.TemporaryDirectory() as d:
            sf, out = Path(d) / "gpqa.jsonl", Path(d) / "out.jsonl"
            sf.write_text(NL.join([json.dumps(GPQA),
                                   json.dumps({"id": "q2", "question": "Which is a bird?",
                                               "choices": ["Shark", "Sparrow", "Bat", "Cod"], "answer": "B"}),
                                   json.dumps({"id": "q3", "question": "Which is a fish?",
                                               "choices": ["Shark", "Sparrow", "Bat", "Cod"], "answer": "A"})]) + NL,
                          encoding="utf-8")
            out.write_text(json.dumps({"id": "q1", "answer": "PRE", "correct": True,
                                       "tokens": 1, "seconds": 0.1}) + NL, encoding="utf-8")
            r = self.cli(base, "--set", "gpqa-diamond", "--set-file", str(sf), out=out)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("resuming: 1 of 3", r.stdout)
            rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 3)                               # q1 appended nothing
            self.assertEqual(rows[0]["answer"], "PRE")                   # ...and its line stayed as it was
            self.assertEqual([x["id"] for x in rows[1:]], ["q2", "q3"])
            self.assertIn("accuracy: 1/3", r.stdout)                     # only the PRE line: C answers neither q2 (B) nor q3 (A)


class ResumeNoServer(unittest.TestCase):
    """resume and the error line with the port dead: ids already scored are skipped and the rest log an error -
    the run still ends and prints, it does not hang or stop at the first refused request."""

    def test_dead_port_logs_errors_and_scores_them_wrong(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.jsonl"
            out.write_text(json.dumps({"id": "q1", "answer": "C", "correct": True,
                                       "tokens": 7, "seconds": 1.0}) + NL + "not json at all" + NL, encoding="utf-8")
            questions = [{"id": "q1", "question": "x", "choices": ["a", "b"], "answer": "A"},
                         {"id": "q2", "question": "y", "choices": ["a", "b"], "answer": "B"}]
            rc = Q.run(f"http://127.0.0.1:{free_port()}", "", questions, out, 5.0, 64)
            self.assertEqual(rc, 0)
            done = Q.read_done(out)
            self.assertEqual(done["q1"]["answer"], "C")                  # the half-written line skipped, q1 kept
            self.assertIn("error", done["q2"])
            self.assertFalse(done["q2"]["correct"])
            self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
