"""tools/check_monitor.py over small fixture files: the Monitor's figures (ui-data.jsonl sampled every 2 s) against
the finished reads and answers the engine log states.  No server, no GPU.

    python -m unittest tools.test_check_monitor
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "saguaro"))
import check_monitor as cm  # noqa: E402

FIX = ROOT / "tools" / "saguaro" / "fixtures"
LOG = (FIX / "engine-reads.txt").read_text(encoding="utf-8")
# the log's own clock as epoch (the ui-data's t is epoch too, same machine, same local time)
T0 = cm.stamp_to_epoch("2026-11-19 10:00:00")


def line(t, state="idle", reading_n=0, prefill=None, tok_s=None, peak_pf=None, peak_dec=None, last_req=None, chats=None):
    return json.dumps({"t": round(t, 3), "state": state, "reading_n": reading_n, "prefill_tok_s": prefill,
                       "tok_s": tok_s, "peak_prefill": ({"tok_s": peak_pf, "tokens": 3072, "time": t} if peak_pf else None),
                       "peak_decode": ({"tok_s": peak_dec, "tokens": 40, "time": t} if peak_dec else None),
                       "chats": chats or [], "last_req": last_req})


def clean_ui():
    """Ten 2 s samples around the fixture log: every figure the Monitor shows agrees with the engine."""
    return [
        line(T0 - 1.0),
        line(T0 + 0.0, state="reading", reading_n=1, prefill=1536.0),                    # (the 3072-token read runs)
        line(T0 + 1.0, peak_pf=1536.0,
             last_req={"time": T0 - 1.05, "prompt_tokens": 4096, "reused": 1024, "prefill_tok_s": 1536.0}),
        line(T0 + 7.0, state="reading", reading_n=1, prefill=171.0),                     # (the side-thread read)
        line(T0 + 9.0, peak_pf=1536.0, peak_dec=8.0,
             last_req={"time": T0 + 5.9, "prompt_tokens": 512, "reused": 0, "prefill_tok_s": 171.0}),
        line(T0 + 11.0,                                                                 # (a fully reused prompt: no rate)
             last_req={"time": T0 + 9.9, "prompt_tokens": 1024, "reused": 1024, "prefill_tok_s": None}),
        line(T0 + 13.0),
        line(T0 + 18.0, state="reading", reading_n=1, prefill=1024.0),                   # (the second side-thread read)
        line(T0 + 21.0, peak_pf=1536.0, peak_dec=8.0,
             last_req={"time": T0 + 15.9, "prompt_tokens": 4096, "reused": 0, "prefill_tok_s": 1024.0}),
        line(T0 + 23.0, peak_pf=1536.0, peak_dec=8.0),
    ]


def run(ui_lines, log_text=None):
    return cm.check("\n".join(ui_lines) + "\n", LOG if log_text is None else log_text)


def bad_in(res, cat):
    return res[cat][1]


class ParseEngineLog(unittest.TestCase):
    def test_the_three_line_kinds(self):
        reads, decodes, errors = cm.parse_engine_log(LOG)
        self.assertEqual(errors, [])
        self.assertEqual([r.kind for r in reads], ["admitted", "thread", "admitted", "thread"])
        r0 = reads[0]
        self.assertEqual((r0.tokens, r0.reused, r0.new, r0.ms), (4096, 1024, 3072, 2000.0))
        self.assertAlmostEqual(r0.rate, 1536.0)
        self.assertEqual(reads[1].target, 512)
        self.assertEqual(reads[2].new, 0)
        self.assertEqual(len(decodes), 2)
        self.assertAlmostEqual(decodes[0].rate, 8.0)                  # (the line's own tok/s wins over n/s)
        self.assertEqual(decodes[1].tokens, 100)

    def test_unstamped_event_is_reported(self):
        log = LOG + "saguaro batch: slot 2 admitted beside the decoding slots: 256 prompt tokens (0 reused) read in 100 ms\n"
        reads, _, errors = cm.parse_engine_log(log)
        self.assertEqual(len(reads), 5)
        self.assertIsNone(reads[4].when)
        self.assertEqual(len(errors), 1)
        res = cm.check("\n".join(clean_ui()) + "\n", log)
        self.assertEqual(len(bad_in(res, "engine-log")), 1)


class CleanPair(unittest.TestCase):
    def test_everything_agrees(self):
        res = run(clean_ui())
        for cat, (checked, bad) in res.items():
            self.assertEqual(bad, [], f"{cat} mismatched: {bad}")
        self.assertEqual(res["read-rate"][0], 1)                      # (the admitted read compared by figure)
        self.assertEqual(res["live-prefill"][0], 3)                   # (the three live read samples)
        self.assertEqual(res["ui-data"][0], 10)


class SeededMismatches(unittest.TestCase):
    def test_read_rate_over_25_percent_off(self):
        ui = clean_ui()
        ui[2] = line(T0 + 1.0, peak_pf=1536.0,
                     last_req={"time": T0 - 1.05, "prompt_tokens": 4096, "reused": 1024, "prefill_tok_s": 900.0})
        res = run(ui)
        bad = bad_in(res, "read-rate")
        self.assertEqual(len(bad), 1)
        self.assertIn("900 tok/s", bad[0])
        self.assertIn("-41%", bad[0])
        self.assertIn("last_req", bad[0])

    def test_finished_read_with_no_monitor_figure(self):
        ui = clean_ui()
        ui[1] = line(T0 + 0.0)                                        # (no live sample caught the read)
        ui[2] = line(T0 + 1.0, peak_pf=1536.0)                        # (and the finished row carries no rate)
        res = run(ui)
        bad = bad_in(res, "read-rate")
        self.assertEqual(len(bad), 1)
        self.assertIn("no figure", bad[0])

    def test_side_thread_read_unseen_by_the_monitor(self):
        ui = clean_ui()
        ui[3] = line(T0 + 7.0)                                        # (the 3 s concurrent read: nothing sampled it)
        res = run(ui)
        bad = bad_in(res, "read-rate")
        self.assertEqual(len(bad), 1)
        self.assertIn("thread read", bad[0])

    def test_peak_above_any_real_read(self):
        ui = [l.replace('"peak_prefill": {"tok_s": 1536.0,', '"peak_prefill": {"tok_s": 4000.0,') for l in clean_ui()]
        res = run(ui)
        bad = bad_in(res, "peak")
        self.assertEqual(len(bad), 1)
        self.assertIn("peak_prefill = 4000", bad[0])

    def test_peak_decode_above_any_real_answer(self):
        ui = [l.replace('"peak_decode": {"tok_s": 8.0,', '"peak_decode": {"tok_s": 30.0,') for l in clean_ui()]
        res = run(ui)
        bad = bad_in(res, "peak")
        self.assertEqual(len(bad), 1)
        self.assertIn("peak_decode = 30", bad[0])

    def test_peak_never_falls(self):
        ui = clean_ui()
        ui[9] = line(T0 + 23.0, peak_pf=500.0, peak_dec=8.0)          # (a maximum that dropped)
        res = run(ui)
        bad = bad_in(res, "peak")
        self.assertEqual(len(bad), 1)
        self.assertIn("fell", bad[0])

    def test_live_figure_taken_from_the_chats_row(self):
        # (no finished-request row: the reading chat's own figure is the Monitor's number for the read)
        ui = clean_ui()
        ui[1] = line(T0 + 0.0, state="reading", reading_n=1, chats=[{"state": "reading", "prefill_tok_s": 1000.0}])
        ui[2] = line(T0 + 1.0, peak_pf=1536.0)
        res = run(ui)
        bad = bad_in(res, "read-rate")
        self.assertEqual(len(bad), 1)
        self.assertIn("1000 tok/s", bad[0])
        self.assertIn("live", bad[0])

    def test_live_prefill_while_nothing_reads(self):
        ui = clean_ui()
        ui[6] = line(T0 + 13.0, state="reading", prefill=300.0)       # (reading_n stays 0, no read covers this)
        res = run(ui)
        bad = bad_in(res, "live-prefill")
        self.assertEqual(len(bad), 1)
        self.assertIn("while nothing reads", bad[0])

    def test_unparsable_ui_line(self):
        res = run(clean_ui() + ["{t: 1,"])
        self.assertEqual(len(bad_in(res, "ui-data")), 1)


class MainExit(unittest.TestCase):
    def _main(self, ui_lines, log_text=None):
        with tempfile.TemporaryDirectory() as d:                       # (scratch-sized temp, nothing left behind)
            p1, p2 = Path(d) / "ui-data.jsonl", Path(d) / "engine.log"
            p1.write_text("\n".join(ui_lines) + "\n", encoding="utf-8")
            p2.write_text(LOG if log_text is None else log_text, encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                rc = cm.main([str(p1), str(p2)])
            return rc, out.getvalue()

    def test_clean_exits_0(self):
        rc, out = self._main(clean_ui())
        self.assertEqual(rc, 0)
        self.assertIn("read-rate", out)                               # (the summary table is printed)
        for cat in ("[read-rate]", "[peak]", "[live-prefill]", "[ui-data]", "[engine-log]"):
            self.assertNotIn(cat, out)                                # (but no mismatch lines)

    def test_mismatch_exits_1_and_prints(self):
        ui = clean_ui()
        ui[6] = line(T0 + 13.0, prefill=300.0)
        rc, out = self._main(ui)
        self.assertEqual(rc, 1)
        self.assertIn("live-prefill", out)
        self.assertIn("while nothing reads", out)

    def test_missing_file_exits_2(self):
        err = io.StringIO()
        with redirect_stdout(err):
            rc = cm.main(["scratch/does-not-exist.jsonl", "scratch/does-not-exist.txt"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
