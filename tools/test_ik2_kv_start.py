"""tools/test_ik2_kv_start.py - the elastic start step's sizing on the PC side (tools/ik2_fit.py, P125e a1): what a
conversation USES at its start (kv_committed, SAGUARO_IK2_KV_START) sizes the resident/offload decision instead of
the --max-context ceiling, so the ceiling-only long context no longer leaves the experts with no tier
(ref/REPORT-p125b-ik2-vram.md's 262144 row) - while streaming, tight cards and the two-card split keep their
answers.  These mirror the engine's helpers (src/ik2/ik2_plan.hpp, tests/ik2_plan_test.cpp).  No GPU, no network.

    python -m unittest tools.test_ik2_kv_start -v
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import ik2_fit as F  # noqa: E402

ONE_T4 = 11.45                                      # Inkling-2-Mini-T4-NVFP4-IQ3_XXS's experts


class TestCommitted(unittest.TestCase):
    def test_the_commit_is_the_start_step_never_past_the_ceiling(self):
        self.assertEqual(F.KV_START, 32768)                          # SAGUARO_IK2_KV_START's default
        self.assertEqual(F.kv_committed(262144), 32768)              # the ceiling stays a ceiling
        self.assertEqual(F.kv_committed(131072), 32768)
        self.assertEqual(F.kv_committed(32768), 32768)               # a ceiling at the start step is itself
        self.assertEqual(F.kv_committed(8192), 8192)                 # a smaller ceiling stays smaller
        self.assertEqual(F.kv_committed(262144, 0), 262144)          # SAGUARO_IK2_KV_START=0: today's layout
        self.assertEqual(F.kv_committed(262144, -1), 262144)         # negative: off, not negative
        self.assertEqual(F.kv_committed(65536, 131072), 65536)       # a start past the ceiling is the ceiling


class TestPlan(unittest.TestCase):
    def test_the_262144_row_that_loses_every_expert_now_keeps_them(self):
        """ref/REPORT-p125b-ik2-vram.md's second mortar row: UD-Q4_K_XL at 262144 on a 24 GB card - the whole
        cache leaves 10.7 of a tier, the committed start step leaves every expert in VRAM and a 32768 commit."""
        p = F.plan(24.0, 262144)
        self.assertTrue(p["resident"])
        self.assertEqual(p["layout"], "resident")
        self.assertEqual(p["kv_committed"], 32768)
        self.assertEqual(p["tier_gib"], F.EXPERTS_GIB)               # every expert stays
        self.assertEqual(p["ram_gib"], F.HOST_GIB)                   # nothing pinned in RAM for the experts
        self.assertFalse(p["kv_streamed"])
        self.assertGreater(p["need_gib"], 24.0 - F.MARGIN_GIB)       # (the ceiling's cache still does not fit)

    def test_a_context_at_the_commit_never_gets_the_key(self):
        for ctx in (8192, 32768):
            self.assertNotIn("kv_committed", F.plan(24.0, ctx))
            self.assertNotIn("kv_committed", F.plan(24.0, ctx, "fp8"))
        self.assertNotIn("kv_committed", F.plan(16.0, 32768, experts_gib=ONE_T4))   # (offload too: the commit is the ceiling)

    def test_a_card_that_cannot_hold_the_experts_even_at_the_commit_is_unchanged(self):
        """The one-T4 quant at 262144 on a 16 GB NVIDIA card: even the committed cache does not leave the experts
        room beside them - the offload path as before (b1 fills its tier at the load, the engine's side)."""
        p = F.plan(16.0, 262144, experts_gib=ONE_T4)
        self.assertFalse(p["resident"])
        self.assertEqual(p["layout"], "offload")
        self.assertNotIn("kv_committed", p)
        self.assertAlmostEqual(p["tier_gib"], 2.7, places=1)

    def test_a_streaming_cache_is_not_replaced_by_a_commit(self):
        """16 GB at the trained 1M: the cache alone does not fit beside a tier - the rows go to RAM and the
        experts stay, which is today's answer and needs no commit (long chats keep working there)."""
        p = F.plan(16.0, 1048576)
        self.assertTrue(p["kv_streamed"])
        self.assertFalse(p["resident"])
        self.assertNotIn("kv_committed", p)

    def test_the_trained_million_fp8_on_24_gb_plans_resident_at_the_commit(self):
        for hip in (False, True):
            p = F.plan(24.0, 1048576, "fp8", hip=hip)
            self.assertEqual((p["layout"], p.get("kv_committed")), ("resident", 32768), f"hip={hip}")

    def test_two_cards_keep_splitting(self):
        """The split plan is untouched: two 15 GB cards still hold every expert across them at 131072."""
        self.assertFalse(F.plan(15.0, 131072)["resident"])           # (no elastic room on one of them)
        p = F.plan_two(15.0, 15.0, 131072)
        self.assertEqual((p["gpus"], p["layout"]), (2, "split"))

    def test_recommended_context_sees_the_whole_slider_on_an_elastic_card(self):
        self.assertEqual(F.recommended_context(24.0), 262144)        # (f16: elastic at 262144)
        self.assertEqual(F.recommended_context(24.0, "fp8"), 262144)  # (and up to RECOMMEND_MAX as before)
        self.assertEqual(F.recommended_context(8.0), 32768)          # (a tight card keeps the smallest)


class TestVerdict(unittest.TestCase):
    def test_the_fast_line_says_the_commit_when_the_ceiling_is_one(self):
        word, line = F.verdict(24.0, 92.0, 262144)
        self.assertEqual(word, "fast")
        self.assertIn("every expert in VRAM", line)
        self.assertIn("32K of the 256K ceiling", line)
        self.assertIn("until it can grow", line)

    def test_the_plain_fast_line_is_word_for_word_as_before(self):
        self.assertEqual(F.verdict(24.0, 92.0, 32768), ("fast", "every expert in VRAM"))
        self.assertEqual(F.verdict(32.0, 92.0, 262144), ("fast", "every expert in VRAM"))   # (fits outright)

    def test_a_card_on_the_offload_path_keeps_its_line(self):
        word, line = F.verdict(16.0, 92.0, 262144, experts_gib=ONE_T4)
        self.assertEqual(word, "runs")
        self.assertIn("pinned in RAM", line)
        self.assertNotIn("until it can grow", line)


if __name__ == "__main__":
    unittest.main()
