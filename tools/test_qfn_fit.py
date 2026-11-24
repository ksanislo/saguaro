"""tools/test_qfn_fit.py - Qwen3.8-Flash-Next on saguaro-qfn on a PC's cards: the files' sizes from the manifest,
the K/V the configs keep in VRAM, the RAM tier, one card (Tesla T4, a 24 GB card, an RX 9070 XT) and the four-card
layer split (the live 4x Tesla T4 configs' own values on their cards, an even split elsewhere), what does not fit,
the T4 env gate and the CLI.  No GPU, no network.

    python -m unittest tools.test_qfn_fit -v
"""
from contextlib import redirect_stdout
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import qfn_fit as F  # noqa: E402


def gpu(i, gb, arch="75", name=None):
    return {"index": i, "name": name or f"GPU {i}", "vram_gb": gb, "arch": arch}


def t4s(n, gb=15.0):
    return [gpu(i, gb, "75", "Tesla T4") for i in range(n)]


def amd(i=0, gb=15.9):
    return {"index": i, "name": "AMD Radeon RX 9070 / AI PRO R9700 (gfx1201)", "vram_gb": gb, "arch": "gfx1201",
            "vendor": "amd"}


class Sizes(unittest.TestCase):
    def test_files_from_the_manifest(self):
        self.assertAlmostEqual(F.quant_gib("ud-q4_k_xl"), 111.3 * 1e9 / F.GIB, delta=0.2)   # ~111 GB
        self.assertAlmostEqual(F.quant_gib("q8_0"), 175.3, delta=0.5)                       # ~177 GB
        self.assertEqual(F.quant_gib("nope"), 0.0)

    def test_kv_and_budget(self):
        self.assertAlmostEqual(F.kv_vram_gib(32768), 32768 * 12672 / F.GIB, delta=1e-9)     # ~0.4 GiB
        self.assertEqual(F.resident_budget_gib(91.7), 71)                                   # the 96 GB PC's tier
        self.assertEqual(F.resident_budget_gib(20.0), 8)


class OneCard(unittest.TestCase):
    def test_a_t4_alone(self):
        p = F.plan([gpu(0, 15.0, "75", "Tesla T4")], 220.0)
        self.assertTrue(p["fits"], p["reason"])
        self.assertEqual((p["n_gpus"], p["layer_split"], p["context"]), (1, None, 65536))
        self.assertEqual(p["kv_resident"], 32768)
        self.assertEqual(p["resident_budget_gib"], 200)
        self.assertEqual(p["args"], [])
        self.assertTrue(p["t4"])                                   # sm_75: the live configs' tuning env applies
        self.assertIn("SAGUARO_KQL", p["env"])
        self.assertFalse(p["measured"])                            # (a lone NVIDIA card: not measured here)

    def test_a_24_gb_card(self):
        p = F.plan([gpu(0, 24.0, "86", "RTX A5000")], 128.0)
        self.assertTrue(p["fits"], p["reason"])
        self.assertEqual((p["context"], p["layer_split"], p["t4"]), (65536, None, False))
        self.assertEqual(p["env"], {})
        self.assertIn("108 GiB RAM tier", p["said"])               # (128 GB of RAM less the 20 GB spare)

    def test_a_16_gb_amd_card(self):
        p = F.plan([amd()], 91.7)                                  # (the RX 9070 XT with 96 GB: measured)
        self.assertTrue(p["fits"], p["reason"])
        self.assertTrue(p["measured"])
        self.assertFalse(p["t4"])
        self.assertEqual(p["env"], {})
        self.assertEqual(p["resident_budget_gib"], 71)


class TooSmall(unittest.TestCase):
    def test_an_8_gb_card(self):
        p = F.plan([gpu(0, 8.0, "61", "Tesla P40")], 96.0)
        self.assertFalse(p["fits"])
        self.assertIn("16 GB or more", p["reason"])
        self.assertIsNone(p["layer_split"])                        # (nothing is laid out over refused cards)

    def test_too_little_ram(self):
        p = F.plan([gpu(0, 15.9, "gfx1201")], 60.0)
        self.assertFalse(p["fits"])
        self.assertIn("96 GB of RAM", p["reason"])

    def test_four_cards_still_need_the_ram(self):
        self.assertFalse(F.plan(t4s(4), 60.0)["fits"])

    def test_unknown_vram_is_not_a_refusal(self):
        self.assertTrue(F.plan([gpu(0, 0)], 0)["fits"])            # (the engine measures at its start)


class FourT4(unittest.TestCase):
    def test_the_live_split_ud_q4_k_xl(self):
        p = F.plan(t4s(4), 220.0, "ud-q4_k_xl")
        self.assertTrue(p["fits"], p["reason"])
        self.assertEqual(p["layer_split"], "11,24,36")             # (./ref/Qwen3.8-Flash-Next-UD-Q4_K_XL.json)
        self.assertEqual(p["context"], 262144)
        self.assertEqual(p["args"], ["--trim-stage-weights", "--prefill-pipe", "1536"])
        self.assertTrue(p["t4"] and p["measured"])
        self.assertIn("SAGUARO_SM75_PERF", p["env"])

    def test_the_live_split_q8_0(self):
        p = F.plan(t4s(4), 220.0, "q8_0")
        self.assertEqual(p["layer_split"], "10,22,35")             # (./ref/Qwen3.8-Flash-Next-Q8_0.json)
        self.assertEqual(p["args"][-1], "6144")
        self.assertTrue(p["measured"])

    def test_equal_but_not_t4_cards_split_evenly(self):
        p = F.plan([gpu(i, 16.0, "86") for i in range(4)], 128.0)
        self.assertEqual(p["layer_split"], "12,24,36")
        self.assertEqual(p["args"], ["--trim-stage-weights", "--prefill-pipe", "1"])
        self.assertFalse(p["t4"] or p["measured"])
        self.assertEqual(p["env"], {})

    def test_one_card_not_sm75_takes_no_t4_env(self):
        p = F.plan(t4s(3) + [gpu(3, 15.0, "86", "RTX A4000")], 220.0)
        self.assertTrue(p["fits"])
        self.assertFalse(p["t4"])
        self.assertEqual(p["env"], {})                             # (mixed kinds: no hardware-specific tuning)

    def test_split_format(self):
        for n in (2, 3, 4):
            p = F.plan([gpu(i, 20.0, "80") for i in range(n)], 128.0)
            bounds = [int(x) for x in p["layer_split"].split(",")]
            self.assertEqual(len(bounds), n - 1, n)
            self.assertTrue(all(b < c for b, c in zip(bounds, bounds[1:])), bounds)


class Choice(unittest.TestCase):
    def test_big_card_alone_equal_cards_together(self):
        self.assertEqual(F.best_gpus([gpu(0, 24.0, "86")]), [gpu(0, 24.0, "86")])
        self.assertEqual([g["index"] for g in F.best_gpus(t4s(4))], [0, 1, 2, 3])
        self.assertEqual([g["index"] for g in F.best_gpus(t4s(6))], [0, 1, 2, 3])          # (up to four)
        mixed = t4s(2) + [gpu(2, 24.0, "86")]
        self.assertEqual([g["index"] for g in F.best_gpus(mixed)], [2])                     # (the 24 GB one alone)
        self.assertEqual([g["index"] for g in F.best_gpus([gpu(0, 15.0), gpu(1, 11.0)])], [0])
        self.assertEqual(F.best_gpus([]), [])


class Facts(unittest.TestCase):
    def test_sm75_gate(self):
        self.assertTrue(F.sm75_only(t4s(4)))
        self.assertFalse(F.sm75_only([gpu(0, 15.0, "75"), gpu(1, 15.0, "86")]))
        self.assertFalse(F.sm75_only([amd()]))
        self.assertFalse(F.sm75_only([]))

    def test_cc_reads_an_amd_arch_as_zero(self):
        self.assertEqual(F._cc(amd()), 0)
        self.assertEqual(F._cc({"arch": 75}), 75)
        self.assertEqual(F._cc({}), 0)


class Cli(unittest.TestCase):
    def test_a_fit_and_a_refusal(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(F.main(["--ram", "220", "16:75", "16:75", "16:75", "16:75"]), 0)
            self.assertEqual(F.main(["--ram", "220", "8"]), 1)
        text = out.getvalue()
        self.assertIn("layer_split: 11,24,36", text)
        self.assertIn("does not fit: Qwen3.8-Flash-Next on saguaro-qfn needs GPUs of 16 GB or more", text)


if __name__ == "__main__":
    unittest.main()
