"""tools/test_qfn_fit.py - Qwen3.8-Flash-Next on saguaro-qfn on a PC's cards: the files' sizes from the manifest,
the K/V the configs keep in VRAM, the RAM tier (RAM - 20 never above the model's expert bytes, and none at all
where the page cache holds the experts), one card (Tesla T4, a 24 GB card, an RX 9070 XT) and the four-card layer
split (the live 4x Tesla T4 configs' own values on their cards, an even split elsewhere), the live configs' tuning
env by class (GENERIC everywhere, T4-ONLY on all-sm_75 cards, MULTI-GPU on a split, TOPOLOGY on a multi-socket
CPU) and the CLI.  v1.2.1-B's fit policy among it all: refused only where the card or the RAM cannot hold the
model at all (the true minimum: a card's share of the dense weights and a 4K resident K/V window, the activations
and the reserve; RAM: dense + buffers + system); between the minimum and the comfortable layout (16 GB cards, 96
GB of RAM - the layout as measured) the plan is a SLOW one: it fits, scaled down (32K context, --kv-resident 4096,
drafting off, all listed in "turned_off") with its cause named ("most experts read from RAM / the SSD") - and the
matrix of the 1.2.1-B plan (T4s at 15.0 and 16.0, V100, 2080 Ti, 3060, 3090 / 4090, A4000, A5000, a P40, AMD, an
8 GB card, a mixed pair, 32 / 64 / 96 / 256 GB of RAM, three quants) says which row is which.  No GPU, no network.

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


def amd(i=0, gb=15.9, arch="gfx1201", name=None):
    return {"index": i, "name": name or "AMD Radeon RX 9070 / AI PRO R9700 (gfx1201)", "vram_gb": gb, "arch": arch,
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
        self.assertEqual(F.resident_budget_gib(220.0, "ud-q4_k_xl"), 103)                   # never over the experts
        self.assertEqual(F.resident_budget_gib(91.7, "q8_0"), 71)                            # (a cap that is no cap)
        self.assertTrue(F.page_cache_tier(252.0, "ud-q4_k_xl"))                              # the live 252 GB box:
        self.assertTrue(F.page_cache_tier(252.0, "q8_0"))                                    #  file cache, no tier
        self.assertFalse(F.page_cache_tier(91.7, "ud-q4_k_xl"))                              # the 96 GB box: tier
        self.assertFalse(F.page_cache_tier(120.0, "ud-q4_k_xl"))                             # 104 + 20 > 120: tier


class OneCard(unittest.TestCase):
    def test_a_t4_alone(self):
        p = F.plan([gpu(0, 15.0, "75", "Tesla T4")], 220.0)
        self.assertTrue(p["fits"], p["reason"])
        self.assertEqual((p["n_gpus"], p["layer_split"], p["context"]), (1, None, 65536))
        self.assertEqual(p["kv_resident"], 32768)
        self.assertIsNone(p["resident_budget_gib"])                    # (220 GB: the page cache holds the experts)
        self.assertEqual(p["args"], F.GENERIC_ARGS + ["--prompt-cache", "64"])   # (generic + the T4 box's quant)
        self.assertTrue(p["t4"])                                   # sm_75: the live configs' tuning env applies
        self.assertIn("SAGUARO_KQL", p["env"])
        self.assertIn("SAGUARO_ELASTIC_SLOTS", p["env"])                 # (GENERIC: what lets the engine lend room
        self.assertIn("SAGUARO_ARGMAX_ROWS_SM70", p["env"])              #  to the picture encoder; T4-ONLY here)
        self.assertNotIn("SAGUARO_SLOT_ADMIT", p["env"])                 # (MULTI-GPU: no split, no stage keys)
        self.assertNotIn("SAGUARO_NUMA_PLACE", p["env"])                 # (TOPOLOGY: no CPU facts handed in)
        self.assertFalse(p["measured"])                            # (a lone NVIDIA card: not measured here)

    def test_a_24_gb_card(self):
        p = F.plan([gpu(0, 24.0, "86", "RTX A5000")], 128.0)
        self.assertTrue(p["fits"], p["reason"])
        self.assertEqual((p["context"], p["layer_split"], p["t4"]), (65536, None, False))
        self.assertEqual({k for k in p["env"] if k.startswith(("SAGUARO_SM75", "SAGUARO_NUMA", "SAGUARO_SLOT_ADMIT"))},
                         set())                                      # (no T4 tuning, no NUMA keys, no split keys)
        self.assertEqual({k: v for k, v in p["env"].items() if k in F.GENERIC_ENV}, F.GENERIC_ENV)   # GENERIC
        self.assertIn("OS file cache", p["said"])                    # (128 GB: the experts all fit in RAM, no tier)

    def test_a_16_gb_amd_card(self):
        p = F.plan([amd()], 91.7)                                  # (the RX 9070 XT with 96 GB: measured)
        self.assertTrue(p["fits"], p["reason"])
        self.assertTrue(p["measured"])
        self.assertFalse(p["t4"])
        self.assertEqual(p["env"], F.GENERIC_ENV)                    # (AMD: the GENERIC keys, and only those)
        self.assertEqual(p["resident_budget_gib"], 71)               # (91.7 GB: the experts do NOT all fit: tier)


class TrueMinimum(unittest.TestCase):
    """v1.2.1-B: refused only under the TRUE minimum, with the numbers - a card: its share of the dense weights
    (3.9 GiB at UD-Q4, scaled by the quant's files) and a 4K-cell resident K/V window, 1.4 GB of activations and
    0.7 GB of reserve; RAM: dense + buffers + system.  Below the comfortable 16 GB / 96 GB a PC still RUNS."""

    def test_an_8_gb_card_runs_ud_q4_and_refuses_q8(self):
        p = F.plan([gpu(0, 8.0, "89", "RTX 4060")], 96.0)          # 6.3 GB needed: it runs, slowly
        self.assertTrue(p["fits"], p["reason"])
        self.assertTrue(p["slow"])
        q8 = F.plan([gpu(0, 8.0, "89", "RTX 4060")], 96.0, "q8_0")     # 9.0 GB needed: it cannot run
        self.assertFalse(q8["fits"])
        self.assertIn("cannot run on this PC's RTX 4060: its 8 GB cannot hold the 9.0 GB", q8["reason"])
        self.assertIsNone(q8["layer_split"])                       # (nothing is laid out over refused cards)

    def test_the_min_per_card_numbers(self):
        one = F.card_min_gib([gpu(0, 24.0, "86")], "ud-q4_k_xl")[0]
        self.assertAlmostEqual(one, 3.9 + 2.8 * 4096 / 32768 + 1.4 + 700 / 1024, delta=0.01)
        big = F.card_min_gib([gpu(0, 24.0, "86")], "q8_0")[0]      # (a Q8_0 card holds ~1.7x the dense weights)
        self.assertAlmostEqual(big, 3.9 * F.quant_gib("q8_0") / F.quant_gib("ud-q4_k_xl")
                               + 2.8 * 4096 / 32768 + 1.4 + 700 / 1024, delta=0.01)
        self.assertGreater(big, 8.0)                               # (what refuses the RTX 4060 above)
        pair = F.card_min_gib([gpu(0, 24.0, "86"), gpu(1, 24.0, "86")], "ud-q4_k_xl")
        self.assertAlmostEqual(pair[0], 0.5 * (3.9 + 2.8 * 4096 / 32768) + 1.4 + 700 / 1024, delta=0.01)
        self.assertEqual(len(pair), 2)
        self.assertAlmostEqual(F.ram_min_gib("ud-q4_k_xl"), 3.9 + 4.0 + 8.0, delta=0.01)

    def test_the_refusal_prices_the_failing_card(self):
        """Review finding 4: on an unequal layout the per-card minimums differ by the split's stage bounds, so
        the line carries THIS card's number and its own share of the layers, not one figure for every card:
        the 4090 beside the GT 710 serves 46 of the 48 layers (6.2 GB of its own), the GT 710 its 2 (2.3 GB)."""
        self.assertAlmostEqual(F.card_min_gib([gpu(0, 24.0, "89", "RTX 4090"), gpu(1, 1.0, "35")], "ud-q4_k_xl")
                               [1], (2 / 48) * (3.9 + 2.8 * 4096 / 32768) + 1.4 + 700 / 1024, delta=0.01)
        why = F.unsupported([gpu(0, 24.0, "89", "RTX 4090"), gpu(1, 1.0, "35", "GeForce GT 710")], 96.0)
        self.assertIn("cannot run on this PC's GeForce GT 710: its 1 GB cannot hold the 2.3 GB this card of the "
                      "layout must (4% of the layers)", why)

    def test_too_little_ram_names_the_minimum(self):
        p = F.plan([gpu(0, 24.0, "86")], 12.0)
        self.assertFalse(p["fits"])
        self.assertIn("cannot run with 12 GB of RAM", p["reason"])
        self.assertIn("at least about 16 GB", p["reason"])

    def test_unknown_vram_is_not_a_refusal(self):
        self.assertTrue(F.plan([gpu(0, 0)], 0)["fits"])            # (the engine measures at its start)

    def test_no_cards_named_is_not_a_fit(self):
        p = F.plan([], 220.0)                                      # (plan([]) must not claim it fits)
        self.assertFalse(p["fits"])
        self.assertIn("no cards were named", p["reason"])
        self.assertIsNone(p["layer_split"])
        self.assertEqual(p["env"], {})


class SlowZone(unittest.TestCase):
    """Between the minimum and the comfortable layout (16 GB cards, 96 GB RAM - the layout as measured, which
    the old rule refused) the plan fits, scaled down: a 32K context, --kv-resident 4096, drafting off - every
    one of them named in turned_off, the cause named in slow_why, and said / the CLI saying so."""

    def test_ram_below_96_is_slow_not_refused(self):
        for ram in (60.0, 32.0):
            with self.subTest(ram=ram):
                p = F.plan([gpu(0, 15.9, "gfx1201")], ram)
                self.assertTrue(p["fits"], p["reason"])
                self.assertTrue(p["slow"])
                self.assertIn(f"{ram:.0f} GB of RAM", p["slow_why"])
                self.assertIn("expect slow decode", p["said"])
                self.assertIn("drafting off (--mtp-max-t 1)", p["turned_off"])
                self.assertIn("--kv-resident 4096 instead of 32768", p["turned_off"])
                self.assertIn("a 32K context instead of 64K", p["turned_off"])
                self.assertEqual((p["context"], p["kv_resident"]), (32768, 4096))

    def test_cards_below_16_are_slow_not_refused(self):
        p = F.plan([gpu(0, 11.0, "75", "RTX 2080 Ti"), gpu(1, 11.0, "75", "RTX 2080 Ti")], 128.0)
        self.assertTrue(p["fits"], p["reason"])
        self.assertTrue(p["slow"])
        self.assertIn("11 GB cards hold", p["slow_why"])
        self.assertEqual((p["context"], p["kv_resident"]), (32768, 4096))
        self.assertIn("a 32K context instead of 256K", p["turned_off"])
        self.assertEqual(p["layer_split"], "24")                   # (the split itself: unchanged by the policy)

    def test_four_cards_still_want_the_ram(self):
        p = F.plan(t4s(4), 60.0)
        self.assertTrue(p["fits"])                                 # (60 GB: the old refusal line - now a warn)
        self.assertTrue(p["slow"])
        self.assertEqual(p["layer_split"], "11,24,36")             # (the live split stays the live split)

    def test_comfortable_is_not_slow(self):
        for cards, ram in ((t4s(4), 220.0), ([gpu(0, 24.0, "86", "RTX A5000")], 128.0), ([amd()], 91.7)):
            p = F.plan(cards, ram)
            self.assertTrue(p["fits"], p["reason"])
            self.assertFalse(p["slow"])
            self.assertEqual(p["turned_off"], [])


class FourT4(unittest.TestCase):
    def test_the_live_split_ud_q4_k_xl(self):
        p = F.plan(t4s(4), 220.0, "ud-q4_k_xl")
        self.assertTrue(p["fits"], p["reason"])
        self.assertEqual(p["layer_split"], "11,24,36")             # (./ref/Qwen3.8-Flash-Next-UD-Q4_K_XL.json)
        self.assertEqual(p["context"], 262144)
        self.assertEqual(p["args"], ["--trim-stage-weights", "--prefill-pipe", "1536"] + F.GENERIC_ARGS +
                         ["--prompt-cache", "64"])                 # (the conversation cache; live-ud-q4's prompt cache)
        self.assertTrue(p["t4"] and p["measured"])
        self.assertIn("SAGUARO_SM75_PERF", p["env"])
        self.assertIn("SAGUARO_SLOT_ADMIT_MULTI", p["env"])        # (MULTI-GPU: the split's pipelined reads)
        self.assertNotIn("SAGUARO_BATCH_READS_OVERTAKE", p["env"])   # (read nowhere: never written)
        self.assertNotIn("SAGUARO_PARK_STAGING_MIB", p["env"])
        self.assertEqual(p["resident_budget_gib"], None)           # (220 GB: all experts in the page cache, as live)

    def test_the_live_split_q8_0(self):
        p = F.plan(t4s(4), 220.0, "q8_0")
        self.assertEqual(p["layer_split"], "10,22,35")             # (./ref/Qwen3.8-Flash-Next-Q8_0.json)
        self.assertEqual(p["args"][:3], ["--trim-stage-weights", "--prefill-pipe", "6144"])
        self.assertNotIn("--prompt-cache", p["args"])              # (live-q8 runs the default: per quant)
        self.assertTrue(p["measured"])
        self.assertEqual(p["env"]["SAGUARO_BATCH_WRAP_SPILL"], "1")     # (the Q8_0 box's server spill, per quant)
        self.assertNotIn("SAGUARO_PARALLEL_LOAD_REGISTER", p["env"])

    def test_the_t4_env_follows_the_quant(self):
        ud = F.plan(t4s(4), 220.0, "ud-q4_k_xl")["env"]            # (./ref/Qwen3.8-Flash-Next-UD-Q4_K_XL.json)
        q8 = F.plan(t4s(4), 220.0, "q8_0")["env"]                  # (./ref/Qwen3.8-Flash-Next-Q8_0.json)
        self.assertEqual(ud["SAGUARO_BF16_TC_SKIP"], "none")
        self.assertEqual(q8["SAGUARO_BF16_TC_SKIP"], "indexer,router,ssm")
        self.assertEqual(ud["SAGUARO_SELECT_SM75"], "1")
        self.assertNotIn("SAGUARO_SELECT_SM75", q8)                # (the live Q8_0 box never ran it)
        self.assertEqual(q8["SAGUARO_KQL"], ud["SAGUARO_KQL"])     # (the shared part stays shared)
        self.assertEqual(ud["SAGUARO_PARALLEL_LOAD_REGISTER"], "overlap")   # (per quant, live-ud-q4's)
        self.assertEqual(ud["SAGUARO_GDN_KEYHEAD"], "1")           # (T4-ONLY: opt-in, same bits on sm_75)

    def test_equal_but_not_t4_cards_split_evenly(self):
        p = F.plan([gpu(i, 16.0, "86") for i in range(4)], 128.0)
        self.assertEqual(p["layer_split"], "12,24,36")
        self.assertEqual(p["args"], ["--trim-stage-weights", "--prefill-pipe", "1"] + F.GENERIC_ARGS)
        self.assertFalse(p["t4"] or p["measured"])
        self.assertEqual(p["env"], {**F.GENERIC_ENV, **F.SPLIT_ENV, **F.SPLIT_ENV_QUANT["ud-q4_k_xl"]})
        # (no T4 keys: the split's, its per-quant part, and the generic)

    def test_one_card_not_sm75_takes_no_t4_env(self):
        p = F.plan(t4s(3) + [gpu(3, 15.0, "86", "RTX A4000")], 220.0)
        self.assertTrue(p["fits"])
        self.assertFalse(p["t4"])
        self.assertNotIn("SAGUARO_SM75_PERF", p["env"])            # (mixed kinds: no hardware-specific tuning)
        self.assertNotIn("SAGUARO_ARGMAX_ROWS_SM70", p["env"])
        self.assertIn("SAGUARO_STAGE_SERVE", p["env"])             # (a split is a split: the MULTI-GPU class fits)


class Topology(unittest.TestCase):
    """The TOPOLOGY class: NUMA keys and --pool-workers only on a multi-socket CPU, the workers across sockets."""

    def test_pool_workers_across_the_sockets(self):
        two = {"sockets": 2, "cores": 20}                          # (mangonel: 2 x 10-core E5-2660 v2)
        self.assertEqual(F.pool_workers(two), 19)                  # (./ref/live-ud-q4.json: --pool-workers 19)
        self.assertIsNone(F.pool_workers({"sockets": 1, "cores": 16}))   # (one socket: the engine's own default)
        self.assertIsNone(F.pool_workers(None))
        self.assertTrue(F.multi_socket(two))
        self.assertFalse(F.multi_socket({"sockets": 1, "cores": 8}))

    def test_the_multi_socket_plan(self):
        p = F.plan(t4s(4), 220.0, "ud-q4_k_xl", {"sockets": 2, "cores": 20})
        self.assertEqual(p["args"][-2:], ["--pool-workers", "19"])
        self.assertEqual(p["env"]["SAGUARO_NUMA_PLACE"], "1")
        self.assertEqual(p["env"]["SAGUARO_NUMA_POOL"], "0")
        one = F.plan(t4s(4), 220.0, "ud-q4_k_xl", {"sockets": 1, "cores": 16})
        self.assertNotIn("--pool-workers", one["args"])
        self.assertNotIn("SAGUARO_NUMA_PLACE", one["env"])

    def test_split_format(self):
        for n in (2, 3, 4):
            p = F.plan([gpu(i, 20.0, "80") for i in range(n)], 128.0)
            bounds = [int(x) for x in p["layer_split"].split(",")]
            self.assertEqual(len(bounds), n - 1, n)
            self.assertTrue(all(b < c for b, c in zip(bounds, bounds[1:])), bounds)


class UnequalCards(unittest.TestCase):
    """U7: the computed split follows each card's VRAM (a smaller card caches less of its layers' experts, so
    it takes fewer layers), and a split that merely hits the live splits' bounds never borrows their pipe rate."""

    def test_the_split_follows_the_vram(self):
        self.assertEqual(F.layer_split([gpu(0, 16.0, "86"), gpu(1, 24.0, "86")]), "19")   # 48 x 16/40 rounded
        self.assertEqual(F.layer_split([gpu(0, 15.0, "86"), gpu(1, 48.0, "86"), gpu(2, 48.0, "86")]), "6,27")
        self.assertEqual(F.layer_split([gpu(i, 20.0, "86") for i in range(2)]), "24")     # equal: the even split
        self.assertEqual(F.layer_split([gpu(i, 0.0, "86") for i in range(3)]), "16,32")   # (VRAM unknown: even)

    def test_every_card_keeps_a_layer_and_the_bounds_rise(self):
        bounds = [int(x) for x in F.layer_split([gpu(0, 15.0, "86"), gpu(1, 96.0, "86")]).split(",")]
        self.assertEqual(bounds, [6])                                                    # 48 x 15/111 rounded
        for vs in ((15.0, 15.0, 96.0), (96.0, 15.0, 15.0), (16.0, 64.0, 64.0, 64.0)):
            s = [int(x) for x in F.layer_split([gpu(i, v, "86") for i, v in enumerate(vs)]).split(",")]
            self.assertEqual([0] + s + [48], sorted(set([0] + s + [48])))                # strictly rising
            self.assertTrue(all(a < b for a, b in zip(s, s[1:] + [48])))                 # none without layers

    def test_a_computed_split_never_borrows_the_live_pipe(self):
        four = [gpu(0, 33.0, "86"), gpu(1, 39.0, "86"), gpu(2, 36.0, "86"), gpu(3, 36.0, "86")]
        p = F.plan(four, 220.0)                                                          # -> bounds "11,24,36"
        self.assertEqual(p["layer_split"], "11,24,36")
        self.assertEqual(p["args"][p["args"].index("--prefill-pipe") + 1], "1")          # (1536 is the T4 box's)
        self.assertFalse(p["measured"])


class SlowReasonByName(unittest.TestCase):
    """The slow line says the quant's own numbers: Q8_0's 175 GiB of experts over 2x Tesla T4 + 64 GB is named
    outright (the plan's example) - as a warn now, not a refusal (v1.2.1-B); and the layout the Q8_0 box was
    measured on, SSD reads and all, is neither refused nor called slow."""

    def test_the_shortfall_names_the_quant(self):
        p = F.plan([gpu(i, 15.0, "75", "Tesla T4") for i in range(2)], 64.0, "q8_0")
        self.assertTrue(p["fits"], p["reason"])               # (the old refusal: now warn + confirm)
        self.assertTrue(p["slow"])
        self.assertIn("64 GB of RAM", p["slow_why"])          # (the thin tier ... )
        self.assertIn("175 GiB", p["slow_why"])               # ( ... and this quant's files ... )
        self.assertIn("~94 GiB", p["slow_why"])               # ( ... over what 2x15 + 64 can hold together)
        self.assertIn("expect slow decode", p["said"])
        measured = F.plan([gpu(0, 15.9, "gfx1201")], 91.7, "q8_0")   # (the measured Q8_0 AMD box: SSD reads are
        self.assertTrue(measured["fits"])                             #  no refusal and no warn - it is measured)
        self.assertFalse(measured["slow"])
        self.assertTrue(F.plan(t4s(4), 60.0, "ud-q4_k_xl")["slow"])  # (60 GB: the thin tier alone is a warn)

    def test_unsupported_names_only_what_cannot_run(self):
        self.assertIsNone(F.unsupported([gpu(0, 15.9, "gfx1201")], 60.0))     # (the old "96 GB of RAM" refusal
        self.assertIsNone(F.unsupported([gpu(i, 15.0, "75", "Tesla T4") for i in range(2)], 64.0, "q8_0"))
        self.assertIn("cannot run", F.unsupported([gpu(0, 4.0, "89")], 96.0, "ud-q4_k_xl"))     # 4 GB card
        self.assertIn("cannot run", F.unsupported([gpu(0, 24.0, "86")], 12.0))                  # 12 GB of RAM


class Choice(unittest.TestCase):
    def test_big_card_alone_equal_cards_together(self):
        self.assertEqual(F.best_gpus([gpu(0, 24.0, "86")]), [gpu(0, 24.0, "86")])
        self.assertEqual([g["index"] for g in F.best_gpus(t4s(4))], [0, 1, 2, 3])
        self.assertEqual([g["index"] for g in F.best_gpus(t4s(6))], [0, 1, 2, 3])          # (up to four)
        mixed = t4s(2) + [gpu(2, 24.0, "86")]
        self.assertEqual([g["index"] for g in F.best_gpus(mixed)], [2])                     # (the 24 GB one alone)
        self.assertEqual([g["index"] for g in F.best_gpus([gpu(0, 15.0), gpu(1, 11.0)])], [0])
        self.assertEqual([g["index"] for g in F.best_gpus(t4s(2, 8.0))], [0, 1])   # (v1.2.1-B: two 8 GB cards
                                                                                    #  are a slow pair, not a
                                                                                    #  refusal - no size filter)
        self.assertEqual(F.best_gpus([]), [])


class Facts(unittest.TestCase):
    def test_sm75_gate(self):
        self.assertTrue(F.sm75_only(t4s(4)))
        self.assertFalse(F.sm75_only([gpu(0, 15.0, "75"), gpu(1, 15.0, "86")]))
        self.assertFalse(F.sm75_only([amd()]))
        self.assertFalse(F.sm75_only([]))

    def test_pascal_is_for_qfn_now(self):
        self.assertEqual(F.MIN_CC, 61)                             # (v1.2.1-B: a Tesla P40 ran QFN by hand)
        p = F.plan([gpu(0, 24.0, "61", "Tesla P40")], 220.0)       # (no T4 tuning on it: the U6 class rule)
        self.assertTrue(p["fits"] and not p["slow"])
        self.assertFalse(p["t4"])
        self.assertEqual({k for k in p["env"] if k.startswith(("SAGUARO_SM75", "SAGUARO_ARGMAX"))
                          or k in ("SAGUARO_KQL", "SAGUARO_GDN_KEYHEAD")}, set())


class PolicyMatrix(unittest.TestCase):
    """The 1.2.1-B plan's matrix, every row judged one of three ways: refused (R: the card or the RAM cannot
    hold the model), slow (S: warned + confirmed, the layout scaled down), or fits as laid out today (F).  The
    one row no quant can run on besides the 4 GB cards of TrueMinimum: an 8 GB card under Q8_0 (its ~6.6 GiB of
    dense weights need 9.0 GB on the card) - its PAIR runs it slow, 5.6 GB a card by the split (review
    finding 2: the minimum is per card of the layout, not of a card alone)."""

    QUANTS = ["ud-q4_k_xl", "q8_0", "cyber-frost-q5_k_m"]
    RAMS = [32.0, 64.0, 96.0, 256.0]                               # (each row's verdict at these RAMs, "S" below
    #  the comfortable 96 GB where the card itself holds up)
    CARDS = {
        "1x T4 (ECC on)": t4s(1), "1x T4": [gpu(0, 16.0, "75", "Tesla T4")],
        "2x T4 (ECC on)": t4s(2), "4x T4 (ECC on)": t4s(4), "4x T4": [gpu(i, 16.0, "75", "Tesla T4")
                                                                      for i in range(4)],
        "1x V100 16": [gpu(0, 16.0, "70", "Tesla V100")], "1x V100 32": [gpu(0, 32.0, "70", "Tesla V100")],
        "2x V100 16": [gpu(i, 16.0, "70", "Tesla V100") for i in range(2)],
        "2080 Ti 11": [gpu(0, 11.0, "75", "RTX 2080 Ti")], "3060 12": [gpu(0, 12.0, "86", "RTX 3060")],
        "3090 24": [gpu(0, 24.0, "86", "RTX 3090")], "4090 24": [gpu(0, 24.0, "89", "RTX 4090")],
        "A4000 16": [gpu(0, 16.0, "86", "RTX A4000")], "A5000 24": [gpu(0, 24.0, "86", "RTX A5000")],
        "2x A4000 16": [gpu(i, 16.0, "86", "RTX A4000") for i in range(2)],
        "2x A5000 24": [gpu(i, 24.0, "86", "RTX A5000") for i in range(2)],
        "P40 24": [gpu(0, 24.0, "61", "Tesla P40")],               # (the hand-run box: 24 GB + 32 GB of RAM)
        "RX 7900 XTX": [amd(0, 24.0, "gfx1100", "AMD Radeon RX 7900 XTX (gfx1100)")],
        "RX 9070 XT": [amd(0, 16.0)],
        "8 GB card": [gpu(0, 8.0, "89", "GeForce RTX 4060")],
        "2x 8 GB": [gpu(0, 8.0, "89", "GeForce RTX 4060"), gpu(1, 8.0, "89", "GeForce RTX 4060")],
        "24 + 16 pair": [gpu(0, 24.0, "86", "RTX A5000"), gpu(1, 16.0, "86", "RTX A4000")],
    }
    EXPECT = dict.fromkeys(CARDS, "SSFF")                          # (the card holds up everywhere: the RAM row)
    EXPECT["2080 Ti 11"] = EXPECT["3060 12"] = EXPECT["8 GB card"] = "SSSS"   # (small cards: slow everywhere)
    EXPECT["2x 8 GB"] = "SSSS"                                     # (Q8_0 included: 5.6 GB a card by the split)

    def test_the_matrix_verdicts(self):
        for label, cards in self.CARDS.items():
            for i, ram in enumerate(self.RAMS):
                for quant in self.QUANTS:
                    p = F.plan(cards, ram, quant)
                    with self.subTest(cards=label, ram=ram, quant=quant):
                        want = self.EXPECT[label][i]
                        if quant == "q8_0" and label == "8 GB card":
                            want = "R"                             # (9.0 GB needed: cannot run)
                        got = "R" if not p["fits"] else "S" if p["slow"] else "F"
                        self.assertEqual(got, want, p["reason"] or p["slow_why"] or p["said"])
                        if want == "S":                            # (the scale-down written, its cause named)
                            self.assertEqual((p["context"], p["kv_resident"]), (32768, 4096))
                            self.assertTrue(p["turned_off"])
                            self.assertTrue(p["slow_why"])
                        if want == "R":
                            self.assertIn("cannot run", p["reason"])

    def test_4x_t4_256_gb_ud_q4_stays_the_live_layout(self):
        p = F.plan(t4s(4), 256.0, "ud-q4_k_xl")                    # (LiveT4Parity writes exactly this config)
        self.assertTrue(p["fits"] and not p["slow"])
        self.assertEqual((p["layer_split"], p["context"], p["kv_resident"]), ("11,24,36", 262144, 32768))
        self.assertTrue(p["measured"])

    def test_cc_reads_an_amd_arch_as_zero(self):
        self.assertEqual(F._cc(amd()), 0)
        self.assertEqual(F._cc({"arch": 75}), 75)
        self.assertEqual(F._cc({}), 0)


class Cli(unittest.TestCase):
    def test_a_fit_a_slow_fit_and_a_refusal(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(F.main(["--ram", "220", "16:75", "16:75", "16:75", "16:75"]), 0)
            self.assertEqual(F.main(["--ram", "220", "8"]), 0)     # (runs, slowly: v1.2.1-B - exit 0 either way)
            self.assertEqual(F.main(["--ram", "220", "4"]), 1)     # (below the true minimum: cannot run)
            self.assertEqual(F.main([]), 1)                        # (no cards named: not a fit either)
        text = out.getvalue()
        self.assertIn("layer_split: 11,24,36", text)
        self.assertIn("slow: its 8 GB card holds few", text)       # (the warn the setup shows too)
        self.assertIn("the setup scales down: drafting off (--mtp-max-t 1)", text)
        self.assertIn("does not fit: Qwen3.8-Flash-Next on saguaro-qfn cannot run on this PC's GPU 0: its 4 GB "
                      "cannot hold the 6.3 GB", text)
        self.assertIn("does not fit: Qwen3.8-Flash-Next on saguaro-qfn runs on one GPU, or up to 4 equal ones", text)


if __name__ == "__main__":
    unittest.main()
