"""tools/test_setup_ik2.py - Inkling 2 Mini (saguaro-ik2) in the setup: its names and setup ids, the pinned manifest,
the layout for one card or two (ik2_fit.py: the two measured on Tesla T4s, a 24 GB card), the one-line refusals, the
model the setup recommends for a small PC, the tokenizer step, the config it writes (its pictures too) and a start of
it.  No GPU, no network.

    python -m unittest tools.test_setup_ik2 -v
"""
from pathlib import Path
import io
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import saguaro as maya  # noqa: E402
import maya_quants as Q  # noqa: E402
import ik2_fit as F  # noqa: E402


def args(**kw):
    base = dict(model=maya.IK2, hy4_quant=None, mm3_quant=None, ds4_quant=None, model_dir=None, models_dir=None,
                gguf_dir=None, draft=None, backend="cuda", yes=True, check=False, port=None, host=None, api_key=None,
                env=[], no_vision=False, context=None, gpus=None, gpu=None, nvcc=None, repack=False, parallel=None,
                drafting=None)
    base.update(kw)
    return SimpleNamespace(**base)


def t4(index=0, gib=15.0, arch="75", name="Tesla T4"):
    return {"index": index, "name": name, "arch": arch, "vram_gb": float(gib), "driver": "580.0"}


def card(gib, index=0, arch="86", name="NVIDIA GeForce RTX 3090"):
    return {"index": index, "name": name, "arch": arch, "vram_gb": float(gib), "driver": "580.0"}


def amd(arch, gib, index=0, **kw):
    return {"index": index, "name": "AMD Radeon", "arch": arch, "vram_gb": float(gib), **kw}


def refusal(fn):
    """The text a fail() printed, and that it stopped (SystemExit)."""
    out = io.StringIO()
    with patch("sys.stdout", new=out), patch.object(maya.S, "UI", None):
        try:
            fn()
        except SystemExit:
            return out.getvalue()
    raise AssertionError("no refusal: " + out.getvalue()[-400:])


class Names(unittest.TestCase):
    def test_shown_and_ids(self):
        self.assertEqual(Q.shown("ik2", Q.IK2_QUANT, 18.27), "Inkling 2 Mini · unsloth UD-Q4_K_XL (18 GB)")
        self.assertEqual(Q.config_stem("ik2", Q.IK2_QUANT), "maya-inkling-2-mini")
        for name in ("inkling-2-mini", "Inkling-2-Mini", "inkling-2-mini-ud-q4_k_xl", "ik2-mini", "INKLING2-MINI"):
            self.assertEqual(Q.resolve(name), ("ik2", Q.IK2_QUANT), name)
        self.assertIsNone(Q.resolve("inkling-2-small"))
        self.assertEqual((maya.IK2, maya.IK2_SHOWN, maya.IK2_MODEL_NAME),
                         ("Inkling-2-Mini", "Inkling 2 Mini", "inkling-2-mini"))
        self.assertIn("ik2", maya.FAMILIES)
        self.assertEqual(maya.IK2_EXE.name, "saguaro-ik2")

    def test_model_option(self):
        a = args(model="inkling-2-mini")
        self.assertIsNone(maya.model_option(a))
        self.assertEqual(a.model, maya.IK2)
        self.assertEqual(maya.choose_family(a, {}), "ik2")

    def test_defaults(self):
        """The setup's config: the model's own sampling, its template's default level by name; a config written by
        hand keeps the engine family's."""
        mini = {"exe": "/b/saguaro-ik2", "installer": {"family": "ik2", "ik2_quant": Q.IK2_QUANT}}
        self.assertEqual(Q.model_defaults(mini), ({"temperature": 1.0, "top_p": 1.0}, "high"))
        by_hand = {"exe": "build/saguaro-ik2", "args": ["--model", "Inkling-2-Mini-UD-Q4_K_XL.gguf"]}
        self.assertEqual(Q.model_defaults(by_hand)[1], "auto")
        self.assertEqual(Q.config_label(mini), "Inkling 2 Mini · unsloth UD-Q4_K_XL")
        self.assertEqual(Q.config_key(mini), Q.IK2_QUANT)


class Manifest(unittest.TestCase):
    def test_pinned(self):
        man = json.loads(maya.IK2_MANIFEST.read_text(encoding="utf-8"))
        self.assertRegex(man["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(man["repo"], "unsloth/Inkling-2-Mini-GGUF")
        self.assertEqual(list(man["files"]), [man["gguf"]])
        for name, (size, sha) in man["files"].items():
            self.assertGreater(size, 100 << 20, name)
            self.assertRegex(sha, r"^[0-9a-f]{64}$", name)

    def test_entry_and_urls(self):
        m = maya.ik2_entry()
        self.assertAlmostEqual(m["download_gb"], 17.63, places=1)
        url = maya.file_url(m, m["gguf"])
        self.assertEqual(url, f"https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/{m['revision']}/"
                              "Inkling-2-Mini-UD-Q4_K_XL.gguf")
        v = maya.ik2_vision_model()
        self.assertEqual(v["files"], {"mmproj-BF16.gguf": 641805760})
        self.assertTrue(maya.file_url(v, "mmproj-BF16.gguf").endswith(f"/{v['revision']}/mmproj-BF16.gguf"))

    def test_drafter_pinned(self):
        m = maya.ik2_drafter_model()
        self.assertEqual((m["repo"], m["revision"]), ("thinkingmachines/Inkling-2-Mini",
                                                      "ed40e7da625f594fc9c4990f5f66be24873c22d7"))
        self.assertEqual(m["files"], {"mtp.safetensors": 635801416})
        self.assertRegex(m["sha256"]["mtp.safetensors"], r"^[0-9a-f]{64}$")

    def test_import_catalog_lists_it(self):
        cat = maya.import_catalog(Path("/models"))
        rels = {c["rel"]: c for c in cat}
        self.assertEqual(rels["Inkling-2-Mini/Inkling-2-Mini-UD-Q4_K_XL.gguf"]["size"], 17629759456)
        self.assertIn("Inkling-2-Mini/mmproj-BF16.gguf", rels)


class Fit(unittest.TestCase):
    def test_one_t4(self):
        """One T4: the offload path (UD-Q4_K_XL's 14.7 GiB of experts do not fit beside the rest), a 32K context."""
        lay = F.layout([t4()])
        self.assertEqual([g["index"] for g in lay["gpus"]], [0])
        self.assertEqual(lay["args"], ["--gpus", "1", "--chunk", "2048", "--threads", "8"])
        self.assertEqual((lay["context"], lay["resident"], lay["path"], lay["measured"]), (32768, False, "offload", True))
        self.assertEqual(lay["env"], {"SAGUARO_IK2_ROOM_MB": "1024"})
        self.assertIn("24 tok/s", F.speeds(lay))
        self.assertIn("1x Tesla T4", F.speeds(lay))

    def test_two_t4s(self):
        """Two T4s: the split, every expert in VRAM across them at 131072 (as measured)."""
        lay = F.layout([t4(2), t4(3)])
        self.assertEqual([g["index"] for g in lay["gpus"]], [2, 3])
        self.assertEqual(lay["args"], ["--gpus", "2", "--chunk", "8192", "--threads", "8"])
        self.assertEqual((lay["context"], lay["resident"], lay["path"], lay["measured"]), (131072, True, "split", True))
        self.assertIn("52 tok/s", F.speeds(lay))
        p = F.plan_two(15.0, 15.0, 131072)
        self.assertEqual((p["gpus"], p["layout"]), (2, "split"))

    def test_four_t4s_take_two_and_one_when_asked(self):
        cards = [t4(i) for i in range(4)]
        self.assertEqual(len(F.layout(cards)["gpus"]), 2)
        self.assertEqual(len(F.layout(cards, 1)["gpus"]), 1)

    def test_a_24_gb_card_holds_it_alone(self):
        lay = F.layout([t4(0), t4(1, 24.0, "86", "NVIDIA GeForce RTX 3090")])
        self.assertEqual(([g["index"] for g in lay["gpus"]], lay["resident"], lay["measured"]), ([1], True, False))
        self.assertEqual(lay["context"], 131072)                # (18.3 GiB + 4 GiB of F16 cache in 23.5)
        self.assertEqual(F.big_context(24.0, "fp8"), 262144)
        self.assertEqual(F.speeds(lay), "")                     # (nothing claimed for a card not measured)
        self.assertNotIn("SAGUARO_IK2_ROOM_MB", lay["env"])

    def test_small_cards_take_the_offload_path(self):
        """8 / 12 GB cards run too: the offload path, a share of the experts on the card growing with it."""
        lay = F.layout([card(8)])
        self.assertEqual((lay["path"], lay["measured"], lay["env"]), ("offload", False, {}))
        tiers = [F.plan(v, 32768) for v in (8.0, 12.0, 16.0)]
        self.assertTrue(all(p["layout"] == "offload" and not p["resident"] for p in tiers))
        self.assertTrue(0 < tiers[0]["tier_gib"] < tiers[1]["tier_gib"] < tiers[2]["tier_gib"] < F.EXPERTS_GIB)
        for p in tiers:
            self.assertAlmostEqual(p["ram_gib"], F.EXPERTS_GIB - p["tier_gib"] + F.OFF_HOST_GIB, places=1)
        self.assertEqual(F.verdict(8.0, 64.0, 32768)[0], "runs")
        self.assertEqual(F.verdict(8.0, 12.0, 32768)[0], "does not fit")

    def test_upstream_figures(self):
        """Upstream's figures (an RTX 3090, the FP8 cache): 262,144 with every expert in VRAM; the trained million
        keeps the cache on the card (the experts leave first) - on AMD too."""
        self.assertEqual(F.recommended_context(24.0, "fp8"), 262144)
        self.assertTrue(F.plan(24.0, 262144, "fp8")["resident"])
        p = F.plan(24.0, 1048576, "fp8")
        self.assertEqual((p["layout"], p["kv_streamed"]), ("offload", False))
        self.assertEqual(F.plan(24.0, 1048576, "fp8", hip=True)["layout"], "offload")
        self.assertEqual(F.plan(0, 262144)["resident"], True)          # (VRAM not read: the engine measures)

    def test_plan_two(self):
        self.assertEqual(F.plan_two(24.0, 24.0, 32768, "fp8")["gpus"], 1)
        p = F.plan_two(12.0, 12.0, 32768, "fp8")
        self.assertEqual((p["gpus"], p["layout"]), (2, "split"))
        self.assertEqual(F.plan_two(8.0, 8.0, 32768, "fp8")["layout"], "split+ram")
        self.assertEqual(F.verdict_two(12.0, 12.0, 64.0, 32768, "fp8")[0], "fast")
        self.assertEqual(F.verdict_two(8.0, 8.0, 8.0, 32768, "fp8")[0], "does not fit")

    def test_experts_of_a_file(self):
        """Another file's experts take UD-Q4_K_XL's place: our one-T4 quant (11.45 GiB) is resident on one T4 at 16K
        with the F16 cache by these sums."""
        self.assertIsNone(F.experts_gib_of(ROOT / "no-such.gguf"))
        self.assertFalse(F.plan(15.0, 32768, experts_gib=F.EXPERTS_GIB)["resident"])
        self.assertLess(F.need_gib(32768, "fp8", 11.45), F.need_gib(32768, "fp8"))
        lay = F.layout([card(24.0)], experts_gib=11.45)
        self.assertEqual(lay["path"], "resident")

    def test_drafting(self):
        self.assertIn("Volta", F.drafting_off_reason("75", 24, 65536, "fp8"))
        self.assertIn("offload", F.drafting_off_reason("86", 8, 65536, "fp8"))
        self.assertIsNone(F.drafting_off_reason("86", 24, 65536, "fp8"))
        self.assertIn("offload", F.drafting_off_reason("gfx1201", 16, 32768, "fp8", hip=True))
        self.assertIsNone(F.drafting_off_reason("gfx1100", 24, 32768, "fp8", hip=True))
        self.assertTrue(F.draft_fits(24, 131072, "fp8"))
        self.assertFalse(F.draft_fits(24, 262144, "fp8"))

    def test_amd(self):
        """AMD as NVIDIA (checked on an RX 9070 XT): every expert in VRAM where the file fits, else the offload path -
        UD-Q4_K_XL on 16 GB offloads, our one-T4 quant (11.45 GiB) stays resident with 1 GiB kept beside it; RDNA 3 / 4
        with 16 GB or more, Linux, one GPU."""
        self.assertIsNone(F.unsupported("hip", False, 1, [amd("gfx1100", 24)]))
        self.assertIsNone(F.unsupported("hip", False, 1, [amd("gfx1201", 15.9)]))
        self.assertIn("16 GB", F.unsupported("hip", False, 1, [amd("gfx1201", 12)]))
        self.assertIn("RDNA 3, 3.5 or 4", F.unsupported("hip", False, 0, [amd("gfx1030", 16)]))
        self.assertIn("one GPU", F.unsupported("hip", False, 2, []))
        self.assertIn("Linux only", F.unsupported("hip", True, 0, []))
        ud = F.amd_layout(amd("gfx1201", 15.9))
        self.assertEqual((ud["path"], ud["context"], ud["env"]), ("offload", 32768, {"SAGUARO_IK2_RESERVE_MB": "1024"}))
        q = F.amd_layout(amd("gfx1201", 15.9), 11.45)
        self.assertEqual((q["path"], q["args"][3]), ("resident", "1024"))
        self.assertEqual(F.amd_layout(amd("gfx1100", 24))["env"], {})

    def test_the_flags_are_the_engines_own(self):
        """saguaro-ik2 stops at a flag it does not know: every flag a layout writes is one its parser reads."""
        src = (ROOT / "src" / "ik2" / "ik2_main.cu").read_text(encoding="utf-8")
        for flag in ("--drafting", "--draft", "--parallel"):
            self.assertIn(f'a == "{flag}"', src, flag)
        for cards in ([t4()], [t4(0), t4(1)], [t4(0, 24.0, "86")]):
            lay = F.layout(cards)
            for flag in lay["args"][::2]:
                self.assertIn(f'a == "{flag}"', src, flag)
            for key in list(lay["env"]) + ["SAGUARO_IK2_WARM", "SAGUARO_IK2_KV"]:
                self.assertIn(f'"{key}"', src, key)

    def test_refusals(self):
        cases = [("metal", False, 0, [], "NVIDIA and AMD GPUs only"),
                 ("cuda", True, 0, [t4()], "Linux only"),
                 ("cuda", False, 3, [t4()], "one or two NVIDIA GPUs"),
                 ("cuda", False, 0, [t4(0, 24, "61", "Tesla P40")], "compute capability 7.0"),
                 ("cuda", False, 0, [card(6)], "8 GB of VRAM or more"),
                 ("cuda", False, 2, [card(24), card(6, 1)], "8 GB of VRAM or more on each")]
        for backend, win, asked, cards, part in cases:
            why = F.unsupported(backend, win, asked, cards)
            self.assertIn(part, why or "", (backend, win))
            self.assertNotIn("\n", why)
        for cards in ([t4()], [card(8)], [t4(0, 32, "70", "Tesla V100")], [card(12), t4(1)], []):
            self.assertIsNone(F.unsupported("cuda", False, 1, cards))
        self.assertIn("RAM and VRAM together", F.unsupported("cuda", False, 1, [card(8)], ram_gb=12.0))
        self.assertIsNone(F.unsupported("cuda", False, 1, [card(8)], ram_gb=15.4))
        self.assertIsNone(F.layout([card(6)]))


class Picker(unittest.TestCase):
    def rows(self, cards, prev=None):
        got = {}

        def pick(title, intro, opts, default, yes, *x):
            got.update(rows=opts, default=default)
            return default
        a = args(model=None)
        with patch.object(maya, "choose", pick), patch.object(maya.S, "gpus", return_value=cards), \
                patch.object(maya, "mem_gb", return_value=(251.0, 200.0)), patch.object(maya, "WIN", False):
            fam = maya.choose_family(a, prev or {})
        return fam, got

    def test_recommended_on_one_or_two_small_cards(self):
        for n in (1, 2):
            fam, got = self.rows([t4(i) for i in range(n)])
            self.assertEqual(fam, "ik2", n)
            label, note, about = got["rows"][maya.FAMILIES.index("ik2")]
            self.assertEqual((label, note), ("Inkling 2 Mini", "recommended for this PC"))
            for part in ("one or two", "16 GB", "8 GB", "AMD", "24 tok/s", "52 tok/s", "Apache 2.0"):
                self.assertIn(part, about)

    def test_four_t4s_get_hy4_and_a_setup_before_wins(self):
        fam, got = self.rows([t4(i) for i in range(4)])
        self.assertEqual(fam, "hy4")
        self.assertEqual(got["rows"][maya.FAMILIES.index("hy4")][1], "recommended for this PC")
        self.assertEqual(got["rows"][maya.FAMILIES.index("ik2")][1], "new")
        prev = {"exe": "/b/saguaro-glm", "args": ["--glm-pack", "/p"], "installer": {"quant": "Maya-S24"}}
        self.assertEqual(self.rows([t4()], prev)[0], "glm")

    def test_no_recommendation_elsewhere(self):
        self.assertIsNone(maya.family_fit([], 64))
        self.assertIsNone(maya.family_fit([t4(i) for i in range(3)], 64))
        self.assertIsNone(maya.family_fit([t4(0, 48, "89", "RTX 6000 Ada")], 64))
        self.assertIsNone(maya.family_fit([t4(i) for i in range(4)], 2))      # (no memory for Hy4's Maya-S)

    def test_gpus(self):
        with patch.object(maya, "choose", lambda *a, **k: 0), patch.object(maya, "WIN", False):
            got = maya.choose_gpus(args(), [t4(i) for i in range(4)], F.MIN_CC, "ik2")
            self.assertEqual(len(got), 2)
            self.assertEqual(len(maya.choose_gpus(args(), [t4()], F.MIN_CC, "ik2")), 1)
            self.assertEqual([g["index"] for g in maya.choose_gpus(args(gpu=3), [t4(i) for i in range(4)],
                                                                   F.MIN_CC, "ik2")], [3])
            self.assertIn("one GPU or two", refusal(lambda: maya.choose_gpus(args(gpus="0,1,2"),
                                                                             [t4(i) for i in range(4)], F.MIN_CC, "ik2")))
            self.assertIn("8 GB of VRAM", refusal(lambda: maya.choose_gpus(args(), [t4(0, 6, "86")], F.MIN_CC, "ik2")))
            got = maya.choose_gpus(args(gpus="0,1"), [t4(0, 12, "86"), t4(1, 16, "89")], F.MIN_CC, "ik2")
            self.assertEqual([g["index"] for g in got], [1, 0])   # (the split: the larger card first)

    def test_check_pc_refuses_first(self):
        with patch.object(maya.S, "gpus", side_effect=AssertionError("no GPU read before the refusal")), \
                patch.object(maya, "WIN", False):
            self.assertIn("NVIDIA and AMD GPUs only", refusal(lambda: maya.check_pc(args(backend="metal"), "ik2")))
            self.assertIn("one or two NVIDIA GPUs", refusal(lambda: maya.check_pc(args(gpus="0,1,2"), "ik2")))
            self.assertIn("on AMD runs on one GPU", refusal(lambda: maya.check_pc(args(backend="hip", gpus="0,1"), "ik2")))
        with patch.object(maya, "WIN", True):
            self.assertIn("Linux only", refusal(lambda: maya.check_pc(args(), "ik2")))
            self.assertIn("Linux only", refusal(lambda: maya.check_pc(args(backend="hip"), "ik2")))


class Steps(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for name in ("ok", "say", "step", "warn"):
            p = patch.object(maya, name, lambda *a, **kw: None)
            p.start()
            self.addCleanup(p.stop)

    def files(self):
        d = self.root / "models" / maya.IK2
        d.mkdir(parents=True)
        gguf = d / "Inkling-2-Mini-UD-Q4_K_XL.gguf"
        gguf.write_bytes(b"GGUF" + b"\0" * 60)
        return d, gguf

    def tok(self, d):
        tok = d / maya.IK2_DIR / "tokenizer"
        tok.mkdir(parents=True)
        (tok / "vocab.json").write_text("{}", encoding="utf-8")
        return tok

    def write(self, pc, gguf, tok, ctx, d, vision=None, a=None):
        lay = F.layout(pc["gpus"], len(pc["gpus"]))
        with patch.object(maya, "ROOT", self.root), \
                patch.object(maya, "write_run_script", lambda p, port: p.with_name("run-" + p.stem + ".sh")):
            path = maya.write_config_ik2(a or args(), pc, {}, gguf, tok, ctx, self.root / "models", d, lay, vision)
        return path, json.loads(path.read_text(encoding="utf-8"))

    def test_tokenizer_with_its_own_template(self):
        d, gguf = self.files()
        seen = []

        def run(cmd, check=True, **kw):
            seen.append(cmd)
            out = Path(cmd[cmd.index("--out") + 1]) / "tokenizer"
            out.mkdir(parents=True, exist_ok=True)
            (out / "vocab.json").write_text("{}", encoding="utf-8")
            (out / "chat_template.jinja").write_text("x", encoding="utf-8")
            return SimpleNamespace(returncode=0)
        with patch.object(maya, "run", run):
            tok = maya.ik2_tokenizer_step(args(), d, gguf)
            self.assertEqual(tok, d / "saguaro-ik2" / "tokenizer")
            cmd = seen[0]
            self.assertEqual(cmd[cmd.index("--chat-template") + 1], str(maya.IK2_TEMPLATE))
            self.assertEqual(cmd[cmd.index("--gguf") + 1], str(gguf))
            self.assertIn("--check", cmd)
            rec = json.loads((d / "saguaro-ik2" / "sources.json").read_text(encoding="utf-8"))
            self.assertEqual([f["name"] for f in rec["files"]], [gguf.name])
            maya.ik2_tokenizer_step(args(), d, gguf)              # (the same tools and GGUF: not exported again)
            self.assertEqual(len(seen), 1)
            maya.ik2_tokenizer_step(args(repack=True), d, gguf)
            self.assertEqual(len(seen), 2)

    def test_the_config_on_one_card(self):
        d, gguf = self.files()
        tok = self.tok(d)
        path, cfg = self.write({"gpus": [t4(2)], "backend": "cuda"}, gguf, tok, 32768, d, a=args(drafting="off"))
        self.assertEqual(path.name, "maya-inkling-2-mini.json")
        self.assertEqual(cfg["exe"], str(maya.IK2_EXE))
        self.assertEqual(cfg["args"], ["--model", str(gguf), "--max-context", "32768", "--gpus", "1", "--chunk", "2048",
                                       "--threads", "8", "--drafting", "off"])
        self.assertEqual(cfg["env"], {"SAGUARO_IK2_ROOM_MB": "1024", "SAGUARO_IK2_WARM": str(maya.IK2_PROFILE)})
        self.assertEqual((cfg["sampling"], cfg["reasoning_effort"]), ({"temperature": 1.0, "top_p": 1.0}, "high"))
        self.assertEqual((cfg["model_name"], cfg["gpu"], cfg["tokenizer"]), ("inkling-2-mini", [2], str(tok)))
        inst = cfg["installer"]
        self.assertEqual((inst["family"], inst["ik2_quant"], inst["cards"], inst["measured"], inst["drafting"]),
                         ("ik2", Q.IK2_QUANT, 1, True, "off"))
        self.assertEqual(Q.family_of(cfg), "ik2")
        self.assertNotIn("vision", cfg)

    def test_the_config_follows_the_cards(self):
        """Two cards after one: the flags are the two-card layout's (a kept --chunk would be the one card's); the
        user's own variable and the server's settings stay."""
        d, gguf = self.files()
        tok = self.tok(d)
        path, cfg = self.write({"gpus": [t4(2)], "backend": "cuda"}, gguf, tok, 32768, d)
        cfg["env"]["MY_VAR"] = "1"
        cfg["env"]["SAGUARO_IK2_PIN"] = "0"                   # (the previous engine's: dropped)
        cfg["api_key"] = "k"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        v = {"exe": "/b/vision-glm", "mmproj": str(d / "mmproj-BF16.gguf"), "gpu": True}
        path, cfg = self.write({"gpus": [t4(2), t4(3)], "backend": "cuda"}, gguf, tok, 131072, d, v, args(drafting="off"))
        self.assertEqual(cfg["args"], ["--model", str(gguf), "--max-context", "131072", "--gpus", "2", "--chunk", "8192",
                                       "--threads", "8", "--drafting", "off"])
        self.assertEqual(cfg["env"], {"SAGUARO_IK2_ROOM_MB": "1024", "SAGUARO_IK2_WARM": str(maya.IK2_PROFILE),
                                      "MY_VAR": "1"})
        self.assertEqual((cfg["gpu"], cfg["api_key"], cfg["vision"]), ([2, 3], "k", v))
        # (the same cards again: a --chunk set by hand stays)
        cfg["args"][cfg["args"].index("--chunk") + 1] = "4096"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        path, cfg = self.write({"gpus": [t4(2), t4(3)], "backend": "cuda"}, gguf, tok, 131072, d)
        self.assertEqual(cfg["args"][cfg["args"].index("--chunk") + 1], "4096")

    def test_vision_step(self):
        d, gguf = self.files()
        va = lambda **kw: args(**dict(dict(rebuild=False, download_model=False, yes=False), **kw))   # noqa: E731
        self.assertIsNone(maya.ik2_vision_step(va(no_vision=True), {}, {}, d, d))
        with patch.object(maya, "offer_download", lambda *a, **k: False):
            self.assertIsNone(maya.ik2_vision_step(va(), {}, {}, d, d))
        sure = [patch.object(maya, "file_incomplete", lambda *a: None), patch.object(maya, "sha256_checked", lambda *a: True)]
        for p in sure:
            p.start()
            self.addCleanup(p.stop)
        with patch.object(maya, "vision_encoder_ready", lambda *a: True):
            v = maya.ik2_vision_step(va(), {}, {}, d, d)
        self.assertEqual(v, {"exe": str(maya.VEXE), "mmproj": str(d / "mmproj-BF16.gguf"), "gpu": True})
        with patch.object(maya, "vision_encoder_ready", lambda *a: False):
            self.assertIsNone(maya.ik2_vision_step(va(), {}, {}, d, d))

    def test_model_step_checks_the_sha256(self):
        d, gguf = self.files()
        a = args(download_model=False)
        with patch.object(maya, "file_incomplete", lambda *x: None), patch.object(maya, "rotational", lambda p: False):
            with patch.object(maya, "sha256_checked", lambda *x: True):
                self.assertEqual(maya.ik2_model_step(a, d), gguf)
            with patch.object(maya, "sha256_checked", lambda *x: False), patch.object(maya, "fail", side_effect=SystemExit):
                self.assertRaises(SystemExit, maya.ik2_model_step, a, d)
        with patch.object(maya, "offer_download", lambda *x, **k: False):       # (a short file, the download declined)
            self.assertIsNone(maya.ik2_model_step(a, d))

    def test_a_start_of_it(self):
        d, gguf = self.files()
        exe = self.root / "saguaro-ik2"
        exe.touch()
        tok = self.tok(d)
        cfg = {"exe": str(exe), "tokenizer": str(tok), "gpu": [0], "model_name": "inkling-2-mini",
               "installer": {"family": "ik2", "ik2_quant": Q.IK2_QUANT},
               "args": ["--model", str(gguf), "--max-context", "32768", "--gpus", "1"]}
        path = self.root / "maya-inkling-2-mini.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        a = SimpleNamespace(parallel=None, port=None, host=None, api_key=None, gpus=None, gpu=None, yes=True)
        for n in ("select_build_backend", "refresh_engine", "rename_programs"):
            p = patch.object(maya, n, return_value=None)
            p.start()
            self.addCleanup(p.stop)
        with patch.object(maya, "speed_calibration", side_effect=AssertionError("no calibration")), \
                patch.object(maya, "prefill_calibration", side_effect=AssertionError("no calibration")), \
                patch.object(maya.S, "gpus", return_value=[t4()]), patch.dict(os.environ, {"INVOCATION_ID": "x"}):
            cmd, env, info = maya.server_command(path, a)
            self.assertIn("serve", " ".join(cmd))
            self.assertEqual((info["model"], info["label"]), ("inkling-2-mini", "Inkling 2 Mini · unsloth UD-Q4_K_XL"))
            gguf.unlink()                                          # (its GGUF gone: said, not started)
            with patch.object(maya, "fail", side_effect=lambda what, how="": (_ for _ in ()).throw(SystemExit(what))):
                with self.assertRaises(SystemExit) as e:
                    maya.server_command(path, a)
        self.assertIn("the model's GGUF file is missing", str(e.exception))

    def test_drafter_step(self):
        d, gguf = self.files()
        va = lambda **kw: args(**dict(dict(rebuild=False, download_model=False, yes=False), **kw))   # noqa: E731
        self.assertIsNone(maya.ik2_drafter_step(va(drafting="off"), d, 24, 65536, "fp8"))
        with patch.object(maya, "offer_download", lambda *a, **k: False):
            self.assertIsNone(maya.ik2_drafter_step(va(), d, 24, 65536, "fp8"))
        with patch.object(maya, "file_incomplete", lambda *a: None), patch.object(maya, "sha256_checked", lambda *a: True):
            self.assertEqual(maya.ik2_drafter_step(va(), d, 24, 65536, "fp8"), d / "mtp.safetensors")

    def test_a_gguf_of_your_own(self):
        """--model-dir with another GGUF (our one-T4 quant): taken as it is, no download, no sha256."""
        d = self.root / "own"
        d.mkdir()
        own = d / "Inkling-2-Mini-T4-NVFP4-IQ3_XXS.gguf"
        own.write_bytes(b"GGUF")
        (d / "mmproj-BF16.gguf").write_bytes(b"GGUF")
        self.assertIsNone(maya.ik2_local_gguf(args(), d))
        self.assertEqual(maya.ik2_local_gguf(args(model_dir=str(d)), d), own)
        with patch.object(maya, "offer_download", side_effect=AssertionError("no download")):
            self.assertEqual(maya.ik2_model_step(args(model_dir=str(d)), d), own)

    def test_set_up_on_a_16_gb_amd_card(self):
        """An RX 9070 XT (16 GB, gfx1201) with a GGUF of your own whose experts fit (our one-T4 quant): resident with
        1 GiB kept beside the experts; drafting off (the setup's sums are the offload path's at 16 GB)."""
        d = self.root / "own"
        d.mkdir()
        own = d / "x.gguf"
        own.write_bytes(b"GGUF")
        pc = {"gpus": [amd("gfx1201", 15.9)], "backend": "hip"}
        seen = {}
        stub = lambda name, ret: patch.object(maya, name, lambda *a, **k: ret)   # noqa: E731
        common = [patch.object(maya, "check_pc", lambda a, fam: pc), stub("tools_step", d), stub("build_step", {}),
                  stub("ik2_tokenizer_step", d), stub("ik2_vision_step", None), patch.object(maya, "IK2_EXE", own),
                  patch.object(maya, "mem_gb", return_value=(96.0, 80.0)), patch.object(maya, "WIN", False),
                  patch.object(maya, "choose_context", lambda a, prev, opts, rec: rec),
                  patch.object(maya, "ik2_drafter_step", lambda *a, **k: d / "mtp.safetensors"),
                  patch.object(maya, "write_config_ik2", lambda a, *r, **k: seen.update(drafting=a.drafting, lay=r[7],
                                                                                    drafter=r[-1]))]
        for p in common:
            p.start()
            self.addCleanup(p.stop)
        with patch.object(F, "experts_gib_of", lambda path: 11.45):
            maya.set_up_ik2(args(backend="hip", model_dir=str(d)), {})
        self.assertEqual((seen["lay"]["env"], seen["lay"]["path"], seen["drafting"], seen["drafter"]),
                         ({"SAGUARO_IK2_RESERVE_MB": "1024"}, "resident", "off", None))

    def test_bench_prompt(self):
        p = maya.bench_chat("ik2", self.root)("Hello?")
        self.assertIn("Thinking effort level: 0<|end_message|>", p)
        self.assertTrue(p.endswith("<|message_user|><|content_text|>Hello?<|end_message|>"
                                   "<|message_model|><|content_text|>"), p[-120:])

    def test_nothing_to_tune_and_its_target(self):
        cfg = {"exe": str(maya.IK2_EXE), "args": ["--model", "/m/x.gguf"], "installer": {"family": "ik2"}}
        self.assertFalse(maya.needs_speed_calibration(cfg))
        path = self.root / "maya-inkling-2-mini.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        self.assertFalse(maya.calibrate_config(path))
        self.assertFalse(maya.speed_calibration(path))
        self.assertEqual(maya.exe_for_target(maya.IK2_TARGET), maya.IK2_EXE)
        self.assertEqual(maya.bench_exe(cfg), maya.IK2_EXE)
        self.assertEqual(maya.family_target("ik2"), "saguaro-ik2")
        self.assertIn("add_executable(saguaro-ik2 ", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))

    def test_set_up_runs_its_steps(self):
        """--model inkling-2-mini on two T4s: the two-card layout into the config, in the setup's order."""
        d, gguf = self.files()
        tok = self.tok(d)
        pc = {"gpus": [t4(2), t4(3)], "backend": "cuda", "archs": [75]}
        seen = {}

        def write(a, pc_, meta, gguf_, tok_, ctx, models, d_, lay, vision=None, drafter=None, **k):
            seen.update(ctx=ctx, lay=lay, vision=vision, d=d_, drafting=a.drafting, drafter=drafter)
            return self.root / "cfg.json"
        with patch.object(maya, "check_pc", lambda a, fam: pc), patch.object(maya, "tools_step", lambda a: self.root), \
                patch.object(maya, "build_step", lambda a, pc_, llama, target: {"target": target}), \
                patch.object(maya, "IK2_EXE", gguf), patch.object(maya, "ik2_model_step", lambda a, d_, q=None: gguf), \
                patch.object(maya, "ik2_tokenizer_step", lambda a, d_, g: tok), \
                patch.object(maya, "ik2_vision_step", lambda *x: None), patch.object(maya, "write_config_ik2", write):
            got = maya.set_up_ik2(args(models_dir=str(self.root / "models"), ik2_quant="ud-q4_k_xl"), {})
        self.assertEqual(got, self.root / "cfg.json")
        self.assertEqual((seen["ctx"], seen["lay"]["args"][:2], seen["d"]), (131072, ["--gpus", "2"], d))
        self.assertEqual((seen["drafting"], seen["drafter"]), ("off", None))   # (two GPUs, Turing: drafting off)


class StrataPacks(unittest.TestCase):
    """Strata's packs of Inkling-2-Mini beside UD-Q4_K_XL: names, the table, the pick for the cards, a pack folder."""

    def test_names(self):
        for name, key in (("inkling-2-mini-sq-15", "strata-sq-15"), ("Inkling-2-Mini-Strata-SQ-20", "strata-sq-20"),
                          ("strata-sq-10", "strata-sq-10")):
            self.assertEqual(Q.resolve(name), ("ik2", key), name)
        self.assertEqual(Q.config_stem("ik2", "strata-sq-15"), "maya-inkling-2-mini-sq-15")
        self.assertEqual(Q.shown("ik2", "strata-sq-15"), "Inkling 2 Mini · Strata SQ-15")

    def test_table(self):
        for k in Q.IK2_QUANTS[1:]:
            m = maya.ik2_entry(k)
            self.assertEqual(m["format"], "ik2-strata")
            self.assertTrue(m["gguf"].endswith(".strata"))
            self.assertIn("tokenizer.json", m["files"])
            self.assertEqual(set(m["files"]), set(m["sha256"]))
            self.assertTrue(all(len(h) == 64 for h in m["sha256"].values()))
        self.assertEqual(maya.ik2_entry("strata-sq-15")["files"]["Inkling-2-Mini-SQ-15.strata"], 14348316672)

    def test_recommended(self):
        rec = lambda v, n=1: maya.ik2_recommend([t4(i, v) for i in range(n)], False)   # noqa: E731
        self.assertEqual(rec(11.0), "strata-sq-10")    # (12 GB: Strata's 12 GB pack)
        self.assertEqual(rec(15.0), "strata-sq-15")    # (a T4: Strata's 16 GB pack, most of its experts on the card)
        self.assertEqual(rec(23.5), "strata-sq-20")    # (24 GB: the original weights, every expert on the card)
        self.assertEqual(rec(15.0, 2), "strata-sq-20")  # (two T4s: every expert across them)

    def test_pack_folder(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "Inkling-2-Mini-SQ-15.strata").write_bytes(b"STRATA\x00\x01")
            a = args(model_dir=str(d))
            self.assertEqual(maya.ik2_pack_in(d), "strata-sq-15")
            self.assertEqual(maya.choose_ik2_quant(a, {}, [t4()], False), "strata-sq-15")
            self.assertIsNone(maya.ik2_local_gguf(a, d))   # (not taken for a GGUF of your own, nor UD offered)

    def test_pieces(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "big.strata"
            self.assertIsNone(maya.piece_paths(f, 10 << 20))
            ps = maya.piece_paths(f, 14348316672)
            self.assertEqual(len(ps), maya.DL_PIECES)
            self.assertEqual((ps[0][1], ps[-1][2]), (0, 14348316672 - 1))
            self.assertTrue(all(ps[i][2] + 1 == ps[i + 1][1] for i in range(len(ps) - 1)))
            f.write_bytes(b"x")                        # (a single stream already started: kept as one)
            self.assertIsNone(maya.piece_paths(f, 14348316672))


if __name__ == "__main__":
    unittest.main()
