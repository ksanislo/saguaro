"""serve/test_new_engines_ui.py - Hy4-Flash and Inkling 2 Mini in the web app's back end and page logic: the Models
page's settings for a Hy4 config (only flags saguaro-hy4 reads, both directions), a config's install state, size and
"needs" line, the download catalog's sizes and its fit notes for one, two and four Tesla T4s, no Turbo for either
engine, and - under Node - the Monitor's experts card from each engine's own protocol lines, the thinking levels, the
names and the page's helpers.  No GPU, no network.

    python -m unittest serve.test_new_engines_ui -v
"""
from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from serve import model_pull, models as M  # noqa: E402
from serve import server  # noqa: E402
from serve.test_models import qfn_cfg  # noqa: E402  (the same QFN config shape the Models page tests run against)

HY4_SRC = (ROOT / "src" / "hy4" / "hy4_main.cpp").read_text(encoding="utf-8")
HY4_FLAGS = set(re.findall(r'a == "(--[a-z0-9-]+)"', HY4_SRC))
HY4_BARE = {"--mtp", "--dfly2", "--serve", "--top", "--verbose", "--room-report"}
QFN_SRC = (ROOT / "engines" / "qfn" / "src" / "program" / "generate.cpp").read_text(encoding="utf-8")
QFN_FLAGS = set(re.findall(r'a == "(--[a-z0-9-]+)"', QFN_SRC))    # every flag saguaro-qfn's parser knows


def t4(i=0, gib=16.0):
    return {"index": i, "name": "Tesla T4", "vram_gb": gib, "arch": "75"}


def info_of(line: str) -> dict:
    """An engine's INFO line as serve/server.py's start loop reads it into engine.info."""
    out = {}
    for kv in line.split()[1:]:
        k, _, v = kv.partition("=")
        out[k] = int(v) if v.lstrip("-").isdigit() else v
    return out


class Hy4Settings(unittest.TestCase):
    """A Hy4-Flash config: the page reads and writes only what saguaro-hy4 knows."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.d = self.root / "models" / "Hy4-Flash-Maya-S"
        self.pack = self.d / "saguaro-hy4"
        self.write_pack("vision.dim 1152\ncontext_length 1048576\nn_mtp 0\n")
        cfg = json.loads((ROOT / "configs" / "Hy4-Flash-Maya-S-IQ2_S.json").read_text(encoding="utf-8"))
        cfg["args"][1] = str(self.pack)
        cfg["tokenizer"] = str(self.pack / "tokenizer")
        cfg["installer"].update(model_dir=str(self.d), models_dir=str(self.root / "models"))
        self.p = self.root / "maya-hy4-flash-maya-s.json"
        self.p.write_text(json.dumps(cfg), encoding="utf-8")
        self.cfg = M.read(self.p)

    def write_pack(self, txt):
        (self.pack / "tokenizer").mkdir(parents=True, exist_ok=True)
        (self.pack / "hy4.txt").write_text(txt, encoding="utf-8")
        for sub, name, n in (("Maya-S", "trunk.gguf", 5000), ("Maya-XS", "ngram.gguf", 7000), ("Maya-XS", "mmproj.gguf", 300)):
            (self.d / sub).mkdir(parents=True, exist_ok=True)
            (self.d / sub / name).write_bytes(b"x" * n)
        (self.pack / "tensors.txt").write_text(
            "file 0 Maya-S/trunk.gguf\nfile 1 Maya-XS/ngram.gguf\nfile 2 Maya-XS/mmproj.gguf\n"
            "file 3 ../outside.gguf\nt blk.0.w Q8_0 0 0 34 32\n", encoding="utf-8")

    def sound(self, cfg):
        """Every argument is a flag saguaro-hy4's parser reads (src/hy4/hy4_main.cpp), each with its value."""
        args, i = [str(a) for a in cfg["args"]], 0
        while i < len(args):
            self.assertIn(args[i], HY4_FLAGS, args)
            i += 1 if args[i] in HY4_BARE else 2
        self.assertEqual(i, len(args), args)
        for bad in ("--parallel", "--vram-reserve-mib", "--expert-cache", "--prefill", "--dflash", "--kv-resident"):
            self.assertNotIn(bad, args)
        return cfg

    def test_family_and_settings(self):
        self.assertEqual(M.family(self.cfg), "hy4")
        s = M.settings_of(self.cfg)
        self.assertEqual({k: s[k] for k in ("context", "gpus", "images", "draft", "tp", "reserve_gb", "prefill_chunk",
                                            "cpu_threads", "kv")},
                         {"context": 131072, "gpus": [0, 1, 2, 3], "images": True, "draft": "none", "tp": 4,
                          "reserve_gb": 0.5, "prefill_chunk": 2048, "cpu_threads": None, "kv": "auto"})
        self.assertEqual(set(s) - {"env"}, set(M.HY4_KEYS) - {"env"})
        self.assertEqual(M.choices(self.cfg), {"draft": ["none"], "images": True})
        same = self.sound(M.apply(self.cfg, M.settings_of(self.cfg)))      # (as it is: nothing moves)
        self.assertEqual(same, self.cfg)

    def test_each_setting_maps_to_its_flag(self):
        for patch, flag, want in (({"context": 65536}, "--max-context", "65536"), ({"tp": 0}, "--tp", "0"),
                                  ({"tp": "auto"}, "--tp", None), ({"tp": 2, "gpus": [0, 1]}, "--tp", "2"),
                                  ({"reserve_gb": 0.75}, "--reserve-gb", "0.75"), ({"reserve_gb": None}, "--reserve-gb", None),
                                  ({"prefill_chunk": 1024}, "--chunk", "1024"), ({"prefill_chunk": None}, "--chunk", None),
                                  ({"cpu_threads": 12}, "--cpu-threads", "12"), ({"kv": "q8"}, "--kv", "q8"),
                                  ({"kv": "f16"}, "--kv", "f16"), ({"kv": "auto"}, "--kv", None),
                                  ({"images": False}, "--vision", "off"), ({"images": True}, "--vision", None)):
            with self.subTest(patch=patch):
                new = self.sound(M.apply(self.cfg, patch))
                self.assertEqual(M.flag(new["args"], flag), want)
                back = M.settings_of(new)                                  # (and read back as it was set)
                for k, v in patch.items():
                    self.assertEqual(back[k], v, k)
        self.assertEqual(M.apply(self.cfg, {"gpus": [2, 3], "tp": "auto"})["gpu"], [2, 3])

    def raised(self, patch):
        with self.assertRaises(ValueError) as e:
            M.apply(self.cfg, patch)
        return str(e.exception)

    def test_what_its_engine_does_not_read_is_refused(self):
        for key in ("reserve_mib", "expert_cache", "prefill", "kv_streaming", "kv_resident", "mmap", "kv_cache", "ram_gb"):
            self.assertIn("not a setting of this model", self.raised({key: 1}), key)
        self.assertIn("one request at a time", self.raised({"parallel": 2}))
        self.sound(M.apply(self.cfg, {"parallel": 1}))                      # (1 is what it does: taken, nothing written)
        self.assertIn("no MTP block", self.raised({"draft": "mtp"}))
        self.assertIn("no DFly2 drafter", self.raised({"draft": "dfly2"}))
        self.assertIn("no drafter", self.raised({"draft": "auto"}))
        self.assertIn("KV cache", self.raised({"kv": "int8"}))
        self.assertIn("a group of 9 needs 9 GPUs", self.raised({"tp": 9}))
        self.assertIn("a group of 4 needs 4 GPUs", self.raised({"gpus": [2, 3]}))   # (the config's --tp 4 on two cards)

    def test_a_pack_with_drafters_offers_them(self):
        self.write_pack("vision.dim 1152\nn_mtp 1\ndfly_layers 5\n")
        self.assertEqual(M.choices(self.cfg)["draft"], ["auto", "mtp", "dfly2", "none"])
        for draft, has, hasnt in (("mtp", ["--mtp"], ["--dfly2", "--drafting"]), ("dfly2", ["--dfly2"], ["--mtp"]),
                                  ("auto", ["--drafting"], ["--mtp", "--dfly2"]), ("none", [], ["--mtp", "--dfly2", "--drafting"])):
            new = self.sound(M.apply(self.cfg, {"draft": draft}))
            for f in has:
                self.assertIn(f, new["args"], draft)
            for f in hasnt:
                self.assertNotIn(f, new["args"], draft)
            self.assertEqual(M.settings_of(new)["draft"], draft)

    def test_a_pack_without_the_tower_reads_text_only(self):
        self.write_pack("n_mtp 0\n")
        self.assertEqual((M.choices(self.cfg)["images"], M.settings_of(self.cfg)["images"]), (False, False))
        self.assertIn("no vision tower", self.raised({"images": True}))

    def test_no_turbo(self):
        info = M.turbo_info(self.cfg)
        self.assertEqual((info["on"], info["supported"], info["plan"]), (False, False, {}))
        self.assertEqual(M.turbo_patch(self.cfg), {})
        with self.assertRaises(ValueError) as e:
            M.turbo_enable(self.p)
        self.assertIn("Hy4-Flash's engine has none", str(e.exception))
        self.assertEqual(server.NO_LIVE_TURBO, {"ik2": "Inkling 2", "hy4": "Hy4-Flash"})

    def test_described(self):
        d = M.describe(self.p, None)
        self.assertEqual((d["family"], d["quant"], d["ok"], d["missing"]), ("hy4", "Maya-S", True, None))
        self.assertEqual(d["folder"], str(self.d))
        self.assertEqual(d["contexts"], M.HY4_CONTEXTS)
        self.assertEqual((d["context_range"]["trained"], d["context_range"]["tested"]), (1048576, 131072))
        # (its files where the repo keeps them, in subfolders: counted by the pack's own index, with the pack)
        pack_files = sum((self.pack / n).stat().st_size for n in ("hy4.txt", "tensors.txt"))
        self.assertEqual(d["size"], 5000 + 7000 + 300 + pack_files)
        self.assertEqual([p.name for p in M.pack_shards(self.pack)], ["trunk.gguf", "ngram.gguf", "mmproj.gguf"])
        self.assertEqual(d["needs"], {"cards": 4, "measured": None, "limits": [   # (a config written by hand)
            "one conversation at a time", "no saved conversations yet", "pictures not measured on Tesla T4s yet"]})
        shutil.rmtree(self.pack)
        self.assertIn("model files are not on this PC", M.describe(self.p, None)["missing"])

    def test_delete_plan_takes_the_files_the_pack_reads(self):
        plan = M.delete_plan(self.root, self.p)
        self.assertEqual((plan["ok"], [i["what"] for i in plan["items"]]), (True, ["folder"]))   # (its own folder, whole)
        self.assertNotIn("outside.gguf", json.dumps(plan))                  # (a path out of the folder: never its own)


class QfnSettings(unittest.TestCase):
    """A QFN config on the Models page: the page reads and writes only what saguaro-qfn itself knows - its engine
    flags (src/program/generate.cpp's parser) and the config's own "parallel" key - and nothing of the CYBER-FROST
    path (its ram_gb / SAGUARO_EXPERT_RAM_GB is refused here).  Turbo is supported."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.p = qfn_cfg(self.root)
        self.cfg = M.read(self.p)

    def sound(self, cfg):
        """Every argument is a flag saguaro-qfn's parser reads; the other engines' flags are never written."""
        args = [str(a) for a in cfg["args"]]
        for a in args:
            if a.startswith("--"):
                self.assertIn(a, QFN_FLAGS, args)
        for bad in ("--parallel", "--tp", "--dspark", "--kv-streaming", "--mmap-experts", "--dflash"):
            self.assertNotIn(bad, args)
        return cfg

    def test_family_and_settings(self):
        self.assertEqual(M.family(self.cfg), "qfn")
        s = M.settings_of(self.cfg)
        self.assertEqual(set(s) - {"env"}, set(M.QFN_KEYS) - {"env"})
        self.assertEqual(s["ram_gib"], 71)                       # (--resident-budget-gib, the setup's tier)
        self.assertEqual(s["kv_resident"], 32768)                # (--kv-resident, the K/V window in VRAM)
        self.assertEqual((s["parallel"], s["draft"], s["images"]), (1, "mtp", False))
        same = self.sound(M.apply(self.cfg, s))                  # (as it is: nothing moves)
        self.assertEqual(same["args"], self.cfg["args"])

    def test_each_setting_maps_to_its_flag(self):
        for patch, flag, want in (({"context": 32768}, "--max-context", "32768"),
                                  ({"ram_gib": 48}, "--resident-budget-gib", "48"),
                                  ({"ram_gib": None}, "--resident-budget-gib", None),
                                  ({"kv_resident": 16384}, "--kv-resident", "16384"),
                                  ({"kv_resident": None}, "--kv-resident", None),
                                  ({"expert_cache": 4000}, "--expert-cache", "4000"),
                                  ({"expert_cache": None}, "--expert-cache", "auto"),
                                  ({"prefill": 1024}, "--prefill", "1024"),
                                  ({"reserve_mib": 400}, "--vram-reserve-mib", "400"),
                                  ({"draft": "none"}, "--mtp-max-t", "1"),
                                  ({"kv": "fp16"}, "--kv", "fp16")):
            with self.subTest(patch=patch):
                new = self.sound(M.apply(self.cfg, patch))
                self.assertEqual(M.flag(new["args"], flag), want)
                back = M.settings_of(new)                        # (and read back as it was set)
                for k, v in patch.items():
                    self.assertEqual(back[k], v, k)
        self.assertEqual(M.apply(self.cfg, {"gpus": [1]})["gpu"], [1])

    def test_answers_at_once_is_the_configs_key(self):
        new = self.sound(M.apply(self.cfg, {"parallel": 6}))     # (the driver turns it into the engine's --batch)
        self.assertEqual((new["parallel"], M.settings_of(new)["parallel"]), (6, 6))
        self.assertNotIn("--parallel", new["args"])
        self.assertNotIn("parallel", self.sound(M.apply(new, {"parallel": 1})))

    def test_the_other_families_settings_are_refused(self):
        for bad in ("ram_gb", "tp", "kv_streaming", "mmap", "dspark_conf", "cpu_threads", "kv_cache", "cache_gb"):
            with self.assertRaises(ValueError, msg=bad) as e:
                M.apply(self.cfg, {bad: 1})
            self.assertIn("not a setting of this model", str(e.exception))
        with self.assertRaisesRegex(ValueError, "Answers at once"):
            M.apply(self.cfg, {"parallel": 9})

    def test_turbo_is_supported(self):
        self.assertNotIn("qfn", M.TURBO_LIVE_DEFAULT)
        self.assertEqual(M.TURBO_DRAFT["qfn"], "mtp")
        info = M.turbo_info(self.cfg)
        self.assertEqual(info["on"], False)


class Ik2Installed(unittest.TestCase):
    """An Inkling 2 config's install state: the page can switch to it once its GGUF and tokenizer folder are there."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.d = root / "models" / "Inkling-2-Mini"
        (self.d / "saguaro-ik2" / "tokenizer").mkdir(parents=True)
        self.gguf = self.d / "Inkling-2-Mini-UD-Q4_K_XL.gguf"
        self.gguf.write_bytes(b"g" * 4000)
        (self.d / "mmproj-BF16.gguf").write_bytes(b"m" * 600)
        self.vocab = self.d / "saguaro-ik2" / "tokenizer" / "vocab.json"
        self.vocab.write_text("{}", encoding="utf-8")
        cfg = {"exe": "/b/saguaro-ik2", "args": ["--model", str(self.gguf), "--max-context", "32768", "--gpus", "1",
                                                 "--chunk", "1024", "--threads", "8"],
               "tokenizer": str(self.vocab.parent), "model_name": "inkling-2-mini", "gpu": [2],
               "vision": {"exe": "/b/vision-glm", "mmproj": str(self.d / "mmproj-BF16.gguf"), "gpu": True},
               "installer": {"family": "ik2", "ik2_quant": "ud-q4_k_xl", "quant": "unsloth UD-Q4_K_XL",
                             "model_dir": str(self.d), "measured": True, "cards": 1}}
        self.p = root / "maya-inkling-2-mini.json"
        self.p.write_text(json.dumps(cfg), encoding="utf-8")

    def test_described_and_startable(self):
        d = M.describe(self.p, None)
        self.assertEqual((d["family"], d["quant"], d["ok"], d["missing"]), ("ik2", "unsloth UD-Q4_K_XL", True, None))
        self.assertEqual((d["folder"], d["size"]), (str(self.d), 4600))
        self.assertEqual(d["contexts"], M.IK2_CONTEXTS)
        self.assertEqual(d["context_range"]["tested"], 131072)
        self.assertEqual(d["needs"], {"cards": 1, "measured": True,
                                      "limits": ["several conversations at once only on one GPU that holds every expert"]})
        self.assertEqual((d["settings"]["draft"], d["settings"]["parallel"]), ("none", 1))

    def test_what_is_missing_is_said(self):
        self.vocab.unlink()
        self.assertIn("tokenizer folder is not on this PC", M.describe(self.p, None)["missing"])
        self.gguf.unlink()
        d = M.describe(self.p, None)
        self.assertEqual(d["ok"], False)
        self.assertIn("model file is not on this PC", d["missing"])
        cfg = M.read(self.p)
        cfg["args"] = ["--max-context", "32768"]
        self.assertIn("not set", M.missing(cfg))

    def test_its_files_can_be_deleted_with_it(self):
        """The setup's saguaro-ik2 folder (its record of the GGUF) is what tells the page the files are Saguaro's."""
        cfg = M.read(self.p)
        self.assertIn("did not download", M.delete_plan(self.p.parent, self.p)["why"])   # (no models folder recorded)
        cfg["installer"]["models_dir"] = str(self.d.parent)
        self.p.write_text(json.dumps(cfg), encoding="utf-8")
        self.assertIn("did not download", M.delete_plan(self.p.parent, self.p)["why"])   # (no record beside the tokenizer)
        (self.d / "saguaro-ik2" / "sources.json").write_text(json.dumps({"files": [{"name": self.gguf.name}]}),
                                                             encoding="utf-8")
        plan = M.delete_plan(self.p.parent, self.p)
        self.assertEqual((plan["ok"], [(i["what"], i["path"]) for i in plan["items"]]),
                         (True, [("folder", str(self.d.resolve()))]))

    def test_other_models_have_no_needs_line(self):
        self.assertIsNone(M.needs({"exe": "/b/saguaro-glm", "args": ["--glm-pack", "/p"]}))


class Catalog(unittest.TestCase):
    def cat(self, cards, ram=64.0, root="/nonexistent"):
        return {c["id"]: c for c in model_pull.catalog(Path(root), None, cards, ram)}

    def test_the_new_models_are_listed_with_their_sizes(self):
        c = self.cat([])
        self.assertEqual(round(c["hy4-flash-maya-s"]["size_gb"]), 117)
        self.assertEqual(c["hy4-flash-maya-s"]["name"], "Hy4-Flash · Maya-S")
        self.assertEqual({k for k in c if k.startswith("hy4-")},
                         {"hy4-flash-maya-xs", "hy4-flash-maya-s", "hy4-flash-maya-m", "hy4-flash-maya-l"})
        ik = c["inkling-2-mini"]
        self.assertEqual((ik["family"], ik["name"], ik["size_gb"]), ("ik2", "Inkling 2 Mini · unsloth UD-Q4_K_XL", 18.3))
        self.assertIn("one or two NVIDIA cards with 8 GB", ik["needs"])
        self.assertIn("four 16 GB NVIDIA cards", c["hy4-flash-maya-s"]["needs"])
        self.assertEqual((ik["fit"], ik["recommended"]), ("", False))          # (no card read: nothing claimed)
        self.assertNotIn("needs", c["glm-maya-s"])

    def test_one_t4_gets_inkling_recommended(self):
        c = self.cat([t4()])
        ik = c["inkling-2-mini"]
        self.assertTrue(ik["recommended"])                  # (UD-Q4_K_XL: the closest to the original on a T4)
        self.assertFalse(c["inkling-2-mini-sq-15"]["recommended"])
        for part in ("one 16 GB card", "the others in RAM (the offload path)", "32K context", "40 tok/s", "1x Tesla T4"):
            self.assertIn(part, ik["fit"])
        self.assertFalse(c["hy4-flash-maya-s"]["recommended"])
        self.assertIn("Inkling 2 Mini is the model for this PC", c["hy4-flash-maya-s"]["fit"])

    def test_one_rx_9070_xt(self):
        """AMD (the setup's own choice, mortar 2026-11-22): one 16 GB RDNA 4 GPU is Inkling 2 Mini's - the offload
        path, measured; the iGPU beside it is not counted; Hy4-Flash does not run on AMD."""
        xt = {"index": 0, "name": "AMD Radeon RX 9070 XT", "vram_gb": 15.9, "arch": "gfx1201", "vendor": "amd"}
        igpu = {"index": 1, "name": "AMD Radeon (gfx1036)", "vram_gb": 2.0, "arch": "gfx1036", "vendor": "amd"}
        c = self.cat([xt, igpu], 92.0)
        ik = c["inkling-2-mini"]
        self.assertTrue(ik["recommended"])
        for part in ("one AMD GPU", "the offload path", "71 tok/s", "RX 9070 XT"):
            self.assertIn(part, ik["fit"])
        self.assertIn("one AMD RDNA 3 / 4 GPU with 12 GB or more", ik["needs"])
        self.assertNotIn("holds every expert", ik["needs"])
        self.assertIn("NVIDIA GPUs only", c["hy4-flash-maya-s"]["fit"])
        self.assertIn("the offload path", self.cat([dict(xt, vram_gb=12.0)], 92.0)["inkling-2-mini"]["fit"])   # (12 GB on)
        self.assertIn("12 GB", self.cat([dict(xt, vram_gb=8.0)])["inkling-2-mini"]["fit"])

    def test_two_t4s(self):
        c = self.cat([t4(0), t4(1)])
        ik = c["inkling-2-mini"]
        self.assertTrue(ik["recommended"])   # (UD-Q4_K_XL's split: packs on two GPUs not proven yet)
        for part in ("two cards: every expert in VRAM", "131K context", "51 tok/s", "1100 tok/s", "2x Tesla T4"):
            self.assertIn(part, ik["fit"])

    def test_four_t4s(self):
        c = self.cat([t4(i) for i in range(4)], 251.0)
        s = c["hy4-flash-maya-s"]
        self.assertTrue(s["recommended"])
        for part in ("four equal cards", "32 tok/s", "590 tok/s", "4x Tesla T4", "clocks locked"):
            self.assertIn(part, s["fit"])
        self.assertFalse(c["hy4-flash-maya-l"]["recommended"])
        self.assertNotIn("tok/s", c["hy4-flash-maya-l"]["fit"])             # (only Maya-S was measured)
        self.assertFalse(c["inkling-2-mini"]["recommended"])                # (it runs - on two of them - but Hy4 fits)
        self.assertIn("two cards", c["inkling-2-mini"]["fit"])
        self.assertFalse(self.cat([t4(i) for i in range(4)], 2.0)["hy4-flash-maya-s"]["recommended"])   # (no RAM for it)

    def test_a_card_too_small_or_too_old(self):
        self.assertIn("8 GB of VRAM or more", self.cat([t4(0, 6.0)])["inkling-2-mini"]["fit"])
        self.assertIn("the offload path", self.cat([t4(0, 8.0)])["inkling-2-mini"]["fit"])
        p40 = [{"index": 0, "name": "Tesla P40", "vram_gb": 24.0, "arch": "61"}]   # (Pascal: second-class, accepted)
        self.assertEqual(self.cat(p40)["inkling-2-mini"]["fit"], "This PC: one card: every expert in VRAM")
        old = [{"index": 0, "name": "Tesla M40", "vram_gb": 24.0, "arch": "52"}]
        self.assertIn("compute capability 6.0", self.cat(old)["inkling-2-mini"]["fit"])

    def test_installed_and_the_download_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, inst in (("maya-inkling-2-mini.json", {"family": "ik2", "ik2_quant": "ud-q4_k_xl"}),
                               ("maya-hy4-flash-maya-s.json", {"family": "hy4", "hy4_quant": "maya-s"})):
                (Path(tmp) / name).write_text(json.dumps({"exe": "/b/x", "installer": inst}), encoding="utf-8")
            c = self.cat([], root=tmp)
        self.assertTrue(c["inkling-2-mini"]["installed"])
        self.assertTrue(c["hy4-flash-maya-s"]["installed"])
        self.assertFalse(c["hy4-flash-maya-xs"]["installed"])
        cmd = model_pull.setup_command("inkling-2-mini", {"installer": {"models_dir": "/nvme/models"}})
        self.assertEqual(cmd[1:], [str(ROOT / "saguaro.py"), "--setup", "--model", "inkling-2-mini", "--download-model",
                                   "--yes", "--plain", "--no-start", "--models-dir", "/nvme/models"])
        import saguaro as maya
        for ident in ("inkling-2-mini", "hy4-flash-maya-s"):                # (the ids the page sends are the setup's)
            self.assertIsNotNone(maya.Q.resolve(ident), ident)

    def test_a_caller_with_its_own_sizes_reads_no_table_and_no_gpu(self):
        ids = {c["id"] for c in model_pull.catalog(Path("/nonexistent"), {})}   # (the pull's check of a model id)
        self.assertIn("inkling-2-mini", ids)
        self.assertIn("hy4-flash-maya-s", ids)


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
class Page(unittest.TestCase):
    def node(self, body):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.mjs"
            p.write_text(NODE_STUBS + body, encoding="utf-8")
            r = subprocess.run(["node", str(p), (ROOT / "serve" / "web").resolve().as_uri()], capture_output=True,
                               encoding="utf-8", timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_the_engines_lines_are_the_ones_read_here(self):
        """The INFO / STAT lines the tests below feed the page are the formats the engines print."""
        self.assertIn('"INFO engine=hy4 engine_kind=hy4 model=Hy4-flash ctx=%lld vram_gb=%.2f dspark=0 mtp=%d experts=%d "',
                      HY4_SRC)
        self.assertIn('"vram_slots=%d expert_slots=%d\\n"', HY4_SRC)   # (upstream 07415b567: INFO says expert_slots too)
        self.assertIn('"STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f cpu=%.3f"', HY4_SRC)
        self.assertIn('" engram_ms=%.3f\\n"', HY4_SRC)
        ik2 = (ROOT / "src" / "ik2" / "ik2_main.cu").read_text(encoding="utf-8")
        self.assertIn('"INFO engine=ik2 engine_kind=ik2 model=Inkling-2 kv=%s ctx=%d gpus=%d dspark=%d lookup=%d "', ik2)
        self.assertIn('"drafting=%s experts=%d parallel=%d watchdog_s=%d path=%s\\n"', ik2)
        self.assertIn('"STAT tok_s=%.1f ms_tok=%.2f active=1 %s%s\\n"', ik2)
        self.assertIn('"vram_used=%d ram_used=%d vram_gb=%.2f ram_gb=%.2f vram_hit=%.4f ram_fetch=%.2f "', ik2)

    def test_monitor_hy4_experts_card(self):
        # the 4x T4 config's first load: 14678 of 20992 routed experts in VRAM (70%)
        eng = info_of("INFO engine=hy4 engine_kind=hy4 model=Hy4-flash ctx=131072 vram_gb=62.00 dspark=0 mtp=0 "
                      "experts=20992 vram_slots=14678")
        stat = info_of("STAT tok_s=32.48 ms_tok=30.79 vram_hit=0.9460 cpu=0.864 engram_ms=0.412")
        stat = {k: float(v) for k, v in stat.items()}
        got = self.node(f"""
const sv = await import(web + "/server.js"), mon = await import(web + "/monitor.js");
const eng = {json.dumps(eng)}, n = {json.dumps(stat)};
console.log(JSON.stringify({{shares: sv.tierShares(n, eng), set: mon.tierSet(eng).map((x) => [x.key, x.label, n[x.key]]),
  glm: mon.tierSet({{engine_kind: "glm-fast"}}).map((x) => x.key), full: sv.tierShares(n, {{...eng, vram_slots: 30000}})}}));""")
        s = got["shares"]
        self.assertEqual((s["vu"], s["ru"], s["total"], s["disk"]), (14678, 6314, 20992, 0))   # (nothing on the SSD)
        self.assertEqual(got["set"], [["cpu", "On the CPU", 0.864], ["engram_ms", "N-gram memory", 0.412]])
        self.assertEqual(got["glm"], ["ram_fetch", "disk", "promo"])        # (the other engines' card as before)
        self.assertEqual((got["full"]["vu"], got["full"]["ru"]), (20992, 0))

    def test_monitor_qfn_stats_and_context_facts(self):
        """T6b: what the page reads from a QFN engine's passed-through fields - the qfn tier tiles filled by its
        STAT line (promo= from T6a), Speculation named by its INFO mtp=, and the Context card's QFN lines
        (contextFacts from engine.context_ram / engine.slot_memory / context.turn_saves)."""
        self.assertIn("mtp_max=%d mtp=%d lookup=%d vram_free_mib=%lld", QFN_SRC)    # (T6a: INFO says mtp=)
        self.assertIn('"STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f ram_fetch=%.3f disk=0.000 promo=%.3f "',
                      QFN_SRC)                                                      # (T6a: STAT says promo=)
        eng = info_of("INFO engine_kind=qfn experts=13824 vram_slots=16384 vram_gb=50.8 ram_slots=0 ram_gb=63.4")
        stat = {k: float(v) for k, v in info_of(
            "STAT tok_s=21.40 ms_tok=46.73 vram_hit=0.9512 ram_fetch=0.487 disk=0.000 promo=0.031 vram_used=4096 "
            "vram_slots=5120 ram_used=9728 ram_gb=71.2 vram_gb=2.4 disk_gb=0.0 draft_acc=37 draft_off=58 active=1"
        ).items()}
        got = self.node(f"""
const sv = await import(web + "/server.js"), mon = await import(web + "/monitor.js");
const eng = {json.dumps(eng)}, n = {json.dumps(stat)};
console.log(JSON.stringify({{
  set: mon.tierSet(eng).map((x) => [x.key, n[x.key]]),
  shares: sv.tierShares(n, eng),
  spec: sv.speculationView({{engine: {{speculation: ["MTP"]}},
                             requests: [{{drafts_offered: n.draft_off, drafts_accepted: n.draft_acc}}]}}),
  facts: mon.contextFacts(
    {{context_ram: {{pinned: 3221225472, whole: 6442450944, slots: [1073741824, 2147483648]}},
     slot_memory: [{{device: 0, total: 4294967296, free: 1073741824, givable: 536870912}}]}},
    {{turn_saves: {{main: {{tokens: 12345, bytes: 35651584, ms: 8.2, at: Date.now() / 1000 - 42}}}}}}),
  none: mon.contextFacts({{}}, {{}})}}));""")
        self.assertEqual(got["set"], [["promo", 0.031], ["ram_fetch", 0.487], ["disk", 0.0]])
        self.assertEqual((got["shares"]["vu"], got["shares"]["ru"], got["shares"]["total"]), (4096, 9728, 13824))
        self.assertEqual((got["spec"]["state"], got["spec"]["name"], got["spec"]["short"]), ("kept", "MTP", "64%"))
        self.assertEqual(got["none"], [])
        self.assertEqual(got["facts"][0], "3.0 GB of RAM pinned for the context's K/V (6.0 GB if every slot were full)")
        self.assertEqual(got["facts"][1], "context memory: 3.0 of 4.5 GB budgeted on its cards")
        self.assertEqual(got["facts"][2], "last turn save on disk: main 12k tokens, 34 MB 42 s ago")

    def test_monitor_ik2_says_where_its_experts_are(self):
        one = info_of("INFO engine=ik2 engine_kind=ik2 model=Inkling-2 kv=f16 ctx=32768 gpus=1 experts=12288 path=offload")
        two = info_of("INFO engine=ik2 engine_kind=ik2 model=Inkling-2 kv=f16 ctx=131072 gpus=2 experts=12288 path=resident")
        got = self.node(f"""
const mon = await import(web + "/monitor.js");
console.log(JSON.stringify([mon.ik2Experts({json.dumps(one)}), mon.ik2Experts({json.dumps(two)})]));""")
        self.assertIn("fetches the others from pinned RAM", got[0])
        self.assertIn("Every expert is in VRAM", got[1])
        self.assertIn("2 GPUs", got[1])

    def test_thinking_levels_and_names(self):
        got = self.node("""
const st = await import(web + "/state.js"), lib = await import(web + "/lib.js");
const out = {};
for (const model of ["hy4-flash", "Hy4-Flash-Maya-S-IQ2_S", "inkling-2-mini", "Inkling-2-Mini-UD-Q4_K_XL", "glm-5.3-flash"]) {
  st.setFamily(model);
  out[model] = {family: st.familyOf(model), levels: st.LEVELS.map((l) => l.id), key: st.prefsKey ? null : null,
                logo: (lib.modelLogo(model) || {}).src || null, name: lib.modelName(model)};
}
console.log(JSON.stringify(out));""")
        self.assertEqual((got["hy4-flash"]["family"], got["hy4-flash"]["levels"]), ("hy4", ["none", "high"]))
        self.assertEqual(got["Hy4-Flash-Maya-S-IQ2_S"]["family"], "hy4")
        self.assertEqual(got["hy4-flash"]["logo"], "web/logo-hy4.svg")
        self.assertTrue((ROOT / "serve" / "web" / "logo-hy4.svg").is_file())
        self.assertEqual(got["hy4-flash"]["name"], "Hy4 Flash")
        self.assertEqual((got["inkling-2-mini"]["family"], got["inkling-2-mini"]["levels"]),
                         ("ik2", ["none", "low", "medium", "high", "max"]))
        self.assertEqual(got["Inkling-2-Mini-UD-Q4_K_XL"]["family"], "ik2")
        self.assertEqual(got["inkling-2-mini"]["name"], "Inkling 2 Mini")
        self.assertEqual(got["glm-5.3-flash"]["family"], "glm")

    def test_models_page_helpers(self):
        got = self.node("""
const mp = await import(web + "/models.js");
const s = {tp: 4}, opt = (d) => mp.hy4TpOptions(s, d).map((o) => [o.id, o.label]);
console.log(JSON.stringify({
  four: opt({tp: 4, gpus: [0, 1, 2, 3]}), two: opt({tp: "auto", gpus: [0, 1]}),
  notes: mp.dlNotes({needs: "N", fit: "F"}), none: mp.dlNotes({}), noFit: mp.dlNotes({needs: "N", fit: ""}),
  line: mp.needsLine({needs: {cards: 4, measured: true, limits: ["one conversation at a time"]}}),
  one: mp.needsLine({needs: {cards: 1, measured: false, limits: []}}), plain: mp.needsLine({}),
  byHand: mp.needsLine({needs: {cards: 2, measured: null, limits: []}})}));""")
        self.assertEqual(got["byHand"], "Set up for 2 GPUs")
        self.assertEqual(got["four"], [["auto", "Auto"], [0, "Off"], [4, "4"]])
        self.assertEqual(got["two"], [["auto", "Auto"], [0, "Off"], [4, "4"], [2, "2"]])
        self.assertEqual((got["notes"], got["none"], got["noFit"]), (["N", "F"], [], ["N"]))
        self.assertEqual(got["line"], "Set up for 4 GPUs (a layout measured on Tesla T4s) · one conversation at a time")
        self.assertEqual(got["one"], "Set up for 1 GPU (a layout not measured yet)")
        self.assertEqual(got["plain"], "")

    def test_the_qfn_settings_card_renders(self):
        """fieldsOf dispatches family "qfn" to qfnFields (models.js) - the card reads its KV list (QFN_KV) and
        must not throw on a missing declaration (T8a review finding 1: node --check never runs the code)."""
        got = self.node("""
const mp = await import(web + "/models.js");
mp.MS.data = {gpus: []};
const s = {context: 262144, kv: "int8", kv_resident: 32768, parallel: 1, draft: "mtp", images: true,
           reserve_mib: 700, ram_gib: null, expert_cache: null, prefill: null, gpus: [0]};
const m = {family: "qfn", running: false, choices: {draft: ["mtp", "none"], images: true}, settings: s, file: "qfn.json"};
console.log(JSON.stringify(mp.fieldsOf(m, {...s}).map((g) => g.label)));""")
        self.assertEqual(got, ["Model", "GPUs and memory", "Prompts and cache"])

    def test_an_open_page_follows_a_model_switch(self):
        """item 10: the server restarts into another config - the page re-reads the identity (on reconnect, or when
        the metrics name another model than the labels show) and every label changes without a reload; the chat
        list and the open chat stay."""
        got = self.node("""
const mem = {};   // (a localStorage that keeps what the model switch check stores between its calls)
globalThis.localStorage = {getItem: (k) => (k in mem ? mem[k] : null), setItem: (k, v) => { mem[k] = String(v); },
                           removeItem: (k) => { delete mem[k]; }};
const sv = await import(web + "/server.js"), st = await import(web + "/state.js"), lib = await import(web + "/lib.js");
const old = {model: "qwen3.8-flash-next", model_label: "Qwen3.8-Flash-Next · Q8_0 (saguaro-qfn)",
             model_name_pretty: "Qwen3.8 Flash Next", model_quant: "Q8_0"};
const out = {
  same: sv.identityChanged(old, {model: old.model, model_label: old.model_label}),
  otherModel: sv.identityChanged(old, {model: "glm-5.3-flash", model_label: "GLM-5.3 · Maya-L"}),
  otherLabel: sv.identityChanged(old, {model: old.model, model_label: "Qwen3.8-Flash-Next · UD-Q4_K_XL (saguaro-qfn)"}),
  saysNothing: sv.identityChanged(old, {}),
  idOnlyOld: sv.identityChanged({model: "x"}, {model: "x", model_label: null}),
  otherId: sv.identityChanged({model: "x"}, {model: "y", model_label: null}),
};
// the switch: /health is re-read, the labels of every page follow - and what was open stays open
st.S.health = old; st.S.chats = [{id: "c", messages: []}]; st.S.current = "c";
st.newModelCheck();                                          // (the page loaded with this model)
const before = lib.modelLabel(st.S.health, {compact: true});
st.S.health = {model: "glm-5.3-flash", model_label: "GLM-5.3 · Maya-L",
               model_name_pretty: "GLM 5.3", model_quant: "Maya-L"};
const switched = st.newModelCheck();
console.log(JSON.stringify({...out, before, after: lib.modelLabel(st.S.health, {compact: true}),
                            switched, chats: st.S.chats.length, current: st.S.current}));""")
        self.assertFalse(got["same"])
        self.assertTrue(got["otherModel"])
        self.assertTrue(got["otherLabel"])          # (two configs of one model: a quant switch too)
        self.assertFalse(got["saysNothing"])        # (an engine that names nothing: never a switch)
        self.assertFalse(got["idOnlyOld"])
        self.assertTrue(got["otherId"])
        self.assertEqual(got["before"], "Qwen3.8 Flash Next (Q8_0)")
        self.assertEqual(got["after"], "GLM 5.3 (Maya-L)")    # (every label re-reads it from S.health)
        self.assertTrue(got["switched"])            # (its own standard parameters, as at a page load)
        self.assertEqual((got["chats"], got["current"]), (1, "c"))

    def test_a_server_behind_its_api_key_says_so(self):
        """/api/context answering 401: the page knows why the context range is missing - Settings says
        "API key needed", not "this server does not say its context range" (serve/web/server.js' loadCtxInfo)."""
        got = self.node("""
globalThis.fetch = async () => ({status: 401, ok: false, json: async () => ({})});
const sv = await import(web + "/server.js"), st = await import(web + "/state.js");
await sv.loadCtxInfo();
console.log(JSON.stringify({conn: st.S.conn, ctx: st.S.ctxInfo === null ? "null" : "data"}));""")
        self.assertEqual((got["conn"], got["ctx"]), ("key", "null"))

    def test_every_page_module_loads(self):
        got = self.node("""
for (const m of ["monitor", "models", "settings", "turbo", "chat-view", "state", "lib", "server"]) await import(web + "/" + m + ".js");
console.log(JSON.stringify({ok: true}));""")
        self.assertTrue(got["ok"])

    def test_no_upstream_program_names_in_the_page(self):
        for f in sorted((ROOT / "serve" / "web").glob("*.js")):
            text = f.read_text(encoding="utf-8")
            self.assertNotIn("maya-hy4", text, f.name)
            self.assertNotIn("maya-ik2", text, f.name)


if __name__ == "__main__":
    unittest.main()
