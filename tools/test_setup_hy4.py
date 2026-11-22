"""tools/test_setup_hy4.py - Hy4-Flash (saguaro-hy4) in the setup: its sizes' names and setup ids, the pinned table
(each file's place in the repo, size and sha256), the layout for the cards a PC has (hy4_fit.py: four Tesla T4s as
measured, the engine's own choices elsewhere), the size it recommends, a folder laid out as the repo is, the pack step,
the config it writes and a start of it.  No GPU, no network.

    python -m unittest tools.test_setup_hy4 -v
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
import hy4_fit as F  # noqa: E402

TRUNK = "Hy4-Flash-Maya-S-IQ2_S-00001-of-00002.gguf"
NGRAM = "Hy4-Flash-Maya-XS-IQ2_XXS-00002-of-00002.gguf"
MMPROJ = "mmproj-Hy4-Flash-Maya-XS.gguf"


def args(**kw):
    base = dict(model=maya.HY4, hy4_quant=None, mm3_quant=None, ds4_quant=None, model_dir=None, models_dir=None,
                gguf_dir=None, draft=None, backend="cuda", yes=True, check=False, port=None, host=None, api_key=None,
                env=[], no_vision=False, context=None, gpus=None, gpu=None, nvcc=None, repack=False, parallel=None,
                download_model=False)
    base.update(kw)
    return SimpleNamespace(**base)


def t4(index=0, gib=15.0, arch="75", name="Tesla T4"):
    return {"index": index, "name": name, "arch": arch, "vram_gb": float(gib), "driver": "580.0"}


def refusal(fn):
    out = io.StringIO()
    with patch("sys.stdout", new=out), patch.object(maya.S, "UI", None):
        try:
            fn()
        except SystemExit:
            return out.getvalue()
    raise AssertionError("no refusal: " + out.getvalue()[-400:])


class Names(unittest.TestCase):
    def test_sizes_and_ids(self):
        self.assertEqual(maya.HY4_QUANTS, {"maya-xs": "Hy4-Flash-Maya-XS", "maya-s": "Hy4-Flash-Maya-S",
                                           "maya-m": "Hy4-Flash-Maya-M", "maya-l": "Hy4-Flash-Maya-L"})
        self.assertEqual(Q.resolve("hy4-flash-maya-s"), ("hy4", "maya-s"))
        self.assertEqual(Q.resolve("Hy4-Flash-Maya-S"), ("hy4", "maya-s"))
        self.assertEqual(Q.resolve("hy4-flash"), ("hy4", None))
        self.assertEqual(Q.resolve("hy4"), ("hy4", None))
        self.assertEqual(Q.shown("hy4", "maya-s", 117.1), "Hy4-Flash · Maya-S (117 GB)")
        self.assertEqual(Q.config_stem("hy4", "maya-s"), "maya-hy4-flash-maya-s")
        self.assertEqual((maya.HY4_EXE.name, maya.HY4_TARGET, maya.HY4_PACK), ("saguaro-hy4", "maya-hy4", "saguaro-hy4"))
        self.assertIn("OUTPUT_NAME saguaro-hy4", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))

    def test_model_option(self):
        a = args(model="hy4-flash-maya-m")
        self.assertIsNone(maya.model_option(a))
        self.assertEqual((a.model, a.hy4_quant), (maya.HY4, "maya-m"))
        self.assertEqual(maya.choose_family(a, {}), "hy4")
        a = args(model="hy4-flash-maya-s", hy4_quant="maya-l")
        self.assertIn("give one of them", maya.model_option(a))
        a = args(model="hy4-flash")                       # (the model alone: its size asked)
        self.assertIsNone(maya.model_option(a))
        self.assertEqual((a.model, a.hy4_quant), (maya.HY4, None))

    def test_config_label_and_defaults(self):
        cfg = {"exe": "/b/saguaro-hy4", "args": ["--pack", "/p"],
               "installer": {"family": "hy4", "hy4_quant": "maya-s", "quant": "Hy4-Flash-Maya-S"}}
        self.assertEqual(Q.config_label(cfg), "Hy4-Flash · Maya-S")
        self.assertEqual(Q.model_defaults(cfg), ({"temperature": 0.9, "top_p": 1.0}, "high"))

    def test_parallel_is_refused(self):
        with patch.object(maya, "fail", side_effect=SystemExit) as f:
            self.assertRaises(SystemExit, maya.check_parallel, args(parallel=2), "hy4")
        self.assertIn("Hy4-Flash serves one request at a time", f.call_args[0][0])


class Manifest(unittest.TestCase):
    def test_every_size_is_pinned(self):
        man = json.loads(maya.HY4_MANIFEST.read_text(encoding="utf-8"))
        self.assertRegex(man["revision"], r"^[0-9a-f]{40}$")
        for k in maya.HY4_QUANTS:
            self.assertTrue(maya.hy4_published(k), k)
            m = maya.hy4_entry(k)
            self.assertEqual(len(m["files"]), 3, k)                       # (the trunk, the n-gram table, the mmproj)
            for name, sha in m["sha256"].items():
                self.assertRegex(sha, r"^[0-9a-f]{64}$", name)
                self.assertGreater(m["files"][name], 100 << 20, name)
            self.assertEqual(m["ngram"], NGRAM)                           # (one file for every size)
            self.assertTrue(m["plan"].get("need_gib"), k)

    def test_maya_s(self):
        m = maya.hy4_entry("maya-s")
        self.assertEqual((m["trunk"], m["ngram"], m["mmproj"]), (TRUNK, NGRAM, MMPROJ))
        self.assertEqual(m["files"], {TRUNK: 52970756096, NGRAM: 63360616160, MMPROJ: 784524096})
        self.assertAlmostEqual(m["download_gb"], 117.1, places=1)
        self.assertEqual(m["memory_gb"], 64)
        base = f"https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF/resolve/{m['revision']}/"
        self.assertEqual(maya.file_url(m, TRUNK), base + "Maya-S/" + TRUNK)
        self.assertEqual(maya.file_url(m, NGRAM), base + "Maya-XS/" + NGRAM)      # (the table: the XS folder's)
        self.assertEqual(maya.file_url(m, MMPROJ), base + "Maya-XS/" + MMPROJ)
        self.assertNotIn("mmproj", maya.hy4_entry("maya-s", vision=False))
        self.assertEqual(len(maya.hy4_entry("maya-s", vision=False)["files"]), 2)

    def test_a_hub_name_is_not_the_local_one(self):
        self.assertEqual(maya.hy4_local_name("mtp/Hy4-Flash-mtp-BF16.gguf_file"), "Hy4-Flash-mtp-BF16.gguf")
        m = {"repo": "r", "revision": "v", "folder": "", "hub_names": {"a.gguf": "a.gguf_file"}}
        self.assertEqual(maya.file_url(m, "a.gguf"), "https://huggingface.co/r/resolve/v/a.gguf_file")

    def test_a_size_switched_off_or_missing(self):
        man = json.loads(maya.HY4_MANIFEST.read_text(encoding="utf-8"))
        man["quants"]["Hy4-Flash-Maya-L"]["published"] = False
        man["quants"]["Hy4-Flash-Maya-M"]["files"] = {}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "download.json"
            p.write_text(json.dumps(man), encoding="utf-8")
            with patch.object(maya, "HY4_MANIFEST", p):
                self.assertEqual([k for k in maya.HY4_QUANTS if maya.hy4_published(k)], ["maya-xs", "maya-s"])
                out = refusal(lambda: maya.hy4_entry("maya-l"))
                self.assertIn("hy4-flash-maya-xs, hy4-flash-maya-s", out)

    def test_import_catalog_lists_them(self):
        rels = {c["rel"]: c for c in maya.import_catalog(Path("/models"))}
        self.assertEqual(rels["Hy4-Flash-Maya-S/" + TRUNK]["size"], 52970756096)
        self.assertIn("Hy4-Flash-Maya-S/" + NGRAM, rels)
        self.assertIn("Hy4-Flash-Maya-XS/" + NGRAM, rels)


class Fit(unittest.TestCase):
    PLAN = {"chunk": 1024, "reserve_gb": 0.75, "need_gib": {"4096": 8.82, "8192": 9.11, "32768": 9.97}}

    def test_four_t4s_as_measured(self):
        lay = F.layout([t4(i) for i in range(4)], self.PLAN, "maya-s")
        self.assertEqual(lay["args"], ["--tp", "4", "--chunk", "2048", "--reserve-gb", "0.5"])
        self.assertEqual(lay["env"], {"HY4_HOT": "1", "HY4_VISION_F16": "1"})
        self.assertEqual((lay["context"], lay["measured"], lay["fits"]), (131072, True, True))
        self.assertIn("32 tok/s", F.speeds(lay, "maya-s"))
        self.assertIn("590 tok/s", F.speeds(lay, "maya-s"))
        self.assertIn("clocks locked", F.speeds(lay, "maya-s"))
        # (another size on the same cards: the layout, nothing claimed for its speed)
        other = F.layout([t4(i) for i in range(4)], self.PLAN, "maya-m")
        self.assertEqual((other["args"], other["measured"], F.speeds(other, "maya-m")), (lay["args"], False, ""))

    def test_it_matches_the_measured_config(self):
        cfg = json.loads((ROOT / "configs" / "Hy4-Flash-Maya-S-IQ2_S.json").read_text(encoding="utf-8"))
        lay = F.layout([t4(i) for i in range(4)], self.PLAN, "maya-s")
        self.assertEqual(cfg["args"][2:], ["--max-context", str(lay["context"])] + lay["args"])
        for k, v in lay["env"].items():
            self.assertEqual(cfg["env"][k], v, k)

    def test_other_cards_get_the_engines_own_choices(self):
        for cards in ([t4()], [t4(0), t4(1)], [t4(i) for i in range(3)], [t4(i, 24.0, "86", "RTX 3090") for i in range(4)],
                      [t4(0), t4(1), t4(2), t4(3, 24.0, "86", "RTX 3090")]):
            lay = F.layout(cards, self.PLAN, "maya-s")
            self.assertEqual((lay["args"], lay["env"], lay["context"], lay["measured"]), ([], {}, 32768, False), cards)
            self.assertTrue(lay["fits"])
            self.assertEqual(F.speeds(lay, "maya-s"), "")

    def test_what_does_not_fit(self):
        lay = F.layout([t4(0, 8.0, "86", "RTX 3060 Ti")], self.PLAN)
        self.assertFalse(lay["fits"])
        self.assertAlmostEqual(lay["need_gib"], 9.97)
        self.assertFalse(F.layout([], self.PLAN)["fits"])
        self.assertFalse(F.layout([t4(0, 32.0, "61", "Tesla P40")], self.PLAN)["fits"])   # (older than the kernels)

    def test_need_grows_with_the_context(self):
        self.assertAlmostEqual(F.need_gib(self.PLAN, 8192), 9.11)
        self.assertAlmostEqual(F.need_gib(self.PLAN, 16384), 9.11 + 8192 * 16640 / 2**30)
        self.assertGreater(F.need_gib(self.PLAN, 131072), 11.0)
        self.assertEqual(F.need_gib({}, 32768), 0.0)

    def test_the_flags_are_the_engines_own(self):
        src = (ROOT / "src" / "hy4" / "hy4_main.cpp").read_text(encoding="utf-8")
        for flag in F.T4_ARGS[::2] + ["--pack", "--max-context", "--vision", "--cpu-threads", "--kv", "--kv-streaming",
                                      "--main-gpu", "--vram-experts"]:
            self.assertIn(f'a == "{flag}"', src, flag)
        self.assertIn('"HY4_HOT"', (ROOT / "src" / "hy4" / "hy4_model.cu").read_text(encoding="utf-8"))

    def test_recommended_size(self):
        sizes = {"maya-xs": 52.0, "maya-s": 64.0, "maya-m": 76.0, "maya-l": 100.0}
        self.assertEqual(F.recommended(sizes, 60, 251), "maya-s")       # (the size measured, though L would fit)
        self.assertEqual(F.recommended(sizes, 16, 40), "maya-xs")
        self.assertIsNone(F.recommended(sizes, 16, 16))


class Picker(unittest.TestCase):
    def test_size_asked_with_maya_s_recommended(self):
        got = {}

        def pick(title, intro, opts, default, yes, *x):
            got.update(title=title, rows=opts, default=default)
            return default
        with patch.object(maya, "choose", pick), patch.object(maya.S, "gpus", return_value=[t4(i) for i in range(4)]), \
                patch.object(maya, "mem_gb", return_value=(251.0, 200.0)):
            self.assertEqual(maya.choose_hy4_quant(args(), {}), "maya-s")
            self.assertEqual(got["rows"][0][:2], ("Hy4-Flash · Maya-S (117 GB)", "recommended"))
            self.assertIn("measured on 4x Tesla T4", got["rows"][0][2])
            self.assertIn("not measured on Tesla T4s yet", got["rows"][1][2])
            self.assertEqual(len(got["rows"]), 4)
            prev = {"installer": {"family": "hy4", "hy4_quant": "maya-xs"}}
            self.assertEqual(maya.choose_hy4_quant(args(), prev), "maya-xs")     # (the setup before's)
        self.assertEqual(maya.choose_hy4_quant(args(hy4_quant="maya-l"), {}), "maya-l")

    def test_the_models_row(self):
        about = maya.hy4_about(" (this PC: 315 GB)")
        for part in ("Tencent's Hy4-Flash", "four cards", "32 tok/s", "590 tok/s", "clocks locked", "this PC: 315 GB",
                     "LICENSE.hy4-flash", "one conversation"):
            self.assertIn(part, about)

    def test_check_pc_refuses_first(self):
        with patch.object(maya.S, "gpus", side_effect=AssertionError("no GPU read before the refusal")), \
                patch.object(maya, "WIN", False):
            self.assertIn("NVIDIA GPUs only", refusal(lambda: maya.check_pc(args(backend="hip"), "hy4")))
        with patch.object(maya, "WIN", True):
            self.assertIn("Linux only", refusal(lambda: maya.check_pc(args(), "hy4")))


class Steps(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.said = []
        for name in ("ok", "say", "step", "warn"):
            p = patch.object(maya, name, lambda *a, _n=name, **kw: self.said.append((_n, " ".join(map(str, a)))))
            p.start()
            self.addCleanup(p.stop)
        for p in (patch.object(maya, "rotational", lambda p: False), patch.object(maya, "configs", lambda: [])):
            p.start()
            self.addCleanup(p.stop)

    def small(self):
        """The table with small files of the same names: an entry the steps can run on."""
        man = json.loads(maya.HY4_MANIFEST.read_text(encoding="utf-8"))
        import hashlib
        body = {TRUNK: b"t" * 64, NGRAM: b"n" * 96, MMPROJ: b"m" * 32}
        sha = {k: hashlib.sha256(v).hexdigest() for k, v in body.items()}
        man["quants"]["Hy4-Flash-Maya-S"]["files"] = {"Maya-S/" + TRUNK: [64, sha[TRUNK]]}
        man["ngram"] = {"Maya-XS/" + NGRAM: [96, sha[NGRAM]]}
        man["vision"] = {"Maya-XS/" + MMPROJ: [32, sha[MMPROJ]]}
        p = self.root / "download.json"
        p.write_text(json.dumps(man), encoding="utf-8")
        patcher = patch.object(maya, "HY4_MANIFEST", p)
        patcher.start()
        self.addCleanup(patcher.stop)
        return body

    def test_files_in_place_are_checked_not_downloaded(self):
        body = self.small()
        d = self.root / "models" / "Hy4-Flash-Maya-S"
        d.mkdir(parents=True)
        for k, v in body.items():
            (d / k).write_bytes(v)
        with patch.object(maya, "offer_download", side_effect=AssertionError("nothing to download")):
            files = maya.hy4_model_step(args(), d, "maya-s", self.root / "models")
        self.assertEqual(files, {"trunk": d / TRUNK, "ngram": d / NGRAM, "mmproj": d / MMPROJ})
        self.assertTrue((d / maya.VERIFIED).exists())
        (d / TRUNK).write_bytes(b"x" * 64)                              # (the right size, damaged)
        with patch.object(maya, "fail", side_effect=SystemExit) as f:
            self.assertRaises(SystemExit, maya.hy4_model_step, args(), d, "maya-s", self.root / "models")
        self.assertIn("sha256 does not match", f.call_args[0][0])

    def test_a_folder_laid_out_as_the_repo(self):
        """The files where the repo keeps them (Maya-S/, Maya-XS/): read where they are, nothing downloaded."""
        body = self.small()
        d = self.root / "models" / "Hy4-Flash-Maya-S-IQ2_S"
        for k, sub in ((TRUNK, "Maya-S"), (NGRAM, "Maya-XS"), (MMPROJ, "Maya-XS")):
            (d / sub).mkdir(parents=True, exist_ok=True)
            (d / sub / k).write_bytes(body[k])
        with patch.object(maya, "offer_download", side_effect=AssertionError("nothing to download")):
            files = maya.hy4_model_step(args(model_dir=str(d)), d, "maya-s", self.root / "models")
        self.assertEqual(files["trunk"], d / "Maya-S" / TRUNK)
        self.assertEqual(files["ngram"], d / "Maya-XS" / NGRAM)
        seen = []

        def run(cmd, **kw):
            seen.append(cmd)
            pack = Path(cmd[cmd.index("--out") + 1])
            (pack / "tokenizer").mkdir(parents=True)
            for f in maya.HY4_PACK_FILES:
                (pack / f).write_text("vision.dim 1\n" if f == "hy4.txt" else "x", encoding="utf-8")
        with patch.object(maya, "run", run):
            pack = maya.hy4_pack_step(args(), d, files)
        cmd = seen[0]
        self.assertEqual(cmd[cmd.index("--files") + 1:], ["Maya-S/" + TRUNK, "Maya-XS/" + NGRAM])
        self.assertEqual(cmd[cmd.index("--mmproj") + 1], "Maya-XS/" + MMPROJ)
        self.assertEqual(pack, d / "saguaro-hy4")

    def test_the_shared_table_is_linked_from_another_size(self):
        body = self.small()
        models = self.root / "models"
        (models / "Hy4-Flash-Maya-XS").mkdir(parents=True)
        (models / "Hy4-Flash-Maya-XS" / NGRAM).write_bytes(body[NGRAM])
        d = models / "Hy4-Flash-Maya-S"
        d.mkdir()
        (d / TRUNK).write_bytes(body[TRUNK])
        asked = []

        def offer(a, quant, d_, shards, m):
            asked.append([s.name for s in shards if maya.file_incomplete(m, s)])
            return False
        with patch.object(maya, "offer_download", offer):
            self.assertIsNone(maya.hy4_model_step(args(), d, "maya-s", models))
        self.assertEqual(asked, [[MMPROJ]])                             # (only the mmproj is still to download)
        self.assertEqual(os.stat(d / NGRAM).st_ino, os.stat(models / "Hy4-Flash-Maya-XS" / NGRAM).st_ino)

    def test_no_vision_leaves_the_mmproj_out(self):
        body = self.small()
        d = self.root / "models" / "Hy4-Flash-Maya-S"
        d.mkdir(parents=True)
        for k in (TRUNK, NGRAM):
            (d / k).write_bytes(body[k])
        files = maya.hy4_model_step(args(no_vision=True), d, "maya-s", self.root / "models")
        self.assertIsNone(files["mmproj"])

    def pack(self, d, vision=True):
        pack = d / "saguaro-hy4"
        (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
        for f in maya.HY4_PACK_FILES:
            (pack / f).write_text("vision.dim 1152\n" if f == "hy4.txt" and vision else "x", encoding="utf-8")
        return pack

    def test_pack_step(self):
        body = self.small()
        d = self.root / "models" / "Hy4-Flash-Maya-S"
        d.mkdir(parents=True)
        for k, v in body.items():
            (d / k).write_bytes(v)
        files = {"trunk": d / TRUNK, "ngram": d / NGRAM, "mmproj": d / MMPROJ}
        seen = []

        def run(cmd, **kw):
            seen.append(cmd)
            self.pack(d, vision=cmd[cmd.index("--mmproj") + 1] != "none")
        with patch.object(maya, "run", run):
            pack = maya.hy4_pack_step(args(), d, files)
            cmd = seen[0]
            self.assertEqual(cmd[1], str(ROOT / "tools" / "hy4" / "pack_gguf.py"))
            self.assertEqual((cmd[cmd.index("--model") + 1], cmd[cmd.index("--out") + 1]), (str(d), str(pack)))
            self.assertEqual((cmd[cmd.index("--mtp") + 1], cmd[cmd.index("--drafter") + 1]), ("none", "none"))
            self.assertEqual(cmd[cmd.index("--files") + 1:], [TRUNK, NGRAM])
            self.assertTrue(maya.hy4_pack_has_vision(pack))
            maya.hy4_pack_step(args(), d, files)                        # (nothing changed: not built again)
            self.assertEqual(len(seen), 1)
            os.utime(d / TRUNK, ns=(1, 1))                              # (a model file changed: built again)
            maya.hy4_pack_step(args(), d, files)
            self.assertEqual(len(seen), 2)
            maya.hy4_pack_step(args(), d, {**files, "mmproj": None})    # (the vision tower left out: built again)
            self.assertEqual(len(seen), 3)
            self.assertEqual(seen[2][seen[2].index("--mmproj") + 1], "none")

    def write(self, cards, ctx=None, a=None, d=None):
        d = d or self.root / "models" / "Hy4-Flash-Maya-S"
        pack = d / "saguaro-hy4"
        if not pack.exists():
            self.pack(d)
        pc = {"gpus": cards, "backend": "cuda"}
        lay = F.layout(cards, Fit.PLAN, "maya-s")
        with patch.object(maya, "ROOT", self.root), patch.object(maya.S, "cpu_info", return_value=("Xeon", False, False)), \
                patch.object(maya, "write_run_script", lambda p, port: p.with_name("run-" + p.stem + ".sh")):
            path = maya.write_config_hy4(a or args(), pc, {}, pack, ctx or lay["context"], self.root / "models", d,
                                         "maya-s", lay)
        return path, json.loads(path.read_text(encoding="utf-8")), pack

    def test_the_config_on_four_t4s(self):
        path, cfg, pack = self.write([t4(i) for i in range(4)])
        self.assertEqual(path.name, "maya-hy4-flash-maya-s.json")
        self.assertEqual(cfg["exe"], str(maya.HY4_EXE))
        self.assertEqual(cfg["args"], ["--pack", str(pack), "--max-context", "131072", "--tp", "4", "--chunk", "2048",
                                       "--reserve-gb", "0.5"])
        self.assertEqual(cfg["env"], {"HY4_HOT": "1", "HY4_VISION_F16": "1", "HY4_Q4X": "0"})      # (no AVX2 here: the 4-bit copies off)
        self.assertEqual((cfg["gpu"], cfg["tokenizer"], cfg["model_name"]),
                         ([0, 1, 2, 3], str(pack / "tokenizer"), "hy4-flash"))
        self.assertEqual((cfg["sampling"], cfg["reasoning_effort"]), ({"temperature": 0.9, "top_p": 1.0}, "high"))
        inst = cfg["installer"]
        self.assertEqual((inst["family"], inst["hy4_quant"], inst["quant"], inst["measured"]),
                         ("hy4", "maya-s", "Hy4-Flash-Maya-S", True))
        self.assertEqual((Q.family_of(cfg), Q.config_key(cfg)), ("hy4", "maya-s"))

    def test_the_servers_learned_scores_are_this_folders(self):
        """The folder the setup downloads Maya-S into is the one the server looks the shipped scores up by."""
        self.assertTrue((ROOT / "data" / f"hy4-expert-hot-{maya.HY4_QUANTS['maya-s']}.bin").is_file())

    def test_the_config_keeps_and_follows(self):
        path, cfg, pack = self.write([t4(i) for i in range(4)])
        cfg["args"] += ["--cpu-threads", "12"]
        cfg["args"][cfg["args"].index("--chunk") + 1] = "1024"
        cfg["env"]["MY_VAR"] = "1"
        cfg["host"] = "0.0.0.0"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        path, cfg, _ = self.write([t4(i) for i in range(4)])               # (the same cards: the hand's values stay)
        self.assertEqual(cfg["args"][cfg["args"].index("--chunk") + 1], "1024")
        self.assertEqual(cfg["args"][cfg["args"].index("--cpu-threads") + 1], "12")
        self.assertEqual((cfg["env"]["MY_VAR"], cfg["host"]), ("1", "0.0.0.0"))
        path, cfg, _ = self.write([t4(0), t4(1)])                           # (two cards now: no tensor-parallel flags)
        self.assertNotIn("--tp", cfg["args"])
        self.assertNotIn("--chunk", cfg["args"])
        self.assertEqual(cfg["args"][cfg["args"].index("--cpu-threads") + 1], "12")
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "32768")
        self.assertFalse(cfg["installer"]["measured"])

    def test_no_vision_writes_the_flag(self):
        _, cfg, _ = self.write([t4(i) for i in range(4)], a=args(no_vision=True))
        self.assertEqual(cfg["args"][-2:], ["--vision", "off"])

    def test_a_start_of_it(self):
        path, cfg, pack = self.write([t4(i) for i in range(4)])
        exe = self.root / "saguaro-hy4"
        exe.touch()
        cfg["exe"] = str(exe)
        path.write_text(json.dumps(cfg), encoding="utf-8")
        a = SimpleNamespace(parallel=None, port=None, host=None, api_key=None, gpus=None, gpu=None, yes=True)
        for n in ("select_build_backend", "refresh_engine", "rename_programs"):
            p = patch.object(maya, n, return_value=None)
            p.start()
            self.addCleanup(p.stop)
        with patch.object(maya, "speed_calibration", side_effect=AssertionError("no calibration")), \
                patch.object(maya, "prefill_calibration", side_effect=AssertionError("no calibration")), \
                patch.object(maya.S, "gpus", return_value=[t4(i) for i in range(4)]), \
                patch.dict(os.environ, {"INVOCATION_ID": "x"}):
            cmd, env, info = maya.server_command(path, a)
        self.assertIn("serve", " ".join(cmd))
        self.assertEqual((info["model"], info["label"]), ("hy4-flash", "Hy4-Flash · Maya-S"))

    def test_nothing_to_tune_and_its_target(self):
        cfg = {"exe": str(maya.HY4_EXE), "args": ["--pack", "/p"], "installer": {"family": "hy4"}}
        self.assertFalse(maya.needs_speed_calibration(cfg))
        path = self.root / "c.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        self.assertFalse(maya.calibrate_config(path))
        self.assertEqual(maya.exe_for_target(maya.HY4_TARGET), maya.HY4_EXE)
        self.assertEqual(maya.bench_exe(cfg), maya.HY4_EXE)

    def test_set_up_refuses_what_cannot_hold_it_and_names_inkling(self):
        self.small()
        pc = {"gpus": [t4(0, 8.0, "86", "RTX 3060 Ti")], "backend": "cuda", "archs": [86]}
        with patch.object(maya, "check_pc", lambda a, fam: pc), patch.object(maya, "mem_gb", return_value=(64.0, 60.0)), \
                patch.object(maya, "fail", side_effect=SystemExit) as f:
            self.assertRaises(SystemExit, maya.set_up_hy4, args(hy4_quant="maya-s"), {})
        self.assertIn("no RAM fallback", f.call_args[0][0])
        self.assertIn("--model inkling-2-mini", f.call_args[0][1])

    def test_set_up_runs_its_steps_and_says_the_clock_tip(self):
        body = self.small()
        models = self.root / "models"
        d = models / "Hy4-Flash-Maya-S"
        d.mkdir(parents=True)
        for k, v in body.items():
            (d / k).write_bytes(v)
        pack = self.pack(d)
        pc = {"gpus": [t4(i) for i in range(4)], "backend": "cuda", "archs": [75]}
        seen = {}

        def write(a, pc_, meta, pack_, ctx, models_, d_, quant, lay):
            seen.update(ctx=ctx, lay=lay, quant=quant, pack=pack_)
            return self.root / "cfg.json"
        with patch.object(maya, "check_pc", lambda a, fam: pc), patch.object(maya, "mem_gb", return_value=(251.0, 200.0)), \
                patch.object(maya, "tools_step", lambda a: self.root), \
                patch.object(maya, "build_step", lambda a, pc_, llama, target: seen.update(target=target) or {}), \
                patch.object(maya, "HY4_EXE", d / TRUNK), patch.object(maya, "hy4_pack_step", lambda a, d_, f: pack), \
                patch.object(maya, "write_config_hy4", write):
            got = maya.set_up_hy4(args(hy4_quant="maya-s", models_dir=str(models)), {})
        self.assertEqual(got, self.root / "cfg.json")
        self.assertEqual((seen["ctx"], seen["quant"], seen["target"], seen["lay"]["args"][:2]),
                         (131072, "maya-s", "maya-hy4", ["--tp", "4"]))
        text = "\n".join(t for _, t in self.said)
        self.assertIn("nvidia-smi -lgc 1590,1590", text)
        self.assertIn("32 tok/s", text)



class HotSeed(unittest.TestCase):
    """The HY4_HOT scores shipped (data/hy4-hot-seed, a file a size, upstream v1.1.75): HY4_HOT_SEED where there is
    one - a layout's folder takes its size's; none for a model without one."""

    def test_seed(self):
        import serve.server as SV
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            seeds = root / "seeds"
            seeds.mkdir()
            (seeds / "hy4-expert-hot-Hy4-Flash-Maya-XS.bin").write_bytes(b"HY4HOT1\0")
            cfg = lambda folder: {"exe": "saguaro-hy4", "args": ["--pack", str(root / folder / "saguaro-hy4")]}   # noqa: E731
            with patch.object(SV, "HY4_HOT_SEEDS", seeds), patch("serve.server.models.family", return_value="hy4"):
                self.assertEqual(SV.child_env(cfg("Hy4-Flash-Maya-XS"))["HY4_HOT_SEED"],
                                 str(seeds / "hy4-expert-hot-Hy4-Flash-Maya-XS.bin"))
                self.assertEqual(SV.hy4_hot_seed(cfg("Hy4-Flash-Maya-XS-8GB")).name, "hy4-expert-hot-Hy4-Flash-Maya-XS.bin")
                self.assertIsNone(SV.hy4_hot_seed(cfg("Hy4-Flash-Maya-S")))
                self.assertNotIn("HY4_HOT_SEED", SV.child_env(cfg("Other")))

if __name__ == "__main__":
    unittest.main()
