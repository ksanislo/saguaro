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
                env=[], no_vision=False, context=None, gpus=None, gpu=None, nvcc=None, repack=False, parallel=None)
    base.update(kw)
    return SimpleNamespace(**base)


def t4(index=0, gib=15.0, arch="75", name="Tesla T4"):
    return {"index": index, "name": name, "arch": arch, "vram_gb": float(gib), "driver": "580.0"}


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

    def test_import_catalog_lists_it(self):
        cat = maya.import_catalog(Path("/models"))
        rels = {c["rel"]: c for c in cat}
        self.assertEqual(rels["Inkling-2-Mini/Inkling-2-Mini-UD-Q4_K_XL.gguf"]["size"], 17629759456)
        self.assertIn("Inkling-2-Mini/mmproj-BF16.gguf", rels)


class Fit(unittest.TestCase):
    def test_one_t4(self):
        lay = F.layout([t4()])
        self.assertEqual([g["index"] for g in lay["gpus"]], [0])
        self.assertEqual(lay["args"], ["--gpus", "1", "--chunk", "2048", "--threads", "8"])
        self.assertEqual((lay["context"], lay["resident"], lay["measured"]), (32768, False, True))
        self.assertEqual(lay["env"], {"SAGUARO_IK2_ROOM_MB": "1024", "SAGUARO_IK2_PIN": "0"})
        self.assertIn("42 tok/s", F.speeds(lay))
        self.assertIn("1x Tesla T4", F.speeds(lay))

    def test_two_t4s(self):
        lay = F.layout([t4(2), t4(3)])
        self.assertEqual([g["index"] for g in lay["gpus"]], [2, 3])
        self.assertEqual(lay["args"], ["--gpus", "2", "--chunk", "8192", "--threads", "8"])
        self.assertEqual((lay["context"], lay["resident"], lay["measured"]), (131072, True, True))
        self.assertIn("52 tok/s", F.speeds(lay))

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

    def test_the_flags_are_the_engines_own(self):
        """saguaro-ik2 stops at a flag it does not know: every flag a layout writes is one its parser reads."""
        src = (ROOT / "src" / "ik2" / "ik2_main.cu").read_text(encoding="utf-8")
        for cards in ([t4()], [t4(0), t4(1)], [t4(0, 24.0, "86")]):
            lay = F.layout(cards)
            for flag in lay["args"][::2]:
                self.assertIn(f'a == "{flag}"', src, flag)
            for key in lay["env"]:
                self.assertIn(f'"{key}"', src, key)

    def test_refusals(self):
        cases = [("hip", False, [t4()], "NVIDIA GPUs only"), ("metal", False, [], "NVIDIA GPUs only"),
                 ("cuda", True, [t4()], "Linux only"),
                 ("cuda", False, [t4(0, 32, "70", "Tesla V100")], "compute capability 7.5"),
                 ("cuda", False, [t4(0, 12, "86", "RTX 3060")], "16 GB of VRAM or more")]
        for backend, win, cards, part in cases:
            why = F.unsupported(backend, win, cards)
            self.assertIn(part, why or "", (backend, win))
            self.assertNotIn("\n", why)
        for cards in ([t4()], [t4(0, 12, "86"), t4(1)], []):
            self.assertIsNone(F.unsupported("cuda", False, cards))
        self.assertIsNone(F.layout([t4(0, 12, "86")]))


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
            for part in ("one or two", "16 GB", "42 tok/s", "52 tok/s", "Apache 2.0", "one conversation"):
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
            self.assertIn("16 GB of VRAM", refusal(lambda: maya.choose_gpus(args(), [t4(0, 8, "86")], F.MIN_CC, "ik2")))

    def test_check_pc_refuses_first(self):
        with patch.object(maya.S, "gpus", side_effect=AssertionError("no GPU read before the refusal")), \
                patch.object(maya, "WIN", False):
            self.assertIn("NVIDIA GPUs only", refusal(lambda: maya.check_pc(args(backend="hip"), "ik2")))
        with patch.object(maya, "WIN", True):
            self.assertIn("Linux only", refusal(lambda: maya.check_pc(args(), "ik2")))


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
        path, cfg = self.write({"gpus": [t4(2)], "backend": "cuda"}, gguf, tok, 32768, d)
        self.assertEqual(path.name, "maya-inkling-2-mini.json")
        self.assertEqual(cfg["exe"], str(maya.IK2_EXE))
        self.assertEqual(cfg["args"], ["--model", str(gguf), "--max-context", "32768", "--gpus", "1", "--chunk", "2048",
                                       "--threads", "8"])
        self.assertEqual(cfg["env"], {"SAGUARO_IK2_ROOM_MB": "1024", "SAGUARO_IK2_PIN": "0"})
        self.assertEqual((cfg["sampling"], cfg["reasoning_effort"]), ({"temperature": 1.0, "top_p": 1.0}, "high"))
        self.assertEqual((cfg["model_name"], cfg["gpu"], cfg["tokenizer"]), ("inkling-2-mini", [2], str(tok)))
        inst = cfg["installer"]
        self.assertEqual((inst["family"], inst["ik2_quant"], inst["cards"], inst["measured"]),
                         ("ik2", Q.IK2_QUANT, 1, True))
        self.assertEqual(Q.family_of(cfg), "ik2")
        self.assertNotIn("vision", cfg)

    def test_the_config_follows_the_cards(self):
        """Two cards after one: the flags are the two-card layout's (a kept --chunk would be the one card's); the
        user's own variable and the server's settings stay."""
        d, gguf = self.files()
        tok = self.tok(d)
        path, cfg = self.write({"gpus": [t4(2)], "backend": "cuda"}, gguf, tok, 32768, d)
        cfg["env"]["MY_VAR"] = "1"
        cfg["api_key"] = "k"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        v = {"exe": "/b/vision-glm", "mmproj": str(d / "mmproj-BF16.gguf"), "gpu": True}
        path, cfg = self.write({"gpus": [t4(2), t4(3)], "backend": "cuda"}, gguf, tok, 131072, d, v)
        self.assertEqual(cfg["args"], ["--model", str(gguf), "--max-context", "131072", "--gpus", "2", "--chunk", "8192",
                                       "--threads", "8"])
        self.assertEqual(cfg["env"], {"SAGUARO_IK2_ROOM_MB": "1024", "MY_VAR": "1"})
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

        def write(a, pc_, meta, gguf_, tok_, ctx, models, d_, lay, vision=None):
            seen.update(ctx=ctx, lay=lay, vision=vision, d=d_)
            return self.root / "cfg.json"
        with patch.object(maya, "check_pc", lambda a, fam: pc), patch.object(maya, "tools_step", lambda a: self.root), \
                patch.object(maya, "build_step", lambda a, pc_, llama, target: {"target": target}), \
                patch.object(maya, "IK2_EXE", gguf), patch.object(maya, "ik2_model_step", lambda a, d_: gguf), \
                patch.object(maya, "ik2_tokenizer_step", lambda a, d_, g: tok), \
                patch.object(maya, "ik2_vision_step", lambda *x: None), patch.object(maya, "write_config_ik2", write):
            got = maya.set_up_ik2(args(models_dir=str(self.root / "models")), {})
        self.assertEqual(got, self.root / "cfg.json")
        self.assertEqual((seen["ctx"], seen["lay"]["args"][:2], seen["d"]), (131072, ["--gpus", "2"], d))


if __name__ == "__main__":
    unittest.main()
