"""tools/test_setup_amd.py - the setup on an AMD PC: which models it offers and recommends (family_fit,
family_unavailable, choose_family), the GLM quant it recommends for the PC's VRAM + RAM, the refusal for a model with
no AMD engine, and the plain-text check line.  Also: no silent defaults without a terminal, the models folder and the
calibration store under Saguaro's names with the old ones read as before.  No GPU, no network.

    python -m unittest tools.test_setup_amd -v
"""
from pathlib import Path
import io
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import saguaro as maya  # noqa: E402
import setup as S  # noqa: E402


def rx9070(index=0, gb=16.0):
    return {"index": index, "name": "AMD Radeon RX 9070 / AI PRO R9700 (gfx1201)", "vram_gb": gb, "arch": "gfx1201",
            "driver": "amdgpu", "vendor": "amd"}


IGPU = {"index": 1, "name": "AMD Radeon (gfx1036)", "vram_gb": 2.0, "arch": "gfx1036", "driver": "amdgpu", "vendor": "amd"}


def t4(index=0, gib=15.0):
    return {"index": index, "name": "Tesla T4", "arch": "75", "vram_gb": gib, "driver": "580.0"}


def args(**kw):
    base = dict(model=None, model_dir=None, gguf_dir=None, draft=None, backend="hip", yes=True, check=False)
    base.update(kw)
    return SimpleNamespace(**base)


def output(fn):
    """What fn printed, and whether it stopped (SystemExit)."""
    out = io.StringIO()
    with patch("sys.stdout", new=out), patch.object(maya.S, "UI", None):
        try:
            fn()
        except SystemExit:
            return out.getvalue(), True
    return out.getvalue(), False


class Picker(unittest.TestCase):
    def pick(self, cards, ram=92.0, backend="hip", answer=None):
        got = {}

        def choose(title, intro, rows, default, yes, *x):
            got.update(rows=rows, default=default)
            return default if answer is None else maya.FAMILIES.index(answer)
        with patch.object(maya, "choose", choose), patch.object(maya.S, "amd_gpus", return_value=cards), \
                patch.object(maya.S, "gpus", return_value=cards if backend == "cuda" else []), \
                patch.object(maya, "mem_gb", return_value=(ram, ram - 5)), patch.object(maya, "WIN", False):
            fam = maya.choose_family(args(backend=backend), {})
        return fam, got

    def test_amd_gets_a_recommendation(self):
        fam, got = self.pick([rx9070(), IGPU])
        self.assertEqual(fam, "glm")
        self.assertEqual(got["rows"][maya.FAMILIES.index("glm")][1], "recommended for this PC")

    def test_models_without_an_amd_engine_are_marked(self):
        _, got = self.pick([rx9070()])
        for fam in ("hy4", "ik2"):
            label, note, about = got["rows"][maya.FAMILIES.index(fam)]
            self.assertEqual(note, "not on this PC's GPU yet")
            self.assertIn("NVIDIA GPUs only", about)
        for fam in ("glm", "qwen"):
            self.assertNotEqual(got["rows"][maya.FAMILIES.index(fam)][1], "not on this PC's GPU yet")

    def test_picking_one_stops_plainly(self):
        text, stopped = output(lambda: self.pick([rx9070()], answer="ik2"))
        self.assertTrue(stopped)
        self.assertIn("not supported on this PC's GPU yet", text)
        self.assertNotIn("Fix the item above", text)

    def test_nvidia_unchanged(self):
        self.assertEqual(self.pick([t4(0), t4(1)], 251.0, "cuda")[0], "ik2")
        self.assertEqual(self.pick([t4(i) for i in range(4)], 251.0, "cuda")[0], "hy4")

    def test_family_fit(self):
        with patch.object(maya, "mem_gb", return_value=(92.0, 86.0)), patch.object(maya, "WIN", False):
            self.assertEqual(maya.family_fit([rx9070()], 92.0, "hip"), "glm")
            self.assertIsNone(maya.family_fit([rx9070()], 32.0, "hip"))      # (no GLM quant fits 48 GB)
            self.assertIsNone(maya.family_fit([], 92.0, "hip"))
            self.assertEqual(maya.family_fit([t4()], 92.0), "ik2")

    def test_unavailable(self):
        with patch.object(maya, "WIN", False):
            self.assertIsNone(maya.family_unavailable("glm", "hip"))
            self.assertIn("NVIDIA GPUs only", maya.family_unavailable("hy4", "hip"))
            self.assertIn("NVIDIA GPUs only", maya.family_unavailable("ik2", "hip"))
            self.assertIsNone(maya.family_unavailable("hy4", "cuda"))
            self.assertIn("GLM-5.3-Flash only", maya.family_unavailable("deepseek", "metal"))
        with patch.object(maya, "WIN", True):
            self.assertIn("Linux only", maya.family_unavailable("hy4", "cuda"))

    def test_check_pc_refusal_is_not_a_fix_list(self):
        a = SimpleNamespace(backend="hip", gpu=None, gpus=None, check=True, yes=True)
        with patch.object(maya.S, "amd_gpus", side_effect=AssertionError("refused before any GPU is read")), \
                patch.object(maya, "WIN", False):
            text, stopped = output(lambda: maya.check_pc(a, "hy4"))
        self.assertTrue(stopped)
        self.assertIn("NVIDIA GPUs only", text)
        self.assertIn("not supported on this PC's GPU yet", text)
        self.assertNotIn("Fix the item above", text)


class GlmQuant(unittest.TestCase):
    def quant(self, backend, cards, ram):
        a = SimpleNamespace(gguf_dir=None, model=None, model_dir=None, yes=True, backend=backend)
        with tempfile.TemporaryDirectory() as d, patch.object(maya.S, "amd_gpus", return_value=cards), \
                patch.object(maya.S, "gpus", return_value=cards if backend == "cuda" else []), \
                patch.object(maya, "mem_gb", return_value=(ram, ram - 5)), patch.object(maya, "say"):
            return maya.choose_model(a, Path(d), {})[1]

    def test_amd_16gb_card_gets_the_quant_that_fits(self):
        self.assertEqual(self.quant("hip", [rx9070(), IGPU], 92.0), "Maya-XS-IQ1_S")

    def test_amd_24gb_card_with_room_gets_s24(self):
        self.assertEqual(self.quant("hip", [rx9070(0, 24.0)], 192.0), "Maya-S24")

    def test_nvidia_with_room_unchanged(self):
        self.assertEqual(self.quant("cuda", [t4(0, 24.0)], 251.0), "Maya-S24")
        self.assertEqual(self.quant("cuda", [t4(0, 32.0)], 251.0), "Maya-S-v2-IQ2_XXS")


class Terminal(unittest.TestCase):
    def test_no_answer_without_a_terminal_stops(self):
        with patch("builtins.input", side_effect=EOFError):
            text, stopped = output(lambda: S.ask("Download them now?", ["y", "n"], "y", False))
        self.assertTrue(stopped)
        self.assertIn("--yes", text)

    def test_yes_takes_the_default(self):
        with patch("builtins.input", side_effect=AssertionError("no question with --yes")):
            self.assertEqual(S.ask("Download them now?", ["y", "n"], "y", True), "y")


class Names(unittest.TestCase):
    def test_models_folder(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "saguaro"
            root.mkdir()
            with patch.object(maya, "ROOT", root):
                self.assertEqual(maya.default_models_dir(), Path(d) / "Saguaro-data" / "models")
                (Path(d) / "Maya-data" / "models").mkdir(parents=True)      # (an earlier setup's)
                self.assertEqual(maya.default_models_dir(), Path(d) / "Maya-data" / "models")
                (Path(d) / "Saguaro-data" / "models").mkdir(parents=True)
                self.assertEqual(maya.default_models_dir(), Path(d) / "Saguaro-data" / "models")

    def test_calibrations_read_from_the_old_place_until_saved(self):
        with tempfile.TemporaryDirectory() as d:
            new, old = Path(d) / "saguaro" / "calibration.json", Path(d) / "project-maya" / "calibration.json"
            old.parent.mkdir()
            old.write_text(json.dumps({"k": 1}))
            with patch.object(maya, "CAL_STORE", new):
                self.assertEqual(maya.load_calibrations(), {"k": 1})
                new.parent.mkdir()
                new.write_text(json.dumps({"k": 2}))
                self.assertEqual(maya.load_calibrations(), {"k": 2})


if __name__ == "__main__":
    unittest.main()
