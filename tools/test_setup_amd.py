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
        self.assertEqual(fam, "ik2")   # (one 16 GB RDNA card: Inkling 2 Mini, measured on an RX 9070 XT)
        self.assertEqual(got["rows"][maya.FAMILIES.index("ik2")][1], "recommended for this PC")

    def test_models_without_an_amd_engine_are_marked(self):
        _, got = self.pick([rx9070()])
        for fam in ("hy4",):
            label, note, about = got["rows"][maya.FAMILIES.index(fam)]
            self.assertEqual(note, "not on this PC's GPU yet")
            self.assertIn("NVIDIA GPUs only", about)
        for fam in ("glm", "qfn", "ik2"):
            self.assertNotEqual(got["rows"][maya.FAMILIES.index(fam)][1], "not on this PC's GPU yet")

    def test_picking_one_stops_plainly(self):
        text, stopped = output(lambda: self.pick([rx9070()], answer="hy4"))
        self.assertTrue(stopped)
        self.assertIn("not supported on this PC's GPU yet", text)
        self.assertNotIn("Fix the item above", text)

    def test_nvidia_unchanged(self):
        self.assertEqual(self.pick([t4(0), t4(1)], 251.0, "cuda")[0], "ik2")
        self.assertEqual(self.pick([t4(i) for i in range(4)], 251.0, "cuda")[0], "hy4")

    def test_family_fit(self):
        with patch.object(maya, "mem_gb", return_value=(92.0, 86.0)), patch.object(maya, "WIN", False):
            self.assertEqual(maya.family_fit([rx9070()], 92.0, "hip"), "ik2")
            self.assertEqual(maya.family_fit([rx9070(), rx9070(1)], 92.0, "hip"), "glm")   # (Inkling: one AMD GPU)
            self.assertIsNone(maya.family_fit([], 92.0, "hip"))
            self.assertEqual(maya.family_fit([t4()], 92.0), "ik2")

    def test_unavailable(self):
        with patch.object(maya, "WIN", False):
            self.assertIsNone(maya.family_unavailable("glm", "hip"))
            self.assertIn("NVIDIA GPUs only", maya.family_unavailable("hy4", "hip"))
            self.assertIsNone(maya.family_unavailable("ik2", "hip"))
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


def sysfs_card(root: Path, connectors: dict, used_mib: int, device="0x7550", revision="0xc0") -> Path:
    """A fake /sys/class/drm/renderD*/device: the card's DRM connectors, its VRAM in use and its PCI IDs."""
    dev = root / "device"
    for name, status in connectors.items():
        (dev / "drm" / "card0" / f"card0-{name}").mkdir(parents=True)
        (dev / "drm" / "card0" / f"card0-{name}" / "status").write_text(status + "\n")
    dev.mkdir(parents=True, exist_ok=True)
    (dev / "mem_info_vram_used").write_text(str(used_mib << 20))
    (dev / "device").write_text(device + "\n")
    (dev / "revision").write_text(revision + "\n")
    return dev


class Reserve(unittest.TestCase):
    """The VRAM an engine leaves free on a card - AMD's and NVIDIA's alike, v1.2.1-A: 3072 MiB where the desktop may
    run on it, 512 on a card with no display connected (AMD's sysfs: and next to no VRAM in use - the RX 9070 XT beside
    an iGPU: Maya-L read prompts token by token with 3072; NVIDIA's display_active says it outright), 1024 on
    a Linux AMD APU.  v1.2.1-D: a headless card keeps 1024 when every card is Pascal sm_6x (the P40 warm-up fell to
    264 MiB free with 512)."""

    def card(self, connectors, used_mib):
        with tempfile.TemporaryDirectory() as d:
            return {**rx9070(), **S.drm_display(sysfs_card(Path(d), connectors, used_mib))}

    def test_headless_card(self):
        g = self.card({"DP-1": "disconnected", "HDMI-A-1": "disconnected", "Writeback-1": "unknown"}, 93)
        self.assertEqual((g["display"], round(g["vram_used_gb"], 2)), (False, 0.09))
        with patch.object(maya, "WIN", False):
            self.assertEqual(maya.gpu_reserve_mib([g]), 512)
            self.assertIn("no display", maya.gpu_reserve_why([g]))

    def test_a_connected_display_keeps_3072(self):
        g = self.card({"DP-1": "connected", "HDMI-A-1": "disconnected"}, 93)
        with patch.object(maya, "WIN", False):
            self.assertEqual(maya.gpu_reserve_mib([g]), 3072)

    def test_vram_in_use_keeps_3072(self):   # (a compositor drawing on the card through another GPU's outputs)
        g = self.card({"DP-1": "disconnected"}, 900)
        with patch.object(maya, "WIN", False):
            self.assertEqual(maya.gpu_reserve_mib([g]), 3072)

    def test_unknown_keeps_3072(self):
        with patch.object(maya, "WIN", False):
            self.assertEqual(maya.gpu_reserve_mib([rx9070()]), 3072)          # (no sysfs facts)
            self.assertEqual(maya.gpu_reserve_mib([{**rx9070(), "display": False, "vram_used_gb": 0.1},
                                                   {**rx9070(1), "display": True, "vram_used_gb": 0.1}]), 3072)
        with patch.object(maya, "WIN", True):
            self.assertEqual(maya.gpu_reserve_mib([rx9070()]), 3072)          # (Windows' AMD: no sysfs, unknown too)

    def test_linux_apu_unchanged(self):
        with patch.object(maya, "WIN", False):
            self.assertEqual(maya.gpu_reserve_mib([{"arch": "gfx1151", "display": True, "vram_gb": 96}]), 1024)

    def test_nvidia_display_active(self):
        """v1.2.1-A: the NVIDIA fact (setup.py's gpus() from nvidia-smi's display_active) - Disabled is headless,
        on Windows too (sysfs' absence is not a question there); Enabled, and unknown - "[N/A]" / "[Not Supported]"
        / an old nvidia-smi, which leave no "display" key - keep the desktop's room."""
        t4 = {"name": "Tesla T4", "vram_gb": 15.0, "arch": "75"}
        for win in (False, True):
            with self.subTest(WIN=win), patch.object(maya, "WIN", win):
                self.assertEqual(maya.gpu_reserve_mib([{**t4, "display": False}]), 512)
                self.assertIn("no display", maya.gpu_reserve_why([{**t4, "display": False}]))
        with patch.object(maya, "WIN", False):
            self.assertEqual(maya.gpu_reserve_mib([{**t4, "display": True}]), 3072)
            self.assertEqual(maya.gpu_reserve_mib([t4]), 3072)               # (no fact: unknown)
            self.assertEqual(maya.gpu_reserve_mib([{**t4, "display": False},
                                                   {**t4, "index": 1, "display": True}]), 3072)

    def test_headless_pascal_keeps_1024(self):
        """v1.2.1-D (the P40 batch): on all-sm_6x headless cards the reserve goes 512 -> 1024 - the warm-up graph
        captures fell to 264-302 MiB free with 512 and held above 760 with 1024.  The display rule is unchanged
        (a Pascal card with a desktop keeps 3072), a mixed list keeps the 512, and --vram-reserve-mib still wins
        above the rule (test_setup_qfn's ReserveRule and the config's own line)."""
        p40 = {"name": "Tesla P40", "vram_gb": 24.0, "arch": "61", "display": False}
        p100 = {"name": "Tesla P100", "vram_gb": 16.0, "arch": "60", "display": False}
        t4 = {"name": "Tesla T4", "vram_gb": 15.0, "arch": "75", "display": False}
        for win in (False, True):
            with self.subTest(WIN=win), patch.object(maya, "WIN", win):
                self.assertEqual(maya.gpu_reserve_mib([p40]), 1024)
                self.assertEqual(maya.gpu_reserve_mib([p40, p100]), 1024)     # (every card sm_6x)
                why = maya.gpu_reserve_why([p40])
                self.assertIn("no display", why)                              # (the rule still said ...)
                self.assertIn("264", why)                                     # ( ... with its measurement)
                self.assertEqual(maya.gpu_reserve_mib([p40, t4]), 512)        # (mixed kinds: the 512 as before)
        with patch.object(maya, "WIN", False):
            self.assertEqual(maya.gpu_reserve_mib([{**p40, "display": True}]), 3072)   # (display rule unchanged)
            self.assertEqual(maya.gpu_reserve_mib([{**p40, "display": False},
                                                   {**p40, "index": 1, "display": True}]), 3072)
        self.assertTrue(maya.gpu_sm6x([p40, {"arch": 60}]))                   # (the int arch reads too)
        self.assertFalse(maya.gpu_sm6x([{"arch": "gfx1201"}]))                # (AMD: no)
        self.assertFalse(maya.gpu_sm6x([]))

    def test_card_name_from_the_pci_revision(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(S.amd_pci_name(sysfs_card(Path(d), {}, 0)), "AMD Radeon RX 9070 XT (gfx1201)")
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(S.amd_pci_name(sysfs_card(Path(d), {}, 0, revision="0xc3")), "AMD Radeon RX 9070 (gfx1201)")
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(S.amd_pci_name(sysfs_card(Path(d), {}, 0, device="0x7551")),
                             "AMD Radeon AI PRO R9700 (gfx1201)")
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(S.amd_pci_name(sysfs_card(Path(d), {}, 0, device="0x744c")))   # (AMD_NAMES then)


class DraftNote(unittest.TestCase):
    def test_one_discrete_gpu_does_not_draft(self):
        a = SimpleNamespace(env=[])
        one = {"backend": "hip", "gpus": [rx9070()]}
        self.assertTrue(maya.draft_note(a, one, "mtp").startswith("none on one GPU"))
        self.assertEqual(maya.draft_note(a, {"backend": "hip", "gpus": [rx9070(), rx9070(1)]}, "mtp"), "the MTP block")
        self.assertEqual(maya.draft_note(a, {"backend": "hip", "gpus": [{"arch": "gfx1151"}]}, "mtp"), "the MTP block")
        self.assertEqual(maya.draft_note(SimpleNamespace(env=["SAGUARO_GLM_MTP_DECODE=1"]), one, "mtp"), "the MTP block")
        self.assertIn("DFlash2", maya.draft_note(a, one, "dflash2"))


class SourceHash(unittest.TestCase):
    def test_bytecode_caches_are_not_sources(self):
        with tempfile.TemporaryDirectory() as d, patch.object(S, "ROOT", Path(d)):
            (Path(d) / "v").mkdir()
            (Path(d) / "v" / "a.py").write_text("x = 1\n")
            before = S.source_hash(("v",))
            (Path(d) / "v" / "__pycache__").mkdir()
            (Path(d) / "v" / "__pycache__" / "a.cpython-313.pyc").write_bytes(b"\0" * 16)
            (Path(d) / "v" / "b.pyc").write_bytes(b"\0")
            self.assertEqual(S.source_hash(("v",)), before)
            (Path(d) / "v" / "a.py").write_text("x = 2\n")
            self.assertNotEqual(S.source_hash(("v",)), before)


class VisionFiles(unittest.TestCase):
    """--yes alone never downloads the vision encoder's files (as the model's); they are found beside the model too."""

    def run_step(self, d: Path, models: Path, **kw):
        a = SimpleNamespace(no_vision=False, download_model=False, yes=True, rebuild=False, vision_dir=None)
        a.__dict__.update(kw)
        fetched = []
        with patch.object(maya, "ROOT", d / "install"), patch.object(maya, "fetch", lambda u, p, w: fetched.append(p)), \
                patch.object(maya, "compile_vision", return_value=False), patch.object(maya, "VEXE", d / "none"), \
                patch.object(maya.S, "UI", None), patch("sys.stdout", new=io.StringIO()):
            try:
                v = maya.vision_step(a, {"gpus": []}, {}, d, models / "m", "Maya-L", models)
            except SystemExit:
                v = "stopped"
        return v, fetched

    def test_yes_does_not_download(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "models" / "m").mkdir(parents=True)
            self.assertEqual(self.run_step(d, d / "models"), (None, []))
            v, fetched = self.run_step(d, d / "models", download_model=True)
            self.assertEqual(len(fetched), 2)

    def test_found_in_the_models_folder_or_vision_dir(self):
        m = maya.MODELS["Maya-L"]["vision"]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "models" / "m").mkdir(parents=True)
            for k in ("mmproj", "vocab"):
                (d / "models" / m[k]).write_bytes(b"x")
            a = SimpleNamespace(vision_dir=None)
            self.assertEqual(maya.vision_files_dir(a, m, d / "models" / "m", d / "models"), d / "models")
            self.assertEqual(maya.vision_files_dir(a, m, d / "models" / "m"), d / "models")     # (the folder above)
            (d / "elsewhere").mkdir()
            own = SimpleNamespace(vision_dir=str(d / "elsewhere"))
            self.assertEqual(maya.vision_files_dir(own, m, d / "models" / "m"), (d / "elsewhere").resolve())
            self.assertEqual(self.run_step(d, d / "models", vision_dir=str(d / "elsewhere")), ("stopped", []))
            (d / "models" / m["vocab"]).unlink()
            self.assertEqual(maya.vision_files_dir(a, m, d / "models" / "m", d / "models"),
                             d / "models" / "m" / "vision")


class PackCheck(unittest.TestCase):
    """A pack an older Maya / Strata install left in the model's folder is never reused or written into."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.shard = self.d / "GLM-5.3-Flash-Maya-L-IQ3_S-00001-of-00004.gguf"
        self.shard.write_bytes(b"GGUF")

    def make_pack(self, pack: Path, sources: bool, fmt=None, head=maya.GLM_PACK_INDEX + "-- generated by tools/iq_pack.py"):
        (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
        for f in maya.PACK_FILES:
            (pack / f).write_text("x")
        (pack / "index.txt").write_text(head + " from " + self.shard.name + "\n")
        (pack / "native_experts.txt").write_text(f"# experts (absolute offsets in {self.shard.name})\n")
        if sources:
            maya.stamp_pack(pack, [self.shard], fmt)

    def build(self, repack=False):
        def fake_iq_pack(cmd, **kw):
            out = Path(cmd[cmd.index("--out") + 1])
            self.built.append(out)
            self.make_pack(out, False)
        self.built = []
        with patch.object(maya, "run", fake_iq_pack), patch.object(maya, "check_sha256"), \
                patch.object(maya, "base_template"), patch("sys.stdout", new=io.StringIO()):
            return maya.pack_step(SimpleNamespace(repack=repack), self.d, [self.shard], self.d, "Maya-L")

    def test_an_old_installs_pack_is_left_alone(self):
        old = self.d / "pack"
        self.make_pack(old, sources=False)
        before = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}
        self.assertIn("no record", maya.pack_foreign(old))
        pack = self.build()
        self.assertEqual(pack, self.d / "pack-saguaro-maya-l")
        self.assertEqual(self.built, [pack])
        self.assertEqual({p: p.read_bytes() for p in old.rglob("*") if p.is_file()}, before)
        self.assertEqual(maya.read_json(pack / maya.PACK_SOURCES)["glm_pack_format"], maya.GLM_PACK_FORMAT)
        self.assertEqual(self.build(), pack)          # (a setup again: the new one is reused, not built again)
        self.assertEqual(self.built, [])
        self.assertEqual(self.build(repack=True), pack)   # (--repack builds the new one again, never the old)
        self.assertEqual(self.built, [pack])

    def test_another_format_is_foreign(self):
        self.make_pack(self.d / "pack", sources=True, fmt=maya.GLM_PACK_FORMAT + 1)
        self.assertIn("format", maya.pack_foreign(self.d / "pack"))
        self.assertEqual(self.build(), self.d / "pack-saguaro-maya-l")
        p2 = self.d / "p2"
        self.make_pack(p2, sources=True, head="# strata pack index v2")
        self.assertIn("not Saguaro's", maya.pack_foreign(p2))

    def test_an_earlier_saguaro_pack_is_reused_and_stamped(self):
        self.make_pack(self.d / "pack", sources=True)      # (sources.json without the format: before this record)
        self.assertIsNone(maya.pack_foreign(self.d / "pack"))
        self.assertEqual(self.build(), self.d / "pack")
        self.assertEqual(self.built, [])
        self.assertEqual(maya.read_json(self.d / "pack" / maya.PACK_SOURCES)["glm_pack_format"], maya.GLM_PACK_FORMAT)


class ConfigName(unittest.TestCase):
    def test_no_double_prefix(self):
        self.assertEqual(maya.glm_config_stem("Maya-S-v2-IQ2_XXS"), "maya-s-v2-iq2_xxs")
        self.assertEqual(maya.glm_config_stem("GSQ-RCO"), "maya-gsq-rco")


class CheckFit(unittest.TestCase):
    def test_check_says_which_quants_fit(self):
        with patch.object(maya, "mem_gb", return_value=(92.0, 80.0)):
            lines = maya.glm_fit_lines([rx9070()])
        self.assertIn("about 92 GB", lines[0])
        self.assertIn("Maya-XS-IQ1_S", lines[1])
        self.assertIn("Maya-L", lines[2])
        self.assertEqual(maya.glm_fit_lines([]), [])


class NoAvx2Message(unittest.TestCase):
    """The setup's no-AVX2 line: an ok about the engine's AVX1 rows where its CPU expert lane
    has them, a warn that it is slower where it does not (the EXL3 quants of deepseek / minimax).  Neither names an
    env switch: the row width is picked when setup.py builds the engine for this CPU, and SAGUARO_GLM_CPU_LANE /
    SAGUARO_DS4_CPU_LANE are the lane's calibrated thread share, not an instruction choice."""

    AVX1 = ("glm", "qfn", "hy4", "ik2")            # (every quant those families run)

    def line(self, family, gguf=False):
        text, stopped = output(lambda: maya.note_no_avx2("Xeon E5-2695 v2", family, gguf))
        self.assertFalse(stopped)
        return text

    def test_avx1_families_say_the_engine_has_the_rows(self):
        for fam in self.AVX1:
            self.assertTrue(maya.cpu_avx1_rows(fam), fam)
            text = self.line(fam)
            self.assertIn("[ok]", text)
            self.assertIn("no AVX2: the engine's AVX1 rows", text)
            self.assertIn("threads", text)

    def test_ds4_and_mm3_gguf_quants_have_the_rows(self):
        for fam in ("deepseek", "minimax"):
            self.assertTrue(maya.cpu_avx1_rows(fam, gguf=True), fam)
            self.assertIn("AVX1 rows", self.line(fam, gguf=True))

    def test_exl3_quants_warn_slower(self):
        for fam in ("deepseek", "minimax"):
            self.assertFalse(maya.cpu_avx1_rows(fam), fam)
            text = self.line(fam)
            self.assertIn("[!]", text)                 # (a warn, not an ok)
            self.assertIn("no AVX1 rows", text)
            self.assertIn("slower", text)
            self.assertNotIn("AVX1 rows, ", text)       # (the ok line's phrasing is not used)

    def test_no_env_switch_is_named_in_any_line(self):
        for fam, gguf in [(f, False) for f in self.AVX1] + [("deepseek", False), ("deepseek", True),
                                                             ("minimax", False), ("minimax", True)]:
            text = self.line(fam, gguf)
            for name in ("SAGUARO_GLM_CPU_LANE", "SAGUARO_DS4_CPU_LANE"):
                self.assertNotIn(name, text)
            self.assertNotIn("turns it off", text)


if __name__ == "__main__":
    unittest.main()
