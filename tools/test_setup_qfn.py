"""tools/test_setup_qfn.py - Qwen3.8-Flash-Next on its own engine (saguaro-qfn) in the setup: its names and setup ids
(told from CYBER-FROST's), the pinned manifest (the model files, the draft layer, the picture tower), the model
question (offered on AMD and on NVIDIA, refused on Windows and on Mac), the memory it was measured with (refused below
it), the RAM tier it gets, the config it writes - AMD's as measured, NVIDIA's mirroring the live 4x Tesla T4 configs,
the vision entry, the drafting choice and the answers-at-once value - the cards it picks, the report's QFN lines, the
--import-model entries and the Models page's entries.  No GPU, no network.

    python -m unittest tools.test_setup_qfn -v
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
import maya_quants as Q  # noqa: E402


def args(**kw):
    base = dict(model=maya.QFN, qfn_quant=None, hy4_quant=None, mm3_quant=None, ds4_quant=None, ik2_quant=None,
                model_dir=None, models_dir=None, gguf_dir=None, draft=None, backend="hip", yes=True, check=False,
                port=None, host=None, api_key=None, env=[], no_vision=False, context=None, gpus=None, gpu=None,
                repack=False, parallel=None, drafting=None, calibrate=False, download_model=False, vision_dir=None,
                rebuild=False)
    base.update(kw)
    return SimpleNamespace(**base)


def refusal(fn):
    out = io.StringIO()
    with patch("sys.stdout", new=out), patch.object(maya.S, "UI", None):
        try:
            fn()
        except SystemExit:
            return out.getvalue()
    raise AssertionError("no refusal: " + out.getvalue()[-400:])


class Names(unittest.TestCase):
    def test_ids_resolve_to_qfn_with_the_quant(self):
        self.assertEqual(Q.resolve("qwen3.8-flash-next"), ("qfn", "ud-q4_k_xl"))
        self.assertEqual(Q.resolve("Qwen3.8-Flash-Next-Q8_0"), ("qfn", "q8_0"))
        self.assertEqual(Q.resolve("qfn"), ("qfn", "ud-q4_k_xl"))
        self.assertEqual(Q.resolve("qfn-q8_0"), ("qfn", "q8_0"))
        self.assertEqual(Q.resolve("cyber-frost"), ("qwen", "q5_k_m"))   # (CYBER-FROST unchanged)

    def test_shown_names_the_engine_and_not_cyber_frost(self):
        s = Q.shown("qfn", "ud-q4_k_xl", 111.3)
        self.assertEqual(s, "Qwen3.8-Flash-Next · UD-Q4_K_XL (saguaro-qfn) (111 GB)")
        self.assertNotIn("CYBER-FROST", s)
        self.assertEqual(Q.shown("qfn", "ud-q4_k_xl"), Q.shown("qwen", "ud-q4_k_xl"))   # (the name is unchanged)
        self.assertIn("CYBER-FROST", Q.shown("qwen", "q5_k_m"))
        self.assertEqual(Q.config_stem("qfn", "ud-q4_k_xl"), "maya-qwen3.8-flash-next-ud-q4_k_xl")
        self.assertEqual(Q.config_stem("qwen", "q5_k_m"), "maya-cyber-frost")

    def test_model_option(self):
        a = args(model="qwen3.8-flash-next-q8_0")
        self.assertIsNone(maya.model_option(a))
        self.assertEqual((a.model, a.qfn_quant), (maya.QFN, "q8_0"))
        self.assertEqual(maya.choose_family(a, {}), "qfn")
        b = args(model="cyber-frost")
        self.assertIsNone(maya.model_option(b))
        self.assertEqual(maya.choose_family(b, {}), "qwen")

    def test_a_setup_config_is_qfn_family_and_told_apart(self):
        cfg = {"exe": "build-hip/qfn/saguaro-qfn", "installer": {"family": "qfn", "qwen_quant": "ud-q4_k_xl"}}
        self.assertEqual(Q.family_of(cfg), "qfn")        # (its own family since the identity split)
        self.assertEqual(maya.setup_family(cfg), "qfn")
        self.assertEqual(Q.config_label(cfg), "Qwen3.8-Flash-Next · UD-Q4_K_XL (saguaro-qfn)")
        # the config a release before the split wrote: read as "qfn" too (maya_quants.family_of's migration)
        old = {"exe": "build-hip/qfn/saguaro-qfn", "installer": {"family": "qwen", "qwen_quant": "ud-q4_k_xl"}}
        self.assertEqual(Q.family_of(old), "qfn")
        self.assertEqual(maya.setup_family(old), "qfn")
        self.assertEqual(Q.config_label(old), "Qwen3.8-Flash-Next · UD-Q4_K_XL (saguaro-qfn)")
        self.assertEqual(maya.setup_family({"installer": {"family": "qwen", "qwen_quant": "q5_k_m"}}), "qwen")
        self.assertEqual(Q.family_of({"exe": "/b/maya", "args": ["--pack", "/p", "--ple-gguf", "/g"],
                                      "installer": {"family": "qwen", "qwen_quant": "q5_k_m"}}), "qwen")

    def test_the_hand_made_configs_the_exe_name_gives_away(self):
        """The live ones (./ref/Qwen3.8-Flash-Next-Q8_0.json, copied): the installer names family "qwen" and no
        quant - the exe is all they carry."""
        cfg = {"exe": "/opt/llm/saguaro/bin/saguaro-qfn", "cwd": "/opt/llm/saguaro/engines/qfn",
               "args": ["--pack", "/opt/llm/models/Qwen3.8-Flash-Next-Q8_0/pack", "--native",
                        "/opt/llm/models/Qwen3.8-Flash-Next-Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf",
                        "--spec", "3", "--mtp", "/opt/llm/models/Qwen3.8-Flash-Next-MTP-Lamina",
                        "--max-context", "1048576", "--kv", "int8", "--vision"],
               "model_name": "Qwen3.8-Flash-Next-Q8_0", "installer": {"family": "qwen"}}
        self.assertEqual(Q.family_of(cfg), "qfn")
        self.assertEqual(maya.is_qfn_config(cfg), True)
        self.assertEqual(maya.setup_family(cfg), "qfn")
        self.assertEqual(Q.config_label(cfg), "Qwen3.8-Flash-Next")     # (no quant recorded: the model's name)
        self.assertEqual(Q.model_defaults(cfg), ({"temperature": 1.0, "top_p": 0.95, "top_k": 20}, "xhigh"))


class Manifest(unittest.TestCase):
    def test_pinned_and_checked(self):
        man = json.loads((ROOT / "tools" / "qfn" / "download.json").read_text())
        self.assertRegex(man["revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(man["mtp"]["revision"], r"^[0-9a-f]{40}$")
        for k, q in man["quants"].items():
            self.assertIn(k, Q.QFN_QUANTS)
            for name, (size, sha) in q["files"].items():
                self.assertTrue(name.startswith(q["name"]) and size > 0 and len(sha) == 64, name)
        m = maya.qfn_entry("ud-q4_k_xl")
        self.assertAlmostEqual(m["download_gb"], 111.3, delta=0.1)
        self.assertEqual(m["name"], "Qwen3.8-Flash-Next-UD-Q4_K_XL")
        self.assertIn("/resolve/efc50b7c3195", maya.file_url(m, "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"))

    def test_the_vision_tower_is_pinned_too(self):
        man = json.loads((ROOT / "tools" / "qfn" / "download.json").read_text())
        vis = man["vision"]
        self.assertRegex(vis["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(vis["repo"], man["repo"])
        for name, (size, sha) in vis["files"].items():
            self.assertTrue(size > 0 and len(sha) == 64, name)
            self.assertEqual(vis["hub_names"].get(name), "mmproj-BF16.gguf")   # (its name on the Hub)
        m = maya.qfn_vision_model()
        self.assertAlmostEqual(m["download_gb"], 0.91, delta=0.01)
        url = maya.file_url(m, "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
        self.assertIn("/resolve/efc50b7c3195", url)
        self.assertTrue(url.endswith("/mmproj-BF16.gguf"), url)


class Offered(unittest.TestCase):
    def test_amd_and_nvidia_offer_it_metal_refuses(self):
        self.assertIsNone(maya.family_unavailable("qfn", "hip"))
        self.assertIsNone(maya.family_unavailable("qfn", "cuda"))    # (NVIDIA too since T2: qfn_fit lays the cards out)
        self.assertIn("Metal", maya.family_unavailable("qfn", "metal"))
        self.assertIsNone(maya.family_unavailable("qwen", "cuda"))   # (CYBER-FROST still everywhere it was)

    def test_the_model_question_has_both_qwen_rows(self):
        got = {}

        def choose(q, intro, rows, default, yes, outro=()):
            got["rows"] = rows
            return maya.FAMILIES.index("qfn")
        with patch.object(maya, "choose", side_effect=choose), patch.object(maya, "setup_cards", return_value=[]):
            self.assertEqual(maya.choose_family(args(model=None), {}), "qfn")
        labels = [r[0] for r in got["rows"]]
        self.assertIn(maya.QFN_SHOWN, labels)
        self.assertTrue(any("CYBER-FROST" in l for l in labels))
        about = got["rows"][maya.FAMILIES.index("qfn")][2]
        self.assertIn("saguaro-qfn", about)
        self.assertIn("Not CYBER-FROST", about)

    def test_refused_below_the_measured_memory(self):
        self.assertIsNone(maya.qfn_quant_fit("ud-q4_k_xl", 15.9, 91.7))   # (an RX 9070 XT with 96 GB: measured)
        self.assertIn("96 GB of RAM", maya.qfn_quant_fit("ud-q4_k_xl", 15.9, 62.0))
        self.assertIn("16 GB or more", maya.qfn_quant_fit("ud-q4_k_xl", 11.9, 91.7))
        out = refusal(lambda: maya.choose_qfn_quant(args(qfn_quant="ud-q4_k_xl"), {}, 15.9, 62.0))
        self.assertIn("96 GB of RAM", out)

    def test_ram_tier(self):
        self.assertEqual(maya.qfn_ram_budget(91.7), 71)
        self.assertEqual(maya.qfn_ram_budget(20.0), 8)

    def test_qfn_fit_contexts_stay_offered_by_the_setup(self):
        for ctx in (maya.QF.DEFAULT_CONTEXT, maya.QF.SPLIT_CONTEXT):
            self.assertIn(ctx, maya.QFN_CONTEXTS)                  # (choose_context offers only QFN_CONTEXTS)


class Config(unittest.TestCase):
    def test_the_measured_amd_settings(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            models, d = t / "models", t / "models" / "Qwen3.8-Flash-Next-UD-Q4_K_XL"
            files = [t / "gguf" / n for n in maya.qfn_entry("ud-q4_k_xl")["files"]]
            pc = {"backend": "hip", "gpus": [{"index": 0, "arch": "gfx1201", "vram_gb": 15.9}]}
            with patch.object(maya, "ROOT", t), patch.object(maya, "mem_gb", return_value=(91.7, 80.0)), \
                    patch.object(maya, "write_run_script", return_value=t / "run.sh"), \
                    patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
                p = maya.write_config_qfn(args(), pc, {"lib_dirs": ["/opt/rocm/lib"]}, "ud-q4_k_xl", files, d / "pack",
                                          models / "Qwen3.8-Flash-Next-MTP-Lamina", 65536, models, d, {})
            cfg = json.loads(p.read_text())
            self.assertEqual(p.name, "maya-qwen3.8-flash-next-ud-q4_k_xl-hip.json")
            self.assertTrue(cfg["exe"].endswith("qfn/saguaro-qfn"))
            import serve.server as SV
            self.assertEqual(SV.engine_kind(cfg), "batch")              # (the server drives it as QFN's engine)
            a = cfg["args"]
            self.assertEqual(a[a.index("--native") + 1], str(files[0]))
            self.assertEqual(a[a.index("--resident-budget-gib") + 1], "71")
            self.assertEqual(a[a.index("--max-context") + 1], "65536")
            self.assertIn("--mtp", a)
            self.assertEqual(cfg["env"]["SAGUARO_MMQ_KQ"], "1")
            self.assertEqual((cfg["backend"], cfg["gpu"]), ("hip", [0]))
            self.assertEqual(cfg["installer"]["qwen_quant"], "ud-q4_k_xl")
            self.assertEqual(cfg["installer"]["family"], "qfn")         # (its own family since the identity split)
            self.assertEqual(Q.family_of(cfg), "qfn")
            self.assertEqual(cfg["sampling"], {"temperature": 1.0, "top_p": 0.95, "top_k": 20})


class NvidiaConfig(unittest.TestCase):
    def test_the_live_t4_settings_on_four_tesla_t4s(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            models, d = t / "models", t / "models" / "Qwen3.8-Flash-Next-UD-Q4_K_XL"
            files = [t / "gguf" / n for n in maya.qfn_entry("ud-q4_k_xl")["files"]]
            pc = {"backend": "cuda", "gpus": [{"index": i, "name": "Tesla T4", "vram_gb": 15.0, "arch": "75"}
                                              for i in range(4)]}
            with patch.object(maya, "ROOT", t), patch.object(maya, "mem_gb", return_value=(220.0, 200.0)), \
                    patch.object(maya, "write_run_script", return_value=t / "run.sh"), \
                    patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
                p = maya.write_config_qfn(args(backend="cuda"), pc, {}, "ud-q4_k_xl", files, d / "pack",
                                          models / "Qwen3.8-Flash-Next-MTP-Lamina", 262144, models, d, {})
            cfg = json.loads(p.read_text())
            self.assertEqual(p.name, "maya-qwen3.8-flash-next-ud-q4_k_xl.json")   # (NVIDIA: no -hip suffix)
            self.assertNotIn("backend", cfg)
            self.assertEqual(cfg["gpu"], [0, 1, 2, 3])
            self.assertEqual(cfg["layer_split"], "11,24,36")                      # (the live server's own split)
            a = cfg["args"]
            self.assertEqual(a[a.index("--max-context") + 1], "262144")
            self.assertEqual(a[a.index("--kv-resident") + 1], "32768")
            self.assertEqual(a[a.index("--resident-budget-gib") + 1], "200")
            self.assertIn("--trim-stage-weights", a)
            self.assertEqual(a[a.index("--prefill-pipe") + 1], "1536")
            self.assertEqual(cfg["env"]["SAGUARO_SM75_PERF"], "1")                # (every card is compute cap 7.5)
            self.assertEqual(cfg["env"]["SAGUARO_KQL"], "3")
            self.assertEqual(cfg["env"]["SAGUARO_MMQ_KQ"], "1")
            import serve.server as SV
            self.assertEqual(SV.engine_kind(cfg), "batch")
            self.assertEqual(Q.family_of(cfg), "qfn")

    def test_a_plain_nvidia_card_gets_no_t4_tuning_or_a_split(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            models, d = t / "models", t / "models" / "Qwen3.8-Flash-Next-UD-Q4_K_XL"
            files = [t / "gguf" / n for n in maya.qfn_entry("ud-q4_k_xl")["files"]]
            pc = {"backend": "cuda", "gpus": [{"index": 0, "name": "RTX A5000", "vram_gb": 24.0, "arch": "86"}]}
            with patch.object(maya, "ROOT", t), patch.object(maya, "mem_gb", return_value=(128.0, 100.0)), \
                    patch.object(maya, "write_run_script", return_value=t / "run.sh"), \
                    patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
                p = maya.write_config_qfn(args(backend="cuda"), pc, {}, "ud-q4_k_xl", files, d / "pack", None,
                                          65536, models, d, {})
            cfg = json.loads(p.read_text())
            self.assertNotIn("layer_split", cfg)
            for k in cfg["env"]:
                self.assertFalse(k.startswith("SAGUARO_SM75") or k in ("SAGUARO_KQL", "SAGUARO_BF16_TC"), k)
            self.assertNotIn("--prefill-pipe", cfg["args"])
            self.assertEqual(cfg["installer"]["family"], "qfn")


class CheckLine(unittest.TestCase):
    def test_the_check_line_names_the_t4_split_only_for_a_split(self):
        t4 = [{"index": i, "name": "Tesla T4", "vram_gb": 15.0, "arch": "75"} for i in range(4)]
        self.assertIn("the live 4x Tesla T4 server's own split",
                      maya.qfn_check_line("ud-q4_k_xl", maya.QF.plan(t4, 220.0)))
        amd = [{"index": 0, "name": "RX 9070 XT", "vram_gb": 15.9, "arch": "gfx1201"}]     # (measured, one card)
        line = maya.qfn_check_line("ud-q4_k_xl", maya.QF.plan(amd, 91.7))
        self.assertNotIn("Tesla T4", line)                                                 # (review finding 2)
        self.assertNotIn("not measured", line)
        self.assertIn("64K context", line)
        plain = [{"index": 0, "name": "RTX A5000", "vram_gb": 24.0, "arch": "86"}]
        self.assertIn("not measured on these cards yet", maya.qfn_check_line("ud-q4_k_xl", maya.QF.plan(plain, 128.0)))
        self.assertIn("needs GPUs of 16 GB or more",
                      maya.qfn_check_line("ud-q4_k_xl", maya.QF.plan([{"index": 0, "name": "RTX 4060",
                                                                       "vram_gb": 8.0, "arch": "89"}], 96.0)))


class Windows(unittest.TestCase):
    def test_qfn_is_refused_there_in_one_plain_line(self):
        with patch.object(maya, "WIN", True):
            # (engines/qfn builds for CUDA and HIP on Linux only - refused on Windows whichever backend)
            self.assertIn("not built on Windows", maya.family_unavailable("qfn", "cuda"))
            self.assertIn("not built on Windows", maya.family_unavailable("qfn", "hip"))
            out = refusal(lambda: maya.check_pc(args(backend="cuda"), "qfn"))
        self.assertIn("Linux for now", out)
        self.assertIsNone(maya.family_unavailable("qfn", "cuda"))      # (Linux: offered)


class QfnGpus(unittest.TestCase):
    def test_four_t4s_all_together_a_big_card_alone(self):
        four = [{"index": i, "name": "Tesla T4", "vram_gb": 15.0, "arch": "75"} for i in range(4)]
        with patch.object(maya, "choose", lambda *x, **k: 0), patch.object(maya, "mem_gb", return_value=(220.0, 200.0)), \
                patch.object(maya, "WIN", False):
            got = maya.choose_gpus(args(backend="cuda"), four, maya.QF.MIN_CC, "qfn")
            self.assertEqual([g["index"] for g in got], [0, 1, 2, 3])       # (the split, as the live server runs)
            big = four + [{"index": 4, "name": "RTX A5000", "vram_gb": 24.0, "arch": "86"}]
            self.assertEqual([g["index"] for g in maya.choose_gpus(args(backend="cuda"), big, maya.QF.MIN_CC, "qfn")],
                             [4])                                           # (24 GB alone: no split over the bus)
            self.assertIn("up to 4 GPUs", refusal(lambda: maya.choose_gpus(args(backend="cuda", gpus="0,1,2,3,4"),
                                                                           big, maya.QF.MIN_CC, "qfn")))

    def test_cards_under_16_gb_are_refused_in_one_line(self):
        small = [{"index": i, "name": "GeForce RTX 4060", "vram_gb": 8.0, "arch": "89"} for i in (0, 1)]
        with patch.object(maya.S, "gpus", return_value=small), patch.object(maya.S, "is_wsl", return_value=False), \
                patch.object(maya, "mem_gb", return_value=(96.0, 80.0)):
            out = refusal(lambda: maya.check_pc(args(backend="cuda"), "qfn"))    # (no --gpus: one line, no crash)
        self.assertIn("needs GPUs of 16 GB or more", out)
        self.assertIn("RTX 4060", out)
        with patch.object(maya.S, "gpus", return_value=[{"index": 2, "name": "RTX A5000", "vram_gb": 24.0,
                                                         "arch": "86"}] + small), \
                patch.object(maya.S, "is_wsl", return_value=False), patch.object(maya, "mem_gb",
                                                                                 return_value=(96.0, 80.0)):
            out = refusal(lambda: maya.check_pc(args(backend="cuda", gpus="1"), "qfn"))   # (--gpus names a small one)
        self.assertIn("needs GPUs of 16 GB or more", out)
        with patch.object(maya.S, "gpus", return_value=small), patch.object(maya.S, "is_wsl", return_value=False), \
                patch.object(maya, "mem_gb", return_value=(96.0, 80.0)):
            out = refusal(lambda: maya.check_pc(args(backend="cuda", gpus="0,1"), "qfn"))  # (both of them named)
        self.assertIn("needs GPUs of 16 GB or more", out)
        with patch.object(maya.S, "gpus", return_value=[{"index": 0, "name": "RTX A5000", "vram_gb": 24.0,
                                                         "arch": "86"}]), \
                patch.object(maya.S, "is_wsl", return_value=False), patch.object(maya, "mem_gb",
                                                                                 return_value=(60.0, 40.0)):
            out = refusal(lambda: maya.check_pc(args(backend="cuda"), "qfn"))   # (and the RAM for the tier)
        self.assertIn("96 GB of RAM", out)


class QfnReport(unittest.TestCase):
    def test_the_engines_speed_lines_are_picked(self):
        for line in ("PP 512 512 120 4266.7 new_read=512 read_ms=120.0 rid=3",
                     "RDONE rid=3 new=512 ms=120.0",
                     "VRAM dev=0 pci=0000:00:03.0 total=15858 free=900 weights=12000 kv=415 kv_ram=0"):
            self.assertIsNotNone(maya.REPORT_LINES.search(line), line)
        self.assertIsNone(maya.REPORT_LINES.search("PP 512 512 120 4266.7"))   # (plain chunk progress)


def pc_hip(gpus=None):
    return {"backend": "hip", "archs": ["gfx1201"],
            "gpus": gpus or [{"index": 0, "arch": "gfx1201", "vram_gb": 15.9}]}


def written_cfg(t: Path, a, vision=None, old=None, gpus=None):
    """write_config_qfn's config in a temp folder, the way set_up_qfn hands it everything."""
    models, d = t / "models", t / "models" / "Qwen3.8-Flash-Next-UD-Q4_K_XL"
    files = [t / "gguf" / n for n in maya.qfn_entry("ud-q4_k_xl")["files"]]
    with patch.object(maya, "ROOT", t), patch.object(maya, "mem_gb", return_value=(91.7, 80.0)), \
            patch.object(maya, "write_run_script", return_value=t / "run.sh"), \
            patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
        p = maya.write_config_qfn(a, pc_hip(gpus), {"lib_dirs": ["/opt/rocm/lib"]}, "ud-q4_k_xl", files, d / "pack",
                                  models / "Qwen3.8-Flash-Next-MTP-Lamina", 65536, models, d, old or {}, vision)
    return json.loads(p.read_text())


class Vision(unittest.TestCase):
    def test_the_step_writes_the_live_config_s_entry(self):
        """The vision entry with every key the live T4 config carries, the elastic ones included; the reader the AMD
        path builds (vision-qfn-hip), the mmproj under the model folder, the first shard for the vocabulary."""
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            d = t / "models" / "Qwen3.8-Flash-Next-UD-Q4_K_XL"
            files = [t / "gguf" / n for n in maya.qfn_entry("ud-q4_k_xl")["files"]]
            with patch.object(maya, "offer_download", return_value=True), \
                    patch.object(maya, "sha256_checked", return_value=True), \
                    patch.object(maya, "compile_vision_qfn", return_value=True), \
                    patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
                v = maya.qfn_vision_step(args(), pc_hip(), {"archs": ["gfx1201"]}, t / "llama", d, files)
            self.assertEqual(v, {"exe": str(maya.qfn_vision_exe(True)),
                                 "mmproj": str(d / "vision" / "mmproj-Qwen3.8-Flash-Next-BF16.gguf"),
                                 "model": str(files[0]), "gpu": True, **maya.QFN_VISION})
            self.assertEqual(v["exe"].endswith("vision-qfn-hip"), True)
            self.assertEqual((v["elastic"], v["elastic_mib"], v["elastic_idle_s"], v["elastic_keep_s"],
                              v["max_tokens"]), (True, 1536, 30, 600, 1024))

    def test_no_vision_leaves_it_out(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            dl = patch.object(maya, "offer_download", side_effect=AssertionError("downloaded despite --no-vision"))
            with dl, patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
                self.assertIsNone(maya.qfn_vision_step(args(no_vision=True), pc_hip(), {}, t / "llama",
                                                       t / "models", [t / "gguf" / "x.gguf"]))
            cfg = written_cfg(t, args())
            self.assertNotIn("vision", cfg)
            self.assertNotIn("--vision", cfg["args"])

    def test_the_config_gets_the_entry_and_the_vision_arg(self):
        with tempfile.TemporaryDirectory() as t:
            v = {"exe": "build-vision-qfn/bin/vision-qfn-hip", "mmproj": "m.gguf", "model": "s.gguf", "gpu": True,
                 **maya.QFN_VISION}
            cfg = written_cfg(Path(t), args(), vision=v)
            self.assertEqual(cfg["vision"], v)
            self.assertIn("--vision", cfg["args"])


class SetupFlow(unittest.TestCase):
    """set_up_qfn with the machine, the build and the downloads stepped around: what the choices write."""

    def flow(self, t: Path, a, vision=None, mtp=True):
        eng = t / "build" / "qfn"
        eng.mkdir(parents=True)
        (eng / "saguaro-qfn").write_text("")
        models = t / "models"
        files = [t / "gguf" / n for n in maya.qfn_entry("ud-q4_k_xl")["files"]]
        with patch.object(maya, "ROOT", t), patch.object(maya, "qfn_exe", return_value=eng / "saguaro-qfn"), \
                patch.object(maya, "check_pc", return_value=pc_hip()), \
                patch.object(maya, "mem_gb", return_value=(91.7, 80.0)), \
                patch.object(maya, "tools_step", return_value=t / "llama"), \
                patch.object(maya, "build_step", return_value={"lib_dirs": ["/opt/rocm/lib"]}), \
                patch.object(maya, "qfn_files_step", return_value=files), \
                patch.object(maya, "qfn_pack_step", return_value=models / "Qwen3.8-Flash-Next-UD-Q4_K_XL" / "pack"), \
                patch.object(maya, "qfn_vision_step", return_value=vision), \
                patch.object(maya, "qfn_mtp_step", return_value=models / "Qwen3.8-Flash-Next-MTP-Lamina" if mtp
                             else None), \
                patch.object(maya, "write_run_script", return_value=t / "run.sh"), \
                patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
            return maya.set_up_qfn(a, {})

    def setup(self, t: Path, **kw):
        vision = kw.pop("_vision", None)
        nomtp = kw.pop("_nomtp", False)
        a = args(qfn_quant="ud-q4_k_xl", context=65536, models_dir=t / "models", **kw)
        p = self.flow(t, a, vision=vision, mtp=not nomtp)
        return json.loads(p.read_text())

    def test_drafting_off_is_the_mtp_max_t_arg(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = self.setup(Path(t), drafting="off")
            a = cfg["args"]
            self.assertIn("--mtp", a)                              # (the server needs the layer loaded either way)
            self.assertEqual(a[a.index("--mtp-max-t") + 1], "1")   # (off: never draft past one token)
            self.assertEqual(cfg["installer"]["mtp_calibration"]["kept"], "none")

    def test_drafting_on_keeps_the_layer(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = self.setup(Path(t), drafting="on")
            self.assertNotIn("--mtp-max-t", cfg["args"])
            self.assertEqual(cfg["installer"]["mtp_calibration"]["kept"], "mtp")

    def test_measure_leaves_drafting_on_for_the_calibration_to_come(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = self.setup(Path(t))                              # (--yes: the question's answer is measure)
            self.assertIn("--mtp", cfg["args"])
            self.assertNotIn("--mtp-max-t", cfg["args"])
            self.assertNotIn("mtp_calibration", cfg["installer"])  # (the measurement of T4's step 4 will record it)

    def test_the_vision_entry_travels_into_the_config(self):
        with tempfile.TemporaryDirectory() as t:
            v = {"exe": "v", "mmproj": "m.gguf", "model": "s.gguf", "gpu": True, **maya.QFN_VISION}
            cfg = self.setup(Path(t), drafting="on", _vision=v)
            self.assertEqual(cfg["vision"], v)
            self.assertIn("--vision", cfg["args"])

    def test_no_draft_layer_no_question_and_no_mtp(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = self.setup(Path(t), _nomtp=True)
            self.assertNotIn("--mtp", cfg["args"])
            self.assertNotIn("mtp_calibration", cfg["installer"])


class Parallel(unittest.TestCase):
    def test_qfn_takes_it_and_the_config_carries_it(self):
        self.assertIsNone(maya.check_parallel(args(parallel=2), "qfn"))   # (no refusal since T3)
        out = refusal(lambda: maya.check_parallel(args(parallel=2), "qwen"))
        self.assertIn("one request at a time", out)
        self.assertIn("Qwen3.8-Flash-Next", out)                           # (listed with the engines that do)

    def test_the_values_written(self):
        two = [{"index": 0, "arch": "gfx1201", "vram_gb": 15.9}, {"index": 1, "arch": "gfx1201", "vram_gb": 15.9}]
        cases = [(args(), None, "the one 16 GB card: off, it lost speed there"),      # (no parallel key)
                 (args(parallel=4), 4, "--parallel 4"),
                 (args(), 2, "several GPUs: 2, as the live T4 configs run it", two),
                 (args(parallel=1), None, "--parallel 1: off")]
        for a, want, why, *gpus in cases:
            with tempfile.TemporaryDirectory() as t:
                cfg = written_cfg(Path(t), a, gpus=(gpus[0] if gpus else None))
                self.assertEqual(cfg.get("parallel"), want, why)
        with tempfile.TemporaryDirectory() as t:                 # (a setup before wrote 3: a setup again keeps it)
            written_cfg(Path(t), args(parallel=3))
            self.assertEqual(written_cfg(Path(t), args())["parallel"], 3)


class ImportCatalog(unittest.TestCase):
    def test_the_qfn_entries(self):
        with patch("sys.stdout", new=io.StringIO()):
            cat = maya.import_catalog(Path("models"))
        rels = {e["rel"] for e in cat}
        qfn = [e for e in cat if "Qwen3.8-Flash-Next" in e["label"]]
        self.assertTrue(qfn)
        self.assertIn("Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", rels)
        self.assertIn("Qwen3.8-Flash-Next-Q8_0/Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf", rels)
        self.assertIn("Qwen3.8-Flash-Next-MTP-Lamina/dense.txt", rels)         # (the draft layer every quant shares)
        self.assertIn("Qwen3.8-Flash-Next-UD-Q4_K_XL/vision/mmproj-Qwen3.8-Flash-Next-BF16.gguf", rels)
        self.assertIn("Qwen3.8-Flash-Next-Q8_0/vision/mmproj-Qwen3.8-Flash-Next-BF16.gguf", rels)
        self.assertNotIn("Qwen3.8-Flash-Next-MTP-Lamina/LICENSE", rels)        # (no published sha256 to match on)
        for e in qfn:
            self.assertTrue(e["size"] and all(len(x) == 64 for x in e["sha"]), e["rel"])


class ModelsPage(unittest.TestCase):
    def test_catalog_entries(self):
        from serve import model_pull as M
        amd = [{"index": 0, "arch": "gfx1201", "vram_gb": 15.9, "vendor": "amd"}]
        rows = {e["id"]: e for e in M.catalog(ROOT, glm_sizes={}, cards=amd, ram_gb=91.7)}
        e = rows["qwen3.8-flash-next-ud-q4_k_xl"]
        self.assertEqual(e["family"], "qfn")           # (its own family since the identity split)
        self.assertIn("saguaro-qfn", e["name"])
        self.assertAlmostEqual(e["size_gb"], 111.3, delta=0.1)
        self.assertIn("fits", e["fit"])
        self.assertIn("measured on an RX 9070 XT (gfx1201) with 96 GB of RAM", e["needs"])
        self.assertNotIn("experimental", e["needs"])
        self.assertIn("qwen3.8-flash-next-q8_0", rows)
        self.assertIn("saguaro-glm", rows["cyber-frost"]["needs"])
        small = {e["id"]: e for e in M.catalog(ROOT, glm_sizes={}, cards=amd, ram_gb=60.0)}
        self.assertIn("96 GB are needed", small["qwen3.8-flash-next-ud-q4_k_xl"]["fit"])


if __name__ == "__main__":
    unittest.main()
