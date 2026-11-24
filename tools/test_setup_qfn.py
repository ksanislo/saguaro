"""tools/test_setup_qfn.py - Qwen3.8-Flash-Next on its own engine (saguaro-qfn) in the setup: its names and setup ids
(told from CYBER-FROST's), the pinned manifest, the model question (offered on AMD and on NVIDIA, refused on Windows
and on Mac), the memory it was measured with (refused below it), the RAM tier it gets, the config it writes - AMD's
as measured, NVIDIA's mirroring the live 4x Tesla T4 configs - the cards it picks, the report's QFN lines, and the
Models page's entries.  No GPU, no network.

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
                repack=False, parallel=None)
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


class QfnReport(unittest.TestCase):
    def test_the_engines_speed_lines_are_picked(self):
        for line in ("PP 512 512 120 4266.7 new_read=512 read_ms=120.0 rid=3",
                     "RDONE rid=3 new=512 ms=120.0",
                     "VRAM dev=0 pci=0000:00:03.0 total=15858 free=900 weights=12000 kv=415 kv_ram=0"):
            self.assertIsNotNone(maya.REPORT_LINES.search(line), line)
        self.assertIsNone(maya.REPORT_LINES.search("PP 512 512 120 4266.7"))   # (plain chunk progress)


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
