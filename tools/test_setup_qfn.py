"""tools/test_setup_qfn.py - Qwen3.8-Flash-Next on its own engine (saguaro-qfn) in the setup: its names and setup ids
(told from CYBER-FROST's), the pinned manifest, the model question (AMD: offered; NVIDIA: listed, not offered), the
memory it was measured with (refused below it), the RAM tier it gets, the config it writes and the Models page's
entries.  No GPU, no network.

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
        for i in Q.QWEN_IDS:                                       # (CYBER-FROST on saguaro-qfn: every id still taken)
            self.assertEqual(Q.resolve(i), ("qfn", Q.CYBER_FROST_QUANT), i)

    def test_shown_names_the_engine_and_not_cyber_frost(self):
        s = Q.shown("qfn", "ud-q4_k_xl", 111.3)
        self.assertEqual(s, "Qwen3.8-Flash-Next · UD-Q4_K_XL (saguaro-qfn) (111 GB)")
        self.assertNotIn("CYBER-FROST", s)
        self.assertEqual(Q.shown("qfn", "ud-q4_k_xl"), Q.shown("qwen", "ud-q4_k_xl"))   # (the name is unchanged)
        self.assertIn("CYBER-FROST", Q.shown("qwen", "q5_k_m"))
        self.assertEqual(Q.shown("qfn", "cyber-frost-q5_k_m"), "Qwen3.8-Flash-Next · CYBER-FROST-3.8 Q5_K_M "
                                                               "(saguaro-qfn)")
        self.assertEqual(Q.config_stem("qfn", "ud-q4_k_xl"), "maya-qwen3.8-flash-next-ud-q4_k_xl")
        self.assertEqual(Q.config_stem("qwen", "q5_k_m"), "maya-cyber-frost")
        self.assertEqual(Q.config_stem("qfn", "cyber-frost-q5_k_m"), "maya-cyber-frost")   # (the name it always had)

    def test_model_option(self):
        a = args(model="qwen3.8-flash-next-q8_0")
        self.assertIsNone(maya.model_option(a))
        self.assertEqual((a.model, a.qfn_quant), (maya.QFN, "q8_0"))
        self.assertEqual(maya.choose_family(a, {}), "qfn")
        b = args(model="cyber-frost")
        self.assertIsNone(maya.model_option(b))
        self.assertEqual((b.model, b.qfn_quant), (maya.QFN, "cyber-frost-q5_k_m"))
        self.assertEqual(maya.choose_family(b, {}), "qfn")        # (CYBER-FROST on saguaro-qfn now)

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

    def test_the_cyber_frost_quant(self):
        """CYBER-FROST as a quant of QFN: the one file, pinned revision, size and sha256 tools/qwen/download.json
        has - copied exactly - from its own repo."""
        old = json.loads((ROOT / "tools" / "qwen" / "download.json").read_text())
        m = maya.qfn_entry(Q.CYBER_FROST_QUANT)
        self.assertEqual((m["repo"], m["revision"]), (old["repo"], old["revision"]))
        self.assertEqual(m["files"], {k: v[0] for k, v in old["files"].items()})
        self.assertEqual(m["sha256"], {k: v[1] for k, v in old["files"].items()})
        self.assertEqual(m["name"], "CYBER-FROST-3.8-PS")
        self.assertEqual(maya.file_url(m, "CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf"),
                         "https://huggingface.co/peasantsmith/CYBER-FROST-3.8-PS-GUFF/resolve/"
                         "32867bb9e228afac427c136dd657b34b7c3f0e86/CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf")


class Offered(unittest.TestCase):
    def test_amd_offers_nvidia_lists_with_the_reason(self):
        self.assertIsNone(maya.family_unavailable("qfn", "hip"))
        self.assertIn("on AMD GPUs", maya.family_unavailable("qfn", "cuda"))
        self.assertIn("on AMD GPUs", maya.family_unavailable("qfn", None))   # (no backend: the CUDA default)
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

    def test_the_cyber_frost_config(self):
        """Its config keeps the name it always had, takes the drafter from inside the pack (the file's own NextN
        block) and carries the pictures the way the live QFN configs do."""
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            models, d = t / "models", t / "models" / "CYBER-FROST-3.8-PS"
            files = [t / "gguf" / n for n in maya.qfn_entry(Q.CYBER_FROST_QUANT)["files"]]
            pc = {"backend": "hip", "gpus": [{"index": 0, "arch": "gfx1201", "vram_gb": 15.9}]}
            vision = {"exe": str(maya.qfn_exe().parent / "vision-qfn"),
                      "mmproj": str(d / "vision" / "mmproj-F16.gguf"), "model": str(files[0]), "gpu": True,
                      "max_tokens": 1024, "elastic": True, "elastic_mib": 1536, "elastic_idle_s": 30,
                      "elastic_keep_s": 600}
            with patch.object(maya, "ROOT", t), patch.object(maya, "mem_gb", return_value=(91.7, 80.0)), \
                    patch.object(maya, "write_run_script", return_value=t / "run.sh"), \
                    patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
                p = maya.write_config_qfn(args(), pc, {}, Q.CYBER_FROST_QUANT, files, d / "pack",
                                          d / "pack" / "mtp", 65536, models, d, {}, vision)
            cfg = json.loads(p.read_text())
            self.assertEqual(p.name, "maya-cyber-frost-hip.json")
            a = cfg["args"]
            self.assertEqual(a[a.index("--native") + 1], str(files[0]))
            self.assertEqual(a[a.index("--mtp") + 1], str(d / "pack" / "mtp"))
            self.assertIn("--vision", a)
            self.assertEqual(cfg["vision"], vision)
            self.assertEqual(cfg["installer"]["family"], "qfn")
            self.assertEqual(cfg["installer"]["qwen_quant"], "cyber-frost-q5_k_m")
            self.assertEqual(Q.family_of(cfg), "qfn")
            self.assertEqual(cfg["sampling"], {"temperature": 1.0, "top_p": 0.95, "top_k": 20})   # (its own)


class Drafter(unittest.TestCase):
    """CYBER-FROST step 7 on QFN: the drafter built from the file's own NextN block (tools/mtp_from_gguf.py) into
    <pack>/mtp - dense.txt, dense.bin, experts.bin, the layout saguaro-qfn's --mtp loader reads - kept while its
    tools and the model file stand; and the pictures step."""

    def built(self, t: Path):
        mtp = t / "pack" / "mtp"
        mtp.mkdir(parents=True)
        for f in ("dense.txt", "dense.bin", "experts.bin"):
            (mtp / f).write_text("x")
        (mtp / maya.PACK_SOURCES).write_text(json.dumps({"tools": "same"}))
        return mtp

    def test_kept_while_unchanged_built_again_when_the_tools_moved(self):
        quiet = [patch.object(maya, n, lambda *a, **k: None) for n in ("ok", "say", "warn", "step")]
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            mtp = self.built(t)
            for p in quiet:
                p.start()
            try:
                with patch.object(maya.S, "source_hash", return_value="same"), \
                        patch.object(maya, "pack_stale", return_value=None), patch.object(maya, "run") as ran:
                    self.assertEqual(maya.qfn_drafter_step(args(), t / "pack", [t / "m.gguf"], t / "llama"), mtp)
                    self.assertEqual(ran.call_count, 0)                     # (already built: nothing runs)
                with patch.object(maya.S, "source_hash", return_value="new"), \
                        patch.object(maya, "pack_stale", return_value=None), patch.object(maya, "run") as ran, \
                        patch.object(maya, "stamp_pack"):
                    self.assertEqual(maya.qfn_drafter_step(args(), t / "pack", [t / "m.gguf"], t / "llama"), mtp)
                self.assertIn("mtp_from_gguf.py", ran.call_args[0][0][1])   # (built from the file's NextN block)
            finally:
                for p in quiet:
                    p.stop()

    def test_vision_entry(self):
        """--no-vision and a declined download give text only; the entry is the live QFN configs' shape."""
        with patch.object(maya, "ok", lambda *x: None):
            self.assertIsNone(maya.qfn_vision_step(args(no_vision=True), Path("/m"), Path("/m/x.gguf")))
        with tempfile.TemporaryDirectory() as t:
            d, g = Path(t), Path(t) / "CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf"
            quiet = [patch.object(maya, n, lambda *x: None) for n in ("ok", "warn")]
            for p in quiet:
                p.start()
            try:
                with patch.object(maya, "offer_download", lambda *x: False):
                    self.assertIsNone(maya.qfn_vision_step(args(), d, g))   # (text only, said why)
                with patch.object(maya, "file_incomplete", lambda *x: None), \
                        patch.object(maya, "sha256_checked", lambda *x: True):
                    v = maya.qfn_vision_step(args(), d, g)
            finally:
                for p in quiet:
                    p.stop()
        self.assertEqual(v, {"exe": str(maya.qfn_vexe()), "mmproj": str(d / "vision" / "mmproj-F16.gguf"),
                             "model": str(g), "gpu": True, "max_tokens": 1024, "elastic": True,
                             "elastic_mib": 1536, "elastic_idle_s": 30, "elastic_keep_s": 600})


class Migration(unittest.TestCase):
    """An existing CYBER-FROST config of the old engine (installer "qwen" / "q5_k_m", the GLM engine with
    --ple-gguf) is rewritten to a saguaro-qfn config of the same file - once: idempotent."""

    OLD = {"exe": "/s/build-hip/saguaro-glm", "backend": "hip",
           "args": ["--pack", "/m/CYBER-FROST-3.8-PS/pack",
                    "--native", "/m/CYBER-FROST-3.8-PS/CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf",
                    "--ple-gguf", "/m/CYBER-FROST-3.8-PS/CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf",
                    "--spec", "4", "--prefill", "auto", "--expert-cache", "auto", "--max-context", "131072",
                    "--kv", "int8", "--expert-profile", "/m/CYBER-FROST-3.8-PS/pack/expert-profile.bin",
                    "--mtp", "/m/CYBER-FROST-3.8-PS/pack/mtp", "--vision"],
           "cwd": "/s", "tokenizer": "/m/CYBER-FROST-3.8-PS/pack/tokenizer", "model_name": "cyber-frost-3.8-ps",
           "gpu": [0], "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}, "reasoning_effort": "xhigh",
           "port": 8090,
           "vision": {"exe": "/s/build-vision/bin/vision-glm", "mmproj": "/m/vision/mmproj-F16.gguf",
                      "model": "/m/CYBER-FROST-3.8-PS/CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf", "gpu": True},
           "installer": {"family": "qwen", "models_dir": "/m", "quant": "CYBER-FROST-3.8-PS",
                         "qwen_quant": "q5_k_m", "model_dir": "/m/CYBER-FROST-3.8-PS",
                         "written": "2026-11-01 10:00"}}

    def migrated(self):
        cfg = json.loads(json.dumps(self.OLD))
        self.assertTrue(maya.migrate_cyber_frost(cfg))
        return cfg

    def test_reads_as_qfn_keeping_files_context_port_and_sampling(self):
        c = self.migrated()
        self.assertEqual(Q.family_of(c), "qfn")
        self.assertEqual(Q.config_key(c), "cyber-frost-q5_k_m")
        self.assertEqual(Q.config_label(c), "Qwen3.8-Flash-Next · CYBER-FROST-3.8 Q5_K_M (saguaro-qfn)")
        self.assertTrue(c["exe"].endswith("qfn/saguaro-qfn"))
        self.assertEqual(c["cwd"], str(maya.ROOT / "engines" / "qfn"))
        self.assertEqual(c["backend"], "hip")                              # (its PC as it was)
        a = c["args"]
        self.assertNotIn("--ple-gguf", a)                                  # (the Qwen path's arguments are gone)
        self.assertEqual(a[a.index("--pack") + 1], "/m/CYBER-FROST-3.8-PS/pack")     # (its model files kept)
        self.assertEqual(a[a.index("--native") + 1],
                         "/m/CYBER-FROST-3.8-PS/CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf")
        self.assertEqual(a[a.index("--mtp") + 1], "/m/CYBER-FROST-3.8-PS/pack/mtp")  # (the NextN drafter's layout)
        self.assertEqual(a[a.index("--max-context") + 1], "131072")        # (the context as it was)
        self.assertEqual(c["port"], 8090)                                  # (the port as it was)
        self.assertEqual(c["sampling"], {"temperature": 1.0, "top_p": 0.95, "top_k": 20})   # (the sampling)
        self.assertTrue(c["vision"]["exe"].endswith("qfn/vision-qfn"))     # (its pictures on QFN's reader)
        self.assertEqual(Q.config_stem("qfn", Q.config_key(c)), "maya-cyber-frost")  # (same file name)

    def test_once_only_and_nothing_else(self):
        self.assertFalse(maya.migrate_cyber_frost(self.migrated()))        # (idempotent: the rewrite runs once)
        for cfg in ({}, {"installer": {}},
                    {"installer": {"family": "qfn", "qwen_quant": "cyber-frost-q5_k_m"}},   # (QFN's own as it is)
                    {"installer": {"family": "minimax", "mm3_quant": "exl3"}},
                    {"args": ["--max-context", "65536"], "installer": {"family": "qwen"}}):
            # (a Qwen-path config that names no files of the setup: left alone too)
            self.assertFalse(maya.migrate_cyber_frost(json.loads(json.dumps(cfg))), str(cfg))

    def test_left_on_the_old_engine_where_qfn_has_no_engine(self):
        """An NVIDIA (or Mac) CYBER-FROST config is not rewritten to saguaro-qfn: --setup would refuse to build the
        engine there, so migrating strands it.  It keeps its old-engine files and runs on saguaro-glm until that
        route is decided."""
        for backend in ("cuda", "metal", None):
            cfg = json.loads(json.dumps(self.OLD))
            cfg["backend"] = backend
            if backend is None:
                cfg.pop("backend")
            self.assertFalse(maya.migrate_cyber_frost(cfg), str(backend))
            self.assertEqual(cfg["installer"]["family"], "qwen")            # (untouched)
            self.assertEqual(cfg["exe"], self.OLD["exe"])                   # (still the GLM engine)
            self.assertIn("--ple-gguf", cfg["args"])                        # (its Qwen-path arguments stand)


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
        self.assertNotIn("qwen3.8-flash-next-cyber-frost-q5_k_m", rows)     # (CYBER-FROST is offered once)
        cf = rows["cyber-frost"]                                            # (and it points at the QFN quant)
        self.assertEqual(cf["family"], "qfn")
        self.assertIn("CYBER-FROST-3.8 Q5_K_M", cf["name"])
        self.assertIn("saguaro-qfn", cf["needs"])
        self.assertAlmostEqual(cf["size_gb"], 112.5, delta=0.1)
        self.assertIn("fits", cf["fit"])
        small = {e["id"]: e for e in M.catalog(ROOT, glm_sizes={}, cards=amd, ram_gb=60.0)}
        self.assertIn("96 GB are needed", small["qwen3.8-flash-next-ud-q4_k_xl"]["fit"])
        self.assertIn("96 GB are needed", small["cyber-frost"]["fit"])


if __name__ == "__main__":
    unittest.main()
