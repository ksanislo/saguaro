"""tools/test_setup_qfn.py - Qwen3.8-Flash-Next on its own engine (saguaro-qfn) in the setup: its names and setup ids
(the old CYBER-FROST ids among them, resolving to its quant), the pinned manifest (the model files, the draft layer,
the picture tower), the model
question (offered on AMD and on NVIDIA, refused on Windows and on Mac), v1.2.1-B's fit policy (refused only under
the true minimum; between it and the measured layout the setup scales the layout down - drafting off, --kv-resident
4096, a 32K context - warns what it turned off and why and confirms, --yes skipping the question and --check only
reporting), Pascal (MIN_CC 61 - a Tesla P40 with a CUDA 12.x toolkit, and only CUDA 13 installed says so plainly),
the RAM tier it gets, the config it writes - AMD's as measured, NVIDIA's mirroring the live 4x Tesla T4 configs,
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
                rebuild=False, nvcc=None, host_compiler=None)
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
        self.assertEqual(Q.shown("qfn", "cyber-frost-q5_k_m"), "Qwen3.8-Flash-Next · CYBER-FROST-3.8 Q5_K_M "
                                                               "(saguaro-qfn)")
        self.assertEqual(Q.config_stem("qfn", "ud-q4_k_xl"), "maya-qwen3.8-flash-next-ud-q4_k_xl")
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

    def test_a_setup_config_is_qfn_family(self):
        cfg = {"exe": "build-hip/qfn/saguaro-qfn", "installer": {"family": "qfn", "qwen_quant": "ud-q4_k_xl"}}
        self.assertEqual(Q.family_of(cfg), "qfn")        # (its own family since the identity split)
        self.assertEqual(maya.setup_family(cfg), "qfn")
        self.assertEqual(Q.config_label(cfg), "Qwen3.8-Flash-Next · UD-Q4_K_XL (saguaro-qfn)")
        # the config a release before the split wrote: read as "qfn" too (maya_quants.family_of's migration)
        old = {"exe": "build-hip/qfn/saguaro-qfn", "installer": {"family": "qwen", "qwen_quant": "ud-q4_k_xl"}}
        self.assertEqual(Q.family_of(old), "qfn")
        self.assertEqual(maya.setup_family(old), "qfn")
        self.assertEqual(Q.config_label(old), "Qwen3.8-Flash-Next · UD-Q4_K_XL (saguaro-qfn)")
        # and the old engine's CYBER-FROST one (T8a: its model is this family's quant; migrate_cyber_frost rewrites it)
        self.assertEqual(maya.setup_family({"installer": {"family": "qwen", "qwen_quant": "q5_k_m"}}), "qfn")
        self.assertEqual(Q.family_of({"exe": "/b/maya", "args": ["--pack", "/p", "--ple-gguf", "/g"],
                                      "installer": {"family": "qwen", "qwen_quant": "q5_k_m"}}), "qfn")

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

    def test_the_cyber_frost_quant(self):
        """CYBER-FROST as a quant of QFN: the one file, at the repo, revision, size and sha256 the old
        tools/qwen/download.json pinned (T8a: that manifest is gone, these values are its copy)."""
        m = maya.qfn_entry(Q.CYBER_FROST_QUANT)
        self.assertEqual(m["repo"], "peasantsmith/CYBER-FROST-3.8-PS-GUFF")
        self.assertEqual(m["revision"], "32867bb9e228afac427c136dd657b34b7c3f0e86")
        self.assertEqual(m["files"], {"CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf": 112534603328})
        self.assertEqual(m["sha256"], {"CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf":
                                       "1c03787fc8f5b53481213f92689e33f8ad124151e834362295458d6a89dca9c5"})
        self.assertEqual(m["name"], "CYBER-FROST-3.8-PS")
        self.assertEqual(maya.file_url(m, "CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf"),
                         "https://huggingface.co/peasantsmith/CYBER-FROST-3.8-PS-GUFF/resolve/"
                         "32867bb9e228afac427c136dd657b34b7c3f0e86/CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf")


class Offered(unittest.TestCase):
    def test_amd_and_nvidia_offer_it_metal_refuses(self):
        self.assertIsNone(maya.family_unavailable("qfn", "hip"))
        self.assertIsNone(maya.family_unavailable("qfn", "cuda"))    # (NVIDIA too since T2: qfn_fit lays the cards out)
        self.assertIsNone(maya.family_unavailable("qfn", None))      # (no backend: the CUDA default, offered too)
        self.assertIn("Metal", maya.family_unavailable("qfn", "metal"))

    def test_the_model_question_has_one_qwen_row(self):
        got = {}

        def choose(q, intro, rows, default, yes, outro=()):
            got["rows"] = rows
            return maya.FAMILIES.index("qfn")
        with patch.object(maya, "choose", side_effect=choose), patch.object(maya, "setup_cards", return_value=[]):
            self.assertEqual(maya.choose_family(args(model=None), {}), "qfn")
        labels = [r[0] for r in got["rows"]]
        self.assertEqual(len(got["rows"]), len(maya.FAMILIES))   # (the old CYBER-FROST row is gone: one Qwen row)
        self.assertIn(maya.QFN_SHOWN, labels)
        self.assertEqual(sum("Qwen" in l for l in labels), 1)
        about = got["rows"][maya.FAMILIES.index("qfn")][2]
        self.assertIn("saguaro-qfn", about)
        self.assertIn("Not CYBER-FROST", about)                  # (the quant list is where it is offered)

    def test_refused_only_what_cannot_run(self):
        """v1.2.1-B: a quant is refused only under its true minimum (6.3 GB on the card at UD-Q4, 9.0 at Q8_0;
        16-19 GB of RAM) - 62 GB of RAM or an 11.9 GB card RUN, slowly: the layout warns and confirms."""
        self.assertIsNone(maya.qfn_quant_fit("ud-q4_k_xl", 15.9, 91.7))   # (an RX 9070 XT with 96 GB: measured)
        self.assertIsNone(maya.qfn_quant_fit("ud-q4_k_xl", 15.9, 62.0))   # (the old "96 GB of RAM" refusal: slow)
        self.assertIsNone(maya.qfn_quant_fit("ud-q4_k_xl", 11.9, 91.7))   # (the old "16 GB or more" refusal)
        self.assertIsNone(maya.qfn_quant_fit("ud-q4_k_xl", 8.0, 91.7))    # (6.3 GB needed: it runs, slowly)
        self.assertIn("8 GB cannot hold the 9.0 GB", maya.qfn_quant_fit("q8_0", 8.0, 91.7))   # (cannot run)
        self.assertIn("cannot run with 12 GB of RAM", maya.qfn_quant_fit("cyber-frost-q5_k_m", 15.9, 12.0))
        out = refusal(lambda: maya.choose_qfn_quant(args(qfn_quant="q8_0"), {}, 8.0, 91.7))
        self.assertIn("cannot run on this GPU", out)
        self.assertIn("8 GB cannot hold the 9.0 GB", out)
        self.assertEqual(maya.choose_qfn_quant(args(qfn_quant="ud-q4_k_xl"), {}, 8.0, 62.0), "ud-q4_k_xl")

    def test_ram_tier(self):
        self.assertEqual(maya.qfn_ram_budget(91.7), 71)
        self.assertEqual(maya.qfn_ram_budget(20.0), 8)

    def test_qfn_fit_contexts_stay_offered_by_the_setup(self):
        for ctx in (maya.QF.DEFAULT_CONTEXT, maya.QF.SPLIT_CONTEXT, maya.QF.SCALED_CONTEXT):
            self.assertIn(ctx, maya.QFN_CONTEXTS)                  # (choose_context offers only QFN_CONTEXTS -
                                                                   #  including the scaled-down one it recommends)


def cpu(sockets=1, cores=8):
    """A patched qfn_cpu_info: the tests never read the TEST machine's /sys (this repo's own dev box has two
    sockets; the one-socket default keeps the TOPOLOGY keys out of the configs that should not carry them)."""
    return patch.object(maya, "qfn_cpu_info", return_value={"sockets": sockets, "cores": cores})


class Config(unittest.TestCase):
    def test_the_measured_amd_settings(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            models, d = t / "models", t / "models" / "Qwen3.8-Flash-Next-UD-Q4_K_XL"
            files = [t / "gguf" / n for n in maya.qfn_entry("ud-q4_k_xl")["files"]]
            pc = {"backend": "hip", "gpus": [{"index": 0, "arch": "gfx1201", "vram_gb": 15.9}]}
            with cpu(), patch.object(maya, "ROOT", t), patch.object(maya, "mem_gb", return_value=(91.7, 80.0)), \
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
            self.assertEqual(a[a.index("--resident-budget-gib") + 1], "71")   # (91.7 GB: the tier as measured)
            self.assertEqual(a[a.index("--max-context") + 1], "65536")
            self.assertIn("--mtp", a)
            self.assertEqual(cfg["env"]["SAGUARO_MMQ_KQ"], "1")
            self.assertEqual((cfg["backend"], cfg["gpu"]), ("hip", [0]))
            self.assertEqual(cfg["installer"]["qwen_quant"], "ud-q4_k_xl")
            self.assertEqual(cfg["installer"]["family"], "qfn")         # (its own family since the identity split)
            self.assertEqual(Q.family_of(cfg), "qfn")
            self.assertEqual(cfg["sampling"], {"temperature": 1.0, "top_p": 0.95, "top_k": 20})
            # U6: AMD's config is the measured one plus the GENERIC additions ONLY (no T4 tuning, no split keys,
            # no NUMA keys on a one-socket CPU), and the conversation cache every serve host gets.
            self.assertEqual(cfg["env"], {"SAGUARO_NO_LARGEPAGES": "1", "SAGUARO_YARN_ELASTIC": "1",
                                          "SAGUARO_MMQ_KQ": "1", **maya.QF.GENERIC_ENV})
            self.assertIn("--conversation-cache-mib", a)                # (auto: the shared park pool)
            self.assertIn("--conversation-cache-slots", a)              # (auto: 64)
            self.assertEqual(a[a.index("--conversation-cache-min-tokens") + 1], "1024")
            self.assertNotIn("--pool-workers", a)                       # (one socket: the engine's own default)
            self.assertNotIn("--prompt-cache", a)                       # (the live T4 box's per-quant tuning)
            self.assertNotIn("--trim-stage-weights", a)                 # (one card: no split, no stage keys)

    def test_the_cyber_frost_config(self):
        """Its config keeps the name it always had and takes the drafter from inside the pack (the file's own NextN
        block); U7: on this one AMD card the pictures are OFF - a single card at parallel 1 has no batch slots to
        give the encoder room from and --vram-elastic is NVIDIA-only, so the vision entry write_config_qfn is
        handed does not go into a config whose server could only start the encoder into a full card."""
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            models, d = t / "models", t / "models" / "CYBER-FROST-3.8-PS"
            files = [t / "gguf" / n for n in maya.qfn_entry(Q.CYBER_FROST_QUANT)["files"]]
            pc = {"backend": "hip", "gpus": [{"index": 0, "arch": "gfx1201", "vram_gb": 15.9}]}
            vision = {"exe": str(maya.qfn_exe().parent / "vision-qfn"),
                      "mmproj": str(d / "vision" / "mmproj-F16.gguf"), "model": str(files[0]), "gpu": True,
                      "max_tokens": 1024, "elastic": True, "elastic_mib": 1536, "elastic_idle_s": 30,
                      "elastic_keep_s": 600}
            with cpu(), patch.object(maya, "ROOT", t), patch.object(maya, "mem_gb", return_value=(91.7, 80.0)), \
                    patch.object(maya, "write_run_script", return_value=t / "run.sh"), \
                    patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
                p = maya.write_config_qfn(args(), pc, {}, Q.CYBER_FROST_QUANT, files, d / "pack",
                                          d / "pack" / "mtp", 65536, models, d, {}, vision)
            cfg = json.loads(p.read_text())
            self.assertEqual(p.name, "maya-cyber-frost-hip.json")
            a = cfg["args"]
            self.assertEqual(a[a.index("--native") + 1], str(files[0]))
            self.assertEqual(a[a.index("--mtp") + 1], str(d / "pack" / "mtp"))
            self.assertNotIn("--vision", a)                          # (no lend path on one AMD card: off)
            self.assertNotIn("vision", cfg)
            self.assertEqual(cfg["installer"]["family"], "qfn")
            self.assertEqual(cfg["installer"]["qwen_quant"], "cyber-frost-q5_k_m")
            self.assertEqual(Q.family_of(cfg), "qfn")
            self.assertEqual(cfg["sampling"], {"temperature": 1.0, "top_p": 0.95, "top_k": 20})   # (its own)


class Drafter(unittest.TestCase):
    """CYBER-FROST step 8 on QFN: the drafter built from the file's own NextN block (tools/mtp_from_gguf.py) into
    <pack>/mtp - dense.txt, dense.bin, experts.bin, the layout saguaro-qfn's --mtp loader reads - kept while its
    tools and the model file stand; and the pictures step (step 7, every quant: HEAD's qfn_vision_step)."""

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
        """CYBER-FROST's pictures on the merged tree: step 7 for every quant - --no-vision and a declined download
        give text only, and the entry is HEAD's qfn_vision_step's (vision-qfn, the QFN manifest's mmproj)."""
        with tempfile.TemporaryDirectory() as t:
            t, d = Path(t), Path(t) / "models" / "CYBER-FROST-3.8-PS"
            files = [t / "CYBER-FROST-3.8-PS-GUFF-Q5_K_M.gguf"]
            with patch.object(maya, "ok", lambda *x: None), patch.object(maya, "step", lambda *x: None):
                self.assertIsNone(maya.qfn_vision_step(args(no_vision=True), pc_hip(), {}, t / "llama", d, files))
            quiet = [patch.object(maya, n, lambda *x: None) for n in ("ok", "warn", "step")]
            for p in quiet:
                p.start()
            try:
                with patch.object(maya, "offer_download", lambda *x: False):
                    self.assertIsNone(maya.qfn_vision_step(args(), pc_hip(), {}, t / "llama", d, files))
                with patch.object(maya, "file_incomplete", lambda *x: None), \
                        patch.object(maya, "sha256_checked", lambda *x: True), \
                        patch.object(maya, "compile_vision_qfn", return_value=True):
                    v = maya.qfn_vision_step(args(), pc_hip(), {"archs": ["gfx1201"]}, t / "llama", d, files)
            finally:
                for p in quiet:
                    p.stop()
        self.assertEqual(v, {"exe": str(maya.qfn_vision_exe(True)),
                             "mmproj": str(d / "vision" / "mmproj-Qwen3.8-Flash-Next-BF16.gguf"),
                             "model": str(files[0]), "gpu": True, **maya.QFN_VISION})


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
        """A Mac (Metal) CYBER-FROST config is not rewritten to saguaro-qfn: --setup would refuse to build the
        engine there, so migrating strands it - it keeps its old-engine files and runs on saguaro-glm.  The same on
        Windows, whatever the backend (saguaro-qfn is not built there).  NVIDIA is offered on Linux since T2
        (qfn_fit lays its cards out), so an NVIDIA config - its backend missing or set - migrates like AMD's."""
        def old(backend):
            cfg = json.loads(json.dumps(self.OLD))
            cfg["backend"] = backend
            if backend is None:
                cfg.pop("backend")
            return cfg

        for backend in ("metal",):
            cfg = old(backend)
            self.assertFalse(maya.migrate_cyber_frost(cfg), str(backend))
            self.assertEqual(cfg["installer"]["family"], "qwen")            # (untouched)
            self.assertEqual(cfg["exe"], self.OLD["exe"])                   # (still the GLM engine)
            self.assertIn("--ple-gguf", cfg["args"])                        # (its Qwen-path arguments stand)
        with patch.object(maya, "WIN", True):
            for backend in ("cuda", "hip", None):
                cfg = old(backend)
                self.assertFalse(maya.migrate_cyber_frost(cfg), str(backend))
                self.assertEqual(cfg["installer"]["family"], "qwen")        # (untouched on Windows too)
        for backend in ("cuda", None):                                      # (Linux CUDA: offered, so it migrates;
            cfg = old(backend)                                              #  a missing backend is the CUDA default)
            self.assertTrue(maya.migrate_cyber_frost(cfg), str(backend))
            self.assertEqual(cfg["installer"]["family"], "qfn")
            self.assertTrue(cfg["exe"].endswith("qfn/saguaro-qfn"))         # (its engine now: saguaro-qfn's)

    def test_pack_still_the_other_engines_blocks_the_start(self):
        """Finding 2: a CYBER-FROST config now on saguaro-qfn is refused while its pack is still the other engine's -
        durable, so a --check that rewrote the config (building nothing) or a hand edit cannot start it against a
        GLM-format pack that only shares the "index v3" header.  It clears once the pack names the QFN pack tools."""
        with tempfile.TemporaryDirectory() as t:
            pack = Path(t)
            cfg = {"backend": "hip", "args": ["--pack", str(pack)],
                   "installer": {"family": "qfn", "qwen_quant": Q.CYBER_FROST_QUANT}}
            with patch.object(maya.S, "source_hash", return_value="qfn-tools"):
                self.assertEqual(maya.cyber_pack_pending(cfg), pack)        # (no sources.json: nothing built here)
                (pack / maya.PACK_SOURCES).write_text(json.dumps({"tools": "glm-tools"}))
                self.assertEqual(maya.cyber_pack_pending(cfg), pack)        # (the Qwen path's pack: still refused)
                (pack / maya.PACK_SOURCES).write_text(json.dumps({"tools": "qfn-tools"}))
                self.assertIsNone(maya.cyber_pack_pending(cfg))            # (qfn_pack_step built it: it may start)
                (pack / maya.PACK_SOURCES).write_text(json.dumps({"tools": "glm-tools"}))
                # (a QFN quant that is not CYBER-FROST, and the old engine, are not this guard's business)
                self.assertIsNone(maya.cyber_pack_pending({**cfg, "installer": {"family": "qfn",
                                                                                "qwen_quant": "ud-q4_k_xl"}}))
                self.assertIsNone(maya.cyber_pack_pending({**cfg, "installer": {"family": "qwen",
                                                                                "qwen_quant": "q5_k_m"}}))


class NvidiaConfig(unittest.TestCase):
    def test_the_live_t4_settings_on_four_tesla_t4s(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            models, d = t / "models", t / "models" / "Qwen3.8-Flash-Next-UD-Q4_K_XL"
            files = [t / "gguf" / n for n in maya.qfn_entry("ud-q4_k_xl")["files"]]
            pc = {"backend": "cuda", "gpus": [{"index": i, "name": "Tesla T4", "vram_gb": 15.0, "arch": "75"}
                                              for i in range(4)]}
            with cpu(2, 20), patch.object(maya, "ROOT", t), patch.object(maya, "mem_gb", return_value=(252.0, 240.0)), \
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
            self.assertNotIn("--resident-budget-gib", a)      # (252 GB: the page cache holds the experts, as live)
            self.assertIn("--trim-stage-weights", a)
            self.assertEqual(a[a.index("--prefill-pipe") + 1], "1536")
            self.assertEqual(a[a.index("--pool-workers") + 1], "19")     # (2 x 10 cores minus the host's)
            self.assertEqual(cfg["env"]["SAGUARO_SM75_PERF"], "1")                # (every card is compute cap 7.5)
            self.assertEqual(cfg["env"]["SAGUARO_KQL"], "3")
            self.assertEqual(cfg["env"]["SAGUARO_MMQ_KQ"], "1")
            self.assertEqual(cfg["env"]["SAGUARO_ELASTIC_SLOTS_GIVE"], "1")       # (what makes the engine lend the
                                                                                  #  picture encoder CUDA0's room)
            import serve.server as SV
            self.assertEqual(SV.engine_kind(cfg), "batch")
            self.assertEqual(Q.family_of(cfg), "qfn")

    def test_a_plain_nvidia_card_gets_no_t4_tuning_or_a_split(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            models, d = t / "models", t / "models" / "Qwen3.8-Flash-Next-UD-Q4_K_XL"
            files = [t / "gguf" / n for n in maya.qfn_entry("ud-q4_k_xl")["files"]]
            pc = {"backend": "cuda", "gpus": [{"index": 0, "name": "RTX A5000", "vram_gb": 24.0, "arch": "86"}]}
            with cpu(), patch.object(maya, "ROOT", t), patch.object(maya, "mem_gb", return_value=(128.0, 100.0)), \
                    patch.object(maya, "write_run_script", return_value=t / "run.sh"), \
                    patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
                p = maya.write_config_qfn(args(backend="cuda"), pc, {}, "ud-q4_k_xl", files, d / "pack", None,
                                          65536, models, d, {})
            cfg = json.loads(p.read_text())
            self.assertNotIn("layer_split", cfg)
            for k in cfg["env"]:
                self.assertFalse(k.startswith("SAGUARO_SM75") or k in ("SAGUARO_KQL", "SAGUARO_BF16_TC"), k)
            self.assertNotIn("SAGUARO_SLOT_ADMIT", cfg["env"])         # (no split: the pipelined reads stay out)
            self.assertNotIn("SAGUARO_NUMA_PLACE", cfg["env"])         # (one socket: no NUMA keys)
            self.assertNotIn("--pool-workers", cfg["args"])
            self.assertNotIn("--prompt-cache", cfg["args"])            # (the per-quant args come from the T4 box)
            self.assertIn("SAGUARO_ELASTIC_SLOTS", cfg["env"])         # (GENERIC: the lend works on one card too)
            self.assertNotIn("--prefill-pipe", cfg["args"])
            self.assertEqual(cfg["installer"]["family"], "qfn")


class LiveT4Parity(unittest.TestCase):
    """U6 step 5: what a fresh `./saguaro.sh --setup --model qwen3.8-flash-next` writes on the live machine's
    shape (4x Tesla T4, 2-socket 10+10-core CPU, 252 GB) is the live UD-Q4 config (./ref/live-ud-q4.json): the
    env and args equal bar the paths, the context and these intended differences:
      - env SAGUARO_BATCH_READS_OVERTAKE, SAGUARO_PARK_STAGING_MIB: read nowhere in engine or server, not written;
      - env SAGUARO_GDN_KEYHEAD: the ref's value is "<removed>" (./ref/live-ud-q4.json:59); the setup writes the
        documented sm_75 opt-in, 1 (prefill/kernels.cu: "SAGUARO_GDN_KEYHEAD=1 also takes it on sm_75");
      - --max-context: the split recommends 262144, the live config was raised to 1048576;
      - --resident-budget-gib: neither has one (252 GB: the page cache holds the experts);
      - the file paths, the port, the log, the installer block: this machine's, not mangonel's."""

    LIVE_ENV = {  # (./ref/live-ud-q4.json's env, copied: ref/ is not in the repository)
        "SAGUARO_NO_LARGEPAGES": "1", "SAGUARO_SM75_PERF": "1", "SAGUARO_SM75_STACK": "1", "SAGUARO_KQL": "3",
        "SAGUARO_SM75_Q8G": "1", "SAGUARO_GDN_KEYHEAD": "1", "SAGUARO_ARGMAX_ROWS_SM70": "1",
        "SAGUARO_BF16_TC": "1", "SAGUARO_BF16_TC_SKIP": "none", "SAGUARO_MMQ_KQ": "1", "SAGUARO_STAGE_SERVE": "1",
        "SAGUARO_NUMA_PLACE": "1", "SAGUARO_PARALLEL_LOAD": "1", "SAGUARO_ELASTIC_SLOTS": "1",
        "SAGUARO_ELASTIC_SLOTS_GIVE": "1", "SAGUARO_PIPED_MTP": "1", "SAGUARO_PIPED_MTP_DRAFTS": "2",
        "SAGUARO_SLOT_ADMIT": "1", "SAGUARO_P2P_HANDOFF": "1", "SAGUARO_SLOT_ADMIT_MULTI": "1",
        "SAGUARO_SHORT_READS": "1", "SAGUARO_READ_YIELD": "1", "SAGUARO_ELASTIC_IDLE": "1",
        "SAGUARO_SLOT_ADMIT_CHUNK": "4096", "SAGUARO_BATCH_READS": "1", "SAGUARO_BATCH_READS_FAST": "512",
        "SAGUARO_BATCH_READS_FAST_SHRINK": "512", "SAGUARO_BATCH_READS_DECODE_WINDOWS": "0",
        "SAGUARO_BATCH_READS_DECODE_SHARE": "0.3", "SAGUARO_BATCH_READS_MAX": "8",
        "SAGUARO_BATCH_READS_OVERTAKE": "0", "SAGUARO_FAST_LANE": "1", "SAGUARO_FAST_LANE_MAX": "96",
        "SAGUARO_PARK_STAGING_MIB": "0", "SAGUARO_NUMA_POOL": "0", "SAGUARO_ELASTIC_IDLE_S": "300",
        "SAGUARO_READ_YIELD_CHUNK": "4096", "SAGUARO_YARN_ELASTIC": "1", "SAGUARO_PIPED_MTP_MAX": "4",
        "SAGUARO_CACHE_MESSAGE_BOUNDARY": "1", "SAGUARO_SLOT_GRID": "6", "SAGUARO_SLOT_COPY_DIRECT": "1",
        "SAGUARO_PARK_FAST": "1", "SAGUARO_PARALLEL_LOAD_REGISTER": "overlap", "SAGUARO_SELECT_SM75": "1",
        "SAGUARO_MMQ_DIRECT": "1", "SAGUARO_MMQ_KTAIL": "1", "SAGUARO_MMQ_SWIGLU_Q8": "1"}
    LIVE_ARGS = ["--pack", "/opt/llm/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/pack", "--native",
                 "/opt/llm/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
                 "--expert-profile", "/opt/llm/var/Qwen3.8-Flash-Next-UD-Q4_K_XL/expert-profile.bin",
                 "--expert-profile-save", "/opt/llm/var/Qwen3.8-Flash-Next-UD-Q4_K_XL/expert-profile.bin",
                 "--expert-cache", "auto", "--prefill", "auto", "--spec", "3", "--spec-min-p", "0.7", "--mtp",
                 "/opt/llm/models/Qwen3.8-Flash-Next-MTP-Lamina", "--max-context", "1048576", "--kv", "int8",
                 "--kv-resident", "32768", "--vision", "--vram-reserve-mib", "700", "--conversation-cache-mib",
                 "auto", "--conversation-cache-slots", "auto", "--prompt-cache", "64", "--session-dir",
                 "/opt/llm/var/Qwen3.8-Flash-Next-UD-Q4_K_XL/sessions", "--pool-workers", "19",
                 "--trim-stage-weights", "--prefill-pipe", "1536", "--conversation-cache-min-tokens", "1024",
                 "--rope-scaling", "yarn", "--rope-scale", "4", "--yarn-orig-ctx", "262144"]
    PATH_FLAGS = ("--pack", "--native", "--expert-profile", "--expert-profile-save", "--mtp", "--session-dir")
    BOOL_FLAGS = ("--vision", "--trim-stage-weights")

    def written(self, t: Path):
        models, d = t / "models", t / "models" / "Qwen3.8-Flash-Next-UD-Q4_K_XL"
        files = [t / "gguf" / n for n in maya.qfn_entry("ud-q4_k_xl")["files"]]
        pc = {"backend": "cuda", "gpus": [{"index": i, "name": "Tesla T4", "vram_gb": 15.0, "arch": "75"}
                                          for i in range(4)]}
        vision = {"exe": "vision-qfn", "mmproj": "mmproj.gguf", "model": str(files[0]), "gpu": True,
                  **maya.QFN_VISION}
        with cpu(2, 20), patch.object(maya, "ROOT", t), patch.object(maya, "mem_gb", return_value=(252.0, 240.0)), \
                patch.object(maya, "write_run_script", return_value=t / "run.sh"), \
                patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
            p = maya.write_config_qfn(args(backend="cuda"), pc, {}, "ud-q4_k_xl", files, d / "pack",
                                      models / "Qwen3.8-Flash-Next-MTP-Lamina", 262144, models, d, {}, vision)
        return json.loads(p.read_text())

    @staticmethod
    def flag_pairs(argv):
        out, i = [], 0
        while i < len(argv):
            if argv[i] in LiveT4Parity.BOOL_FLAGS:
                out.append((argv[i], None))
                i += 1
            else:
                out.append((argv[i], argv[i + 1]))
                i += 2
        return out

    def test_the_written_env_is_the_live_env(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = self.written(Path(t))
        want = {k: v for k, v in self.LIVE_ENV.items() if k not in ("SAGUARO_BATCH_READS_OVERTAKE",
                                                                    "SAGUARO_PARK_STAGING_MIB")}
        self.assertEqual(cfg["env"], want)
        self.assertEqual(cfg["env"]["SAGUARO_GDN_KEYHEAD"], "1")       # ("<removed>" in the ref: see the class note)

    def test_the_written_args_are_the_live_args(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = self.written(Path(t))

        def machine_free(pairs):     # (the paths and --max-context are the intended differences, listed above)
            return {(f, v) for f, v in pairs if f not in self.PATH_FLAGS and f != "--max-context"}
        self.assertEqual(machine_free(self.flag_pairs(cfg["args"])), machine_free(self.flag_pairs(self.LIVE_ARGS)))
        a = cfg["args"]
        self.assertEqual(a[a.index("--max-context") + 1], "262144")    # (the split's recommendation; live: 1M)
        self.assertNotIn("--resident-budget-gib", a)                   # (252 GB: neither config pins a tier)
        self.assertEqual(a[a.index("--pool-workers") + 1], "19")       # (computed here: 2 x 10 cores minus one)

    def test_the_ref_copy_still_matches_the_live_file_when_it_is_here(self):
        """ref/ is a worktree-local folder (git-excluded): when the live file IS present next to the tests, the
        embedded copy above must be it, save the two dead keys and the removed GDN value."""
        ref = ROOT / "ref" / "live-ud-q4.json"
        if not ref.is_file():
            self.skipTest("ref/ is not in the checkout")
        live = json.loads(ref.read_text())
        env = {k: v for k, v in live["env"].items()}
        env["SAGUARO_GDN_KEYHEAD"] = "1"                               # ("<removed>" -> the documented 1)
        self.assertEqual(env, self.LIVE_ENV)
        self.assertEqual(live["args"], self.LIVE_ARGS)


class Layouts1To4T4(unittest.TestCase):
    """U7 step 3, judged by the v1.2.1-B policy: one case per (1, 2, 3, 4 Tesla T4) x (UD-Q4_K_XL, Q8_0,
    CYBER-FROST Q5_K_M) x (32, 64, 96, 256 GB RAM), as a fresh setup writes it on a single-socket CPU (no NUMA
    keys, no --pool-workers).  The T4s themselves are always the comfortable card (15.0 = ECC on, 16.0 is the
    other row, in test_qfn_fit's matrix), so the RAM decides: 96 and 256 GB get the layout as laid out - the
    split, the context, the K/V window, the RAM tier, the answers-at-once value, the pipeline, the class env and
    the drafting; 32 and 64 GB fit too (v1.2.1-B refused nothing here), slowly: the written config comes scaled
    down (32K, --kv-resident 4096, drafting off and recorded).  4x T4 + 256 GB UD-Q4 is U6's pinned
    live-equivalent config.  Nothing of the matrix is refused: 32 GB is over the RAM minimum, the T4 over the
    card's (9.0 GB even at Q8_0)."""

    QUANTS = ["ud-q4_k_xl", "q8_0", "cyber-frost-q5_k_m"]
    RAMS = [32.0, 64.0, 96.0, 256.0]
    SLOW_RAM = [32.0, 64.0]                                      # (under the comfortable MIN_RAM_GB: warn + confirm)
    SPLIT = {"ud-q4_k_xl": "11,24,36", "q8_0": "10,22,35", "cyber-frost-q5_k_m": "12,24,36"}

    def pc(self, n):
        return {"backend": "cuda", "gpus": [{"index": i, "name": "Tesla T4", "vram_gb": 15.0, "arch": "75"}
                                            for i in range(n)]}

    def written(self, t: Path, n, quant, ram, sockets=1, cores=8, vision=None, **kw):
        models, d = t / "models", t / "models" / maya.qfn_entry(quant)["name"]
        files = [t / "gguf" / nm for nm in maya.qfn_entry(quant)["files"]]
        ctx = maya.QF.plan(self.pc(n)["gpus"], ram, quant)["context"]
        with cpu(sockets, cores), patch.object(maya, "ROOT", t), \
                patch.object(maya, "mem_gb", return_value=(ram, max(8.0, ram - 12.0))), \
                patch.object(maya, "write_run_script", return_value=t / "run.sh"), \
                patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
            p = maya.write_config_qfn(args(backend="cuda", **kw), self.pc(n), {}, quant, files, d / "pack",
                                      d / "pack" / "mtp", ctx, models, d, {}, vision)
        return json.loads(p.read_text())

    def test_the_thin_ram_rows_are_slow_not_refused(self):  # (v1.2.1-B: the old "96 GB of RAM" refusals become
        for n, quant in ((n, q) for n in (1, 2, 3, 4) for q in self.QUANTS):   #  the warn rows)
            for ram in self.SLOW_RAM:
                with self.subTest(cards=n, quant=quant, ram=ram):
                    lay = maya.QF.plan(self.pc(n)["gpus"], ram, quant)
                    self.assertTrue(lay["fits"], lay["reason"])
                    self.assertTrue(lay["slow"])
                    self.assertIn(f"{ram:.0f} GB of RAM", lay["slow_why"])     # (the thin tier is named)
                    experts, room = maya.QF.quant_gib(quant), ram + 15.0 * n
                    if room < experts:                                         # (and only where the experts really
                        self.assertIn("GiB more than these", lay["slow_why"])  #  are over the layout: Q8_0 always,
                        self.assertIn(f"~{room:.0f} GiB", lay["slow_why"])     #  CF up to two cards)

    def test_the_slow_rows_write_the_scaled_layout(self):  # (v1.2.1-B: the config is still written for these -
        for n, quant, ram in ((n, q, r) for n in (1, 2, 3, 4)          #  as it stands today, only scaled down:
                               for q in self.QUANTS for r in self.SLOW_RAM):   #  32K, --kv-resident 4096, drafting
            with self.subTest(cards=n, quant=quant, ram=ram), tempfile.TemporaryDirectory() as t:             #  off
                cfg = self.written(Path(t), n, quant, ram)
                a = cfg["args"]
                self.assertEqual(cfg["gpu"], list(range(n)))
                self.assertEqual(a[a.index("--max-context") + 1], "32768")
                self.assertEqual(a[a.index("--kv-resident") + 1], "4096")
                self.assertEqual(a[a.index("--mtp-max-t") + 1], "1")   # (off: said, and recorded, no question)
                self.assertEqual(cfg["installer"]["mtp_calibration"]["kept"], "none")
                self.assertEqual(cfg["installer"]["mtp_calibration"]["skipped"], "slow-pc")
                self.assertIn("--mtp", a)                              # (the layer stays loaded either way)
                self.assertEqual(a[a.index("--resident-budget-gib") + 1], str(int(ram - 20)))   # (12 / 44 GiB)
                even = {2: "24", 3: "16,32"}
                self.assertEqual(cfg.get("layer_split"),
                                 even.get(n) or (self.SPLIT[quant] if n == 4 else None))

    def test_the_written_configs(self):
        even = {2: "24", 3: "16,32"}
        for n, quant, ram in ((n, q, r) for n in (1, 2, 3, 4) for q in self.QUANTS
                              for r in self.RAMS if r not in self.SLOW_RAM):
            with self.subTest(cards=n, quant=quant, ram=ram), tempfile.TemporaryDirectory() as t:
                cfg = self.written(Path(t), n, quant, ram)
                a = cfg["args"]
                self.assertEqual(cfg["gpu"], list(range(n)))
                self.assertEqual(cfg["installer"]["qwen_quant"], quant)
                # the split: none on one card, balanced on 2/3 by the 48 layers and the cards' VRAM, the live
                # server's own values on four T4s (CYBER-FROST never measured: the even split, pipe 1)
                self.assertEqual(cfg.get("layer_split"), even.get(n) or (self.SPLIT[quant] if n == 4 else None),
                                 f"{n} {quant}")
                self.assertEqual(a[a.index("--max-context") + 1], "65536" if n == 1 else "262144")
                self.assertEqual(a[a.index("--kv-resident") + 1], "32768")
                # the RAM tier (U6's rule, per quant's bytes): at 96 GB no quant's experts fit the page cache
                # beside the 20 GB spare, so all three pin the 76 GiB tier; at 256 GB all leave it to the cache
                tier = 76 if ram == 96.0 else None
                if tier:
                    self.assertEqual(a[a.index("--resident-budget-gib") + 1], str(tier), f"{n} {quant} {ram}")
                else:
                    self.assertNotIn("--resident-budget-gib", a, f"{n} {quant} {ram}")
                # answers at once: one card stays at 1 (no key - the loss the one-card box measured), 2 on a split
                self.assertEqual(cfg.get("parallel"), 2 if n > 1 else None)
                # the pipeline: the split's own keys; the live pipe rate only on the four T4s that were measured
                self.assertEqual("--trim-stage-weights" in a, n > 1)
                if n > 1:
                    pipe = {"ud-q4_k_xl": "1536", "q8_0": "6144"}.get(quant, "1") if n == 4 else "1"
                    self.assertEqual(a[a.index("--prefill-pipe") + 1], pipe)
                # drafting: the draft layer stays in (off only when the person says so)
                self.assertIn("--mtp", a)
                self.assertNotIn("--mtp-max-t", a)
                # the env by class: GENERIC everywhere, the T4 tuning (every card sm_75), the split's on a split,
                # nothing of a multi-socket CPU on this one
                self.assertEqual({k: v for k, v in cfg["env"].items() if k in maya.QF.GENERIC_ENV},
                                 maya.QF.GENERIC_ENV)
                self.assertEqual("SAGUARO_SM75_PERF" in cfg["env"], True)
                self.assertEqual("SAGUARO_SLOT_ADMIT" in cfg["env"], n > 1)
                self.assertEqual("SAGUARO_NUMA_PLACE" in cfg["env"], False)
                self.assertNotIn("--pool-workers", a)
                self.assertEqual(cfg["env"].get("SAGUARO_BATCH_WRAP_SPILL"), "1" if (n > 1 and quant == "q8_0")
                                 else None)

    def test_a_two_socket_cpu_adds_only_the_topology_class(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = self.written(Path(t), 2, "ud-q4_k_xl", 256.0, sockets=2, cores=20)
        self.assertEqual(cfg["env"]["SAGUARO_NUMA_PLACE"], "1")
        self.assertEqual(cfg["env"]["SAGUARO_NUMA_POOL"], "0")
        self.assertEqual(cfg["args"][cfg["args"].index("--pool-workers") + 1], "19")
        self.assertEqual(cfg["layer_split"], "24")          # (the split itself: the CPU does not touch it)

    def test_four_t4s_256_gb_ud_q4_is_the_pinned_live_config(self):
        """U6 pinned the live UD-Q4 machine's env/args (LiveT4Parity, 252 GB); the same shape at 256 GB
        (its experts all sit in the page cache either way) writes exactly that."""
        vision = {"exe": "vision-qfn", "mmproj": "m.gguf", "model": "s.gguf", "gpu": True, **maya.QFN_VISION}
        with tempfile.TemporaryDirectory() as t:
            cfg = self.written(Path(t), 4, "ud-q4_k_xl", 256.0, sockets=2, cores=20, vision=vision)
        want = {k: v for k, v in LiveT4Parity.LIVE_ENV.items() if k not in ("SAGUARO_BATCH_READS_OVERTAKE",
                                                                            "SAGUARO_PARK_STAGING_MIB")}
        self.assertEqual(cfg["env"], want)
        machine_free = lambda pairs: {(f, v) for f, v in pairs
                                      if f not in LiveT4Parity.PATH_FLAGS and f != "--max-context"}
        self.assertEqual(machine_free(LiveT4Parity.flag_pairs(cfg["args"])),
                         machine_free(LiveT4Parity.flag_pairs(LiveT4Parity.LIVE_ARGS)))
        self.assertNotIn("--resident-budget-gib", cfg["args"])   # (256 GB: the file cache is the tier, as live)
        self.assertEqual(cfg["layer_split"], "11,24,36")
        self.assertEqual(cfg["args"][cfg["args"].index("--prefill-pipe") + 1], "1536")
        self.assertEqual(cfg["parallel"], 2)                     # (fresh: 2; the live box raised it to 8)


class PicturesOnEveryLayout(unittest.TestCase):
    """U7 step 4: pictures on every 1-4 T4 layout.  The elastic encoder starts on VRAM the engine lends it and
    there are exactly two lend paths: the batch slots' give path (answers at once 2 or more -> --batch N; the
    engine then says "INFO vision_lend=1", generate.cpp:13119-13164, and the server sends VLEND,
    serve/server.py:3105-3139), and the VRAM command on a --vram-elastic expert cache (the config's
    "vram_elastic": true -> engines/qfn/serve/server.py engine_args:3762-3768; NVIDIA only,
    expert_cache.cpp:301, and the engine answers VRAM only with it, generate.cpp:12738-12769).  A one-card config
    (parallel 1, no batch slots) writes the second; four T4s keep U6's give-path config unchanged; a server whose
    engine says neither in its INFO refuses a picture plainly instead of starting the encoder onto a full card
    (the SIGKILL of U6's ./ref/vision-fail.log)."""

    VISION = {"exe": "vision-qfn", "mmproj": "m.gguf", "model": "s.gguf", "gpu": True}

    def setUp(self):
        self.VISION = {**self.VISION, **maya.QFN_VISION}

    def written(self, t: Path, n, **kw):
        models, d = t / "models", t / "models" / "Qwen3.8-Flash-Next-UD-Q4_K_XL"
        files = [t / "gguf" / nm for nm in maya.qfn_entry("ud-q4_k_xl")["files"]]
        gpus = [{"index": i, "name": "Tesla T4", "vram_gb": 15.0, "arch": "75"} for i in range(n)]
        with cpu(), patch.object(maya, "ROOT", t), patch.object(maya, "mem_gb", return_value=(252.0, 240.0)), \
                patch.object(maya, "write_run_script", return_value=t / "run.sh"), \
                patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
            p = maya.write_config_qfn(args(backend="cuda", **kw), {"backend": "cuda", "gpus": gpus}, {},
                                      "ud-q4_k_xl", files, d / "pack", models / "MTP", 262144 if n > 1 else 65536,
                                      models, d, {}, dict(self.VISION))
        return json.loads(p.read_text())

    def test_one_t4_gets_the_vram_elastic_lend_path(self):
        for kw in ({}, {"parallel": 1}):                     # (the fresh default and a plain --parallel 1)
            with self.subTest(**kw), tempfile.TemporaryDirectory() as t:
                cfg = self.written(Path(t), 1, **kw)
                self.assertIs(cfg["vram_elastic"], True)     # (engine_args:3762-3768 turns it into --vram-elastic)
                self.assertEqual(cfg["vision"], self.VISION)
                self.assertIn("--vision", cfg["args"])
                self.assertNotIn("parallel", cfg)            # (still one conversation at a time)
                self.assertNotIn("--vram-elastic", cfg["args"])   # (the flag comes at the start, from the key)

    def test_splits_keep_the_give_path_and_no_vram_elastic(self):
        for n in (2, 3, 4):
            with self.subTest(cards=n), tempfile.TemporaryDirectory() as t:
                cfg = self.written(Path(t), n)
                self.assertEqual(cfg["parallel"], 2)         # (--batch 2 -> the batch slots the give path needs)
                self.assertNotIn("vram_elastic", cfg)        # (the give path lends; U6's 4-card config as left)
                self.assertEqual(cfg["vision"], self.VISION)
                self.assertEqual(cfg["env"]["SAGUARO_ELASTIC_SLOTS_GIVE"], "1")   # (GENERIC: the give keys, all)

    def test_a_two_card_setup_at_parallel_one_reads_text_only(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = self.written(Path(t), 2, parallel=1)
        self.assertNotIn("parallel", cfg)
        self.assertNotIn("vram_elastic", cfg)                # (--vram-elastic is off on a split: no lend path)
        self.assertNotIn("vision", cfg)
        self.assertNotIn("--vision", cfg["args"])

    def test_the_engine_driver_says_what_it_can_lend(self):
        from serve import qfn_engine as QE
        e = QE.StrataEngine.__new__(QE.StrataEngine)
        for info, want in (({}, False), ({"vram_elastic": 0}, False), ({"vram_elastic": "0"}, False),
                           ({"vram_elastic": 1}, True), ({"vision_lend": 1}, True), ({"vision_lend": 0}, False),
                           ({"vram_elastic": 0, "vision_lend": 1}, True)):
            e.info = info
            self.assertEqual(e.vision_room(), want, str(info))

    def test_the_server_refuses_a_picture_with_no_lend_path(self):
        import serve.server as SV
        from serve import qfn_engine as QE

        class Eng:
            info = {"vram_elastic": 0}                       # (the engine's own INFO: neither lend path)
            alive = lambda s: True
            vision_room = QE.StrataEngine.vision_room        # (the extracted predicate, the real one)

            def vram_reserve(s, mib):                        # (what _vision_up's fallback calls first)
                raise ValueError("VRAM needs an engine started with --vram-elastic (one NVIDIA GPU, --serve)")

        class EngLends(Eng):
            info = {"vram_elastic": 1}

            def vram_reserve(s, mib):
                return {"expert_cache_mib": 8000, "expert_cache_full_mib": 14000}

        class Vis:
            elastic, elastic_mib, elastic_idle_s = True, 1536, 30.0

            def __init__(self):
                self.started = False

            def alive(self):
                return False

            def restart(self):
                self.started = True

        class Svc:
            _vision_up, _vision_lends, _engine_ready = SV.Service._vision_up, SV.Service._vision_lends, \
                SV.Service._engine_ready

            def __init__(self, eng):
                self.engine, self.vision, self.vision_held = eng, Vis(), False

        with patch("time.sleep", lambda s: None), patch("sys.stdout", new=io.StringIO()):
            svc = Svc(Eng())
            with self.assertRaises(ValueError) as got:       # (no room: a plain error, no encoder, no SIGKILL)
                svc._vision_up()
            self.assertIn("--vram-elastic", str(got.exception))
            self.assertFalse(svc.vision.started)
            svc = Svc(EngLends())
            svc._vision_up()                                 # (the --vram-elastic path: reserve, then start)
            self.assertTrue(svc.vision.started)
            self.assertTrue(svc.vision_held)


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
        slow = maya.qfn_check_line("ud-q4_k_xl", maya.QF.plan([{"index": 0, "name": "RTX 4060", "vram_gb": 8.0,
                                                                "arch": "89"}], 96.0))
        self.assertIn("below the measured layout", slow)               # (v1.2.1-B: --check reports the slow row:
        self.assertIn("expect slow decode", slow)                      #  the cause, and the scale-down it would
        self.assertIn("the setup scales this down: drafting off (--mtp-max-t 1)", slow)   #  write - no question)
        self.assertIn("cannot run on this PC's RTX 4060",
                      maya.qfn_check_line("q8_0", maya.QF.plan([{"index": 0, "name": "RTX 4060", "vram_gb": 8.0,
                                                                 "arch": "89"}], 96.0, "q8_0")))

    def test_the_check_line_names_a_thin_ram_row_as_slow(self):
        t4 = [{"index": i, "name": "Tesla T4", "vram_gb": 15.0, "arch": "75"} for i in range(2)]
        line = maya.qfn_check_line("q8_0", maya.QF.plan(t4, 64.0, "q8_0"))    # (the plan's own example PC)
        self.assertIn("below the measured layout", line)
        self.assertIn("175 GiB", line)                                       # (the quant's numbers, said)
        self.assertIn("--kv-resident 4096 instead of 32768", line)
        self.assertNotIn("does not fit", line)


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

    def test_cards_below_the_minimum_are_refused_in_one_line(self):
        """v1.2.1-B: only a card that cannot hold the model at all, or RAM under the minimum, stops check_pc -
        in one plain line carrying the numbers, before anything else is asked."""
        tiny = [{"index": i, "name": "GeForce RTX 4060", "vram_gb": 4.0, "arch": "89", "driver": "550"}
                for i in (0, 1)]
        big = [{"index": 0, "name": "RTX A5000", "vram_gb": 24.0, "arch": "86", "driver": "550"}]
        with patch.object(maya.S, "gpus", return_value=tiny), patch.object(maya.S, "is_wsl", return_value=False), \
                patch.object(maya, "mem_gb", return_value=(96.0, 80.0)):
            out = refusal(lambda: maya.check_pc(args(backend="cuda"), "qfn"))    # (no --gpus: one line, no crash)
        # (4.2 GB: the 4060s are equal, so the refusal prices the 2-card split - half the model per card)
        self.assertIn("cannot run on this PC's GeForce RTX 4060: its 4 GB cannot hold the 4.2 GB", out)
        self.assertIn("50% of the layers", out)
        with patch.object(maya.S, "gpus", return_value=big + tiny), \
                patch.object(maya.S, "is_wsl", return_value=False), patch.object(maya, "mem_gb",
                                                                                 return_value=(96.0, 80.0)):
            out = refusal(lambda: maya.check_pc(args(backend="cuda", gpus="1"), "qfn"))  # (--gpus names a tiny one)
        self.assertIn("cannot run on this PC's GeForce RTX 4060", out)
        with patch.object(maya.S, "gpus", return_value=big), patch.object(maya.S, "is_wsl", return_value=False), \
                patch.object(maya, "mem_gb", return_value=(12.0, 8.0)):
            out = refusal(lambda: maya.check_pc(args(backend="cuda"), "qfn"))   # (and RAM under the minimum)
        self.assertIn("cannot run with 12 GB of RAM", out)
        self.assertIn("at least about 16 GB", out)

    def test_an_8_gb_pair_is_not_refused_by_the_fit_rule(self):
        """The 8 GB pair the old rule refused in check_pc is a warn row now (test_qfn_fit's matrix): the fit
        guards say nothing about it and the picker takes both cards."""
        small = [{"index": i, "name": "GeForce RTX 4060", "vram_gb": 8.0, "arch": "89"} for i in (0, 1)]
        self.assertIsNone(maya.QF.unsupported(small, 96.0))
        self.assertEqual([g["index"] for g in maya.QF.best_gpus(small)], [0, 1])
        lay = maya.QF.plan(small, 96.0)
        self.assertTrue(lay["fits"], lay["reason"])
        self.assertTrue(lay["slow"])


class PascalForQfn(unittest.TestCase):
    """v1.2.1-B step 3: MIN_CC 61 - a Tesla P40 is set up for QFN (24 GB + 32 GB is the hand-run box: warned
    as slow, never refused).  sm_6x / sm_70 build through engines/qfn's experimental pre-Turing path, which needs
    a CUDA 12.x toolkit - CUDA 13 dropped them (how the Inkling/GLM path has always handled Pascal): the setup
    prefers a fitting 12.x and, with only 13 installed, says so plainly.  The sm_75 class rule (U6) keeps the
    T4 tuning off it."""

    P40 = {"index": 0, "name": "Tesla P40", "vram_gb": 24.0, "arch": "61", "driver": "550"}

    def patches(self, nvcc):
        return [patch.object(maya.S, "gpus", return_value=[dict(self.P40)]),
                patch.object(maya.S, "is_wsl", return_value=False),
                patch.object(maya.S, "cpu_info", return_value=("CPU", True, True)),
                patch.object(maya.S, "out", return_value="g++ (test) 13.0"),
                patch.object(maya, "oomd_active", return_value=False),
                patch.object(maya, "mem_gb", return_value=(32.0, 20.0)),
                patch.object(maya, "find_nvcc", return_value=nvcc),
                patch.object(maya.shutil, "which", side_effect=lambda c, *r: "/usr/bin/g++" if c == "g++" else None),
                patch.object(maya, "pick_cmake", return_value=None),
                patch.object(maya, "select_build_backend", lambda b: None)]

    def run_check_pc(self, nvcc):
        out, got = io.StringIO(), []
        ps = self.patches(nvcc)
        for p in ps:
            p.start()
        try:
            with patch("sys.stdout", new=out), patch.object(maya.S, "UI", None):
                try:
                    got.append(maya.check_pc(args(backend="cuda"), "qfn"))
                except SystemExit:
                    pass
        finally:
            for p in ps:
                p.stop()
        return out.getvalue(), (got[0] if got else None)

    def test_the_p40_is_usable_and_lays_out_slow(self):
        self.assertEqual(maya.QF.MIN_CC, 61)
        lay = maya.QF.plan([dict(self.P40, index=0)], 32.0)
        self.assertTrue(lay["fits"], lay["reason"])
        self.assertTrue(lay["slow"])                                   # (32 GB: the warn box, not a refusal)
        self.assertFalse(lay["t4"])
        self.assertEqual({k for k in lay["env"] if k.startswith("SAGUARO_SM75")
                          or k in ("SAGUARO_KQL", "SAGUARO_GDN_KEYHEAD", "SAGUARO_ARGMAX_ROWS_SM70")}, set())

    def test_the_toolkit_rule_prefers_a_fitting_12_x(self):
        self.assertEqual(maya.nvcc_range([61]), ((12, 0), (13, 0)))    # (CUDA 13 dropped Pascal: hi)
        self.assertEqual(maya.pick_nvcc([("/c13", (13, 2)), ("/c128", (12, 8))], [61]), ("/c128", (12, 8)))
        self.assertEqual(maya.pick_nvcc([("/c13", (13, 2))], [61]), ("/c13", (13, 2)))   # (only 13: reported)
        self.assertIn("Pascal need CUDA 12.x", maya.toolkit_hint([61]))

    def test_only_cuda_13_installed_says_pascal_needs_12(self):
        out, pc = self.run_check_pc(("/usr/local/cuda-13.2/bin/nvcc", (13, 2)))
        self.assertIsNone(pc)                                          # (the setup stops there)
        self.assertIn("can be used (Pascal: experimental)", out)       # (the card itself: seen, second-class)
        self.assertIn("cannot build for these GPUs", out)
        self.assertIn("CUDA 12.0 or newer but older than 13.0", out)   # ("they need", plainly)
        self.assertIn("CUDA 13 dropped Volta and Pascal", out)
        self.assertIn("Pascal need CUDA 12.x", out)                    # (the hint with the packages to install)

    def test_cuda_12_installed_sets_the_p40_up(self):
        out, pc = self.run_check_pc(("/usr/local/cuda-12.8/bin/nvcc", (12, 8)))
        self.assertIsNotNone(pc)                                       # (past the checks: the P40 is chosen)
        self.assertEqual([g["index"] for g in pc["gpus"]], [0])
        self.assertIn("CUDA toolkit 12.8", out)
        self.assertNotIn("cannot build", out)


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
    with cpu(), patch.object(maya, "ROOT", t), patch.object(maya, "mem_gb", return_value=(91.7, 80.0)), \
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

    def test_the_one_amd_card_with_no_lend_path_turns_pictures_off(self):
        """U7 step 4: write_config_qfn is handed the elastic entry, but a single AMD card at parallel 1 has
        neither lend path (no --batch slots to give, --vram-elastic is CUDA-only) - the entry and --vision stay
        out of the config, and the setup says so in one plain line."""
        with tempfile.TemporaryDirectory() as t:
            v = {"exe": "build-vision-qfn/bin/vision-qfn-hip", "mmproj": "m.gguf", "model": "s.gguf", "gpu": True,
                 **maya.QFN_VISION}
            said = []
            with patch.object(maya, "warn", side_effect=lambda *x: said.append(" ".join(map(str, x)))):
                cfg = written_cfg(Path(t), args(), vision=v)
            self.assertNotIn("vision", cfg)
            self.assertNotIn("--vision", cfg["args"])
            self.assertTrue(any("pictures: off" in s for s in said), said)


class SetupFlow(unittest.TestCase):
    """set_up_qfn with the machine, the build and the downloads stepped around: what the choices write - and,
    v1.2.1-B, what a slow PC gets: the scaled config, the loud warning, the one confirmation question."""

    def flow(self, t: Path, a, vision=None, mtp=True, pc=None, mem=(91.7, 80.0)):
        eng = t / "build" / "qfn"
        eng.mkdir(parents=True)
        (eng / "saguaro-qfn").write_text("")
        models = t / "models"
        files = [t / "gguf" / n for n in maya.qfn_entry("ud-q4_k_xl")["files"]]
        with patch.object(maya, "ROOT", t), patch.object(maya, "qfn_exe", return_value=eng / "saguaro-qfn"), \
                patch.object(maya, "check_pc", return_value=pc or pc_hip()), \
                patch.object(maya, "mem_gb", return_value=mem), \
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
        pc = kw.pop("_pc", None)
        mem = kw.pop("_mem", (91.7, 80.0))
        a = args(qfn_quant="ud-q4_k_xl", context=65536, models_dir=t / "models", **kw)
        p = self.flow(t, a, vision=vision, mtp=not nomtp, pc=pc, mem=mem)
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
        """... on an NVIDIA one-card layout through its lend path: with no batch slots (parallel 1) the config
        gets "vram_elastic": true, which is what gives the engine the --vram-elastic expert cache the elastic
        encoder's room comes from (serve/server.py's _vision_up vram_reserve fallback)."""
        with tempfile.TemporaryDirectory() as t, tempfile.TemporaryDirectory() as t2:
            v = {"exe": "v", "mmproj": "m.gguf", "model": "s.gguf", "gpu": True, **maya.QFN_VISION}
            cfg = self.setup(Path(t), drafting="on", _vision=v, _pc={"backend": "cuda", "gpus": [
                {"index": 0, "name": "RTX A5000", "vram_gb": 24.0, "arch": "86"}]})
            self.assertEqual(cfg["vision"], v)
            self.assertIn("--vision", cfg["args"])
            self.assertIs(cfg["vram_elastic"], True)
            self.assertNotIn("vram_elastic", self.setup(Path(t2), drafting="on"))    # (text only: nothing to lend)

    def test_no_draft_layer_no_question_and_no_mtp(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = self.setup(Path(t), _nomtp=True)
            self.assertNotIn("--mtp", cfg["args"])
            self.assertNotIn("mtp_calibration", cfg["installer"])

    # v1.2.1-B step 2: a PC between the true minimum and the measured layout gets the loud warning with the
    # cause, the scale-down said out and recorded in the config, and the confirmation - one question, which
    # --yes skips and a "no" answers by stopping the setup.

    def test_a_slow_pc_gets_the_scaled_config_and_no_questions_with_yes(self):
        asked = []
        with tempfile.TemporaryDirectory() as t, cpu(), \
                patch.object(maya, "ask", side_effect=lambda q, ch, d, y: asked.append(q) or d):
            cfg = self.setup(Path(t), _mem=(64.0, 50.0))
        a = cfg["args"]
        self.assertEqual(a[a.index("--kv-resident") + 1], "4096")          # (scaled: the plan's window)
        self.assertEqual(a[a.index("--mtp-max-t") + 1], "1")               # (drafting off)
        self.assertIn("--mtp", a)
        self.assertEqual(cfg["installer"]["mtp_calibration"]["kept"], "none")
        self.assertEqual(cfg["installer"]["mtp_calibration"]["skipped"], "slow-pc")
        self.assertFalse([q for q in asked if "very slow" in q])           # (--yes skips the confirmation)
        self.assertFalse([q for q in asked if "m = measure" in q])         # (the drafting question: not asked)

    def test_a_slow_pc_is_asked_once_and_a_no_stops_the_setup(self):
        asked = []

        def answer(q, ch, dflt, yes):
            asked.append(q)
            return "n" if "very slow" in q else dflt
        with tempfile.TemporaryDirectory() as t, cpu(), patch.object(maya, "ask", side_effect=answer):
            self.assertRaises(SystemExit, self.setup, Path(t), **{"_mem": (64.0, 50.0), "yes": False})
        self.assertEqual([q for q in asked if "very slow" in q],
                         ["This will probably be very slow. Are you sure? (y/n)"])

    def test_a_slow_pc_answered_yes_is_written_scaled_and_drafting_stays_off(self):
        asked = []

        def answer(q, ch, dflt, yes):
            asked.append(q)
            return "y" if "very slow" in q else dflt
        with tempfile.TemporaryDirectory() as t, cpu(), patch.object(maya, "ask", side_effect=answer):
            cfg = self.setup(Path(t), **{"_mem": (64.0, 50.0), "yes": False})
        self.assertEqual(cfg["args"][cfg["args"].index("--mtp-max-t") + 1], "1")   # (measure leaves it off: the
        self.assertEqual(cfg["installer"]["mtp_calibration"]["skipped"], "slow-pc")  #  slow PC's own choice)
        self.assertEqual(len([q for q in asked if "very slow" in q]), 1)

    def test_a_measured_pc_is_neither_asked_nor_scaled(self):
        asked = []
        with tempfile.TemporaryDirectory() as t, cpu(), \
                patch.object(maya, "ask", side_effect=lambda q, ch, d, y: asked.append(q) or d):
            cfg = self.setup(Path(t))                                          # (91.7 GB: comfortable)
        self.assertEqual(cfg["args"][cfg["args"].index("--kv-resident") + 1], "32768")
        self.assertNotIn("--mtp-max-t", cfg["args"])
        self.assertFalse([q for q in asked if "very slow" in q])

    def test_drafting_on_overrides_the_scale_down(self):
        with tempfile.TemporaryDirectory() as t, cpu():
            cfg = self.setup(Path(t), _mem=(64.0, 50.0), drafting="on")        # (--drafting on says otherwise)
        self.assertNotIn("--mtp-max-t", cfg["args"])
        self.assertEqual(cfg["installer"]["mtp_calibration"]["kept"], "mtp")   # (the question's own record)


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
        with tempfile.TemporaryDirectory() as t:                 # (a setup before wrote 3 on several GPUs: a
            written_cfg(Path(t), args(parallel=3), gpus=two)     #  several-GPUs setup again keeps it)
            self.assertEqual(written_cfg(Path(t), args(), gpus=two)["parallel"], 3)
        with tempfile.TemporaryDirectory() as t:                 # (U7: a setup that shrinks to ONE card drops it:
            written_cfg(Path(t), args(parallel=3), gpus=two)     #  parallel decoding on one 16 GB card measured
            self.assertIsNone(written_cfg(Path(t), args()).get("parallel"))   #  a loss: one card -> parallel 1
        with tempfile.TemporaryDirectory() as t:                 # (--parallel N still says it on one card)
            self.assertEqual(written_cfg(Path(t), args(parallel=2))["parallel"], 2)


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
