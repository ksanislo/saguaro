"""serve/test_models.py - the web app's Models page: the run configs listed with their engine settings, the settings
written back into a config (checked first, its other keys kept), and a switch to another model - against the mock
engine, no GPU.

    python -m unittest serve.test_models -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import models as M  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
import serve.server as SV  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def glm_cfg(d: Path, name="maya-maya-m.json", **extra) -> Path:
    pack = d / "GLM-5.3-Flash" / "Maya-M" / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack.parent / "shard.gguf").write_bytes(b"x" * 1000)
    cfg = {"exe": "/b/maya", "args": ["--glm-pack", str(pack), "--max-context", "131072", "--expert-cache", "auto",
                                      "--prefill", "auto", "--kv-streaming", "on"],
           "model_name": "glm-5.3-flash", "gpu": [0, 1], "port": 8080, "vision": {"mmproj": "/v/mmproj.gguf"},
           "env": {"SAGUARO_GLM_RAM_HEADROOM_GB": "4"}, "sampling": {"temperature": 1.0},
           "installer": {"models_dir": str(d), "quant": "Maya-M", "draft": "mtp", "written": "2026-10-20 09:00"},
           **extra}
    p = d / name
    p.write_text(json.dumps(cfg, indent=1))
    return p


def ds4_cfg(d: Path, name="maya-gguf-q8.json", drafter=True, vision=False, quant="gguf-q8") -> Path:
    pack = d / "DeepSeek-V4.1-Flash" / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "tensors.txt").write_text("t blk.0.x\n" + ("t mtp.0.w\n" if drafter else ""))
    (pack / "ds4.txt").write_text("hidden 4096\n" + ("vision.num_hidden_layers 24\n" if vision else ""))
    args = ["--pack", str(pack), "--max-context", "65536", "--tp", "2", "--cpu-threads", "16", "--vision", "off"]
    if drafter:
        args += ["--dspark", "on"]
    cfg = {"exe": "/b/maya-ds4", "args": args, "model_name": "deepseek-v4.1-flash", "gpu": [0, 1],
           "installer": {"family": "deepseek", "ds4_quant": quant, "written": "2026-10-09 10:00"}}
    p = d / name
    p.write_text(json.dumps(cfg, indent=1))
    return p


def mm3_cfg(d: Path, name="maya-mm3.json", drafter=True, vision=True) -> Path:
    """MiniMax's config as saguaro.py's write_config_mm3 writes it."""
    pack = d / "MiniMax-M3.1-Flash-EXL3-3.5bpw" / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "mm3.txt").write_text("hidden_size 3072\nmax_position_embeddings 1048576\n" +
                                  ("dspark.num_hidden_layers 2\n" if drafter else "") +
                                  ("vision.hidden_size 1280\n" if vision else ""))
    args = ["--pack", str(pack), "--max-context", "65536", "--kv-streaming", "on", "--tp", "4",
            "--dspark", "off", "--vision", "on"]
    cfg = {"exe": "/b/maya-mm3", "args": args, "model_name": "minimax-m3.1-flash", "gpu": list(range(9)),
           "reasoning_effort": "auto", "installer": {"family": "minimax", "quant": "MiniMax-M3.1-Flash",
                                                     "written": "2026-10-27 10:00"}}
    p = d / name
    p.write_text(json.dumps(cfg, indent=1))
    return p


QFN_SHARD = "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"


def qfn_cfg(d: Path, name="maya-qwen3.8-flash-next-ud-q4_k_xl.json", mtp=True, vision=False, **extra) -> Path:
    """A QFN config as saguaro.py's write_config_qfn writes it (ref/amd-setup-written-qfn.json, in small): the
    Lamina draft layer beside the models folder, and with `vision` the encoder entry + --vision of a CUDA setup."""
    folder = d / "Qwen3.8-Flash-Next-UD-Q4_K_XL"
    pack = folder / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (folder / QFN_SHARD).write_bytes(b"g" * 1000)
    args = ["--pack", str(pack), "--native", str(folder / QFN_SHARD), "--expert-cache", "auto", "--prefill", "auto",
            "--spec", "3", "--spec-min-p", "0.7", "--max-context", "65536", "--kv", "int8",
            "--kv-resident", "32768", "--vram-reserve-mib", "700", "--resident-budget-gib", "71",
            "--rope-scaling", "yarn", "--rope-scale", "4", "--yarn-orig-ctx", "262144"]
    if mtp:
        drafter = d / "Qwen3.8-Flash-Next-MTP-Lamina"
        drafter.mkdir(exist_ok=True)
        (drafter / "ggml-model-f16.gguf").write_bytes(b"m" * 500)
        args += ["--mtp", str(drafter)]
    cfg = {"exe": "/b/saguaro-qfn", "args": args, "model_name": "Qwen3.8-Flash-Next-UD-Q4_K_XL", "gpu": [0],
           "port": 8080, "log": str(d / "qfn.log"),
           "env": {"SAGUARO_NO_LARGEPAGES": "1", "SAGUARO_YARN_ELASTIC": "1", "SAGUARO_MMQ_KQ": "1"},
           "installer": {"family": "qfn", "qwen_quant": "ud-q4_k_xl", "engine": "saguaro-qfn",
                         "models_dir": str(d), "written": "2026-11-23 20:20"}, **extra}
    if vision:
        cfg["vision"] = {"exe": "/b/vision-qfn", "mmproj": str(d / "mmproj-Qwen3.8-Flash-Next-BF16.gguf"),
                         "model": str(folder / QFN_SHARD), "gpu": True, "elastic": True}
        cfg["args"] += ["--vision"]
    p = d / name
    p.write_text(json.dumps(cfg, indent=1))
    return p


class Settings(unittest.TestCase):
    """A config's engine arguments as the page's settings, and back."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_glm_read(self):
        s = M.settings_of(M.read(glm_cfg(self.d)))
        self.assertEqual(s, {"context": 131072, "gpus": [0, 1], "env": {"SAGUARO_GLM_RAM_HEADROOM_GB": "4"},
                             "images": True, "reserve_mib": None, "parallel": 1, "draft": "mtp", "expert_cache": None,
                             "prefill": None, "kv_streaming": True, "kv_resident": None})

    def test_glm_written_back_with_the_rest_kept(self):
        cfg = M.read(glm_cfg(self.d))
        new = M.apply(cfg, {"context": 65536, "parallel": 2, "reserve_mib": 1024, "expert_cache": 1200,
                            "prefill": 4096, "kv_resident": 49152, "gpus": [1], "env": {"A": "1"}, "images": False})
        a = new["args"]
        for f, v in (("--max-context", "65536"), ("--parallel", "2"), ("--vram-reserve-mib", "1024"),
                     ("--expert-cache", "1200"), ("--prefill", "4096"), ("--kv-resident", "49152")):
            self.assertEqual(M.flag(a, f), v, f)
        self.assertEqual((new["gpu"], new["env"], new["sampling"], new["port"]), ([1], {"A": "1"}, {"temperature": 1.0}, 8080))
        self.assertNotIn("vision", new)                       # (Images off: the encoder's entry kept aside ...)
        back = M.apply(new, {"images": True, "reserve_mib": None, "parallel": 1, "env": {}})
        self.assertEqual(back["vision"], {"mmproj": "/v/mmproj.gguf"})   # (... and back when it is switched on)
        self.assertIsNone(M.flag(back["args"], "--vram-reserve-mib"))
        self.assertNotIn("env", back)
        self.assertEqual(M.settings_of(back)["parallel"], 1)

    def test_glm_dflash2_only_when_its_drafter_is_here(self):
        cfg = M.read(glm_cfg(self.d))
        self.assertEqual(M.choices(cfg)["draft"], ["mtp"])
        with self.assertRaisesRegex(ValueError, "DFlash2"):
            M.apply(cfg, {"draft": "dflash2"})
        f = self.d / "GLM-5.3-Flash-DFlash2" / "drafter.gguf"
        f.parent.mkdir()
        f.write_bytes(b"x")
        new = M.apply(cfg, {"draft": "dflash2"})
        self.assertEqual(M.flag(new["args"], "--dflash"), str(f))
        self.assertEqual(M.settings_of(M.apply(new, {"draft": "mtp"}))["draft"], "mtp")

    def test_checked_before_anything_changes(self):
        cfg = M.read(glm_cfg(self.d))
        for bad in ({"context": "lots"}, {"parallel": 9}, {"gpus": []}, {"gpus": ["0"]}, {"reserve_mib": -1},
                    {"env": {"not a name": "1"}}, {"tp": 2}, {"prefill": 1.5}):
            with self.assertRaises(ValueError, msg=bad):
                M.apply(cfg, bad)

    def test_an_older_glm_config_keeps_its_command(self):
        """A config from before --kv-streaming (no flag: every cell in VRAM) reads as streaming off, and a save writes
        only what changed - its engine, built with it, may not know the newer flags (a Maya-S v2 setup of 2026-10-10)."""
        cfg = M.read(glm_cfg(self.d))
        cfg["args"] = ["--glm-pack", "/m/pack", "--max-context", "32768"]
        self.assertFalse(M.settings_of(cfg)["kv_streaming"])
        self.assertTrue(M.settings_of(dict(cfg, env={"SAGUARO_GLM_KV_RESIDENT": "65536"}))["kv_streaming"])
        new = M.apply(cfg, {"context": 65536})
        self.assertEqual(new["args"], ["--glm-pack", "/m/pack", "--max-context", "65536"])
        on = M.apply(cfg, {"kv_streaming": True})
        self.assertEqual(M.flag(on["args"], "--kv-streaming"), "on")
        self.assertNotIn("--expert-cache", on["args"])

    def test_loader_variables_not_from_the_page(self):
        """LD_PRELOAD, PATH and the like change what the engine's process loads: refused, a config's own kept."""
        cfg = M.read(glm_cfg(self.d))
        for k in ("LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "PATH", "Path", "PYTHONPATH", "BASH_ENV"):
            with self.assertRaises(ValueError, msg=k):
                M.apply(cfg, {"env": {k: "/tmp/x"}})
        own = dict(cfg, env={"LD_LIBRARY_PATH": "/opt/rocm/lib"})
        kept = M.apply(own, {"env": {"LD_LIBRARY_PATH": "/opt/rocm/lib", "SAGUARO_X": "1"}})
        self.assertEqual(kept["env"], {"LD_LIBRARY_PATH": "/opt/rocm/lib", "SAGUARO_X": "1"})
        with self.assertRaises(ValueError):
            M.apply(own, {"env": {"LD_LIBRARY_PATH": "/tmp/evil"}})

    def test_deepseek_read_and_written(self):
        p = ds4_cfg(self.d)
        cfg = M.read(p)
        s = M.settings_of(cfg)
        self.assertEqual((s["draft"], s["tp"], s["cpu_threads"], s["images"], s["mmap"]), ("dspark", 2, 16, False, "auto"))
        new = M.apply(cfg, {"draft": "none", "tp": 1, "mmap": "off", "prefill_chunk": 2048, "cpu_threads": None})
        a = new["args"]
        self.assertEqual((M.flag(a, "--dspark"), M.flag(a, "--tp"), M.flag(a, "--mmap-experts"), M.flag(a, "--prefill-chunk")),
                         ("none", "1", "off", "2048"))
        self.assertIsNone(M.flag(a, "--cpu-threads"))
        with self.assertRaisesRegex(ValueError, "vision tower"):
            M.apply(cfg, {"images": True})
        self.assertEqual(M.choices(M.read(ds4_cfg(self.d, "maya-x.json", drafter=False)))["draft"], ["none"])

    def test_deepseek_prompt_lookup_and_draft_confidence(self):
        cfg = M.read(ds4_cfg(self.d))
        s = M.settings_of(cfg)
        self.assertEqual((s["prompt_lookup"], s["dspark_conf"]), (True, None))   # (the engine's: --suffix-draft 3, 0.5)
        off = M.apply(cfg, {"prompt_lookup": False, "dspark_conf": "0.6"})
        self.assertEqual((M.flag(off["args"], "--suffix-draft"), M.flag(off["args"], "--dspark-conf")), ("0", "0.6"))
        self.assertEqual((M.settings_of(off)["prompt_lookup"], M.settings_of(off)["dspark_conf"]), (False, 0.6))
        on = M.apply(off, {"prompt_lookup": True, "dspark_conf": None})["args"]
        self.assertFalse("--suffix-draft" in on or "--dspark-conf" in on)        # (back to the engine's own)
        own = dict(cfg, args=cfg["args"] + ["--suffix-draft", "5"])              # (a config's own match length: kept)
        self.assertEqual(M.flag(M.apply(own, {"prompt_lookup": True})["args"], "--suffix-draft"), "5")
        for bad, why in ((1.5, "from 0 to 1"), ("x", "a number"), (True, "a number")):
            with self.assertRaisesRegex(ValueError, why):
                M.apply(cfg, {"dspark_conf": bad})
        with self.assertRaisesRegex(ValueError, "Prompt lookup"):
            M.apply(cfg, {"prompt_lookup": "on"})

    def test_minimax_memory_prompt_chunk_and_draft_confidence(self):
        cfg = M.read(mm3_cfg(self.d))
        new = M.apply(cfg, {"reserve_gb": "0.5", "cache_gb": 2, "prefill_chunk": 2048, "dspark_conf": 0.65})
        a = new["args"]
        for f, v in (("--reserve-gb", "0.5"), ("--cache-gb", "2"), ("--chunk", "2048"), ("--dspark-conf", "0.65")):
            self.assertEqual(M.flag(a, f), v, f)
        self.assertNotIn("--prefill-chunk", a)                                    # (maya-mm3's own name: --chunk)
        s = M.settings_of(new)
        self.assertEqual((s["reserve_gb"], s["cache_gb"], s["prefill_chunk"], s["dspark_conf"]), (0.5, 2.0, 2048, 0.65))
        back = M.apply(new, {"reserve_gb": "", "cache_gb": None, "prefill_chunk": None, "dspark_conf": None})["args"]
        for f in ("--reserve-gb", "--cache-gb", "--chunk", "--dspark-conf"):
            self.assertNotIn(f, back)
        with self.assertRaisesRegex(ValueError, "VRAM kept free"):
            M.apply(cfg, {"reserve_gb": 100})
        with self.assertRaisesRegex(ValueError, "Prompt chunk"):
            M.apply(cfg, {"prefill_chunk": 8})

    def test_deepseek_answers_at_once(self):
        """maya-ds4 --parallel: read (1 where the config names none), written where it changes, 1-8."""
        cfg = M.read(ds4_cfg(self.d))
        self.assertEqual(M.settings_of(cfg)["parallel"], 1)
        self.assertNotIn("--parallel", M.apply(cfg, {"cpu_threads": 8})["args"])   # (an older config: as it was)
        new = M.apply(cfg, {"parallel": 4})
        self.assertEqual(M.flag(new["args"], "--parallel"), "4")
        self.assertEqual(M.settings_of(new)["parallel"], 4)
        self.assertEqual(M.flag(M.apply(new, {"parallel": 1})["args"], "--parallel"), "1")
        for bad in (0, 9):
            with self.assertRaisesRegex(ValueError, "Answers at once"):
                M.apply(cfg, {"parallel": bad})

    def test_deepseek_tensor_parallel_beyond_two(self):
        # a 9-GPU config the calibration left at --tp 8: read as it is, and its other settings still save
        cfg = M.read(ds4_cfg(self.d))
        cfg["gpu"] = list(range(9))
        cfg["args"] = M.set_flag(cfg["args"], "--tp", 8)
        self.assertEqual(M.settings_of(cfg)["tp"], 8)
        new = M.apply(cfg, {"prefill_chunk": 2048})
        self.assertEqual((M.flag(new["args"], "--tp"), M.flag(new["args"], "--prefill-chunk")), ("8", "2048"))
        self.assertEqual(M.flag(M.apply(cfg, {"tp": 9})["args"], "--tp"), "9")
        with self.assertRaisesRegex(ValueError, "Tensor parallel: from 1 to 9"):
            M.apply(cfg, {"tp": 10})
        with self.assertRaisesRegex(ValueError, "Tensor parallel: from 1 to 2"):
            M.apply(cfg, {"gpus": [0, 1]})               # (fewer GPUs than its group: tp too)
        self.assertEqual(M.flag(M.apply(cfg, {"gpus": [0], "tp": 2})["args"], "--tp"), "2")   # (one GPU: 2 as before)
        # no --tp: the engine's own 2, which a save keeps
        cfg["args"] = M.drop_flag(cfg["args"], "--tp")
        self.assertEqual(M.settings_of(cfg)["tp"], 2)
        self.assertEqual(M.flag(M.apply(cfg, {"cpu_threads": 8})["args"], "--tp"), "2")

    def test_minimax_read_and_written_with_its_own_flags(self):
        cfg = M.read(mm3_cfg(self.d))
        self.assertEqual(M.family(cfg), "minimax")
        self.assertEqual(M.family({"exe": "/b/maya-mm3", "args": ["--pack", "/p"]}), "minimax")   # (no installer)
        self.assertEqual(M.settings_of(cfg), {"context": 65536, "gpus": list(range(9)), "env": {}, "images": True,
                                              "parallel": 1, "draft": "none", "tp": 4, "kv_cache": "int8",
                                              "kv_streaming": True, "kv_resident": None, "cpu_threads": None,
                                              "reserve_gb": None, "prefill_chunk": None, "cache_gb": None,
                                              "dspark_conf": None})
        self.assertEqual(M.choices(cfg), {"draft": ["dspark", "none"], "images": True})
        new = M.apply(cfg, {"context": 262144, "draft": "dspark", "images": False, "tp": 2, "kv_cache": "fp8",
                            "kv_resident": 65536, "cpu_threads": 24})
        a = new["args"]
        for f, v in (("--max-context", "262144"), ("--dspark", "on"), ("--vision", "off"), ("--tp", "2"),
                     ("--kv-cache", "fp8"), ("--kv-streaming", "on"), ("--kv-resident", "65536"),
                     ("--cpu-threads", "24")):
            self.assertEqual(M.flag(a, f), v, f)
        # maya-mm3 stops at a flag it does not know: none of DeepSeek's or GLM's are written
        for f in ("--mmap-experts", "--prefill-chunk", "--reserve-mib", "--vram-reserve-mib", "--prefill"):
            self.assertNotIn(f, a)
        self.assertNotIn("--parallel", a)                                       # (1 and not in the config: as it was)
        with self.assertRaisesRegex(ValueError, "not a setting of this model: mmap"):
            M.apply(cfg, {"mmap": "on"})
        # tp 0: the layers split (no --tp); 3 is not a group the engine takes
        self.assertNotIn("--tp", M.apply(cfg, {"tp": 0})["args"])
        with self.assertRaisesRegex(ValueError, "1, 2 or 4"):
            M.apply(cfg, {"tp": 3})
        with self.assertRaisesRegex(ValueError, "KV cache"):
            M.apply(cfg, {"kv_cache": "q4_0"})
        # KV streaming off: the window goes with it; INT8 (the engine's default) not written where it was not
        off = M.apply(cfg, {"kv_streaming": False, "kv_resident": 65536})["args"]
        self.assertEqual((M.flag(off, "--kv-streaming"), "--kv-resident" in off, "--kv-cache" in off), ("off", False, False))
        # a pack without the drafter or the tower: neither offered
        bare = M.read(mm3_cfg(self.d, "maya-mm3-b.json", drafter=False, vision=False))
        self.assertEqual(M.choices(bare), {"draft": ["none"], "images": False})
        with self.assertRaisesRegex(ValueError, "no DSpark drafter"):
            M.apply(bare, {"draft": "dspark"})
        self.assertEqual(M.quant_label(cfg, Path("maya-mm3.json")), "EXL3 3.5 bpw")

    def test_minimax_answers_at_once(self):
        """maya-mm3 --parallel: read (1 where the config names none), written where it changes, 1-8; the KV streaming and
        DSpark choices stay as they are for the day it is 1 again (the engine leaves both out while it is above 1)."""
        cfg = M.read(mm3_cfg(self.d))
        self.assertEqual(M.settings_of(cfg)["parallel"], 1)
        self.assertNotIn("--parallel", M.apply(cfg, {"cpu_threads": 8})["args"])   # (an older config: as it was)
        new = M.apply(cfg, {"parallel": 4, "draft": "dspark"})
        self.assertEqual(M.flag(new["args"], "--parallel"), "4")
        self.assertEqual((M.flag(new["args"], "--kv-streaming"), M.flag(new["args"], "--dspark")), ("on", "on"))
        s = M.settings_of(new)
        self.assertEqual((s["parallel"], s["kv_streaming"], s["draft"]), (4, True, "dspark"))
        self.assertEqual(M.flag(M.apply(new, {"parallel": 1})["args"], "--parallel"), "1")
        for bad in (0, 9):
            with self.assertRaisesRegex(ValueError, "Answers at once"):
                M.apply(cfg, {"parallel": bad})

    def test_listing(self):
        g, ds = glm_cfg(self.d), ds4_cfg(self.d)
        (self.d / "maya-maya-m.shared-settings.json").write_text("{}")
        os.utime(g, (time.time() - 100, time.time() - 100))
        with mock.patch.object(M, "machine_gpus", return_value=[{"index": 0, "name": "RTX 3090", "vram_gb": 24.0}]):
            out = M.listing(self.d, str(g))
        self.assertEqual([m["file"] for m in out["models"]], [ds.name, g.name])   # (the most recently used first)
        mm = out["models"][1]
        self.assertEqual((mm["running"], mm["quant"], mm["family"], mm["size"]), (True, "Maya-M", "glm", 1000))
        self.assertEqual(out["models"][0]["quant"], "UD-Q8_K_XL")
        # each model's context choices are its setup's: DeepSeek's reach 1M, GLM's 128K (its own size is offered too)
        self.assertEqual((out["models"][0]["contexts"], mm["contexts"]), (M.DS4_CONTEXTS, M.CONTEXTS))
        self.assertEqual([x["index"] for x in out["gpus"]], [0, 1])               # (GPU 1: named by a config only)


    def test_context_range_per_model(self):
        # the Engine settings' Context size slider: Settings > Model's range for each model - up to what the model was
        # trained for (MiniMax's pack: 1M), else 256K (a GLM GGUF header it cannot read, a DeepSeek pack that does not
        # say), never below the config's own size; untested past 256K except on DeepSeek
        g, ds, mm = glm_cfg(self.d), ds4_cfg(self.d), mm3_cfg(self.d)
        M._trained.clear()
        with mock.patch.object(M, "machine_gpus", return_value=[]):
            got = {m["family"]: m["context_range"] for m in M.listing(self.d, None)["models"]}
        self.assertEqual(got["minimax"], {"min": 4096, "max": 1048576, "trained": 1048576, "tested": 262144})
        self.assertEqual(got["glm"], {"min": 4096, "max": 262144, "trained": None, "tested": 262144})
        self.assertEqual(got["deepseek"], {"min": 4096, "max": 262144, "trained": None, "tested": 1048576})
        cfg = json.loads(ds.read_text())
        cfg["args"][cfg["args"].index("--max-context") + 1] = "524288"
        ds.write_text(json.dumps(cfg))
        self.assertEqual(M.describe(ds, None)["context_range"]["max"], 524288)
        # GLM: the GGUF header beside its pack (the server's own reading), kept once read
        with mock.patch.object(M, "trained_context", return_value=1048576) as tc:
            for _ in range(2):
                self.assertEqual(M.describe(g, None)["context_range"]["max"], 1048576)
        tc.assert_called_once_with(str((self.d / "GLM-5.3-Flash" / "Maya-M").resolve()))
        M._trained.clear()

    def test_installs_the_setup_no_longer_lists_are_still_listed(self):
        """v1.1.34: unsloth's UD-Q4_K_XL (gguf-q4) and GLM's EXL3-sc_4.10bpw are out of the setup's pickers; a config of
        either is listed, shown and settable on the Models page as before."""
        g = glm_cfg(self.d, name="maya-exl3-sc_4.10bpw.json",
                    installer={"models_dir": str(self.d), "quant": "EXL3-sc_4.10bpw", "written": "2026-10-20 09:00"})
        ds = ds4_cfg(self.d, name="maya-gguf-q4.json", quant="gguf-q4")
        with mock.patch.object(M, "machine_gpus", return_value=[]):
            out = M.listing(self.d, None)
        by = {m["file"]: m for m in out["models"]}
        self.assertEqual(by[ds.name]["quant"], "UD-Q4_K_XL")
        self.assertEqual(by[g.name]["quant"], "EXL3-sc_4.10bpw")
        self.assertTrue(by[ds.name]["ok"] and by[g.name]["ok"])
        cfg = M.read(ds)
        self.assertEqual(M.settings_of(cfg)["context"], 65536)
        self.assertEqual(M.apply(cfg, {"context": 131072})["args"][cfg["args"].index("--max-context") + 1], "131072")
        self.assertEqual(M.maya_quants.config_label(cfg), "DeepSeek V4.1 · UD-Q4_K_XL")

    def test_the_cyber_frost_config_reads_as_qfn(self):
        """The old CYBER-FROST config reads through the QFN family since T8a: its quant is the family's
        cyber-frost one, and the label the page writes names the model and the engine."""
        cfg = {"exe": "/b/maya", "args": ["--pack", "/p", "--ple-gguf", "/g.gguf"], "model_name": "cyber-frost-3.8-ps",
               "installer": {"family": "qwen", "quant": "CYBER-FROST-3.8-PS", "qwen_quant": "q5_k_m"}}
        self.assertEqual(M.family(cfg), "qfn")
        self.assertEqual(M.quant_label(cfg, Path("maya-cyber-frost.json")), "cyber-frost-q5_k_m")
        self.assertEqual(M.maya_quants.config_label(cfg), "Qwen3.8-Flash-Next · CYBER-FROST-3.8 Q5_K_M "
                                                          "(saguaro-qfn)")

    def test_qfn_read(self):
        """A QFN config reads through its own field set: the RAM tier is --resident-budget-gib (not the Qwen path's
        SAGUARO_EXPERT_RAM_GB), the K/V window in VRAM is --kv-resident, and Answers at once is the config's own
        "parallel" key."""
        cfg = M.read(qfn_cfg(self.d))
        self.assertEqual(M.family(cfg), "qfn")
        s = M.settings_of(cfg)
        self.assertEqual(s, {"context": 65536, "gpus": [0], "images": False, "reserve_mib": 700, "env":
                             {"SAGUARO_NO_LARGEPAGES": "1", "SAGUARO_YARN_ELASTIC": "1", "SAGUARO_MMQ_KQ": "1"},
                             "parallel": 1, "draft": "mtp", "expert_cache": None, "prefill": None, "kv": "int8",
                             "kv_resident": 32768, "ram_gib": 71})
        self.assertEqual(set(s), set(M.QFN_KEYS))
        self.assertNotIn("ram_gb", M.QFN_KEYS)                    # (SAGUARO_EXPERT_RAM_GB: QFN ignores it)
        self.assertEqual(M.choices(cfg), {"draft": ["mtp", "none"], "images": False})
        bare = M.read(qfn_cfg(self.d, "maya-qfn-bare.json", mtp=False))     # (the draft layer was not downloaded)
        self.assertEqual(M.choices(bare)["draft"], ["none"])
        self.assertEqual(M.settings_of(bare)["draft"], "none")
        par = M.read(qfn_cfg(self.d, "maya-qfn-par.json", parallel=8))      # (the config's own key, like mangonel's)
        self.assertEqual(M.settings_of(par)["parallel"], 8)

    def test_qfn_written_back(self):
        cfg = M.read(qfn_cfg(self.d))
        new = M.apply(cfg, {"kv_resident": 16384, "ram_gib": 48, "parallel": 4, "expert_cache": 6000,
                            "prefill": 2048, "reserve_mib": 512, "kv": "q4_0"})
        a = new["args"]
        for f, v in (("--kv-resident", "16384"), ("--resident-budget-gib", "48"), ("--expert-cache", "6000"),
                     ("--prefill", "2048"), ("--vram-reserve-mib", "512"), ("--kv", "q4_0")):
            self.assertEqual(M.flag(a, f), v, f)
        self.assertEqual(new["parallel"], 4)                      # (Answers at once: the config's key ...)
        self.assertNotIn("--parallel", a)                         # ( ... never an engine flag)
        self.assertEqual(M.settings_of(new)["ram_gib"], 48)
        self.assertNotIn("SAGUARO_EXPERT_RAM_GB", new.get("env") or {})
        self.assertEqual((new["port"], new["log"], new["model_name"]), (8080, str(self.d / "qfn.log"),
                                                                        "Qwen3.8-Flash-Next-UD-Q4_K_XL"))
        back = M.apply(new, {"parallel": 1, "kv_resident": None, "ram_gib": None})
        self.assertNotIn("parallel", back)                        # (1 or absent: the engine answers one at a time)
        self.assertIsNone(M.flag(back["args"], "--kv-resident"))   # (empty: as much as fits, no flag)
        self.assertIsNone(M.flag(back["args"], "--resident-budget-gib"))
        s = M.settings_of(back)
        self.assertEqual((s["parallel"], s["kv_resident"], s["ram_gib"]), (1, None, None))
        for bad in ({"ram_gb": 8}, {"tp": 2}, {"kv_streaming": True}, {"parallel": 9}, {"parallel": 0},
                    {"kv": "k8v5"}, {"kv_resident": 512}, {"ram_gib": 0}):
            with self.assertRaises(ValueError, msg=bad):
                M.apply(cfg, bad)
        with self.assertRaisesRegex(ValueError, "not a setting of this model: ram_gb"):
            M.apply(cfg, {"ram_gb": 8})      # (the CYBER-FROST RAM knob: QFN reads nothing of it)

    def test_qfn_drafting(self):
        """Drafting on/off: the config's --mtp args stay (the drafter loads either way) and off is --mtp-max-t 1;
        without the draft layer on this PC, "mtp" is neither offered nor taken."""
        cfg = M.read(qfn_cfg(self.d))
        off = M.apply(cfg, {"draft": "none"})
        self.assertEqual(M.flag(off["args"], "--mtp-max-t"), "1")
        self.assertEqual(M.flag(off["args"], "--mtp"), M.flag(cfg["args"], "--mtp"))
        self.assertEqual(M.settings_of(off)["draft"], "none")
        on = M.apply(off, {"draft": "mtp"})
        self.assertIsNone(M.flag(on["args"], "--mtp-max-t"))
        self.assertEqual(M.settings_of(on)["draft"], "mtp")
        bare = M.read(qfn_cfg(self.d, "maya-qfn-bare.json", mtp=False))
        with self.assertRaisesRegex(ValueError, "MTP draft layer"):
            M.apply(bare, {"draft": "mtp"})

    def test_qfn_images(self):
        """Images with QFN's vision entry (vision-qfn with the model's mmproj): off keeps the entry as vision_off and
        takes --vision out of the command; a text-only setup has nothing to switch on."""
        text_only = M.read(qfn_cfg(self.d))
        with self.assertRaisesRegex(ValueError, "text only"):
            M.apply(text_only, {"images": True})
        vis = M.read(qfn_cfg(self.d, "maya-qfn-vis.json", vision=True))
        self.assertEqual((M.settings_of(vis)["images"], M.choices(vis)["images"]), (True, True))
        off = M.apply(vis, {"images": False})
        self.assertIsNone(off.get("vision"))
        self.assertEqual(off["vision_off"], vis["vision"])
        self.assertNotIn("--vision", off["args"])
        on = M.apply(off, {"images": True})
        self.assertEqual(on["vision"], vis["vision"])
        self.assertIn("--vision", on["args"])
        self.assertFalse((on.get("vision_off") is not None))

    def test_qfn_reset_of_an_old_as_set_up(self):
        """A config whose installer.as_set_up was written while QFN still shared the CYBER-FROST keys (its ram_gb,
        none of QFN's own): the page's Reset takes what QFN has and drops what it does not."""
        p = qfn_cfg(self.d)
        cfg = json.loads(p.read_text())
        cfg["installer"]["as_set_up"] = {"context": 32768, "gpus": [0], "env": {}, "images": False,
                                         "reserve_mib": 700, "draft": "mtp", "expert_cache": None, "prefill": None,
                                         "kv": "int8", "ram_gb": 40}
        p.write_text(json.dumps(cfg))
        setup = M.describe(p, None)["setup"]
        self.assertNotIn("ram_gb", setup)
        self.assertEqual(setup["context"], 32768)
        new = M.apply(M.read(p), setup)                # (the Reset as the page sends it: taken, nothing refused)
        self.assertEqual(M.flag(new["args"], "--max-context"), "32768")

    def test_the_turbo_sidecar_is_not_a_model(self):
        """The live Turbo sidecar serve/server.py keeps beside the config (<config minus .json>.turbo.json) matches
        the maya-*.json glob: it is no more a model than the .shared-settings.json file."""
        g = glm_cfg(self.d)
        side = g.with_name(g.stem + ".turbo.json")
        side.write_text('{"drop": 1, "skip": "off"}')
        self.assertEqual([p.name for p in M.config_paths(self.d)], [g.name])
        with mock.patch.object(M, "machine_gpus", return_value=[]):
            out = M.listing(self.d, None)
        self.assertEqual([m["file"] for m in out["models"]], [g.name])


class QfnTurbo(unittest.TestCase):
    """QFN's Turbo: in the config it turns the Lamina drafting on and Answers at once back to one (no no-op: QFN has
    no expert-skipping environment knobs); the engine's lossy half is the live per-request Turbo, so the server puts
    the model's live Turbo to its default while Turbo is on and restores what it was when Turbo goes off."""

    def setUp(self):
        os.environ.pop("SAGUARO_RESTART_ON_UPDATE", None)
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.p = qfn_cfg(self.d)
        cfg = json.loads(self.p.read_text())
        cfg["args"] += ["--mtp-max-t", "1"]            # (drafting off and 8 at once: what Turbo is asked to change)
        cfg["parallel"] = 8
        self.p.write_text(json.dumps(cfg))
        self.before = M.read(self.p)

    def tearDown(self):
        os.environ.pop("SAGUARO_RESTART_ON_UPDATE", None)
        self.tmp.cleanup()

    def test_the_patch_is_not_empty(self):
        cfg = self.before
        self.assertEqual(M.turbo_patch(cfg), {"parallel": 1, "draft": "mtp"})
        info = M.turbo_info(cfg)
        self.assertEqual((info["on"], info["family"], sorted(info["plan"])), (False, "qfn", ["draft", "parallel"]))
        self.assertEqual(M.turbo_patch(M.apply(cfg, M.turbo_patch(cfg))), {})   # (already Turbo: nothing left to do)

    def test_on_then_off_returns_the_command(self):
        self.assertTrue(M.turbo_enable(self.p)["changed"])
        on = M.read(self.p)
        self.assertTrue(M.turbo_on(on))
        self.assertEqual(M.settings_of(on)["draft"], "mtp")
        self.assertNotIn("parallel", on)
        self.assertTrue(M.turbo_disable(self.p)["changed"])
        back = M.read(self.p)
        for k in ("args", "env", "gpu", "parallel", "port"):
            self.assertEqual(back.get(k), self.before.get(k), k)
        self.assertNotIn("turbo", back)

    def test_one_at_a_time_alone_is_held_by_the_server(self):
        cfg = M.read(self.p)
        cfg = M.apply(cfg, {"draft": "mtp"})                           # (only "parallel" left for Turbo to change)
        M.write(self.p, cfg)
        self.assertEqual(set(M.turbo_patch(cfg)), {"parallel"})
        self.assertEqual(M.turbo_info(cfg)["plan"], {})
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "x", max_context=32768), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.config_path, svc.turbo_path = str(self.p), str(self.p.with_suffix("")) + ".turbo.json"
        svc.fifo.set_capacity(8)
        svc.set_turbo_default({"drop": 2, "skip": "off"})
        self.assertEqual(svc.set_turbo(True)["changed"], False)        # (no restart)
        on = M.read(self.p)
        self.assertEqual((on["parallel"], on["args"]), (cfg["parallel"], cfg["args"]))
        self.assertFalse(M.turbo_info(on)["off_restarts"])
        self.assertTrue(svc.fifo.acquire(False, units=1))
        self.assertFalse(svc.fifo.acquire(False, units=1))             # (8 places, one answer at a time)
        self.assertTrue(svc.fifo.claim_yield(0.0) is False)            # (nobody waits)
        svc.turbo_live_before = None                                   # (as after a restart: the note keeps it)
        self.assertEqual(svc.set_turbo(False)["changed"], False)
        self.assertEqual(svc.turbo_default, {"drop": 2, "skip": "off"})
        self.assertTrue(svc.fifo.acquire(False, units=1))              # (a flip with one running: the places still add up)
        svc.fifo.release(units=1), svc.fifo.release(units=1)
        self.assertFalse(svc.fifo.locked())
        self.assertNotIn("turbo", M.read(self.p))

    def test_a_turbo_that_restarts_says_so_both_ways(self):
        self.assertIn("draft", M.turbo_info(M.read(self.p))["plan"])
        M.turbo_enable(self.p)
        self.assertTrue(M.turbo_info(M.read(self.p))["off_restarts"])

    def test_the_server_turns_the_live_turbo_with_it(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "x", max_context=32768), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.config_path = str(self.p)
        svc.turbo_path = str(self.p.with_suffix("")) + ".turbo.json"    # (where main() keeps the live Turbo)
        self.assertIsNone(svc.turbo_default)
        out = svc.set_turbo(True)
        self.assertEqual(out["changed"], True)
        self.assertEqual(svc.turbo_default, {"drop": 3, "skip": "off"})
        self.assertEqual(json.loads(Path(svc.turbo_path).read_text()), {"drop": 3, "skip": "off"})
        self.assertEqual(M.settings_of(M.read(self.p))["draft"], "mtp")
        svc.set_turbo(False)
        self.assertIsNone(svc.turbo_default)
        self.assertFalse(Path(svc.turbo_path).exists())               # (off: the sidecar goes, as any live Turbo off)
        back = M.read(self.p)
        self.assertEqual(back["args"], self.before["args"])
        self.assertEqual(back.get("parallel"), 8)                     # (drafting and the 8 at once back as they were)


class Served(unittest.TestCase):
    """The server on the mock engine, its config beside another model's."""

    def setUp(self):
        os.environ.pop("SAGUARO_RESTART_ON_UPDATE", None)
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nhello", max_context=32768), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.cfg, self.other = glm_cfg(self.d), ds4_cfg(self.d)
        self.svc.config_path = str(self.cfg)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.gpus = mock.patch.object(M, "machine_gpus", return_value=[])
        self.gpus.start()

    def tearDown(self):
        self.gpus.stop()
        os.environ.pop("SAGUARO_RESTART_ON_UPDATE", None)
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def req(self, method, path, body=None, headers=None):
        h = {"Content-Type": "application/json"} if body is not None else {}
        h.update(headers or {})
        r = urllib.request.Request(self.base + path, method=method, headers=h,
                                   data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())


class Page(Served):
    """GET /api/models, POST /api/models/settings and /api/models/switch."""

    def test_listed(self):
        s, b = self.req("GET", "/api/models")
        self.assertEqual((s, b["running"], b["restarts"], len(b["models"])), (200, self.cfg.name, False, 2))
        self.assertTrue(next(m for m in b["models"] if m["file"] == self.cfg.name)["running"])

    def test_secrets_in_the_environment_stay_on_this_pc(self):
        """A config's HF_TOKEN / *_API_KEY reach the page masked (as the report leaves them out); a save that sends the
        mask back keeps the value, a new value replaces it."""
        cfg = json.loads(self.cfg.read_text())
        cfg["env"] = {"HF_TOKEN": "hf_secret", "SAGUARO_GLM_RAM_HEADROOM_GB": "4"}
        self.cfg.write_text(json.dumps(cfg))
        s, b = self.req("GET", "/api/models")
        self.assertNotIn("hf_secret", json.dumps(b))
        mine = next(m for m in b["models"] if m["file"] == self.cfg.name)
        self.assertEqual((mine["settings"]["env"]["HF_TOKEN"], mine["command"]["env"]["HF_TOKEN"]), (M.MASK, M.MASK))
        s, b = self.req("POST", "/api/models/settings", {"file": self.cfg.name, "dry_run": True,
                                                         "settings": {"env": mine["settings"]["env"]}})
        self.assertNotIn("hf_secret", json.dumps(b))
        s, _ = self.req("POST", "/api/models/settings", {"file": self.cfg.name,
                                                         "settings": {"context": 65536, "env": mine["settings"]["env"]}})
        self.assertEqual(json.loads(self.cfg.read_text())["env"]["HF_TOKEN"], "hf_secret")
        self.req("POST", "/api/models/settings", {"file": self.cfg.name, "settings": {"env": {"HF_TOKEN": "hf_new"}}})
        self.assertEqual(json.loads(self.cfg.read_text())["env"], {"HF_TOKEN": "hf_new"})

    def test_dry_run_saves_nothing(self):
        before = self.cfg.read_text()
        s, b = self.req("POST", "/api/models/settings", {"file": self.cfg.name, "settings": {"context": 65536},
                                                         "dry_run": True})
        self.assertEqual((s, M.flag(b["command"]["args"], "--max-context")), (200, "65536"))
        self.assertEqual(self.cfg.read_text(), before)

    def test_another_models_settings_leave_which_one_starts_next(self):
        old = time.time() - 1000
        os.utime(self.other, (old, old))
        s, b = self.req("POST", "/api/models/settings", {"file": self.other.name, "settings": {"tp": 1}})
        self.assertEqual((s, b["changed"], b["running"], b["restarts"]), (202, True, False, False))
        self.assertEqual(M.flag(json.loads(self.other.read_text())["args"], "--tp"), "1")
        self.assertAlmostEqual(self.other.stat().st_mtime, old, delta=1)   # (still the one used longest ago)

    def test_saved_without_a_restart_here(self):
        s, b = self.req("POST", "/api/models/settings", {"file": self.cfg.name, "settings": {"context": 65536}})
        self.assertEqual((s, b["changed"], b["running"], b["restarts"]), (202, True, True, False))
        cfg = json.loads(self.cfg.read_text())
        self.assertEqual(M.flag(cfg["args"], "--max-context"), "65536")
        self.assertEqual(cfg["installer"]["as_set_up"]["context"], 131072)      # (the page's Reset)
        self.assertEqual(M.flag(json.loads((self.d / (self.cfg.name + ".prev")).read_text())["args"], "--max-context"),
                         "131072")
        s, b = self.req("POST", "/api/models/settings", {"file": self.cfg.name, "settings": {"context": "x"}})
        self.assertEqual(s, 400)

    def test_running_one_starts_again_with_its_new_settings(self):
        os.environ["SAGUARO_RESTART_ON_UPDATE"] = "1"
        with mock.patch.object(SV.os, "_exit") as ex, mock.patch.object(self.svc, "close_for_restart"):
            s, b = self.req("POST", "/api/models/settings", {"file": self.cfg.name, "settings": {"parallel": 2}})
            self.assertEqual((s, b["restarts"]), (202, True))
            for _ in range(50):
                if ex.called:
                    break
                time.sleep(0.05)
        ex.assert_called_once_with(SV.RESTART_EXIT)
        note = json.loads((self.d / M.SWITCH_FILE).read_text())
        self.assertEqual((note["why"], Path(note["config"]).name, note["port"]),
                         ("settings", self.cfg.name, self.httpd.server_address[1]))

    def test_switch(self):
        old = time.time() - 1000
        os.utime(self.other, (old, old))
        s, b = self.req("POST", "/api/models/switch", {"file": self.other.name})
        self.assertEqual((s, b["to"], b["restarts"]), (202, self.other.name, False))
        self.assertGreater(self.other.stat().st_mtime, old + 900)            # (the next start picks it)
        self.assertFalse((self.d / M.SWITCH_FILE).exists())                   # (no restart: no note for saguaro.py)
        os.environ["SAGUARO_RESTART_ON_UPDATE"] = "1"
        with mock.patch.object(SV.os, "_exit") as ex, mock.patch.object(self.svc, "close_for_restart"):
            s, b = self.req("POST", "/api/models/switch", {"file": self.other.name})
            for _ in range(50):
                if ex.called:
                    break
                time.sleep(0.05)
        self.assertEqual((s, b["restarts"]), (202, True))
        ex.assert_called_once_with(SV.RESTART_EXIT)
        self.assertEqual(json.loads((self.d / M.SWITCH_FILE).read_text())["why"], "switch")
        s, b = self.req("POST", "/api/models/switch", {"file": self.other.name})
        self.assertEqual(s, 400)                                              # (already starting again)

    def test_switch_writes_the_models_own_defaults(self):
        """A switch to a config without its model's sampling and level (an older setup's), or with another model's,
        writes the model's own into it (the setup's MODEL_DEFAULTS), the config before kept beside it."""
        cfg = json.loads(self.other.read_text())                             # (DeepSeek's)
        cfg.pop("reasoning_effort", None)
        cfg["sampling"] = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}      # (not DeepSeek's)
        self.other.write_text(json.dumps(cfg))
        s, b = self.req("POST", "/api/models/switch", {"file": self.other.name})
        self.assertEqual(s, 202, b)
        got = json.loads(self.other.read_text())
        self.assertEqual((got["sampling"], got["reasoning_effort"]), ({"temperature": 1.0, "top_p": 1.0}, "high"))
        self.assertTrue((self.d / (self.other.name + ".prev")).exists())
        self.assertFalse(M.own_defaults(self.other))                         # (its own already: nothing written)

    def test_refused(self):
        for f in (self.cfg.name, "../maya-x.json", "maya-none.json", "maya-maya-m.shared-settings.json", None):
            s, _ = self.req("POST", "/api/models/switch", {"file": f})
            self.assertEqual(s, 400, f)
        s, _ = self.req("POST", "/api/models/switch", {"file": self.other.name}, {"Origin": "http://elsewhere.example"})
        self.assertEqual(s, 403)

    def test_files_gone(self):
        # a config whose model was deleted: listed as such, and no switch to it (saguaro.py would stop there, no model
        # running)
        gone = glm_cfg(self.d, "maya-gone.json")
        cfg = json.loads(gone.read_text())
        cfg["args"] = M.set_flag(cfg["args"], "--glm-pack", str(self.d / "deleted" / "pack"))
        gone.write_text(json.dumps(cfg))
        s, b = self.req("GET", "/api/models")
        m = next(x for x in b["models"] if x["file"] == gone.name)
        self.assertEqual((m["ok"], m["missing"]), (False, f"its model files are not on this PC ({self.d / 'deleted'})"))
        self.assertTrue(next(x for x in b["models"] if x["file"] == self.other.name)["ok"])
        s, b = self.req("POST", "/api/models/switch", {"file": gone.name})
        self.assertEqual(s, 400)
        self.assertIn("its model files are not on this PC", json.dumps(b))
        s, _ = self.req("POST", "/api/models/settings", {"file": self.cfg.name, "settings": {"context": 8192}},
                        {"Origin": "http://elsewhere.example"})
        self.assertEqual(s, 403)
        self.assertEqual(M.flag(json.loads(self.cfg.read_text())["args"], "--max-context"), "131072")



def glm_quant(d: Path, quant: str, folder: Path | None = None, pack_name="pack", shards=3, dflash=False,
              models: Path | None = None) -> Path:
    """A GLM config as the setup writes it for `quant`: its GGUF shards and pack in <folder> (<models>/<quant>), the
    image encoder in vision/, the DFlash2 drafter in <models>/GLM-5.3-Flash-DFlash2 when it drafts with it."""
    models = models or d
    folder = folder or models / "GLM-5.3-Flash" / quant
    pack = folder / pack_name
    pack.mkdir(parents=True, exist_ok=True)
    names = [f"GLM-5.3-Flash-{quant}-{i:05d}-of-{shards:05d}.gguf" for i in range(1, shards + 1)]
    for n in names:
        (folder / n).write_bytes(b"g" * 1000)
    (pack / "index.txt").write_bytes(b"i" * 100)
    (pack / "sources.json").write_text(json.dumps({"files": [{"name": n, "size": 1000} for n in names]}))
    (folder / "vision").mkdir(exist_ok=True)
    (folder / "vision" / "mmproj.gguf").write_bytes(b"v" * 200)
    args = ["--glm-pack", str(pack), "--max-context", "65536"]
    if dflash:
        drafter = models / "GLM-5.3-Flash-DFlash2" / "GLM-5.3-Flash-DFlash2-Q4_K_M.gguf"
        drafter.parent.mkdir(parents=True, exist_ok=True)
        drafter.write_bytes(b"f" * 700)
        args += ["--dflash", str(drafter)]
    cfg = {"exe": "/b/maya", "args": args, "model_name": "glm-5.3-flash", "gpu": [0],
           "vision": {"mmproj": str(folder / "vision" / "mmproj.gguf")},
           "installer": {"models_dir": str(models), "quant": quant, "draft": "dflash2" if dflash else "mtp"}}
    p = d / f"maya-{quant.lower()}.json"
    p.write_text(json.dumps(cfg, indent=1))
    return p


class Delete(unittest.TestCase):
    """A model off the list, and with its files: what its setup downloaded for it and the pack Saguaro built - never what
    another model runs from, nor a GGUF of your own."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_its_own_folder_and_drafter(self):
        m = glm_quant(self.d, "Maya-M")
        l = glm_quant(self.d, "Maya-L", dflash=True)
        (self.d / f"run-{l.stem}.sh").write_text("#!/bin/sh\n")
        plan = M.delete_plan(self.d, l)
        folder = self.d / "GLM-5.3-Flash" / "Maya-L"
        self.assertTrue(plan["ok"])
        self.assertEqual([(Path(i["path"]).name, i["what"]) for i in plan["items"]],
                         [("Maya-L", "folder"), ("GLM-5.3-Flash-DFlash2-Q4_K_M.gguf", "drafter")])
        self.assertEqual((plan["gguf"], plan["pack"], plan["vision"], plan["drafter"], plan["drafter_users"]),
                         (3, True, True, 700, []))
        self.assertEqual(plan["bytes"], 3 * 1000 + 100 + len((folder / "pack" / "sources.json").read_bytes()) + 200 + 700)
        out = M.delete(self.d, l, files=True)
        self.assertEqual((out["deleted"], out["freed"]), (l.name, plan["bytes"]))
        self.assertFalse(folder.exists() or l.exists() or (self.d / f"run-{l.stem}.sh").exists())
        self.assertFalse((self.d / "GLM-5.3-Flash-DFlash2").exists())       # (its folder, empty now)
        self.assertTrue(m.exists() and (self.d / "GLM-5.3-Flash" / "Maya-M" / "pack").is_dir())

    def test_off_the_list_only(self):
        l = glm_quant(self.d, "Maya-L")
        prev, shared = l.with_name(l.name + ".prev"), l.with_name(l.stem + ".shared-settings.json")
        prev.write_text("{}")
        shared.write_text("{}")
        out = M.delete(self.d, l)
        self.assertEqual((out["freed"], out["files"]), (0, False))
        self.assertFalse(l.exists() or prev.exists() or shared.exists())
        self.assertTrue((self.d / "GLM-5.3-Flash" / "Maya-L" / "pack").is_dir())

    def test_a_folder_another_model_runs_from(self):
        # two quants' files in one folder (the setup's pack-<model>/ beside pack/): only its own pack and GGUF files
        folder = self.d / "GLM-5.3-Flash"
        glm_quant(self.d, "Maya-M", folder=folder)
        l = glm_quant(self.d, "Maya-L", folder=folder, pack_name="pack-maya-l")
        plan = M.delete_plan(self.d, l)
        self.assertEqual(sorted(i["what"] for i in plan["items"]), ["gguf", "gguf", "gguf", "pack"])
        self.assertFalse(plan["vision"])                    # (vision/: Maya-M's too)
        M.delete(self.d, l, files=True)
        self.assertEqual(sorted(p.name for p in folder.iterdir()),
                         [f"GLM-5.3-Flash-Maya-M-{i:05d}-of-00003.gguf" for i in (1, 2, 3)] + ["pack", "vision"])

    def test_your_own_gguf_stays(self):
        own = self.d / "elsewhere"
        l = glm_quant(self.d, "Maya-L", folder=own, models=self.d / "models")
        plan = M.delete_plan(self.d, l)
        self.assertEqual([i["what"] for i in plan["items"]], ["pack"])
        self.assertIn("Your own model files", plan["kept"][0])
        M.delete(self.d, l, files=True)
        self.assertEqual(len(list(own.glob("*.gguf"))), 3)
        self.assertFalse((own / "pack").exists())

    def test_a_drafter_another_model_drafts_with(self):
        m = glm_quant(self.d, "Maya-M", dflash=True)
        l = glm_quant(self.d, "Maya-L", dflash=True)
        os.utime(m, (1_700_000_000, 1_700_000_000))
        plan = M.delete_plan(self.d, l)
        self.assertEqual(plan["drafter_users"], [m.name])
        out = M.delete(self.d, l, files=True)
        self.assertEqual(out["drafter_users"], [m.name])
        cfg = json.loads(m.read_text())
        self.assertNotIn("--dflash", cfg["args"])           # (it drafts with MTP now: its drafter is gone)
        self.assertEqual(M.settings_of(cfg)["draft"], "mtp")
        self.assertEqual(m.stat().st_mtime, 1_700_000_000)  # (which model starts next: as it was)
        self.assertIn("--dflash", json.loads(m.with_name(m.name + ".prev").read_text())["args"])

    def test_without_the_setups_record(self):
        p = ds4_cfg(self.d)                                 # (no models_dir: which files are its own is not known)
        plan = M.delete_plan(self.d, p)
        self.assertEqual((plan["ok"], plan["items"]), (False, []))
        with self.assertRaises(ValueError):
            M.delete(self.d, p, files=True)
        self.assertTrue(p.exists())


class Leftovers(unittest.TestCase):
    """The files in the models folder no model here runs from: a whole folder, or one quant's beside another's."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.models = self.d / "models"
        self.big = mock.patch.object(M, "LEFTOVER_MIN", 500)       # (the fixtures' files are small)
        self.big.start()

    def tearDown(self):
        self.big.stop()
        self.tmp.cleanup()

    def test_what_no_model_uses(self):
        glm_quant(self.d, "Maya-M", models=self.models)
        l = glm_quant(self.d, "Maya-L", models=self.models)
        self.assertEqual(M.leftovers(self.d)["items"], [])     # (both GLM quants are on the list)
        old = self.models / "DSV4"                            # (a model whose config was deleted by hand)
        (old / "pack").mkdir(parents=True)
        (old / "pack" / "tensors.txt").write_text("t blk.0.x\n")
        (old / "model-00001-of-00002.safetensors").write_bytes(b"s" * 2000)
        (old / ".cache" / "huggingface").mkdir(parents=True)
        (self.models / "download.log").write_text("small\n")
        l.unlink()
        out = M.leftovers(self.d)
        self.assertEqual(out["folders"], [str(self.models.resolve())])
        got = {Path(i["path"]).name: (i["folder"], i["pack"], i["hub"]) for i in out["items"]}
        self.assertEqual(got, {"DSV4": (True, True, True), "Maya-L": (True, True, False)})
        self.assertEqual(out["bytes"], sum(i["bytes"] for i in out["items"]))
        gone = M.delete_leftover(self.d, out["items"][0]["path"])
        self.assertFalse(Path(gone["deleted"]).exists())
        self.assertTrue((self.models / "GLM-5.3-Flash" / "Maya-M" / "pack").is_dir())

    def test_never_a_folder_a_model_reads_through_its_pack(self):
        # DeepSeek's drafter and vision tower beside its GGUF folder: no config names them, the pack reads them
        p = ds4_cfg(self.d)
        cfg = json.loads(p.read_text())
        root = self.models / "DeepSeek-V4.1-Flash"
        self.models.mkdir(parents=True, exist_ok=True)   # (root itself: the rename makes it; Windows will not rename onto a folder)
        (self.d / "DeepSeek-V4.1-Flash").rename(root)
        cfg["args"][1] = str(root / "pack")
        cfg["installer"]["models_dir"] = str(self.models)
        p.write_text(json.dumps(cfg))
        for side in ("dspark", "vision"):
            (root / side).mkdir()
            (root / side / f"{side}.bin").write_bytes(b"d" * 2000)
        self.assertEqual(M.leftovers(self.d)["items"], [])

    def test_only_what_is_listed_is_deleted(self):
        m = glm_quant(self.d, "Maya-M", models=self.models)
        with self.assertRaisesRegex(ValueError, "not a file no model uses"):
            M.delete_leftover(self.d, str(self.models / "GLM-5.3-Flash" / "Maya-M"))
        with self.assertRaisesRegex(ValueError, "not a file no model uses"):
            M.delete_leftover(self.d, str(self.d))
        self.assertTrue(m.exists() and (self.models / "GLM-5.3-Flash" / "Maya-M").is_dir())


class DeletePage(Served):
    """POST /api/models/delete: not the model running, the app's own page only, dry_run deletes nothing."""

    def test_delete(self):
        l = glm_quant(self.d, "Maya-L", dflash=True)
        s, b = self.req("POST", "/api/models/delete", {"file": l.name, "files": True, "dry_run": True})
        self.assertEqual((s, b["file"], b["ok"], b["gguf"]), (200, l.name, True, 3))
        self.assertTrue(l.exists())
        s, _ = self.req("POST", "/api/models/delete", {"file": l.name, "files": True},
                        {"Origin": "http://elsewhere.example"})
        self.assertEqual(s, 403)
        s, b = self.req("POST", "/api/models/delete", {"file": l.name, "files": True})
        self.assertEqual((s, b["deleted"], b["freed"] > 3000), (200, l.name, True))
        s, b = self.req("GET", "/api/models")
        self.assertNotIn(l.name, [m["file"] for m in b["models"]])
        self.assertFalse((self.d / "GLM-5.3-Flash" / "Maya-L").exists())

    def test_not_the_one_running(self):
        for f in (self.cfg.name, "../maya-x.json", "maya-none.json", None):
            s, _ = self.req("POST", "/api/models/delete", {"file": f})
            self.assertEqual(s, 400, f)
        # the running one with a model to switch to: only where Saguaro can start again by itself
        s, b = self.req("POST", "/api/models/delete", {"file": self.cfg.name, "then": self.other.name})
        self.assertEqual(s, 400)
        self.assertIn("cannot switch", b["error"]["message"])
        self.assertTrue(self.cfg.exists())
        self.assertFalse((self.d / M.DELETE_FILE).exists())

    def test_the_running_one_after_a_switch(self):
        os.environ["SAGUARO_RESTART_ON_UPDATE"] = "1"
        with mock.patch.object(Service, "_restart_into") as restart:
            s, b = self.req("POST", "/api/models/delete", {"file": self.cfg.name, "dry_run": True})
            self.assertEqual((s, b["running"]), (200, True))
            s, b = self.req("POST", "/api/models/delete", {"file": self.cfg.name})       # (no model to switch to)
            self.assertEqual(s, 400)
            s, b = self.req("POST", "/api/models/delete", {"file": self.cfg.name, "then": self.cfg.name})
            self.assertEqual(s, 400)                                                     # (not itself)
            self.assertFalse((self.d / M.DELETE_FILE).exists())
            s, b = self.req("POST", "/api/models/delete", {"file": self.cfg.name, "then": self.other.name})
        self.assertEqual((s, b["deleting"], b["to"]), (202, self.cfg.name, self.other.name))
        self.assertEqual(restart.call_args[0][0].name, self.other.name)
        self.assertTrue(self.cfg.exists())                    # (it runs until the switch)
        # the next server, on the other model: it deletes it, once
        self.svc.config_path = str(self.other)
        self.svc.finish_delete()
        self.assertFalse(self.cfg.exists() or (self.d / M.DELETE_FILE).exists())
        self.assertEqual((self.svc.deleted["deleted"], self.svc.deleted["model"]), (self.cfg.name, "glm-5.3-flash"))
        s, b = self.req("GET", "/api/models")
        self.assertEqual((b["deleted"]["deleted"], [m["file"] for m in b["models"]]), (self.cfg.name, [self.other.name]))

    def test_a_switch_that_did_not_happen_deletes_nothing(self):
        note = {"config": str(self.cfg.resolve()), "files": True, "time": time.time()}
        (self.d / M.DELETE_FILE).write_text(json.dumps(note))
        self.svc.finish_delete()                              # (the same model runs again)
        self.assertTrue(self.cfg.exists())
        self.assertIn("running now", self.svc.deleted["error"])
        (self.d / M.DELETE_FILE).write_text(json.dumps(dict(note, time=time.time() - 3600)))
        self.svc.config_path = str(self.other)
        self.svc.finish_delete()                              # (a stale note: an hour old)
        self.assertTrue(self.cfg.exists())
        self.assertFalse((self.d / M.DELETE_FILE).exists())

    def test_leftovers_from_the_page_only(self):
        models = self.d / "models"
        glm_quant(self.d, "Maya-S", models=models)            # (the folder stays known: a config downloaded into it)
        glm_quant(self.d, "Maya-L", models=models).unlink()
        with mock.patch.object(M, "LEFTOVER_MIN", 500):
            s, b = self.req("GET", "/api/models")
            [item] = b["leftovers"]["items"]
            s, _ = self.req("POST", "/api/models/leftovers", {"path": item["path"]}, {"Origin": "http://elsewhere.example"})
            self.assertEqual(s, 403)
            s, out = self.req("POST", "/api/models/leftovers", {"path": item["path"]})
        self.assertEqual((s, out["freed"]), (200, item["bytes"]))
        self.assertFalse(Path(item["path"]).exists())
        s, _ = self.req("POST", "/api/models/leftovers", {"path": item["path"]})
        self.assertEqual(s, 400)


# saguaro-ik2 stops at a flag it does not know: all a config of it may carry (--serve is the server's own)
IK2_VALUE_FLAGS = {"--model", "--max-context", "--ctx", "--cache-gb", "--gpus", "--threads", "--chunk", "--draft",
                   "--eos", "--image-token", "--dump", "--drafting", "--parallel", "--kv-streaming", "--kv-resident",
                   "--user-token"}
IK2_BARE_FLAGS = {"--tokens", "--gen"}


class Inkling2(unittest.TestCase):
    """An Inkling 2 config (saguaro-ik2): the page writes only what its engine reads; Turbo is refused."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.p = Path(self.tmp.name) / "maya-inkling-2-mini.json"
        self.p.write_text((ROOT / "configs" / "Inkling-2-Mini-UD-Q4_K_XL.json").read_text(encoding="utf-8"))
        self.cfg = M.read(self.p)

    def sound(self, cfg):
        args, i = [str(a) for a in cfg.get("args") or []], 0
        while i < len(args):
            self.assertIn(args[i], IK2_VALUE_FLAGS | IK2_BARE_FLAGS, args)
            i += 2 if args[i] in IK2_VALUE_FLAGS else 1
        self.assertEqual(i, len(args), args)
        for k in cfg.get("env") or {}:
            self.assertNotRegex(k, r"^SAGUARO_(GLM|MM3|DS4)_")
        self.assertEqual(M.flag(args, "--gpus"), str(len(M.gpus_of(cfg))))
        self.assertIn(len(M.gpus_of(cfg)), (1, 2))
        self.assertNotIn("layer_split", cfg)
        return cfg

    def test_family_and_settings(self):
        self.assertEqual(M.family(self.cfg), "ik2")
        self.assertEqual(M.settings_of(self.cfg),
                         {"context": 131072, "gpus": [2, 3], "images": True, "prefill_chunk": 8192,
                          "cpu_threads": 8, "kv": "f16", "draft": "none", "parallel": 1,
                          "env": {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "SAGUARO_IK2_ROOM_MB": "1024",
                                  "SAGUARO_IK2_STATS": "1",
                                  "SAGUARO_IK2_WARM": "data/expert-profile-ik2mini.bin",
                                  "SAGUARO_IK2_SLOT_DIR": "var/Inkling-2-Mini-UD-Q4_K_XL/sessions"}})
        self.assertEqual(M.choices(self.cfg), {"draft": ["none"], "images": True})   # (two GPUs: the engine does not draft yet)
        tp = dict(self.cfg, args=self.cfg["args"] + ["--tp", "2"])
        self.assertEqual(M.choices(tp)["draft"], ["none"])                 # (--tp 2: no verify window)
        self.assertEqual(self.sound(M.apply(self.cfg, M.settings_of(self.cfg))), self.cfg)   # (as it is: nothing moves)

    def test_each_setting(self):
        for patch, flag, want in (({"context": 65536}, "--max-context", "65536"), ({"gpus": [2]}, "--gpus", "1"),
                                  ({"gpus": [0, 1]}, "--gpus", "2"), ({"prefill_chunk": 4096}, "--chunk", "4096"),
                                  ({"prefill_chunk": None}, "--chunk", None), ({"cpu_threads": 16}, "--threads", "16"),
                                  ({"cpu_threads": None}, "--threads", None), ({"kv": "fp8"}, "--gpus", "2"),
                                  ({"images": False}, "--gpus", "2"), ({"env": {"SAGUARO_IK2_ROOM_MB": "512"}}, "--gpus", "2")):
            with self.subTest(patch=patch):
                new = self.sound(M.apply(self.cfg, patch))
                self.assertEqual(M.flag(new["args"], flag), want)
                self.assertEqual(self.sound(M.apply(new, M.settings_of(new))), new)   # (and read back the same)
        self.assertEqual(M.apply(self.cfg, {"gpus": [3]})["gpu"], [3])
        fp8 = M.apply(self.cfg, {"kv": "fp8"})
        self.assertEqual(fp8["env"]["SAGUARO_IK2_KV"], "fp8")
        self.assertEqual(M.apply(fp8, {"kv": "f16"})["env"]["SAGUARO_IK2_KV"], "f16")
        self.assertEqual(M.settings_of(M.apply(self.cfg, {"env": {"SAGUARO_IK2_KV": "fp8"}}))["kv"], "fp8")
        off = M.apply(self.cfg, {"images": False})
        self.assertEqual((off.get("vision"), off["vision_off"]), (None, self.cfg["vision"]))
        self.assertEqual(M.apply(off, {"images": True})["vision"], self.cfg["vision"])

    def test_answers_at_once_and_drafting(self):
        """--parallel N written where it changes (the engine takes it on one GPU holding every expert, else says so);
        drafting where it runs - Volta on, the offload path where its cache stays in VRAM, not Pascal."""
        two = self.sound(M.apply(self.cfg, {"parallel": 2}))
        self.assertEqual(M.flag(two["args"], "--parallel"), "2")
        self.assertEqual(M.settings_of(two)["parallel"], 2)
        self.assertEqual(M.flag(M.apply(two, {"parallel": 1})["args"], "--parallel"), "1")
        one = dict(self.cfg, gpu=[0], args=[a if a != "2" else "1" for a in self.cfg["args"]])
        with mock.patch.object(M, "machine_gpus", lambda b="cuda": [{"index": 0, "arch": "86", "vram_gb": 24.0}]):
            self.assertEqual(M.choices(dict(one, args=one["args"][:3] + ["32768"] + one["args"][4:]))["draft"],
                             ["auto", "none"])
        with mock.patch.object(M, "machine_gpus", lambda b="cuda": [{"index": 0, "arch": "75", "vram_gb": 15.0}]):
            self.assertEqual(M.choices(dict(one, args=one["args"][:3] + ["32768"] + one["args"][4:]))["draft"],
                             ["auto", "none"])   # (a T4's offload path drafts: Ken 11-23)
        with mock.patch.object(M, "machine_gpus", lambda b="cuda": [{"index": 0, "arch": "61", "vram_gb": 24.0}]):
            self.assertEqual(M.choices(one)["draft"], ["none"])   # (Pascal: no drafting)

    def test_its_other_name_for_the_context(self):
        cfg = dict(self.cfg, args=["--ctx" if a == "--max-context" else a for a in self.cfg["args"]])
        self.assertEqual(M.settings_of(cfg)["context"], 131072)
        new = self.sound(M.apply(cfg, {"context": 32768}))
        self.assertEqual((M.flag(new["args"], "--ctx"), M.flag(new["args"], "--max-context")), ("32768", None))

    def test_what_its_engine_does_not_read_is_refused(self):
        for bad in ({"parallel": 9}, {"draft": "auto"}, {"kv_streaming": True}, {"kv_resident": 4096}, {"reserve_mib": 700}, {"tp": 2},
                    {"draft": "mtp"}, {"expert_cache": 100}, {"prefill": 512}, {"kv_cache": "int8"}, {"dspark_conf": 0.3},
                    {"reserve_gb": 1}, {"cache_gb": 2}, {"mmap": "on"}, {"prompt_lookup": True}, {"ram_gb": 8},
                    {"gpus": [0, 1, 2]}, {"gpus": []}, {"kv": "int8"}, {"context": 0},
                    {"env": {"SAGUARO_GLM_MISS_SKIP_W": "0.16"}}, {"env": {"SAGUARO_MM3_SLOT_DIR": "/x"}},
                    {"env": {"SAGUARO_DS4_RAM_SKIP_W": "0.14"}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                M.apply(self.cfg, bad)
        self.assertIn("not a setting of this model", self.raised({"tp": 2}))
        tp = dict(self.cfg, args=self.cfg["args"] + ["--tp", "2"])
        with self.assertRaises(ValueError) as e:
            M.apply(tp, {"draft": "auto"})
        self.assertIn("Drafting: not on this GPU", str(e.exception))
        self.assertIn("one or two", self.raised({"gpus": [0, 1, 2]}))
        self.assertIn("SAGUARO_IK2_", self.raised({"env": {"SAGUARO_GLM_KV_RESIDENT": "1"}}))

    def raised(self, patch):
        with self.assertRaises(ValueError) as e:
            M.apply(self.cfg, patch)
        return str(e.exception)

    def test_one_card(self):
        cfg = json.loads((ROOT / "configs" / "Inkling-2-Mini-UD-Q4_K_XL.1card.json").read_text(encoding="utf-8"))
        self.assertEqual(self.sound(M.apply(cfg, M.settings_of(cfg))), cfg)
        two = self.sound(M.apply(cfg, {"gpus": [2, 3]}))
        self.assertEqual((M.flag(two["args"], "--gpus"), two["env"]["SAGUARO_IK2_ROOM_MB"]), ("2", "1024"))

    def test_saved_and_listed(self):
        out = M.save(self.p, {"context": 65536, "kv": "fp8"})
        self.assertTrue(out["changed"])
        new = self.sound(M.read(self.p))
        self.assertEqual(new["installer"]["as_set_up"], M.settings_of(self.cfg))
        self.sound(M.apply(new, new["installer"]["as_set_up"]))            # (the page's Reset)
        d = M.describe(self.p, None)
        self.assertEqual((d["family"], d["settings"]["kv"], d["command"]["exe"]), ("ik2", "fp8", "saguaro-ik2"))

    def test_prompt_chunk_is_checked_against_the_fit(self):
        """The Models page checks Prompt chunk as the setup does (tools/ik2_fit.py chunk_max): the one-T4 quant resident
        on a 16 GB AMD card takes at most 1024 (2048 ran it out of memory on a 10K prompt); UD-Q4_K_XL's offload
        path there, or a card the page cannot read, has no cap."""
        cfg = {"exe": "/s/build-hip/saguaro-ik2", "backend": "hip", "gpu": [0],
               "args": ["--model", "/m/Inkling-2-Mini-T4-NVFP4-IQ3_XXS.gguf", "--max-context", "32768", "--gpus", "1",
                        "--chunk", "1024", "--threads", "8", "--drafting", "auto"],
               "installer": {"family": "ik2", "ik2_file": "Inkling-2-Mini-T4-NVFP4-IQ3_XXS.gguf"}}
        xt = [{"index": 0, "arch": "gfx1201", "vram_gb": 15.9}]
        M._ik2_ex["/m/Inkling-2-Mini-T4-NVFP4-IQ3_XXS.gguf"] = 11.45
        self.addCleanup(M._ik2_ex.pop, "/m/Inkling-2-Mini-T4-NVFP4-IQ3_XXS.gguf", None)
        with mock.patch.object(M, "machine_gpus", lambda backend: xt):
            self.assertEqual(M.flag(M.apply(cfg, {"prefill_chunk": 512})["args"], "--chunk"), "512")
            with self.assertRaises(ValueError) as e:
                M.apply(cfg, {"prefill_chunk": 2048})
            self.assertIn("at most 1024", str(e.exception))
            ud = dict(cfg, args=[a.replace("T4-NVFP4-IQ3_XXS", "UD-Q4_K_XL").replace("auto", "off")
                                 for a in cfg["args"]], installer={})
            self.assertEqual(M.flag(M.apply(ud, {"prefill_chunk": 2048})["args"], "--chunk"), "2048")
        with mock.patch.object(M, "machine_gpus", lambda backend: []):
            self.assertEqual(M.flag(M.apply(cfg, {"prefill_chunk": 2048})["args"], "--chunk"), "2048")

    def test_turbo_is_refused(self):
        self.assertEqual(M.turbo_patch(self.cfg), {})
        self.assertEqual(M.turbo_info(self.cfg), {"on": False, "family": "ik2", "plan": {}, "env": {}, "supported": False})
        with self.assertRaises(ValueError) as e:
            M.turbo_enable(self.p)
        self.assertIn("Inkling 2", str(e.exception))
        self.assertEqual(M.read(self.p), self.cfg)                          # (nothing written, no "turbo" note)
        self.assertFalse(self.p.with_name(self.p.name + ".prev").exists())

    def test_the_server_refuses_turbo_too(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "x", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.config_path, svc.engine_kind = str(self.p), "ik2"
        self.assertFalse(svc.turbo_info()["supported"])
        with self.assertRaises(ValueError):
            svc.set_turbo(True)
        with self.assertRaises(ValueError):
            svc.set_turbo_default({"drop": 2, "skip": "ram"})
        self.assertIsNone(svc.set_turbo_default(None))
        self.assertEqual(M.read(self.p), self.cfg)
        out = svc.model_settings(self.p.name, {"cpu_threads": 12}, dry_run=True)
        self.assertEqual(M.flag(out["command"]["args"], "--threads"), "12")
        with self.assertRaises(ValueError):
            svc.model_settings(self.p.name, {"tp": 2})
        self.assertEqual(M.read(self.p), self.cfg)


if __name__ == "__main__":
    unittest.main()
