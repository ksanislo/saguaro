"""tools/test_setup_qfn.py - Qwen3.8-Flash-Next on its own engine (saguaro-qfn) in the setup: its names and setup ids
(the old CYBER-FROST ids among them, resolving to its quant), the pinned manifest (the model files, the draft layer,
the picture tower), the model
question (offered on AMD and on NVIDIA, refused on Windows and on Mac), v1.2.1-B's fit policy (refused only under
the true minimum; between it and the measured layout the setup scales the layout down - drafting off, --kv-resident
4096, a 32K context - warns what it turned off and why and confirms, --yes skipping the question and --check only
reporting), Pascal (MIN_CC 61 - a Tesla P40 with a CUDA 12.x toolkit, and only CUDA 13 installed says so plainly),
the RAM tier it gets, the memlock soft limit beside a planned RAM tier (P3: one warning with the fix; unlimited,
unknown and at-or-over-the-tier say nothing), the config it writes - AMD's as measured, NVIDIA's mirroring the live
4x Tesla T4 configs,
the vision entry, the drafting choice and the answers-at-once value - the cards it picks, the report's QFN lines, the
--import-model entries and the Models page's entries.  No GPU, no network.

    python -m unittest tools.test_setup_qfn -v
"""
from pathlib import Path
import contextlib
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


class GsqQuants(unittest.TestCase):
    """U8: the two GSQ-RCO quants (IST-DASLab's repos): the manifest pins (repo, revision, folder, sizes, sha256,
    apache-2.0), the ids and shown names (the Coder one shown as the code-tuned variant), the shard 2 the two files
    sets share (the PLE table, the same bytes in every GSQ quant), and --gguf-dir taking the files in place."""

    GSQ = {
        "gsq-iq3_s": ("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "ed59f92082b1e93c0e96d60a8b11aab089b52f09",
                      "IQ3_S", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S", 83.6,
                      "4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3"),
        "gsq-coder-iq1_m": ("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
                            "5348543e0147355ac9cbcb031184a3546350988e", "IQ1_M",
                            "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M", 58.4,
                            "e11083ba855e7666b48ea3f2db6a9c3a20c18751a012cc24f948de91b7087fad"),
    }
    SHARD2 = "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"      # the PLE table, both repos

    def test_the_manifest_pins(self):
        for k, (repo, rev, folder, stem, gb, sha1) in self.GSQ.items():
            m = maya.qfn_entry(k)
            self.assertEqual((m["repo"], m["revision"], m["folder"]), (repo, rev, folder), k)   # (their own repos)
            self.assertEqual(list(m["files"]), [f"{stem}-00001-of-00002.gguf", f"{stem}-00002-of-00002.gguf"], k)
            self.assertAlmostEqual(m["download_gb"], gb, delta=0.1)
            self.assertEqual(m["sha256"][f"{stem}-00001-of-00002.gguf"], sha1, k)
            self.assertEqual(m["needs"], {"vram_gb": 15, "ram_gb": 88}, k)            # (estimated, marked so)
            self.assertIn("apache-2.0", m["about"], k)                                 # (not the Qwen license)
            self.assertEqual(maya.file_url(m, f"{stem}-00001-of-00002.gguf"),
                             f"https://huggingface.co/{repo}/resolve/{rev}/{folder}/{stem}-00001-of-00002.gguf", k)

    def test_the_shared_shard_2(self):
        """Shard 2 is the PLE table: the same size and sha256 in both GSQ file sets (and in every other GSQ quant
        those repos publish); shard 1 is what differs."""
        man = json.loads((ROOT / "tools" / "qfn" / "download.json").read_text())
        quants = man["quants"]
        s2 = {k: [v for n, v in quants[k]["files"].items() if n.endswith("-00002-of-00002.gguf")][0]
              for k in self.GSQ}
        self.assertEqual(s2["gsq-iq3_s"], s2["gsq-coder-iq1_m"])
        self.assertEqual(s2["gsq-iq3_s"][1], self.SHARD2)
        self.assertNotEqual([v for n, v in quants["gsq-iq3_s"]["files"].items()
                             if n.endswith("-00001-of-00002.gguf")][0],
                            [v for n, v in quants["gsq-coder-iq1_m"]["files"].items()
                             if n.endswith("-00001-of-00002.gguf")][0])

    def test_the_ids_shown_names_and_configs(self):
        self.assertEqual(Q.resolve("qwen3.8-flash-next-iq3_s"), ("qfn", "gsq-iq3_s"))
        self.assertEqual(Q.resolve("qwen3.8-flash-next-coder-iq1_m"), ("qfn", "gsq-coder-iq1_m"))
        self.assertEqual(Q.shown("qfn", "gsq-iq3_s"), "Qwen3.8-Flash-Next · GSQ-RCO IQ3_S (saguaro-qfn)")
        self.assertEqual(Q.shown("qfn", "gsq-coder-iq1_m", 54.4),
                         "Qwen3.8-Flash-Next · GSQ-RCO Coder IQ1_M (saguaro-qfn) (54 GB)")
        self.assertIn("Coder", Q.QFN_QUANTS["gsq-coder-iq1_m"])                  # (shown as the code-tuned variant)
        self.assertIn("code-tuned", maya.qfn_entry("gsq-coder-iq1_m")["about"])   # (... and said in its about)
        self.assertEqual(Q.config_stem("qfn", "gsq-iq3_s"), "maya-qwen3.8-flash-next-iq3_s")
        self.assertEqual(Q.config_stem("qfn", "gsq-coder-iq1_m"), "maya-qwen3.8-flash-next-coder-iq1_m")

    def test_gguf_dir_takes_the_files_in_place(self):
        """The friend's box keeps its GSQ files: --gguf-dir with a quant's two shards is used in place (sizes and
        sha256 checked against the pins), nothing downloaded - for both quants; a folder missing one shard is refused."""
        for quant, (_repo, _rev, folder, stem, _gb, _sha) in self.GSQ.items():
            with self.subTest(quant=quant), tempfile.TemporaryDirectory() as t:
                src = Path(t) / folder                                    # (IQ3_S / IQ1_M, as on the Hub)
                src.mkdir()
                for n, size in maya.qfn_entry(quant)["files"].items():
                    with (src / n).open("wb") as f:
                        f.truncate(size)
                quiet = [patch.object(maya, k, lambda *a, **kw: None) for k in ("ok", "warn", "step")]
                for q_ in quiet:
                    q_.start()
                try:
                    with patch.object(maya, "sha256_checked", return_value=True), \
                            patch.object(maya, "rotational", return_value=False), \
                            patch.object(maya, "offer_download",
                                         side_effect=AssertionError("downloaded despite --gguf-dir")):
                        got = maya.qfn_files_step(args(gguf_dir=src), Path(t) / "models", quant)
                    self.assertEqual([f.name for f in got], list(maya.qfn_entry(quant)["files"]))
                    self.assertEqual({f.parent for f in got}, {src})       # (in place: nothing copied to models/)
                    (src / f"{stem}-00002-of-00002.gguf").unlink()         # (a missing shard: refused, named)
                    out = refusal(lambda: maya.qfn_files_step(args(gguf_dir=src), Path(t) / "models", quant))
                finally:
                    for q_ in quiet:
                        q_.stop()
                self.assertIn("does not hold all of", out)
                self.assertIn(f"{stem}-00002-of-00002.gguf", out)


class ExpertProfileShapes(unittest.TestCase):
    """v1.2.1-D fix (f): the shipped 48x512 ranking is copied into a quant's var/ only when its shape matches the
    quant's own expert count - read from the pack's index (the Coder keeps 256 of the 512 a layer), not hard-coded.
    With no matching profile the --expert-profile / --expert-profile-save pair stays out; a layout that pins a RAM
    tier REQUIRES a static profile of the model's own shape, so the setup stops there with a plain line naming the
    file and how to make one instead of writing a config that cannot start."""

    @staticmethod
    def index(n_experts):                                   # (a pack's index.txt: 48 router rows at the quant's width)
        rows = ["# strata pack index v3 -- generated by tools/iq_pack.py (native experts) from x.gguf",
                "# align 64 pool 0 tensors 48"]
        rows += [f"blk.{l}.ffn_gate_inp.weight 0 4 0 12 0 12 4096 {n_experts} 0 0 1 " + "0 " * 7
                 for l in range(48)]
        return "\n".join(rows) + "\n"

    @staticmethod
    def profile_bytes(n_layers, n_experts):                 # (the header the engine reads: "STRP", v1, layers, experts)
        import struct
        return struct.pack("<4sIII", b"STRP", 1, n_layers, n_experts)

    def written(self, t: Path, quant, n_experts, shipped=(48, 512), mem=(220.0, 200.0), old=None, out=None):
        models, d = t / "models", t / "models" / maya.qfn_entry(quant)["name"]
        (d / "pack").mkdir(parents=True, exist_ok=True)
        (d / "pack" / "index.txt").write_text(self.index(n_experts))
        (t / "data").mkdir(parents=True, exist_ok=True)
        if shipped:                                         # (the shipped ranking under the patched ROOT)
            (t / "data" / "expert-profile-qfn.bin").write_bytes(self.profile_bytes(*shipped))
        files = [t / "gguf" / n for n in maya.qfn_entry(quant)["files"]]
        with cpu(), patch.object(maya, "ROOT", t), patch.object(maya, "mem_gb", return_value=mem), \
                patch.object(maya, "write_run_script", return_value=t / "run.sh"), \
                patch("sys.stdout", new=out or io.StringIO()), patch.object(maya.S, "UI", None):
            p = maya.write_config_qfn(args(backend="cuda"), {"backend": "cuda", "gpus": [
                {"index": 0, "name": "RTX A5000", "vram_gb": 24.0, "arch": "86"}]}, {}, quant, files,
                d / "pack", None, 65536, models, d, old or {})
        return json.loads(p.read_text()), d / "var" / "expert-profile.bin"

    def stopped(self, t: Path, **kw) -> str:               # (the setup's plain line when it refuses to write)
        out = io.StringIO()
        try:
            self.written(t, out=out, **kw)
        except SystemExit:
            return out.getvalue()
        raise AssertionError("no refusal: " + out.getvalue()[-400:])

    def test_the_header_and_the_index_are_read(self):
        self.assertEqual(maya.qfn_profile_shape(ROOT / "data" / "expert-profile-qfn.bin"), (48, 512))
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            self.assertEqual(maya.qfn_profile_shape(t / "nope.bin"), None)
            (t / "junk.bin").write_bytes(b"nope" * 8)
            self.assertEqual(maya.qfn_profile_shape(t / "junk.bin"), None)
            self.assertEqual(maya.qfn_expert_shape(t / "pack"), None)          # (no pack: no shape, old rule)
            (t / "pack").mkdir()
            (t / "pack" / "index.txt").write_text(self.index(256))
            self.assertEqual(maya.qfn_expert_shape(t / "pack"), (48, 256))     # (the Coder's width from the router)

    def test_a_matching_shape_is_copied_and_wired(self):
        with tempfile.TemporaryDirectory() as t:
            cfg, profile = self.written(Path(t), "ud-q4_k_xl", 512)            # (512 a layer: the shipped shape)
            self.assertTrue(profile.is_file())                                 # (copied into var/ as before)
            self.assertIn("--expert-profile", cfg["args"])
            self.assertIn("--expert-profile-save", cfg["args"])

    def test_the_coder_does_not_get_the_512_profile(self):
        with tempfile.TemporaryDirectory() as t:                               # (220 GB: no tier: no static profile)
            cfg, profile = self.written(Path(t), "gsq-coder-iq1_m", 256)
            self.assertFalse(profile.exists())                                 # (never copied into var/)
            self.assertNotIn("--expert-profile", cfg["args"])                  # (the read side stays out ...)
            self.assertNotIn("--expert-profile-save", cfg["args"])             # ( ... and the save's file with it)
            self.assertNotIn("--resident-budget-gib", cfg["args"])

    def test_the_coder_with_a_tier_stops_and_says_how_to_make_one(self):
        with tempfile.TemporaryDirectory() as t:                               # (60 GB: a tier: needs a profile)
            out = self.stopped(Path(t), quant="gsq-coder-iq1_m", n_experts=256, mem=(60.0, 40.0))
        self.assertIn("expert-profile.bin", out)                               # (the file that is missing ...)
        self.assertIn("48 layers x 256 experts", out)                          # ( ... and its shape)
        self.assertIn("tools/make_profile.py --no-base --n-expert 256", out)   # ( ... and how to make one)
        self.assertIn("48 x 512", out)                                         # (what the shipped one is)

    def test_a_stale_512_copy_is_not_read_and_a_tier_stops_on_it(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            cfg, profile = self.written(t, "gsq-coder-iq1_m", 256)             # (no tier: nothing written ...)
            profile.parent.mkdir(parents=True, exist_ok=True)                  # (Paul's old setup already copied ...)
            profile.write_bytes(self.profile_bytes(48, 512))                   # ( ... the 48x512 file in)
            cfg, _ = self.written(t, "gsq-coder-iq1_m", 256)                   # (no tier: it is left, unread)
            self.assertEqual(profile.read_bytes(), self.profile_bytes(48, 512))   # (not overwritten)
            self.assertNotIn("--expert-profile", cfg["args"])
            out = self.stopped(t, quant="gsq-coder-iq1_m", n_experts=256, mem=(60.0, 40.0))
            self.assertIn("a different artifact", out)                         # (named as one, not read)

    def test_a_pack_that_does_not_read_keeps_the_old_rule(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            models, d = t / "models", t / "models" / "Qwen3.8-Flash-Next-UD-Q4_K_XL"
            (t / "data").mkdir(parents=True)
            (t / "data" / "expert-profile-qfn.bin").write_bytes(self.profile_bytes(48, 512))
            files = [t / "gguf" / n for n in maya.qfn_entry("ud-q4_k_xl")["files"]]
            with cpu(), patch.object(maya, "ROOT", t), patch.object(maya, "mem_gb", return_value=(220.0, 200.0)), \
                    patch.object(maya, "write_run_script", return_value=t / "run.sh"), \
                    patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
                p = maya.write_config_qfn(args(backend="cuda"), {"backend": "cuda", "gpus": [
                    {"index": 0, "name": "RTX A5000", "vram_gb": 24.0, "arch": "86"}]}, {}, "ud-q4_k_xl", files,
                    d / "pack", None, 65536, models, d, {})       # (no pack at all: the pre-(f) copy and wiring)
            cfg = json.loads(p.read_text())
            self.assertTrue((d / "var" / "expert-profile.bin").is_file())
            self.assertIn("--expert-profile", cfg["args"])


class YesAloneNamesDownloadModel(unittest.TestCase):
    """v1.2.1-D: --yes without --download-model answers the download question no, so the picture reader and the
    draft layer are skipped by design - and the two warnings that say so must name the flag that says yes
    (the queue note 11-24 20:45: "the message could say so")."""

    def test_the_picture_reader_warning_names_the_flag(self):
        with tempfile.TemporaryDirectory() as t:
            t, d = Path(t), Path(t) / "models" / "Qwen3.8-Flash-Next-UD-Q4_K_XL"
            said = []
            with patch.object(maya, "step", lambda *x: None), patch.object(maya, "warn", said.append), \
                    patch.object(maya, "offer_download", return_value=False):
                self.assertIsNone(maya.qfn_vision_step(args(), pc_hip(), {}, t / "llama", d, []))
        self.assertIn("text only", said[0])
        self.assertIn("--download-model", said[0])               # (the flag that answers the question yes)
        self.assertIn("--yes", said[0])                          # (and the one that answered it no)

    def test_the_draft_layer_warning_names_the_flag(self):
        with tempfile.TemporaryDirectory() as t:
            said = []
            with patch.object(maya, "warn", said.append), patch.object(maya, "ok", lambda *x: None), \
                    patch.object(maya, "step", lambda *x: None), \
                    patch.object(maya, "offer_download", return_value=False):
                self.assertIsNone(maya.qfn_mtp_step(args(), Path(t) / "models"))
        self.assertIn("drafting off", said[0])
        self.assertIn("--download-model", said[0])
        self.assertIn("--yes", said[0])


def fake_gguf(path: Path, tensors, truncate=0):
    """A whole little GGUF v3 file: F32 tensors written in the given order after the header
    (tensors: [(name, [shape], [floats])]); truncate lops the last bytes off (a short file)."""
    import struct

    def s(b):
        return struct.pack("<Q", len(b)) + b
    hdr = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), 1))
    hdr += s(b"general.architecture") + struct.pack("<I", 8) + s(b"qwen35next")   # (the only kv)
    data = b""
    offs = []
    for name, shape, vals in tensors:
        offs.append(len(data))
        data += struct.pack(f"<{len(vals)}f", *vals)
    for (name, shape, _), off in zip(tensors, offs):
        hdr += s(name.encode()) + struct.pack(f"<I{len(shape)}Q", len(shape), *shape)
        hdr += struct.pack("<IQ", 0, off)                # type 0 = F32
    path.write_bytes(bytes(hdr) + b"\0" * (-len(hdr) % 32) + data)   # (the data section starts aligned to 32)
    if truncate:
        with open(path, "r+b") as f:
            f.truncate(f.seek(0, 2) - truncate)


class MergedGguf(unittest.TestCase):
    """v1.2.1-D: --gguf-dir holding ONE merged .gguf (gguf-split --merge, an Ollama blob hard-linked to a name)
    instead of the pinned shards: accepted when the tensor names, shapes and types match the pinned shards'
    index and the bytes of a sample of tensors (the first, the last, 8 spread between) match the shards' - and,
    where the shards are not on the PC at all, on the file's own complete tensor directory, said with "not
    sha256-checked".  A merged file that differs is refused by what differed."""

    QUANT = "gsq-iq3_s"
    TS = [("token_embd.weight", [4], [0.0, 1.0, 2.0, 3.0]),
          ("blk.0.attn_norm.weight", [3], [4.0, 5.0, 6.0]),
          ("blk.0.ffn_gate_inp.weight", [2, 2], [7.0, 8.0, 9.0, 10.0]),
          ("blk.47.ffn_exps.7.weight", [5], [11.0, 12.0, 13.0, 14.0, 15.0]),
          ("output.weight", [6], [16.0, 17.0, 18.0, 19.0, 20.0, 21.0])]

    @contextlib.contextmanager
    def quiet(self, said=None):
        olds = dict(warn=lambda *x: None, step=lambda *x: None, say=lambda *x: None,
                    rotational=lambda *x: False,
                    offer_download=lambda *x: (_ for _ in ()).throw(AssertionError("downloaded despite --gguf-dir")))
        if said is not None:
            olds["ok"] = lambda *x: said.append(" ".join(str(v) for v in x))
        else:
            olds["ok"] = lambda *x: None
        with contextlib.ExitStack() as st:
            for k, v in olds.items():
                st.enter_context(patch.object(maya, k, v))
            yield

    def shards(self, t: Path):
        """The pinned shard names as whole little GGUFs in t, the five tensors spread over the two files."""
        parts = {0: (0, 2, 4), 1: (1, 3)}               # (shard 1: embd, router, out; shard 2: norm, expert)
        for i, name in enumerate(maya.qfn_entry(self.QUANT)["files"]):
            fake_gguf(t / name, [self.TS[j] for j in parts[i]])
        return t

    def test_merged_file_accepted_against_the_shards_next_door(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            src, prev = t / "merged", t / "shards"
            src.mkdir()
            prev.mkdir()
            self.shards(prev)
            fake_gguf(src / "Qwen-merged.gguf", [self.TS[j] for j in (3, 0, 4, 1, 2)])   # (another order: names
            said = []                                                   #  must decide, not the order in the file)
            a = args(gguf_dir=src)
            a.qfn_prev_files_dir = prev              # (the folder the config before read the files from)
            with self.quiet(said), patch.object(maya, "sha256_checked",
                                                side_effect=AssertionError("sha256 of a merged file")):
                got = maya.qfn_files_step(a, t / "models", self.QUANT)
            self.assertEqual([f.name for f in got], ["Qwen-merged.gguf"])
            self.assertIn("match the pinned shards' index", said[0])     # (what was checked, said)
            self.assertIn("sample of 5", said[0])                        # (all five sampled: first, last, 8 spread)
            self.assertNotIn("not sha256-checked", said[0])

    def test_one_flipped_sample_tensor_refuses_it(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            src, prev = t / "merged", t / "shards"
            src.mkdir()
            prev.mkdir()
            self.shards(prev)
            ts = [list(x) for x in self.TS]
            ts[2][2][0] = 99.0                                           # (same names, shapes and types ...)
            fake_gguf(src / "Qwen-merged.gguf", [self.TS[j][:2] + (tuple(ts[j][2]),) for j in (3, 0, 4, 1, 2)])
            a = args(gguf_dir=src)
            a.qfn_prev_files_dir = prev                                  # (the shards are on the PC: the index is)
            out = refusal(lambda: maya.qfn_files_step(a, t / "models", self.QUANT))
        self.assertIn("bytes differ", out)
        self.assertIn("blk.0.ffn_gate_inp.weight", out)

    def test_differing_tensors_refuse_it(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            src, prev = t / "merged", t / "shards"
            src.mkdir()
            prev.mkdir()
            self.shards(prev)
            fake_gguf(src / "other.gguf", self.TS[:-1])                  # (no output.weight: another model's file)
            a = args(gguf_dir=src)
            a.qfn_prev_files_dir = prev
            out = refusal(lambda: maya.qfn_files_step(a, t / "models", self.QUANT))
        self.assertIn("its tensors are not the pinned shards'", out)
        self.assertIn("output.weight", out)

    def test_no_shards_on_the_pc_accepts_it_not_sha256_checked(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            src = t / "merged"
            src.mkdir()
            fake_gguf(src / "ollama-blob.gguf", self.TS)
            said = []
            with self.quiet(said), patch.object(maya, "sha256_checked",
                                                side_effect=AssertionError("sha256 of a merged file")):
                got = maya.qfn_files_step(args(gguf_dir=src), t / "models", self.QUANT)
            self.assertEqual([f.name for f in got], ["ollama-blob.gguf"])
            self.assertIn("not sha256-checked", said[0])                 # (said plainly: nothing to compare with)
            self.assertIn("tensor names, shapes and types", said[0])

    def test_a_truncated_merged_file_refuses_it(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            src = t / "merged"
            src.mkdir()
            fake_gguf(src / "half.gguf", self.TS, truncate=8)            # (its tensor bytes no longer reach the end)
            out = refusal(lambda: maya.qfn_files_step(args(gguf_dir=src), t / "models", self.QUANT))
        self.assertIn("does not hold", out)
        self.assertIn("whole", out)

    def test_two_unknown_ggufs_are_the_usual_refusal(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            src = t / "merged"
            src.mkdir()
            fake_gguf(src / "a.gguf", self.TS)
            fake_gguf(src / "b.gguf", self.TS)
            out = refusal(lambda: maya.qfn_files_step(args(gguf_dir=src), t / "models", self.QUANT))
        self.assertIn("does not hold all of", out)                       # (nothing to name: the usual line)


class PascalBuild(unittest.TestCase):
    """U8 step 2: the P40 build line. An sm_61 card builds with CUDA 12.x (nvcc_range, test_maya_pascal) and a host
    g++ that toolkit accepts: NVIDIA's supported-matrix stops CUDA 12.0 - 12.3 at gcc 12, so on a PC whose default
    g++ is newer the setup tries g++-12 first (the P40's hand-made build used gcc-12 / g++-12 for C, C++ and CUDA's
    host code alike) and names the package to install when none fits. compile_engine for QFN_TARGET then writes the
    cmake line; the top CMakeLists turns an all-below-sm_75 arch list into engines/qfn's -DSTRATA_EXPERIMENTAL_SM60=ON."""

    def test_the_cuda_gcc_matrix(self):
        self.assertEqual(maya.cuda_gcc_max((12, 0)), 12)
        self.assertEqual(maya.cuda_gcc_max((12, 3)), 12)
        self.assertEqual(maya.cuda_gcc_max((12, 4)), 13)
        self.assertEqual(maya.cuda_gcc_max((12, 8)), 14)
        self.assertIsNone(maya.cuda_gcc_max((13, 0)))         # (no ceiling pinned: the probe tries)
        self.assertIsNone(maya.cuda_gcc_max(None))

    def test_the_host_order(self):
        base = [None, "g++-14", "g++-13", "g++-12"]           # (the default g++ first, then installed g++-N)
        with patch.object(maya, "gcc_major", lambda: 14):
            self.assertEqual(maya.host_order(base, (12, 0)), ["g++-12", None, "g++-14", "g++-13"])
            self.assertEqual(maya.host_order(base, (12, 4)), ["g++-13", None, "g++-14", "g++-12"])
            self.assertEqual(maya.host_order(base, (13, 0)), base)
        with patch.object(maya, "gcc_major", lambda: 12):
            self.assertEqual(maya.host_order(base, (12, 0)), base)          # (the default fits: as installed)
        with patch.object(maya, "gcc_major", lambda: 14):
            self.assertEqual(maya.host_order([None, "g++-14"], (12, 0)), [None, "g++-14"])   # (nothing fitting)

    def test_the_install_hint_names_the_package(self):
        with patch.object(maya, "gcc_major", lambda: 14), \
                patch.object(maya, "gpp_versions", lambda: ["g++-14", "g++-13"]):
            self.assertIn("sudo apt-get install -y g++-12", maya.missing_gpp_hint([("/usr/bin/nvcc", (12, 0))]))
        with patch.object(maya, "gcc_major", lambda: 14), patch.object(maya, "gpp_versions", lambda: ["g++-12"]):
            self.assertEqual(maya.missing_gpp_hint([("/usr/bin/nvcc", (12, 0))]), "")    # (installed: nothing to say)

    def test_toolchain_picks_gpp12_over_the_default_gpp14(self):
        """The P40 box of the plan: CUDA 12.0's nvcc, default g++ 14, g++-12 installed. The probe starts at g++-12
        and that pair is taken - the default g++ is not even tried (nvcc would only answer 'unsupported GNU
        version')."""
        probes = []

        def fake_run(cmd, **kw):
            probes.append(cmd)
            return SimpleNamespace(returncode=0 if any(str(x).endswith("g++-12") for x in cmd) else 1)

        quiet = [patch.object(maya, n, lambda *x, **k: None) for n in ("say", "ok", "warn")]
        for q_ in quiet:
            q_.start()
        try:
            with patch.object(maya, "nvcc_candidates", lambda: [str(Path(maya.__file__))]), \
                    patch.object(maya.S, "nvcc_version", lambda c: (12, 0)), \
                    patch.object(maya, "gcc_major", lambda: 14), \
                    patch.object(maya, "gpp_versions", lambda: ["g++-14", "g++-13", "g++-12"]), \
                    patch.object(maya.shutil, "which", lambda n: f"/usr/bin/{n}"), \
                    patch("subprocess.run", side_effect=fake_run):
                self.assertEqual(maya.toolchain_for([61], None, None), (str(Path(maya.__file__)), "g++-12"))
        finally:
            for q_ in quiet:
                q_.stop()
        self.assertIn("/usr/bin/g++-12", probes[0])            # (the first probe already uses it)
        self.assertEqual(len(probes), 1)                       # (no wasted round on the default g++ or g++-14)

    def test_toolchain_says_which_package_to_install(self):
        """The same box without g++-12: nothing compiles, and the closing line says to install it."""
        said = []

        def fake_run(cmd, **kw):
            return SimpleNamespace(returncode=1)

        quiet = [patch.object(maya, n, lambda *x, **k: None) for n in ("say", "ok")]
        for q_ in quiet:
            q_.start()
        try:
            with patch.object(maya, "nvcc_candidates", lambda: [str(Path(maya.__file__))]), \
                    patch.object(maya.S, "nvcc_version", lambda c: (12, 0)), \
                    patch.object(maya, "gcc_major", lambda: 14), \
                    patch.object(maya, "gpp_versions", lambda: ["g++-14", "g++-13"]), \
                    patch.object(maya.shutil, "which", lambda n: f"/usr/bin/{n}"), \
                    patch.object(maya, "warn", side_effect=lambda *x: said.append(" ".join(map(str, x)))), \
                    patch("subprocess.run", side_effect=fake_run):
                nvcc, hc = maya.toolchain_for([61], None, None)
        finally:
            for q_ in quiet:
                q_.stop()
        self.assertEqual(hc, None)                             # (nothing better than the default to hand)
        self.assertTrue(any("sudo apt-get install -y g++-12" in s for s in said), said)

    def test_the_p40_qfn_cmake_line(self):
        """compile_engine for saguaro-qfn on the sm_61 card with CUDA 12.0's nvcc and the g++-12 it takes: arch 61,
        the QFN target, and one host compiler named for nvcc, C and C++ (the P40 build recipe; engines/qfn's
        ExternalProject inherits it from the top project)."""
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            build, stamp = root / "build", root / "build" / "MAYA-BUILD.json"
            build.mkdir()
            got = []

            def fake_steps(conf, build_cmd, env, bat, nvcc=None):
                got.extend([conf, build_cmd])
                return None

            quiet = [patch.object(maya, n, lambda *x, **k: None) for n in ("say", "ok")]
            for q_ in quiet:
                q_.start()
            try:
                with patch.object(maya, "BUILD", build), patch.object(maya, "STAMP", stamp), \
                        patch.object(maya, "pick_cmake", return_value="cmake"), \
                        patch.object(maya, "cached_generator", return_value=None), \
                        patch.object(maya, "venv_tool", return_value=None), \
                        patch.object(maya, "toolkit_root_defs", return_value=[]), \
                        patch.object(maya, "cuda_lib_dirs", return_value=[]), \
                        patch.object(maya, "mem_gb", return_value=(31.0, 24.0)), \
                        patch.object(maya.shutil, "which",
                                     lambda n: f"/usr/bin/{n}" if n in ("gcc-12", "g++-12") else None), \
                        patch.object(maya, "cmake_steps", side_effect=fake_steps):
                    meta = maya.compile_engine([61], [0], "/usr/bin/nvcc", "g++-12", root / "llama", "src-sha",
                                               target=maya.QFN_TARGET)
            finally:
                for q_ in quiet:
                    q_.stop()
        conf, build_cmd = got
        for want in ("-DCMAKE_CUDA_ARCHITECTURES=61", "-DSAGUARO_QFN=ON", "-DCMAKE_CUDA_COMPILER=/usr/bin/nvcc",
                     "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12", "-DCMAKE_C_COMPILER=/usr/bin/gcc-12",
                     "-DCMAKE_CXX_COMPILER=/usr/bin/g++-12"):
            self.assertIn(want, conf)
        self.assertNotIn("-DCUDAToolkit_ROOT=/usr", conf)      # (the distribution's nvcc: CMake's own search)
        self.assertEqual(build_cmd[build_cmd.index("--target") + 1], maya.QFN_TARGET)
        self.assertEqual(meta["host_compiler"], "g++-12")
        # engines/qfn's Pascal switch rides with it: the top CMakeLists gives the external project
        # STRATA_EXPERIMENTAL_SM60=ON when every arch is below sm_75 (and only then)
        top = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("if(_qfn_old AND NOT _qfn_new)", top)
        self.assertIn("-DSTRATA_EXPERIMENTAL_SM60=ON", top)


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
        """v1.2.1-B: a quant is refused only under its true minimum (6.0 GB on the card at UD-Q4, 8.7 at Q8_0;
        16-19 GB of RAM) - 62 GB of RAM or an 11.9 GB card RUN, slowly: the layout warns and confirms.
        Review finding 2: the minimum is per card OF THE SPLIT the layout would use - a pair of 8 GB cards runs
        Q8_0 slowly (5.4 GB a card), where one 8 GB card alone cannot hold it."""
        g8 = {"index": 0, "name": "GeForce RTX 4060", "vram_gb": 8.0, "arch": "89"}
        g16 = {"index": 0, "name": "AMD Radeon RX 9070", "vram_gb": 15.9, "arch": "gfx1201"}
        pair = [g8, dict(g8, index=1)]
        self.assertIsNone(maya.qfn_quant_fit("ud-q4_k_xl", [g16], 91.7))   # (an RX 9070 XT with 96 GB: measured)
        self.assertIsNone(maya.qfn_quant_fit("ud-q4_k_xl", [g16], 62.0))   # (the old "96 GB of RAM" refusal: slow)
        self.assertIsNone(maya.qfn_quant_fit("ud-q4_k_xl", [{"vram_gb": 11.9}], 91.7))   # (the "16 GB or more")
        self.assertIsNone(maya.qfn_quant_fit("ud-q4_k_xl", [g8], 91.7))    # (6.0 GB needed: it runs, slowly)
        self.assertIsNone(maya.qfn_quant_fit("q8_0", pair, 91.7))          # (the 8 GB pair: 5.4 GB a card by the
                                                           #  split - review finding 2: runs, slowly, not refused)
        self.assertIn("8 GB cannot hold the 8.7 GB", maya.qfn_quant_fit("q8_0", [g8], 91.7))   # (alone: cannot)
        self.assertIn("cannot run with 12 GB of RAM",
                      maya.qfn_quant_fit("cyber-frost-q5_k_m", [g16], 12.0))
        out = refusal(lambda: maya.choose_qfn_quant(args(qfn_quant="q8_0"), {}, [g8], 91.7))
        self.assertIn("cannot run on this GPU", out)
        self.assertIn("8 GB cannot hold the 8.7 GB", out)
        self.assertEqual(maya.choose_qfn_quant(args(qfn_quant="q8_0"), {}, pair, 91.7), "q8_0")   # (the pair: the
        self.assertEqual(maya.choose_qfn_quant(args(qfn_quant="ud-q4_k_xl"), {}, [g8], 62.0),     #  menu takes it)
                         "ud-q4_k_xl")

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
    --ple-gguf) is rewritten to a saguaro-qfn config of the same file - once: idempotent.  The class asks no GPU:
    the cards a migrating config names come from a patched S.gpus / S.amd_gpus ([] unless the test says)."""

    def setUp(self):
        for name in ("gpus", "amd_gpus"):
            p = patch.object(maya.S, name, lambda *a: [])
            p.start()
            self.addCleanup(p.stop)

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

    def test_the_reserve_is_its_own_or_the_rule(self):
        """v1.2.1-A: the migration keeps the --vram-reserve-mib the old config had, and writes the display rule of
        this PC's cards when it had none (the class's patched S.gpus / S.amd_gpus answer for the cards here)."""
        old = json.loads(json.dumps(self.OLD))
        old["args"] += ["--vram-reserve-mib", "1234"]
        self.assertTrue(maya.migrate_cyber_frost(old))
        self.assertEqual(old["args"][old["args"].index("--vram-reserve-mib") + 1], "1234")   # (kept as it was)
        self.assertEqual(self.migrated()["args"][self.migrated()["args"].index("--vram-reserve-mib") + 1],
                         "3072")                                          # (no cards here: the rule's unknown case)
        with patch.object(maya.S, "amd_gpus", lambda *a: [{"index": 0, "display": False, "vram_used_gb": 0.1}]):
            self.assertEqual(self.migrated()["args"][self.migrated()["args"].index("--vram-reserve-mib") + 1],
                             "512")                                       # (the PC's headless card: the rule)

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


class ReserveRule(unittest.TestCase):
    """v1.2.1-A: what --vram-reserve-mib a fresh QFN config writes, per vendor - the display rule (3072 where a
    desktop may run or the display is unknown, 512 where no card has one, 1024 on a Linux AMD APU) with the user's
    --vram-reserve-mib above the rule."""

    def written(self, gpus, backend="cuda", ram=252.0, **kw):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            models, d = t / "models", t / "models" / maya.qfn_entry("ud-q4_k_xl")["name"]
            files = [t / "gguf" / n for n in maya.qfn_entry("ud-q4_k_xl")["files"]]
            with cpu(), patch.object(maya, "ROOT", t), patch.object(maya, "WIN", False), \
                    patch.object(maya, "mem_gb", return_value=(ram, ram - 12.0)), \
                    patch.object(maya, "write_run_script", return_value=t / "run.sh"), \
                    patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
                p = maya.write_config_qfn(args(backend=backend, **kw),
                                          {"backend": backend, "gpus": [dict(g, index=i) for i, g in enumerate(gpus)]},
                                          {}, "ud-q4_k_xl", files, d / "pack", None, 65536, models, d, {})
            a = json.loads(p.read_text())["args"]
        return a[a.index("--vram-reserve-mib") + 1]

    def test_nvidia(self):
        t4 = {"name": "Tesla T4", "vram_gb": 15.0, "arch": "75"}
        self.assertEqual(self.written([{**t4, "display": True}]), "3072")     # (a GeForce driving the desktop)
        self.assertEqual(self.written([t4]), "3072")                          # (display_active unknown)
        self.assertEqual(self.written([{**t4, "display": False}]), "512")     # (no more 700 on a headless card)
        self.assertEqual(self.written([{**t4, "display": False}, {**t4, "display": False}]), "512")
        self.assertEqual(self.written([{**t4, "display": False}, {**t4, "display": True}]), "3072")
        self.assertEqual(self.written([t4], vram_reserve_mib=1234), "1234")   # (asked for: above the rule)

    def test_amd(self):
        card = {"arch": "gfx1201", "vram_gb": 15.9}
        self.assertEqual(self.written([card], backend="hip"), "3072")         # (no sysfs facts: the desktop's room)
        self.assertEqual(self.written([{**card, "display": True}], backend="hip"), "3072")
        self.assertEqual(self.written([{**card, "display": False, "vram_used_gb": 0.1}], backend="hip"), "512")
        self.assertEqual(self.written([{**card, "display": False, "vram_used_gb": 0.1}], backend="hip",
                                      vram_reserve_mib=2048), "2048")
        apu = {"arch": "gfx1151", "name": "AMD Radeon 8060S", "vram_gb": 96}
        self.assertEqual(self.written([apu], backend="hip"), "1024")          # (the Linux APU case, unchanged)


class NvidiaConfig(unittest.TestCase):
    def test_the_live_t4_settings_on_four_tesla_t4s(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            models, d = t / "models", t / "models" / "Qwen3.8-Flash-Next-UD-Q4_K_XL"
            files = [t / "gguf" / n for n in maya.qfn_entry("ud-q4_k_xl")["files"]]
            pc = {"backend": "cuda", "gpus": [{"index": i, "name": "Tesla T4", "vram_gb": 15.0, "arch": "75",
                                               "display": False}    # (nvidia-smi's display_active: Disabled)
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
      - --vram-reserve-mib: the live box was set up before v1.2.1-A and holds 700; a fresh setup writes the
        display rule's 512 - nvidia-smi's display_active says Disabled on these compute-only T4s;
      - --resident-budget-gib: neither has one (252 GB: the page cache holds the experts);
      - the file paths, the port, the log, the installer block: this machine's, not the 4 x T4 box's."""

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
        pc = {"backend": "cuda", "gpus": [{"index": i, "name": "Tesla T4", "vram_gb": 15.0, "arch": "75",
                                           "display": False}    # (nvidia-smi's display_active: Disabled, the live box)
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

        def machine_free(pairs):     # (the paths, --max-context and --vram-reserve-mib are the intended
            return {(f, v) for f, v in pairs                           #  differences, listed above)
                    if f not in self.PATH_FLAGS and f not in ("--max-context", "--vram-reserve-mib")}
        self.assertEqual(machine_free(self.flag_pairs(cfg["args"])), machine_free(self.flag_pairs(self.LIVE_ARGS)))
        a = cfg["args"]
        self.assertEqual(a[a.index("--max-context") + 1], "262144")    # (the split's recommendation; live: 1M)
        self.assertEqual(a[a.index("--vram-reserve-mib") + 1], "512")  # (the display rule: no display on a T4;
                                                                       #  the live box holds its setup-time 700)
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
    card's (8.7 GB even at Q8_0)."""

    QUANTS = ["ud-q4_k_xl", "q8_0", "cyber-frost-q5_k_m"]
    RAMS = [32.0, 64.0, 96.0, 256.0]
    SLOW_RAM = [32.0, 64.0]                                      # (under the comfortable MIN_RAM_GB: warn + confirm)
    SPLIT = {"ud-q4_k_xl": "11,24,36", "q8_0": "10,22,35", "cyber-frost-q5_k_m": "12,24,36"}

    def pc(self, n):
        return {"backend": "cuda", "gpus": [{"index": i, "name": "Tesla T4", "vram_gb": 15.0, "arch": "75",
                                             "display": False}   # (nvidia-smi's display_active: Disabled, these boxes)
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
        machine_free = lambda pairs: {(f, v) for f, v in pairs if f not in LiveT4Parity.PATH_FLAGS
                                      and f not in ("--max-context", "--vram-reserve-mib")}
        self.assertEqual(machine_free(LiveT4Parity.flag_pairs(cfg["args"])),
                         machine_free(LiveT4Parity.flag_pairs(LiveT4Parity.LIVE_ARGS)))
        self.assertEqual(cfg["args"][cfg["args"].index("--vram-reserve-mib") + 1], "512")   # (headless T4s: the
                                                                                            #  display rule; live: 700)
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
        gpus = [{"index": i, "name": "Tesla T4", "vram_gb": 15.0, "arch": "75", "display": False} for i in range(n)]
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
                self.assertEqual(cfg["vision"], {**self.VISION, "elastic_mib": cfg["vision"]["elastic_mib"]})
                self.assertEqual(cfg["vision"]["elastic_mib"], 1536 + 512)   # (the reader and the engine's own room beside it)
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
        self.assertIn("the setup scales this down: drafting off", slow)   #  write - no question; no --mtp-max-t
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


class Memlock(unittest.TestCase):
    """P3: the soft RLIMIT_MEMLOCK beside a planned RAM tier (the field case: 2x Tesla T4, `ulimit -l` 8 GiB, the
    71 GiB tier pinned 8 and said nothing).  One warning - the limit, the planned tier, and the fix - only when the
    limit is a number under the tier; unlimited, unknown, at/over the tier, or no tier planned: nothing is said."""

    def warned(self, soft, planned):
        said = []
        with patch.object(maya, "memlock_soft_gib", lambda: soft), \
                patch.object(maya, "warn", side_effect=lambda *x: said.append(" ".join(map(str, x)))):
            maya.memlock_tier_warn(planned)
        return "\n".join(said)

    def test_a_64_kib_limit_under_a_71_gib_tier_says_so(self):
        out = self.warned(64 * 1024 / 2**30, 71)                      # (the defaults'-era 64 KiB: named in KiB,
        self.assertIn("64 KiB", out)                                  #  as `ulimit -l` shows it)
        self.assertIn("71 GB", out)

    def test_an_8_gib_limit_under_a_71_gib_tier_says_so(self):        # (the field case)
        out = self.warned(8.0, 71)
        self.assertIn("8 GiB", out)
        self.assertIn("71 GB", out)

    def test_unlimited_and_unknown_say_nothing(self):
        self.assertEqual(self.warned(float("inf"), 71), "")           # (unlimited: fine)
        self.assertEqual(self.warned(None, 71), "")                   # (unknown: no warning)

    def test_a_limit_at_or_over_the_tier_says_nothing(self):
        self.assertEqual(self.warned(71.0, 71), "")                   # (exactly the tier: enough to pin it)
        self.assertEqual(self.warned(80.0, 71), "")
        self.assertEqual(self.warned(8.0, None), "")                  # (no tier planned: nothing to compare)

    def test_the_wording_once(self):
        out = self.warned(8.0, 71)
        self.assertIn("memory-lock limit (`ulimit -l`)", out)         # (the limit, named as the shell shows it)
        self.assertIn("pinned RAM tier", out)                         # (what is planned)
        self.assertIn("LimitMEMLOCK=infinity", out)                   # (systemd: the unit line...)
        self.assertIn("systemctl daemon-reload", out)                 # (... and the reload / restart)
        self.assertIn('"<user> - memlock unlimited"', out)            # (login shells: the limits.conf line...)
        self.assertIn("/etc/security/limits.conf, then log out and in", out)   # (... and logging back in)

    def test_the_reader_reads_the_limit_and_maps_unlimited_to_inf(self):
        """The unpatched reader on Linux: the soft limit in GiB, unlimited as inf (non-Linux: None - skipped)."""
        if not sys.platform.startswith("linux"):
            self.skipTest("the reader is Linux-only: non-Linux skips")
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        want = float("inf") if soft == resource.RLIM_INFINITY else soft / 2**30
        self.assertEqual(maya.memlock_soft_gib(), want)

    def test_the_setup_fit_step_and_check_call_it(self):
        """set_up_qfn's fit step (step 2, where the plan is made) makes the call: the flow with an 8 GiB limit
        prints the line, unlimited prints nothing; and main's --check branch carries the same call beside its
        qfn_check_line."""
        import inspect
        with tempfile.TemporaryDirectory() as t, patch.object(maya, "memlock_soft_gib", lambda: 8.0):
            out = io.StringIO()
            SetupFlow().setup(Path(t), _out=out)
        self.assertIn("memory-lock limit (`ulimit -l`) is 8 GiB", out.getvalue())
        with tempfile.TemporaryDirectory() as t, patch.object(maya, "memlock_soft_gib",
                                                              lambda: float("inf")):
            out2 = io.StringIO()
            SetupFlow().setup(Path(t), _out=out2)
        self.assertNotIn("memory-lock", out2.getvalue())
        self.assertIn('memlock_tier_warn(lay["resident_budget_gib"])', inspect.getsource(maya.set_up_qfn))
        self.assertIn('memlock_tier_warn(lay["resident_budget_gib"])', inspect.getsource(maya.main))


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
        # (4.1 GB: the 4060s are equal, so the refusal prices the 2-card split - half the model per card)
        self.assertIn("cannot run on this PC's GeForce RTX 4060: its 4 GB cannot hold the 4.1 GB", out)
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

    def test_a_display_card_beside_a_good_one_is_not_a_refusal(self):
        """Review finding 1: a 4090 plus a 1 GB GT 710 (cc 3.5 - below MIN_CC 61, a card choose_gpus can never
        take) is a PC the code before this branch set up fine: the pre-guard prices the layout the PICKER would
        use (best_gpus: the 4090 alone), not one split over every card nvidia-smi lists."""
        cards = [{"index": 0, "name": "RTX 4090", "vram_gb": 24.0, "arch": "89", "driver": "550"},
                 {"index": 1, "name": "GeForce GT 710", "vram_gb": 1.0, "arch": "35", "driver": "550"}]
        out, got = io.StringIO(), []
        ps = [patch.object(maya.S, "gpus", return_value=cards),
              patch.object(maya.S, "is_wsl", return_value=False),
              patch.object(maya.S, "cpu_info", return_value=("CPU", True, True)),
              patch.object(maya.S, "out", return_value="g++ (test) 13.0"),
              patch.object(maya, "oomd_active", return_value=False),
              patch.object(maya, "mem_gb", return_value=(96.0, 80.0)),
              patch.object(maya, "find_nvcc", return_value=("/usr/local/cuda-12.8/bin/nvcc", (12, 8))),
              patch.object(maya.shutil, "which", side_effect=lambda c, *r: "/usr/bin/g++" if c == "g++" else None),
              patch.object(maya, "pick_cmake", return_value=None),
              patch.object(maya, "select_build_backend", lambda b: None)]
        for p in ps:
            p.start()
        try:
            with patch("sys.stdout", new=out), patch.object(maya.S, "UI", None):
                got.append(maya.check_pc(args(backend="cuda"), "qfn"))
        finally:
            for p in ps:
                p.stop()
        self.assertIsNotNone(got[0])                                    # (the PC sets up: no refusal)
        self.assertEqual([g["index"] for g in got[0]["gpus"]], [0])     # (the 4090 alone)
        self.assertNotIn("cannot run", out.getvalue())
        self.assertIn("too old", out.getvalue())                        # (the GT 710: listed, never chosen)

    def test_a_tiny_equal_pair_beside_a_comfortable_card_is_not_a_refusal(self):
        """Review finding 1: an equal pair of tiny cards beside a card that runs the model alone - the pre-guard
        priced best_gpus, which ranked by count and took the pair's split, and it did not apply the cc filter
        choose_gpus applies.  Both shapes now set the PC up on the comfortable card alone: the GT 710s (cc 3.5,
        unnameable by --gpus) beside a 16 GB A4000, and a cc-61 GT 1030 pair beside an 11 GB 2080 Ti."""
        for cards, want in (
                ([{"index": 0, "name": "RTX A4000", "vram_gb": 16.0, "arch": "86", "driver": "550"},
                  {"index": 1, "name": "GeForce GT 710", "vram_gb": 1.0, "arch": "35", "driver": "550"},
                  {"index": 2, "name": "GeForce GT 710", "vram_gb": 1.0, "arch": "35", "driver": "550"}], [0]),
                ([{"index": 0, "name": "RTX 2080 Ti", "vram_gb": 11.0, "arch": "75", "driver": "550"},
                  {"index": 1, "name": "GeForce GT 1030", "vram_gb": 4.0, "arch": "61", "driver": "550"},
                  {"index": 2, "name": "GeForce GT 1030", "vram_gb": 4.0, "arch": "61", "driver": "550"}], [0])):
            out, got = io.StringIO(), []
            ps = [patch.object(maya.S, "gpus", return_value=cards),
                  patch.object(maya.S, "is_wsl", return_value=False),
                  patch.object(maya.S, "cpu_info", return_value=("CPU", True, True)),
                  patch.object(maya.S, "out", return_value="g++ (test) 13.0"),
                  patch.object(maya, "oomd_active", return_value=False),
                  patch.object(maya, "mem_gb", return_value=(96.0, 80.0)),
                  patch.object(maya, "find_nvcc", return_value=("/usr/local/cuda-12.8/bin/nvcc", (12, 8))),
                  patch.object(maya.shutil, "which", side_effect=lambda c, *r: "/usr/bin/g++" if c == "g++" else None),
                  patch.object(maya, "pick_cmake", return_value=None),
                  patch.object(maya, "select_build_backend", lambda b: None)]
            for p in ps:
                p.start()
            try:
                with patch("sys.stdout", new=out), patch.object(maya.S, "UI", None):
                    got.append(maya.check_pc(args(backend="cuda"), "qfn"))
            finally:
                for p in ps:
                    p.stop()
            self.assertIsNotNone(got[0], out.getvalue()[-300:])               # (the PC sets up: no refusal)
            self.assertEqual([g["index"] for g in got[0]["gpus"]], want)      # (the comfortable card alone)
            self.assertNotIn("cannot run", out.getvalue())

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

    def run_check_pc(self, nvcc, cards=None, a=None):
        out, got = io.StringIO(), []
        ps = self.patches(nvcc)
        if cards is not None:
            ps[0] = patch.object(maya.S, "gpus", return_value=cards)
        for p in ps:
            p.start()
        try:
            with patch("sys.stdout", new=out), patch.object(maya.S, "UI", None):
                try:
                    got.append(maya.check_pc(a or args(backend="cuda"), "qfn"))
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

    def test_the_headless_p40_writes_1024_as_its_reserve(self):
        """v1.2.1-D: the written config of a headless Pascal card keeps 1024 MiB free (the warm-up graph captures
        fell to 264 MiB with 512), the --vram-reserve-mib still wins above the rule, and the ok line says why."""
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            models, d = t / "models", t / "models" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S"
            files = [t / "gguf" / n for n in maya.qfn_entry("gsq-iq3_s")["files"]]
            gpus = {"backend": "cuda", "gpus": [dict(self.P40, display=False)]}
            for kw, want in (({}, "1024"), ({"vram_reserve_mib": 2048}, "2048")):
                with self.subTest(**kw), tempfile.TemporaryDirectory() as t2:
                    t2 = Path(t2)
                    out = io.StringIO()
                    m2, d2 = t2 / "models", t2 / "models" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S"
                    f2 = [t2 / "gguf" / n for n in maya.qfn_entry("gsq-iq3_s")["files"]]
                    with cpu(), patch.object(maya, "ROOT", t2), patch.object(maya, "mem_gb",
                                                                             return_value=(220.0, 200.0)), \
                            patch.object(maya, "write_run_script", return_value=t2 / "run.sh"), \
                            patch("sys.stdout", new=out), patch.object(maya.S, "UI", None):
                        p = maya.write_config_qfn(args(backend="cuda", **kw), gpus, {}, "gsq-iq3_s", f2,
                                                  d2 / "pack", None, 65536, m2, d2, {})
                    a = json.loads(p.read_text())["args"]
                    self.assertEqual(a[a.index("--vram-reserve-mib") + 1], want)
                    if not kw:
                        self.assertIn("264", out.getvalue())             # (the measured reason, said)

    def test_the_toolkit_rule_prefers_a_fitting_12_x(self):
        self.assertEqual(maya.nvcc_range([61]), ((12, 0), (13, 0)))    # (CUDA 13 dropped Pascal: hi)
        self.assertEqual(maya.pick_nvcc([("/c13", (13, 2)), ("/c128", (12, 8))], [61]), ("/c128", (12, 8)))
        self.assertEqual(maya.pick_nvcc([("/c13", (13, 2))], [61]), ("/c13", (13, 2)))   # (only 13: reported)
        self.assertIn("Pascal need CUDA 12.x", maya.toolkit_hint([61]))

    def test_only_cuda_13_installed_says_pascal_needs_12(self):
        out, pc = self.run_check_pc(("/usr/local/cuda-13.2/bin/nvcc", (13, 2)))
        self.assertIsNone(pc)                                          # (the setup stops there)
        self.assertIn("can be used (Pascal: second class)", out)       # (the card itself: seen, second class ...)
        self.assertNotIn("Pascal: experimental", out)                  # ( ... never labelled experimental)
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
        # v1.2.1-D: the Pascal line states what was measured (and what was not) rather than calling it experimental
        self.assertNotIn("experimental", out)
        self.assertIn("Pascal (compute capability 6.x) is second class", out)
        self.assertIn("Measured on a Tesla P40", out)
        self.assertIn("not measured on a Pascal card yet", out)

    def test_a_mixed_pre_turing_and_newer_list_is_said_plainly(self):
        """Review finding 4: CMakeLists.txt's qfn rule (1787-1808) builds engines/qfn's pre-Turing path only when
        EVERY arch is < 75, so a P40 named beside a T4 builds sm_75 only - the P40 would get a binary with no
        kernel image for it.  check_pc says that plainly (MIN_CC 61 extended this to Pascal; the engine's C++ is
        off-limits, and the PC runs on either class alone, so it is a warning with the way out, not a refusal)."""
        cards = [dict(self.P40), {"index": 1, "name": "Tesla T4", "vram_gb": 15.0, "arch": "75", "driver": "550"}]
        out, pc = self.run_check_pc(("/usr/local/cuda-12.8/bin/nvcc", (12, 8)), cards=cards,
                                    a=args(backend="cuda", gpus="0,1"))
        self.assertIsNotNone(pc)                                       # (warned, not refused: either class alone runs)
        self.assertIn("no kernel image for it", out)                   # (the mixed list, said plainly)
        self.assertIn("--gpu N for one class", out)
        out1, pc1 = self.run_check_pc(("/usr/local/cuda-12.8/bin/nvcc", (12, 8)), cards=cards,
                                      a=args(backend="cuda", gpus="0"))  # (the P40 alone: pre-Turing build, no warn)
        self.assertIsNotNone(pc1)
        self.assertNotIn("no kernel image", out1)


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

    def flow(self, t: Path, a, vision=None, mtp=True, pc=None, mem=(91.7, 80.0), out=None, prev=None):
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
                patch("sys.stdout", new=out or io.StringIO()), patch.object(maya.S, "UI", None):
            return maya.set_up_qfn(a, prev or {})

    def setup(self, t: Path, **kw):
        vision = kw.pop("_vision", None)
        nomtp = kw.pop("_nomtp", False)
        pc = kw.pop("_pc", None)
        mem = kw.pop("_mem", (91.7, 80.0))
        out = kw.pop("_out", None)
        prev = kw.pop("_prev", None)
        kw.setdefault("context", 65536)
        a = args(qfn_quant="ud-q4_k_xl", models_dir=t / "models", **kw)
        p = self.flow(t, a, vision=vision, mtp=not nomtp, pc=pc, mem=mem, out=out, prev=prev)
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
            self.assertEqual(cfg["vision"], {**v, "elastic_mib": 1536 + 1024})   # (the reader and the reserve, at most 1024)
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

    def test_a_16_gb_pc_pins_no_tier_and_streams_through_the_cache(self):
        """Review finding 3 through the setup: a 16 GB PC clears the RAM minimum by 0.07 GiB - the written layout
        scales the pinned tier down to nothing instead of pinning 8 GiB beside the ~8 the minimum leaves."""
        with tempfile.TemporaryDirectory() as t:
            cfg = self.setup(Path(t), _mem=(16.0, 8.0))
        a = cfg["args"]
        self.assertNotIn("--resident-budget-gib", a)               # (no tier: the file cache is it, scaled slow)
        self.assertEqual(a[a.index("--kv-resident") + 1], "4096")  # (still the scaled layout - the context here
                                                                   #  is the 65536 the flow itself names)

    def test_drafting_on_overrides_the_scale_down(self):
        """Review finding 3: with --drafting on the step-2 line says drafting STAYS on - it does not announce a
        drafting off the flag overrides (and without the flag the scale-down line still says drafting off)."""
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as t, cpu():
            cfg = self.setup(Path(t), _mem=(64.0, 50.0), drafting="on", _out=out)   # (--drafting on says otherwise)
        self.assertNotIn("--mtp-max-t", cfg["args"])
        self.assertEqual(cfg["installer"]["mtp_calibration"]["kept"], "mtp")   # (the question's own record)
        said = [ln for ln in out.getvalue().splitlines() if "scales this layout down" in ln]
        self.assertEqual(len(said), 1)                                         # (the scale-down line, once)
        self.assertNotIn("drafting off", said[0])                              # (not announced: it stays on)
        self.assertIn("drafting stays on: --drafting on says otherwise", said[0])
        self.assertIn("--kv-resident 4096 instead of 32768", said[0])          # (the rest of the scale-down: said)
        out2 = io.StringIO()
        with tempfile.TemporaryDirectory() as t, cpu():
            self.setup(Path(t), _mem=(64.0, 50.0), _out=out2)                  # (no --drafting: the default)
        said = [ln for ln in out2.getvalue().splitlines() if "scales this layout down" in ln][0]
        self.assertIn("drafting off", said)                                    # (still scaled down as before; the
        self.assertNotIn("--mtp-max-t", said)                                  #  line claims no flag - finding 5)
        self.assertNotIn("drafting stays on", said)

    def test_the_scale_down_line_says_nothing_of_the_context_when_one_was_given(self):
        """v1.2.1-D: --context named a context, so the written --max-context is that one and the scale-down line
        must not claim "a 32K context instead of 64K" (the P40's Paul runs --context 262144).  The rest of the
        scale-down is still said."""
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as t, cpu():
            cfg = self.setup(Path(t), _mem=(64.0, 50.0), context=262144, _out=out)
        said = [ln for ln in out.getvalue().splitlines() if "scales this layout down" in ln][0]
        self.assertNotIn("context instead of", said)
        self.assertIn("kept, not scaled down", said)
        self.assertIn("--kv-resident 4096 instead of 32768", said)
        self.assertIn("drafting off", said)
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "262144")

    def test_a_slow_pc_with_no_drafter_claims_no_flag(self):
        """Review finding 5: turned_off always said "drafting off (--mtp-max-t 1)", but write_config_qfn writes
        that flag only when a drafter exists (saguaro.py:7576) - a slow PC whose draft layer was not downloaded
        gets the scale-down line without the flag, and a config with neither --mtp nor --mtp-max-t."""
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as t, cpu():
            cfg = self.setup(Path(t), _mem=(64.0, 50.0), _nomtp=True, _out=out)
        self.assertNotIn("--mtp", cfg["args"])                    # (no drafter: nothing loaded, nothing capped)
        self.assertNotIn("--mtp-max-t", cfg["args"])
        said = [ln for ln in out.getvalue().splitlines() if "scales this layout down" in ln][0]
        self.assertNotIn("--mtp-max-t", said)                     # (no flag claimed where there is no drafter)
        self.assertIn("drafting off", said)                       # (still plain and true: nothing drafts here)
        self.assertIn("--kv-resident 4096 instead of 32768", said)


class LongContextC18(SetupFlow):
    """C18 step 1: the context question offers the elastic YaRN sizes past 256K and the config follows:
    --rope-scale = max / 262144 as the engine parses it (a double: 512K's 2, 1M's 4, 384K's 1.5 - a
    fractional factor is written as computed, never rounded), --yarn-orig-ctx 262144 and the elastic
    SAGUARO_YARN_ELASTIC stay, a --context past 1M or no whole 64K step is refused plainly, and a 256K
    config is written exactly as before."""

    def yarn_tail(self, cfg):
        a = cfg["args"]
        return a[a.index("--rope-scaling"):a.index("--rope-scaling") + 6]

    def test_the_512k_config(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = self.setup(Path(t), context=524288)
        a = cfg["args"]
        self.assertEqual(a[a.index("--max-context") + 1], "524288")
        self.assertEqual(self.yarn_tail(cfg), ["--rope-scaling", "yarn", "--rope-scale", "2",
                                               "--yarn-orig-ctx", "262144"])
        self.assertEqual(cfg["env"]["SAGUARO_YARN_ELASTIC"], "1")          # (the elastic pair, as today)
        self.assertEqual(a[a.index("--kv-resident") + 1], "32768")         # (the cards' window: unchanged)

    def test_the_1m_config(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = self.setup(Path(t), context=1048576)
        a = cfg["args"]
        self.assertEqual(a[a.index("--max-context") + 1], "1048576")
        self.assertEqual(self.yarn_tail(cfg), ["--rope-scaling", "yarn", "--rope-scale", "4",
                                               "--yarn-orig-ctx", "262144"])   # (the live 4 x T4 box's line)
        self.assertEqual(cfg["env"]["SAGUARO_YARN_ELASTIC"], "1")

    def test_a_64k_step_writes_its_fraction_as_computed(self):
        """The engine's --rope-scale is a double (generate.cpp: atof into a double, its check is only ">= 1"),
        so 384K gets 1.5 - not rounded up to 2, and not refused."""
        with tempfile.TemporaryDirectory() as t:
            cfg = self.setup(Path(t), context=393216)
        a = cfg["args"]
        self.assertEqual(a[a.index("--max-context") + 1], "393216")
        self.assertEqual(a[a.index("--rope-scale") + 1], "1.5")

    def test_256k_writes_what_it_always_wrote(self):
        """Byte-identity for the recommended default: the 262144 config keeps the args it had before the
        YaRN sizes joined - --rope-scale 4 among them (the elastic engine never reads the table inside 256K,
        so the factor there was and stays a formality; the parity classes compare the whole arg lists)."""
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            big = self.setup(Path(t1), context=262144)
            small = self.setup(Path(t2), context=65536)
        self.assertEqual(self.yarn_tail(big), ["--rope-scaling", "yarn", "--rope-scale", "4",
                                               "--yarn-orig-ctx", "262144"])
        norm = lambda cfg, tmp: [x for x in zip(cfg["args"][::2], cfg["args"][1::2])   # noqa: E731
                                 if "--max-context" not in x and str(tmp) not in str(x)]
        self.assertEqual(norm(big, t1), norm(small, t2))   # (the args, bar --max-context and the paths: the
                                                           #  file was and is that one flag's difference)
        self.assertEqual(big["args"][big["args"].index("--max-context") + 1], "262144")

    def test_past_1m_is_refused(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as t, self.assertRaises(SystemExit):
            self.setup(Path(t), context=2097152, _out=out)
        said = out.getvalue()
        self.assertIn("whole 64K steps", said)
        self.assertIn("1M", said)
        self.assertNotIn("context: 2097152 tokens", said)    # (review nit: refused before it is said)

    def test_a_part_step_past_256k_is_refused(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as t, self.assertRaises(SystemExit):
            self.setup(Path(t), context=300000, _out=out)
        said = out.getvalue()
        self.assertIn("whole 64K steps", said)
        self.assertIn("524288", said)
        self.assertNotIn("context: 300000 tokens", said)     # (review nit: refused before it is said)


class AnswersAtOnceC18(SetupFlow):
    """C18 step 2: the setup asks "answers at once" right after the context - the layout's default pre-set
    (1 on one card, 2 on a split, a config written before keeps its value on a split), 1 to 8, and a given
    --parallel N is not asked - and states one combined cost line (context x answers: 1M x 2 = about 27 GB
    of pinned RAM) with the RAM and memlock cross-checks: warn on a tight box, refuse only what cannot fit."""

    def asked(self, t: Path, **kw):
        seen = []

        def answer(q, ch, dflt, yes):
            seen.append((q, tuple(ch), dflt))
            return dflt
        with cpu(), patch.object(maya, "ask", side_effect=answer):
            cfg = self.setup(t, **kw)
        return cfg, [s for s in seen if "Answers at once" in s[0]]

    def test_the_question_after_the_context(self):
        with tempfile.TemporaryDirectory() as t:
            cfg, asked = self.asked(Path(t), context=65536, yes=False)
        self.assertEqual(len(asked), 1)
        self.assertEqual(asked[0][1], tuple(str(i) for i in range(1, 9)))    # (1 to 8)
        self.assertEqual(asked[0][2], "1")                                   # (the one card: 1, as today)
        self.assertIsNone(cfg.get("parallel"))

    def test_the_split_default_is_two_and_a_config_before_keeps_its_value(self):
        two = {"backend": "cuda", "gpus": [{"index": 0, "name": "Tesla T4", "vram_gb": 15.0, "arch": "75",
                                            "display": False},
                                           {"index": 1, "name": "Tesla T4", "vram_gb": 15.0, "arch": "75",
                                            "display": False}]}
        with tempfile.TemporaryDirectory() as t:
            cfg, asked = self.asked(Path(t), context=65536, yes=False, _pc=two)
        self.assertEqual(asked[0][2], "2")                                   # (a split: 2, as the live boxes)
        self.assertEqual(cfg["parallel"], 2)
        with tempfile.TemporaryDirectory() as t:                             # (a setup again: the value a
            cfg, asked = self.asked(Path(t), context=65536, yes=False, _pc=two,   #  config wrote before is
                                    _prev={"parallel": 5})                   #   the default on a split)
        self.assertEqual(asked[0][2], "5")
        self.assertEqual(cfg["parallel"], 5)

    def test_parallel_given_is_not_asked(self):
        with tempfile.TemporaryDirectory() as t:
            cfg, asked = self.asked(Path(t), context=65536, parallel=4, yes=False)
        self.assertEqual(asked, [])                                          # (--parallel N: not asked)
        self.assertEqual(cfg["parallel"], 4)

    def test_above_one_on_the_one_amd_card_hears_what_it_measured(self):
        """Review finding 4: on the one AMD card parallel decoding was measured slower - an answer of 2 reaches
        the config now that the question asks (until the question, only a deliberate --parallel did), so the
        answer hears the measurement.  It keeps its 2: that is the one card's picture-encoder lend path."""
        said = []
        with tempfile.TemporaryDirectory() as t, \
                patch.object(maya, "ask", side_effect=lambda q, c, d, y: "2" if "Answers at once" in q else d), \
                patch.object(maya, "warn", side_effect=lambda *x: said.append(" ".join(map(str, x)))):
            cfg = self.setup(Path(t), context=65536, yes=False)               # (the flow's card is the AMD one)
        self.assertEqual(cfg["parallel"], 2)                                  # (the answer stands)
        self.assertEqual(len([s for s in said if "lost speed" in s]), 1, said)

    def test_parallel_given_on_that_card_is_deliberate_and_untold(self):
        """"--parallel N says" holds: the AMD card given 2 on the command line is asked nothing and lectured
        of nothing - the warning belongs to the answer, not to the flag."""
        def answer(q, ch, dflt, yes):
            self.assertNotIn("Answers at once", q)
            return dflt
        said = []
        with tempfile.TemporaryDirectory() as t, patch.object(maya, "ask", side_effect=answer), \
                patch.object(maya, "warn", side_effect=lambda *x: said.append(" ".join(map(str, x)))):
            cfg = self.setup(Path(t), context=65536, parallel=2, yes=False)
        self.assertEqual(cfg["parallel"], 2)
        self.assertEqual([s for s in said if "lost speed" in s], [])

    def test_the_combined_cost_line(self):
        """The one line beside the questions: 12.7 KB a token a conversation, context x answers - Ken's
        1M x 2 = about 27 GB, and the newest 32K of each conversation on the cards too."""
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as t:
            self.setup(Path(t), context=1048576, parallel=2, _out=out)
        line = [ln for ln in out.getvalue().splitlines() if "the cost:" in ln]
        self.assertEqual(len(line), 1)
        self.assertIn("12.7 KB of K/V a context token a conversation in INT8", line[0])
        self.assertIn("1M x 2 = about 27 GB", line[0])
        self.assertIn("32K of each conversation on the cards too", line[0])

    def test_a_ram_poor_box_gets_the_warning_not_a_refusal(self):
        """64 GB with 1M x 2: about 27 GB of K/V beside the 44 GiB expert tier - warn, and the config is
        still written (the plan: warn+confirm, not refuse - v1.2.1-B's own rule for the layout)."""
        said = []
        with tempfile.TemporaryDirectory() as t, cpu(), \
                patch.object(maya, "warn", side_effect=lambda *x: said.append(" ".join(map(str, x)))):
            cfg = self.setup(Path(t), context=1048576, parallel=2, _mem=(64.0, 50.0))
        warns = [s for s in said if "GiB of RAM: the K/V alone pins about 27 GB" in s]   # (GiB: mem_gb's unit,
                                                                                         #  now labelled as such)
        self.assertEqual(len(warns), 1, said)
        self.assertIn("44 GiB expert tier", warns[0])
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "1048576")

    def test_it_refuses_only_what_cannot_fit(self):
        """40 GB of RAM and 1M x 4 = about 53 GB of K/V, all of it pinned: the RAM cannot hold it - refuse,
        and say what would fit."""
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as t, cpu(), self.assertRaises(SystemExit):
            self.setup(Path(t), context=1048576, parallel=4, _mem=(40.0, 30.0), _out=out)
        said = out.getvalue()
        self.assertIn("does not fit", said)
        self.assertIn("fewer answers at once", said)
        self.assertIn("768K", said)     # (40 GiB / (12672 B x 4): 768K fits per conversation, 832K would not)

    def test_memlock_between_the_tier_and_the_k_v_warns_once(self):
        """The 96 GB PC's 71 GiB tier under an 80 GiB memlock: the tier's own warning says nothing, the cost
        check's does - once - and names the tier plus the K/V (71 + ~12 = 83)."""
        said = []
        with tempfile.TemporaryDirectory() as t, cpu(), \
                patch.object(maya, "memlock_soft_gib", return_value=80.0), \
                patch.object(maya, "warn", side_effect=lambda *x: said.append(" ".join(map(str, x)))):
            self.setup(Path(t), context=1048576)
        warns = [s for s in said if "memory-lock" in s]
        self.assertEqual(len(warns), 1, said)
        self.assertIn("83 GB pinned RAM tier", warns[0])

    def test_a_memlock_under_the_tier_is_still_one_warning(self):
        said = []
        with tempfile.TemporaryDirectory() as t, cpu(), \
                patch.object(maya, "memlock_soft_gib", return_value=40.0), \
                patch.object(maya, "warn", side_effect=lambda *x: said.append(" ".join(map(str, x)))):
            self.setup(Path(t), context=1048576)
        self.assertEqual(len([s for s in said if "memory-lock" in s]), 1)     # (the tier's own: not a second)


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
        self.assertIn("measured on about 96 GB", small["qwen3.8-flash-next-ud-q4_k_xl"]["fit"])   # (v1.2.1-B
        self.assertIn("runs it too, slowly", small["qwen3.8-flash-next-ud-q4_k_xl"]["fit"])       #  nit: the
        self.assertIn("measured on about 96 GB", small["cyber-frost"]["fit"])                     #  measured box,
                                                                                                  #  not a need)


if __name__ == "__main__":
    unittest.main()
