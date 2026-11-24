"""tools/test_qfn_calibrate.py - QFN's --calibrate / --speed-calibration / --bench in unit tests with fakes (no GPU,
no engine, no network): the drafter-path lookup on the config's own --mtp folder, the drafting-on/off calibration
decision on fake timings, the QFN routing of calibrate_config / speed_calibration (the old CYBER-FROST config
routing there too since T8a),
the expert-profile copy the calibration measures on, and --bench keeping saguaro-qfn and starting the engine the
way its server does.

    python -m unittest tools.test_qfn_calibrate -v
"""
from pathlib import Path
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import saguaro as maya  # noqa: E402
import qfn_calibrate as CQ  # noqa: E402
import maya_bench as MB  # noqa: E402


def cfg_on(tmp):
    """A QFN config like the live ones (./ref/*.json): its own --mtp folder, expert profile, layer split."""
    mtp = Path(tmp) / "Qwen3.8-Flash-Next-MTP-Lamina"
    mtp.mkdir()
    (mtp / "dense.txt").write_text("lamina index")
    return {"exe": "/opt/llm/saguaro/bin/saguaro-qfn", "backend": "hip", "gpu": [0],
            "args": ["--pack", str(Path(tmp) / "pack"), "--mtp", str(mtp), "--spec", "3", "--spec-min-p", "0.7",
                     "--expert-profile", str(Path(tmp) / "expert-profile.bin"),
                     "--expert-profile-save", str(Path(tmp) / "expert-profile.bin"), "--max-context", "65536"],
            "tokenizer": str(Path(tmp) / "pack" / "tokenizer"), "model_name": "Qwen3.8-Flash-Next-UD-Q4_K_XL",
            "installer": {"family": "qfn"}, "parallel": 2}


def write_cfg(tmp, cfg):
    p = Path(tmp) / "maya-qwen3.8-flash-next-ud-q4_k_xl-hip.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


class DrafterPath(unittest.TestCase):
    """The drafter is the folder the config's own --mtp names - not CYBER-FROST's <pack>/mtp."""

    def test_the_configs_own_mtp_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = cfg_on(tmp)
            self.assertEqual(CQ.mtp_dir(cfg), Path(tmp) / "Qwen3.8-Flash-Next-MTP-Lamina")
            self.assertTrue(CQ.has_drafter(cfg))

    def test_no_mtp_arg_no_drafter(self):
        cfg = {"args": ["--pack", "/m/pack"]}
        self.assertIsNone(CQ.mtp_dir(cfg))
        self.assertFalse(CQ.has_drafter(cfg))

    def test_a_folder_without_its_layer_is_no_drafter(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = cfg_on(tmp)
            (Path(tmp) / "Qwen3.8-Flash-Next-MTP-Lamina" / "dense.txt").unlink()
            self.assertEqual(CQ.mtp_dir(cfg), Path(tmp) / "Qwen3.8-Flash-Next-MTP-Lamina")   # the path is still there
            self.assertFalse(CQ.has_drafter(cfg))                                             # the layer is not

    def test_with_drafting_on_and_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = cfg_on(tmp)
            off = CQ.with_drafting(cfg, False)
            a = off["args"]
            self.assertEqual(a[a.index("--mtp-max-t") + 1], "1")          # off: --mtp-max-t 1
            self.assertEqual(a[a.index("--mtp") + 1], cfg["args"][cfg["args"].index("--mtp") + 1])   # --mtp stays
            on = CQ.with_drafting(CQ.with_drafting(cfg, False), True)
            self.assertNotIn("--mtp-max-t", on["args"])                   # on: the config's --mtp args as they stand
            self.assertEqual(on["args"], cfg["args"])                     # nothing else moved

    def test_the_profile_is_measured_on_a_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = cfg_on(tmp)
            (tmp / "expert-profile.bin").write_bytes(b"STRP")
            with tempfile.TemporaryDirectory() as t2:
                c = CQ.profile_copy(cfg, Path(t2))
                a = c["args"]
                self.assertEqual(a[a.index("--expert-profile") + 1], str(Path(t2) / "expert-profile.bin"))
                self.assertEqual(a[a.index("--expert-profile-save") + 1], str(Path(t2) / "expert-profile.bin"))
                self.assertEqual((Path(t2) / "expert-profile.bin").read_bytes(), b"STRP")   # (the PC's own copied in)
                self.assertEqual((tmp / "expert-profile.bin").read_bytes(), b"STRP")        # (the original untouched)


class CalibrationDecision(unittest.TestCase):
    """Fake timings through CQ.run: the faster side wins, drafts only by more than MIN_GAIN."""

    def fake(self, off_rate, on_rate):
        import calibrate_draft as CD

        def measure(c, label, warm, questions, n_doc, start_engine, say=print, tag=""):
            started.append(tuple(c["args"]))
            rate = on_rate if "--mtp-max-t" not in c["args"] else off_rate
            return {"tok_s": rate, "tokens": 2000, "answers": 4, "seconds": 160.0, "load_s": 60,
                    "drafts_accepted": 30, "drafts_offered": 40}
        started = []
        with patch.object(CD, "requests", lambda cfg: ([1], [[2]], 10)), patch.object(CD, "measure", measure):
            res = CQ.run({"args": ["--mtp", "/m/Lamina"]}, say=lambda *a: None, start_engine=lambda c: None)
        return res, started

    def test_drafts_kept_when_clearly_faster(self):
        res, started = self.fake(10.0, 13.0)
        self.assertEqual(res["keep"], "on")
        self.assertAlmostEqual(res["gain"], 0.3)
        self.assertEqual(len(started), 2)                                  # (both sides started, one engine each)

    def test_tie_keeps_decoding_without_drafts(self):
        res, _ = self.fake(10.0, 10.1)
        self.assertEqual(res["keep"], "off")

    def test_the_off_side_runs_with_mtp_max_t_1(self):
        res, started = self.fake(10.0, 10.1)
        self.assertIn("--mtp-max-t", started[0])                           # (no-drafting side first)
        self.assertNotIn("--mtp-max-t", started[1])

    def test_decide_boundaries(self):
        self.assertEqual(CQ.decide(10.0, 10.0), "off")
        self.assertEqual(CQ.decide(10.0, 10.3), "off")                     # (exactly MIN_GAIN is not "more than")
        self.assertEqual(CQ.decide(10.0, 10.31), "on")
        self.assertEqual(CQ.decide(0.0, 5.0), "off")                       # (no base speed: nothing to keep)

    def test_sides_that_wrote_too_little_raise(self):
        import calibrate_draft as CD
        with patch.object(CD, "requests", lambda cfg: ([1], [[2]], 10)), \
                patch.object(CD, "measure", lambda *a, **kw: {"tok_s": 9.0, "tokens": 10}):
            with self.assertRaises(RuntimeError):
                CQ.run({"args": ["--mtp", "/m/Lamina"]}, say=lambda *a: None, start_engine=lambda c: None)


class SaguaroRouting(unittest.TestCase):
    """--calibrate / --speed-calibration take the qfn branch to QFN's own check - the old CYBER-FROST config too,
    which reads "qfn" since T8a (its model is the family's quant)."""

    def test_calibrate_config_routes_qfn_to_its_own_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_cfg(tmp, cfg_on(tmp))
            with patch.object(maya, "qfn_calibrate_config", return_value=True) as qfn:
                self.assertTrue(maya.calibrate_config(p))
                self.assertTrue(qfn.called)

    def test_calibrate_config_routes_old_cyber_frost_to_qfn(self):
        cfg = {"exe": "/b/maya", "args": ["--glm-pack", "/p", "--ple-gguf", "/g"],
               "installer": {"family": "qwen", "qwen_quant": "q5_k_m"}}
        with tempfile.TemporaryDirectory() as tmp:
            p = write_cfg(tmp, cfg)
            with patch.object(maya, "qfn_calibrate_config", return_value=True) as qfn:
                self.assertTrue(maya.calibrate_config(p))
                self.assertTrue(qfn.called)

    def test_speed_calibration_routes_qfn_to_its_own_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_cfg(tmp, cfg_on(tmp))
            with patch.object(maya, "qfn_draft_calibration", return_value=True) as qfn, \
                    patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
                self.assertTrue(maya.speed_calibration(p, "asked for"))
                self.assertTrue(qfn.called)

    def test_draft_calibration_keeps_the_faster_side_in_the_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_cfg(tmp, cfg_on(tmp))
            res = {"off": {"tok_s": 10.0}, "on": {"tok_s": 14.0}, "keep": "on", "gain": 0.4}
            out = io.StringIO()
            with patch.object(maya, "busy_gpus", return_value=[]), patch.object(maya.CQ, "run", return_value=res), \
                    patch("sys.stdout", new=out), patch.object(maya.S, "UI", None):
                self.assertTrue(maya.qfn_draft_calibration(p))
            new = json.loads(p.read_text())
            self.assertNotIn("--mtp-max-t", new["args"])                   # (drafts kept: the config's own --mtp args)
            self.assertEqual(new["installer"]["mtp_calibration"]["kept"], "mtp")
            self.assertIn("Lamina drafter is faster", out.getvalue())      # (it says what it chose)

    def test_draft_calibration_turns_drafts_off_with_mtp_max_t_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_cfg(tmp, cfg_on(tmp))
            res = {"off": {"tok_s": 12.0}, "on": {"tok_s": 10.0}, "keep": "off", "gain": -0.167}
            out = io.StringIO()
            with patch.object(maya, "busy_gpus", return_value=[]), patch.object(maya.CQ, "run", return_value=res), \
                    patch("sys.stdout", new=out), patch.object(maya.S, "UI", None):
                self.assertFalse(maya.qfn_draft_calibration(p))
            new = json.loads(p.read_text())
            a = new["args"]
            self.assertEqual(a[a.index("--mtp-max-t") + 1], "1")           # (off: --mtp-max-t 1, --mtp stays loaded)
            self.assertEqual(new["installer"]["mtp_calibration"]["kept"], "none")
            self.assertIn("not faster", out.getvalue())

    def test_draft_calibration_without_a_drafter_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = cfg_on(tmp)
            (Path(tmp) / "Qwen3.8-Flash-Next-MTP-Lamina" / "dense.txt").unlink()
            p = write_cfg(tmp, cfg)
            with patch.object(maya, "busy_gpus", return_value=[]), patch.object(maya.CQ, "run") as run, \
                    patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
                self.assertIsNone(maya.qfn_draft_calibration(p))
            self.assertFalse(run.called)
            self.assertEqual(json.loads(p.read_text())["args"], cfg["args"])

    def test_draft_calibration_survives_a_failed_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = cfg_on(tmp)
            p = write_cfg(tmp, cfg)
            with patch.object(maya, "busy_gpus", return_value=[]), \
                    patch.object(maya.CQ, "run", side_effect=RuntimeError("the engine stopped")), \
                    patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
                self.assertIsNone(maya.qfn_draft_calibration(p))
            self.assertEqual(json.loads(p.read_text())["args"], cfg["args"])   # (nothing changed)

    def test_hardware_calibration_writes_the_won_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_cfg(tmp, cfg_on(tmp))
            fake = {"settings": {"--pcie-frac": "0.35"}, "report": {"tok_s": 42.0}}
            with patch.object(maya, "busy_gpus", return_value=[]), patch.object(maya.CALH, "run", return_value=fake), \
                    patch("sys.stdout", new=io.StringIO()), patch.object(maya.S, "UI", None):
                self.assertTrue(maya.qfn_hardware_calibration(p))
            new = json.loads(p.read_text())
            a = new["args"]
            self.assertEqual(a[a.index("--pcie-frac") + 1], "0.35")
            self.assertEqual(a[a.index("--spec-min-p") + 1], "0.5")        # (not won: back to the product default)
            self.assertEqual(new["installer"]["calibration"]["settings"], {"--pcie-frac": "0.35"})

    def test_hardware_calibration_with_the_gpus_busy_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_cfg(tmp, cfg_on(tmp))
            out = io.StringIO()
            with patch.object(maya, "busy_gpus", return_value=[(0, 40000)]), patch.object(maya.CALH, "run") as run, \
                    patch("sys.stdout", new=out), patch.object(maya.S, "UI", None):
                self.assertFalse(maya.qfn_hardware_calibration(p))
        self.assertFalse(run.called)
        self.assertIn("GPU 0 has 40000 MiB in use", out.getvalue())


class BenchQfn(unittest.TestCase):
    """--bench for QFN: the engine binary stays saguaro-qfn, the engine starts as its server starts it."""

    def test_bench_exe_keeps_saguaro_qfn(self):
        with patch.object(maya, "BUILD", Path("/build-hip")):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = cfg_on(tmp)                                          # (backend hip, family qfn)
                self.assertEqual(maya.bench_exe(cfg), Path("/build-hip/qfn/saguaro-qfn"))

    def test_bench_exe_does_not_drift_into_the_glm_binary(self):
        # the old bug: family "qfn" fell through family_target to "maya" and --bench swapped in saguaro-glm.
        with patch.object(maya, "BUILD", Path("/build-hip")), patch.object(maya, "EXE", Path("/build-hip/maya")):
            with tempfile.TemporaryDirectory() as tmp:
                self.assertNotEqual(maya.bench_exe(cfg_on(tmp)), Path("/build-hip/maya"))

    def test_bench_start_command_uses_the_batch_driver(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = cfg_on(tmp)
            cmd, env = maya.bench_start_command(cfg)
            self.assertEqual(cmd[:2], ["/opt/llm/saguaro/bin/saguaro-qfn", "--serve"])
            a = cmd[2:]
            self.assertEqual(a[a.index("--batch") + 1], "2")               # ("parallel": 2 -> the batch driver's --batch)
            self.assertEqual(a[a.index("--mtp") + 1], cfg["args"][cfg["args"].index("--mtp") + 1])
            self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0")              # (the driver's own child_env, HIP numbering)

    def test_bench_start_command_other_families_stay_with_the_line_driver(self):
        glm = {"exe": "/b/maya", "backend": "hip", "args": ["--glm-pack", "/p/pack"]}
        cmd, env = maya.bench_start_command(glm)
        self.assertEqual(cmd[:2], ["/b/maya", "--serve"])
        self.assertEqual(cmd[2:], glm["args"])                             # (serve.server.engine_args, one GPU: as is)

    def test_speed_lines_pick_qfn_stat_lines(self):
        lines = ["STAT tok_s=21.40 ms_tok=46.73 vram_hit=0.9512 ram_fetch=0.487 disk=0.000 promo=0.031 active=1",
                 "loading the model ...", "saguaro serve: prompt chunk 1536 -> 1024 tokens so its buffers fit"]
        picked = maya.speed_lines(lines)
        self.assertIn(lines[0], picked)
        self.assertIn(lines[2], picked)
        self.assertEqual([x for x in picked if x.startswith("loading")], [])

    def test_drafting_choice_reads_qfn_info_and_args(self):
        info = "INFO logit_bias=1 context=1048576 kv=int8 mtp_max=3 mtp=1 lookup=1 pool_workers=19 pcie_frac=0.55"
        self.assertEqual(MB.drafting_choice([], {}, [info], [], "qfn"), "MTP (Lamina)")
        self.assertEqual(MB.drafting_choice([], {}, [info.replace("mtp_max=3", "mtp_max=1")], [], "qfn"),
                         "MTP drafts off (--mtp-max-t 1)")
        self.assertEqual(MB.drafting_choice([], {}, [info.replace("mtp=1", "mtp=0")], [], "qfn"), "no drafting")
        self.assertEqual(MB.drafting_choice(["--mtp", "/m/Lamina", "--mtp-max-t", "1"], {}, [], [], "qfn"),
                         "MTP drafts off (--mtp-max-t 1)")
        self.assertEqual(MB.drafting_choice(["--mtp", "/m/Lamina"], {}, [], [], "qfn"), "MTP (Lamina)")


if __name__ == "__main__":
    unittest.main()
