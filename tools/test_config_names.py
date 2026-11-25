"""tools/test_config_names.py - C122's config name family: a new setup writes saguaro-<model>.json, an old install
is found, shown, switched, deleted and updated under its maya- name, a model under both names is the new one, the
sidecars beside a config move with its rename and stay out of its list, a --config path given as the old name
after the rename still resolves, a rename that cannot happen (a read-only folder) leaves everything working, and
the server's notes are read under either name, written under the new one and never survive the server.

    python -m unittest tools.test_config_names -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import config_names as CN  # noqa: E402
import saguaro as maya  # noqa: E402
from serve import model_api, models  # noqa: E402


def cfg_file(d: Path, name: str, **extra) -> Path:
    """an installed model's config: a pack beside it so it reads as a real one, its log named the way a setup names it"""
    pack = d / "GLM-5.3-Flash" / "Maya-M" / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (pack.parent / "shard.gguf").write_bytes(b"x" * 1000)
    stem = name.removesuffix(".json")
    cfg = {"exe": "/b/saguaro-glm", "args": ["--glm-pack", str(pack), "--max-context", "131072"],
           "model_name": "glm-5.3-flash", "gpu": 0, "port": 8080, "log": str(d / (stem + ".log")),
           "installer": {"models_dir": str(d), "quant": "Maya-M", "draft": "mtp", "written": "2026-11-01 09:00"},
           **extra}
    p = d / name
    p.write_text(json.dumps(cfg, indent=1))
    return p


class Names(unittest.TestCase):
    def test_bare(self):
        self.assertEqual(CN.bare("maya-glm-maya-s"), "glm-maya-s")
        self.assertEqual(CN.bare("saguaro-glm-maya-s"), "glm-maya-s")
        self.assertEqual(CN.bare("glm-maya-s"), "glm-maya-s")

    def test_is_config_name(self):
        for name in ("maya-maya-m.json", "saguaro-mm3.json", "saguaro-x.turbo.json.json"):
            self.assertTrue(CN.is_config_name(name), name)
        for name in ("maya-x.shared-settings.json", "saguaro-x.turbo.json", "run-saguaro-x.sh", ".saguaro-switch.json",
                     "not-a-config.json", "maya-x.json.prev", ""):
            self.assertFalse(CN.is_config_name(name), name)

    def test_matches(self):
        for want in ("saguaro-glm-maya-s", "glm-maya-s", "maya-glm-maya-s"):   # the old name of the moved file works
            self.assertTrue(CN.matches("saguaro-glm-maya-s", want), want)
        for want in ("saguaro-glm-maya-s", "glm-maya-s"):
            self.assertTrue(CN.matches("maya-glm-maya-s", want), want)         # the new name finds an unmoved install
        for want in ("maya-", "saguaro-", "", "glm-maya-xs", "other"):
            self.assertFalse(CN.matches("saguaro-glm-maya-s", want), want)


class Listing(unittest.TestCase):
    """both families listed, a model under both names is the new one, the sidecars are not models, and a switch
    (serve/models.py: its mtime touched) moves it to the front - under either name"""

    def test_both_families_and_shadowing(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            old = cfg_file(d, "maya-maya-m.json")
            new = cfg_file(d, "saguaro-ds4.json")
            cfg_file(d, "maya-mm3.json")
            cfg_file(d, "maya-shadow.json")
            shadow = cfg_file(d, "saguaro-shadow.json")
            (d / "saguaro-shadow.shared-settings.json").write_text("{}")
            (d / "maya-maya-m.turbo.json").write_text("{}")
            (d / "saguaro-ds4.json.prev").write_text("{}")
            (d / "run-saguaro-ds4.sh").write_text("#!/bin/sh\n")
            found = {p.name for p in models.config_paths(d)}
            self.assertEqual(found, {"maya-maya-m.json", "saguaro-ds4.json", "maya-mm3.json", "saguaro-shadow.json"})
            with mock.patch.object(maya, "ROOT", d):
                self.assertEqual({p.name for p in maya.configs()}, found)      # saguaro.py's own list: same rules
            self.assertEqual(CN.locate(d, "maya-shadow"), shadow)              # both names: the new one, ...
            self.assertEqual(CN.locate(d, "maya-mm3"), d / "maya-mm3.json")    # ... and only the old: the old one
            self.assertEqual(CN.locate(d, "never-set-up"), d / "saguaro-never-set-up.json")

    def test_switched_order_and_shown_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            a = cfg_file(d, "maya-maya-m.json")
            b = cfg_file(d, "saguaro-ds4.json")
            self.assertEqual(models.config_paths(d)[0], b)                     # b was written last
            now = time.time() + 1                                              # (serve/models.py's switch: its mtime touched)
            os.utime(a, (now, now))
            self.assertEqual(models.config_paths(d)[0], a)
            self.assertEqual(models.quant_label({}, Path("maya-maya-m.json")), "maya-m")
            self.assertEqual(models.quant_label({}, Path("saguaro-maya-m.json")), "maya-m")
            self.assertEqual(models.quant_label({}, Path("maya-maya-m-hip.json")), "maya-m")

    def test_sidecars_are_not_models_for_saguaro_py(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            cfg_file(d, "maya-ds4.json")
            (d / "maya-ds4.turbo.json").write_text("{}")                       # (the live Turbo sidecar: not a model,
            (d / "maya-ds4.shared-settings.json").write_text("{}")             #  as serve/models.py always said)
            with mock.patch.object(maya, "ROOT", d):
                self.assertEqual([p.name for p in maya.configs()], ["maya-ds4.json"])


class Api(unittest.TestCase):
    """the model API's names across both families and the rename"""

    def test_find_and_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            old = cfg_file(d, "maya-maya-m.json")
            new = cfg_file(d, "saguaro-ds4.json")
            cfg_file(d, "maya-shadow.json")
            shadow = cfg_file(d, "saguaro-shadow.json")
            self.assertEqual(model_api.find_config(d, "maya-maya-m"), old)
            self.assertEqual(model_api.find_config(d, "saguaro-maya-m"), old)      # the name of after the move
            self.assertEqual(model_api.find_config(d, "maya-m"), old)               # and the display name
            self.assertEqual(model_api.find_config(d, "ds4:latest"), new)
            self.assertEqual(model_api.find_config(d, "maya-shadow"), shadow)      # a moved file answers the old API
            self.assertIsNone(model_api.find_config(d, "maya-"))                   # ... never the family name alone
            entries = [{"file": "saguaro-ds4.json", "model": "deepseek-v4.1-flash"},
                       {"file": "saguaro-shadow.json", "model": "x"}]
            self.assertEqual(model_api.match_name("maya-shadow", entries), entries[1])
            self.assertEqual(model_api.match_name("maya-ds4.json", entries), entries[0])
            self.assertIsNone(model_api.match_name("saguaro-", entries))

    def test_resolve_after_the_rename(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            gone = d / "maya-maya-m.json"
            now = cfg_file(d, "saguaro-maya-m.json")
            self.assertEqual(CN.resolve(gone), now)                            # (--config / llama-swap / systemd)
            self.assertEqual(CN.resolve(now), now)
            self.assertEqual(CN.resolve(d / "maya-none.json"), d / "maya-none.json")
            self.assertEqual(CN.resolve(gone / "sub" / "maya-x.json"), gone / "sub" / "maya-x.json")


class Migration(unittest.TestCase):
    """saguaro.py's migrate_config_names (a start and an update run it): once, atomically, sidecars with it, only
    where the new name is free; a shadowed pair goes aside as <old name>.shadowed and a failed move leaves
    everything working"""

    def test_moved_with_its_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            p = cfg_file(d, "maya-maya-m.json")
            (d / "maya-maya-m.log").write_text("a log line\n")
            (d / "maya-maya-m.shared-settings.json").write_text('{"chat": 1}')
            (d / "maya-maya-m.turbo.json").write_text('{"turbo": 1}')
            (d / "maya-maya-m.json.prev").write_text("{}")
            (d / "maya-maya-m.json.bak-20261101-pre-env-rename").write_text("{}")
            (d / "run-maya-maya-m.sh").write_text("exec saguaro.py --config " + str(p) + " --port 8080\n")
            (d / "run-maya-maya-m.bat").write_text("saguaro.py --config " + str(p) + "\n")
            with mock.patch.object(maya, "ROOT", d), mock.patch.object(maya, "warn") as warn:
                self.assertEqual(maya.migrate_config_names(), ["saguaro-maya-m.json"])
                self.assertEqual(maya.migrate_config_names(), [])              # once: nothing to do again
                warn.assert_not_called()
            self.assertEqual(sorted(f.name for f in d.iterdir() if f.is_file()),
                             ["run-saguaro-maya-m.bat", "run-saguaro-maya-m.sh", "saguaro-maya-m.json",
                              "saguaro-maya-m.json.bak-20261101-pre-env-rename", "saguaro-maya-m.json.prev",
                              "saguaro-maya-m.log", "saguaro-maya-m.shared-settings.json", "saguaro-maya-m.turbo.json"])
            self.assertTrue((d / "saguaro-maya-m.log").read_text() == "a log line\n")   # its contents came along
            self.assertTrue((d / "GLM-5.3-Flash" / "Maya-M" / "shard.gguf").is_file())  # the model files are left
            moved = d / "saguaro-maya-m.json"
            self.assertEqual(json.loads(moved.read_text())["log"], str(d / "saguaro-maya-m.log"))   # it follows
            self.assertIn("saguaro-maya-m.json", (d / "run-saguaro-maya-m.sh").read_text())          # the script too
            self.assertIn("--config " + str(moved), (d / "run-saguaro-maya-m.sh").read_text())
            self.assertIn("saguaro-maya-m.json", (d / "run-saguaro-maya-m.bat").read_text())
            self.assertEqual(CN.resolve(p), moved)                             # the old path still starts it
            with mock.patch.object(maya, "ROOT", d):
                self.assertEqual([f.name for f in maya.configs()], ["saguaro-maya-m.json"])

    def test_a_crash_between_the_moves_is_finished_at_the_next_start(self):
        """the process died right after the config moved, before its sidecars followed: the leftovers under the
        old stem are no orphans - the next start finds them beside the moved config and finishes the set"""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            p = cfg_file(d, "maya-maya-m.json")
            (d / "maya-maya-m.log").write_text("a log line\n")
            (d / "maya-maya-m.turbo.json").write_text('{"turbo": 1}')
            (d / "run-maya-maya-m.sh").write_text("exec saguaro.py --config " + str(p) + " --port 8080\n")
            with mock.patch.object(maya, "ROOT", d), mock.patch.object(maya, "warn"), \
                    mock.patch.object(maya, "_finish_config_rename"):          # (dies right after the config)
                self.assertEqual(maya.migrate_config_names(), ["saguaro-maya-m.json"])
            self.assertTrue((d / "maya-maya-m.turbo.json").is_file())          # ... the sidecars stayed behind
            with mock.patch.object(maya, "ROOT", d), mock.patch.object(maya, "warn") as warn:
                self.assertEqual(maya.migrate_config_names(), [])              # the next start: finished, said of
                warn.assert_not_called()                                       #  nothing
                self.assertEqual(maya.migrate_config_names(), [])              # idempotent: nothing left to do
            self.assertEqual(sorted(f.name for f in d.iterdir() if f.is_file()),
                             ["run-saguaro-maya-m.sh", "saguaro-maya-m.json", "saguaro-maya-m.log",
                              "saguaro-maya-m.turbo.json"])
            self.assertTrue((d / "saguaro-maya-m.log").read_text() == "a log line\n")   # contents came along
            self.assertEqual(json.loads((d / "saguaro-maya-m.json").read_text())["log"],
                             str(d / "saguaro-maya-m.log"))                    # the "log" key follows as ever
            self.assertIn("--config " + str(d / "saguaro-maya-m.json"), (d / "run-saguaro-maya-m.sh").read_text())

    def test_a_locked_sidecar_moves_at_the_next_start(self):
        """one file beside the config is open elsewhere while everything else moves: it is said and stays - and
        the next start's leftover pass moves it once it has let go, so the page's Turbo file reads again"""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            cfg_file(d, "maya-maya-m.json")
            (d / "maya-maya-m.turbo.json").write_text('{"turbo": 1}')
            real = os.replace

            def locked(src, dst, **kw):                     # (the one file Windows would not let go)
                if str(src).endswith("maya-maya-m.turbo.json"):
                    raise OSError(13, "Permission denied")
                return real(src, dst, **kw)

            with mock.patch.object(maya, "ROOT", d), mock.patch.object(maya, "warn") as warn, \
                    mock.patch.object(maya.os, "replace", locked):
                self.assertEqual(maya.migrate_config_names(), ["saguaro-maya-m.json"])
                self.assertEqual(warn.call_count, 1)       # said: it stays where it is
            self.assertTrue((d / "maya-maya-m.turbo.json").is_file())     # the config moved, this one did not
            self.assertTrue((d / "saguaro-maya-m.json").is_file())
            with mock.patch.object(maya, "ROOT", d), mock.patch.object(maya, "warn") as warn:
                self.assertEqual(maya.migrate_config_names(), [])         # it let go: the next start finishes
                warn.assert_not_called()
            self.assertFalse((d / "maya-maya-m.turbo.json").exists())
            self.assertEqual(json.loads((d / "saguaro-maya-m.turbo.json").read_text()), {"turbo": 1})

    def test_a_neighbouring_config_is_no_sidecar(self):
        """dot-named variants (maya-2.json beside maya-2.5.json): the first config's rename must not sweep the
        second's files - both move with their own sidecars, run scripts rewritten, nothing said"""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            a = cfg_file(d, "maya-2.json")
            b = cfg_file(d, "maya-2.5.json")
            (d / "maya-2.turbo.json").write_text('{"for": 2}')
            (d / "maya-2.5.turbo.json").write_text('{"for": "2.5"}')
            (d / "run-maya-2.sh").write_text("exec saguaro.py --config " + str(a) + " --port 8080\n")
            (d / "run-maya-2.5.sh").write_text("exec saguaro.py --config " + str(b) + " --port 8080\n")
            with mock.patch.object(maya, "ROOT", d), mock.patch.object(maya, "warn") as warn:
                self.assertEqual(sorted(maya.migrate_config_names()), ["saguaro-2.5.json", "saguaro-2.json"])
                self.assertEqual(maya.migrate_config_names(), [])              # once: nothing to do again
                warn.assert_not_called()
            self.assertEqual(sorted(f.name for f in d.iterdir() if f.is_file()),
                             ["run-saguaro-2.5.sh", "run-saguaro-2.sh", "saguaro-2.5.json", "saguaro-2.5.turbo.json",
                              "saguaro-2.json", "saguaro-2.turbo.json"])
            self.assertEqual(json.loads((d / "saguaro-2.turbo.json").read_text()), {"for": 2})     # each sidecar
            self.assertEqual(json.loads((d / "saguaro-2.5.turbo.json").read_text()), {"for": "2.5"})   # followed its own
            self.assertIn("--config " + str(d / "saguaro-2.5.json"), (d / "run-saguaro-2.5.sh").read_text())

    def test_a_neighbour_does_not_drag_a_shadowed_pair_around(self):
        """maya-2.json beside the pair maya-2.5.json + saguaro-2.5.json: the shadow goes aside and the
        neighbour's rename must not drag its sidecar into the live model's name"""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            a = cfg_file(d, "maya-2.json")
            cfg_file(d, "maya-2.5.json")
            cfg_file(d, "saguaro-2.5.json")
            (d / "maya-2.5.turbo.json").write_text('{"the shadow": 1}')
            (d / "run-maya-2.sh").write_text("exec saguaro.py --config " + str(a) + "\n")
            with mock.patch.object(maya, "ROOT", d), mock.patch.object(maya, "warn") as warn:
                self.assertEqual(maya.migrate_config_names(), ["saguaro-2.json"])
                self.assertEqual(warn.call_count, 1)                           # (only the pair being set aside)
            self.assertFalse((d / "maya-2.5.json").exists())                   # the shadow went aside, ...
            self.assertTrue((d / "maya-2.5.json.shadowed").is_file())          #  its sidecar stayed its own:
            self.assertEqual(json.loads((d / "maya-2.5.turbo.json").read_text()), {"the shadow": 1})   #  nobody
            self.assertFalse((d / "saguaro-2.5.turbo.json").exists())          #  else's Turbo state

    def test_both_names_the_new_one_and_the_old_goes_aside(self):
        """a hand-made pair: the new one is the model; the old file is set aside once as maya-shadow.json.shadowed
        - a name nothing loads - rather than waiting to be renamed back, and kept readable, not lost"""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            old = cfg_file(d, "maya-shadow.json", model_name="the old one")
            new = cfg_file(d, "saguaro-shadow.json", model_name="the new one")
            os.utime(old)                                                      # the old one even touched last
            with mock.patch.object(maya, "ROOT", d), mock.patch.object(maya, "warn") as warn:
                self.assertEqual(maya.migrate_config_names(), [])              # nothing is *renamed*: it goes aside
                self.assertEqual(warn.call_count, 1)                           # and it is said, once
                self.assertEqual([f.name for f in maya.configs()], ["saguaro-shadow.json"])
                self.assertEqual(maya.migrate_config_names(), [])              # once: the aside is no config
                self.assertEqual(warn.call_count, 1)
            aside = d / "maya-shadow.json.shadowed"
            self.assertEqual(json.loads(aside.read_text())["model_name"], "the old one")   # kept, not lost
            self.assertFalse(old.exists())
            self.assertEqual(models.config_paths(d), [new])
            self.assertEqual(model_api.find_config(d, "maya-shadow"), new)
            self.assertEqual(CN.resolve(old), new)                             # the old path still starts it

    def test_a_deleted_model_does_not_return_from_its_shadow(self):
        """the page shows and deletes only the new one of a hand-made pair: its old-name shadow goes with it, so
        the next start's rename has nothing to move back - the deleted model stays deleted"""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            old = cfg_file(d, "maya-shadow.json", model_name="the old one")
            new = cfg_file(d, "saguaro-shadow.json", model_name="the new one")
            (d / "maya-shadow.shared-settings.json").write_text("{}")
            (d / "run-maya-shadow.sh").write_text("#!/bin/sh\n")
            out = models.delete(d, new)
            self.assertEqual(out["deleted"], "saguaro-shadow.json")
            self.assertFalse(old.exists())                                     # the shadow went with the model:
            self.assertFalse((d / "maya-shadow.shared-settings.json").exists())
            self.assertFalse((d / "run-maya-shadow.sh").exists())
            self.assertEqual(models.config_paths(d), [])
            with mock.patch.object(maya, "ROOT", d), mock.patch.object(maya, "warn") as warn:
                self.assertEqual(maya.migrate_config_names(), [])              # the next start resurrects nothing
                warn.assert_not_called()
            self.assertEqual(models.config_paths(d), [])
            self.assertEqual(list(d.glob("maya-shadow*")), [])

    def test_read_only_folder_leaves_everything_working(self):
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("a read-only folder stops nobody but the owner (running as root)")
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            p = cfg_file(d, "maya-maya-m.json")
            (d / "run-maya-maya-m.sh").write_text("exec saguaro.py --config " + str(p) + "\n")
            os.chmod(d, 0o500)
            try:
                with mock.patch.object(maya, "ROOT", d), mock.patch.object(maya, "warn"):
                    self.assertEqual(maya.migrate_config_names(), [])          # said, not raised
            finally:
                os.chmod(d, 0o700)
            self.assertTrue(p.is_file())                                       # still there, still working:
            with mock.patch.object(maya, "ROOT", d):
                self.assertEqual([f.name for f in maya.configs()], ["maya-maya-m.json"])
            self.assertEqual(models.config_paths(d), [p])
            self.assertEqual(model_api.find_config(d, "maya-m"), p)        # the display name, under the old name
            self.assertEqual(models.read(p)["model_name"], "glm-5.3-flash")   # still readable
            os.remove(d / "maya-maya-m.json")                                  # (and deletable again)

    def test_updated_after_the_move(self):
        """About > Updates: the server ends with UPDATE_EXIT and the new version is started again with the same
        --config path it was given - the old name of the file this very move renamed, which its main() resolves"""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            p = cfg_file(d, "maya-maya-m.json")
            with mock.patch.object(maya, "ROOT", d), mock.patch.object(maya, "warn"):
                maya.migrate_config_names()
            moved = d / "saguaro-maya-m.json"
            a = SimpleNamespace(port=None, host=None, api_key=None, gpu=None, gpus=None, backend="cuda", config=p)
            with mock.patch.object(maya, "ROOT", d), mock.patch.object(maya, "say"), \
                    mock.patch.object(maya.os, "execv") as execv:
                self.assertEqual(maya.after_server(maya.UPDATE_EXIT, a), 0)
            argv = execv.call_args.args[1]
            self.assertEqual(argv[argv.index("--config") + 1], str(p))         # the path as it was given, and:
            self.assertEqual(CN.resolve(Path(argv[argv.index("--config") + 1])), moved)   # the start finds the file

    def test_deleted_after_the_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            cfg_file(d, "maya-maya-m.json")
            (d / "maya-maya-m.shared-settings.json").write_text("{}")
            (d / "run-maya-maya-m.sh").write_text("#!/bin/sh\n")
            with mock.patch.object(maya, "ROOT", d), mock.patch.object(maya, "warn"):
                maya.migrate_config_names()
            moved = d / "saguaro-maya-m.json"
            cfg = models.read(moved)
            models.write(moved, {**cfg, "port": 8090})                         # settings saved into it: as before
            self.assertEqual(models.read(moved)["port"], 8090)
            out = models.delete(d, moved)                                      # and it deletes with them
            self.assertEqual(out["deleted"], "saguaro-maya-m.json")
            for gone in ("saguaro-maya-m.json", "run-saguaro-maya-m.sh", "saguaro-maya-m.shared-settings.json"):
                self.assertFalse((d / gone).exists(), gone)
            self.assertEqual(models.config_paths(d), [])


class Notes(unittest.TestCase):
    """the server's notes: read under either name (the new one first), written under the new one alone"""

    def test_note_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertEqual(CN.note_paths(d, models.SWITCH_FILE), [])
            (d / ".maya-switch.json").write_text("{}")
            self.assertEqual(CN.note_paths(d, models.SWITCH_FILE), [d / ".maya-switch.json"])
            (d / models.SWITCH_FILE).write_text('{"a": 1}')
            self.assertEqual(CN.note_paths(d, models.SWITCH_FILE),
                             [d / ".saguaro-switch.json", d / ".maya-switch.json"])   # the new name first
            self.assertEqual(CN.note_paths(d, "unrelated.txt"), [])        # only notes that are there

    def test_new_names(self):
        self.assertEqual(models.SWITCH_FILE, ".saguaro-switch.json")
        self.assertEqual(models.DELETE_FILE, ".saguaro-delete.json")
        self.assertEqual(maya.SWITCH_FILE, ".saguaro-switch.json")

    def test_a_stray_note_does_not_survive_the_server(self):
        """a crash between the server writing its switch note and restarting: a leftover of either family goes
        with the next non-restart exit (as 1.2.1 cleared it) instead of lingering over starts and updates"""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            a = SimpleNamespace(port=None, host=None, api_key=None, gpu=None, gpus=None, backend="cuda", config=None)
            for rc in (0, maya.UPDATE_EXIT):                                   # a normal stop and an update
                (d / ".maya-switch.json").write_text(json.dumps({"why": "switch", "time": time.time()}))
                (d / models.SWITCH_FILE).write_text(json.dumps({"why": "settings", "time": time.time()}))
                with mock.patch.object(maya, "ROOT", d), mock.patch.object(maya, "say"), \
                        mock.patch.object(maya, "WIN", False), mock.patch.object(maya.os, "execv"):
                    maya.after_server(rc, a)
                self.assertFalse((d / ".maya-switch.json").exists(), rc)
                self.assertFalse((d / models.SWITCH_FILE).exists(), rc)

    def test_switch_note_after_the_rename(self):
        """a switch note naming the old path of a config this very start renamed: saguaro.py follows it anyway"""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            old = cfg_file(d, "maya-maya-m.json", backend="hip")
            with mock.patch.object(maya, "ROOT", d), mock.patch.object(maya, "warn"):
                maya.migrate_config_names()
            moved = d / "saguaro-maya-m.json"
            (d / ".maya-switch.json").write_text(json.dumps({"config": str(old), "why": "switch", "port": 8123,
                                                             "time": time.time()}))   # a note of before the rename
            a = SimpleNamespace(port=8090, host=None, api_key=None, gpu=None, gpus=None, backend="cuda", config=None)
            with mock.patch.object(maya, "ROOT", d), mock.patch.object(maya, "say"), \
                    mock.patch.object(maya.os, "execv") as execv:
                maya.after_server(maya.RESTART_EXIT, a)
            argv = execv.call_args.args[1]
            self.assertEqual(argv[argv.index("--config") + 1], str(moved))     # followed, to the moved file
            self.assertFalse((d / ".maya-switch.json").exists())               # and read once


if __name__ == "__main__":
    unittest.main()
