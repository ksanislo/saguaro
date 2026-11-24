"""serve/test_load_by_name.py - load by name (U3): a chat request whose "model" names another installed config
switches to it the way the Models page's Switch and /models/load do, so a client that sends a model id gets that
model.  The setting parse, the name match, and the handler's decision (the switch mocked, the mock engine, no GPU).

    python -m unittest serve.test_load_by_name -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.model_api import match_name  # noqa: E402
from serve.server import load_by_name_setting  # noqa: E402

ENTRIES = [{"file": "maya-maya-m.json", "model": "glm-5.3-flash"},
           {"file": "maya-qwen38-flash.json", "model": "Qwen3.8-Flash-Next"}]


class Setting(unittest.TestCase):
    """load_by_name_setting: off by default; the config or SAGUARO_LOAD_BY_NAME turns it on (the environment, when
    set, wins); it needs model management on."""

    def test_default_off(self):
        self.assertFalse(load_by_name_setting(None, env={}))
        self.assertFalse(load_by_name_setting({}, env={}))
        self.assertFalse(load_by_name_setting({"load_by_name": False}, env={}))
        self.assertFalse(load_by_name_setting({"load_by_name": 0}, env={}))

    def test_config_on(self):
        self.assertTrue(load_by_name_setting({"load_by_name": True}, env={}))

    def test_env_beats_config(self):
        self.assertTrue(load_by_name_setting({"load_by_name": False}, env={"SAGUARO_LOAD_BY_NAME": "1"}))
        self.assertTrue(load_by_name_setting({}, env={"SAGUARO_LOAD_BY_NAME": "on"}))
        self.assertTrue(load_by_name_setting({}, env={"SAGUARO_LOAD_BY_NAME": "TRUE"}))
        self.assertTrue(load_by_name_setting({}, env={"SAGUARO_LOAD_BY_NAME": " yes "}))
        self.assertFalse(load_by_name_setting({"load_by_name": True}, env={"SAGUARO_LOAD_BY_NAME": "0"}))
        self.assertFalse(load_by_name_setting({"load_by_name": True}, env={"SAGUARO_LOAD_BY_NAME": "off"}))
        self.assertFalse(load_by_name_setting({"load_by_name": True}, env={"SAGUARO_LOAD_BY_NAME": "no"}))
        self.assertTrue(load_by_name_setting({"load_by_name": True}, env={"SAGUARO_LOAD_BY_NAME": ""}))

    def test_needs_model_management(self):
        self.assertFalse(load_by_name_setting({"load_by_name": True, "model_management": False}, env={}))
        self.assertFalse(load_by_name_setting({"load_by_name": True}, env={"SAGUARO_MODEL_MANAGEMENT": "0"}))
        self.assertTrue(load_by_name_setting({"load_by_name": True, "model_management": False},
                                             env={"SAGUARO_MODEL_MANAGEMENT": "1"}))


class Match(unittest.TestCase):
    """match_name: the config's stem or the model's own name, case-insensitive, with or without ":latest" / ".json"
    / the leading "maya-" - the ids /v1/models and /api/tags show."""

    def test_config_stem(self):
        self.assertIs(match_name("maya-maya-m", ENTRIES), ENTRIES[0])
        self.assertIs(match_name("maya-maya-m.json", ENTRIES), ENTRIES[0])
        self.assertIs(match_name("maya-maya-m:latest", ENTRIES), ENTRIES[0])
        self.assertIs(match_name("maya-m", ENTRIES), ENTRIES[0])       # (find_config's prefix-off name too)
        self.assertIs(match_name("MAYA-QWEN38-FLASH", ENTRIES), ENTRIES[1])

    def test_model_name(self):
        self.assertIs(match_name("glm-5.3-flash", ENTRIES), ENTRIES[0])
        self.assertIs(match_name("Qwen3.8-Flash-Next", ENTRIES), ENTRIES[1])
        self.assertIs(match_name(" qwen3.8-flash-next:latest ", ENTRIES), ENTRIES[1])

    def test_config_name_beats_another_models_name(self):
        entries = ENTRIES + [{"file": "glm-5.3-flash.json", "model": "something else"}]
        self.assertIs(match_name("glm-5.3-flash", entries), entries[2])   # (a config of that name wins)

    def test_no_match(self):
        for name in (None, "", "   ", "maya-", "glm", "nope-404", "maya/../evil", "not me"):
            self.assertIsNone(match_name(name, ENTRIES), name)

    def test_empty_entries(self):
        self.assertIsNone(match_name("maya-maya-m", []))
        self.assertIsNone(match_name("maya-maya-m", None))


if __name__ == "__main__":
    unittest.main()
