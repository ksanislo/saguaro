"""tools/run_qfn_serve_tests.py - run a QFN serve test module standalone (default: the plan-named
serve/test_fatal_recovery.py, the drain/kill protocol of P124x d).

engines/qfn/serve/server.py carries `from serve import turbo_live` (its own line since 936183783) whose file
lives in the repo-root serve package, so importing the QFN serve package standalone fails at that line - on
this checkout's base as well as now (pre-existing, noted in PLAN-p124x's REPORT 3, not fixed there).  This
runner loads the QFN serve package the tests import and registers the root's serve/turbo_live.py under the
name that import asks for, before anything else imports serve.server.  The test module itself is untouched:
same file, same assertions.  (re-review L3: this existed in scratch/ while the branch was reviewed - it is
committed so the named module runs on a clean checkout.)

    python tools/run_qfn_serve_tests.py [serve.test_module ...]
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines" / "qfn"))

import serve  # noqa: E402  the QFN serve package

_spec = importlib.util.spec_from_file_location("serve.turbo_live", str(ROOT / "serve" / "turbo_live.py"))
_turbo = importlib.util.module_from_spec(_spec)
sys.modules["serve.turbo_live"] = _turbo
_spec.loader.exec_module(_turbo)
serve.turbo_live = _turbo

names = [("serve." + a if "." not in a else a) for a in (sys.argv[1:] or ["test_fatal_recovery"])]
suite = unittest.defaultTestLoader.loadTestsFromNames(names)
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
