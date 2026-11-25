#!/usr/bin/env python3
"""tools/saguaro/rename_internal_maya.py - upstream Maya v1.1.75's internal rename (its dc6e746a4), applied to our tree.

    python3 tools/saguaro/rename_internal_maya.py [--check] [--no-move]

include/strata -> include/maya, `strata::` / `namespace strata` -> maya, CMake targets and variables strata_* /
_strata_* -> maya_* / _maya_*, build options and compile defines STRATA_X -> MAYA_X.  Our run-time settings are
already SAGUARO_* (tools/saguaro/rename_env.py) and are not touched.  Kept: engines/qfn, tools/qfn and the generated serve/qfn_engine.py (Strata's own tree and
CMake project, its ExternalProject arguments too), the Strata pack format's names (src/strata_fmt, strata_pack,
strata_inkling, .strata), tools/strata/, file names, history files and the rename tools.

Doing what upstream did lets later upstream commits apply as they are.  A branch still written against
include/strata runs this once before it merges (idempotent; --check exits 1 while anything is left to rename).
The skips and the exception strings that stay are the shared lists of rename_rules.py (T.internal_skipped,
T.EXC_TEXT), the same ones rename_upstream.py uses.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rename_rules as T                          # the shared rule table (rename_rules.py)

ROOT = Path(__file__).resolve().parents[2]
OLD, NEW = T.INT_OLD, T.INT_NEW                    # split in the table, so this file holds no complete old word
UP = T.UP
rename = T.rename_internal                         # the per-file rewrite, shared with rename_upstream.py

TEXT_SUFFIXES = T.INT_TEXT_SUFFIXES
CMAKE = T.is_cmake


def files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode()
    for rel in out.split("\0"):
        if not rel or T.internal_skipped(rel):
            continue
        p = ROOT / rel
        if p.suffix.lower() not in TEXT_SUFFIXES or not p.is_file() or p.is_symlink():
            continue
        yield rel, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-move", action="store_true")
    a = ap.parse_args()
    left = 0
    if (ROOT / "include" / OLD).is_dir():
        left += 1
        if not a.check and not a.no_move:
            subprocess.run(["git", "mv", f"include/{OLD}", f"include/{NEW}"], cwd=ROOT, check=True)
    for rel, p in files():
        try:
            raw = p.read_bytes()
            t = raw.decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        masked, hidden = T.mask_exceptions(t, rel)      # the exception list's names stay, as in rename_upstream
        n = T.unmask_exceptions(rename(masked, p), hidden)
        if n != t:
            left += 1
            if a.check:
                print(rel)
            else:
                p.write_bytes(n.encode("utf-8"))
    if a.check and left:
        sys.exit(1)


if __name__ == "__main__":
    main()
