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
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD, NEW = "str" + "ata", "maya"           # split so this file holds no complete old word to find
UP = OLD.upper()

SKIP_PREFIX = ("engines/qfn/", "tools/qfn/", "serve/qfn_engine.py", "third_party/", "tools/saguaro/", "bench/results/", "ref/")
SKIP_NAME = re.compile(r"^(CHANGELOG\.md|NOTICE\.md|(PLAN|REPORT|NEXT)-.*\.md)$")
TEXT_SUFFIXES = {".c", ".h", ".hpp", ".cpp", ".cc", ".cu", ".cuh", ".hip", ".inc", ".inl", ".metal", ".mm",
                 ".py", ".sh", ".ps1", ".bat", ".cmd", ".cmake", ".txt", ".md", ".yml", ".yaml", ".json", ".in", ""}
CMAKE = lambda p: p.suffix == ".cmake" or p.name == "CMakeLists.txt"

# lowercase strata_* words that are code names upstream renamed (cmake targets, a few helpers); others (the
# Strata pack format's strata_fmt / strata_pack / strata_inkling, libstrata_kq) stay
CODE_WORDS = {"mmq", "cpu_pause", "store_fence", "engine", "mmq_pool_slab", "dp4a_sm60", "q2_kernel", "kernels_cpu",
              "kernels", "hipblas_set_workspace_noop", "profile_version_test", "profile_save_test", "ik_oracle",
              "ik_oracle_help", "glm_maya_sync_test", "gguf_reader_test", "ds4_ablate", "ds4_ablate_junk",
              "mm3_ablate_missing", "warnings", "prefill", "platform", "core", "spec", "program", "artifact",
              "ngram", "plan"}
KEEP_LOWER = re.compile(r"^(fmt|pack|inkling|kq|kq_tab|fmt_test|fmt_gpu_test|ref|tokenizer)\b")

R_INC = re.compile(r"\binclude/" + OLD + r"(?=[/\s\"'`)\]>;:,]|$)", re.M)
R_DIR = re.compile(r"(?<![\w.-])(?<!tools/)(?<!docs/)" + OLD + r"/(?=(core|kernels|spec|platform|artifact|prefill|program|plan|ngram|hip_compat)/)")
R_HDR = re.compile(r"([\"<])" + OLD + r"/")
R_NS = re.compile(r"\b" + OLD + r"(?=::)")
R_NSD = re.compile(r"\bnamespace " + OLD + r"\b")
R_UP = re.compile(r"(?:\b|(?<=-D)|(?<=/D))" + UP + r"_([A-Z0-9][A-Z0-9_]*)\b")
R_LOW = re.compile(r"(?<![/\w.])(_?)" + OLD + r"_(\w+)")


def low_sub(cmake: bool):
    def f(m):
        pre, rest = m.group(1), m.group(2)
        if KEEP_LOWER.match(rest):
            return m.group(0)
        if cmake or pre or rest in CODE_WORDS or rest.split("_")[0] in ("ds4", "mm3", "glm", "ik") and "ablate" in rest:
            return pre + NEW + "_" + rest
        if rest.startswith(("ds4_ablate", "mm3_ablate", "glm_dsa_loader")):
            return pre + NEW + "_" + rest
        return m.group(0)
    return f


def up_sub(line: str):
    """STRATA_X -> MAYA_X, but not where the line also names MAYA_X (the older spelling beside the new)"""
    def f(m):
        if "MAYA_" + m.group(1) in line:
            return m.group(0)
        return "MAYA_" + m.group(1)
    return f


def qfn_block(text: str):
    """the QFN sub-build's lines (after `if(SAGUARO_QFN AND ...)` up to its custom target): engines/qfn's own CMake
    project keeps its STRATA_ options, so its -D arguments and option names there stay"""
    a = text.find("if(SAGUARO_QFN AND")
    if a < 0:
        return None
    a = text.find("\n", a)
    b = text.find("add_custom_target(saguaro-qfn", a)
    return (a, b) if b > 0 else None


def rename(text: str, path: Path) -> str:
    cm = CMAKE(path)
    keep = qfn_block(text) if cm else None
    head, mid, tail = (text[:keep[0]], text[keep[0]:keep[1]], text[keep[1]:]) if keep else (text, "", "")

    def one(t, up=True):
        t = R_INC.sub("include/" + NEW, t)
        t = R_HDR.sub(lambda m: m.group(1) + NEW + "/", t)
        t = R_NS.sub(NEW, t)
        t = R_NSD.sub("namespace " + NEW, t)
        t = R_DIR.sub(NEW + "/", t)
        if up:
            t = "\n".join(R_UP.sub(up_sub(ln), ln) for ln in t.split("\n"))
        t = R_LOW.sub(low_sub(cm), t)
        return t

    return one(head) + one(mid, up=False) + one(tail) if keep else one(text)


def files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode()
    for rel in out.split("\0"):
        if not rel or rel.startswith(SKIP_PREFIX) or SKIP_NAME.match(Path(rel).name):
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
        n = rename(t, p)
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
