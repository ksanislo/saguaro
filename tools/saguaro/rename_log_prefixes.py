#!/usr/bin/env python3
"""tools/saguaro/rename_log_prefixes.py - the engines' log lines say "saguaro <part>:", not the two old engine names.

    python3 tools/saguaro/rename_log_prefixes.py [--check] [paths...]   rewrite in place, or check without writing

A string literal that opens with `<engine> <part>:` or `<engine>-<part>: ` (or `<engine> <part> (`) is an engine log
or stderr prefix: the engine name becomes `saguaro` and the part after it stays as it is - "strata generate:" and
"maya generate:" both become "saguaro generate:", "maya-ds4:" becomes "saguaro ds4:", "ds4:" becomes "saguaro ds4:"
(the ds4 program's own name is the part it keeps), "strata-vision:" and "maya-vision:" become "saguaro vision:".
Names with other text after them are NOT prefixes and stay: include paths ("strata/core.h"), GGUF keys
("strata.ple.format"), file names ("ds4.txt: ..."), session and temp names ("strata-session-..."), dlsym symbols,
invocation and usage lines ("maya generate --pack ..."), metric namespaces ("strata:requests_total" - no space
after the colon), bare identifiers ("strata") and prose.  Comments and docstrings are not literals and stay; the
readers of these lines (serve, saguaro.py, tools) are updated with the engines, in the same commit.

Default paths: the engines' sources - src/, engines/qfn/src/ and both vision encoders (tools/vision,
engines/qfn/tools/vision).  Pass .py files explicitly to apply the same rules to their literals (the tests that
simulate an engine's stderr).  Deterministic and idempotent: re-run after merges.  `--check` exits 1 while a
renameable prefix is left.  See REPORT-visible-names.md.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NEW = "saguaro"
OLD_P = ("strata", "maya")                              # split up so scanning this file finds no complete old name
NAME = "(?:" + "|".join(OLD_P) + ")"                    # "strata" or "maya", not strata_x / strata-x as a whole word
P = r"[a-z0-9][a-z0-9 ._()#-]*"                         # the part after the engine name, up to the colon or the (
RULES = [
    (re.compile(r'(["\'])(' + NAME + r')(?![A-Za-z0-9_-])(?=: )'), r"\1" + NEW),                       # "strata: x"
    (re.compile(r'(["\'])(' + NAME + r')( ' + P + r')(?=: )'), r"\1" + NEW + r"\3"),                   # "strata part: x"
    (re.compile(r'(["\'])(' + NAME + r')( ' + P + r')(?= \()'), r"\1" + NEW + r"\3"),                  # "strata part (x"
    (re.compile(r'(["\'])(' + NAME + r')-([a-z0-9]+)(?=(?:: | [A-Za-z(]))'), r"\1" + NEW + r" \3"),    # "maya-ds4: x"
    (re.compile(r'(["\'])(ds4)(?=[: ])'), r"\1" + NEW + r" ds4"),                                # "ds4: x", "ds4 x"
    (re.compile(r'(["\'])(' + NAME + r') (batch GPU stages|decode GPU stages|KV prefetch|DMA|DBG|Q2_0)'),
     r"\1" + NEW + r" \3"),                                                                      # "strata GPU/DBG ... :"
    (re.compile(r'(["\'])(' + NAME + r')(?=" \+)'), r"\1" + NEW),                                # "strata " + label
    (re.compile(r'(["\'])strata-dsv41(?= %s build)'), r"\1" + NEW + " dsv41"),                   # the --version line
]
CPP_SUFFIXES = {".c", ".h", ".hpp", ".cpp", ".cc", ".cxx", ".cu", ".cuh"}
# In files passed explicitly (a test simulating an engine's log, a docstring showing its lines), a line inside a
# multi-line string starts with the prefix too: the same rules with a line anchor instead of a quote.
LINE_RULES = [(re.compile(r"^(" + NAME + r")(?![A-Za-z0-9_-])(?=: )", re.M), NEW),
              (re.compile(r"^(" + NAME + r")( [a-z0-9][a-z0-9 ._()#-]*)(?=: )", re.M), NEW + r"\2"),
              (re.compile(r"^(" + NAME + r")( [a-z0-9][a-z0-9 ._()#-]*)(?= \()", re.M), NEW + r"\2"),
              (re.compile(r"^(" + NAME + r")-([a-z0-9]+)(?=(?:: | [A-Za-z(]))", re.M), NEW + r" \2"),
              (re.compile(r"^(ds4)(?=[: ])", re.M), NEW + r" ds4"),
              (re.compile(r"^(" + NAME + r")( (?:batch GPU stages|decode GPU stages|KV prefetch|DMA|DBG|Q2_0))", re.M),
               NEW + r"\2")]
DEFAULT_PATHS = [ROOT / "src", ROOT / "engines" / "qfn" / "src",
                 ROOT / "engines" / "qfn" / "tools" / "vision", ROOT / "tools" / "vision"]


def targets(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out += [f for f in sorted(p.rglob("*")) if f.is_file() and f.suffix in CPP_SUFFIXES]
        elif p.is_file():
            out.append(p)
        else:
            sys.exit(f"{p}: no such file or directory")
    return out


def rewrite(text: str, line_rules: bool = False) -> tuple[str, int]:
    n = 0
    for rx, to in (RULES + (LINE_RULES if line_rules else [])):
        text, k = rx.subn(to, text)
        n += k
    return text, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report what is left and exit 1, write nothing")
    ap.add_argument("paths", nargs="*", type=Path, help="files or trees (default: the engines' sources)")
    a = ap.parse_args()
    files, changed = targets(a.paths or DEFAULT_PATHS), 0
    for f in files:
        old = f.read_text(encoding="utf-8", errors="replace")
        new, n = rewrite(old, line_rules=bool(a.paths))
        if not n:
            continue
        changed += n
        print(f"{f.relative_to(ROOT) if f.is_absolute() else f}: {n}")
        if not a.check:
            f.write_text(new, encoding="utf-8")
    print(("check: " + ("no renameable prefix left" if not changed else f"{changed} prefixes left to rename"))
          if a.check else f"renamed {changed} prefixes in {len(files)} files")
    return 1 if (a.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
