#!/usr/bin/env python3
"""tools/saguaro/rename_upstream.py - the whole rename layer as ONE tested command (docs/UPSTREAM.md section 2).

    python3 tools/saguaro/rename_upstream.py [--check] [--diff] [--paths] [paths...]

One run over an upstream checkout (or the files a merge brings in) turns upstream's names into ours, so what gets
merged shows only REAL conflicts:

  internal    include/strata -> include/maya, strata:: / namespace strata -> maya, strata_* cmake words,
              STRATA_X build options -> MAYA_X                    (the table of rename_internal_maya.py)
  settings    MAYA_X / STRATA_X -> SAGUARO_X by pattern - a NEW upstream MAYA_FOO is caught without being
              listed - where the name reaches the environment, and its proven mentions everywhere
                                                                  (the logic of rename_env.py)
  logprefix   "strata part:" / "maya-ds4:" / "ds4:" -> "saguaro ..."  (the table of rename_log_prefixes.py)
  names       maya.py -> saguaro.py, maya.sh -> saguaro.sh, START-MAYA.bat, maya-nightly*, the saguaro-*
              temp-dir prefixes, [maya] -> [saguaro], "maya --speed-calibration" -> "saguaro.py ..."
  brand       "restart Maya" / "Maya checks ..." -> Saguaro in user-visible docs; the credits, "Project Maya",
              "Maya v1.1.x" and the Maya quant names (Maya-S/M/L/XS/S24) stay (289431642's rule, .md only)

Idempotent: no rule's replacement is ever a rule's match, so a second run changes nothing.  Exception list
(rename_rules.EXC_PATHS / EXC_TEXT): record texts quoting upstream commands, the maya-*.json config names, the
`maya.` browser keys, the maya_kernels / maya_mmq targets, ds4/hy4 internal names, vendored dirs, LICENSE texts,
the history files and the three leftovers of C.4 of scratch/notes.md - kept, and REPORTED (reported kept hits do
not fail --check; a renameable use does).

--check writes nothing and exits 1 while a renameable name is left (the intake's gate, like the three tools').
--diff prints what would change.  --paths prints the PATH renames (old<TAB>new per line: maya.py, maya.sh,
START-MAYA.bat, the MAYA_* docs, include/strata/) for `git mv`; it changes nothing either.
Default paths: the whole worktree.  The three older scripts keep working on their own; all four read the one
rule table, tools/saguaro/rename_rules.py.  See scratch/notes.md and REPORT-rn1-rename-tool.md.
"""
from __future__ import annotations

import argparse
import difflib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rename_rules as T
import rename_env as renv

ROOT = Path(__file__).resolve().parents[2]
FAM_NAME = "internal settings logprefix names brand".split()


def collect(paths: list[Path]) -> dict:
    """{rel: (path, family|None, text, bom, explicit)} over paths (dirs walk, files pass) minus every skip"""
    out: dict = {}
    for root in paths:
        explicit = root.is_file()
        cand = [root] if explicit else []
        if not cand:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in T.SKIP_DIR_NAMES and not T.SKIP_DIR_RE.match(d)]
                cand += [Path(dirpath) / fn for fn in filenames]
        for p in cand:
            try:
                rel = p.relative_to(ROOT).as_posix()
            except ValueError:
                sys.exit(f"{p}: outside the worktree {ROOT}")
            if T.skipped(rel) or rel.startswith(T.INTERNAL_SKIP_PATHS):   # QFN's tree: see the docstring
                continue
            fam = renv.family_of(p)
            if fam is None and p.suffix.lower() not in T.INT_TEXT_SUFFIXES:
                continue
            got = renv.load(p)
            if got is None:
                continue
            out[rel] = [p, fam, got[0], got[1], explicit or root.is_file()]
    return out


def apply_names(text: str, rel: str, fam, explicit: bool) -> tuple[str, list]:
    """the new (M1 hand-rename) families plus the engine log prefixes for one file: (text, families changed)"""
    fams = []
    if not T.exc_path(rel):
        masked, hidden = T.mask_exceptions(text, rel)
        n = 0
        if fam in T.NAMES_FAM:
            for rx, to in T.NAMES_RULES + T.CMD_RULES:
                masked, k = rx.subn(to, masked)
                n += k
            if fam in T.LOGTAG_FAM:
                for rx, to in T.LOGTAG_RULES:
                    masked, k = rx.subn(to, masked)
                    n += k
            if n:
                fams.append("names")
        if fam in T.BRAND_FAM:
            b = 0
            for rx, to in T.BRAND_RULES:
                masked, k = rx.subn(to, masked)
                b += k
            if b:
                fams.append("brand")
        text = T.unmask_exceptions(masked, hidden)
    if explicit or (fam == "cpp" and rel.startswith(T.LOG_SRC_PATHS)
                    and Path(rel).suffix.lower() in T.LOG_CPP_SUFFIXES):
        rules = T.LOG_RULES + (T.LOG_LINE_RULES if explicit else [])
        n = 0
        for rx, to in rules:
            text, k = rx.subn(to, text)
            n += k
        if n:
            fams.append("logprefix")
    return text, fams


def exception_hits(files: dict, texts: dict) -> list:
    """the stale old-name mentions the exception list keeps (scratch/notes.md C): reported, never rewritten"""
    hits = []
    for exc_rel, rx in T.EXC_TEXT:
        if exc_rel in texts:
            for m in rx.finditer(texts[exc_rel]):
                line = texts[exc_rel][:m.start()].count("\n") + 1
                hits.append(f"{exc_rel}:{line}: {m.group(0)}")
    return hits


def plan_tree(files: dict):
    """run every family over the collected tree; returns (texts, changed {rel: families}, setting_report)"""
    texts = {rel: f[2] for rel, f in files.items()}
    fams = {rel: f[1] for rel, f in files.items()}
    changed: dict = {rel: [] for rel in files}

    def bump(rel, fam):
        if fam not in changed[rel]:
            changed[rel].append(fam)

    for rel, (p, fam, text, bom, explicit) in files.items():
        if p.suffix.lower() in T.INT_TEXT_SUFFIXES and not T.exc_path(rel):
            masked, hidden = T.mask_exceptions(texts[rel], rel)   # the exception list's STRATA_ names stay
            t = T.unmask_exceptions(T.rename_internal(masked, p), hidden)   # (their docs and tests too)
            if t != texts[rel]:
                texts[rel] = t
                bump(rel, "internal")
    # settings: proved over the internal-renamed texts, exactly rename_env's decision pass
    sfiles = {rel: (f[0], f[1], texts[rel], f[3]) for rel, f in files.items() if f[1] is not None}
    renames, collisions, allow_hits, kept_words, names, merged, both = renv.plan(sfiles)
    for rel, oc in renames.items():
        texts[rel] = renv.apply(texts[rel], oc)
        bump(rel, "settings")
    for rel, (p, fam, text, bom, explicit) in files.items():
        t, fams = apply_names(texts[rel], rel, fam, explicit)
        if fams:
            texts[rel] = t
            changed[rel] += fams
    setting_report = dict(total=sum(len(v) for v in renames.values()), names=sorted(names), collisions=collisions,
                          allow=len(allow_hits), kept=len(kept_words))
    return texts, {r: sorted(set(c)) for r, c in changed.items() if c}, setting_report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="write nothing; exit 1 while a renameable name is left")
    ap.add_argument("--diff", action="store_true", help="print the changes (writes nothing)")
    ap.add_argument("--paths", dest="paths_mode", action="store_true",
                    help="print the file/dir renames (old<TAB>new) for `git mv`")
    ap.add_argument("paths", nargs="*", default=None)
    a = ap.parse_args()
    roots = [Path(x).resolve() for x in (a.paths or [ROOT])]

    if a.paths_mode:
        for root in roots:
            for p in ([root] if root.is_file() else sorted(root.rglob("*"))):
                if not p.is_file():
                    continue
                try:
                    rel = p.relative_to(ROOT).as_posix()
                except ValueError:
                    sys.exit(f"{p}: outside the worktree {ROOT}")
                name = p.name
                new = next((n for o, n in T.NAME_PATHS if name == o), None)
                if new is None:
                    pre = next(((pre, to) for pre, to in T.NAME_PATH_PREFIXES if name.startswith(pre)), None)
                    new = pre[1] + name[len(pre[0]):] if pre else None
                if new:
                    print(rel + "\t" + (rel[:len(rel) - len(name)] if "/" in rel else "") + new)
        for old, new in T.INTERNAL_PATHS:
            if (ROOT / old).is_dir():
                print(old + "\t" + new)
        return 0

    files = collect(roots)
    texts, changed, rep = plan_tree(files)
    hits = exception_hits(files, texts)

    by_fam = {f: 0 for f in FAM_NAME}
    for fams in changed.values():
        for f in fams:
            by_fam[f] += 1
    if changed:
        print(f"rename_upstream: {len(files)} files read; changed: "
              + ", ".join(f"{n} {f}" for f, n in by_fam.items() if n))
    else:
        print(f"rename_upstream: {len(files)} files read; nothing to rename")
    if rep["collisions"]:
        print(f"collisions (a SAGUARO_* of the same name already exists - left, review): "
              f"{', '.join(sorted(rep['collisions']))}")
    if hits:
        print(f"kept by the exception list ({len(hits)}) - names that stay; the stale ones wait for a product pass:")
        for h in hits:
            print("  " + h)

    if a.diff or not a.check:
        for rel in sorted(changed):
            if a.diff:
                old = files[rel][2].splitlines(keepends=True)
                new = texts[rel].splitlines(keepends=True)
                sys.stdout.writelines(difflib.unified_diff(old, new, fromfile=rel, tofile=rel))
            elif not a.check:
                p, fam, text, bom, explicit = files[rel]
                p.write_bytes((b"\xef\xbb\xbf" if bom else b"") + texts[rel].encode("utf-8"))
    if a.check:
        if changed:
            print(f"CHECK FAILED: {len(changed)} file(s) would change, e.g.: " + ", ".join(sorted(changed)[:10]))
            return 1
        print("CHECK OK: the rename layer is applied (exceptions and excluded paths aside)")
    elif not a.diff:
        print(f"applied: rewrote {len(changed)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
