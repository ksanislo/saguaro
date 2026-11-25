#!/usr/bin/env python3
"""tools/saguaro/rename_env.py - the settings are SAGUARO_*, not the two old prefixes (same suffix, one name, no aliases).

    python3 tools/saguaro/rename_env.py [--check] [paths...]   rewrite in place, or check without writing

A `\\bOLD_([A-Z0-9_]+)\\b` occurrence is renamed to `SAGUARO_<suffix>` when the name reaches the environment: a string
literal the code uses as a name (in a getenv/setenv/environ context, or quoted as a dict/config key), a json/yaml key,
a shell assignment (`NAME=`, `$NAME`, `%NAME%`, `set NAME`, `env NAME=`) or `$ENV{..}`/`ENVIRONMENT NAME=` in cmake, a
Python `NAME=` keyword argument, a JS `env.NAME`/`{NAME:` property, or `cmake -E env NAME=`, and - for a name proved a
setting that way - every further mention of it: messages, comments, and docs.

A name proved nowhere stays a word and is kept everywhere, docs included: C/C++/CUDA macros, enums, header guards,
CMake option and variable words, `option(STRATA_..)`/`#define STRATA_..` texts in build scripts, module constants,
`-DNAME=`/`/DNAME=` build flags, Docker ARG lines without `ENV`, and prose that names the old binaries.  This is what
protects e.g. the API-export macro and the build options; skipped words are printed.

Collisions (SAGUARO_X already exists where OLD_X would land) are reported and not renamed.  Both old prefixes on one
suffix are NOT a collision: the old alias code made them one setting; they merge into one SAGUARO_X (printed).

Excluded: the shared skip list (rename_rules.skipped(): third_party/, .git, build trees, node_modules, ref-configs/
(migrate_configs.py reports those instead), the results/records dirs, *.log/*.jsonl/*.patch, the history files and
PLAN/REPORT/NEXT-*.md, ref/ and scratch/, this tool's own folder (it must spell the old prefixes to find them) and
its test module, whose fixtures are no longer collisions of themselves).  Deterministic and idempotent: re-run after
merges.  `--check` exits 1 while a renameable use is left.  See REPORT-env-rename.md.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rename_rules as T                          # the shared rule table (rename_rules.py)

ROOT = Path(__file__).resolve().parents[2]
NEW = T.NEW
OLD_P = T.OLD_ENV                                  # split up in the table, so scans find no complete old name
NAME = T.NAME
HAS_SAGUARO = T.HAS_SAGUARO

# the skip lists (history, vendored, records, this folder) are the shared ones in rename_rules.py: T.skipped()
CONF_SUFFIXES = {".json", ".yaml", ".yml"}
DOC_SUFFIXES = {".md", ".markdown", ".txt", ".in", ".example"}
CPP_SUFFIXES = {".c", ".h", ".hpp", ".cpp", ".cc", ".cxx", ".cu", ".cuh", ".hip", ".metal", ".inl", ".inc", ".mm",
                ".rs", ".ts", ".tsx"}
PY_SUFFIXES = {".py", ".pyw"}
JS_SUFFIXES = {".js", ".mjs", ".cjs"}
CMAKE_SUFFIXES = {".cmake"}
SHELL_SUFFIXES = {".sh", ".bash", ".zsh", ".bat", ".cmd", ".ps1"}
SPECIAL_DOC_NAMES = set()
SHELL_NAMES = {"Dockerfile", "Containerfile"}

# the explicit allowlist (words that stay though a script or a config spells them) lives in the shared table
ALLOW = T.ALLOW

KEYWORD = re.compile(r"[(,{][\s]*\Z")                                    # kwarg / object-key position
ENV_ACCESS = re.compile(r"[\w$]*(?:env|environ)\??\.\Z", re.I)           # process.env.X, env?.X, baseEnv.X, environ.X
ENV_NAME_CTX = re.compile(r"(?i)(?:get|set|put|unset)env\w*\(|\w*env\w*\(|environ|\benv\b|\bENV\b|export [A-Z_]*$")
SHELL_ASSIGN = re.compile(r"(?:^|[\s;&(${" + '"' + r"`])\Z")             # NAME= in a command position
SHELL_READ = re.compile(r"[%$]|env:$|\\$\{?\Z|defined\s+\Z|\benv\s+\Z", re.I)


def family_of(p: Path):
    if p.name == "CMakeLists.txt":
        return "cmake"
    if p.name in SHELL_NAMES:
        return "shell"
    ext = p.suffix.lower()
    if ext in CMAKE_SUFFIXES:
        return "cmake"
    if ext in PY_SUFFIXES:
        return "py"
    if ext in JS_SUFFIXES:
        return "js"
    if ext in CPP_SUFFIXES:
        return "cpp"
    if ext in SHELL_SUFFIXES:
        return "shell"
    if ext in CONF_SUFFIXES:
        return "conf"
    if ext in DOC_SUFFIXES:
        return "doc"
    return None


def code_regions(text, fam):
    """(string spans, comment spans) of a code file; (None, comment-only spans) for conf (yaml comments)."""
    string, comment = [], []
    i, n = 0, len(text)
    hash_c = fam in ("py", "cmake", "conf")
    while i < n:
        c = text[i]
        two = text[i:i + 2]
        if fam in ("py", "js") and text[i:i + 3] in ('"""', "'''"):
            q = text[i:i + 3]
            j = text.find(q, i + 3)
            j = n if j < 0 else j + 3
            string.append((i, j)); i = j
        elif two == "//" and fam in ("cpp", "py", "js", "cmake"):
            j = text.find("\n", i); j = n if j < 0 else j
            comment.append((i, j)); i = j
        elif two == "/*" and fam in ("cpp", "py", "js", "cmake"):
            j = text.find("*/", i + 2); j = n if j < 0 else j + 2
            comment.append((i, j)); i = j
        elif two == "--" and fam == "cmake":
            j = text.find("\n", i); j = n if j < 0 else j
            comment.append((i, j)); i = j
        elif c == "#" and hash_c and (fam != "conf" or (i == 0 or text[i - 1] in " \t\n")):
            j = text.find("\n", i); j = n if j < 0 else j
            comment.append((i, j)); i = j
        elif c in "\"'" and fam in ("cpp", "py", "js", "cmake"):
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2; continue
                if text[j] == c:
                    j += 1; break
                if text[j] == "\n":
                    break
                j += 1
            string.append((i, j)); i = j
        elif c == "`" and fam == "js":
            j = text.find("`", i + 1); j = n if j < 0 else j + 1
            string.append((i, j)); i = j
        else:
            i += 1
    return string, comment


def marks(spans, n):
    m = bytearray(n)
    for a, b in spans:
        m[a:b] = b"\1" * (b - a)
    return m


def is_build_flag(text, s):
    return text[max(0, s - 2):s] in ("-D", "/D", "/d")


def line_before(text, s):
    return text[text.rfind("\n", 0, s) + 1:s]


def cmake_env(text, s):
    lb = line_before(text, s)
    return (bool(re.search(r"\bENVIRONMENT\b", lb)) or bool(re.search(r"-E[ \t]+env\b", lb))
            or text[max(0, s - 6):s].endswith("ENV{"))


def string_is_name(text, s, e):
    """a string occurrence the code uses as a NAME (not message prose): in an env-access context, or quoted as a key."""
    if ENV_NAME_CTX.search(text[max(0, s - 60):s]):
        return True
    before_q, after = text[s - 1:s], text[e:e + 3]
    return before_q in "\"'" and (after[0] == '"' and after[1:3] == '":' or re.match(r"['\"]\s*[,:}\]]", after)
                                  or after[:2] == '":' or after[:2] == "',")


def env_position(text, fam, s, e):
    """a bare code occurrence in a name position: env.X / kwarg / object key / regex literal (js)."""
    lb = text[max(0, s - 400):s]
    if ENV_ACCESS.search(lb):
        return True
    if fam in ("py", "js") and KEYWORD.search(lb) and re.match(r"\s*[=:]", text[e:]):
        return True
    if fam == "js" and text[s - 1:s] in (".", "/"):
        return True
    return False


def shell_env_position(text, s, e):
    lb = text[max(0, s - 40):s]
    if SHELL_ASSIGN.search(lb) and re.match(r"\s*=(?!=)", text[e:]):
        return True                                    # `NAME=value ...` / `set "NAME=..."` / env NAME=
    if SHELL_READ.search(lb) and not re.match(r"\s*=", text[e:]):
        return True                                    # $NAME, ${NAME}, %NAME%, $env:NAME, if defined NAME
    return False


def regs_for(fam, text):
    if fam in ("doc", "shell"):
        return None
    st, cm = code_regions(text, fam)
    return marks(st, len(text)), marks(cm, len(text))


def strong_use(text, fam, m, reg):
    """an occurrence that PROVES the name is a setting (decides messages, comments, docs and shell prose)."""
    s, e = m.start(), m.end()
    if m.group(0) in ALLOW or is_build_flag(text, s):
        return False
    if fam == "conf":
        return not (reg and reg[1][s])                 # a json/yaml key (not inside a #comment)
    if fam == "doc":
        return False
    if fam == "shell":
        return shell_env_position(text, s, e)
    if fam == "cmake":
        return cmake_env(text, s)
    st, cm = reg
    if st[s]:
        return string_is_name(text, s, e)
    if cm[s]:
        return False
    return env_position(text, fam, s, e)


def decide(text, fam, m, reg, sufs):
    """(rename?, where) for one occurrence.  sufs = the suffixes proved settings: a mention of either old name of a
    suffix (the alias code made them ONE setting) renames once the suffix is known to be a setting."""
    s, e = m.start(), m.end()
    if m.group(0) in ALLOW:
        return False, "allowlist"
    if is_build_flag(text, s):
        return False, "build flag"
    suf = m.group(2)
    known = suf in sufs or (re.fullmatch(r"[A-Z0-9_]*\d{1,3}", suf) and re.sub(r"\d{1,3}$", "", suf) in sufs)
    if fam == "conf":
        if reg is None or not reg[1][s]:
            return True, "setting"                     # a json key, or a yaml value on a non-comment line
        return (True, "setting") if known else (False, "comment of a word")
    if fam == "doc":
        return (True, "setting") if known else (False, "doc word")
    if fam == "shell":
        if known or shell_env_position(text, s, e):
            return True, "setting"
        return False, "script word"
    if fam == "cmake":
        return (True, "setting") if cmake_env(text, s) else (False, "cmake word")
    st, cm = reg
    if st[s]:
        if string_is_name(text, s, e) or known:
            return True, "setting"
        return False, "message of a word"
    if cm[s]:
        return (True, "setting") if known else (False, "comment of a word")
    if env_position(text, fam, s, e):
        return True, "setting"
    return False, "identifier"


def load(p: Path):
    try:
        raw = p.read_bytes()
    except OSError:
        return None
    if b"\0" in raw[:8000]:
        return None
    try:
        return raw.decode("utf-8-sig"), raw.startswith(b"\xef\xbb\xbf")
    except UnicodeDecodeError:
        return None


def collect(paths):
    out = {}
    for root in paths:
        cand = [root] if root.is_file() else []
        if not cand:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in T.SKIP_DIR_NAMES and not T.SKIP_DIR_RE.match(d)]
                for fn in filenames:
                    cand.append(Path(dirpath) / fn)
        for p in cand:
            try:
                rel = p.relative_to(ROOT).as_posix()
            except ValueError:
                sys.exit(f"{p}: outside the worktree {ROOT}")
            if T.skipped(rel):
                continue
            fam = family_of(p)
            if fam is None:
                continue
            got = load(p)
            if got is None:
                continue
            out[rel] = (p, fam, got[0], got[1])
    return out


BARE_NAME = re.compile(r"([\x27\x22])([A-Z][A-Z0-9_]{2,})\1")
ENV_DIRECT = re.compile(r"(?i)(?:\w*env\w*|environ\.get|environ|getenv\w*|setenv|putenv_s|--env|ENVIRONMENT)[\s]*[\(\[]?\s*$")


def plan(files: dict):
    """the whole decision pass over {rel: (path, family, text, bom)} (collect()'s shape, or a caller's own
    in-memory texts): returns (renames, collisions, allow_hits, kept_words, names, merged, both).  Shared with
    tools/saguaro/rename_upstream.py, which feeds it the tree after the internal rename."""
    strong, existing, regs = set(), {}, {}
    for rel, (p, fam, text, bom) in files.items():
        reg = regs[rel] = regs_for(fam, text)
        for m in NAME.finditer(text):
            if strong_use(text, fam, m, reg):
                strong.add(m.group(0))
        for m in HAS_SAGUARO.finditer(text):
            existing.setdefault(m.group(1), set()).add(rel)
    sufs = {suf for nm in strong for suf in [nm.split("_", 1)[1]]}
    for rel, (p, fam, text, bom) in files.items():        # a fallback-read name: env("DS4_X") means SAGUARO_DS4_X
        if fam in ("doc", "shell", "conf", "cmake"):
            continue
        reg = regs[rel]
        for m in BARE_NAME.finditer(text):
            s, e = m.start(), m.end()
            if reg is not None and reg[0][s] and ENV_DIRECT.search(text[max(0, s - 30):s]):
                sufs.add(m.group(2))          # only a string an env call READS directly: ds4_env("DS4_X"), not ROOT / "VERSION"
    occ = {rel: [(m.start(), m.end(), m.group(0), m.group(2),
                  *decide(text, fam, m, regs[rel], sufs)) for m in NAME.finditer(text)]
           for rel, (p, fam, text, bom) in files.items()}

    renames, kept_words, allow_hits, collisions = {}, {}, {}, {}
    for rel, oc in occ.items():
        for (s, e, old, suf, ok, where) in oc:
            if ok and suf in existing:
                collisions.setdefault(old, set()).add(rel)
                continue
            if ok:
                renames.setdefault(rel, []).append((s, e, old, suf))
            elif where == "allowlist":
                allow_hits.setdefault(old, set()).add(rel)
            else:
                kept_words.setdefault(old, set()).add(where)
    names = {o[2] for oc in renames.values() for o in oc}
    both = {}
    for oc in occ.values():
        for (s, e, old, suf, ok, where) in oc:
            both.setdefault(suf, set()).add(old.split("_")[0])
    merged = sorted(s for s, ps in both.items() if len(ps) == 2 and ("MAYA_" + s in strong or "STRATA_" + s in strong))
    return renames, collisions, allow_hits, kept_words, names, merged, both


def apply(text: str, oc: list) -> str:
    """rewrite one file's text at the spans plan() found: OLD_X -> SAGUARO_<suffix>"""
    out, last = [], 0
    for (s, e, old, suf) in oc:
        out.append(text[last:s]); out.append(NEW + suf); last = e
    out.append(text[last:])
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="change nothing; exit 1 if a renameable setting name is left")
    ap.add_argument("paths", nargs="*", default=None)
    a = ap.parse_args()
    files = collect([Path(x).resolve() for x in (a.paths or [ROOT])])
    renames, collisions, allow_hits, kept_words, names, merged, both = plan(files)

    print(f"rename_env: {sum(len(v) for v in renames.values())} uses of {len(names)} settings -> SAGUARO_* in "
          f"{len(renames)} files")
    print(f"rename_env: {len(both)} distinct old-style names seen; {len(merged)} suffixes where BOTH old prefixes "
          f"appeared (one setting behind the alias code: merged, not collisions)")
    if collisions:
        print("COLLISIONS - left as they are, a SAGUARO_* of the same name already exists:")
        for old in sorted(collisions):
            print(f"  {old}: {', '.join(sorted(collisions[old])[:4])} ...")
    if allow_hits:
        print(f"skipped by the explicit allowlist ({len(allow_hits)}):")
        for n in sorted(allow_hits):
            print(f"  {n}")
    print(f"kept as words / build identifiers / docs ({len(kept_words)} names):")
    for n in sorted(kept_words):
        print(f"  {n}  [{', '.join(sorted(kept_words[n]))}]")

    if a.check:
        if renames:
            print(f"CHECK FAILED: {sum(len(v) for v in renames.values())} renameable uses left in {len(renames)} "
                  f"files, e.g.:")
            for rel in sorted(renames)[:12]:
                print(f"  {rel}: {len(renames[rel])}")
            return 1
        print("CHECK OK: no renameable setting name left (words, allowlist and excluded paths aside)")
        return 0
    written = 0
    for rel, oc in renames.items():
        p, fam, text, bom = files[rel]
        p.write_bytes((b"\xef\xbb\xbf" if bom else b"") + apply(text, oc).encode("utf-8"))
        written += 1
    print(f"applied: rewrote {written} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
