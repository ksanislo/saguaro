#!/usr/bin/env python3
"""tools/saguaro/rename_rules.py - the rename layer's rule table in ONE place (docs/UPSTREAM.md section 2).

The four rename tools read their patterns from here and hold no duplicated rules:
  rename_env.py             the settings family (MAYA_X / STRATA_X -> SAGUARO_X, context proved)
  rename_internal_maya.py   the internal family (upstream v1.1.75's strata -> maya, kept as ours)
  rename_log_prefixes.py    the engine log prefixes ("strata part:" / "ds4:" -> "saguaro ...")
  rename_upstream.py        ALL of them plus the name families below, as one command

Name families (the hand renames the M1 intake needed, scratch/notes.md section B):
  NAMES_RULES      the program's and the launcher's names in text: maya.py / maya.sh -> saguaro.py / saguaro.sh,
                   START-MAYA.bat, maya-nightly*, the saguaro-* temp-dir prefixes, `maya --speed-calibration`
  BRAND_RULES      brand words in user-visible docs: "restart Maya" / "Maya checks ..." -> Saguaro; the credits,
                   "Project Maya", "Maya v1.1.x" and the Maya quant names (Maya-S/M/L/XS/S24) stay
  INTERNAL_PATHS   the path renames the families assume (include/strata -> include/maya, maya.py -> saguaro.py ...)

Words are split in the definitions so scanning this folder finds no complete old name to rename (the same trick
the three scripts use).  Everything here is pure data plus the two small sub-functions the internal family needs;
idempotence holds because no rule's replacement is ever a rule's match.
"""
from __future__ import annotations

import re
from pathlib import Path

# ---- the settings family (tools/saguaro/rename_env.py) ---------------------------------------------------------

NEW = "SA" + "GUARO_"                                    # our one setting prefix
OLD_ENV = ("MA" + "YA", "ST" + "RATA")                   # the two old prefixes (one setting behind the alias code)
NAME = re.compile(r"\b(" + "|".join(OLD_ENV) + r")_([A-Z0-9_]+)\b")
HAS_SAGUARO = re.compile(r"\b" + "SAGUARO" + r"_([A-Z0-9_]+)\b")

# not settings, and not recognised as words automatically because a script or a config spells them anyway (a usage
# example's placeholder for the metal engine's binary, and the hip build scripts' CMake variable / C++ macro names)
ALLOW = {"MA" + "YA_METAL", "MAYA_USE_HIP", "STRATA_USE_HIP", "STRATA_HIP_ARCHS", "MAYA_HIP_ARCHS",
         "STRATA_VISION_HIP_ARCHS", "MAYA_VISION_HIP_ARCHS"}

# ---- the internal family (tools/saguaro/rename_internal_maya.py) ------------------------------------------------

INT_OLD, INT_NEW = "st" + "rata", "ma" + "ya"          # upstream v1.1.75's own internal rename, kept as ours
UP = INT_OLD.upper()

# lowercase words that are code names upstream renamed (cmake targets, a few helpers); others (the Strata pack
# format's strata_fmt / strata_pack / strata_inkling, libstrata_kq) stay
INT_CODE_WORDS = {"mmq", "cpu_pause", "store_fence", "engine", "mmq_pool_slab", "dp4a_sm60", "q2_kernel",
                  "kernels_cpu", "kernels", "hipblas_set_workspace_noop", "profile_version_test",
                  "profile_save_test", "ik_oracle", "ik_oracle_help", "glm_maya_sync_test", "gguf_reader_test",
                  "ds4_ablate", "ds4_ablate_junk", "mm3_ablate_missing", "warnings", "prefill", "platform",
                  "core", "spec", "program", "artifact", "ngram", "plan"}
INT_KEEP_LOWER = re.compile(r"^(fmt|pack|inkling|kq|kq_tab|fmt_test|fmt_gpu_test|ref|tokenizer)\b")

R_INC = re.compile(r"\binclude/" + INT_OLD + r"(?=[/\s\"'`)\]>;:,]|$)", re.M)
R_DIR = re.compile(r"(?<![\w.-])(?<!tools/)(?<!docs/)" + INT_OLD +
                   r"/(?=(core|kernels|spec|platform|artifact|prefill|program|plan|ngram|hip_compat)/)")
R_HDR = re.compile(r"([\"<])" + INT_OLD + r"/")
R_PYP = re.compile(r"([\"'])include\1 / ([\"'])" + INT_OLD + r"\2")   # Python's "include" / "strata" path pieces
R_NS = re.compile(r"\b" + INT_OLD + r"(?=::)")
R_NSD = re.compile(r"\bnamespace " + INT_OLD + r"\b")
R_UP = re.compile(r"(?:\b|(?<=-D)|(?<=/D))" + UP + r"_([A-Z0-9][A-Z0-9_]*)\b")
R_LOW = re.compile(r"(?<![/\w.])(_?)" + INT_OLD + r"_(\w+)")

INT_TEXT_SUFFIXES = {".c", ".h", ".hpp", ".cpp", ".cc", ".cu", ".cuh", ".hip", ".inc", ".inl", ".metal", ".mm",
                     ".py", ".sh", ".ps1", ".bat", ".cmd", ".cmake", ".txt", ".md", ".yml", ".yaml", ".json", ".in", ""}


def is_cmake(p: Path) -> bool:
    return p.suffix == ".cmake" or p.name == "CMakeLists.txt"


def int_low_sub(cmake: bool):
    def f(m):
        pre, rest = m.group(1), m.group(2)
        if INT_KEEP_LOWER.match(rest):
            return m.group(0)
        if cmake or pre or rest in INT_CODE_WORDS or rest.split("_")[0] in ("ds4", "mm3", "glm", "ik") and "ablate" in rest:
            return pre + INT_NEW + "_" + rest
        if rest.startswith(("ds4_ablate", "mm3_ablate", "glm_dsa_loader")):
            return pre + INT_NEW + "_" + rest
        return m.group(0)
    return f


def int_up_sub(line: str):
    """STRATA_X -> MAYA_X, but not where the line also names MAYA_X (the older spelling beside the new)"""
    def f(m):
        if "MAYA_" + m.group(1) in line:
            return m.group(0)
        return "MAYA_" + m.group(1)
    return f


def qfn_block(text: str):
    """the QFN sub-builds' lines of CMakeLists.txt: engines/qfn's own CMake project keeps its STRATA_ options, so
    its -D arguments and option names there stay"""
    a = text.find("if(" + NEW + "QFN AND")
    if a < 0:
        return None
    a = text.find("\n", a)
    b = text.rfind("add_custom_target(saguaro-qfn")
    b = b if b > a else -1
    return (a, b) if b > 0 else None


def rename_internal(text: str, path: Path) -> str:
    cm = is_cmake(path)
    keep = qfn_block(text) if cm else None
    head, mid, tail = (text[:keep[0]], text[keep[0]:keep[1]], text[keep[1]:]) if keep else (text, "", "")

    def one(t, up=True):
        t = R_INC.sub("include/" + INT_NEW, t)
        t = R_HDR.sub(lambda m: m.group(1) + INT_NEW + "/", t)
        t = R_PYP.sub(lambda m: f"{m.group(1)}include{m.group(1)} / {m.group(2)}{INT_NEW}{m.group(2)}", t)
        t = R_NS.sub(INT_NEW, t)
        t = R_NSD.sub("namespace " + INT_NEW, t)
        t = R_DIR.sub(INT_NEW + "/", t)
        if up:
            t = "\n".join(R_UP.sub(int_up_sub(ln), ln) for ln in t.split("\n"))
        t = R_LOW.sub(int_low_sub(cm), t)
        return t

    return one(head) + one(mid, up=False) + one(tail) if keep else one(text)


# ---- the engine log prefixes (tools/saguaro/rename_log_prefixes.py) ---------------------------------------------

LOG_NEW = "sa" + "guaro"
LOG_OLD = ("st" + "rata", "ma" + "ya")                  # the two old engine names
LOG_NAME = "(?:" + "|".join(LOG_OLD) + ")"               # not strata_x / strata-x as a whole word
_P = r"[a-z0-9][a-z0-9 ._()#-]*"                         # the part after the engine name, up to the colon or the (
LOG_RULES = [
    (re.compile(r'(["\'])(' + LOG_NAME + r')(?![A-Za-z0-9_-])(?=: )'), r"\1" + LOG_NEW),                       # "strata: x"
    (re.compile(r'(["\'])(' + LOG_NAME + r')( ' + _P + r')(?=: )'), r"\1" + LOG_NEW + r"\3"),                  # "strata part: x"
    (re.compile(r'(["\'])(' + LOG_NAME + r')( ' + _P + r')(?= \()'), r"\1" + LOG_NEW + r"\3"),                 # "strata part (x"
    (re.compile(r'(["\'])(' + LOG_NAME + r')-([a-z0-9]+)(?=(?:: | [A-Za-z(]))'), r"\1" + LOG_NEW + r" \3"),    # "maya-ds4: x"
    (re.compile(r'(["\'])(ds4)(?=[: ])'), r"\1" + LOG_NEW + r" ds4"),                                   # "ds4: x", "ds4 x"
    (re.compile(r'(["\'])(' + LOG_NAME + r') (batch GPU stages|decode GPU stages|KV prefetch|DMA|DBG|Q2_0)'),
     r"\1" + LOG_NEW + r" \3"),                                                                           # "strata GPU/DBG ... :"
    (re.compile(r'(["\'])(' + LOG_NAME + r')(?=" \+)'), r"\1" + LOG_NEW),                             # "strata " + label
    (re.compile(r'(["\'])strata-dsv41(?= %s build)'), r"\1" + LOG_NEW + " dsv41"),                     # the --version line
]
# In files passed explicitly (a test simulating an engine's log, a docstring showing its lines), a line inside a
# multi-line string starts with the prefix too: the same rules with a line anchor instead of a quote.
LOG_LINE_RULES = [
    (re.compile(r"^(" + LOG_NAME + r")(?![A-Za-z0-9_-])(?=: )", re.M), LOG_NEW),
    (re.compile(r"^(" + LOG_NAME + r")( [a-z0-9][a-z0-9 ._()#-]*)(?=: )", re.M), LOG_NEW + r"\2"),
    (re.compile(r"^(" + LOG_NAME + r")( [a-z0-9][a-z0-9 ._()#-]*)(?= \()", re.M), LOG_NEW + r"\2"),
    (re.compile(r"^(" + LOG_NAME + r")-([a-z0-9]+)(?=(?:: | [A-Za-z(]))", re.M), LOG_NEW + r" \2"),
    (re.compile(r"^(ds4)(?=[: ])", re.M), LOG_NEW + r" ds4"),
    (re.compile(r"^(" + LOG_NAME + r")( (?:batch GPU stages|decode GPU stages|KV prefetch|DMA|DBG|Q2_0))", re.M),
     LOG_NEW + r"\2")]
LOG_CPP_SUFFIXES = {".c", ".h", ".hpp", ".cpp", ".cc", ".cxx", ".cu", ".cuh"}
# the whole-tree scope of the prefix rules: the engines' sources (rename_log_prefixes.py's default paths;
# engines/qfn's are skipped here as everywhere else); tests' own banners are not engine logs and are not swept
LOG_SRC_PATHS = ("src/", "tools/vision/")


# ---- the program's and the launcher's names (the M1 hand renames: b2e85aa52, ef27c2650, fe22cd559, a4a45bd71) ------

PROG_NEW = "sa" + "guaro"
BRAND_OLD, BRAND_NEW = "Ma" + "ya", "Sa" + "guaro"

# text files (not json/yaml data: the settings family already owns those keys, and the maya-*.json config file
# names stay): doc / py / js / cpp / cmake / shell
NAMES_FAM = ("doc", "py", "js", "cpp", "cmake", "shell")
# the program and launcher names as written in text: `maya.py`, `./maya.sh` (not `serve-maya.sh`, not `maya.py`
# inside a word), the Windows launcher, the nightly run's files, the saguaro-* temp-dir prefixes (upstream:
# PICTURE_DIR_PREFIXES ("maya-vision-", "maya-ds4-pictures-", ...) and a test's "maya-vision-http-"), and the
# server's own log tag.  The CMake target names (maya, maya-ds4, maya-vision, maya_kernels, maya_mmq) and the
# maya-*.json config file names are NOT matched: they stay as they are.
NAMES_RULES = [
    (re.compile(r"(?<![\w-])maya\.py\b"), PROG_NEW + ".py"),
    (re.compile(r"(?<![\w-])maya\.sh\b"), PROG_NEW + ".sh"),
    (re.compile("START-" + BRAND_OLD.upper() + r"\.bat"), "START-SAGUARO.bat"),
    (re.compile(r"(?<![\w-])maya-nightly\b"), PROG_NEW + "-nightly"),
    (re.compile(r"(?<![\w-])maya-vision-"), PROG_NEW + "-vision-"),
    (re.compile(r"(?<![\w-])maya-ds4-pictures-"), PROG_NEW + "-ds4-pictures-"),
    (re.compile(r"(?<![\w-])maya-hy4-pictures-"), PROG_NEW + "-hy4-pictures-"),
    (re.compile(r"(?<![\w-])maya-mm3-pictures-"), PROG_NEW + "-mm3-pictures-"),
]
# the server's log tag: main has no `[maya]` in any .py / .js / .md (ef27c2650: the merge's two new lines are
# the case); only in those text families, where a bracketed lowercase word is always OUR print prefix
LOGTAG_RULES = [(re.compile(r"\[ma" + r"ya\]"), "[" + PROG_NEW + "]")]
LOGTAG_FAM = ("py", "js", "doc")
# `--speed-calibration` is the launcher's switch, not the engine's (a4a45bd71): "maya --speed-calibration" in a
# comment is the program as it is run, so it says `saguaro.py --speed-calibration`
CMD_RULES = [(re.compile(r"(?<![\w./-])ma" + r"ya(?= --speed-calibration)"), PROG_NEW + ".py")]


# ---- brand words in user-visible docs (2d4246a26; the shape of the whole-project rebrand 289431642) ----------------

BRAND_FAM = ("doc",)
BRAND_RULES = [
    (re.compile(r"\b(restart|start|stop|launch|install|update|upgrade) " + BRAND_OLD + r"\b(?![-\w])(?! v?\d)"),
     r"\1 " + BRAND_NEW),
    (re.compile(r"(?<!Project )\b" + BRAND_OLD + r"\b(?= (checks|will|needs|says|supports|writes|refuses|stores|keeps)\b)"),
     BRAND_NEW),
]


# ---- the paths the tool's rules assume (rename_upstream.py --paths prints these for `git mv`) ----------------------

INTERNAL_PATHS = [("include/" + INT_OLD, "include/" + INT_NEW)]
NAME_PATHS = [
    ("maya.py", PROG_NEW + ".py"),
    ("maya.sh", PROG_NEW + ".sh"),
    ("START-" + BRAND_OLD.upper() + ".bat", "START-SAGUARO.bat"),
    ("AMD_" + BRAND_OLD.upper() + ".md", "AMD_SAGUARO.md"),
]
NAME_PATH_PREFIXES = [(BRAND_OLD.upper() + "_", "SAGUARO_")]        # docs named MAYA_* go by their SAGUARO_* name


# ---- exceptions: names and files that stay (docs/UPSTREAM.md section 2's "explicit exceptions list") ---------------

# record texts quoting upstream's commands and targets verbatim (credit and history): never rewritten
EXC_PATHS = ("docs/MI50_VALIDATION.md", "docs/DGX_SPARK.md", "docs/triage/")
# stale old-name mentions the rename pass must not rewrite by itself, kept and REPORTED by --check:
#   engines/qfn's own names: the -D options of its CMake project (it keeps STRATA_), their docs and tests
#   tools/test_maya_hip.py    a comment naming the launcher as ./maya.sh (main restarts through ./saguaro.sh)
#   src/hy4/hy4_main.cpp      two comments saying "maya --speed-calibration" (a4a45bd71 fixed glm_fast_path.cu only)
#   saguaro.py                "start Maya again" in three user strings (two tests assert the wording)
EXC_TEXT = [
    ("saguaro.py", re.compile(UP + r"_[A-Z0-9_]+")),
    ("tools/qfn_fit.py", re.compile(UP + r"_[A-Z0-9_]+")),
    ("tools/test_setup_qfn.py", re.compile(UP + r"_[A-Z0-9_]+")),
    ("docs/SETTINGS.md", re.compile(UP + r"_MMQ_TAB")),
    ("tools/test_maya_hip.py", re.compile(r"\./maya\.sh\b")),
    ("src/hy4/hy4_model.cu", re.compile(r"(?<![\w-])maya\.py")),
    ("src/hy4/hy4_main.cpp", re.compile(r"(?<![\w./-])ma" + r"ya(?= --speed-calibration)")),
    ("saguaro.py", re.compile(r"start " + BRAND_OLD + r" again")),
]


def exc_path(rel: str) -> bool:
    """rel is one of the record texts EXC_PATHS (never rewritten by any family's in-tree pass)"""
    return rel.startswith(EXC_PATHS)


def mask_exceptions(text: str, rel: str):
    """hide one file's EXC_TEXT occurrences from the rules: (masked text, the hidden strings) - the shared
    masking rename_upstream.py and rename_internal_maya.py both wrap their internal-family run with"""
    spans = []
    for exc_rel, rx in EXC_TEXT:
        if exc_rel == rel:
            spans += [text[m.start():m.end()] for m in rx.finditer(text)]
    for i, s in enumerate(spans):
        text = text.replace(s, f"\x00EXC{i}\x00")
    return text, spans


def unmask_exceptions(text: str, spans: list) -> str:
    """put the strings mask_exceptions hid back"""
    for i, s in enumerate(spans):
        text = text.replace(f"\x00EXC{i}\x00", s)
    return text


# every family skips these (history, vendored, generated, and this folder, which must spell the old names):
SKIP_DIR_NAMES = {".git", "third_party", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".scons"}
SKIP_DIR_RE = re.compile(r"^(build[^/\\]*|.*-build|cmake-build.*)$", re.I)
SKIP_PATHS = ("ref-configs/", "bench/results/", "engines/qfn/bench/results/", "tools/saguaro/", "docs/triage/",
              "ref/", "scratch/", "tools/test_rename_upstream.py",          # the tool's own tests and its manual
              "docs/UPSTREAM.md")                                           # must spell the old names
SKIP_NAMES = {"CHANGELOG.md", "NOTICE.md", "DIVERGENCE.md", "UPSTREAM.md"}   # (the register and the intake guide name both sides)
SKIP_NAME_RE = re.compile(r"^(PLAN|REPORT|NEXT)-.*\.md$|^LICENSE(\.|$)")
SKIP_SUFFIXES = {".log", ".jsonl", ".patch", ".orig"}

# the internal family stops before QFN's own tree (its CMake project keeps STRATA_, serve/qfn_engine.py is the
# extracted copy); the settings and log-prefix families still sweep engines/qfn, as rename_env.py and
# rename_log_prefixes.py always did
INTERNAL_SKIP_PATHS = ("engines/qfn/", "tools/qfn/", "serve/qfn_engine.py")


def skipped(rel: str) -> bool:
    """rel: a root-relative posix path - True when no rename family may touch it"""
    if rel.startswith(SKIP_PATHS):
        return True
    *dirs, name = rel.split("/")
    for d in dirs:
        if d in SKIP_DIR_NAMES or SKIP_DIR_RE.match(d):
            return True
    if name in SKIP_NAMES or SKIP_NAME_RE.match(name):
        return True
    return any(name.lower().endswith(s) for s in SKIP_SUFFIXES)


def internal_skipped(rel: str) -> bool:
    """skipped(rel), plus what only the internal family stops before: QFN's own tree and the EXC_PATHS records"""
    return skipped(rel) or rel.startswith(INTERNAL_SKIP_PATHS) or exc_path(rel)
