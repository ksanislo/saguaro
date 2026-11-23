"""tools/check_hip_compat.py - every CUDA runtime / cuBLAS name the AMD (HIP) build compiles must have a mapping in
include/maya/hip_compat/*.h.  The HIP build compiles the same CUDA-shaped sources with those headers forced in
(cmake/hip_backend.cmake), so a CUDA call added without its mapping breaks every AMD build - v1.0.21 did, with three
device queries from #44 (fixed in v1.0.22 by #52).  No ROCm or GPU is needed: it reads the sources.

    python tools/check_hip_compat.py [repo]      # exit 1 and the file:line of each unmapped name

Names inside CUDA-only code (#ifndef MAYA_USE_HIP, #if !defined(MAYA_USE_HIP), the #else of a MAYA_USE_HIP
block), comments and string literals are not counted.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

NAME = re.compile(r"\b(cuda[A-Z]\w*|cublas[A-Z]\w*|CUBLAS_[A-Z0-9_]+|CUDA_R_\w+)\b")
SOURCES = ("*.cu", "*.cuh", "*.cpp", "*.hpp", "*.h")


def mapped_names(compat: Path) -> set:
    """What the compat headers define: #define NAME, and the inline functions/templates named like CUDA's."""
    out = set()
    for h in compat.glob("*.h"):
        text = h.read_text(encoding="utf-8", errors="replace")
        out |= set(re.findall(r"^\s*#\s*define\s+(\w+)", text, re.M))
        out |= {m for m in re.findall(r"\b(cuda\w+|cublas\w+)\s*\(", text) if f"{m}(" in text}
    return out


def strip_comments_and_strings(text: str) -> str:
    """The source with comments and string/char literals blanked (newlines kept, so line numbers stay)."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * text.count("\n", i, j))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(" ")
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def kind(cond: str) -> str:
    """'hip' when the condition holds only in the HIP build, 'cuda' when only in CUDA's, else 'other'."""
    c = cond.replace(" ", "")
    if "MAYA_USE_HIP" not in c or "||" in c:
        return "other"
    if "!defined(MAYA_USE_HIP)" in c or "!MAYA_USE_HIP" in c:
        return "cuda"
    return "hip"


def hip_lines(text: str):
    """(line number, code) for each line the HIP build compiles, as far as MAYA_USE_HIP decides."""
    stack = []          # per open #if: [kind of the condition, in its #else branch]
    for no, line in enumerate(text.split("\n"), 1):
        s = line.strip()
        m = re.match(r"#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)", s)
        if m:
            d, rest = m.group(1), m.group(2)
            if d == "ifdef":
                stack.append(["hip" if rest.strip() == "MAYA_USE_HIP" else "other", False])
            elif d == "ifndef":
                stack.append(["cuda" if rest.strip() == "MAYA_USE_HIP" else "other", False])
            elif d == "if":
                stack.append([kind(rest), False])
            elif d == "elif" and stack:
                stack[-1] = ["other", False]        # (an #elif chain: counted as compiled)
            elif d == "else" and stack:
                stack[-1][1] = True
            elif d == "endif" and stack:
                stack.pop()
            continue
        skipped = any((k == "cuda" and not in_else) or (k == "hip" and in_else) for k, in_else in stack)
        if not skipped:
            yield no, line


def unmapped(repo: Path) -> list:
    """[(file, line, name)] for every name the HIP build compiles without a mapping."""
    compat = repo / "include" / "strata" / "hip_compat"
    have = mapped_names(compat)
    found = []
    for top in ("src", "include", "tests"):
        for pat in SOURCES:
            for f in sorted((repo / top).rglob(pat)):
                if compat in f.parents:
                    continue
                text = strip_comments_and_strings(f.read_text(encoding="utf-8", errors="replace"))
                for no, line in hip_lines(text):
                    for name in NAME.findall(line):
                        if name not in have:
                            found.append((str(f.relative_to(repo)), no, name))
    return found


# NVIDIA's inline assembly: a PTX mnemonic with its type suffix, or a PTX register name.  AMD's own (v_, s_, ds_
# mnemonics, DPP controls) has no such dotted instruction and stays allowed.
PTX = re.compile(r"\b(mma\.sync|ldmatrix|cp\.async|mov\.[usb]\d|ld\.(global|shared|volatile)|st\.(global|shared)|"
                 r"prmt\.|lop3\.|shf\.|cvt\.|red\.|atom\.|fence\.|membar|bar\.sync|setp\.|selp\.|"
                 r"%%(globaltimer|clock|laneid|smid))")
ASM = re.compile(r"\b(asm|__asm__)\s*(volatile\s*|__volatile__\s*)?\(")
# CUDA headers the HIP build must find (the compat folder is on its include path, ROCm provides <mma.h>-free code)
HEADER = re.compile(r"#\s*include\s*<((cuda|cublas|cufft|curand|cusparse|nvtx|mma)[\w./]*)>")


def raw_asm_text(lines: list, no: int) -> str:
    """the raw text of the asm statement starting on line `no` (1-based), up to its closing ');'"""
    out = []
    for k in range(no - 1, min(len(lines), no + 12)):
        out.append(lines[k])
        if ");" in lines[k]:
            break
    return "\n".join(out)


# what an AMD device pass defines (gfx1201: RDNA 4; gfx1100: RDNA 3 - both checked)
HIP_DEVICE = {"gfx1201": {"MAYA_USE_HIP": "1", "__HIPCC__": "1", "__HIP_DEVICE_COMPILE__": "1", "__HIP__": "1",
                          "__GFX12__": "1", "__gfx1201__": "1", "__AMDGCN__": "1"},
              "gfx1100": {"MAYA_USE_HIP": "1", "__HIPCC__": "1", "__HIP_DEVICE_COMPILE__": "1", "__HIP__": "1",
                          "__GFX11__": "1", "__gfx1100__": "1", "__AMDGCN__": "1"}}


def _eval_if(expr: str, macros: dict) -> bool:
    """a #if expression under `macros` (object-like defines; an unknown name is 0, as the preprocessor has it)"""
    e = re.sub(r"defined\s*\(\s*(\w+)\s*\)|defined\s+(\w+)",
               lambda m: "1" if (m.group(1) or m.group(2)) in macros else "0", expr)
    e = re.sub(r"\b([A-Za-z_]\w*)\b", lambda m: macros.get(m.group(1), "0") if m.group(1) not in ("and", "or", "not")
               else m.group(1), e)
    e = e.replace("&&", " and ").replace("||", " or ")
    e = re.sub(r"!(?!=)", " not ", e)
    e = re.sub(r"(\d+)[uUlL]+\b", r"\1", e)
    try:
        return bool(eval(e, {"__builtins__": {}}, {}))
    except Exception:  # noqa: BLE001 - an expression this does not read: counted as compiled
        return True


def device_lines(text: str, defs: dict):
    """(line number, code) for each line an AMD device pass compiles: #if / #ifdef / #elif evaluated, with the file's
    own object-like #defines and #undefs tracked (macros from included headers are not seen: 0)"""
    macros = dict(defs)
    stack = []   # [this branch taken, some branch of the chain taken, the enclosing taken]
    on = True
    for no, line in enumerate(text.split("\n"), 1):
        s = line.strip()
        m = re.match(r"#\s*(ifdef|ifndef|if|elif|else|endif|define|undef)\b(.*)", s)
        if m:
            d, rest = m.group(1), m.group(2).strip()
            if d in ("ifdef", "ifndef", "if"):
                c = (rest.split()[0] in macros) if d == "ifdef" and rest else \
                    (rest.split()[0] not in macros) if d == "ifndef" and rest else _eval_if(rest, macros)
                stack.append([on and c, c, on])
                on = on and c
            elif d == "elif" and stack:
                t, any_, outer = stack[-1]
                c = not any_ and _eval_if(rest, macros)
                stack[-1] = [outer and c, any_ or c, outer]
                on = outer and c
            elif d == "else" and stack:
                t, any_, outer = stack[-1]
                stack[-1] = [outer and not any_, True, outer]
                on = outer and not any_
            elif d == "endif" and stack:
                on = stack.pop()[2]
            elif d == "define" and on:
                mm = re.match(r"(\w+)(\s+(.*))?$", rest)
                if mm and "(" not in rest.split()[0]:
                    macros[mm.group(1)] = (mm.group(3) or "1").strip()
            elif d == "undef" and on and rest:
                macros.pop(rest.split()[0], None)
            continue
        if on:
            yield no, line


def ptx_in_hip(repo: Path) -> list:
    """[(file, line)] for each inline PTX statement an AMD device pass compiles (clang cannot assemble PTX for AMD):
    the engines' sources (src/, not src/ik2 - no HIP target), each file's #if chain evaluated for gfx1201 and gfx1100"""
    found = set()
    for pat in SOURCES:
        for f in sorted((repo / "src").rglob(pat)):
            if "ik2" in f.relative_to(repo).parts[:2]:
                continue
            raw = f.read_text(encoding="utf-8", errors="replace")
            lines = raw.split("\n")
            text = strip_comments_and_strings(raw)
            for defs in HIP_DEVICE.values():
                for no, line in device_lines(text, defs):
                    if ASM.search(line) and PTX.search(raw_asm_text(lines, no)):
                        found.add((str(f.relative_to(repo)), no))
    return sorted(found)


def missing_headers(repo: Path) -> list:
    """[(file, line, header)] for each CUDA header an AMD device pass includes that include/maya/hip_compat/ lacks
    (the engines' sources, as ptx_in_hip)"""
    compat = repo / "include" / "strata" / "hip_compat"
    have = {h.name for h in compat.glob("*")}
    found = set()
    for top in ("src", "include"):
        for pat in SOURCES:
            for f in sorted((repo / top).rglob(pat)):
                if compat in f.parents or "ik2" in f.relative_to(repo).parts[:2]:
                    continue
                raw = f.read_text(encoding="utf-8", errors="replace")
                for defs in HIP_DEVICE.values():
                    for no, line in device_lines(_strip_keep_includes(raw), defs):
                        m = HEADER.search(line)
                        if m and m.group(1) not in have:
                            found.add((str(f.relative_to(repo)), no, m.group(1)))
    return sorted(found)


def _strip_keep_includes(text: str) -> str:
    """comments blanked, strings kept (an #include <...> line is all the header check reads)"""
    return re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), re.sub(r"//[^\n]*", "", text), flags=re.S)


def main() -> int:
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    bad = unmapped(repo)
    for f, no, name in bad:
        print(f"{f}:{no}: {name} has no mapping in include/maya/hip_compat/ (the AMD build will not compile)")
    ptx = ptx_in_hip(repo)
    for f, no in ptx:
        print(f"{f}:{no}: inline PTX the HIP build compiles - guard it with #if !defined(MAYA_USE_HIP)")
    hdr = missing_headers(repo)
    for f, no, h in hdr:
        print(f"{f}:{no}: <{h}> is not in include/maya/hip_compat/ (the AMD build will not find it)")
    if not bad and not ptx and not hdr:
        print("hip_compat: every CUDA name and header the HIP build compiles has a mapping, and no inline PTX")
    return 1 if bad or ptx or hdr else 0


if __name__ == "__main__":
    sys.exit(main())
