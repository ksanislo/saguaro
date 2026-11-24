#!/usr/bin/env python3
"""tools/hip_syntax_check.py - every HIP source of the AMD build through ROCm's clang with -fsyntax-only (host and device
passes, no GPU, no link): a HIP build break shows on a PC without an AMD GPU too, if it has ROCm's compiler and headers.
check_hip_compat.py's scans miss what only a compiler sees (a member a HIP stub lacks, an asm constraint, a missing
overload); this sees those.  The flags mirror CMakeLists.txt's HIP targets (saguaro-glm / -ds4 / -mm3 / -hy4 / -ik2).

    python tools/hip_syntax_check.py --rocm /opt/rocm [--arch gfx1201] [--ggml third_party/llama.cpp] [-j 8] [FILE ...]

--rocm: a ROCm install, or a copy of its llvm/bin/clang-20 (+ the clang / clang++ links and *.cfg), llvm/lib/clang/20/include,
include/ and share/hip/version (without it clang takes an old HIP and leaves its math wrapper out) - 46 MB as tar.xz;
--ggml: the llama.cpp at setup.py's LLAMA_CPP_COMMIT (src/prefill's ggml kernels compile against its ggml-cuda);
SAGUARO_HIP_SYNTAX_ROCM names it for tools/test_hip_compat.py, which skips the check when it is not set.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CUDA_ONLY = {"src/kernels/peer_probe.cu"}   # (not in any HIP target)
GGML_FILES = {"src/prefill/ggml_cuda_host.cu", "src/prefill/moe_fused_rdna4.cu", "src/prefill/moe_mmq.cu"}
IQ_ONLY = ["-DMAYA_DS4_IQ_ONLY"]
MIN_FORMS = ["-DDS4_IQ_MIN_FORMS", "-DMAYA_DS4_IQ_ONLY"]


def inc(*dirs) -> list:
    return [f"-I{REPO / d}" for d in dirs]


HY4_INC = ("src/hy4", "src/hy4/ds4", "src/ds4", "src/mm3")   # (saguaro-hy4's order: its own ds4 copy first)


def variants(rel: str) -> list:
    """The extra flags of each target that compiles the file (one entry a distinct compile)."""
    if rel.startswith("src/ik2/"):
        return [["-ffp-contract=on"] + (MIN_FORMS if rel.endswith(("ik2_draft.cu", "ik2_main.cu")) else []) +
                inc(*HY4_INC, "src/ik2", "src/strata_fmt")]
    if rel.startswith("src/hy4/"):
        return [MIN_FORMS + inc(*HY4_INC)]
    if rel.startswith("src/mm3/"):
        return [IQ_ONLY + inc("src/mm3")]
    if rel in ("src/ds4/ds4_iq.cu", "src/ds4/ds4_lt.cu"):   # saguaro-ds4's and saguaro-mm3's builds of them
        return [[], IQ_ONLY]
    if rel in ("src/kernels/cuda/elementwise.cu", "src/kernels/cuda/quantize_act.cu"):
        return [["-ffp-contract=off"]]
    if rel == "src/kernels/cuda/exl3.cu":
        return [["-ffp-contract=on"]]
    return [[]]


def sources(names) -> list:
    if names:
        return [str(Path(n).resolve().relative_to(REPO)) for n in names]
    return sorted(str(p.relative_to(REPO)) for p in (REPO / "src").rglob("*.cu")
                  if str(p.relative_to(REPO)) not in CUDA_ONLY)


def command(clang: Path, rocm: Path, arch: str, ggml: Path, gen: Path, rel: str, extra: list) -> list:
    base = [str(clang), "-x", "hip", f"--offload-arch={arch}", f"--rocm-path={rocm}", "-nogpulib", "-std=c++20",
            "-fsyntax-only", "-w", "-D__HIP_PLATFORM_AMD__=1", "-D__HIP_ROCclr__=1", "-DNDEBUG",
            f"-I{REPO / 'include'}", f"-I{ggml / 'ggml/include'}", f"-I{rocm / 'include'}"]
    if rel in GGML_FILES:   # (llama.cpp's own HIP kernels: ggml-cuda's headers, no hip_compat)
        return base + ["-DGGML_USE_HIP=1", f"-I{ggml / 'ggml/src'}", f"-I{ggml / 'ggml/src/ggml-cuda'}",
                       str(REPO / rel)]
    return base + ["-DMAYA_USE_HIP=1", "-DMAYA_HAS_WMMA=1", f'-DMAYA_HIP_ARCHS="{arch}"',
                   "-DMAYA_NATIVE_EXPERTS=1", "-DMAYA_HIPBLASLT_AVAILABLE=1", "-DMAYA_PREFILL_MMQ=1",
                   "-DMAYA_HIP_PREFILL_FUSED=1", "-DDS4_CUDA_PEER_AR=1", "-DHIPBLASLT_USE_ROCROLLER",
                   '-DMAYA_VERSION="0"', f"-I{REPO / 'include/maya/hip_compat'}",
                   f"-I{REPO / 'third_party/ggml'}", f"-I{gen}",
                   "-include", str(REPO / "include/maya/hip_compat/cuda_runtime.h")] + extra + [str(REPO / rel)]


def check(rocm: Path, arch: str = "gfx1201", ggml: Path | None = None, jobs: int = os.cpu_count() or 4,
          names=()) -> list:
    """[(file, flags, first error lines)] for every compile that fails."""
    clang = rocm / "llvm" / "bin" / "clang++"
    ggml = ggml or REPO / "third_party" / "llama.cpp"
    if not clang.exists():
        raise FileNotFoundError(f"{clang} not found")
    if not (ggml / "ggml" / "include").exists():
        raise FileNotFoundError(f"{ggml}/ggml/include not found (--ggml: the llama.cpp checkout the setup fetches)")
    with tempfile.TemporaryDirectory() as gen:
        Path(gen, "maya_build_id.h").write_text('#define MAYA_BUILD_ID "syntax"\n#define MAYA_RELEASE "0"\n')
        jobs_ = [(rel, ex) for rel in sources(names) for ex in variants(rel)]

        def one(job):
            rel, ex = job
            r = subprocess.run(command(clang, rocm, arch, ggml, Path(gen), rel, ex), capture_output=True, text=True)
            if r.returncode == 0:
                return None
            errs = [ln for ln in r.stderr.splitlines() if "error:" in ln]
            return rel, " ".join(ex), "\n".join(errs[:8] or r.stderr.splitlines()[-8:])

        with cf.ThreadPoolExecutor(max(1, jobs)) as pool:
            return [x for x in pool.map(one, jobs_) if x is not None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rocm", default=os.environ.get("SAGUARO_HIP_SYNTAX_ROCM") or os.environ.get("ROCM_PATH") or "/opt/rocm")
    ap.add_argument("--arch", default="gfx1201")
    ap.add_argument("--ggml", help="llama.cpp checkout (default third_party/llama.cpp)")
    ap.add_argument("-j", type=int, default=os.cpu_count() or 4)
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    bad = check(Path(a.rocm), a.arch, Path(a.ggml) if a.ggml else None, a.j, a.files)
    for rel, ex, msg in bad:
        print(f"FAIL {rel} {ex}\n{msg}\n")
    n = sum(len(variants(r)) for r in sources(a.files))
    print(f"{n - len(bad)} of {n} HIP compiles clean for {a.arch}" + ("" if not bad else f", {len(bad)} FAIL"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
