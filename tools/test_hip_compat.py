"""tools/test_hip_compat.py - the AMD build compiles the CUDA-shaped sources through include/maya/hip_compat/: every
CUDA runtime / cuBLAS name it compiles must have a mapping there (v1.0.21 shipped three without, and no AMD build
compiled).  No ROCm or GPU needed.

    python -m unittest tools.test_hip_compat
"""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_hip_compat as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class HipCompat(unittest.TestCase):
    def test_every_cuda_name_the_hip_build_compiles_is_mapped(self):
        bad = C.unmapped(REPO)
        self.assertEqual(bad, [], "\n".join(f"{f}:{n}: {name} - add it to include/maya/hip_compat/"
                                            for f, n, name in bad))

    def test_the_check_sees_what_it_must(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            compat = root / "include" / "strata" / "hip_compat"
            compat.mkdir(parents=True)
            (compat / "cuda_runtime.h").write_text("#define cudaMalloc hipMalloc\n#define cudaFree hipFree\n")
            (root / "src").mkdir()
            (root / "src" / "a.cu").write_text(
                "void f() {\n"
                "  cudaMalloc(&p, 4);                       // mapped\n"
                "  cudaDeviceGetPCIBusId(b, 16, 0);         // not: the AMD build breaks\n"
                "  puts(\"cudaMemcpyPeer failed\");         // a string\n"
                "#ifndef MAYA_USE_HIP\n"
                "  cudaMemcpyPeerAsync(a, 1, b, 0, n, s);   // CUDA only\n"
                "#endif\n"
                "#if defined(MAYA_USE_HIP)\n"
                "  hipFree(p);\n"
                "#else\n"
                "  cudaFuncGetAttributes(&x, k);            // CUDA only: the #else of a HIP block\n"
                "#endif\n"
                "}\n")
            self.assertEqual([(n, name) for _, n, name in C.unmapped(root)], [(3, "cudaDeviceGetPCIBusId")])

    def test_no_inline_ptx_or_missing_header_in_the_hip_build(self):
        self.assertEqual(C.ptx_in_hip(REPO), [], "inline PTX an AMD device pass compiles: guard it")
        self.assertEqual(C.missing_headers(REPO), [], "CUDA headers include/maya/hip_compat/ lacks")

    def test_ptx_and_headers_seen_where_compiled(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "include" / "strata" / "hip_compat").mkdir(parents=True)
            (root / "include" / "strata" / "hip_compat" / "cuda_fp16.h").write_text("")
            (root / "src").mkdir()
            (root / "src" / "k.cu").write_text(
                "#include <cuda_fp16.h>\n"                                   # 1: mapped
                "#include <cuda_bf16.h>\n"                                   # 2: not
                "__device__ void f(float* c, unsigned a) {\n"
                "    asm volatile(\"mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0}, {%1};\" : \"+f\"(c[0]) : \"r\"(a));\n"
                "#if !defined(MAYA_USE_HIP)\n"
                "    asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(t));\n"   # CUDA only
                "#endif\n"
                "#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800\n"
                "#define TC 1\n"
                "#endif\n"
                "#if defined(TC) && !defined(__HIPCC__)\n"
                "    asm volatile(\"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0}, [%1];\" : \"=r\"(a) : \"r\"(a));\n"
                "#endif\n"
                "    asm volatile(\"v_alignbit_b32 %0, %1, %2, 8\" : \"=v\"(a) : \"v\"(a), \"v\"(a));\n"   # AMD's own
                "}\n")
            self.assertEqual(C.ptx_in_hip(root), [("src/k.cu", 4)])
            self.assertEqual(C.missing_headers(root), [("src/k.cu", 2, "cuda_bf16.h")])

if __name__ == "__main__":
    unittest.main()
