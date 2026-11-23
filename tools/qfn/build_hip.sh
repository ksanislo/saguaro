#!/bin/bash
# tools/qfn/build_hip.sh <build dir> - the QFN engine (engines/qfn) for AMD GPUs (HIP, ROCm 7): the engine, and with
# TESTS=1 (default) every test target, so a HIP compile break shows up here as it does in the CUDA build.
# HIP_ARCHS (default gfx1201) = the architectures; ROCM (default $ROCM_PATH or /opt/rocm) = the ROCm installation;
# QFN_GGML_DIR (default third_party/llama.cpp) = the llama.cpp checkout at the pin (the same one tools/qfn/build.sh uses).
# Output: <dir>/saguaro-qfn (no kq libraries on HIP: the K-quant MMQ is in the engine). Then on a machine with the GPU:
#   ctest --test-dir <dir> --output-on-failure        (the HIP parity tests run on HIP_VISIBLE_DEVICES' first GPU)
set -eu
R=$(cd "$(dirname "$0")/../.." && pwd); S=$R/engines/qfn; B=${1:?build dir}
GGML=${QFN_GGML_DIR:-$R/third_party/llama.cpp}
ROCM=${ROCM:-${ROCM_PATH:-/opt/rocm}}
[ -f $GGML/ggml/CMakeLists.txt ] || { echo "no llama.cpp checkout at $GGML: set QFN_GGML_DIR"; exit 1; }
[ -x $ROCM/llvm/bin/clang++ ] || { echo "no ROCm at $ROCM (its llvm/bin/clang++): set ROCM"; exit 1; }
[ -e $B ] && { echo "$B exists - never rebuild into a dir something may load from; pick a new one"; exit 1; }
cmake -S $S -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=$ROCM/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=$ROCM -DCMAKE_HIP_ARCHITECTURES="${HIP_ARCHS:-gfx1201}" -DSTRATA_ENABLE_HIP=ON \
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_DSV41=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_GGML_DIR=$GGML > $B.cmake.log 2>&1 \
  || { tail -30 $B.cmake.log; exit 1; }
T=strata; [ "${TESTS:-1}" = 1 ] && T=all
nice -n 5 ninja -C $B $T > $B.build.log 2>&1 || { grep -E "error:|FAILED:" $B.build.log | head -30; exit 1; }
[ -e $B/saguaro-qfn ] || { echo "ninja left no $B/saguaro-qfn"; exit 1; }
echo "QFN HIP BUILD OK $(git -C $R rev-parse --short HEAD) ${HIP_ARCHS:-gfx1201} -> $B"
