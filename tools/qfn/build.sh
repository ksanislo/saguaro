#!/bin/bash
# tools/qfn/build.sh <build dir> - the QFN engine (engines/qfn, Strata's tree with its own CMake project) for mangonel:
# the live QFN build's options (bench/qfn-exact3/build.sh, build-x3b), so the code is the live engine's bit for bit.
# Output: <dir>/maya-qfn (= the strata target), libstrata_kq.so + libstrata_kq_tab.so (one set, ABI 2, loaded from the
# binary's folder). The version string stays Strata's 0.1.49.2 (saved QFN conversations are keyed on it).
# VISION=1 also builds <dir>-vision/bin/maya-qfn-vision (= strata-vision) from engines/qfn/tools/vision.
set -eu
R=$(cd "$(dirname "$0")/../.." && pwd); S=$R/engines/qfn; B=${1:?build dir}
GGML=${QFN_GGML_DIR:-/data/llm/strata-main/third_party/llama.cpp}   # (3cf03257f, the pin of both trees)
[ -e $B ] && { echo "$B exists - never rebuild into a dir something may load from; pick a new one"; exit 1; }
cmake -S $S -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ISA_FLOOR=avx -DSTRATA_BUILD_DSV41=ON \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_GGML_DIR=$GGML \
  -DSTRATA_MMQ_TAB=ON -DSTRATA_MMQ_TAB_TYPES= "-DSTRATA_MMQ_TAB_KQ_TYPES=q4_k;q5_k;q5_1;q8_0" > $B.cmake.log 2>&1
nice -n 5 ninja -C $B ${QFN_TARGETS:-strata strata_kq strata_kq_tab} > $B.build.log 2>&1 || { tail -30 $B.build.log; exit 1; }
cp -p $B/strata $B/maya-qfn
if [ "${VISION:-0}" = 1 ]; then
  cmake -G Ninja -S $S/tools/vision -B $B-vision -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=$GGML -DSTRATA_VISION_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc > $B-vision.cmake.log 2>&1
  ninja -C $B-vision > $B-vision.build.log 2>&1 || { tail -30 $B-vision.build.log; exit 1; }
  V=$(ls $B-vision/bin/strata-vision $B-vision/strata-vision 2>/dev/null | head -1); cp -p $V $(dirname $V)/maya-qfn-vision
fi
echo "QFN BUILD OK $(git -C $R rev-parse --short HEAD) -> $B"
