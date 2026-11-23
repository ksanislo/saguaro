#pragma once
// hipBLAS 2.x (ROCm 6.x) keeps the hipDataType / hipblasComputeType_t GemmEx API behind HIPBLAS_V2 (ROCm 7's
// hipBLAS 3 has it by default): opt in before the header, as a distribution's ROCm 6 (Debian's 6.4) needs it.
// (a hipBLAS without the version header builds as before: no version, so neither ROCm 6 branch below)
#if defined(__has_include)
#if __has_include(<hipblas/hipblas-version.h>)
#include <hipblas/hipblas-version.h>
#endif
#endif
#if defined(hipblasVersionMajor) && hipblasVersionMajor < 3 && !defined(HIPBLAS_V2)
#define HIPBLAS_V2
#endif
#include <hipblas/hipblas.h>
#if defined(hipblasVersionMajor) && hipblasVersionMajor == 0
// (engines/qfn) Ubuntu's hipBLAS 0.54 has no workspace API or compute-type enum: its underlying rocBLAS API directly,
// including ownership of the real handle, so the engine's explicit workspace contract is preserved.
#include <rocblas/rocblas.h>
#define CUBLAS_COMPUTE_32F rocblas_datatype_f32_r
#define CUBLAS_DEFAULT_MATH 0
#define CUBLAS_GEMM_DEFAULT rocblas_gemm_algo_standard
#define CUBLAS_OP_N rocblas_operation_none
#define CUBLAS_OP_T rocblas_operation_transpose
#define CUBLAS_STATUS_SUCCESS rocblas_status_success
#define CUDA_R_16BF rocblas_datatype_bf16_r
#define CUDA_R_16F rocblas_datatype_f16_r
#define CUDA_R_32F rocblas_datatype_f32_r
#define cublasCreate rocblas_create_handle
#define cublasDestroy rocblas_destroy_handle
#define cublasHandle_t rocblas_handle
#define cublasSetStream rocblas_set_stream
#define cublasSetWorkspace rocblas_set_workspace
#define cublasStatus_t rocblas_status

inline rocblas_status cublasSetMathMode(rocblas_handle h, int mode) {
    // This API has only its default arithmetic; GEMM below explicitly selects
    // FP32 accumulation. Reject any request for a different math mode.
    if (!h) return rocblas_status_invalid_handle;
    return mode == CUBLAS_DEFAULT_MATH ? rocblas_status_success : rocblas_status_invalid_value;
}
inline rocblas_status cublasGemmEx(rocblas_handle h, rocblas_operation trans_a,
    rocblas_operation trans_b, int m, int n, int k, const void* alpha,
    const void* a, rocblas_datatype a_type, int lda,
    const void* b, rocblas_datatype b_type, int ldb, const void* beta,
    void* c, rocblas_datatype c_type, int ldc,
    rocblas_datatype compute_type, rocblas_gemm_algo algo) {
    return rocblas_gemm_ex(h, trans_a, trans_b, m, n, k, alpha,
        a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc,
        c, c_type, ldc, compute_type, algo, 0, 0);
}
#else
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
// (F32 products in TF32 where the GPU has it - maya-hy4's vision attention; on AMD plain F32, the same or better)
#define CUBLAS_COMPUTE_32F_FAST_TF32 HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_PEDANTIC HIPBLAS_COMPUTE_32F   // (maya-ik2: FP32 without TF32 - what hipBLAS does anyway)
#define cublasComputeType_t hipblasComputeType_t
#define CUBLAS_DEFAULT_MATH HIPBLAS_DEFAULT_MATH
// cuBLAS' pedantic mode keeps F32 GEMMs in F32 (no TF32); hipBLAS' default math already does (XF32 is opt-in), and
// HIPBLAS_DEFAULT_MATH exists in every hipBLAS the build supports.
#define CUBLAS_PEDANTIC_MATH HIPBLAS_DEFAULT_MATH
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
// hipBLAS selects the available matrix instructions with its default algorithm.
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define cublasGemmAlgo_t hipblasGemmAlgo_t   // (the DeepSeek vision tower's FP16 path picks its algorithm)
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUDA_R_16BF HIP_R_16BF
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_32F HIP_R_32F
#define cudaDataType hipDataType
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasGemmEx hipblasGemmEx
// (the same argument order: hipDataType operands, hipblasStride strides, hipblasComputeType_t; glm-dsa's prompt MLA)
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#define cublasSgemm hipblasSgemm
#define cublasSgemv hipblasSgemv
#define cublasSgemmStridedBatched hipblasSgemmStridedBatched
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#define cublasHandle_t hipblasHandle_t
#define cublasSetMathMode hipblasSetMathMode
#define cublasSetStream hipblasSetStream
#if defined(hipblasVersionMajor) && hipblasVersionMajor < 3
// hipBLAS 2.x has no hipblasSetWorkspace: rocBLAS then sizes its own workspace (a no-op, the GEMMs are the same)
static inline hipblasStatus_t strata_hipblas_set_workspace_noop(hipblasHandle_t, void*, size_t) {
    return HIPBLAS_STATUS_SUCCESS;
}
#define cublasSetWorkspace strata_hipblas_set_workspace_noop
#else
#define cublasSetWorkspace hipblasSetWorkspace
#endif
#define cublasStatus_t hipblasStatus_t
#endif  // hipBLAS 0.x
