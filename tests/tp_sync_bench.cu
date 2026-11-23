// tests/tp_sync_bench.cu - what tensor parallelism's sums cost a decode token: 96 all-reduces of one 2304-wide F32 row
// between two GPUs (each: a tiny kernel on both, the partial copied over both ways, the sum on both), as a chain in one
// CUDA graph - the critical path a TP-2 Inkling token would add.  Prints us a sum and ms a token.
//     tp_sync_bench A B [n_sums=96] [C=2304] [reps=200]
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CK(x)                                                                                   \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            std::fprintf(stderr, "%s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__); \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

__global__ void work(float* x, int n) {   // (stands for a member's share of a visit: touches its row)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 0.999f + 1.0f;
}
__global__ void add2(const float* a, const float* b, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: tp_sync_bench A B [n_sums] [C] [reps]\n");
        return 2;
    }
    const int A = std::atoi(argv[1]), B = std::atoi(argv[2]);
    const int NS = argc > 3 ? std::atoi(argv[3]) : 96, C = argc > 4 ? std::atoi(argv[4]) : 2304;
    const int R = argc > 5 ? std::atoi(argv[5]) : 200;
    int ab = 0, ba = 0;
    CK(cudaDeviceCanAccessPeer(&ab, A, B));
    CK(cudaDeviceCanAccessPeer(&ba, B, A));
    const bool peer = ab && ba;
    if (peer) {
        CK(cudaSetDevice(A));
        cudaDeviceEnablePeerAccess(B, 0);
        CK(cudaSetDevice(B));
        cudaDeviceEnablePeerAccess(A, 0);
        (void) cudaGetLastError();
    }
    const size_t bytes = (size_t) C * 4;
    float *xa, *pa, *ia, *xb, *pb, *ib;
    cudaStream_t sa, sb;
    cudaEvent_t ea, eb, fork;
    CK(cudaSetDevice(A));
    CK(cudaMalloc(&xa, bytes)); CK(cudaMalloc(&pa, bytes)); CK(cudaMalloc(&ia, bytes));
    CK(cudaMemset(xa, 0, bytes));
    CK(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&ea, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    CK(cudaSetDevice(B));
    CK(cudaMalloc(&xb, bytes)); CK(cudaMalloc(&pb, bytes)); CK(cudaMalloc(&ib, bytes));
    CK(cudaMemset(xb, 0, bytes));
    CK(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&eb, cudaEventDisableTiming));
    const int th = 256, bl = (C + th - 1) / th;
    double ms = -1;
    if (peer) {
    // the chain, captured on A's stream with B's joined
    CK(cudaSetDevice(A));
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(sa, cudaStreamCaptureModeGlobal));
    CK(cudaEventRecord(fork, sa));
    CK(cudaStreamWaitEvent(sb, fork, 0));
    for (int i = 0; i < NS; ++i) {
        CK(cudaSetDevice(A));
        work<<<bl, th, 0, sa>>>(xa, C);
        CK(cudaSetDevice(B));
        work<<<bl, th, 0, sb>>>(xb, C);
        // each member's partial to the other (A's xa -> ib, B's xb -> ia), then both add in the same order
        CK(cudaMemcpyAsync(ib, xa, bytes, cudaMemcpyDefault, sa));
        CK(cudaMemcpyAsync(ia, xb, bytes, cudaMemcpyDefault, sb));
        CK(cudaEventRecord(ea, sa));
        CK(cudaEventRecord(eb, sb));
        CK(cudaStreamWaitEvent(sa, eb, 0));
        CK(cudaStreamWaitEvent(sb, ea, 0));
        CK(cudaSetDevice(A));
        add2<<<bl, th, 0, sa>>>(xa, ia, pa, C);
        CK(cudaSetDevice(B));
        add2<<<bl, th, 0, sb>>>(ib, xb, pb, C);
    }
    CK(cudaEventRecord(eb, sb));
    CK(cudaStreamWaitEvent(sa, eb, 0));
    CK(cudaSetDevice(A));
    CK(cudaStreamEndCapture(sa, &g));
    cudaGraphExec_t ge;
    CK(cudaGraphInstantiate(&ge, g, 0));
    for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ge, sa));
    CK(cudaStreamSynchronize(sa));
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < R; ++i) CK(cudaGraphLaunch(ge, sa));
    CK(cudaStreamSynchronize(sa));
    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / R;
    }
    // the same chain launched without a graph (as the engine's two-GPU path runs)
    auto chain = [&]() {
        for (int i = 0; i < NS; ++i) {
            CK(cudaSetDevice(A));
            work<<<bl, th, 0, sa>>>(xa, C);
            CK(cudaSetDevice(B));
            work<<<bl, th, 0, sb>>>(xb, C);
            CK(cudaMemcpyAsync(ib, xa, bytes, cudaMemcpyDefault, sa));
            CK(cudaMemcpyAsync(ia, xb, bytes, cudaMemcpyDefault, sb));
            CK(cudaEventRecord(ea, sa));
            CK(cudaEventRecord(eb, sb));
            CK(cudaStreamWaitEvent(sa, eb, 0));
            CK(cudaStreamWaitEvent(sb, ea, 0));
            CK(cudaSetDevice(A));
            add2<<<bl, th, 0, sa>>>(xa, ia, pa, C);
            CK(cudaSetDevice(B));
            add2<<<bl, th, 0, sb>>>(ib, xb, pb, C);
        }
        CK(cudaSetDevice(A));
    };
    chain();
    CK(cudaDeviceSynchronize());
    const auto ts = std::chrono::steady_clock::now();
    for (int i = 0; i < R / 4 + 1; ++i) chain();
    CK(cudaStreamSynchronize(sa));
    CK(cudaSetDevice(B));
    CK(cudaStreamSynchronize(sb));
    CK(cudaSetDevice(A));
    const double ms_s = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count() / (R / 4 + 1);
    // the same chain without the copies and waits: the kernels' own time
    cudaGraph_t g2;
    CK(cudaStreamBeginCapture(sa, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < NS; ++i) {
        work<<<bl, th, 0, sa>>>(xa, C);
        add2<<<bl, th, 0, sa>>>(xa, ia, pa, C);
    }
    CK(cudaStreamEndCapture(sa, &g2));
    cudaGraphExec_t ge2;
    CK(cudaGraphInstantiate(&ge2, g2, 0));
    CK(cudaGraphLaunch(ge2, sa));
    CK(cudaStreamSynchronize(sa));
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < R; ++i) CK(cudaGraphLaunch(ge2, sa));
    CK(cudaStreamSynchronize(sa));
    const double ms0 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / R;
    std::printf("GPUs %d,%d (%s): %d sums of %d floats: graph %.3f ms, stream launches %.3f ms (kernels alone %.3f ms)"
                " -> %.1f / %.1f us a sum\n", A, B, peer ? "peer" : "no peer access: the driver's copies", NS, C, ms, ms_s,
                ms0, (ms - ms0) * 1e3 / NS, (ms_s - ms0) * 1e3 / NS);
    return 0;
}
