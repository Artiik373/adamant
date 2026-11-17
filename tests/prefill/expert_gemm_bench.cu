// tests/prefill/expert_gemm_bench.cu - the fp16 expert GEMMs of the prompt path, at the real shapes, in the variants that
// could replace the per-expert cublasGemmEx (fp16 in, fp32 out, fp32 accumulate).  Prints microseconds and TFLOPs per
// call; weights rotate over 16 buffers so none stays in L2 between calls.  Not a pass/fail test.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { std::printf("%s:%d cublas %d\n", __FILE__, __LINE__, (int) s_); return -1.f; } } while (0)

static cublasHandle_t H;
static cudaStream_t S;
constexpr int NW = 16;   // rotating weight buffers

struct Shape { const char* name; int N, K; };

template <class F>
static float timeit(F f, int iters = 60) {
    for (int i = 0; i < 6; ++i) if (f(i) < 0.f) return -1.f;
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    CK(cudaStreamSynchronize(S));
    cudaEventRecord(a, S);
    for (int i = 0; i < iters; ++i) f(i);
    cudaEventRecord(b, S);
    CK(cudaEventSynchronize(b));
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    return ms * 1000.f / iters;
}

int main() {
    CK(cudaStreamCreate(&S));
    cublasCreate(&H);
    cublasSetStream(H, S);
    void* ws = nullptr;
    CK(cudaMalloc(&ws, 32 << 20));
    cublasSetWorkspace(H, ws, 32 << 20);
    cudaDeviceProp pr;
    cudaGetDeviceProperties(&pr, 0);
    std::printf("%s\n", pr.name);
    const Shape shapes[2] = {{"gate/up N=1280 K=2560", 1280, 2560}, {"down    N=2560 K=640 ", 2560, 640}};
    const int Ms[] = {40, 80, 160, 320, 640};
    for (const Shape& sh : shapes) {
        for (int M : Ms) {
            const size_t wsz = (size_t) sh.N * sh.K, xsz = (size_t) M * sh.K, ysz = (size_t) M * sh.N;
            std::vector<__half*> W(NW), X(NW);
            std::vector<float*> Y32(NW);
            std::vector<__half*> Y16(NW);
            for (int i = 0; i < NW; ++i) {
                CK(cudaMalloc(&W[i], wsz * 2)); CK(cudaMalloc(&X[i], xsz * 2 * 8));
                CK(cudaMalloc(&Y32[i], ysz * 4 * 8)); CK(cudaMalloc(&Y16[i], ysz * 2 * 8));
                CK(cudaMemset(W[i], 0, wsz * 2)); CK(cudaMemset(X[i], 0, xsz * 2 * 8));
            }
            const double fl = 2.0 * M * sh.N * sh.K;
            auto rep = [&](const char* nm, float us, int per) {
                if (us < 0) std::printf("  M=%-4d %-34s unsupported\n", M, nm);
                else std::printf("  M=%-4d %-34s %8.1f us/call  %6.1f TFLOPs\n", M, nm, us / per, fl * per / (us * 1e-6) / 1e12);
            };
            const float one = 1.f, zero = 0.f;
            const __half h1 = __float2half(1.f), h0 = __float2half(0.f);
            rep("cublasGemmEx f16->f32 (current)", timeit([&](int i) {
                CB(cublasGemmEx(H, CUBLAS_OP_T, CUBLAS_OP_N, sh.N, M, sh.K, &one, W[i % NW], CUDA_R_16F, sh.K, X[i % NW], CUDA_R_16F, sh.K,
                                &zero, Y32[i % NW], CUDA_R_32F, sh.N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)); return 0.f; }), 1);
            rep("f16->f16 out, acc f32", timeit([&](int i) {
                CB(cublasGemmEx(H, CUBLAS_OP_T, CUBLAS_OP_N, sh.N, M, sh.K, &one, W[i % NW], CUDA_R_16F, sh.K, X[i % NW], CUDA_R_16F, sh.K,
                                &zero, Y16[i % NW], CUDA_R_16F, sh.N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)); return 0.f; }), 1);
            rep("f16->f16 out, acc f16", timeit([&](int i) {
                CB(cublasGemmEx(H, CUBLAS_OP_T, CUBLAS_OP_N, sh.N, M, sh.K, &h1, W[i % NW], CUDA_R_16F, sh.K, X[i % NW], CUDA_R_16F, sh.K,
                                &h0, Y16[i % NW], CUDA_R_16F, sh.N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT)); return 0.f; }), 1);
            // 8 experts of the same M in one strided-batched call (weights stride = one buffer; here buffers are separate, so
            // pointer-array batched is the honest form)
            std::vector<const void*> hw(8), hx(8);
            std::vector<void*> hy32(8), hy16(8);
            for (int e = 0; e < 8; ++e) { hw[e] = W[e]; hx[e] = (char*) X[0] + (size_t) e * xsz * 2; hy32[e] = (char*) Y32[0] + (size_t) e * ysz * 4; hy16[e] = (char*) Y16[0] + (size_t) e * ysz * 2; }
            const void **dw, **dx; void **dy32, **dy16;
            CK(cudaMalloc(&dw, 64)); CK(cudaMalloc(&dx, 64)); CK(cudaMalloc(&dy32, 64)); CK(cudaMalloc(&dy16, 64));
            CK(cudaMemcpy(dw, hw.data(), 64, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx, hx.data(), 64, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dy32, hy32.data(), 64, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dy16, hy16.data(), 64, cudaMemcpyHostToDevice));
            rep("batched x8 f16->f32 out", timeit([&](int) {
                CB(cublasGemmBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, sh.N, M, sh.K, &one, (const void* const*) dw, CUDA_R_16F, sh.K,
                                       (const void* const*) dx, CUDA_R_16F, sh.K, &zero, dy32, CUDA_R_32F, sh.N, 8, CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT)); return 0.f; }), 8);
            rep("batched x8 f16->f16 out", timeit([&](int) {
                CB(cublasGemmBatchedEx(H, CUBLAS_OP_T, CUBLAS_OP_N, sh.N, M, sh.K, &one, (const void* const*) dw, CUDA_R_16F, sh.K,
                                       (const void* const*) dx, CUDA_R_16F, sh.K, &zero, dy16, CUDA_R_16F, sh.N, 8, CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT)); return 0.f; }), 8);
            for (int i = 0; i < NW; ++i) { cudaFree(W[i]); cudaFree(X[i]); cudaFree(Y32[i]); cudaFree(Y16[i]); }
            cudaFree(dw); cudaFree(dx); cudaFree(dy32); cudaFree(dy16);
        }
        std::printf("\n");
    }
    return 0;
}
