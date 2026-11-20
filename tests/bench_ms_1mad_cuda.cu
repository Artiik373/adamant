// Ampere (sm_86) RTX 3090 VRAM bandwidth benchmark for MS_C6 dense spine + MS_C4 hot experts.
// Build: nvcc -O3 -arch=sm_86 -std=c++20 -Isrc tests/bench_ms_1mad_cuda.cu -o build/bench_ms_1mad_cuda
// Run:   ./build/bench_ms_1mad_cuda

#include "kernels/ms_1mad_cuda.cuh"

#include <cstdio>
#include <cstdint>
#include <vector>

using namespace adamant::kernels::ms_cuda;

template <int K>
static void bench_shape(const char* label, int out_f, int in_f, int iters) {
    size_t w_bytes = static_cast<size_t>(out_f) * ms_row_bytes(in_f, K);
    uint8_t* d_w = nullptr;
    float* d_x = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_w, w_bytes);
    cudaMalloc(&d_x, in_f * sizeof(float));
    cudaMalloc(&d_out, out_f * sizeof(float));
    cudaMemset(d_w, 0x5A, w_bytes);
    cudaMemset(d_x, 0, in_f * sizeof(float));

    // Warmup
    for (int i = 0; i < 10; ++i) launch_ms_1mad_gemv<K>(d_w, d_x, d_out, out_f, in_f);
    cudaDeviceSynchronize();

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);
    for (int i = 0; i < iters; ++i) {
        launch_ms_1mad_gemv<K>(d_w, d_x, d_out, out_f, in_f);
    }
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);

    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, ev0, ev1);
    double us_per_call = (ms_total * 1000.0) / iters;
    double gb_per_s = (static_cast<double>(w_bytes) * iters) / (ms_total * 1e6);

    std::printf("[CUDA %-14s MS_C%d %4dx%-4d] %6.1f us | %.1f GB/s\n",
                label, K, out_f, in_f, us_per_call, gb_per_s);

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_out);
}

int main() {
    bench_shape<6>("Dense spine", 2560, 2560, 500);
    bench_shape<6>("lm_head",    32768, 2560, 100);
    bench_shape<4>("Expert gate",  640, 2560, 500);
    bench_shape<4>("Expert down", 2560,  640, 500);
    bench_shape<3>("Expert gate",  640, 2560, 500);
    bench_shape<3>("Expert down", 2560,  640, 500);
    return 0;
}
