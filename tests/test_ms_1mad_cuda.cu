// Standalone Ampere (sm_86) CUDA round-trip test for adamant::kernels::ms_cuda.
// Build: nvcc -O3 -arch=sm_86 -std=c++20 -Isrc tests/test_ms_1mad_cuda.cu -o build/test_ms_1mad_cuda && ./build/test_ms_1mad_cuda

#include "kernels/ms_1mad_avx2.h"
#include "kernels/ms_1mad_cuda.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x03FFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

static void pack_row_ref(uint8_t* dst, float scale, const std::vector<uint32_t>& states, int k) {
    const int in_f = static_cast<int>(states.size());
    const uint16_t d16 = fp32_to_fp16(scale);
    const uint16_t s0 = static_cast<uint16_t>(states[0] & adamant::kernels::ms::kStateMask);
    std::memcpy(dst, &d16, 2);
    std::memcpy(dst + 2, &s0, 2);

    const size_t n_sym_bytes = static_cast<size_t>(((in_f - 1) * k + 7) / 8);
    uint8_t* qs = dst + 4;
    std::memset(qs, 0, n_sym_bytes);

    const uint32_t sym_mask = (1u << k) - 1u;
    int bit_pos = 0;
    for (int c = 1; c < in_f; ++c) {
        uint32_t sym = states[c] & sym_mask;
        for (int b = k - 1; b >= 0; --b) {
            if ((sym >> b) & 1u) {
                qs[bit_pos >> 3] |= static_cast<uint8_t>(1u << (7 - (bit_pos & 7)));
            }
            ++bit_pos;
        }
    }
}

template <int K>
static bool test_cuda_gemv(int out_f, int in_f, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist_x(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_d(0.001f, 0.02f);
    std::uniform_int_distribution<uint32_t> dist_s0(0, adamant::kernels::ms::kStateMask);
    std::uniform_int_distribution<uint32_t> dist_sym(0, (1u << K) - 1u);

    const size_t stride = adamant::kernels::ms::ms_row_bytes(in_f, K);
    std::vector<uint8_t> h_w(static_cast<size_t>(out_f) * stride, 0);
    std::vector<float> h_x(in_f);
    for (int c = 0; c < in_f; ++c) h_x[c] = dist_x(rng);

    std::vector<uint32_t> row_states(in_f);
    for (int r = 0; r < out_f; ++r) {
        uint32_t s = dist_s0(rng);
        row_states[0] = s;
        for (int c = 1; c < in_f; ++c) {
            s = ((s << K) | dist_sym(rng)) & adamant::kernels::ms::kStateMask;
            row_states[c] = s;
        }
        pack_row_ref(h_w.data() + static_cast<size_t>(r) * stride, dist_d(rng), row_states, K);
    }

    std::vector<float> cpu_out(out_f, 0.0f);
    adamant::kernels::ms::ms_1mad_gemv<K>(h_w.data(), h_x.data(), cpu_out.data(), out_f, in_f);

    uint8_t* d_w = nullptr;
    float* d_x = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_w, h_w.size());
    cudaMalloc(&d_x, h_x.size() * sizeof(float));
    cudaMalloc(&d_out, out_f * sizeof(float));
    cudaMemcpy(d_w, h_w.data(), h_w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);

    adamant::kernels::ms_cuda::launch_ms_1mad_gemv<K>(d_w, d_x, d_out, out_f, in_f);
    cudaDeviceSynchronize();

    std::vector<float> gpu_out(out_f, 0.0f);
    cudaMemcpy(gpu_out.data(), d_out, out_f * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_out);

    float max_err = 0.0f;
    for (int r = 0; r < out_f; ++r) {
        max_err = std::max(max_err, std::fabs(gpu_out[r] - cpu_out[r]));
    }
    std::printf("[CUDA MS_C%d %dx%d] max_abs_err vs CPU = %.6g\n", K, out_f, in_f, max_err);
    return max_err < 1e-3f;
}

int main() {
    bool ok = true;
    ok &= test_cuda_gemv<3>(64, 2560, 201);
    ok &= test_cuda_gemv<3>(64, 640, 202);
    ok &= test_cuda_gemv<4>(64, 2560, 203);
    ok &= test_cuda_gemv<4>(64, 640, 204);
    ok &= test_cuda_gemv<6>(64, 2560, 205);
    ok &= test_cuda_gemv<6>(64, 640, 206);
    if (!ok) {
        std::fprintf(stderr, "CUDA TESTS FAILED\n");
        return 1;
    }
    std::printf("ALL CUDA MS_1MAD TESTS PASSED\n");
    return 0;
}
