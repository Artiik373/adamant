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

// n_experts > 1 runs the grouped launch with a separate weight matrix and input per expert.
template <int K>
static bool test_cuda_gemv(int out_f, int in_f, uint32_t seed, int n_experts = 1) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist_x(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_d(0.001f, 0.02f);
    std::uniform_int_distribution<uint32_t> dist_s0(0, adamant::kernels::ms::kStateMask);
    std::uniform_int_distribution<uint32_t> dist_sym(0, (1u << K) - 1u);

    const size_t stride = adamant::kernels::ms::ms_row_bytes(in_f, K);
    const size_t w_bytes = static_cast<size_t>(out_f) * stride;
    std::vector<uint8_t> h_w(w_bytes * n_experts, 0);
    std::vector<float> h_x(static_cast<size_t>(in_f) * n_experts);
    for (float& v : h_x) v = dist_x(rng);

    std::vector<uint32_t> row_states(in_f);
    for (int r = 0; r < out_f * n_experts; ++r) {
        uint32_t s = dist_s0(rng);
        row_states[0] = s;
        for (int c = 1; c < in_f; ++c) {
            s = ((s << K) | dist_sym(rng)) & adamant::kernels::ms::kStateMask;
            row_states[c] = s;
        }
        pack_row_ref(h_w.data() + static_cast<size_t>(r) * stride, dist_d(rng), row_states, K);
    }

    std::vector<float> cpu_out(static_cast<size_t>(out_f) * n_experts, 0.0f);
    for (int e = 0; e < n_experts; ++e) {
        adamant::kernels::ms::ms_1mad_gemv<K>(h_w.data() + e * w_bytes, h_x.data() + static_cast<size_t>(e) * in_f,
                                              cpu_out.data() + static_cast<size_t>(e) * out_f, out_f, in_f);
    }

    uint8_t* d_w = nullptr;
    float* d_x = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_w, h_w.size());
    cudaMalloc(&d_x, h_x.size() * sizeof(float));
    cudaMalloc(&d_out, cpu_out.size() * sizeof(float));
    cudaMemcpy(d_w, h_w.data(), h_w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<const uint8_t*> w_ptrs(n_experts);
    std::vector<const float*> x_ptrs(n_experts);
    std::vector<float*> out_ptrs(n_experts);
    for (int e = 0; e < n_experts; ++e) {
        w_ptrs[e] = d_w + e * w_bytes;
        x_ptrs[e] = d_x + static_cast<size_t>(e) * in_f;
        out_ptrs[e] = d_out + static_cast<size_t>(e) * out_f;
    }
    const cudaError_t err = n_experts == 1
        ? adamant::kernels::ms_cuda::launch_ms_1mad_gemv<K>(d_w, d_x, d_out, out_f, in_f)
        : adamant::kernels::ms_cuda::launch_ms_1mad_gemv_grouped<K>(
              w_ptrs.data(), x_ptrs.data(), out_ptrs.data(), n_experts, out_f, in_f);
    if (err != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("[CUDA MS_C%d %dx%d x%d] launch failed\n", K, out_f, in_f, n_experts);
        return false;
    }

    std::vector<float> gpu_out(cpu_out.size(), 0.0f);
    cudaMemcpy(gpu_out.data(), d_out, gpu_out.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_out);

    float max_err = 0.0f;
    for (size_t i = 0; i < gpu_out.size(); ++i) {
        max_err = std::max(max_err, std::fabs(gpu_out[i] - cpu_out[i]));
    }
    std::printf("[CUDA MS_C%d %dx%d x%d] max_abs_err vs CPU = %.6g\n", K, out_f, in_f, n_experts, max_err);
    return max_err < 1e-3f * std::sqrt(std::max(1.0f, in_f / 2560.0f));  // fp32 sum-order error grows with width
}

// The engine path: Q8_1 input columns of the unrotated x, signs at the end of the payload, rotation in the kernel.
// Reference: the CPU GEMV on fwht_rotate_input(dequant(q8)).
template <int K>
static bool test_cuda_gemv_q8(int out_f, int in_f, int ncols, uint32_t seed) {
    namespace ms = adamant::kernels::ms;
    namespace mc = adamant::kernels::ms_cuda;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist_x(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_d(0.001f, 0.02f);
    std::uniform_int_distribution<uint32_t> dist_s0(0, ms::kStateMask);
    std::uniform_int_distribution<uint32_t> dist_sym(0, (1u << K) - 1u);

    const size_t stride = ms::ms_row_bytes(in_f, K);
    std::vector<uint8_t> payload(static_cast<size_t>(out_f) * stride + in_f, 0);
    std::vector<uint32_t> row_states(in_f);
    for (int r = 0; r < out_f; ++r) {
        uint32_t s = dist_s0(rng);
        row_states[0] = s;
        for (int c = 1; c < in_f; ++c) row_states[c] = s = ((s << K) | dist_sym(rng)) & ms::kStateMask;
        pack_row_ref(payload.data() + static_cast<size_t>(r) * stride, dist_d(rng), row_states, K);
    }
    int8_t* signs = reinterpret_cast<int8_t*>(payload.data() + static_cast<size_t>(out_f) * stride);
    for (int c = 0; c < in_f; ++c) signs[c] = (rng() & 1u) ? 1 : -1;

    struct Q81 { uint16_t d, s; int8_t qs[32]; };
    const int nb = in_f / 32;
    std::vector<Q81> xq(static_cast<size_t>(nb) * ncols);
    std::vector<float> cpu_out(static_cast<size_t>(out_f) * ncols);
    std::vector<float> x_deq(in_f), x_rot(in_f);
    const int had = mc::ms_hadamard_block(in_f);
    for (int j = 0; j < ncols; ++j) {
        for (int b = 0; b < nb; ++b) {
            float v[32], amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[i] = dist_x(rng)));
            const uint16_t d16 = fp32_to_fp16(amax / 127.0f);
            const float d = ms::fp16_to_fp32(d16);
            Q81& q = xq[static_cast<size_t>(j) * nb + b];
            q.d = d16;
            q.s = 0;
            for (int i = 0; i < 32; ++i) {
                q.qs[i] = static_cast<int8_t>(std::lrint(d > 0 ? v[i] / d : 0.0f));
                x_deq[b * 32 + i] = d * q.qs[i];
            }
        }
        ms::fwht_rotate_input(x_deq.data(), signs, x_rot.data(), in_f, had);
        ms::ms_1mad_gemv<K>(payload.data(), x_rot.data(), cpu_out.data() + static_cast<size_t>(j) * out_f, out_f, in_f);
    }

    uint8_t* d_w = nullptr;
    void* d_x = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_w, payload.size());
    cudaMalloc(&d_x, xq.size() * sizeof(Q81));
    cudaMalloc(&d_out, cpu_out.size() * sizeof(float));
    cudaMemcpy(d_w, payload.data(), payload.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, xq.data(), xq.size() * sizeof(Q81), cudaMemcpyHostToDevice);
    const bool launched = mc::launch_ms_1mad_gemv_q8<K>(d_w, d_x, d_out, out_f, in_f, ncols, nullptr) == cudaSuccess &&
                          cudaDeviceSynchronize() == cudaSuccess;
    std::vector<float> gpu_out(cpu_out.size(), 0.0f);
    cudaMemcpy(gpu_out.data(), d_out, gpu_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_out);
    if (!launched) {
        std::printf("[CUDA q8 MS_C%d %dx%d c%d] launch failed\n", K, out_f, in_f, ncols);
        return false;
    }
    float max_err = 0.0f;
    for (size_t i = 0; i < gpu_out.size(); ++i) max_err = std::max(max_err, std::fabs(gpu_out[i] - cpu_out[i]));
    std::printf("[CUDA q8 MS_C%d %dx%d c%d had=%d] max_abs_err vs CPU = %.6g\n", K, out_f, in_f, ncols, had, max_err);
    return max_err < 1e-3f * std::sqrt(std::max(1.0f, in_f / 2560.0f));  // fp32 sum-order error grows with width
}

int main() {
    bool ok = true;
    ok &= test_cuda_gemv<3>(64, 2560, 201);
    ok &= test_cuda_gemv<3>(64, 640, 202);
    ok &= test_cuda_gemv<4>(64, 2560, 203);
    ok &= test_cuda_gemv<4>(64, 640, 204);
    ok &= test_cuda_gemv<6>(64, 2560, 205);
    ok &= test_cuda_gemv<6>(64, 640, 206);
    ok &= test_cuda_gemv<3>(64, 132, 212);   // row stride padded from 54 to 56 bytes
    // More rows than resident warps: exercises the grid-stride prefetch loop.
    ok &= test_cuda_gemv<6>(8192, 2560, 207);
    ok &= test_cuda_gemv<4>(8192, 640, 208);
    // Grouped expert launch; 40 experts also covers the split into several launches.
    ok &= test_cuda_gemv<4>(640, 2560, 209, 8);
    ok &= test_cuda_gemv<4>(2560, 640, 210, 8);
    ok &= test_cuda_gemv<3>(640, 2560, 211, 40);
    ok &= test_cuda_gemv_q8<6>(2560, 2560, 1, 213);
    ok &= test_cuda_gemv_q8<6>(4096, 2560, 3, 214);
    ok &= test_cuda_gemv_q8<4>(2560, 640, 8, 215);
    ok &= test_cuda_gemv_q8<3>(640, 2560, 2, 216);
    ok &= test_cuda_gemv_q8<4>(256, 96, 1, 217);   // Hadamard block 32
    // Rows past one shared-memory chunk: the GDN / attention output projections take 6144 inputs.
    ok &= test_cuda_gemv<6>(64, 6144, 218);
    ok &= test_cuda_gemv_q8<6>(2560, 6144, 2, 219);
    ok &= test_cuda_gemv_q8<4>(512, 6144, 1, 220);
    ok &= test_cuda_gemv_q8<3>(512, 12288, 1, 221);
    if (!ok) {
        std::fprintf(stderr, "CUDA TESTS FAILED\n");
        return 1;
    }
    std::printf("ALL CUDA MS_1MAD TESTS PASSED\n");
    return 0;
}
