// tests/kernels/ms_gpu_parity.cpp - MS (1MAD trellis) dense projections through the engine's native_mmvq against the
// CPU kernels, on random trellis rows at the model's dense geometry, 1..8 columns.  The input is quantized to Q8_1 by
// native_quantize_q8_1 on the GPU; the reference rotates the same dequantized Q8_1 values on the CPU.  Also checks that
// the GGUF payload size of an MS tensor equals native_mmvq_weight_bytes.
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "kernels/ms_1mad_avx2.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace ms = adamant::kernels::ms;
namespace gpu = basalt::kernels;

namespace {
int g_fail = 0;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

uint16_t to_fp16(float f) { return __half_as_ushort(__float2half(f)); }

// A random MS payload: n_out trellis rows of n_in states, then int8 signs[n_in].
std::vector<uint8_t> random_payload(int k, int n_in, int n_out, std::mt19937& rng) {
    const size_t stride = ms::ms_row_bytes(n_in, k);
    std::vector<uint8_t> p((size_t) n_out * stride + n_in, 0);
    std::uniform_real_distribution<float> dd(0.001f, 0.02f);
    for (int r = 0; r < n_out; ++r) {
        uint8_t* row = p.data() + (size_t) r * stride;
        const uint16_t d16 = to_fp16(dd(rng)), s0 = (uint16_t) (rng() & ms::kStateMask);
        std::memcpy(row, &d16, 2);
        std::memcpy(row + 2, &s0, 2);
        for (size_t b = 4; b < 4 + ((size_t) (n_in - 1) * k + 7) / 8; ++b) row[b] = (uint8_t) rng();
    }
    for (int c = 0; c < n_in; ++c) p[(size_t) n_out * stride + c] = (rng() & 1u) ? 1 : (uint8_t) -1;
    return p;
}

template <int K>
void ref_gemv(const std::vector<uint8_t>& p, const float* x, float* y, int n_in, int n_out) {
    std::vector<float> xr(n_in);
    const int8_t* signs = (const int8_t*) p.data() + (size_t) n_out * ms::ms_row_bytes(n_in, K);
    ms::fwht_rotate_input(x, signs, xr.data(), n_in, basalt::ms_hadamard_block((uint64_t) n_in));
    ms::ms_1mad_gemv<K>(p.data(), xr.data(), y, n_out, n_in);
}

template <int K>
void run(int n_in, int n_out, int ncols, uint32_t seed) {
    const int type = 120 + K;
    std::mt19937 rng(seed);
    const std::vector<uint8_t> payload = random_payload(K, n_in, n_out, rng);

    basalt::TensorInfo ti;
    ti.shape = {(uint64_t) n_in, (uint64_t) n_out};
    ti.type = (uint32_t) type;
    const uint64_t gguf_bytes = basalt::tensor_payload_bytes(ti);
    const size_t mmvq_bytes = gpu::native_mmvq_weight_bytes(type, n_in, n_out);
    if (!gpu::native_mmvq_supported(type) || gguf_bytes != payload.size() || mmvq_bytes != payload.size()) {
        std::printf("  %s %dx%d: payload %zu, gguf %llu, mmvq %zu FAIL\n", basalt::ggml_type_name(type), n_out, n_in,
                    payload.size(), (unsigned long long) gguf_bytes, mmvq_bytes);
        ++g_fail;
        return;
    }

    std::vector<float> x((size_t) n_in * ncols);
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    for (float& v : x) v = dx(rng);

    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    void *dw, *dq;
    float *dx_, *dy;
    const size_t q_bytes = gpu::native_q8_1_bytes(n_in, ncols);
    CK(cudaMalloc(&dw, payload.size()));
    CK(cudaMalloc(&dq, q_bytes));
    CK(cudaMalloc(&dx_, x.size() * sizeof(float)));
    CK(cudaMalloc(&dy, (size_t) n_out * ncols * sizeof(float)));
    CK(cudaMemcpy(dw, payload.data(), payload.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx_, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    gpu::native_quantize_q8_1(dx_, dq, n_in, ncols, s);
    gpu::native_mmvq(type, dw, dq, dy, n_in, n_out, ncols, s);
    CK(cudaStreamSynchronize(s));

    std::vector<uint8_t> q(q_bytes);
    std::vector<float> y((size_t) n_out * ncols);
    CK(cudaMemcpy(q.data(), dq, q_bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaFree(dw));
    CK(cudaFree(dq));
    CK(cudaFree(dx_));
    CK(cudaFree(dy));
    CK(cudaStreamDestroy(s));

    // the reference input: the same Q8_1 blocks (fp16 d, fp16 s, int8 qs[32]) dequantized
    double err = 0, mag = 0;
    std::vector<float> xd(n_in), yr(n_out);
    for (int j = 0; j < ncols; ++j) {
        for (int i = 0; i < n_in; ++i) {
            const uint8_t* b = q.data() + ((size_t) j * (n_in / 32) + i / 32) * 36;
            uint16_t d16;
            std::memcpy(&d16, b, 2);
            xd[i] = ms::fp16_to_fp32(d16) * (float) (int8_t) b[4 + i % 32];
        }
        ref_gemv<K>(payload, xd.data(), yr.data(), n_in, n_out);
        for (int r = 0; r < n_out; ++r) {
            err = std::max(err, (double) std::fabs(y[(size_t) j * n_out + r] - yr[r]));
            mag = std::max(mag, (double) std::fabs(yr[r]));
        }
    }
    const double rel = err / (mag + 1e-30), tol = 1e-5;
    const bool ok = rel <= tol;
    std::printf("  %s %5dx%-5d c%d  max err %.2e of max |y| %.2e  (rel %.1e, tol %.0e) %s\n",
                basalt::ggml_type_name(type), n_out, n_in, ncols, err, mag, rel, tol, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    run<6>(2560, 2560, 1, 1);    // a GDN / attention input projection
    run<6>(2560, 4096, 8, 2);    // a verify window's columns
    run<6>(6144, 2560, 2, 3);    // ssm_out / attn_output: rows past one shared-memory chunk
    run<6>(2560, 32768, 1, 4);   // a head slice
    run<4>(2560, 640, 3, 5);     // shared-expert gate / up
    run<4>(640, 2560, 1, 6);     // shared-expert down: Hadamard block 128
    run<3>(2560, 640, 4, 7);
    std::printf(g_fail ? "ms_gpu_parity: %d FAILED\n" : "ms_gpu_parity: all ok\n", g_fail);
    return g_fail ? 1 : 0;
}
