// tests/kernels/ms_gpu_parity.cpp - MS (1MAD trellis) dense projections through the engine's native_mmvq against the
// CPU kernels, on random trellis rows at the model's dense geometry, 1..8 columns.  The input is quantized to Q8_1 by
// native_quantize_q8_1 on the GPU; the reference rotates the same dequantized Q8_1 values on the CPU.  Also checks that
// the GGUF payload size of an MS tensor equals native_mmvq_weight_bytes, and runs the prompt path's fp16 dequantizer
// (ms_dequant_dense_f16) on a row slice: its rows times x against the same CPU reference.
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/kernels/ms_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "kernels/ms_1mad_avx2.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
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
// native_mmvq on 8 columns against each column alone: the verifier needs a column's value whatever rides along.
// Then the time of one call at 1 and 8 columns (the weights decoded once a column chunk).
template <int K>
void run_cols(int n_in, int n_out, uint32_t seed) {
    const int type = 120 + K, NC = 8;
    std::mt19937 rng(seed);
    const std::vector<uint8_t> payload = random_payload(K, n_in, n_out, rng);
    std::vector<float> x((size_t) n_in * NC);
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    for (float& v : x) v = dx(rng);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    void *dw, *dq;
    float *dx_, *dy, *d1;
    CK(cudaMalloc(&dw, payload.size()));
    CK(cudaMalloc(&dq, gpu::native_q8_1_bytes(n_in, NC)));
    CK(cudaMalloc(&dx_, x.size() * 4));
    CK(cudaMalloc(&dy, (size_t) n_out * NC * 4));
    CK(cudaMalloc(&d1, (size_t) n_out * 4));
    CK(cudaMemcpy(dw, payload.data(), payload.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx_, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    gpu::native_quantize_q8_1(dx_, dq, n_in, NC, s);
    gpu::native_mmvq(type, dw, dq, dy, n_in, n_out, NC, s);
    std::vector<float> all((size_t) n_out * NC), one((size_t) n_out);
    CK(cudaStreamSynchronize(s));
    CK(cudaMemcpy(all.data(), dy, all.size() * 4, cudaMemcpyDeviceToHost));
    int diff = 0;
    for (int c = 0; c < NC; ++c) {
        gpu::native_mmvq(type, dw, (const uint8_t*) dq + (size_t) c * (n_in / 32) * 36, d1, n_in, n_out, 1, s);
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(one.data(), d1, one.size() * 4, cudaMemcpyDeviceToHost));
        diff += std::memcmp(one.data(), all.data() + (size_t) c * n_out, one.size() * 4) != 0;
    }
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const double mb = (double) payload.size() / 1e6;
    std::printf("  %s %5dx%-5d columns alone vs 8: %s; bench", basalt::ggml_type_name(type), n_out, n_in,
                diff ? "DIFF FAIL" : "bitwise equal");
    for (int nc : {1, 2, 3, 4, 8}) {
        const int it = 50;
        gpu::native_mmvq(type, dw, dq, dy, n_in, n_out, nc, s);
        CK(cudaEventRecord(e0, s));
        for (int i = 0; i < it; ++i) gpu::native_mmvq(type, dw, dq, dy, n_in, n_out, nc, s);
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms_ = 0;
        CK(cudaEventElapsedTime(&ms_, e0, e1));
        const double us = ms_ * 1e3 / it;
        std::printf(" c%d %.1f us (%.0f GB/s)", nc, us, mb / us * 1e3);
    }
    std::printf("\n");
    if (diff) ++g_fail;
    for (void* p : {dw, dq, (void*) dx_, (void*) dy, (void*) d1}) cudaFree(p);
    CK(cudaStreamDestroy(s));
}

// The prompt path's dequantizer of one expert blob (2560 x 640): microseconds an expert, fp16 bytes written a second.
template <int K>
void bench_dequant_expert() {
    const int type = 120 + K, H = 2560, FF = 640;
    const gpu::NativeExpertLayout L = gpu::native_expert_layout(type, type, H, FF, 0.f);
    std::mt19937 rng(31);
    std::vector<uint8_t> b = random_payload(K, H, FF, rng), u = random_payload(K, H, FF, rng),
                         d = random_payload(K, FF, H, rng);
    b.insert(b.end(), u.begin(), u.end());
    b.insert(b.end(), d.begin(), d.end());
    void* db;
    uint16_t *gu, *dd;
    CK(cudaMalloc(&db, b.size()));
    CK(cudaMalloc(&gu, (size_t) 2 * FF * H * 2));
    CK(cudaMalloc(&dd, (size_t) H * FF * 2));
    CK(cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    gpu::ms_dequant_expert_f16(type, db, H, FF, L.up_off, L.down_off, gu, dd, s);
    // the layout the prompt path's GEMMs read: gate row r at 2 r, up row r at 2 r + 1, then down, each the bits of
    // the single-tensor dequantizer
    {
        uint16_t* one;
        CK(cudaMalloc(&one, (size_t) H * H * 2));
        std::vector<uint16_t> got((size_t) 2 * FF * H), gd((size_t) H * FF), ref((size_t) H * H);
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(got.data(), gu, got.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gd.data(), dd, gd.size() * 2, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int p = 0; p < 2; ++p) {
            gpu::ms_dequant_dense_f16(type, (const uint8_t*) db + (p ? L.up_off : 0), H, FF, 0, FF, one, s);
            CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(ref.data(), one, (size_t) FF * H * 2, cudaMemcpyDeviceToHost));
            for (int r = 0; r < FF; ++r)
                bad += std::memcmp(&got[(size_t) (2 * r + p) * H], &ref[(size_t) r * H], (size_t) H * 2) != 0;
        }
        gpu::ms_dequant_dense_f16(type, (const uint8_t*) db + L.down_off, FF, H, 0, H, one, s);
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(ref.data(), one, (size_t) H * FF * 2, cudaMemcpyDeviceToHost));
        bad += std::memcmp(gd.data(), ref.data(), gd.size() * 2) != 0;
        std::printf("  %s expert dequant layout (gate/up interleaved, down): %s\n", basalt::ggml_type_name(type),
                    bad ? "FAIL" : "bitwise equal");
        if (bad) ++g_fail;
        cudaFree(one);
        // the Q8_0 conversion: every value within half a step (plus the fp16 copy's rounding) of the fp16 weights
        const size_t bgu = (size_t) 2 * FF * (H / 32) * 34, bd = (size_t) H * (FF / 32) * 34;
        void *qgu, *qd;
        CK(cudaMalloc(&qgu, bgu));
        CK(cudaMalloc(&qd, bd));
        gpu::ms_expert_to_q8_0(type, db, H, FF, L.up_off, L.down_off, qgu, qd, s);
        CK(cudaStreamSynchronize(s));
        std::vector<uint8_t> hq(bgu), hd(bd);
        CK(cudaMemcpy(hq.data(), qgu, bgu, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hd.data(), qd, bd, cudaMemcpyDeviceToHost));
        double worst = 0;   // in units of the block's step
        auto cmp = [&](const std::vector<uint8_t>& q, int rows, int cols, auto ref_of) {
            for (int r = 0; r < rows; ++r)
                for (int bk = 0; bk < cols / 32; ++bk) {
                    const uint8_t* blk = q.data() + ((size_t) r * (cols / 32) + bk) * 34;
                    uint16_t d16;
                    std::memcpy(&d16, blk, 2);
                    const float dd = ms::fp16_to_fp32(d16);
                    for (int i = 0; i < 32; ++i) {
                        const float ref = ref_of(r, bk * 32 + i);
                        const double err = std::fabs(dd * (float) (int8_t) blk[2 + i] - ref) -
                                           2e-3 * std::fabs(ref);   // the fp16 copy's rounding
                        if (dd > 0) worst = std::max(worst, err / dd);
                    }
                }
        };
        cmp(hq, 2 * FF, H, [&](int r, int c) {   // gate rows, then up rows; got holds them interleaved
            const int src = r < FF ? 2 * r : 2 * (r - FF) + 1;
            return ms::fp16_to_fp32(got[(size_t) src * H + c]);
        });
        cmp(hd, H, FF, [&](int r, int c) { return ms::fp16_to_fp32(gd[(size_t) r * FF + c]); });
        const bool ok = worst <= 0.5 + 1e-3;
        std::printf("  %s expert to Q8_0: worst error %.3f of a step (tol 0.5) %s\n", basalt::ggml_type_name(type), worst,
                    ok ? "ok" : "FAIL");
        if (!ok) ++g_fail;
        cudaEvent_t q0, q1;
        CK(cudaEventCreate(&q0));
        CK(cudaEventCreate(&q1));
        CK(cudaEventRecord(q0, s));
        for (int i = 0; i < 100; ++i) gpu::ms_expert_to_q8_0(type, db, H, FF, L.up_off, L.down_off, qgu, qd, s);
        CK(cudaEventRecord(q1, s));
        CK(cudaEventSynchronize(q1));
        float qms = 0;
        CK(cudaEventElapsedTime(&qms, q0, q1));
        std::printf("  bench %s expert to Q8_0: %.1f us an expert alone", basalt::ggml_type_name(type), qms * 10.0);
        {   // a prompt group: 16 blobs in one launch (the same blob 16 times; the timing is what counts here)
            void *g16, *d16;
            CK(cudaMalloc(&g16, bgu * 16));
            CK(cudaMalloc(&d16, bd * 16));
            const uint8_t* bl[16];
            for (auto& p : bl) p = (const uint8_t*) db;
            gpu::ms_experts_to_q8_0(type, bl, 16, H, FF, L.up_off, L.down_off, g16, bgu, d16, bd, s);
            CK(cudaEventRecord(q0, s));
            for (int i = 0; i < 20; ++i) gpu::ms_experts_to_q8_0(type, bl, 16, H, FF, L.up_off, L.down_off, g16, bgu, d16, bd, s);
            CK(cudaEventRecord(q1, s));
            CK(cudaEventSynchronize(q1));
            CK(cudaEventElapsedTime(&qms, q0, q1));
            std::vector<uint8_t> last(bgu);   // the 16th slot holds the same rows as the single conversion
            CK(cudaMemcpy(last.data(), (uint8_t*) g16 + 15 * bgu, bgu, cudaMemcpyDeviceToHost));
            const bool same = last == hq;
            std::printf(", %.1f us an expert in a group of 16 (slot 16 %s)\n", qms * 1e3 / (20 * 16),
                        same ? "bitwise equal" : "DIFF FAIL");
            if (!same) ++g_fail;
            cudaFree(g16);
            cudaFree(d16);
        }
        cudaFree(qgu);
        cudaFree(qd);
    }
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const int it = 100;
    CK(cudaEventRecord(e0, s));
    for (int i = 0; i < it; ++i) gpu::ms_dequant_expert_f16(type, db, H, FF, L.up_off, L.down_off, gu, dd, s);
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    float ms_ = 0;
    CK(cudaEventElapsedTime(&ms_, e0, e1));
    const double us = ms_ * 1e3 / it;
    std::printf("  bench %s expert dequant f16: %.1f us an expert (%.0f GB/s of fp16 written)\n",
                basalt::ggml_type_name(type), us, 3.0 * H * FF * 2 / (us * 1e3));
    for (void* p : {db, (void*) gu, (void*) dd}) cudaFree(p);
    CK(cudaStreamDestroy(s));
}

// native_expert_grouped on MS experts (2560 x 640): groups of 1, 3, 6 (two passes of the block's 4) and 2 entries on
// random tokens and output rows, against gate/up -> SwiGLU -> down on the CPU from the same dequantized q8_1 inputs.
template <int K>
void run_grouped(float limit, uint32_t seed) {
    const int type = 120 + K, H = 2560, FF = 640, NT = 5;
    const gpu::NativeExpertLayout L = gpu::native_expert_layout(type, type, H, FF, limit);
    if (!gpu::native_expert_supported(type, type, H, FF) ||
        L.bytes != 2 * (FF * ms::ms_row_bytes(H, K) + H) + H * ms::ms_row_bytes(FF, K) + FF) {
        std::printf("  %s grouped: unsupported or blob %zu FAIL\n", basalt::ggml_type_name(type), L.bytes);
        ++g_fail;
        return;
    }
    std::mt19937 rng(seed);
    const std::vector<int> sizes = {1, 3, 6, 2};
    const int G = (int) sizes.size();
    std::vector<std::vector<uint8_t>> blobs;
    for (int g = 0; g < G; ++g) {
        std::vector<uint8_t> b = random_payload(K, H, FF, rng), u = random_payload(K, H, FF, rng),
                             d = random_payload(K, FF, H, rng);
        b.insert(b.end(), u.begin(), u.end());
        b.insert(b.end(), d.begin(), d.end());
        blobs.push_back(std::move(b));
    }
    std::vector<int32_t> start(G + 1, 0), tok, dst;
    for (int g = 0; g < G; ++g) start[g + 1] = start[g] + sizes[g];
    const int E = start[G];
    for (int e = 0; e < E; ++e) {
        tok.push_back((int32_t) (rng() % NT));
        dst.push_back(e);
    }
    std::shuffle(dst.begin(), dst.end(), rng);
    std::vector<float> x((size_t) NT * H);
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    for (float& v : x) v = dx(rng);

    std::vector<void*> dblob(G);
    std::vector<unsigned long long> hp(G);
    for (int g = 0; g < G; ++g) {
        CK(cudaMalloc(&dblob[g], blobs[g].size()));
        CK(cudaMemcpy(dblob[g], blobs[g].data(), blobs[g].size(), cudaMemcpyHostToDevice));
        hp[g] = (unsigned long long) dblob[g];
    }
    unsigned long long* d_ptr;
    int32_t *d_start, *d_ng, *d_dst, *d_tok;
    float *d_x, *d_out;
    void *d_q, *d_scr;
    const size_t qb = gpu::native_q8_1_bytes(H, NT);
    CK(cudaMalloc(&d_ptr, G * 8));
    CK(cudaMalloc(&d_start, (G + 1) * 4));
    CK(cudaMalloc(&d_ng, 4));
    CK(cudaMalloc(&d_dst, E * 4));
    CK(cudaMalloc(&d_tok, E * 4));
    CK(cudaMalloc(&d_x, x.size() * 4));
    CK(cudaMalloc(&d_q, qb));
    CK(cudaMalloc(&d_out, (size_t) E * H * 4));
    CK(cudaMalloc(&d_scr, gpu::native_expert_scratch_bytes(E, FF)));
    CK(cudaMemcpy(d_ptr, hp.data(), G * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_start, start.data(), (G + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ng, &G, 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dst, dst.data(), E * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tok, tok.data(), E * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    gpu::native_quantize_q8_1(d_x, d_q, H, NT, s);
    if (limit > 0.f)
        gpu::native_expert_grouped_clamped(L, limit, d_ptr, d_start, d_ng, d_dst, d_tok, G, E, d_q, d_scr, d_out, s);
    else
        gpu::native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, G, E, d_q, d_scr, d_out, s);
    CK(cudaStreamSynchronize(s));
    std::vector<uint8_t> q(qb);
    std::vector<float> got((size_t) E * H);
    CK(cudaMemcpy(q.data(), d_q, qb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaStreamDestroy(s));
    for (void* pp : dblob) cudaFree(pp);
    for (void* pp : {(void*) d_ptr, (void*) d_start, (void*) d_ng, (void*) d_dst, (void*) d_tok, (void*) d_x, d_q,
                     (void*) d_out, d_scr})
        cudaFree(pp);

    double err = 0, mag = 0;
    std::vector<float> xd(H), gg(FF), uu(FF), hh(FF), y(H);
    for (int g = 0; g < G; ++g) {
        const std::vector<uint8_t>& b = blobs[g];
        const std::vector<uint8_t> pg(b.begin(), b.begin() + L.up_off), pu(b.begin() + L.up_off, b.begin() + L.down_off),
            pd(b.begin() + L.down_off, b.end());
        for (int e = start[g]; e < start[g + 1]; ++e) {
            for (int i = 0; i < H; ++i) {
                const uint8_t* qb8 = q.data() + ((size_t) tok[e] * (H / 32) + i / 32) * 36;
                uint16_t d16;
                std::memcpy(&d16, qb8, 2);
                xd[i] = ms::fp16_to_fp32(d16) * (float) (int8_t) qb8[4 + i % 32];
            }
            ref_gemv<K>(pg, xd.data(), gg.data(), H, FF);
            ref_gemv<K>(pu, xd.data(), uu.data(), H, FF);
            for (int i = 0; i < FF; ++i) {
                float a = gg[i], c = uu[i];
                if (limit > 0.f) {
                    a = std::fmin(a, limit);
                    c = std::fmax(std::fmin(c, limit), -limit);
                }
                hh[i] = a / (1.f + std::exp(-a)) * c;
            }
            ref_gemv<K>(pd, hh.data(), y.data(), FF, H);
            for (int r = 0; r < H; ++r) {
                err = std::max(err, (double) std::fabs(got[(size_t) dst[e] * H + r] - y[r]));
                mag = std::max(mag, (double) std::fabs(y[r]));
            }
        }
    }
    const double rel = err / (mag + 1e-30), tol = 1e-4;
    const bool ok = rel <= tol;
    std::printf("  %s grouped experts limit %g, %d groups, %d entries  max err %.2e of max |y| %.2e  (rel %.1e, tol %.0e) %s\n",
                basalt::ggml_type_name(type), limit, G, E, err, mag, rel, tol, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// One layer's experts as decode runs them: `groups` experts of `per` entries each, blobs rotating through more memory
// than the L2 holds, replayed in a CUDA graph (the engine's launch form).
template <int K>
void bench_grouped(int groups, int per) {
    const int type = 120 + K, H = 2560, FF = 640, SETS = 8;
    const gpu::NativeExpertLayout L = gpu::native_expert_layout(type, type, H, FF, 0.f);
    std::mt19937 rng(9);
    std::vector<uint8_t> b = random_payload(K, H, FF, rng), u = random_payload(K, H, FF, rng),
                         d = random_payload(K, FF, H, rng);
    b.insert(b.end(), u.begin(), u.end());
    b.insert(b.end(), d.begin(), d.end());
    std::vector<unsigned long long> hp((size_t) groups * SETS);
    std::vector<void*> dbl(hp.size());
    for (size_t i = 0; i < hp.size(); ++i) {
        CK(cudaMalloc(&dbl[i], b.size()));
        CK(cudaMemcpy(dbl[i], b.data(), b.size(), cudaMemcpyHostToDevice));
        hp[i] = (unsigned long long) dbl[i];
    }
    const int E = groups * per;
    std::vector<int32_t> start(groups + 1), dst(E), tok(E);
    for (int g = 0; g <= groups; ++g) start[g] = g * per;
    for (int e = 0; e < E; ++e) {
        dst[e] = e;
        tok[e] = e % per;
    }
    unsigned long long* d_ptr;
    int32_t *d_start, *d_ng, *d_dst, *d_tok;
    float *d_x, *d_out;
    void *d_q, *d_scr;
    CK(cudaMalloc(&d_ptr, hp.size() * 8));
    CK(cudaMalloc(&d_start, (groups + 1) * 4));
    CK(cudaMalloc(&d_ng, 4));
    CK(cudaMalloc(&d_dst, E * 4));
    CK(cudaMalloc(&d_tok, E * 4));
    CK(cudaMalloc(&d_x, (size_t) per * H * 4));
    CK(cudaMalloc(&d_q, gpu::native_q8_1_bytes(H, per)));
    CK(cudaMalloc(&d_out, (size_t) E * H * 4));
    CK(cudaMalloc(&d_scr, gpu::native_expert_scratch_bytes(E, FF)));
    CK(cudaMemcpy(d_ptr, hp.data(), hp.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_start, start.data(), (groups + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ng, &groups, 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dst, dst.data(), E * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tok, tok.data(), E * 4, cudaMemcpyHostToDevice));
    std::vector<float> x((size_t) per * H, 0.5f);
    CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    gpu::native_quantize_q8_1(d_x, d_q, H, per, st);
    constexpr int IT = 200, PER = 8;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < PER; ++i)
        gpu::native_expert_grouped(L, d_ptr + (i % SETS) * groups, d_start, d_ng, d_dst, d_tok, groups, E, d_q, d_scr,
                                   d_out, st);
    CK(cudaStreamEndCapture(st, &graph));
    CK(cudaGraphInstantiate(&exec, graph, 0));
    for (int i = 0; i < 4; ++i) CK(cudaGraphLaunch(exec, st));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaStreamSynchronize(st));
    CK(cudaEventRecord(e0, st));
    for (int i = 0; i < IT / PER; ++i) CK(cudaGraphLaunch(exec, st));
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms_ = 0;
    CK(cudaEventElapsedTime(&ms_, e0, e1));
    const double us = ms_ * 1e3 / (IT / PER * PER);
    std::printf("  bench %s grouped: %.1f us per layer (%d experts x %d entries, %.1f MB) = %.0f GB/s\n",
                basalt::ggml_type_name(type), us, groups, per, (double) L.bytes * groups / 1e6,
                (double) L.bytes * groups / (us * 1e-6) / 1e9);
    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(graph));
    CK(cudaStreamDestroy(st));
    for (void* pp : dbl) cudaFree(pp);
    for (void* pp : {(void*) d_ptr, (void*) d_start, (void*) d_ng, (void*) d_dst, (void*) d_tok, (void*) d_x, d_q,
                     (void*) d_out, d_scr})
        cudaFree(pp);
}

// ms_dequant_dense_f16 rows [row0, row0 + rows) times a float x, against the CPU rotation + GEMV on the same x.
template <int K>
void run_dequant(int n_in, int n_out, int row0, int rows, uint32_t seed) {
    const int type = 120 + K;
    std::mt19937 rng(seed);
    const std::vector<uint8_t> payload = random_payload(K, n_in, n_out, rng);
    std::vector<float> x(n_in), yr(n_out);
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    for (float& v : x) v = dx(rng);
    ref_gemv<K>(payload, x.data(), yr.data(), n_in, n_out);

    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    void* dw;
    uint16_t* dd;
    CK(cudaMalloc(&dw, payload.size()));
    CK(cudaMalloc(&dd, (size_t) rows * n_in * 2));
    CK(cudaMemcpy(dw, payload.data(), payload.size(), cudaMemcpyHostToDevice));
    gpu::ms_dequant_dense_f16(type, dw, n_in, n_out, row0, rows, dd, s);
    CK(cudaStreamSynchronize(s));
    std::vector<uint16_t> w((size_t) rows * n_in);
    CK(cudaMemcpy(w.data(), dd, w.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaFree(dw));
    CK(cudaFree(dd));
    CK(cudaStreamDestroy(s));

    double err = 0, mag = 0;
    for (int r = 0; r < rows; ++r) {
        double y = 0;
        for (int c = 0; c < n_in; ++c) y += (double) ms::fp16_to_fp32(w[(size_t) r * n_in + c]) * x[c];
        err = std::max(err, std::fabs(y - yr[row0 + r]));
        mag = std::max(mag, (double) std::fabs(yr[row0 + r]));
    }
    const double rel = err / (mag + 1e-30), tol = 1e-3;   // the weights rounded to fp16
    const bool ok = rel <= tol;
    std::printf("  %s %5dx%-5d rows %d+%d dequant f16  max err %.2e of max |y| %.2e  (rel %.1e, tol %.0e) %s\n",
                basalt::ggml_type_name(type), n_out, n_in, row0, rows, err, mag, rel, tol, ok ? "ok" : "FAIL");
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
    run_dequant<6>(2560, 4096, 0, 4096, 8);
    run_dequant<6>(6144, 2560, 1000, 700, 9);   // a row slice of a wide input
    run_dequant<4>(640, 2560, 17, 300, 10);     // Hadamard block 128
    run_dequant<3>(2560, 640, 0, 640, 11);
    run_cols<6>(2560, 10240, 20);   // GDN qkv
    run_cols<6>(6144, 2560, 21);    // ssm_out
    run_cols<6>(2560, 248320, 22);  // the head
    run_cols<4>(640, 2560, 23);     // a shared expert's down
    bench_dequant_expert<4>();
    bench_dequant_expert<3>();
    run_grouped<4>(0.f, 12);
    run_grouped<3>(0.f, 13);
    run_grouped<6>(0.f, 14);
    run_grouped<4>(7.f, 15);
    bench_grouped<4>(10, 1);
    bench_grouped<3>(10, 1);
    bench_grouped<4>(20, 2);
    std::printf(g_fail ? "ms_gpu_parity: %d FAILED\n" : "ms_gpu_parity: all ok\n", g_fail);
    return g_fail ? 1 : 0;
}
