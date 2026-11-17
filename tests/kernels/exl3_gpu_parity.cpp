// tests/kernels/exl3_gpu_parity.cpp - EXL3 experts on the GPU against the CPU kernels (themselves checked against a
// bit-by-bit oracle in exl3_cpu_test), on random weights in the experts' geometry, all four sign placements.
//   (a) native_expert_grouped (q8_1 activations): the whole expert, gate/up -> SwiGLU -> down, plain and limited
//   (b) the prompt path's fp16 dequantizers, as matrix-vector products
//   (c) dense projections (native_mmvq on an EXL3 type, 1..8 columns) and the dense row-sliced dequantizer
#include "basalt/kernels/cpu/exl3.hpp"
#include "basalt/kernels/exl3_kernels.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"

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

namespace cpu = basalt::kernels::cpu;
namespace gpu = basalt::kernels;

namespace {
constexpr int T = 3, H = 2560, FF = 640;
int g_fail = 0;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

void report(const char* what, double err, double tol) {
    const bool ok = err <= tol;
    std::printf("  %-36s rel err %.2e (tol %.0e) %s\n", what, err, tol, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

// dtype: the down projection's type when it differs from the gate/up's (the drafter's experts; -1: the same)
void run(int type, float limit, int variant, int dtype = -1) {
    cpu::exl3_set_variant(variant);
    if (dtype < 0) dtype = type;
    const int w = cpu::exl3_words(type), wd = cpu::exl3_words(dtype);
    std::printf("type %d/%d (%g/%g bits) limit %g variant %d\n", type, dtype, w / 16.0, wd / 16.0, limit, variant);
    const gpu::NativeExpertLayout L = gpu::native_expert_layout(type, dtype, H, FF, limit);
    std::mt19937 rng(100u + (unsigned) w + (unsigned) variant);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint8_t> blob(L.bytes);
    for (auto& b : blob) b = (uint8_t) rng();
    // suh / svh as fp16 of sane magnitudes
    auto fill = [&](size_t rec, int k, int n, int wk) {
        cpu::Exl3Proj p = cpu::exl3_proj_at(blob.data() + rec, k, n, wk);
        __half* suh = (__half*) p.suh;
        __half* svh = (__half*) p.svh;
        for (int i = 0; i < k; ++i) suh[i] = __float2half(nd(rng) * 0.02f);
        for (int i = 0; i < n; ++i) svh[i] = __float2half(nd(rng));
    };
    fill(0, H, FF, w);
    fill(L.up_off, H, FF, w);
    fill(L.down_off, FF, H, wd);

    std::vector<float> x((size_t) T * H);
    for (auto& v : x) v = nd(rng);
    uint8_t* d_blob;
    float *d_x, *d_out, *d_xs;
    void *d_q, *d_scr;
    unsigned long long* d_ptr;
    int32_t *d_start, *d_ng, *d_dst, *d_tok;
    CK(cudaMalloc(&d_blob, blob.size()));
    CK(cudaMemcpy(d_blob, blob.data(), blob.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_x, x.size() * 4));
    CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    const size_t qbytes = (size_t) T * (H / 32) * 36;
    CK(cudaMalloc(&d_q, qbytes));
    gpu::quantize_q8_1_rows(d_x, T, H, d_q, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> hq(qbytes);
    CK(cudaMemcpy(hq.data(), d_q, qbytes, cudaMemcpyDeviceToHost));
    std::vector<float> xh((size_t) T * H);   // exactly what the kernels see
    for (int t = 0; t < T; ++t)
        for (int b = 0; b < H / 32; ++b) {
            const uint8_t* blk = hq.data() + ((size_t) t * (H / 32) + b) * 36;
            __half d;
            std::memcpy(&d, blk, 2);
            for (int i = 0; i < 32; ++i) xh[(size_t) t * H + b * 32 + i] = __half2float(d) * (float) (int8_t) blk[4 + i];
        }
    CK(cudaMalloc(&d_out, (size_t) T * H * 4));
    CK(cudaMalloc(&d_scr, gpu::native_expert_scratch_bytes(T, FF)));
    CK(cudaMalloc(&d_ptr, 8));
    CK(cudaMalloc(&d_start, 8));
    CK(cudaMalloc(&d_ng, 4));
    CK(cudaMalloc(&d_dst, T * 4));
    CK(cudaMalloc(&d_tok, T * 4));
    const unsigned long long hp = (unsigned long long) d_blob;
    const int32_t hs[2] = {0, T}, hn = 1, hd[T] = {0, 1, 2};
    CK(cudaMemcpy(d_ptr, &hp, 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_start, hs, 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ng, &hn, 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dst, hd, T * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tok, hd, T * 4, cudaMemcpyHostToDevice));
    gpu::native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, 1, T, d_q, d_scr, d_out, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<float> got((size_t) T * H);
    CK(cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost));

    // the CPU reference on the same activations
    auto proj = [&](size_t rec, int k, int n, const float* in, float* out, int wk) {
        const cpu::Exl3Proj p = cpu::exl3_proj_at(blob.data() + rec, k, n, wk);
        std::vector<float> x2((size_t) k);
        cpu::exl3_prepare_input(in, p.suh, k, variant, x2.data());
        cpu::exl3_rows_ref(p, x2.data(), out, 0, n, variant);
    };
    std::vector<float> ref((size_t) T * H);
    for (int t = 0; t < T; ++t) {
        std::vector<float> g(FF), u(FF), h(FF);
        proj(0, H, FF, &xh[(size_t) t * H], g.data(), w);
        proj(L.up_off, H, FF, &xh[(size_t) t * H], u.data(), w);
        for (int i = 0; i < FF; ++i) {
            float a = g[(size_t) i], b = u[(size_t) i];
            if (limit > 0.f) { a = std::fmin(a, limit); b = std::fmax(std::fmin(b, limit), -limit); }
            h[(size_t) i] = (a / (1.f + std::exp(-a))) * b;
        }
        proj(L.down_off, FF, H, h.data(), &ref[(size_t) t * H], wd);
    }
    report("grouped expert (q8_1 in)", rel(got, ref), 2e-4);

    // (b) the prompt path's fp16 weights: W[row][k] . x against the CPU projections
    uint16_t *d_gu, *d_d;
    CK(cudaMalloc(&d_gu, (size_t) 2 * FF * H * 2));
    CK(cudaMalloc(&d_d, (size_t) H * FF * 2));
    gpu::exl3_dequant_gu_f16(type, d_blob, L.up_off, FF, H, d_gu, nullptr);
    gpu::exl3_dequant_down_f16(dtype, d_blob + L.down_off, H, FF, d_d, nullptr);
    CK(cudaDeviceSynchronize());
    if (variant == 0 && limit == 0) {   // what one expert's prompt-path dequant costs (a layer has 512 of them)
        cudaEvent_t t0, t1;
        CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
        CK(cudaEventRecord(t0));
        for (int i = 0; i < 200; ++i) {
            gpu::exl3_dequant_gu_f16(type, d_blob, L.up_off, FF, H, d_gu, nullptr);
            gpu::exl3_dequant_down_f16(dtype, d_blob + L.down_off, H, FF, d_d, nullptr);
        }
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, t0, t1));
        {   // the same launches replayed from a CUDA graph: the GPU's own time per expert without the host's launch cost
            cudaStream_t gs;
            CK(cudaStreamCreate(&gs));
            cudaGraph_t gr;
            cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(gs, cudaStreamCaptureModeGlobal));
            for (int i = 0; i < 50; ++i) {
                gpu::exl3_dequant_gu_f16(type, d_blob, L.up_off, FF, H, d_gu, gs);
                gpu::exl3_dequant_down_f16(dtype, d_blob + L.down_off, H, FF, d_d, gs);
            }
            CK(cudaStreamEndCapture(gs, &gr));
            CK(cudaGraphInstantiate(&ge, gr, 0));
            CK(cudaGraphLaunch(ge, gs));
            CK(cudaStreamSynchronize(gs));
            cudaEvent_t g0, g1;
            CK(cudaEventCreate(&g0)); CK(cudaEventCreate(&g1));
            CK(cudaEventRecord(g0, gs));
            for (int r = 0; r < 4; ++r) CK(cudaGraphLaunch(ge, gs));
            CK(cudaEventRecord(g1, gs));
            CK(cudaEventSynchronize(g1));
            float gms = 0;
            CK(cudaEventElapsedTime(&gms, g0, g1));
            std::printf("  dequant from a CUDA graph: %.1f us per expert (the GPU alone)\n", gms * 1000.f / 200.f);
            CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(gr)); CK(cudaStreamDestroy(gs));
        }
        {   // where a CTA's cycles go
            unsigned long long p[6] = {};
            gpu::exl3_dequant_profile(true, true, nullptr);
            for (int i = 0; i < 50; ++i) {
                gpu::exl3_dequant_gu_f16(type, d_blob, L.up_off, FF, H, d_gu, nullptr);
                gpu::exl3_dequant_down_f16(dtype, d_blob + L.down_off, H, FF, d_d, nullptr);
            }
            gpu::exl3_dequant_profile(false, false, p);
            int khz = 0;
            cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);
            const double n = (double) p[5];
            std::printf("  dequant CTA cycles (avg of %.0f CTAs, %.2f GHz): decode %.0f, input transform %.0f, output transform %.0f, "
                        "store %.0f, total %.0f (= %.1f us)\n", n, khz / 1e6, p[0] / n, p[1] / n, p[2] / n, p[3] / n, p[4] / n,
                        p[4] / n / (khz / 1e3));
        }
        std::printf("  dequant timing: %.1f us per expert (gate+up+down) -> %.1f ms for 512 experts\n", ms * 1000.f / 200.f,
                    ms * 512.f / 200.f);
    }
    std::vector<uint16_t> gu((size_t) 2 * FF * H), dn((size_t) H * FF);
    CK(cudaMemcpy(gu.data(), d_gu, gu.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dn.data(), d_d, dn.size() * 2, cudaMemcpyDeviceToHost));
    const float* x0 = x.data();
    std::vector<float> yg(FF), yu(FF), rg(FF), ru(FF), yd(H), rd(H);
    for (int r = 0; r < FF; ++r) {
        double a = 0, b = 0;
        for (int k = 0; k < H; ++k) {
            __half hg, hu;
            std::memcpy(&hg, &gu[((size_t) 2 * r) * H + k], 2);
            std::memcpy(&hu, &gu[((size_t) 2 * r + 1) * H + k], 2);
            a += (double) __half2float(hg) * x0[k];
            b += (double) __half2float(hu) * x0[k];
        }
        yg[(size_t) r] = (float) a;
        yu[(size_t) r] = (float) b;
    }
    proj(0, H, FF, x0, rg.data(), w);
    proj(L.up_off, H, FF, x0, ru.data(), w);
    report("dequant gate (fp16 W . x)", rel(yg, rg), 3e-3);
    report("dequant up (fp16 W . x)", rel(yu, ru), 3e-3);
    for (int o = 0; o < H; ++o) {
        double a = 0;
        for (int k = 0; k < FF; ++k) {
            __half hd2;
            std::memcpy(&hd2, &dn[(size_t) o * FF + k], 2);
            a += (double) __half2float(hd2) * x0[k];
        }
        yd[(size_t) o] = (float) a;
    }
    proj(L.down_off, FF, H, x0, rd.data(), wd);
    report("dequant down (fp16 W . x)", rel(yd, rd), 3e-3);

    // (b2) the fused prompt-path projections (no W_full): M tokens of fp16 activations against the CPU projections
    for (int M : {1, 37, 160, 333}) {
        std::vector<uint16_t> hx((size_t) M * H), hh((size_t) M * FF);
        std::vector<float> fx((size_t) M * H), fh((size_t) M * FF);
        for (size_t i = 0; i < hx.size(); ++i) { __half hv = __float2half(nd(rng)); std::memcpy(&hx[i], &hv, 2); fx[i] = __half2float(hv); }
        for (size_t i = 0; i < hh.size(); ++i) { __half hv = __float2half(nd(rng)); std::memcpy(&hh[i], &hv, 2); fh[i] = __half2float(hv); }
        uint16_t *d_fx, *d_fh;
        float *d_fgu, *d_fdn;
        CK(cudaMalloc(&d_fx, hx.size() * 2)); CK(cudaMalloc(&d_fh, hh.size() * 2));
        CK(cudaMalloc(&d_fgu, (size_t) M * 2 * FF * 4)); CK(cudaMalloc(&d_fdn, (size_t) M * H * 4));
        CK(cudaMemcpy(d_fx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_fh, hh.data(), hh.size() * 2, cudaMemcpyHostToDevice));
        gpu::exl3_expert_gu_fused(type, d_blob, L.up_off, d_fx, M, d_fgu, nullptr);
        gpu::exl3_expert_down_fused(dtype, d_blob + L.down_off, d_fh, M, d_fdn, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<float> fgu((size_t) M * 2 * FF), fdn((size_t) M * H);
        CK(cudaMemcpy(fgu.data(), d_fgu, fgu.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(fdn.data(), d_fdn, fdn.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<float> a, b;
        std::vector<float> rg2(FF), ru2(FF), rd2(H);
        for (int t : {0, M / 2, M - 1}) {
            proj(0, H, FF, &fx[(size_t) t * H], rg2.data(), w);
            proj(L.up_off, H, FF, &fx[(size_t) t * H], ru2.data(), w);
            for (int r = 0; r < FF; ++r) {
                a.push_back(fgu[(size_t) t * 2 * FF + 2 * r]); b.push_back(rg2[(size_t) r]);
                a.push_back(fgu[(size_t) t * 2 * FF + 2 * r + 1]); b.push_back(ru2[(size_t) r]);
            }
        }
        char label[96];
        std::snprintf(label, sizeof label, "fused gate/up, %d tokens", M);
        report(label, rel(a, b), 3e-3);
        a.clear(); b.clear();
        for (int t : {0, M / 2, M - 1}) {
            proj(L.down_off, FF, H, &fh[(size_t) t * FF], rd2.data(), wd);
            for (int o = 0; o < H; ++o) { a.push_back(fdn[(size_t) t * H + o]); b.push_back(rd2[(size_t) o]); }
        }
        std::snprintf(label, sizeof label, "fused down, %d tokens", M);
        report(label, rel(a, b), 3e-3);
        if (variant == 0 && limit == 0 && (M == 160 || M == 333)) {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            for (int i = 0; i < 5; ++i) {
                gpu::exl3_expert_gu_fused(type, d_blob, L.up_off, d_fx, M, d_fgu, nullptr);
                gpu::exl3_expert_down_fused(dtype, d_blob + L.down_off, d_fh, M, d_fdn, nullptr);
            }
            cudaEventRecord(e0);
            for (int i = 0; i < 100; ++i) gpu::exl3_expert_gu_fused(type, d_blob, L.up_off, d_fx, M, d_fgu, nullptr);
            cudaEventRecord(e1);
            CK(cudaEventSynchronize(e1));
            float ms_gu = 0, ms_dn = 0;
            cudaEventElapsedTime(&ms_gu, e0, e1);
            cudaEventRecord(e0);
            for (int i = 0; i < 100; ++i) gpu::exl3_expert_down_fused(dtype, d_blob + L.down_off, d_fh, M, d_fdn, nullptr);
            cudaEventRecord(e1);
            CK(cudaEventSynchronize(e1));
            cudaEventElapsedTime(&ms_dn, e0, e1);
            std::printf("  fused, %d tokens: gate/up %.1f us, down %.1f us (the dequant path: 74 us + cuBLAS ~50 us at 160)\n", M,
                        ms_gu * 10.f, ms_dn * 10.f);
        }
        for (void* p2 : {(void*) d_fx, (void*) d_fh, (void*) d_fgu, (void*) d_fdn}) cudaFree(p2);
    }

    for (void* p : {(void*) d_blob, (void*) d_x, (void*) d_out, (void*) d_q, d_scr, (void*) d_ptr, (void*) d_start,
                    (void*) d_ng, (void*) d_dst, (void*) d_tok, (void*) d_gu, (void*) d_d})
        cudaFree(p);
    (void) d_xs;
}


// A dense projection of k inputs to n outputs through native_mmvq, against the CPU reference on the same q8_1 activations.
void run_dense(int type, int k, int n, int variant) {
    cpu::exl3_set_variant(variant);
    const int w = cpu::exl3_words(type);
    std::mt19937 rng(7u + (unsigned) w + (unsigned) k + (unsigned) variant);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint8_t> rec(gpu::native_mmvq_weight_bytes(type, k, n));
    for (auto& b : rec) b = (uint8_t) rng();
    const cpu::Exl3Proj p = cpu::exl3_proj_at(rec.data(), k, n, w);
    for (int i = 0; i < k; ++i) ((__half*) p.suh)[i] = __float2half(nd(rng) * 0.02f);
    for (int i = 0; i < n; ++i) ((__half*) p.svh)[i] = __float2half(nd(rng));
    uint8_t* d_w;
    CK(cudaMalloc(&d_w, rec.size()));
    CK(cudaMemcpy(d_w, rec.data(), rec.size(), cudaMemcpyHostToDevice));
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    char label[96];
    for (const int nc : {1, 2, 3, 5, 8}) {
        std::vector<float> x((size_t) nc * k);
        for (auto& v : x) v = nd(rng);
        float *d_x, *d_y;
        void* d_q;
        CK(cudaMalloc(&d_x, x.size() * 4));
        CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_q, gpu::native_q8_1_bytes(k, nc)));
        CK(cudaMalloc(&d_y, (size_t) nc * n * 4));
        gpu::native_quantize_q8_1(d_x, d_q, k, nc, st);
        gpu::native_mmvq(type, d_w, d_q, d_y, k, n, nc, st);
        CK(cudaStreamSynchronize(st));
        const size_t qbytes = (size_t) nc * (k / 32) * 36;
        std::vector<uint8_t> hq(qbytes);
        CK(cudaMemcpy(hq.data(), d_q, qbytes, cudaMemcpyDeviceToHost));
        std::vector<float> got((size_t) nc * n), ref((size_t) nc * n), xh(k), x2((size_t) k);
        CK(cudaMemcpy(got.data(), d_y, got.size() * 4, cudaMemcpyDeviceToHost));
        for (int c = 0; c < nc; ++c) {
            for (int b = 0; b < k / 32; ++b) {
                const uint8_t* blk = hq.data() + ((size_t) c * (k / 32) + b) * 36;
                __half d;
                std::memcpy(&d, blk, 2);
                for (int i = 0; i < 32; ++i) xh[(size_t) b * 32 + i] = __half2float(d) * (float) (int8_t) blk[4 + i];
            }
            cpu::exl3_prepare_input(xh.data(), p.suh, k, variant, x2.data());
            cpu::exl3_rows_ref(p, x2.data(), &ref[(size_t) c * n], 0, n, variant);
        }
        std::snprintf(label, sizeof label, "dense %dx%d w%d cols %d v%d", k, n, w, nc, variant);
        report(label, rel(got, ref), 2e-4);
        for (void* q : {(void*) d_x, d_q, (void*) d_y}) cudaFree(q);
    }
    // the row-sliced dequantizer (rows 128..383 when the matrix has them)
    if (n >= 384) {
        uint16_t* d_h;
        CK(cudaMalloc(&d_h, (size_t) 256 * k * 2));
        gpu::exl3_dequant_dense_f16(type, d_w, k, n, 128, 256, d_h, st);
        CK(cudaStreamSynchronize(st));
        std::vector<uint16_t> h((size_t) 256 * k);
        CK(cudaMemcpy(h.data(), d_h, h.size() * 2, cudaMemcpyDeviceToHost));
        std::vector<float> x(k), x2((size_t) k), ref(n), got(256);
        for (auto& v : x) v = nd(rng);
        cpu::exl3_prepare_input(x.data(), p.suh, k, variant, x2.data());
        cpu::exl3_rows_ref(p, x2.data(), ref.data(), 0, n, variant);
        for (int r = 0; r < 256; ++r) {
            double a = 0;
            for (int i = 0; i < k; ++i) {
                __half hv;
                std::memcpy(&hv, &h[(size_t) r * k + i], 2);
                a += (double) __half2float(hv) * x[(size_t) i];
            }
            got[(size_t) r] = (float) a;
        }
        std::snprintf(label, sizeof label, "dense dequant rows 128..383 %dx%d w%d", k, n, w);
        report(label, rel(got, std::vector<float>(ref.begin() + 128, ref.begin() + 384)), 3e-3);
        cudaFree(d_h);
    }
    cudaStreamDestroy(st);
    cudaFree(d_w);
}


// Timing of one layer's experts as decode runs them: 10 groups of one entry each, blobs rotating through more memory
// than the L2 holds.
void bench(int type, bool mixed) {
    const gpu::NativeExpertLayout L = gpu::native_expert_layout(type, type, H, FF, 0.f);
    // mixed: a 3-token window as the engine sees it, 22 experts taking 29 entries
    static const int kCounts[22] = {1, 1, 2, 1, 1, 3, 1, 2, 1, 1, 1, 2, 1, 1, 3, 1, 1, 2, 1, 1, 1, 1};
    const int G = mixed ? 22 : 10, SETS = 8;
    int E = 0;
    for (int i = 0; i < G; ++i) E += mixed ? kCounts[i] : 1;
    std::mt19937 rng(5);
    std::vector<uint8_t> blob(L.bytes);
    for (auto& b : blob) b = (uint8_t) rng();
    std::vector<uint8_t*> blobs(G * SETS);
    for (auto& d : blobs) { CK(cudaMalloc(&d, L.bytes)); CK(cudaMemcpy(d, blob.data(), L.bytes, cudaMemcpyHostToDevice)); }
    std::vector<unsigned long long> hp((size_t) G * SETS);
    for (size_t i = 0; i < hp.size(); ++i) hp[i] = (unsigned long long) blobs[i];
    unsigned long long* d_ptr;
    CK(cudaMalloc(&d_ptr, hp.size() * 8));
    CK(cudaMemcpy(d_ptr, hp.data(), hp.size() * 8, cudaMemcpyHostToDevice));
    std::vector<int32_t> start(G + 1), dst(E), tok(E, 0);
    start[0] = 0;
    for (int i = 0; i < G; ++i) start[(size_t) i + 1] = start[(size_t) i] + (mixed ? kCounts[i] : 1);
    for (int i = 0; i < E; ++i) dst[(size_t) i] = i;
    int32_t *d_start, *d_ng, *d_dst, *d_tok;
    CK(cudaMalloc(&d_start, (G + 1) * 4)); CK(cudaMalloc(&d_ng, 4)); CK(cudaMalloc(&d_dst, E * 4)); CK(cudaMalloc(&d_tok, E * 4));
    const int ng = G;
    CK(cudaMemcpy(d_start, start.data(), (G + 1) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ng, &ng, 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dst, dst.data(), E * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tok, tok.data(), E * 4, cudaMemcpyHostToDevice));
    std::vector<float> x(H, 0.5f);
    float *d_x, *d_out;
    void *d_q, *d_scr;
    CK(cudaMalloc(&d_x, H * 4)); CK(cudaMemcpy(d_x, x.data(), H * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_q, (size_t) (H / 32) * 36));
    gpu::quantize_q8_1_rows(d_x, 1, H, d_q, nullptr);
    CK(cudaMalloc(&d_out, (size_t) E * H * 4));
    CK(cudaMalloc(&d_scr, gpu::native_expert_scratch_bytes(E, FF)));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    // the engine runs these inside a CUDA graph: replay one, so the host's launch rate is not what is timed
    constexpr int IT = 200, PER = 8;
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < PER; ++i)
        gpu::native_expert_grouped(L, d_ptr + (i % SETS) * G, d_start, d_ng, d_dst, d_tok, G, E, d_q, d_scr, d_out, st);
    CK(cudaStreamEndCapture(st, &graph));
    CK(cudaGraphInstantiate(&exec, graph, 0));
    for (int i = 0; i < 4; ++i) CK(cudaGraphLaunch(exec, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaEventRecord(e0, st));
    for (int i = 0; i < IT / PER; ++i) CK(cudaGraphLaunch(exec, st));
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    const double us = ms * 1e3 / (IT / PER * PER);
    std::printf("bench%s type %d: %.1f us per layer (%d experts, %d entries, %.0f MB) = %.0f GB/s\n", mixed ? " mixed" : "", type, us, G, E,
                (double) L.bytes * G / 1e6, (double) L.bytes * G / (us * 1e-6) / 1e9);
}

// Timing of dense projections as decode runs them (weights rotate through more memory than L2 holds), us per call.
void bench_dense() {
    struct Shape { const char* what; int k, n; };
    const Shape shapes[] = {{"shared gate/up", 2560, 640}, {"shared down", 640, 2560}, {"gdn qkv", 2560, 10240},
                            {"gdn z", 2560, 6144},         {"gdn/attn out", 6144, 2560}, {"attn q", 2560, 12288},
                            {"attn k/v", 2560, 512},       {"head", 2560, 248320}};
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    for (const int type : {cpu::kExl3TypeBase + 12}) {
        const int w = cpu::exl3_words(type);
        for (const Shape& sh : shapes) {
            const size_t bytes = gpu::native_mmvq_weight_bytes(type, sh.k, sh.n);
            const int sets = (int) std::min<size_t>(8, std::max<size_t>(1, (96u << 20) / bytes + 1));
            std::mt19937 rng(3);
            std::vector<uint8_t> rec(bytes);
            for (auto& b : rec) b = (uint8_t) rng();
            const cpu::Exl3Proj p = cpu::exl3_proj_at(rec.data(), sh.k, sh.n, w);
            for (int i = 0; i < sh.k; ++i) ((__half*) p.suh)[i] = __float2half(0.02f);
            for (int i = 0; i < sh.n; ++i) ((__half*) p.svh)[i] = __float2half(1.f);
            std::vector<uint8_t*> d_w(sets);
            for (auto& d : d_w) { CK(cudaMalloc(&d, bytes)); CK(cudaMemcpy(d, rec.data(), bytes, cudaMemcpyHostToDevice)); }
            for (const int nc : {1, 3}) {
                std::vector<float> x((size_t) nc * sh.k, 0.5f);
                float *d_x, *d_y;
                void* d_q;
                CK(cudaMalloc(&d_x, x.size() * 4));
                CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
                CK(cudaMalloc(&d_q, gpu::native_q8_1_bytes(sh.k, nc)));
                CK(cudaMalloc(&d_y, (size_t) nc * sh.n * 4));
                gpu::native_quantize_q8_1(d_x, d_q, sh.k, nc, st);
                for (int i = 0; i < 10; ++i) gpu::native_mmvq(type, d_w[(size_t) i % sets], d_q, d_y, sh.k, sh.n, nc, st);
                cudaEvent_t e0, e1;
                CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                constexpr int IT = 100;
                CK(cudaEventRecord(e0, st));
                for (int i = 0; i < IT; ++i) gpu::native_mmvq(type, d_w[(size_t) i % sets], d_q, d_y, sh.k, sh.n, nc, st);
                CK(cudaEventRecord(e1, st));
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                std::printf("dense %-15s %6d x %6d w%d ncols %d: %8.1f us  %6.0f GB/s\n", sh.what, sh.k, sh.n, w, nc,
                            ms * 1e3 / IT, (double) bytes * IT / (ms * 1e-3) / 1e9);
                cudaFree(d_x); cudaFree(d_q); cudaFree(d_y);
            }
            for (auto d : d_w) cudaFree(d);
        }
    }
    cudaStreamDestroy(st);
}

}  // namespace

int main(int argc, char** argv) {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) { std::printf("no CUDA device\n"); return 77; }
    if (argc > 1 && std::strcmp(argv[1], "--bench-dense") == 0) { bench_dense(); return 0; }
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        for (const int type : {cpu::kExl3TypeBase + 6, cpu::kExl3TypeBase + 7})
            for (const bool mixed : {false, true}) bench(type, mixed);
        return 0;
    }
    for (const int type : {cpu::kExl3TypeBase + 6, cpu::kExl3TypeBase + 7})   // 3 and 3.5 bits per weight
        for (const float limit : {0.f, 7.f})
            for (int variant = 0; variant < 4; ++variant) run(type, limit, variant);
    // the drafter's experts: gate/up 2 bits, down 3 bits
    for (const float limit : {0.f})
        for (int variant = 0; variant < 2; ++variant) run(cpu::kExl3TypeBase + 4, limit, variant, cpu::kExl3TypeBase + 6);
    std::printf("dense projections\n");
    for (const int type : {cpu::kExl3TypeBase + 10, cpu::kExl3TypeBase + 11, cpu::kExl3TypeBase + 12, cpu::kExl3TypeBase + 14, cpu::kExl3TypeBase + 16})
        for (const auto& kn : {std::pair<int, int>{2560, 640}, {2560, 1536}, {6144, 2560}, {640, 2560}, {2560, 512}})
            run_dense(type, kn.first, kn.second, 0);
    run_dense(cpu::kExl3TypeBase + 14, 2560, 640, 1);
    run_dense(cpu::kExl3TypeBase + 14, 2560, 640, 2);
    run_dense(cpu::kExl3TypeBase + 14, 2560, 640, 3);
    std::printf(g_fail ? "FAILED\n" : "ok\n");
    return g_fail ? 1 : 0;
}
