// src/models/gigachat4/kernels/kernels_parity.cpp - the GigaChat4 kernels against plain CPU references (double accumulation) at
// the model's geometry: GatedNorm, the MLA preparation and attention, the per-head Q8_0 products, the indexer's keys, queries,
// scores and top-k, the GDN conv and delta-rule step, the MoE plan and the softcap. Microbench lines at the end. Exit 77 without
// a GPU.
#include "basalt/core/device_copy.hpp"
#include "basalt/models/gigachat4/kernels.hpp"
#include "models/gigachat4/meta.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace gc = basalt::models::gigachat4;

namespace {

std::mt19937 rng(17);
std::vector<float> randn(size_t n, float s = 1.f) {
    std::normal_distribution<float> d(0.f, s);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
template <class T> T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    cudaMalloc(&p, h.size() * sizeof(T) + 16);
    basalt::core::device_copy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T> std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    basalt::core::device_copy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
template <class T> T* alloc(size_t n) {
    T* p = nullptr;
    cudaMalloc(&p, n * sizeof(T) + 16);
    cudaMemset(p, 0, n * sizeof(T) + 16);
    return p;
}
double sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }
float h2f(uint16_t b) {
    __half h;
    std::memcpy(&h, &b, 2);
    return __half2float(h);
}
std::vector<float> from_half(const std::vector<uint16_t>& v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = h2f(v[i]);
    return o;
}

int failures = 0;
void report(const char* what, const std::vector<float>& got, const std::vector<double>& ref, double tol) {
    double max_abs = 0, max_ref = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
        max_ref = std::max(max_ref, std::fabs(ref[i]));
    }
    const double rel = max_abs / std::max(max_ref, 1e-30);
    const bool ok = rel <= tol && std::isfinite(max_abs);
    std::printf("%-34s max abs %.3e  max rel %.3e  %s\n", what, max_abs, rel, ok ? "ok" : "FAIL");
    failures += !ok;
}

const gc::Geometry G;

std::vector<double> rms(const float* x, int n, const float* w, double eps) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const double r = 1.0 / std::sqrt(ss / n + eps);
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) y[i] = x[i] * r * w[i];
    return y;
}

void test_gated_norm(int n, int rows, int stride) {
    auto x = randn((size_t) rows * stride), w = randn(n, 0.3f), up = randn(16 * n, 0.05f), down = randn(n * 16, 0.3f);
    for (auto& v : w) v += 1.f;
    std::vector<double> ref((size_t) rows * stride);
    for (int r = 0; r < rows; ++r) {
        const auto y = rms(x.data() + (size_t) r * stride, n, w.data(), G.rms_eps);
        double g[16];
        for (int j = 0; j < 16; ++j) {
            double d = 0;
            for (int i = 0; i < n; ++i) d += up[(size_t) j * n + i] * y[i];
            g[j] = d * sig(d);
        }
        for (int i = 0; i < n; ++i) {
            double d = 0;
            for (int j = 0; j < 16; ++j) d += down[(size_t) i * 16 + j] * g[j];
            ref[(size_t) r * stride + i] = y[i] * G.norm_gain * sig(d);
        }
        for (int i = n; i < stride; ++i) ref[(size_t) r * stride + i] = x[(size_t) r * stride + i];   // untouched tail (in place)
    }
    float *dx = dev(x), *dw = dev(w), *du = dev(up), *dd = dev(down);
    gc::gated_norm(dx, stride, dw, du, dd, G.rms_eps, G.norm_gain, n, rows, dx, stride, nullptr);
    char name[64];
    std::snprintf(name, sizeof name, "gated_norm n=%d rows=%d", n, rows);
    report(name, host(dx, x.size()), ref, 2e-5);
    cudaFree(dx), cudaFree(dw), cudaFree(du), cudaFree(dd);
}

// The rope pair (x0, x1) turned by pos * f.
// The angle is the fp32 product pos * f, as transformers computes it.
void rot(double x0, double x1, float f, int pos, double& o0, double& o1) {
    const float ang = (float) pos * f;
    const double c = std::cos((double) ang), s = std::sin((double) ang);
    o0 = x0 * c - x1 * s;
    o1 = x0 * s + x1 * c;
}

// mla_prepare (fp32 and fp16 caches) then mla_attend over the fp16 cache, cells 0..pos and a list of cells, against a plain
// softmax.
void test_mla() {
    const gc::MlaShape s;
    const int W = s.rank + s.rope, Q = s.nope + s.rope, T = 4, pos0 = 237, cells = pos0 + T, ring = 0;
    const auto freqs = gc::yarn_inv_freq(G);
    const float scale = gc::mla_scale(G);
    auto q = randn((size_t) T * s.heads * Q), q_abs = randn((size_t) T * s.heads * s.rank, 0.2f), kv = randn((size_t) T * W);
    auto past = randn((size_t) cells * W);
    std::vector<int32_t> pos(T);
    std::iota(pos.begin(), pos.end(), pos0);
    std::vector<double> qref((size_t) T * s.heads * W), cref(past.begin(), past.end());
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < s.heads; ++h) {
            const size_t hr = (size_t) t * s.heads + h;
            for (int i = 0; i < s.rank; ++i) qref[hr * W + i] = q_abs[hr * s.rank + i];
            for (int i = 0; i < s.rope / 2; ++i)
                rot(q[hr * Q + s.nope + 2 * i], q[hr * Q + s.nope + 2 * i + 1], freqs[i], pos[t], qref[hr * W + s.rank + 2 * i],
                    qref[hr * W + s.rank + 2 * i + 1]);
        }
        double* c = cref.data() + (size_t) pos[t] * W;
        for (int i = 0; i < s.rank; ++i) c[i] = kv[(size_t) t * W + i];
        for (int i = 0; i < s.rope / 2; ++i)
            rot(kv[(size_t) t * W + s.rank + 2 * i], kv[(size_t) t * W + s.rank + 2 * i + 1], freqs[i], pos[t], c[s.rank + 2 * i], c[s.rank + 2 * i + 1]);
    }
    float *dq = dev(q), *da = dev(q_abs), *dkv = dev(kv), *df = dev(freqs), *dqc = alloc<float>(qref.size()), *dc = dev(past);
    int32_t* dp = dev(pos);
    gc::mla_prepare(dq, da, dkv, df, s, dp, T, ring, dqc, dc, nullptr);
    report("mla_prepare query", host(dqc, qref.size()), qref, 2e-6);
    report("mla_prepare cache fp32", host(dc, cref.size()), cref, 2e-6);
    std::vector<uint16_t> past_h(past.size());
    for (size_t i = 0; i < past.size(); ++i) {
        const __half h = __float2half_rn(past[i]);
        std::memcpy(&past_h[i], &h, 2);
    }
    uint16_t* dch = dev(past_h);
    gc::mla_prepare(dq, da, dkv, df, s, dp, T, ring, dqc, dch, nullptr);
    std::vector<double> cref_h(cref.size());
    for (size_t i = 0; i < cref.size(); ++i) cref_h[i] = __half2float(__float2half_rn((float) cref[i]));
    report("mla_prepare cache fp16", from_half(host(dch, past_h.size())), cref_h, 1e-3);   // one fp16 step

    const auto qc = host(dqc, qref.size());
    const auto cache = from_half(host(dch, past_h.size()));
    std::vector<int32_t> list((size_t) T * G.idx_topk, 0), nlist(T);
    for (int t = 0; t < T; ++t) {
        int c = 0;
        for (int j = 0; j <= pos[t]; j += 2) list[(size_t) t * G.idx_topk + c++] = j;
        nlist[t] = c;
    }
    for (int listed = 0; listed < 2; ++listed) {
        std::vector<double> oref((size_t) T * s.heads * s.rank, 0.0);
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < s.heads; ++h) {
                const float* qh = qc.data() + ((size_t) t * s.heads + h) * W;
                std::vector<int> js;
                if (listed)
                    js.assign(list.begin() + (size_t) t * G.idx_topk, list.begin() + (size_t) t * G.idx_topk + nlist[t]);
                else
                    for (int j = 0; j <= pos[t]; ++j) js.push_back(j);
                std::vector<double> sc(js.size());
                double m = -1e300, z = 0;
                for (size_t jj = 0; jj < js.size(); ++jj) {
                    double d = 0;
                    for (int i = 0; i < W; ++i) d += (double) qh[i] * cache[(size_t) js[jj] * W + i];
                    sc[jj] = d * scale;
                    m = std::max(m, sc[jj]);
                }
                for (auto& v : sc) z += (v = std::exp(v - m));
                double* o = oref.data() + ((size_t) t * s.heads + h) * s.rank;
                for (size_t jj = 0; jj < js.size(); ++jj)
                    for (int i = 0; i < s.rank; ++i) o[i] += sc[jj] / z * cache[(size_t) js[jj] * W + i];
            }
        std::vector<int32_t> counts(T);
        for (int t = 0; t < T; ++t) counts[t] = listed ? nlist[t] : pos[t] + 1;
        int32_t *dco = dev(counts), *dl = dev(list);
        const int max_cells = 300;   // more than two blocks of cells
        float* work = alloc<float>(gc::mla_work_floats(s.heads, s.rank, max_cells, T));
        float* dout = alloc<float>(oref.size());
        gc::mla_attend(dqc, dch, listed ? dl : nullptr, G.idx_topk, dco, max_cells, s, scale, T, work, dout, nullptr);
        report(listed ? "mla_attend (listed cells)" : "mla_attend (cells 0..pos)", host(dout, oref.size()), oref, 1e-5);
        cudaFree(dco), cudaFree(dl), cudaFree(work), cudaFree(dout);
    }
    // FP8 cache: mla_prepare and mla_attend
    {
        std::vector<uint8_t> f8_init((size_t) cells * gc::kFp8RowBytes, 0);
        uint8_t* dcf8 = dev(f8_init);
        std::vector<int32_t> all_pos(cells);
        std::iota(all_pos.begin(), all_pos.end(), 0);
        auto all_q = randn((size_t) cells * s.heads * Q), all_qa = randn((size_t) cells * s.heads * s.rank, 0.2f);
        float *daq = dev(all_q), *daa = dev(all_qa), *dpast = dev(past), *dqc_all = alloc<float>((size_t) cells * s.heads * W);
        int32_t* dap = dev(all_pos);
        gc::mla_prepare(daq, daa, dpast, df, s, dap, cells, ring, dqc_all, dcf8, nullptr);
        cudaFree(dqc_all);
        const auto f8 = host(dcf8, f8_init.size());
        auto e4m3 = [](uint8_t b) -> float {
            const int sign = (b >> 7) & 1, exp = (b >> 3) & 15, man = b & 7;
            float v = exp == 0 ? (float) man / 8.f * std::ldexp(1.f, -6) : (1.f + (float) man / 8.f) * std::ldexp(1.f, exp - 7);
            return sign ? -v : v;
        };
        std::vector<double> f8_dec((size_t) cells * W);
        for (int c = 0; c < cells; ++c) {
            const uint8_t* row = f8.data() + (size_t) c * gc::kFp8RowBytes;
            float sc[4];
            std::memcpy(sc, row + 512, 16);
            for (int i = 0; i < s.rank; ++i) f8_dec[(size_t) c * W + i] = (double) e4m3(row[i]) * (double) sc[i >> 7];
            uint16_t rp[64];
            std::memcpy(rp, row + 528, 128);
            for (int i = 0; i < s.rope; ++i) {
                __nv_bfloat16 b;
                std::memcpy(&b, &rp[i], 2);
                f8_dec[(size_t) c * W + s.rank + i] = __bfloat162float(b);
            }
        }
        std::vector<double> past_rot(past.begin(), past.end());
        for (int c = 0; c < cells; ++c)
            for (int i = 0; i < s.rope / 2; ++i)
                rot(past[(size_t) c * W + s.rank + 2 * i], past[(size_t) c * W + s.rank + 2 * i + 1], freqs[i], c,
                    past_rot[(size_t) c * W + s.rank + 2 * i], past_rot[(size_t) c * W + s.rank + 2 * i + 1]);
        std::vector<float> f8_dec_f(f8_dec.size());
        for (size_t i = 0; i < f8_dec.size(); ++i) f8_dec_f[i] = (float) f8_dec[i];
        report("mla_prepare cache fp8", f8_dec_f, past_rot, 0.15);   // E4M3 quantization (~6%)
        gc::mla_prepare(dq, da, dkv, df, s, dp, T, ring, dqc, dcf8, nullptr);
        const auto qcf8 = host(dqc, qref.size());
        const auto f8_2 = host(dcf8, f8_init.size());
        for (int c = 0; c < cells; ++c) {
            const uint8_t* row = f8_2.data() + (size_t) c * gc::kFp8RowBytes;
            float sc[4];
            std::memcpy(sc, row + 512, 16);
            for (int i = 0; i < s.rank; ++i) f8_dec[(size_t) c * W + i] = (double) e4m3(row[i]) * (double) sc[i >> 7];
            uint16_t rp[64];
            std::memcpy(rp, row + 528, 128);
            for (int i = 0; i < s.rope; ++i) {
                __nv_bfloat16 b;
                std::memcpy(&b, &rp[i], 2);
                f8_dec[(size_t) c * W + s.rank + i] = __bfloat162float(b);
            }
        }
        std::vector<double> oref((size_t) T * s.heads * s.rank, 0.0);
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < s.heads; ++h) {
                const float* qh = qcf8.data() + ((size_t) t * s.heads + h) * W;
                const int cnt = nlist[t];
                std::vector<double> sc(cnt);
                double m = -1e300, z = 0;
                for (int jj = 0; jj < cnt; ++jj) {
                    const int cell = list[(size_t) t * G.idx_topk + jj];
                    double d = 0;
                    for (int i = 0; i < W; ++i) d += (double) qh[i] * f8_dec[(size_t) cell * W + i];
                    sc[jj] = d * scale;
                    m = std::max(m, sc[jj]);
                }
                for (auto& v : sc) z += (v = std::exp(v - m));
                double* o = oref.data() + ((size_t) t * s.heads + h) * s.rank;
                for (int jj = 0; jj < cnt; ++jj) {
                    const int cell = list[(size_t) t * G.idx_topk + jj];
                    for (int i = 0; i < s.rank; ++i) o[i] += sc[jj] / z * f8_dec[(size_t) cell * W + i];
                }
            }
        int32_t *dco = dev(nlist), *dl = dev(list);
        const int max_cells = 300;
        float* work = alloc<float>(gc::mla_work_floats(s.heads, s.rank, max_cells, T));
        float* dout = alloc<float>(oref.size());
        gc::mla_attend(dqc, dcf8, dl, G.idx_topk, dco, max_cells, s, scale, T, work, dout, nullptr);
        report("mla_attend fp8 (listed cells)", host(dout, oref.size()), oref, 1e-5);
        // Tensor-core prefill path (n_tok > 8) over both fp16 and fp8 caches
        const int TP = 12;
        auto pq = randn((size_t) TP * s.heads * W, 0.2f);
        float *dpq = dev(pq), *dpo = alloc<float>((size_t) TP * s.heads * s.rank), *dwo = alloc<float>(gc::mla_work_floats(s.heads, s.rank, max_cells, TP));
        std::vector<int32_t> pcnt(TP, cells), plist((size_t) TP * G.idx_topk, 0);
        for (int t = 0; t < TP; ++t) {
            pcnt[t] = (cells / 2) + t * 5;
            for (int j = 0; j < pcnt[t]; ++j) plist[(size_t) t * G.idx_topk + j] = j;
        }
        int32_t *dpc = dev(pcnt), *dpl = dev(plist);
        std::vector<double> mref((size_t) TP * s.heads * s.rank, 0.0);
        for (int t = 0; t < TP; ++t)
            for (int h = 0; h < s.heads; ++h) {
                const float* qh = pq.data() + ((size_t) t * s.heads + h) * W;
                std::vector<double> sc(pcnt[t]);
                double m = -1e300, z = 0;
                for (int j = 0; j < pcnt[t]; ++j) {
                    double d = 0;
                    for (int i = 0; i < W; ++i) d += (double) qh[i] * f8_dec[(size_t) j * W + i];
                    sc[j] = d * scale;
                    m = std::max(m, sc[j]);
                }
                for (auto& v : sc) z += (v = std::exp(v - m));
                double* o = mref.data() + ((size_t) t * s.heads + h) * s.rank;
                for (int j = 0; j < pcnt[t]; ++j)
                    for (int i = 0; i < s.rank; ++i) o[i] += sc[j] / z * f8_dec[(size_t) j * W + i];
            }
        gc::mla_attend(dpq, dcf8, dpl, G.idx_topk, dpc, max_cells, s, scale, TP, dwo, dpo, nullptr);
        report("mla_attend fp8 prefill mma", host(dpo, mref.size()), mref, 5e-3);
        cudaFree(dpq), cudaFree(dpo), cudaFree(dwo), cudaFree(dpc), cudaFree(dpl);
        cudaFree(dco), cudaFree(dl), cudaFree(work), cudaFree(dout), cudaFree(dcf8), cudaFree(daq), cudaFree(daa), cudaFree(dpast), cudaFree(dap);
    }
    cudaFree(dq), cudaFree(da), cudaFree(dkv), cudaFree(df), cudaFree(dqc), cudaFree(dc), cudaFree(dch), cudaFree(dp);
}

// Q8_0 blocks of 32: an fp16 scale amax / 127, then 32 int8 codes.
std::vector<uint8_t> q8_0(const std::vector<float>& w, std::vector<float>& deq) {
    const size_t nb = w.size() / 32;
    std::vector<uint8_t> out(nb * 34);
    deq.resize(w.size());
    for (size_t b = 0; b < nb; ++b) {
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(w[b * 32 + i]));
        const __half dh = __float2half_rn(amax / 127.f);
        const float d = __half2float(dh);
        std::memcpy(&out[b * 34], &dh, 2);
        for (int i = 0; i < 32; ++i) {
            const int q = d > 0 ? (int) std::lround(w[b * 32 + i] / d) : 0;
            out[b * 34 + 2 + i] = (uint8_t) (int8_t) std::max(-127, std::min(127, q));
            deq[b * 32 + i] = d * (int8_t) out[b * 34 + 2 + i];
        }
    }
    return out;
}

// k_b (rows 512 over 128, the query's head stride nope + rope) and v_b (rows 128 over 512): per-head Q8_0 products.
void test_heads_gemv(int rows, int k, int xh) {
    const int heads = G.heads, T = 3;
    std::vector<float> deq;
    const auto w = q8_0(randn((size_t) heads * rows * k, 0.1f), deq);
    const auto x = randn((size_t) T * heads * xh);
    std::vector<double> ref((size_t) T * heads * rows);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < heads; ++h)
            for (int r = 0; r < rows; ++r) {
                double d = 0;
                for (int i = 0; i < k; ++i) d += (double) deq[((size_t) h * rows + r) * k + i] * x[((size_t) t * heads + h) * xh + i];
                ref[((size_t) t * heads + h) * rows + r] = d;
            }
    uint8_t* dw = dev(w);
    float *dx = dev(x), *dy = alloc<float>(ref.size());
    gc::heads_q8_gemv(dw, dx, xh, heads * xh, heads, rows, k, T, dy, nullptr);
    char name[64];
    std::snprintf(name, sizeof name, "heads_q8_gemv rows=%d k=%d", rows, k);
    report(name, host(dy, ref.size()), ref, 1e-5);
    cudaFree(dw), cudaFree(dx), cudaFree(dy);
}

// The indexer: keys and queries roped in halves, then idx_topk's scores sum_h w_h relu(q_h . k) and top-2048.
void test_indexer() {
    const int H = G.idx_heads, D = G.idx_dim, R = G.rope, n_keys = 3000, k = G.idx_topk, T = 2, ring = 4096;
    const auto freqs = gc::yarn_inv_freq(G);
    auto kr = randn((size_t) n_keys * D), nw = randn(D, 0.2f), q = randn((size_t) T * H * D);
    for (auto& v : nw) v += 1.f;
    std::vector<int32_t> kpos(n_keys), qpos = {n_keys - 2, n_keys - 1};
    std::iota(kpos.begin(), kpos.end(), 0);
    std::vector<double> kref((size_t) n_keys * D), qref(q.begin(), q.end());
    for (int p = 0; p < n_keys; ++p) {
        const auto y = rms(kr.data() + (size_t) p * D, D, nw.data(), 1e-6);
        double* o = kref.data() + (size_t) p * D;
        for (int i = 0; i < D; ++i) o[i] = y[i];
        for (int i = 0; i < R / 2; ++i) rot(y[i], y[i + R / 2], freqs[i], p, o[i], o[i + R / 2]);
    }
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            double* o = qref.data() + ((size_t) t * H + h) * D;
            double a[64], b[64];
            for (int i = 0; i < R / 2; ++i) rot(o[i], o[i + R / 2], freqs[i], qpos[t], a[i], b[i]);
            for (int i = 0; i < R / 2; ++i) o[i] = a[i], o[i + R / 2] = b[i];
        }
    float *dk = dev(kr), *dnw = dev(nw), *df = dev(freqs), *dq = dev(q);
    int32_t *dkp = dev(kpos), *dqp = dev(qpos);
    uint16_t* keys = alloc<uint16_t>((size_t) ring * D);
    gc::idx_keys(dk, dnw, 1e-6f, df, D, R, dkp, n_keys, ring, keys, nullptr);
    gc::idx_rope_queries(dq, df, H, D, R, dqp, T, nullptr);
    const auto kh = from_half(host(keys, (size_t) n_keys * D));
    std::vector<double> kref_h(kref.size());
    for (size_t i = 0; i < kref.size(); ++i) kref_h[i] = __half2float(__float2half_rn((float) kref[i]));
    report("idx_keys (fp16)", kh, kref_h, 2e-3);
    report("idx_rope_queries", host(dq, q.size()), qref, 2e-6);

    // row 1 sees every key, row 0 all but the last
    const auto w = randn((size_t) T * H, gc::idx_weight_scale(G));
    const auto qd = host(dq, q.size());
    std::vector<int32_t> nc = {n_keys - 1, n_keys};
    float *dw = dev(w), *dsc = alloc<float>((size_t) T * 2 * k);
    int32_t *dnc = dev(nc), *dcells = alloc<int32_t>((size_t) T * k), *dns = alloc<int32_t>(T);
    gc::idx_topk(dq, dw, keys, dnc, n_keys, H, D, k, T, dsc, dcells, dns, nullptr);
    const auto sc = host(dsc, (size_t) T * n_keys);
    const auto got = host(dcells, (size_t) T * k);
    const auto ns = host(dns, T);
    for (int t = 0; t < T; ++t) {
        std::vector<double> sref(n_keys);
        std::vector<float> sg(n_keys);
        for (int p = 0; p < n_keys; ++p) {
            double tot = 0;
            for (int h = 0; h < H; ++h) {
                double d = 0;
                for (int i = 0; i < D; ++i) d += (double) qd[((size_t) t * H + h) * D + i] * kh[(size_t) p * D + i];
                tot += w[t * H + h] * std::max(d, 0.0);
            }
            sref[p] = p < nc[t] ? tot : -3.4028234663852886e38;
            sg[p] = sc[(size_t) t * n_keys + p];
        }
        report(t ? "idx_topk scores (all keys)" : "idx_topk scores (last masked)", sg, sref, 1e-5);
        std::vector<int> order(n_keys);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return sg[x] > sg[y]; });
        std::vector<int> want(order.begin(), order.begin() + k), have(got.begin() + (size_t) t * k, got.begin() + (size_t) (t + 1) * k);
        std::sort(want.begin(), want.end());
        std::sort(have.begin(), have.end());
        const bool same = ns[t] == k && want == have;
        std::printf("%-34s %s\n", t ? "idx_topk top-2048 (all keys)" : "idx_topk top-2048 (last masked)", same ? "ok" : "FAIL");
        failures += !same;
    }
    std::vector<int32_t> few = {5, 1};
    int32_t* dfew = dev(few);
    gc::idx_topk(dq, dw, keys, dfew, 2 * k, H, D, k, T, dsc, dcells, dns, nullptr);
    const auto g2 = host(dcells, (size_t) T * k);
    const auto n2 = host(dns, T);
    const bool few_ok = n2[0] == 5 && n2[1] == 1 && g2[0] == 0 && g2[4] == 4 && g2[k] == 0;
    std::printf("%-34s %s\n", "idx_topk fewer cells than k", few_ok ? "ok" : "FAIL");
    failures += !few_ok;
    cudaFree(dk), cudaFree(dnw), cudaFree(df), cudaFree(dq), cudaFree(dkp), cudaFree(dqp), cudaFree(keys), cudaFree(dw), cudaFree(dsc),
        cudaFree(dcells), cudaFree(dnc), cudaFree(dns), cudaFree(dfew);
}

// gdn_conv then gdn_step over T rows from a random state, against the release's recurrence (torch_recurrent_gated_delta_rule
// with the sigmoid-gated output norm): value head h reads key head h / 2.
void test_gdn(int T = 3) {
    const gc::GdnShape s;
    const int NK = s.hk * s.dk, NV = s.hv * s.dv, CH = 2 * NK + NV, NP = CH + NV;
    auto proj = randn((size_t) T * NP), w = randn((size_t) CH * 4, 0.5f), st0 = randn((size_t) CH * 3);
    auto ba = randn((size_t) T * 2 * s.hv), a_log = randn(s.hv, 0.5f), dtb = randn(s.hv, 0.5f), nw = randn(s.dv, 0.2f);
    auto S0 = randn((size_t) s.hv * s.dk * s.dv, 0.1f);
    for (auto& v : nw) v += 2.f;
    std::vector<double> conv((size_t) T * CH), stc(st0.begin(), st0.end());
    for (int c = 0; c < CH; ++c)
        for (int t = 0; t < T; ++t) {
            const double xi = proj[(size_t) t * NP + c];
            const double v = w[c * 4] * stc[3 * c] + w[c * 4 + 1] * stc[3 * c + 1] + w[c * 4 + 2] * stc[3 * c + 2] + w[c * 4 + 3] * xi;
            stc[3 * c] = stc[3 * c + 1], stc[3 * c + 1] = stc[3 * c + 2], stc[3 * c + 2] = xi;
            conv[(size_t) t * CH + c] = v * sig(v);
        }
    std::vector<double> S(S0.begin(), S0.end()), y((size_t) T * NV);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < s.hv; ++h) {
            const int kh = h / (s.hv / s.hk);
            const double* r = conv.data() + (size_t) t * CH;
            double qn = 0, kn = 0;
            for (int i = 0; i < s.dk; ++i) qn += r[kh * s.dk + i] * r[kh * s.dk + i], kn += r[NK + kh * s.dk + i] * r[NK + kh * s.dk + i];
            qn = 1 / std::sqrt(qn + 1e-6) / std::sqrt((double) s.dk), kn = 1 / std::sqrt(kn + 1e-6);
            const double a = ba[(size_t) t * 2 * s.hv + s.hv + h] + dtb[h], b = ba[(size_t) t * 2 * s.hv + h];
            const double decay = std::exp(-std::exp((double) a_log[h]) * std::log1p(std::exp(a))), beta = sig(b);
            double* M = S.data() + (size_t) h * s.dk * s.dv;
            std::vector<double> o(s.dv);
            for (int j = 0; j < s.dv; ++j) {
                double kv = 0;
                for (int i = 0; i < s.dk; ++i) kv += (M[i * s.dv + j] *= decay) * r[NK + kh * s.dk + i] * kn;
                const double delta = (r[2 * NK + h * s.dv + j] - kv) * beta;
                double acc = 0;
                for (int i = 0; i < s.dk; ++i) {
                    M[i * s.dv + j] += r[NK + kh * s.dk + i] * kn * delta;
                    acc += M[i * s.dv + j] * r[kh * s.dk + i] * qn;
                }
                o[j] = acc;
            }
            double ss = 0;
            for (double v : o) ss += v * v;
            const double inv = 1 / std::sqrt(ss / s.dv + G.rms_eps);
            for (int j = 0; j < s.dv; ++j)
                y[(size_t) t * NV + h * s.dv + j] = o[j] * inv * nw[j] * sig(proj[(size_t) t * NP + CH + h * s.dv + j]);
        }
    float *dp = dev(proj), *dw = dev(w), *dst = dev(st0), *dco = alloc<float>((size_t) T * CH), *dba = dev(ba), *dal = dev(a_log),
          *ddt = dev(dtb), *dnw = dev(nw), *dS = dev(S0), *dy = alloc<float>(y.size());
    gc::gdn_conv(dp, NP, dst, dw, CH, T, dco, nullptr);
    char l1[64], l2[64], l3[64];
    std::snprintf(l1, sizeof(l1), "gdn_conv (silu, %d rows)", T);
    std::snprintf(l2, sizeof(l2), "gdn_step output (%d rows)", T);
    std::snprintf(l3, sizeof(l3), "gdn_step state (%d rows)", T);
    report(l1, host(dco, conv.size()), conv, 1e-5);
    gc::gdn_step(dS, dco, dba, dal, ddt, dp + CH, NP, dnw, G.rms_eps, s, T, dy, nullptr);
    report(l2, host(dy, y.size()), y, 1e-4);
    report(l3, host(dS, S.size()), S, 1e-4);
    cudaFree(dp), cudaFree(dw), cudaFree(dst), cudaFree(dco), cudaFree(dba), cudaFree(dal), cudaFree(ddt), cudaFree(dnw), cudaFree(dS), cudaFree(dy);
}

// moe_plan: groups are the distinct experts in order, each entry once, counts added.
void test_moe_plan(int rows = 8) {
    const int k = 8, E = G.n_expert;
    std::vector<int32_t> ids(rows * k);
    for (int t = 0; t < rows; ++t)
        for (int j = 0; j < k; ++j) ids[t * k + j] = (int32_t) ((t * 7 + j * 23) % 40);
    std::vector<unsigned long long> table(E);
    for (int e = 0; e < E; ++e) table[e] = 1000 + e;
    int32_t *di = dev(ids), *dgs = alloc<int32_t>(rows * k + 1), *dng = alloc<int32_t>(1), *dd = alloc<int32_t>(rows * k), *dt = alloc<int32_t>(rows * k);
    unsigned long long *dtab = dev(table), *dgp = alloc<unsigned long long>(rows * k);
    uint32_t* dc = alloc<uint32_t>(E);
    gc::moe_plan(di, rows, k, E, dtab, dgp, dgs, dng, dd, dt, dc, nullptr);
    const int ng = host(dng, 1)[0];
    const auto gp = host(dgp, rows * k);
    const auto gs = host(dgs, rows * k + 1);
    const auto ed = host(dd, rows * k), et = host(dt, rows * k);
    const auto cn = host(dc, E);
    std::vector<int> want(ids.begin(), ids.end());
    std::sort(want.begin(), want.end());
    want.erase(std::unique(want.begin(), want.end()), want.end());
    bool ok = ng == (int) want.size() && gs[ng] == rows * k;
    std::vector<int> seen(rows * k, 0);
    for (int gi = 0; ok && gi < ng; ++gi) {
        ok = gp[gi] == 1000ull + want[gi];
        for (int e = gs[gi]; ok && e < gs[gi + 1]; ++e) ok = ids[ed[e]] == want[gi] && et[e] == ed[e] / k && !seen[ed[e]]++;
    }
    for (int e = 0; ok && e < E; ++e) ok = (int) cn[e] == (int) std::count(ids.begin(), ids.end(), e);
    char lbl[64];
    std::snprintf(lbl, sizeof(lbl), "moe_plan (%d row(s), shared)", rows);
    std::printf("%-34s %s\n", lbl, ok ? "ok" : "FAIL");
    failures += !ok;
    cudaFree(di), cudaFree(dgs), cudaFree(dng), cudaFree(dd), cudaFree(dt), cudaFree(dtab), cudaFree(dgp), cudaFree(dc);
}

void test_moe_sort() {
    const int rows = 256, k = 8, E = G.n_expert, P = rows * k;
    std::vector<int32_t> ids(P);
    for (int i = 0; i < P; ++i) ids[i] = (i * 37 + 13) % E;
    int32_t *d_ids = dev(ids), *d_b = alloc<int32_t>(E + 1), *d_t = alloc<int32_t>(P), *d_d = alloc<int32_t>(P), *d_e = alloc<int32_t>(P);
    uint8_t* scr = alloc<uint8_t>(gc::moe_sort_scratch(P, E));
    gc::moe_sort(d_ids, rows, k, E, scr, d_b, d_t, d_d, d_e, nullptr);
    const auto b = host(d_b, E + 1), tk = host(d_t, P), ds = host(d_d, P), ex = host(d_e, P);
    std::vector<int> want_b(E + 1, 0), want_d, want_t, want_e;
    std::vector<std::vector<int>> buckets(E);
    for (int i = 0; i < P; ++i) buckets[ids[i]].push_back(i);
    for (int e = 0; e < E; ++e) {
        want_b[e] = (int) want_d.size();
        for (int i : buckets[e]) { want_d.push_back(i); want_t.push_back(i / k); want_e.push_back(e); }
    }
    want_b[E] = P;
    const bool ok = b == want_b && tk == want_t && ds == want_d && ex == want_e;
    std::printf("%-34s %s\n", "moe_sort (256 rows x 8)", ok ? "ok" : "FAIL");
    failures += !ok;
    cudaFree(d_ids), cudaFree(d_b), cudaFree(d_t), cudaFree(d_d), cudaFree(d_e), cudaFree(scr);
}

void test_softcap() {
    const auto x = randn(G.n_vocab, 40.f);
    std::vector<double> ref(x.size());
    for (size_t i = 0; i < x.size(); ++i) ref[i] = G.logit_softcap * std::tanh(x[i] / G.logit_softcap);
    float* dx = dev(x);
    gc::softcap(dx, (int64_t) x.size(), G.logit_softcap, nullptr);
    report("softcap", host(dx, x.size()), ref, 1e-6);
    cudaFree(dx);
}

void test_argmax_prob() {
    const auto x = randn(G.n_vocab, 3.f);
    double mx = -1e300;
    int best = 0;
    for (int i = 0; i < G.n_vocab; ++i)
        if ((double) x[i] > mx) mx = x[i], best = i;
    double z = 0;
    for (int i = 0; i < G.n_vocab; ++i) z += std::exp((double) x[i] - mx);
    const double want_p = 1.0 / z;
    float *dx = dev(x), *dp = alloc<float>(1);
    int32_t* di = alloc<int32_t>(1);
    gc::argmax_prob_row(dx, G.n_vocab, di, dp, nullptr);
    const int got_i = host(di, 1)[0];
    const float got_p = host(dp, 1)[0];
    const double rel = std::fabs((double) got_p - want_p) / std::max(want_p, 1e-30);
    const bool ok = got_i == best && rel < 1e-5;
    std::printf("%-34s id %d/%d  rel %.3e  %s\n", "argmax_prob_row", got_i, best, rel, ok ? "ok" : "FAIL");
    failures += !ok;
    cudaFree(dx), cudaFree(di), cudaFree(dp);
}

template <class F> double time_us(F f, int reps = 200) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    f();
    cudaEventRecord(e0);
    for (int i = 0; i < reps; ++i) f();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0), cudaEventDestroy(e1);
    return 1000.0 * ms / reps;
}

void bench() {
    cudaStream_t st;
    cudaStreamCreate(&st);
    const int n = G.n_embd;
    float *x = alloc<float>((size_t) 8 * n), *w = alloc<float>(n), *up = alloc<float>(16 * n), *down = alloc<float>((size_t) n * 16);
    for (int rows : {1, 8})
        std::printf("bench: gated_norm %d row(s) of %d %.2f us a launch\n", rows, n,
                    time_us([&] { gc::gated_norm(x, n, w, up, down, G.rms_eps, G.norm_gain, n, rows, x, n, st); }));
    const gc::MlaShape s;
    const int W = s.rank + s.rope;
    float *q = alloc<float>((size_t) 8 * s.heads * (s.nope + s.rope)), *qa = alloc<float>((size_t) 8 * s.heads * s.rank),
          *kv = alloc<float>((size_t) 8 * W), *f = alloc<float>(s.rope / 2), *qc = alloc<float>((size_t) 8 * s.heads * W);
    uint16_t* ch = alloc<uint16_t>((size_t) 4096 * W);
    int32_t* p = alloc<int32_t>(8);
    std::printf("bench: mla_prepare 1 row %.2f us a launch\n", time_us([&] { gc::mla_prepare(q, qa, kv, f, s, p, 1, 0, qc, ch, st); }));
    {
        int32_t* nc = alloc<int32_t>(8);
        const std::vector<int32_t> cells(8, 1000);
        basalt::core::device_copy(nc, cells.data(), 32, cudaMemcpyHostToDevice);
        float* work = alloc<float>(gc::mla_work_floats(s.heads, s.rank, 1024, 1));
        float* out = alloc<float>((size_t) s.heads * s.rank);
        std::printf("bench: mla_attend 1 row, 1000 cells (bucket 1024) %.2f us a launch\n",
                    time_us([&] { gc::mla_attend(qc, ch, nullptr, 0, nc, 1024, s, 0.1f, 1, work, out, st); }));
        uint8_t* kb = alloc<uint8_t>((size_t) 32 * 512 * 4 * 34);
        std::printf("bench: heads_q8_gemv k_b (32 x 512 x 128) %.2f us a launch\n",
                    time_us([&] { gc::heads_q8_gemv(kb, q, 192, 32 * 192, 32, 512, 128, 1, qa, st); }));
        std::printf("bench: heads_q8_gemv v_b (32 x 128 x 512) %.2f us a launch\n",
                    time_us([&] { gc::heads_q8_gemv(kb, qa, 512, 32 * 512, 32, 128, 512, 1, out, st); }));
        cudaFree(nc), cudaFree(work), cudaFree(out), cudaFree(kb);
    }
    {
        const gc::GdnShape gs;
        float *S = alloc<float>((size_t) gs.hv * gs.dk * gs.dv), *qkv = alloc<float>(256 * 8192), *ba = alloc<float>(256 * 64),
              *z = alloc<float>(256 * 4096), *y = alloc<float>(256 * 4096), *cw = alloc<float>(8192 * 4), *cs = alloc<float>(8192 * 3),
              *co = alloc<float>(256 * 8192);
        for (int rows : {1, 8, 256}) {
            std::printf("bench: gdn_conv %d row(s) %.2f us a launch\n", rows, time_us([&] { gc::gdn_conv(qkv, 8192, cs, cw, 8192, rows, co, st); }, rows == 256 ? 50 : 200));
            std::printf("bench: gdn_step %d row(s) %.2f us a launch\n", rows,
                        time_us([&] { gc::gdn_step(S, qkv, ba, ba, ba, z, 4096, ba, G.rms_eps, gs, rows, y, st); }, rows == 256 ? 50 : 200));
        }
        cudaFree(S), cudaFree(qkv), cudaFree(ba), cudaFree(z), cudaFree(y), cudaFree(cw), cudaFree(cs), cudaFree(co);
    }
    {
        float *qx = alloc<float>((size_t) 256 * 64 * 128), *qw = alloc<float>(256 * 64), *sc = alloc<float>((size_t) 32 * 8192);
        uint16_t* keys = alloc<uint16_t>((size_t) 8192 * 128);
        std::vector<int32_t> ncv(256, 8192);
        int32_t *dnc = dev(ncv), *dcl = alloc<int32_t>((size_t) 256 * 2048), *dns = alloc<int32_t>(256);
        std::printf("bench: idx_topk 1 row 8192 %.2f us a launch\n",
                    time_us([&] { gc::idx_topk(qx, qw, keys, dnc, 8192, 64, 128, 2048, 1, sc, dcl, dns, st); }, 50));
        std::printf("bench: idx_topk 8 rows 8192 %.2f us a launch\n",
                    time_us([&] { gc::idx_topk(qx, qw, keys, dnc, 8192, 64, 128, 2048, 8, sc, dcl, dns, st); }, 50));
        cudaFree(qx), cudaFree(qw), cudaFree(sc), cudaFree(keys), cudaFree(dnc), cudaFree(dcl), cudaFree(dns);
    }
    float* logits = alloc<float>(G.n_vocab);
    std::printf("bench: softcap %d logits %.2f us a launch\n", G.n_vocab, time_us([&] { gc::softcap(logits, G.n_vocab, G.logit_softcap, st); }));
    {
        int32_t *di = alloc<int32_t>(64), *dgs = alloc<int32_t>(65), *dng = alloc<int32_t>(1), *dd = alloc<int32_t>(64), *dt = alloc<int32_t>(64);
        unsigned long long *dtab = alloc<unsigned long long>(G.n_expert), *dgp = alloc<unsigned long long>(64);
        uint32_t* dc = alloc<uint32_t>(G.n_expert);
        std::printf("bench: moe_plan 1 row %.2f us a launch\n",
                    time_us([&] { gc::moe_plan(di, 1, 8, G.n_expert, dtab, dgp, dgs, dng, dd, dt, dc, st); }));
        float *ex = alloc<float>(8 * n), *rw = alloc<float>(8), *sh = alloc<float>(n);
        std::printf("bench: moe_combine 1 row %.2f us a launch\n",
                    time_us([&] { gc::moe_combine(ex, rw, sh, 8, n, 1, x, st); }));
        cudaFree(di), cudaFree(dgs), cudaFree(dng), cudaFree(dd), cudaFree(dt), cudaFree(dtab), cudaFree(dgp), cudaFree(dc), cudaFree(ex), cudaFree(rw), cudaFree(sh);
    }
    cudaFree(x), cudaFree(w), cudaFree(up), cudaFree(down), cudaFree(q), cudaFree(qa), cudaFree(kv), cudaFree(f), cudaFree(qc), cudaFree(ch),
        cudaFree(p), cudaFree(logits);
    cudaStreamDestroy(st);
}

}  // namespace

int main() {
    int n = 0;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
    test_gated_norm(G.n_embd, 5, G.n_embd);
    test_gated_norm(G.q_rank, 3, G.q_rank);
    test_gated_norm(G.kv_rank, 4, G.kv_rank + G.rope);   // the latent of a kv_a row, in place
    test_mla();
    test_heads_gemv(G.kv_rank, G.nope, G.nope + G.rope);
    test_heads_gemv(G.v_dim, G.kv_rank, G.kv_rank);
    test_indexer();
    test_gdn(3);
    test_gdn(16);
    test_moe_plan(1);
    test_moe_plan(8);
    test_moe_sort();
    test_softcap();
    test_argmax_prob();
    bench();
    std::printf(failures ? "FAILED\n" : "all ok\n");
    return failures ? 1 : 0;
}
