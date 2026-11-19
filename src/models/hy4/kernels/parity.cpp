// src/models/hy4/kernels/parity.cpp - the Hy4-flash decode kernels against plain double-precision CPU references, on
// random inputs at the real geometry: the hyper-connection mix, apply and head, the row norm, GPT-J RoPE, the router,
// the Engram rows and post-op, the absorbed MLA attention (a window and a selected list, both head counts), and the
// drafter's helpers.
//
//     build/hy4_kernels_parity        (needs a CUDA device; exits 77 without one)
#include "basalt/models/hy4/geometry.hpp"
#include "basalt/models/hy4/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace basalt::models::hy4;

namespace {

int g_fail = 0;
std::mt19937 g_rng(1234);

std::vector<float> randn(size_t n, float s = 1.f) {
    std::normal_distribution<float> d(0.f, s);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g_rng);
    return v;
}

template <typename T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    cudaMalloc((void**) &p, std::max<size_t>(1, h.size()) * sizeof(T));
    cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}

template <typename T>
std::vector<T> host(const T* p, size_t n) {
    std::vector<T> h(n);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

// max |got - want| / (max |want| + tiny): one number per check against `tol`
void check(const char* what, const std::vector<float>& got, const std::vector<double>& want, double tol) {
    double err = 0, mag = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        err = std::max(err, std::fabs((double) got[i] - want[i]));
        mag = std::max(mag, std::fabs(want[i]));
    }
    const double rel = err / (mag + 1e-30);
    const bool ok = rel <= tol && std::isfinite(rel);
    if (!ok) ++g_fail;
    std::printf("  %-44s max rel err %.2e (tol %.0e)  %s\n", what, rel, tol, ok ? "ok" : "<-- WRONG");
}

std::vector<uint16_t> to_half(const std::vector<float>& v) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        const __half x = __float2half(v[i]);
        h[i] = *reinterpret_cast<const uint16_t*>(&x);
    }
    return h;
}
double half_val(uint16_t b) { return (double) __half2float(*reinterpret_cast<const __half*>(&b)); }
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

void test_hc(int T) {
    std::printf("hyper-connections, T=%d\n", T);
    auto streams = randn((size_t) T * kHcFlat);
    auto fn = randn((size_t) kHcMixes * kHcFlat, 0.01f), base = randn(kHcMixes, 0.5f), scale = randn(2, 0.5f);
    float *d_s = dev(streams), *d_fn = dev(fn), *d_b = dev(base), *d_sc = dev(scale);
    float *d_pre, *d_post, *d_y;
    cudaMalloc(&d_pre, T * kHc * 4);
    cudaMalloc(&d_post, T * kHc * 4);
    cudaMalloc(&d_y, (size_t) T * kEmbd * 4);
    hc_mix(d_s, {d_fn, d_b, d_sc, kHcMixes}, T, d_pre, d_post, d_y, nullptr);
    std::vector<double> want_pre(T * kHc), want_post(T * kHc), want_y((size_t) T * kEmbd);
    for (int t = 0; t < T; ++t) {
        const float* x = streams.data() + (size_t) t * kHcFlat;
        double ss = 0, mix[kHcMixes] = {};
        for (int i = 0; i < kHcFlat; ++i) {
            ss += (double) x[i] * x[i];
            for (int j = 0; j < kHcMixes; ++j) mix[j] += (double) fn[(size_t) j * kHcFlat + i] * x[i];
        }
        const double r = 1.0 / std::sqrt(ss / kHcFlat + kRmsEps);
        for (int s = 0; s < kHc; ++s) {
            want_pre[t * kHc + s] = sigmoid(mix[s] * r * scale[0] + base[s]) + kHcEps;
            want_post[t * kHc + s] = kHcMagnitude * sigmoid(mix[kHc + s] * r * scale[1] + base[kHc + s]) + kHcEps;
        }
        for (int d = 0; d < kEmbd; ++d) {
            double s = 0;
            for (int c = 0; c < kHc; ++c) s += want_pre[t * kHc + c] * x[c * kEmbd + d];
            want_y[(size_t) t * kEmbd + d] = s;
        }
    }
    check("hc_mix pre", host(d_pre, T * kHc), want_pre, 1e-5);
    check("hc_mix post", host(d_post, T * kHc), want_post, 1e-5);
    check("hc_mix collapsed y", host(d_y, (size_t) T * kEmbd), want_y, 1e-5);
    // apply, from the kernel's own y and post (the reference recomputes nothing)
    auto y = host(d_y, (size_t) T * kEmbd), post = host(d_post, T * kHc);
    hc_apply(d_s, d_y, d_post, T, nullptr);
    std::vector<double> want_s(streams.size());
    for (size_t i = 0; i < streams.size(); ++i) {
        const size_t t = i / kHcFlat, s = (i / kEmbd) % kHc, d = i % kEmbd;
        want_s[i] = streams[i] + (double) post[t * kHc + s] * y[t * kEmbd + d];
    }
    check("hc_apply", host(d_s, streams.size()), want_s, 1e-6);
    // the head: 4 mixes, pre only
    hc_mix(d_s, {d_fn, d_b, d_sc, kHc}, T, d_pre, nullptr, d_y, nullptr);
    auto s2 = host(d_s, streams.size());
    for (int t = 0; t < T; ++t) {
        const float* x = s2.data() + (size_t) t * kHcFlat;
        double ss = 0, mix[kHc] = {};
        for (int i = 0; i < kHcFlat; ++i) {
            ss += (double) x[i] * x[i];
            for (int j = 0; j < kHc; ++j) mix[j] += (double) fn[(size_t) j * kHcFlat + i] * x[i];
        }
        const double r = 1.0 / std::sqrt(ss / kHcFlat + kRmsEps);
        double p[kHc];
        for (int s = 0; s < kHc; ++s) p[s] = sigmoid(mix[s] * r * scale[0] + base[s]) + kHcEps;
        for (int d = 0; d < kEmbd; ++d) {
            double s = 0;
            for (int c = 0; c < kHc; ++c) s += p[c] * x[c * kEmbd + d];
            want_y[(size_t) t * kEmbd + d] = s;
        }
    }
    check("hc head collapse", host(d_y, (size_t) T * kEmbd), want_y, 1e-5);
    // expand
    auto e = randn((size_t) T * kEmbd);
    float* d_e = dev(e);
    hc_expand(d_e, d_s, T, nullptr);
    std::vector<double> want_e(streams.size());
    for (size_t i = 0; i < want_e.size(); ++i) want_e[i] = e[(i / kHcFlat) * kEmbd + i % kEmbd];
    check("hc_expand", host(d_s, streams.size()), want_e, 0);
    for (void* p : {(void*) d_s, (void*) d_fn, (void*) d_b, (void*) d_sc, (void*) d_pre, (void*) d_post, (void*) d_y,
                    (void*) d_e})
        cudaFree(p);
}

void test_norm_rope(int T) {
    std::printf("norm and RoPE, T=%d\n", T);
    const int width = kQRank;
    auto x = randn((size_t) T * width, 3.f), w = randn(width);
    float *d_x = dev(x), *d_w = dev(w), *d_o;
    cudaMalloc(&d_o, x.size() * 4);
    rms_norm_rows(d_x, d_w, width, T, width, kRmsEps, d_o, nullptr);
    std::vector<double> want(x.size());
    for (int t = 0; t < T; ++t) {
        double ss = 0;
        for (int i = 0; i < width; ++i) ss += (double) x[t * width + i] * x[t * width + i];
        const double r = 1.0 / std::sqrt(ss / width + kRmsEps);
        for (int i = 0; i < width; ++i) want[t * width + i] = x[t * width + i] * r * w[i];
    }
    check("rms_norm_rows", host(d_o, x.size()), want, 1e-5);
    // RoPE on the last 64 dims of 32 heads of 256, positions far out (the angle's accuracy)
    const int heads = kHeadsSparse;
    auto q = randn((size_t) T * heads * kQkHead);
    std::vector<int32_t> pos(T);
    for (int t = 0; t < T; ++t) pos[t] = 100000 + 17 * t;
    float* d_q = dev(q);
    int32_t* d_pos = dev(pos);
    for (const float theta : {kRopeThetaSliding, kRopeThetaSparse}) {
        cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
        rope_gptj(d_q, T * heads, kQkHead, kNope, kRope, heads, d_pos, theta, nullptr);
        std::vector<double> wq(q.begin(), q.end());
        for (int r = 0; r < T * heads; ++r)
            for (int k = 0; k < kRope / 2; ++k) {
                const double inv = std::pow((double) theta, -2.0 * k / kRope), ang = pos[r / heads] * inv;
                const size_t i = (size_t) r * kQkHead + kNope + 2 * k;
                wq[i] = q[i] * std::cos(ang) - q[i + 1] * std::sin(ang);
                wq[i + 1] = q[i] * std::sin(ang) + q[i + 1] * std::cos(ang);
            }
        check(theta == kRopeThetaSliding ? "rope_gptj theta 1e4" : "rope_gptj theta 1e7", host(d_q, q.size()), wq,
              1e-3);
    }
    for (void* p : {(void*) d_x, (void*) d_w, (void*) d_o, (void*) d_q, (void*) d_pos}) cudaFree(p);
}

void test_router(int T) {
    std::printf("router, T=%d\n", T);
    auto logits = randn((size_t) T * kExperts, 2.f), bias = randn(kExperts, 0.1f);
    float *d_l = dev(logits), *d_b = dev(bias), *d_w;
    int32_t* d_i;
    cudaMalloc(&d_i, T * kExpertsUsed * 4);
    cudaMalloc(&d_w, T * kExpertsUsed * 4);
    route_sigmoid_topk(d_l, d_b, T, d_i, d_w, nullptr);
    auto ids = host(d_i, T * kExpertsUsed);
    auto w = host(d_w, T * kExpertsUsed);
    std::vector<double> want_w(T * kExpertsUsed);
    int id_bad = 0;
    for (int t = 0; t < T; ++t) {
        std::vector<int> order(kExperts);
        std::iota(order.begin(), order.end(), 0);
        auto key = [&](int e) { return sigmoid(logits[t * kExperts + e]) + bias[e]; };
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key(a) > key(b); });
        double sum = 0;
        for (int k = 0; k < kExpertsUsed; ++k) sum += sigmoid(logits[t * kExperts + order[k]]);
        for (int k = 0; k < kExpertsUsed; ++k) {
            if (ids[t * kExpertsUsed + k] != order[k]) ++id_bad;
            want_w[t * kExpertsUsed + k] = sigmoid(logits[t * kExperts + order[k]]) / sum * kRoutedScale;
        }
    }
    if (id_bad) ++g_fail;
    std::printf("  %-44s %d of %d differ  %s\n", "route ids (in order)", id_bad, T * kExpertsUsed,
                id_bad ? "<-- WRONG" : "ok");
    check("route weights", w, want_w, 1e-5);
    for (void* p : {(void*) d_l, (void*) d_b, (void*) d_w, (void*) d_i}) cudaFree(p);
}

void test_engram(int T) {
    std::printf("Engram, T=%d\n", T);
    const float eps_e = 0.0078125f;   // bfloat16's eps: the checkpoint's dtype
    auto rows = randn((size_t) T * kEngramHeads * kEngramHeadDim, 0.05f), hn = randn(kEngramHeads * kEngramHeadDim);
    float *d_r = dev(rows), *d_hn = dev(hn), *d_o;
    cudaMalloc(&d_o, (size_t) T * kEngramDim * 4);
    engram_rows(d_r, d_hn, eps_e, T, d_o, nullptr);
    std::vector<double> want((size_t) T * kEngramDim);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < kEngramHeads; ++h) {
            const float* x = rows.data() + ((size_t) t * kEngramHeads + h) * kEngramHeadDim;
            double amax = 0;
            for (int i = 0; i < kEngramHeadDim; ++i) amax = std::max(amax, (double) std::fabs(x[i]));
            const float sc = (float) std::max(amax, 1e-12) / 448.f;
            double v[kEngramHeadDim], ss = 0;
            for (int i = 0; i < kEngramHeadDim; ++i) {
                const float q = std::min(std::max(x[i] / sc, -448.f), 448.f);
                v[i] = (double) float(__nv_fp8_e4m3(q)) * sc;   // the host conversion: E4M3, round to nearest even
                ss += v[i] * v[i];
            }
            const double r = 1.0 / std::sqrt(ss / kEngramHeadDim + eps_e);
            for (int i = 0; i < kEngramHeadDim; ++i)
                want[(size_t) t * kEngramDim + h * kEngramHeadDim + i] = v[i] * r * hn[h * kEngramHeadDim + i];
        }
    check("engram_rows (fp8 round trip + head norms)", host(d_o, want.size()), want, 1e-5);

    auto key = randn((size_t) T * kHc * kEmbd), value = randn((size_t) T * kEmbd), streams = randn((size_t) T * kHcFlat);
    auto kb = randn(kHc * kEmbd, 0.1f), nk = randn(kHc * kEmbd), nq = randn(kHc * kEmbd), vb = randn(kEmbd, 0.1f),
         pg = randn(kEmbd, 0.1f), phs = randn(kHc * kEmbd, 0.1f);
    float *d_k = dev(key), *d_v = dev(value), *d_s = dev(streams), *d_kb = dev(kb), *d_nk = dev(nk), *d_nq = dev(nq),
          *d_vb = dev(vb), *d_pg = dev(pg), *d_phs = dev(phs);
    EngramWeights ew{d_kb, d_nk, d_nq, d_vb, d_pg, d_phs, kRmsEps, eps_e};
    engram_apply(d_k, d_v, ew, T, d_s, nullptr);
    std::vector<double> want_s(streams.begin(), streams.end());
    for (int t = 0; t < T; ++t) {
        double main[kEmbd], vs = 0;
        for (int d = 0; d < kEmbd; ++d) vs += std::pow((double) value[t * kEmbd + d] + vb[d], 2);
        const double vr = 1.0 / std::sqrt(vs / kEmbd + eps_e);
        for (int d = 0; d < kEmbd; ++d) main[d] = pg[d] * ((double) value[t * kEmbd + d] + vb[d]) * vr;
        for (int s = 0; s < kHc; ++s) {
            double ks = 0, qs = 0, dot = 0;
            for (int d = 0; d < kEmbd; ++d) {
                const double kv = (double) key[((size_t) t * kHc + s) * kEmbd + d] + kb[s * kEmbd + d];
                const double qv = streams[(size_t) t * kHcFlat + s * kEmbd + d];
                ks += kv * kv;
                qs += qv * qv;
                dot += kv * nk[s * kEmbd + d] * qv * nq[s * kEmbd + d];
            }
            const double g = dot / std::sqrt(ks / kEmbd + kRmsEps) / std::sqrt(qs / kEmbd + kRmsEps) / std::sqrt(kEmbd);
            const double gate = sigmoid(std::sqrt(std::max(std::fabs(g), 1e-6)) * (g > 0 ? 1 : (g < 0 ? -1 : 0)));
            for (int d = 0; d < kEmbd; ++d)
                want_s[(size_t) t * kHcFlat + s * kEmbd + d] += gate * main[d] + phs[s * kEmbd + d] * main[d];
        }
    }
    check("engram_apply", host(d_s, streams.size()), want_s, 1e-5);
    for (void* p : {(void*) d_r, (void*) d_hn, (void*) d_o, (void*) d_k, (void*) d_v, (void*) d_s, (void*) d_kb,
                    (void*) d_nk, (void*) d_nq, (void*) d_vb, (void*) d_pg, (void*) d_phs})
        cudaFree(p);
}

void test_mla(int heads, int T, bool selected) {
    std::printf("MLA, %d heads, T=%d, %s\n", heads, T, selected ? "selected positions" : "window");
    constexpr int cap = 1024;
    const int kvb_rows = kNope + kVHead;
    auto q = randn((size_t) T * heads * kQkHead), gate = randn((size_t) T * heads * kVHead);
    auto kvb = to_half(randn((size_t) heads * kvb_rows * kKvRank, 0.05f));
    auto ckv = to_half(randn((size_t) cap * kKvRank)), kpe = to_half(randn((size_t) cap * kRope));
    auto sink = randn(heads, 1.f);
    const int qpos0 = 1500;   // the ring holds positions 1500 - 1023 .. 1500 + T
    std::vector<int32_t> lo(T), hi(T), sel, sel_n(T);
    const int sel_stride = 300;
    for (int t = 0; t < T; ++t) {
        hi[t] = qpos0 + t + 1;
        lo[t] = std::max(0, hi[t] - kWindow);
    }
    if (selected) {
        sel.resize((size_t) T * sel_stride);
        std::uniform_int_distribution<int> u(qpos0 - 1000, qpos0 + T + 3);   // some past the row: masked
        for (int t = 0; t < T; ++t) {
            sel_n[t] = sel_stride - 7 * t;
            for (int j = 0; j < sel_stride; ++j) sel[(size_t) t * sel_stride + j] = u(g_rng);
        }
    }
    float *d_q = dev(q), *d_gate = dev(gate), *d_sink = dev(sink), *d_ql, *d_ol, *d_out;
    uint16_t *d_kvb = dev(kvb), *d_ckv = dev(ckv), *d_kpe = dev(kpe);
    int32_t *d_lo = dev(lo), *d_hi = dev(hi), *d_sel = selected ? dev(sel) : nullptr, *d_seln = dev(sel_n);
    cudaMalloc(&d_ql, (size_t) T * heads * kKvRank * 4);
    cudaMalloc(&d_ol, (size_t) T * heads * kKvRank * 4);
    cudaMalloc(&d_out, (size_t) T * heads * kVHead * 4);
    const float scale = 1.f / std::sqrt((float) kQkHead);
    mla_absorb_q(d_q, d_kvb, heads, T, d_ql, nullptr);
    mla_attend(d_ql, d_q, {d_ckv, d_kpe, cap}, d_lo, d_hi, d_sel, d_seln, sel_stride, d_sink, heads, T, scale, d_ol,
               nullptr);
    mla_value_out(d_ol, d_kvb, d_gate, heads, T, d_out, nullptr);

    std::vector<double> want_ql((size_t) T * heads * kKvRank), want_ol(want_ql.size()), want_out((size_t) T * heads * kVHead);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < heads; ++h) {
            const float* qh = q.data() + ((size_t) t * heads + h) * kQkHead;
            const uint16_t* wk = kvb.data() + (size_t) h * kvb_rows * kKvRank;
            double* ql = want_ql.data() + ((size_t) t * heads + h) * kKvRank;
            for (int c = 0; c < kKvRank; ++c) {
                double s = 0;
                for (int d = 0; d < kNope; ++d) s += qh[d] * half_val(wk[(size_t) d * kKvRank + c]);
                ql[c] = s;
            }
            std::vector<int> keys;
            if (selected)
                for (int j = 0; j < sel_n[t]; ++j) keys.push_back(sel[(size_t) t * sel_stride + j]);
            else
                for (int p = lo[t]; p < hi[t]; ++p) keys.push_back(p);
            std::vector<double> sc;
            std::vector<int> slots;
            double mx = sink[h];
            for (int p : keys) {
                if (p < 0 || p >= hi[t]) continue;
                const int slot = p % cap;
                double s = 0;
                for (int c = 0; c < kKvRank; ++c) s += ql[c] * half_val(ckv[(size_t) slot * kKvRank + c]);
                for (int d = 0; d < kRope; ++d) s += qh[kNope + d] * half_val(kpe[(size_t) slot * kRope + d]);
                s *= scale;
                sc.push_back(s);
                slots.push_back(slot);
                mx = std::max(mx, s);
            }
            double den = std::exp(sink[h] - mx);
            for (double s : sc) den += std::exp(s - mx);
            double* ol = want_ol.data() + ((size_t) t * heads + h) * kKvRank;
            for (int c = 0; c < kKvRank; ++c) {
                double s = 0;
                for (size_t k = 0; k < sc.size(); ++k)
                    s += std::exp(sc[k] - mx) / den * half_val(ckv[(size_t) slots[k] * kKvRank + c]);
                ol[c] = s;
            }
            const uint16_t* wv = wk + (size_t) kNope * kKvRank;
            for (int v = 0; v < kVHead; ++v) {
                double s = 0;
                for (int c = 0; c < kKvRank; ++c) s += ol[c] * half_val(wv[(size_t) v * kKvRank + c]);
                const size_t o = ((size_t) t * heads + h) * kVHead + v;
                want_out[o] = s * sigmoid(gate[o]);
            }
        }
    check("mla_absorb_q", host(d_ql, want_ql.size()), want_ql, 1e-5);
    check("mla_attend", host(d_ol, want_ol.size()), want_ol, 1e-4);
    check("mla_value_out", host(d_out, want_out.size()), want_out, 1e-4);
    for (void* p : {(void*) d_q, (void*) d_gate, (void*) d_sink, (void*) d_ql, (void*) d_ol, (void*) d_out, (void*) d_kvb,
                    (void*) d_ckv, (void*) d_kpe, (void*) d_lo, (void*) d_hi, (void*) d_sel, (void*) d_seln})
        cudaFree(p);
}

void test_indexer(int T, int n_pos) {
    std::printf("indexer, T=%d, %d positions\n", T, n_pos);
    const int groups = n_pos / kIdxRatio;
    auto tk = randn((size_t) n_pos * kIdxDim), kvw = randn(kIdxDim * kIdxDim, 0.1f), gw = randn(kIdxDim * kIdxDim, 0.1f),
         ape = randn(kIdxRatio * kIdxDim, 0.5f), kn = randn(kIdxDim);
    float *d_tk = dev(tk), *d_kvw = dev(kvw), *d_gw = dev(gw), *d_ape = dev(ape), *d_kn = dev(kn), *d_gk;
    cudaMalloc(&d_gk, (size_t) std::max(groups, 1) * kIdxDim * 4);
    indexer_pool(d_tk, {d_kvw, d_gw, d_ape, d_kn, kRmsEps}, 0, groups, d_gk, nullptr);
    std::vector<double> want_gk((size_t) groups * kIdxDim);
    for (int g = 0; g < groups; ++g) {
        double pooled[kIdxDim], ss = 0;
        for (int o = 0; o < kIdxDim; ++o) {
            double val[kIdxRatio], gate[kIdxRatio], mx = -1e300;
            for (int i = 0; i < kIdxRatio; ++i) {
                double v = 0, gt = 0;
                for (int d = 0; d < kIdxDim; ++d) {
                    v += (double) kvw[o * kIdxDim + d] * tk[((size_t) g * kIdxRatio + i) * kIdxDim + d];
                    gt += (double) gw[o * kIdxDim + d] * tk[((size_t) g * kIdxRatio + i) * kIdxDim + d];
                }
                val[i] = v;
                gate[i] = gt + ape[i * kIdxDim + o];
                mx = std::max(mx, gate[i]);
            }
            double den = 0, p = 0;
            for (int i = 0; i < kIdxRatio; ++i) den += std::exp(gate[i] - mx);
            for (int i = 0; i < kIdxRatio; ++i) p += val[i] * std::exp(gate[i] - mx) / den;
            pooled[o] = p;
            ss += p * p;
        }
        const double r = 1.0 / std::sqrt(ss / kIdxDim + kRmsEps);
        double* out = want_gk.data() + (size_t) g * kIdxDim;
        for (int o = 0; o < kIdxDim; ++o) out[o] = pooled[o] * r * kn[o];
        for (int k = 0; k < kRope / 2; ++k) {
            const double ang = (double) (g * kIdxRatio) * std::pow((double) kRopeThetaSparse, -2.0 * k / kRope);
            double* p = out + (kIdxDim - kRope) + 2 * k;
            const double a = p[0], b = p[1];
            p[0] = a * std::cos(ang) - b * std::sin(ang);
            p[1] = a * std::sin(ang) + b * std::cos(ang);
        }
    }
    auto gk = host(d_gk, want_gk.size());
    check("indexer_pool", gk, want_gk, 1e-5);

    auto q = randn((size_t) T * kIdxHeads * kIdxDim), hw = randn(T * kIdxHeads);
    std::vector<int32_t> pos(T);
    for (int t = 0; t < T; ++t) pos[t] = n_pos - T + t;
    float *d_q = dev(q), *d_hw = dev(hw), *d_scr;
    int32_t *d_pos = dev(pos), *d_sel, *d_seln;
    cudaMalloc(&d_scr, indexer_scratch_floats(groups, T) * 4);
    cudaMalloc(&d_sel, (size_t) T * kIdxGroupTopk * kIdxRatio * 4);
    cudaMalloc(&d_seln, T * 4);
    indexer_select(d_q, d_hw, d_gk, groups, d_pos, T, d_scr, d_sel, d_seln, nullptr);
    auto sel = host(d_sel, (size_t) T * kIdxGroupTopk * kIdxRatio);
    auto seln = host(d_seln, T);
    int bad = 0;
    for (int t = 0; t < T; ++t) {
        const int own = pos[t] / kIdxRatio, valid = std::min(own, groups), k = std::min(kIdxGroupTopk - 1, valid);
        std::vector<double> lg(valid);
        for (int g = 0; g < valid; ++g) {   // from the kernel's own pooled keys: the selection alone is checked here
            double l = 0;
            for (int h = 0; h < kIdxHeads; ++h) {
                double s = 0;
                for (int d = 0; d < kIdxDim; ++d) s += (double) q[((size_t) t * kIdxHeads + h) * kIdxDim + d] * gk[(size_t) g * kIdxDim + d];
                l += hw[t * kIdxHeads + h] / std::sqrt((double) kIdxHeads * kIdxDim) * std::max(s, 0.0);
            }
            lg[g] = l;
        }
        std::vector<int> order(valid);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        std::vector<char> want_g(valid, 0), got_g(valid, 0);
        for (int i = 0; i < k; ++i) want_g[order[i]] = 1;
        const double thr = k > 0 ? lg[order[k - 1]] : 0;
        int n_want = k * kIdxRatio;
        for (int i = 0; i < kIdxRatio; ++i) n_want += own * kIdxRatio + i <= pos[t];
        if (seln[t] != n_want) ++bad;
        for (int j = 0; j < seln[t] - (n_want - k * kIdxRatio) && j < seln[t]; j += kIdxRatio) {
            const int g = sel[(size_t) t * kIdxGroupTopk * kIdxRatio + j] / kIdxRatio;
            if (g >= 0 && g < valid) got_g[g] = 1;
        }
        for (int g = 0; g < valid; ++g)
            if (got_g[g] != want_g[g] && std::fabs(lg[g] - thr) > 1e-5 * (std::fabs(thr) + 1e-30)) ++bad;
        for (int j = 1; j < seln[t]; ++j)   // ascending positions: the sum order is fixed
            if (sel[(size_t) t * kIdxGroupTopk * kIdxRatio + j] <= sel[(size_t) t * kIdxGroupTopk * kIdxRatio + j - 1]) ++bad;
    }
    if (bad) ++g_fail;
    std::printf("  %-44s %d differ (beyond ties at the k-th)  %s\n", "indexer_select groups and order", bad,
                bad ? "<-- WRONG" : "ok");
    for (void* p : {(void*) d_tk, (void*) d_kvw, (void*) d_gw, (void*) d_ape, (void*) d_kn, (void*) d_gk, (void*) d_q,
                    (void*) d_hw, (void*) d_scr, (void*) d_pos, (void*) d_sel, (void*) d_seln})
        cudaFree(p);
}

// the drafter's helpers: y += g * f, a vocab-wide argmax with ties, the target picks with the block appended
void test_dfly_helpers() {
    std::printf("dfly helpers\n");
    auto y = randn(kEmbd), g = randn(kEmbd), f = randn(kEmbd);
    std::vector<double> want(kEmbd);
    for (int i = 0; i < kEmbd; ++i) want[i] = (double) y[i] + (double) g[i] * f[i];
    float *d_y = dev(y), *d_g = dev(g), *d_f = dev(f);
    add_scaled(d_y, d_g, d_f, kEmbd, nullptr);
    check("add_scaled", host(d_y, kEmbd), want, 1e-6);

    int bad = 0;
    for (int trial = 0; trial < 3; ++trial) {
        auto x = randn(kVocab);
        const int at = (int) (g_rng() % kVocab);
        x[at] = 50.f;
        if (trial > 0) x[(at + 7919 * trial) % kVocab] = 50.f;   // a tie: the lower index wins
        const int want_i = (int) (std::max_element(x.begin(), x.end()) - x.begin());
        float* d_x = dev(x);
        int32_t* d_o;
        cudaMalloc(&d_o, 4);
        argmax_row(d_x, kVocab, d_o, nullptr);
        bad += host(d_o, 1)[0] != want_i;
        cudaFree(d_x);
        cudaFree(d_o);
    }
    if (bad) ++g_fail;
    std::printf("  %-44s %d of 3 differ  %s\n", "argmax_row (vocab, ties)", bad, bad ? "<-- WRONG" : "ok");

    bad = 0;
    for (int n : {-1, 0, 5, kIdxGroupTopk * kIdxRatio}) {
        std::vector<int32_t> src(kIdxGroupTopk * kIdxRatio);
        for (size_t i = 0; i < src.size(); ++i) src[i] = (int32_t) (g_rng() % 100000);
        const std::vector<int32_t> sn{n}, p0{123456};
        int32_t *d_src = dev(src), *d_sn = dev(sn), *d_p0 = dev(p0), *d_dst, *d_dn;
        cudaMalloc(&d_dst, (size_t) kDflyBlock * kDflySelStride * 4);
        cudaMalloc(&d_dn, kDflyBlock * 4);
        sel_with_block(d_src, d_sn, d_p0, kDflyBlock, kDflyBlock, d_dst, d_dn, nullptr);
        auto dst = host(d_dst, (size_t) kDflyBlock * kDflySelStride);
        auto dn = host(d_dn, kDflyBlock);
        for (int t = 0; t < kDflyBlock; ++t) {
            if (n < 0) { bad += dn[t] != -1; continue; }
            bad += dn[t] != n + kDflyBlock;
            for (int i = 0; i < n; ++i) bad += dst[(size_t) t * kDflySelStride + i] != src[i];
            for (int i = 0; i < kDflyBlock; ++i) bad += dst[(size_t) t * kDflySelStride + n + i] != 123456 + i;
        }
        for (void* p : {(void*) d_src, (void*) d_sn, (void*) d_p0, (void*) d_dst, (void*) d_dn}) cudaFree(p);
    }
    if (bad) ++g_fail;
    std::printf("  %-44s %d differ  %s\n", "sel_with_block", bad, bad ? "<-- WRONG" : "ok");
    for (void* p : {(void*) d_y, (void*) d_g, (void*) d_f}) cudaFree(p);
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    // the layer pattern against the published swa_pattern
    const int pattern[kLayers] = {0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
                                  0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0};
    int bad = 0;
    for (int l = 0; l < kLayers; ++l) bad += sliding_layer(l) != (pattern[l] == 1);
    if (bad) ++g_fail;
    std::printf("layer pattern: %d of %d differ  %s\n", bad, kLayers, bad ? "<-- WRONG" : "ok");
    for (int T : {1, 3}) {
        test_hc(T);
        test_norm_rope(T);
        test_router(T);
        test_engram(T);
    }
    test_mla(kHeadsSliding, 3, false);
    test_mla(kHeadsSparse, 1, false);
    test_mla(kHeadsSparse, 3, true);
    test_indexer(1, 1000);    // fewer groups than the top-k: every earlier group
    test_indexer(3, 12000);   // 3000 groups, 511 picked
    test_dfly_helpers();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(e));
        ++g_fail;
    }
    std::printf("hy4_kernels_parity: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
