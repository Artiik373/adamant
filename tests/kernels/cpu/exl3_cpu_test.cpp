// tests/kernels/cpu/exl3_cpu_test.cpp - the EXL3 CPU kernels against a bit-by-bit oracle, on random trellis data in
// the experts' real geometry (gate/up 2560 -> 640, down 640 -> 2560; 3 and 3.5 bits per weight).
#include "basalt/kernels/cpu/exl3.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace basalt::kernels::cpu;

namespace {

float half_to_f(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const float v = e == 0 ? std::ldexp((float) m, -24) : std::ldexp((float) (1024 + m), e - 25);
    return (h & 0x8000) ? -v : v;
}

uint16_t f_to_half(float f) {   // exact for the values the test draws: round to nearest by search
    uint16_t best = 0;
    float bd = 1e30f;
    for (uint32_t h = 0; h < 0x7C00; ++h) {
        const float d = std::fabs(half_to_f((uint16_t) h) - std::fabs(f));
        if (d < bd) { bd = d; best = (uint16_t) h; }
    }
    return f < 0 ? (uint16_t) (best | 0x8000) : best;
}

// W_inner[k][n] by walking each tile's ring bit by bit
std::vector<double> oracle_inner(const std::vector<uint8_t>& tr, int k, int n, int w) {
    const int nbits = w * 16, ka = w / 16;
    const bool half = (w % 16) == 8;
    std::vector<double> out((size_t) k * n);
    const double kinv = 0.0067672729;   // fp16 0x1EEE
    for (int kt = 0; kt < k / 16; ++kt)
        for (int nt = 0; nt < n / 16; ++nt) {
            const uint8_t* t = tr.data() + ((size_t) kt * (n / 16) + nt) * (size_t) w * 2;
            int s = 0;
            for (int i = 0; i < 256; ++i) {
                s += ka + (half && (i & 1) ? 1 : 0);
                uint32_t win = 0;
                for (int b = 0; b < 16; ++b) {
                    const int pos = ((s - 16 + b) % nbits + nbits) % nbits;
                    uint32_t word;
                    std::memcpy(&word, t + (pos >> 5) * 4, 4);
                    win = (win << 1) | ((word >> (31 - (pos & 31))) & 1u);
                }
                const uint32_t x = win * 0x83DCD12Du;
                const int sum = (int) (x & 255) + (int) ((x >> 8) & 255) + (int) ((x >> 16) & 255) + (int) (x >> 24);
                const int tt = i >> 3, j = i & 7;
                const int r = (tt & 3) * 2 + (j & 1) + ((j & 2) ? 8 : 0), c = (tt >> 2) + ((j & 4) ? 8 : 0);
                out[(size_t) (kt * 16 + r) * n + (size_t) nt * 16 + c] = (sum - 510) * kinv;
            }
        }
    return out;
}

bool parity(int v) {
    bool p = false;
    for (; v; v &= v - 1) p = !p;
    return p;
}

void block_h(std::vector<double>& v) {
    std::vector<double> o(v.size());
    for (size_t b = 0; b < v.size() / 128; ++b)
        for (int i = 0; i < 128; ++i) {
            double a = 0;
            for (int j = 0; j < 128; ++j) a += (parity(i & j) ? -1.0 : 1.0) * v[b * 128 + j];
            o[b * 128 + i] = a / std::sqrt(128.0);
        }
    v = o;
}

int g_fail = 0;

void check(const char* what, double err, double scale, double tol) {
    const bool ok = err <= tol * scale;
    std::printf("  %-34s max err %.2e (scale %.3f) %s\n", what, err, scale, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

void run(int k, int n, int w, bool avx2) {
    std::printf("k %d n %d w %d\n", k, n, w);
    std::mt19937 rng(7u + (unsigned) w + (unsigned) k);
    std::vector<uint8_t> rec(exl3_proj_bytes(k, n, w));
    for (auto& b : rec) b = (uint8_t) rng();
    Exl3Proj p = exl3_proj_at(rec.data(), k, n, w);
    std::normal_distribution<float> nd(0.f, 1.f);
    uint16_t* suh = (uint16_t*) p.suh;
    uint16_t* svh = (uint16_t*) p.svh;
    for (int i = 0; i < k; ++i) suh[i] = f_to_half(nd(rng) * 0.02f);
    for (int i = 0; i < n; ++i) svh[i] = f_to_half(nd(rng));
    std::vector<float> x((size_t) k);
    for (auto& v : x) v = nd(rng);
    const std::vector<double> wi = oracle_inner(rec, k, n, w);
    for (int variant = 0; variant < 4; ++variant) {
        std::vector<double> x1((size_t) k);
        for (int i = 0; i < k; ++i) x1[(size_t) i] = (double) x[(size_t) i] * (variant & 1 ? 1.0 : half_to_f(suh[i]));
        block_h(x1);
        if (variant & 1) for (int i = 0; i < k; ++i) x1[(size_t) i] *= half_to_f(suh[i]);
        std::vector<double> y1((size_t) n, 0.0);
        for (int i = 0; i < k; ++i)
            for (int c = 0; c < n; ++c) y1[(size_t) c] += x1[(size_t) i] * wi[(size_t) i * n + c];
        if (variant & 2) for (int c = 0; c < n; ++c) y1[(size_t) c] *= half_to_f(svh[c]);
        block_h(y1);
        if (!(variant & 2)) for (int c = 0; c < n; ++c) y1[(size_t) c] *= half_to_f(svh[c]);
        double scale = 0;
        for (double v : y1) scale = std::fmax(scale, std::fabs(v));

        std::vector<float> x2((size_t) k), x2v((size_t) k), yr((size_t) n), ya((size_t) n), ys((size_t) n, -1.f);
        exl3_prepare_input(x.data(), p.suh, k, variant, x2.data());
        exl3_rows_ref(p, x2.data(), yr.data(), 0, n, variant);
        double er = 0, ea = 0, ep = 0, es = 0;
        for (int i = 0; i < n; ++i) er = std::fmax(er, std::fabs(yr[(size_t) i] - y1[(size_t) i]));
        char name[64];
        std::snprintf(name, sizeof name, "variant %d ref", variant);
        check(name, er, scale, 1e-5);
        if (!avx2) continue;
        exl3_prepare_input_avx2(x.data(), p.suh, k, variant, x2v.data());
        for (int i = 0; i < k; ++i) ep = std::fmax(ep, std::fabs(x2v[(size_t) i] - x2[(size_t) i]));
        exl3_rows_avx2(p, x2v.data(), ya.data(), 0, n, variant);
        for (int i = 0; i < n; ++i) ea = std::fmax(ea, std::fabs(ya[(size_t) i] - y1[(size_t) i]));
        std::snprintf(name, sizeof name, "variant %d avx2 input", variant);
        check(name, ep, 1.0, 1e-5);
        std::snprintf(name, sizeof name, "variant %d avx2 rows", variant);
        check(name, ea, scale, 1e-4);
        // a row range split on 128-blocks, and one that is not, store the same rows as the whole
        for (const int cut : {128, 256, 130}) {
            std::fill(ys.begin(), ys.end(), -1.f);
            exl3_rows_avx2(p, x2v.data(), ys.data(), 0, cut, variant);
            exl3_rows_avx2(p, x2v.data(), ys.data(), cut, n, variant);
            double e = 0;
            for (int i = 0; i < n; ++i) e = std::fmax(e, std::fabs(ys[(size_t) i] - ya[(size_t) i]));
            es = std::fmax(es, e);
        }
        std::snprintf(name, sizeof name, "variant %d split rows", variant);
        check(name, es, scale, 0.0);
        // the column kernel: a token among others, and a split into column ranges, give the bits of the token alone
        std::vector<float> xo((size_t) k), c1((size_t) n), cm((size_t) n * 3, -1.f);
        for (int i = 0; i < k; ++i) xo[(size_t) i] = x2v[(size_t) (k - 1 - i)] * 1.5f;
        const float* one = x2v.data();
        float* o1 = c1.data();
        exl3_cols_avx2(p, &one, 1, &o1, 0, n);
        const float* xs3[3] = {xo.data(), x2v.data(), xo.data()};
        float* o3[3] = {cm.data(), cm.data() + n, cm.data() + 2 * (size_t) n};
        exl3_cols_avx2(p, xs3, 3, o3, 0, 48);
        exl3_cols_avx2(p, xs3, 3, o3, 48, n);
        double em = 0;
        for (int i = 0; i < n; ++i) em = std::fmax(em, std::fabs(cm[(size_t) n + (size_t) i] - c1[(size_t) i]));
        std::snprintf(name, sizeof name, "variant %d columns, 3 tokens", variant);
        check(name, em, scale, 0.0);
    }
}

// us per projection for 1 and 3 tokens: `cold` walks many records so each one comes from memory, hot repeats one.
void bench(int k, int n, int w) {
    const size_t bytes = exl3_proj_bytes(k, n, w), recs = ((size_t) 1 << 29) / bytes;
    std::vector<uint8_t> buf(bytes * recs);
    std::mt19937_64 rng(1);
    for (size_t i = 0; i + 8 <= buf.size(); i += 8) { const uint64_t v = rng(); std::memcpy(&buf[i], &v, 8); }
    std::vector<float> x2((size_t) k, 0.5f), y((size_t) n * 3);
    const float* xs[3] = {x2.data(), x2.data(), x2.data()};
    float* ys[3] = {y.data(), y.data() + n, y.data() + 2 * (size_t) n};
    for (const int nt : {1, 3})
        for (const bool cold : {true, false}) {
            const auto t0 = std::chrono::steady_clock::now();
            float sink = 0;
            for (size_t r = 0; r < recs; ++r) {
                const Exl3Proj p = exl3_proj_at(buf.data() + (cold ? r : 0) * bytes, k, n, w);
                exl3_cols_avx2(p, xs, nt, ys, 0, n);
                sink += y[0];
            }
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            std::printf("bench k %d n %d w %d, %d tokens, %s: %.1f us/proj (%g)\n", k, n, w, nt, cold ? "cold" : "hot",
                        us / (double) recs, sink);
        }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && !std::strcmp(argv[1], "--bench")) {
        bench(2560, 640, 48);
        bench(640, 2560, 48);
        return 0;
    }
    const bool avx2 = cpu_avx2_ok();
    if (!avx2) std::printf("no AVX2: the plain C rows only\n");
    // fp16 activations round to nearest even
    for (const float v : {0.f, 1.f, -2.5f, 0.1f, 65504.f, 1e-6f, 3.14159f}) {
        uint16_t h;
        exl3_pack_act(&v, 1, &h);
        float back;
        exl3_unpack_act(&h, 1, &back);
        if (std::fabs(back - v) > std::fabs(v) * 1e-3 + 1e-7) { std::printf("fp16 round trip %g -> %g FAIL\n", v, back); ++g_fail; }
    }
    // the 8-wide conversions give the bits of the one-value (plain C) ones: ties, subnormals, overflow, random values
    {
        std::mt19937 rng(11);
        std::vector<float> v(4096);
        for (size_t i = 0; i < v.size(); ++i) {
            const uint32_t r = rng();
            v[i] = std::ldexp((float) (int) (r & 0xFFFFFF) - 8388608.f, (int) ((r >> 24) % 48) - 50);
        }
        const float edge[8] = {1.00048828125f, 1.00146484375f, 65520.f, 1e30f, 5.9e-8f, 2.98e-8f, -0.f, 6.1e-5f};
        std::copy(edge, edge + 8, v.begin());
        std::vector<uint16_t> hv(v.size()), hs(v.size());
        std::vector<float> fv(v.size()), fs(v.size());
        exl3_pack_act(v.data(), (int) v.size(), hv.data());
        exl3_unpack_act(hv.data(), (int) v.size(), fv.data());
        for (size_t i = 0; i < v.size(); ++i) {
            exl3_pack_act(&v[i], 1, &hs[i]);
            exl3_unpack_act(&hs[i], 1, &fs[i]);
        }
        if (std::memcmp(hv.data(), hs.data(), hv.size() * 2) != 0 || std::memcmp(fv.data(), fs.data(), fv.size() * 4) != 0) {
            std::printf("fp16 conversions: 8-wide and one-value bits differ FAIL\n");
            ++g_fail;
        }
    }
    run(2560, 640, 48, avx2);
    run(2560, 640, 56, avx2);
    run(640, 2560, 48, avx2);
    run(640, 2560, 56, avx2);
    std::printf(g_fail ? "FAILED\n" : "ok\n");
    return g_fail ? 1 : 0;
}
