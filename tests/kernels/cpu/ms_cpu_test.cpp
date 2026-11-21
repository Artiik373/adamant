// tests/kernels/cpu/ms_cpu_test.cpp - the MS CPU rows against the plain C rows (same integers) and against W x in
// double (the int8 rotated input's error), on random payloads in the experts' geometry (gate/up 2560 -> 640, down
// 640 -> 2560; K 3, 4, 6), 1 to 5 tokens.  Prints a microbench line per shape.
#include "basalt/kernels/cpu/ms_expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/artifact/gguf_reader.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace basalt::kernels::cpu;

namespace basalt::kernels::cpu::ms_detail {
void rows_avx2(int k, const uint8_t* rows, int n_in, const MsAct* const* a, int nt, float* const* out, int r0, int r1);
void rows_vnni(int k, const uint8_t* rows, int n_in, const MsAct* const* a, int nt, float* const* out, int r0, int r1);
}

namespace {

int g_fail = 0;

void expect(bool ok, const char* what, int k, int n_in, int nt, double v) {
    if (!ok) {
        std::printf("FAIL %s: K %d n_in %d nt %d (%g)\n", what, k, n_in, nt, v);
        ++g_fail;
    }
}

double half_d(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const double v = e == 0 ? std::ldexp((double) m, -24) : std::ldexp((double) (1024 + m), e - 25);
    return (h & 0x8000) ? -v : v;
}

// W (n_out x n_in) in double: each row's states walked symbol by symbol, then H and the signs
std::vector<double> oracle_w(const std::vector<uint8_t>& p, int k, int n_in, int n_out) {
    const size_t stride = basalt::ms_row_bytes((uint64_t) n_in, k);
    const int8_t* signs = (const int8_t*) p.data() + (size_t) n_out * stride;
    const int hb = (int) basalt::ms_hadamard_block((uint64_t) n_in);
    std::vector<double> w((size_t) n_out * n_in), v((size_t) n_in);
    for (int r = 0; r < n_out; ++r) {
        const uint8_t* row = p.data() + (size_t) r * stride;
        uint16_t d16, s0;
        std::memcpy(&d16, row, 2);
        std::memcpy(&s0, row + 2, 2);
        uint32_t st = s0 & 0x3FFFu;
        for (int c = 0; c < n_in; ++c) {
            if (c > 0)
                for (int b = 0; b < k; ++b) {
                    const int bit = (c - 1) * k + b;
                    st = ((st << 1) | ((row[4 + bit / 8] >> (7 - bit % 8)) & 1u)) & 0x3FFFu;
                }
            const uint32_t x = st * 34038481u + 76625530u;
            v[(size_t) c] = (double) ((x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24)) - 510.0;
        }
        // W_rot H: row r of W is H^T applied to the decoded row, H symmetric
        for (int h = 1; h < hb; h <<= 1)
            for (int i = 0; i < n_in; i += 2 * h)
                for (int j = i; j < i + h; ++j) {
                    const double a = v[(size_t) j], b = v[(size_t) (j + h)];
                    v[(size_t) j] = a + b;
                    v[(size_t) (j + h)] = a - b;
                }
        for (int c = 0; c < n_in; ++c) w[(size_t) r * n_in + c] = half_d(d16) / std::sqrt((double) hb) * v[(size_t) c] * signs[c];
    }
    return w;
}

std::vector<uint8_t> payload(std::mt19937& g, int k, int n_in, int n_out) {
    const size_t stride = basalt::ms_row_bytes((uint64_t) n_in, k);
    std::vector<uint8_t> p((size_t) n_out * stride + (size_t) n_in);
    for (auto& b : p) b = (uint8_t) g();
    for (int r = 0; r < n_out; ++r) {
        const uint16_t d16 = (uint16_t) (0x1400 + g() % 0x400);   // fp16 scales near 1e-3
        std::memcpy(p.data() + (size_t) r * stride, &d16, 2);
    }
    for (int c = 0; c < n_in; ++c) p[(size_t) n_out * stride + c] = (g() & 1) ? 1 : 0xFF;
    return p;
}

using RowsFn = void (*)(int, const uint8_t*, int, const MsAct* const*, int, float* const*, int, int);

void check(std::mt19937& g, int k, int n_in, int n_out) {
    const auto p = payload(g, k, n_in, n_out);
    const auto w = oracle_w(p, k, n_in, n_out);
    const int8_t* signs = (const int8_t*) p.data() + (size_t) n_out * basalt::ms_row_bytes((uint64_t) n_in, k);
    std::normal_distribution<float> nd;
    for (int nt = 1; nt <= 5; ++nt) {
        std::vector<std::vector<float>> x((size_t) nt, std::vector<float>((size_t) n_in));
        std::vector<MsAct> act((size_t) nt);
        std::vector<const MsAct*> ap((size_t) nt);
        for (int t = 0; t < nt; ++t) {
            for (auto& v : x[(size_t) t]) v = nd(g);
            ms_act_prepare(x[(size_t) t].data(), signs, n_in, act[(size_t) t]);
            ap[(size_t) t] = &act[(size_t) t];
        }
        std::vector<std::vector<float>> ref((size_t) nt, std::vector<float>((size_t) n_out)), got = ref;
        std::vector<float*> rp((size_t) nt), gp((size_t) nt);
        for (int t = 0; t < nt; ++t) { rp[(size_t) t] = ref[(size_t) t].data(); gp[(size_t) t] = got[(size_t) t].data(); }
        ms_rows_ref(k, p.data(), n_in, ap.data(), nt, rp.data(), 0, n_out);
        RowsFn fns[2] = {ms_detail::rows_avx2, cpu_avxvnni_ok() ? ms_detail::rows_vnni : nullptr};
        for (RowsFn fn : fns) {
            if (fn == nullptr) continue;
            for (int t = 0; t < nt; ++t) std::fill(got[(size_t) t].begin(), got[(size_t) t].end(), NAN);
            fn(k, p.data(), n_in, ap.data(), nt, gp.data(), 3, n_out);   // r0 > 0: the row offset
            double worst = 0, ss = 0;   // against the output's RMS: rows near 0 have no relative error
            for (int t = 0; t < nt; ++t)
                for (int r = 3; r < n_out; ++r) {
                    worst = std::fmax(worst, std::fabs(got[(size_t) t][r] - ref[(size_t) t][r]));
                    ss += (double) ref[(size_t) t][r] * ref[(size_t) t][r];
                }
            worst /= std::sqrt(ss / ((double) nt * (n_out - 3)));
            expect(worst < 1e-5, fn == ms_detail::rows_avx2 ? "avx2 vs ref" : "vnni vs ref", k, n_in, nt, worst);
        }
        double num = 0, den = 0;   // the quantized input against W x
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < n_out; ++r) {
                double y = 0;
                for (int c = 0; c < n_in; ++c) y += w[(size_t) r * n_in + c] * x[(size_t) t][c];
                num += (ref[(size_t) t][r] - y) * (ref[(size_t) t][r] - y);
                den += y * y;
            }
        expect(std::sqrt(num / den) < 1e-2, "ref vs W x", k, n_in, nt, std::sqrt(num / den));
    }
}

void bench(std::mt19937& g, int k, int n_in, int n_out) {
    const auto p = payload(g, k, n_in, n_out);
    const int8_t* signs = (const int8_t*) p.data() + (size_t) n_out * basalt::ms_row_bytes((uint64_t) n_in, k);
    std::vector<float> x((size_t) n_in, 0.5f), y((size_t) n_out * 4);
    MsAct a[4];
    const MsAct* ap[4];
    float* yp[4];
    for (int t = 0; t < 4; ++t) {
        ms_act_prepare(x.data(), signs, n_in, a[t]);
        ap[t] = &a[t];
        yp[t] = y.data() + (size_t) t * n_out;
    }
    for (int nt : {1, 3}) {
        const int reps = 200;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) ms_rows(k, p.data(), n_in, ap, nt, yp, 0, n_out);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("ms_cpu bench: K %d %d -> %d, %d token(s): %.1f us, %.2f G weights/s, one thread (%s)\n", k, n_in,
                    n_out, nt, s / reps * 1e6, (double) n_in * n_out * reps / s * 1e-9,
                    cpu_avxvnni_ok() ? "AVX-VNNI" : "AVX2");
    }
}

}  // namespace

int main() {
    if (!cpu_avx2_ok()) {
        std::printf("ms_cpu_test: no AVX2, skipped\n");
        return 0;
    }
    std::mt19937 g(7);
    for (int k : {3, 4, 6}) {
        check(g, k, 2560, 64);   // gate/up rows (a slice: the oracle is O(rows x n_in))
        check(g, k, 640, 256);   // down rows
    }
    for (int k : {3, 4}) {
        bench(g, k, 2560, 640);
        bench(g, k, 640, 2560);
    }
    if (g_fail) {
        std::printf("ms_cpu_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("ms_cpu_test: ok\n");
    return 0;
}
