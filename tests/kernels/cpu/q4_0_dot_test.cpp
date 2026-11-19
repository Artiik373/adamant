// tests/kernels/cpu/q4_0_dot_test.cpp - the Q4_0 x Q8_0 dot product of both CPU tables against a scalar reference.
//
//     build/q4_0_dot_test
//
// Random rows at a model's row widths are quantized to Q4_0 (weights) and Q8_0 (activations); each table's dot
// must match the sum over the dequantized blocks within float rounding.
#define GGML_COMMON_DECL_CPP
#include "basalt/ggml/ggml.h"
#include "basalt/ggml/ggml_common.h"
#include "basalt/kernels/cpu/expert_layout.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

double reference(const block_q4_0* x, const block_q8_0* y, int nb) {
    double s = 0;
    for (int b = 0; b < nb; ++b) {
        int sumi = 0;
        for (int j = 0; j < QK4_0 / 2; ++j) {
            sumi += ((x[b].qs[j] & 0x0F) - 8) * y[b].qs[j];
            sumi += ((x[b].qs[j] >> 4) - 8) * y[b].qs[j + QK4_0 / 2];
        }
        s += (double) sumi * ggml_fp16_to_fp32(x[b].d) * ggml_fp16_to_fp32(y[b].d);
    }
    return s;
}

}  // namespace

int main() {
    namespace cpu = basalt::kernels::cpu;
    const bool on[2] = {cpu::cpu_avx2_ok(), cpu::cpu_avx512_ok()};
    const ggml_type_traits_cpu* tables[2] = {on[0] ? ggml_get_type_traits_cpu_avx2(GGML_TYPE_Q4_0) : nullptr,
                                             on[1] ? ggml_get_type_traits_cpu_avx512(GGML_TYPE_Q4_0) : nullptr};
    const char* names[2] = {"avx2", "avx512"};
    if (!on[0]) {
        std::printf("q4_0_dot_test SKIPPED: this CPU lacks AVX2\n");
        return 0;
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fails = 0;
    for (const int n : {32, 256, 640, 2560, 5120}) {
        std::vector<float> w((size_t) n), a((size_t) n);
        std::vector<block_q4_0> x((size_t) n / QK4_0);
        std::vector<block_q8_0> y((size_t) n / QK8_0);
        for (int rep = 0; rep < 20; ++rep) {
            for (auto& v : w) v = nd(rng);
            for (auto& v : a) v = nd(rng) * 4.f;
            ggml_quantize_chunk(GGML_TYPE_Q4_0, w.data(), x.data(), 0, 1, n);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, a.data(), y.data(), 0, 1, n);
            const double want = reference(x.data(), y.data(), n / QK4_0);
            for (int k = 0; k < 2; ++k) {
                if (tables[k] == nullptr || tables[k]->vec_dot == nullptr) {
                    if (tables[k] != nullptr) { std::printf("%s: no Q4_0 dot   <-- WRONG\n", names[k]); ++fails; }
                    continue;
                }
                float got = 0.f;
                tables[k]->vec_dot(n, &got, 0, x.data(), 0, y.data(), 0, 1);
                if (std::fabs(got - want) > 1e-4 * (1.0 + std::fabs(want))) {
                    std::printf("%s n=%d: %g, reference %g   <-- WRONG\n", names[k], n, got, want);
                    ++fails;
                }
            }
        }
    }
    std::printf("q4_0_dot_test %s\n", fails ? "FAILED" : "passed");
    return fails ? 1 : 0;
}
