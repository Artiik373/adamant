// Standalone round-trip verification for adamant::kernels::ms (AVX2 + scalar 1MAD & FWHT).
// Build: g++ -O3 -march=native -std=c++20 -Isrc tests/test_ms_1mad.cpp -o build/test_ms_1mad && ./build/test_ms_1mad

#include "kernels/ms_1mad_avx2.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace adamant::kernels::ms;

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

static void pack_row_ref(
    uint8_t* dst,
    float scale,
    const std::vector<uint32_t>& states,
    int k
) {
    const int in_f = static_cast<int>(states.size());
    const uint16_t d16 = fp32_to_fp16(scale);
    const uint16_t s0 = static_cast<uint16_t>(states[0] & kStateMask);
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
static bool test_gemv_shape(int out_f, int in_f, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist_x(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_d(0.001f, 0.02f);
    std::uniform_int_distribution<uint32_t> dist_s0(0, kStateMask);
    std::uniform_int_distribution<uint32_t> dist_sym(0, (1u << K) - 1u);

    const size_t stride = ms_row_bytes(in_f, K);
    std::vector<uint8_t> weights(static_cast<size_t>(out_f) * stride, 0);
    std::vector<float> x_rot(in_f);
    for (int c = 0; c < in_f; ++c) x_rot[c] = dist_x(rng);

    std::vector<float> ref_out(out_f, 0.0f);
    std::vector<uint32_t> row_states(in_f);

    for (int r = 0; r < out_f; ++r) {
        float d_raw = dist_d(rng);
        float d = fp16_to_fp32(fp32_to_fp16(d_raw));
        uint32_t s = dist_s0(rng);
        row_states[0] = s;
        for (int c = 1; c < in_f; ++c) {
            s = ((s << K) | dist_sym(rng)) & kStateMask;
            row_states[c] = s;
        }
        pack_row_ref(weights.data() + static_cast<size_t>(r) * stride, d_raw, row_states, K);

        double acc = 0.0;
        for (int c = 0; c < in_f; ++c) {
            uint32_t x = row_states[c] * kMadMul + kMadAdd;
            int32_t v = static_cast<int32_t>((x & 0xFFu) + ((x >> 8) & 0xFFu) +
                                             ((x >> 16) & 0xFFu) + (x >> 24)) - 510;
            acc += static_cast<double>(v) * static_cast<double>(x_rot[c]);
        }
        ref_out[r] = static_cast<float>(acc * static_cast<double>(d));
    }

    std::vector<float> got_out(out_f, 0.0f);
    ms_1mad_gemv<K>(weights.data(), x_rot.data(), got_out.data(), out_f, in_f);

    float max_err = 0.0f;
    for (int r = 0; r < out_f; ++r) {
        max_err = std::max(max_err, std::fabs(got_out[r] - ref_out[r]));
    }
    std::printf("[MS_C%d %dx%d] max_abs_err = %.6g\n", K, out_f, in_f, max_err);
    return max_err < 1e-3f;
}

static bool test_fwht_orthonormal(int n, int block_size) {
    std::mt19937 rng(42 + n);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(n), y(n), z(n);
    std::vector<int8_t> signs(n);
    for (int i = 0; i < n; ++i) {
        x[i] = dist(rng);
        signs[i] = (rng() & 1u) ? 1 : -1;
    }

    fwht_rotate_input(x.data(), signs.data(), y.data(), n, block_size);

    // Check norm preservation: ||y||_2 == ||x||_2
    double nx = 0.0, ny = 0.0;
    for (int i = 0; i < n; ++i) {
        nx += static_cast<double>(x[i]) * x[i];
        ny += static_cast<double>(y[i]) * y[i];
    }
    double norm_err = std::fabs(std::sqrt(nx) - std::sqrt(ny));
    std::printf("[FWHT n=%d blk=%d] norm_err = %.6g\n", n, block_size, norm_err);
    return norm_err < 1e-4;
}

int main() {
    bool ok = true;
    ok &= test_fwht_orthonormal(2560, 512);
    ok &= test_fwht_orthonormal(640, 128);
    ok &= test_gemv_shape<3>(64, 2560, 101);
    ok &= test_gemv_shape<3>(64, 640, 102);
    ok &= test_gemv_shape<4>(64, 2560, 103);
    ok &= test_gemv_shape<4>(64, 640, 104);
    ok &= test_gemv_shape<6>(64, 2560, 105);
    ok &= test_gemv_shape<6>(64, 640, 106);

    if (!ok) {
        std::fprintf(stderr, "FAILED\n");
        return 1;
    }
    std::printf("ALL MS_1MAD TESTS PASSED\n");
    return 0;
}
