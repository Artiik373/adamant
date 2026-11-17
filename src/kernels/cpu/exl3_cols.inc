// src/kernels/cpu/exl3_cols.inc - the EXL3 column kernel, built once for AVX2 and once for AVX-VNNI (the byte sum in
// one instruction).  The includer defines BASALT_EXL3_VNNI and BASALT_EXL3_COLS (the function's name).

namespace basalt::kernels::cpu::exl3_detail {
namespace {

struct Avx2Consts {
    __m256i mul, ones8, ones16;
};

// The byte sums of one group of 8 weights (ring positions 8t..8t+7) as floats.  `T` is a compile-time constant so the
// table rows are fixed addresses.
template <int T>
inline __m256 group8(const RingTable& rt, const uint32_t* wd, const Avx2Consts& k) {
    __m256i win;
    if (T == 0 && !rt.fast[0]) {
        alignas(32) uint32_t w8[8];
        for (int j = 0; j < 8; ++j) w8[j] = be_bits(wd, rt, j);
        win = _mm256_load_si256((const __m256i*) w8);
    } else {
        const __m128i chunk = _mm_loadu_si128((const __m128i*) (wd + rt.u0[T]));
        const __m256i v = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(chunk),
                                              _mm256_load_si256((const __m256i*) rt.shuf[T]));
        win = _mm256_srli_epi32(_mm256_sllv_epi32(v, _mm256_load_si256((const __m256i*) rt.bit[T])), 16);
    }
    const __m256i x = _mm256_mullo_epi32(win, k.mul);
#if BASALT_EXL3_VNNI
    return _mm256_cvtepi32_ps(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), x, k.ones8));   // the sum of the 4 bytes
#else
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(x, k.ones8), k.ones16));
#endif
}

template <int C>
inline void quad(const RingTable& rt, const uint32_t* wd, const Avx2Consts& k, __m256* cd) {
    cd[C * 4 + 0] = group8<C * 4 + 0>(rt, wd, k);
    cd[C * 4 + 1] = group8<C * 4 + 1>(rt, wd, k);
    cd[C * 4 + 2] = group8<C * 4 + 2>(rt, wd, k);
    cd[C * 4 + 3] = group8<C * 4 + 3>(rt, wd, k);
}

constexpr int kColsTok = 8;   // tokens one decode serves

// NT > 0: the token count is a constant (one token keeps its sums in registers); NT = 0: `ntok` tokens.
template <int NT>
void cols_n(const Exl3Proj& p, const RingTable& rt, const float* const* x2, int ntok, float* const* y1, int c0, int c1) {
    const int nt = NT ? NT : ntok;
    const int ktiles = p.k / 16;
    // per token, k tile and t&3: the activations the 8 lanes of a group multiply (rows 2q, 2q+1, 2q+8, 2q+9, twice)
    thread_local std::vector<float> xp_store;
    const size_t xp_tok = (size_t) ktiles * 32;
    xp_store.resize(xp_tok * (size_t) nt + 8);
    float* xp = (float*) (((uintptr_t) xp_store.data() + 31) & ~(uintptr_t) 31);
    float bias[kColsTok];
    for (int t = 0; t < nt; ++t) {
        const float* xt = x2[t];
        for (int kt = 0; kt < ktiles; ++kt)
            for (int q = 0; q < 4; ++q) {
                const float* x = xt + kt * 16 + q * 2;
                _mm256_store_ps(xp + (size_t) t * xp_tok + ((size_t) kt * 4 + q) * 8,
                                _mm256_setr_ps(x[0], x[1], x[8], x[9], x[0], x[1], x[8], x[9]));
            }
        float sumx = 0.f;
        for (int i = 0; i < p.k; ++i) sumx += xt[i];
        bias[t] = 510.f * sumx;
    }
    Avx2Consts k;
    k.mul = _mm256_set1_epi32((int) kMul);
    k.ones8 = _mm256_set1_epi8(1);
    k.ones16 = _mm256_set1_epi16(1);
    const size_t tile_stride = (size_t) (p.n / 16) * (size_t) p.w / 2;   // words between k tiles of one column of tiles
    for (int col = c0 / 16; col < c1 / 16; ++col) {
        __m256 acc[kColsTok][8];
        for (int t = 0; t < nt; ++t)
            for (int c = 0; c < 8; ++c) acc[t][c] = _mm256_setzero_ps();
        const uint32_t* wd = tile_words(p, 0, col);
        for (int kt = 0; kt < ktiles; ++kt, wd += tile_stride) {
            __m256 cd[32];
            quad<0>(rt, wd, k, cd);
            quad<1>(rt, wd, k, cd);
            quad<2>(rt, wd, k, cd);
            quad<3>(rt, wd, k, cd);
            quad<4>(rt, wd, k, cd);
            quad<5>(rt, wd, k, cd);
            quad<6>(rt, wd, k, cd);
            quad<7>(rt, wd, k, cd);
            for (int t = 0; t < nt; ++t) {
                const float* xk = xp + (size_t) t * xp_tok + (size_t) kt * 32;
                const __m256 x0 = _mm256_load_ps(xk), x1 = _mm256_load_ps(xk + 8), x2v = _mm256_load_ps(xk + 16),
                             x3 = _mm256_load_ps(xk + 24);
                for (int c = 0; c < 8; ++c) {
                    __m256 a = acc[t][c];
                    a = _mm256_fmadd_ps(x0, cd[c * 4], a);
                    a = _mm256_fmadd_ps(x1, cd[c * 4 + 1], a);
                    a = _mm256_fmadd_ps(x2v, cd[c * 4 + 2], a);
                    acc[t][c] = _mm256_fmadd_ps(x3, cd[c * 4 + 3], a);
                }
            }
        }
        // lanes 0..3 of accumulator c are column c, lanes 4..7 column c + 8
        for (int t = 0; t < nt; ++t)
            for (int c = 0; c < 8; ++c) {
                alignas(32) float a[8];
                _mm256_store_ps(a, acc[t][c]);
                y1[t][col * 16 + c] = (a[0] + a[1] + a[2] + a[3] - bias[t]) * kScale;
                y1[t][col * 16 + c + 8] = (a[4] + a[5] + a[6] + a[7] - bias[t]) * kScale;
            }
    }
}

}  // namespace

void BASALT_EXL3_COLS(const Exl3Proj& p, const float* const* x2, int nt, float* const* y1, int c0, int c1) {
    const RingTable& rt = ring_table(p.w);
    if (!rt.fast[1]) {   // under 2 bits per weight: the windows of the first groups wrap; not a rate EXL3 ships here
        for (int t = 0; t < nt; ++t) exl3_cols_ref(p, x2[t], y1[t], c0, c1);
        return;
    }
    for (; nt > kColsTok; nt -= kColsTok, x2 += kColsTok, y1 += kColsTok) cols_n<0>(p, rt, x2, kColsTok, y1, c0, c1);
    if (nt == 1) cols_n<1>(p, rt, x2, 1, y1, c0, c1);
    else cols_n<0>(p, rt, x2, nt, y1, c0, c1);
}

}  // namespace basalt::kernels::cpu::exl3_detail
