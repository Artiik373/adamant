// src/models/hy4flash/engram.hpp - the Engram row lookup of Hy4-flash: the n-gram hash and the rows (F8_E4M3 or Q4_0) as floats.
#pragma once

#include "models/hy4flash/pack.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace basalt::models::hy4flash {

/// The n_cols table rows of module `e` (the e-th Engram layer) for the token at `pos` whose predecessors are prev1 (pos - 1)
/// and prev2 (pos - 2). A history that starts before the sequence is padded; `valid` is 0 for a token with no n-gram (an
/// image position, or an id outside the map), whose contribution is masked.
inline void engram_ids(const EngramInfo& p, int e, int32_t token, int32_t prev1, int32_t prev2, int64_t pos, bool embedding,
                       uint32_t* ids, float& valid) {
    const int n_gram = p.n_gram, n_head = p.n_head, n_cols = p.n_cols();
    const uint64_t* mult = p.mult.data() + (size_t) e * n_gram;
    const uint64_t* primes = p.primes.data() + (size_t) e * n_cols;
    const uint64_t* offs = p.offsets.data() + (size_t) e * n_cols;
    const int64_t pad = p.token_map[(size_t) p.pad_token_id];
    const auto cid = [&](int32_t t) -> int64_t {
        return t < 0 || (size_t) t >= p.token_map.size() ? -1 : p.token_map[(size_t) t];
    };
    const int32_t prev[2] = {prev1, prev2};
    int64_t c = embedding ? -1 : cid(token);
    const bool dead = c < 0;
    uint64_t toks[4];
    toks[0] = (uint64_t) (dead ? pad : c);
    for (int s = 1; s < n_gram; ++s) {
        c = pos - s < 0 ? -1 : cid(prev[s - 1]);
        toks[s] = (uint64_t) (c < 0 ? pad : c);
    }
    uint64_t rolling = toks[0] * mult[0];
    for (int o = 1; o < n_gram; ++o) {
        rolling ^= toks[o] * mult[o];
        for (int h = 0; h < n_head; ++h) {
            const int col = (o - 1) * n_head + h;
            ids[col] = (uint32_t) (rolling % primes[col] + offs[col]);
        }
    }
    valid = dead ? 0.f : 1.f;
}

/// One E4M3 code as a float (bias 7, no infinities; 0x7f and 0xff are NaN).
inline float e4m3_to_f32(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN;
    const float v = e ? std::ldexp(1.f + (float) m / 8.f, e - 7) : std::ldexp((float) m, -9);
    return (b & 0x80) ? -v : v;
}

/// One fp16 as a float.
inline float f16_to_f32(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp((float) m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(1.f + (float) m / 1024.f, e - 15);
    return (h & 0x8000) ? -v : v;
}

/// The p.row_values values of one table row `src` (p.row_bytes bytes) into `row`: an E4M3 code each (`lut`, 256 floats), or
/// ggml's Q4_0 blocks (fp16 d, 16 bytes of nibbles: value i is the low nibble of byte i, value i + 16 the high one, (q - 8) d).
inline void engram_row(const EngramInfo& p, const float* lut, const uint8_t* src, float* row) {
    if (p.type != 2) {
        for (int d = 0; d < p.row_values; ++d) row[d] = lut[src[d]];
        return;
    }
    for (int b = 0; b < p.row_values / 32; ++b) {
        const uint8_t* blk = src + b * 18;
        uint16_t h;
        std::memcpy(&h, blk, 2);
        const float d = f16_to_f32(h);
        for (int i = 0; i < 16; ++i) {
            row[b * 32 + i] = (float) ((blk[2 + i] & 15) - 8) * d;
            row[b * 32 + i + 16] = (float) ((blk[2 + i] >> 4) - 8) * d;
        }
    }
}

}  // namespace basalt::models::hy4flash
