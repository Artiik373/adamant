// src/kernels/cpu/exl3_ring.hpp - where the trellis windows of an EXL3 tile sit; shared by the builds of the column
// kernel.
#pragma once

#include "basalt/kernels/cpu/exl3.hpp"

#include <cstdint>
#include <vector>

namespace basalt::kernels::cpu::exl3_detail {

constexpr uint32_t kMul = 0x83DCD12Du;
constexpr float kScale = 0.00676727f;   // fp16 0x1EEE: 1 / the byte sum's standard deviation

/// Where each ring position's 16-bit window starts: 32-bit word index (and its successor, wrapped) and bit offset.
struct RingTable {
    int w = 0;
    uint32_t w0[256], w1[256], sh[256];       // per ring position: the window's words and bit offset (scalar path)
    // per group of 8 ring positions: a 16-byte chunk at word u0 holds all 8 windows (fast[t]); `shuf` turns each lane's
    // 4 stream bytes into a big-endian dword and `bit` is the window's bit offset in it
    bool fast[32];
    uint32_t u0[32];
    alignas(32) uint8_t shuf[32][32];
    alignas(32) uint32_t bit[32][8];
};

struct RingTables {
    RingTable t[kExl3MaxWords / 8 + 1];
    RingTables() {
        for (int q = 1; q <= kExl3MaxWords / 8; ++q) {
            RingTable& r = t[q];
            r.w = q * 8;
            const int ka = r.w / 16, half = r.w % 16;           // 0 or 8: alternating KA / KA+1 bits
            const int nbits = r.w * 16, nwords = r.w / 2;
            int s = 0;
            for (int i = 0; i < 256; ++i) {
                s += ka + (half && (i & 1) ? 1 : 0);             // mask 0xAAAA: odd steps take the extra bit
                const int s0 = ((s - 16) % nbits + nbits) % nbits;
                r.w0[i] = (uint32_t) (s0 >> 5);
                r.w1[i] = (uint32_t) (((s0 >> 5) + 1) % nwords);
                r.sh[i] = (uint32_t) (s0 & 31);
            }
            std::vector<int> end(256);
            s = 0;
            for (int i = 0; i < 256; ++i) { s += ka + (half && (i & 1) ? 1 : 0); end[(size_t) i] = s; }
            for (int t = 0; t < 32; ++t) {
                const int first = end[(size_t) t * 8] - 16;
                r.fast[t] = first >= 0;
                if (!r.fast[t]) continue;
                int u0 = first >> 5;
                if (u0 > nwords - 4) u0 = nwords - 4;
                r.u0[t] = (uint32_t) u0;
                for (int j = 0; j < 8; ++j) {
                    const int pos = end[(size_t) t * 8 + (size_t) j] - 16 - 32 * u0;   // bit offset in the chunk
                    const int ob = pos >> 3;
                    r.bit[t][j] = (uint32_t) (pos & 7);
                    for (int p = 0; p < 4; ++p) {
                        const int sb = ob + 3 - p;                                       // stream byte -> memory byte
                        r.shuf[t][(j >> 2) * 16 + (j & 3) * 4 + p] =
                            sb > 15 ? (uint8_t) 0x80 : (uint8_t) ((sb & ~3) + 3 - (sb & 3));
                    }
                }
            }
        }
    }
};

inline const RingTable& ring_table(int w) {
    static const RingTables tabs;
    return tabs.t[w / 8];
}

inline uint32_t be_bits(const uint32_t* words, const RingTable& r, int i) {
    const uint32_t hi = words[r.w0[i]], lo = words[r.w1[i]], b = r.sh[i];
    const uint64_t v = ((uint64_t) hi << 32) | lo;
    return (uint32_t) ((v << b) >> 48);
}

/// The 32-bit words of a tile: the trellis is little-endian int16 and each pair is read as one MSB-first word.
inline const uint32_t* tile_words(const Exl3Proj& p, int kt, int nt) {
    return (const uint32_t*) (p.trellis + ((size_t) kt * (size_t) (p.n / 16) + (size_t) nt) * (size_t) p.w * 2);
}

}  // namespace basalt::kernels::cpu::exl3_detail
