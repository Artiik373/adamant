// include/basalt/kernels/exl3_lane.hpp - how one warp reads an EXL3 16x16 tile: which weights a lane decodes, where
// their trellis windows sit, and which activations they multiply.  Plain integer code shared by the CUDA kernels and
// the host test that checks the mapping.
//
// A tile is a ring of 16*w bits (w 16-bit words).  Lane `l` of a warp decodes ring positions 8l .. 8l+7:
//   weight j of the lane (0..7) sits in tile row  2*(l&3) + (j&1) + 8*((j>>1)&1)  and column  (l>>2) + 8*(j>>2),
// and its 16-bit window is the ring bits that end at S(8l+j), S(i) = (bits of steps 0..i).
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define EXL3_HD __host__ __device__
#else
#define EXL3_HD
#endif

namespace basalt::kernels {

/// Ring position at which weight i's window ends: w words per tile, an integer rate takes w/16 bits a step and a
/// half-integer rate alternates w/16 and w/16 + 1 (the odd steps take the extra bit).
EXL3_HD inline int exl3_ring_end(int w, int i) {
    const int ka = w / 16;
    return ka * (i + 1) + ((w % 16) ? ((i + 1) >> 1) : 0);
}

/// One lane's trellis geometry.  Its eight windows lie within 4 consecutive 32-bit words of the (wrapped) ring,
/// starting at word `wb`; window j starts `pos[j]` bits into those words.
struct Exl3Lane {
    int wb;
    int pos[8];
};

EXL3_HD inline Exl3Lane exl3_lane(int w, int lane) {
    Exl3Lane r;
    const int nbits = w * 16;
    const int first = exl3_ring_end(w, lane * 8) - 16;
    const int start = ((first % nbits) + nbits) % nbits;
    r.wb = start >> 5;
    const int bo = start & 31;
    for (int j = 0; j < 8; ++j) r.pos[j] = bo + (exl3_ring_end(w, lane * 8 + j) - 16 - first);
    return r;
}

/// The window starting `pos` bits into x[0..3] (big-endian bit order: bit 31 of x[0] first).
EXL3_HD inline uint32_t exl3_window(const uint32_t* x, int pos) {
    const int i = pos >> 5, s = pos & 31;
    const uint32_t hi = i == 0 ? x[0] : i == 1 ? x[1] : i == 2 ? x[2] : x[3];
    const uint32_t lo = i == 0 ? x[1] : i == 1 ? x[2] : i == 2 ? x[3] : 0u;
    return (s == 0 ? hi : ((hi << s) | (lo >> (32 - s)))) >> 16;
}

/// Tile row of weight j of lane l, and its column.
EXL3_HD inline int exl3_row(int lane, int j) { return 2 * (lane & 3) + (j & 1) + ((j & 2) ? 8 : 0); }
EXL3_HD inline int exl3_col(int lane, int j) { return (lane >> 2) + ((j & 4) ? 8 : 0); }

/// The computed code of a window as an integer: the byte sum of window * 0x83DCD12D minus 510; the value is that times
/// exl3_code_scale().
EXL3_HD inline int exl3_code_int(uint32_t win) {
    const uint32_t x = win * 0x83DCD12Du;
    return (int) (x & 255u) + (int) ((x >> 8) & 255u) + (int) ((x >> 16) & 255u) + (int) (x >> 24) - 510;
}
EXL3_HD inline float exl3_code_scale() { return 0.0067672729f; }

}  // namespace basalt::kernels
