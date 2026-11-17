// tests/kernels/cpu/exl3_lane_test.cpp - the warp-level tile mapping the EXL3 GPU kernels use (exl3_lane.hpp), run
// lane by lane on the host: the windows each lane reads, the tile rows and columns, the shuffle Hadamard and the
// partial sums, against a bit-by-bit ring oracle.
#include "basalt/kernels/exl3_lane.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace basalt::kernels;

namespace {

int g_fail = 0;

void check(const char* what, int w, bool ok) {
    std::printf("  %-28s w %3d %s\n", what, w, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

unsigned ring_bit(const std::vector<uint32_t>& t, int w, int pos) {
    const int n = w * 16;
    pos = ((pos % n) + n) % n;
    return (t[(size_t) (pos >> 5)] >> (31 - (pos & 31))) & 1u;
}

uint32_t ring_window(const std::vector<uint32_t>& t, int w, int i) {
    uint32_t win = 0;
    for (int b = 0; b < 16; ++b) win = (win << 1) | ring_bit(t, w, exl3_ring_end(w, i) - 16 + b);
    return win;
}

// had128 of the kernel, lane l holding values 4l..4l+3 (xor shuffles written as array reads)
void had_warp(float v[32][4]) {
    for (int l = 0; l < 32; ++l) {
        float* p = v[l];
        const float a = p[0] + p[1], b = p[0] - p[1], c = p[2] + p[3], d = p[2] - p[3];
        p[0] = a + c; p[1] = b + d; p[2] = a - c; p[3] = b - d;
    }
    for (int s = 0; s < 5; ++s) {
        const int bit = 1 << s;
        for (int m = 0; m < 4; ++m) {
            float o[32];
            for (int l = 0; l < 32; ++l) o[l] = v[l ^ bit][m];
            for (int l = 0; l < 32; ++l) v[l][m] = (l & bit) ? o[l] - v[l][m] : v[l][m] + o[l];
        }
    }
    for (int l = 0; l < 32; ++l)
        for (int m = 0; m < 4; ++m) v[l][m] *= 0.08838834764831845f;
}

}  // namespace

int main() {
    for (const int w : {32, 40, 48, 56, 64, 96, 112, 128}) {
        std::mt19937 rng((unsigned) w);
        const int nw = w / 2;
        std::vector<uint32_t> tile((size_t) nw);
        for (auto& x : tile) x = rng();
        bool windows = true, cover = true;
        bool seen[256] = {};
        for (int lane = 0; lane < 32; ++lane) {
            const Exl3Lane g = exl3_lane(w, lane);
            uint32_t x[4];
            for (int i = 0; i < 4; ++i) x[i] = tile[(size_t) ((g.wb + i) % nw)];
            for (int j = 0; j < 8; ++j) {
                windows = windows && g.pos[j] + 16 <= 128 && exl3_window(x, g.pos[j]) == ring_window(tile, w, lane * 8 + j);
                const int el = exl3_row(lane, j) * 16 + exl3_col(lane, j);
                cover = cover && !seen[el];
                seen[el] = true;
            }
        }
        for (const bool s : seen) cover = cover && s;
        check("lane windows", w, windows);
        check("tile coverage", w, cover);
    }
    {   // the shuffle Hadamard against the dense transform
        std::mt19937 rng(1);
        std::normal_distribution<float> nd;
        float x[128], v[32][4];
        for (auto& a : x) a = nd(rng);
        for (int l = 0; l < 32; ++l)
            for (int m = 0; m < 4; ++m) v[l][m] = x[l * 4 + m];
        had_warp(v);
        double e = 0;
        for (int i = 0; i < 128; ++i) {
            double a = 0;
            for (int j = 0; j < 128; ++j) {
                int p = i & j, q = 0;
                for (; p; p &= p - 1) q ^= 1;
                a += (q ? -1.0 : 1.0) * (double) x[j];
            }
            e = std::fmax(e, std::fabs(a / std::sqrt(128.0) - v[i / 4][i % 4]));
        }
        check("warp hadamard", 128, e < 1e-5);
    }
    for (const int w : {48, 56}) {   // a column of tiles: per-lane partial sums, reduced over the 4 lanes of a column
        const int K = 640, ktiles = K / 16, nw = w / 2;
        std::mt19937 rng((unsigned) w + 5u);
        std::normal_distribution<float> nd;
        std::vector<std::vector<uint32_t>> tiles((size_t) ktiles, std::vector<uint32_t>((size_t) nw));
        for (auto& t : tiles)
            for (auto& x : t) x = rng();
        std::vector<float> x1((size_t) K);
        for (auto& a : x1) a = nd(rng);
        double ref[16] = {};
        for (int kt = 0; kt < ktiles; ++kt)
            for (int i = 0; i < 256; ++i) {
                const int tt = i >> 3, j = i & 7;
                const int r = (tt & 3) * 2 + (j & 1) + ((j & 2) ? 8 : 0), c = (tt >> 2) + ((j & 4) ? 8 : 0);
                ref[c] += (double) x1[(size_t) (kt * 16 + r)] * exl3_code_int(ring_window(tiles[(size_t) kt], w, i));
            }
        float acc[32][2] = {};
        for (int kt = 0; kt < ktiles; ++kt)
            for (int lane = 0; lane < 32; ++lane) {
                const Exl3Lane g = exl3_lane(w, lane);
                uint32_t x[4];
                for (int i = 0; i < 4; ++i) x[i] = tiles[(size_t) kt][(size_t) ((g.wb + i) % nw)];
                for (int j = 0; j < 8; ++j)
                    acc[lane][j >> 2] = std::fma(x1[(size_t) (kt * 16 + exl3_row(lane, j))],
                                                 (float) exl3_code_int(exl3_window(x, g.pos[j])), acc[lane][j >> 2]);
            }
        double e = 0, sc = 0;
        for (int c = 0; c < 2; ++c)
            for (int lane = 0; lane < 32; lane += 4) {
                const double s = (double) acc[lane][c] + acc[lane + 1][c] + acc[lane + 2][c] + acc[lane + 3][c];
                const double r = ref[(lane >> 2) + 8 * c];
                e = std::fmax(e, std::fabs(s - r));
                sc = std::fmax(sc, std::fabs(r));
            }
        check("column partial sums", w, e < 1e-4 * sc);
    }
    std::printf(g_fail ? "FAILED\n" : "ok\n");
    return g_fail ? 1 : 0;
}
