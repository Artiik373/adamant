// src/models/gigachat4/meta_test.cpp - the GigaChat4 layer kinds and attention constants against transformers' values.
#include "models/gigachat4/meta.hpp"

#include <cmath>
#include <cstdio>

using namespace basalt::models::gigachat4;

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("%-40s %s\n", what, ok ? "ok" : "FAIL");
        failures += !ok;
    };
    const Geometry g;
    int full = 0;
    for (int l = 0; l < g.n_layers; ++l) full += g.is_full_attention(l);
    check(full == 8 && g.is_full_attention(3) && g.is_full_attention(31) && !g.is_full_attention(0), "full attention on 3, 7, ..., 31");
    check(!g.is_moe(0) && g.is_moe(1), "layer 0 dense, then MoE");
    // transformers' _compute_yarn_parameters (rope 64, theta 1e5, factor 32, beta 32/1, original 32768): pairs 0, 5, 10, 15, 20, 31
    const auto f = yarn_inv_freq(g);
    const int at[] = {0, 5, 10, 15, 20, 31};
    const double want[] = {1.0, 0.1654817, 0.027384195, 0.0040925867, 0.00031401819, 4.4781646e-07};
    bool ok = f.size() == 32;
    for (int i = 0; i < 6 && ok; ++i) ok = std::fabs(f[at[i]] - want[i]) <= 1e-6 * want[i] + 1e-12;
    check(ok, "yarn_inv_freq matches transformers");
    const double m = 0.1 * std::log(32.0) + 1.0;
    check(std::fabs(mla_scale(g) - m * m / std::sqrt(192.0)) < 1e-7, "mla_scale = mscale^2 / sqrt(192)");
    check(std::fabs(idx_weight_scale(g) - 1.0 / std::sqrt(64.0 * 128.0)) < 1e-9, "idx_weight_scale");
    std::printf(failures ? "FAILED\n" : "all ok\n");
    return failures ? 1 : 0;
}
