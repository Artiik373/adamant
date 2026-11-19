// The resident expert copy gives RAM back under memory pressure and takes it back: host-VMM pieces go from the top down,
// their experts read from the file meanwhile, and come back bottom up with the file's bytes.  A synthetic expert file
// of three pieces; needs a CUDA device with host VMM (exits 77 without one).
#include "basalt/core/expert_cache.hpp"
#include "basalt/core/expert_source.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace basalt::core;

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

struct TempDirectory {
    fs::path path;
    TempDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("basalt-give-back-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }
    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

int run() {
    constexpr int64_t layers = 1, experts = 512, held = 8;
    TempDirectory dir;
    basalt::kernels::cpu::NativeFmt fmt;
    std::string err;
    require(basalt::kernels::cpu::native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, 2560, 640, fmt, err), err);
    const uint64_t blob = (uint64_t) fmt.bytes;
    std::ostringstream text;
    text << 0 << ' ' << GGML_TYPE_IQ3_XXS << ' ' << GGML_TYPE_IQ4_NL << ' ' << 0 << ' ' << blob << '\n';
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary);
        std::vector<char> b((size_t) blob);
        for (int64_t i = 0; i < layers * experts; ++i) {
            for (size_t k = 0; k < b.size(); ++k) b[k] = (char) (i * 31 + (int64_t) k * 7 + 1);
            out.write(b.data(), (std::streamsize) b.size());
        }
        require((bool) out, "could not write the synthetic experts");
    }
    require(basalt::kernels::cpu::expert_layout_load_text(text.str(), layers, experts, err, 2560, 640), err);

    FileExpertSource src;
    require(src.open(FileRange((dir.path / "experts.bin").string()), layers, experts, err), err);
    ExpertCache cache;
    require(cache.open_sized(std::vector<int64_t>((size_t) (layers * held), (int64_t) blob), layers, experts, err), err);
    for (int64_t e = 0; e < held; ++e) {
        const int32_t slot = cache.admit(0, e);
        require(slot >= 0 && cache.fill_slot_blocking(slot, src.blob(0, e), err, (int64_t) blob), "cache fill " + err);
    }
    require(src.pin_cache_complement(cache, err, true, {}, -1, 0), err);
    if (!src.complement_allocation_pinned()) {
        std::puts("no host VMM: skipped");
        return 77;
    }
    const uint64_t piece = vmm::host_piece_bytes();
    auto all_match = [&](const char* when) {
        int64_t resident = 0;
        for (int64_t e = held; e < experts; ++e) {
            require(src.equals_file(0, e, src.blob(0, e)), std::string(when) + ": expert " + std::to_string(e));
            resident += src.has_resident(0, e) ? 1 : 0;
        }
        return resident;
    };
    const int64_t all = all_match("start");
    require(all == experts - held && src.ram_can_give_back() && src.ram_given_back() == 0, "start state");

    // one piece: the experts on it read from the file, the rest stay in RAM
    const uint64_t top = src.ram_give_back(1);   // the top piece: the copy's remainder above the whole pieces
    require(top > 0 && top <= piece && src.ram_given_back() == top, "one piece given back");
    const int64_t after_one = all_match("one piece out");
    require(after_one > 0 && after_one < all, "the piece's experts left RAM, the others stayed");
    require(src.has_resident(0, held), "the lowest expert stays in RAM while the top piece is out");

    // everything: nothing more to give, every expert from the file
    require(src.ram_give_back(~0ull) > 0 && !src.ram_can_give_back() && all_match("all out") == 0, "all given back");
    require(src.ram_give_back(1) == 0, "a second give-back of nothing freed bytes");

    // take back, bottom up; a budget below a piece takes nothing
    require(src.ram_take_back(piece - 1) == 0, "a take-back over budget");
    uint64_t taken = 0, step = 0;
    while ((step = src.ram_take_back(~0ull)) > 0) {
        taken += step;
        (void) all_match("partly back");
    }
    require(src.ram_given_back() == 0 && taken > 0 && all_match("all back") == all && src.ram_can_give_back(),
            "every expert back in RAM with its file bytes");
    return 0;
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    try {
        const int rc = run();
        if (rc == 0) std::puts("file_expert_give_back_test: PASS");
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "file_expert_give_back_test: %s\n", e.what());
        return 1;
    }
}
