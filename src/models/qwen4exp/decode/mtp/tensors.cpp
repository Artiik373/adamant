#include "common.hpp"

namespace basalt::core {

using namespace mtp_detail;

const float* MtpDrafter::f32(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "f32") return (const float*) (dense_ + t.off);
    return nullptr;
}
const uint16_t* MtpDrafter::bf16(const char* name) const {
    for (const auto& t : tensors_)
        if (t.name == name && t.kind == "bf16") return t.dev ? (const uint16_t*) t.dev : (const uint16_t*) (dense_ + t.off);
    return nullptr;
}
const void* MtpDrafter::q8(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.type >= 0) return dense_ + t.off;
    return nullptr;
}
int MtpDrafter::wtype(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name) return t.type;
    return -1;
}
uint64_t MtpDrafter::wbytes(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.type >= 0) return t.bytes;
    return 0;
}

bool MtpDrafter::parse_index(const FileRange& index, std::vector<Tensor>& out, ExpertFormat& ex, std::string& err) {
    std::vector<uint8_t> idx_text;
    if (!read_file(index, idx_text)) { err = "mtp: cannot open " + index.path; return false; }
    std::istringstream idx(std::string(idx_text.begin(), idx_text.end()));
    std::string line;
    while (std::getline(idx, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {   // `#experts exl3|ms GU_TYPE D_TYPE BLOB_BYTES`
            std::istringstream hs(line);
            std::string tag, fmt;
            hs >> tag >> fmt;
            if (tag == "#experts") {
                const bool read = (fmt == "exl3" || fmt == "ms") && (hs >> ex.gu_type >> ex.d_type >> ex.blob) && ex.blob != 0;
                const bool typed = fmt == "exl3" ? basalt::kernels::cpu::exl3_type(ex.gu_type) && basalt::kernels::cpu::exl3_type(ex.d_type)
                                                 : ms_tensor_type((uint32_t) ex.gu_type) && ex.d_type == ex.gu_type;
                if (!read || !typed) {
                    err = "mtp: malformed experts line: " + line;
                    return false;
                }
                ex.native = true;
            }
            continue;
        }
        std::istringstream is(line);
        Tensor t;
        is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;
        if (!is) { err = "mtp: malformed dense index line: " + line; return false; }
        if (t.kind == "q8_0") t.type = 8;   // GGML_TYPE_Q8_0
        else if (t.kind == "exl3") {   // bytes = (k/16)(n/16) W 2 + 2 (k + n): the rate is implied
            const int64_t k = t.cols, n = t.rows;
            const int64_t body = (int64_t) t.bytes - 2 * (k + n), per = k * n / 128;   // bytes per word of W
            if (k % 128 || n % 128 || body <= 0 || body % per || (body / per) % 8) {
                err = "mtp: bad EXL3 tensor in the dense index: " + line;
                return false;
            }
            t.type = basalt::kernels::cpu::kExl3TypeBase + (int) (body / per) / 8;
            if (!basalt::kernels::cpu::exl3_type(t.type)) { err = "mtp: bad EXL3 rate: " + line; return false; }
        } else if (t.kind == "ms") {   // rows of ms_row_bytes(cols, K), then int8 signs[cols]: K is implied
            for (int k = 2; k <= 6 && t.type < 0; ++k)
                if ((uint64_t) t.rows * ms_row_bytes((uint64_t) t.cols, k) + (uint64_t) t.cols == t.bytes) t.type = 120 + k;
            if (t.type < 0) { err = "mtp: bad MS tensor in the dense index: " + line; return false; }
        }
        out.push_back(t);
    }
    return true;
}

bool MtpDrafter::is_hc8_matrix(const Tensor& t) {
    return t.kind == "bf16" && t.name.find("hyper_connection") != std::string::npos &&
           t.name.find("input_mix_weight_") != std::string::npos && (uint64_t) t.rows * (uint64_t) t.cols * 2 == t.bytes;
}

bool MtpDrafter::plan_bytes(const MtpFiles& files, const ModelGeometry& g, uint64_t& out, std::string& err) {
    uint64_t dense = files.dense.size();
    if (basalt::kernels::hc8_only()) {   // the dense blob without the matrices that are held as 8-bit copies only
        std::vector<Tensor> tensors;
        ExpertFormat ex;
        if (!parse_index(files.index, tensors, ex, err)) return false;
        uint64_t kept = 0;
        for (const Tensor& t : tensors) {
            if (is_hc8_matrix(t)) continue;
            kept = ((kept + 255) & ~(uint64_t) 255) + t.bytes;
        }
        dense = kept;
    }
    ExpertFormat ex;
    {
        std::vector<Tensor> tensors;
        if (!parse_index(files.index, tensors, ex, err)) return false;
    }
    out = dense + (uint64_t) g.n_expert * (ex.native ? ex.blob : (uint64_t) basalt::kernels::cpu::BLOB);
    return true;
}

}
