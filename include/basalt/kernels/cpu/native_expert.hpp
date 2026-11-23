// include/basalt/kernels/cpu/native_expert.hpp - one routed expert in its GGUF form on the CPU.
//
// The IQ2_XS / IQ3_XXS model files keep their experts in i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S gate/up; Q2_0 or IQ4_NL down) whose values cannot be re-expressed in the Q2_0 pack form.  A native expert
// blob is the three GGUF slices back to back, [gate rows | up rows | down rows], and the arithmetic is ggml's
// own (`ggml_get_type_traits_cpu`, extracted in src/kernels/cpu/ggml): the activation is quantized with the weight type's `vec_dot_type` and each row
// is one `vec_dot`, exactly what llama.cpp's CPU backend computes for the same tensor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace basalt::kernels::cpu {

/// Bytes of the largest activation any native layer uses (5120 values as Q8_K: 20 x 292; 2560 fp32 values: MS).
inline constexpr size_t kNativeActBytes = 10240;
/// The widest expert intermediate a native layer may have (the float buffers between gate/up and down).
inline constexpr int kNativeMaxFF = 2304;
/// Bytes of the largest quantized down activation (2304 values as Q8_K: 9 x 292, or Q8_0 72 x 34).
inline constexpr size_t kNativeHBytes = 3072;

/// One layer's native expert geometry.  Gate and up usually share a type; a layer whose up rows have their own type
/// (Unsloth's UD-IQ1_S mixes IQ2_S gate with IQ2_XXS up on one layer) keeps it in `up_type` / `up_row`.
struct NativeFmt {
    int gu_type = -1, d_type = -1;      ///< ggml types of gate (and of up unless up_type says otherwise) and of down
    int up_type = -1;                   ///< the up rows' type (gu_type when they agree)
    int gu_act = -1, d_act = -1;        ///< their vec_dot_type (the activation formats; gate and up share one)
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       ///< bytes per gate row and per down row
    size_t up_row = 0;                  ///< bytes per up row (gu_row when the types agree)
    size_t up_off = 0, down_off = 0;    ///< inside the blob
    size_t bytes = 0;                   ///< the whole blob
    size_t tail = 0;                    ///< NVFP4: the blob's last 16 bytes, FP32 {s_gate, s_up, s_down, 0}
    size_t act_bytes = 0, h_bytes = 0;  ///< quantized activation sizes (n_embd of gu_act, n_ff of d_act)
    float swiglu_limit = 0.f;           ///< > 0: gate capped at it and up clamped to +-it before the product
    int exl3_w = 0;                     ///< EXL3 layer: 16-bit words per 16x16 tile (0: a GGUF layer); the blob is three
                                        ///< EXL3 projection records [gate | up | down] (exl3.hpp), activations are fp16
    int ms_k = 0;                       ///< MS layer: K bits a weight (0: another format); the blob is three MS payloads
                                        ///< [gate | up | down] (ms_expert.hpp), activations are fp32
    float gate_softcap = 0.f, up_softcap = 0.f;   ///< > 0: the softcapped SwiGLU (swiglu.hpp); the caller sets them
};

/// Fills `f` for a layer (`swiglu_limit`: the model's SwiGLU limit, 0 for none; `up_type`: the up rows' type, -1 for
/// gu_type's); false (with a reason) when there is no dot product for a type or gate and up need different activations.
bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err,
                float swiglu_limit = 0.f, int up_type = -1);

/// x (n_embd floats) -> the gate/up activation (act_bytes).
void native_quant_act(const NativeFmt& f, const float* x, void* dst);
/// h (n_ff floats) -> the down activation (h_bytes).
void native_quant_h(const NativeFmt& f, const float* h, void* dst);

/// NVFP4 blobs end in FP32 {s_gate, s_up, s_down, 0}: the scale tail's byte count.
inline constexpr size_t kNvfp4Tail = 16;

/// ff[t][r] = swiglu(gate_r . a[t], up_r . a[t]) for rows r in [r0, r1), `nt` tokens, the layer's SwiGLU (swiglu.hpp;
/// NVFP4: gate and up times s_gate and s_up first).
void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1);
/// out[t][r] = down_r . hq[t] for rows r in [r0, r1) (NVFP4: times s_down).
void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1);

/// An EXL3 layer's gate and up in two steps, so that tasks can be cut finer than the 128-block of the output transform:
/// g1[t][c] and u1[t][c] for output columns [c0, c1) (multiples of 16), the weights decoded once for the `nt` tokens.
/// `stamp` (not 0) names the expert of a batch: a thread prepares its inputs once per stamp.
void exl3_gu_cols(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* g1,
                  float* const* u1, int c0, int c1, uint64_t stamp);
/// Then the output transforms and SwiGLU over 128-blocks [b0, b1): ff (holding g1) becomes the intermediate.
void exl3_gu_finish(const NativeFmt& f, const uint8_t* blob, int nt, float* const* ff, float* const* u1, int b0, int b1);

}  // namespace basalt::kernels::cpu
