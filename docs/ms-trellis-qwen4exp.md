# MS (Mad Shift) Trellis Quantization for Qwen3.8-Flash-Next in Amperite

This branch (`feat/ms-trellis-qwen4exp`) adds a per-row 14-bit `1MAD` bitshift-trellis quantization pipeline and AVX2 + Ampere (`sm_86`) inference kernels tailored to **Qwen3.8-Flash-Next** (`qwen4_exp`) on single-GPU + DDR5 workstations (e.g., RTX 3090 24 GB + i5-13600KF + 64 GB DDR5).

## Target ≤ 55 GB Main Pack Allocation

| Component | Params | Format | Effective bpw | Est. Size |
| :--- | :--- | :--- | :--- | :--- |
| Routed MoE Experts (11 outer / `GDN-2` layers) | ~25B | `MS_C4` | ~4.025 bpw | ~12.6 GB |
| Routed MoE Experts (37 remaining layers) | ~90B | `MS_C3` | ~3.025 bpw | ~34.0 GB |
| Dense Spine (QSA, GDN, Shared Experts) | ~7.5B | `MS_C6` | ~6.025 bpw | ~5.65 GB |
| Output Head (`lm_head`, unsplit) | ~1.0B | `MS_C6` | ~6.012 bpw | ~0.75 GB |
| Token Embeddings + Routers/Norms/SSM | ~1.5B | `Q4_K` / `F32` / `Q8_0` | — | ~1.00 GB |
| **Main Pack Total** | **125B** | **All-MS Trellis** | **~3.46 bpw** | **~54.0 GB** |
| PLE / N-gram Table (separate host `mmap` shard) | 51B | `Q4_K` / `IQ4_XS` | 3.5–4.5 bpw | ~22–28 GB |

## Per-Row `.basalt` Binary Layout

Unlike `ms-llama.cpp`'s 512-weight block layout (which cannot natively code `moe_intermediate_size = 640`), Amperite stores one continuous 14-bit trellis stream per row:

```text
fp16   d        (2 bytes, IEEE 754 binary16 row scale)
uint16 s0       (2 bytes, 14-bit initial trellis state)
uint8  qs[...]  ceil((in_features - 1) * K / 8) bytes, MSB-first K-bit symbols
                zero padding to a 4-byte row stride (ms_row_bytes)
```

A GGUF tensor `[n_in, n_out]` of type `120 + K` (`MS_C2` = 122 to `MS_C6` = 126) stores `n_out` such rows,
then `int8 signs[n_in]` (+1 / -1). The engine computes `y = W_rot * H(signs * x)`, with `H` the orthonormal block
Hadamard of size `largest_pow2_divisor(n_in)`, at most 512. `n_in` must be a multiple of 32.

- `hidden_size = 2560`: Hadamard `block_size = 512` (`0.0125 bpw` overhead).
- `moe_intermediate_size = 640`: Hadamard `block_size = 128` (`0.05 bpw` overhead).

## Engine Support

- **Decode, dense spine and head:** `native_mmvq` runs MS tensors (`src/kernels/cuda/ms_mmvq.cu`) on the shared Q8_1
  input, 1 to 8 columns in one kernel: each row is decoded once for all columns of a chunk, and a column's sum has one
  order whatever rides along (the verifier's requirement). The block stages its columns rotated in shared memory.
- **Routed experts:** an expert blob is three MS payloads `[gate | up | down]`; `experts.layout` v8 with type 120 + K.
  GPU: `native_expert_grouped` (`src/kernels/cuda/ms_experts.cu`), the input rotated once per entry. CPU tier:
  `src/kernels/cpu/ms_expert.cpp`, fp32 activations rotated and quantized to int8 per 128 columns, AVX-VNNI byte dots.
  With an MS head the drafter ranks its token subset on a Q8_0 copy of the dequantized head rows.
- **Prompt:** dense MS rows are dequantized to fp16 with the rotation undone, then an fp16 GEMM. Routed experts are
  converted to Q8_0 per gather group (`ms_experts_to_q8_0`) and run on the int8 MMQ kernels.
- **Parity tests:** `ms_cpu_test` (CPU rows against plain C and W x in double, the whole expert), `ms_gpu_parity`
  (dense columns bitwise alone vs 8, grouped experts, dequant layout, Q8_0 within half a step); both print bench lines.

## Converter Search

Rows are coded after a random block Hadamard rotation with `d = gain_k * std(row) / 147.8` (`gain` 1.0 / 1.15 / 1.2 / 1.2 / 1.3 / 1.2 for
`k` = 1 / 2 / 3 / 4 / 5 / 6). The Viterbi search runs over windows of 32 columns, commits the first 8, feeds their LDLQ error
forward (damping 1% of the mean Hessian diagonal) and slides on by 8. Branch costs are weighted by `1 / Hu[c, c]^2`.
The Qwen3.8-Flash-Next converter (`ms_flashnext_convert.py`) uses windows of 48 committing 12 (`--window`, `--commit`),
and afterwards refits each row's fp16 `d` for its fixed codes in closed form, `d = (c H w^T) / (c H c^T)`, kept only where it
lowers the H-weighted error. On the 8 busiest experts of layers 0 and 24 the proxy error tr(E H E^T) / tr(W H W^T) is 2.5 to 4%
below 32 / 8 at the same search time, and the refit takes 0.1 to 0.7% more.

Qwen3-1.7B, decoder linears only, against BF16 (`smoke_kld_qwen3.py`, 64 calibration and 32 eval sequences of 512 tokens
spread over all four calibration pools, RTX 3090 at 350 W):

| Coding | bpw | KLD mean | top-1 |
| :--- | :--- | :--- | :--- |
| MS_C3 | 3.025 | 0.0336 | 89.37% |
| MS_C3, GuidedQuant Hessians (`--calib-mode guided`) | 3.025 | 0.0295 | 90.09% |
| MS_C3, two-sided H = E[g g^T] (x) E[x x^T], data labels (`--calib-mode kron-emp`) | 3.025 | 0.0339 | 89.88% |
| MS_C3, the same with labels sampled from the model (`--calib-mode kron-true`) | 3.025 | 0.0327 | 89.42% |
| MS_C4 | 4.025 | 0.0091 | 93.90% |
| Q4_K_M (unsloth / bartowski GGUF, `--gguf`) | 4.80 | 0.0201 / 0.0185 | 91.78% / 91.82% |

Windows of 24 committing 16 measured MS_C3 KLD 0.0423 against 0.0378 for 32 / 8 on one math-only eval split.

## Building a Pack

`ms_convert_dense_spine.py` codes one tensor into a record (`<name>.json` + payload) with the GDN value heads already in
the GGUF order; `ms_flashnext_convert.py` codes the whole model layer by layer (spine, head and routed experts, at the
widths of a `--plan`). `ms_pack.py` copies a base `.basalt` and replaces the recorded `gguf` tensors; with an `experts/`
folder of per-layer records it rebuilds `experts` and `experts.layout` too:

```sh
python3 tools/convert/ms_pack.py --base models/Qwen3.8-FN-UD-IQ3_XXS.basalt --ms-dir out_ms --out models/Qwen3.8-FN-MS.basalt --verify
```

## Verification Commands

```sh
# 1. Verify AVX2 1MAD + FWHT CPU kernels
g++ -O3 -march=native -std=c++20 -Isrc -Iinclude tests/test_ms_1mad.cpp -o build/test_ms_1mad
./build/test_ms_1mad

# 2. Verify Ampere sm_86 CUDA 1MAD kernels against CPU reference
nvcc -O3 -arch=sm_86 -std=c++20 -Isrc -Iinclude tests/test_ms_1mad_cuda.cu -o build/test_ms_1mad_cuda
./build/test_ms_1mad_cuda

# 3. Verify Python converter (Viterbi + LDLQ + RMT denoising + bit-exact packing)
python3 tests/test_ms_converter.py

# 4. The engine paths (dense, routed experts on CPU and GPU, prompt conversion) and the pack writer
build/Release/ms_cpu_test          # needs -DBUILD_TESTING=ON
build/Release/ms_gpu_parity
python3 -m unittest discover -s tools/convert -p "test_ms_pack.py"

# 5. Inspect the default <= 55 GB layer bit plan
python3 tools/convert/ms_moe_converter.py --hf-dir /path/to/Qwen3.8-Flash-Next --calib-pt calib.pt --out-dir out_ms --dry-run
```
