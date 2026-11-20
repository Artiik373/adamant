# MS (Mad Shift) Trellis Quantization for Qwen3.8-Flash-Next in Adamant

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

Unlike `ms-llama.cpp`'s 512-weight block layout (which cannot natively code `moe_intermediate_size = 640`), Adamant stores one continuous 14-bit trellis stream per row:

```text
fp16   d        (2 bytes, IEEE 754 binary16 row scale)
uint16 s0       (2 bytes, 14-bit initial trellis state)
uint8  qs[...]  ceil((in_features - 1) * K / 8) bytes, MSB-first K-bit symbols
```

- `hidden_size = 2560`: Hadamard `block_size = 512` (`0.0125 bpw` overhead).
- `moe_intermediate_size = 640`: Hadamard `block_size = 128` (`0.05 bpw` overhead).

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

# 4. Inspect the default <= 55 GB layer bit plan
python3 tools/convert/ms_moe_converter.py --hf-dir /path/to/Qwen3.8-Flash-Next --calib-pt calib.pt --out-dir out_ms --dry-run
```
