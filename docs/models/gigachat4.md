# GigaChat4-37B-A4B

Status: architecture, kernels, converter (`tools/convert/gigachat4_convert.py`), runtime (`gigachat4-run`), `ModelAdapter`
registration (`BASALT_BUILTIN_MODELS`), `--kv fp16|fp8`, recurrent MTP drafting (`--spec`), and `serve/models/gigachat4.py`
are implemented. No quality number exists yet.

## Architecture

From `config.json` and `modeling_gigachat4.py` of `ai-sage/GigaChat4-37B-A4B`; the code is `src/models/gigachat4/meta.hpp`.

- 32 layers, width 2560, vocabulary 128,256. Every layer: `x += post_attn_norm(attn(pre_attn_norm(x)))`, then the same for the
  feed-forward. All four norms, and the MLA's `q_a` and `kv_a` norms, are GatedNorm: RMS norm times a weight, then times
  `2 * sigmoid(down . silu(up . y))` with a 16-wide bottleneck.
- Attention: Gated DeltaNet on 24 layers (16 key heads, 32 value heads, dims 128, conv 4; the output norm is gated by a sigmoid and
  scaled by 2). Layers 3, 7, ..., 31 are full attention: MLA (32 heads, `q_rank` 1536, `kv_rank` 512, nope 128, rope 64, v 128)
  with a sigmoid output gate and a DSA indexer (64 heads of 128, top-2048 keys). `alpha_q = sqrt(2560/1536)` and
  `alpha_kv = sqrt(2560/512)` are folded into `q_b` and `kv_b` by the converter.
- Rope: YaRN (theta 1e5, factor 32, beta 32/1, original 32,768), attention factor 1. The MLA rotates consecutive pairs; the indexer
  rotates halves of its first 64 values. The MLA score scale is `mscale^2 / sqrt(192)`, `mscale = 0.1 ln 32 + 1`.
- Feed-forward: dense SwiGLU (8192) on layer 0; then 192 routed experts of 768 (sigmoid router with a bias, top-8, normalised,
  times 2.5) plus one shared expert. Every SwiGLU clamps gate to at most 10 and up to [-10, 10].
- Logits are softcapped at 30. The release is FP8 E4M3 in 128x128 blocks.
- Drafters: a recurrent MTP layer (EAGLE-3 style, checkpoint layer 32): `eh_proj` over `[enorm(embed(token)); hnorm(h)]`, one
  MLA layer with the indexer and a dense SwiGLU, then `shared_head`; each draft step feeds its output back into the same layer.
  The release also has a separate 5-layer BF16 DFlash2 drafter (`dflash_draft/`), not planned.

## Kernels

`include/basalt/models/gigachat4/kernels.hpp`: `gated_norm`, `mla_prepare` (YaRN, fp32 or fp16 cache), `idx_keys`,
`idx_rope_queries`, `gdn_gate_norm`, `softcap`. Reused: the Hy4-flash MLA attention (sinks of -inf), its per-head Q8_0 product
(now also over 128 values, for `k_b`), its indexer scores and top-k (pool 1), the shared router and the clamped SwiGLU.

The kernels keep activations in fp32 where the reference rounds to BF16 between steps; no quality number exists yet.

Tests: `gigachat4_meta_test` (CPU; YaRN frequencies against transformers) and `gigachat4_kernels_parity` (GPU; against CPU
references at the model's geometry, with microbench lines).
