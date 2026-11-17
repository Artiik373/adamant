# EXL3 support for basalt: format notes and status

Source model: `G:\llm\tabbyAPI\models\exl3-318bpw` (Qwen3.8-Flash-Next, EXL3 3.18 bpw). The format follows QTIP
(Tseng, Sun, Hou, De Sa 2024): a bitshift trellis, tail-biting, computed code, random Hadamard incoherence.
`exl3_ref.py` is an independent numpy reference decoder. `probe_structure.py` tests decoding hypotheses on real tensors.
No third-party EXL3 code is used or vendored, and nothing is compared against it.

## What the checkpoint contains (from `quantization_config.json` and the safetensors index)

- Every quantized linear has four tensors: `trellis` (int16 `[K/16, N/16, W]`), `suh` (fp16 `[K]`), `svh` (fp16 `[N]`),
  `mul1` (int32 scalar: the codebook id).
- `W` is 16 x bits-per-weight in 16-bit words per 16x16 tile. Routed experts (gate/up/down, 512 per layer, 48 layers):
  layers 0-3 and 44-47 use `W = 56` (3.5 bits per weight, a half-integer rate), layers 4-43 use `W = 48` (3 bits). Shared experts and the MTP drafter, the attention and
  linear-attention projections: integer rates 5 to 7 (`W` = 80..112). `lm_head`: 6 bits (`W` = 96).
- Not quantized (fp16/bf16): token embedding, router gates, hyper-connection weights, norms, conv/dt/A parameters.
- The n-gram table (`ngram_embedding.safetensors`, 51.8 GB) is itself trellis-coded: 128 shards `[2500012, 81]` int16.
  Its row layout is not analysed yet.
- Vision tower: quantized, 6 bit. Some of its dimensions (4304) are not multiples of 128; padding is not yet understood.

## Decoding, as confirmed on layer 0 expert 0 (`probe_structure.py`)

- Tile: 256 weights in a ring of `16 * K` bits. The bit stream is MSB-first within 32-bit words; two int16 words form
  one 32-bit word (low half first). Weight `i` has a 16-bit window ending after its own bits; a half-integer rate
  alternates `KA` and `KA + 1` bits per step (period 16, mask 0xAAAA).
- Ring position to tile element: lane-interleaved 16x16 order (`ring_to_element`), with tile rows along the input
  dimension K. **Confirmed:** with this orientation the decoded matrix has real structure (top-1% singular energy 3.0%
  against 2.0% for noise); the transposed orientation is indistinguishable from noise.
- Computed code `mul1`: `x = window * 0x83DCD12D (mod 2^32)`, value `= (sum of the 4 bytes of x - 510) / 147.7`.
  The decoded inner matrix has unit-scale, kurtosis 2.96 (Gaussian), as QTIP's computed codes are meant to give.

## NOT yet settled

The placement of the Hadamard transforms and the scale vectors cannot be decided from weight statistics: every
variant I tried gives a plausible weight matrix (see the table in the session log). The working hypothesis is the
standard incoherence-processing form `W = diag(suh) . H . W_inner . H . diag(svh)` (QuIP#/QTIP: `y = H_n(H_k(x * suh) W_inner) * svh`),
with H the normalized 128-block Sylvester Hadamard. Verification needs a functional test (model output quality), so the
engine must carry a switch between the few remaining variants until output is coherent.

Also open: `suh` has varying magnitudes (up to about 0.024, signs ~50/50) and `svh` has magnitudes around 1 to 1.6, so
both carry scale as well as sign.

## Plan

1. (done) Reference decoder + structure probe.
2. Converter: pack the EXL3 tensors into `.basalt` sections without re-quantizing ("keep EXL3 as is").
3. CPU expert kernel (this machine has AVX2 only), with parity tests against the reference decoder.
4. GPU kernels: decode, grouped expert prompt path, peer tier.
5. Dense projections, head, MTP drafter, n-gram table, vision.
6. Functional check of the transform placement by output quality.

## Building the expert pack

    python tools/exl3/build_experts.py --exl3 <exl3 folder> --base <existing .basalt> --out <new .basalt> [--dry-run] [--verify]

Copies every section of the base pack except `experts`, `experts.layout` and `meta`, and writes the EXL3 experts
unchanged as layout v8 (see the script's docstring for the blob format). Dry run on exl3-318bpw: experts 43.80 GiB,
output about 78.5 GiB (needs that much free disk).

## Engine status

- CPU: `src/kernels/cpu/exl3.cpp` (AVX2 and AVX-VNNI columns, plain C reference) is wired into the native-expert path (`NativeFmt::exl3_w`,
  layout v8, 128-aligned pool parts). `tests/kernels/cpu/exl3_cpu_test.cpp` checks it against a bit-by-bit oracle.
- GPU: `src/kernels/cuda/exl3_kernels.cu` has the grouped decode kernels (reached through `native_expert_grouped`) and the
  prompt path's fp16 dequantizers. `exl3_lane_test` checks the warp mapping on the host; `exl3_gpu_parity` needs a GPU.
  The device-planned MoE tail (`dec_moe_*`) does not cover EXL3 layers and turns itself off.
- Not done: dense projections, head, shared experts, MTP drafter, n-gram table and vision in EXL3 form, and the peer/remote
  tiers' own paths.
- The sign placement relative to the Hadamard transforms is `--exl3-variant` (0..3, default 0); not yet checked against
  model output.

## Dense tensors (`--dense`)

`build_experts.py --dense` also replaces the dense projections of the base pack's `gguf` section (GDN
`attn_qkv`/`attn_gate`/`ssm_out`, attention `q/k/v/output`, shared experts, `output.weight`) with EXL3 records
(`trellis | suh | svh`, type `96 + W/8`, shape `[K, N]`), taken from the checkpoint unchanged.  The GGUF order of the
linear-attention value heads differs from the checkpoint's; it is applied as a permutation of whole 128-blocks of the
trellis and of `suh`/`svh` (`dense_pack.py`), which changes no weight.  Untouched tensors (embeddings, norms, routers,
`ssm_*`) are copied as they are.  The MTP drafter, vision tower and n-gram table are not converted.

## The MTP drafter (`--mtp FILE`)

`--mtp mtp-2.6bpw.safetensors` (the EXL3 sidecar of mixed-rate drafter heads) replaces the drafter's `mtp/dense`
projections (fc_embedding, fc_hidden, q/k/v/o, shared expert) and its 512 routed experts (gate/up and down at different
rates, so a blob is `[gate | up | down]` with two types).  The index gets a first line
`#experts exl3 GU_TYPE D_TYPE BLOB_BYTES`; its tensors of kind `exl3` carry `rows` = outputs, `cols` = inputs.  Norms,
hyper-connection matrices, the router and the shared-expert gate stay from the base pack.  The base pack's self-distilled
dense tuning is dropped with the section.

    python tools/exl3/build_experts.py --exl3 CKPT --base BASE.basalt --dense --mtp mtp-2.6bpw.safetensors --out OUT.basalt

The output head is converted too (`--keep-head` leaves it).  An EXL3 head has no rows of its own (the output transform
mixes 128 of them), so at load the engine dequantizes the drafter's token subset into a Q8_0 copy; the drafter ranks on
the copy and the head verifies every draft.

## Swapping only the drafter (`swap_mtp.py`)

`swap_mtp.py PACK --from OTHER.basalt` (another pack's drafter, e.g. the IQ3 pack's) or
`swap_mtp.py PACK --sidecar mtp-2.6bpw.safetensors --base OTHER.basalt` replaces just `mtp/*` and `meta` in place: about
1 GB is appended and one header write switches over, nothing else moves.  The old drafter stays as a gap;
`--undo DIR` (printed at the end) restores the file bit for bit.  Stop every process that has the pack open first.
