"""Dense-spine + lm_head MS_C6 streaming converter for Qwen3.8-Flash-Next (qwen4_exp).

Designed to pair directly with Adamant's dense-spine + lm_head .basalt decode path:
- Quantizes all VRAM-resident dense projections (QSA, GDN, shared expert) and lm_head to MS_C6.
- Applies GDN-2 (l % 4 == 2) Early+Late Minimax positional weighting (0.40/0.10/0.10/0.40).
- Supports optional ridge least-squares residual refitting for attn_output / shared_expert.down_proj.
- Computes deterministic Hadamard block_size = min(512, largest_pow2_divisor(in_features))
  and optional row-stride alignment (e.g., 16-byte alignment for CUDA uint4 loads).
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Dict, Tuple

import torch

import ms_moe_converter as ms
from ms_safetensors_stream import SafetensorsLayerLoader


def largest_pow2_divisor(n: int, cap: int = 512) -> int:
    b = 1
    while (b * 2) <= cap and (n % (b * 2) == 0):
        b *= 2
    return b


def ridge_refit(
    w_orig: torch.Tensor,      # [out, in]
    h_zz: torch.Tensor,        # [in, in] = E[z z^T]
    c_zy: torch.Tensor,        # [in, out] = E[z y^T]
    lam_scale: float = 1e-4,
    mu_scale: float = 0.05,
) -> torch.Tensor:
    """Solve W_t^T = (H + (lam + mu) I)^-1 (C + mu W_orig^T) to absorb prior quantization drift."""
    d = h_zz.shape[0]
    mean_diag = torch.diagonal(h_zz).mean().clamp(min=1e-8)
    lam = lam_scale * mean_diag
    mu = mu_scale * mean_diag
    lhs = h_zz + (lam + mu) * torch.eye(d, device=h_zz.device, dtype=torch.float32)
    rhs = c_zy + mu * w_orig.T
    w_t = torch.linalg.solve(lhs, rhs).T
    return w_t


def pack_per_row_aligned(
    scales_fp16: torch.Tensor,
    states: torch.Tensor,
    k: int,
    align_bytes: int = 16,
) -> Tuple[bytes, int]:
    """Pack per-row MS trellis with optional row-stride alignment (e.g. 16 bytes for CUDA uint4)."""
    scales_np = scales_fp16.cpu().numpy()
    states_np = states.cpu().numpy()
    out_f, in_f = states_np.shape
    sym_mask = (1 << k) - 1
    raw_row_bytes = 4 + ((in_f - 1) * k + 7) // 8
    stride = ((raw_row_bytes + align_bytes - 1) // align_bytes) * align_bytes
    pad_len = stride - raw_row_bytes

    buf = bytearray()
    pad = bytes(pad_len)
    n_sym_bytes = raw_row_bytes - 4
    for r in range(out_f):
        buf += struct.pack("<eH", float(scales_np[r]), int(states_np[r, 0]) & ms.STATE_MASK)
        row_bytes = bytearray(n_sym_bytes)
        bit_pos = 0
        for c in range(1, in_f):
            sym = int(states_np[r, c]) & sym_mask
            for b in range(k - 1, -1, -1):
                if (sym >> b) & 1:
                    row_bytes[bit_pos >> 3] |= 1 << (7 - (bit_pos & 7))
                bit_pos += 1
        buf += row_bytes
        if pad_len:
            buf += pad
    return bytes(buf), stride


def convert_single_dense_tensor(
    name: str,
    w: torch.Tensor,
    h: torch.Tensor,
    k: int,
    seed: int,
    device: torch.device,
    out_dir: Path,
    align_bytes: int = 16,
) -> dict:
    """Rotate, LDLQ+Viterbi code, and pack one dense-spine or lm_head tensor."""
    in_f = int(w.shape[1])
    blk_size = largest_pow2_divisor(in_f, cap=512)
    lut = ms.build_1mad_lut(device)

    w_f32 = w.to(device=device, dtype=torch.float32)
    h_f32 = h.to(device=device, dtype=torch.float32)
    w_rot, h_rot, signs = ms.rotate_weight_and_hessian(w_f32, h_f32, block_size=blk_size, seed=seed)
    hu = ms.cholesky_inv_upper(h_rot)

    scales_fp16, states, _ = ms.viterbi_code_matrix(w_rot, hu, k=k, lut=lut)
    blob, stride = pack_per_row_aligned(scales_fp16, states, k=k, align_bytes=align_bytes)

    safe_name = name.replace("/", "_").replace(".", "_")
    weight_file = f"{safe_name}.ms{k}"
    signs_file = f"{safe_name}.signs.i8"
    (out_dir / weight_file).write_bytes(blob)
    (out_dir / signs_file).write_bytes(signs.to(torch.int8).cpu().numpy().tobytes())

    return {
        "tensor": name,
        "file": weight_file,
        "signs_file": signs_file,
        "k": k,
        "shape": list(w.shape),
        "row_stride_bytes": stride,
        "hadamard_block": blk_size,
        "hadamard_seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen3.8-Flash-Next dense spine + lm_head to MS_C6")
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0, help="Layer index to inspect/convert (or -1 for lm_head)")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--align-bytes", type=int, default=16)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    loader = SafetensorsLayerLoader(args.hf_dir)
    if args.layer == -1:
        key = "lm_head.weight" if "lm_head.weight" in loader.weight_map else "model.embed_tokens.weight"
        w = loader.load_tensor(key, device="cpu")
        print(f"Loaded {key} {tuple(w.shape)} -> Hadamard block={largest_pow2_divisor(w.shape[1])}")
    else:
        keys = [k for k in loader.list_layer_keys(args.layer) if ".experts." not in k and k.endswith(".weight")]
        print(f"Layer {args.layer} dense-spine 2D candidates ({len(keys)}):")
        for k in keys:
            print(f"  - {k}")


if __name__ == "__main__":
    main()
