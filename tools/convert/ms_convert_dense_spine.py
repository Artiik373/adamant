"""Dense-spine + lm_head MS_C6 streaming converter for Qwen3.8-Flash-Next (qwen4_exp).

Designed to pair directly with Amperite's dense-spine + lm_head .basalt decode path:
- Quantizes all VRAM-resident dense projections (QSA, GDN, shared expert) and lm_head to MS_C6.
- Applies GDN-2 (l % 4 == 2) Early+Late Minimax positional weighting (0.40/0.10/0.10/0.40).
- Supports optional ridge least-squares residual refitting for attn_output / shared_expert.down_proj.
- Computes deterministic Hadamard block_size = min(512, largest_pow2_divisor(in_features)), the engine's rule.
- Puts the GDN value heads into the GGUF order before coding (attn_qkv / attn_gate rows, ssm_out columns), and writes
  one engine payload per tensor (ms_pack.write_record) for ms_pack.py to place into a .basalt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import model_sections as msec
import ms_moe_converter as ms
from ms_pack import write_record
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


def to_gguf_order(gguf_name: str, w: torch.Tensor, h: torch.Tensor):
    """W [out, in] and H [in, in] in the GGUF's value-head order: attn_qkv / attn_gate permute output rows, ssm_out
    permutes input columns (before the rotation mixes them)."""
    rest = gguf_name.split(".", 2)[-1]
    if rest in ("attn_qkv.weight", "attn_gate.weight"):
        n = int(w.shape[0])
        first = msec.GDN_QK_ROWS if n > msec.GDN_V_HEADS * msec.GDN_HEAD else 0
        return w[torch.as_tensor(msec.gdn_index(n, first), device=w.device)], h
    if rest == "ssm_out.weight":
        idx = torch.as_tensor(msec.gdn_index(int(w.shape[1]), 0), device=w.device)
        return w[:, idx], h[idx][:, idx]
    return w, h


def convert_single_dense_tensor(
    gguf_name: str,
    w: torch.Tensor,
    h: torch.Tensor,
    k: int,
    seed: int,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """Rotate, LDLQ+Viterbi code, and write one dense-spine or lm_head tensor (HF W [out, in]) as an engine record."""
    out_f, in_f = int(w.shape[0]), int(w.shape[1])
    blk_size = largest_pow2_divisor(in_f, cap=512)
    lut = ms.build_1mad_lut(device)

    w_f32 = w.to(device=device, dtype=torch.float32)
    h_f32 = h.to(device=device, dtype=torch.float32)
    w_f32, h_f32 = to_gguf_order(gguf_name, w_f32, h_f32)
    w_rot, h_rot, signs = ms.rotate_weight_and_hessian(w_f32, h_f32, block_size=blk_size, seed=seed)
    hu = ms.cholesky_inv_upper(h_rot)

    scales_fp16, states, _ = ms.viterbi_code_matrix(w_rot, hu, k=k, lut=lut)
    rows = ms.pack_per_row_ms(scales_fp16, states, k=k)
    rec = write_record(out_dir, gguf_name, k, in_f, out_f, rows, signs.to(torch.int8).cpu().numpy())
    rec.update({"hadamard_block": blk_size, "hadamard_seed": seed})
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen3.8-Flash-Next dense spine + lm_head to MS_C6")
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0, help="Layer index to inspect/convert (or -1 for lm_head)")
    parser.add_argument("--k", type=int, default=6)
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
