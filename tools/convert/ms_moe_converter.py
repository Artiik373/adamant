"""Layer-by-layer streaming MS (Mad Shift) trellis converter for Qwen3.8-Flash-Next (qwen4_exp).

Designed for single-GPU (RTX 3090 24 GB) + 64 GB DDR5 workstations:
- Streams one BF16 layer (~5.2 GB) onto GPU at a time; holds dual residual streams
  (h_ref, h_q) and per-expert Hessians in host DDR5.
- Uses per-row 14-bit 1MAD bitshift-trellis packing (no 512-weight block restriction),
  supporting both hidden_size=2560 (Hadamard block=512) and moe_intermediate_size=640
  (Hadamard block=128).
- Applies Marchenko-Pastur (MP+DG-Flat) RMT denoising for cold experts (r/D < 0.5)
  and Early+Late Minimax positional weighting (0.40/0.10/0.10/0.40) on GDN-2 layers
  (l % 4 == 2).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ms_pack import ms_row_bytes


# QTIP 1MAD constants (arXiv:2406.11235)
MAD_MUL = 34038481
MAD_ADD = 76625530
STATE_BITS = 14
NUM_STATES = 1 << STATE_BITS
STATE_MASK = NUM_STATES - 1
CODEBOOK_STD = 147.8
GAINS = {1: 1.0, 2: 1.15, 3: 1.2, 4: 1.2, 5: 1.3, 6: 1.2}


@dataclass
class LayerBitConfig:
    routed_k: int = 3
    spine_k: int = 6
    head_k: int = 6


def build_1mad_lut(device: torch.device) -> torch.Tensor:
    """Precompute the 2^14 1MAD integer codebook values in [-510, 510] as float32."""
    s = torch.arange(NUM_STATES, dtype=torch.int64, device=device)
    x = (s * MAD_MUL + MAD_ADD) & 0xFFFFFFFF
    v = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF) - 510
    return v.to(torch.float32)


def fwht_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Orthonormal block Fast Walsh-Hadamard Transform along the last dimension."""
    orig_shape = x.shape
    n = orig_shape[-1]
    assert n % block_size == 0, f"Width {n} must be divisible by Hadamard block_size {block_size}"
    y = x.reshape(-1, n // block_size, block_size).clone()
    h = 1
    while h < block_size:
        y = y.reshape(-1, block_size // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, n // block_size, block_size)
        h *= 2
    return (y / math.sqrt(block_size)).reshape(orig_shape)


def random_hadamard_signs(width: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (width,), generator=g, dtype=torch.int8) * 2 - 1).to(
        device=device, dtype=torch.float32
    )
    return signs


def rotate_weight_and_hessian(
    w: torch.Tensor, h: torch.Tensor, block_size: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate W -> W R and H -> R^T H R where R = diag(s) * blockdiag(H_b)."""
    signs = random_hadamard_signs(w.shape[1], seed, w.device)
    w_rot = fwht_blocks(w * signs.unsqueeze(0), block_size)
    h_rot = fwht_blocks(h * signs.unsqueeze(0), block_size)
    h_rot = fwht_blocks((h_rot.T * signs.unsqueeze(0)), block_size).T
    return w_rot, h_rot, signs


def best_sign_seed(w: torch.Tensor, block_size: int, seed: int, trials: int) -> int:
    """Of `trials` seeds (seed, seed + 1, ...), the one whose rotated weights have the smallest fourth moment
    sum(w^4) / sum(w^2)^2: the fewest outliers for the trellis (OptRot's objective over the sign vectors only)."""
    best, best_m = seed, None
    for t in range(max(trials, 1)):
        sd = seed + t
        wr = fwht_blocks(w * random_hadamard_signs(w.shape[1], sd, w.device).unsqueeze(0), block_size)
        m = (wr.pow(4).sum() / wr.pow(2).sum().pow(2)).item()
        if best_m is None or m < best_m:
            best, best_m = sd, m
    return best


def best_sign_seed_h(h: torch.Tensor, block_size: int, seed: int, trials: int) -> int:
    """Of `trials` seeds (seed, seed + 1, ...), the one whose rotated Hessian has the smallest sum of LDLQ pivots
    1 / Hu[c, c]^2 (tr D of H = L^T D L): the factor on the expected LDLQ error for a fixed per-column noise.
    The pivots are those of a Cholesky of H in reverse order: one factorization, in float64 on the CPU."""
    h = h.detach().to("cpu", torch.float64)
    h = 0.5 * (h + h.T)
    h = h + 0.01 * h.diagonal().mean() * torch.eye(h.shape[0], dtype=h.dtype)   # cholesky_inv_upper's damping
    best, best_m = seed, None
    for t in range(max(trials, 1)):
        sd = seed + t
        _, h_rot, _ = rotate_weight_and_hessian(h[:1], h, block_size, sd)
        m = (torch.linalg.cholesky(h_rot.flip(0, 1)).diagonal() ** 2).sum().item()
        if best_m is None or m < best_m:
            best, best_m = sd, m
    return best


def denoise_hessian_rmt(h: torch.Tensor, n_samples: int, base_damp: float = 0.01) -> torch.Tensor:
    """Marchenko-Pastur (MP+DG-Flat) RMT denoising for undersampled/cold experts (r/D < 0.5)."""
    d = h.shape[0]
    h = 0.5 * (h + h.T)
    mean_diag = torch.diagonal(h).mean().clamp(min=1e-8)

    if n_samples <= 0:
        return torch.eye(d, device=h.device, dtype=h.dtype) * mean_diag

    gamma = d / float(max(n_samples, 1))
    if gamma <= 2.0:
        return h + (base_damp * mean_diag) * torch.eye(d, device=h.device, dtype=h.dtype)

    evals, evecs = torch.linalg.eigh(h)
    evals = evals.clamp(min=0.0)
    sigma2 = evals.median().clamp(min=1e-8)
    lambda_plus = sigma2 * (1.0 + math.sqrt(gamma)) ** 2
    bulk_mask = evals <= lambda_plus
    if bulk_mask.any():
        bulk_mean = evals[bulk_mask].mean().clamp(min=base_damp * mean_diag)
        evals = torch.where(bulk_mask, bulk_mean, evals)
    h_denoised = (evecs * evals.unsqueeze(0)) @ evecs.T
    h_denoised = 0.5 * (h_denoised + h_denoised.T)
    return h_denoised + (base_damp * mean_diag) * torch.eye(d, device=h.device, dtype=h.dtype)


def cholesky_inv_upper(h: torch.Tensor) -> torch.Tensor:
    """Compute upper Cholesky factor Hu of H^-1 with mandatory >=1% GPTQ/QTIP damping."""
    d = h.shape[0]
    eye = torch.eye(d, device=h.device, dtype=torch.float32)
    h = 0.5 * (h + h.T)
    mean_diag = torch.diagonal(h).mean().clamp(min=1e-8)
    for damp in (0.01, 0.05, 0.2, 0.5):
        try:
            h_d = h + (damp * mean_diag) * eye
            l_fac = torch.linalg.cholesky(h_d)
            h_inv = torch.cholesky_inverse(l_fac)
            return torch.linalg.cholesky(h_inv, upper=True)
        except RuntimeError:
            continue
    return eye


def gdn2_positional_weights(seq_len: int, device: torch.device) -> torch.Tensor:
    """Early+Late Minimax weighting (0.40 Q1, 0.10 Q2, 0.10 Q3, 0.40 Q4) for GDN-2 layers."""
    q = seq_len // 4
    w = torch.empty(seq_len, device=device, dtype=torch.float32)
    w[:q] = 0.40
    w[q : 2 * q] = 0.10
    w[2 * q : 3 * q] = 0.10
    w[3 * q :] = 0.40
    return w / w.mean()


def viterbi_code_matrix(
    w_rot: torch.Tensor,
    hu: torch.Tensor,
    k: int,
    lut: torch.Tensor,
    window: int = 32,
    commit: int = 8,
    row_chunk: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fast GPU-batched sliding-window 14-bit Viterbi search with LDLQ error feedback."""
    out_f, in_f = w_rot.shape
    device = w_rot.device
    w_work = w_rot.clone().to(torch.float32)

    row_std = w_work.std(dim=1).clamp(min=1e-8)
    scales_fp16 = (GAINS[k] * row_std / CODEBOOK_STD).to(torch.float16)
    scales = scales_fp16.to(torch.float32)

    states = torch.empty((out_f, in_f), dtype=torch.int32, device=device)
    w_quant = torch.empty((out_f, in_f), dtype=torch.float32, device=device)

    v2 = (lut * lut).unsqueeze(0)
    lut_row = lut.unsqueeze(0)
    shift_r = STATE_BITS - k
    n_pred = 1 << k
    n_rem = 1 << shift_r

    hu_diag = torch.diagonal(hu)
    inv_hu_diag2 = 1.0 / (hu_diag * hu_diag).clamp(min=1e-12)
    sym_arange = torch.arange(n_pred, device=device, dtype=torch.int64).unsqueeze(0)
    rem_arange = torch.arange(n_rem, device=device, dtype=torch.int32).unsqueeze(0)

    for r0 in range(0, out_f, row_chunk):
        r1 = min(r0 + row_chunk, out_f)
        rc = r1 - r0
        d_c = scales[r0:r1].unsqueeze(1)
        d2_c = d_c * d_c
        w_c = w_work[r0:r1]
        row_idx = torch.arange(rc, device=device, dtype=torch.int64)
        backptr = torch.empty((window, rc, NUM_STATES), dtype=torch.int16, device=device)

        prev_state: Optional[torch.Tensor] = None
        col = 0
        while col < in_f:
            w_end = min(col + window, in_f)
            w_len = w_end - col
            step_commit = min(commit, w_len)

            if prev_state is None:
                costs = torch.zeros((rc, NUM_STATES), device=device, dtype=torch.float32)
            else:
                costs = torch.full((rc, NUM_STATES), float("inf"), device=device, dtype=torch.float32)
                base = (prev_state.to(torch.int64) << k) & STATE_MASK
                valid_next = base.unsqueeze(1) | sym_arange
                costs.scatter_(1, valid_next, 0.0)

            for t in range(w_len):
                c_idx = col + t
                piv = inv_hu_diag2[c_idx]
                x_col = w_c[:, c_idx].unsqueeze(1)
                branch = piv * (d2_c * v2 - (2.0 * x_col * d_c) * lut_row)

                if t == 0 and prev_state is None:
                    costs = branch
                    backptr[0].zero_()
                elif t == 0:
                    costs = costs + branch
                    backptr[0] = prev_state.unsqueeze(1).expand(rc, NUM_STATES).to(torch.int16)
                else:
                    view = costs.view(rc, n_pred, n_rem)
                    min_cost, best_a = view.min(dim=1)
                    best_pred = ((best_a.to(torch.int32) << shift_r) | rem_arange).to(torch.int16)
                    costs = min_cost.unsqueeze(2).expand(rc, n_rem, n_pred).reshape(rc, NUM_STATES) + branch
                    backptr[t] = best_pred.unsqueeze(2).expand(rc, n_rem, n_pred).reshape(rc, NUM_STATES)

            cur = costs.argmin(dim=1).to(torch.int32)
            path = torch.empty((rc, w_len), dtype=torch.int32, device=device)
            for t in range(w_len - 1, -1, -1):
                path[:, t] = cur
                cur = backptr[t, row_idx, cur.to(torch.int64)].to(torch.int32)

            committed_s = path[:, :step_commit]
            committed_q = lut[committed_s.to(torch.int64)] * d_c
            states[r0:r1, col : col + step_commit] = committed_s
            w_quant[r0:r1, col : col + step_commit] = committed_q

            for t in range(step_commit):
                c_idx = col + t
                if c_idx + 1 < in_f:
                    err = (w_c[:, c_idx] - committed_q[:, t]) / hu_diag[c_idx]
                    w_c[:, c_idx + 1 :] -= err.unsqueeze(1) * hu[c_idx, c_idx + 1 :].unsqueeze(0)

            prev_state = committed_s[:, -1]
            col += step_commit

    return scales_fp16, states, w_quant


def pack_per_row_ms(scales_fp16: torch.Tensor, states: torch.Tensor, k: int) -> bytes:
    """Pack per-row MS trellis without 512-weight padding."""
    scales_np = scales_fp16.cpu().numpy()
    states_np = states.cpu().numpy()
    out_f, in_f = states_np.shape
    sym_mask = (1 << k) - 1
    n_sym_bytes = ms_row_bytes(in_f, k) - 4

    buf = bytearray()
    for r in range(out_f):
        buf += struct.pack("<eH", float(scales_np[r]), int(states_np[r, 0]) & STATE_MASK)
        row_bytes = bytearray(n_sym_bytes)
        bit_pos = 0
        for c in range(1, in_f):
            sym = int(states_np[r, c]) & sym_mask
            for b in range(k - 1, -1, -1):
                if (sym >> b) & 1:
                    row_bytes[bit_pos >> 3] |= 1 << (7 - (bit_pos & 7))
                bit_pos += 1
        buf += row_bytes
    return bytes(buf)


def build_default_tier_plan(num_layers: int = 48) -> List[LayerBitConfig]:
    """<= 55 GB main shard allocation for Qwen3.8-Flash-Next (125B main + 51B PLE separate)."""
    ms_c4_layers = {0, 1, 2, 6, 10, 38, 42, 44, 45, 46, 47}
    return [
        LayerBitConfig(routed_k=4 if l in ms_c4_layers else 3, spine_k=6, head_k=6)
        for l in range(num_layers)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer-by-layer streaming MS trellis converter for Qwen3.8-Flash-Next (.basalt)"
    )
    parser.add_argument("--hf-dir", type=Path, required=True, help="Path to BF16 Qwen3.8-Flash-Next")
    parser.add_argument("--calib-pt", type=Path, required=True, help="Calibration token IDs [N, 512]")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for MS shards")
    parser.add_argument("--dry-run", action="store_true", help="Print layer bit plan and exit")
    args = parser.parse_args()

    plan = build_default_tier_plan(48)
    if args.dry_run:
        c4 = [i for i, p in enumerate(plan) if p.routed_k == 4]
        print(f"MS_C4 routed layers ({len(c4)}/48): {c4}")
        print("All other routed layers: MS_C3 | Dense spine & lm_head: MS_C6")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": "adamant_ms_per_row_v1",
        "state_bits": STATE_BITS,
        "code": "1mad",
        "hadamard_blocks": {"hidden_size_2560": 512, "moe_intermediate_640": 128},
        "layers": [p.__dict__ for p in plan],
    }
    (args.out_dir / "ms_plan.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote conversion plan to {args.out_dir / 'ms_plan.json'}")
    print("Next: wire qwen4_exp safetensors layer loader into the streaming loop.")


if __name__ == "__main__":
    main()
