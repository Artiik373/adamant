"""JIT-compiled CUDA extension wrapper for 14-bit 1MAD Viterbi trellis coding.

Compiles `ms_viterbi_cuda.cu` once via `torch.utils.cpp_extension.load` and caches
the binary. Delivers ~50-100x speedup over Python Viterbi on RTX 3090.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load

import ms_moe_converter as ms

_EXT = None


def get_viterbi_ext():
    global _EXT
    if _EXT is None:
        cu_path = Path(__file__).resolve().parent / "ms_viterbi_cuda.cu"
        cuda_flags = ["-O3", "--use_fast_math"]
        if os.name == "nt":
            cuda_flags += ["-Xcompiler", "/Zc:preprocessor"]
        _EXT = load(
            name="ms_viterbi_cuda_ext",
            sources=[str(cu_path)],
            extra_cuda_cflags=cuda_flags,
            verbose=False,
        )
    return _EXT


@torch.no_grad()
def viterbi_code_matrix_cuda(
    w_rot: torch.Tensor,
    hu: torch.Tensor,
    k: int,
    lut: torch.Tensor,
    window: int = 24,
    commit: int = 16,
    row_chunk: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full 14-bit 1MAD Viterbi + LDLQ using the compiled CUDA kernel."""
    ext = get_viterbi_ext()
    out_f, in_f = w_rot.shape
    device = w_rot.device
    w_work = w_rot.clone().to(torch.float32).contiguous()

    row_std = w_work.std(dim=1).clamp(min=1e-8)
    scales_fp16 = (ms.GAINS[k] * row_std / ms.CODEBOOK_STD).to(torch.float16)
    scales = scales_fp16.to(torch.float32).contiguous()

    states = torch.empty((out_f, in_f), dtype=torch.int32, device=device)
    w_quant = torch.empty((out_f, in_f), dtype=torch.float32, device=device)

    hu_diag = torch.diagonal(hu)
    inv_hu_diag2 = (1.0 / (hu_diag * hu_diag).clamp(min=1e-12)).cpu().contiguous()

    costs_a = torch.empty((row_chunk, ms.NUM_STATES), dtype=torch.float32, device=device)
    costs_b = torch.empty((row_chunk, ms.NUM_STATES), dtype=torch.float32, device=device)
    backptr = torch.empty((window, row_chunk, ms.NUM_STATES), dtype=torch.int16, device=device)

    for r0 in range(0, out_f, row_chunk):
        r1 = min(r0 + row_chunk, out_f)
        rc = r1 - r0
        d_c = scales[r0:r1].contiguous()
        w_c = w_work[r0:r1]

        ca = costs_a[:rc]
        cb = costs_b[:rc]
        bp = backptr[:, :rc, :]

        prev_state: Optional[torch.Tensor] = None
        col = 0
        while col < in_f:
            w_end = min(col + window, in_f)
            w_len = w_end - col
            step_commit = min(commit, w_len)

            committed_s = ext.viterbi_window_cuda(
                w_c[:, col:w_end].contiguous(),
                d_c,
                inv_hu_diag2[col:w_end],
                prev_state,
                k,
                step_commit,
                ca,
                cb,
                bp[:w_len],
            )
            committed_q = lut[committed_s.to(torch.int64)] * d_c.unsqueeze(1)
            states[r0:r1, col : col + step_commit] = committed_s
            w_quant[r0:r1, col : col + step_commit] = committed_q

            for t in range(step_commit):
                c_idx = col + t
                if c_idx + 1 < in_f:
                    err = (w_c[:, c_idx] - committed_q[:, t]) / hu_diag[c_idx]
                    w_c[:, c_idx + 1 :] -= err.unsqueeze(1) * hu[c_idx, c_idx + 1 :].unsqueeze(0)

            prev_state = committed_s[:, -1].contiguous()
            col += step_commit

    return scales_fp16, states, w_quant
