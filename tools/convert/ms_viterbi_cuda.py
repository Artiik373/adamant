"""JIT-compiled CUDA extension wrapper for 14-bit 1MAD Viterbi trellis coding.

Compiles `ms_viterbi_cuda.cu` once via `torch.utils.cpp_extension.load` and caches
the binary. 
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load

import ms_moe_converter as ms

_EXT: dict = {}


def get_viterbi_ext(fp4_c: Optional[float] = None):
    """fp4_c: a probe build whose decode snaps each value to the E2M1 grid times fp4_c."""
    if fp4_c not in _EXT:
        import importlib.util
        import sysconfig
        from torch.utils.cpp_extension import _get_build_directory

        cu_path = Path(__file__).resolve().parent / "ms_viterbi_cuda.cu"
        name = "ms_viterbi_cuda_ext"
        if fp4_c is not None:
            name += "_fp4_" + f"{fp4_c!r}".replace(".", "p")
        build_dir = Path(_get_build_directory(name, verbose=False))
        lib_path = build_dir / f"{name}{'.pyd' if os.name == 'nt' else '.so'}"
        if lib_path.exists() and lib_path.stat().st_mtime >= cu_path.stat().st_mtime:
            spec = importlib.util.spec_from_file_location(name, str(lib_path))
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _EXT[fp4_c] = mod
                return mod
        cuda_flags = ["-O3", "--use_fast_math"]
        if os.name == "nt":
            cuda_flags += ["-Xcompiler", "/Zc:preprocessor"]
        if fp4_c is not None:
            cuda_flags.append(f"-DMS_FP4_C={fp4_c!r}f")
        _EXT[fp4_c] = load(name=name, sources=[str(cu_path)], extra_cuda_cflags=cuda_flags, verbose=False)
    return _EXT[fp4_c]


@torch.no_grad()
def viterbi_code_matrix_cuda(
    w_rot: torch.Tensor,
    hu: torch.Tensor,
    k: int,
    lut: torch.Tensor,
    window: int = 32,
    commit: int = 8,
    scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full 14-bit 1MAD Viterbi + LDLQ using the compiled CUDA kernel."""
    sc = scales.unsqueeze(0) if scales is not None else None
    s, st, q = viterbi_code_batched_cuda(w_rot.unsqueeze(0), hu.unsqueeze(0), k, lut, window, commit, scales=sc)
    return s[0], st[0], q[0]


@torch.no_grad()
def viterbi_code_batched_cuda(
    w_rot: torch.Tensor,
    hu: torch.Tensor,
    k: int,
    lut: torch.Tensor,
    window: int = 32,
    commit: int = 8,
    fp4_c: Optional[float] = None,
    scales: Optional[torch.Tensor] = None,
    return_quant: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Viterbi + LDLQ for E matrices at once: w_rot [E, out, in], hu [E, in, in]. fp4_c: see get_viterbi_ext;
    lut must then hold the same snapped values. scales: optional [E, out] initial fp16/fp32 row scales."""
    ext = get_viterbi_ext(fp4_c)
    n_mat, out_f, in_f = w_rot.shape
    if not (torch.isfinite(w_rot).all() and torch.isfinite(hu).all()):
        raise ValueError("viterbi input has non-finite weights or Hessian factor")
    if hu.shape != (n_mat, in_f, in_f) or not 1 <= k <= 6:
        raise ValueError(f"bad viterbi input: w {tuple(w_rot.shape)}, hu {tuple(hu.shape)}, k {k}")
    rows = n_mat * out_f
    device = w_rot.device
    w_work = w_rot.to(torch.float32).clone().contiguous()
    w_work_2d = w_work.view(rows, in_f)
    hu = hu.to(torch.float32)

    if scales is None:
        row_std = w_work.std(dim=2).clamp(min=1e-8)
        scales_fp16 = (ms.GAINS[k] * row_std / ms.CODEBOOK_STD).to(torch.float16)
    else:
        if scales.shape != (n_mat, out_f):
            raise ValueError(f"bad scales shape: expected {(n_mat, out_f)}, got {tuple(scales.shape)}")
        scales_fp16 = scales.to(device=device, dtype=torch.float16)
    d_flat = scales_fp16.to(torch.float32).reshape(rows).contiguous()
    d_flat_1 = d_flat.unsqueeze(1)

    states = torch.empty((n_mat, out_f, in_f), dtype=torch.int32, device=device)
    w_quant = torch.empty((n_mat, out_f, in_f), dtype=torch.float32, device=device) if return_quant else None

    hu_diag = torch.diagonal(hu, dim1=1, dim2=2)  # [E, in]
    inv_hu_diag2 = (1.0 / (hu_diag * hu_diag).clamp(min=1e-12)).contiguous()
    backptr = torch.empty(rows * window * (ms.NUM_STATES >> k), dtype=torch.uint8, device=device)

    n_full = in_f // commit
    u_inv_full = None
    if n_full > 0:
        u_blks = torch.stack(
            [hu[:, c : c + commit, c : c + commit] for c in range(0, n_full * commit, commit)], dim=1
        )
        eye_c = torch.eye(commit, device=device, dtype=torch.float32).expand(n_mat, n_full, commit, commit)
        u_inv_full = torch.linalg.solve_triangular(u_blks, eye_c, upper=True, left=True)

    prev_state: Optional[torch.Tensor] = None
    col = 0
    step = 0
    while col < in_f:
        w_end = min(col + window, in_f)
        c_end = col + min(commit, w_end - col)
        n = c_end - col

        committed_s = ext.viterbi_window_cuda(
            w_work_2d[:, col:w_end].contiguous(),
            d_flat,
            inv_hu_diag2[:, col:w_end].contiguous(),
            prev_state,
            k,
            n,
            backptr,
        )
        committed_q = (lut[committed_s.to(torch.int64)] * d_flat_1).view(n_mat, out_f, n)
        states[:, :, col:c_end] = committed_s.view(n_mat, out_f, n)
        if w_quant is not None:
            w_quant[:, :, col:c_end] = committed_q

        # LDLQ: the errors e of the committed block solve e U_blk = w - q (U_blk upper triangular), then one
        # batched GEMM feeds them to the columns after it in place.
        if c_end < in_f:
            diff = w_work[:, :, col:c_end] - committed_q
            if u_inv_full is not None and step < n_full and n == commit:
                err = torch.bmm(diff, u_inv_full[:, step])
            else:
                err = torch.linalg.solve_triangular(hu[:, col:c_end, col:c_end], diff, upper=True, left=False)
            w_work[:, :, c_end:].baddbmm_(err, hu[:, col:c_end, c_end:], beta=1.0, alpha=-1.0)

        prev_state = committed_s[:, -1].contiguous()
        col = c_end
        step += 1

    return scales_fp16, states, w_quant
