"""End-to-end verification for tools/convert/ms_moe_converter.py.

Run: python3 tests/test_ms_converter.py
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "convert"))
import ms_moe_converter as ms


def unpack_per_row_ms(blob: bytes, out_f: int, in_f: int, k: int) -> torch.Tensor:
    """Reference bit-exact unpacker matching src/kernels/ms_1mad_avx2.h."""
    lut = ms.build_1mad_lut(torch.device("cpu"))
    stride = ms.ms_row_bytes(in_f, k)
    assert len(blob) == out_f * stride

    out = torch.empty((out_f, in_f), dtype=torch.float32)
    sym_mask = (1 << k) - 1
    for r in range(out_f):
        row = blob[r * stride : (r + 1) * stride]
        d_fp16, s0 = struct.unpack_from("<eH", row, 0)
        d = float(torch.tensor(d_fp16, dtype=torch.float16).to(torch.float32))
        qs = row[4:]

        state = s0 & ms.STATE_MASK
        out[r, 0] = lut[state].item() * d
        bit_pos = 0
        for c in range(1, in_f):
            sym = 0
            for _ in range(k):
                byte_val = qs[bit_pos >> 3]
                bit = (byte_val >> (7 - (bit_pos & 7))) & 1
                sym = (sym << 1) | bit
                bit_pos += 1
            state = ((state << k) | (sym & sym_mask)) & ms.STATE_MASK
            out[r, c] = lut[state].item() * d
    return out


def test_rmt_denoising() -> None:
    torch.manual_seed(0)
    d = 128
    # Cold expert: only 32 samples (r/D = 0.25 < 0.5)
    x = torch.randn(32, d)
    h = (x.T @ x) / 32.0
    h_den = ms.denoise_hessian_rmt(h, n_samples=32)
    evals = torch.linalg.eigvalsh(h_den)
    assert evals.min().item() > 0.0, "Denoised Hessian must be strictly positive definite"
    print(f"[RMT] cold expert min_eval={evals.min().item():.4e}, max_eval={evals.max().item():.4e}")


def test_viterbi_and_pack(out_f: int, in_f: int, k: int, block_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123 + k + in_f)
    w = torch.randn(out_f, in_f, device=device) * 0.02
    x = torch.randn(256, in_f, device=device)
    h = (x.T @ x) / 256.0

    h_den = ms.denoise_hessian_rmt(h, n_samples=256)
    w_rot, h_rot, _ = ms.rotate_weight_and_hessian(w, h_den, block_size=block_size, seed=7)
    hu = ms.cholesky_inv_upper(h_rot)
    lut = ms.build_1mad_lut(device)

    scales_fp16, states, w_q = ms.viterbi_code_matrix(w_rot, hu, k=k, lut=lut)
    blob = ms.pack_per_row_ms(scales_fp16, states, k=k)
    w_unpacked = unpack_per_row_ms(blob, out_f, in_f, k)

    max_diff = (w_unpacked - w_q.cpu()).abs().max().item()
    rel_rmse = (w_rot.cpu() - w_unpacked).norm().item() / w_rot.cpu().norm().item()
    print(f"[MS_C{k} {out_f}x{in_f} blk={block_size}] pack_err={max_diff:.2e}, rel_rmse={rel_rmse:.4f}")
    assert max_diff < 1e-6, f"Pack/unpack mismatch: {max_diff}"


def main() -> None:
    test_rmt_denoising()
    test_viterbi_and_pack(16, 640, k=3, block_size=128)
    test_viterbi_and_pack(16, 2560, k=3, block_size=512)
    test_viterbi_and_pack(16, 640, k=4, block_size=128)
    test_viterbi_and_pack(16, 2560, k=6, block_size=512)
    print("ALL MS CONVERTER TESTS PASSED")


if __name__ == "__main__":
    main()
