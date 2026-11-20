"""Unit tests for MS trellis conversion utilities (no GPU or full model required)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "convert"))

import ms_moe_converter as ms  # noqa: E402
from ms_convert_dense_spine import (  # noqa: E402
    largest_pow2_divisor,
    pack_per_row_aligned,
    ridge_refit,
)
from eval_ms_kld import compute_kld_metrics  # noqa: E402


def test_largest_pow2_divisor() -> None:
    assert largest_pow2_divisor(2048, cap=512) == 512
    assert largest_pow2_divisor(1536, cap=512) == 512
    assert largest_pow2_divisor(768, cap=512) == 256
    assert largest_pow2_divisor(192, cap=512) == 64
    assert largest_pow2_divisor(25, cap=512) == 1


def test_hadamard_orthogonality() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 128)
    y = ms.block_hadamard_rows(x, block_size=64)
    x_rec = ms.block_hadamard_rows(y, block_size=64)
    assert torch.allclose(x, x_rec, atol=1e-5)


def test_pack_aligned_stride() -> None:
    out_f, in_f = 8, 64
    scales = torch.ones(out_f, dtype=torch.float16)
    states = torch.randint(0, 65536, (out_f, in_f), dtype=torch.int32)
    for k in (2, 3, 4, 6):
        blob, stride = pack_per_row_aligned(scales, states, k=k, align_bytes=16)
        assert stride % 16 == 0
        assert len(blob) == out_f * stride


def test_ridge_refit_reduces_error() -> None:
    torch.manual_seed(42)
    out_f, in_f, n = 16, 32, 256
    w_true = torch.randn(out_f, in_f)
    w_noisy = w_true + 0.25 * torch.randn_like(w_true)
    z = torch.randn(n, in_f)
    y = z @ w_true.T
    h_zz = (z.T @ z) / n
    c_zy = (z.T @ y) / n
    w_refit = ridge_refit(w_noisy, h_zz, c_zy, lam_scale=1e-4, mu_scale=0.05)
    err_before = ((z @ w_noisy.T - y) ** 2).mean().item()
    err_after = ((z @ w_refit.T - y) ** 2).mean().item()
    assert err_after < err_before * 0.25


def test_kld_exactness() -> None:
    logits = torch.randn(32, 128)
    m_same = compute_kld_metrics(logits, logits)
    assert abs(m_same["mean_kld"]) < 1e-6
    assert m_same["top1_agreement"] == 1.0

    m_diff = compute_kld_metrics(logits, logits + 0.5 * torch.randn_like(logits))
    assert m_diff["mean_kld"] > 0.0
    assert 0.0 <= m_diff["top1_agreement"] <= 1.0


if __name__ == "__main__":
    test_largest_pow2_divisor()
    test_hadamard_orthogonality()
    test_pack_aligned_stride()
    test_ridge_refit_reduces_error()
    test_kld_exactness()
    print("All MS converter unit tests passed.")
