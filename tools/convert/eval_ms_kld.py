"""Teacher-forced full-vocab KL divergence (KLD) evaluator for MS vs BF16 / WV-v5 / EXL3.

Computes exact mean KLD = E[sum_v p_ref(v) * log(p_ref(v) / p_cand(v))] and top-1 agreement
between saved BF16 reference logits and candidate model logits over held-out evaluation windows.

Usage:
  python tools/convert/eval_ms_kld.py --ref-logits ref_bf16.pt --cand-logits cand_ms.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def compute_kld_metrics(ref_logits: torch.Tensor, cand_logits: torch.Tensor, chunk_tokens: int = 256) -> dict:
    """Compute exact full-vocab KLD, max KLD, and top-1 match rate in fp32 chunks."""
    ref_flat = ref_logits.reshape(-1, ref_logits.shape[-1])
    cand_flat = cand_logits.reshape(-1, cand_logits.shape[-1])
    assert ref_flat.shape == cand_flat.shape, f"Shape mismatch: {ref_flat.shape} vs {cand_flat.shape}"

    n_tok = ref_flat.shape[0]
    kld_sum = 0.0
    kld_max = 0.0
    top1_match = 0

    for start in range(0, n_tok, chunk_tokens):
        end = min(start + chunk_tokens, n_tok)
        r = ref_flat[start:end].to(torch.float32)
        c = cand_flat[start:end].to(torch.float32)

        log_p = F.log_softmax(r, dim=-1)
        log_q = F.log_softmax(c, dim=-1)
        p = log_p.exp()

        tok_kld = (p * (log_p - log_q)).sum(dim=-1)
        kld_sum += float(tok_kld.sum().item())
        kld_max = max(kld_max, float(tok_kld.max().item()))
        top1_match += int((r.argmax(dim=-1) == c.argmax(dim=-1)).sum().item())

    return {
        "tokens": n_tok,
        "mean_kld": kld_sum / max(n_tok, 1),
        "max_kld": kld_max,
        "top1_agreement": top1_match / max(n_tok, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate teacher-forced full-vocab KLD vs BF16")
    parser.add_argument("--ref-logits", type=Path, required=True, help="BF16 reference logits .pt")
    parser.add_argument("--cand-logits", type=Path, required=True, help="Quantized candidate logits .pt")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    ref = torch.load(args.ref_logits, map_location="cpu", weights_only=True)
    cand = torch.load(args.cand_logits, map_location="cpu", weights_only=True)
    metrics = compute_kld_metrics(ref, cand)

    print(
        f"Tokens: {metrics['tokens']} | Mean KLD: {metrics['mean_kld']:.5f} | "
        f"Max KLD: {metrics['max_kld']:.5f} | Top-1 Match: {metrics['top1_agreement'] * 100:.2f}%"
    )
    if args.out_json:
        args.out_json.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
