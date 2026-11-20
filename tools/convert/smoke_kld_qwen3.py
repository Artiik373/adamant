"""End-to-end BF16 -> real Hessian -> MS_C{k} -> exact KLD smoke test for dense Qwen3 models.

Runs on a single GPU (e.g. RTX 3090) in ~2-3 minutes for Qwen3-1.7B:
1. Loads BF16 HF model + calibration tokens (calib_2m.pt or random fallback).
2. Evaluates BF16 reference logits on held-out sequences.
3. Collects real per-linear Hessians H = E[x x^T] via forward hooks.
4. Quantizes all linear layers (excluding lm_head unless --quant-lm-head) using
   Hadamard rotation + LDLQ/Viterbi (ms_moe_converter) and replaces weights in-place
   with the exact dequantized MS weights.
5. Computes exact full-vocab KLD and top-1 agreement vs BF16 via eval_ms_kld.py.

Usage:
  python tools/convert/smoke_kld_qwen3.py \
      --model-dir G:\\llm\\models\\Qwen3-1.7B \
      --calib G:\\llm\\calib\\calib_2m.pt \
      --k 4 --calib-seqs 64 --eval-seqs 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import ms_moe_converter as ms  # noqa: E402
from eval_ms_kld import compute_kld_metrics  # noqa: E402


def largest_pow2_divisor(n: int, cap: int = 512) -> int:
    b = 1
    while (b * 2) <= cap and (n % (b * 2) == 0):
        b *= 2
    return b


@torch.no_grad()
def collect_logits(model: nn.Module, eval_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    outs = []
    for i in range(eval_ids.shape[0]):
        inp = eval_ids[i : i + 1].to(device=device, dtype=torch.long)
        logits = model(input_ids=inp).logits.cpu().to(torch.float16)
        outs.append(logits)
    return torch.cat(outs, dim=0)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="E2E BF16 -> MS_C{k} real-Hessian KLD smoke test")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True, help="Path to calib_2m.pt")
    parser.add_argument("--k", type=int, default=4, choices=[2, 3, 4, 6])
    parser.add_argument("--calib-seqs", type=int, default=64, help="Number of 512-tok sequences for Hessian")
    parser.add_argument("--eval-seqs", type=int, default=8, help="Held-out 512-tok sequences for KLD")
    parser.add_argument("--quant-lm-head", action="store_true", help="Also quantize lm_head to MS_C6")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Loading {args.model_dir} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    calib_all = torch.load(args.calib, map_location="cpu", weights_only=True)
    vocab_size = model.config.vocab_size
    calib_all = calib_all.clamp(0, vocab_size - 1)
    calib_ids = calib_all[: args.calib_seqs]
    eval_ids = calib_all[-args.eval_seqs :]

    t0 = time.time()
    print(f"Evaluating BF16 reference logits on {args.eval_seqs} held-out sequences...")
    ref_logits = collect_logits(model, eval_ids, device)

    # Collect per-layer Hessians H = E[x x^T]
    hessians: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_mod, inp, _out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float32)
            h_batch = x.T @ x
            if name not in hessians:
                hessians[name] = h_batch.cpu()
                counts[name] = x.shape[0]
            else:
                hessians[name] += h_batch.cpu()
                counts[name] += x.shape[0]
        return hook

    target_linears = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and (args.quant_lm_head or name != "lm_head")
    ]
    for name, mod in target_linears:
        hooks.append(mod.register_forward_hook(make_hook(name)))

    print(f"Collecting Hessians across {args.calib_seqs} sequences ({calib_ids.numel():,} tokens)...")
    for i in range(args.calib_seqs):
        inp = calib_ids[i : i + 1].to(device=device, dtype=torch.long)
        model(input_ids=inp)
    for h in hooks:
        h.remove()

    lut = ms.build_1mad_lut(device)
    print(f"Quantizing {len(target_linears)} linear layers to MS_C{args.k} (lm_head=MS_C6 if enabled)...")
    for idx, (name, mod) in enumerate(target_linears):
        k_use = 6 if name == "lm_head" else args.k
        w_f32 = mod.weight.detach().to(device=device, dtype=torch.float32)
        h_f32 = (hessians[name] / max(counts[name], 1)).to(device=device, dtype=torch.float32)
        blk = largest_pow2_divisor(int(w_f32.shape[1]), cap=512)
        seed = (idx + 1) * 10007

        w_rot, h_rot, signs = ms.rotate_weight_and_hessian(w_f32, h_f32, block_size=blk, seed=seed)
        hu = ms.cholesky_inv_upper(h_rot)
        _scales, _states, w_rot_q = ms.viterbi_code_matrix(w_rot, hu, k=k_use, lut=lut)

        # Invert rotation: W_q = (W_rot_q @ H) * signs
        w_q = ms.block_hadamard_rows(w_rot_q, block_size=blk) * signs.unsqueeze(0)
        mod.weight.data.copy_(w_q.to(dtype=mod.weight.dtype))

        if (idx + 1) % 28 == 0 or (idx + 1) == len(target_linears):
            print(f"  [{idx + 1}/{len(target_linears)}] quantized ({time.time() - t0:.1f}s)")

    print("Evaluating quantized MS logits...")
    cand_logits = collect_logits(model, eval_ids, device)
    metrics = compute_kld_metrics(ref_logits, cand_logits)
    print(
        f"\n=== Qwen3 MS_C{args.k} Smoke Result ===\n"
        f"Tokens:         {metrics['tokens']}\n"
        f"Mean KLD:       {metrics['mean_kld']:.5f}\n"
        f"Max KLD:        {metrics['max_kld']:.5f}\n"
        f"Top-1 Match:    {metrics['top1_agreement'] * 100:.2f}%\n"
        f"Elapsed:        {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
