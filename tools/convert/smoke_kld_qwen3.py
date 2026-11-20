"""End-to-end BF16 -> real Hessian -> MS_C{k} -> exact KLD smoke test for dense Qwen3 models.

Runs on a single GPU (e.g. RTX 3090) for Qwen3-1.7B:
- Batched GPU Hessian collection (~2-9s)
- Default: compiled CUDA 14-bit 1MAD Viterbi kernel (`ms_viterbi_cuda.cu`)
- `--fast-greedy`: 1-step greedy trellis + LDLQ (~15-140s pure PyTorch)

Usage:
  python tools/convert/smoke_kld_qwen3.py \
      --model-dir G:\\llm\\models\\Qwen3-1.7B \
      --calib C:\\llm\\calib\\calib_2m.pt \
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
def greedy_trellis_code_matrix(
    w_rot: torch.Tensor,
    hu: torch.Tensor,
    k: int,
    lut: torch.Tensor,
) -> torch.Tensor:
    """Fast GPU 1-step greedy 1MAD trellis + LDLQ error feedback."""
    out_f, in_f = w_rot.shape
    device = w_rot.device
    w_work = w_rot.clone().to(torch.float32)

    row_std = w_work.std(dim=1).clamp(min=1e-8)
    scales = (ms.GAINS[k] * row_std / ms.CODEBOOK_STD).to(torch.float16).to(torch.float32)
    inv_scales = 1.0 / scales.clamp(min=1e-12)

    w_quant = torch.empty((out_f, in_f), dtype=torch.float32, device=device)
    hu_diag = torch.diagonal(hu)
    n_pred = 1 << k
    sym = torch.arange(n_pred, device=device, dtype=torch.int64).unsqueeze(0)

    target0 = (w_work[:, 0] * inv_scales).unsqueeze(1)
    state = torch.empty(out_f, dtype=torch.int64, device=device)
    for r0 in range(0, out_f, 512):
        r1 = min(r0 + 512, out_f)
        state[r0:r1] = (lut.unsqueeze(0) - target0[r0:r1]).abs().argmin(dim=1)

    q0 = lut[state] * scales
    w_quant[:, 0] = q0
    if in_f > 1:
        err = (w_work[:, 0] - q0) / hu_diag[0]
        w_work[:, 1:] -= err.unsqueeze(1) * hu[0, 1:].unsqueeze(0)

    for c in range(1, in_f):
        cand_states = ((state << k) & ms.STATE_MASK).unsqueeze(1) | sym
        cand_vals = lut[cand_states] * scales.unsqueeze(1)
        best_idx = (cand_vals - w_work[:, c].unsqueeze(1)).abs().argmin(dim=1)
        state = cand_states.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        qc = cand_vals.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        w_quant[:, c] = qc
        if c + 1 < in_f:
            err = (w_work[:, c] - qc) / hu_diag[c]
            w_work[:, c + 1 :] -= err.unsqueeze(1) * hu[c, c + 1 :].unsqueeze(0)

    return w_quant


@torch.no_grad()
def collect_logits(model: nn.Module, eval_ids: torch.Tensor, device: torch.device, batch_size: int = 4) -> torch.Tensor:
    outs = []
    for i in range(0, eval_ids.shape[0], batch_size):
        inp = eval_ids[i : i + batch_size].to(device=device, dtype=torch.long)
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
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for Hessian collection")
    parser.add_argument("--fast-greedy", action="store_true", help="Use 1-step greedy trellis instead of CUDA Viterbi")
    parser.add_argument("--quant-lm-head", action="store_true", help="Also quantize lm_head to MS_C6")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Loading {args.model_dir} (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    calib_all = torch.load(args.calib, map_location="cpu", weights_only=True)
    vocab_size = model.config.vocab_size
    calib_all = calib_all.clamp(0, vocab_size - 1)
    calib_ids = calib_all[: args.calib_seqs]
    eval_ids = calib_all[-args.eval_seqs :]

    viterbi_cuda_fn = None
    if not args.fast_greedy and device.type == "cuda":
        print("Compiling/loading CUDA 14-bit Viterbi kernel (ms_viterbi_cuda.cu)...", flush=True)
        from ms_viterbi_cuda import viterbi_code_matrix_cuda
        viterbi_cuda_fn = viterbi_code_matrix_cuda

    t0 = time.time()
    print(f"Evaluating BF16 reference logits on {args.eval_seqs} held-out sequences...", flush=True)
    ref_logits = collect_logits(model, eval_ids, device, batch_size=args.batch_size)

    hessians: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_mod, inp, _out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float32)
            h_batch = x.T @ x
            if name not in hessians:
                hessians[name] = h_batch
                counts[name] = x.shape[0]
            else:
                hessians[name] += h_batch
                counts[name] += x.shape[0]
        return hook

    target_linears = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and (args.quant_lm_head or name != "lm_head")
    ]
    for name, mod in target_linears:
        hooks.append(mod.register_forward_hook(make_hook(name)))

    print(f"Collecting Hessians across {args.calib_seqs} sequences (batch={args.batch_size})...", flush=True)
    for i in range(0, args.calib_seqs, args.batch_size):
        inp = calib_ids[i : i + args.batch_size].to(device=device, dtype=torch.long)
        model(input_ids=inp)
    for h in hooks:
        h.remove()
    torch.cuda.synchronize()
    print(f"Hessians collected in {time.time() - t0:.1f}s.", flush=True)

    lut = ms.build_1mad_lut(device)
    mode_str = "fast-greedy" if args.fast_greedy else ("viterbi-cuda" if viterbi_cuda_fn else "viterbi-py")
    print(f"Quantizing {len(target_linears)} linear layers to MS_C{args.k} ({mode_str})...", flush=True)
    for idx, (name, mod) in enumerate(target_linears):
        k_use = 6 if name == "lm_head" else args.k
        w_f32 = mod.weight.detach().to(device=device, dtype=torch.float32)
        h_f32 = hessians.pop(name) / max(counts[name], 1)
        blk = largest_pow2_divisor(int(w_f32.shape[1]), cap=512)
        seed = (idx + 1) * 10007

        w_rot, h_rot, signs = ms.rotate_weight_and_hessian(w_f32, h_f32, block_size=blk, seed=seed)
        hu = ms.cholesky_inv_upper(h_rot)

        if args.fast_greedy:
            w_rot_q = greedy_trellis_code_matrix(w_rot, hu, k=k_use, lut=lut)
        elif viterbi_cuda_fn is not None:
            _scales, _states, w_rot_q = viterbi_cuda_fn(w_rot, hu, k=k_use, lut=lut)
        else:
            _scales, _states, w_rot_q = ms.viterbi_code_matrix(w_rot, hu, k=k_use, lut=lut)

        w_q = ms.fwht_blocks(w_rot_q, block_size=blk) * signs.unsqueeze(0)
        mod.weight.data.copy_(w_q.to(dtype=mod.weight.dtype))

        if (idx + 1) == 1 or (idx + 1) % 28 == 0 or (idx + 1) == len(target_linears):
            rel_err = ((w_q - w_f32).norm() / w_f32.norm().clamp(min=1e-8)).item()
            print(f"  [{idx + 1}/{len(target_linears)}] {name} rel_err={rel_err:.4f} ({time.time() - t0:.1f}s)", flush=True)

    print("Evaluating quantized MS logits...", flush=True)
    cand_logits = collect_logits(model, eval_ids, device, batch_size=args.batch_size)
    metrics = compute_kld_metrics(ref_logits, cand_logits)
    print(
        f"\n=== Qwen3 MS_C{args.k} ({mode_str}) Smoke Result ===\n"
        f"Tokens:         {metrics['tokens']}\n"
        f"Mean KLD:       {metrics['mean_kld']:.5f}\n"
        f"Max KLD:        {metrics['max_kld']:.5f}\n"
        f"Top-1 Match:    {metrics['top1_agreement'] * 100:.2f}%\n"
        f"Elapsed:        {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
