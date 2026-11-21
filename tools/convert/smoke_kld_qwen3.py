"""End-to-end BF16 -> real Hessian -> MS_C{k} -> exact KLD smoke test for dense Qwen3 models.

Runs on a single GPU (e.g. RTX 3090) for Qwen3-1.7B:
- Batched GPU Hessian collection (~2-9s)
- Default: compiled CUDA 14-bit 1MAD Viterbi kernel (`ms_viterbi_cuda.cu`)
- `--fast-greedy`: 1-step greedy trellis + LDLQ (~15-140s pure PyTorch)
- `--gguf FILE`: measure a GGUF's linear layers instead (same tokens, same BF16 reference)

Usage:
  python tools/convert/smoke_kld_qwen3.py \
      --model-dir G:\\llm\\models\\Qwen3-1.7B \
      --calib C:\\llm\\calib\\calib_2m.pt \
      --k 4 --calib-seqs 64 --eval-seqs 8
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import ms_moe_converter as ms  # noqa: E402
from eval_ms_kld import compute_kld_metrics, split_calib  # noqa: E402


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


GGUF_NAMES = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
}


def load_gguf_linears(model: nn.Module, path: Path) -> str:
    """Replace each decoder-layer linear weight with its dequantized GGUF tensor; returns the type mix and bpw."""
    import gguf
    from gguf.quants import dequantize

    mods = dict(model.named_modules())
    bits = n = 0
    types: dict[str, int] = {}
    for t in gguf.GGUFReader(path).tensors:
        parts = t.name.split(".")
        if parts[0] != "blk" or parts[2] not in GGUF_NAMES:
            continue
        mod = mods[f"model.layers.{parts[1]}.{GGUF_NAMES[parts[2]]}"]
        w = torch.from_numpy(dequantize(t.data, t.tensor_type).copy())
        assert w.shape == mod.weight.shape, t.name
        mod.weight.data.copy_(w.to(mod.weight.dtype))
        types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + 1
        bits += t.n_bytes * 8
        n += w.numel()
    assert sum(types.values()) == len(GGUF_NAMES) * model.config.num_hidden_layers, types
    return f"{types}, {bits / n:.3f} bpw"


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
    parser.add_argument("--gguf", type=Path, help="Measure this GGUF's decoder linears instead of quantizing")
    parser.add_argument(
        "--calib-mode", choices=["fp", "guided", "kron-emp", "kron-true", "swiglu"], default="fp",
        help="fp: Hessians from the BF16 model; "
        "guided: GuidedQuant, x x^T weighted by the squared loss gradient of each output-row group; "
        "kron-emp / kron-true: two-sided rounding on H = E[g g^T] (x) E[x x^T], g from data labels (empirical Fisher) "
        "or from labels sampled from the model (true Fisher, the Hessian of the KL); "
        "swiglu: as fp, but each group of gate / up rows weights a token's x x^T by its rows' local output "
        "sensitivity through SwiGLU and down (up: silu(g) |W_down[:, r]|, gate: u silu'(g) |W_down[:, r]|), no gradients",
    )
    parser.add_argument("--guided-groups", type=int, default=4, help="output-row groups per layer for --calib-mode guided")
    parser.add_argument("--calib-tokenizer", type=Path,
                        help="the tokenizer --calib was built with, when it is not the model's: re-tokenized as text")
    parser.add_argument("--spine-k", type=int, default=None, help="bits of the non-expert linears (default --k)")
    parser.add_argument("--hess-tau", type=float, default=0.0,
                        help="MoE: shrink each expert's Hessians toward the layer's with this many tokens' weight")
    parser.add_argument("--hess-affinity", action="store_true",
                        help="MoE: weight each routed token in its expert's Hessians by its router score squared")
    parser.add_argument("--sign-trials", type=int, default=1,
                        help="per matrix, keep the best of this many sign vectors (smallest rotated fourth moment)")
    parser.add_argument("--window", type=int, default=32, help="Viterbi search window (columns)")
    parser.add_argument("--commit", type=int, default=8, help="columns committed per window")
    parser.add_argument("--ls-scales", action="store_true",
                        help="after each matrix's search, refit every row's scale by least squares under its Hessian")
    parser.add_argument("--tune-scales", type=int, default=0, metavar="STEPS",
                        help="after coding, train each coded row's scale (codes fixed: the format's fp16 row scale) on "
                        "KL(BF16 || quantized) over the calibration sequences for STEPS optimizer steps")
    parser.add_argument("--tune-lr", type=float, default=5e-4, help="Adam learning rate of the row-scale factors")
    parser.add_argument("--tune-norms", action="store_true", help="with --tune-scales: train the RMSNorm weights too")
    parser.add_argument("--tune-cache", action="store_true",
                        help="with --tune-scales: distill from the calibration logits cached before quantizing "
                        "(no second model copy: MoE models)")
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
    teacher = None
    if args.tune_scales and not args.tune_cache:   # the BF16 model the tuning distills from
        teacher = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16, device_map=device,
                                                       trust_remote_code=True).eval()
        teacher.requires_grad_(False)

    calib_all = torch.load(args.calib, map_location="cpu", weights_only=True)
    vocab_size = model.config.vocab_size
    calib_all = calib_all.clamp(0, vocab_size - 1)
    if args.calib_tokenizer:   # decode with the pool's tokenizer, encode with the model's, repack into windows
        from transformers import AutoTokenizer
        src, dst = AutoTokenizer.from_pretrained(args.calib_tokenizer), AutoTokenizer.from_pretrained(args.model_dir)
        n_len = calib_all.shape[1]
        rows, buf = [], []
        for r in calib_all.tolist():
            buf += dst.encode(src.decode(r), add_special_tokens=False)
            while len(buf) >= n_len:
                rows.append(buf[:n_len])
                del buf[:n_len]
        calib_all = torch.tensor(rows, dtype=torch.long)
        print(f"re-tokenized the pool: {calib_all.shape[0]} windows of {n_len}", flush=True)
    calib_ids, eval_ids = split_calib(calib_all, args.calib_seqs, args.eval_seqs)

    viterbi_cuda_fn = None
    if not args.fast_greedy and device.type == "cuda":
        print("Compiling/loading CUDA 14-bit Viterbi kernel (ms_viterbi_cuda.cu)...", flush=True)
        from ms_viterbi_cuda import viterbi_code_matrix_cuda
        viterbi_cuda_fn = viterbi_code_matrix_cuda

    t0 = time.time()
    print(f"Evaluating BF16 reference logits on {args.eval_seqs} held-out sequences...", flush=True)
    ref_logits = collect_logits(model, eval_ids, device, batch_size=args.batch_size)
    if args.tune_scales and args.tune_cache:   # the teacher's calibration logits, before the model is quantized
        teacher = [model(input_ids=calib_ids[i:i + 1].to(device=device, dtype=torch.long)).logits.to("cpu", torch.float16)
                   for i in range(calib_ids.shape[0])]

    if args.gguf:
        print(f"Loading decoder linears from {args.gguf}: {load_gguf_linears(model, args.gguf)}", flush=True)
        report(f"GGUF {args.gguf.name}", ref_logits, collect_logits(model, eval_ids, device, args.batch_size), t0)
        return

    target_linears = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and (args.quant_lm_head or name != "lm_head")
    ]
    lut = ms.build_1mad_lut(device)
    mode_str = "fast-greedy" if args.fast_greedy else ("viterbi-cuda" if viterbi_cuda_fn else "viterbi-py")

    def quantize_linear(idx: int, name: str, mod: nn.Linear, w_target: torch.Tensor, h_f32: torch.Tensor) -> None:
        k_use = 6 if name == "lm_head" else (args.spine_k or args.k)
        w_f32 = mod.weight.detach().to(device=device, dtype=torch.float32)
        blk = largest_pow2_divisor(int(w_f32.shape[1]), cap=512)
        seed = (idx + 1) * 10007
        if args.sign_trials > 1:
            seed = ms.best_sign_seed(w_target.float(), blk, seed, args.sign_trials)

        # h_f32 [in, in], or [groups, in, in] with one Hessian per contiguous block of output rows
        hs = h_f32 if h_f32.dim() == 3 else h_f32.unsqueeze(0)
        n_g = hs.shape[0]
        w_g = w_target.reshape(n_g, -1, w_target.shape[1])
        rot = [ms.rotate_weight_and_hessian(w_g[j], hs[j], block_size=blk, seed=seed) for j in range(n_g)]
        signs = rot[0][2]  # same seed: every group shares one rotation
        hu = torch.stack([ms.cholesky_inv_upper(r[1]) for r in rot])
        w_rot = torch.stack([r[0] for r in rot])
        if viterbi_cuda_fn is not None and not args.fast_greedy:
            from ms_viterbi_cuda import viterbi_code_batched_cuda

            w_rot_q = viterbi_code_batched_cuda(w_rot, hu, k_use, lut, window=args.window, commit=args.commit)[2]
        elif args.fast_greedy:
            w_rot_q = torch.stack([greedy_trellis_code_matrix(w_rot[j], hu[j], k=k_use, lut=lut) for j in range(n_g)])
        else:
            w_rot_q = torch.stack([ms.viterbi_code_matrix(w_rot[j], hu[j], k=k_use, lut=lut)[2] for j in range(n_g)])
        if args.ls_scales:   # each row's scale refit: min (w - a q) H (w - a q)^T, a = <w, q>_H / <q, q>_H
            qh = torch.einsum("gro,gpo->grp", w_rot_q, torch.stack([r[1] for r in rot]))
            a = (qh * w_rot).sum(-1) / (qh * w_rot_q).sum(-1).clamp(min=1e-20)
            w_rot_q = w_rot_q * a.unsqueeze(-1)
        w_q = ms.fwht_blocks(w_rot_q.reshape(w_target.shape), block_size=blk) * signs.unsqueeze(0)
        mod.weight.data.copy_(w_q.to(dtype=mod.weight.dtype))

        if (idx + 1) == 1 or (idx + 1) % 28 == 0 or (idx + 1) == len(target_linears):
            rel_err = ((w_q - w_f32).norm() / w_f32.norm().clamp(min=1e-8)).item()
            print(f"  [{idx + 1}/{len(target_linears)}] {name} rel_err={rel_err:.4f} ({time.time() - t0:.1f}s)", flush=True)

    def quantize_two_sided_many(items, rows: int = 128):
        """LDLQ on H = B (x) A for many matrices at once. Rows go in order: each row is searched with A, and a
        finished block of rows feeds (Ub[i, i'] / Ub[i, i]) (w_i - q_i) to every later row i'. Rows inside one
        block ignore B. Step r searches block r of every matrix with the same input width in one batched call."""
        from ms_viterbi_cuda import viterbi_code_batched_cuda

        jobs = []
        for idx, name, mod, w_target, (a, b) in items:
            blk = largest_pow2_divisor(int(w_target.shape[1]), cap=512)
            w_rot, a_rot, signs = ms.rotate_weight_and_hessian(w_target, a, block_size=blk, seed=(idx + 1) * 10007)
            jobs.append({"idx": idx, "name": name, "mod": mod, "blk": blk, "signs": signs, "w": w_rot,
                         "q": torch.empty_like(w_rot), "ua": ms.cholesky_inv_upper(a_rot), "ub": ms.cholesky_inv_upper(b)})
        for in_f in sorted({j["w"].shape[1] for j in jobs}):
            group = sorted((j for j in jobs if j["w"].shape[1] == in_f), key=lambda j: -j["w"].shape[0])
            assert all(j["w"].shape[0] % rows == 0 for j in group)
            hu = torch.stack([j["ua"] for j in group])
            for r0 in range(0, group[0]["w"].shape[0], rows):
                act = [j for j in group if j["w"].shape[0] > r0]  # a prefix: the group is sorted by rows
                r1 = r0 + rows
                q = viterbi_code_batched_cuda(torch.stack([j["w"][r0:r1] for j in act]), hu[: len(act)], args.k, lut)[2]
                for j, qj in zip(act, q):
                    j["q"][r0:r1] = qj
                    if r1 < j["w"].shape[0]:
                        coef = j["ub"][r0:r1, r1:] / torch.diagonal(j["ub"])[r0:r1, None]
                        j["w"][r1:] -= coef.T @ (j["w"][r0:r1] - qj)
            del hu
        for j in jobs:
            w_q = ms.fwht_blocks(j["q"], block_size=j["blk"]) * j["signs"].unsqueeze(0)
            j["mod"].weight.data.copy_(w_q.to(dtype=j["mod"].weight.dtype))
            if (j["idx"] + 1) % 28 == 0:
                w0 = items[[i[0] for i in items].index(j["idx"])][3]
                print(f"  [{j['idx'] + 1}/{len(target_linears)}] {j['name']} rel_err="
                      f"{((w_q - w0).norm() / w0.norm()).item():.4f} ({time.time() - t0:.1f}s)", flush=True)

    def finish(label: str) -> None:
        report(label, ref_logits, collect_logits(model, eval_ids, device, args.batch_size), t0)
        if args.tune_scales:
            tune_scales(model, teacher, target_linears, calib_ids, device, args)
            report(f"{label} + tuned row scales{' and norms' if args.tune_norms else ''}", ref_logits,
                   collect_logits(model, eval_ids, device, args.batch_size), t0)

    if args.calib_mode not in ("fp", "swiglu"):
        quantize_guided(model, target_linears, quantize_linear, quantize_two_sided_many, calib_ids, device, args)
        finish(f"MS_C{args.k} ({mode_str}, {args.calib_mode})")
        return

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

    swiglu: dict[str, torch.Tensor] = {}   # gate / up name -> [groups, in, in]

    def make_mlp_hook(prefix: str, mlp):
        n_g = args.guided_groups
        nd = mlp.down_proj.weight.detach().float().norm(dim=0)   # |W_down[:, r]| of each intermediate row

        def hook(_mod, inp, _out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            g = x @ mlp.gate_proj.weight.detach().float().T
            u = x @ mlp.up_proj.weight.detach().float().T
            sg = torch.sigmoid(g)
            for nm, c in ((f"{prefix}.up_proj", g * sg * nd), (f"{prefix}.gate_proj", u * sg * (1 + g * (1 - sg)) * nd)):
                wt = (c * c).reshape(c.shape[0], n_g, -1).mean(-1)   # [tokens, groups]
                h = torch.stack([(x * wt[:, j:j + 1]).T @ x for j in range(n_g)])
                swiglu[nm] = swiglu[nm] + h if nm in swiglu else h
        return hook

    # MoE blocks (fused experts, as Qwen4Exp's): their inputs kept on the host, the experts coded after the pass
    moe_blocks = [(n, m) for n, m in model.named_modules() if hasattr(m, "experts") and hasattr(m, "gate")
                  and hasattr(getattr(m, "experts"), "gate_up_proj")]
    moe_x: dict[str, list] = {n: [] for n, _ in moe_blocks}

    def make_moe_hook(name: str):
        def hook(_mod, inp, _out):
            moe_x[name].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).to("cpu"))
        return hook

    for name, mod in moe_blocks:
        hooks.append(mod.register_forward_hook(make_moe_hook(name)))
    for name, mod in target_linears:
        if args.calib_mode == "swiglu" and name.endswith((".gate_proj", ".up_proj")):
            continue
        hooks.append(mod.register_forward_hook(make_hook(name)))
    if args.calib_mode == "swiglu":
        for name, mod in model.named_modules():
            if name.endswith(".mlp") and hasattr(mod, "gate_proj"):
                hooks.append(mod.register_forward_hook(make_mlp_hook(name, mod)))

    print(f"Collecting Hessians across {args.calib_seqs} sequences (batch={args.batch_size})...", flush=True)
    for i in range(0, args.calib_seqs, args.batch_size):
        inp = calib_ids[i : i + args.batch_size].to(device=device, dtype=torch.long)
        model(input_ids=inp)
    for h in hooks:
        h.remove()
    torch.cuda.synchronize()
    print(f"Hessians collected in {time.time() - t0:.1f}s.", flush=True)

    print(f"Quantizing {len(target_linears)} linear layers to MS_C{args.k} ({mode_str})...", flush=True)
    for idx, (name, mod) in enumerate(target_linears):
        w_f32 = mod.weight.detach().to(device=device, dtype=torch.float32)
        h_l = swiglu.pop(name) if name in swiglu else hessians.pop(name) / max(counts[name], 1)
        quantize_linear(idx, name, mod, w_f32, h_l)

    if moe_blocks:
        from ms_flashnext_convert import code, expert_hessians
        for li, (name, blk) in enumerate(moe_blocks):
            x = torch.cat(moe_x.pop(name)).to(device)
            _, wts, idx = blk.gate(x)
            ex = blk.experts
            h_all = x_ref = None
            if args.hess_tau > 0:
                xf = x.float()
                h_all = xf.T @ xf / xf.shape[0]
                x_ref = x[torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(li))[:2048].to(device)]
            zero = 0
            for e0 in range(0, ex.gate_up_proj.shape[0], 16):
                es = list(range(e0, min(e0 + 16, ex.gate_up_proj.shape[0])))
                hs = [expert_hessians(x, idx, e, ex.gate_up_proj[e].float(), wts if args.hess_affinity else None,
                                      args.hess_tau, h_all, x_ref) for e in es]
                zero += sum(1 for h in hs if h[2] == 0)
                gu = code(ex.gate_up_proj[es].float(), torch.stack([h[0] for h in hs]), args.k, lut,
                          [li * 4096 + 2 * e + 1 for e in es])
                dn = code(ex.down_proj[es].float(), torch.stack([h[1] for h in hs]), args.k, lut,
                          [li * 4096 + 2 * e + 2 for e in es])
                for i, e in enumerate(es):
                    ex.gate_up_proj.data[e].copy_(gu[i][2].to(ex.gate_up_proj.dtype))
                    ex.down_proj.data[e].copy_(dn[i][2].to(ex.down_proj.dtype))
            del x
            print(f"  experts of {name}: MS_C{args.k}, zero-token {zero} ({time.time() - t0:.1f}s)", flush=True)

    print("Evaluating quantized MS logits...", flush=True)
    finish(f"MS_C{args.k} ({mode_str})")


def _scaled_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """The fused experts' forward (OlmoeExperts / Qwen4ExpTextExperts) with each expert's row factors ms_a_gu, ms_a_d
    on its gate/up and down output channels: the same as scaling the rows, without a copy of the weights."""
    out = torch.zeros_like(hidden_states)
    mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
    for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()[:, 0].tolist():
        if e == self.num_experts:
            continue
        pos, tok = torch.where(mask[e])
        gu = nn.functional.linear(hidden_states[tok], self.gate_up_proj[e]) * self.ms_a_gu[e].to(hidden_states.dtype)
        g, u = gu.chunk(2, dim=-1)
        y = nn.functional.linear(self.act_fn(g) * u, self.down_proj[e]) * self.ms_a_d[e].to(hidden_states.dtype)
        out.index_add_(0, tok, (y * top_k_weights[tok, pos, None]).to(out.dtype))
    return out


class _RowScale(nn.Module):
    """W -> alpha[:, None] * W: a coded row times a factor, which the format stores in its fp16 row scale."""

    def __init__(self, rows: int, device):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(rows, device=device, dtype=torch.float32))

    def forward(self, w):
        return w * self.alpha.to(w.dtype)[:, None]


@torch.enable_grad()
def tune_scales(model, teacher, target_linears, calib_ids, device, args) -> None:
    """Codes fixed, row scales (and with --tune-norms the RMSNorm weights) trained on KL(teacher || model) over the
    calibration sequences, one sequence a step, cycling; the factors are then folded into the weights."""
    import torch.nn.utils.parametrize as P

    model.requires_grad_(False)
    params = []
    for _name, mod in target_linears:
        P.register_parametrization(mod, "weight", _RowScale(mod.weight.shape[0], device))
        params.append(mod.parametrizations.weight[0].alpha)
    fused = [m for m in model.modules() if hasattr(m, "gate_up_proj") and hasattr(m, "down_proj")
             and isinstance(getattr(m, "gate_up_proj"), nn.Parameter)]
    for m in fused:   # each expert's rows: the format's row scales, applied to its output channels while training
        m.ms_a_gu = nn.Parameter(torch.ones(m.gate_up_proj.shape[:2], device=device))
        m.ms_a_d = nn.Parameter(torch.ones(m.down_proj.shape[:2], device=device))
        params += [m.ms_a_gu, m.ms_a_d]
        m.forward = _scaled_experts_forward.__get__(m)
    norm_params = []
    if args.tune_norms:
        for name, mod in model.named_modules():
            if "norm" in type(mod).__name__.lower() and getattr(mod, "weight", None) is not None:
                mod.weight.requires_grad_(True)
                norm_params.append(mod.weight)
    groups = [{"params": params, "lr": args.tune_lr}]
    if norm_params:
        groups.append({"params": norm_params, "lr": args.tune_lr * 0.2})
    opt = torch.optim.Adam(groups)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    t0 = time.time()
    run = 0.0
    for step in range(args.tune_scales):
        ids = calib_ids[step % calib_ids.shape[0]: step % calib_ids.shape[0] + 1].to(device=device, dtype=torch.long)
        with torch.no_grad():
            t_lg = teacher[step % len(teacher)].to(device) if isinstance(teacher, list) else teacher(input_ids=ids).logits
            t_lp = torch.log_softmax(t_lg.float(), dim=-1)
        s_lp = torch.log_softmax(model(input_ids=ids).logits.float(), dim=-1)
        loss = (t_lp.exp() * (t_lp - s_lp)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run = loss.item() if step == 0 else 0.95 * run + 0.05 * loss.item()
        if (step + 1) % 50 == 0 or step + 1 == args.tune_scales:
            a = torch.cat([p.detach().reshape(-1) for p in params])
            print(f"  tune {step + 1}/{args.tune_scales}: KL {run:.5f}, factor range [{a.min().item():.4f}, "
                  f"{a.max().item():.4f}] ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    model.gradient_checkpointing_disable()
    for _name, mod in target_linears:
        P.remove_parametrizations(mod, "weight", leave_parametrized=True)
    with torch.no_grad():
        for m in fused:   # the factors folded into the rows
            m.gate_up_proj.mul_(m.ms_a_gu.to(m.gate_up_proj.dtype)[..., None])
            m.down_proj.mul_(m.ms_a_d.to(m.down_proj.dtype)[..., None])
            del m.ms_a_gu, m.ms_a_d, m.forward   # the class forward again
    for w in norm_params:
        w.requires_grad_(False)


def quantize_guided(model, target_linears, quantize_linear, quantize_many, calib_ids, device, args, chunk_layers: int = 7) -> None:
    """Gradient-weighted Hessians from the BF16 model, a few layers at a time. g is the gradient of the next-token
    loss with respect to the layer output. guided (GuidedQuant): per output-row group j, H_j = sum_t (sum_{i in j}
    g_ti^2) x_t x_t^T. kron-*: the pair (A, B) = (sum x x^T, sum g g^T); kron-true samples the labels from the
    model, so E[g g^T] is the Fisher of the model's own distribution."""
    kron = args.calib_mode.startswith("kron")
    grad_model = copy.deepcopy(model)
    grad_model.requires_grad_(False)
    gmods = dict(grad_model.named_modules())
    n_groups = args.guided_groups
    emb_hook = grad_model.get_input_embeddings().register_forward_hook(lambda _m, _i, out: out.requires_grad_(True))

    def layer_of(name: str) -> int:
        return int(name.split(".")[2]) if name.startswith("model.layers.") else model.config.num_hidden_layers

    for c in range(0, model.config.num_hidden_layers + 1, chunk_layers):
        chunk = [(i, n, m) for i, (n, m) in enumerate(target_linears) if c <= layer_of(n) < c + chunk_layers]
        if not chunk:
            continue
        hess: dict[str, torch.Tensor] = {}
        hooks = []
        for _, name, mod in chunk:
            def fwd(_m, inp, out, name=name):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()

                def bwd_kron(g):
                    g = g.reshape(x.shape[0], -1).float()
                    a, b = x.T @ x, g.T @ g
                    if name in hess:
                        hess[name][0].add_(a)
                        hess[name][1].add_(b)
                    else:
                        hess[name] = (a, b)

                if kron:
                    out.register_hook(bwd_kron)
                    return

                def bwd(g):
                    gw = g.reshape(x.shape[0], n_groups, -1).float().pow(2).sum(-1)  # [tokens, groups]
                    xg = x.unsqueeze(0) * gw.T.sqrt().unsqueeze(-1)  # [groups, tokens, in]
                    h = torch.bmm(xg.transpose(1, 2), xg)
                    if name in hess:
                        hess[name] += h
                    else:
                        hess[name] = h

                out.register_hook(bwd)

            hooks.append(gmods[name].register_forward_hook(fwd))
        t0 = time.time()
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True  # Hessian sums only: TF32 inputs, FP32 accumulation
        with torch.enable_grad():
            for i in range(0, calib_ids.shape[0], args.batch_size):
                inp = calib_ids[i : i + args.batch_size].to(device=device, dtype=torch.long)
                if args.calib_mode == "kron-true":
                    logits = grad_model(input_ids=inp).logits.float()
                    y = torch.distributions.Categorical(logits=logits.detach()).sample()
                    F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).backward()
                    del logits
                else:
                    grad_model(input_ids=inp, labels=inp).loss.backward()
        torch.backends.cuda.matmul.allow_tf32 = tf32
        for hk in hooks:
            hk.remove()
        torch.cuda.synchronize()
        t1 = time.time()
        if kron:
            quantize_many([(idx, name, mod, mod.weight.detach().float(), hess.pop(name)) for idx, name, mod in chunk])
        for idx, name, mod in chunk if not kron else []:
            quantize_linear(idx, name, mod, mod.weight.detach().float(), hess.pop(name))
        torch.cuda.synchronize()
        print(f"  layers {c}-{c + chunk_layers - 1}: gradients+Hessians {t1 - t0:.0f}s, quantize {time.time() - t1:.0f}s", flush=True)
    emb_hook.remove()
    del grad_model


def report(label: str, ref_logits: torch.Tensor, cand_logits: torch.Tensor, t0: float) -> None:
    metrics = compute_kld_metrics(ref_logits, cand_logits)
    print(
        f"\n=== Qwen3 {label} Smoke Result ===\n"
        f"Tokens:         {metrics['tokens']}\n"
        f"Mean KLD:       {metrics['mean_kld']:.5f}\n"
        f"Max KLD:        {metrics['max_kld']:.5f}\n"
        f"Top-1 Match:    {metrics['top1_agreement'] * 100:.2f}%\n"
        f"Elapsed:        {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
