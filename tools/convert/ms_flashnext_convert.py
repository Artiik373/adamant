"""Convert Qwen3.8-Flash-Next to MS (1MAD trellis) records, one BF16 layer at a time.

Streams each decoder layer through the calibration tokens on the BF16 reference chain, collects the Hessian of every
coded matrix from its real inputs, and codes:

- the dense spine (GDN attn_qkv / attn_gate / ssm_out, QSA attn_q / k / v / output, shared-expert gate / up / down) at
  `--spine-k`, as 2D records for ms_pack.py (`<out>/<gguf name>.json` + payload);
- the routed experts at the layer's width from `--plan`, as `<out>/experts/blk.<l>.ffn_{gate,up,down}_exps.ms`: per
  expert, the 2D payload of ms_pack.py (rows, then int8 signs), experts in order; gate and up share their signs;
- the output head at `--head-k` on the final hyper-connection mix.

Resumes after the last finished layer. The router, norms, small mixing matrices, the token table and the PLE n-gram
table are not coded here; the base pack keeps them.

    python tools/convert/ms_flashnext_convert.py --hf-dir C:/llm/models/Qwen3.8-Flash-Next \
        --calib G:/llm/calib/calib_2m.pt --plan plan.json --out-dir G:/llm/ms_fn
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

import ms_moe_converter as ms
from eval_ms_kld import split_calib
from ms_convert_dense_spine import largest_pow2_divisor, to_gguf_order
from ms_pack import ms_row_bytes, write_record
from ms_sensitivity_scan import PRE, load_layer, prelude, run_layer
from ms_viterbi_cuda import viterbi_code_batched_cuda

SPINE = {  # HF module -> GGUF tensor suffix
    "linear_attn.in_proj_qkv": "attn_qkv.weight",
    "linear_attn.in_proj_z": "attn_gate.weight",
    "linear_attn.out_proj": "ssm_out.weight",
    "self_attn.q_proj": "attn_q.weight",
    "self_attn.k_proj": "attn_k.weight",
    "self_attn.v_proj": "attn_v.weight",
    "self_attn.o_proj": "attn_output.weight",
    "mlp.shared_expert.gate_proj": "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj": "ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj": "ffn_down_shexp.weight",
}


def pack_rows(states: torch.Tensor, scales: torch.Tensor, k: int) -> bytes:
    """MS rows (ms_pack.py format): fp16 scale, uint16 start state, MSB-first k-bit symbols of states[:, 1:], zero pad."""
    rows, n = states.shape
    rb = ms_row_bytes(n, k)
    sym = (states[:, 1:] & ((1 << k) - 1)).to(torch.uint8)
    bits = (sym.unsqueeze(-1) >> torch.arange(k - 1, -1, -1, device=sym.device, dtype=torch.uint8)) & 1
    bits = bits.reshape(rows, -1)
    nbytes = (bits.shape[1] + 7) // 8
    bits = torch.nn.functional.pad(bits, (0, nbytes * 8 - bits.shape[1])).reshape(rows, nbytes, 8)
    packed = (bits << torch.arange(7, -1, -1, device=sym.device, dtype=torch.uint8)).sum(-1, dtype=torch.uint8)
    out = torch.zeros(rows, rb, dtype=torch.uint8, device=sym.device)
    out[:, 0:2] = scales.to(torch.float16).view(torch.uint8).reshape(rows, 2)
    out[:, 2:4] = (states[:, 0] & ms.STATE_MASK).to(torch.int16).view(torch.uint8).reshape(rows, 2)
    out[:, 4 : 4 + nbytes] = packed
    return out.cpu().numpy().tobytes()


def code(ws: torch.Tensor, hs: torch.Tensor, k: int, lut, seeds) -> list:
    """Code E matrices [E, out, in] on Hessians [E, in, in]: (rows bytes, int8 signs, decoded W) per matrix."""
    blk = largest_pow2_divisor(int(ws.shape[2]), cap=512)
    rot = [ms.rotate_weight_and_hessian(w, h, block_size=blk, seed=s) for w, h, s in zip(ws, hs, seeds)]
    hu = torch.stack([ms.cholesky_inv_upper(r[1]) for r in rot])
    scales, states, q = viterbi_code_batched_cuda(torch.stack([r[0] for r in rot]), hu, k, lut)
    out = []
    for i, r in enumerate(rot):
        signs = r[2]
        deq = ms.fwht_blocks(q[i], block_size=blk) * signs.unsqueeze(0)
        out.append((pack_rows(states[i], scales[i], k), signs.to(torch.int8).cpu().numpy(), deq))
    return out


def code_rows(w: torch.Tensor, h: torch.Tensor, k: int, lut, seed: int, chunk: int = 16384):
    """One large matrix (the head) coded in row chunks on one Hessian; same rotation for every chunk."""
    blk = largest_pow2_divisor(int(w.shape[1]), cap=512)
    _, h_rot, signs = ms.rotate_weight_and_hessian(w[:1], h, block_size=blk, seed=seed)
    hu = ms.cholesky_inv_upper(h_rot).unsqueeze(0)
    parts = []
    for r0 in range(0, w.shape[0], chunk):
        w_rot = ms.fwht_blocks(w[r0 : r0 + chunk].float() * signs.unsqueeze(0), blk)
        scales, states, _ = viterbi_code_batched_cuda(w_rot.unsqueeze(0), hu, k, lut)
        parts.append(pack_rows(states[0], scales[0], k))
    return b"".join(parts), signs.to(torch.int8).cpu().numpy()


def expert_hessians(x: torch.Tensor, idx: torch.Tensor, e: int, w_gu: torch.Tensor, wts=None, tau: float = 0.0,
                    h_all=None, x_ref=None):
    """Input Hessians of expert e's gate/up and down rows and its routed-token count.  Plain: sums over the tokens
    routed to it (identity Hessians when none is).  wts (the router's top-k scores): each token weighted by its score
    squared, as its error reaches the layer output times the score.  tau > 0: each Hessian, as a mean, shrunk toward
    the layer's (gate/up: h_all, every token's x x^T mean; down: the expert's activations on the x_ref tokens) with
    tau tokens' weight, so a rarely routed expert keeps the layer's input statistics instead of a few tokens' or none."""
    hit = idx == e
    rows = hit.any(-1)
    xe = x[rows].float()
    n = xe.shape[0]
    if tau <= 0 and not n:
        return torch.eye(x.shape[1], device=x.device), torch.eye(w_gu.shape[0] // 2, device=x.device), 0
    p = (wts.float()[rows] * hit[rows]).sum(-1) ** 2 if wts is not None else torch.ones(n, device=x.device)
    g, u = (xe @ w_gu.T).chunk(2, -1)
    a = torch.nn.functional.silu(g) * u
    s_in, s_dn, m = (xe * p[:, None]).T @ xe, (a * p[:, None]).T @ a, p.sum()
    if tau <= 0:
        return s_in, s_dn, n
    gr, ur = (x_ref.float() @ w_gu.T).chunk(2, -1)
    ar = torch.nn.functional.silu(gr) * ur
    hd_all = ar.T @ ar / ar.shape[0]
    # tau weighs the prior as tau tokens of the mean routed token's weight
    w_mean = m / n if n else torch.tensor(1.0, device=x.device)
    t = tau * w_mean
    return (s_in + t * h_all) / (m + t), (s_dn + t * hd_all) / (m + t), n


def code_experts(moe, x, idx, k, lut, l, out_dir: Path, wts=None, tau: float = 0.0) -> dict:
    ex = moe.experts
    h_all = x_ref = None
    if tau > 0:   # the layer-wide statistics the shrinkage pulls toward: all tokens, and a fixed sample of them
        h_all = sum(c.float().T @ c.float() for c in x.split(65536)) / x.shape[0]   # in chunks: no fp32 copy
        g = torch.Generator(device="cpu").manual_seed(l)
        x_ref = x[torch.randperm(x.shape[0], generator=g)[:2048].to(x.device)]
    n_exp, two_ff, n_embd = ex.gate_up_proj.shape
    n_ff = two_ff // 2
    files = {n: open(out_dir / f"blk.{l}.ffn_{n}_exps.ms", "wb") for n in ("gate", "up", "down")}
    eb = 16
    zero = 0
    try:
        for e0 in range(0, n_exp, eb):
            es = list(range(e0, min(e0 + eb, n_exp)))
            hs = [expert_hessians(x, idx, e, ex.gate_up_proj[e].float(), wts, tau, h_all, x_ref) for e in es]
            zero += sum(1 for h in hs if h[2] == 0)
            gu = code(ex.gate_up_proj[es].float(), torch.stack([h[0] for h in hs]), k, lut, [l * 4096 + 2 * e + 1 for e in es])
            dn = code(ex.down_proj[es].float(), torch.stack([h[1] for h in hs]), k, lut, [l * 4096 + 2 * e + 2 for e in es])
            rb_gu = ms_row_bytes(n_embd, k) * n_ff
            for (rows, signs, _), (rows_d, signs_d, _) in zip(gu, dn):
                files["gate"].write(rows[:rb_gu] + signs.tobytes())
                files["up"].write(rows[rb_gu:] + signs.tobytes())
                files["down"].write(rows_d + signs_d.tobytes())
    finally:
        for f in files.values():
            f.close()
    rec = {
        "layer": l, "k": k, "n_expert": n_exp, "zero_token_experts": zero,
        "tensors": {f"blk.{l}.ffn_{n}_exps.weight": {"file": f"blk.{l}.ffn_{n}_exps.ms", "shape": s}
                    for n, s in (("gate", [n_embd, n_ff]), ("up", [n_embd, n_ff]), ("down", [n_ff, n_embd]))},
    }
    (out_dir / f"blk.{l}.experts.json").write_text(json.dumps(rec, indent=1))
    return rec


def spine_hooks(layer, hess: dict, counts: dict) -> list:
    hooks = []
    for name, mod in layer.named_modules():
        if name in SPINE:
            def hk(_m, inp, _o, name=name):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                h = x.T @ x
                if name in hess:
                    hess[name] += h
                    counts[name] += x.shape[0]
                else:
                    hess[name], counts[name] = h, x.shape[0]
            hooks.append(mod.register_forward_hook(hk))
    return hooks


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3.8-Flash-Next -> MS records, layer-streamed")
    p.add_argument("--hf-dir", type=Path, required=True)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True, help='JSON {"experts_k": [k per layer]}')
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--calib-seqs", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--spine-k", type=int, default=6)
    p.add_argument("--head-k", type=int, default=6)
    p.add_argument("--stop-after", type=int, default=None, help="last layer to convert in this run")
    p.add_argument("--reuse-from", type=Path, help="an earlier output folder (same plan) to copy layers from")
    p.add_argument("--reuse-layers", type=int, default=0, help="copy layers below this from --reuse-from")
    p.add_argument("--hess-tau", type=float, default=0.0,
                   help="shrink each expert's Hessians toward the layer's with this many tokens' weight (0: off)")
    p.add_argument("--hess-affinity", action="store_true",
                   help="weight each routed token in its expert's Hessians by its router score squared")
    args = p.parse_args()
    if args.reuse_layers and not args.reuse_from:
        p.error("--reuse-layers needs --reuse-from")

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True  # Hessian sums: TF32 inputs, FP32 accumulation
    out = args.out_dir
    (out / "experts").mkdir(parents=True, exist_ok=True)
    tc = json.loads((args.hf_dir / "config.json").read_text())["text_config"]
    cfg = Qwen4ExpTextConfig(**tc)
    nl = cfg.num_hidden_layers
    plan = json.loads(args.plan.read_text())["experts_k"]
    assert len(plan) == nl and all(k in (3, 4, 6) for k in plan), plan
    wmap = json.loads((args.hf_dir / "model.safetensors.index.json").read_text())["weight_map"]
    calib = torch.load(args.calib, map_location="cpu", weights_only=True)
    cal_ids, _ = split_calib(calib, args.calib_seqs, 0)
    bs = args.batch_size

    # a large calibration set's stream (tokens x hyper-connection streams x hidden, bf16) stays in host RAM
    host_h = cal_ids.numel() * cfg.hc_count * cfg.hidden_size * 2 > (4 << 30)
    h, kws = prelude(cfg, args.hf_dir, wmap, cal_ids, dev, bs, host=host_h)
    state = out / "state.pt"
    start = 0
    if state.exists():
        st = torch.load(state, map_location="cpu" if host_h else dev, weights_only=True)
        start, h = st["next_layer"], st["h"]
        print(f"resuming at layer {start}", flush=True)
    lut = ms.build_1mad_lut(dev)
    last = nl - 1 if args.stop_after is None else args.stop_after

    for l in range(start, min(last, nl - 1) + 1):
        t0 = time.time()
        layer = load_layer(cfg, args.hf_dir, wmap, l, dev)
        if l < args.reuse_layers:
            # the layer's records come from an earlier run with the same plan; only advance the stream
            h = run_layer(layer, h, kws, bs)
            for src in list(args.reuse_from.glob(f"blk.{l}.*")) + list((args.reuse_from / "experts").glob(f"blk.{l}.*")):
                shutil.copy2(src, out / src.relative_to(args.reuse_from))
            del layer
            torch.save({"next_layer": l + 1, "h": h}, state)
            print(f"layer {l:2d}: records reused from {args.reuse_from}, forward {time.time() - t0:.0f}s", flush=True)
            continue
        hess, counts, xs = {}, {}, []
        hooks = spine_hooks(layer, hess, counts)
        h = run_layer(layer, h, kws, bs, grab_in=xs)
        for hk in hooks:
            hk.remove()
        x = torch.cat(xs)
        del xs
        t_fwd = time.time() - t0

        mods = dict(layer.named_modules())
        for name in sorted(hess):
            gguf = f"blk.{l}.{SPINE[name]}"
            w, hh = to_gguf_order(gguf, mods[name].weight.float(), hess.pop(name) / counts[name])
            rows, signs, _ = code(w.unsqueeze(0), hh.unsqueeze(0), args.spine_k, lut, [l * 4096 + 3000 + len(gguf)])[0]
            write_record(out, gguf, args.spine_k, w.shape[1], w.shape[0], rows, signs)
        t_spine = time.time() - t0 - t_fwd

        moe = layer.mlp
        with torch.no_grad():
            _, wts, idx = moe.gate(x)
        rec = code_experts(moe, x, idx, plan[l], lut, l, out / "experts", wts if args.hess_affinity else None,
                           args.hess_tau)
        del layer, moe, x, idx, wts
        torch.cuda.empty_cache()
        torch.save({"next_layer": l + 1, "h": h}, state)
        print(f"layer {l:2d} {cfg.layer_types[l]:>17}: experts MS_C{plan[l]} (zero-token {rec['zero_token_experts']}) | "
              f"fwd {t_fwd:.0f}s spine {t_spine:.0f}s experts {time.time() - t0 - t_fwd - t_spine:.0f}s "
              f"total {time.time() - t0:.0f}s", flush=True)

    if last >= nl - 1 and not (out / "output.weight.json").exists():
        t0 = time.time()
        with torch.device("meta"):
            mixer = Qwen4ExpTextGatedResidual(cfg, use_combine=False)
        pre = f"{PRE}hyper_connection_mixer."
        sd = {}
        for k in [k for k in wmap if k.startswith(pre)]:
            with safe_open(args.hf_dir / wmap[k], "pt") as s:
                sd[k[len(pre):]] = s.get_tensor(k)
        mixer.load_state_dict(sd, assign=True)
        mixer = mixer.to(dev, torch.bfloat16)
        hh = torch.zeros(cfg.hidden_size, cfg.hidden_size, device=dev)
        with torch.no_grad():
            for i in range(0, h.shape[0], bs):
                m = mixer(h[i : i + bs].to(dev)).reshape(-1, cfg.hidden_size).float()
                hh += m.T @ m
        hh /= h.shape[0] * h.shape[1]
        with safe_open(args.hf_dir / wmap["lm_head.weight"], "pt") as s:
            w = s.get_tensor("lm_head.weight").to(dev)
        rows, signs = code_rows(w, hh, args.head_k, lut, seed=12)
        write_record(out, "output.weight", args.head_k, w.shape[1], w.shape[0], rows, signs)
        print(f"head MS_C{args.head_k}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
