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
from ms_pack import MS_BITS, ms_row_bytes, write_record
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


VITERBI = (32, 8)   # search window, commit length; set from --window/--commit


def code(ws: torch.Tensor, hs: torch.Tensor, k: int, lut, seeds) -> list:
    """Code E matrices [E, out, in] on Hessians [E, in, in]: (rows bytes, int8 signs, decoded W) per matrix."""
    blk = largest_pow2_divisor(int(ws.shape[2]), cap=512)
    rot = [ms.rotate_weight_and_hessian(w, h, block_size=blk, seed=s) for w, h, s in zip(ws, hs, seeds)]
    hu = torch.stack([ms.cholesky_inv_upper(r[1]) for r in rot])
    scales, states, q = viterbi_code_batched_cuda(torch.stack([r[0] for r in rot]), hu, k, lut, *VITERBI)
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
        scales, states, _ = viterbi_code_batched_cuda(w_rot.unsqueeze(0), hu, k, lut, *VITERBI)
        parts.append(pack_rows(states[0], scales[0], k))
    return b"".join(parts), signs.to(torch.int8).cpu().numpy()


def expert_hessians(x: torch.Tensor, idx: torch.Tensor, e: int, w_gu: torch.Tensor, wts=None, tau: float = 0.0,
                    h_all=None, x_ref=None, tok_w=None):
    """Input Hessians of expert e's gate/up and down rows and its routed-token count.  Plain: sums over the tokens
    routed to it (identity Hessians when none is).  wts (the router's top-k scores): each token weighted by its score
    squared, as its error reaches the layer output times the score.  tau > 0: each Hessian, as a mean, shrunk toward
    the layer's (gate/up: h_all, every token's x x^T mean; down: the expert's activations on the x_ref tokens) with
    tau tokens' weight, so a rarely routed expert keeps the layer's input statistics instead of a few tokens' or none."""
    hit = idx == e
    rows = hit.any(-1)
    sel = rows.nonzero().squeeze(1)
    n = sel.numel()
    if tau <= 0 and not n:
        return torch.eye(x.shape[1], device=x.device), torch.eye(w_gu.shape[0] // 2, device=x.device), 0
    p = (wts.float()[sel] * hit[sel]).sum(-1) ** 2 if wts is not None else torch.ones(n, device=x.device)
    if tok_w is not None:   # per-token weights (Fisher): on top of the router's
        p = p * tok_w[sel]
    s_in = torch.zeros(x.shape[1], x.shape[1], device=x.device)
    s_dn = torch.zeros(w_gu.shape[0] // 2, w_gu.shape[0] // 2, device=x.device)
    for c in range(0, n, 65536):   # in chunks: a busy expert's tokens in fp32 would not fit next to x
        xe, pc = x[sel[c : c + 65536]].float(), p[c : c + 65536, None]
        g, u = (xe @ w_gu.T).chunk(2, -1)
        a = torch.nn.functional.silu(g) * u
        s_in += (xe * pc).T @ xe
        s_dn += (a * pc).T @ a
    m = p.sum()
    if tau <= 0:
        return s_in, s_dn, n
    gr, ur = (x_ref.float() @ w_gu.T).chunk(2, -1)
    ar = torch.nn.functional.silu(gr) * ur
    hd_all = ar.T @ ar / ar.shape[0]
    # tau weighs the prior as tau tokens against the expert's effective sample size (n when all weights are equal)
    ess = m * m / (p * p).sum() if n else torch.tensor(1.0, device=x.device)
    t = tau * (m / ess if n else torch.tensor(1.0, device=x.device))
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
    eb = 8 * 32 // VITERBI[0]   # experts per batch: the back-pointer buffer grows with the window
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


# The decoder layer's forward in its parts (Qwen4ExpTextDecoderLayer.forward, no cache): the two-stream step needs
# the state between them. two_stream_layer checks the parts against the layer's own forward.
KNOWN_KW = ("position_embeddings", "attention_mask", "conv_mask", "ple_input_ids")


def attn_in(layer, h, kw):
    """(attention input, hyper-connection input, injection weights)"""
    if layer.ple is not None:
        h = h + layer.ple(h, kw["ple_input_ids"], None, conv_mask=kw.get("conv_mask"))
    return layer.attn_hyper_connection(h)


def attn(layer, x, kw):
    rest = {k: v for k, v in kw.items() if k not in KNOWN_KW}
    if layer.layer_type == "linear_attention":
        return layer.linear_attn(x, cache_params=None, attention_mask=kw.get("conv_mask"), **rest)
    return layer.self_attn(x, kw["position_embeddings"], attention_mask=kw["attention_mask"], **rest)[0]


def inject(hyper, y, w):
    return hyper + (y.unsqueeze(-2) * w.unsqueeze(-1)).flatten(-2)


def stream_target(d, w):
    """The block output y that best gives d = sum over streams of y * w_s, per token: (target, weight), the weighted
    least squares fit over the hyper-connection streams (d [..., hc * hidden], w [..., hc])."""
    d = d.float().unflatten(-1, (w.shape[-1], -1))
    w = w.float()
    c = (w * w).sum(-1)
    return (w.unsqueeze(-1) * d).sum(-2) / c.unsqueeze(-1), c


def input_map(w, hqq, crq):
    """W M with M = (C_rq + rI)(H_qq + rI)^-1, r = 1% of H_qq's mean diagonal: minimises
    sum ||W x_ref - W M x_q||^2 + r ||W M - W||^2 (means over the tokens)."""
    eye = torch.eye(hqq.shape[0], device=hqq.device, dtype=hqq.dtype)
    r = 0.01 * hqq.diagonal().mean()
    return w @ torch.linalg.solve((hqq + r * eye).T, (crq + r * eye).T).T


def ridge_refit(w, hh, c):
    """W_t with W_t^T = (HH + (lam + mu) I)^-1 (C + mu W^T), lam = 1e-4 and mu = 0.05 of HH's mean diagonal: the
    least squares fit of the targets in C = sum z y^T, held toward W."""
    d = hh.diagonal().mean()
    eye = torch.eye(hh.shape[0], device=hh.device, dtype=hh.dtype)
    return torch.linalg.solve(hh + (1e-4 + 0.05) * d * eye, c + 0.05 * d * w.T).T.contiguous()


@torch.no_grad()
def two_stream_layer(layer, l, nl, h, hq, kws, bs, x, code_spine, code_moe, load):
    """One layer of the two-stream conversion (the MS method's section 6.3 on the hyper-connection layer). h, the
    original stream, and hq, the converted one, advance in place; x gets the converted MoE inputs. code_spine(name, W,
    H) and code_moe(x) write records; load(names, experts) puts their decoded values into the layer."""
    dev = next(layer.parameters()).device
    tm, t0 = {}, time.time()
    mark = lambda k: tm.__setitem__(k, time.time() - t0 - sum(tm.values()))
    mods = dict(layer.named_modules())
    reads = [n for n in ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "self_attn.q_proj", "self_attn.k_proj",
                         "self_attn.v_proj") if n in mods]
    out_name = "linear_attn.out_proj" if "linear_attn.out_proj" in mods else "self_attn.o_proj"
    sh = layer.mlp.shared_expert
    n, hid = x.shape
    att_ref = torch.empty(n, hid, dtype=torch.bfloat16, device=dev)
    ffn_ref = torch.empty(n, hid, dtype=torch.bfloat16, device=dev)
    mid_ref = torch.empty(n, h.shape[-1], dtype=torch.bfloat16, device=dev)
    sl = lambda b: slice(b * bs, (b + 1) * bs)
    rows = lambda b: slice(b * bs * h.shape[1], (b + 1) * bs * h.shape[1])

    # the original stream through the original layer; its parts checked against the layer's forward once
    for b, kw in enumerate(kws):
        hb = h[sl(b)].to(dev)
        xa, hy, wi = attn_in(layer, hb, kw)
        mid = inject(hy, attn(layer, xa, kw), wi)
        xf, hy2, wi2 = layer.mlp_hyper_connection(mid)
        o = inject(hy2, layer.mlp(xf), wi2)
        if b == 0:
            assert torch.equal(o, layer(hb, **kw)), "the layer's parts differ from its forward"
        att_ref[rows(b)], mid_ref[rows(b)], ffn_ref[rows(b)] = (t.reshape(-1, t.shape[-1]) for t in (xa, mid, xf))
        h[sl(b)] = o.to(h.device)
    mark("ref")

    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False   # attention and shared gate/up statistics in fp32
    hqq = crq = 0
    for b, kw in enumerate(kws):
        xq = attn_in(layer, hq[sl(b)].to(dev), kw)[0].reshape(-1, hid).float()
        hqq, crq = hqq + xq.T @ xq, crq + att_ref[rows(b)].float().T @ xq
    for name in reads:
        code_spine(name, input_map(mods[name].weight.float(), hqq / n, crq / n), hqq / n)
    load(reads, False)
    mark("reads")

    z_in = []
    hk = mods[out_name].register_forward_pre_hook(lambda m, a: z_in.append(a[0]))
    hh = cc = 0
    for b, kw in enumerate(kws):
        xa, hy, wi = attn_in(layer, hq[sl(b)].to(dev), kw)
        attn(layer, xa, kw)
        z = z_in.pop().reshape(-1, mods[out_name].in_features).float()
        t, c = stream_target(mid_ref[rows(b)].float() - hy.reshape(-1, hy.shape[-1]).float(), wi.reshape(-1, wi.shape[-1]))
        hh, cc = hh + (z * c[:, None]).T @ z, cc + (z * c[:, None]).T @ t
    hk.remove()
    del mid_ref
    code_spine(out_name, ridge_refit(mods[out_name].weight.float(), hh, cc), hh / n)
    load([out_name], False)
    mark("out")

    # FFN inputs of the converted stream; the shared expert's target: in the first half of the layers what the layer
    # output needs (absorbing the error so far), from the half on the original FFN on the converted inputs
    hnn = cnn = 0
    tgt = torch.empty(n, hid, dtype=torch.float32, device=dev)
    tw = torch.empty(n, dtype=torch.float32, device=dev)
    for b, kw in enumerate(kws):
        xa, hy, wi = attn_in(layer, hq[sl(b)].to(dev), kw)
        xf, hy2, wi2 = layer.mlp_hyper_connection(inject(hy, attn(layer, xa, kw), wi))
        xq = xf.reshape(-1, hid)
        x[rows(b)] = xq
        xq = xq.float()
        hnn, cnn = hnn + xq.T @ xq, cnn + ffn_ref[rows(b)].float().T @ xq
        d = (h[sl(b)].to(dev).float() - hy2.float() if l < nl // 2 else
             (layer.mlp(xf).float().unsqueeze(-2) * wi2.float().unsqueeze(-1)).flatten(-2))
        tgt[rows(b)], tw[rows(b)] = stream_target(d.reshape(-1, d.shape[-1]), wi2.reshape(-1, wi2.shape[-1]))
    del ffn_ref
    torch.backends.cuda.matmul.allow_tf32 = tf32
    for name in ("mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj"):
        code_spine(name, input_map(mods[name].weight.float(), hnn / n, cnn / n), hnn / n)
    mark("ffn-in")
    rec = code_moe(x)
    mark("experts")
    load(["mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj"], True)
    mark("load")

    # the shared expert's down refit after the coded routed experts: its output enters times the shared gate
    moe, ha, ca = layer.mlp, 0, 0
    for c0 in range(0, n, bs * h.shape[1]):
        xc = x[c0 : c0 + bs * h.shape[1]]
        _, wts, sel = moe.gate(xc)
        rest = tgt[c0 : c0 + xc.shape[0]] - moe.experts(xc, sel, wts).float()
        s = torch.sigmoid(moe.shared_expert_gate(xc)).float()
        a = (sh.act_fn(sh.gate_proj(xc)) * sh.up_proj(xc)).float()
        om = tw[c0 : c0 + xc.shape[0], None]
        ha, ca = ha + (a * om * s * s).T @ a, ca + (a * om * s).T @ rest
    del tgt, tw
    code_spine("mlp.shared_expert.down_proj", ridge_refit(sh.down_proj.weight.float(), ha, ca), ha / n)
    load(["mlp.shared_expert.down_proj"], False)
    mark("down")
    run_layer(layer, hq, kws, bs, inplace=True)
    mark("advance")
    rec["times"] = tm
    return rec


MTP_PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj")   # MS-coded
MTP_Q8 = ("fc_embedding", "fc_hidden", "self_attn.indexer.index_qk_proj")   # Q8_0 (the drafter's loader reads them so)


def mtp_index_entry(name: str, t: torch.Tensor, coded: dict) -> tuple:
    """(kind, rows, cols, bytes) of one drafter tensor in the engine's mtp/dense forms: MS records for MTP_PROJ (rows,
    int8 signs), Q8_0 for MTP_Q8, f32 1 + w for the norm vectors, bf16 for the other matrices."""
    import gguf.quants as gq
    from gguf import GGMLQuantizationType as T
    base = name[: -len(".weight")] if name.endswith(".weight") else name
    if base in coded:
        rows, signs, n_out, n_in = coded[base]
        return "ms", n_out, n_in, rows + signs.tobytes()
    if t.dim() == 1:
        return "f32", 1, t.shape[0], (1.0 + t.float()).cpu().numpy().astype("<f4").tobytes()
    if base in MTP_Q8:
        return "q8_0", t.shape[0], t.shape[1], gq.quantize(t.float().cpu().numpy(), T.Q8_0).tobytes()
    return "bf16", t.shape[0], t.shape[1], t.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().tobytes()


@torch.no_grad()
def code_mtp(cfg, hf: Path, wmap: dict, h, cal_ids, kws, bs, dev, lut, k_proj: int, k_exp: int, tau: float, out: Path):
    """The MTP drafter from the checkpoint's mtp.* weights, coded on Hessians from the original model's last residual
    stream h: <out>/mtp/mtp.dense + mtp.dense.index (the pack's mtp/dense forms, first line `#experts ms T T BLOB`)
    and the experts as one layer's records (code_experts). The drafter (decode/mtp/forward.cpp): R at t and the
    embedding of token t+1 -> fc_hidden(norm(R)) per stream + fc_embedding(norm(e)) -> one full-attention layer."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
    d = out / "mtp"
    d.mkdir(parents=True, exist_ok=True)
    get = lambda k: safe_open(hf / wmap[k], "pt").get_tensor(k)
    c1 = Qwen4ExpTextConfig(**{**cfg.to_dict(), "num_hidden_layers": 1, "layer_types": ["full_attention"],
                               "ple_layer_ids": []})
    c1._attn_implementation = "sdpa"
    with torch.device("meta"):
        layer = Qwen4ExpTextDecoderLayer(c1, 0)
    lp = "mtp.layers.0."
    layer.load_state_dict({k[len(lp):]: get(k) for k in wmap if k.startswith(lp)}, strict=True, assign=True)
    layer = layer.to(dev, torch.bfloat16).eval()
    top = {k[4:]: get(k).to(dev) for k in wmap if k.startswith("mtp.") and ".layers." not in k}
    emb = get(f"{PRE}embed_tokens.weight").to(dev)
    hc, n = cfg.hc_count, cfg.hidden_size

    def norm(x, w, group=None):
        x = x.float()
        if group:
            x = x.unflatten(-1, (-1, group))
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
        return (x.flatten(-2) if group else x) * (1.0 + w.float())

    mods = dict(layer.named_modules())
    hs = {}
    x = torch.empty(cal_ids.numel(), n, dtype=torch.bfloat16, device=dev)
    fill = [0]

    def hk(name):
        def f(_m, a):
            t = a[0].reshape(-1, a[0].shape[-1]).float()
            hs[name] = hs.get(name, 0) + t.T @ t
        return f

    def grab(_m, a):
        t = a[0].reshape(-1, n)
        x[fill[0] : fill[0] + t.shape[0]] = t
        fill[0] += t.shape[0]
    hooks = [mods[m].register_forward_pre_hook(hk(m)) for m in MTP_PROJ] + [layer.mlp.register_forward_pre_hook(grab)]
    for b, kw in enumerate(kws):
        t = cal_ids[b * bs : (b + 1) * bs].to(dev, torch.long)
        nxt = torch.cat([t[:, 1:], t[:, -1:]], 1)
        h2 = norm(h[b * bs : (b + 1) * bs].to(dev), top["pre_fc_norm_hidden.weight"], n).unflatten(-1, (hc, n))
        h2 = h2 @ top["fc_hidden.weight"].float().T
        e2 = norm(emb[nxt], top["pre_fc_norm_embedding.weight"]) @ top["fc_embedding.weight"].float().T
        layer((h2 + e2.unsqueeze(-2)).flatten(-2).to(torch.bfloat16), **kw)
    for hk_ in hooks:
        hk_.remove()
    assert fill[0] == x.shape[0], (fill[0], x.shape)
    coded = {}
    for i, m in enumerate(MTP_PROJ):
        w = mods[m].weight.float()
        rows, signs, _ = code(w[None], (hs.pop(m) / x.shape[0])[None], k_proj, lut, [9000 + i])[0]
        coded[m] = (rows, signs, w.shape[0], w.shape[1])
    _, wts, idx = layer.mlp.gate(x)
    rec = code_experts(layer.mlp, x, idx, k_exp, lut, 0, d, None, tau)
    ex = rec["tensors"]
    blob = sum(int(ms_row_bytes(int(t["shape"][0]), k_exp)) * int(t["shape"][1]) + int(t["shape"][0]) for t in ex.values())
    t_exp = 120 + k_exp
    lines, parts, off = [f"#experts ms {t_exp} {t_exp} {blob}"], [], 0
    names = [k[4:] for k in wmap if k.startswith("mtp.") and ".experts." not in k]
    for name in names:
        short = name.replace("layers.0.", "")
        kind, r, c, raw = mtp_index_entry(short, get("mtp." + name), coded)
        pad = (-off) % 256
        parts.append(b"\0" * pad)
        off += pad
        lines.append(f"{short} {kind} {r} {c} {off} {len(raw)}")
        parts.append(raw)
        off += len(raw)
    (d / "mtp.dense").write_bytes(b"".join(parts))
    (d / "mtp.dense.index").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"k_proj": k_proj, "k_exp": k_exp, "zero_token_experts": rec["zero_token_experts"], "dense_bytes": off}


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3.8-Flash-Next -> MS records, layer-streamed")
    p.add_argument("--hf-dir", type=Path, required=True)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True, help='JSON {"experts_k": [k per layer]}')
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--calib-seqs", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--spine-k", type=int, default=6, choices=MS_BITS)
    p.add_argument("--head-k", type=int, default=6, choices=MS_BITS)
    p.add_argument("--stop-after", type=int, default=None, help="last layer to convert in this run")
    p.add_argument("--reuse-from", type=Path, help="an earlier output folder (same plan) to copy layers from")
    p.add_argument("--reuse-layers", type=int, default=0, help="copy layers below this from --reuse-from")
    p.add_argument("--window", type=int, default=32, help="Viterbi search window")
    p.add_argument("--commit", type=int, default=8, help="Viterbi commit length")
    p.add_argument("--hess-tau", type=float, default=0.0,
                   help="shrink each expert's Hessians toward the layer's with this many tokens' weight (0: off)")
    p.add_argument("--two-stream", action="store_true",
                   help="keep the converted model's stream beside the original's (the MS method's section 6.3): read "
                        "projections mapped toward the original inputs, output projections refit toward the original "
                        "stream, every matrix coded on the converted model's inputs")
    p.add_argument("--mtp-k", type=int, default=5, choices=MS_BITS, help="the MTP drafter's projections")
    p.add_argument("--mtp-k-exp", type=int, default=2, choices=MS_BITS, help="the MTP drafter's routed experts")
    p.add_argument("--mtp-only", action="store_true",
                   help="only the MTP drafter: the original layers run uncoded, then the drafter is coded")
    p.add_argument("--hess-affinity", action="store_true",
                   help="weight each routed token in its expert's Hessians by its router score squared")
    args = p.parse_args()
    if args.reuse_layers and not args.reuse_from:
        p.error("--reuse-layers needs --reuse-from")
    if args.two_stream and args.reuse_layers:
        p.error("--reuse-layers advances one stream only: not with --two-stream")
    if args.mtp_only and (args.two_stream or args.reuse_layers or args.stop_after is not None):
        p.error("--mtp-only runs every original layer: not with --two-stream, --reuse-layers or --stop-after")

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True  # Hessian sums: TF32 inputs, FP32 accumulation
    global VITERBI
    VITERBI = (args.window, args.commit)
    out = args.out_dir
    (out / "experts").mkdir(parents=True, exist_ok=True)
    tc = json.loads((args.hf_dir / "config.json").read_text())["text_config"]
    cfg = Qwen4ExpTextConfig(**tc)
    nl = cfg.num_hidden_layers
    plan = json.loads(args.plan.read_text())["experts_k"]
    assert len(plan) == nl and all(k in MS_BITS for k in plan), plan
    wmap = json.loads((args.hf_dir / "model.safetensors.index.json").read_text())["weight_map"]
    calib = torch.load(args.calib, map_location="cpu", weights_only=True)
    cal_ids, _ = split_calib(calib, args.calib_seqs, 0)
    bs = args.batch_size

    # a large calibration set's stream (tokens x hyper-connection streams x hidden, bf16) stays in host RAM
    host_h = cal_ids.numel() * cfg.hc_count * cfg.hidden_size * 2 > (4 << 30)
    h, kws = prelude(cfg, args.hf_dir, wmap, cal_ids, dev, bs, host=host_h)
    hq = h.clone() if args.two_stream else None   # the converted model's stream
    state = out / "state.pt"
    start = 0
    if args.mtp_only:
        for l in range(nl):
            h = run_layer(load_layer(cfg, args.hf_dir, wmap, l, dev), h, kws, bs, inplace=True)
            torch.cuda.empty_cache()
        print(f"original stream through {nl} layers", flush=True)
        t0 = time.time()
        r = code_mtp(cfg, args.hf_dir, wmap, h, cal_ids, kws, bs, dev, ms.build_1mad_lut(dev), args.mtp_k,
                     args.mtp_k_exp, args.hess_tau, out)
        print(f"mtp: projections MS_C{r['k_proj']}, experts MS_C{r['k_exp']} (zero-token {r['zero_token_experts']}), "
              f"dense {r['dense_bytes'] / 2**20:.1f} MiB: {time.time() - t0:.0f}s", flush=True)
        return
    if state.exists():
        st = torch.load(state, map_location="cpu" if host_h else dev, weights_only=True)
        start, h, hq = st["next_layer"], st["h"], st.get("hq")
        assert (hq is not None) == args.two_stream, "the state was written with another --two-stream setting"
        print(f"resuming at layer {start}", flush=True)
    lut = ms.build_1mad_lut(dev)
    last = nl - 1 if args.stop_after is None else args.stop_after

    for l in range(start, min(last, nl - 1) + 1):
        t0 = time.time()
        layer = load_layer(cfg, args.hf_dir, wmap, l, dev)
        if l < args.reuse_layers:
            # the layer's records come from an earlier run with the same plan; only advance the stream
            h = run_layer(layer, h, kws, bs, inplace=True)
            for src in list(args.reuse_from.glob(f"blk.{l}.*")) + list((args.reuse_from / "experts").glob(f"blk.{l}.*")):
                shutil.copy2(src, out / src.relative_to(args.reuse_from))
            del layer
            torch.save({"next_layer": l + 1, "h": h}, state)
            print(f"layer {l:2d}: records reused from {args.reuse_from}, forward {time.time() - t0:.0f}s", flush=True)
            continue
        if hq is not None:
            from ms_flashnext_eval import load_ms_layer   # here: the eval module imports this one
            mods = dict(layer.named_modules())

            def code_spine(name, w, hh):
                gguf = f"blk.{l}.{SPINE[name]}"
                w, hh = to_gguf_order(gguf, w, hh)
                rows, signs, _ = code(w.unsqueeze(0), hh.unsqueeze(0), args.spine_k, lut, [l * 4096 + 3000 + len(gguf)])[0]
                write_record(out, gguf, args.spine_k, w.shape[1], w.shape[0], rows, signs)

            def code_moe(xq):
                with torch.no_grad():
                    parts = [layer.mlp.gate(c)[1:] for c in xq.split(65536)]
                    wts, idx = torch.cat([q[0] for q in parts]), torch.cat([q[1] for q in parts])
                return code_experts(layer.mlp, xq, idx, plan[l], lut, l, out / "experts",
                                    wts if args.hess_affinity else None, args.hess_tau)

            x = torch.empty(cal_ids.numel(), cfg.hidden_size, dtype=torch.bfloat16, device=dev)
            rec = two_stream_layer(layer, l, nl, h, hq, kws, bs, x, code_spine, code_moe,
                                   lambda names, ex: load_ms_layer(layer, l, out, lut, dev, names, ex))
            del layer, mods, x
            torch.cuda.empty_cache()
            torch.save({"next_layer": l + 1, "h": h, "hq": hq}, state)
            print(f"layer {l:2d} {cfg.layer_types[l]:>17}: experts MS_C{plan[l]} (zero-token {rec['zero_token_experts']}) | "
                  + " ".join(f"{k} {v:.0f}s" for k, v in rec["times"].items()) + f" total {time.time() - t0:.0f}s", flush=True)
            continue
        hess, counts = {}, {}
        hooks = spine_hooks(layer, hess, counts)
        # the MoE inputs go straight into one buffer: a list plus torch.cat would hold them twice
        x = torch.empty(cal_ids.numel(), cfg.hidden_size, dtype=torch.bfloat16, device=dev)
        fill = [0]

        def grab(_m, a):
            t = a[0].reshape(-1, a[0].shape[-1])
            x[fill[0] : fill[0] + t.shape[0]] = t
            fill[0] += t.shape[0]
        hooks.append(layer.mlp.register_forward_pre_hook(grab))
        h = run_layer(layer, h, kws, bs, inplace=True)
        for hk in hooks:
            hk.remove()
        assert fill[0] == x.shape[0], (fill[0], x.shape)
        t_fwd = time.time() - t0

        mods = dict(layer.named_modules())
        for name in sorted(hess):
            gguf = f"blk.{l}.{SPINE[name]}"
            w, hh = to_gguf_order(gguf, mods[name].weight.float(), hess.pop(name) / counts[name])
            rows, signs, _ = code(w.unsqueeze(0), hh.unsqueeze(0), args.spine_k, lut, [l * 4096 + 3000 + len(gguf)])[0]
            write_record(out, gguf, args.spine_k, w.shape[1], w.shape[0], rows, signs)
        t_spine = time.time() - t0 - t_fwd

        moe = layer.mlp
        with torch.no_grad():   # in chunks: only the top-k ids and scores stay, not every token's logits
            parts = [moe.gate(c)[1:] for c in x.split(65536)]
            wts, idx = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
            del parts
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
    if last >= nl - 1 and not (out / "mtp" / "mtp.dense.index").exists():
        t0 = time.time()
        r = code_mtp(cfg, args.hf_dir, wmap, h, cal_ids, kws, bs, dev, lut, args.mtp_k, args.mtp_k_exp, args.hess_tau, out)
        print(f"mtp: projections MS_C{r['k_proj']}, experts MS_C{r['k_exp']} (zero-token {r['zero_token_experts']}), "
              f"dense {r['dense_bytes'] / 2**20:.1f} MiB: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
