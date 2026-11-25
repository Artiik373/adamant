"""Per-layer MS width sensitivity scan (default MS_C3 / MS_C4) for Qwen3.8-Flash-Next routed experts.

Streams one BF16 decoder layer at a time through calibration tokens (the reference chain stays BF16), builds
per-expert Hessians from the real routing, quantizes all experts of the layer at each k, and measures the
routed, MoE and layer-output error on held-out tokens. One JSON line per layer; resumes from the saved
hidden state. The PLE n-gram table is read from the shards by memory map, not loaded.

    python tools/convert/ms_sensitivity_scan.py --hf-dir C:/llm/models/Qwen3.8-Flash-Next \
        --calib G:/llm/calib/calib_2m.pt --out-dir G:/llm/results/ms_scan
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextModel,
    Qwen4ExpTextQSAIndexer,
)

import ms_moe_converter as ms
from ms_pack import MS_BITS
from eval_ms_kld import split_calib
from ms_viterbi_cuda import viterbi_code_batched_cuda

PRE = "model.language_model."

_indexer_forward = Qwen4ExpTextQSAIndexer.forward


def _indexer_all(self, hidden_states, position_embeddings, attention_mask, past_key_values):
    """Without a cache and with no more keys than the token budget, the indexer selects every key: skip it."""
    if past_key_values is None and attention_mask.shape[-1] <= self.token_budget:
        return torch.ones((), dtype=torch.bool, device=hidden_states.device)
    return _indexer_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values)


Qwen4ExpTextQSAIndexer.forward = _indexer_all
NGRAM = "ple.ple_embedding.ngram_embedding."


class MemmapEmbedding(torch.nn.Module):
    """Row lookup into the n-gram table shards on disk (BF16, equal rows per shard)."""

    def __init__(self, shards):
        super().__init__()
        self.shards = shards
        self.rows = shards[0].shape[0]
        self.weight = torch.empty(0)  # the caller moves ids to weight.device

    def forward(self, ids):
        flat = ids.reshape(-1).cpu().numpy()
        out = np.empty((flat.size, self.shards[0].shape[1]), dtype=np.uint16)
        shard, row = np.divmod(flat, self.rows)

        def fetch(s):
            m = shard == s
            out[m] = self.shards[s][row[m]]

        # parallel page faults keep the disk queue full on a cold cache
        with ThreadPoolExecutor(32) as pool:
            list(pool.map(fetch, np.unique(shard)))
        return torch.from_numpy(out).view(torch.bfloat16).reshape(*ids.shape, -1)


def tensor_memmap(path: Path, key: str) -> np.memmap:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))[key]
    assert h["dtype"] == "BF16", key
    return np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + h["data_offsets"][0], shape=tuple(h["shape"]))


def load_layer(cfg, hf: Path, wmap: dict, l: int, dev) -> Qwen4ExpTextDecoderLayer:
    # built from the config, not from_pretrained: without this the eager attention adds the boolean causal
    # mask to the scores instead of masking, and the full-attention layers see future tokens
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        layer = Qwen4ExpTextDecoderLayer(cfg, l)
    pre = f"{PRE}layers.{l}."
    keys = [k for k in wmap if k.startswith(pre) and NGRAM not in k]
    sd = {}
    for f in sorted({wmap[k] for k in keys}):
        with safe_open(hf / f, "pt") as s:
            for k in keys:
                if wmap[k] == f:
                    sd[k[len(pre):]] = s.get_tensor(k)
    layer.load_state_dict(sd, strict=False, assign=True)
    if layer.ple is not None:
        emb = layer.ple.ple_embedding
        n_parts = cfg.split_ngram_parts
        shards = [tensor_memmap(hf / wmap[f"{pre}{NGRAM}shard_{i}.weight"], f"{pre}{NGRAM}shard_{i}.weight") for i in range(n_parts)]
        emb.ngram_embedding = MemmapEmbedding(shards)
        emb.ngram_heads_vocab_sizes = torch.tensor(emb.head_vocab_sizes, dtype=torch.long)
        emb.ngram_heads_offsets = torch.tensor(emb.head_offsets, dtype=torch.long)
        assert sum(sh.shape[0] for sh in shards) >= emb.total_vocab_size
    meta = [n for n, t in list(layer.named_parameters()) + list(layer.named_buffers()) if t.is_meta]
    if meta:
        raise RuntimeError(f"layer {l}: weights not loaded: {meta}")
    return layer.to(dev, torch.bfloat16).eval()


def prelude(cfg, hf: Path, wmap: dict, ids: torch.Tensor, dev, batch: int, host: bool = False):
    """Layer-0 inputs and the per-batch layer kwargs (masks, positions, PLE ids) from the model's own forward.
    host: the inputs collected in host RAM (a calibration set too large for the GPU twice over)."""
    c1 = Qwen4ExpTextConfig(**{**cfg.to_dict(), "num_hidden_layers": 2, "layer_types": cfg.layer_types[:2]})  # 2 layers keep ple_layer_ids valid
    c1._attn_implementation = "sdpa"  # the masks the layers get are built for sdpa (see load_layer)
    with torch.device("meta"):
        model = Qwen4ExpTextModel(c1)
    k = f"{PRE}embed_tokens.weight"
    with safe_open(hf / wmap[k], "pt") as s:
        model.embed_tokens.weight = torch.nn.Parameter(s.get_tensor(k).to(dev), requires_grad=False)
    model.rotary_emb = type(model.rotary_emb)(config=c1).to(dev)
    got = []

    class Stop(Exception):
        pass

    def hook(_m, args, kwargs):
        got.append((args[0], kwargs))
        raise Stop

    model.layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    out, kws = None, []
    with torch.no_grad():
        for i in range(0, ids.shape[0], batch):
            try:
                model(input_ids=ids[i : i + batch].to(dev, torch.long), use_cache=False)
            except Stop:
                pass
            h, kw = got.pop()
            if out is None:   # one buffer: a list plus torch.cat would hold the stream twice
                out = torch.empty((ids.shape[0], *h.shape[1:]), dtype=h.dtype, device="cpu" if host else dev)
            out[i : i + h.shape[0]] = h
            kws.append({n: v for n, v in kw.items() if n != "past_key_values"})
    del model
    return out, kws


FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def nvfp4(w):
    """Round to NVFP4: E2M1 values, an E4M3 scale per 16 inputs, an FP32 scale per tensor."""
    b = w.float().reshape(-1, 16)
    g = (b.abs().amax() / (6 * 448)).clamp_min(1e-30)
    s = ((b.abs().amax(1, keepdim=True) / 6 / g).to(torch.float8_e4m3fn).float() * g).clamp_min(1e-30)
    grid = FP4_GRID.to(w.device)
    a = (b / s).abs().clamp_max(6)
    i = torch.bucketize(a, (grid[1:] + grid[:-1]) / 2)   # nearest grid point
    return (grid[i] * b.sign() * s).reshape(w.shape)


def fp4_lut(lut):
    """The 1MAD values snapped to the E2M1 grid times c, with the c of least squared snap error."""
    def snap(c):
        a = (lut.abs() / c).clamp_max(6)
        g = FP4_GRID.to(lut.device)
        return g[torch.bucketize(a, (g[1:] + g[:-1]) / 2, right=True)] * lut.sign() * c   # ties up, as the kernel
    c = min((float(c) for c in torch.linspace(20, 120, 1001)), key=lambda c: (snap(c) - lut).square().sum().item())
    return snap(c), c


@torch.no_grad()
def gptq_nvfp4(w, hu):
    """GPTQ onto NVFP4: w [E, out, in] rotated, hu [E, in, in] upper factor of H^-1. Block scales are set from the
    error-fed weights when the column loop reaches each block of 16."""
    w = w.float().clone()
    g = (w.abs().amax((1, 2), keepdim=True) / (6 * 448)).clamp_min(1e-30)
    grid = FP4_GRID.to(w.device)
    mid = (grid[1:] + grid[:-1]) / 2
    q = torch.empty_like(w)
    for j in range(w.shape[2]):
        if j % 16 == 0:
            b = w[:, :, j : j + 16]
            s = ((b.abs().amax(2) / 6 / g[:, :, 0]).to(torch.float8_e4m3fn).float() * g[:, :, 0]).clamp_min(1e-30)
        col = w[:, :, j]
        q[:, :, j] = grid[torch.bucketize((col / s).abs().clamp_max(6), mid)] * col.sign() * s
        if j + 1 < w.shape[2]:
            err = (col - q[:, :, j]) / hu[:, j, j].unsqueeze(1)
            w[:, :, j + 1 :] -= err.unsqueeze(2) * hu[:, j, j + 1 :].unsqueeze(1)
    return q


@torch.no_grad()
def fast_ldlq_batch(ws, hs, k, seeds, blk_sz: int = 64):
    """Fast non-Viterbi proxy: Hadamard rotate + clipped uniform K-bit block-LDLQ (no 2^14 trellis search)."""
    blk = 512 if ws[0].shape[1] % 512 == 0 else 128
    rot = [ms.rotate_weight_and_hessian(w, h, block_size=blk, seed=sd) for w, h, sd in zip(ws, hs, seeds)]
    hu = torch.stack([ms.cholesky_inv_upper(r[1]) for r in rot])
    w_work = torch.stack([r[0] for r in rot]).float().clone()
    std = w_work.std(dim=2, keepdim=True).clamp(min=1e-8)
    clip = (1.5 + 0.45 * k) * std
    half_levels = (1 << (k - 1)) - 0.5
    step = clip / half_levels
    q = torch.empty_like(w_work)
    in_f = w_work.shape[2]
    for c0 in range(0, in_f, blk_sz):
        c1 = min(c0 + blk_sz, in_f)
        qc = torch.round(w_work[:, :, c0:c1] / step).clamp(-half_levels, half_levels) * step
        q[:, :, c0:c1] = qc
        if c1 < in_f:
            err = torch.linalg.solve_triangular(hu[:, c0:c1, c0:c1], w_work[:, :, c0:c1] - qc, upper=True, left=False)
            w_work[:, :, c1:] -= torch.bmm(err, hu[:, c0:c1, c1:])
    return [ms.fwht_blocks(q[i], block_size=blk) * rot[i][2].unsqueeze(0) for i in range(len(ws))]


def quant_batch(ws, hs, k, seeds, lut, mode="ms", viterbi=(32, 8)):
    """mode: ms; ldlq (fast block-64 LDLQ proxy); fp4round; e2m1; gptqfp4. All in the rotated domain."""
    if mode == "ldlq":
        return fast_ldlq_batch(ws, hs, k, seeds)
    blk = 512 if ws[0].shape[1] % 512 == 0 else 128
    rot = [ms.rotate_weight_and_hessian(w, h, block_size=blk, seed=sd) for w, h, sd in zip(ws, hs, seeds)]
    hu = torch.stack([ms.cholesky_inv_upper(r[1]) for r in rot])
    w_rot = torch.stack([r[0] for r in rot])
    if mode == "gptqfp4":
        q = gptq_nvfp4(w_rot, hu)
    elif mode == "e2m1":
        lut4, c = fp4_lut(lut)
        _, _, q = viterbi_code_batched_cuda(w_rot, hu, k, lut4, *viterbi, fp4_c=c)
    else:
        _, _, q = viterbi_code_batched_cuda(w_rot, hu, k, lut, *viterbi)
    if mode == "fp4round":
        q = torch.stack([nvfp4(qi) for qi in q])
    return [ms.fwht_blocks(q[i], block_size=blk) * rot[i][2].unsqueeze(0) for i in range(len(ws))]


@torch.no_grad()
def expert_out(xs, gu, dn):
    g, u = (xs @ gu.T).chunk(2, -1)
    return (F.silu(g) * u) @ dn.T


def quantize_experts(moe, x, idx, k, lut, experts, eb: int = 8, tau: float = 0.0, l: int = 0, errs=None, mode="ms",
                     errs_proj=None, viterbi=(32, 8)):
    """Quantize the listed experts in place; all others keep their BF16 weights. tau > 0: the converter's shrunk
    Hessians. Each expert's Hessians are built before its weights are overwritten. errs (a dict): per expert, the
    squared output error of the coded expert on its own calibration tokens (unweighted by the router). errs_proj:
    optional {"gate": {}, "up": {}, "down": {}} for each projection's isolated squared output error."""
    from ms_flashnext_convert import expert_hessians
    ex = moe.experts
    dev = ex.gate_up_proj.device   # x may be in host RAM: each expert's tokens go to the GPU on their own
    h_all = x_ref = None
    if tau > 0:   # as code_experts: all tokens' x x^T mean, and a fixed sample of tokens
        h_all = sum((c := b.to(dev).float()).T @ c for b in x.split(65536)) / x.shape[0]
        x_ref = x[torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(l))[:2048].to(x.device)].to(dev)
    gu_q, dn_q = ex.gate_up_proj.data, ex.down_proj.data
    n_ff = gu_q.shape[1] // 2
    for e0 in range(0, len(experts), eb):
        es = experts[e0 : e0 + eb]
        gus, dns, hgu, hdn = [], [], [], []
        for e in es:
            wgu = ex.gate_up_proj[e].float().clone()
            gus.append(wgu)
            dns.append(ex.down_proj[e].float().clone())
            sel = (idx == e).any(-1).nonzero().squeeze(1)
            h_gu, h_dn, _ = expert_hessians(x[sel.to(x.device)].to(dev), idx[sel], e, wgu, None, tau, h_all, x_ref)
            hgu.append(h_gu)
            hdn.append(h_dn)
        for e, q in zip(es, quant_batch(gus, hgu, k, [2 * e + 1 for e in es], lut, mode, viterbi)):
            gu_q[e] = q.to(gu_q.dtype)
        for e, q in zip(es, quant_batch(dns, hdn, k, [2 * e + 2 for e in es], lut, mode, viterbi)):
            dn_q[e] = q.to(dn_q.dtype)
        del hgu, hdn   # the Hessians are done: free them before the error pass
        if errs is not None or errs_proj is not None:
            for e, wgu, wdn in zip(es, gus, dns):
                sel = (idx == e).any(-1).nonzero().squeeze(1).to(x.device)
                err = eg = eu = ed = 0.0
                qg, qd = gu_q[e].float(), dn_q[e].float()
                if errs_proj is not None:
                    gu_gate = torch.cat([qg[:n_ff], wgu[n_ff:]], dim=0)
                    gu_up = torch.cat([wgu[:n_ff], qg[n_ff:]], dim=0)
                for c in range(0, sel.numel(), 8192):
                    xs = x[sel[c : c + 8192]].to(dev).float()
                    y0 = expert_out(xs, wgu, wdn)
                    if errs is not None:
                        d = expert_out(xs, qg, qd) - y0
                        err += (d * d).sum().item()
                    if errs_proj is not None:
                        dg = expert_out(xs, gu_gate, wdn) - y0
                        du = expert_out(xs, gu_up, wdn) - y0
                        dd = expert_out(xs, wgu, qd) - y0
                        eg += (dg * dg).sum().item()
                        eu += (du * du).sum().item()
                        ed += (dd * dd).sum().item()
                if errs is not None:
                    errs[e] = err
                if errs_proj is not None:
                    errs_proj["gate"][e] = eg
                    errs_proj["up"][e] = eu
                    errs_proj["down"][e] = ed
        torch.cuda.empty_cache()   # the error pass's blocks back before the next batch's Hessians
        if e0 % 128 == 0:
            print(f"    k {k} experts {e0}: peak alloc {torch.cuda.max_memory_allocated() / 2**30:.1f} "
                  f"reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB", flush=True)
    return gu_q, dn_q


def run_layer(layer, h, kws, batch, grab_in=None, grab_out=None, inplace=False):
    hooks = []
    if grab_in is not None:
        hooks.append(layer.mlp.register_forward_pre_hook(lambda m, a: grab_in.append(a[0].reshape(-1, a[0].shape[-1]))))
    if grab_out is not None:
        hooks.append(layer.mlp.register_forward_hook(lambda m, a, o: grab_out.append(o.reshape(-1, o.shape[-1]).float())))
    out = h if inplace else torch.empty_like(h)   # inplace: each batch is on the device before its slot is overwritten
    dev = next(layer.parameters()).device   # h may live on the host (a large calibration set): a batch at a time
    with torch.no_grad():
        for b, kw in enumerate(kws):
            out[b * batch : (b + 1) * batch] = layer(h[b * batch : (b + 1) * batch].to(dev), **kw).to(out.device)
    for hk in hooks:
        hk.remove()
    return out


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def report(out_dir: Path) -> None:
    rows = [json.loads(s) for s in (out_dir / "scan.jsonl").read_text().splitlines() if s.strip()]
    ws = [c for c in rows[0] if c[0] == "C" and c[1:].isdigit()]
    ws.sort(key=lambda c: int(c[1:]))
    # one width step costs the same bits on every layer, so rank by the drop in squared layer-output error
    lo, hi = ws[0], ws[1] if len(ws) > 1 else ws[0]
    for r in rows:
        r["gain"] = r[lo]["out_rel"] ** 2 - r[hi]["out_rel"] ** 2
    print(f"{'layer':>5} {'type':>17} " + " ".join(f"{c + ' out':>8}" for c in ws) + " "
          + " ".join(f"{c + ' moe':>8}" for c in ws) + f" {'zero-tok':>8}")
    for r in rows:
        print(f"{r['layer']:5d} {r['type']:>17} " + " ".join(f"{r[c]['out_rel']:8.5f}" for c in ws) + " "
              + " ".join(f"{r[c]['moe_rel']:8.4f}" for c in ws) + f" {r['zero_token_experts']:8d}")
    order = sorted(rows, key=lambda r: -r["gain"])
    print(f"layers by {lo}->{hi} gain:", [r["layer"] for r in order])


def main() -> None:
    p = argparse.ArgumentParser(description="Per-layer MS width sensitivity scan (Qwen3.8-Flash-Next experts)")
    p.add_argument("--hf-dir", type=Path, required=True)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--calib-seqs", type=int, default=256)
    p.add_argument("--eval-seqs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--experts", type=int, default=64, help="experts quantized per layer, spread over token counts")
    p.add_argument("--widths", type=lambda s: [int(k) for k in s.split(",")], default=[3, 4],
                   help="comma-separated trellis widths K to scan, e.g. 2,3,4,5,6")
    p.add_argument("--hess-tau", type=float, default=0.0, help="the converter's Hessian shrinkage (0: plain sums)")
    p.add_argument("--stop-after", type=int, default=None, help="last layer to scan in this run")
    p.add_argument("--mix", action="store_true", help="with --per-expert: measure the per-expert mixes too")
    p.add_argument("--expert-batch", type=int, default=8, help="experts coded per Viterbi call (fewer: less GPU memory)")
    p.add_argument("--window", type=int, default=24, help="Viterbi search window for scan (default 24)")
    p.add_argument("--commit", type=int, default=12, help="Viterbi commit length for scan (default 12)")
    p.add_argument("--fp4", action="store_true",
                   help="also: NVFP4 RTN and GPTQ, and per width the MS output rounded to NVFP4 and the trellis on the E2M1 grid")
    p.add_argument("--per-expert", action="store_true",
                   help="also: each expert's own output error at every width (experts_<layer>.json)")
    p.add_argument("--proxy", action="store_true",
                   help="use fast block-64 LDLQ proxy instead of 14-bit Viterbi for sensitivity ranking (~5.5x faster)")
    p.add_argument("--max-vram-gib", type=float, default=22.0,
                   help="hard per-process VRAM cap in GiB to prevent WDDM paging (0: off)")
    args = p.parse_args()
    assert args.widths and all(k in MS_BITS for k in args.widths), args.widths

    dev = torch.device("cuda")
    if args.max_vram_gib > 0:
        tot_vram = torch.cuda.get_device_properties(dev).total_memory
        frac = min(1.0, (args.max_vram_gib * (1 << 30)) / tot_vram)
        torch.cuda.set_per_process_memory_fraction(frac, dev)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tc = json.loads((args.hf_dir / "config.json").read_text())["text_config"]
    cfg = Qwen4ExpTextConfig(**tc)
    wmap = json.loads((args.hf_dir / "model.safetensors.index.json").read_text())["weight_map"]
    calib = torch.load(args.calib, map_location="cpu", weights_only=True)
    bs = args.batch_size
    assert args.calib_seqs % bs == 0 and args.eval_seqs % bs == 0

    cal_ids, ev_ids = split_calib(calib, args.calib_seqs, args.eval_seqs)
    host = cal_ids.numel() * cfg.hc_count * cfg.hidden_size * 2 > (4 << 30)   # as the converter: a large stream in RAM
    h_cal, kw_cal = prelude(cfg, args.hf_dir, wmap, cal_ids, dev, bs, host=host)
    h_ev, kw_ev = prelude(cfg, args.hf_dir, wmap, ev_ids, dev, bs)
    state = args.out_dir / "state.pt"
    start = 0
    if state.exists():
        st = torch.load(state, map_location="cpu", weights_only=True)
        start, h_cal, h_ev = st["next_layer"], st["h_cal"].to("cpu" if host else dev), st["h_ev"].to(dev)
        print(f"resuming at layer {start}", flush=True)
    lut = ms.build_1mad_lut(dev)
    last = cfg.num_hidden_layers - 1 if args.stop_after is None else args.stop_after

    for l in range(start, last + 1):
        t0 = time.time()
        layer = load_layer(cfg, args.hf_dir, wmap, l, dev)
        t_load = time.time() - t0
        # the MoE inputs go straight into one buffer (in host RAM for a large set): a list plus torch.cat holds them twice
        x = torch.empty(cal_ids.numel(), cfg.hidden_size, dtype=torch.bfloat16, device="cpu" if host else dev)
        fill = [0]

        def grab(_m, a):
            t = a[0].reshape(-1, a[0].shape[-1])
            x[fill[0] : fill[0] + t.shape[0]] = t
            fill[0] += t.shape[0]
        hk = layer.mlp.register_forward_pre_hook(grab)
        h_cal = run_layer(layer, h_cal, kw_cal, bs, inplace=host)
        hk.remove()
        assert fill[0] == x.shape[0], (fill[0], x.shape)
        torch.cuda.empty_cache()
        if x.device != dev:
            free_b, _ = torch.cuda.mem_get_info(dev)
            if free_b > x.numel() * x.element_size() + (5 << 30):
                x = x.to(dev)
        moe = layer.mlp
        with torch.no_grad():   # in chunks, as the converter: only the top-k ids stay
            idx = torch.cat([moe.gate(c.to(dev))[2] for c in x.split(65536)])
        counts = torch.bincount(idx.flatten(), minlength=moe.experts.num_experts)
        y_ref, xe_ev = [], []
        h_ref = run_layer(layer, h_ev, kw_ev, bs, grab_in=xe_ev, grab_out=y_ref)
        y_ref = torch.cat(y_ref)
        xe_ev = torch.cat(xe_ev)
        with torch.no_grad():
            shared = (torch.sigmoid(moe.shared_expert_gate(xe_ev)) * moe.shared_expert(xe_ev)).float()
        routed_ref = y_ref - shared
        torch.cuda.empty_cache()   # the forward's cached blocks back to the driver before the coder
        rec = {
            "layer": l,
            "type": cfg.layer_types[l],
            "tokens_per_expert": [counts.min().item(), counts.median().item(), counts.max().item()],
            "zero_token_experts": int((counts == 0).sum().item()),
            "experts_quantized": args.experts,
        }
        # a ranking needs the same sample rule on every layer, not every expert: every n-th by token count
        order = torch.argsort(counts).tolist()
        sample = order[:: max(1, len(order) // args.experts)][: args.experts]
        t_fwd = time.time() - t0 - t_load
        gu_ref, dn_ref = moe.experts.gate_up_proj.data.cpu(), moe.experts.down_proj.data.cpu()   # originals in RAM

        def measure(plan, name, mode="ms"):
            """Code each width's experts, measure on the held-out tokens, put the BF16 weights back.
            k 0: plain NVFP4 round-to-nearest of the BF16 weights."""
            tq = time.time()
            for k, es in plan.items():
                if k == 0:
                    for e in es:
                        moe.experts.gate_up_proj.data[e] = nvfp4(moe.experts.gate_up_proj[e]).to(gu_ref.dtype)
                        moe.experts.down_proj.data[e] = nvfp4(moe.experts.down_proj[e]).to(dn_ref.dtype)
                    continue
                per_e = name[0] == "C" and args.per_expert and mode in ("ms", "ldlq")
                quantize_experts(moe, x, idx, k, lut, es, eb=args.expert_batch, tau=args.hess_tau, l=l, mode=mode,
                                 errs=errs.setdefault(k, {}) if per_e else None,
                                 errs_proj={p: errs_proj[p].setdefault(k, {}) for p in ("gate", "up", "down")} if per_e else None,
                                 viterbi=(args.window, args.commit))
            y_q = []
            h_q = run_layer(layer, h_ev, kw_ev, bs, grab_out=y_q)
            y_q = torch.cat(y_q)
            moe.experts.gate_up_proj.data.copy_(gu_ref)
            moe.experts.down_proj.data.copy_(dn_ref)
            rec[name] = {
                "routed_rel": ((y_q - y_ref).norm() / routed_ref.norm()).item(),
                "moe_rel": rel(y_q, y_ref),
                "out_rel": rel(h_q.float(), h_ref.float()),
                "seconds": round(time.time() - tq, 1),
            }
            print(f"  {name}: out {rec[name]['out_rel']:.5f} moe {rec[name]['moe_rel']:.4f}, {rec[name]['seconds']}s, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

        errs: dict = {}
        errs_proj: dict = {"gate": {}, "up": {}, "down": {}}
        print(f"layer {l}: forward done, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        scan_mode = "ldlq" if args.proxy else "ms"
        for k in args.widths:
            measure({k: sample}, f"C{k}", scan_mode)
        if args.fp4:
            measure({0: sample}, "NVFP4")
            for k in args.widths:
                measure({k: sample}, f"C{k}fp4", "fp4round")
                measure({k: sample}, f"C{k}e2m1", "e2m1")
            measure({4: sample}, "GPTQfp4", "gptqfp4")
        if args.per_expert:
            from ms_plan_solver import layer_bytes, solve
            size = lambda k: layer_bytes(k, n_exp=1)
            (args.out_dir / f"experts_{l}.json").write_text(
                json.dumps({"counts": counts.tolist(), "errs": errs, "errs_proj": errs_proj}))
            for k0 in args.widths[1:-1] if args.mix else []:   # a mix needs a narrower and a wider width than its budget
                pick = solve({e: {k: errs[k][e] for k in args.widths} for e in sample}, len(sample) * size(k0), size)
                plan = {k: [e for e in sample if pick[e] == k] for k in args.widths}
                measure({k: es for k, es in plan.items() if es}, f"mix{k0}")
                rec[f"mix{k0}"]["experts_per_k"] = {k: len(es) for k, es in plan.items()}
        h_ev = h_ref
        del layer, moe, x, idx, gu_ref, dn_ref
        torch.cuda.empty_cache()
        with open(args.out_dir / "scan.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        torch.save({"next_layer": l + 1, "h_cal": h_cal, "h_ev": h_ev}, state)
        names = [n for n in rec if n[0] in "CmN" and isinstance(rec[n], dict)]
        widths = " | ".join(f"{n} out {rec[n]['out_rel']:.5f} moe {rec[n]['moe_rel']:.4f}" for n in names)
        quant = sum(rec[n]["seconds"] for n in names)
        print(f"layer {l:2d} {rec['type']:>17}: {widths} | zero-token experts "
              f"{rec['zero_token_experts']} | load {t_load:.0f}s fwd {t_fwd:.0f}s quant {quant:.0f}s total {time.time() - t0:.0f}s", flush=True)

    report(args.out_dir)


if __name__ == "__main__":
    main()
