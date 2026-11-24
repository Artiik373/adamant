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
    hs, kws = [], []
    with torch.no_grad():
        for i in range(0, ids.shape[0], batch):
            try:
                model(input_ids=ids[i : i + batch].to(dev, torch.long), use_cache=False)
            except Stop:
                pass
            h, kw = got.pop()
            hs.append(h.cpu() if host else h)
            kws.append({n: v for n, v in kw.items() if n != "past_key_values"})
    del model
    return torch.cat(hs), kws


def quant_batch(ws, hs, k, seeds, lut):
    blk = 512 if ws[0].shape[1] % 512 == 0 else 128
    rot = [ms.rotate_weight_and_hessian(w, h, block_size=blk, seed=sd) for w, h, sd in zip(ws, hs, seeds)]
    hu = torch.stack([ms.cholesky_inv_upper(r[1]) for r in rot])
    _, _, q = viterbi_code_batched_cuda(torch.stack([r[0] for r in rot]), hu, k, lut)
    return [ms.fwht_blocks(q[i], block_size=blk) * rot[i][2].unsqueeze(0) for i in range(len(ws))]


@torch.no_grad()
def quantize_experts(moe, x, idx, k, lut, experts, eb: int = 8, tau: float = 0.0, l: int = 0):
    """Quantize the listed experts in place; all others keep their BF16 weights. tau > 0: the converter's shrunk
    Hessians. Each expert's Hessians are built before its weights are overwritten."""
    from ms_flashnext_convert import expert_hessians
    ex = moe.experts
    dev = ex.gate_up_proj.device   # x may be in host RAM: each expert's tokens go to the GPU on their own
    h_all = x_ref = None
    if tau > 0:   # as code_experts: all tokens' x x^T mean, and a fixed sample of tokens
        h_all = sum((c := b.to(dev).float()).T @ c for b in x.split(65536)) / x.shape[0]
        x_ref = x[torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(l))[:2048].to(x.device)].to(dev)
    gu_q, dn_q = ex.gate_up_proj.data, ex.down_proj.data
    for e0 in range(0, len(experts), eb):
        es = experts[e0 : e0 + eb]
        gus, dns, hgu, hdn = [], [], [], []
        for e in es:
            wgu = ex.gate_up_proj[e].float()
            gus.append(wgu)
            dns.append(ex.down_proj[e].float())
            sel = (idx == e).any(-1).nonzero().squeeze(1)
            h_gu, h_dn, _ = expert_hessians(x[sel.to(x.device)].to(dev), idx[sel], e, wgu, None, tau, h_all, x_ref)
            hgu.append(h_gu)
            hdn.append(h_dn)
        for e, q in zip(es, quant_batch(gus, hgu, k, [2 * e + 1 for e in es], lut)):
            gu_q[e] = q.to(gu_q.dtype)
        for e, q in zip(es, quant_batch(dns, hdn, k, [2 * e + 2 for e in es], lut)):
            dn_q[e] = q.to(dn_q.dtype)
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
    args = p.parse_args()
    assert args.widths and all(k in MS_BITS for k in args.widths), args.widths

    dev = torch.device("cuda")
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
        xs = []
        h_cal = run_layer(layer, h_cal, kw_cal, bs, grab_in=xs, inplace=host)
        x = torch.cat(xs)
        del xs
        moe = layer.mlp
        with torch.no_grad():   # in chunks, as the converter: only the top-k ids stay
            idx = torch.cat([moe.gate(c)[2] for c in x.split(65536)])
        if host:
            x = x.cpu()
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
        for k in args.widths:
            tq = time.time()
            gu_q, dn_q = quantize_experts(moe, x, idx, k, lut, sample, tau=args.hess_tau, l=l)
            y_q = []
            h_q = run_layer(layer, h_ev, kw_ev, bs, grab_out=y_q)
            y_q = torch.cat(y_q)
            moe.experts.gate_up_proj.data.copy_(gu_ref)
            moe.experts.down_proj.data.copy_(dn_ref)
            del gu_q, dn_q
            rec[f"C{k}"] = {
                "routed_rel": ((y_q - y_ref).norm() / routed_ref.norm()).item(),
                "moe_rel": rel(y_q, y_ref),
                "out_rel": rel(h_q.float(), h_ref.float()),
                "seconds": round(time.time() - tq, 1),
            }
        h_ev = h_ref
        del layer, moe, x, idx, gu_ref, dn_ref
        torch.cuda.empty_cache()
        with open(args.out_dir / "scan.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        torch.save({"next_layer": l + 1, "h_cal": h_cal, "h_ev": h_ev}, state)
        widths = " | ".join(f"C{k} out {rec[f'C{k}']['out_rel']:.5f} moe {rec[f'C{k}']['moe_rel']:.4f}" for k in args.widths)
        quant = sum(rec[f"C{k}"]["seconds"] for k in args.widths)
        print(f"layer {l:2d} {rec['type']:>17}: {widths} | zero-token experts "
              f"{rec['zero_token_experts']} | load {t_load:.0f}s fwd {t_fwd:.0f}s quant {quant:.0f}s total {time.time() - t0:.0f}s", flush=True)

    report(args.out_dir)


if __name__ == "__main__":
    main()
