"""Score converted Qwen3.8-Flash-Next MS records against a BF16 KL-divergence base, the way the quality gate does.

Reads the base file (tools/bf16_reference.py, include/basalt/program/ppl_gate.hpp): the chunk tokens and, per chunk,
the BF16 log-probs of rows n_ctx/2 .. n_ctx-2. Streams the model one layer at a time with every coded matrix decoded
from its MS record (routed experts, dense spine, head); every other tensor is the BF16 original. Prints PPL, mean KLD
and top-1 agreement over the same rows.

    python tools/convert/ms_flashnext_eval.py --hf-dir C:/llm/models/Qwen3.8-Flash-Next --ms-dir G:/llm/ms_fn \
        --base G:/llm/adamant/data/flashnext-bf16-base.kld [--chunks 100]

With `--write-base OUT --text FILE` and no `--ms-dir` it runs the BF16 original instead and writes that base (the
format and row quantization of tools/bf16_reference.py) from `--chunks` chunks of `--ctx` tokens of the text.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bf16_reference import quantize_row, tokenize  # noqa: E402

import ms_moe_converter as ms  # noqa: E402
import model_sections as msec  # noqa: E402
from ms_convert_dense_spine import largest_pow2_divisor
from ms_flashnext_convert import SPINE
from ms_pack import ms_row_bytes
from ms_sensitivity_scan import PRE, load_layer, prelude, run_layer


def read_base(path: Path, n_chunks: int | None):
    """Tokens [n_chunk, n_ctx] and a reader of each chunk's scored rows as fp32 log-probs [n_ctx/2 - 1, n_vocab]."""
    f = open(path, "rb")
    assert f.read(8) == b"_logits_", f"{path}: not a KL-divergence base"
    n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", f.read(12))
    toks = np.frombuffer(f.read(4 * n_ctx * n_chunk), dtype=np.int32).reshape(n_chunk, n_ctx)
    rows = n_ctx // 2 - 1
    row_bytes = 8 + 2 * (n_vocab + n_vocab % 2)
    data0 = f.tell()

    def chunk_logprobs(i: int, dev) -> torch.Tensor:
        f.seek(data0 + i * rows * row_bytes)
        raw = np.frombuffer(f.read(rows * row_bytes), dtype=np.uint8).reshape(rows, row_bytes)
        scale_min = torch.from_numpy(raw[:, :8].copy().view(np.float32)).to(dev)
        q = torch.from_numpy(raw[:, 8 : 8 + 2 * n_vocab].copy().view(np.uint16).astype(np.int32)).to(dev)
        return q.float() * scale_min[:, :1] + scale_min[:, 1:2]

    n = n_chunk if n_chunks is None else min(n_chunk, n_chunks)
    return n_ctx, n_vocab, torch.from_numpy(toks[:n].astype(np.int64)), chunk_logprobs


def decode_rows(payload: torch.Tensor, n_out: int, n_in: int, k: int, lut: torch.Tensor) -> torch.Tensor:
    """W [n_out, n_in] from one MS payload (rows, then int8 signs) on the GPU."""
    rb = ms_row_bytes(n_in, k)
    buf = payload[: n_out * rb].view(n_out, rb)
    signs = payload[n_out * rb : n_out * rb + n_in].view(torch.int8).float()
    scale = buf[:, 0:2].contiguous().view(torch.float16).float().reshape(n_out)
    state = buf[:, 2:4].contiguous().view(torch.int16).to(torch.int64).reshape(n_out) & ms.STATE_MASK
    nbits = (n_in - 1) * k
    bits = (buf[:, 4 : 4 + (nbits + 7) // 8].unsqueeze(-1) >> torch.arange(7, -1, -1, device=buf.device, dtype=torch.uint8)) & 1
    bits = bits.reshape(n_out, -1)[:, :nbits].reshape(n_out, n_in - 1, k).to(torch.int64)
    sym = (bits * (1 << torch.arange(k - 1, -1, -1, device=buf.device))).sum(-1)
    del bits
    # the state after c symbols holds the last 14 bits of (s0, sym_1 .. sym_c): build it from a sliding window
    per = (ms.STATE_BITS + k - 1) // k
    hist = torch.cat([torch.zeros(n_out, per - 1, dtype=torch.int64, device=buf.device), sym], 1)
    states = torch.empty(n_out, n_in, dtype=torch.int64, device=buf.device)
    states[:, 0] = state
    acc = torch.zeros(n_out, n_in - 1, dtype=torch.int64, device=buf.device)
    for j in range(per):  # acc_c = sum over the last `per` symbols, oldest first
        acc = (acc << k) | hist[:, j : j + n_in - 1]
    # symbols older than the window come from s0: shift s0 in front where the window reaches before symbol 1
    for c in range(1, min(per, n_in)):
        acc[:, c - 1] |= state << (k * c)
    states[:, 1:] = acc & ms.STATE_MASK
    w_rot = lut[states] * scale.unsqueeze(1)
    return ms.fwht_blocks(w_rot, block_size=largest_pow2_divisor(n_in, cap=512)) * signs.unsqueeze(0)


def read_payload(path: Path, offset: int, nbytes: int, dev) -> torch.Tensor:
    with open(path, "rb") as f:
        f.seek(offset)
        return torch.frombuffer(bytearray(f.read(nbytes)), dtype=torch.uint8).to(dev)


def load_ms_layer(layer, l: int, ms_dir: Path, lut, dev) -> None:
    """Replace the layer's coded matrices with their decoded MS records (spine back from the GGUF order)."""
    mods = dict(layer.named_modules())
    for hf_name, suffix in SPINE.items():
        if hf_name not in mods:
            continue
        rec = json.loads((ms_dir / f"blk.{l}.{suffix}.json").read_text())
        n_in, n_out = rec["shape"]
        k = rec["k"]
        w = decode_rows(read_payload(ms_dir / rec["file"], 0, n_out * ms_row_bytes(n_in, k) + n_in, dev), n_out, n_in, k, lut)
        if suffix in ("attn_qkv.weight", "attn_gate.weight"):
            idx = torch.as_tensor(msec.gdn_index(n_out, msec.GDN_QK_ROWS if n_out > msec.GDN_V_HEADS * msec.GDN_HEAD else 0), device=dev)
            hf = torch.empty_like(w)
            hf[idx] = w
            w = hf
        elif suffix == "ssm_out.weight":
            idx = torch.as_tensor(msec.gdn_index(n_in, 0), device=dev)
            hf = torch.empty_like(w)
            hf[:, idx] = w
            w = hf
        mods[hf_name].weight.data.copy_(w.to(mods[hf_name].weight.dtype))
    ex = layer.mlp.experts
    rec = json.loads((ms_dir / "experts" / f"blk.{l}.experts.json").read_text())
    k = rec["k"]
    n_embd, n_ff = ex.hidden_dim, ex.intermediate_dim
    for name, n_in, n_out in (("gate", n_embd, n_ff), ("up", n_embd, n_ff), ("down", n_ff, n_embd)):
        per = n_out * ms_row_bytes(n_in, k) + n_in
        path = ms_dir / "experts" / f"blk.{l}.ffn_{name}_exps.ms"
        for e0 in range(0, ex.num_experts, 64):
            blob = read_payload(path, e0 * per, 64 * per, dev)
            for e in range(64):
                w = decode_rows(blob[e * per : (e + 1) * per], n_out, n_in, k, lut).to(ex.down_proj.dtype)
                if name == "gate":
                    ex.gate_up_proj.data[e0 + e, :n_ff] = w
                elif name == "up":
                    ex.gate_up_proj.data[e0 + e, n_ff:] = w
                else:
                    ex.down_proj.data[e0 + e] = w


def load_mixer(cfg, hf: Path, wmap: dict, dev):
    with torch.device("meta"):
        mixer = Qwen4ExpTextGatedResidual(cfg, use_combine=False)
    pre = f"{PRE}hyper_connection_mixer."
    sd = {}
    for key in [x for x in wmap if x.startswith(pre)]:
        with safe_open(hf / wmap[key], "pt") as s:
            sd[key[len(pre):]] = s.get_tensor(key)
    mixer.load_state_dict(sd, assign=True)
    return mixer.to(dev, torch.bfloat16)


def bf16_head(hf: Path, wmap: dict, dev):
    with safe_open(hf / wmap["lm_head.weight"], "pt") as s:
        return s.get_tensor("lm_head.weight").to(dev, torch.bfloat16)


def ms_head(ms_dir: Path, lut, dev):
    rec = json.loads((ms_dir / "output.weight.json").read_text())
    n_in, n_out = rec["shape"]
    head = torch.empty(n_out, n_in, dtype=torch.bfloat16, device=dev)
    rb = ms_row_bytes(n_in, rec["k"])
    signs = read_payload(ms_dir / rec["file"], n_out * rb, n_in, dev)
    for r0 in range(0, n_out, 16384):
        r1 = min(r0 + 16384, n_out)
        rows = read_payload(ms_dir / rec["file"], r0 * rb, (r1 - r0) * rb, dev)
        head[r0:r1] = decode_rows(torch.cat([rows, signs]), r1 - r0, n_in, rec["k"], lut).to(torch.bfloat16)
    return head


def eval_cases(args, cfg, wmap, dev, lut, prefix: int = 55) -> None:
    """KL(BF16 || MS) per case of a corpus file ({"cases": [{"id", "token_ids"}]}), every position scored, over the
    valid token ids only, FP64 normalization. Both models run here, the BF16 one first on each streamed layer.
    `wiki2_*` cases are also scored without their chat-template prefix rows, as their published results are."""
    cases = json.loads(args.cases.read_text())["cases"]
    vocab = torch.tensor(json.loads(args.vocab.read_text())["valid_token_ids"], dtype=torch.long, device=dev)
    t0 = time.time()
    states = []
    for c in cases:
        ids = torch.tensor(c["token_ids"], dtype=torch.long).unsqueeze(0)
        h, kws = prelude(cfg, args.hf_dir, wmap, ids, dev, 1)
        states.append([h, h.clone(), kws])
    for l in range(cfg.num_hidden_layers):
        tl = time.time()
        layer = load_layer(cfg, args.hf_dir, wmap, l, dev)
        for st in states:
            st[0] = run_layer(layer, st[0], st[2], 1)
        load_ms_layer(layer, l, args.ms_dir, lut, dev)
        for st in states:
            st[1] = run_layer(layer, st[1], st[2], 1)
        del layer
        torch.cuda.empty_cache()
        print(f"layer {l:2d}: {time.time() - tl:.0f}s", flush=True)
    mixer = load_mixer(cfg, args.hf_dir, wmap, dev)
    heads = (bf16_head(args.hf_dir, wmap, dev), ms_head(args.ms_dir, lut, dev))
    all_kl, wiki_kl, all_top, wiki_top, case_means = [], [], [], [], []
    with torch.no_grad():
        for c, st in zip(cases, states):
            kl_c, top_c = [], []
            for r0 in range(0, st[0].shape[1], 512):
                lp = [torch.log_softmax(((mixer(s[:, r0 : r0 + 512]) @ hd.T)[0][:, vocab]).double(), dim=-1)
                      for s, hd in zip(st[:2], heads)]
                kl_c.append((lp[0].exp() * (lp[0] - lp[1])).sum(-1).clamp(min=0).float().cpu())
                top_c.append((lp[0].argmax(-1) == lp[1].argmax(-1)).cpu())
            kl_c, top_c = torch.cat(kl_c), torch.cat(top_c)
            all_kl.append(kl_c)
            all_top.append(top_c)
            case_means.append(kl_c.mean().item())
            if c["id"].startswith("wiki2_"):
                wiki_kl.append(kl_c[prefix:])
                wiki_top.append(top_c[prefix:])
            print(f"  {c['id']:>16}: KL {kl_c.mean().item():.5f}  top-1 {100 * top_c.float().mean().item():.2f}%  "
                  f"({kl_c.numel()} rows)", flush=True)
    a, t = torch.cat(all_kl), torch.cat(all_top).float()
    print(f"\n=== MS records {args.ms_dir} on {args.cases.name} ===\n"
          f"all rows       KL {a.mean().item():.5f}  median {a.median().item():.5f}  top-1 {100 * t.mean().item():.2f}%  "
          f"({a.numel()} rows)\nmacro by case  KL {np.mean(case_means):.5f}")
    if wiki_kl:
        w, wt = torch.cat(wiki_kl), torch.cat(wiki_top).float()
        print(f"wiki2 scored   KL {w.mean().item():.5f}  top-1 {100 * wt.mean().item():.2f}%  ({w.numel()} rows, "
              f"prefix {prefix} dropped)")
    print(f"time {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3.8-Flash-Next MS records against a BF16 KLD base")
    p.add_argument("--hf-dir", type=Path, required=True)
    p.add_argument("--ms-dir", type=Path, help="the converter's records; omit with --write-base")
    p.add_argument("--base", type=Path, help="the BF16 base to score against")
    p.add_argument("--write-base", type=Path, help="run the BF16 original and write its base here")
    p.add_argument("--text", type=Path, help="the text for --write-base")
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--chunks", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--cases", type=Path, help="a corpus of token cases to score both models on, with --ms-dir")
    p.add_argument("--vocab", type=Path, help='{"valid_token_ids": [...]} for --cases')
    args = p.parse_args()

    dev = torch.device("cuda")
    tc = json.loads((args.hf_dir / "config.json").read_text())["text_config"]
    cfg = Qwen4ExpTextConfig(**tc)
    wmap = json.loads((args.hf_dir / "model.safetensors.index.json").read_text())["weight_map"]
    if args.cases:
        if not (args.ms_dir and args.vocab) or args.base or args.write_base:
            p.error("--cases needs --ms-dir and --vocab, and no --base or --write-base")
        eval_cases(args, cfg, wmap, dev, ms.build_1mad_lut(dev))
        return
    writing = args.write_base is not None
    if writing == (args.ms_dir is not None) or writing != (args.text is not None) or writing == (args.base is not None):
        p.error("either --ms-dir with --base, or --write-base with --text")
    if writing:
        ids = tokenize(str(args.hf_dir), str(args.text))
        n_ctx = args.ctx
        n_chunk = min(args.chunks or 100, len(ids) // n_ctx)
        toks = torch.tensor(ids[: n_chunk * n_ctx], dtype=torch.long).reshape(n_chunk, n_ctx)
        print(f"{len(ids)} tokens, {n_chunk} chunks of {n_ctx}", flush=True)
    else:
        n_ctx, n_vocab, toks, chunk_logprobs = read_base(args.base, args.chunks)
    bs = args.batch_size
    assert toks.shape[0] % bs == 0, "chunks must be a multiple of the batch size"
    lut = ms.build_1mad_lut(dev)
    t0 = time.time()
    h, kws = prelude(cfg, args.hf_dir, wmap, toks, dev, bs)
    for l in range(cfg.num_hidden_layers):
        tl = time.time()
        layer = load_layer(cfg, args.hf_dir, wmap, l, dev)
        if not writing:
            load_ms_layer(layer, l, args.ms_dir, lut, dev)
        td = time.time() - tl
        h = run_layer(layer, h, kws, bs)
        del layer
        torch.cuda.empty_cache()
        print(f"layer {l:2d}: decode {td:.0f}s forward {time.time() - tl - td:.0f}s", flush=True)

    mixer = load_mixer(cfg, args.hf_dir, wmap, dev)
    head = bf16_head(args.hf_dir, wmap, dev) if writing else ms_head(args.ms_dir, lut, dev)

    lo, hi = n_ctx // 2, n_ctx - 1
    nll = kld = 0.0
    top1 = rows_n = 0
    out = open(args.write_base, "wb") if writing else None
    if writing:
        out.write(b"_logits_")
        out.write(struct.pack("<Iii", n_ctx, head.shape[0], toks.shape[0]))
        out.write(toks.to(torch.int32).numpy().tobytes())
    with torch.no_grad():
        for i in range(toks.shape[0]):
            x = mixer(h[i : i + 1, lo:hi])
            logits = (x @ head.T).float()[0]
            lp = torch.log_softmax(logits, dim=-1)
            tgt = toks[i, lo + 1 : hi + 1].to(dev)
            if writing:
                for r in range(logits.shape[0]):
                    scale, low, q = quantize_row(logits[r])
                    out.write(struct.pack("<ff", scale, low))
                    out.write(q.tobytes())
                    if head.shape[0] % 2:
                        out.write(bytes(2))
                nll -= lp.gather(1, tgt[:, None]).sum().item()
                rows_n += lp.shape[0]
                continue
            ref = chunk_logprobs(i, dev)
            nll -= lp.gather(1, tgt[:, None]).sum().item()
            kld += (ref.exp() * (ref - lp)).sum().item()
            top1 += (ref.argmax(-1) == lp.argmax(-1)).sum().item()
            rows_n += lp.shape[0]
    if writing:
        out.close()
        print(f"wrote {args.write_base}: {toks.shape[0]} chunks, {rows_n} rows, BF16 PPL {np.exp(nll / rows_n):.4f}, "
              f"{time.time() - t0:.0f}s", flush=True)
        return
    print(f"\n=== MS records {args.ms_dir} on {args.base.name} ({toks.shape[0]} chunks, {rows_n} rows) ===\n"
          f"PPL       {np.exp(nll / rows_n):.4f}\nKLD mean  {kld / rows_n:.5f}\ntop-1     {100 * top1 / rows_n:.2f}%\n"
          f"time      {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
