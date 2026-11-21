"""Deterministic calibration pool builder for Qwen3.8-Flash-Next MS conversion.

Builds a [N_seq, 512] int32 tensor (default 2,048 x 512 = 1.05M tokens, or 4,096 x 512 = 2.1M)
mixing FineWeb-Edu, Wikipedia, The Stack Smol (30 languages), OpenWebMath, and zero-repeat
on-policy `<think>...</think>` trajectories, and writes a SHA-256 manifest.

Usage:
  python tools/convert/build_calib_pool.py \
      --tokenizer Qwen/Qwen3.8-Flash-Next \
      --think-jsonl data/selfgen_think.jsonl \
      --out calib_2m.pt --total-seqs 4096

The on-policy chat mix: Flash-Next's own generations in its chat template (ProCreations/Qwen3.8-Flash-Next-DSpine-
generations: prompts of 16 sources rendered with each row's tools and thinking settings, then the stored response
token ids; Vemloir/Qwen3.8-Flash-Next-traces: agent and chat sessions already tokenized), plus a share of an existing
web pool, shuffled so that evenly spaced picks follow the mix.  The DSpine validation rows go to `--eval-out`.

  python tools/convert/build_calib_pool.py --tokenizer C:/llm/models/Qwen3.8-Flash-Next \
      --dspine data/dl_dspine --vemloir data/dl_vemloir --base-pool G:/llm/calib/calib_2m.pt \
      --out G:/llm/calib/calib_chat.pt --eval-out G:/llm/calib/eval_chat.pt --total-seqs 4096
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from transformers import AutoTokenizer


def pack_token_stream(
    texts: Iterable[str],
    tokenizer,
    max_seqs: int,
    seq_len: int = 512,
    doc_cap: int = 1536,
) -> torch.Tensor:
    """Tokenize documents, cap each at `doc_cap` tokens, join with '\n\n', and slice into [N, seq_len]."""
    sep_ids: List[int] = tokenizer.encode("\n\n", add_special_tokens=False)
    buf: List[int] = []
    rows: List[List[int]] = []

    for text in texts:
        if len(rows) >= max_seqs:
            break
        if not text or not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)[:doc_cap]
        buf.extend(ids)
        buf.extend(sep_ids)
        while len(buf) >= seq_len and len(rows) < max_seqs:
            rows.append(buf[:seq_len])
            del buf[:seq_len]

    if not rows:
        return torch.empty((0, seq_len), dtype=torch.int32)
    return torch.tensor(rows, dtype=torch.int32)


def iter_jsonl_texts(path: Path, field: str = "text") -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if field in obj:
                yield str(obj[field])
            elif "content" in obj:
                yield str(obj["content"])


def iter_hf_dataset(repo_id: str, config: Optional[str], split: str, field: str, limit_docs: int) -> Iterable[str]:
    from datasets import load_dataset

    ds = load_dataset(repo_id, config, split=split, streaming=True)
    for idx, row in enumerate(ds):
        if idx >= limit_docs:
            break
        yield str(row.get(field, ""))


def windows(ids: List[int], seq_len: int, cap: int) -> List[List[int]]:
    """Up to `cap` whole seq_len windows of one token stream, from its start."""
    return [ids[i:i + seq_len] for i in range(0, len(ids) - seq_len + 1, seq_len)][:cap]


def dspine_streams(folder: Path, tok, validation: bool) -> dict:
    """{source: [token ids of prompt + response]} of the DSpine rows (validation or training)."""
    import pyarrow.parquet as pq

    out: dict = {}
    for f in sorted((folder / "data").glob("*.parquet")):
        for r in pq.read_table(f).to_pylist():
            if bool(r["is_validation"]) != validation:
                continue
            kw = json.loads(r["chat_template_kwargs"]) if r.get("chat_template_kwargs") else {}
            tools = json.loads(r["tools"]) if r.get("tools") else None
            msgs = json.loads(r["prompt_messages_full_json"])
            prompt = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True, **kw)
            if hasattr(prompt, "keys"):   # a BatchEncoding
                prompt = prompt["input_ids"]
            out.setdefault(r["source"], []).append(list(prompt) + list(r["output_token_ids"]))
    return out


def round_robin(groups: List[List[List[int]]], n: int) -> List[List[int]]:
    """n rows taken one group at a time, so no group dominates while the others last."""
    rows, at = [], [0] * len(groups)
    while len(rows) < n and any(at[g] < len(groups[g]) for g in range(len(groups))):
        for g in range(len(groups)):
            if at[g] < len(groups[g]) and len(rows) < n:
                rows.append(groups[g][at[g]])
                at[g] += 1
    return rows


def chat_pool(args, tok) -> None:
    g = torch.Generator().manual_seed(0)
    n_web = int(args.total_seqs * args.web_frac)
    n_vem = int(args.total_seqs * args.vemloir_frac) if args.vemloir else 0
    n_dsp = args.total_seqs - n_web - n_vem
    manifest = {"seq_len": args.seq_len, "pools": []}
    parts = []

    by_src = dspine_streams(args.dspine, tok, validation=False)
    groups = [[w for ids in v for w in windows(ids, args.seq_len, 3)] for _, v in sorted(by_src.items())]
    for grp in groups:
        perm = torch.randperm(len(grp), generator=g).tolist()
        grp[:] = [grp[i] for i in perm]
    rows = round_robin(groups, n_dsp)
    parts.append(torch.tensor(rows, dtype=torch.int32))
    manifest["pools"].append({"name": "dspine", "seqs": len(rows), "sources": len(groups)})

    if args.vemloir:
        vg = []
        for part in ("agents", "chat"):
            sess = []
            with (args.vemloir / part / "data.jsonl").open(encoding="utf-8") as f:
                for line in f:
                    ids = json.loads(line)["input_ids"]
                    ids = json.loads(ids) if isinstance(ids, str) else ids
                    sess.append(windows(ids, args.seq_len, 16))
            vg.append([w for s_ in sess for w in s_])
        rows = round_robin(vg, n_vem)
        parts.append(torch.tensor(rows, dtype=torch.int32))
        manifest["pools"].append({"name": "vemloir", "seqs": len(rows)})

    base = torch.load(args.base_pool, map_location="cpu", weights_only=True)
    if args.seq_len > base.shape[1]:   # the pool was packed from one stream: consecutive rows join into longer windows
        m = args.seq_len // base.shape[1]
        base = base[: base.shape[0] // m * m].reshape(-1, m * base.shape[1])
    pick = torch.linspace(0, base.shape[0] - 1, n_web).round().long()   # evenly over its sources
    parts.append(base[pick, :args.seq_len].to(torch.int32))
    manifest["pools"].append({"name": f"web ({args.base_pool.name})", "seqs": n_web})

    calib = torch.cat(parts)
    calib = calib[torch.randperm(calib.shape[0], generator=g)].contiguous()
    torch.save(calib, args.out)
    manifest["total_seqs"] = int(calib.shape[0])
    manifest["total_tokens"] = int(calib.numel())
    manifest["sha256"] = hashlib.sha256(calib.numpy().tobytes()).hexdigest()
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {tuple(calib.shape)} -> {args.out}: {manifest['pools']}")

    if args.eval_out:
        ev = dspine_streams(args.dspine, tok, validation=True)
        rows = round_robin([[ids[:args.seq_len] for ids in v if len(ids) >= args.seq_len] for _, v in sorted(ev.items())],
                           args.eval_seqs)
        t = torch.tensor(rows, dtype=torch.int32)
        torch.save(t, args.eval_out)
        print(f"Saved {tuple(t.shape)} held-out DSpine validation windows -> {args.eval_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed + on-policy calibration pool (.pt)")
    parser.add_argument("--tokenizer", type=str, required=True, help="HF model dir or repo ID")
    parser.add_argument("--think-jsonl", type=Path, default=None, help="Optional on-policy <think> JSONL")
    parser.add_argument("--out", type=Path, required=True, help="Output .pt path")
    parser.add_argument("--total-seqs", type=int, default=2048, help="Total 512-token sequences (2048=1M, 4096=2.1M)")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dspine", type=Path, help="DSpine generations folder: build the on-policy chat mix")
    parser.add_argument("--vemloir", type=Path, help="Vemloir traces folder (agents/, chat/)")
    parser.add_argument("--base-pool", type=Path, help="with --dspine: the web pool its share comes from")
    parser.add_argument("--web-frac", type=float, default=0.2)
    parser.add_argument("--vemloir-frac", type=float, default=0.2)
    parser.add_argument("--eval-out", type=Path, help="with --dspine: held-out validation windows")
    parser.add_argument("--eval-seqs", type=int, default=128)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if args.dspine:
        if not args.base_pool:
            parser.error("--dspine needs --base-pool")
        chat_pool(args, tok)
        return
    pools = []
    manifest = {"seq_len": args.seq_len, "pools": []}

    think_quota = args.total_seqs // 4 if args.think_jsonl else 0
    if args.think_jsonl and args.think_jsonl.exists():
        t_think = pack_token_stream(iter_jsonl_texts(args.think_jsonl), tok, think_quota, args.seq_len)
        pools.append(t_think)
        manifest["pools"].append({"name": "on_policy_think", "seqs": int(t_think.shape[0])})

    rem = args.total_seqs - sum(int(p.shape[0]) for p in pools)
    per_source = max(1, rem // 4)
    sources = [
        ("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
        ("wikipedia", "wikimedia/wikipedia", "20231101.en", "train", "text"),
        ("the_stack_smol", "bigcode/the-stack-smol", None, "train", "content"),
        ("open_web_math", "open-web-math/open-web-math", None, "train", "text"),
    ]

    for name, repo, cfg, split, field in sources:
        quota = min(per_source, args.total_seqs - sum(int(p.shape[0]) for p in pools))
        if quota <= 0:
            break
        try:
            t = pack_token_stream(iter_hf_dataset(repo, cfg, split, field, quota * 3), tok, quota, args.seq_len)
            pools.append(t)
            manifest["pools"].append({"name": name, "seqs": int(t.shape[0])})
            print(f"[{name}] packed {t.shape[0]} sequences")
        except Exception as exc:
            print(f"[{name}] skipped ({exc})")

    if not pools:
        raise RuntimeError("No calibration sequences were collected.")

    calib = torch.cat(pools, dim=0)[: args.total_seqs].contiguous()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calib, args.out)

    sha256 = hashlib.sha256(calib.numpy().tobytes()).hexdigest()
    manifest["total_seqs"] = int(calib.shape[0])
    manifest["total_tokens"] = int(calib.numel())
    manifest["sha256"] = sha256
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved {calib.shape} ({calib.numel():,} tokens) -> {args.out} (sha256={sha256[:12]}...)")


if __name__ == "__main__":
    main()
