"""Deterministic calibration pool builder for Qwen3.8-Flash-Next MS conversion.

Builds a [N_seq, 512] int32 tensor (default 2,048 x 512 = 1.05M tokens, or 4,096 x 512 = 2.1M)
mixing FineWeb-Edu, Wikipedia, The Stack Smol (30 languages), OpenWebMath, and zero-repeat
on-policy `<think>...</think>` trajectories, and writes a SHA-256 manifest.

Usage:
  python tools/convert/build_calib_pool.py \
      --tokenizer Qwen/Qwen3.8-Flash-Next \
      --think-jsonl data/selfgen_think.jsonl \
      --out calib_2m.pt --total-seqs 4096
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed + on-policy calibration pool (.pt)")
    parser.add_argument("--tokenizer", type=str, required=True, help="HF model dir or repo ID")
    parser.add_argument("--think-jsonl", type=Path, default=None, help="Optional on-policy <think> JSONL")
    parser.add_argument("--out", type=Path, required=True, help="Output .pt path")
    parser.add_argument("--total-seqs", type=int, default=2048, help="Total 512-token sequences (2048=1M, 4096=2.1M)")
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
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
