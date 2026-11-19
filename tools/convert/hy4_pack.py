#!/usr/bin/env python3
"""Build the .basalt pack of Hy4-flash (HYV4FlashForCausalLM) from an EXL3 checkpoint directory.

    python tools/convert/hy4_pack.py --src C:/llm-dl/hy4-2p5 --out models/Hy4-flash-EXL3-2.5bpw.basalt [--dry-run]

Sections:
  meta                      JSON: arch "hy4", the source config, the trunk index, the Engram tables, the expert layers
  config.json               the source config, byte for byte
  tokenizer/tokenizer.json  the source tokenizer, and its tokenizer_config.json and chat_template.jinja
  experts, experts.layout   the routed experts of layers 1..41 as EXL3 blobs (tools/exl3/build_experts.py's layout,
                            expert layer i is model layer i + 1), the checkpoint's trellis words kept as they are
  trunk                     every other text tensor (attention, indexer, shared and dense FFN, LatentMoE projections,
                            router, iHC, norms, Engram projections, embedding, head, the DFly2 drafter) as stored:
                            BF16 / F32 / I32, each part 4096-byte aligned; the engine quantizes the matrices at load

The Engram tables (4 x 31.7 GB, BF16) stay in the source shards: `meta.engram` names each table's file, byte offset and
row count, and the engine reads their rows there. The vision tower is not packed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "exl3"))
import build_experts as bx                 # noqa: E402
import exl3_ref as e                       # noqa: E402
from model_file import ModelWriter         # noqa: E402

ARCH = "hy4"
N_LAYERS = 42
EXPERT_LAYERS = 41                          # model layers 1..41; layer 0 is dense
N_EXPERT = 512
LATENT = 1536
FF = 1536
ENGRAM_IDS = (2, 12, 22, 32)                # 1-based in the config; the tables sit on layers id - 1
ENGRAM_DIM = 96
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "I16": 2}


class PackError(Exception):
    pass


class Hy4Experts(bx.ExpertSource):
    """build_experts' blob layout over Hy4's names: expert layer i is model layer i + 1, K = N = 1536."""

    @staticmethod
    def prefix(layer: int, expert: int, role: str) -> str:
        return f"model.layers.{layer + 1}.mlp.experts.{expert}.{role}"


def engram_multipliers(seed: int, layer_id: int, compressed_vocab: int, max_ngram: int) -> list[int]:
    """The hash multipliers of one Engram layer, as the reference draws them (numpy PCG64): odd int64 values."""
    half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab // 2)
    rng = np.random.default_rng(seed + 10007 * layer_id)
    return [int(x) * 2 + 1 for x in rng.integers(0, half_bound, size=max_ngram, dtype=np.int64)]


def tensor_place(ck: e.Checkpoint, name: str):
    """(absolute file, byte offset, bytes, dtype, shape) of a tensor in its shard."""
    shard = ck.weight_map[name]
    if shard not in ck._open:
        ck._open[shard] = e.SafeTensors(ck.dir / shard)
    st = ck._open[shard]
    h = st.header[name]
    a, b = h["data_offsets"]
    return str((ck.dir / shard).resolve()), st.base + a, b - a, h["dtype"], h["shape"]


def trunk_names(ck: e.Checkpoint) -> list[str]:
    out = []
    for n in ck.weight_map:
        if ".mlp.experts." in n or "multi_head_embedding" in n or n.startswith("vit."):
            continue
        out.append(n)
    return sorted(out)


def check_config(cfg: dict) -> None:
    t = cfg.get("text_config", cfg)
    want = {"hidden_size": 3072, "num_hidden_layers": N_LAYERS, "n_routed_experts": N_EXPERT, "num_experts_per_tok": 16,
            "moe_intermediate_size": FF, "routed_expert_hidden_size": LATENT, "kv_lora_rank": 512, "q_lora_rank": 1024,
            "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256, "hc_mult": 4, "sliding_window": 512,
            "vocab_size": 120832, "engram_n_head_per_ngram": 8, "engram_max_ngram_size": 3}
    bad = [f"{k}: {t.get(k)} (expected {v})" for k, v in want.items() if t.get(k) != v]
    if list(t.get("engram_layer_ids", [])) != list(ENGRAM_IDS):
        bad.append(f"engram_layer_ids: {t.get('engram_layer_ids')} (expected {list(ENGRAM_IDS)})")
    if bad:
        raise PackError("the config is not Hy4-flash's geometry: " + "; ".join(bad))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="the EXL3 checkpoint directory (model.safetensors.index.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--dry-run", action="store_true", help="check and plan, write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    a = ap.parse_args(argv)
    src = Path(a.src)
    try:
        cfg = json.load(open(src / "config.json", encoding="utf-8"))
        check_config(cfg)
        qcfg = json.load(open(src / "quantization_config.json", encoding="utf-8"))
        if qcfg.get("quant_method") != "exl3":
            raise PackError("quantization_config.json: not an EXL3 checkpoint")
        ck = e.Checkpoint(src)
        experts = Hy4Experts(ck, EXPERT_LAYERS, N_EXPERT, LATENT, FF)
        trunk = trunk_names(ck)
        places = {n: tensor_place(ck, n) for n in trunk}
        engram = []
        for lid in ENGRAM_IDS:
            name = f"model.layers.{lid - 1}.engram_layer.multi_head_embedding.embedding.weight"
            if name not in ck.weight_map:
                raise PackError(f"missing Engram table {name}")
            path, off, size, dt, shape = tensor_place(ck, name)
            if dt != "BF16" or len(shape) != 2 or shape[1] != ENGRAM_DIM or size != shape[0] * ENGRAM_DIM * 2:
                raise PackError(f"{name}: {dt} {shape}, expected BF16 [rows, {ENGRAM_DIM}]")
            lookup = ck.get(f"model.layers.{lid - 1}.engram_layer.lookup_table").astype(np.int64)
            t = cfg.get("text_config", cfg)
            sizes = [int(s) for order in t["engram_layer_head_vocab_sizes"][str(lid)] for s in order]
            if sum(sizes) != shape[0] or len(sizes) != 16:
                raise PackError(f"{name}: {shape[0]} rows, the head sizes sum to {sum(sizes)} over {len(sizes)} heads")
            engram.append({"id": lid, "layer": lid - 1, "file": path, "offset": off, "rows": shape[0],
                           "row_bytes": ENGRAM_DIM * 2, "dtype": "BF16", "head_sizes": sizes,
                           "multipliers": engram_multipliers(int(t.get("engram_seed", 0)), lid,
                                                             int(lookup.max()) + 1, 3),
                           "pad": int(lookup[int(t["engram_pad_id"])])})
    except (PackError, bx.BuildError, KeyError, OSError) as err:
        print(f"hy4_pack: {err}", file=sys.stderr)
        return 1

    trunk_bytes = sum(p[2] for p in places.values())
    print(f"experts: {EXPERT_LAYERS} layers x {N_EXPERT}, {experts.total_bytes() / 2**30:.2f} GiB, "
          f"tile width {sorted(set(experts.words))} words")
    print(f"trunk: {len(trunk)} tensors, {trunk_bytes / 2**30:.2f} GiB")
    print(f"Engram tables left in the source: {sum(t['rows'] * t['row_bytes'] for t in engram) / 2**30:.2f} GiB")
    if a.dry_run:
        return 0
    if os.path.exists(a.out) and not a.force:
        print(f"hy4_pack: {a.out} exists (use --force)", file=sys.stderr)
        return 1

    w = ModelWriter(a.out)
    w.add_file("config.json", str(src / "config.json"))
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (src / f).exists():
            w.add_file(f"tokenizer/{f}", str(src / f))
    print("writing experts ...", flush=True)
    w.add_chunks("experts", experts.chunks())
    w.add_bytes("experts.layout", experts.layout_text())
    print("writing trunk ...", flush=True)
    offsets = w.add_parts("trunk", [(p[0], p[1], p[2]) for p in (places[n] for n in trunk)])
    index = {n: {"dtype": places[n][3], "shape": places[n][4], "offset": o, "bytes": places[n][2]}
             for n, o in zip(trunk, offsets)}
    meta = {"arch": ARCH, "name": a.name or f"Hy4-flash {src.name}", "config": cfg.get("text_config", cfg),
            "special": {"bos": 120000, "eos": 120025, "pad": 120002},
            "experts": {"first_layer": 1, "layers": EXPERT_LAYERS, "n_expert": N_EXPERT, "format": "exl3",
                        "mul1": e.MUL1_MULT},
            "trunk": index, "engram": engram}
    w.add_bytes("meta", json.dumps(meta, separators=(",", ":")).encode("utf-8"))
    w.close()
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
