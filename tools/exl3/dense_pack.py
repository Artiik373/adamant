"""EXL3 dense tensors for a .basalt: the non-expert projections the engine serves natively (the GDN and attention
projections, the shared experts, the head) taken from the EXL3 checkpoint, unchanged, into the `gguf` section.

A tensor [K, N] (GGUF order: K the input size, ne0) becomes the record the experts use, with type id 96 + W/8:

    trellis   int16[K/16][N/16][W]
    suh       fp16[K]
    svh       fp16[N]

The GGUF files this engine reads keep llama.cpp's order of the linear-attention value heads, the HF checkpoint does
not (tools/convert/model_sections.py: gdn_block_source).  The reordering moves whole 128-wide blocks, and an EXL3
tensor is a grid of independent 128 x 128 blocks (the Hadamard size), so it is applied to the trellis tiles and to
suh / svh slices with no change to any weight:  attn_qkv and attn_gate permute output blocks, ssm_out input blocks.
"""
from __future__ import annotations

import numpy as np

import exl3_ref as e

EXL3_TYPE_BASE = 96
BLOCK = 128
GDN_V_HEADS, GDN_K_HEADS, GDN_QK_ROWS = 48, 16, 4096     # model_sections.py's constants for this architecture

# GGUF suffix -> (HF module suffix, which dimension's blocks the GGUF order permutes, first block that moves)
LAYER = {
    "attn_qkv.weight": ("linear_attn.in_proj_qkv", "n", GDN_QK_ROWS // BLOCK),
    "attn_gate.weight": ("linear_attn.in_proj_z", "n", 0),
    "ssm_out.weight": ("linear_attn.out_proj", "k", 0),
    "attn_q.weight": ("self_attn.q_proj", None, 0),
    "attn_k.weight": ("self_attn.k_proj", None, 0),
    "attn_v.weight": ("self_attn.v_proj", None, 0),
    "attn_output.weight": ("self_attn.o_proj", None, 0),
    "ffn_gate_shexp.weight": ("mlp.shared_expert.gate_proj", None, 0),
    "ffn_up_shexp.weight": ("mlp.shared_expert.up_proj", None, 0),
    "ffn_down_shexp.weight": ("mlp.shared_expert.down_proj", None, 0),
}
GLOBAL = {"output.weight": "lm_head"}


class DenseError(Exception):
    pass


def block_source() -> np.ndarray:
    j = np.arange(GDN_V_HEADS)
    return (j % GDN_K_HEADS) * (GDN_V_HEADS // GDN_K_HEADS) + j // GDN_K_HEADS


def block_order(blocks: int, first: int) -> np.ndarray:
    """New block b takes old block order[b]: identity below `first`, the value-head order above."""
    if blocks - first != GDN_V_HEADS:
        raise DenseError(f"{blocks - first} value-head blocks, expected {GDN_V_HEADS}")
    order = np.arange(blocks)
    order[first:] = first + block_source()
    return order


def hf_prefix(name: str):
    """(HF prefix, dimension, first block) of a GGUF tensor name, or None when it is not converted."""
    if name in GLOBAL:
        return GLOBAL[name], None, 0
    if name.startswith("blk."):
        _, layer, rest = name.split(".", 2)
        if rest in LAYER:
            mod, dim, first = LAYER[rest]
            return f"model.language_model.layers.{layer}.{mod}", dim, first
    return None


def record(ck: e.Checkpoint, prefix: str, k: int, n: int, dim, first: int):
    """The record bytes of one tensor and its tile width W (checked against the GGUF's [k, n])."""
    if f"{prefix}.trellis" not in ck.weight_map:
        raise DenseError(f"the checkpoint has no {prefix}")
    lin = ck.linear(prefix)
    tr, suh, svh = lin["trellis"], lin["suh"], lin["svh"]
    w = int(tr.shape[2])
    if tr.shape != (k // 16, n // 16, w) or tr.dtype != np.int16 or k % BLOCK or n % BLOCK or w % 8:
        raise DenseError(f"{prefix}: trellis {tr.shape} {tr.dtype} does not fit a [{k}, {n}] tensor in whole 128-blocks")
    if suh.shape != (k,) or svh.shape != (n,) or suh.dtype != np.float16 or svh.dtype != np.float16:
        raise DenseError(f"{prefix}: suh/svh must be fp16 of lengths {k} and {n}")
    if int(lin["mul1"]) & 0xFFFFFFFF != e.MUL1_MULT:
        raise DenseError(f"{prefix}.mul1 is not the computed-code multiplier")
    tr, suh, svh = np.ascontiguousarray(tr), np.ascontiguousarray(suh), np.ascontiguousarray(svh)
    if dim == "n":   # output blocks: 8 tile columns and 128 svh entries each
        order = block_order(n // BLOCK, first)
        tr = tr.reshape(k // 16, n // BLOCK, 8, w)[:, order].reshape(k // 16, n // 16, w)
        svh = svh.reshape(n // BLOCK, BLOCK)[order].reshape(n)
    elif dim == "k":   # input blocks: 8 tile rows and 128 suh entries each
        order = block_order(k // BLOCK, first)
        tr = tr.reshape(k // BLOCK, 8, n // 16, w)[order].reshape(k // 16, n // 16, w)
        suh = suh.reshape(k // BLOCK, BLOCK)[order].reshape(k)
    data = b"".join((tr.astype("<i2", copy=False).tobytes(), suh.astype("<f2", copy=False).tobytes(),
                     svh.astype("<f2", copy=False).tobytes()))
    if len(data) != (k // 16) * (n // 16) * w * 2 + (k + n) * 2:
        raise DenseError(f"{prefix}: record is {len(data)} B")
    return data, w


def plan(meta_tensors: dict, ck: e.Checkpoint, head: bool = True):   # head: also output.weight
    """[(name, prefix, dim, first, k, n)] for every converted tensor of the base pack."""
    out = []
    for name, t in meta_tensors.items():
        hp = hf_prefix(name)
        if hp is None or "section" in t or (not head and name in GLOBAL):
            continue
        shape = t["shape"]
        if len(shape) != 2:
            raise DenseError(f"{name}: shape {shape}")
        out.append((name, hp[0], hp[1], hp[2], int(shape[0]), int(shape[1])))
    if not out:
        raise DenseError("the base pack has no tensor this converter replaces")
    return out
