"""The MTP drafter from an EXL3 sidecar (a single safetensors of `mtp.*` tensors), into a pack's mtp/* sections.

Replaced, from the sidecar and unchanged: the projections the drafter reads through its dense kernels (fc_embedding,
fc_hidden, q/k/v/o, the shared expert) and the 512 routed experts.  Everything else of mtp/dense (norms, hyper-connection
matrices, router, shared-expert gate, the unused indexer) is copied from the base pack's section.

mtp/dense.index lines are `name kind rows cols offset bytes`; an EXL3 tensor has kind `exl3` (rows = outputs, cols = inputs,
the record [trellis | suh | svh], its rate implied by the bytes).  A first line `#experts exl3 GU_TYPE D_TYPE BLOB_BYTES`
says the experts section is 512 blobs of three records [gate | up | down] (gate and up at GU_TYPE, down at D_TYPE).
"""
from __future__ import annotations

import os

import numpy as np

import dense_pack as d
import exl3_ref as e

EXL3_TYPE_BASE = 96
N_EXPERT, HIDDEN, FF = 512, 2560, 640
PREFIX = "mtp."
LAYER = "mtp.layers.0."
# index name -> (sidecar prefix, outputs, inputs)
DENSE = {
    "fc_embedding.weight": (PREFIX + "fc_embedding", HIDDEN, HIDDEN),
    "fc_hidden.weight": (PREFIX + "fc_hidden", HIDDEN, HIDDEN),
    "self_attn.q_proj.weight": (LAYER + "self_attn.q_proj", 12288, HIDDEN),
    "self_attn.k_proj.weight": (LAYER + "self_attn.k_proj", 512, HIDDEN),
    "self_attn.v_proj.weight": (LAYER + "self_attn.v_proj", 512, HIDDEN),
    "self_attn.o_proj.weight": (LAYER + "self_attn.o_proj", HIDDEN, 6144),
    "mlp.shared_expert.gate_proj.weight": (LAYER + "mlp.shared_expert.gate_proj", FF, HIDDEN),
    "mlp.shared_expert.up_proj.weight": (LAYER + "mlp.shared_expert.up_proj", FF, HIDDEN),
    "mlp.shared_expert.down_proj.weight": (LAYER + "mlp.shared_expert.down_proj", HIDDEN, FF),
}


class MtpError(Exception):
    pass


class Sidecar:
    """The e.Checkpoint interface over a safetensors file of mtp.* tensors, or over a checkpoint folder (its mtp.* tensors)."""

    def __init__(self, path):
        self.dir = None
        if os.path.isdir(path):
            self.dir = e.Checkpoint(path)
            self.weight_map = {n: s for n, s in self.dir.weight_map.items() if n.startswith(PREFIX)}
            self.st = None
        else:
            self.st = e.SafeTensors(path)
            self.weight_map = {n: "sidecar" for n in self.st.names()}

    def get(self, name):
        return self.dir.get(name) if self.dir else self.st.get(name)

    def header(self, name):
        if self.dir is None:
            return self.st.header.get(name)
        shard = self.weight_map.get(name)
        if shard is None:
            return None
        if shard not in self.dir._open:
            self.dir._open[shard] = e.SafeTensors(self.dir.dir / shard)
        return self.dir._open[shard].header.get(name)

    def linear(self, prefix):
        return {k: self.get(f"{prefix}.{k}") for k in ("trellis", "suh", "svh", "mul1")}

    def words(self, prefix, k, n):
        h = self.header(prefix + ".trellis")
        if h is None:
            raise MtpError(f"the sidecar has no {prefix}")
        sh = h["shape"]
        if sh[0] != k // 16 or sh[1] != n // 16 or sh[2] % 8:
            raise MtpError(f"{prefix}.trellis is {sh}, expected [{k // 16}, {n // 16}, W] for {k} -> {n}")
        return int(sh[2])


def expert_words(sc: Sidecar):
    """(gate/up words, down words) of the experts, the same for every one."""
    got = set()
    for x in range(N_EXPERT):
        p = f"{LAYER}mlp.experts.{x}."
        got.add((sc.words(p + "gate_proj", HIDDEN, FF), sc.words(p + "up_proj", HIDDEN, FF),
                 sc.words(p + "down_proj", FF, HIDDEN)))
    if len(got) != 1:
        raise MtpError(f"the experts do not share one rate: {sorted(got)}")
    (g, u, dn), = got
    if g != u:
        raise MtpError(f"gate ({g} words) and up ({u}) differ")
    return g, dn


def blob_bytes(wg: int, wd: int) -> int:
    rec = lambda k, n, w: (k // 16) * (n // 16) * w * 2 + (k + n) * 2
    return 2 * rec(HIDDEN, FF, wg) + rec(FF, HIDDEN, wd)


def expert_chunks(sc: Sidecar, wg: int, wd: int):
    for x in range(N_EXPERT):
        p = f"{LAYER}mlp.experts.{x}."
        parts = []
        for role, k, n, w in (("gate_proj", HIDDEN, FF, wg), ("up_proj", HIDDEN, FF, wg), ("down_proj", FF, HIDDEN, wd)):
            data, got = d.record(sc, p + role, k, n, None, 0)
            if got != w:
                raise MtpError(f"{p}{role}: {got} words, expected {w}")
            parts.append(data)
        yield b"".join(parts)


def dense_chunks(sc: Sidecar, base_index: str, base_blob_reader, out_lines: list):
    """The new dense section, each tensor at a multiple of 256; fills `out_lines`.  base_blob_reader(off, size) -> bytes."""
    off = 0
    seen = set()
    for line in base_index.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, kind, rows, cols, o, size = line.split()
        rows, cols, o, size = int(rows), int(cols), int(o), int(size)
        if name in DENSE:
            prefix, n, k = DENSE[name]
            if (rows, cols) != (n, k):
                raise MtpError(f"{name}: the base pack has [{rows}, {cols}], expected [{n}, {k}]")
            raw, _ = d.record(sc, prefix, k, n, None, 0)
            kind, size = "exl3", len(raw)
            seen.add(name)
        else:
            raw = base_blob_reader(o, size)
        pad = (-off) % 256
        if pad:
            yield b"\0" * pad
            off += pad
        yield raw
        out_lines.append(f"{name} {kind} {rows} {cols} {off} {size}")
        off += size
    missing = set(DENSE) - seen
    if missing:
        raise MtpError("the base pack's mtp/dense.index lacks: " + ", ".join(sorted(missing)))
