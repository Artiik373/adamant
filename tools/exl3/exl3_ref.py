"""Reference reader/decoder for EXL3 (QTIP-style) tensors, written for basalt.

Format model (QTIP, Tseng et al. 2024: bitshift trellis, tail-biting, computed code, random Hadamard incoherence):
  * a linear layer W[K_in, N_out] is stored as a trellis (tiles of 16x16 = 256 weights), per-input and per-output
    sign/scale vectors `suh` / `svh`, and a codebook id (`mul1`).
  * each 16x16 tile is a ring of 256*K bits (K bits per weight; K may be a half-integer, in which case the trellis
    steps alternate between floor(K) and floor(K)+1 bits). Weight i's state is the 16-bit window of the ring that ends
    after its own bits; the window is mapped to a value by the computed code.
  * W_full = diag(suh) . H128 . W_inner . H128 . diag(svh), H128 the normalized block Hadamard.

Everything here is a hypothesis about the on-disk conventions until `probe_structure.py` confirms it on real tensors.
No third-party quantization code is used; only numpy.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

MUL1_MULT = 0x83DCD12D
K_INV = float(np.array([0x1EEE], dtype=np.uint16).view(np.float16)[0])   # 1 / 147.7: the byte-sum's std


# ------------------------------------------------------------------------------------------------ safetensors

_DT = {"F16": np.float16, "F32": np.float32, "I16": np.int16, "I32": np.int32, "I64": np.int64, "U8": np.uint8,
       "BF16": None}


class SafeTensors:
    """Minimal safetensors reader over a memory map (no torch, no safetensors package)."""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.map = np.memmap(self.path, dtype=np.uint8, mode="r")

    def names(self):
        return [k for k in self.header if k != "__metadata__"]

    def get(self, name) -> np.ndarray:
        h = self.header[name]
        a, b = h["data_offsets"]
        raw = self.map[self.base + a:self.base + b]
        if h["dtype"] == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(h["shape"])
        return np.frombuffer(raw, dtype=_DT[h["dtype"]]).reshape(h["shape"])


class Checkpoint:
    """A sharded checkpoint directory (model.safetensors.index.json)."""

    def __init__(self, folder):
        self.dir = Path(folder)
        idx = json.load(open(self.dir / "model.safetensors.index.json"))
        self.weight_map = idx["weight_map"]
        self._open: dict[str, SafeTensors] = {}

    def get(self, name) -> np.ndarray:
        shard = self.weight_map[name]
        if shard not in self._open:
            self._open[shard] = SafeTensors(self.dir / shard)
        return self._open[shard].get(name)

    def linear(self, prefix) -> dict:
        return {k: self.get(f"{prefix}.{k}") for k in ("trellis", "suh", "svh", "mul1")}


# ------------------------------------------------------------------------------------------------ trellis

def tile_geometry(words_per_tile: int):
    """(ka, mask, bits_per_tile): integer K gives mask 0; a half-integer K alternates KA / KA+1 (period 16)."""
    ka, rem = divmod(words_per_tile, 16)
    if rem == 0:
        return ka, 0x0000, words_per_tile * 16
    if rem == 8:
        return ka, 0xAAAA, words_per_tile * 16
    raise ValueError(f"unsupported tile width: {words_per_tile} 16-bit words")


def ring_ends(ka: int, mask: int) -> np.ndarray:
    """S(i), the ring position at which weight i's window ends (prefix sum of the per-step bit counts)."""
    d = np.array([ka + ((mask >> (i & 15)) & 1) for i in range(256)], dtype=np.int64)
    return np.cumsum(d)


# Ring position -> row-major element of the 16x16 tile (the lane order the decoder walks the tile in).
def ring_to_element(variant: str = "A") -> np.ndarray:
    out = np.empty(256, dtype=np.int64)
    for i in range(256):
        t, j = i >> 3, i & 7
        r0, c0 = (t & 3) * 2, t >> 2
        r = r0 + (1 if j & 1 else 0) + (8 if j & 2 else 0)
        c = c0 + (8 if j & 4 else 0)
        out[i] = r * 16 + c if variant == "A" else c * 16 + r
    return out


def trellis_windows(trellis: np.ndarray) -> np.ndarray:
    """uint16 window per ring position for every tile: [tiles_k, tiles_n, 256]."""
    tk, tn, w = trellis.shape
    ka, mask, rbits = tile_geometry(w)
    flat = np.ascontiguousarray(trellis.reshape(tk * tn, w)).astype(np.uint16)
    words = flat.view("<u4")                                           # two 16-bit words per 32-bit word
    stream = np.unpackbits(words.astype(">u4").view(np.uint8).reshape(len(words), -1), axis=1)   # MSB first
    s = ring_ends(ka, mask)
    idx = (s[:, None] - 16 + np.arange(16)[None, :]) % rbits           # [256, 16] ring bit indices
    bits = stream[:, idx].astype(np.uint32)                            # [tiles, 256, 16]
    win = (bits << (15 - np.arange(16, dtype=np.uint32))).sum(axis=2)
    return win.reshape(tk, tn, 256).astype(np.uint32)


def code_mul1(win: np.ndarray) -> np.ndarray:
    """The computed code: multiply by a constant, sum the four product bytes, centre and scale."""
    x = (win.astype(np.uint64) * MUL1_MULT) & 0xFFFFFFFF
    s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
    return (s.astype(np.float32) - 510.0) * np.float32(K_INV)


def inner_matrix(trellis: np.ndarray, variant: str = "A") -> np.ndarray:
    """W_inner[K_in, N_out] as float32 (the decoded trellis values, before the Hadamard / sign transforms)."""
    tk, tn, _ = trellis.shape
    vals = code_mul1(trellis_windows(trellis))                         # [tk, tn, 256] in ring order
    tile = np.empty_like(vals)
    tile[..., ring_to_element(variant)] = vals                         # [tk, tn, 256] row-major 16x16
    tile = tile.reshape(tk, tn, 16, 16)
    return tile.transpose(0, 2, 1, 3).reshape(tk * 16, tn * 16)


# ------------------------------------------------------------------------------------------------ transforms

def hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.float32(np.sqrt(n))


def full_weight(lin: dict, variant: str = "A", had: np.ndarray | None = None, order: str = "had_then_sign") -> np.ndarray:
    """W_full[K_in, N_out] = diag(suh) H W_inner H diag(svh); `order` selects hypotheses for the sign placement."""
    w = inner_matrix(lin["trellis"], variant)
    k, n = w.shape
    h = hadamard(128) if had is None else had
    suh = lin["suh"].astype(np.float32)
    svh = lin["svh"].astype(np.float32)
    w = (h @ w.reshape(k // 128, 128, n)).reshape(k, n)                # block Hadamard along the input dimension
    if order == "had_then_sign":
        w = w * suh[:, None]
    w = (w.reshape(-1, 128) @ h).reshape(k, n)                         # block Hadamard along the output dimension
    return w * svh[None, :]
