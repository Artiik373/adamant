"""The GigaChat4 pack from the FP8 release (ai-sage/GigaChat4-37B-A4B): `gigachat4_convert.py MODEL_DIR OUT.basalt`.

  meta        JSON: arch, geometry, and the table of the `dense` section's tensors {name: {type, shape, offset, bytes}}
  dense       every tensor but the routed experts: matrices as Q8_0, the embeddings as BF16, everything else F32
  experts     per MoE layer (1..31) and expert: [gate | up | down] as NVFP4 blocks (ggml type 40) and the FP32 tail
              {s_gate, s_up, s_down, 0}; expert (l, e) at ((l - 1) * 192 + e) * blob
  tokenizer/* the release's tokenizer.json, tokenizer_config.json and chat template

FP8 blocks are dequantized with their 128 x 128 scales, then rounded once. The converter folds: alpha_q into q_b and the
indexer's wq_b, alpha_kv into k_b and v_b, the indexer's head-weight scale into its weights_proj, the GDN output gain into
its norm weight. GDN rows are reordered to [q | k | v | z] and [b | a] (value heads in the release's order; value head h
reads key head h / 2). kv_b splits into k_b [head][512][128] (transposed, for the absorbed query) and v_b [head][128][512].
Needs numpy.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import checkpoint as C  # noqa: E402
import checkpoint_qwen35 as Q  # noqa: E402
import model_sections as S  # noqa: E402
from model_file import ModelWriter  # noqa: E402

TYPE_ID = {"F32": 0, "Q8_0": 8, "BF16": 30, "NVFP4": 40}
L, E, D, FF = 32, 192, 2560, 768
FULL = set(range(3, 32, 4))
EXPERTS_PER_JOB = 4

_src = None


def e4m3_lut() -> np.ndarray:
    """The 256 E4M3 (fn) codes as float32: no infinities, 0x7f / 0xff are NaN."""
    c = np.arange(256)
    s, e, m = (c >> 7) & 1, (c >> 3) & 15, c & 7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7.0))
    v = np.where((e == 15) & (m == 7), np.nan, v)
    return np.where(s == 1, -v, v).astype(np.float32)


E4M3 = e4m3_lut()


class Source:
    """The release's safetensors read in place (no torch): BF16, F32 and F8_E4M3 with its 128 x 128 scales."""

    def __init__(self, md: str):
        self.md = Path(md)
        self.where = json.load(open(self.md / "model.safetensors.index.json"))["weight_map"]
        self.head = {}
        for f in set(self.where.values()):
            with open(self.md / f, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                h = json.loads(fh.read(n))
            h.pop("__metadata__", None)
            self.head[f] = (8 + n, h)

    def has(self, name):
        return name in self.where

    def raw(self, name):
        f = self.where[name]
        base, h = self.head[f]
        t = h[name]
        o0, o1 = t["data_offsets"]
        dt = {"BF16": "<u2", "F32": "<f4", "F8_E4M3": np.uint8}[t["dtype"]]
        return t["dtype"], np.fromfile(self.md / f, dtype=dt, count=(o1 - o0) // np.dtype(dt).itemsize, offset=base + o0).reshape(t["shape"])

    def get(self, name) -> np.ndarray:
        ty, a = self.raw(name)
        if ty == "BF16":
            x = (a.astype(np.uint32) << 16).view(np.float32)
        elif ty == "F32":
            x = a.astype(np.float32)
        else:
            _, s = self.raw(name + "_scale_inv")
            r, c = a.shape
            x = E4M3[a] * np.repeat(np.repeat(s, 128, 0)[:r], 128, 1)[:, :c]
        if not np.isfinite(x).all():
            raise ValueError(f"{name}: a non-finite value")
        return x


def _init(md):
    global _src
    _src = Source(md)


def nvfp4(x: np.ndarray):
    gs = Q.fp4q_global(float(np.abs(x).max()))
    codes, sc = Q.fp4q_rows(np.ascontiguousarray(x, np.float32), gs)
    return C.nvfp4_rows(codes, sc).tobytes(), gs


def _expert_job(args):
    l, e0, e1 = args
    out = []
    for e in range(e0, e1):
        tail = []
        for r in ("gate", "up", "down"):
            b, gs = nvfp4(_src.get(f"model.layers.{l}.mlp.experts.{e}.{r}_proj.weight"))
            out.append(b)
            tail.append(gs)
        out.append(np.array(tail + [0.0], dtype="<f4").tobytes())
    return b"".join(out)


def bf16_bits(x: np.ndarray) -> bytes:
    """float32 values that came from BF16 -> their BF16 bits."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    if (u & 0xFFFF).any():
        raise ValueError("a value that is not a BF16 number")
    return (u >> 16).astype("<u2").tobytes()


class Dense:
    def __init__(self):
        self.parts, self.table, self.pos = [], {}, 0

    def add(self, name, x: np.ndarray, ty: str):
        x = np.ascontiguousarray(x, np.float32)
        b = {"F32": lambda: x.astype("<f4").tobytes(), "Q8_0": lambda: S.q8_0(x.reshape(-1)), "BF16": lambda: bf16_bits(x)}[ty]()
        self.pos += (-self.pos) % 256
        self.table[name] = {"type": TYPE_ID[ty], "shape": list(x.shape), "offset": self.pos, "bytes": len(b)}
        self.parts.append(b)
        self.pos += len(b)


def gated_norm(dn: Dense, src: Source, hf: str, name: str):
    dn.add(name + ".w", src.get(hf + ".weight"), "F32")
    dn.add(name + ".up", src.get(hf + ".gate_up_projection.weight"), "F32")      # [16, n]
    dn.add(name + ".down", src.get(hf + ".gate_down_projection.weight"), "F32")  # [n, 16]


def mla(dn: Dense, src: Source, p: str, b: str):
    aq, akv = math.sqrt(D / 1536), math.sqrt(D / 512)
    dn.add(b + "mla.q_a", src.get(p + "self_attn.q_a_proj.weight"), "Q8_0")
    gated_norm(dn, src, p + "self_attn.q_a_layernorm", b + "mla.q_a_norm")
    dn.add(b + "mla.q_b", src.get(p + "self_attn.q_b_proj.weight") * aq, "Q8_0")
    dn.add(b + "mla.kv_a", src.get(p + "self_attn.kv_a_proj_with_mqa.weight"), "Q8_0")
    gated_norm(dn, src, p + "self_attn.kv_a_layernorm", b + "mla.kv_norm")
    kvb = (src.get(p + "self_attn.kv_b_proj.weight") * akv).reshape(32, 256, 512)
    dn.add(b + "mla.k_b", kvb[:, :128].transpose(0, 2, 1).reshape(32 * 512, 128), "Q8_0")
    dn.add(b + "mla.v_b", kvb[:, 128:].reshape(32 * 128, 512), "Q8_0")
    dn.add(b + "mla.o", src.get(p + "self_attn.o_proj.weight"), "Q8_0")
    dn.add(b + "mla.gate", src.get(p + "self_attn.gate_proj.weight"), "Q8_0")
    dn.add(b + "idx.wq_b", src.get(p + "self_attn.indexer.wq_b.weight") * aq, "Q8_0")
    dn.add(b + "idx.wk", src.get(p + "self_attn.indexer.wk.weight"), "Q8_0")
    dn.add(b + "idx.k_norm", src.get(p + "self_attn.indexer.k_norm.weight"), "F32")
    dn.add(b + "idx.w", src.get(p + "self_attn.indexer.weights_proj.weight") / math.sqrt(64 * 128), "F32")


def gdn(dn: Dense, src: Source, p: str, b: str):
    w = src.get(p + "self_attn.in_proj_qkvz.weight").reshape(16, 768, D)
    q, k, v, z = w[:, :128], w[:, 128:256], w[:, 256:512], w[:, 512:]
    dn.add(b + "gdn.qkvz", np.concatenate([x.reshape(-1, D) for x in (q, k, v, z)]), "Q8_0")
    ba = src.get(p + "self_attn.in_proj_ba.weight").reshape(16, 4, D)
    dn.add(b + "gdn.ba", np.concatenate([ba[:, :2].reshape(32, D), ba[:, 2:].reshape(32, D)]), "Q8_0")
    dn.add(b + "gdn.conv", src.get(p + "self_attn.conv1d.weight").reshape(8192, 4), "F32")
    dn.add(b + "gdn.a_log", src.get(p + "self_attn.A_log"), "F32")
    dn.add(b + "gdn.dt_bias", src.get(p + "self_attn.dt_bias"), "F32")
    dn.add(b + "gdn.norm", src.get(p + "self_attn.norm.weight") * 2.0, "F32")
    dn.add(b + "gdn.out", src.get(p + "self_attn.out_proj.weight"), "Q8_0")


def swiglu(dn: Dense, src: Source, p: str, name: str):
    gu = np.concatenate([src.get(p + "gate_proj.weight"), src.get(p + "up_proj.weight")])
    dn.add(name + ".gu", gu, "Q8_0")
    dn.add(name + ".down", src.get(p + "down_proj.weight"), "Q8_0")


def layer(dn: Dense, src: Source, l: int, b: str, moe: bool):
    p = f"model.layers.{l}."
    for hf, n in (("input_layernorm", "attn_norm"), ("post_self_attn_layernorm", "attn_post_norm"),
                  ("post_attention_layernorm", "ffn_norm"), ("post_feedforward_layernorm", "ffn_post_norm")):
        gated_norm(dn, src, p + hf, b + n)
    (mla if l in FULL or l == L else gdn)(dn, src, p, b)
    if moe:
        dn.add(b + "router", src.get(p + "mlp.gate.weight"), "F32")
        dn.add(b + "router_bias", src.get(p + "mlp.gate.e_score_correction_bias"), "F32")
        swiglu(dn, src, p + "mlp.shared_experts.", b + "shexp")
    else:
        swiglu(dn, src, p + "mlp.", b + "ffn")


def dense(src: Source) -> Dense:
    dn = Dense()
    dn.add("token_embd", src.get("model.embed_tokens.weight"), "BF16")
    for l in range(L):
        layer(dn, src, l, f"blk.{l}.", l >= 1)
        print(f"  layer {l} dense", flush=True)
    gated_norm(dn, src, "model.norm", "output_norm")
    head = src.get("lm_head.weight")
    dn.add("output", head, "Q8_0")
    # the drafter: layer 32
    p = f"model.layers.{L}."
    layer(dn, src, L, "mtp.", False)
    dn.add("mtp.eh", src.get(p + "eh_proj.weight"), "Q8_0")
    gated_norm(dn, src, p + "enorm", "mtp.enorm")
    gated_norm(dn, src, p + "hnorm", "mtp.hnorm")
    gated_norm(dn, src, p + "shared_head.norm", "mtp.head_norm")
    mh = src.get(p + "shared_head.head.weight")
    if not np.array_equal(mh, head):
        dn.add("mtp.head", mh, "Q8_0")
    me = src.get(p + "embed_tokens.weight")
    if not np.array_equal(me, src.get("model.embed_tokens.weight")):
        dn.add("mtp.token_embd", me, "BF16")
    print("  drafter dense", flush=True)
    return dn


GEOMETRY = {"n_layers": L, "n_embd": D, "n_vocab": 128256, "n_expert": E, "n_ff_exp": FF, "n_expert_used": 8,
            "full_attention_layers": sorted(FULL), "mtp_layers": 1}


def main(md: str, out: str, workers: int = 16):
    t0 = time.time()
    src = Source(md)
    dn = dense(src)
    blob = 3 * FF * D // 64 * 36 + 16
    w = ModelWriter(out + ".partial")
    meta = {"arch": "gigachat4", "geometry": GEOMETRY, "expert_blob": blob, "tensors": dn.table}
    w.add_bytes("meta", json.dumps(meta).encode())
    w.add_parts("dense", dn.parts, align=256)
    del dn
    jobs = [(l, e, min(e + EXPERTS_PER_JOB, E)) for l in range(1, L) for e in range(0, E, EXPERTS_PER_JOB)]
    with multiprocessing.Pool(workers, _init, (md,)) as pool:
        def chunks():
            for i, b in enumerate(pool.imap(_expert_job, jobs, chunksize=1)):
                if len(b) != blob * (jobs[i][2] - jobs[i][1]):
                    raise ValueError("expert blob size")
                if jobs[i][2] == E:
                    print(f"  layer {jobs[i][0]} experts, {time.time() - t0:.0f} s", flush=True)
                yield b
        n = w.add_chunks("experts", chunks())
    if n != (L - 1) * E * blob:
        raise ValueError("experts section size")
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        w.add_file("tokenizer/" + f, str(Path(md) / f))
    w.close()
    Path(out + ".partial").replace(out)
    print(f"done: {Path(out).stat().st_size / (1 << 30):.2f} GiB, {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16)
