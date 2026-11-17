#!/usr/bin/env python3
"""Build a .basalt whose routed experts are EXL3 (QTIP-style trellis) tensors, kept exactly as the checkpoint stores them.

    python tools/exl3/build_experts.py --exl3 G:/llm/tabbyAPI/models/exl3-318bpw \
        --base models/Qwen3.8-FN-UD-IQ3_XXS.basalt --out models/Qwen3.8-FN-EXL3.basalt [--dry-run] [--verify]

Every section of `--base` except `experts`, `experts.layout` and `meta` is copied unchanged (the tokenizer, dense and
quantized non-expert tensors, the n-gram table, the drafter, vision).  The `experts` section is rebuilt from the EXL3
checkpoint without re-quantizing anything; `meta` gets an `exl3` record.

Expert blob (per expert, layer by layer, expert 0..n-1 contiguous), for each projection in the order gate, up, down:

    trellis   int16[K/16][N/16][W]   exactly the checkpoint's tensor, little endian, W = 16 * bits per weight
    suh       fp16[K]                input-side vector
    svh       fp16[N]                output-side vector

K and N are the projection's input and output sizes (gate, up: K = hidden, N = ff; down: K = ff, N = hidden).  The
experts' tiles are 16x16, so a record's size is a multiple of 16 bytes and no padding is added.

`experts.layout` is written as `# experts v8`: the v7 columns (layer gu_type d_type offset blob_bytes up_type), where
the type of an EXL3 layer is  EXL3_TYPE_BASE + W / 8  (3 bits: W = 48 -> 102; 3.5 bits: W = 56 -> 103), the same
for gate, up and down.  The header also records the codebook multiplier all tensors carry (`mul1`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import dense_pack                          # noqa: E402
import mtp_pack                            # noqa: E402
import exl3_ref as e                       # noqa: E402
from model_file import ModelFile, ModelWriter   # noqa: E402

EXL3_TYPE_BASE = 96
LAYOUT_VERSION = 8
ROLES = ("gate_proj", "up_proj", "down_proj")


class BuildError(Exception):
    pass


def type_of(words_per_tile: int) -> int:
    if words_per_tile % 8:
        raise BuildError(f"tile width {words_per_tile} words is not a multiple of 8 (half-integer rates only)")
    return EXL3_TYPE_BASE + words_per_tile // 8


def words_of(type_id: int) -> int:
    return (type_id - EXL3_TYPE_BASE) * 8


def record_bytes(k: int, n: int, w: int) -> int:
    return (k // 16) * (n // 16) * w * 2 + (k + n) * 2


class ExpertSource:
    """The checkpoint's routed experts, checked against the geometry before anything is written."""

    def __init__(self, ck: e.Checkpoint, n_layers: int, n_expert: int, hidden: int, ff: int):
        self.ck, self.n_layers, self.n_expert, self.hidden, self.ff = ck, n_layers, n_expert, hidden, ff
        self.words: list[int] = []
        for layer in range(n_layers):
            ws = set()
            for expert in range(n_expert):
                for role in ROLES:
                    pre = self.prefix(layer, expert, role)
                    t = ck.weight_map.get(pre + ".trellis")
                    if t is None:
                        raise BuildError(f"missing tensor {pre}.trellis")
                    ws.add(self._shape(pre)[2])
            if len(ws) != 1:
                raise BuildError(f"layer {layer}: experts differ in bit rate (tile widths {sorted(ws)}); "
                                 "a layer needs one blob size")
            self.words.append(ws.pop())

    @staticmethod
    def prefix(layer: int, expert: int, role: str) -> str:
        return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{role}"

    def _shape(self, pre: str):
        return self.ck.get(pre + ".trellis").shape

    def dims(self, role: str):
        return (self.hidden, self.ff) if role != "down_proj" else (self.ff, self.hidden)

    def blob_bytes(self, layer: int) -> int:
        w = self.words[layer]
        return sum(record_bytes(*self.dims(r), w) for r in ROLES)

    def blob(self, layer: int, expert: int) -> bytes:
        w = self.words[layer]
        parts = []
        for role in ROLES:
            pre = self.prefix(layer, expert, role)
            k, n = self.dims(role)
            lin = self.ck.linear(pre)
            tr, suh, svh = lin["trellis"], lin["suh"], lin["svh"]
            if tr.shape != (k // 16, n // 16, w) or tr.dtype != np.int16:
                raise BuildError(f"{pre}.trellis: shape {tr.shape} {tr.dtype}, expected {(k // 16, n // 16, w)} int16")
            if suh.shape != (k,) or svh.shape != (n,) or suh.dtype != np.float16 or svh.dtype != np.float16:
                raise BuildError(f"{pre}: suh/svh must be fp16 of lengths {k} and {n}")
            if int(lin["mul1"]) & 0xFFFFFFFF != e.MUL1_MULT:
                raise BuildError(f"{pre}.mul1 is {int(lin['mul1']) & 0xFFFFFFFF:#x}, not {e.MUL1_MULT:#x}")
            parts += [tr.astype("<i2", copy=False).tobytes(), suh.astype("<f2", copy=False).tobytes(),
                      svh.astype("<f2", copy=False).tobytes()]
        data = b"".join(parts)
        if len(data) != self.blob_bytes(layer):
            raise BuildError(f"layer {layer} expert {expert}: blob is {len(data)} B, expected {self.blob_bytes(layer)}")
        return data

    def layout_text(self) -> bytes:
        total, lines, offset = 0, [], 0
        for layer in range(self.n_layers):
            blob = self.blob_bytes(layer)
            t = type_of(self.words[layer])
            lines.append(f"{layer} {t} {t} {offset} {blob} {t}\n")
            offset += blob * self.n_expert
        total = offset
        head = (f"# experts v{LAYOUT_VERSION}: layer gu_type d_type offset blob_bytes up_type "
                f"(n_expert {self.n_expert}, total {total}); EXL3 blobs: per projection gate, up, down: "
                f"trellis int16[K/16][N/16][W], suh fp16[K], svh fp16[N]; type = {EXL3_TYPE_BASE} + W/8; "
                f"mul1 {e.MUL1_MULT:#x}\n")
        return (head + "".join(lines)).encode("utf-8")

    def chunks(self, group: int = 16):
        for layer in range(self.n_layers):
            for s in range(0, self.n_expert, group):
                yield b"".join(self.blob(layer, x) for x in range(s, min(s + group, self.n_expert)))

    def total_bytes(self) -> int:
        return sum(self.blob_bytes(l) * self.n_expert for l in range(self.n_layers))


def write_gguf(out, base, base_path, meta, dense_plan, ck) -> None:
    """The `gguf` section with each planned tensor replaced by its EXL3 record, the others copied unchanged."""
    off0, _ = base.section("gguf")
    tensors = meta["tensors"]
    replaced = {p[0]: p for p in dense_plan}
    order = sorted((n for n, t in tensors.items() if "section" not in t), key=lambda n: tensors[n]["offset"])
    done = []

    def parts():
        for i, name in enumerate(order):
            t = tensors[name]
            if name in replaced:
                _, prefix, dim, first, k, n = replaced[name]
                data, w = dense_pack.record(ck, prefix, k, n, dim, first)
                done.append((name, type_of(w), len(data)))
                if i % 40 == 0:
                    print(f"  dense {name} -> type {type_of(w)} ({len(data) / 2**20:.1f} MiB)", file=sys.stderr)
                yield data
            else:
                done.append((name, None, t["bytes"]))
                yield (base_path, off0 + t["offset"], t["bytes"])

    offsets = out.add_parts("gguf", parts())
    for (name, tid, size), o in zip(done, offsets):
        t = tensors[name]
        t["offset"], t["bytes"] = o, size
        if tid is not None:
            t["type"], t["type_id"] = "EXL3", tid


def write_mtp(out, base, sc, words) -> None:
    """mtp/dense, mtp/dense.index and mtp/experts from the sidecar (see mtp_pack.py)."""
    wg, wd = words
    base_off, base_size = base.section("mtp/dense")
    base_index = base.read("mtp/dense.index").decode("utf-8")
    lines: list = []
    with open(base.path, "rb") as f:
        def reader(o, size):
            f.seek(base_off + o)
            return f.read(size)
        print("write mtp/dense ...", file=sys.stderr)
        out.add_chunks("mtp/dense", mtp_pack.dense_chunks(sc, base_index, reader, lines))
    head = f"#experts exl3 {type_of(wg)} {type_of(wd)} {mtp_pack.blob_bytes(wg, wd)}\n"
    out.add_bytes("mtp/dense.index", (head + "\n".join(lines) + "\n").encode("utf-8"))
    print("write mtp/experts ...", file=sys.stderr)
    out.add_chunks("mtp/experts", mtp_pack.expert_chunks(sc, wg, wd))


def read_geometry(folder: Path):
    cfg = json.load(open(folder / "config.json"))
    t = cfg["text_config"]
    q = cfg.get("quantization_config", {})
    if q.get("quant_method") != "exl3":
        raise BuildError(f"{folder}: quantization_config.quant_method is {q.get('quant_method')!r}, not 'exl3'")
    return t["num_hidden_layers"], t["num_experts"], t["hidden_size"], t["moe_intermediate_size"], q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exl3", required=True, help="the EXL3 checkpoint folder")
    ap.add_argument("--base", required=True, help="a .basalt providing every section except the experts")
    ap.add_argument("--out", help="the .basalt to write")
    ap.add_argument("--dense", action="store_true",
                    help="also take the dense projections (GDN/attention, shared experts, lm_head) from the checkpoint")
    ap.add_argument("--keep-head", action="store_true", help="with --dense: leave the base pack's output head as it is")
    ap.add_argument("--mtp", help="an EXL3 sidecar (safetensors of mtp.* tensors): the drafter's projections and "
                                  "experts come from it")
    ap.add_argument("--dry-run", action="store_true", help="check the checkpoint and print the plan; write nothing")
    ap.add_argument("--verify", action="store_true", help="after writing, re-read and hash-check the new experts")
    a = ap.parse_args(argv)
    if not a.dry_run and not a.out:
        ap.error("--out is required unless --dry-run")

    folder = Path(a.exl3)
    n_layers, n_expert, hidden, ff, q = read_geometry(folder)
    print(f"checkpoint: {n_layers} layers x {n_expert} experts, hidden {hidden}, ff {ff}, "
          f"EXL3 {q.get('version')} {q.get('bits')} bpw", file=sys.stderr)
    src = ExpertSource(e.Checkpoint(folder), n_layers, n_expert, hidden, ff)
    base = ModelFile(a.base)
    for need in ("experts", "experts.layout", "meta"):
        if need not in base.sections:
            raise BuildError(f"{a.base} has no `{need}` section")
    rates = {}
    for l, w in enumerate(src.words):
        rates.setdefault(w, []).append(l)
    for w, ls in sorted(rates.items()):
        print(f"  {w / 16:g} bits/weight (tile {w} words, type {type_of(w)}, blob {src.blob_bytes(ls[0]):,} B): "
              f"layers {ls[0]}..{ls[-1]} ({len(ls)})", file=sys.stderr)
    old = base.section("experts")[1]
    print(f"experts: {src.total_bytes() / 2**30:.2f} GiB (base: {old / 2**30:.2f} GiB)", file=sys.stderr)
    other = sum(sz for n, (_, sz, _) in base.sections.items() if n not in ("experts", "experts.layout", "meta"))
    print(f"output: about {(other + src.total_bytes()) / 2**30:.1f} GiB", file=sys.stderr)
    meta = json.loads(base.read("meta"))
    dense_plan, dense_words = [], {}
    if a.dense:
        ck = src.ck
        dense_plan = dense_pack.plan(meta["tensors"], ck, not a.keep_head)
        for name, prefix, dim, first, k, n in dense_plan:
            w = int(ck.linear(prefix)["trellis"].shape[2])
            dense_words.setdefault(w, []).append(name)
            if (k % dense_pack.BLOCK) or (n % dense_pack.BLOCK):
                raise BuildError(f"{name}: [{k}, {n}] is not whole 128-blocks")
        for w, ns in sorted(dense_words.items()):
            print(f"dense {w / 16:g} bits/weight (type {type_of(w)}): {len(ns)} tensors", file=sys.stderr)
    sc, mtp_words = None, None
    if a.mtp:
        for need in ("mtp/dense", "mtp/dense.index", "mtp/experts"):
            if need not in base.sections:
                raise BuildError(f"{a.base} has no `{need}` section")
        sc = mtp_pack.Sidecar(a.mtp)
        try:
            mtp_words = mtp_pack.expert_words(sc)
            for name, (prefix, n, k) in mtp_pack.DENSE.items():
                sc.words(prefix, k, n)
        except mtp_pack.MtpError as ex:
            raise BuildError(str(ex))
        wg, wd = mtp_words
        print(f"drafter: experts gate/up {wg / 16:g} bits (type {type_of(wg)}), down {wd / 16:g} bits (type {type_of(wd)}), "
              f"blob {mtp_pack.blob_bytes(wg, wd):,} B", file=sys.stderr)
    if a.dry_run:
        return 0

    meta["name"] = f"{meta.get('name', 'model')} (EXL3 experts)"
    meta["exl3"] = {"experts": {"source": folder.name, "quant_method": "exl3", "version": q.get("version"),
                                "bits": q.get("bits"), "codebook": "mul1", "mul1": f"{e.MUL1_MULT:#x}",
                                "layout_version": LAYOUT_VERSION, "type_base": EXL3_TYPE_BASE,
                                "words_per_tile": {str(w): ls for w, ls in sorted(rates.items())}}}
    if a.dense:
        meta["exl3"]["dense"] = {"count": len(dense_plan), "type_base": EXL3_TYPE_BASE,
                                 "words_per_tile": {str(w): len(ns) for w, ns in sorted(dense_words.items())},
                                 "value_head_blocks": "permuted to the GGUF order"}
        meta["name"] = meta["name"].replace("(EXL3 experts)", "(EXL3)")
    if a.mtp:
        meta["exl3"]["mtp"] = {"source": Path(a.mtp).name, "expert_words": list(mtp_words)}
        meta.setdefault("mtp", {})["expert_format"] = "exl3"
        meta["mtp"].pop("dense", None)   # the base pack's tuned dense section is replaced
    out = ModelWriter(a.out)
    try:
        for name in base.names():
            if name in ("experts", "experts.layout", "meta"):
                continue
            if a.mtp and name in ("mtp/dense", "mtp/dense.index", "mtp/experts"):
                if name == "mtp/dense":
                    write_mtp(out, base, sc, mtp_words)
                continue
            if name == "gguf" and a.dense:
                write_gguf(out, base, a.base, meta, dense_plan, src.ck)
                continue
            off, size = base.section(name)
            print(f"copy {name} ({size / 2**30:.2f} GiB)", file=sys.stderr)
            out.add_file(name, a.base, off, size)
        print("write experts ...", file=sys.stderr)
        out.add_chunks("experts", src.chunks())
        out.add_bytes("experts.layout", src.layout_text())
        out.add_bytes("meta", json.dumps(meta, indent=1).encode("utf-8"))
    finally:
        out.close()
    if a.verify:
        m = ModelFile(a.out)
        bad = [n for n in m.names() if not m.verify(n)]
        if bad:
            raise BuildError("hash mismatch in: " + ", ".join(bad))
        print("verified", file=sys.stderr)
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildError as ex:
        print(f"build_experts: {ex}", file=sys.stderr)
        sys.exit(1)
