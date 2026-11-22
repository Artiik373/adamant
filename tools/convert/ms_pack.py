#!/usr/bin/env python3
"""Build a .basalt whose listed dense tensors are MS (Mad Shift 1MAD trellis) tensors, from a converter output folder.

    python tools/convert/ms_pack.py --base models/Qwen3.8-FN-UD-IQ3_XXS.basalt --ms-dir out_ms \
        --out models/Qwen3.8-FN-MS.basalt [--dry-run] [--verify]

Every section of `--base` except `gguf` and `meta` is copied unchanged.  In the `gguf` section each tensor that has a
record in `--ms-dir` is replaced by its MS payload, the others are copied; `meta` gets the new types and an `ms` record.

A record is `<name>.json` (written by `write_record`, e.g. from ms_convert_dense_spine.py) plus its payload file.  The
payload of a GGUF [n_in, n_out] tensor is the engine's (include/basalt/artifact/gguf_reader.hpp, ms_tensor_type):

    n_out rows of ms_row_bytes(n_in, K)   fp16 scale, uint16 14-bit start state, MSB-first K-bit symbols, zero pad
    int8 signs[n_in]                      +1 / -1, the input rotation y = W_rot * H(signs * x)

The tensor type id is 120 + K (MS_C2 = 122 .. MS_C6 = 126).  The rows must already be in the GGUF order.

With `<ms-dir>/experts/blk.<l>.experts.json` for every layer (ms_flashnext_convert.py), the `experts` section and
`experts.layout` are rebuilt too.  An expert blob is three such payloads back to back, [gate | up | down], one K per
layer; the layout is `# experts v8` with type 120 + K for gate, up and down.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from model_file import ModelFile, ModelWriter    # noqa: E402

MS_TYPE_BASE = 120
MS_BITS = (2, 3, 4, 5, 6)
LAYOUT_VERSION = 8
ROLES = ("gate", "up", "down")


class PackError(Exception):
    pass


def ms_row_bytes(n_in: int, k: int) -> int:
    """Row stride: 4-byte header plus symbols, padded to 4 bytes (the C++ ms_row_bytes)."""
    return (4 + ((n_in - 1) * k + 7) // 8 + 3) & ~3


def payload_bytes(n_in: int, n_out: int, k: int) -> int:
    return n_out * ms_row_bytes(n_in, k) + n_in


def write_record(out_dir: Path, name: str, k: int, n_in: int, n_out: int, rows: bytes, signs: np.ndarray) -> dict:
    """Write one tensor's payload (rows, then int8 signs) and its record; `name` is the GGUF tensor name."""
    signs = np.asarray(signs, dtype=np.int8)
    if k not in MS_BITS or signs.shape != (n_in,) or len(rows) != n_out * ms_row_bytes(n_in, k):
        raise PackError(f"{name}: K {k}, {len(rows)} row bytes and {signs.shape} signs do not fit [{n_in}, {n_out}]")
    stem = name.replace("/", "_")
    (out_dir / f"{stem}.ms").write_bytes(rows + signs.tobytes())
    rec = {"tensor": name, "file": f"{stem}.ms", "k": k, "shape": [n_in, n_out]}
    (out_dir / f"{stem}.json").write_text(json.dumps(rec), encoding="utf-8")
    return rec


def read_records(ms_dir: Path, tensors: dict) -> dict:
    """{GGUF name: (record, payload path)}, each checked against the base pack's tensor."""
    out = {}
    for p in sorted(ms_dir.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        name, k = rec["tensor"], int(rec["k"])
        n_in, n_out = (int(d) for d in rec["shape"])
        t = tensors.get(name)
        if t is None or "section" in t:
            raise PackError(f"{p.name}: the base pack has no gguf tensor {name}")
        if [int(d) for d in t["shape"]] != [n_in, n_out]:
            raise PackError(f"{name}: record shape {[n_in, n_out]}, base pack {t['shape']}")
        if k not in MS_BITS or n_in % 32:
            raise PackError(f"{name}: K {k} or n_in {n_in} (a multiple of 32) is not supported")
        path = ms_dir / rec["file"]
        if path.stat().st_size != payload_bytes(n_in, n_out, k):
            raise PackError(f"{name}: payload is {path.stat().st_size} B, expected {payload_bytes(n_in, n_out, k)}")
        with open(path, "rb") as f:
            f.seek(n_out * ms_row_bytes(n_in, k))
            signs = np.frombuffer(f.read(n_in), dtype=np.int8)
        if not np.all(np.abs(signs) == 1):
            raise PackError(f"{name}: signs must be +1 or -1")
        if name in out:
            raise PackError(f"{name}: two records")
        out[name] = (rec, path)
    if not out:
        raise PackError(f"{ms_dir}: no records")
    return out


class Experts:
    """The routed experts of `<ms-dir>/experts`, one record per layer, checked before anything is written."""

    def __init__(self, exp_dir: Path, n_layers: int):
        self.layers = []
        for l in range(n_layers):
            p = exp_dir / f"blk.{l}.experts.json"
            if not p.exists():
                raise PackError(f"{p}: missing (every layer needs its experts)")
            rec = json.loads(p.read_text(encoding="utf-8"))
            k, n = int(rec["k"]), int(rec["n_expert"])
            if k not in MS_BITS:
                raise PackError(f"{p.name}: K {k}")
            files, sizes = [], []
            for role in ROLES:
                t = rec["tensors"][f"blk.{l}.ffn_{role}_exps.weight"]
                n_in, n_out = (int(d) for d in t["shape"])
                if n_in % 32:
                    raise PackError(f"{p.name}: {role} n_in {n_in} is not a multiple of 32")
                per = payload_bytes(n_in, n_out, k)
                f = exp_dir / t["file"]
                if f.stat().st_size != per * n:
                    raise PackError(f"{f.name}: {f.stat().st_size} B, expected {n} x {per}")
                files.append(f)
                sizes.append(per)
            if self.layers and n != self.layers[0][1]:
                raise PackError(f"{p.name}: {n} experts, layer 0 has {self.layers[0][1]}")
            self.layers.append((k, n, files, sizes))
        self.n_expert = self.layers[0][1]

    def blob_bytes(self, l: int) -> int:
        return sum(self.layers[l][3])

    def total_bytes(self) -> int:
        return sum(self.blob_bytes(l) * self.n_expert for l in range(len(self.layers)))

    def chunks(self, group: int = 16):
        for k, n, files, sizes in self.layers:
            fs = [open(f, "rb") for f in files]
            try:
                for s in range(0, n, group):
                    m = min(group, n - s)
                    parts = [f.read(per * m) for f, per in zip(fs, sizes)]
                    yield b"".join(p[e * per:(e + 1) * per] for e in range(m) for p, per in zip(parts, sizes))
            finally:
                for f in fs:
                    f.close()

    def layout_text(self) -> bytes:
        lines, off = [], 0
        for l, (k, n, _, _) in enumerate(self.layers):
            t, blob = MS_TYPE_BASE + k, self.blob_bytes(l)
            lines.append(f"{l} {t} {t} {off} {blob} {t}\n")
            off += blob * n
        head = (f"# experts v{LAYOUT_VERSION}: layer gu_type d_type offset blob_bytes up_type (n_expert {self.n_expert}, "
                f"total {off}); MS blobs: per projection gate, up, down: rows, int8 signs[n_in]; type = "
                f"{MS_TYPE_BASE} + K\n")
        return (head + "".join(lines)).encode("utf-8")


def write_gguf(out: ModelWriter, base: ModelFile, meta: dict, records: dict) -> None:
    """The `gguf` section with each recorded tensor replaced by its payload, the others copied unchanged."""
    off0, _ = base.section("gguf")
    tensors = meta["tensors"]
    order = sorted((n for n, t in tensors.items() if "section" not in t), key=lambda n: tensors[n]["offset"])

    def parts():
        for name in order:
            if name in records:
                yield records[name][1].read_bytes()
            else:
                t = tensors[name]
                yield (base.path, off0 + t["offset"], t["bytes"])

    offsets = out.add_parts("gguf", parts())
    for name, o in zip(order, offsets):
        t = tensors[name]
        t["offset"] = o
        if name in records:
            rec = records[name][0]
            k = int(rec["k"])
            t["type"], t["type_id"] = f"MS_C{k}", MS_TYPE_BASE + k
            t["bytes"] = payload_bytes(*(int(d) for d in rec["shape"]), k)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, help="a .basalt providing every section and the tensors not replaced")
    ap.add_argument("--ms-dir", required=True, help="the converter's output folder (<name>.json records)")
    ap.add_argument("--out", help="the .basalt to write")
    ap.add_argument("--dry-run", action="store_true", help="check the records and print the plan; write nothing")
    ap.add_argument("--verify", action="store_true", help="after writing, re-read and hash-check every section")
    a = ap.parse_args(argv)
    if not a.dry_run and not a.out:
        ap.error("--out is required unless --dry-run")
    base = ModelFile(a.base)
    for need in ("gguf", "meta"):
        if need not in base.sections:
            raise PackError(f"{a.base} has no `{need}` section")
    meta = json.loads(base.read("meta"))
    records = read_records(Path(a.ms_dir), meta["tensors"])
    experts = None
    if (Path(a.ms_dir) / "experts").is_dir():
        for need in ("experts", "experts.layout"):
            if need not in base.sections:
                raise PackError(f"{a.base} has no `{need}` section")
        n_layers = next(int(v) for kk, v in meta["gguf_kv"].items() if kk.endswith(".block_count"))
        experts = Experts(Path(a.ms_dir) / "experts", n_layers)
        by_kx: dict = {}
        for l, (k, *_rest) in enumerate(experts.layers):
            by_kx.setdefault(k, []).append(l)
        for k, ls in sorted(by_kx.items()):
            print(f"experts MS_C{k}: {len(ls)} layers, blob {experts.blob_bytes(ls[0]):,} B", file=sys.stderr)
        print(f"experts: {experts.total_bytes() / 2**30:.2f} GiB (base {base.section('experts')[1] / 2**30:.2f} GiB)",
              file=sys.stderr)
    by_k: dict = {}
    for name, (rec, _) in records.items():
        by_k.setdefault(int(rec["k"]), []).append(name)
    for k, names in sorted(by_k.items()):
        print(f"MS_C{k} (type {MS_TYPE_BASE + k}): {len(names)} tensors", file=sys.stderr)
    if a.dry_run:
        return 0

    meta["ms"] = {"dense": {f"MS_C{k}": len(v) for k, v in sorted(by_k.items())}, "type_base": MS_TYPE_BASE}
    if experts is not None:
        meta["ms"]["experts"] = {"k": [k for k, *_ in experts.layers], "layout_version": LAYOUT_VERSION}
    out = ModelWriter(a.out)
    try:
        for name in base.names():
            if name == "meta" or (experts is not None and name == "experts.layout"):
                continue
            if experts is not None and name == "experts":
                print("write experts ...", file=sys.stderr)
                out.add_chunks("experts", experts.chunks())
                out.add_bytes("experts.layout", experts.layout_text())
                continue
            if name == "gguf":
                write_gguf(out, base, meta, records)
                continue
            off, size = base.section(name)
            print(f"copy {name} ({size / 2**30:.2f} GiB)", file=sys.stderr)
            out.add_file(name, a.base, off, size)
        out.add_bytes("meta", json.dumps(meta, indent=1).encode("utf-8"))
    finally:
        out.close()
    if a.verify:
        m = ModelFile(a.out)
        bad = [n for n in m.names() if not m.verify(n)]
        if bad:
            raise PackError("hash mismatch in: " + ", ".join(bad))
        print("verified", file=sys.stderr)
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackError as ex:
        print(f"ms_pack: {ex}", file=sys.stderr)
        sys.exit(1)
