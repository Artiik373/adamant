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

The tensor type id is 120 + K (MS_C3 = 123, MS_C4 = 124, MS_C6 = 126).  The rows must already be in the GGUF order.
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
MS_BITS = (3, 4, 6)


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
    by_k: dict = {}
    for name, (rec, _) in records.items():
        by_k.setdefault(int(rec["k"]), []).append(name)
    for k, names in sorted(by_k.items()):
        print(f"MS_C{k} (type {MS_TYPE_BASE + k}): {len(names)} tensors", file=sys.stderr)
    if a.dry_run:
        return 0

    meta["ms"] = {"dense": {f"MS_C{k}": len(v) for k, v in sorted(by_k.items())}, "type_base": MS_TYPE_BASE}
    out = ModelWriter(a.out)
    try:
        for name in base.names():
            if name == "meta":
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
