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
layer; the layout is `# experts v8` with type 120 + K for gate, up and down.  A record whose k is a list (one K per
expert) makes a mixed layer: layout v9, type 127, the K list as the line's last column, each blob a 16-byte head
(byte 0 = K) then the payloads at that K, blobs at their own sizes; blob_bytes is the widest, and the section ends in
one slot of zeros.

With `<ms-dir>/mtp/` (ms_flashnext_convert.py's drafter: mtp.dense, mtp.dense.index and one layer of expert records)
the `mtp/dense`, `mtp/dense.index` and `mtp/experts` sections are replaced: the index names the same tensors as the
base pack's, projections of kind `ms`, and its first line `#experts ms T T BLOB` says the experts are such blobs.
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


MS_MIX = 127      # a layer whose experts carry their own K (layout v9)
MIX_HEAD = 16     # a mixed layer's blob: byte 0 = K, zero pad, then gate | up | down at that K


class Experts:
    """The routed experts of `<ms-dir>/experts`, one record per layer, checked before anything is written.
    A record's k is one width, or a list of one width per expert (a mixed layer: blobs at their own sizes)."""

    def __init__(self, exp_dir: Path, n_layers: int):
        self.layers = []
        for l in range(n_layers):
            p = exp_dir / f"blk.{l}.experts.json"
            if not p.exists():
                raise PackError(f"{p}: missing (every layer needs its experts)")
            rec = json.loads(p.read_text(encoding="utf-8"))
            n = int(rec["n_expert"])
            k = rec["k"] if isinstance(rec["k"], int) else [int(v) for v in rec["k"]]
            ks = [k] * n if isinstance(k, int) else k
            if len(ks) != n or not set(ks) <= set(MS_BITS):
                raise PackError(f"{p.name}: K {k}")
            files, sizes = [], []
            for role in ROLES:
                t = rec["tensors"][f"blk.{l}.ffn_{role}_exps.weight"]
                n_in, n_out = (int(d) for d in t["shape"])
                if n_in % 32:
                    raise PackError(f"{p.name}: {role} n_in {n_in} is not a multiple of 32")
                per = [payload_bytes(n_in, n_out, ke) for ke in ks]
                f = exp_dir / t["file"]
                if f.stat().st_size != sum(per):
                    raise PackError(f"{f.name}: {f.stat().st_size} B, expected {sum(per)}")
                files.append(f)
                sizes.append(per)
            if self.layers and n != self.layers[0][1]:
                raise PackError(f"{p.name}: {n} experts, layer 0 has {self.layers[0][1]}")
            self.layers.append((k, n, files, sizes))
        self.n_expert = self.layers[0][1]
        self.mixed = any(not isinstance(k, int) for k, *_ in self.layers)

    def expert_bytes(self, l: int, e: int) -> int:
        k, _, _, sizes = self.layers[l]
        return (0 if isinstance(k, int) else MIX_HEAD) + sum(s[e] for s in sizes)

    def blob_bytes(self, l: int) -> int:
        """The layer's slot: its widest blob."""
        return max(self.expert_bytes(l, e) for e in range(self.n_expert))

    def layer_bytes(self, l: int) -> int:
        return sum(self.expert_bytes(l, e) for e in range(self.n_expert))

    def pad_bytes(self) -> int:
        """Zeros after the last layer: a slot-sized copy of a mixed layer's last expert stays in the section."""
        return max(self.blob_bytes(l) for l in range(len(self.layers))) if self.mixed else 0

    def total_bytes(self) -> int:
        return sum(self.layer_bytes(l) for l in range(len(self.layers))) + self.pad_bytes()

    def chunks(self, group: int = 16):
        for k, n, files, sizes in self.layers:
            fs = [open(f, "rb") for f in files]
            try:
                for s in range(0, n, group):
                    es = range(s, min(s + group, n))
                    parts = [f.read(sum(sz[e] for e in es)) for f, sz in zip(fs, sizes)]
                    at = [0, 0, 0]
                    out = []
                    for e in es:
                        if not isinstance(k, int):
                            out.append(bytes([k[e]]) + bytes(MIX_HEAD - 1))
                        for r, (part, sz) in enumerate(zip(parts, sizes)):
                            out.append(part[at[r]:at[r] + sz[e]])
                            at[r] += sz[e]
                    yield b"".join(out)
            finally:
                for f in fs:
                    f.close()
        if self.pad_bytes():
            yield bytes(self.pad_bytes())

    def layout_text(self) -> bytes:
        lines, off = [], 0
        for l, (k, n, _, _) in enumerate(self.layers):
            blob = self.blob_bytes(l)
            if isinstance(k, int):
                t = MS_TYPE_BASE + k
                lines.append(f"{l} {t} {t} {off} {blob} {t}\n")
            else:
                lines.append(f"{l} {MS_MIX} {MS_MIX} {off} {blob} {MS_MIX} {''.join(str(v) for v in k)}\n")
            off += self.layer_bytes(l)
        ver = 9 if self.mixed else LAYOUT_VERSION
        head = (f"# experts v{ver}: layer gu_type d_type offset blob_bytes up_type (n_expert {self.n_expert}, "
                f"total {off}); MS blobs: per projection gate, up, down: rows, int8 signs[n_in]; type = "
                f"{MS_TYPE_BASE} + K")
        if self.mixed:
            head += (f"; type {MS_MIX}: each expert its own K (the last column, one digit an expert), its blob a "
                     f"{MIX_HEAD}-byte head (byte 0 = K) then gate, up, down at that K, blobs back to back, "
                     f"blob_bytes the widest")
        return (head + "\n" + "".join(lines)).encode("utf-8")


MTP_SECTIONS = ("mtp/dense", "mtp/dense.index", "mtp/experts")


def read_mtp(mtp_dir: Path, base: ModelFile):
    """(index text, dense path, Experts) of the converter's drafter, checked against the base pack's drafter."""
    for need in MTP_SECTIONS:
        if need not in base.sections:
            raise PackError(f"the base pack has no `{need}` section")
    index = (mtp_dir / "mtp.dense.index").read_text(encoding="utf-8")
    dense = mtp_dir / "mtp.dense"
    lines = index.splitlines()
    ex = Experts(mtp_dir, 1)
    k = ex.layers[0][0]
    want = f"#experts ms {MS_TYPE_BASE + k} {MS_TYPE_BASE + k} {ex.blob_bytes(0)}"
    if not lines or lines[0] != want:
        raise PackError(f"{mtp_dir}: the index starts with {lines[0] if lines else '(nothing)'!r}, expected {want!r}")
    names, end = set(), 0
    for line in lines[1:]:
        name, kind, rows, cols, off, size = line.split()
        rows, cols, off, size = int(rows), int(cols), int(off), int(size)
        if kind == "ms" and not any(rows * ms_row_bytes(cols, kk) + cols == size for kk in MS_BITS):
            raise PackError(f"{name}: {size} B is no MS payload of [{rows}, {cols}]")
        names.add(name)
        end = max(end, off + size)
    if end > dense.stat().st_size:
        raise PackError(f"{dense}: {dense.stat().st_size} B, the index reaches {end}")
    base_names = {ln.split()[0] for ln in base.read("mtp/dense.index").decode("utf-8").splitlines()
                  if ln.strip() and not ln.startswith("#")}
    if names != base_names:
        raise PackError(f"the drafter's tensors differ from the base pack's: {sorted(names ^ base_names)}")
    return index, dense, ex


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
            by_kx.setdefault(k if isinstance(k, int) else MS_MIX, []).append(l)
        for k, ls in sorted(by_kx.items()):
            name = "mixed" if k == MS_MIX else f"MS_C{k}"
            print(f"experts {name}: {len(ls)} layers, slot {experts.blob_bytes(ls[0]):,} B", file=sys.stderr)
        print(f"experts: {experts.total_bytes() / 2**30:.2f} GiB (base {base.section('experts')[1] / 2**30:.2f} GiB)",
              file=sys.stderr)
    mtp = None
    if (Path(a.ms_dir) / "mtp").is_dir():
        mtp = read_mtp(Path(a.ms_dir) / "mtp", base)
        print(f"mtp: experts MS_C{mtp[2].layers[0][0]} ({mtp[2].total_bytes() / 2**30:.2f} GiB, base "
              f"{base.section('mtp/experts')[1] / 2**30:.2f} GiB), dense {mtp[1].stat().st_size / 2**20:.1f} MiB",
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
        meta["ms"]["experts"] = {"k": [k for k, *_ in experts.layers], "layout_version": 9 if experts.mixed else LAYOUT_VERSION}
    if mtp is not None:   # the drafter is the checkpoint's own, coded: no tuned dense, no Q2_0 experts
        meta["mtp"] = {k: v for k, v in meta.get("mtp", {}).items() if k not in ("dense", "source_sha256")}
        meta["mtp"]["expert_format"] = "ms"
        meta["ms"]["mtp"] = {"experts_k": mtp[2].layers[0][0]}
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
            if mtp is not None and name in MTP_SECTIONS:
                if name == "mtp/dense":
                    out.add_file(name, str(mtp[1]), 0, mtp[1].stat().st_size)
                elif name == "mtp/dense.index":
                    out.add_bytes(name, mtp[0].encode("utf-8"))
                else:
                    out.add_chunks(name, mtp[2].chunks())
                print(f"mtp: {name} from {a.ms_dir}", file=sys.stderr)
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
