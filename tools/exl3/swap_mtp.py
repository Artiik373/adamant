"""Swap the MTP drafter of a finished .basalt in place: only the mtp/* sections and meta are written (about 1 GB past
the file's end, one header write), nothing else moves.  The replaced drafter's bytes stay as a gap; --undo restores the
file bit for bit.

    # the drafter of another pack (e.g. the IQ3 pack's tuned Q8_0 / Q2_0 one)
    python tools/exl3/swap_mtp.py PACK --from OTHER.basalt
    # a checkpoint's own drafter (a folder) or an EXL3 sidecar file; the dense tensors it does not replace come from --base (a pack with a drafter)
    python tools/exl3/swap_mtp.py PACK --sidecar mtp-2.6bpw.safetensors --base OTHER.basalt
    python tools/exl3/swap_mtp.py PACK --undo UNDO_DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import mtp_pack                                  # noqa: E402
from build_experts import BuildError, type_of    # noqa: E402
from model_file import COPY_CHUNK, ModelFile, replace_sections, undo_replace   # noqa: E402

SECTIONS = ("mtp/dense", "mtp/dense.index", "mtp/experts")


def file_chunks(path: str, off: int, size: int):
    with open(path, "rb") as f:
        f.seek(off)
        left = size
        while left > 0:
            b = f.read(min(COPY_CHUNK, left))
            if not b:
                raise IOError(f"{path}: short read")
            yield b
            left -= len(b)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pack", help="the .basalt to change")
    ap.add_argument("--from", dest="src", help="take the drafter of this pack")
    ap.add_argument("--sidecar", help="an EXL3 sidecar (safetensors of mtp.* tensors) or a checkpoint folder (its own mtp.* tensors)")
    ap.add_argument("--base", help="with --sidecar: the pack whose mtp/dense tensors the sidecar does not replace")
    ap.add_argument("--undo", metavar="DIR", help="restore the pack from DIR (kept by an earlier swap)")
    ap.add_argument("--keep", metavar="DIR", help="where the undo information goes (default: PACK.undo-<time>)")
    a = ap.parse_args(argv)
    if a.undo:
        undo_replace(a.pack, a.undo)
        print(f"restored {a.pack}", file=sys.stderr)
        return 0
    if bool(a.src) == bool(a.sidecar) or (a.sidecar and not a.base):
        ap.error("give --from PACK, or --sidecar FILE with --base PACK")

    target = ModelFile(a.pack)
    for need in SECTIONS:
        if need not in target.sections:
            raise BuildError(f"{a.pack} has no `{need}` section")
    meta = json.loads(target.read("meta"))
    donor_path = a.src or a.base
    donor = ModelFile(donor_path)
    for need in SECTIONS:
        if need not in donor.sections:
            raise BuildError(f"{donor_path} has no `{need}` section")
    donor_meta = json.loads(donor.read("meta"))
    exl3 = meta.setdefault("exl3", {})
    if a.src:
        new = {n: file_chunks(donor_path, *donor.section(n)) for n in SECTIONS}
        meta["mtp"] = donor_meta.get("mtp", {})
        if "mtp" in donor_meta.get("exl3", {}):
            exl3["mtp"] = donor_meta["exl3"]["mtp"]
        else:
            exl3.pop("mtp", None)
        what = f"the drafter of {Path(donor_path).name}"
    else:
        sc = mtp_pack.Sidecar(a.sidecar)
        try:
            wg, wd = mtp_pack.expert_words(sc)
            for name, (prefix, n, k) in mtp_pack.DENSE.items():
                sc.words(prefix, k, n)
        except mtp_pack.MtpError as ex:
            raise BuildError(str(ex))
        off, _ = donor.section("mtp/dense")
        index = donor.read("mtp/dense.index").decode("utf-8")
        lines: list = []
        with open(donor_path, "rb") as f:
            def reader(o, size):
                f.seek(off + o)
                return f.read(size)
            dense = b"".join(mtp_pack.dense_chunks(sc, index, reader, lines))   # about 100 MB
        head = f"#experts exl3 {type_of(wg)} {type_of(wd)} {mtp_pack.blob_bytes(wg, wd)}\n"
        new = {"mtp/dense": [dense],
               "mtp/dense.index": [(head + "\n".join(lines) + "\n").encode("utf-8")],
               "mtp/experts": mtp_pack.expert_chunks(sc, wg, wd)}
        meta["mtp"] = dict(donor_meta.get("mtp", {}), expert_format="exl3")
        meta["mtp"].pop("dense", None)
        exl3["mtp"] = {"source": Path(a.sidecar).name, "expert_words": [wg, wd]}
        what = f"the EXL3 sidecar {Path(a.sidecar).name}"
    keep = a.keep or f"{a.pack}.undo-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"{a.pack}: swapping in {what}", file=sys.stderr)
    replace_sections(a.pack, new, json.dumps(meta, indent=1).encode("utf-8"), keep)
    m = ModelFile(a.pack)
    bad = [n for n in (*SECTIONS, "meta") if not m.verify(n)]
    if bad:
        raise BuildError("hash mismatch in: " + ", ".join(bad))
    print(f"done; undo with: python tools/exl3/swap_mtp.py {a.pack} --undo {keep}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (BuildError, ValueError, OSError) as ex:
        print(f"swap_mtp: {ex}", file=sys.stderr)
        sys.exit(1)
