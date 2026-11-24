"""ms_pack.py on a synthetic .basalt: one dense tensor replaced by an MS_C6 record, the rest copied; bad records refused.
Also ms_convert_dense_spine.to_gguf_order: the GDN reorder keeps W x for an input in the GGUF order.

    python -m unittest discover -s tools/convert -p "test_ms_pack.py"
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import ms_pack                                   # noqa: E402
from model_file import ModelFile, ModelWriter    # noqa: E402

HAVE_TORCH = importlib.util.find_spec("torch") is not None

N_IN, N_OUT, K = 96, 16, 6


def base_pack(path: Path) -> dict:
    """A .basalt with a `tokenizer` section and two gguf tensors: blk.0.attn_q.weight (to replace) and a norm."""
    rng = np.random.default_rng(1)
    q = rng.integers(0, 256, N_OUT * N_IN // 32 * 34, dtype=np.uint8).tobytes()   # Q8_0 [96, 16]
    norm = rng.standard_normal(N_IN).astype("<f4").tobytes()
    w = ModelWriter(str(path))
    w.add_bytes("tokenizer", b"tok" * 100)
    offs = w.add_parts("gguf", [q, norm])
    meta = {"name": "test", "gguf_kv": {}, "tensors": {
        "blk.0.attn_q.weight": {"type": "Q8_0", "type_id": 8, "shape": [N_IN, N_OUT], "offset": offs[0], "bytes": len(q)},
        "blk.0.attn_norm.weight": {"type": "F32", "type_id": 0, "shape": [N_IN], "offset": offs[1], "bytes": len(norm)},
    }}
    w.add_bytes("meta", json.dumps(meta).encode())
    w.close()
    return {"norm": norm}


def ms_record(out_dir: Path, name: str = "blk.0.attn_q.weight", shape=(N_IN, N_OUT), signs=None, k=K) -> bytes:
    n_in, n_out = shape
    rows = np.random.default_rng(2).integers(0, 256, n_out * ms_pack.ms_row_bytes(n_in, k), dtype=np.uint8).tobytes()
    if signs is None:
        signs = np.where(np.arange(n_in) % 3, 1, -1)
    ms_pack.write_record(out_dir, name, k, n_in, n_out, rows, signs)
    return rows + np.asarray(signs, dtype=np.int8).tobytes()


def check_pack(k: int = K) -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base = base_pack(td / "base.basalt")
        (td / "ms").mkdir()
        payload = ms_record(td / "ms", k=k)
        assert len(payload) == ms_pack.payload_bytes(N_IN, N_OUT, k)
        out = td / "out.basalt"
        assert ms_pack.main(["--base", str(td / "base.basalt"), "--ms-dir", str(td / "ms"), "--out", str(out),
                             "--verify"]) == 0
        m = ModelFile(str(out))
        assert m.read("tokenizer") == b"tok" * 100
        meta = json.loads(m.read("meta"))
        t = meta["tensors"]["blk.0.attn_q.weight"]
        assert (t["type"], t["type_id"], t["bytes"]) == (f"MS_C{k}", 120 + k, len(payload)), t
        assert t["offset"] % 4096 == 0
        g0, _ = m.section("gguf")
        with open(out, "rb") as f:
            f.seek(g0 + t["offset"])
            assert f.read(t["bytes"]) == payload
            n = meta["tensors"]["blk.0.attn_norm.weight"]
            f.seek(g0 + n["offset"])
            assert f.read(n["bytes"]) == base["norm"]
        assert meta["ms"]["dense"] == {f"MS_C{k}": 1}
    print("ms_pack: replace + copy + verify ok")


def refused(make) -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base_pack(td / "base.basalt")
        (td / "ms").mkdir()
        make(td / "ms")
        try:
            ms_pack.main(["--base", str(td / "base.basalt"), "--ms-dir", str(td / "ms"), "--dry-run"])
        except ms_pack.PackError:
            return
        raise AssertionError("a bad record was accepted")


def check_refused() -> None:
    refused(lambda d: ms_record(d, name="blk.9.attn_q.weight"))       # no such tensor
    refused(lambda d: ms_record(d, shape=(N_IN, N_OUT + 1)))           # another shape
    refused(lambda d: ms_record(d, name="blk.0.attn_norm.weight", shape=(N_IN, 1)))   # shape of another rank

    def bad_signs(d: Path) -> None:
        ms_record(d)
        p = d / "blk.0.attn_q.weight.ms"
        b = bytearray(p.read_bytes())
        b[-1] = 0
        p.write_bytes(bytes(b))
    refused(bad_signs)
    print("ms_pack: bad records refused")


def check_gguf_order() -> None:
    import torch
    import model_sections as msec
    from ms_convert_dense_spine import to_gguf_order
    g = torch.Generator().manual_seed(3)
    n_v = msec.GDN_V_HEADS * msec.GDN_HEAD
    w = torch.randn(8, n_v, generator=g)
    h = torch.randn(n_v, n_v, generator=g)
    x = torch.randn(n_v, generator=g)
    w2, h2 = to_gguf_order("blk.0.ssm_out.weight", w, h)
    x2 = x[torch.as_tensor(msec.gdn_index(n_v, 0))]   # the GGUF order of the input
    assert torch.allclose(w2 @ x2, w @ x, atol=1e-4)
    assert torch.allclose(x2 @ h2 @ x2, x @ h @ x, rtol=1e-4)
    rows = msec.GDN_QK_ROWS + n_v
    wq = torch.randn(rows, 4, generator=g)
    wq2, _ = to_gguf_order("blk.0.attn_qkv.weight", wq, torch.eye(4))
    assert torch.equal(wq2, wq[torch.as_tensor(msec.gdn_index(rows, msec.GDN_QK_ROWS))])
    assert torch.equal(wq2[:msec.GDN_QK_ROWS], wq[:msec.GDN_QK_ROWS])
    print("to_gguf_order: ssm_out columns and attn_qkv rows ok")


def check_experts(ks=(4, 3)) -> None:
    """Two layers of three experts (one K each): the blobs are [gate | up | down] per expert, the layout v8."""
    hid, ff, n_exp = 64, 32, 3
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        w = ModelWriter(str(td / "base.basalt"))
        q = bytes(N_OUT * N_IN // 32 * 34)
        offs = w.add_parts("gguf", [q])
        w.add_bytes("experts", b"old" * 10)
        w.add_bytes("experts.layout", b"# experts v5\n")
        meta = {"name": "t", "gguf_kv": {"qwen4exp.block_count": 2}, "tensors": {
            "blk.0.attn_q.weight": {"type": "Q8_0", "type_id": 8, "shape": [N_IN, N_OUT], "offset": offs[0],
                                    "bytes": len(q)}}}
        w.add_bytes("meta", json.dumps(meta).encode())
        w.close()
        ms = td / "ms"
        (ms / "experts").mkdir(parents=True)
        ms_record(ms)
        rng = np.random.default_rng(4)
        want = []
        for l, k in enumerate(ks):
            tens, per = {}, {}
            for role, (n_in, n_out) in zip(ms_pack.ROLES, ((hid, ff), (hid, ff), (ff, hid))):
                per[role] = [rng.integers(0, 256, ms_pack.payload_bytes(n_in, n_out, k), dtype=np.uint8).tobytes()
                             for _ in range(n_exp)]
                (ms / "experts" / f"blk.{l}.ffn_{role}_exps.ms").write_bytes(b"".join(per[role]))
                tens[f"blk.{l}.ffn_{role}_exps.weight"] = {"file": f"blk.{l}.ffn_{role}_exps.ms", "shape": [n_in, n_out]}
            (ms / "experts" / f"blk.{l}.experts.json").write_text(json.dumps(
                {"layer": l, "k": k, "n_expert": n_exp, "tensors": tens}))
            want += [per["gate"][e] + per["up"][e] + per["down"][e] for e in range(n_exp)]
        out = td / "out.basalt"
        assert ms_pack.main(["--base", str(td / "base.basalt"), "--ms-dir", str(ms), "--out", str(out), "--verify"]) == 0
        m = ModelFile(str(out))
        assert m.read("experts") == b"".join(want)
        lines = m.read("experts.layout").decode().splitlines()
        assert lines[0].startswith("# experts v8:") and "(n_expert 3," in lines[0], lines[0]
        b0, b1 = len(want[0]), len(want[3])
        t0, t1 = (120 + k for k in ks)
        assert lines[1:] == [f"0 {t0} {t0} 0 {b0} {t0}", f"1 {t1} {t1} {3 * b0} {b1} {t1}"], lines
        assert json.loads(m.read("meta"))["ms"]["experts"]["k"] == list(ks)
    print("ms_pack: experts interleaved, layout v8")


def check_mixed() -> None:
    """Layer 0 one K, layer 1 a K per expert: v9, mixed blobs = 16-byte head (K) + payloads, section padded by a slot."""
    hid, ff, n_exp, ks = 64, 32, 3, (4, [2, 5, 3])
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        w = ModelWriter(str(td / "base.basalt"))
        q = bytes(N_OUT * N_IN // 32 * 34)
        offs = w.add_parts("gguf", [q])
        w.add_bytes("experts", b"old" * 10)
        w.add_bytes("experts.layout", b"# experts v5\n")
        meta = {"name": "t", "gguf_kv": {"qwen4exp.block_count": 2}, "tensors": {
            "blk.0.attn_q.weight": {"type": "Q8_0", "type_id": 8, "shape": [N_IN, N_OUT], "offset": offs[0],
                                    "bytes": len(q)}}}
        w.add_bytes("meta", json.dumps(meta).encode())
        w.close()
        ms = td / "ms"
        (ms / "experts").mkdir(parents=True)
        ms_record(ms)
        rng = np.random.default_rng(5)
        want = []
        for l, k in enumerate(ks):
            kl = [k] * n_exp if isinstance(k, int) else k
            tens, per = {}, {}
            for role, (n_in, n_out) in zip(ms_pack.ROLES, ((hid, ff), (hid, ff), (ff, hid))):
                per[role] = [rng.integers(0, 256, ms_pack.payload_bytes(n_in, n_out, kl[e]), dtype=np.uint8).tobytes()
                             for e in range(n_exp)]
                (ms / "experts" / f"blk.{l}.ffn_{role}_exps.ms").write_bytes(b"".join(per[role]))
                tens[f"blk.{l}.ffn_{role}_exps.weight"] = {"file": f"blk.{l}.ffn_{role}_exps.ms", "shape": [n_in, n_out]}
            (ms / "experts" / f"blk.{l}.experts.json").write_text(json.dumps(
                {"layer": l, "k": k, "n_expert": n_exp, "tensors": tens}))
            head = (lambda e: b"") if isinstance(k, int) else (lambda e: bytes([kl[e]]) + bytes(15))
            want.append([head(e) + per["gate"][e] + per["up"][e] + per["down"][e] for e in range(n_exp)])
        out = td / "out.basalt"
        assert ms_pack.main(["--base", str(td / "base.basalt"), "--ms-dir", str(ms), "--out", str(out), "--verify"]) == 0
        m = ModelFile(str(out))
        slot1 = max(len(b) for b in want[1])
        body = b"".join(want[0]) + b"".join(want[1])
        assert m.read("experts") == body + bytes(max(len(want[0][0]), slot1))
        lines = m.read("experts.layout").decode().splitlines()
        assert lines[0].startswith("# experts v9:"), lines[0]
        l0 = len(b"".join(want[0]))
        assert lines[1:] == ["0 124 124 0 %d 124" % len(want[0][0]), "1 127 127 %d %d 127 253" % (l0, slot1)], lines
        assert json.loads(m.read("meta"))["ms"]["experts"]["k"] == [4, [2, 5, 3]]
    print("ms_pack: mixed experts, layout v9")


class MsPackTest(unittest.TestCase):
    def test_experts(self) -> None:
        check_experts()
        check_experts((5, 2))
        check_mixed()

    def test_pack(self) -> None:
        for k in ms_pack.MS_BITS:
            check_pack(k)

    def test_refused(self) -> None:
        check_refused()

    @unittest.skipUnless(HAVE_TORCH, "torch is not installed")
    def test_gguf_order(self) -> None:
        check_gguf_order()


if __name__ == "__main__":
    unittest.main()
