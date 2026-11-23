"""The MS drafter's mtp/dense forms (ms_flashnext_convert.mtp_index_entry) and ms_pack's check of a drafter folder."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import ms_pack


class StubBase:
    """The two things read_mtp reads from a base pack."""

    def __init__(self, index: str):
        self.sections = {n: (0, 0) for n in ms_pack.MTP_SECTIONS}
        self.index = index

    def read(self, name):
        assert name == "mtp/dense.index"
        return self.index.encode("utf-8")


def write_experts(d: Path, k: int, n_expert: int, n_embd: int, n_ff: int) -> int:
    """One layer of expert records as code_experts writes them; returns the blob bytes."""
    tensors, blob = {}, 0
    for role, n_in, n_out in (("gate", n_embd, n_ff), ("up", n_embd, n_ff), ("down", n_ff, n_embd)):
        per = ms_pack.payload_bytes(n_in, n_out, k)
        (d / f"blk.0.ffn_{role}_exps.ms").write_bytes(bytes(per * n_expert))
        tensors[f"blk.0.ffn_{role}_exps.weight"] = {"file": f"blk.0.ffn_{role}_exps.ms", "shape": [n_in, n_out]}
        blob += per
    (d / "blk.0.experts.json").write_text(json.dumps({"layer": 0, "k": k, "n_expert": n_expert, "tensors": tensors}))
    return blob


class ReadMtpTest(unittest.TestCase):
    def folder(self, d: Path, first=None, ms_bytes=None, names=("a.weight", "b.weight")):
        blob = write_experts(d, 2, 2, 64, 32)
        ms = ms_bytes or 8 * ms_pack.ms_row_bytes(64, 5) + 64
        lines = [first or f"#experts ms 122 122 {blob}", f"{names[0]} ms 8 64 0 {ms}", f"{names[1]} f32 1 64 {ms} 256"]
        (d / "mtp.dense.index").write_text("\n".join(lines) + "\n")
        (d / "mtp.dense").write_bytes(bytes(ms + 256))

    def test_good(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            self.folder(d)
            index, dense, ex = ms_pack.read_mtp(d, StubBase("a.weight q8_0 8 64 0 1\nb.weight f32 1 64 0 1\n"))
            self.assertEqual(ex.layers[0][0], 2)
            self.assertEqual(dense, d / "mtp.dense")
            self.assertTrue(index.startswith("#experts ms 122 122 "))

    def test_refused(self):
        base = StubBase("a.weight q8_0 8 64 0 1\nb.weight f32 1 64 0 1\n")
        for kw in ({"first": "#experts ms 122 122 1"},          # blob size not the records'
                   {"ms_bytes": 999},                           # no MS payload of [8, 64]
                   {"names": ("a.weight", "c.weight")}):        # tensors differ from the base pack's
            with tempfile.TemporaryDirectory() as t:
                self.folder(Path(t), **kw)
                with self.assertRaises(ms_pack.PackError):
                    ms_pack.read_mtp(Path(t), base)


class IndexEntryTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("needs torch")
        from ms_flashnext_convert import mtp_index_entry
        self.torch, self.entry = torch, mtp_index_entry

    def test_forms(self):
        torch = self.torch
        w = torch.randn(4, 64)
        kind, r, c, raw = self.entry("self_attn.q_norm.weight", torch.full((64,), 0.25), {})
        self.assertEqual((kind, r, c), ("f32", 1, 64))
        self.assertTrue(np.allclose(np.frombuffer(raw, "<f4"), 1.25))   # the engine's norms hold 1 + w
        kind, r, c, raw = self.entry("fc_hidden.weight", w, {})
        self.assertEqual((kind, r, c, len(raw)), ("q8_0", 4, 64, 4 * 2 * 34))
        kind, r, c, raw = self.entry("mlp.gate.weight", w, {})
        self.assertEqual((kind, r, c, len(raw)), ("bf16", 4, 64, 4 * 64 * 2))
        coded = {"self_attn.o_proj": (b"x" * 12, np.ones(64, np.int8), 4, 64)}
        kind, r, c, raw = self.entry("self_attn.o_proj.weight", w, coded)
        self.assertEqual((kind, r, c, len(raw)), ("ms", 4, 64, 12 + 64))


if __name__ == "__main__":
    unittest.main()
