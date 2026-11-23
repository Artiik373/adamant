"""ms_viterbi_cuda.py against the plain PyTorch Viterbi in ms_moe_converter: same states and error, batched == single.

    python -m unittest discover -s tools/convert -p "test_ms_viterbi_cuda.py"
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ms_moe_converter as ms  # noqa: E402


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class ViterbiCudaParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ms_viterbi_cuda as mv

        cls.mv = mv
        cls.dev = torch.device("cuda")
        cls.lut = ms.build_1mad_lut(cls.dev)

    def case(self, out_f, in_f, seed):
        g = torch.Generator(device=self.dev).manual_seed(seed)
        w = torch.randn(out_f, in_f, device=self.dev, generator=g)
        x = torch.randn(4 * in_f, in_f, device=self.dev, generator=g)
        return w, ms.cholesky_inv_upper(x.T @ x)

    def test_matches_reference(self):
        for k in (2, 3, 4, 5, 6):
            w, hu = self.case(32, 128, k)
            _, s_ref, q_ref = ms.viterbi_code_matrix(w, hu, k=k, lut=self.lut)
            _, s, q = self.mv.viterbi_code_matrix_cuda(w, hu, k=k, lut=self.lut)
            match = (s.cpu() == s_ref.cpu()).float().mean().item()
            e_ref = ((q_ref.to(self.dev) - w).norm() / w.norm()).item()
            e = ((q - w).norm() / w.norm()).item()
            # LDLQ summation order differs, so a few paths may diverge; the error must not
            self.assertGreater(match, 0.98, f"k={k}")
            self.assertLess(abs(e - e_ref), 1e-3 * e_ref + 1e-5, f"k={k}")

    def test_batched_equals_single(self):
        w0, hu0 = self.case(64, 256, 10)
        w1, hu1 = self.case(64, 256, 11)
        _, s, _ = self.mv.viterbi_code_batched_cuda(torch.stack([w0, w1]), torch.stack([hu0, hu1]), 4, self.lut)
        for i, (w, hu) in enumerate(((w0, hu0), (w1, hu1))):
            _, s1, _ = self.mv.viterbi_code_matrix_cuda(w, hu, k=4, lut=self.lut)
            self.assertTrue(torch.equal(s[i], s1))


class PackRows(unittest.TestCase):
    def test_matches_reference_packer(self):
        from ms_flashnext_convert import pack_rows

        g = torch.Generator().manual_seed(3)
        for k in (2, 3, 4, 5, 6):
            states = torch.randint(0, ms.NUM_STATES, (5, 64), generator=g, dtype=torch.int32)
            scales = torch.rand(5, generator=g).to(torch.float16)
            self.assertEqual(pack_rows(states, scales, k), ms.pack_per_row_ms(scales, states, k), f"k={k}")


def decode_payload(rows: bytes, signs: np.ndarray, n_out: int, n_in: int, k: int, lut: torch.Tensor) -> torch.Tensor:
    """W from MS rows as the format describes them: states from the start state and k-bit symbols, 1MAD value times the
    row scale, then the inverse of the input rotation."""
    from ms_convert_dense_spine import largest_pow2_divisor
    from ms_pack import ms_row_bytes

    rb = ms_row_bytes(n_in, k)
    buf = np.frombuffer(rows, dtype=np.uint8).reshape(n_out, rb)
    scale = torch.from_numpy(buf[:, 0:2].copy().view(np.float16).astype(np.float32)).reshape(n_out)
    state = buf[:, 2:4].copy().view(np.uint16).astype(np.int64).reshape(n_out)
    bits = np.unpackbits(buf[:, 4:], axis=1)[:, : (n_in - 1) * k].reshape(n_out, n_in - 1, k)
    sym = (bits * (1 << np.arange(k - 1, -1, -1))).sum(-1)
    states = np.empty((n_out, n_in), dtype=np.int64)
    states[:, 0] = state & ms.STATE_MASK
    for c in range(1, n_in):
        states[:, c] = ((states[:, c - 1] << k) | sym[:, c - 1]) & ms.STATE_MASK
    w_rot = lut.cpu()[torch.from_numpy(states)] * scale.unsqueeze(1)
    blk = largest_pow2_divisor(n_in, cap=512)
    return ms.fwht_blocks(w_rot, block_size=blk) * torch.from_numpy(signs.astype(np.float32)).unsqueeze(0)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class ConverterPayload(unittest.TestCase):
    def test_payload_decodes_to_coded_weights(self):
        from ms_flashnext_convert import code

        dev = torch.device("cuda")
        lut = ms.build_1mad_lut(dev)
        g = torch.Generator(device=dev).manual_seed(5)
        for k, (n_out, n_in) in ((2, (16, 640)), (3, (16, 640)), (4, (24, 512)), (5, (16, 512)), (6, (8, 1024))):
            w = torch.randn(2, n_out, n_in, device=dev, generator=g)
            x = torch.randn(4 * n_in, n_in, device=dev, generator=g)
            h = (x.T @ x).expand(2, n_in, n_in).contiguous()
            for rows, signs, deq in code(w, h, k, lut, [7, 8]):
                got = decode_payload(rows, signs, n_out, n_in, k, lut)
                self.assertLess((got - deq.cpu()).abs().max().item(), 1e-3 * deq.abs().max().item(), f"k={k}")

    def test_mse_falls_with_k(self):
        from ms_flashnext_convert import code

        dev = torch.device("cuda")
        lut = ms.build_1mad_lut(dev)
        n_out, n_in = 64, 512
        w = torch.randn(1, n_out, n_in, device=dev, generator=torch.Generator(device=dev).manual_seed(9))
        h = torch.eye(n_in, device=dev).unsqueeze(0)
        mse = {}
        for k in (2, 3, 4, 5, 6):
            rows, signs, _ = code(w, h, k, lut, [3])[0]
            mse[k] = ((decode_payload(rows, signs, n_out, n_in, k, lut) - w[0].cpu()) ** 2).mean().item()
        print("MSE by K:", {k: f"{v:.5f}" for k, v in mse.items()})
        self.assertTrue(all(mse[k] > mse[k + 1] for k in (2, 3, 4, 5)), mse)


if __name__ == "__main__":
    unittest.main()
