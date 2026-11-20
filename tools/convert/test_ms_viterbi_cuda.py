"""ms_viterbi_cuda.py against the plain PyTorch Viterbi in ms_moe_converter: same states and error, batched == single.

    python -m unittest discover -s tools/convert -p "test_ms_viterbi_cuda.py"
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

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
        for k in (2, 3, 4, 6):
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


if __name__ == "__main__":
    unittest.main()
