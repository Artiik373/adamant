"""ms_viterbi_cuda.py against the plain PyTorch Viterbi in ms_moe_converter: same states and error, batched == single.

    python -m unittest discover -s tools/convert -p "test_ms_viterbi_cuda.py"
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    raise unittest.SkipTest("needs torch")

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

    def test_scale_refit_never_worse(self):
        from ms_flashnext_convert import refit_scales
        import ms_viterbi_cuda as mv

        dev = torch.device("cuda")
        lut = ms.build_1mad_lut(dev)
        g = torch.Generator(device=dev).manual_seed(4)
        w = torch.randn(2, 32, 256, device=dev, generator=g)
        x = torch.randn(1024, 256, device=dev, generator=g)
        h = (x.T @ x / 1024).expand(2, 256, 256).contiguous()
        hu = torch.stack([ms.cholesky_inv_upper(m) for m in h])
        sc, st, q = mv.viterbi_code_batched_cuda(w, hu, 3, lut)
        s2, q2 = refit_scales(w, st, sc, h, lut)
        err = lambda q_: (torch.bmm(w - q_, h) * (w - q_)).sum(-1)
        self.assertTrue(bool((err(q2) <= err(q) * (1 + 1e-6)).all()))
        self.assertLess(err(q2).sum().item(), err(q).sum().item())

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

    def test_sign_trials_and_two_pass_scale_improve_heavy_tailed(self):
        from ms_flashnext_convert import code

        dev = torch.device("cuda")
        lut = ms.build_1mad_lut(dev)
        g = torch.Generator(device=dev).manual_seed(17)
        # Real down_proj geometry: n_out=2560, n_in=640 (blk=128) with heavy-tailed outliers
        w = torch.randn(1, 256, 640, device=dev, generator=g)
        w[0, :32, :16] *= 8.0
        x = torch.randn(1024, 640, device=dev, generator=g)
        x[:, :64] *= 3.0
        h = (x.T @ x / 1024).unsqueeze(0)

        _, _, deq_base = code(w, h, 3, lut, [1], sign_trials=1, scale_passes=1)[0]
        _, _, deq_2pass = code(w, h, 3, lut, [1], sign_trials=1, scale_passes=2)[0]
        _, _, deq_opt = code(w, h, 3, lut, [1], sign_trials=8, scale_passes=2)[0]

        def h_err(deq):
            e = w[0] - deq
            return ((e @ h[0]) * e).sum().item()

        e_base, e_2pass, e_opt = h_err(deq_base), h_err(deq_2pass), h_err(deq_opt)
        print(f"H-err heavy-tailed 256x640 MS_C3: base={e_base:.4f}, 2pass={e_2pass:.4f}, 2pass+sign8={e_opt:.4f}")
        self.assertLessEqual(e_2pass, e_base)
        self.assertLess(e_opt, e_base)

    def test_expert_hessians_tok_w(self):
        from ms_flashnext_convert import expert_hessians

        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(21)
        x = torch.randn(32, 64, device=dev, generator=g)
        idx = torch.zeros(32, 2, dtype=torch.long, device=dev)
        w_gu = torch.randn(32, 64, device=dev, generator=g)
        tok_w = torch.linspace(0.5, 1.5, 32, device=dev)
        s1, d1, n1 = expert_hessians(x, idx, 0, w_gu)
        s2, d2, n2 = expert_hessians(x, idx, 0, w_gu, tok_w=tok_w)
        self.assertEqual(n1, 32)
        self.assertEqual(n2, 32)
        self.assertFalse(torch.allclose(s1, s2))
        self.assertFalse(torch.allclose(d1, d2))

    def test_code_experts_cold_frac(self):
        import tempfile
        from types import SimpleNamespace
        from ms_flashnext_convert import code_experts

        dev = torch.device("cuda")
        lut = ms.build_1mad_lut(dev)
        g = torch.Generator(device=dev).manual_seed(33)
        n_exp, n_ff, n_embd = 4, 32, 64
        moe = SimpleNamespace(
            experts=SimpleNamespace(
                gate_up_proj=torch.randn(n_exp, 2 * n_ff, n_embd, device=dev, generator=g),
                down_proj=torch.randn(n_exp, n_embd, n_ff, device=dev, generator=g),
            )
        )
        x = torch.randn(40, n_embd, device=dev, generator=g)
        # Experts 0,1 hot; experts 2,3 cold (zero tokens)
        idx = torch.zeros(40, 2, dtype=torch.long, device=dev)
        idx[:, 1] = 1
        with tempfile.TemporaryDirectory() as td:
            rec = code_experts(moe, x, idx, [3, 4, 3, 4], lut, 0, Path(td), cold_frac=0.5)
            self.assertEqual(rec["zero_token_experts"], 2)
            for proj in ("gate", "up", "down"):
                self.assertTrue((Path(td) / f"blk.0.ffn_{proj}_exps.ms").stat().st_size > 0)

    def test_code_experts_per_proj(self):
        import tempfile
        from types import SimpleNamespace
        from ms_flashnext_convert import code_experts
        from ms_pack import payload_bytes

        dev = torch.device("cuda")
        lut = ms.build_1mad_lut(dev)
        g = torch.Generator(device=dev).manual_seed(44)
        n_exp, n_ff, n_embd = 2, 32, 64
        moe = SimpleNamespace(
            experts=SimpleNamespace(
                gate_up_proj=torch.randn(n_exp, 2 * n_ff, n_embd, device=dev, generator=g),
                down_proj=torch.randn(n_exp, n_embd, n_ff, device=dev, generator=g),
            )
        )
        x = torch.randn(20, n_embd, device=dev, generator=g)
        idx = torch.zeros(20, 2, dtype=torch.long, device=dev)
        idx[:, 1] = 1
        ks = [[4, 2, 5], [3, 3, 4]]
        with tempfile.TemporaryDirectory() as td:
            code_experts(moe, x, idx, ks, lut, 0, Path(td))
            self.assertEqual((Path(td) / "blk.0.ffn_gate_exps.ms").stat().st_size,
                             sum(payload_bytes(n_embd, n_ff, k[0]) for k in ks))
            self.assertEqual((Path(td) / "blk.0.ffn_up_exps.ms").stat().st_size,
                             sum(payload_bytes(n_embd, n_ff, k[1]) for k in ks))
            self.assertEqual((Path(td) / "blk.0.ffn_down_exps.ms").stat().st_size,
                             sum(payload_bytes(n_ff, n_embd, k[2]) for k in ks))

    def test_sensitivity_scan_errs_proj(self):
        from types import SimpleNamespace
        from ms_sensitivity_scan import quantize_experts

        dev = torch.device("cuda")
        lut = ms.build_1mad_lut(dev)
        g = torch.Generator(device=dev).manual_seed(55)
        n_exp, n_ff, n_embd = 2, 128, 128
        moe = SimpleNamespace(
            experts=SimpleNamespace(
                gate_up_proj=torch.randn(n_exp, 2 * n_ff, n_embd, device=dev, generator=g),
                down_proj=torch.randn(n_exp, n_embd, n_ff, device=dev, generator=g),
            )
        )
        x = torch.randn(16, n_embd, device=dev, generator=g)
        idx = torch.zeros(16, 2, dtype=torch.long, device=dev)
        idx[:, 1] = 1
        errs, errs_proj = {}, {"gate": {}, "up": {}, "down": {}}
        quantize_experts(moe, x, idx, 3, lut, [0, 1], eb=2, errs=errs, errs_proj=errs_proj)
        self.assertEqual(set(errs.keys()), {0, 1})
        for p in ("gate", "up", "down"):
            self.assertEqual(set(errs_proj[p].keys()), {0, 1})
            self.assertTrue(all(v > 0.0 for v in errs_proj[p].values()))

    def test_sensitivity_scan_ldlq_proxy(self):
        from types import SimpleNamespace
        from ms_sensitivity_scan import quantize_experts

        dev = torch.device("cuda")
        lut = ms.build_1mad_lut(dev)
        g = torch.Generator(device=dev).manual_seed(66)
        n_exp, n_ff, n_embd = 2, 128, 128
        moe = SimpleNamespace(
            experts=SimpleNamespace(
                gate_up_proj=torch.randn(n_exp, 2 * n_ff, n_embd, device=dev, generator=g),
                down_proj=torch.randn(n_exp, n_embd, n_ff, device=dev, generator=g),
            )
        )
        x = torch.randn(16, n_embd, device=dev, generator=g)
        idx = torch.zeros(16, 2, dtype=torch.long, device=dev)
        idx[:, 1] = 1
        errs, errs_proj = {}, {"gate": {}, "up": {}, "down": {}}
        quantize_experts(moe, x, idx, 3, lut, [0, 1], eb=2, errs=errs, mode="ldlq", errs_proj=errs_proj)
        self.assertEqual(set(errs.keys()), {0, 1})
        self.assertTrue(all(v > 0.0 for v in errs.values()))
        for p in ("gate", "up", "down"):
            self.assertTrue(all(v > 0.0 for v in errs_proj[p].values()))

    def test_sensitivity_scan_proxy_multi_parity(self):
        from types import SimpleNamespace
        from ms_sensitivity_scan import quantize_experts, quantize_experts_proxy_multi

        dev = torch.device("cuda")
        lut = ms.build_1mad_lut(dev)
        g = torch.Generator(device=dev).manual_seed(67)
        n_exp, n_ff, n_embd = 3, 128, 128
        gu0 = torch.randn(n_exp, 2 * n_ff, n_embd, device=dev, dtype=torch.bfloat16, generator=g)
        dn0 = torch.randn(n_exp, n_embd, n_ff, device=dev, dtype=torch.bfloat16, generator=g)
        moe = SimpleNamespace(
            experts=SimpleNamespace(gate_up_proj=gu0.clone(), down_proj=dn0.clone())
        )
        x = torch.randn(24, n_embd, device=dev, dtype=torch.bfloat16, generator=g)
        idx = torch.randint(0, n_exp, (24, 2), device=dev, generator=g)
        errs_m, ep_m = quantize_experts_proxy_multi(moe, x, idx, [2, 3, 4], [0, 1, 2], eb=2, tau=16.0, l=0)
        for k in (2, 3, 4):
            moe.experts.gate_up_proj.copy_(gu0)
            moe.experts.down_proj.copy_(dn0)
            errs_s, ep_s = {}, {"gate": {}, "up": {}, "down": {}}
            quantize_experts(moe, x, idx, k, lut, [0, 1, 2], eb=2, tau=16.0, l=0,
                             errs=errs_s, mode="ldlq", errs_proj=ep_s)
            for e in (0, 1, 2):
                self.assertLess(abs(errs_m[k][e] - errs_s[e]) / max(1.0, errs_s[e]), 1e-5)
                for p in ("gate", "up", "down"):
                    self.assertLess(abs(ep_m[p][k][e] - ep_s[p][e]) / max(1.0, ep_s[p][e]), 1e-5)


if __name__ == "__main__":
    unittest.main()
