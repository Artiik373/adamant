"""The two-stream conversion's closed forms in ms_flashnext_convert against their objectives."""
import unittest

import torch

from ms_flashnext_convert import inject, input_map, ridge_refit, stream_target


class TwoStreamTest(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(5)
        self.g = g
        self.n, self.d_in, self.d_out = 4096, 48, 64
        self.xr = torch.randn(self.n, self.d_in, generator=g, dtype=torch.float64)
        self.xq = self.xr + 0.2 * torch.randn(self.n, self.d_in, generator=g, dtype=torch.float64)   # converted inputs
        self.w = torch.randn(self.d_out, self.d_in, generator=g, dtype=torch.float64)

    def test_stream_target(self):
        # y minimises sum_s ||d_s - w_s y||^2; with d = inject(0, y0, w) it recovers y0 exactly
        hc, hid = 4, 16
        w = 2 * torch.rand(32, hc, generator=self.g, dtype=torch.float64)
        y0 = torch.randn(32, hid, generator=self.g, dtype=torch.float64)
        t, c = stream_target(inject(torch.zeros(32, hc * hid, dtype=torch.float64), y0, w), w)
        self.assertTrue(torch.allclose(t.double(), y0, atol=1e-5))
        self.assertTrue(torch.allclose(c.double(), (w * w).sum(-1), atol=1e-5))
        # on any d, the residual is orthogonal to w: sum_s w_s (d_s - w_s t) = 0
        d = torch.randn(32, hc * hid, generator=self.g, dtype=torch.float64)
        t, _ = stream_target(d, w)
        res = d.unflatten(-1, (hc, hid)) - w.unsqueeze(-1) * t.double().unsqueeze(-2)
        self.assertLess((w.unsqueeze(-1) * res).sum(-2).abs().max().item(), 1e-5)

    def test_input_map(self):
        # W M minimises sum ||W x_ref - W M x_q||^2 + r ||W M - W||^2: its gradient in M vanishes
        h, c = self.xq.T @ self.xq / self.n, self.xr.T @ self.xq / self.n
        wm = input_map(self.w, h, c)
        r = 0.01 * h.diagonal().mean()
        m = torch.linalg.solve(self.w.T @ self.w, self.w.T @ wm)   # back to M: W has more rows than columns
        eye = torch.eye(self.d_in, dtype=h.dtype)
        self.assertLess((m @ (h + r * eye) - (c + r * eye)).abs().max().item(), 1e-8)
        err = lambda ww: ((self.xr @ self.w.T - self.xq @ ww.T) ** 2).mean().item()
        self.assertLess(err(wm), err(self.w))

    def test_ridge_refit(self):
        # weighted targets: gradient of sum_i c_i ||y_i - W' z_i||^2 + lam ||W'||^2 + mu ||W' - W||^2 vanishes
        y = self.xr @ self.w.T + 0.1 * torch.randn(self.n, self.d_out, generator=self.g, dtype=torch.float64)
        c = torch.rand(self.n, generator=self.g, dtype=torch.float64) + 0.5
        z = self.xq
        hh, cc = (z * c[:, None]).T @ z, (z * c[:, None]).T @ y
        wt = ridge_refit(self.w, hh, cc)
        d = hh.diagonal().mean()
        grad = (z * c[:, None]).T @ (z @ wt.T - y) + 1e-4 * d * wt.T + 0.05 * d * (wt - self.w).T
        self.assertLess(grad.abs().max().item() / cc.abs().max().item(), 1e-8)


if __name__ == "__main__":
    unittest.main()
