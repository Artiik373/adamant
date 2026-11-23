"""Best-of-4 1MAD constants probe: codes real expert rows (Flash-Next) or MLP rows (a dense model) with the format's 1MAD constants and with 3
others (identity Hessian, same rotation), and reports how much a per-row choice of the 4 would lower the MSE.
Decides whether a 2-bit hash selector in the row header is worth a format change.

python ms_hash_salt_probe.py --hf-dir C:/llm/models/Qwen3.8-Flash-Next --rows 200 --k 3
"""
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load

import ms_moe_converter as ms
import ms_viterbi_cuda as mvc
from ms_convert_dense_spine import largest_pow2_divisor

# (multiplier, addend); the first pair is the format's
CONSTS = [(34038481, 76625530), (89226354 | 1, 64248391), (2654435761, 40503), (1103515245, 12345)]


def use_consts(i: int, mul: int, add: int, dev):
    flags = ["-O3", "--use_fast_math", f"-DMS_MAD_MUL={mul}u", f"-DMS_MAD_ADD={add}u"]
    if os.name == "nt":
        flags += ["-Xcompiler", "/Zc:preprocessor"]
    mvc._EXT = load(name=f"ms_viterbi_cuda_probe{i}", sources=[str(Path(mvc.__file__).parent / "ms_viterbi_cuda.cu")],
                    extra_cuda_cflags=flags, verbose=False)
    ms.MAD_MUL, ms.MAD_ADD = mul, add
    lut = ms.build_1mad_lut(dev)
    ms.CODEBOOK_STD = float(lut.std())   # the row scale assumes the codebook's own std
    return lut


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dir", type=Path, required=True)
    p.add_argument("--rows", type=int, default=200, help="rows per tensor kind (gate_up, down)")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--layers", type=int, nargs="+", default=[2, 20, 44])
    args = p.parse_args()
    dev = torch.device("cuda")
    wmap = json.loads((args.hf_dir / "model.safetensors.index.json").read_text())["weight_map"]
    g = torch.Generator().manual_seed(0)
    mats = {"gate_up": [], "down": []}
    for l in args.layers:
        for kind in mats:
            key = f"model.language_model.layers.{l}.mlp.experts.{kind}_proj"
            dense = key not in wmap   # a dense model (Qwen3-1.7B): its MLP's gate and down
            if dense:
                key = f"model.layers.{l}.mlp.{'gate' if kind == 'gate_up' else 'down'}_proj.weight"
            with safe_open(args.hf_dir / wmap[key], "pt") as f:
                t = f.get_slice(key)
                if dense:
                    w = t[:].float()
                else:
                    e = int(torch.randint(t.get_shape()[0], (1,), generator=g))
                    w = t[e].float()
            r = torch.randperm(w.shape[0], generator=g)[: args.rows // len(args.layers)]
            mats[kind].append(w[r])
    keep = (ms.MAD_MUL, ms.MAD_ADD, ms.CODEBOOK_STD)
    for kind, ws in mats.items():
        w = torch.cat(ws).to(dev)
        blk = largest_pow2_divisor(w.shape[1], cap=512)
        w_rot, h_rot, _ = ms.rotate_weight_and_hessian(w, torch.eye(w.shape[1], device=dev), block_size=blk, seed=1)
        hu = ms.cholesky_inv_upper(h_rot).unsqueeze(0)
        mse = []
        for i, (mul, add) in enumerate(CONSTS):
            lut = use_consts(i, mul, add, dev)
            _, _, q = mvc.viterbi_code_batched_cuda(w_rot.unsqueeze(0), hu, args.k, lut, 32, 8)
            mse.append(((q[0] - w_rot) ** 2).mean(1) / (w_rot ** 2).mean(1))   # relative MSE per row
            print(f"{kind:7s} consts {i} lut std {ms.CODEBOOK_STD:6.1f}: mean rel MSE {mse[-1].mean():.5f}", flush=True)
        m = torch.stack(mse)
        best = m.min(0).values.mean()
        print(f"{kind:7s} rows {m.shape[1]}: format {m[0].mean():.5f}, best single {m.mean(1).min():.5f}, "
              f"per-row best of 4 {best:.5f} ({100 * (1 - best / m[0].mean()):.2f}% below format), "
              f"rows where format wins {(m.argmin(0) == 0).float().mean():.2f}")
    ms.MAD_MUL, ms.MAD_ADD, ms.CODEBOOK_STD = keep
    mvc._EXT = None


if __name__ == "__main__":
    main()
