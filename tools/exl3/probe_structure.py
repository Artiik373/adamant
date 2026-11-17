"""Does a decoding hypothesis produce a real weight matrix?  (no reference model involved)

A random i.i.d. matrix has a flat (Marchenko-Pastur) singular-value spectrum.  Trained weights have a decaying one:
a few directions carry a disproportionate share of the energy.  We decode one expert projection under several
hypotheses about the on-disk conventions and report how concentrated each spectrum is; a wrong convention gives
noise.  The null row is the same shape of i.i.d. Gaussian noise.
"""
import sys
import numpy as np

sys.path.insert(0, ".")
import exl3_ref as e


def spectrum_stats(w: np.ndarray) -> str:
    s = np.linalg.svd(w.astype(np.float64), compute_uv=False)
    e2 = s ** 2
    tot = e2.sum()
    n = len(s)
    top1 = e2[: max(1, n // 100)].sum() / tot
    top10 = e2[: max(1, n // 10)].sum() / tot
    kurt = float(((w - w.mean()) ** 4).mean() / (w.var() ** 2))
    return f"top1% energy {top1:6.1%} | top10% {top10:6.1%} | kurtosis {kurt:6.2f} | rms {np.sqrt((w ** 2).mean()):.4f}"


def main(folder: str, layer: int = 0, expert: int = 0) -> None:
    ck = e.Checkpoint(folder)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        pre = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
        lin = ck.linear(pre)
        k, n = lin["trellis"].shape[0] * 16, lin["trellis"].shape[1] * 16
        print(f"\n{pre}  [{k} x {n}]  words/tile={lin['trellis'].shape[2]}  "
              f"suh: mean|x|={np.abs(lin['suh'].astype(np.float32)).mean():.4f}  svh: mean|x|={np.abs(lin['svh'].astype(np.float32)).mean():.4f}")
        rng = np.random.default_rng(0)
        print("  null (gaussian)         ", spectrum_stats(rng.standard_normal((k, n)).astype(np.float32) * 0.02))
        for variant in ("A", "B"):
            if variant == "B" and k != n:
                pass
            try:
                w_in = e.inner_matrix(lin["trellis"], variant)
            except Exception as ex:                                       # transposed layout may not fit the shape
                print(f"  variant {variant}: {ex}")
                continue
            print(f"  inner, variant {variant}       ", spectrum_stats(w_in))
            w_full = e.full_weight(lin, variant)
            print(f"  full,  variant {variant}       ", spectrum_stats(w_full))


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
