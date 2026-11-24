"""Per-layer MS expert widths under a byte budget: the plan with the least predicted KL.

Costs: JSON {"<layer>": {"<k>": predicted KL, ...}, ...}, one entry per layer and every width allowed for it.
Each layer starts at its narrowest width; the greedy then takes, step by step, the one-width step with the largest KL
drop per byte while it fits. With KL convex in bytes per layer this is optimal (Fox 1966); a layer whose costs are
not convex is made so (its lower hull) and the skipped width is never chosen.

    python ms_plan_solver.py --costs costs.json --budget-gib 46.84 --out plan.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ms_pack import payload_bytes


def layer_bytes(k: int, n_embd: int = 2560, n_ff: int = 640, n_exp: int = 512) -> int:
    """One layer's routed experts at width k: gate and up [n_ff, n_embd], down [n_embd, n_ff], each with its signs."""
    return n_exp * (2 * payload_bytes(n_embd, n_ff, k) + payload_bytes(n_ff, n_embd, k))


def hull(costs: dict, size) -> list:
    """Widths on the lower convex hull of (bytes, KL), narrowest first."""
    pts = sorted((size(k), f, k) for k, f in costs.items())
    out: list = []
    for p in pts:
        if out and p[1] >= out[-1][1]:   # a wider width that is not better is never worth its bytes
            continue
        while len(out) >= 2 and (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0]) >= (p[1] - out[-2][1]) * (out[-1][0] - out[-2][0]):
            out.pop()
        out.append(p)
    return [k for _, _, k in out]


def solve(costs: dict, budget: int, size=layer_bytes, min_k: int = 0) -> dict:
    """{layer: k >= min_k} with sum of bytes <= budget and the least sum of predicted KL."""
    # The floor keeps an additive cost sum away from widths whose real (joint) loss it underestimates
    f = {l: {int(k): float(v) for k, v in c.items() if int(k) >= min_k} for l, c in costs.items()}
    if not all(f.values()):
        raise SystemExit(f"no width >= {min_k} for layer(s) {[l for l, c in f.items() if not c]}")
    steps = {l: hull(c, size) for l, c in f.items()}
    pick = {l: 0 for l in steps}
    used = sum(size(s[0]) for s in steps.values())
    if used > budget:
        raise SystemExit(f"budget {budget} B is below the narrowest plan ({used} B)")
    while True:
        best = None
        for l, s in steps.items():
            if pick[l] + 1 < len(s):
                k0, k1 = s[pick[l]], s[pick[l] + 1]
                db = size(k1) - size(k0)
                if used + db <= budget and (best is None or (f[l][k0] - f[l][k1]) / db > best[0]):
                    best = ((f[l][k0] - f[l][k1]) / db, l, db)
        if best is None:
            return {l: steps[l][pick[l]] for l in steps}
        pick[best[1]] += 1
        used += best[2]


def solve_items(costs: dict, budget: int, size) -> dict:
    """solve() for many items (every expert of every layer): the same greedy with a heap of each item's next step.
    A step that no longer fits ends its item (the steps left are smaller in gain per byte)."""
    import heapq
    steps = {i: hull(c, size) for i, c in costs.items()}
    pick = {i: 0 for i in steps}
    used = sum(size(s[0]) for s in steps.values())
    if used > budget:
        raise SystemExit(f"budget {budget} B is below the narrowest plan ({used} B)")

    def push(i):
        s = steps[i]
        if pick[i] + 1 < len(s):
            k0, k1 = s[pick[i]], s[pick[i] + 1]
            db = size(k1) - size(k0)
            heapq.heappush(heap, (-(costs[i][k0] - costs[i][k1]) / db, i, db))
    heap: list = []
    for i in steps:
        push(i)
    while heap:
        _, i, db = heapq.heappop(heap)
        if used + db > budget:
            continue
        pick[i] += 1
        used += db
        push(i)
    return {i: steps[i][pick[i]] for i in steps}


def expert_costs(scan_dir: Path, ref_k: int = 3, max_k: int = 6) -> dict:
    """{(layer, expert): {k: cost}} from ms_sensitivity_scan --per-expert: each expert's output error, scaled per layer
    so the layer's sum at ref_k is its measured squared layer-output error there (comparable across layers)."""
    rows = {r["layer"]: r for r in map(json.loads, (scan_dir / "scan.jsonl").read_text().splitlines()) if r}
    out = {}
    for l, r in sorted(rows.items()):
        errs = json.loads((scan_dir / f"experts_{l}.json").read_text())["errs"]
        ks = sorted(int(k) for k in errs if int(k) <= max_k)
        tot = sum(errs[str(ref_k)].values())
        c = r[f"C{ref_k}"]["out_rel"] ** 2 / tot if tot > 0 else 0.0
        for e in errs[str(ref_k)]:
            out[(l, int(e))] = {k: c * errs[str(k)][e] for k in ks}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--costs", type=Path, help="per-layer costs JSON")
    p.add_argument("--experts-dir", type=Path,
                   help="instead: an ms_sensitivity_scan --per-expert folder; the plan gets one K per expert")
    p.add_argument("--max-k", type=int, default=6, help="with --experts-dir: the widest K any expert may take")
    p.add_argument("--budget-gib", type=float, required=True, help="routed experts of all layers")
    p.add_argument("--out", type=Path, required=True, help='the converter plan {"experts_k": [k per layer]}')
    p.add_argument("--min-k", type=int, default=0, help="narrowest width any layer may take (default: no floor)")
    args = p.parse_args()
    if args.experts_dir is not None:
        costs = expert_costs(args.experts_dir, max_k=args.max_k)
        n_l = 1 + max(l for l, _ in costs)
        n_e = 1 + max(e for _, e in costs)
        size = lambda k: layer_bytes(k, n_exp=1)
        plan = solve_items(costs, int(args.budget_gib * 2**30), size)
        if args.min_k:
            raise SystemExit("--min-k is for per-layer plans")
        ks = [[plan[(l, e)] for e in range(n_e)] for l in range(n_l)]
        args.out.write_text(json.dumps({"experts_k": ks, "source": str(args.experts_dir)}))
        flat = [k for row in ks for k in row]
        gib = sum(size(k) for k in flat) / 2**30
        print(f"plan: " + ", ".join(f"{flat.count(k)} C{k}" for k in sorted(set(flat))) + f"; {gib:.2f} GiB; "
              f"predicted {sum(costs[i][plan[i]] for i in plan):.5f}")
        return
    costs = json.loads(args.costs.read_text())
    plan = solve(costs, int(args.budget_gib * 2**30), min_k=args.min_k)
    ks = [plan[l] for l in sorted(plan, key=int)]
    args.out.write_text(json.dumps({"experts_k": ks}))
    gib = sum(layer_bytes(k) for k in ks) / 2**30
    kl = sum(float(costs[l][str(plan[l])]) for l in plan)
    print(f"plan: {ks.count(2)} C2, {ks.count(3)} C3, {ks.count(4)} C4, {ks.count(5)} C5; {gib:.2f} GiB; predicted KL {kl:.5f}")


if __name__ == "__main__":
    main()
