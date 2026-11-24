"""ms_plan_solver against a brute force over all plans of a few layers.

    python -m unittest discover -s tools/convert -p "test_ms_plan_solver.py"
"""
import itertools
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ms_plan_solver import layer_bytes, solve, solve_items  # noqa: E402


class PlanSolver(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = random.Random(1)
        for _ in range(30):
            # convex KL per layer: roughly a quarter per extra bit, a random scale per layer
            costs = {str(l): {str(k): rng.uniform(0.2, 2.0) * 4.0 ** -k * rng.uniform(0.8, 1.2) ** (k == 3) for k in (2, 3, 4, 5)}
                     for l in range(5)}
            budget = int(rng.uniform(5 * layer_bytes(2), 5 * layer_bytes(5)))
            plan = solve(costs, budget)
            best = min((sum(costs[str(l)][str(k)] for l, k in enumerate(ks)), ks)
                       for ks in itertools.product((2, 3, 4, 5), repeat=5)
                       if sum(layer_bytes(k) for k in ks) <= budget)
            got = sum(costs[l][str(k)] for l, k in plan.items())
            self.assertAlmostEqual(got, best[0], places=12, msg=(plan, best))

    def test_items_match_solve(self):
        rng = random.Random(2)
        size = lambda k: layer_bytes(k, n_exp=1)
        for _ in range(20):
            costs = {(0, e): {k: rng.uniform(0.2, 2.0) * 4.0 ** -k for k in (2, 3, 4, 5, 6)} for e in range(40)}
            budget = int(rng.uniform(40 * size(2), 40 * size(6)))
            a = solve_items(costs, budget, size)
            b = solve({str(i[1]): {str(k): v for k, v in c.items()} for i, c in costs.items()}, budget, size)
            self.assertLessEqual(sum(size(k) for k in a.values()), budget)
            self.assertAlmostEqual(sum(costs[i][k] for i, k in a.items()),
                                   sum(costs[(0, int(e))][k] for e, k in b.items()), places=12)

    def test_floor(self):
        costs = {"0": {"2": 1.0, "3": 0.99, "4": 0.98}, "1": {"2": 9.0, "3": 0.1, "4": 0.01}}
        budget = layer_bytes(2) + layer_bytes(4)
        self.assertEqual(solve(costs, budget), {"0": 2, "1": 4})
        self.assertEqual(solve(costs, budget, min_k=3), {"0": 3, "1": 3})
        with self.assertRaises(SystemExit):
            solve(costs, budget, min_k=5)

    def test_budget_too_small(self):
        with self.assertRaises(SystemExit):
            solve({"0": {"2": 1.0, "3": 0.5}}, layer_bytes(2) - 1)


if __name__ == "__main__":
    unittest.main()
