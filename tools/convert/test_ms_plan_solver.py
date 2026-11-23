"""ms_plan_solver against a brute force over all plans of a few layers.

    python -m unittest discover -s tools/convert -p "test_ms_plan_solver.py"
"""
import itertools
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ms_plan_solver import layer_bytes, solve  # noqa: E402


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

    def test_budget_too_small(self):
        with self.assertRaises(SystemExit):
            solve({"0": {"2": 1.0, "3": 0.5}}, layer_bytes(2) - 1)


if __name__ == "__main__":
    unittest.main()
