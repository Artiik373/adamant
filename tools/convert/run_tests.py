"""Run all MS converter and packer unit tests in one command.

    python tools/convert/run_tests.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = [
    "test_ms_pack.py",
    "test_ms_plan_solver.py",
    "test_two_stream.py",
    "test_ms_viterbi_cuda.py",
]


def main() -> int:
    os.environ.setdefault("TRITON_CACHE_DIR", str(Path(tempfile.gettempdir()) / "adamant_triton_cache"))
    sys.path.insert(0, str(HERE))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in TESTS:
        suite.addTests(loader.discover(str(HERE), pattern=name))
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
