import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from solver import StreetSolver


def test_street_solver_accepts_torch_cpu_backend_and_returns_strategy():
    board = [0, 5, 10, 15, 20]
    solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)

    solver.solve(n_iterations=1, backend="torch", device="cpu")
    strategy = solver.get_strategy((30, 31))

    assert strategy
    assert set(strategy).issubset(set(range(9)))
    assert np.isclose(sum(strategy.values()), 1.0)


def test_street_solver_rejects_unknown_backend():
    board = [0, 5, 10, 15, 20]
    solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)

    try:
        solver.solve(n_iterations=1, backend="bogus")
    except ValueError as exc:
        assert "Unknown solver backend" in str(exc)
    else:
        raise AssertionError("Expected unknown backend to raise ValueError")
