import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from solver import StreetSolver
from solver import solve_street


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


def test_solve_street_prunes_low_probability_ranges_and_keeps_hero_hand():
    board = [0, 5, 10, 15, 20]
    probe = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)
    hero_hand = (30, 31)
    hero_idx = probe.hand_to_idx[hero_hand]
    villain_idx = probe.hand_to_idx[(32, 33)]
    hero_range = np.zeros(probe.n, dtype=np.float64)
    villain_range = np.zeros(probe.n, dtype=np.float64)
    hero_range[hero_idx] = 1.0
    villain_range[villain_idx] = 1.0

    action, strategy, solver, node = solve_street(
        list(hero_hand),
        board,
        pot=400,
        hero_stack=19800,
        villain_stack=19800,
        hero_first=True,
        n_iterations=1,
        hero_range=hero_range,
        villain_range=villain_range,
        range_prune_threshold=1e-4,
    )

    assert action in strategy
    assert node is not None
    assert solver.n < probe.n
    assert hero_hand in solver.hand_to_idx
    assert (32, 33) in solver.hand_to_idx
