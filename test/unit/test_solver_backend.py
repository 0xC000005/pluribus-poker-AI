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


def test_street_solver_explicit_cfr_plus_matches_default_update():
    board = [0, 5, 10, 15, 20]
    default_solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)
    explicit_solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)

    default_solver.solve(n_iterations=2)
    explicit_solver.solve(n_iterations=2, solver_update="cfr_plus")

    np.testing.assert_allclose(explicit_solver._regret_sum, default_solver._regret_sum, atol=1e-5)
    np.testing.assert_allclose(explicit_solver._strategy_sum, default_solver._strategy_sum, atol=1e-5)


def test_street_solver_accepts_dcfr_plus_cpu_update():
    board = [0, 5, 10, 15, 20]
    solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)

    solver.solve(n_iterations=2, solver_update="dcfr_plus")
    strategy = solver.get_strategy((30, 31))

    assert strategy
    assert set(strategy).issubset(set(range(9)))
    assert np.isclose(sum(strategy.values()), 1.0)
    assert np.isfinite(solver._regret_sum).all()
    assert np.isfinite(solver._strategy_sum).all()


def test_street_solver_accepts_pdcfr_plus_cpu_update():
    board = [0, 5, 10, 15, 20]
    solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)

    solver.solve(n_iterations=2, solver_update="pdcfr_plus")
    strategy = solver.get_strategy((30, 31))

    assert strategy
    assert set(strategy).issubset(set(range(9)))
    assert np.isclose(sum(strategy.values()), 1.0)
    assert np.isfinite(solver._regret_sum).all()
    assert np.isfinite(solver._strategy_sum).all()


def test_street_solver_rejects_unknown_backend():
    board = [0, 5, 10, 15, 20]
    solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)

    try:
        solver.solve(n_iterations=1, backend="bogus")
    except ValueError as exc:
        assert "Unknown solver backend" in str(exc)
    else:
        raise AssertionError("Expected unknown backend to raise ValueError")


def test_street_solver_rejects_unknown_solver_update():
    board = [0, 5, 10, 15, 20]
    solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)

    try:
        solver.solve(n_iterations=1, solver_update="bogus")
    except ValueError as exc:
        assert "Unknown solver_update" in str(exc)
    else:
        raise AssertionError("Expected unknown solver update to raise ValueError")


def test_torch_backend_rejects_non_default_solver_updates():
    board = [0, 5, 10, 15, 20]

    for update in ("dcfr_plus", "pdcfr_plus"):
        solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)
        try:
            solver.solve(n_iterations=1, backend="torch", device="cpu", solver_update=update)
        except ValueError as exc:
            assert "only supported by the CPU CFR backend" in str(exc)
        else:
            raise AssertionError(f"Expected torch {update} update to raise ValueError")


def test_trace_callback_receives_per_action_counterfactual_values():
    board = [0, 5, 10, 15, 20]
    solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)
    records = []

    def capture_trace(**kwargs):
        records.append(kwargs)

    solver.solve(n_iterations=1, trace_node_indices=[0], trace_node_fn=capture_trace)

    assert records
    record = records[0]
    assert record["hero_action_values"].shape == (1, solver._tree["n_actions"], solver.n)
    assert record["villain_action_values"].shape == (1, solver._tree["n_actions"], solver.n)
    for action in solver._tree["decision_actions"][0]:
        assert np.isfinite(record["hero_action_values"][0, action]).all()
        assert np.isfinite(record["villain_action_values"][0, action]).all()


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
