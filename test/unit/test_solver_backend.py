import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fast_cfr import _regret_matching_strategy, solve_cfr_levelsync, solve_cfr_levelsync_torch
from solver import _CARD_TO_EVAL
from solver import _EVALUATOR
from solver import _evaluate_seven_eval_cards
from solver import StreetSolver
from solver import solve_street


def test_regret_matching_strategy_uses_positive_regrets_or_uniform_fallback():
    regret_rows = np.array(
        [
            [-2.0, 2.0, 0.0, 0.5],
            [0.0, 1.0, -4.0, 0.5],
            [-1.0, -3.0, 3.0, -2.0],
        ],
        dtype=np.float32,
    )

    strategy = _regret_matching_strategy(regret_rows)

    expected = np.array(
        [
            [1.0 / 3.0, 2.0 / 3.0, 0.0, 0.5],
            [1.0 / 3.0, 1.0 / 3.0, 0.0, 0.5],
            [1.0 / 3.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(strategy, expected, atol=1e-7)
    assert strategy.dtype == np.float32


def test_fast_seven_card_evaluator_matches_reference_on_sampled_showdowns():
    samples = [
        (0, 5, 10, 15, 20, 25, 30),
        (3, 7, 11, 19, 27, 35, 43),
        (12, 13, 14, 28, 32, 40, 51),
        (1, 9, 17, 21, 29, 37, 45),
        (0, 4, 8, 12, 16, 20, 24),
        (0, 1, 2, 3, 4, 8, 12),
    ]

    for cards in samples:
        eval_cards = [int(_CARD_TO_EVAL[card]) for card in cards]
        fast_rank = _evaluate_seven_eval_cards(*eval_cards)
        reference_rank = _EVALUATOR.evaluate(eval_cards[:2], eval_cards[2:])
        assert fast_rank == reference_rank


def test_levelsync_cfr_matches_reference_cpu_cfrplus_on_river_smoke():
    board = [0, 1, 2, 3, 4]
    reference = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)
    levelsync = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)

    reference.solve(n_iterations=3, backend="cpu")
    levelsync._regret_sum, levelsync._strategy_sum = solve_cfr_levelsync(
        levelsync._tree,
        levelsync.n,
        levelsync.win_m,
        levelsync.lose_m,
        levelsync.tie_m,
        levelsync.valid,
        levelsync.pot_start,
        levelsync.hero_stack_start,
        levelsync.villain_stack_start,
        n_iterations=3,
    )

    np.testing.assert_allclose(
        levelsync._regret_sum,
        reference._regret_sum,
        rtol=1e-4,
        atol=25.0,
    )
    np.testing.assert_allclose(levelsync._strategy_sum, reference._strategy_sum, atol=1e-5)


def test_street_solver_accepts_cpu_levelsync_backend():
    board = [0, 1, 2, 3, 4]
    solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)

    solver.solve(n_iterations=1, backend="cpu-levelsync")

    strategy = solver.get_strategy((30, 31))
    assert strategy
    assert np.isclose(sum(strategy.values()), 1.0)


def test_torch_levelsync_cfr_matches_reference_cpu_cfrplus_on_river_smoke():
    board = [0, 1, 2, 3, 4]
    reference = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)
    levelsync = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)

    reference.solve(n_iterations=2, backend="cpu")
    levelsync._regret_sum, levelsync._strategy_sum = solve_cfr_levelsync_torch(
        levelsync._tree,
        levelsync.n,
        levelsync.win_m,
        levelsync.lose_m,
        levelsync.tie_m,
        levelsync.valid,
        levelsync.pot_start,
        levelsync.hero_stack_start,
        levelsync.villain_stack_start,
        n_iterations=2,
        device="cpu",
    )

    np.testing.assert_allclose(
        levelsync._regret_sum,
        reference._regret_sum,
        rtol=1e-4,
        atol=25.0,
    )
    np.testing.assert_allclose(levelsync._strategy_sum, reference._strategy_sum, atol=1e-5)


def test_street_solver_accepts_torch_levelsync_cpu_backend():
    board = [0, 1, 2, 3, 4]
    solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)

    solver.solve(n_iterations=1, backend="torch-levelsync-cpu")

    strategy = solver.get_strategy((30, 31))
    assert strategy
    assert np.isclose(sum(strategy.values()), 1.0)


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
