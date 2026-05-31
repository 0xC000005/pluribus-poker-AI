import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fast_cfr import (
    _regret_matching_strategy,
    clear_torch_matrix_tensor_cache,
    solve_cfr,
    solve_cfr_levelsync,
    solve_cfr_levelsync_torch,
    solve_cfr_levelsync_torch_batched_same_topology,
    solve_cfr_levelsync_torch_ragged_terminals,
    torch_matrix_tensor_cache_info,
)
from solver import _CARD_TO_EVAL
from solver import _EVALUATOR
from solver import _evaluate_seven_eval_cards_with_six_base
from solver import _evaluate_six_eval_cards
from solver import _evaluate_seven_eval_cards
from solver import _rank_outcome_matrices
from solver import StreetSolver
from solver import clear_terminal_matrix_cache
from solver import resolve_solver_backend
from solver import solve_street
from solver import terminal_matrix_cache_info


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


def test_incremental_seven_card_evaluator_matches_full_evaluator():
    samples = [
        (0, 5, 10, 15, 20, 25, 30),
        (3, 7, 11, 19, 27, 35, 43),
        (12, 13, 14, 28, 32, 40, 51),
        (1, 9, 17, 21, 29, 37, 45),
    ]

    for cards in samples:
        eval_cards = [int(_CARD_TO_EVAL[card]) for card in cards]
        base_rank = _evaluate_six_eval_cards(*eval_cards[:6])
        incremental_rank = _evaluate_seven_eval_cards_with_six_base(
            *eval_cards,
            base_rank,
        )
        full_rank = _evaluate_seven_eval_cards(*eval_cards)
        assert incremental_rank == full_rank


def test_rank_outcome_matrices_preserve_win_loss_tie_semantics():
    ranks = np.array([1, 3, 3], dtype=np.int32)
    valid = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    win_m, lose_m, tie_m = _rank_outcome_matrices(ranks, valid)

    assert win_m.dtype == np.float32
    assert lose_m.dtype == np.float32
    assert tie_m.dtype == np.float32
    assert win_m[0, 1] == 1.0
    assert lose_m[1, 0] == 1.0
    assert tie_m[1, 2] == 1.0
    assert tie_m[1, 1] == 0.0


def test_street_solver_reuses_terminal_matrix_cache_for_same_board():
    clear_terminal_matrix_cache()
    board = [0, 1, 2, 3, 4]

    first = StreetSolver(
        board,
        pot=400,
        hero_stack=19800,
        villain_stack=19800,
        hero_first=True,
    )
    after_first = terminal_matrix_cache_info()
    second = StreetSolver(
        board,
        pot=800,
        hero_stack=19600,
        villain_stack=19600,
        hero_first=False,
    )
    after_second = terminal_matrix_cache_info()

    assert after_first["misses"] == 1
    assert after_second["hits"] == 1
    assert first.valid is second.valid
    assert first.win_m is second.win_m
    assert first.lose_m is second.lose_m
    assert first.tie_m is second.tie_m
    np.testing.assert_allclose(first.win_m, second.win_m)


def test_terminal_matrix_cache_keeps_active_hand_subset_isolated():
    clear_terminal_matrix_cache()
    board = [0, 1, 2, 3, 4]

    full = StreetSolver(
        board,
        pot=400,
        hero_stack=19800,
        villain_stack=19800,
        hero_first=True,
    )
    subset = StreetSolver(
        board,
        pot=400,
        hero_stack=19800,
        villain_stack=19800,
        hero_first=True,
        active_indices=[0, 1, 2],
    )
    info = terminal_matrix_cache_info()

    assert info["misses"] == 2
    assert info["hits"] == 0
    assert full.win_m is not subset.win_m
    assert full.win_m.shape != subset.win_m.shape


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


def test_batched_torch_levelsync_matches_serial_for_same_topology_tree():
    board = [0, 1, 2, 3, 4]
    first = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)
    second = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)
    hero_range = np.linspace(1.0, 2.0, first.n, dtype=np.float32)
    villain_range = np.linspace(2.0, 1.0, first.n, dtype=np.float32)

    serial_first = solve_cfr_levelsync_torch(
        first._tree,
        first.n,
        first.win_m,
        first.lose_m,
        first.tie_m,
        first.valid,
        first.pot_start,
        first.hero_stack_start,
        first.villain_stack_start,
        n_iterations=2,
        hero_range=hero_range,
        villain_range=villain_range,
        device="cpu",
    )
    serial_second = solve_cfr_levelsync_torch(
        second._tree,
        second.n,
        second.win_m,
        second.lose_m,
        second.tie_m,
        second.valid,
        second.pot_start,
        second.hero_stack_start,
        second.villain_stack_start,
        n_iterations=2,
        hero_range=villain_range,
        villain_range=hero_range,
        device="cpu",
    )

    batched_regret, batched_strategy = solve_cfr_levelsync_torch_batched_same_topology(
        [first._tree, second._tree],
        first.n,
        [first.win_m, second.win_m],
        [first.lose_m, second.lose_m],
        [first.tie_m, second.tie_m],
        [first.valid, second.valid],
        [first.pot_start, second.pot_start],
        [first.hero_stack_start, second.hero_stack_start],
        [first.villain_stack_start, second.villain_stack_start],
        n_iterations=2,
        hero_ranges=[hero_range, villain_range],
        villain_ranges=[villain_range, hero_range],
        device="cpu",
    )

    np.testing.assert_allclose(batched_regret[0], serial_first[0], rtol=1e-4, atol=10.0)
    np.testing.assert_allclose(batched_strategy[0], serial_first[1], rtol=1e-5, atol=2e-4)
    np.testing.assert_allclose(batched_regret[1], serial_second[0], rtol=1e-4, atol=10.0)
    np.testing.assert_allclose(batched_strategy[1], serial_second[1], rtol=1e-5, atol=2e-4)

    loop_regret, loop_strategy = solve_cfr_levelsync_torch_batched_same_topology(
        [first._tree, second._tree],
        first.n,
        [first.win_m, second.win_m],
        [first.lose_m, second.lose_m],
        [first.tie_m, second.tie_m],
        [first.valid, second.valid],
        [first.pot_start, second.pot_start],
        [first.hero_stack_start, second.hero_stack_start],
        [first.villain_stack_start, second.villain_stack_start],
        n_iterations=2,
        hero_ranges=[hero_range, villain_range],
        villain_ranges=[villain_range, hero_range],
        terminal_eval_mode="loop",
        device="cpu",
    )

    np.testing.assert_allclose(loop_regret[0], serial_first[0], rtol=1e-4, atol=10.0)
    np.testing.assert_allclose(loop_strategy[0], serial_first[1], rtol=1e-5, atol=2e-4)
    np.testing.assert_allclose(loop_regret[1], serial_second[0], rtol=1e-4, atol=10.0)
    np.testing.assert_allclose(loop_strategy[1], serial_second[1], rtol=1e-5, atol=2e-4)

    for terminal_eval_mode in ("loop_showdown", "loop_folds"):
        hybrid_regret, hybrid_strategy = solve_cfr_levelsync_torch_batched_same_topology(
            [first._tree, second._tree],
            first.n,
            [first.win_m, second.win_m],
            [first.lose_m, second.lose_m],
            [first.tie_m, second.tie_m],
            [first.valid, second.valid],
            [first.pot_start, second.pot_start],
            [first.hero_stack_start, second.hero_stack_start],
            [first.villain_stack_start, second.villain_stack_start],
            n_iterations=2,
            hero_ranges=[hero_range, villain_range],
            villain_ranges=[villain_range, hero_range],
            terminal_eval_mode=terminal_eval_mode,
            device="cpu",
        )

        np.testing.assert_allclose(hybrid_regret[0], serial_first[0], rtol=1e-4, atol=25.0)
        np.testing.assert_allclose(hybrid_strategy[0], serial_first[1], rtol=1e-5, atol=2e-4)
        np.testing.assert_allclose(hybrid_regret[1], serial_second[0], rtol=1e-4, atol=25.0)
        np.testing.assert_allclose(hybrid_strategy[1], serial_second[1], rtol=1e-5, atol=2e-4)


def test_batched_torch_levelsync_rejects_topology_mismatch():
    first = StreetSolver([0, 1, 2, 3, 4], pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)
    second = StreetSolver([0, 1, 2, 3, 5], pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)
    second._tree["children"] = second._tree["children"].copy()
    root_action = int(np.nonzero(second._tree["children"][0] >= 0)[0][0])
    second._tree["children"][0, root_action] = -1

    with pytest.raises(ValueError, match="same topology"):
        solve_cfr_levelsync_torch_batched_same_topology(
            [first._tree, second._tree],
            first.n,
            [first.win_m, second.win_m],
            [first.lose_m, second.lose_m],
            [first.tie_m, second.tie_m],
            [first.valid, second.valid],
            [first.pot_start, second.pot_start],
            [first.hero_stack_start, second.hero_stack_start],
            [first.villain_stack_start, second.villain_stack_start],
            n_iterations=1,
            device="cpu",
        )


def test_ragged_terminal_levelsync_matches_serial_for_mismatched_topology():
    first = StreetSolver([0, 1, 2, 3, 4], pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)
    second = StreetSolver([0, 1, 2, 3, 5], pot=800, hero_stack=19600, villain_stack=19600, hero_first=False)
    assert first._tree["n_nodes"] != second._tree["n_nodes"]
    assert first.n == second.n

    hero_range = np.linspace(1.0, 2.0, first.n, dtype=np.float32)
    villain_range = np.linspace(2.0, 1.0, first.n, dtype=np.float32)
    serial_first = solve_cfr(
        first._tree,
        first.n,
        first.win_m,
        first.lose_m,
        first.tie_m,
        first.valid,
        first.pot_start,
        first.hero_stack_start,
        first.villain_stack_start,
        n_iterations=2,
        hero_range=hero_range,
        villain_range=villain_range,
    )
    serial_second = solve_cfr(
        second._tree,
        second.n,
        second.win_m,
        second.lose_m,
        second.tie_m,
        second.valid,
        second.pot_start,
        second.hero_stack_start,
        second.villain_stack_start,
        n_iterations=2,
        hero_range=villain_range,
        villain_range=hero_range,
    )

    ragged = solve_cfr_levelsync_torch_ragged_terminals(
        [first._tree, second._tree],
        first.n,
        [first.win_m, second.win_m],
        [first.lose_m, second.lose_m],
        [first.tie_m, second.tie_m],
        [first.valid, second.valid],
        [first.pot_start, second.pot_start],
        [first.hero_stack_start, second.hero_stack_start],
        [first.villain_stack_start, second.villain_stack_start],
        n_iterations=2,
        hero_ranges=[hero_range, villain_range],
        villain_ranges=[villain_range, hero_range],
        device="cpu",
    )

    np.testing.assert_allclose(ragged[0][0], serial_first[0], rtol=1e-4, atol=25.0)
    np.testing.assert_allclose(ragged[0][1], serial_first[1], rtol=1e-5, atol=2e-4)
    np.testing.assert_allclose(ragged[1][0], serial_second[0], rtol=1e-4, atol=25.0)
    np.testing.assert_allclose(ragged[1][1], serial_second[1], rtol=1e-5, atol=2e-4)


def test_torch_levelsync_reuses_static_tensor_cache_between_solves():
    board = [0, 1, 2, 3, 4]
    solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)

    solve_cfr_levelsync_torch(
        solver._tree,
        solver.n,
        solver.win_m,
        solver.lose_m,
        solver.tie_m,
        solver.valid,
        solver.pot_start,
        solver.hero_stack_start,
        solver.villain_stack_start,
        n_iterations=1,
        device="cpu",
    )

    cache = solver._tree.get("_torch_levelsync_cache")
    assert cache
    first_entry = next(iter(cache.values()))
    first_group_parents = next(
        group["parents"] for group in first_entry["groups"] if group is not None
    )
    first_win_tensor = first_entry["win"]

    solve_cfr_levelsync_torch(
        solver._tree,
        solver.n,
        solver.win_m,
        solver.lose_m,
        solver.tie_m,
        solver.valid,
        solver.pot_start,
        solver.hero_stack_start,
        solver.villain_stack_start,
        n_iterations=1,
        device="cpu",
    )

    second_entry = next(iter(solver._tree["_torch_levelsync_cache"].values()))
    second_group_parents = next(
        group["parents"] for group in second_entry["groups"] if group is not None
    )
    assert second_entry["win"] is first_win_tensor
    assert second_group_parents is first_group_parents


def test_torch_levelsync_reuses_workspace_tensors_between_solves():
    board = [0, 1, 2, 3, 4]
    solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)

    solve_cfr_levelsync_torch(
        solver._tree,
        solver.n,
        solver.win_m,
        solver.lose_m,
        solver.tie_m,
        solver.valid,
        solver.pot_start,
        solver.hero_stack_start,
        solver.villain_stack_start,
        n_iterations=1,
        device="cpu",
    )
    entry = next(iter(solver._tree["_torch_levelsync_cache"].values()))
    first_workspace = entry["_workspace"]
    first_hr_at = first_workspace["hr_at"]
    first_hvals = first_workspace["hvals"]

    solve_cfr_levelsync_torch(
        solver._tree,
        solver.n,
        solver.win_m,
        solver.lose_m,
        solver.tie_m,
        solver.valid,
        solver.pot_start,
        solver.hero_stack_start,
        solver.villain_stack_start,
        n_iterations=1,
        device="cpu",
    )
    second_workspace = entry["_workspace"]

    assert second_workspace["hr_at"] is first_hr_at
    assert second_workspace["hvals"] is first_hvals


def test_torch_levelsync_reuses_terminal_tensor_cache_between_same_board_solvers():
    clear_terminal_matrix_cache()
    clear_torch_matrix_tensor_cache()
    board = [0, 1, 2, 3, 4]
    first_solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)
    second_solver = StreetSolver(board, pot=800, hero_stack=19600, villain_stack=19600, hero_first=False)

    solve_cfr_levelsync_torch(
        first_solver._tree,
        first_solver.n,
        first_solver.win_m,
        first_solver.lose_m,
        first_solver.tie_m,
        first_solver.valid,
        first_solver.pot_start,
        first_solver.hero_stack_start,
        first_solver.villain_stack_start,
        n_iterations=1,
        device="cpu",
    )
    after_first = torch_matrix_tensor_cache_info()
    first_entry = next(iter(first_solver._tree["_torch_levelsync_cache"].values()))

    solve_cfr_levelsync_torch(
        second_solver._tree,
        second_solver.n,
        second_solver.win_m,
        second_solver.lose_m,
        second_solver.tie_m,
        second_solver.valid,
        second_solver.pot_start,
        second_solver.hero_stack_start,
        second_solver.villain_stack_start,
        n_iterations=1,
        device="cpu",
    )
    after_second = torch_matrix_tensor_cache_info()
    second_entry = next(iter(second_solver._tree["_torch_levelsync_cache"].values()))

    assert after_first["misses"] == 1
    assert after_second["hits"] == 1
    assert second_entry["win"] is first_entry["win"]
    assert second_entry["valid_T"] is first_entry["valid_T"]


def test_street_solver_accepts_torch_levelsync_cpu_backend():
    board = [0, 1, 2, 3, 4]
    solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)

    solver.solve(n_iterations=1, backend="torch-levelsync-cpu")

    strategy = solver.get_strategy((30, 31))
    assert strategy
    assert np.isclose(sum(strategy.values()), 1.0)


def test_street_solver_accepts_segmented_cpu_backend_and_returns_strategy():
    board = [0, 1, 2, 3, 4]
    solver = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)

    solver.solve(n_iterations=1, backend="segmented-cpu")

    strategy = solver.get_strategy((30, 31))
    assert strategy
    assert set(strategy).issubset(set(range(9)))
    assert np.isclose(sum(strategy.values()), 1.0)
    assert solver._regret_sum.shape == (solver._tree["n_nodes"], solver._tree["n_actions"], solver.n)
    assert solver._strategy_sum.shape == solver._regret_sum.shape


def test_segmented_cpu_backend_matches_cpu_root_strategy_smoke():
    board = [0, 1, 2, 3, 4]
    reference = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)
    segmented = StreetSolver(board, pot=400, hero_stack=19800, villain_stack=19800, hero_first=True)
    hand = (30, 31)

    reference.solve(n_iterations=2, backend="cpu")
    segmented.solve(n_iterations=2, backend="segmented-cpu")

    ref_strategy = reference.get_strategy(hand)
    segmented_strategy = segmented.get_strategy(hand)
    assert ref_strategy.keys() == segmented_strategy.keys()
    for action in ref_strategy:
        assert np.isclose(segmented_strategy[action], ref_strategy[action], atol=1e-4)


def test_auto_solver_backend_prefers_torch_levelsync_cuda_when_available(monkeypatch):
    monkeypatch.setattr("solver.torch.cuda.is_available", lambda: True)

    backend, device = resolve_solver_backend("auto")

    assert backend == "torch-levelsync"
    assert device == "cuda"


def test_auto_solver_backend_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr("solver.torch.cuda.is_available", lambda: False)

    backend, device = resolve_solver_backend("auto")

    assert backend == "cpu"
    assert device is None


def test_explicit_segmented_backends_resolve_without_changing_auto(monkeypatch):
    monkeypatch.setattr("solver.torch.cuda.is_available", lambda: True)

    assert resolve_solver_backend("segmented-cpu") == ("segmented", "cpu")
    assert resolve_solver_backend("segmented-cuda") == ("segmented", "cuda")
    assert resolve_solver_backend("auto") == ("torch-levelsync", "cuda")


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
