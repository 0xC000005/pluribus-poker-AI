import json

import numpy as np

from poker_ai.research.empirical_game import (
    build_empirical_payoff_matrix,
    solve_zero_sum_meta_strategy,
)


def _write_record(path, candidate, baseline, payoff, *, n_games=100):
    payload = {
        "candidate_checkpoint": candidate,
        "baseline_checkpoint": baseline,
        "mean_candidate_payoff": payoff,
        "lower95_candidate_payoff": payoff - 0.01,
        "n_games": n_games,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_empirical_payoff_matrix_uses_duplicate_swapped_antisymmetry(tmp_path):
    ab = tmp_path / "a_vs_b.json"
    bc = tmp_path / "b_vs_c.json"
    ac = tmp_path / "a_vs_c.json"
    _write_record(ab, "a.pt", "b.pt", 0.2, n_games=100)
    _write_record(bc, "b.pt", "c.pt", 0.1, n_games=200)
    _write_record(ac, "a.pt", "c.pt", -0.05, n_games=300)

    summary = build_empirical_payoff_matrix([str(ab), str(bc), str(ac)])

    assert summary["policies"] == ["a.pt", "b.pt", "c.pt"]
    matrix = np.asarray(summary["payoff_matrix"], dtype=float)
    assert matrix.tolist() == [
        [0.0, 0.2, -0.05],
        [-0.2, 0.0, 0.1],
        [0.05, -0.1, 0.0],
    ]
    assert summary["off_diagonal_coverage"] == 1.0
    assert summary["recommendation"] == "solve_meta_strategy"


def test_build_empirical_payoff_matrix_reports_missing_pairs(tmp_path):
    ab = tmp_path / "a_vs_b.json"
    _write_record(ab, "a.pt", "b.pt", 0.2)

    summary = build_empirical_payoff_matrix([str(ab)], policies=["a.pt", "b.pt", "c.pt"])

    assert summary["off_diagonal_coverage"] == 1 / 3
    assert summary["missing_pairs"] == [["a.pt", "c.pt"], ["b.pt", "c.pt"]]
    assert summary["recommendation"] == "fill_pairwise_payoff_matrix"


def test_solve_zero_sum_meta_strategy_uses_supplied_nashpy_module():
    class FakeGame:
        def __init__(self, matrix):
            self.matrix = matrix

        def support_enumeration(self):
            yield ([0.25, 0.75], [0.4, 0.6])

    class FakeNashpy:
        Game = FakeGame

    solution = solve_zero_sum_meta_strategy(
        [[0.0, 1.0], [-1.0, 0.0]],
        nashpy_module=FakeNashpy,
    )

    assert solution["solved"] is True
    assert solution["row_strategy"] == [0.25, 0.75]
    assert solution["column_strategy"] == [0.4, 0.6]
    assert solution["solver"] == "nashpy.support_enumeration"


def test_analyze_empirical_game_cli_writes_summary(tmp_path):
    from scripts import analyze_poker_empirical_game as cli

    ab = tmp_path / "a_vs_b.json"
    _write_record(ab, "a.pt", "b.pt", 0.2)
    output = tmp_path / "matrix.json"

    exit_code = cli.main([str(ab), "--output-json", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["algorithm"] == "native_empirical_payoff_matrix"
    assert payload["recommendation"] == "solve_meta_strategy"
    assert payload["meta_strategy"]["status"] == "not_requested"
