import json

from poker_ai.research.h2h_league_summary import summarize_h2h_league


def _write_record(path, *, baseline: str, mean: float, lower95: float):
    path.write_text(
        json.dumps(
            {
                "algorithm": "tianshou_rainbow_vs_tianshou_rainbow_h2h",
                "candidate_checkpoint": "candidate.pt",
                "baseline_checkpoint": baseline,
                "mean_candidate_payoff": mean,
                "lower95_candidate_payoff": lower95,
                "n_games": 1000,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_h2h_league_summary_passes_when_all_lower_bounds_positive(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_record(first, baseline="a.pt", mean=0.01, lower95=0.002)
    _write_record(second, baseline="b.pt", mean=0.03, lower95=0.005)

    summary = summarize_h2h_league([str(first), str(second)], min_lower95=0.0)

    assert summary["algorithm"] == "poker_h2h_league_summary"
    assert summary["passed"] is True
    assert summary["candidate_checkpoint"] == "candidate.pt"
    assert summary["worst_lower95_candidate_payoff"] == 0.002
    assert summary["n_records"] == 2


def test_h2h_league_summary_fails_on_any_nonpositive_lower_bound(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_record(first, baseline="a.pt", mean=0.01, lower95=0.002)
    _write_record(second, baseline="b.pt", mean=0.01, lower95=-0.001)

    summary = summarize_h2h_league([str(first), str(second)], min_lower95=0.0)

    assert summary["passed"] is False
    assert summary["worst_lower95_candidate_payoff"] == -0.001
    assert summary["failed_baselines"] == ["b.pt"]
