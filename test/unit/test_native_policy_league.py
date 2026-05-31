import json

from poker_ai.research.native_policy_league import (
    load_native_policy_league_records,
    summarize_native_policy_league,
)


def test_summarize_native_policy_league_selects_checkpoint_by_worst_lower95():
    records = [
        {
            "candidate_checkpoint": "a.pt",
            "baseline_checkpoint": "control.pt",
            "mean_candidate_payoff": 0.02,
            "lower95_candidate_payoff": 0.01,
        },
        {
            "candidate_checkpoint": "a.pt",
            "baseline_checkpoint": "incumbent.pt",
            "mean_candidate_payoff": 0.01,
            "lower95_candidate_payoff": -0.001,
        },
        {
            "candidate_checkpoint": "b.pt",
            "baseline_checkpoint": "control.pt",
            "mean_candidate_payoff": 0.015,
            "lower95_candidate_payoff": 0.005,
        },
        {
            "candidate_checkpoint": "b.pt",
            "baseline_checkpoint": "incumbent.pt",
            "mean_candidate_payoff": 0.012,
            "lower95_candidate_payoff": 0.004,
        },
    ]

    summary = summarize_native_policy_league(records)

    assert summary["best_checkpoint"] == "b.pt"
    assert summary["best_worst_lower95"] == 0.004
    assert summary["candidate_summaries"]["a.pt"]["passes_all_positive_lower95"] is False
    assert summary["candidate_summaries"]["b.pt"]["passes_all_positive_lower95"] is True


def test_summarize_native_policy_league_marks_empty_records_invalid():
    summary = summarize_native_policy_league([])

    assert summary["passed"] is False
    assert summary["best_checkpoint"] is None
    assert "no_pairwise_records" in summary["errors"]


def test_load_native_policy_league_records_reads_existing_pairwise_json(tmp_path):
    record_path = tmp_path / "pairwise.json"
    record_path.write_text(
        json.dumps(
            {
                "algorithm": "native_policy_h2h",
                "candidate_checkpoint": "candidate.pt",
                "baseline_checkpoint": "baseline.pt",
                "mean_candidate_payoff": 0.2,
                "lower95_candidate_payoff": 0.1,
            }
        ),
        encoding="utf-8",
    )

    records = load_native_policy_league_records([str(record_path)])

    assert records == [
        {
            "algorithm": "native_policy_h2h",
            "candidate_checkpoint": "candidate.pt",
            "baseline_checkpoint": "baseline.pt",
            "mean_candidate_payoff": 0.2,
            "lower95_candidate_payoff": 0.1,
        }
    ]
