import pytest

from poker_ai.research.early_street_calibration import summarize_early_street_rows


def test_early_street_summary_reports_allin_and_reference_drift():
    rows = [
        {
            "street": 0,
            "candidate_strategy": [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
            "reference_strategy": [0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "legal_mask": [0, 1, 1, 0, 0, 0, 0, 0, 1],
        },
        {
            "street": 0,
            "candidate_strategy": [0.0, 0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "reference_strategy": [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "legal_mask": [0, 1, 1, 0, 0, 0, 0, 0, 1],
        },
        {
            "street": 1,
            "candidate_strategy": [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
            "reference_strategy": [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
            "legal_mask": [0, 1, 0, 0, 0, 0, 0, 0, 1],
        },
    ]

    metrics = summarize_early_street_rows(
        rows,
        max_candidate_preflop_allin_rate=0.25,
        max_candidate_flop_allin_rate=0.75,
        max_mean_l1_to_reference=1.0,
    )

    assert metrics["passed"] is False
    assert metrics["n_rows"] == 3
    assert metrics["by_street"]["preflop"]["candidate_top_allin_rate"] == pytest.approx(0.5)
    assert metrics["by_street"]["preflop"]["reference_top_allin_rate"] == pytest.approx(0.0)
    assert metrics["by_street"]["flop"]["candidate_top_allin_rate"] == pytest.approx(1.0)
    assert metrics["by_street"]["preflop"]["mean_l1_to_reference"] == pytest.approx(0.9)
    assert "preflop_allin_rate" in metrics["failed_checks"]
    assert "flop_allin_rate" in metrics["failed_checks"]
