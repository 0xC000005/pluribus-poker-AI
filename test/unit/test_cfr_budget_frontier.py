import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_cfr_budget_frontier import summarize_budget_frontier_records  # noqa: E402


def _record():
    return {
        "passed": True,
        "reference_action": 1,
        "reference_allin_prob": 0.05,
        "reference_allin_selected": False,
        "reference_illegal_mass": 0.0,
        "budgets": {
            "5": {
                "l1_to_reference": 0.52,
                "kl_to_reference": 0.26,
                "action": 0,
                "allin_prob": 0.12,
                "allin_selected": False,
                "illegal_mass": 0.0,
                "latency_ms": 180.0,
            },
            "10": {
                "l1_to_reference": 0.36,
                "kl_to_reference": 0.13,
                "action": 1,
                "allin_prob": 0.08,
                "allin_selected": False,
                "illegal_mass": 0.0,
                "latency_ms": 350.0,
            },
        },
    }


def test_budget_frontier_summary_compares_quality_and_latency_by_budget():
    metrics = summarize_budget_frontier_records([_record()], budgets=[5, 10])

    assert metrics["passed"] is True
    assert metrics["n_evaluated"] == 1
    assert metrics["best_l1_budget"] == 10
    assert metrics["budgets"]["5"]["mean_l1_to_reference"] == 0.52
    assert metrics["budgets"]["10"]["action_agreement"] == 1.0
    assert metrics["budgets"]["10"]["latency_ratio_to_min_budget"] == 1.94444444


def test_budget_frontier_summary_fails_when_any_budget_has_illegal_mass():
    record = _record()
    record["budgets"]["10"]["illegal_mass"] = 0.25

    metrics = summarize_budget_frontier_records([record], budgets=[5, 10])

    assert metrics["passed"] is False
    assert metrics["max_illegal_mass"] == 0.25


def test_budget_frontier_summary_fails_when_reference_has_illegal_mass():
    record = _record()
    record["reference_illegal_mass"] = 0.5

    metrics = summarize_budget_frontier_records([record], budgets=[5, 10])

    assert metrics["passed"] is False
    assert metrics["max_illegal_mass"] == 0.5
