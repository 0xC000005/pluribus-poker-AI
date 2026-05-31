import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_solver_backend_parity import summarize_backend_parity_records  # noqa: E402


def test_summarize_backend_parity_requires_decision_match_and_speedup():
    records = [
        {
            "passed": True,
            "strategy_l1": 0.00001,
            "strategy_kl": 0.0,
            "reference_action": 8,
            "candidate_action": 8,
            "reference_latency_ms": 100.0,
            "candidate_latency_ms": 50.0,
            "reference_illegal_mass": 0.0,
            "candidate_illegal_mass": 0.0,
        },
        {
            "passed": True,
            "strategy_l1": 0.00002,
            "strategy_kl": 0.0,
            "reference_action": 1,
            "candidate_action": 1,
            "reference_latency_ms": 200.0,
            "candidate_latency_ms": 100.0,
            "reference_illegal_mass": 0.0,
            "candidate_illegal_mass": 0.0,
        },
    ]

    summary = summarize_backend_parity_records(records, min_evaluated=2)

    assert summary["passed"] is True
    assert summary["action_agreement"] == 1.0
    assert summary["latency_ratio_candidate_to_reference"] == 0.5


def test_summarize_backend_parity_rejects_slow_candidate_even_with_parity():
    records = [
        {
            "passed": True,
            "strategy_l1": 0.0,
            "strategy_kl": 0.0,
            "reference_action": 8,
            "candidate_action": 8,
            "reference_latency_ms": 50.0,
            "candidate_latency_ms": 75.0,
            "reference_illegal_mass": 0.0,
            "candidate_illegal_mass": 0.0,
        }
    ]

    summary = summarize_backend_parity_records(records, min_evaluated=1)

    assert summary["action_agreement"] == 1.0
    assert summary["latency_ratio_candidate_to_reference"] == 1.5
    assert summary["passed"] is False
