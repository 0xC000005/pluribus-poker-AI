import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_solver_update_gate import summarize_solver_update_records  # noqa: E402


def _record(*, candidate_l1, candidate_kl, candidate_agrees=True):
    return {
        "passed": True,
        "low_l1_to_reference": 0.5,
        "candidate_l1_to_reference": candidate_l1,
        "low_kl_to_reference": 0.25,
        "candidate_kl_to_reference": candidate_kl,
        "low_agrees_with_reference": True,
        "candidate_agrees_with_reference": candidate_agrees,
        "low_allin_selected": False,
        "candidate_allin_selected": False,
        "reference_allin_selected": False,
        "low_allin_prob": 0.1,
        "candidate_allin_prob": 0.05,
        "reference_allin_prob": 0.0,
        "low_illegal_mass": 0.0,
        "candidate_illegal_mass": 0.0,
        "reference_illegal_mass": 0.0,
        "low_latency_ms": 10.0,
        "candidate_latency_ms": 11.0,
        "reference_latency_ms": 30.0,
    }


def test_solver_update_summary_passes_when_candidate_improves_low_solver():
    metrics = summarize_solver_update_records(
        [_record(candidate_l1=0.25, candidate_kl=0.1)],
        candidate_update="dcfr_plus",
    )

    assert metrics["passed"] is True
    assert metrics["candidate_update"] == "dcfr_plus"
    assert metrics["mean_candidate_l1_to_reference"] < metrics["mean_low_l1_to_reference"]


def test_solver_update_summary_fails_when_candidate_worsens_reference_distance():
    metrics = summarize_solver_update_records(
        [_record(candidate_l1=0.75, candidate_kl=0.3, candidate_agrees=False)],
        candidate_update="dcfr_plus",
    )

    assert metrics["passed"] is False
    assert metrics["mean_candidate_l1_to_reference"] > metrics["mean_low_l1_to_reference"]
