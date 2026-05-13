import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fit_joint_pbs_cut_impact_predictor import fit_cut_impact_predictor_from_rows  # noqa: E402


def _row(label: str, shape: str, drift: float, *, pot: int = 1000) -> dict:
    return {
        "label": label,
        "action_l1_drift": drift,
        "action_shape": shape,
        "actor_to_act": 0,
        "bet_count": 5,
        "client_pos": 0,
        "cut_pos": 0,
        "hero_first": True,
        "hero_stack": 10000,
        "legal_action_count": 3,
        "pot": pot,
        "root_action_shape": "root/" + shape,
        "root_bet_count": 4,
        "villain_stack": 10000,
    }


def test_cut_impact_predictor_selects_lower_predicted_drift_rows():
    train = [
        _row("train-safe-1", "safe", 0.05, pot=1000),
        _row("train-safe-2", "safe", 0.06, pot=1100),
        _row("train-risky-1", "risky", 1.0, pot=9000),
        _row("train-risky-2", "risky", 1.1, pot=9100),
    ]
    holdout = [
        _row("holdout-safe", "safe", 0.07, pot=1200),
        _row("holdout-risky", "risky", 1.2, pot=9200),
    ]

    metrics = fit_cut_impact_predictor_from_rows(train, holdout)

    assert metrics["passed"] is True
    assert metrics["abstention"]["selected_count"] == 1
    assert metrics["abstention"]["rejected_count"] == 1
    assert (
        metrics["abstention"]["selected_mean_action_l1"]
        < metrics["abstention"]["rejected_mean_action_l1"]
    )
