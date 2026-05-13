import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_joint_pbs_error_predictor import fit_error_predictor_from_rows


def test_error_predictor_fits_synthetic_metadata_signal():
    train_rows = [
        {"label": "a", "mae": 0.05, "legal_action_count": 8, "action_shape": "x"},
        {"label": "b", "mae": 0.07, "legal_action_count": 7, "action_shape": "x"},
        {"label": "c", "mae": 0.40, "legal_action_count": 2, "action_shape": "y"},
        {"label": "d", "mae": 0.45, "legal_action_count": 2, "action_shape": "y"},
    ]
    holdout_rows = [
        {"label": "e", "mae": 0.06, "legal_action_count": 8, "action_shape": "x"},
        {"label": "f", "mae": 0.42, "legal_action_count": 2, "action_shape": "y"},
    ]

    metrics = fit_error_predictor_from_rows(train_rows, holdout_rows)

    assert metrics["passed"] is True
    assert metrics["holdout"]["pearson_mae"] > 0.9
    assert metrics["worst_predicted_holdout"][0]["label"] == "f"
    assert metrics["model"]["abstention_rule"]
    assert metrics["model"]["abstention_predicted_mae_cut"] > 0.0
    assert metrics["structural_only"]["model"]["numeric_fields"] == [
        "actor_to_act",
        "bet_count",
        "client_pos",
        "cut_pos",
        "legal_action_count",
    ]
