import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_joint_pbs_policy_warm_start import _summarize_records  # noqa: E402


def test_joint_pbs_policy_warm_start_summary_requires_real_improvement():
    records = [
        {
            "passed": True,
            "low_l1_to_reference": 0.8,
            "warm_l1_to_reference": 0.4,
            "low_allin_selected": True,
            "warm_allin_selected": False,
            "reference_allin_selected": False,
            "low_agrees_with_reference": False,
            "warm_agrees_with_reference": True,
        },
        {
            "passed": True,
            "low_l1_to_reference": 0.6,
            "warm_l1_to_reference": 0.5,
            "low_allin_selected": False,
            "warm_allin_selected": False,
            "reference_allin_selected": False,
            "low_agrees_with_reference": True,
            "warm_agrees_with_reference": True,
        },
    ]

    summary = _summarize_records(records)

    assert summary["passed"]
    assert summary["mean_warm_l1_to_reference"] < summary["mean_low_l1_to_reference"]
    assert summary["warm_action_agreement"] > summary["low_action_agreement"]
    assert summary["warm_allin_gap"] < summary["low_allin_gap"]


def test_joint_pbs_policy_warm_start_summary_rejects_l1_regression():
    records = [
        {
            "passed": True,
            "low_l1_to_reference": 0.2,
            "warm_l1_to_reference": 0.4,
            "low_allin_selected": False,
            "warm_allin_selected": False,
            "reference_allin_selected": False,
            "low_agrees_with_reference": True,
            "warm_agrees_with_reference": True,
        }
    ]

    assert not _summarize_records(records)["passed"]
