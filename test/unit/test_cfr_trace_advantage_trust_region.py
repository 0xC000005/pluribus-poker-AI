import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_cfr_trace_advantage_trust_region import evaluate_advantage_trust_region  # noqa: E402


def _record(label: str, iteration: int, *, low, uniform, reference, advantage):
    policy = low if iteration == 5 else uniform if iteration == 10 else reference
    return {
        "label": label,
        "iteration": iteration,
        "legal_actions": [0, 1],
        "strategy_policy": policy,
        "counterfactual_advantage": advantage,
        "l1_to_final_strategy": sum(abs(a - b) for a, b in zip(policy, reference)),
    }


def _payload() -> dict:
    records = []
    for label in ("a", "b", "c", "d"):
        advantage = [1.0, -1.0] if label in ("a", "b") else [-1.0, 1.0]
        reference = [0.8, 0.2] if label in ("a", "b") else [0.2, 0.8]
        for iteration in (5, 10, 24):
            records.append(
                _record(
                    label,
                    iteration,
                    low=[0.5, 0.5],
                    uniform=[0.55, 0.45],
                    reference=reference,
                    advantage=advantage,
                )
            )
    return {"records": records}


def test_advantage_trust_region_selects_eta_on_train_and_passes_holdout(tmp_path):
    train = tmp_path / "train.json"
    holdout = tmp_path / "holdout.json"
    import json

    train.write_text(json.dumps(_payload()), encoding="utf-8")
    holdout.write_text(json.dumps(_payload()), encoding="utf-8")

    metrics = evaluate_advantage_trust_region(
        train_trace_json=train,
        holdout_trace_json=holdout,
        etas=[0.0, 0.1, 0.5, 1.0],
        low_trace_iteration=5,
        uniform_trace_iteration=10,
        reference_trace_iteration=24,
    )

    assert metrics["promotion"] is False
    assert metrics["selected_eta"] > 0.0
    assert metrics["holdout"]["mean_update_l1_to_reference"] < metrics["holdout"]["mean_low_l1_to_reference"]
    assert metrics["holdout"]["mean_update_l1_to_reference"] < metrics["holdout"]["mean_uniform_l1_to_reference"]
    assert metrics["passed"] is True
