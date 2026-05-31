import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_cfr_trace_oracle_eta_selector import (  # noqa: E402
    fit_oracle_eta_selector_from_payloads,
    oracle_eta_targets,
)


def _record(
    label: str,
    iteration: int,
    *,
    low,
    uniform,
    reference,
    advantage,
    marker: float,
):
    if iteration <= 5:
        blend = iteration / 5.0
        policy = [
            low[0] * (1.0 - blend) + low[0] * blend,
            low[1] * (1.0 - blend) + low[1] * blend,
        ]
    elif iteration == 10:
        policy = uniform
    else:
        policy = reference
    return {
        "label": label,
        "iteration": iteration,
        "street": 2,
        "legal_actions": [0, 1],
        "regret_mass": 1.0 + iteration,
        "strategy_mass": 1.0 + iteration,
        "hero_reach_mass": 1.0,
        "villain_reach_mass": 1.0,
        "regret_policy": policy,
        "strategy_policy": policy,
        "counterfactual_advantage": advantage,
        "public_belief_features": [marker],
        "l1_to_final_strategy": sum(abs(a - b) for a, b in zip(policy, reference)),
    }


def _payload(labels):
    rows = []
    for label, direction, marker in labels:
        if direction == "left":
            reference = [0.88, 0.12]
            advantage = [1.0, 0.0]
        else:
            reference = [0.12, 0.88]
            advantage = [0.0, 1.0]
        for iteration in (0, 1, 2, 3, 4, 5, 10, 24):
            rows.append(
                _record(
                    label,
                    iteration,
                    low=[0.5, 0.5],
                    uniform=[0.55, 0.45],
                    reference=reference,
                    advantage=advantage,
                    marker=marker,
                )
            )
    return {"records": rows}


def test_oracle_eta_targets_choose_decision_best_update():
    payload = _payload([("root-left", "left", 1.0), ("root-right", "right", -1.0)])

    targets = oracle_eta_targets(
        payload,
        etas=[0.0, 1.0, 4.0],
        low_trace_iteration=5,
        uniform_trace_iteration=10,
        reference_trace_iteration=24,
    )

    assert targets["labels"] == ["root-left", "root-right"]
    assert targets["oracle_eta_by_label"] == {"root-left": 1.0, "root-right": 1.0}
    assert targets["oracle_beats_uniform_count"] == 2


def test_oracle_eta_selector_beats_low_and_uniform_on_learnable_payload():
    train_payload = _payload(
        [
            ("train-left-a", "left", 1.0),
            ("train-left-b", "left", 0.8),
            ("train-right-a", "right", -1.0),
            ("train-right-b", "right", -0.8),
        ]
    )
    holdout_payload = _payload(
        [
            ("holdout-left", "left", 0.9),
            ("holdout-right", "right", -0.9),
        ]
    )

    metrics = fit_oracle_eta_selector_from_payloads(
        train_payload,
        holdout_payload,
        etas=[0.0, 1.0, 4.0],
        low_trace_iteration=5,
        uniform_trace_iteration=10,
        reference_trace_iteration=24,
    )

    assert metrics["promotion"] is False
    assert metrics["passed"] is True
    assert metrics["holdout"]["mean_selected_l1_to_reference"] < metrics["holdout"]["mean_low_l1_to_reference"]
    assert metrics["holdout"]["mean_selected_l1_to_reference"] < metrics["holdout"]["mean_uniform_l1_to_reference"]
    assert metrics["holdout"]["selected_top_match_rate"] == 1.0
