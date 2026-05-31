import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_cfr_trace_mirror_update_operator import (  # noqa: E402
    fit_mirror_update_operator_from_payloads,
    mirror_update_policy,
)


def _record(
    label: str,
    iteration: int,
    *,
    low,
    uniform,
    reference,
    advantage,
    public_belief_features=None,
):
    policy = low if iteration == 5 else uniform if iteration == 10 else reference
    return {
        "label": label,
        "iteration": iteration,
        "street": 2,
        "legal_actions": [0, 1],
        "regret_mass": 1.0,
        "strategy_mass": 1.0,
        "hero_reach_mass": 1.0,
        "villain_reach_mass": 1.0,
        "regret_policy": policy,
        "strategy_policy": policy,
        "counterfactual_advantage": advantage,
        "public_belief_features": public_belief_features or [],
        "l1_to_final_strategy": sum(abs(a - b) for a, b in zip(policy, reference)),
    }


def test_mirror_update_policy_respects_legality_and_eta():
    record = {
        "strategy_policy": [0.0, 0.5, 0.5],
        "legal_actions": [1, 2],
        "counterfactual_advantage": [0.0, -2.0, 2.0],
    }

    no_update = mirror_update_policy(record, eta=0.0, min_base_prob=1e-6)
    updated = mirror_update_policy(record, eta=2.0, min_base_prob=1e-6)

    np.testing.assert_allclose(no_update, [0.0, 0.5, 0.5], atol=1e-8)
    assert updated[0] == 0.0
    assert updated[2] > updated[1]
    assert updated.sum() == np.float64(1.0)


def test_mirror_update_operator_learns_decision_improving_update_strength():
    rows = []
    specs = [
        ("left-a", [1.0, 0.0], [0.9, 0.1], [1.0, 0.0]),
        ("left-b", [1.0, 0.0], [0.85, 0.15], [1.0, 0.0]),
        ("right-a", [0.0, 1.0], [0.1, 0.9], [0.0, 1.0]),
        ("right-b", [0.0, 1.0], [0.15, 0.85], [0.0, 1.0]),
    ]
    for label, advantage, reference, belief in specs:
        for iteration in (5, 10, 24):
            rows.append(
                _record(
                    label,
                    iteration,
                    low=[0.5, 0.5],
                    uniform=[0.55, 0.45],
                    reference=reference,
                    advantage=advantage,
                    public_belief_features=belief,
                )
            )
    payload = {"records": rows}

    metrics = fit_mirror_update_operator_from_payloads(
        payload,
        payload,
        low_trace_iteration=5,
        target_trace_iteration=24,
        uniform_trace_iteration=10,
        reference_trace_iteration=24,
        hidden_dim=16,
        epochs=250,
        learning_rate=0.02,
        batch_size=4,
        seed=17,
        device="cpu",
        max_eta=8.0,
    )

    assert metrics["promotion"] is False
    assert metrics["target_fit_passed"] is True
    assert metrics["decision_passed"] is True
    assert metrics["passed"] is True
    assert metrics["holdout"]["mean_update_l1_to_reference"] < metrics["holdout"]["mean_low_l1_to_reference"]
    assert metrics["holdout"]["mean_update_l1_to_reference"] < metrics["holdout"]["mean_uniform_l1_to_reference"]
    assert metrics["holdout"]["update_top_match_rate"] == 1.0
