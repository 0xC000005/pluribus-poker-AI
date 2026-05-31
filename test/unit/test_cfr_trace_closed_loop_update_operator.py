import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_cfr_trace_closed_loop_update_operator import (  # noqa: E402
    fit_closed_loop_update_operator_from_payloads,
)


def _policy_for_progress(progress: float, *, final_a: float) -> list[float]:
    value = 0.5 + (float(final_a) - 0.5) * float(progress)
    value = max(0.01, min(0.99, value))
    return [value, 1.0 - value, 0.0]


def _record(label: str, *, iteration: int, final_a: float) -> dict:
    progress = min(1.0, 0.05 + 0.04 * float(iteration))
    if iteration == 10:
        progress = 0.55
    if iteration == 24:
        progress = 1.0
    policy = _policy_for_progress(progress, final_a=final_a)
    return {
        "label": label,
        "iteration": iteration,
        "street": 2,
        "legal_actions": [0, 1],
        "regret_mass": 1.0 + iteration,
        "strategy_mass": 1.0 + 0.25 * iteration,
        "hero_reach_mass": 1.0,
        "villain_reach_mass": 1.0,
        "public_belief_features": [final_a, 1.0 - final_a],
        "counterfactual_advantage": [final_a - 0.5, 0.5 - final_a, 0.0],
        "regret_policy": policy,
        "strategy_policy": policy,
        "top_matches_final": True,
        "l1_to_final_strategy": abs(final_a - policy[0]) * 2.0,
    }


def _payload(prefix: str) -> dict:
    records = []
    final_values = [0.12, 0.22, 0.32, 0.68, 0.78, 0.88]
    for index, final_a in enumerate(final_values):
        label = f"{prefix}-{index}"
        for iteration in range(0, 25):
            records.append(_record(label, iteration=iteration, final_a=final_a))
    return {"records": records}


def test_closed_loop_update_operator_rolls_forward_decision_state():
    metrics = fit_closed_loop_update_operator_from_payloads(
        _payload("train"),
        _payload("holdout"),
        start_iteration=5,
        rollout_end_iteration=24,
        uniform_iteration=10,
        reference_iteration=24,
        hidden_dim=32,
        epochs=300,
        batch_size=32,
        seed=21,
        device="cpu",
    )

    assert metrics["promotion"] is False
    assert metrics["target_fit_passed"] is True
    assert metrics["decision_passed"] is True
    assert metrics["passed"] is True
    assert metrics["model"]["max_step_l1"] > 0.0
    assert metrics["model"]["trust_region_quantile"] == 0.95
    assert metrics["holdout"]["mean_closed_loop_l1_to_reference"] < metrics["holdout"]["mean_low_l1_to_reference"]
    assert metrics["holdout"]["closed_loop_top_match_rate"] >= metrics["holdout"]["uniform_top_match_rate"]
