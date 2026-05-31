import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_cfr_trace_sequence_model import fit_trace_sequence_model_from_payloads  # noqa: E402


def _policy_for_progress(progress: float, *, final_a: float) -> list[float]:
    value = 0.5 + (float(final_a) - 0.5) * float(progress)
    value = max(0.01, min(0.99, value))
    return [value, 1.0 - value, 0.0]


def _record(label: str, *, iteration: int, final_a: float) -> dict:
    if iteration <= 5:
        progress = 0.20 + 0.08 * iteration
    elif iteration == 10:
        progress = 0.70
    else:
        progress = 1.0
    policy = _policy_for_progress(progress, final_a=final_a)
    return {
        "label": label,
        "iteration": iteration,
        "street": 2,
        "legal_actions": [0, 1],
        "regret_mass": 1.0 + iteration,
        "strategy_mass": 1.0 + 0.5 * iteration,
        "hero_reach_mass": 1.0,
        "villain_reach_mass": 1.0,
        "public_belief_features": [final_a, 1.0 - final_a],
        "regret_policy": policy,
        "strategy_policy": policy,
        "top_matches_final": True,
        "l1_to_final_strategy": abs(final_a - policy[0]) * 2.0,
    }


def _payload(prefix: str) -> dict:
    records = []
    final_values = [0.15, 0.25, 0.35, 0.65, 0.75, 0.85]
    for index, final_a in enumerate(final_values):
        label = f"{prefix}-{index}"
        for iteration in (0, 1, 2, 3, 4, 5, 10, 24):
            records.append(_record(label, iteration=iteration, final_a=final_a))
    return {"records": records}


def test_trace_sequence_model_learns_reference_policy_from_ordered_trace():
    metrics = fit_trace_sequence_model_from_payloads(
        _payload("train"),
        _payload("holdout"),
        low_trace_iteration=5,
        uniform_trace_iteration=10,
        target_trace_iteration=24,
        reference_trace_iteration=24,
        hidden_dim=24,
        epochs=250,
        batch_size=6,
        seed=7,
        device="cpu",
    )

    assert metrics["promotion"] is False
    assert metrics["target_fit_passed"] is True
    assert metrics["decision_passed"] is True
    assert metrics["passed"] is True
    assert metrics["mean_pred_l1_to_reference"] < metrics["mean_uniform_l1_to_reference"]
    assert metrics["pred_top_match_rate"] >= metrics["uniform_top_match_rate"]
