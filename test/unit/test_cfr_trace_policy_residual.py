import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_cfr_trace_policy_residual import fit_trace_policy_residual_from_payloads  # noqa: E402


def _record(
    label: str,
    *,
    iteration: int,
    low_policy: list[float],
    regret_policy: list[float],
    final_policy: list[float],
) -> dict:
    policy = final_policy if iteration == 24 else low_policy
    return {
        "label": label,
        "iteration": iteration,
        "street": 2,
        "legal_actions": [0, 1],
        "regret_mass": 10.0,
        "strategy_mass": 1.0,
        "hero_reach_mass": 1.0,
        "villain_reach_mass": 1.0,
        "regret_policy": regret_policy,
        "strategy_policy": policy,
        "top_matches_final": policy.index(max(policy)) == final_policy.index(max(final_policy)),
        "l1_to_final_strategy": sum(abs(a - b) for a, b in zip(policy, final_policy)),
    }


def _payload() -> dict:
    rows = []
    specs = [
        ("a", [0.2, 0.8], [0.9, 0.1], [0.9, 0.1]),
        ("b", [0.25, 0.75], [0.85, 0.15], [0.85, 0.15]),
        ("c", [0.8, 0.2], [0.2, 0.8], [0.2, 0.8]),
        ("d", [0.75, 0.25], [0.15, 0.85], [0.15, 0.85]),
    ]
    for label, low_policy, regret_policy, final_policy in specs:
        rows.append(
            _record(
                label,
                iteration=5,
                low_policy=low_policy,
                regret_policy=regret_policy,
                final_policy=final_policy,
            )
        )
        rows.append(
            _record(
                label,
                iteration=10,
                low_policy=[0.5, 0.5],
                regret_policy=regret_policy,
                final_policy=final_policy,
            )
        )
        rows.append(
            _record(
                label,
                iteration=24,
                low_policy=final_policy,
                regret_policy=regret_policy,
                final_policy=final_policy,
            )
        )
    return {"records": rows}


def test_trace_policy_residual_can_learn_final_policy_mapping():
    metrics = fit_trace_policy_residual_from_payloads(
        _payload(),
        _payload(),
        low_trace_iteration=5,
        uniform_trace_iteration=10,
        reference_trace_iteration=24,
    )

    assert metrics["passed"] is True
    assert metrics["mean_pred_l1_to_reference"] < metrics["mean_low_l1_to_reference"]
    assert metrics["mean_pred_l1_to_reference"] < metrics["mean_uniform_l1_to_reference"]
    assert metrics["pred_top_match_rate"] == 1.0
