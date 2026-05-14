import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_cfr_trace_delta_mlp import fit_trace_delta_mlp_from_payloads  # noqa: E402


def _record(
    label: str,
    *,
    iteration: int,
    low_policy: list[float],
    regret_policy: list[float],
    target_policy: list[float],
    uniform_policy: list[float],
    public_belief_features: list[float] | None = None,
) -> dict:
    if iteration == 5:
        policy = low_policy
    elif iteration == 10:
        policy = uniform_policy
    else:
        policy = target_policy
    return {
        "label": label,
        "iteration": iteration,
        "street": 2,
        "legal_actions": [0, 1],
        "regret_mass": 8.0,
        "strategy_mass": 1.0,
        "hero_reach_mass": 1.0,
        "villain_reach_mass": 1.0,
        "regret_policy": regret_policy,
        "strategy_policy": policy,
        "public_belief_features": public_belief_features or [],
        "top_matches_final": policy.index(max(policy)) == target_policy.index(max(target_policy)),
        "l1_to_final_strategy": sum(abs(a - b) for a, b in zip(policy, target_policy)),
    }


def _payload() -> dict:
    rows = []
    specs = [
        ("a", [1.0, 0.0, 0.0], [0.88, 0.12, 0.0]),
        ("b", [0.9, 0.1, 0.0], [0.84, 0.16, 0.0]),
        ("c", [0.0, 1.0, 0.0], [0.12, 0.88, 0.0]),
        ("d", [0.1, 0.9, 0.0], [0.16, 0.84, 0.0]),
    ]
    for label, regret_policy, target_policy in specs:
        low_policy = [0.5, 0.5, 0.0]
        uniform_policy = [0.55, 0.45, 0.0]
        for iteration in (5, 10, 24):
            rows.append(
                _record(
                    label,
                    iteration=iteration,
                    low_policy=low_policy,
                    regret_policy=regret_policy,
                    target_policy=target_policy,
                    uniform_policy=uniform_policy,
                )
            )
    return {"records": rows}


def test_trace_delta_mlp_learns_non_linear_update_mapping():
    metrics = fit_trace_delta_mlp_from_payloads(
        _payload(),
        _payload(),
        low_trace_iteration=5,
        target_trace_iteration=24,
        uniform_trace_iteration=10,
        reference_trace_iteration=24,
        hidden_dim=16,
        epochs=250,
        learning_rate=0.02,
        seed=7,
        device="cpu",
    )

    assert metrics["promotion"] is False
    assert metrics["target_fit_passed"] is True
    assert metrics["decision_passed"] is True
    assert metrics["passed"] is True
    assert metrics["mean_pred_l1_to_reference"] < metrics["mean_low_l1_to_reference"]
    assert metrics["mean_pred_l1_to_reference"] < metrics["mean_uniform_l1_to_reference"]
    assert metrics["pred_top_match_rate"] == 1.0
    for record in metrics["records"]:
        assert record["pred_policy"][2] == 0.0


def test_trace_delta_mlp_does_not_promote_target_fit_alone():
    payload = _payload()
    metrics = fit_trace_delta_mlp_from_payloads(
        payload,
        payload,
        low_trace_iteration=5,
        target_trace_iteration=10,
        uniform_trace_iteration=10,
        reference_trace_iteration=24,
        hidden_dim=16,
        epochs=200,
        learning_rate=0.02,
        seed=11,
        device="cpu",
    )

    assert metrics["promotion"] is False
    assert metrics["target_fit_passed"] is True
    assert metrics["decision_passed"] is False
    assert metrics["passed"] is False


def test_trace_delta_mlp_uses_optional_public_belief_features():
    rows = []
    specs = [
        ("belief-a", [1.0, 0.0], [0.9, 0.1, 0.0]),
        ("belief-b", [0.0, 1.0], [0.1, 0.9, 0.0]),
        ("belief-c", [1.0, 0.1], [0.85, 0.15, 0.0]),
        ("belief-d", [0.1, 1.0], [0.15, 0.85, 0.0]),
    ]
    for label, belief_features, target_policy in specs:
        for iteration in (5, 10, 24):
            rows.append(
                _record(
                    label,
                    iteration=iteration,
                    low_policy=[0.5, 0.5, 0.0],
                    regret_policy=[0.5, 0.5, 0.0],
                    target_policy=target_policy,
                    uniform_policy=[0.5, 0.5, 0.0],
                    public_belief_features=belief_features,
                )
            )
    payload = {"records": rows}

    metrics = fit_trace_delta_mlp_from_payloads(
        payload,
        payload,
        low_trace_iteration=5,
        target_trace_iteration=24,
        uniform_trace_iteration=10,
        reference_trace_iteration=24,
        hidden_dim=16,
        epochs=300,
        learning_rate=0.02,
        seed=13,
        device="cpu",
    )

    assert metrics["passed"] is True
    assert metrics["mean_pred_l1_to_reference"] < 0.1
