import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_solver_budget_policy_integration import evaluate_budget_policy_integration  # noqa: E402


def _policy(threshold):
    return {
        "policy_type": "solver_budget_selective_profile",
        "feature_source": "solver-native",
        "feature_dim": 10,
        "feature_mean": [0.0] * 10,
        "feature_std": [1.0] * 10,
        "ridge_weights": [0.0] * 11,
        "score_threshold": threshold,
    }


def _record(label, *, fast_strategy, live_strategy, fast_action, live_action, fast_latency, live_latency):
    return {
        "label": label,
        "passed": True,
        "profiles": {
            "fast-live": {
                "action": fast_action,
                "latency_ms": fast_latency,
                "strategy": fast_strategy,
                "illegal_mass": 0.0,
            },
            "live": {
                "action": live_action,
                "latency_ms": live_latency,
                "strategy": live_strategy,
                "illegal_mass": 0.0,
            },
        },
    }


def test_budget_policy_integration_charges_fast_plus_live_for_escalation():
    policy = _policy(threshold=0.5)
    policy["ridge_weights"][1] = 1.0
    profile = {
        "records": [
            _record(
                "case-0000-street2",
                fast_strategy=[0.8, 0.2],
                live_strategy=[0.2, 0.8],
                fast_action=0,
                live_action=1,
                fast_latency=10,
                live_latency=30,
            ),
            _record(
                "case-0001-street2",
                fast_strategy=[0.1, 0.9],
                live_strategy=[0.1, 0.9],
                fast_action=1,
                live_action=1,
                fast_latency=10,
                live_latency=30,
            ),
        ],
    }

    metrics = evaluate_budget_policy_integration(profile, policy)

    assert metrics["passed"] is True
    assert metrics["n_escalated"] == 1
    assert metrics["fast_action_agreement"] == 0.5
    assert metrics["selective_action_agreement"] == 1.0
    assert metrics["selective_online_mean_latency_ms"] == 25.0
    assert metrics["live_mean_latency_ms"] == 30.0
