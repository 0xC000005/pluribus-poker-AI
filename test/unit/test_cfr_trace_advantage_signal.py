import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_cfr_trace_advantage_signal import analyze_trace_advantage_signal_from_payload  # noqa: E402


def _record(
    label: str,
    *,
    iteration: int,
    strategy_policy: list[float],
    regret_policy: list[float],
    advantage_policy: list[float],
    reference_policy: list[float],
) -> dict:
    return {
        "label": label,
        "iteration": iteration,
        "legal_actions": [0, 1],
        "strategy_policy": strategy_policy,
        "regret_policy": regret_policy,
        "advantage_policy": advantage_policy,
        "top_matches_final": strategy_policy.index(max(strategy_policy))
        == reference_policy.index(max(reference_policy)),
        "l1_to_final_strategy": sum(abs(a - b) for a, b in zip(strategy_policy, reference_policy)),
    }


def test_advantage_signal_compares_against_low_and_uniform_baselines():
    records = []
    for label, reference in (("a", [0.9, 0.1, 0.0]), ("b", [0.1, 0.9, 0.0])):
        records.append(
            _record(
                label,
                iteration=5,
                strategy_policy=[0.5, 0.5, 0.0],
                regret_policy=[0.55, 0.45, 0.0],
                advantage_policy=reference,
                reference_policy=reference,
            )
        )
        records.append(
            _record(
                label,
                iteration=10,
                strategy_policy=[0.6, 0.4, 0.0],
                regret_policy=[0.6, 0.4, 0.0],
                advantage_policy=[0.6, 0.4, 0.0],
                reference_policy=reference,
            )
        )
        records.append(
            _record(
                label,
                iteration=24,
                strategy_policy=reference,
                regret_policy=reference,
                advantage_policy=reference,
                reference_policy=reference,
            )
        )

    metrics = analyze_trace_advantage_signal_from_payload(
        {"records": records},
        low_trace_iteration=5,
        uniform_trace_iteration=10,
        reference_trace_iteration=24,
    )

    assert metrics["passed"] is True
    assert metrics["n_eval"] == 2
    assert metrics["mean_advantage_l1_to_reference"] == 0.0
    assert metrics["mean_advantage_l1_to_reference"] < metrics["mean_low_l1_to_reference"]
    assert metrics["mean_advantage_l1_to_reference"] < metrics["mean_uniform_l1_to_reference"]
    assert metrics["advantage_top_match_rate"] == 1.0
