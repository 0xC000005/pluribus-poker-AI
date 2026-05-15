import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_cfr_trace_sequence_predictor import (  # noqa: E402
    fit_trace_sequence_predictor_from_payloads,
)


def _policy_for_slope(slope: float) -> list[float]:
    value = max(0.05, min(0.95, 0.5 + float(slope)))
    return [value, 1.0 - value, 0.0]


def _policy_at_l1(l1: float) -> list[float]:
    half = max(0.0, min(1.0, float(l1) / 2.0))
    return [1.0 - half, half, 0.0]


def _record(label: str, *, iteration: int, slope: float, low_l1: float, uniform_l1: float) -> dict:
    if iteration <= 5:
        l1 = low_l1
        policy = _policy_at_l1(low_l1 * (iteration + 1) / 6.0)
    elif iteration == 10:
        l1 = uniform_l1
        policy = _policy_at_l1(uniform_l1)
    else:
        l1 = 0.0
        policy = [1.0, 0.0, 0.0]
    return {
        "label": label,
        "iteration": iteration,
        "street": 2,
        "legal_actions": [0, 1],
        "regret_mass": 1.0 + abs(slope) * (iteration + 1),
        "strategy_mass": 1.0 + 0.25 * iteration,
        "hero_reach_mass": 1.0,
        "villain_reach_mass": 1.0,
        "regret_policy": _policy_for_slope(slope * (iteration + 1) / 6.0),
        "strategy_policy": policy,
        "top_matches_final": l1 < 0.2,
        "l1_to_final_strategy": l1,
    }


def _payload(prefix: str) -> dict:
    records = []
    specs = [
        ("safe-a", 0.02, 0.10, 0.30),
        ("safe-b", 0.03, 0.12, 0.32),
        ("improve-a", 0.30, 0.90, 0.15),
        ("improve-b", 0.35, 0.95, 0.20),
    ]
    for suffix, slope, low_l1, uniform_l1 in specs:
        label = f"{prefix}-{suffix}"
        for iteration in (0, 1, 2, 3, 4, 5, 10, 24):
            records.append(
                _record(
                    label,
                    iteration=iteration,
                    slope=slope,
                    low_l1=low_l1,
                    uniform_l1=uniform_l1,
                )
            )
    return {"records": records}


def test_trace_sequence_predictor_finds_continuation_signal():
    metrics = fit_trace_sequence_predictor_from_payloads(
        _payload("train"),
        _payload("holdout"),
        low_trace_iteration=5,
        uniform_trace_iteration=10,
        reference_trace_iteration=24,
        selection_fraction=0.5,
    )

    assert metrics["promotion"] is False
    assert metrics["prediction_signal_found"] is True
    assert metrics["passed"] is True
    assert metrics["ridge_mae"] < metrics["train_mean_mae"]
    assert metrics["selected_count"] == 2
    assert metrics["selected_mean_target_improvement"] > metrics["rejected_mean_target_improvement"]
    assert metrics["mean_adaptive_l1_to_reference"] < metrics["mean_uniform_l1_to_reference"]
