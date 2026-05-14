import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_cfr_trace_budget_gate import eval_trace_budget_gate_from_payloads  # noqa: E402


def _record(label: str, *, iteration: int, risky: bool, l1: float) -> dict:
    return {
        "label": label,
        "iteration": iteration,
        "street": 2,
        "legal_actions": [0, 1],
        "regret_mass": 12.0 if risky else 2.0,
        "strategy_mass": 1.0,
        "hero_reach_mass": 1.0,
        "villain_reach_mass": 1.0,
        "regret_policy": [1.0, 0.0] if risky else [0.8, 0.2],
        "strategy_policy": [0.0, 1.0] if risky else [0.75, 0.25],
        "top_matches_final": not risky or iteration == 24,
        "l1_to_final_strategy": l1,
    }


def _payload(iteration_l1: dict[int, tuple[float, float]]) -> dict:
    records = []
    for iteration, (safe_l1, risky_l1) in iteration_l1.items():
        records.extend(
            [
                _record("safe-1", iteration=iteration, risky=False, l1=safe_l1),
                _record("safe-2", iteration=iteration, risky=False, l1=safe_l1),
                _record("risky-1", iteration=iteration, risky=True, l1=risky_l1),
                _record("risky-2", iteration=iteration, risky=True, l1=risky_l1),
            ]
        )
    return {"records": records}


def test_trace_budget_gate_compares_adaptive_to_uniform_compute():
    train = _payload({5: (0.1, 1.0), 10: (0.05, 0.8), 24: (0.0, 0.0)})
    holdout = _payload({5: (0.1, 1.0), 10: (0.05, 0.8), 24: (0.0, 0.0)})

    metrics = eval_trace_budget_gate_from_payloads(
        train,
        holdout,
        low_trace_iteration=5,
        uniform_trace_iteration=10,
        reference_trace_iteration=24,
        selection_fraction=0.5,
    )

    assert metrics["passed"] is True
    assert metrics["selected_count"] == 2
    assert metrics["mean_adaptive_l1_to_reference"] < metrics["mean_uniform_l1_to_reference"]
    assert metrics["adaptive_mean_trace_iterations"] == 15.5
