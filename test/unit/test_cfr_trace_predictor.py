import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_cfr_trace_predictor import fit_trace_predictor_from_payloads  # noqa: E402


def _record(label: str, *, iteration: int, risky: bool, l1: float) -> dict:
    if risky:
        regret_policy = [1.0, 0.0, 0.0]
        strategy_policy = [0.0, 1.0, 0.0]
    else:
        regret_policy = [0.8, 0.2, 0.0]
        strategy_policy = [0.75, 0.25, 0.0]
    return {
        "label": label,
        "iteration": iteration,
        "street": 2,
        "legal_actions": [0, 1],
        "regret_mass": 10.0 if risky else 2.0,
        "strategy_mass": 1.0,
        "hero_reach_mass": 1.0,
        "villain_reach_mass": 1.0,
        "regret_policy": regret_policy,
        "strategy_policy": strategy_policy,
        "top_matches_final": not risky,
        "l1_to_final_strategy": l1,
    }


def test_trace_predictor_finds_holdout_error_signal():
    train = {
        "records": [
            _record("train-safe-1", iteration=5, risky=False, l1=0.05),
            _record("train-safe-2", iteration=5, risky=False, l1=0.07),
            _record("train-risky-1", iteration=5, risky=True, l1=0.9),
            _record("train-risky-2", iteration=5, risky=True, l1=1.0),
        ]
    }
    holdout = {
        "records": [
            _record("holdout-safe-1", iteration=5, risky=False, l1=0.06),
            _record("holdout-safe-2", iteration=5, risky=False, l1=0.08),
            _record("holdout-risky-1", iteration=5, risky=True, l1=0.85),
            _record("holdout-risky-2", iteration=5, risky=True, l1=1.1),
        ]
    }

    metrics = fit_trace_predictor_from_payloads(train, holdout, iterations=(5,))

    assert metrics["passed"] is True
    assert metrics["best_iteration"] == 5
    iter_metrics = metrics["iterations"]["5"]
    assert iter_metrics["ridge_mae"] < iter_metrics["train_mean_mae"]
    assert iter_metrics["predicted_high_mean_l1"] > iter_metrics["predicted_low_mean_l1"]
    assert iter_metrics["top_quintile_recall"] == 1.0
