import importlib
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _frontier_record(label, *, low_action, low_l1, uniform_l1, high_l1):
    return {
        "label": label,
        "passed": True,
        "budgets": {
            "5": {
                "action": low_action,
                "allin_prob": 0.8 if low_action == 8 else 0.1,
                "allin_selected": low_action == 8,
                "illegal_mass": 0.0,
                "l1_to_reference": low_l1,
                "kl_to_reference": low_l1 / 2.0,
                "latency_ms": 10.0,
            },
            "10": {
                "action": low_action,
                "allin_prob": 0.5 if low_action == 8 else 0.1,
                "allin_selected": low_action == 8,
                "illegal_mass": 0.0,
                "l1_to_reference": uniform_l1,
                "kl_to_reference": uniform_l1 / 2.0,
                "latency_ms": 20.0,
            },
            "25": {
                "action": 1,
                "allin_prob": 0.2 if low_action == 8 else 0.1,
                "allin_selected": False,
                "illegal_mass": 0.0,
                "l1_to_reference": high_l1,
                "kl_to_reference": high_l1 / 2.0,
                "latency_ms": 30.0,
            },
        },
    }


def test_budget_selector_beats_uniform_when_low_budget_features_predict_improvement():
    selector = importlib.import_module("eval_public_belief_search_budget_selector")
    frontier = {
        "records": [
            _frontier_record("root-0000-street2", low_action=8, low_l1=0.7, uniform_l1=0.5, high_l1=0.1),
            _frontier_record("root-0001-street2", low_action=1, low_l1=0.3, uniform_l1=0.25, high_l1=0.24),
            _frontier_record("root-0002-street2", low_action=8, low_l1=0.8, uniform_l1=0.6, high_l1=0.1),
            _frontier_record("root-0003-street2", low_action=1, low_l1=0.2, uniform_l1=0.2, high_l1=0.18),
        ]
    }

    metrics = selector.evaluate_search_budget_selector(
        frontier,
        train_start_index=0,
        train_limit=2,
        holdout_start_index=2,
        holdout_limit=2,
        select_train_top_k=1,
        low_budget=5,
        uniform_budget=10,
        high_budget=25,
    )

    assert metrics["passed"] is True
    assert metrics["holdout_selected"] == 1
    assert metrics["selected_labels"] == ["root-0002-street2"]
    assert metrics["low_mean_l1"] == 0.5
    assert metrics["uniform_mean_l1"] == 0.4
    assert metrics["selective_mean_l1"] == 0.15
    assert metrics["selective_mean_latency_ms"] == metrics["uniform_mean_latency_ms"]


def test_budget_selector_requires_frontier_rows_and_budgets():
    selector = importlib.import_module("eval_public_belief_search_budget_selector")

    try:
        selector.evaluate_search_budget_selector(
            {"records": []},
            train_start_index=0,
            train_limit=1,
            holdout_start_index=1,
            holdout_limit=1,
        )
    except ValueError as exc:
        assert "no passed frontier records" in str(exc)
    else:
        raise AssertionError("expected missing frontier records to fail")


def test_budget_selector_uses_train_selection_fraction_on_holdout():
    selector = importlib.import_module("eval_public_belief_search_budget_selector")
    frontier = {
        "records": [
            _frontier_record("root-0000-street2", low_action=8, low_l1=0.7, uniform_l1=0.5, high_l1=0.1),
            _frontier_record("root-0001-street2", low_action=1, low_l1=0.3, uniform_l1=0.25, high_l1=0.24),
            _frontier_record("root-0002-street2", low_action=8, low_l1=0.8, uniform_l1=0.6, high_l1=0.1),
            _frontier_record("root-0003-street2", low_action=8, low_l1=0.75, uniform_l1=0.55, high_l1=0.12),
        ]
    }

    metrics = selector.evaluate_search_budget_selector(
        frontier,
        train_start_index=0,
        train_limit=2,
        holdout_start_index=2,
        holdout_limit=2,
        select_train_top_k=1,
        low_budget=5,
        uniform_budget=10,
        high_budget=25,
    )

    assert metrics["selection_rule"] == "top_predicted_fraction_from_train_split"
    assert metrics["holdout_selected"] == 1
    assert metrics["selective_mean_latency_ms"] == metrics["uniform_mean_latency_ms"]


def test_budget_selector_uses_strategy_vectors_when_exported():
    selector = importlib.import_module("eval_public_belief_search_budget_selector")
    frontier = {
        "records": [
            _frontier_record("root-0000-street2", low_action=8, low_l1=0.7, uniform_l1=0.5, high_l1=0.1),
            _frontier_record("root-0001-street2", low_action=1, low_l1=0.3, uniform_l1=0.25, high_l1=0.24),
            _frontier_record("root-0002-street2", low_action=8, low_l1=0.8, uniform_l1=0.6, high_l1=0.1),
            _frontier_record("root-0003-street2", low_action=1, low_l1=0.2, uniform_l1=0.2, high_l1=0.18),
        ]
    }
    for record in frontier["records"]:
        for budget in record["budgets"].values():
            action = int(budget["action"])
            strategy = [0.0] * 9
            strategy[action] = 1.0 - float(budget["allin_prob"])
            strategy[8] = float(budget["allin_prob"])
            budget["strategy"] = strategy

    metrics = selector.evaluate_search_budget_selector(
        frontier,
        train_start_index=0,
        train_limit=2,
        holdout_start_index=2,
        holdout_limit=2,
        select_train_top_k=1,
        low_budget=5,
        uniform_budget=10,
        high_budget=25,
    )

    assert metrics["policy"]["feature_source"] == "low_budget_strategy_vector"
    assert metrics["policy"]["feature_dim"] > 4
