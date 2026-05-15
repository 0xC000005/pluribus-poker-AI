import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_solver_budget_boundary_predictor as predictor  # noqa: E402


def test_boundary_predictor_uses_cheap_features_for_holdout_selection():
    frontier_metrics = {
        "records": [
            {
                "label": f"root-{idx:04d}-street2",
                "passed": True,
                "budgets": {
                    "150": {"l1_to_reference": live_l1, "kl_to_reference": live_kl, "latency_ms": 100.0},
                    "350": {"l1_to_reference": high_l1, "kl_to_reference": high_kl, "latency_ms": 200.0},
                },
            }
            for idx, live_l1, live_kl, high_l1, high_kl in [
                (0, 0.4, 0.2, 0.1, 0.05),
                (1, 0.3, 0.1, 0.2, 0.08),
                (2, 0.5, 0.3, 0.15, 0.06),
                (3, 0.2, 0.09, 0.18, 0.07),
            ]
        ]
    }
    profile_metrics = {
        "records": [
            {
                "label": "root-0000-street2",
                "passed": True,
                "profiles": {
                    "live": {"iterations": 150, "strategy": [0.8, 0.2]},
                    "fast-live": {"iterations": 100, "strategy": [0.4, 0.6]},
                },
            },
            {
                "label": "root-0001-street2",
                "passed": True,
                "profiles": {
                    "live": {"iterations": 150, "strategy": [0.55, 0.45]},
                    "fast-live": {"iterations": 100, "strategy": [0.5, 0.5]},
                },
            },
            {
                "label": "root-0002-street2",
                "passed": True,
                "profiles": {
                    "live": {"iterations": 150, "strategy": [0.9, 0.1]},
                    "fast-live": {"iterations": 100, "strategy": [0.45, 0.55]},
                },
            },
            {
                "label": "root-0003-street2",
                "passed": True,
                "profiles": {
                    "live": {"iterations": 150, "strategy": [0.6, 0.4]},
                    "fast-live": {"iterations": 100, "strategy": [0.58, 0.42]},
                },
            },
        ]
    }
    feature_labels = np.asarray([f"root-{idx:04d}-street2" for idx in range(4)])
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    metrics = predictor.evaluate_boundary_predictor(
        frontier_metrics,
        profile_metrics,
        features,
        feature_labels,
        train_start_index=0,
        train_limit=2,
        holdout_start_index=2,
        holdout_limit=2,
        select_train_top_k=1,
        escalation_budget=350,
    )

    assert metrics["passed"] is True
    assert metrics["holdout_selected"] == 1
    assert metrics["selected_labels"] == ["root-0002-street2"]
    assert metrics["oracle_selected_labels"] == ["root-0002-street2"]
    assert metrics["top_k_recall"] == 1.0
    assert metrics["holdout_profile_l1_mae"] < metrics["holdout_profile_l1_mean_baseline_mae"]
    assert metrics["live_mean_l1"] == 0.35
    assert metrics["predicted_selective_mean_l1"] == 0.175
    assert metrics["uniform_escalation_mean_l1"] == 0.165
    assert metrics["predicted_selective_mean_latency_ms"] == 150.0
    assert metrics["predicted_latency_ratio_to_live"] == 1.5


def test_trace_feature_loader_uses_requested_iteration_and_context():
    payload = {
        "records": [
            {
                "label": "root-0000-street2",
                "iteration": 5,
                "legal_actions": [0, 1],
                "regret_policy": [0.75, 0.25],
                "strategy_policy": [0.6, 0.4],
                "regret_mass": 3.0,
                "strategy_mass": 4.0,
                "hero_reach_mass": 5.0,
                "villain_reach_mass": 6.0,
                "street": 2,
                "public_belief_features": [0.1, 0.2, 0.3],
            },
            {
                "label": "root-0000-street2",
                "iteration": 10,
                "legal_actions": [0, 1],
                "regret_policy": [0.5, 0.5],
                "strategy_policy": [0.5, 0.5],
                "regret_mass": 1.0,
                "strategy_mass": 1.0,
                "hero_reach_mass": 1.0,
                "villain_reach_mass": 1.0,
                "street": 2,
                "public_belief_features": [9.0],
            },
        ]
    }

    features, labels = predictor.trace_features_from_payloads([payload], iteration=5)

    assert labels.tolist() == ["root-0000-street2"]
    assert features.shape == (1, 15)
    np.testing.assert_allclose(features[0, -3:], [0.1, 0.2, 0.3])
