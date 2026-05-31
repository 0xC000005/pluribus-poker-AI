import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_solver_budget_selective_profile import (  # noqa: E402
    build_solver_native_features,
    evaluate_selective_profile,
    fit_selective_policy,
    score_selective_policy,
)


def _record(case_idx, *, fast_action, live_action, fast_latency, live_latency, fast_p):
    live = np.array([1.0 - fast_p, fast_p], dtype=float)
    fast = live[::-1]
    return {
        "label": f"case-{case_idx:04d}-street2",
        "passed": True,
        "profiles": {
            "fast-live": {
                "action": fast_action,
                "latency_ms": fast_latency,
                "strategy": fast.tolist(),
            },
            "live": {
                "action": live_action,
                "latency_ms": live_latency,
                "strategy": live.tolist(),
            },
        },
    }


def test_selective_profile_escalates_predicted_boundaries():
    profile = {
        "records": [
            _record(0, fast_action=0, live_action=1, fast_latency=10, live_latency=20, fast_p=0.9),
            _record(1, fast_action=1, live_action=1, fast_latency=10, live_latency=20, fast_p=0.55),
            _record(2, fast_action=0, live_action=1, fast_latency=10, live_latency=20, fast_p=0.95),
            _record(3, fast_action=1, live_action=1, fast_latency=10, live_latency=20, fast_p=0.52),
        ],
    }
    features = np.array(
        [
            [1.0],
            [0.0],
            [1.2],
            [0.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([record["label"] for record in profile["records"]])

    metrics = evaluate_selective_profile(
        profile,
        features,
        labels,
        train_start_index=0,
        train_limit=2,
        holdout_start_index=2,
        holdout_limit=2,
        select_train_top_k=1,
        target_name="profile-l1",
    )

    assert metrics["passed"] is True
    assert metrics["holdout_selected"] == 1
    assert metrics["fast_action_agreement"] == 0.5
    assert metrics["predicted_selective_action_agreement"] == 1.0
    assert metrics["predicted_selective_mean_l1_to_live"] < metrics["fast_mean_l1_to_live"]
    assert metrics["predicted_latency_ratio_to_live"] < 1.0


def test_solver_native_features_can_drive_boundary_escalation():
    profile = {
        "records": [
            {
                "label": "case-0000-street2",
                "passed": True,
                "profiles": {
                    "fast-live": {"action": 0, "latency_ms": 10, "strategy": [0.55, 0.45]},
                    "live": {"action": 1, "latency_ms": 20, "strategy": [0.45, 0.55]},
                },
            },
            {
                "label": "case-0001-street2",
                "passed": True,
                "profiles": {
                    "fast-live": {"action": 0, "latency_ms": 10, "strategy": [0.9, 0.1]},
                    "live": {"action": 0, "latency_ms": 20, "strategy": [0.9, 0.1]},
                },
            },
            {
                "label": "case-0002-street2",
                "passed": True,
                "profiles": {
                    "fast-live": {"action": 0, "latency_ms": 10, "strategy": [0.52, 0.48]},
                    "live": {"action": 1, "latency_ms": 20, "strategy": [0.48, 0.52]},
                },
            },
            {
                "label": "case-0003-street2",
                "passed": True,
                "profiles": {
                    "fast-live": {"action": 0, "latency_ms": 10, "strategy": [0.95, 0.05]},
                    "live": {"action": 0, "latency_ms": 20, "strategy": [0.95, 0.05]},
                },
            },
        ],
    }
    features, labels = build_solver_native_features(profile, candidate_profile="fast-live")

    metrics = evaluate_selective_profile(
        profile,
        features,
        labels,
        train_start_index=0,
        train_limit=2,
        holdout_start_index=2,
        holdout_limit=2,
        select_train_top_k=1,
        target_name="action-disagreement",
    )

    assert metrics["passed"] is True
    assert metrics["feature_dim"] > 2
    assert metrics["holdout_selected"] == 1
    assert metrics["predicted_selective_action_agreement"] == 1.0

    policy = fit_selective_policy(
        profile,
        features,
        labels,
        train_start_index=0,
        train_limit=2,
        select_train_top_k=1,
        target_name="action-disagreement",
        feature_source="solver-native",
    )
    holdout_scores = [score_selective_policy(policy, features[idx]) for idx in (2, 3)]
    selected = [score >= policy["score_threshold"] for score in holdout_scores]
    assert policy["feature_source"] == "solver-native"
    assert selected == [True, False]
