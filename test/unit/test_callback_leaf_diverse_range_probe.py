import json
import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_callback_leaf_diverse_range_probe import (  # noqa: E402
    aggregate_value_set_predictions,
    load_high_margin_flip_records,
    perturb_public_belief,
    summarize_diverse_range_records,
    summarize_integrated_value_sets,
)
from poker_ai.research.belief_probe import N_HANDS  # noqa: E402


def test_perturb_public_belief_preserves_normalized_support():
    belief = np.zeros(2 * N_HANDS, dtype=np.float32)
    belief[0] = 2.0
    belief[1] = 1.0
    belief[N_HANDS + 10] = 0.8
    belief[N_HANDS + 11] = 0.2

    mixed = perturb_public_belief(belief, "villain_uniform_mix_0.50")
    sharp = perturb_public_belief(belief, "villain_temp_2.0")

    assert np.isclose(float(mixed[:N_HANDS].sum()), 1.0)
    assert np.isclose(float(mixed[N_HANDS:].sum()), 1.0)
    assert np.isclose(float(sharp[N_HANDS:].sum()), 1.0)
    assert np.count_nonzero(mixed[N_HANDS:]) == 2
    assert np.count_nonzero(sharp[N_HANDS:]) == 2
    assert mixed[N_HANDS + 12] == 0.0


def test_load_high_margin_flip_records_selects_worst_flips(tmp_path):
    payload = {
        "start_index": 40,
        "records": [
            {
                "label": "low-margin",
                "passed": True,
                "action_agreement": False,
                "action_l1_drift": 1.9,
                "baseline_strategy": [0.51, 0.49],
            },
            {
                "label": "high-margin-small",
                "passed": True,
                "action_agreement": False,
                "action_l1_drift": 0.8,
                "baseline_strategy": [0.9, 0.1],
            },
            {
                "label": "high-margin-large",
                "passed": True,
                "action_agreement": False,
                "action_l1_drift": 1.2,
                "baseline_strategy": [0.85, 0.15],
            },
            {
                "label": "agrees",
                "passed": True,
                "action_agreement": True,
                "action_l1_drift": 2.0,
                "baseline_strategy": [0.9, 0.1],
            },
        ],
    }
    path = tmp_path / "leaf_ab.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    selected = load_high_margin_flip_records([path], min_margin=0.5, limit=2)

    assert [row["label"] for row in selected] == [
        "high-margin-large",
        "high-margin-small",
    ]
    assert [row["case_index"] for row in selected] == [42, 41]


def test_summarize_diverse_range_records_reports_action_agreement_rates():
    metrics = summarize_diverse_range_records(
        [
            {
                "baseline_action": 1,
                "original_learned_action": 2,
                "original_l1": 1.0,
                "range_avg_agree": True,
                "range_avg_l1": 0.4,
                "best_variant_agree": True,
                "best_variant_l1": 0.2,
                "plurality_action_agree": False,
            },
            {
                "baseline_action": 0,
                "original_learned_action": 0,
                "original_l1": 0.5,
                "range_avg_agree": False,
                "range_avg_l1": 0.6,
                "best_variant_agree": True,
                "best_variant_l1": 0.1,
                "plurality_action_agree": True,
            },
        ]
    )

    assert metrics["n_roots"] == 2
    assert metrics["original_mean_l1"] == 0.75
    assert metrics["range_avg_action_agreement"] == 0.5
    assert metrics["best_variant_oracle_action_agreement"] == 1.0
    assert metrics["plurality_action_agreement"] == 0.5


def test_aggregate_value_set_predictions_supports_mean_and_pessimistic_modes():
    predictions = np.zeros((2, 2, 1, N_HANDS), dtype=np.float32)
    predictions[0, 0, 0, 3] = 2.0
    predictions[1, 0, 0, 3] = -4.0
    predictions[0, 1, 0, 5] = 1.0
    predictions[1, 1, 0, 5] = 7.0

    mean = aggregate_value_set_predictions(predictions, "mean")
    pessimistic = aggregate_value_set_predictions(predictions, "hero_pessimistic")

    assert mean.shape == (2, 1, N_HANDS)
    assert mean[0, 0, 3] == -1.0
    assert mean[1, 0, 5] == 4.0
    assert pessimistic[0, 0, 3] == -4.0
    assert pessimistic[1, 0, 5] == 7.0


def test_aggregate_value_set_predictions_villain_best_response_chooses_consistent_variant():
    predictions = np.zeros((2, 2, 1, N_HANDS), dtype=np.float32)
    predictions[0, 0, 0, 2] = 3.0
    predictions[0, 1, 0, 10] = 1.0
    predictions[1, 0, 0, 2] = -5.0
    predictions[1, 1, 0, 10] = 9.0
    belief = np.zeros((1, 2 * N_HANDS), dtype=np.float32)
    belief[0, 0] = 1.0
    belief[0, N_HANDS + 10] = 1.0

    chosen = aggregate_value_set_predictions(
        predictions,
        "villain_best_response",
        belief=belief,
    )

    assert chosen[0, 0, 2] == -5.0
    assert chosen[1, 0, 10] == 9.0


def test_summarize_integrated_value_sets_groups_passed_modes_only():
    summary = summarize_integrated_value_sets(
        [
            {
                "integrated_value_sets": [
                    {
                        "aggregation": "mean",
                        "passed": True,
                        "l1": 0.4,
                        "action_agree": True,
                        "learned_solve_ms": 10.0,
                        "leaf_prediction_ms": 2.0,
                    },
                    {
                        "aggregation": "mean",
                        "passed": False,
                        "error": "skipped",
                    },
                ]
            },
            {
                "integrated_value_sets": [
                    {
                        "aggregation": "mean",
                        "passed": True,
                        "l1": 0.8,
                        "action_agree": False,
                        "learned_solve_ms": 20.0,
                        "leaf_prediction_ms": 4.0,
                    }
                ]
            },
        ]
    )

    assert summary["mean"]["n_roots"] == 2
    assert np.isclose(summary["mean"]["mean_l1"], 0.6)
    assert summary["mean"]["action_agreement"] == 0.5
    assert summary["mean"]["mean_solve_ms"] == 15.0
