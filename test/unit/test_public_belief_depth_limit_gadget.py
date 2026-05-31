import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_public_belief_depth_limit_gadget import (  # noqa: E402
    ExactCutTrace,
    ExactCutValueCallback,
    summarize_exact_cut_records,
)


def test_exact_cut_value_callback_replays_iteration_values_in_requested_node_order():
    trace = ExactCutTrace(
        node_indices=np.asarray([10, 3], dtype=np.int32),
        hero_values_by_iter=np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=np.float32,
        ),
        villain_values_by_iter=np.asarray(
            [
                [[-1.0, -2.0], [-3.0, -4.0]],
                [[-5.0, -6.0], [-7.0, -8.0]],
            ],
            dtype=np.float32,
        ),
    )
    callback = ExactCutValueCallback(trace, mode="replay")

    hero_0, villain_0 = callback(cut_indices=np.asarray([3, 10], dtype=np.int32))
    hero_1, villain_1 = callback(cut_indices=np.asarray([10, 3], dtype=np.int32))
    hero_2, villain_2 = callback(cut_indices=np.asarray([3], dtype=np.int32))

    np.testing.assert_allclose(hero_0, [[3.0, 4.0], [1.0, 2.0]])
    np.testing.assert_allclose(villain_0, [[-3.0, -4.0], [-1.0, -2.0]])
    np.testing.assert_allclose(hero_1, [[5.0, 6.0], [7.0, 8.0]])
    np.testing.assert_allclose(villain_1, [[-5.0, -6.0], [-7.0, -8.0]])
    np.testing.assert_allclose(hero_2, [[7.0, 8.0]])
    np.testing.assert_allclose(villain_2, [[-7.0, -8.0]])
    assert callback.calls == 3


def test_exact_cut_value_callback_final_mode_reuses_last_iteration():
    trace = ExactCutTrace(
        node_indices=np.asarray([4], dtype=np.int32),
        hero_values_by_iter=np.asarray([[[1.0]], [[9.0]]], dtype=np.float32),
        villain_values_by_iter=np.asarray([[[-1.0]], [[-9.0]]], dtype=np.float32),
    )
    callback = ExactCutValueCallback(trace, mode="final")

    hero_0, villain_0 = callback(cut_indices=np.asarray([4], dtype=np.int32))
    hero_1, villain_1 = callback(cut_indices=np.asarray([4], dtype=np.int32))

    np.testing.assert_allclose(hero_0, [[9.0]])
    np.testing.assert_allclose(villain_0, [[-9.0]])
    np.testing.assert_allclose(hero_1, [[9.0]])
    np.testing.assert_allclose(villain_1, [[-9.0]])
    assert callback.calls == 2


def test_exact_cut_value_callback_fixed_mode_reuses_selected_iteration():
    trace = ExactCutTrace(
        node_indices=np.asarray([4], dtype=np.int32),
        hero_values_by_iter=np.asarray([[[1.0]], [[5.0]], [[9.0]]], dtype=np.float32),
        villain_values_by_iter=np.asarray([[[-1.0]], [[-5.0]], [[-9.0]]], dtype=np.float32),
    )
    callback = ExactCutValueCallback(trace, mode="fixed", fixed_iteration=1)

    hero_0, villain_0 = callback(cut_indices=np.asarray([4], dtype=np.int32))
    hero_1, villain_1 = callback(cut_indices=np.asarray([4], dtype=np.int32))

    np.testing.assert_allclose(hero_0, [[5.0]])
    np.testing.assert_allclose(villain_0, [[-5.0]])
    np.testing.assert_allclose(hero_1, [[5.0]])
    np.testing.assert_allclose(villain_1, [[-5.0]])
    assert callback.calls == 2


def test_exact_cut_value_callback_fixed_mode_rejects_missing_iteration():
    trace = ExactCutTrace(
        node_indices=np.asarray([1], dtype=np.int32),
        hero_values_by_iter=np.asarray([[[0.0]]], dtype=np.float32),
        villain_values_by_iter=np.asarray([[[0.0]]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="fixed_iteration"):
        ExactCutValueCallback(trace, mode="fixed")


def test_exact_cut_value_callback_rejects_unknown_mode():
    trace = ExactCutTrace(
        node_indices=np.asarray([1], dtype=np.int32),
        hero_values_by_iter=np.asarray([[[0.0]]], dtype=np.float32),
        villain_values_by_iter=np.asarray([[[0.0]]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="unknown exact cut callback mode"):
        ExactCutValueCallback(trace, mode="bad")


def test_summarize_exact_cut_records_groups_passed_modes_only():
    summary = summarize_exact_cut_records(
        [
            {
                "mode_results": {
                    "replay": {
                        "passed": True,
                        "action_l1": 0.2,
                        "action_agreement": True,
                        "solve_ms": 10.0,
                    },
                    "final": {"passed": False, "action_l1": 1.0},
                }
            },
            {
                "mode_results": {
                    "replay": {
                        "passed": True,
                        "action_l1": 0.6,
                        "action_agreement": False,
                        "solve_ms": 30.0,
                    },
                    "final": {
                        "passed": True,
                        "action_l1": 0.4,
                        "action_agreement": True,
                        "solve_ms": 20.0,
                    },
                }
            },
        ]
    )

    assert summary["replay"]["n_roots"] == 2
    assert np.isclose(summary["replay"]["mean_l1"], 0.4)
    assert summary["replay"]["max_l1"] == 0.6
    assert summary["replay"]["action_agreement"] == 0.5
    assert summary["replay"]["mean_solve_ms"] == 20.0
    assert summary["final"]["n_roots"] == 1
    assert summary["final"]["mean_l1"] == 0.4
