import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_regret_policy_iteration_hook import apply_regret_policy_field_update  # noqa: E402


def test_apply_regret_policy_field_update_replaces_selected_node_only():
    regret_sum = np.ones((2, 4, 3), dtype=np.float32)
    strategy_sum = np.ones((2, 4, 3), dtype=np.float32) * 2.0
    predicted_regret = np.zeros((3, 4), dtype=np.float32)
    predicted_strategy = np.zeros((3, 4), dtype=np.float32)
    predicted_regret[:, 1] = 5.0
    predicted_regret[:, 3] = 9.0
    predicted_strategy[:, 1] = 7.0
    predicted_strategy[:, 3] = 11.0

    update = apply_regret_policy_field_update(
        regret_sum=regret_sum,
        strategy_sum=strategy_sum,
        node_idx=1,
        legal_actions=[1, 3],
        predicted_regret=predicted_regret,
        predicted_strategy=predicted_strategy,
    )

    np.testing.assert_array_equal(update["regret_sum"][0], regret_sum[0])
    np.testing.assert_array_equal(update["strategy_sum"][0], strategy_sum[0])
    np.testing.assert_allclose(update["regret_sum"][1, 1, :], [5.0, 5.0, 5.0])
    np.testing.assert_allclose(update["regret_sum"][1, 3, :], [9.0, 9.0, 9.0])
    np.testing.assert_allclose(update["strategy_sum"][1, 1, :], [7.0, 7.0, 7.0])
    np.testing.assert_allclose(update["strategy_sum"][1, 3, :], [11.0, 11.0, 11.0])
    assert update["regret_sum"][1, 0, :].sum() == pytest.approx(0.0)
    assert update["strategy_sum"][1, 2, :].sum() == pytest.approx(0.0)


def test_apply_regret_policy_field_update_rejects_bad_hand_shape():
    regret_sum = np.zeros((1, 4, 3), dtype=np.float32)
    strategy_sum = np.zeros((1, 4, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="predicted_regret shape"):
        apply_regret_policy_field_update(
            regret_sum=regret_sum,
            strategy_sum=strategy_sum,
            node_idx=0,
            legal_actions=[1],
            predicted_regret=np.zeros((2, 4), dtype=np.float32),
            predicted_strategy=np.zeros((3, 4), dtype=np.float32),
        )
