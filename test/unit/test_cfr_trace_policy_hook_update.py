import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_cfr_trace_policy_hook_update import apply_policy_update_to_cfr_state  # noqa: E402


def test_apply_policy_update_changes_only_selected_node_and_masks_illegal_actions():
    regret_sum = np.ones((3, 4, 2), dtype=np.float32)
    strategy_sum = np.ones((3, 4, 2), dtype=np.float32) * 2.0

    update = apply_policy_update_to_cfr_state(
        regret_sum=regret_sum,
        strategy_sum=strategy_sum,
        node_idx=1,
        legal_actions=[0, 2],
        target_policy=np.asarray([0.25, 0.25, 0.75, 0.0], dtype=np.float32),
        regret_mass=4.0,
        strategy_mass=8.0,
    )

    updated_regret = update["regret_sum"]
    updated_strategy = update["strategy_sum"]

    np.testing.assert_array_equal(updated_regret[0], regret_sum[0])
    np.testing.assert_array_equal(updated_regret[2], regret_sum[2])
    np.testing.assert_array_equal(updated_strategy[0], strategy_sum[0])
    np.testing.assert_array_equal(updated_strategy[2], strategy_sum[2])
    np.testing.assert_allclose(updated_regret[1, :, 0], [1.0, 0.0, 3.0, 0.0])
    np.testing.assert_allclose(updated_regret[1, :, 1], [1.0, 0.0, 3.0, 0.0])
    np.testing.assert_allclose(updated_strategy[1, :, 0], [2.0, 0.0, 6.0, 0.0])
    np.testing.assert_allclose(updated_strategy[1, :, 1], [2.0, 0.0, 6.0, 0.0])


def test_apply_policy_update_falls_back_to_legal_uniform_when_prediction_is_invalid():
    regret_sum = np.zeros((1, 3, 1), dtype=np.float32)
    strategy_sum = np.zeros((1, 3, 1), dtype=np.float32)

    update = apply_policy_update_to_cfr_state(
        regret_sum=regret_sum,
        strategy_sum=strategy_sum,
        node_idx=0,
        legal_actions=[1, 2],
        target_policy=np.asarray([np.nan, -1.0, 0.0], dtype=np.float32),
        regret_mass=2.0,
        strategy_mass=4.0,
    )

    np.testing.assert_allclose(update["regret_sum"][0, :, 0], [0.0, 1.0, 1.0])
    np.testing.assert_allclose(update["strategy_sum"][0, :, 0], [0.0, 2.0, 2.0])
