import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_bounded_regret_residual_hook import apply_bounded_residual_field_update  # noqa: E402


def test_bounded_residual_field_update_masks_and_caps_selected_node():
    regret = np.zeros((2, 4, 2), dtype=np.float32)
    strategy = np.zeros_like(regret)
    regret[0, 1, :] = [10.0, 20.0]
    regret[1, 1, :] = [3.0, 3.0]
    strategy[0, 1, :] = [5.0, 5.0]
    predicted_regret = np.array(
        [
            [0.0, 100.0, 100.0, 100.0],
            [0.0, 200.0, 200.0, 200.0],
        ],
        dtype=np.float32,
    )
    predicted_strategy = np.array(
        [
            [0.0, 50.0, 50.0, 50.0],
            [0.0, 60.0, 60.0, 60.0],
        ],
        dtype=np.float32,
    )

    update = apply_bounded_residual_field_update(
        regret_sum=regret,
        strategy_sum=strategy,
        node_idx=0,
        legal_actions=[1, 2],
        predicted_regret=predicted_regret,
        predicted_strategy=predicted_strategy,
        residual_l1_fraction=0.25,
        residual_l1_floor=4.0,
    )

    updated_regret = update["regret_sum"]
    updated_strategy = update["strategy_sum"]
    assert updated_regret[1].tolist() == regret[1].tolist()
    assert updated_regret[0, 0].tolist() == [0.0, 0.0]
    assert updated_regret[0, 3].tolist() == [0.0, 0.0]
    assert updated_strategy[0, 0].tolist() == [0.0, 0.0]
    assert updated_strategy[0, 3].tolist() == [0.0, 0.0]
    assert float(np.abs(updated_regret[0, :, 0] - regret[0, :, 0]).sum()) <= 4.0001
    assert float(np.abs(updated_regret[0, :, 1] - regret[0, :, 1]).sum()) <= 5.0001
    assert updated_regret[0, 1, 0] > regret[0, 1, 0]
    assert updated_regret[0, 2, 0] > 0.0
