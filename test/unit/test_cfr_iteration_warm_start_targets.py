import sys
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_cfr_iteration_warm_start_targets import materialize_iteration_target_rows  # noqa: E402
from poker_ai.deep_cfr.fast_state import N_ACTIONS  # noqa: E402


def test_materialize_iteration_target_rows_exports_current_state_with_reference_targets():
    public_feature = np.arange(N_ACTIONS, dtype=np.float32)
    belief = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    policy_features = np.stack(
        [
            np.full(N_ACTIONS, 10.0, dtype=np.float32),
            np.full(N_ACTIONS, 20.0, dtype=np.float32),
        ],
        axis=0,
    )
    legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)
    legal_mask[[1, 2]] = 1.0
    current_regret_by_iteration = {
        0: np.zeros((2, N_ACTIONS), dtype=np.float32),
        2: np.zeros((2, N_ACTIONS), dtype=np.float32),
    }
    current_regret_by_iteration[0][:, [1, 2]] = [[1, 2], [3, 4]]
    current_regret_by_iteration[2][:, [1, 2]] = [[5, 6], [7, 8]]
    current_strategy_by_iteration = {
        0: np.zeros((2, N_ACTIONS), dtype=np.float32),
        2: np.zeros((2, N_ACTIONS), dtype=np.float32),
    }
    current_strategy_by_iteration[0][:, [1, 2]] = [[2, 6], [1, 1]]
    current_strategy_by_iteration[2][:, [1, 2]] = [[4, 4], [3, 9]]
    target_regret = np.zeros((2, N_ACTIONS), dtype=np.float32)
    target_regret[:, [1, 2]] = [[9, 1], [2, 8]]
    target_strategy = np.zeros((2, N_ACTIONS), dtype=np.float32)
    target_strategy[:, [1, 2]] = [[3, 1], [5, 15]]

    rows = materialize_iteration_target_rows(
        root_label="root-a",
        public_feature=public_feature,
        policy_features=policy_features,
        belief=belief,
        legal_mask=legal_mask,
        current_regret_by_iteration=current_regret_by_iteration,
        current_strategy_by_iteration=current_strategy_by_iteration,
        target_regret=target_regret,
        target_strategy=target_strategy,
        hand_indices=np.array([11, 22], dtype=np.int32),
    )

    assert rows["features"].shape == (4, N_ACTIONS)
    assert rows["belief"].shape == (4, 6)
    assert rows["legal_masks"].shape == (4, N_ACTIONS)
    assert rows["low_regret_sum"][0, [1, 2]].tolist() == [1.0, 2.0]
    assert rows["low_regret_sum"][2, [1, 2]].tolist() == [5.0, 6.0]
    assert rows["target_regret_sum"][0, [1, 2]].tolist() == [9.0, 1.0]
    assert rows["target_strategy_sum"][3, [1, 2]].tolist() == [5.0, 15.0]
    assert np.allclose(rows["target_probs"][0, [1, 2]], [0.75, 0.25])
    assert float(rows["target_probs"][0, [0, 3, 4, 5, 6, 7, 8]].sum()) == 0.0
    assert rows["root_labels"].tolist() == ["root-a"] * 4
    assert rows["labels"].tolist() == [
        "root-a-iter000-hand0011",
        "root-a-iter000-hand0022",
        "root-a-iter002-hand0011",
        "root-a-iter002-hand0022",
    ]
