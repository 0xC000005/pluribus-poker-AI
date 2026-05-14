import sys
from pathlib import Path

import numpy as np
import pytest

from poker_ai.deep_cfr.fast_state import N_ACTIONS

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_regret_policy_warm_start import build_regret_policy_field_warm_start  # noqa: E402
from solver import StreetSolver  # noqa: E402


def test_build_regret_policy_field_warm_start_masks_and_seeds_selected_node_only():
    solver = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    node = solver.root
    node_idx = solver._tree["all_nodes"].index(node)
    predicted_regret = np.zeros((solver.n, N_ACTIONS), dtype=np.float32)
    predicted_strategy = np.zeros((solver.n, N_ACTIONS), dtype=np.float32)
    legal_action = max(node.children)
    illegal_action = min(set(range(N_ACTIONS)) - set(node.children))
    predicted_regret[:, legal_action] = 3.0
    predicted_regret[:, illegal_action] = 99.0
    predicted_strategy[:, legal_action] = 1.5
    predicted_strategy[:, illegal_action] = 42.0

    initial_regret, initial_strategy = build_regret_policy_field_warm_start(
        solver=solver,
        node=node,
        predicted_regret=predicted_regret,
        predicted_strategy=predicted_strategy,
    )

    assert initial_regret.shape == (
        solver._tree["n_nodes"],
        solver._tree["n_actions"],
        solver.n,
    )
    assert initial_regret[node_idx, legal_action, 0] == pytest.approx(3.0)
    assert initial_strategy[node_idx, legal_action, 0] == pytest.approx(1.5)
    assert initial_regret[node_idx, illegal_action, :].sum() == pytest.approx(0.0)
    assert initial_strategy[node_idx, illegal_action, :].sum() == pytest.approx(0.0)
    assert np.count_nonzero(initial_regret[:node_idx]) == 0
    assert np.count_nonzero(initial_regret[node_idx + 1 :]) == 0
    assert np.count_nonzero(initial_strategy[:node_idx]) == 0
    assert np.count_nonzero(initial_strategy[node_idx + 1 :]) == 0
