import sys
from pathlib import Path

import numpy as np
import pytest

from poker_ai.deep_cfr.fast_state import N_ACTIONS

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_regret_policy_warm_start import (  # noqa: E402
    _apply_uniform_budget_baseline,
    build_regret_policy_field_warm_start,
)
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


def test_uniform_budget_baseline_rejects_warm_start_that_only_beats_low_budget():
    summary = {
        "passed": True,
        "mean_warm_l1_to_reference": 0.4,
        "mean_warm_kl_to_reference": 0.2,
        "warm_action_agreement": 0.5,
        "warm_allin_gap": 0.0,
        "warm_allin_prob_gap": 0.0,
    }
    records = [
        {
            "passed": True,
            "baseline_l1_to_reference": 0.3,
            "baseline_kl_to_reference": 0.1,
            "baseline_latency_ms": 20.0,
            "baseline_allin_selected": False,
            "reference_allin_selected": False,
            "baseline_allin_prob": 0.05,
            "reference_allin_prob": 0.05,
            "baseline_agrees_with_reference": True,
        }
    ]

    updated = _apply_uniform_budget_baseline(summary, records, baseline_iterations=10)

    assert updated["baseline_iterations"] == 10
    assert updated["mean_baseline_l1_to_reference"] == pytest.approx(0.3)
    assert not updated["warm_beats_uniform_budget"]
    assert not updated["passed"]
