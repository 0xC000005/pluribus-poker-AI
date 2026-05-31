import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fast_cfr import build_tree_arrays, get_average_strategy, solve_cfr  # noqa: E402
from solver import Node, StreetSolver  # noqa: E402


def _two_action_tree():
    root = Node(player=0, pot=100, stacks=(1000, 1000), to_call=0, n_raises=0)
    root.children[1] = Node(
        player=-1,
        pot=100,
        stacks=(1000, 1000),
        to_call=0,
        n_raises=0,
        terminal_type="hero_fold",
    )
    root.children[2] = Node(
        player=-1,
        pot=200,
        stacks=(1000, 900),
        to_call=0,
        n_raises=0,
        terminal_type="villain_fold",
    )
    return build_tree_arrays(root)


def _solve_with_hook(iteration_update_fn=None):
    tree = _two_action_tree()
    one = np.ones((1, 1), dtype=np.float32)
    zero = np.zeros((1, 1), dtype=np.float32)
    regrets, strategies = solve_cfr(
        tree,
        1,
        one,
        zero,
        zero,
        one,
        pot_start=100,
        hero_stack_start=1000,
        villain_stack_start=1000,
        n_iterations=2,
        iteration_update_fn=iteration_update_fn,
    )
    return tree, regrets, strategies


def test_iteration_update_fn_modifies_state_consumed_by_next_cfr_iteration():
    updates = []

    def update_hook(**payload):
        updates.append(int(payload["iteration"]))
        regret_sum = payload["regret_sum"].copy()
        regret_sum[0, 1, 0] = 5000.0
        regret_sum[0, 2, 0] = 0.0
        return {"regret_sum": regret_sum}

    plain_tree, _plain_regrets, plain_strategies = _solve_with_hook()
    hooked_tree, hooked_regrets, hooked_strategies = _solve_with_hook(update_hook)

    assert updates == [1, 2]
    plain = get_average_strategy(plain_strategies, 0, [1, 2], 0)
    hooked = get_average_strategy(hooked_strategies, 0, [1, 2], 0)
    assert hooked_tree["n_nodes"] == plain_tree["n_nodes"]
    assert hooked[1] > plain[1]
    assert hooked[1] > 0.45
    assert hooked_regrets[0, 1, 0] >= 5000.0


def test_street_solver_rejects_iteration_update_hook_on_non_cpu_backends():
    solver = StreetSolver([0, 1, 2, 3, 4], pot=200, hero_stack=1000, villain_stack=1000, hero_first=True)

    with pytest.raises(ValueError, match="iteration_update_fn"):
        solver.solve(n_iterations=1, backend="cpu-levelsync", iteration_update_fn=lambda **_: None)
