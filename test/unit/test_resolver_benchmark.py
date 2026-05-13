import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES
from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.research.belief_probe import N_HANDS, _HAND_TO_INDEX
from poker_ai.research.resolver_benchmark import (
    ResolverBenchmarkCase,
    run_resolver_benchmark,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_learned_river_leaf_resolver_ab import (  # noqa: E402
    _action_path_to_street_string,
    _global_board_mask,
    _local_ranges_from_belief,
)
from build_learned_river_leaf_cases import (  # noqa: E402
    _rotated_cards,
    _summarize_leaf_records,
)
from solver import StreetSolver  # noqa: E402


def _small_checkpoint(path: Path) -> None:
    checkpoint = {
        "iteration": 9,
        "n_players": 2,
        "hidden_dim": 16,
        "n_layers": 1,
        "initial_chips": 20000,
        "value_net": {
            "net.0.weight": torch.randn(16, N_FEATURES),
            "net.0.bias": torch.randn(16),
            "net.2.weight": torch.randn(N_ACTIONS, 16),
            "net.2.bias": torch.randn(N_ACTIONS),
        },
    }
    torch.save(checkpoint, path)


def test_resolver_benchmark_reports_fixed_state_policy_and_solver_metrics():
    torch.manual_seed(0)
    value_net = ValueNetwork(N_FEATURES, 16, N_ACTIONS, n_layers=1)
    case = ResolverBenchmarkCase(
        label="river-check",
        hole_cards=("Ac", "Kd"),
        board=("2c", "7d", "Jh", "4s", "9c"),
        action_str="ck/kk/kk/",
        client_pos=0,
    )

    metrics = run_resolver_benchmark(
        value_net,
        torch.device("cpu"),
        cases=[case],
        solver_iterations=1,
    )

    assert metrics["passed"] is True
    assert metrics["n_cases"] == 1
    assert metrics["n_solver_cases"] == 1
    result = metrics["cases"][0]
    assert result["label"] == "river-check"
    assert 0 <= result["blueprint_action"] < N_ACTIONS
    assert 0 <= result["solver_action"] < N_ACTIONS
    assert result["blueprint_action_legal"] is True
    assert 0 <= result["blueprint_no_allin_action"] < N_ACTIONS
    assert result["blueprint_no_allin_action_legal"] is True
    assert isinstance(result["blueprint_allin_selected"], bool)
    assert isinstance(result["allin_removed_action_changed"], bool)
    assert 0 <= result["policy_head_action"] < N_ACTIONS
    assert result["policy_head_action_legal"] is True
    assert isinstance(result["policy_head_allin_selected"], bool)
    assert 0.0 <= result["policy_head_action_l1_drift"] <= 2.0
    assert result["solver_action_legal"] is True
    assert result["solver_latency_ms"] >= 0.0
    assert 0.0 <= result["action_l1_drift"] <= 2.0
    assert isinstance(result["advantage_delta_proxy"], float)
    assert "blueprint_allin_rate" in metrics
    assert "no_allin_changed_rate" in metrics
    assert "policy_head_allin_rate" in metrics
    assert "policy_head_mean_action_l1_drift" in metrics


def test_street_solver_omits_under_minimum_raise_buckets():
    solver = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )

    assert 2 not in solver.root.children
    assert 3 in solver.root.children


def test_street_solver_showdown_leaf_callback_can_reproduce_default():
    base = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    hooked = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    calls = []

    def passthrough_leaf(**kwargs):
        calls.append(kwargs["showdown_indices"].shape[0])
        return kwargs["default_hero_values"], kwargs["default_villain_values"]

    base.solve(n_iterations=2)
    hooked.solve(n_iterations=2, showdown_leaf_fn=passthrough_leaf)

    assert calls
    np.testing.assert_allclose(hooked._regret_sum, base._regret_sum, atol=1e-5)
    np.testing.assert_allclose(hooked._strategy_sum, base._strategy_sum, atol=1e-5)


def test_street_solver_showdown_leaf_callback_rejects_torch_backend():
    solver = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )

    with pytest.raises(ValueError, match="CPU CFR backend"):
        solver.solve(n_iterations=1, backend="torch-cpu", showdown_leaf_fn=lambda **_: None)


def test_learned_leaf_action_path_reconstructs_turn_sequence():
    solver = StreetSolver(
        board=[0, 1, 2, 3],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    node = solver.root.children[1].children[1]
    node_idx = solver._tree["all_nodes"].index(node)

    street = _action_path_to_street_string(
        solver._tree,
        node_idx,
        hero_stack_start=solver.hero_stack_start,
        villain_stack_start=solver.villain_stack_start,
    )

    assert street == "kk"


def test_learned_leaf_range_and_mask_helpers_normalize_legal_hands():
    solver_hands = [(0, 1), (4, 5)]
    belief = np.zeros(2 * N_HANDS, dtype=np.float32)
    belief[_HAND_TO_INDEX[(0, 1)]] = 2.0
    belief[N_HANDS + _HAND_TO_INDEX[(4, 5)]] = 3.0

    hero, villain = _local_ranges_from_belief(belief, solver_hands)
    board_mask = _global_board_mask([0, 1, 2, 3, 4])

    np.testing.assert_allclose(hero, [1.0, 0.0])
    np.testing.assert_allclose(villain, [0.0, 1.0])
    assert int(board_mask.sum()) == 47 * 46 // 2


def test_river_leaf_record_summary_reports_dataset_skew():
    records = [
        {
            "source_case": "turn-a",
            "leaf_action_str": "ck/kk/kk/",
            "river_card": "2c",
            "terminal_node_idx": 7,
        },
        {
            "source_case": "turn-a",
            "leaf_action_str": "ck/kk/kk/",
            "river_card": "3d",
            "terminal_node_idx": 7,
        },
        {
            "source_case": "turn-b",
            "leaf_action_str": "ck/kk/b400",
            "river_card": "2c",
            "terminal_node_idx": 9,
        },
    ]

    summary = _summarize_leaf_records(records)

    assert summary["sources"]["n_unique"] == 2
    assert summary["sources"]["max_share"] == pytest.approx(2 / 3)
    assert summary["terminals"]["n_unique"] == 2
    assert summary["river_cards"]["top"][0] == {"key": "2c", "count": 2}
    assert summary["leaf_action_parse_errors"] == 0
    assert summary["leaf_total_last_bet_to"]["max"] == 500.0


def test_river_leaf_card_rotation_is_deterministic_and_keyed():
    cards = [1, 2, 3, 4, 5]

    first = _rotated_cards(cards, key="seed:a")
    second = _rotated_cards(cards, key="seed:a")
    other = _rotated_cards(cards, key="seed:b")

    assert first == second
    assert sorted(first) == cards
    assert sorted(other) == cards
    assert first != cards or other != cards


def test_resolver_benchmark_cli_emits_json_for_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "candidate.pt"
    _small_checkpoint(checkpoint_path)
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_resolver_benchmark.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint_path),
            "--device",
            "cpu",
            "--max-cases",
            "1",
            "--solver-iterations",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    metrics = json.loads(result.stdout)
    assert metrics["passed"] is True
    assert metrics["checkpoint_iteration"] == 9
    assert metrics["n_cases"] == 1
    assert metrics["promotion_blockers"] == [
        "fixed_state_resolver_benchmark_is_not_slumbot_confidence"
    ]
