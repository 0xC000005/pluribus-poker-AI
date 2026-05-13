import sys
from pathlib import Path

import numpy as np

from poker_ai.research.belief_probe import N_HANDS, _HAND_TO_INDEX
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_dynamic_successor_cut_pbs_targets import (  # noqa: E402
    DynamicTargetCollector,
    _card_to_str,
    _count_value_targets,
    _normalize,
    _normalized_strategy_target,
    _root_policy_target_hands,
)


def test_dynamic_target_normalize_handles_zero_mass():
    values = _normalize(np.zeros(4, dtype=np.float32))

    assert np.allclose(values, np.full(4, 0.25, dtype=np.float32))


def test_dynamic_target_normalize_preserves_distribution():
    values = _normalize(np.asarray([0.0, 2.0, 2.0], dtype=np.float32))

    assert np.allclose(values, [0.0, 0.5, 0.5])


def test_normalized_strategy_target_respects_legal_mask_and_uniform_fallback():
    target = _normalized_strategy_target(
        np.asarray([0.4, 0.6, 0.5], dtype=np.float32),
        np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
    )
    fallback = _normalized_strategy_target(
        np.zeros(3, dtype=np.float32),
        np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
    )

    np.testing.assert_allclose(target, [4.0 / 9.0, 0.0, 5.0 / 9.0])
    np.testing.assert_allclose(fallback, [0.0, 0.5, 0.5])


def test_root_policy_target_hands_support_observed_or_all_hands():
    observed = _root_policy_target_hands(
        [(3, 1), (5, 4)],
        [8, 7],
        include_all_hands=False,
    )
    all_hands = _root_policy_target_hands(
        [(3, 1), (5, 4)],
        [8, 7],
        include_all_hands=True,
    )

    assert observed == [(7, 8)]
    assert all_hands == [(1, 3), (4, 5)]
    assert _card_to_str(0) == "2c"
    assert _card_to_str(51) == "As"


def test_dynamic_target_cut_budget_counts_value_rows_only():
    assert _count_value_targets([0.0, 1.0, 0.0, 1.0, 1.0]) == 2


def test_dynamic_target_collector_exports_denominator_squared_weights():
    case = ResolverBenchmarkCase(
        label="case",
        hole_cards=("Ac", "Kd"),
        board=("2c", "3d", "4h", "5s"),
        action_str="ck/kk/b100",
        client_pos=0,
    )
    hand = (8, 9)
    hand_idx = _HAND_TO_INDEX[tuple(sorted(hand))]
    collector = DynamicTargetCollector(
        case=case,
        board=list(case.board),
        board_idx=[0, 5, 10, 15],
        board_mask=np.ones(N_HANDS, dtype=np.float32),
        solver_hands=[hand],
        records_by_node={
            3: {
                "action_str": "ck/kk/b100",
                "action_shape": "ck/kk/b",
                "bet_count": 1,
                "cut_pos": 0,
            }
        },
        value_scale=10.0,
        max_targets=1,
        value_weight_mode="denominator_squared",
    )

    collector(
        iteration=0,
        node_indices=np.asarray([3], dtype=np.int32),
        hero_reach=np.asarray([[0.2]], dtype=np.float32),
        villain_reach=np.asarray([[0.3]], dtype=np.float32),
        hero_values=np.asarray([[3.0]], dtype=np.float32),
        villain_values=np.asarray([[4.0]], dtype=np.float32),
        valid_m=np.asarray([[1.0]], dtype=np.float32),
    )

    assert collector.hero_masks[0][hand_idx] == 1.0
    assert collector.villain_masks[0][hand_idx] == 1.0
    assert collector.hero_values[0][hand_idx] == np.float32(1.0)
    assert collector.villain_values[0][hand_idx] == np.float32(2.0)
    assert np.isclose(collector.hero_value_weights[0][hand_idx], np.float32(0.09))
    assert np.isclose(collector.villain_value_weights[0][hand_idx], np.float32(0.04))


def test_dynamic_target_collector_defaults_to_mask_value_weights():
    case = ResolverBenchmarkCase(
        label="case",
        hole_cards=("Ac", "Kd"),
        board=("2c", "3d", "4h", "5s"),
        action_str="ck/kk/b100",
        client_pos=0,
    )
    hand = (8, 9)
    hand_idx = _HAND_TO_INDEX[tuple(sorted(hand))]
    collector = DynamicTargetCollector(
        case=case,
        board=list(case.board),
        board_idx=[0, 5, 10, 15],
        board_mask=np.ones(N_HANDS, dtype=np.float32),
        solver_hands=[hand],
        records_by_node={
            3: {
                "action_str": "ck/kk/b100",
                "action_shape": "ck/kk/b",
                "bet_count": 1,
                "cut_pos": 0,
            }
        },
        value_scale=10.0,
        max_targets=1,
    )

    collector(
        iteration=0,
        node_indices=np.asarray([3], dtype=np.int32),
        hero_reach=np.asarray([[0.2]], dtype=np.float32),
        villain_reach=np.asarray([[0.3]], dtype=np.float32),
        hero_values=np.asarray([[3.0]], dtype=np.float32),
        villain_values=np.asarray([[4.0]], dtype=np.float32),
        valid_m=np.asarray([[1.0]], dtype=np.float32),
    )

    assert collector.hero_value_weights[0][hand_idx] == 1.0
    assert collector.villain_value_weights[0][hand_idx] == 1.0
