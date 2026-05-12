import sys
from pathlib import Path

import torch

from scripts.range_tracker import (
    RangeTracker,
    map_slumbot_action_to_idx,
    _parse_action,
    SMALL_BLIND,
    BIG_BLIND,
)


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import (
    action_to_slumbot,
    get_legal_mask_from_parsed,
    parse_action,
)
from solver import StreetSolver


def test_soft_mapping_between_buckets():
    # Preflop, SB acts first. Before string empty → pot=SB+BB=150, street_last_bet_to=100.
    before = ''
    parsed = _parse_action(before)
    acting_pos = 1  # SB
    pot = SMALL_BLIND + BIG_BLIND  # 150
    # Target bet_to so that (bet_to - 100)/150 = 0.375 (midway between 0.25 and 0.5)
    bet_to = 100 + int(round(0.375 * pot))  # 100 + 56 = 156

    mapping = map_slumbot_action_to_idx('b', bet_to, before, acting_pos, parsed)
    # Expect split between indices 2 (0.25x) and 3 (0.5x) with roughly 50/50.
    mapping = sorted(mapping)
    assert mapping[0][0] == 2 and mapping[1][0] == 3
    assert abs(mapping[0][1] + mapping[1][1] - 1.0) < 1e-6
    # Approximately equal weights.
    assert abs(mapping[0][1] - 0.5) < 0.2


def test_low_and_high_mapping_bounds():
    before = ''
    parsed = _parse_action(before)
    acting_pos = 1
    pot = SMALL_BLIND + BIG_BLIND

    # Very small raise_by (close to 0) → bucket 0.25x → idx=2
    bet_to_small = 100 + 1
    m_small = map_slumbot_action_to_idx('b', bet_to_small, before, acting_pos, parsed)
    assert m_small == [(2, 1.0)]

    # Very large raise_by (>> 2.0x pot) → bucket 2.0x → idx=7
    bet_to_large = 100 + int(3.0 * pot)
    m_large = map_slumbot_action_to_idx('b', bet_to_large, before, acting_pos, parsed)
    assert m_large == [(7, 1.0)]


def test_near_all_in_maps_allin():
    # Construct a before string where SB has already bet to some amount.
    # For simplicity, use before='' and pick bet_to that implies bet_from_stack ~ all-in.
    before = ''
    parsed = _parse_action(before)
    acting_pos = 1
    # bet_to equal to our full stack from preflop street bet baseline (SB=50) → all-in
    # our_sb = 50, stack=20000, so bet_to ≈ 50 + stack
    bet_to_allin = 50 + 20000
    m = map_slumbot_action_to_idx('b', bet_to_allin, before, acting_pos, parsed)
    assert m == [(8, 1.0)]


def test_slumbot_mask_excludes_preflop_under_min_raise_bucket():
    parsed = parse_action('')

    mask = get_legal_mask_from_parsed(parsed, '', client_pos=1)

    assert mask[3] == 0.0
    assert mask[4] == 1.0


def test_legal_slumbot_raises_round_trip_to_intended_bucket():
    action_str = ''
    parsed = parse_action(action_str)
    client_pos = 1
    mask = get_legal_mask_from_parsed(parsed, action_str, client_pos)

    for action_idx in range(2, 8):
        if mask[action_idx] <= 0:
            continue
        incr = action_to_slumbot(action_idx, parsed, action_str, client_pos)
        bet_to = int(incr[1:])
        mapped = map_slumbot_action_to_idx(
            'b', bet_to, action_str, client_pos, parsed,
        )
        intended_weight = sum(weight for idx, weight in mapped if idx == action_idx)
        assert intended_weight >= 0.95


def test_range_tracker_maps_positive_hero_mass_to_actual_hand():
    our_cards = [0, 1]
    board = [8, 12, 16, 20]
    net = torch.nn.Linear(126, 9)
    tracker = RangeTracker(our_cards, net, torch.device("cpu"))
    tracker.update_board(board)
    solver = StreetSolver(
        board,
        pot=200,
        hero_stack=19850,
        villain_stack=19950,
        hero_first=True,
    )

    hero_range, villain_range = tracker.get_solver_ranges(
        solver.hands,
        solver.hand_to_idx,
    )
    actual_hand = tuple(sorted(our_cards))
    actual_idx = solver.hand_to_idx[actual_hand]

    assert actual_hand in tracker.hero_hand_to_idx
    assert actual_hand not in tracker.opponent_hand_to_idx
    assert hero_range[actual_idx] > 0.0
    assert villain_range[actual_idx] == 0.0
    assert abs(float(hero_range.sum()) - 1.0) < 1e-9
    assert abs(float(villain_range.sum()) - 1.0) < 1e-9
