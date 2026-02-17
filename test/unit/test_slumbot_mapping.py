from scripts.range_tracker import (
    map_slumbot_action_to_idx,
    _parse_action,
    SMALL_BLIND,
    BIG_BLIND,
)


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
