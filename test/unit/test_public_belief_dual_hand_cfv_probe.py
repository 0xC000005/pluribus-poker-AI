import numpy as np

from poker_ai.research.belief_probe import N_HANDS
from scripts.eval_public_belief_dual_hand_cfv_probe import (
    _HAND_TO_INDEX,
    _factorize_pair_values,
    _opponent_reach_factors,
    _reconstruct_pair_values,
)


def test_opponent_reach_factors_sum_compatible_opponent_belief():
    hero_hand = _HAND_TO_INDEX[(0, 1)]
    villain_hand = _HAND_TO_INDEX[(2, 3)]
    villain_compatible = _HAND_TO_INDEX[(4, 5)]
    villain_blocked = _HAND_TO_INDEX[(0, 2)]
    hero_blocked = _HAND_TO_INDEX[(2, 4)]

    raw_belief = np.zeros((1, 2 * N_HANDS), dtype=np.float32)
    raw_belief[0, hero_hand] = 0.4
    raw_belief[0, hero_blocked] = 0.6
    raw_belief[0, N_HANDS + villain_hand] = 0.2
    raw_belief[0, N_HANDS + villain_compatible] = 0.5
    raw_belief[0, N_HANDS + villain_blocked] = 0.3

    factors = _opponent_reach_factors(
        raw_belief,
        case_idx=np.asarray([0, 0]),
        hand_idx=np.asarray([hero_hand, villain_hand]),
        player_idx=np.asarray([0, 1]),
    )

    assert np.allclose(factors, [0.7, 0.4])


def test_opponent_reach_ev_factorization_reconstructs_cfv_values():
    values = np.asarray([7.0, -2.0], dtype=np.float32)
    reach = np.asarray([0.7, 0.25], dtype=np.float32)

    factored = _factorize_pair_values(
        values,
        reach,
        value_factorization="opponent-reach-ev",
    )
    reconstructed = _reconstruct_pair_values(
        factored,
        reach,
        value_factorization="opponent-reach-ev",
    )

    assert np.allclose(factored, [10.0, -8.0])
    assert np.allclose(reconstructed, values)
