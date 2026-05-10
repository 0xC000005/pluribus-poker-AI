import numpy as np

from poker_ai.deep_cfr.deep_cfr import get_legal_mask as cpu_get_legal_mask
from poker_ai.games.full_deck.state import (
    ACTION_TO_INDEX,
    INDEX_TO_ACTION,
    RAISE_FRACTIONS as CPU_RAISE_FRACS,
    new_game,
)
from poker_ai.deep_cfr.fast_state import (
    FastPokerState,
    N_ACTIONS,
    RAISE_FRACTIONS as FAST_RAISE_FRACS,
    new_fast_game,
)


def idx_to_action_str(idx: int) -> str:
    if idx == 0:
        return "fold"
    if idx == 1:
        return "call"
    if 2 <= idx <= 7:
        frac = CPU_RAISE_FRACS[idx - 2]
        return f"raise_{frac}"
    if idx == 8:
        return "all_in"
    raise ValueError(f"Invalid idx: {idx}")


def choose_action(mask: np.ndarray) -> int:
    legal = np.where(mask > 0.0)[0]
    # Prefer call if available, else smallest raise, else fold/all-in.
    if 1 in legal:
        return 1
    raises = [a for a in legal if 2 <= a <= 7]
    if raises:
        return min(raises)
    if 0 in legal:
        return 0
    if 8 in legal:
        return 8
    return int(legal[0]) if len(legal) else 1


def test_legal_mask_parity_random_walk():
    # Create parallel games with identical blinds/stacks (card identities differ).
    state_cpu = new_game(2, initial_chips=10000)
    state_fast = new_fast_game(2, initial_chips=10000)

    for _ in range(20):
        # Compare legal masks.
        mask_cpu = cpu_get_legal_mask(state_cpu)
        mask_fast = state_fast.get_legal_mask()
        assert mask_cpu.shape == (N_ACTIONS,)
        assert mask_fast.shape == (N_ACTIONS,)
        np.testing.assert_array_equal((mask_cpu > 0), (mask_fast > 0))

        # Apply the same action to both states.
        action_idx = choose_action(mask_cpu)
        action_str = idx_to_action_str(action_idx)

        # CPU state returns a new immutable state.
        state_cpu = state_cpu.apply_action(action_str)

        # Fast state mutates in place.
        state_fast.apply_action(action_idx)

        if state_fast.is_terminal and state_cpu.is_terminal:
            break


def test_initial_preflop_mask_excludes_under_min_raise_buckets():
    state_cpu = new_game(2, initial_chips=20000)
    state_fast = new_fast_game(2, initial_chips=20000)

    mask_cpu = cpu_get_legal_mask(state_cpu)
    mask_fast = state_fast.get_legal_mask()

    # SB facing the big blind must raise at least to 200 total. The 0.5x-pot
    # bucket would only spend 125 chips from the stack, so exposing it causes
    # Slumbot play to clamp the action into a different bucket.
    assert mask_cpu[3] == 0.0
    assert mask_fast[3] == 0.0
    assert mask_cpu[4] == 1.0
    assert mask_fast[4] == 1.0
