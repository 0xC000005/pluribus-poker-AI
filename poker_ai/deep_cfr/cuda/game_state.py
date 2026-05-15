"""Structure-of-Arrays GPU game state for N simultaneous poker games.

Holds all game state in flat device arrays suitable for Numba CUDA kernels.
Replicates the game logic of fast_state.py (FastPokerState) but for batch
GPU execution — one CUDA thread per game.

Card index convention: card_idx = (rank - 2) * 4 + suit_idx
  rank: 2-14 (2=deuce, ..., 14=ace)
  suit_idx: 0=clubs, 1=diamonds, 2=hearts, 3=spades
"""

import numpy as np
from numba import cuda, int8, int16, int32, int64

from poker_ai.deep_cfr.cuda.lookup_tables import (
    CARD_INDEX_TO_EVAL_CARD,
    get_gpu_tables,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SMALL_BLIND = 50
BIG_BLIND = 100
INITIAL_CHIPS = 10000

# Stage constants (must match FastPokerState).
PREFLOP = 0
FLOP = 1
TURN = 2
RIVER = 3
SHOWDOWN = 4
TERMINAL = 5

# Maximum players supported.
MAX_PLAYERS = 6

# ---------------------------------------------------------------------------
# Player order device arrays (module-level, uploaded once)
# ---------------------------------------------------------------------------

# 6-player orders.
_PREFLOP_ORDER_6 = np.array([2, 3, 4, 5, 0, 1], dtype=np.int8)
_POSTFLOP_ORDER_6 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int8)

# 2-player orders.
_PREFLOP_ORDER_2 = np.array([0, 1], dtype=np.int8)
_POSTFLOP_ORDER_2 = np.array([1, 0], dtype=np.int8)  # BB first postflop in HU

# Lazy GPU copies.
_d_orders = None
_d_card_eval_lookup = None


def _get_device_orders():
    """Return device arrays for player orders. Cached after first call."""
    global _d_orders
    if _d_orders is None:
        _d_orders = {
            2: (cuda.to_device(_PREFLOP_ORDER_2),
                cuda.to_device(_POSTFLOP_ORDER_2)),
            6: (cuda.to_device(_PREFLOP_ORDER_6),
                cuda.to_device(_POSTFLOP_ORDER_6)),
        }
    return _d_orders


def get_card_eval_lookup():
    """Return CARD_INDEX_TO_EVAL_CARD[52] as a GPU device array. Cached."""
    global _d_card_eval_lookup
    if _d_card_eval_lookup is None:
        _d_card_eval_lookup = cuda.to_device(CARD_INDEX_TO_EVAL_CARD)
    return _d_card_eval_lookup


# ---------------------------------------------------------------------------
# xorshift128+ RNG (device function)
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def xorshift128plus(s0, s1):
    """One step of xorshift128+. Returns (result, new_s0, new_s1)."""
    s1 ^= s1 << 23
    s1 ^= s1 >> 17
    s1 ^= s0
    s1 ^= s0 >> 26
    return s0 + s1, s0, s1


# ---------------------------------------------------------------------------
# GameBatch — SoA container for N games
# ---------------------------------------------------------------------------

class GameBatch:
    """Structure-of-Arrays device storage for N simultaneous poker games."""

    def __init__(self, n_games: int, n_players: int):
        self.n_games = n_games
        self.n_players = n_players

        # Per-player arrays.
        self.chips = cuda.device_array((n_games, n_players), dtype=np.int32)
        self.bets = cuda.device_array((n_games, n_players), dtype=np.int32)
        self.active = cuda.device_array((n_games, n_players), dtype=np.int8)
        self.hole_cards = cuda.device_array((n_games, n_players, 2), dtype=np.int8)
        self.payout = cuda.device_array((n_games, n_players), dtype=np.int32)

        # Community and deck.
        self.community = cuda.device_array((n_games, 5), dtype=np.int8)
        self.deck = cuda.device_array((n_games, 52), dtype=np.int8)
        self.deck_cursor = cuda.device_array(n_games, dtype=np.int32)

        # Scalar per-game state.
        self.stage = cuda.device_array(n_games, dtype=np.int8)
        self.n_raises = cuda.device_array(n_games, dtype=np.int8)
        self.player_i_index = cuda.device_array(n_games, dtype=np.int8)
        self.n_actions = cuda.device_array(n_games, dtype=np.int16)
        self.pot_total = cuda.device_array(n_games, dtype=np.int32)
        self.n_players_started_round = cuda.device_array(n_games, dtype=np.int8)

        # Per-round action history: (N, 4 rounds, 3 types: calls/raises/folds).
        self.history = cuda.device_array((n_games, 4, 3), dtype=np.int8)

        # Game-finished flag (1 = done, 0 = still playing).
        self.is_done = cuda.device_array(n_games, dtype=np.int8)


# ---------------------------------------------------------------------------
# init_games_kernel — initialize N games in parallel
# ---------------------------------------------------------------------------

@cuda.jit
def init_games_kernel(
    chips, bets, active, hole_cards, community, deck, deck_cursor,
    stage, n_raises, player_i_index, n_actions, pot_total,
    n_players_started_round, history, payout, is_done,
    seeds, n_players, n_games,
    preflop_order, initial_chips,
):
    """Initialize game i with thread i.

    Parameters
    ----------
    seeds : (n_games, 2) int64 device array — per-game RNG seeds
    preflop_order : (n_players,) int8 device array
    """
    i = cuda.grid(1)
    if i >= n_games:
        return

    # --- RNG state from seed pair ---
    s0 = seeds[i, 0]
    s1 = seeds[i, 1]

    # --- Initialize deck as [0, 1, ..., 51] then Fisher-Yates shuffle ---
    for c in range(52):
        deck[i, c] = int8(c)

    for c in range(51, 0, -1):
        rnd, s0, s1 = xorshift128plus(s0, s1)
        # Map to [0, c] using unsigned modulo.
        j = int32(rnd & 0x7FFFFFFFFFFFFFFF) % int32(c + 1)
        tmp = deck[i, c]
        deck[i, c] = deck[i, j]
        deck[i, j] = tmp

    # --- Deal hole cards ---
    cursor = int32(0)
    for p in range(n_players):
        for ci in range(2):
            hole_cards[i, p, ci] = deck[i, cursor]
            cursor += 1
    deck_cursor[i] = cursor

    # --- Community cards: all undealt ---
    for c in range(5):
        community[i, c] = int8(-1)

    # --- Per-player state ---
    for p in range(n_players):
        chips[i, p] = int32(initial_chips)
        bets[i, p] = int32(0)
        active[i, p] = int8(1)
        payout[i, p] = int32(0)

    # --- Post blinds ---
    sb = int32(SMALL_BLIND)
    if chips[i, 0] < sb:
        sb = chips[i, 0]
    chips[i, 0] -= sb
    bets[i, 0] = sb

    bb = int32(BIG_BLIND)
    if chips[i, 1] < bb:
        bb = chips[i, 1]
    chips[i, 1] -= bb
    bets[i, 1] = bb

    pot_total[i] = sb + bb

    # --- Scalar state ---
    stage[i] = int8(PREFLOP)
    n_raises[i] = int8(0)
    n_actions[i] = int16(0)
    n_started = int8(0)
    for p in range(n_players):
        if active[i, p] == int8(1) and chips[i, p] > 0:
            n_started += 1
    n_players_started_round[i] = n_started
    is_done[i] = int8(0)

    # --- History: zero ---
    for r in range(4):
        for a in range(3):
            history[i, r, a] = int8(0)

    # --- Skip to first active player in preflop order ---
    # For standard games all players are active at start, so first in order.
    player_i_index[i] = int8(0)
    for idx in range(n_players):
        p = preflop_order[idx]
        if active[i, p] == int8(1) and chips[i, p] > 0:
            player_i_index[i] = int8(idx)
            break


# ---------------------------------------------------------------------------
# copy_game_kernel — fork game states for tree traversal
# ---------------------------------------------------------------------------

@cuda.jit
def copy_game_kernel(
    # Source batch arrays
    src_chips, src_bets, src_active, src_hole_cards, src_community,
    src_deck, src_deck_cursor, src_stage, src_n_raises,
    src_player_i_index, src_n_actions, src_pot_total,
    src_n_players_started_round, src_history, src_payout, src_is_done,
    # Destination batch arrays
    dst_chips, dst_bets, dst_active, dst_hole_cards, dst_community,
    dst_deck, dst_deck_cursor, dst_stage, dst_n_raises,
    dst_player_i_index, dst_n_actions, dst_pot_total,
    dst_n_players_started_round, dst_history, dst_payout, dst_is_done,
    # Index mapping
    src_indices, dst_indices, n_copies, n_players,
):
    """Copy game src_indices[tid] from src_batch to dst_indices[tid] in dst_batch."""
    tid = cuda.grid(1)
    if tid >= n_copies:
        return

    si = src_indices[tid]
    di = dst_indices[tid]

    # Per-player arrays.
    for p in range(n_players):
        dst_chips[di, p] = src_chips[si, p]
        dst_bets[di, p] = src_bets[si, p]
        dst_active[di, p] = src_active[si, p]
        dst_payout[di, p] = src_payout[si, p]
        for c in range(2):
            dst_hole_cards[di, p, c] = src_hole_cards[si, p, c]

    # Community cards.
    for c in range(5):
        dst_community[di, c] = src_community[si, c]

    # Deck.
    for c in range(52):
        dst_deck[di, c] = src_deck[si, c]

    # Scalar state.
    dst_deck_cursor[di] = src_deck_cursor[si]
    dst_stage[di] = src_stage[si]
    dst_n_raises[di] = src_n_raises[si]
    dst_player_i_index[di] = src_player_i_index[si]
    dst_n_actions[di] = src_n_actions[si]
    dst_pot_total[di] = src_pot_total[si]
    dst_n_players_started_round[di] = src_n_players_started_round[si]
    dst_is_done[di] = src_is_done[si]

    # History.
    for r in range(4):
        for a in range(3):
            dst_history[di, r, a] = src_history[si, r, a]


# ---------------------------------------------------------------------------
# Convenience: create_game_batch
# ---------------------------------------------------------------------------

def create_game_batch(n_games: int, n_players: int, initial_chips: int = 10000) -> GameBatch:
    """Create and initialize a batch of N poker games on GPU.

    Seeds are generated from numpy RNG. Returns a GameBatch with all
    games set to initial preflop state.
    """
    batch = GameBatch(n_games, n_players)

    # Generate per-game seeds on host, upload to device.
    rng = np.random.default_rng()
    seeds = rng.integers(1, 2**62, size=(n_games, 2), dtype=np.int64)
    d_seeds = cuda.to_device(seeds)

    # Get player order device array.
    orders = _get_device_orders()
    if n_players not in orders:
        # Build on the fly for uncommon player counts.
        preflop = np.array(
            list(range(2, n_players)) + [0, 1], dtype=np.int8
        )
        d_preflop = cuda.to_device(preflop)
    else:
        d_preflop = orders[n_players][0]

    # Launch kernel: one thread per game.
    threads_per_block = 256
    blocks = (n_games + threads_per_block - 1) // threads_per_block

    init_games_kernel[blocks, threads_per_block](
        batch.chips, batch.bets, batch.active, batch.hole_cards,
        batch.community, batch.deck, batch.deck_cursor,
        batch.stage, batch.n_raises, batch.player_i_index,
        batch.n_actions, batch.pot_total,
        batch.n_players_started_round, batch.history, batch.payout,
        batch.is_done,
        d_seeds, n_players, n_games,
        d_preflop, initial_chips,
    )

    return batch
