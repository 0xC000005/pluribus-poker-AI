"""CUDA kernels for poker game logic.

Ports the game logic from fast_state.py to GPU kernels operating on
GameBatch Structure-of-Arrays.  One thread handles one game.

All kernels expect GameBatch device arrays passed individually
(Numba CUDA cannot pass Python objects to kernels).
"""

from numba import cuda, int8, int16, int32, float32
from poker_ai.deep_cfr.cuda.hand_eval import eval_5cards

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SMALL_BLIND = 50
BIG_BLIND = 100
INITIAL_CHIPS = 10000

PREFLOP = 0
FLOP = 1
TURN = 2
RIVER = 3
SHOWDOWN = 4
TERMINAL = 5

N_FEATURES = 126
N_ACTIONS = 3

# Precomputed C(7,5) index pairs for 7-card evaluation.
# 21 combos of 5 from 7.
_COMBO_5_FROM_7 = (
    (0,1,2,3,4), (0,1,2,3,5), (0,1,2,3,6), (0,1,2,4,5), (0,1,2,4,6),
    (0,1,2,5,6), (0,1,3,4,5), (0,1,3,4,6), (0,1,3,5,6), (0,1,4,5,6),
    (0,2,3,4,5), (0,2,3,4,6), (0,2,3,5,6), (0,2,4,5,6), (0,3,4,5,6),
    (1,2,3,4,5), (1,2,3,4,6), (1,2,3,5,6), (1,2,4,5,6), (1,3,4,5,6),
    (2,3,4,5,6),
)


# ---------------------------------------------------------------------------
# Device helper: current player index
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def _current_player(player_i_index, stage, n_players, preflop_order, postflop_order):
    """Return the actual player index for the current position."""
    if stage == PREFLOP:
        return int32(preflop_order[player_i_index])
    return int32(postflop_order[player_i_index])


# ---------------------------------------------------------------------------
# Device helper: is betting finished
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def _is_betting_finished(chips, bets, active, n_players):
    """True when all active non-all-in players have equal bets."""
    first_bet = int32(-1)
    for p in range(n_players):
        if active[p] and chips[p] > 0:
            if first_bet == -1:
                first_bet = bets[p]
            elif bets[p] != first_bet:
                return False
    return True


# ---------------------------------------------------------------------------
# Device helper: deal community cards
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def _deal_community(community, deck, deck_cursor, n_cards):
    """Deal n_cards from deck to community. Returns updated cursor."""
    cursor = deck_cursor
    for _ in range(n_cards):
        # Find next empty slot.
        slot = int32(-1)
        for s in range(5):
            if community[s] < 0:
                slot = s
                break
        if slot < 0 or slot >= 5:
            break
        community[slot] = deck[cursor]
        cursor += 1
    return cursor


# ---------------------------------------------------------------------------
# Device helper: advance to next player / stage
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def _advance(g, chips, bets, active, community, deck, history,
             n_players, preflop_order, postflop_order):
    """Advance game state after an action.

    g is a tuple-like accessor: (stage, n_raises, player_i_index,
    n_actions, pot_total, deck_cursor, n_players_started_round).
    Returns updated values as a tuple.

    Because Numba CUDA can't pass mutable scalars by reference, we
    pass and return them as local variables.
    """
    stage = g[0]
    n_raises = g[1]
    player_i_index = g[2]
    n_actions = g[3]
    pot_total = g[4]
    deck_cursor = g[5]
    n_players_started_round = g[6]

    for _iter in range(200):  # Safety bound (replaces while True).
        player_i_index = (player_i_index + 1) % n_players
        pi = _current_player(player_i_index, stage, n_players,
                             preflop_order, postflop_order)

        # Check if betting round is finished.
        betting_done = _is_betting_finished(chips, bets, active, n_players)
        if betting_done and n_actions >= n_players_started_round:
            # Increment stage.
            if stage == PREFLOP:
                stage = FLOP
                deck_cursor = _deal_community(community, deck, deck_cursor, 3)
            elif stage == FLOP:
                stage = TURN
                deck_cursor = _deal_community(community, deck, deck_cursor, 1)
            elif stage == TURN:
                stage = RIVER
                deck_cursor = _deal_community(community, deck, deck_cursor, 1)
            elif stage == RIVER:
                stage = SHOWDOWN

            # Reset round.
            n_actions = 0
            n_raises = 0
            player_i_index = 0
            n_players_started_round = 0
            for p in range(n_players):
                if active[p]:
                    n_players_started_round += 1

            # Skip to first active in new round.
            found_active = False
            for _ in range(n_players):
                pi2 = _current_player(player_i_index, stage, n_players,
                                      preflop_order, postflop_order)
                if active[pi2]:
                    found_active = True
                    break
                player_i_index += 1
            pi = _current_player(player_i_index, stage, n_players,
                                 preflop_order, postflop_order)

        if not active[pi]:
            continue

        # Check terminal: count active players with chips.
        n_with_moves = int32(0)
        for p in range(n_players):
            if active[p] and chips[p] > 0:
                n_with_moves += 1
        if n_with_moves <= 1:
            stage = TERMINAL
            # Deal remaining community if needed.
            n_dealt = int32(0)
            for s in range(5):
                if community[s] >= 0:
                    n_dealt += 1
            if n_dealt == 0:
                deck_cursor = _deal_community(community, deck, deck_cursor, 3)

        if stage >= SHOWDOWN:
            break
        break

    return (stage, n_raises, player_i_index, n_actions, pot_total,
            deck_cursor, n_players_started_round)


# ---------------------------------------------------------------------------
# Kernel: apply_action
# ---------------------------------------------------------------------------

@cuda.jit
def apply_action_kernel(
    # Game state arrays (from GameBatch).
    chips, bets, active, hole_cards, community, deck,
    deck_cursor, stage, n_raises, player_i_index,
    n_actions, pot_total, history, n_players_started_round,
    # Inputs.
    actions,  # (N,) int8: 0=fold, 1=call, 2=raise, -1=skip(inactive)
    # Constants.
    n_games, n_players,
    preflop_order, postflop_order,
):
    """Apply one action to each of N games in parallel."""
    i = cuda.grid(1)
    if i >= n_games:
        return
    if stage[i] >= SHOWDOWN:
        return  # Game already finished.

    pi = _current_player(player_i_index[i], stage[i], n_players,
                         preflop_order, postflop_order)
    action = actions[i]

    if action == 0:  # Fold.
        active[i, pi] = 0
    elif action == 1:  # Call.
        if chips[i, pi] > 0:
            biggest = int32(0)
            for p in range(n_players):
                if bets[i, p] > biggest:
                    biggest = bets[i, p]
            to_call = biggest - bets[i, pi]
            if to_call > chips[i, pi]:
                to_call = chips[i, pi]
            chips[i, pi] -= to_call
            bets[i, pi] += to_call
            pot_total[i] += to_call
    elif action == 2:  # Raise.
        # Pot-sized raise (min 1 BB).
        bet_amount = pot_total[i]
        if bet_amount < BIG_BLIND:
            bet_amount = BIG_BLIND
        biggest = int32(0)
        for p in range(n_players):
            if bets[i, p] > biggest:
                biggest = bets[i, p]
        to_call = biggest - bets[i, pi]
        raise_chips = bet_amount + to_call
        if raise_chips > chips[i, pi]:
            raise_chips = chips[i, pi]
        chips[i, pi] -= raise_chips
        bets[i, pi] += raise_chips
        pot_total[i] += raise_chips
        n_raises[i] += 1

    # Record in history.
    if action >= 0:
        rd = stage[i]
        if rd > 3:
            rd = 3
        if action == 1:
            history[i, rd, 0] += 1
        elif action == 2:
            history[i, rd, 1] += 1
        elif action == 0:
            history[i, rd, 2] += 1

    n_actions[i] += 1

    # Advance.
    g = _advance(
        (stage[i], n_raises[i], player_i_index[i], n_actions[i],
         pot_total[i], deck_cursor[i], n_players_started_round[i]),
        chips[i], bets[i], active[i], community[i], deck[i], history[i],
        n_players, preflop_order, postflop_order,
    )
    stage[i] = int8(g[0])
    n_raises[i] = int8(g[1])
    player_i_index[i] = int8(g[2])
    n_actions[i] = int16(g[3])
    pot_total[i] = int32(g[4])
    deck_cursor[i] = int32(g[5])
    n_players_started_round[i] = int8(g[6])


# ---------------------------------------------------------------------------
# Kernel: compute_winners
# ---------------------------------------------------------------------------

@cuda.jit
def compute_winners_kernel(
    chips, bets, active, hole_cards, community,
    payout, stage, n_games, n_players,
    card_lookup,
    flush_keys, flush_vals, flush_size,
    unsuited_keys, unsuited_vals, unsuited_size,
):
    """Compute winners and update chips/payout for finished games."""
    i = cuda.grid(1)
    if i >= n_games:
        return
    if stage[i] < SHOWDOWN:
        return  # Not finished.

    # Count active.
    n_active = int32(0)
    for p in range(n_players):
        if active[i, p]:
            n_active += 1

    if n_active == 0:
        for p in range(n_players):
            payout[i, p] = chips[i, p] - INITIAL_CHIPS
        return

    if n_active == 1:
        # Single winner takes the pot.
        total_pot = int32(0)
        for p in range(n_players):
            total_pot += bets[i, p]
        for p in range(n_players):
            if active[i, p]:
                chips[i, p] += total_pot
                break
        for p in range(n_players):
            payout[i, p] = chips[i, p] - INITIAL_CHIPS
        return

    # Multiple active players — evaluate hands.
    # Build board eval cards.
    board = cuda.local.array(5, dtype=int32)
    n_board = int32(0)
    for c in range(5):
        if community[i, c] >= 0:
            board[n_board] = card_lookup[community[i, c]]
            n_board += 1

    # Evaluate each active player's hand.
    ranks = cuda.local.array(6, dtype=int32)  # Max 6 players.
    for p in range(n_players):
        if active[i, p]:
            # Build 7-card hand: 2 hole + up to 5 community.
            cards7 = cuda.local.array(7, dtype=int32)
            cards7[0] = card_lookup[hole_cards[i, p, 0]]
            cards7[1] = card_lookup[hole_cards[i, p, 1]]
            n_cards = int32(2)
            for c in range(n_board):
                cards7[n_cards] = board[c]
                n_cards += 1
            # Pad remaining with dummy if < 7.
            # For proper evaluation we need at least 5 cards.
            if n_cards >= 5:
                best = int32(7463)
                # Evaluate all C(n_cards, 5) combinations.
                for a in range(n_cards):
                    for b in range(a + 1, n_cards):
                        for c2 in range(b + 1, n_cards):
                            for d in range(c2 + 1, n_cards):
                                for e in range(d + 1, n_cards):
                                    r = eval_5cards(
                                        cards7[a], cards7[b], cards7[c2],
                                        cards7[d], cards7[e],
                                        flush_keys, flush_vals, flush_size,
                                        unsuited_keys, unsuited_vals,
                                        unsuited_size,
                                    )
                                    if r < best:
                                        best = r
                ranks[p] = best
            else:
                ranks[p] = 7463
        else:
            ranks[p] = 7463

    # Side pot distribution.
    remaining = cuda.local.array(6, dtype=int32)
    for p in range(n_players):
        remaining[p] = bets[i, p]

    for _pot_iter in range(n_players):  # At most n_players side pots.
        # Find minimum nonzero remaining bet.
        min_bet = int32(999999)
        any_remaining = False
        for p in range(n_players):
            if remaining[p] > 0:
                any_remaining = True
                if remaining[p] < min_bet:
                    min_bet = remaining[p]
        if not any_remaining:
            break

        # Build this side pot.
        pot_amount = int32(0)
        eligible_count = int32(0)
        best_rank = int32(7463)
        for p in range(n_players):
            if remaining[p] > 0:
                contrib = min_bet
                if remaining[p] < contrib:
                    contrib = remaining[p]
                pot_amount += contrib
                remaining[p] -= contrib
                if active[i, p] and ranks[p] < best_rank:
                    best_rank = ranks[p]

        # Count winners for this pot.
        n_winners = int32(0)
        for p in range(n_players):
            if active[i, p] and ranks[p] == best_rank:
                n_winners += 1

        if n_winners > 0:
            per_winner = pot_amount // n_winners
            leftover = pot_amount - per_winner * n_winners
            winner_idx = int32(0)
            for p in range(n_players):
                if active[i, p] and ranks[p] == best_rank:
                    bonus = int32(0)
                    if winner_idx < leftover:
                        bonus = 1
                    chips[i, p] += per_winner + bonus
                    winner_idx += 1

    # Compute payout.
    for p in range(n_players):
        payout[i, p] = chips[i, p] - INITIAL_CHIPS


# ---------------------------------------------------------------------------
# Kernel: get features
# ---------------------------------------------------------------------------

@cuda.jit
def get_features_kernel(
    chips, bets, active, hole_cards, community,
    stage, n_raises, player_i_index, pot_total, history,
    n_players, preflop_order, postflop_order,
    out_features,  # (N, 126) float32
    n_games,
):
    """Compute 126-dim feature vector for each game."""
    i = cuda.grid(1)
    if i >= n_games:
        return

    pi = _current_player(player_i_index[i], stage[i], n_players,
                         preflop_order, postflop_order)

    # Zero the output.
    for f in range(N_FEATURES):
        out_features[i, f] = 0.0

    # Hole cards: 52-dim binary.
    for c in range(2):
        card = hole_cards[i, pi, c]
        if card >= 0:
            out_features[i, card] = 1.0

    # Community cards: 52-dim binary.
    for c in range(5):
        card = community[i, c]
        if card >= 0:
            out_features[i, 52 + card] = 1.0

    # Betting round: 4-dim one-hot.
    round_idx = stage[i]
    if round_idx > 3:
        round_idx = 3
    if stage[i] < 4:
        out_features[i, 104 + round_idx] = 1.0

    # Scalar features.
    total_chips = float32(INITIAL_CHIPS * n_players)
    out_features[i, 108] = float32(pot_total[i]) / total_chips
    out_features[i, 109] = float32(chips[i, pi]) / float32(INITIAL_CHIPS)
    out_features[i, 110] = float32(bets[i, pi]) / float32(INITIAL_CHIPS)

    active_count = float32(0)
    for p in range(n_players):
        if active[i, p]:
            active_count += 1.0
    out_features[i, 111] = active_count / float32(n_players)

    denom = float32(n_players - 1)
    if denom < 1.0:
        denom = 1.0
    out_features[i, 112] = float32(pi) / denom
    out_features[i, 113] = float32(n_raises[i]) / 3.0

    # Per-round action summary.
    for r in range(4):
        offset = 114 + r * 3
        np_denom = float32(n_players)
        if np_denom < 1.0:
            np_denom = 1.0
        out_features[i, offset] = float32(history[i, r, 0]) / np_denom
        out_features[i, offset + 1] = float32(history[i, r, 1]) / 3.0
        out_features[i, offset + 2] = float32(history[i, r, 2]) / np_denom


# ---------------------------------------------------------------------------
# Kernel: get legal masks
# ---------------------------------------------------------------------------

@cuda.jit
def get_legal_mask_kernel(
    active, chips, n_raises, stage,
    player_i_index, n_players,
    preflop_order, postflop_order,
    out_masks,  # (N, 3) float32
    n_games,
):
    """Compute legal action masks for each game."""
    i = cuda.grid(1)
    if i >= n_games:
        return

    out_masks[i, 0] = 0.0
    out_masks[i, 1] = 0.0
    out_masks[i, 2] = 0.0

    if stage[i] >= SHOWDOWN:
        return

    pi = _current_player(player_i_index[i], stage[i], n_players,
                         preflop_order, postflop_order)

    if active[i, pi]:
        out_masks[i, 0] = 1.0  # fold
        out_masks[i, 1] = 1.0  # call
        if n_raises[i] < 3:
            out_masks[i, 2] = 1.0  # raise
