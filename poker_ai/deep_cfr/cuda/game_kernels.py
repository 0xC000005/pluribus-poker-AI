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
N_ACTIONS = 9

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
    """True when active players are folded, all-in, or matched."""
    active_count = int32(0)
    biggest = int32(0)
    for p in range(n_players):
        if active[p]:
            active_count += 1
            if bets[p] > biggest:
                biggest = bets[p]
    if active_count <= 1:
        return True
    for p in range(n_players):
        if active[p] and chips[p] > 0:
            if bets[p] < biggest:
                return False
    return True


@cuda.jit(device=True)
def _n_active_players(active, n_players):
    count = int32(0)
    for p in range(n_players):
        if active[p]:
            count += 1
    return count


@cuda.jit(device=True)
def _n_players_with_moves(chips, active, n_players):
    count = int32(0)
    for p in range(n_players):
        if active[p] and chips[p] > 0:
            count += 1
    return count


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


@cuda.jit(device=True)
def _deal_remaining_to_showdown(stage, community, deck, deck_cursor):
    while stage < SHOWDOWN:
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
    return stage, deck_cursor


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

        if _n_active_players(active, n_players) == 1:
            stage = TERMINAL
            break

        # Check if betting round is finished.
        betting_done = _is_betting_finished(chips, bets, active, n_players)
        if betting_done and _n_players_with_moves(chips, active, n_players) <= 1:
            stage, deck_cursor = _deal_remaining_to_showdown(
                stage, community, deck, deck_cursor
            )
            break

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

            if stage >= SHOWDOWN:
                break

            if _n_players_with_moves(chips, active, n_players) <= 1:
                stage, deck_cursor = _deal_remaining_to_showdown(
                    stage, community, deck, deck_cursor
                )
                break

            # Reset round.
            n_actions = 0
            n_raises = 0
            player_i_index = 0
            n_players_started_round = 0
            for p in range(n_players):
                if active[p] and chips[p] > 0:
                    n_players_started_round += 1

            # Skip to first active in new round.
            found_active = False
            for _ in range(n_players):
                pi2 = _current_player(player_i_index, stage, n_players,
                                      preflop_order, postflop_order)
                if active[pi2] and chips[pi2] > 0:
                    found_active = True
                    break
                player_i_index += 1
            pi = _current_player(player_i_index, stage, n_players,
                                 preflop_order, postflop_order)

        if not active[pi] or chips[pi] <= 0:
            continue

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
    actions,  # (N,) int8: 0=fold, 1=call, 2-7=frac raise, 8=all-in, -1=skip
    # Constants.
    n_games, n_players,
    preflop_order, postflop_order,
    raise_fractions,  # (6,) float32 device array
):
    """Apply one action to each of N games in parallel."""
    i = cuda.grid(1)
    if i >= n_games:
        return
    if stage[i] >= SHOWDOWN:
        return  # Game already finished.

    action = actions[i]
    if action < 0:
        return

    pi = _current_player(player_i_index[i], stage[i], n_players,
                         preflop_order, postflop_order)

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
    elif action >= 2 and action <= 7:  # Fractional raise.
        frac = raise_fractions[action - 2]
        biggest = int32(0)
        for p in range(n_players):
            if bets[i, p] > biggest:
                biggest = bets[i, p]
        to_call = biggest - bets[i, pi]
        raise_chips = int32(frac * float32(pot_total[i])) + to_call
        if to_call > 0:
            min_raise_by = to_call
            if min_raise_by < BIG_BLIND:
                min_raise_by = int32(BIG_BLIND)
            min_raise = to_call + min_raise_by
        else:
            min_raise = int32(BIG_BLIND)
        if raise_chips < min_raise:
            raise_chips = min_raise
        if raise_chips > chips[i, pi]:
            raise_chips = chips[i, pi]
        chips[i, pi] -= raise_chips
        bets[i, pi] += raise_chips
        pot_total[i] += raise_chips
        n_raises[i] += 1
    elif action == 8:  # All-in.
        all_in = chips[i, pi]
        chips[i, pi] = int32(0)
        bets[i, pi] += all_in
        pot_total[i] += all_in
        n_raises[i] += 1

    # Record in history (3 categories: calls, raises, folds).
    if action >= 0:
        rd = stage[i]
        if rd > 3:
            rd = 3
        if action == 1:
            history[i, rd, 0] += 1  # calls
        elif action >= 2:
            history[i, rd, 1] += 1  # raises (all sizes)
        elif action == 0:
            history[i, rd, 2] += 1  # folds

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


@cuda.jit
def apply_action_mapped_kernel(
    # Game state arrays (from GameBatch).
    chips, bets, active, hole_cards, community, deck,
    deck_cursor, stage, n_raises, player_i_index,
    n_actions, pot_total, history, n_players_started_round,
    # Inputs.
    actions,  # (max_pool,) int8: per-slot actions.
    frontier_indices,  # (N,) int32: stable slot ids to process.
    n_frontier, n_players,
    preflop_order, postflop_order,
    raise_fractions,  # (6,) float32 device array
):
    """Apply actions to a compact frontier without moving tree slots."""
    gid = cuda.grid(1)
    if gid >= n_frontier:
        return
    i = frontier_indices[gid]
    if stage[i] >= SHOWDOWN:
        return

    action = actions[i]
    if action < 0:
        return

    pi = _current_player(player_i_index[i], stage[i], n_players,
                         preflop_order, postflop_order)

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
    elif action >= 2 and action <= 7:  # Fractional raise.
        frac = raise_fractions[action - 2]
        biggest = int32(0)
        for p in range(n_players):
            if bets[i, p] > biggest:
                biggest = bets[i, p]
        to_call = biggest - bets[i, pi]
        raise_chips = int32(frac * float32(pot_total[i])) + to_call
        if to_call > 0:
            min_raise_by = to_call
            if min_raise_by < BIG_BLIND:
                min_raise_by = int32(BIG_BLIND)
            min_raise = to_call + min_raise_by
        else:
            min_raise = int32(BIG_BLIND)
        if raise_chips < min_raise:
            raise_chips = min_raise
        if raise_chips > chips[i, pi]:
            raise_chips = chips[i, pi]
        chips[i, pi] -= raise_chips
        bets[i, pi] += raise_chips
        pot_total[i] += raise_chips
        n_raises[i] += 1
    elif action == 8:  # All-in.
        all_in = chips[i, pi]
        chips[i, pi] = int32(0)
        bets[i, pi] += all_in
        pot_total[i] += all_in
        n_raises[i] += 1

    rd = stage[i]
    if rd > 3:
        rd = 3
    if action == 1:
        history[i, rd, 0] += 1
    elif action >= 2:
        history[i, rd, 1] += 1
    elif action == 0:
        history[i, rd, 2] += 1

    n_actions[i] += 1

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
    initial_chips,
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
            payout[i, p] = chips[i, p] - initial_chips
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
            payout[i, p] = chips[i, p] - initial_chips
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
        payout[i, p] = chips[i, p] - initial_chips


# ---------------------------------------------------------------------------
# Kernel: get features
# ---------------------------------------------------------------------------

@cuda.jit
def get_features_kernel(
    chips, bets, active, hole_cards, community,
    stage, n_raises, player_i_index, pot_total, history,
    n_players, preflop_order, postflop_order,
    out_features,  # (N, 126) float32
    n_games, initial_chips,
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
    total_chips = float32(initial_chips * n_players)
    out_features[i, 108] = float32(pot_total[i]) / total_chips
    out_features[i, 109] = float32(chips[i, pi]) / float32(initial_chips)
    out_features[i, 110] = float32(bets[i, pi]) / float32(initial_chips)

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


@cuda.jit
def get_features_mapped_kernel(
    chips, bets, active, hole_cards, community,
    stage, n_raises, player_i_index, pot_total, history,
    n_players, preflop_order, postflop_order,
    frontier_indices,  # (N,) int32: stable source slot ids.
    out_features,  # (N, 126) float32: compact output rows.
    n_frontier, initial_chips,
):
    """Compute compact feature rows from stable frontier slot ids."""
    row = cuda.grid(1)
    if row >= n_frontier:
        return
    i = frontier_indices[row]

    pi = _current_player(player_i_index[i], stage[i], n_players,
                         preflop_order, postflop_order)

    for f in range(N_FEATURES):
        out_features[row, f] = 0.0

    for c in range(2):
        card = hole_cards[i, pi, c]
        if card >= 0:
            out_features[row, card] = 1.0

    for c in range(5):
        card = community[i, c]
        if card >= 0:
            out_features[row, 52 + card] = 1.0

    round_idx = stage[i]
    if round_idx > 3:
        round_idx = 3
    if stage[i] < 4:
        out_features[row, 104 + round_idx] = 1.0

    total_chips = float32(initial_chips * n_players)
    out_features[row, 108] = float32(pot_total[i]) / total_chips
    out_features[row, 109] = float32(chips[i, pi]) / float32(initial_chips)
    out_features[row, 110] = float32(bets[i, pi]) / float32(initial_chips)

    active_count = float32(0)
    for p in range(n_players):
        if active[i, p]:
            active_count += 1.0
    out_features[row, 111] = active_count / float32(n_players)

    denom = float32(n_players - 1)
    if denom < 1.0:
        denom = 1.0
    out_features[row, 112] = float32(pi) / denom
    out_features[row, 113] = float32(n_raises[i]) / 3.0

    for r in range(4):
        offset = 114 + r * 3
        np_denom = float32(n_players)
        if np_denom < 1.0:
            np_denom = 1.0
        out_features[row, offset] = float32(history[i, r, 0]) / np_denom
        out_features[row, offset + 1] = float32(history[i, r, 1]) / 3.0
        out_features[row, offset + 2] = float32(history[i, r, 2]) / np_denom


# ---------------------------------------------------------------------------
# Kernel: get legal masks
# ---------------------------------------------------------------------------

@cuda.jit
def get_legal_mask_kernel(
    active, chips, bets, n_raises, stage, pot_total,
    player_i_index, n_players,
    preflop_order, postflop_order,
    raise_fractions,  # (6,) float32 device array
    out_masks,  # (N, 9) float32
    n_games,
):
    """Compute legal action masks for each game (9-action space)."""
    i = cuda.grid(1)
    if i >= n_games:
        return

    for a in range(N_ACTIONS):
        out_masks[i, a] = float32(0.0)

    if stage[i] >= SHOWDOWN:
        return

    pi = _current_player(player_i_index[i], stage[i], n_players,
                         preflop_order, postflop_order)

    if active[i, pi] and chips[i, pi] > 0:
        out_masks[i, 0] = float32(1.0)  # fold
        out_masks[i, 1] = float32(1.0)  # call
        if n_raises[i] < 3:
            biggest = int32(0)
            for p in range(n_players):
                if bets[i, p] > biggest:
                    biggest = bets[i, p]
            to_call = biggest - bets[i, pi]
            player_chips = chips[i, pi]
            if to_call > 0:
                min_raise_by = to_call
                if min_raise_by < BIG_BLIND:
                    min_raise_by = int32(BIG_BLIND)
                min_raise = to_call + min_raise_by
            else:
                min_raise = int32(BIG_BLIND)
            # Check each fractional raise (actions 2-7).
            for fi in range(6):
                raise_amount = int32(raise_fractions[fi] * float32(pot_total[i])) + to_call
                if raise_amount >= min_raise and raise_amount <= player_chips:
                    out_masks[i, 2 + fi] = float32(1.0)
            # All-in (action 8).
            if player_chips > 0:
                out_masks[i, 8] = float32(1.0)


@cuda.jit
def get_legal_mask_mapped_kernel(
    active, chips, bets, n_raises, stage, pot_total,
    player_i_index, n_players,
    preflop_order, postflop_order,
    raise_fractions,  # (6,) float32 device array
    frontier_indices,  # (N,) int32: stable source slot ids.
    out_masks,  # (N, 9) float32: compact output rows.
    n_frontier,
):
    """Compute compact legal-mask rows from stable frontier slot ids."""
    row = cuda.grid(1)
    if row >= n_frontier:
        return
    i = frontier_indices[row]

    for a in range(N_ACTIONS):
        out_masks[row, a] = float32(0.0)

    if stage[i] >= SHOWDOWN:
        return

    pi = _current_player(player_i_index[i], stage[i], n_players,
                         preflop_order, postflop_order)

    if active[i, pi] and chips[i, pi] > 0:
        out_masks[row, 0] = float32(1.0)
        out_masks[row, 1] = float32(1.0)
        if n_raises[i] < 3:
            biggest = int32(0)
            for p in range(n_players):
                if bets[i, p] > biggest:
                    biggest = bets[i, p]
            to_call = biggest - bets[i, pi]
            player_chips = chips[i, pi]
            if to_call > 0:
                min_raise_by = to_call
                if min_raise_by < BIG_BLIND:
                    min_raise_by = int32(BIG_BLIND)
                min_raise = to_call + min_raise_by
            else:
                min_raise = int32(BIG_BLIND)
            for fi in range(6):
                raise_amount = int32(raise_fractions[fi] * float32(pot_total[i])) + to_call
                if raise_amount >= min_raise and raise_amount <= player_chips:
                    out_masks[row, 2 + fi] = float32(1.0)
            if player_chips > 0:
                out_masks[row, 8] = float32(1.0)
