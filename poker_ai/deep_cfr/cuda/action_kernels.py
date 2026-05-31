"""CUDA kernels for regret matching, action sampling, and tree traversal.

Kernels:
  regret_match_kernel: advantages -> strategy probabilities
  sample_action_kernel: strategy + RNG -> sampled action per game
  classify_and_sample_kernel: classify traverser/opponent + sample
  fork_kernel: allocate children for traverser nodes (replaces CPU fork loop)
  copy_from_parent_kernel: copy game state from parent to child slots
  propagate_kernel: propagate terminal values up tree, collect regret samples
  collect_policy_targets_kernel: collect legal policy targets from opponent nodes
"""

from numba import cuda, int8, int32, float32
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_uniform_float32,
)

N_ACTIONS = 9
PREFLOP = 0


# ---------------------------------------------------------------------------
# Device helper: current player (duplicated to avoid cross-module import)
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def _current_player_ak(player_i_index, stage, n_players,
                       preflop_order, postflop_order):
    """Return the actual player index for the current position."""
    if stage == PREFLOP:
        return int32(preflop_order[player_i_index])
    return int32(postflop_order[player_i_index])


# ---------------------------------------------------------------------------
# Kernel: regret matching (advantages -> strategy probabilities)
# ---------------------------------------------------------------------------

@cuda.jit
def regret_match_kernel(
    advantages,     # (N, 9) float32 — NN output
    legal_masks,    # (N, 9) float32 — from get_legal_mask_kernel
    strategies,     # (N, 9) float32 — OUTPUT
    n_games,        # int32
):
    """Convert advantages to strategy via regret matching. One thread per game.

    strategy[a] = max(adv[a], 0) * legal[a] / sum(...)
    If sum == 0: uniform over legal actions.
    """
    gid = cuda.grid(1)
    if gid >= n_games:
        return

    # Regret matching: clamp to positive, mask illegal.
    total = float32(0.0)
    for a in range(N_ACTIONS):
        val = advantages[gid, a]
        if val > float32(0.0):
            s = val * legal_masks[gid, a]
        else:
            s = float32(0.0)
        strategies[gid, a] = s
        total += s

    if total > float32(0.0):
        inv = float32(1.0) / total
        for a in range(N_ACTIONS):
            strategies[gid, a] *= inv
    else:
        # Uniform over legal actions.
        n_legal = float32(0.0)
        for a in range(N_ACTIONS):
            n_legal += legal_masks[gid, a]
        if n_legal > float32(0.0):
            inv = float32(1.0) / n_legal
            for a in range(N_ACTIONS):
                strategies[gid, a] = legal_masks[gid, a] * inv
        else:
            for a in range(N_ACTIONS):
                strategies[gid, a] = float32(0.0)


# ---------------------------------------------------------------------------
# Kernel: sample action from strategy (opponent nodes)
# ---------------------------------------------------------------------------

@cuda.jit
def sample_action_kernel(
    strategies,     # (N, 9) float32
    legal_masks,    # (N, 9) float32
    stages,         # (N,) int8 — game stage (skip finished games)
    rng_states,     # xoroshiro128p states array
    out_actions,    # (N,) int8 — OUTPUT: sampled action (0-8), -1 for skipped
    n_games,        # int32
):
    """Sample one action from strategy distribution. One thread per game.

    Skips games where stage >= 4 (finished) or no legal actions.
    """
    gid = cuda.grid(1)
    if gid >= n_games:
        return

    if stages[gid] >= int8(4):
        out_actions[gid] = int8(-1)
        return

    # Check if any legal action.
    n_legal = float32(0.0)
    for a in range(N_ACTIONS):
        n_legal += legal_masks[gid, a]
    if n_legal <= float32(0.0):
        out_actions[gid] = int8(-1)
        return

    # Draw uniform random in [0, 1).
    u = xoroshiro128p_uniform_float32(rng_states, gid)

    # Categorical sampling via cumulative sum.
    cumsum = float32(0.0)
    chosen = int8(N_ACTIONS - 1)  # Default to last action.
    for a in range(N_ACTIONS):
        cumsum += strategies[gid, a]
        if u < cumsum:
            chosen = int8(a)
            break
    out_actions[gid] = chosen


# ---------------------------------------------------------------------------
# Kernel: classify traverser vs opponent + sample for opponents
# ---------------------------------------------------------------------------

@cuda.jit
def classify_and_sample_kernel(
    strategies,         # (N, 9) float32
    legal_masks,        # (N, 9) float32
    stages,             # (N,) int8
    player_i_indices,   # (N,) int8
    n_players,          # int32
    traverser,          # int32
    preflop_order,      # (n_players,) int8
    postflop_order,     # (n_players,) int8
    rng_states,         # xoroshiro128p states
    out_actions,        # (N,) int8 — sampled action for opponents, -1 for traverser/skipped
    out_is_traverser,   # (N,) int8 — 1 if traverser node, 0 otherwise
    n_games,            # int32
):
    """Classify each game as traverser/opponent and sample actions for opponents.

    For traverser nodes: out_actions = -1, out_is_traverser = 1
    For opponent nodes: out_actions = sampled action, out_is_traverser = 0
    For finished/no-legal: out_actions = -1, out_is_traverser = 0
    """
    gid = cuda.grid(1)
    if gid >= n_games:
        return

    out_actions[gid] = int8(-1)
    out_is_traverser[gid] = int8(0)

    if stages[gid] >= int8(4):
        return

    n_legal = float32(0.0)
    for a in range(N_ACTIONS):
        n_legal += legal_masks[gid, a]
    if n_legal <= float32(0.0):
        return

    # Determine current player.
    pi = _current_player_ak(
        player_i_indices[gid], stages[gid], n_players,
        preflop_order, postflop_order,
    )

    if pi == traverser:
        out_is_traverser[gid] = int8(1)
        return  # Action decided by CPU fork logic.

    # Opponent: sample action from strategy.
    u = xoroshiro128p_uniform_float32(rng_states, gid)
    cumsum = float32(0.0)
    chosen = int8(N_ACTIONS - 1)
    for a in range(N_ACTIONS):
        cumsum += strategies[gid, a]
        if u < cumsum:
            chosen = int8(a)
            break
    out_actions[gid] = chosen


@cuda.jit
def classify_and_sample_mapped_kernel(
    strategies,         # (N, 9) float32 compact frontier rows
    legal_masks,        # (N, 9) float32 compact frontier rows
    stages,             # (max_pool,) int8
    player_i_indices,   # (max_pool,) int8
    frontier_indices,   # (N,) int32 stable slot ids
    n_players,          # int32
    traverser,          # int32
    preflop_order,      # (n_players,) int8
    postflop_order,     # (n_players,) int8
    rng_states,         # xoroshiro128p states indexed by stable slot id
    out_actions,        # (max_pool,) int8 per-slot actions
    out_is_traverser,   # (max_pool,) int8 per-slot traverser flags
    n_frontier,         # int32
):
    """Classify/sample compact frontier rows while writing stable slot outputs."""
    row = cuda.grid(1)
    if row >= n_frontier:
        return
    slot = frontier_indices[row]

    out_actions[slot] = int8(-1)
    out_is_traverser[slot] = int8(0)

    if stages[slot] >= int8(4):
        return

    n_legal = float32(0.0)
    for a in range(N_ACTIONS):
        n_legal += legal_masks[row, a]
    if n_legal <= float32(0.0):
        return

    pi = _current_player_ak(
        player_i_indices[slot], stages[slot], n_players,
        preflop_order, postflop_order,
    )

    if pi == traverser:
        out_is_traverser[slot] = int8(1)
        return

    u = xoroshiro128p_uniform_float32(rng_states, slot)
    cumsum = float32(0.0)
    chosen = int8(N_ACTIONS - 1)
    for a in range(N_ACTIONS):
        cumsum += strategies[row, a]
        if u < cumsum:
            chosen = int8(a)
            break
    out_actions[slot] = chosen


@cuda.jit
def collect_policy_targets_kernel(
    features,            # (N, 126) float32
    legal_masks,         # (N, 9) float32
    strategies,          # (N, 9) float32
    stages,              # (N,) int8
    player_i_indices,    # (N,) int8
    n_players,           # int32
    traverser,           # int32
    preflop_order,       # (n_players,) int8
    postflop_order,      # (n_players,) int8
    collected_features,  # (policy_capacity, 126) float32
    collected_masks,     # (policy_capacity, 9) float32
    collected_targets,   # (policy_capacity, 9) float32
    n_collected,         # (1,) int32
    policy_capacity,     # int32
    n_games,             # int32
):
    """Collect opponent-node strategy targets for average-policy training."""
    gid = cuda.grid(1)
    if gid >= n_games:
        return
    if stages[gid] >= int8(4):
        return

    n_legal = float32(0.0)
    for a in range(N_ACTIONS):
        n_legal += legal_masks[gid, a]
    if n_legal <= float32(0.0):
        return

    pi = _current_player_ak(
        player_i_indices[gid], stages[gid], n_players,
        preflop_order, postflop_order,
    )
    if pi == traverser:
        return

    out_idx = cuda.atomic.add(n_collected, 0, int32(1))
    if out_idx >= policy_capacity:
        return
    for f in range(N_FEATURES):
        collected_features[out_idx, f] = features[gid, f]
    for a in range(N_ACTIONS):
        collected_masks[out_idx, a] = legal_masks[gid, a]
        collected_targets[out_idx, a] = strategies[gid, a]


# ---------------------------------------------------------------------------
# Constants for tree traversal kernels
# ---------------------------------------------------------------------------

N_FEATURES = 126


# ---------------------------------------------------------------------------
# Kernel: fork traverser nodes (replaces CPU fork loop)
# ---------------------------------------------------------------------------

@cuda.jit
def fork_kernel(
    is_traverser_flag,    # (N,) int8 — 1=traverser (IN/OUT: 0 on pool exhaust)
    stages,               # (N,) int8
    features,             # (N, 126) float32
    strategies,           # (N, 9) float32
    legal_masks,          # (N, 9) float32
    parent_idx,           # (max_pool,) int32
    parent_action,        # (max_pool,) int8
    is_traverser_node,    # (max_pool,) int8 — marks forked nodes
    traverser_features,   # (max_pool, 126) float32
    slot_strategy,        # (max_pool, 9) float32
    n_children_expected,  # (max_pool,) int32
    child_values,         # (max_pool, 9) float32
    n_children_done,      # (max_pool,) int32
    next_free,            # (1,) int32 — atomic counter
    pool_exhausted_count, # (1,) int32 — traverser nodes demoted by pool exhaustion
    pool_exhausted_by_depth, # (100,) int32
    pool_exhausted_by_stage, # (4,) int32
    max_pool,             # int32
    actions_out,          # (max_pool,) int8
    rng_states,
    depth,                # int32
    n_active,             # int32
):
    """Allocate child slots for traverser nodes. One thread per slot."""
    gid = cuda.grid(1)
    if gid >= n_active:
        return
    if stages[gid] >= int8(4):
        return
    if is_traverser_flag[gid] != int8(1):
        return
    if is_traverser_node[gid] == int8(1) and n_children_expected[gid] > int32(0):
        return

    n_legal = int32(0)
    for a in range(N_ACTIONS):
        if legal_masks[gid, a] > float32(0.0):
            n_legal += 1
    if n_legal == 0:
        return

    start = cuda.atomic.add(next_free, 0, n_legal)
    if start >= max_pool or start + n_legal > max_pool:
        # Pool exhausted — demote to opponent, sample action.
        cuda.atomic.add(pool_exhausted_count, 0, int32(1))
        if depth >= 0 and depth < 100:
            cuda.atomic.add(pool_exhausted_by_depth, depth, int32(1))
        stage_i = int32(stages[gid])
        if stage_i >= 0 and stage_i < 4:
            cuda.atomic.add(pool_exhausted_by_stage, stage_i, int32(1))
        is_traverser_flag[gid] = int8(0)
        u = xoroshiro128p_uniform_float32(rng_states, gid)
        cumsum_f = float32(0.0)
        chosen_f = int8(N_ACTIONS - 1)
        for a in range(N_ACTIONS):
            cumsum_f += strategies[gid, a]
            if u < cumsum_f:
                chosen_f = int8(a)
                break
        actions_out[gid] = chosen_f
        return

    is_traverser_node[gid] = int8(1)
    n_children_expected[gid] = n_legal
    n_children_done[gid] = int32(0)
    for f in range(N_FEATURES):
        traverser_features[gid, f] = features[gid, f]
    for a in range(N_ACTIONS):
        slot_strategy[gid, a] = strategies[gid, a]
        child_values[gid, a] = float32(0.0)

    legal_i = int32(0)
    for a in range(N_ACTIONS):
        if legal_masks[gid, a] > float32(0.0):
            child = start + legal_i
            parent_idx[child] = int32(gid)
            parent_action[child] = int8(a)
            actions_out[child] = int8(a)
            is_traverser_node[child] = int8(0)
            n_children_done[child] = int32(0)
            n_children_expected[child] = int32(0)
            legal_i += 1


@cuda.jit
def fork_mapped_kernel(
    is_traverser_flag,    # (max_pool,) int8 — 1=traverser (IN/OUT)
    stages,               # (max_pool,) int8
    features,             # (N, 126) float32 compact frontier rows
    strategies,           # (N, 9) float32 compact frontier rows
    legal_masks,          # (N, 9) float32 compact frontier rows
    frontier_indices,     # (N,) int32 stable slot ids
    parent_idx,           # (max_pool,) int32
    parent_action,        # (max_pool,) int8
    is_traverser_node,    # (max_pool,) int8
    traverser_features,   # (max_pool, 126) float32
    slot_strategy,        # (max_pool, 9) float32
    n_children_expected,  # (max_pool,) int32
    child_values,         # (max_pool, 9) float32
    n_children_done,      # (max_pool,) int32
    next_free,            # (1,) int32
    pool_exhausted_count, # (1,) int32
    pool_exhausted_by_depth, # (100,) int32
    pool_exhausted_by_stage, # (4,) int32
    max_pool,             # int32
    actions_out,          # (max_pool,) int8
    rng_states,
    depth,                # int32
    n_frontier,           # int32
):
    """Allocate children for compact frontier rows while keeping slot ids stable."""
    row = cuda.grid(1)
    if row >= n_frontier:
        return
    gid = frontier_indices[row]
    if stages[gid] >= int8(4):
        return
    if is_traverser_flag[gid] != int8(1):
        return
    if is_traverser_node[gid] == int8(1) and n_children_expected[gid] > int32(0):
        return

    n_legal = int32(0)
    for a in range(N_ACTIONS):
        if legal_masks[row, a] > float32(0.0):
            n_legal += 1
    if n_legal == 0:
        return

    start = cuda.atomic.add(next_free, 0, n_legal)
    if start >= max_pool or start + n_legal > max_pool:
        cuda.atomic.add(pool_exhausted_count, 0, int32(1))
        if depth >= 0 and depth < 100:
            cuda.atomic.add(pool_exhausted_by_depth, depth, int32(1))
        stage_i = int32(stages[gid])
        if stage_i >= 0 and stage_i < 4:
            cuda.atomic.add(pool_exhausted_by_stage, stage_i, int32(1))
        is_traverser_flag[gid] = int8(0)
        u = xoroshiro128p_uniform_float32(rng_states, gid)
        cumsum_f = float32(0.0)
        chosen_f = int8(N_ACTIONS - 1)
        for a in range(N_ACTIONS):
            cumsum_f += strategies[row, a]
            if u < cumsum_f:
                chosen_f = int8(a)
                break
        actions_out[gid] = chosen_f
        return

    is_traverser_node[gid] = int8(1)
    n_children_expected[gid] = n_legal
    n_children_done[gid] = int32(0)
    for f in range(N_FEATURES):
        traverser_features[gid, f] = features[row, f]
    for a in range(N_ACTIONS):
        slot_strategy[gid, a] = strategies[row, a]
        child_values[gid, a] = float32(0.0)

    legal_i = int32(0)
    for a in range(N_ACTIONS):
        if legal_masks[row, a] > float32(0.0):
            child = start + legal_i
            parent_idx[child] = int32(gid)
            parent_action[child] = int8(a)
            actions_out[child] = int8(a)
            is_traverser_node[child] = int8(0)
            n_children_done[child] = int32(0)
            n_children_expected[child] = int32(0)
            legal_i += 1


# ---------------------------------------------------------------------------
# Kernel: copy game state from parent to newly forked children
# ---------------------------------------------------------------------------

@cuda.jit
def copy_from_parent_kernel(
    chips, bets, active, hole_cards, community, deck,
    deck_cursor, stage, n_raises, player_i_index,
    n_actions, pot_total, n_players_started_round, history,
    payout, is_done,
    parent_idx, start_slot, end_slot, n_players,
):
    """Copy game state from parent_idx[slot] to slot, for [start, end)."""
    tid = cuda.grid(1)
    slot = start_slot + tid
    if slot >= end_slot:
        return

    si = parent_idx[slot]
    di = slot

    for p in range(n_players):
        chips[di, p] = chips[si, p]
        bets[di, p] = bets[si, p]
        active[di, p] = active[si, p]
        payout[di, p] = payout[si, p]
        for c in range(2):
            hole_cards[di, p, c] = hole_cards[si, p, c]

    for c in range(5):
        community[di, c] = community[si, c]
    for c in range(52):
        deck[di, c] = deck[si, c]

    deck_cursor[di] = deck_cursor[si]
    stage[di] = stage[si]
    n_raises[di] = n_raises[si]
    player_i_index[di] = player_i_index[si]
    n_actions[di] = n_actions[si]
    pot_total[di] = pot_total[si]
    n_players_started_round[di] = n_players_started_round[si]
    is_done[di] = is_done[si]

    for r in range(4):
        for a in range(3):
            history[di, r, a] = history[si, r, a]


# ---------------------------------------------------------------------------
# Kernel: propagate terminal values up tree and collect regret samples
# ---------------------------------------------------------------------------

@cuda.jit
def propagate_kernel(
    stages,              # (N,) int8
    payouts,             # (N, n_players) int32
    traverser,           # int32
    parent_idx,          # (max_pool,) int32
    parent_action,       # (max_pool,) int8
    is_traverser_node,   # (max_pool,) int8
    traverser_features,  # (max_pool, 126) float32
    slot_strategy,       # (max_pool, 9) float32
    child_values,        # (max_pool, 9) float32
    n_children_done,     # (max_pool,) int32
    n_children_expected, # (max_pool,) int32
    propagated,          # (max_pool,) int8
    collected_features,  # (max_pool, 126) float32
    collected_regrets,   # (max_pool, 9) float32
    n_collected,         # (1,) int32 — atomic counter
    initial_chips,       # float32
    n_slots,             # int32
):
    """Propagate terminal values up tree. One thread per terminal."""
    gid = cuda.grid(1)
    if gid >= n_slots:
        return
    if stages[gid] < int8(4):
        return
    if propagated[gid] == int8(1):
        return

    propagated[gid] = int8(1)
    current_value = float32(payouts[gid, traverser])
    current_idx = int32(gid)

    # Max collected = n_slots (one per traverser node at most).
    max_collected = n_slots

    for _safety in range(200):
        pidx = parent_idx[current_idx]
        if pidx < int32(0):
            break

        if is_traverser_node[pidx] == int8(1):
            action = parent_action[current_idx]
            child_values[pidx, action] = current_value
            old = cuda.atomic.add(n_children_done, pidx, int32(1))

            if old + int32(1) >= n_children_expected[pidx]:
                state_value = float32(0.0)
                for a in range(N_ACTIONS):
                    state_value += slot_strategy[pidx, a] * child_values[pidx, a]

                out_idx = cuda.atomic.add(n_collected, 0, int32(1))
                if out_idx < max_collected:
                    ichips = float32(initial_chips)
                    for f in range(N_FEATURES):
                        collected_features[out_idx, f] = traverser_features[pidx, f]
                    for a in range(N_ACTIONS):
                        collected_regrets[out_idx, a] = (
                            child_values[pidx, a] - state_value
                        ) / ichips

                current_value = state_value
                current_idx = pidx
            else:
                break
        else:
            current_idx = pidx


# ---------------------------------------------------------------------------
# Kernel: reset traversal bookkeeping arrays
# ---------------------------------------------------------------------------

@cuda.jit
def reset_traversal_state_kernel(
    parent_idx,          # (max_pool,) int32
    parent_action,       # (max_pool,) int8
    is_traverser_node,   # (max_pool,) int8
    n_children_done,     # (max_pool,) int32
    n_children_expected, # (max_pool,) int32
    propagated,          # (max_pool,) int8
    n_slots,             # int32
):
    """Reset traversal bookkeeping arrays for [0, n_slots)."""
    gid = cuda.grid(1)
    if gid >= n_slots:
        return
    parent_idx[gid] = int32(-1)
    parent_action[gid] = int8(-1)
    is_traverser_node[gid] = int8(0)
    n_children_done[gid] = int32(0)
    n_children_expected[gid] = int32(0)
    propagated[gid] = int8(0)


# ---------------------------------------------------------------------------
# Kernel: count non-terminal slots
# ---------------------------------------------------------------------------

@cuda.jit
def count_nonterminal_kernel(
    stages,     # (N,) int8
    out_count,  # (1,) int32
    n_slots,    # int32
):
    """Count slots where stage < 4 using one atomic counter."""
    gid = cuda.grid(1)
    if gid >= n_slots:
        return
    if stages[gid] < int8(4):
        cuda.atomic.add(out_count, 0, int32(1))


@cuda.jit
def count_active_frontier_kernel(
    stages,              # (N,) int8
    is_traverser_node,   # (N,) int8
    n_children_expected, # (N,) int32
    out_count,           # (1,) int32
    n_slots,             # int32
):
    """Count non-terminal slots that can still advance in the wavefront."""
    gid = cuda.grid(1)
    if gid >= n_slots:
        return
    if stages[gid] >= int8(4):
        return
    if is_traverser_node[gid] == int8(1) and n_children_expected[gid] > int32(0):
        return
    cuda.atomic.add(out_count, 0, int32(1))
