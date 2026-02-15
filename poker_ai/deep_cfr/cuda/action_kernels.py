"""CUDA kernels for regret matching and action sampling.

Moves the CPU-bound Python loop (regret matching + np.random.choice)
to GPU. Combined with zero-copy Numba<->PyTorch interop, this
eliminates the dominant bottleneck in GPU traversal.

Kernels:
  regret_match_kernel: advantages -> strategy probabilities
  sample_action_kernel: strategy + RNG -> sampled action per game
  classify_and_sample_kernel: classify traverser/opponent + sample
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
