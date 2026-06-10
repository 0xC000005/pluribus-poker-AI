"""Per-belief V* river target emission for the PBS net (p1_design.json STEP 8).

The training-row interface the P2/P4 board-general river PBS net consumes,
mirroring the committed row conventions of ``poker_ai/rebel/iig_selfplay.py``
``_row`` (x = public encoding + padded ranges; y = padded values; w = reach
weights) and ``poker_ai/rebel/river_pbs_net.py::sample_river_targets``
(X = normalized ranges, Y = per-hand CFVs, W = the same normalized ranges;
Dirichlet beliefs with a in {0.3, 1.0, 3.0} per
``scripts/run_rebel_b0_perbelief_targets.py``), lifted to the board-general
GLOBAL 1326 hand index:

    x [X_DIM]  = board 52-hot | pot/scale, eff_stack/scale | r0n [1326] | r1n [1326]
    y [Y_DIM]  = v0 [1326] | v1 [1326]   (local solve values scattered global,
                                          zero padding on board-conflict hands)
    w [Y_DIM]  = r0n [1326] | r1n [1326] (reach weights == the x range slices)

``r0n``/``r1n`` are normalized over the board's local support and scattered
back, so board-conflicting hands carry exactly zero range, weight, and target.
``v0``/``v1`` come from the population solver's batched average-strategy root
value pass: per-hand COUNTERFACTUAL values in the net-from-street-start
convention (opponent-reach-weighted, own range factored out) -- the B0 finding
made board-general (per-belief targets break the target-coherence floor).

River-exact targets only at this stage (the DeepStack-lite gate needs exactly
river-exact -> turn); the net architecture itself is P2 scope.
"""
from __future__ import annotations

import numpy as np

from poker_ai.rebel.hunl import subgame_spec as sgs

__all__ = [
    "DIRICHLET_ALPHAS",
    "DEFAULT_SCALE",
    "N_BOARD",
    "N_SCALARS",
    "X_DIM",
    "Y_DIM",
    "encode_x",
    "result_row",
    "generate_river_targets",
]

# Belief-sampling convention of scripts/run_rebel_b0_perbelief_targets.py /
# river_pbs_net.sample_river_targets / iig_selfplay.coverage_rows.
DIRICHLET_ALPHAS = (0.3, 1.0, 3.0)

DEFAULT_SCALE = 20000.0  # census protocol stack (200bb at BIG_BLIND=100)

N_BOARD = sgs.N_CARDS                              # 52-hot board encoding
N_SCALARS = 2                                      # pot, effective stack
X_DIM = N_BOARD + N_SCALARS + 2 * sgs.N_GLOBAL_HANDS  # 2706
Y_DIM = 2 * sgs.N_GLOBAL_HANDS                        # 2652


def _normalized_global(board, r_global):
    """Normalize a global range over the board's local support, scatter back.

    Guarantees sum == 1 over live hands and exact zeros on board-conflicting
    hands (the padded-global convention).
    """
    local = sgs.gather_local(board, np.asarray(r_global, dtype=np.float64))
    total = local.sum()
    if total > 1e-12:
        local = local / total
    else:
        local = np.full(local.shape, 1.0 / local.shape[0])
    return sgs.scatter_global(board, local)


def encode_x(spec, scale=DEFAULT_SCALE) -> np.ndarray:
    """[X_DIM] float32 PBS-net input row for one river spec."""
    x = np.zeros(X_DIM, dtype=np.float32)
    x[list(spec.board)] = 1.0
    x[N_BOARD] = spec.pot / scale
    x[N_BOARD + 1] = min(spec.stack0, spec.stack1) / scale
    x[N_BOARD + N_SCALARS:N_BOARD + N_SCALARS + sgs.N_GLOBAL_HANDS] = (
        _normalized_global(spec.board, spec.r0))
    x[N_BOARD + N_SCALARS + sgs.N_GLOBAL_HANDS:] = (
        _normalized_global(spec.board, spec.r1))
    return x


def result_row(result, scale=DEFAULT_SCALE):
    """(x, y, w) float32 training row from one solved river spec.

    The global twin of ``iig_selfplay._row``: y carries the value-pass V*
    targets scattered to the global index (zero padding off-board), w carries
    the normalized reach weights (identical to x's range slices).
    """
    spec = result.spec
    if spec.street != "river":
        raise ValueError(
            f"river target rows need river specs, got {spec.street!r}")
    if result.v0 is None or result.v1 is None:
        raise ValueError(
            "result carries no value-pass targets "
            "(solve with compute_value_pass=True)")
    x = encode_x(spec, scale=scale)
    y = np.zeros(Y_DIM, dtype=np.float32)
    y[:sgs.N_GLOBAL_HANDS] = result.v0_global()
    y[sgs.N_GLOBAL_HANDS:] = result.v1_global()
    w = np.zeros(Y_DIM, dtype=np.float32)
    w[:sgs.N_GLOBAL_HANDS] = x[N_BOARD + N_SCALARS:
                               N_BOARD + N_SCALARS + sgs.N_GLOBAL_HANDS]
    w[sgs.N_GLOBAL_HANDS:] = x[N_BOARD + N_SCALARS + sgs.N_GLOBAL_HANDS:]
    return x, y, w


def generate_river_targets(n_beliefs, boards, config_sampler, rng, queue,
                           scale=DEFAULT_SCALE, factory=None):
    """Sample beliefs, fill the population queue, flush, emit training rows.

    Pure-CPU-capable end-to-end driver: for each river ``board`` (outer loop)
    and each of ``n_beliefs`` (inner loop) it samples a Dirichlet belief pair
    over the board's local hands (``a0, a1 ~ choice(DIRICHLET_ALPHAS)`` then
    ``rng.dirichlet`` -- the exact rng call order of
    run_rebel_b0_perbelief_targets.py, reproducible from the seed), draws
    ``(pot, stack0, stack1, first_to_act) = config_sampler(rng)``, enqueues the
    spec, and finally drains the queue. Same-topology specs across boards and
    beliefs fuse into single batched solves at the queue's memory-derived
    B_max; nothing global is ever enumerated.

    Args:
        n_beliefs: belief pairs per board.
        boards: iterable of 5-card river boards.
        config_sampler: callable ``rng -> (pot, stack0, stack1, first_to_act)``
            (principled choice: the trunk-reach config distribution; tests use
            a fixed config).
        rng: ``numpy.random.Generator``.
        queue: a :class:`lazy_subgames.PopulationQueue` (must compute the
            value pass).
        scale: chip normalizer for the pot/eff-stack scalars.
        factory: optional :class:`lazy_subgames.SubgameFactory` (spec
            accounting); one is created if omitted.

    Returns:
        (X, Y, W) float32 arrays of shape [n_beliefs * len(boards), X_DIM /
        Y_DIM / Y_DIM], rows in queue-flush order.
    """
    from poker_ai.rebel.hunl import lazy_subgames as lzs

    if factory is None:
        factory = lzs.SubgameFactory()
    alphas = list(DIRICHLET_ALPHAS)
    results = []
    for board in boards:
        key = sgs.board_key(board)
        n_local = len(sgs.local_hands(key))
        for _ in range(int(n_beliefs)):
            a0 = float(rng.choice(alphas))
            a1 = float(rng.choice(alphas))
            r0_local = rng.dirichlet(np.full(n_local, a0))
            r1_local = rng.dirichlet(np.full(n_local, a1))
            pot, stack0, stack1, first_to_act = config_sampler(rng)
            spec = factory.from_state(
                street="river", board=key, pot=pot, stack0=stack0,
                stack1=stack1, first_to_act=first_to_act,
                r0=sgs.scatter_global(key, r0_local),
                r1=sgs.scatter_global(key, r1_local))
            results.extend(queue.add(spec))
    results.extend(queue.flush_all())

    rows = [result_row(result, scale=scale) for result in results]
    if not rows:
        return (np.zeros((0, X_DIM), np.float32),
                np.zeros((0, Y_DIM), np.float32),
                np.zeros((0, Y_DIM), np.float32))
    X, Y, W = (np.stack(cols, axis=0) for cols in zip(*rows))
    return X, Y, W
