"""Per-board terminal evaluation for HUNL street subgames (p1_design.json STEP 4).

River showdowns: a per-board RANK VECTOR [H] int32 (4KB/board, the compact
representation) computed through ``scripts/solver.py``'s trusted evaluator path
(``_CARD_TO_EVAL`` + ``_evaluate_seven_eval_cards`` over the same Cactus-Kev
lookup tables ``StreetSolver._compute_showdown_matrices`` uses). W/L/T matrices
are materialized from the rank vector by broadcasting via the equally trusted
``solver._rank_outcome_matrices``; ``valid`` comes from the global-conflict
gather in ``subgame_spec`` -- gate P-D asserts all four exactly equal
StreetSolver's own matrices on real boards.

Turn trees carry NO equity matrices here: their ``showdown`` terminals are
river-deal CUT leaves valued by a pluggable leaf hook (this module defines the
hook contract + the exact-river injection used for target generation and
GO/NO-GO checks now; the PBS river net replaces it later). The hook mirrors
``turn_river.make_exact_river_showdown_fn``'s conventions, including the
net-from-river -> net-from-turn counterfactual offset (turn_river.py L241-242).

``scripts/solver.py`` is a protected surface: imported and reused, never edited.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from poker_ai.rebel.hunl import subgame_spec as sgs  # wires scripts/ via tree_builder

import solver as _solver  # noqa: E402  (protected surface; imported, never edited)

__all__ = [
    "board_rank_vector",
    "showdown_matrices",
    "spec_terminal_matrices",
    "exact_river_cfv_population",
    "make_exact_river_leaf",
]

# Leaf-hook contract (the pluggable river-deal cut valuation; PBS net later).
#
# A leaf hook is ``leaf(ctx) -> (hero_values, villain_values)`` with both
# outputs shaped [B, n_show, H] (numpy, float). It is called by the population
# kernel once per CFR iteration (``ctx["iteration"]`` = 0-based int) and once
# inside the average-strategy root value pass (``ctx["iteration"] is None``),
# mirroring ``fast_cfr.solve_cfr``'s ``showdown_leaf_fn`` semantics: returned
# values REPLACE the default showdown values at ``ctx["showdown_indices"]`` and
# must be counterfactual numerators (opponent-reach-weighted, own reach NOT
# included) in the NET-FROM-STREET-START convention of the solved tree.
#
# ctx keys:
#   iteration            int CFR iteration, or None for the value pass
#   specs                list[SubgameSpec] per batch element (None at the raw
#                        kernel level when no specs were provided)
#   trees                list[dict] per-element tree arrays
#   showdown_indices     [n_show] int node indices (shared topology)
#   hero_reach           [B, n_show, H] float64: range*strategy reach (hr_at)
#   villain_reach        [B, n_show, H] float64 (vr_at)
#   default_hero_values  [B, n_show, H] float64 (zeros when no W/L/T given)
#   default_villain_values  [B, n_show, H] float64
#   valid_ms             list[np.ndarray [H, H] float32] per element
#   pot_starts, hero_stack_starts, villain_stack_starts   [B] float64
#   terminal_pots        [B, n_show] pot at each cut node
#   terminal_stacks_h    [B, n_show] hero stack at each cut node
#   terminal_stacks_v    [B, n_show] villain stack at each cut node


@lru_cache(maxsize=4096)
def _board_rank_vector_cached(key: tuple) -> np.ndarray:
    hands = sgs.local_hands(key)
    board_eval = [int(_solver._CARD_TO_EVAL[c]) for c in key]
    ranks = np.empty(len(hands), dtype=np.int32)
    for i, (c0, c1) in enumerate(hands):
        ranks[i] = _solver._evaluate_seven_eval_cards(
            int(_solver._CARD_TO_EVAL[c0]),
            int(_solver._CARD_TO_EVAL[c1]),
            *board_eval,
        )
    ranks.setflags(write=False)
    return ranks


def board_rank_vector(board) -> np.ndarray:
    """[H=1081] int32 7-card hand rank (lower = stronger) per local river hand.

    Exactly the rank loop of ``StreetSolver._compute_showdown_matrices`` on the
    same evaluator tables -- the compact per-board representation (4KB instead
    of the 18.7MB [H, H] float matrices).
    """
    key = sgs.board_key(board)
    if len(key) != 5:
        raise ValueError(f"river rank vector needs a 5-card board, got {len(key)}")
    return _board_rank_vector_cached(key)


def showdown_matrices(board):
    """(win, lose, tie, valid) float32 [H, H] for a river board.

    W/L/T broadcast from the rank vector via the trusted
    ``solver._rank_outcome_matrices``; ``valid`` from the global conflict
    gather. Bit-identical to StreetSolver's matrices (gate P-D).
    """
    ranks = board_rank_vector(board)
    valid = sgs.local_valid_matrix(board)
    win_m, lose_m, tie_m = _solver._rank_outcome_matrices(ranks, valid)
    return win_m, lose_m, tie_m, valid


def spec_terminal_matrices(spec):
    """Terminal tensors for one SubgameSpec.

    Returns dict(win, lose, tie, valid). River: exact showdown matrices.
    Turn: ``valid`` only (fold terminals); win/lose/tie are None because turn
    ``showdown`` terminals are river-deal cut leaves that MUST be valued by a
    leaf hook.
    """
    if spec.street == "river":
        win_m, lose_m, tie_m, valid = showdown_matrices(spec.board)
        return {"win": win_m, "lose": lose_m, "tie": tie_m, "valid": valid}
    if spec.street == "turn":
        return {"win": None, "lose": None, "tie": None,
                "valid": sgs.local_valid_matrix(spec.board)}
    raise ValueError(
        f"terminal evaluation implemented for turn/river only, got {spec.street!r}")


# ---------------------------------------------------------------------------
# Exact-river injection (the control leaf; PBS net replaces it later)
# ---------------------------------------------------------------------------

def exact_river_cfv_population(
    turn_board,
    pot,
    stack0,
    stack1,
    first_to_act,
    hero_reach,
    villain_reach,
    n_iterations=150,
    dtype=None,
    device="cpu",
    b_max=8,
):
    """Exact river-continuation CFVs at one turn river-deal cut, fused-batched.

    The population twin of ``turn_river.turn_leaf_river_cfv``: the 48 runouts
    share one betting topology (board changes only rank vectors + hand sets),
    so all 48 river subgames solve as one same-topology population via
    ``solve_population``; per-runout per-belief values come from the batched
    root value pass. Ranges are mapped turn-local -> global -> river-local
    (card-removal masking is implicit: hands containing the runout card simply
    do not exist in that river's local hand set).

    Returns ``(hero_cfv, villain_cfv)`` per TURN-local hand in the
    NET-FROM-RIVER convention (they award the cut pot), averaged over the 44
    legal runouts per valid pair (52 - 4 board - 2 - 2; the per-pair exclusion
    zeroes the rest automatically -- exactly the reference's /44 mapping).
    """
    import torch

    from poker_ai.rebel.hunl import lazy_subgames as _lzs
    from poker_ai.rebel.hunl import population_solver as _pop

    key = sgs.board_key(turn_board)
    if len(key) != 4:
        raise ValueError(f"turn board must have 4 cards, got {len(key)}")
    if dtype is None:
        dtype = torch.float64
    n_turn = len(sgs.local_hands(key))
    hero_reach = np.asarray(hero_reach, dtype=np.float64)
    villain_reach = np.asarray(villain_reach, dtype=np.float64)
    if hero_reach.shape != (n_turn,) or villain_reach.shape != (n_turn,):
        raise ValueError(
            f"reaches must have shape ({n_turn},), got "
            f"{hero_reach.shape} / {villain_reach.shape}")

    # The runout expansion + card-removal masking is SHARED with
    # SubgameFactory.expand_to_river (lazy_subgames.river_runout_specs).
    specs = _lzs.river_runout_specs(
        key, pot, stack0, stack1, first_to_act,
        sgs.scatter_global(key, hero_reach),
        sgs.scatter_global(key, villain_reach))
    results = _pop.solve_population(
        specs, n_iterations=n_iterations, dtype=dtype, device=device, b_max=b_max)

    l2g_turn = sgs.local_to_global(key)
    cut_h = np.zeros(n_turn, dtype=np.float64)
    cut_v = np.zeros(n_turn, dtype=np.float64)
    for spec, res in zip(specs, results):
        cut_h += sgs.scatter_global(spec.board, res.v0)[l2g_turn]
        cut_v += sgs.scatter_global(spec.board, res.v1)[l2g_turn]
    # 44 legal runouts per valid (a, b) pair: 52 - 4 board - 2 hero - 2 villain
    # (turn_leaf_river_cfv's exact normalization).
    cut_h /= 44.0
    cut_v /= 44.0
    return cut_h, cut_v


def make_exact_river_leaf(n_iterations=150, dtype=None, device="cpu", b_max=8):
    """Leaf hook valuing turn river-deal cuts by exact river re-solving.

    The batched twin of ``turn_river.make_exact_river_showdown_fn``: per cut
    node, solve the 48-runout river population at the cut's (pot, stacks) and
    current reaches, then convert net-from-river -> net-from-turn-start by
    subtracting the hi_cut/vi_cut counterfactual offsets (turn_river.py
    L241-242; the offset differs per cut node so it does NOT cancel in
    regrets). Per-iteration use is exact-but-slow by design (use on tiny spots
    or for offline target generation; the PBS net is the fast replacement).
    """

    def leaf(ctx):
        specs = ctx["specs"]
        if specs is None:
            raise ValueError("exact-river leaf needs ctx['specs'] (boards/streets)")
        hero_reach = np.asarray(ctx["hero_reach"], dtype=np.float64)
        villain_reach = np.asarray(ctx["villain_reach"], dtype=np.float64)
        terminal_pots = np.asarray(ctx["terminal_pots"])
        terminal_sh = np.asarray(ctx["terminal_stacks_h"])
        terminal_sv = np.asarray(ctx["terminal_stacks_v"])
        hero_starts = np.asarray(ctx["hero_stack_starts"])
        villain_starts = np.asarray(ctx["villain_stack_starts"])

        out_h = np.zeros_like(hero_reach)
        out_v = np.zeros_like(villain_reach)
        for b, spec in enumerate(specs):
            if spec.street != "turn":
                raise ValueError(
                    f"exact-river leaf applies to turn trees, got {spec.street!r}")
            valid64 = sgs.local_valid_matrix(spec.board).astype(np.float64)
            for k in range(hero_reach.shape[1]):
                hr = hero_reach[b, k]
                vr = villain_reach[b, k]
                pot_cut = int(terminal_pots[b, k])
                hs_cut = int(terminal_sh[b, k])
                vs_cut = int(terminal_sv[b, k])
                hi_cut = float(hero_starts[b]) - hs_cut
                vi_cut = float(villain_starts[b]) - vs_cut
                cut_h, cut_v = exact_river_cfv_population(
                    spec.board, pot_cut, hs_cut, vs_cut, spec.first_to_act,
                    hr, vr, n_iterations=n_iterations,
                    dtype=dtype, device=device, b_max=b_max)
                # net-from-river -> net-from-turn-start (turn_river.py L241-242)
                out_h[b, k] = cut_h - hi_cut * (vr @ valid64.T)
                out_v[b, k] = cut_v - vi_cut * (hr @ valid64)
        return out_h, out_v

    return leaf
