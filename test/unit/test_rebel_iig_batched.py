"""G1 parity: the cross-key batched subgame solver (poker_ai/rebel/iig_batched) must reproduce the trusted
per-key numpy DepthLimitedGame.solve_subgame_equilibrium to float64 precision, across poker + non-poker
games with chance, variable depth, and imperfect-info repeated infosets. CPU device for deterministic CI."""
import numpy as np
import pyspiel
import pytest

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import (leduc_is_cut, first_decision_is_cut, load_goofspiel,
                                     goofspiel_is_cut, goofspiel_public_key)
from poker_ai.rebel.iig_batched import solve_all_keys, solve_all_keys_soa, trunk_solve_batched

ITERS = 60


def _cases():
    return [
        ("leduc", pyspiel.load_game("leduc_poker"), leduc_is_cut, None),
        ("goofspiel4", load_goofspiel(4), goofspiel_is_cut, goofspiel_public_key),
        ("liars_dice", pyspiel.load_game("liars_dice", {"dice_sides": 2}),
         first_decision_is_cut, (lambda s: "root")),
    ]


@pytest.mark.parametrize("name,game,cut,pkf", _cases(), ids=lambda c: c if isinstance(c, str) else "")
def test_cross_key_batched_matches_numpy(name, game, cut, pkf):
    dlg = DepthLimitedGame(game, cut, public_key_fn=pkf)
    keys = sorted({n[1] for n in dlg.cut_nodes})
    rng = np.random.default_rng(7)
    ranges = {k: (rng.dirichlet(np.ones(dlg.n_priv(k, 0))), rng.dirichlet(np.ones(dlg.n_priv(k, 1))))
              for k in keys}

    got = solve_all_keys(dlg, ranges, ITERS, device="cpu")
    got_soa = solve_all_keys_soa(dlg, ranges, ITERS, device="cpu")  # level-grouped flat-SoA kernel

    max_l1 = max_l1_soa = 0.0
    for k in keys:
        ref = dlg.solve_subgame_equilibrium(k, ranges[k][0], ranges[k][1], iters=ITERS)
        for iid in ref:
            r = np.asarray(ref[iid])
            max_l1 = max(max_l1, float(np.abs(r - np.asarray(got[k][iid])).sum()))
            max_l1_soa = max(max_l1_soa, float(np.abs(r - np.asarray(got_soa[k][iid])).sum()))
    assert max_l1 < 1e-4, f"{name}: cross-key walk vs numpy max L1 {max_l1:.2e}"
    assert max_l1_soa < 1e-4, f"{name}: cross-key SoA kernel vs numpy max L1 {max_l1_soa:.2e}"


def test_trunk_solve_batched_matches_serial_resolve():
    """G3: the GPU batched solver used as the per-belief re-solve leaf in the depth-limited trunk solve
    reproduces the serial per_belief_equilibrium_leaf_fn trunk strategy (CPU exact)."""
    dlg = DepthLimitedGame(pyspiel.load_game("leduc_poker"), leduc_is_cut)
    ti, si = 8, 40
    serial = dlg.trunk_solve(dlg.per_belief_equilibrium_leaf_fn(iters=si), iters=ti)
    batched = trunk_solve_batched(dlg, subgame_iters=si, iters=ti, device="cpu")
    max_l1 = max(float(np.abs(np.asarray(serial[i]) - np.asarray(batched[i])).sum()) for i in serial)
    assert max_l1 < 1e-4, f"trunk_solve_batched vs serial re-solve max sigma1 L1 {max_l1:.2e}"
