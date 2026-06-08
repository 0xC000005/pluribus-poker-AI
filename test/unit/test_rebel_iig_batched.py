"""G1 parity: the cross-key batched subgame solver (poker_ai/rebel/iig_batched) must reproduce the trusted
per-key numpy DepthLimitedGame.solve_subgame_equilibrium to float64 precision, across poker + non-poker
games with chance, variable depth, and imperfect-info repeated infosets. CPU device for deterministic CI."""
import numpy as np
import pyspiel
import pytest

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import (leduc_is_cut, first_decision_is_cut, load_goofspiel,
                                     goofspiel_is_cut, goofspiel_public_key)
from poker_ai.rebel.iig_batched import solve_all_keys, solve_all_keys_soa

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
