"""Generic depth-limited solving substrate (poker_ai/rebel/iig_solve.py): the game-agnostic
generalization of leduc.py/loop.py/leaf_eval.py. Headline gate = the CONVENTION ROUND-TRIP (Stage-0,
generalized): valuing the trunk through the exact normalized-PBS leaf reproduces the full-game above-cut
values to machine precision, and the un-normalized convention breaks it. Plus a depth-limited
trunk_solve smoke."""
import pytest

pytest.importorskip("pyspiel")

import numpy as np
import pyspiel

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import leduc_is_cut


def _leduc():
    return DepthLimitedGame(pyspiel.load_game("leduc_poker"), leduc_is_cut)


def test_structure():
    g = _leduc()
    assert sum(g.iset_above_cut) == 36          # round-1 (trunk) infosets
    assert len(g.cut_nodes) == 150              # (c0,c1) board-deal instances
    assert g.n_priv(sorted({n[1] for n in g.cut_nodes})[0], 0) == 6   # 6 private cards/player


def test_convention_roundtrip_normalized_closes():
    """Depth-limited leaf consumption with the normalized PBS convention reproduces the exact
    full-game above-cut values (the generic Stage-0 round-trip)."""
    g = _leduc()
    rng = np.random.default_rng(0)
    for _ in range(3):
        err = g.roundtrip_error(g.random_policy(rng), normalize=True)
        assert err < 1e-12, f"round-trip broke: {err}"


def test_unnormalized_convention_breaks_roundtrip():
    """Negative control: the un-normalized counterfactual value does NOT round-trip (the reason the
    normalized convention is pinned)."""
    g = _leduc()
    rng = np.random.default_rng(1)
    err = g.roundtrip_error(g.random_policy(rng), normalize=False)
    assert err > 0.1


def test_trunk_solve_smoke():
    """Depth-limited trunk_solve with a fixed (blueprint) exact leaf runs and returns valid
    above-cut strategies."""
    g = _leduc()
    leaf = g.blueprint_leaf_fn(g.uniform_policy())
    avg = g.trunk_solve(leaf, iters=80)
    above = [i for i in range(g.n_iset) if g.iset_above_cut[i]]
    assert set(avg.keys()) == set(above)
    for i, pr in avg.items():
        assert abs(pr.sum() - 1.0) < 1e-9 and (pr >= -1e-12).all()


def test_exploitability_infra_and_single_value_trunk_wall():
    """The OpenSpiel NashConv infra works (full-CFR near-eq is ~Nash), and the SINGLE-VALUE depth-limited
    trunk is provably exploitable (the documented trunk wall, reproduced on the generic substrate -- the
    motivation for Build 4 soundness)."""
    g = _leduc()
    cont = g.cfr_plus(300)
    nash_floor = g.nash_conv(cont)
    assert nash_floor < 0.05                       # infra sanity: full CFR+ ~ Nash
    sigma1 = g.trunk_solve(g.blueprint_leaf_fn(cont), iters=200)
    exploit = g.nash_conv(g.assemble(sigma1, cont))
    assert exploit > 0.15                           # single-value trunk is far from Nash (the wall)
    assert exploit > 10 * nash_floor


def test_subgame_equilibrium_solver_smoke():
    """solve_subgame_equilibrium (below-cut re-solve, reusable on >=3-level games) returns valid
    strategies at a cut public state."""
    g = _leduc()
    key = sorted({n[1] for n in g.cut_nodes})[0]
    H = g.n_priv(key, 0)
    import numpy as np
    eq = g.solve_subgame_equilibrium(key, np.ones(H) / H, np.ones(g.n_priv(key, 1)) / g.n_priv(key, 1),
                                     iters=40)
    assert len(eq) > 0
    for pr in eq.values():
        assert abs(pr.sum() - 1.0) < 1e-9 and (pr >= -1e-12).all()
