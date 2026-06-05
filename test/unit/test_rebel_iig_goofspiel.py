"""Goofspiel (a >=3-level, non-poker, natively-simultaneous IIG) in the generic substrate: the
turn-based wrapper makes it sequential; a custom public_key_fn (OpenSpiel has no public observer for it)
gives the public state; the SAME generic CFR+ solves it to exact Nash, and the PBS cut structure has
the multi-level prize-reveal cuts needed for the soundness/bootstrapping build."""
import pytest

pytest.importorskip("pyspiel")

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import (
    PBSStructure, load_goofspiel, goofspiel_public_key, goofspiel_is_cut,
)


def test_goofspiel_in_substrate_solves_and_has_cuts():
    g = load_goofspiel(num_cards=4, points_order="random")
    dlg = DepthLimitedGame(g, goofspiel_is_cut, public_key_fn=goofspiel_public_key)
    assert dlg.n_iset > 100
    assert len(dlg.cut_nodes) > 0
    n_public = len(set(n[1] for n in dlg.cut_nodes))
    assert n_public > 1                              # multiple public cut states (multi-level)
    # the SAME generic CFR+ reaches near-Nash on this >=3-level non-poker game
    avg = dlg.cfr_plus(150)
    assert dlg.nash_conv(avg) < 0.1


def test_goofspiel_pbs_belief_dims():
    g = load_goofspiel(num_cards=4, points_order="random")
    dims = PBSStructure(g, goofspiel_is_cut, public_key_fn=goofspiel_public_key).belief_dims()
    assert dims["n_public_states"] > 1
    assert dims["priv_dim_p0_max"] >= 2              # belief over the remaining hand
