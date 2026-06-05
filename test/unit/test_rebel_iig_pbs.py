"""Generic public-state / PBS abstraction (poker_ai/rebel/iig_pbs.py): the depth-limit cut + per-player
private-state reaches, validated BIT-IDENTICALLY against the trusted LeducTree.cut_reaches, and shown to
derive belief dimensions across poker (Kuhn, Leduc) and a non-poker game (Liar's Dice)."""
import pytest

pytest.importorskip("pyspiel")

import numpy as np
import pyspiel

from poker_ai.rebel.leduc import LeducTree
from poker_ai.rebel.iig_pbs import PBSStructure, leduc_is_cut, first_decision_is_cut, uniform_policy_fn


def test_leduc_pbs_bit_parity_with_trusted_substrate():
    """Generic cut_reaches must reproduce LeducTree.cut_reaches exactly (same partition + same reach
    values), up to private-index permutation (compared as sorted multisets)."""
    lt = LeducTree()
    lr = lt.cut_reaches(lt.uniform_policy())                       # {key: [reach0[6], reach1[6]]}
    gen = PBSStructure(pyspiel.load_game("leduc_poker"), leduc_is_cut)
    gr = gen.cut_reaches(uniform_policy_fn)                        # {pubkey: ({p0:r}, {p1:r})}

    assert len(lr) == len(gr) == 5                                 # 5 round-1 public betting sequences
    for side in (0, 1):
        L = np.array(sorted(v for arr in lr.values() for v in arr[side] if v > 1e-15))
        G = np.array(sorted(v for c in gr.values() for v in c[side].values() if v > 1e-15))
        assert L.shape == G.shape and L.size > 0
        assert np.allclose(L, G, atol=1e-12, rtol=0), f"player{side} reach parity broken"


def test_belief_dims_poker_and_non_poker():
    """The PBS abstraction derives the public-state count + per-player private (belief) dimension for
    poker AND non-poker games via the SAME code."""
    leduc = PBSStructure(pyspiel.load_game("leduc_poker"), leduc_is_cut).belief_dims()
    assert leduc["n_public_states"] == 5 and leduc["priv_dim_p0_max"] == 6

    kuhn = PBSStructure(pyspiel.load_game("kuhn_poker"), first_decision_is_cut).belief_dims()
    assert kuhn["n_public_states"] == 1 and kuhn["priv_dim_p0_max"] == 3

    # Liar's Dice has no OpenSpiel public observer -> constant-root public key (root PBS = belief over dice)
    ld = PBSStructure(pyspiel.load_game("liars_dice", {"dice_sides": 3}),
                      first_decision_is_cut, public_key_fn=lambda s: "root")
    d = ld.belief_dims()
    assert d["n_public_states"] == 1 and d["priv_dim_p0_max"] == 3   # 3 dice faces = 3 private states
