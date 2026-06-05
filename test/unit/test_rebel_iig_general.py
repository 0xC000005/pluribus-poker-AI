"""Generic OpenSpiel-backed IIG solver: the SAME CFR+ code reaches exact Nash on Kuhn, Leduc (with a
PARITY gate vs the trusted LeducTree), and Liar's Dice (a non-poker game) -- the "same computation
solves any imperfect-info game" foundation."""
import pytest

pytest.importorskip("pyspiel")

from poker_ai.rebel.iig import PublicTree
from poker_ai.rebel.leduc import LeducTree


def test_tree_structure():
    t = PublicTree("kuhn_poker")
    assert t.game.num_players() == 2
    assert t.n_iset == 12               # Kuhn has 12 information sets
    assert t.root[0] in ("term", "chance", "dec")
    assert t.n_nodes > t.n_iset


def test_kuhn_reaches_nash():
    t = PublicTree("kuhn_poker")
    avg, _ = t.cfr_plus(400, averaging="linear")
    assert t.nash_conv(avg) < 0.001     # base case: one-round game, full-tree CFR


def test_leduc_parity_with_trusted_substrate():
    """The generic solver must EXACTLY reproduce the trusted LeducTree.cfr_plus (identical math:
    private deals + the public cut are both just 'chance')."""
    it = 150
    g = PublicTree("leduc_poker")
    gnc = g.nash_conv(g.cfr_plus(it, averaging="linear")[0])
    l = LeducTree()
    lnc = l.nash_conv(l.cfr_plus(it, averaging="linear")[0])
    assert abs(gnc - lnc) < 1e-9, f"parity broken: generic {gnc} vs leduc {lnc}"
    assert gnc < 0.05                    # both converging


def test_liars_dice_generality_non_poker():
    """SAME code on a NON-poker game: small Liar's Dice converges toward Nash."""
    t = PublicTree("liars_dice", {"dice_sides": 3})
    avg, hist = t.cfr_plus(80, eval_every=40, averaging="linear")
    assert hist[-1][1] < hist[0][1]     # exploitability decreasing
    assert t.nash_conv(avg) < 0.05      # near-Nash with the generic solver
