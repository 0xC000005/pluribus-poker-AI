"""Lock the small-NLHE OpenSpiel game used by the PPO-vs-R-NaD GO/NO-GO A/B.

Measured values (this session; /tmp/up_final.json) are asserted so a silent OpenSpiel/param change
that alters the game (and thus the exact-NashConv baseline) is caught.
"""
import pytest


def test_small_nlhe_loads_and_matches_measured():
    pytest.importorskip("pyspiel")
    from poker_ai.rnad.small_nlhe import load_small_nlhe
    g = load_small_nlhe()
    assert g.num_distinct_actions() == 4
    assert g.max_game_length() == 7
    assert g.num_players() == 2


def test_small_nlhe_exact_nashconv_tractable():
    pytest.importorskip("pyspiel")
    from open_spiel.python import policy as policy_lib
    from open_spiel.python.algorithms import exploitability
    from poker_ai.rnad.small_nlhe import load_small_nlhe
    g = load_small_nlhe()
    nc = float(exploitability.nash_conv(g, policy_lib.TabularPolicy(g)))
    # uniform-policy NashConv measured at 1.7; assert it's the same game (tight tol).
    assert abs(nc - 1.7) < 1e-3, nc
