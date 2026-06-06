"""Game-agnostic infoset abstraction (poker_ai/rebel/iig_clustering.py): feature collection + clustering
+ the strategy-incoherence gate work end-to-end on a small game (Goofspiel-3, 90 infosets)."""
import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("sklearn")

from poker_ai.rebel.iig_solve import DepthLimitedGame
from poker_ai.rebel.iig_pbs import load_goofspiel, goofspiel_public_key, goofspiel_is_cut
from poker_ai.rebel.iig_clustering import collect_features, cluster_infosets, strategy_incoherence


def test_clustering_pipeline_smoke():
    g = load_goofspiel(num_cards=3, points_order="random")
    dlg = DepthLimitedGame(g, goofspiel_is_cut, public_key_fn=goofspiel_public_key)
    feats = collect_features(g, dlg)
    assert len(feats) == dlg.n_iset and dlg.n_iset > 0
    # one feature vector per infoset, fixed length
    L = len(next(iter(feats.values())))
    assert all(len(v) == L for v in feats.values())

    cluster_of, n_clusters = cluster_infosets(dlg, feats, ratio=3)
    assert set(cluster_of) == set(feats)
    assert 1 <= n_clusters <= dlg.n_iset

    pol = dlg.cfr_plus(60)
    inc = strategy_incoherence(dlg, pol, cluster_of)
    assert 0.0 <= inc <= 2.0          # within-cluster L1 strategy deviation is a valid simplex distance
