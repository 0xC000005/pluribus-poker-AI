"""Generic depth-limited PBS-net self-play loop (poker_ai/rebel/iig_selfplay.py): the game-agnostic
ReBeL training loop. On Leduc (a 2-level game) it validates net-LEARNABILITY + leaf-amortization
(net-leaf sigma1 tracks the exact-leaf control); bootstrapping needs a >=3-level game (Step-3 finding)."""
import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

import pyspiel

from poker_ai.rebel.iig_selfplay import self_play
from poker_ai.rebel.iig_pbs import leduc_is_cut


def test_self_play_loop_learns_on_leduc():
    hist, net, dlg = self_play(
        pyspiel.load_game("leduc_poker"), leduc_is_cut,
        n_iters=2, trunk_iters=60, coverage_per_cut=60, train_epochs=150,
        cont_iters=200, hidden=64, seed=0, verbose=False)
    assert len(hist) == 2
    # the net learns the leaf value (reach-weighted MAE well below the value scale)
    assert hist[-1]["mae_frac"] < 0.15
    # net-leaf sigma1 tracks the exact-leaf control (finite, bounded well under a full simplex L1)
    assert 0.0 <= hist[-1]["sigma1_l1_vs_exact"] < 1.0
    # learnability improved over the loop
    assert hist[-1]["mae_frac"] <= hist[0]["mae_frac"] + 1e-6
