import torch

from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.h2h_transfer_attribution import (
    evaluate_h2h_transfer_attribution,
)


def _write_checkpoint(path, *, seed: int = 0):
    torch.manual_seed(seed)
    net = ValueNetwork(N_FEATURES, 32, N_ACTIONS, n_layers=1)
    torch.save(
        {
            "value_net": net.state_dict(),
            "hidden_dim": 32,
            "n_layers": 1,
            "n_players": 2,
            "initial_chips": 1000,
            "iteration": 3,
            "uses_betting_history": True,
        },
        path,
    )


def test_h2h_transfer_attribution_self_compare_is_neutral_and_records_actions(tmp_path):
    checkpoint = tmp_path / "self.pt"
    _write_checkpoint(checkpoint, seed=13)

    metrics = evaluate_h2h_transfer_attribution(
        candidate_checkpoint=checkpoint,
        baseline_checkpoint=checkpoint,
        n_games=8,
        initial_chips=1000,
        seed=20260521,
        device="cpu",
    )

    assert metrics["mode"] == "duplicate_swapped_transfer_attribution"
    assert metrics["passed"] is True
    assert metrics["avg_chips_per_hand"] == 0.0
    assert metrics["candidate"]["n_actions"] > 0
    assert metrics["baseline"]["n_actions"] > 0
    assert metrics["candidate"]["street_action_counts"]
    assert metrics["baseline"]["first_action_outcomes"]
    assert metrics["promotion"] is False
