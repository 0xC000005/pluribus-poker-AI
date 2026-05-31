import torch

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.native_nfsp import _MLP
from poker_ai.research.native_ppo_average_policy_fit import diagnose_average_policy_fit


def test_average_policy_fit_identical_actor_average_positive_control(tmp_path):
    checkpoint = tmp_path / "same_actor_avg.pt"
    net = _MLP(hidden_dim=16)
    torch.save(
        {
            "algorithm": "native_ppo_policy",
            "environment": "poker_ai:full_deck_hu_nlhe",
            "num_actions": N_ACTIONS,
            "num_features": N_FEATURES,
            "hidden_dim": 16,
            "policy_net_state_dict": net.state_dict(),
            "avg_net_state_dict": net.state_dict(),
            "config": {
                "fsp_average_policy": True,
                "initial_chips": 1000,
                "max_steps_per_hand": 16,
            },
        },
        checkpoint,
    )

    metrics = diagnose_average_policy_fit(
        checkpoint,
        n_games=2,
        device="cpu",
        seed=20260521,
    )

    assert metrics["algorithm"] == "native_ppo_average_policy_fit"
    assert metrics["n_states"] > 0
    assert metrics["mean_l1_actor_to_average"] < 1e-6
    assert metrics["mean_kl_actor_to_average"] < 1e-6
    assert metrics["top_action_agreement"] == 1.0
    assert metrics["promotion"] is False
