import torch

from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.frozen_best_response import (
    FrozenBestResponseConfig,
    build_monte_carlo_transitions,
    run_frozen_best_response,
)


def test_frozen_best_response_smoke_uses_full_deck_contract(tmp_path):
    checkpoint = tmp_path / "frozen.pt"
    net = ValueNetwork(N_FEATURES, hidden_dim=16, output_dim=N_ACTIONS, n_layers=1)
    torch.save(
        {
            "value_net": net.state_dict(),
            "hidden_dim": 16,
            "n_layers": 1,
            "n_players": 2,
            "initial_chips": 1000,
            "iteration": 1,
        },
        checkpoint,
    )

    metrics = run_frozen_best_response(
        FrozenBestResponseConfig(
            checkpoint=str(checkpoint),
            train_episodes=2,
            eval_games=2,
            hidden_dim=16,
            batch_size=8,
            min_buffer_size_to_learn=100,
            device="cpu",
            seed=20260515,
        )
    )

    assert metrics["algorithm"] == "frozen_checkpoint_best_response"
    assert metrics["environment"] == "poker_ai:full_deck_hu_nlhe"
    assert metrics["num_actions"] == 9
    assert metrics["source_checkpoint"] == str(checkpoint)
    assert metrics["train_episodes"] == 2
    assert metrics["eval_games"] == 2
    assert metrics["promotion"] is False


def test_frozen_best_response_cli_builds_config():
    from scripts.run_frozen_best_response import build_config

    cfg = build_config(
        [
            "--checkpoint",
            "models/candidate.pt",
            "--train-episodes",
            "7",
            "--eval-games",
            "3",
            "--hidden-dim",
            "32",
            "--batch-size",
            "16",
            "--br-player",
            "0",
            "--device",
            "cpu",
            "--seed",
            "42",
        ]
    )

    assert cfg.checkpoint == "models/candidate.pt"
    assert cfg.train_episodes == 7
    assert cfg.eval_games == 3
    assert cfg.hidden_dim == 32
    assert cfg.batch_size == 16
    assert cfg.br_player == 0
    assert cfg.device == "cpu"
    assert cfg.seed == 42


def test_monte_carlo_transitions_assign_final_payoff_to_each_br_decision():
    records = [
        (1, torch.ones(N_FEATURES).numpy(), torch.ones(N_ACTIONS).numpy(), 2, True),
        (1, torch.zeros(N_FEATURES).numpy(), torch.ones(N_ACTIONS).numpy(), 4, True),
    ]

    transitions = build_monte_carlo_transitions(records, payoff=0.75)

    assert len(transitions) == 2
    assert all(transition.reward == 0.75 for transition in transitions)
    assert all(transition.done for transition in transitions)
