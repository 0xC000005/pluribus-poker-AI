import numpy as np
import torch

from poker_ai.games.full_deck.state import N_ACTIONS
from poker_ai.research.native_regularized_policy import (
    NativeRegularizedPolicyConfig,
    evaluate_native_regularized_policy_head_to_head,
    run_native_regularized_policy_pilot,
    sampled_regularized_policy_target,
)


def test_sampled_regularized_policy_target_is_legal_and_reward_sensitive():
    current = np.array([0.30, 0.30, 0.40], dtype=np.float32)
    legal_mask = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    target = sampled_regularized_policy_target(
        current,
        legal_mask,
        action=1,
        payoff=1.0,
        step_size=0.5,
        regularization_strength=0.1,
    )

    assert target[2] == 0.0
    np.testing.assert_allclose(target.sum(), 1.0)
    assert target[1] > current[1]


def test_run_native_regularized_policy_pilot_smoke_uses_full_deck_contract(tmp_path):
    checkpoint_path = tmp_path / "native_regularized_policy.pt"

    metrics = run_native_regularized_policy_pilot(
        NativeRegularizedPolicyConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=8,
            min_buffer_size_to_learn=100,
            device="cpu",
            seed=2026,
            checkpoint_path=str(checkpoint_path),
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["algorithm"] == "native_regularized_policy"
    assert metrics["num_actions"] == N_ACTIONS
    assert metrics["resolved_device"] == "cpu"
    assert metrics["promotion"] is False
    assert payload["algorithm"] == "native_regularized_policy"
    assert "policy_net_state_dict" in payload


def test_native_regularized_policy_self_h2h_is_zero(tmp_path):
    checkpoint_path = tmp_path / "native_regularized_policy.pt"
    run_native_regularized_policy_pilot(
        NativeRegularizedPolicyConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=8,
            min_buffer_size_to_learn=100,
            device="cpu",
            seed=2027,
            checkpoint_path=str(checkpoint_path),
        )
    )

    metrics = evaluate_native_regularized_policy_head_to_head(
        str(checkpoint_path),
        str(checkpoint_path),
        n_games=2,
        device="cpu",
        seed=2028,
    )

    assert metrics["algorithm"] == "native_regularized_policy_h2h"
    assert metrics["n_games"] == 2
    assert abs(metrics["mean_candidate_payoff"]) < 1e-6
    assert abs(metrics["lower95_candidate_payoff"]) < 1e-6
    assert abs(metrics["upper95_candidate_payoff"]) < 1e-6
    assert metrics["promotion"] is False
