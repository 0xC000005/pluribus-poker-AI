import numpy as np
import torch

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
import poker_ai.research.native_ppo_policy as native_ppo_policy
from poker_ai.research.native_ppo_policy import (
    NativePPOConfig,
    _DecisionRecord,
    _train_ppo_batch,
    q_boosted_behavior_policy,
    _policy_surrogate_loss,
    evaluate_native_ppo_policy_head_to_head,
    run_native_ppo_policy_pilot,
)


def test_run_native_ppo_policy_pilot_smoke_uses_full_deck_policy_only_contract(tmp_path):
    checkpoint_path = tmp_path / "native_ppo_policy.pt"

    metrics = run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=8,
            rollout_episodes_per_update=1,
            ppo_epochs=1,
            device="cpu",
            seed=2026,
            checkpoint_path=str(checkpoint_path),
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["algorithm"] == "native_ppo_policy"
    assert metrics["role"] == "policy_only_self_play_falsifier"
    assert metrics["num_actions"] == N_ACTIONS
    assert metrics["uses_resolver"] is False
    assert metrics["uses_explicit_belief"] is False
    assert metrics["promotion"] is False
    assert metrics["advantage_mode"] == "value"
    assert payload["algorithm"] == "native_ppo_policy"
    assert payload["config"]["rollout_episodes_per_update"] == 1
    assert payload["config"]["advantage_mode"] == "value"
    assert payload["config"].get("fsp_average_policy", False) is False
    assert "policy_net_state_dict" in payload
    assert "value_net_state_dict" in payload


def test_native_ppo_policy_self_h2h_is_zero(tmp_path):
    checkpoint_path = tmp_path / "native_ppo_policy.pt"
    run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=8,
            ppo_epochs=1,
            device="cpu",
            seed=2027,
            checkpoint_path=str(checkpoint_path),
        )
    )

    metrics = evaluate_native_ppo_policy_head_to_head(
        str(checkpoint_path),
        str(checkpoint_path),
        n_games=2,
        device="cpu",
        seed=2028,
    )

    assert metrics["algorithm"] == "native_policy_h2h"
    assert metrics["n_games"] == 2
    assert abs(metrics["mean_candidate_payoff"]) < 1e-6
    assert abs(metrics["lower95_candidate_payoff"]) < 1e-6
    assert abs(metrics["upper95_candidate_payoff"]) < 1e-6
    assert metrics["promotion"] is False


def test_native_ppo_policy_batches_rollouts_before_update(tmp_path):
    checkpoint_path = tmp_path / "native_ppo_policy.pt"

    metrics = run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=4,
            eval_games=1,
            hidden_dim=16,
            batch_size=10000,
            rollout_episodes_per_update=2,
            ppo_epochs=1,
            device="cpu",
            seed=2030,
            checkpoint_path=str(checkpoint_path),
        )
    )

    assert metrics["rollout_episodes_per_update"] == 2
    assert metrics["rollout_updates"] == 2
    assert metrics["policy_updates"] == 2


def test_native_ppo_policy_q_expected_mc_advantage_smoke(tmp_path):
    checkpoint_path = tmp_path / "native_ppo_q_policy.pt"

    metrics = run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=32,
            rollout_episodes_per_update=2,
            ppo_epochs=1,
            advantage_mode="q_expected_mc",
            device="cpu",
            seed=2031,
            checkpoint_path=str(checkpoint_path),
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["advantage_mode"] == "q_expected_mc"
    assert payload["config"]["advantage_mode"] == "q_expected_mc"
    assert "q_net_state_dict" in payload


def test_native_ppo_policy_q_expected_lambda_advantage_smoke(tmp_path):
    checkpoint_path = tmp_path / "native_ppo_q_lambda_policy.pt"

    metrics = run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=32,
            rollout_episodes_per_update=2,
            ppo_epochs=1,
            advantage_mode="q_expected_lambda",
            trace_lambda=0.5,
            device="cpu",
            seed=2032,
            checkpoint_path=str(checkpoint_path),
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["advantage_mode"] == "q_expected_lambda"
    assert metrics["trace_lambda"] == 0.5
    assert payload["config"]["advantage_mode"] == "q_expected_lambda"
    assert payload["config"]["trace_lambda"] == 0.5
    assert "q_net_state_dict" in payload


def test_native_ppo_policy_q_expected_lambda_target_advantage_smoke(tmp_path):
    checkpoint_path = tmp_path / "native_ppo_q_lambda_target_policy.pt"

    metrics = run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=32,
            rollout_episodes_per_update=2,
            ppo_epochs=1,
            advantage_mode="q_expected_lambda_target",
            trace_lambda=0.5,
            device="cpu",
            seed=2033,
            checkpoint_path=str(checkpoint_path),
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["advantage_mode"] == "q_expected_lambda_target"
    assert metrics["uses_target_q"] is True
    assert payload["config"]["advantage_mode"] == "q_expected_lambda_target"
    assert payload["config"]["uses_target_q"] is True
    assert "q_net_state_dict" in payload
    assert "target_q_net_state_dict" in payload


def test_q_boosted_behavior_policy_moves_toward_high_q_legal_action():
    policy_probs = torch.tensor([0.8, 0.1, 0.1, 0.0])
    q_values = torch.tensor([0.0, 3.0, -1.0, 100.0])
    legal_mask = torch.tensor([1.0, 1.0, 1.0, 0.0])

    boosted = q_boosted_behavior_policy(
        policy_probs,
        q_values,
        legal_mask,
        beta=3.0,
    )

    assert torch.isclose(boosted.sum(), torch.tensor(1.0))
    assert boosted[3] == 0.0
    assert boosted[1] > policy_probs[1]
    assert int(torch.argmax(boosted)) == 1


def test_native_ppo_policy_q_boosted_behavior_exports_config(tmp_path):
    checkpoint_path = tmp_path / "native_ppo_q_boosted_policy.pt"

    metrics = run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=32,
            rollout_episodes_per_update=2,
            ppo_epochs=1,
            advantage_mode="q_expected_lambda",
            behavior_mode="q_boosted",
            q_boost_beta=0.75,
            device="cpu",
            seed=2040,
            checkpoint_path=str(checkpoint_path),
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["advantage_mode"] == "q_expected_lambda"
    assert metrics["behavior_mode"] == "q_boosted"
    assert metrics["q_boost_beta"] == 0.75
    assert payload["config"]["behavior_mode"] == "q_boosted"
    assert payload["config"]["q_boost_beta"] == 0.75
    assert "q_net_state_dict" in payload


def test_q_expected_ppo_uses_rollout_frozen_expected_q_baseline():
    feature_dim = 4
    policy_net = torch.nn.Linear(feature_dim, N_ACTIONS)
    value_net = torch.nn.Linear(feature_dim, 1)
    q_net = torch.nn.Linear(feature_dim, N_ACTIONS)
    with torch.no_grad():
        policy_net.weight.zero_()
        policy_net.bias.zero_()
        value_net.weight.zero_()
        value_net.bias.zero_()
        q_net.weight.zero_()
        q_net.bias.zero_()
        q_net.bias[0] = 20.0

    optimizer = torch.optim.SGD(
        list(policy_net.parameters()) + list(value_net.parameters()) + list(q_net.parameters()),
        lr=0.1,
    )
    record = _DecisionRecord(
        player=0,
        features=torch.zeros(feature_dim).numpy(),
        critic_features=torch.zeros(feature_dim).numpy(),
        legal_mask=torch.tensor([1.0, 1.0] + [0.0] * (N_ACTIONS - 2)).numpy(),
        action=0,
        old_log_prob=float(torch.log(torch.tensor(0.5))),
        value=0.0,
        expected_q=0.0,
    )

    updates, _ = _train_ppo_batch(
        policy_net,
        value_net,
        q_net,
        optimizer,
        [record],
        torch.tensor([1.0]).numpy(),
        NativePPOConfig(
            hidden_dim=16,
            batch_size=1,
            ppo_epochs=1,
            advantage_mode="q_expected_lambda",
            value_loss_weight=0.0,
            entropy_weight=0.0,
        ),
        np.random.default_rng(7),
        torch.device("cpu"),
    )

    assert updates == 1
    with torch.no_grad():
        logits = policy_net(torch.zeros(feature_dim))
    assert logits[0] > logits[1]


def test_native_ppo_policy_centralized_q_critic_keeps_actor_observation_contract(tmp_path):
    checkpoint_path = tmp_path / "native_ppo_centralized_q_policy.pt"

    metrics = run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=32,
            rollout_episodes_per_update=2,
            ppo_epochs=1,
            advantage_mode="q_expected_lambda",
            centralized_q_critic=True,
            device="cpu",
            seed=2036,
            checkpoint_path=str(checkpoint_path),
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["uses_centralized_q_critic"] is True
    assert metrics["q_critic_feature_dim"] > payload["num_features"]
    assert payload["num_features"] == 126
    assert payload["config"]["centralized_q_critic"] is True
    assert payload["config"]["q_critic_feature_dim"] == metrics["q_critic_feature_dim"]
    assert "q_net_state_dict" in payload


def test_native_ppo_policy_fsp_average_policy_exports_average_net(tmp_path):
    checkpoint_path = tmp_path / "native_ppo_fsp_policy.pt"

    metrics = run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=4,
            eval_games=1,
            hidden_dim=16,
            batch_size=32,
            rollout_episodes_per_update=2,
            ppo_epochs=1,
            advantage_mode="q_expected_lambda",
            fsp_average_policy=True,
            average_policy_batch_size=32,
            device="cpu",
            seed=2034,
            checkpoint_path=str(checkpoint_path),
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["fsp_average_policy"] is True
    assert metrics["average_policy_buffer_size"] > 0
    assert metrics["average_policy_updates"] == metrics["rollout_updates"]
    assert payload["config"]["fsp_average_policy"] is True
    assert "policy_net_state_dict" in payload
    assert "avg_net_state_dict" in payload

    h2h = evaluate_native_ppo_policy_head_to_head(
        str(checkpoint_path),
        str(checkpoint_path),
        n_games=2,
        device="cpu",
        seed=2035,
    )
    assert h2h["candidate_strategy_source"] == "auto"
    assert abs(h2h["mean_candidate_payoff"]) < 1e-6


def test_native_ppo_policy_historical_opponent_pool_exports_config(tmp_path):
    checkpoint_path = tmp_path / "native_ppo_historical_policy.pt"

    metrics = run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=4,
            eval_games=1,
            hidden_dim=16,
            batch_size=32,
            rollout_episodes_per_update=2,
            ppo_epochs=1,
            advantage_mode="q_expected_lambda",
            feature_mode="raw_sequence",
            historical_opponent_interval=1,
            historical_opponent_capacity=2,
            device="cpu",
            seed=2037,
            checkpoint_path=str(checkpoint_path),
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["uses_historical_opponents"] is True
    assert metrics["historical_opponent_interval"] == 1
    assert metrics["historical_opponent_capacity"] == 2
    assert metrics["historical_policy_pool_size"] == 2
    assert metrics["historical_snapshots_created"] >= 2
    assert payload["config"]["feature_mode"] == "raw_sequence"
    assert payload["config"]["historical_opponent_interval"] == 1
    assert payload["config"]["historical_opponent_capacity"] == 2


def test_native_ppo_policy_k_best_historical_opponent_pool_exports_config(tmp_path):
    checkpoint_path = tmp_path / "native_ppo_k_best_historical_policy.pt"

    metrics = run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=4,
            eval_games=1,
            hidden_dim=16,
            batch_size=32,
            rollout_episodes_per_update=2,
            ppo_epochs=1,
            advantage_mode="q_expected_lambda",
            feature_mode="raw_sequence",
            historical_opponent_interval=1,
            historical_opponent_capacity=2,
            historical_opponent_selection="k_best",
            device="cpu",
            seed=2038,
            checkpoint_path=str(checkpoint_path),
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["historical_opponent_selection"] == "k_best"
    assert metrics["historical_policy_pool_size"] == 2
    assert payload["config"]["historical_opponent_selection"] == "k_best"


def test_native_ppo_policy_external_native_opponent_exports_config(tmp_path):
    opponent_path = tmp_path / "opponent.pt"
    checkpoint_path = tmp_path / "native_ppo_external_opponent_policy.pt"
    run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=8,
            ppo_epochs=1,
            device="cpu",
            seed=2039,
            checkpoint_path=str(opponent_path),
        )
    )

    metrics = run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=32,
            rollout_episodes_per_update=2,
            ppo_epochs=1,
            advantage_mode="q_expected_lambda",
            feature_mode="raw_sequence",
            external_opponent_checkpoint=str(opponent_path),
            external_opponent_kind="native-ppo",
            device="cpu",
            seed=2041,
            checkpoint_path=str(checkpoint_path),
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["uses_external_opponent"] is True
    assert metrics["external_opponent_checkpoint"] == str(opponent_path)
    assert metrics["external_opponent_kind"] == "native-ppo"
    assert metrics["external_opponent_algorithm"] == "native_ppo_policy"
    assert metrics["uses_historical_opponents"] is False
    assert metrics["fsp_average_policy"] is False
    assert payload["config"]["uses_external_opponent"] is True
    assert payload["config"]["external_opponent_checkpoint"] == str(opponent_path)
    assert payload["config"]["external_opponent_kind"] == "native-ppo"


def test_native_ppo_policy_external_rainbow_response_opponent_exports_config(tmp_path):
    from poker_ai.research.native_ppo_policy import _RainbowDistributionNet

    opponent_path = tmp_path / "compiled_rainbow_response.pt"
    checkpoint_path = tmp_path / "native_ppo_external_rainbow_policy.pt"
    model = _RainbowDistributionNet(
        hidden_dim=8,
        num_atoms=3,
        device=torch.device("cpu"),
    )
    torch.save(
        {
            "algorithm": "compiled_tianshou_rainbow_response_oracle",
            "environment": "poker_ai:full_deck_hu_nlhe",
            "num_actions": N_ACTIONS,
            "num_features": N_FEATURES,
            "hidden_dim": 8,
            "num_atoms": 3,
            "model_state_dict": model.state_dict(),
        },
        opponent_path,
    )

    metrics = run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=32,
            rollout_episodes_per_update=2,
            ppo_epochs=1,
            advantage_mode="q_expected_lambda",
            feature_mode="raw_sequence",
            external_opponent_checkpoint=str(opponent_path),
            external_opponent_kind="auto",
            device="cpu",
            seed=2042,
            checkpoint_path=str(checkpoint_path),
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["uses_external_opponent"] is True
    assert metrics["external_opponent_kind"] == "tianshou-rainbow"
    assert metrics["external_opponent_algorithm"] == "compiled_tianshou_rainbow_response_oracle"
    assert payload["config"]["external_opponent_kind"] == "tianshou-rainbow"


def test_native_ppo_policy_external_opponent_rejects_combined_modes(tmp_path):
    opponent_path = tmp_path / "missing.pt"

    with np.testing.assert_raises_regex(
        ValueError,
        "external_opponent_checkpoint cannot be combined with fsp_average_policy",
    ):
        run_native_ppo_policy_pilot(
            NativePPOConfig(
                train_episodes=1,
                eval_games=1,
                external_opponent_checkpoint=str(opponent_path),
                fsp_average_policy=True,
                device="cpu",
            )
        )


def test_trinal_clip_caps_large_negative_advantage_policy_loss():
    ratios = torch.tensor([10.0])
    advantages = torch.tensor([-1.0])

    standard = _policy_surrogate_loss(
        ratios,
        advantages,
        clip_epsilon=0.2,
        loss_mode="standard",
    )
    trinal = _policy_surrogate_loss(
        ratios,
        advantages,
        clip_epsilon=0.2,
        loss_mode="trinal_clip",
    )

    assert float(standard) == 10.0
    assert float(trinal) == 3.0


def test_native_ppo_policy_trinal_clip_exports_config(tmp_path):
    checkpoint_path = tmp_path / "native_ppo_trinal_policy.pt"

    metrics = run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=32,
            rollout_episodes_per_update=2,
            ppo_epochs=1,
            advantage_mode="q_expected_lambda",
            ppo_loss_mode="trinal_clip",
            feature_mode="raw_sequence",
            device="cpu",
            seed=2039,
            checkpoint_path=str(checkpoint_path),
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["ppo_loss_mode"] == "trinal_clip"
    assert payload["config"]["ppo_loss_mode"] == "trinal_clip"


def test_neurd_actor_loss_uses_selected_legal_logit_without_illegal_gradients():
    logits = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32))
    legal_masks = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    actions = torch.tensor([1], dtype=torch.long)
    advantages = torch.tensor([2.0], dtype=torch.float32)

    loss = native_ppo_policy._neurd_actor_loss(logits, legal_masks, actions, advantages)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad[0, 1] < 0.0
    assert logits.grad[0, 0] == 0.0
    assert logits.grad[0, 2] == 0.0


def test_neurd_actor_update_increases_positive_advantage_action_logit():
    feature_dim = 4
    policy_net = torch.nn.Linear(feature_dim, N_ACTIONS)
    value_net = torch.nn.Linear(feature_dim, 1)
    q_net = torch.nn.Linear(feature_dim, N_ACTIONS)
    with torch.no_grad():
        policy_net.weight.zero_()
        policy_net.bias.zero_()
        value_net.weight.zero_()
        value_net.bias.zero_()
        q_net.weight.zero_()
        q_net.bias.zero_()

    optimizer = torch.optim.SGD(
        list(policy_net.parameters()) + list(value_net.parameters()) + list(q_net.parameters()),
        lr=0.1,
    )
    record = _DecisionRecord(
        player=0,
        features=torch.zeros(feature_dim).numpy(),
        critic_features=torch.zeros(feature_dim).numpy(),
        legal_mask=torch.tensor([1.0, 1.0] + [0.0] * (N_ACTIONS - 2)).numpy(),
        action=1,
        old_log_prob=float(torch.log(torch.tensor(0.5))),
        value=0.0,
        expected_q=0.0,
    )

    updates, _ = _train_ppo_batch(
        policy_net,
        value_net,
        q_net,
        optimizer,
        [record],
        torch.tensor([1.0]).numpy(),
        NativePPOConfig(
            hidden_dim=16,
            batch_size=1,
            ppo_epochs=1,
            advantage_mode="q_expected_lambda",
            actor_update_mode="neurd",
            value_loss_weight=0.0,
            entropy_weight=0.0,
        ),
        np.random.default_rng(8),
        torch.device("cpu"),
    )

    assert updates == 1
    with torch.no_grad():
        logits = policy_net(torch.zeros(feature_dim))
    assert logits[1] > logits[0]


def test_native_ppo_policy_neurd_actor_update_exports_config(tmp_path):
    checkpoint_path = tmp_path / "native_ppo_neurd_policy.pt"

    metrics = run_native_ppo_policy_pilot(
        NativePPOConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=32,
            rollout_episodes_per_update=2,
            ppo_epochs=1,
            advantage_mode="q_expected_lambda",
            actor_update_mode="neurd",
            feature_mode="raw_sequence",
            device="cpu",
            seed=2041,
            checkpoint_path=str(checkpoint_path),
        )
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert metrics["actor_update_mode"] == "neurd"
    assert payload["config"]["actor_update_mode"] == "neurd"
