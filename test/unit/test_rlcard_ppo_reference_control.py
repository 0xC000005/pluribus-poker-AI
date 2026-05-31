from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rlcard")
pytest.importorskip("tianshou")


def test_rlcard_single_agent_env_exposes_obs_and_legal_mask():
    from poker_ai.research.rlcard_tianshou_ppo import RLCardNoLimitHoldemSingleAgentEnv

    env = RLCardNoLimitHoldemSingleAgentEnv(seed=17)

    obs, info = env.reset(seed=17)

    assert obs["obs"].shape == (54,)
    assert obs["mask"].shape == (5,)
    assert obs["mask"].sum() >= 1
    assert info["current_player"] == 0
    assert env.action_space.n == 5


def test_rlcard_single_agent_env_reset_skips_terminal_opponent_only_hands():
    from poker_ai.research.rlcard_tianshou_ppo import RLCardNoLimitHoldemSingleAgentEnv

    env = RLCardNoLimitHoldemSingleAgentEnv(seed=20260622)

    obs, info = env.reset(seed=20260622)

    assert info["environment"] == "rlcard:no-limit-holdem"
    assert env._rl_env.game.is_over() is False
    assert obs["mask"].shape == (5,)
    assert obs["mask"].any()


def test_run_rlcard_ppo_control_writes_env_native_checkpoint(tmp_path):
    from scripts.run_tianshou_ppo_rlcard_reference_control import run_control

    checkpoint = tmp_path / "rlcard_ppo.pt"
    metrics = run_control(
        rollout_steps=16,
        updates=1,
        repeat=1,
        batch_size=8,
        hidden_dim=16,
        replay_size=128,
        eval_games=2,
        seed=18,
        device="cpu",
        checkpoint_out=str(checkpoint),
    )

    assert checkpoint.exists()
    assert metrics["algorithm"] == "tianshou_ppo_rlcard"
    assert metrics["environment"] == "rlcard:no-limit-holdem"
    assert metrics["trained_environment_native"] is True
    assert metrics["native_action_projection"] is False
    assert metrics["uses_slumbot_data"] is False
    assert metrics["uses_alphanlholdem_training_data"] is False
    assert metrics["checkpoint_path"] == str(checkpoint)


def test_run_rlcard_ppo_control_supports_checkpoint_opponent(tmp_path):
    from poker_ai.research.rlcard_tianshou_ppo import save_untrained_ppo_checkpoint
    from scripts.run_tianshou_ppo_rlcard_reference_control import run_control

    opponent = tmp_path / "opponent.pt"
    checkpoint = tmp_path / "child.pt"
    save_untrained_ppo_checkpoint(opponent, hidden_dim=16)

    metrics = run_control(
        rollout_steps=16,
        updates=1,
        repeat=1,
        batch_size=8,
        hidden_dim=16,
        replay_size=128,
        eval_games=2,
        seed=19,
        device="cpu",
        opponent_checkpoint=str(opponent),
        checkpoint_out=str(checkpoint),
    )

    assert checkpoint.exists()
    assert metrics["opponent_kind"] == "rlcard-ppo"
    assert metrics["opponent_checkpoint"] == str(opponent)
    assert metrics["population_training"] is True
    assert metrics["trained_environment_native"] is True
    assert metrics["native_action_projection"] is False


def test_rlcard_ppo_policy_loader_returns_legal_action(tmp_path):
    from poker_ai.research.rlcard_tianshou_ppo import RLCardPPOPolicy, save_untrained_ppo_checkpoint

    checkpoint = tmp_path / "rlcard_ppo.pt"
    save_untrained_ppo_checkpoint(checkpoint, hidden_dim=16)
    policy = RLCardPPOPolicy.from_checkpoint(checkpoint, device="cpu")
    obs = {
        "obs": np.zeros((54,), dtype=np.float32),
        "legal_actions": {1: None, 4: None},
        "raw_legal_actions": [type("Action", (), {"value": 1})(), type("Action", (), {"value": 4})()],
    }

    decision = policy.act_rlcard_state(obs)

    assert decision.action in {1, 4}
    assert decision.legal_actions == [1, 4]
    assert np.isneginf(decision.masked_logits[0])
    assert np.isneginf(decision.masked_logits[2])
    assert np.isneginf(decision.masked_logits[3])


def test_eval_alphanlholdem_cli_accepts_rlcard_ppo_candidate(monkeypatch, tmp_path):
    from scripts import eval_alphanlholdem_rlcard_reference as cli

    candidate = tmp_path / "candidate.pt"
    baseline = tmp_path / "baseline.pkl"
    candidate.write_bytes(b"candidate")
    baseline.write_bytes(b"baseline")
    output = tmp_path / "h2h.json"
    load_calls = []

    class _Policy:
        def __init__(self, name):
            self.policy_name = name

    def fake_load_policy(kind, *, weights, seed, device):
        load_calls.append((kind, weights, seed, device))
        return _Policy(kind)

    monkeypatch.setattr(cli, "_load_policy", fake_load_policy)
    monkeypatch.setattr(
        cli,
        "evaluate_rlcard_reference_h2h",
        lambda **kwargs: {
            "algorithm": "alphanlholdem_rlcard_reference_h2h",
            "candidate_policy": kwargs["candidate"].policy_name,
            "baseline_policy": kwargs["baseline"].policy_name,
            "games": kwargs["games_per_seat"] * 2,
            "promotion": False,
        },
    )

    exit_code = cli.main(
        [
            "--candidate",
            "rlcard-ppo",
            "--candidate-weights",
            str(candidate),
            "--baseline",
            "alphanlholdem",
            "--baseline-weights",
            str(baseline),
            "--games-per-seat",
            "2",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    assert load_calls[0][:2] == ("rlcard-ppo", candidate)
