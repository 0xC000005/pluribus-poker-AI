import json

import numpy as np
import pytest
import torch

pytest.importorskip("rlcard")


def test_local_vtrace_rlcard_learner_writes_rlcard_native_checkpoint(tmp_path):
    from poker_ai.research.rlcard_tianshou_ppo import RLCardPPOPolicy, N_RLCARD_ACTIONS
    from scripts.run_local_vtrace_rlcard_learner import run_learner

    checkpoint = tmp_path / "rlcard_vtrace.pt"
    output = tmp_path / "rlcard_vtrace.json"
    metrics = run_learner(
        n_envs=2,
        unroll_length=4,
        train_iterations=1,
        hidden_dim=16,
        seed=20260561,
        device="cpu",
        checkpoint_out=checkpoint,
        output_json=output,
    )

    assert checkpoint.exists()
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["checkpoint_path"] == str(checkpoint)
    assert metrics["algorithm"] == "local_vtrace_rlcard"
    assert metrics["environment"] == "rlcard:no-limit-holdem"
    assert metrics["num_actions"] == N_RLCARD_ACTIONS
    assert metrics["trained_environment_native"] is True
    assert metrics["native_action_projection"] is False
    assert metrics["rlcard_candidate"] is True
    assert metrics["uses_slumbot_data"] is False
    assert metrics["uses_alphanlholdem_training_data"] is False
    assert metrics["promotion"] is False
    assert metrics["passed"] is True

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["algorithm"] == "local_vtrace_rlcard"
    assert payload["environment"] == "rlcard:no-limit-holdem"
    assert payload["num_actions"] == N_RLCARD_ACTIONS
    assert payload["trained_environment_native"] is True
    assert payload["native_action_projection"] is False

    policy = RLCardPPOPolicy.from_checkpoint(checkpoint, device="cpu")
    decision = policy.act_rlcard_state(
        {
            "obs": np.zeros((54,), dtype=np.float32),
            "legal_actions": {1: None, 3: None},
            "raw_legal_actions": [1, 3],
        }
    )
    assert decision.action in {1, 3}
    assert decision.legal_actions == [1, 3]
    assert decision.policy_name == "local_vtrace_rlcard"


def test_local_vtrace_rlcard_learner_supports_checkpoint_in(tmp_path):
    from poker_ai.research.rlcard_tianshou_ppo import (
        _RLCardCritic,
        _RLCardMaskedActor,
        save_untrained_ppo_checkpoint,
    )
    from scripts.run_local_vtrace_rlcard_learner import _load_checkpoint_in

    parent = tmp_path / "parent.pt"
    save_untrained_ppo_checkpoint(parent, hidden_dim=16, device="cpu")
    actor = _RLCardMaskedActor(hidden_dim=16, device=torch.device("cpu"))
    critic = _RLCardCritic(hidden_dim=16, device=torch.device("cpu"))

    metrics = _load_checkpoint_in(
        parent,
        actor=actor,
        critic=critic,
        hidden_dim=16,
        device=torch.device("cpu"),
    )

    assert metrics["initialized_from_checkpoint"] is True
    assert metrics["checkpoint_in"] == str(parent)
    assert metrics["checkpoint_in_algorithm"] == "tianshou_ppo_rlcard"
    assert metrics["checkpoint_in_contract"] == "same_environment_continuation_only"


def test_local_vtrace_rlcard_learner_rejects_cross_environment_checkpoint_in(tmp_path):
    from poker_ai.research.rlcard_tianshou_ppo import (
        _RLCardCritic,
        _RLCardMaskedActor,
        N_RLCARD_ACTIONS,
        N_RLCARD_FEATURES,
    )
    from scripts.run_local_vtrace_rlcard_learner import _load_checkpoint_in

    native_checkpoint = tmp_path / "native.pt"
    actor = _RLCardMaskedActor(hidden_dim=16, device=torch.device("cpu"))
    critic = _RLCardCritic(hidden_dim=16, device=torch.device("cpu"))
    torch.save(
        {
            "algorithm": "native_control",
            "environment": "poker_ai:full_deck_hu_nlhe",
            "num_actions": N_RLCARD_ACTIONS,
            "num_features": N_RLCARD_FEATURES,
            "hidden_dim": 16,
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
        },
        native_checkpoint,
    )

    with pytest.raises(ValueError, match="rlcard:no-limit-holdem"):
        _load_checkpoint_in(
            native_checkpoint,
            actor=actor,
            critic=critic,
            hidden_dim=16,
            device=torch.device("cpu"),
        )


def test_local_vtrace_rlcard_learner_assigns_population_opponents():
    from scripts.run_local_vtrace_rlcard_learner import (
        _assign_opponent_checkpoints,
        _normalize_opponent_checkpoints,
    )

    checkpoints = _normalize_opponent_checkpoints(
        opponent_checkpoint="parent.pt",
        opponent_checkpoints=["control_a.pt", "control_b.pt"],
    )
    assignments = _assign_opponent_checkpoints(checkpoints, n_envs=8, seed=20260574)

    assert checkpoints == ["parent.pt", "control_a.pt", "control_b.pt"]
    assert len(assignments) == 8
    assert set(assignments) == set(checkpoints)


def test_eval_alphanlholdem_cli_accepts_rlcard_vtrace_candidate(monkeypatch, tmp_path):
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
            "rlcard-vtrace",
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
    assert load_calls[0][:2] == ("rlcard-vtrace", candidate)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["candidate_kind"] == "rlcard-vtrace"
    assert payload["trained_environment_native"] is True
    assert payload["native_action_projection"] is False
