import json

import pytest
import torch


def test_compiled_rainbow_response_oracle_writes_checkpoint(tmp_path):
    pytest.importorskip("tianshou")

    from poker_ai.research.compiled_rainbow_response import run_compiled_rainbow_response_oracle
    from poker_ai.research.mixed_policy_h2h import load_policy_adapter

    checkpoint = tmp_path / "compiled_rainbow_response.pt"
    output = tmp_path / "compiled_rainbow_response.json"

    metrics = run_compiled_rainbow_response_oracle(
        collect_iterations=1,
        games_per_iteration=8,
        collector_batch_size=8,
        max_steps_per_game=32,
        updates_per_collect=1,
        batch_size=8,
        hidden_dim=16,
        num_atoms=11,
        replay_size=256,
        seed=20260624,
        device="cpu",
        checkpoint_out=checkpoint,
        output_json=output,
    )

    assert checkpoint.exists()
    assert output.exists()
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["checkpoint_path"] == str(checkpoint)
    assert metrics["algorithm"] == "compiled_tianshou_rainbow_response_oracle"
    assert metrics["collector_backend"] == "compiled-fast-state"
    assert metrics["trained_environment_native"] is True
    assert metrics["native_action_projection"] is False
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["semi_mdp_transitions"] is True
    assert metrics["learner_seat"] == 0
    assert metrics["promotion"] is False
    assert metrics["passed"] is True
    assert metrics["replay_transitions"] > 0
    assert metrics["collector_transitions"] > 0
    assert metrics["updates"] == 1

    adapter = load_policy_adapter(
        str(checkpoint),
        kind="tianshou-rainbow",
        device=torch.device("cpu"),
    )
    assert adapter.kind == "tianshou-rainbow"
    assert adapter.algorithm == "compiled_tianshou_rainbow_response_oracle"


def test_compiled_rainbow_response_oracle_cli_writes_metrics(monkeypatch, tmp_path):
    from scripts import run_compiled_rainbow_response_oracle as cli

    checkpoint = tmp_path / "candidate.pt"
    output = tmp_path / "metrics.json"
    calls = []

    def _fake_run(**kwargs):
        calls.append(kwargs)
        checkpoint.write_bytes(b"checkpoint")
        output.write_text(json.dumps({"passed": True}) + "\n", encoding="utf-8")
        return {
            "algorithm": "compiled_tianshou_rainbow_response_oracle",
            "passed": True,
            "checkpoint_path": str(checkpoint),
        }

    monkeypatch.setattr(cli, "run_compiled_rainbow_response_oracle", _fake_run)

    metrics = cli.main(
        [
            "--collect-iterations",
            "2",
            "--games-per-iteration",
            "16",
            "--opponent-policy",
            "tianshou-rainbow:parent.pt",
            "--opponent-meta-strategy",
            "1.0",
            "--checkpoint-in",
            "parent.pt",
            "--checkpoint-out",
            str(checkpoint),
            "--output-json",
            str(output),
        ]
    )

    assert metrics["passed"] is True
    assert calls[0]["collect_iterations"] == 2
    assert calls[0]["games_per_iteration"] == 16
    assert calls[0]["opponent_policy_specs"] == ["tianshou-rainbow:parent.pt"]
    assert calls[0]["opponent_meta_strategy"] == [1.0]
    assert calls[0]["checkpoint_in"].name == "parent.pt"
    assert calls[0]["checkpoint_out"] == checkpoint
    assert calls[0]["output_json"] == output


def test_compiled_rainbow_response_oracle_supports_checkpoint_continuation(tmp_path):
    pytest.importorskip("tianshou")

    from poker_ai.research.compiled_rainbow_response import run_compiled_rainbow_response_oracle

    parent = tmp_path / "parent.pt"
    child = tmp_path / "child.pt"

    run_compiled_rainbow_response_oracle(
        collect_iterations=1,
        games_per_iteration=8,
        collector_batch_size=8,
        max_steps_per_game=32,
        updates_per_collect=1,
        batch_size=8,
        hidden_dim=16,
        num_atoms=11,
        replay_size=256,
        seed=20260625,
        device="cpu",
        checkpoint_out=parent,
    )

    metrics = run_compiled_rainbow_response_oracle(
        collect_iterations=1,
        games_per_iteration=8,
        collector_batch_size=8,
        max_steps_per_game=32,
        updates_per_collect=1,
        batch_size=8,
        hidden_dim=16,
        num_atoms=11,
        replay_size=256,
        seed=20260626,
        device="cpu",
        checkpoint_in=parent,
        checkpoint_out=child,
    )

    payload = torch.load(child, map_location="cpu", weights_only=False)
    assert metrics["checkpoint_in"] == str(parent)
    assert metrics["checkpoint_in_algorithm"] == "compiled_tianshou_rainbow_response_oracle"
    assert payload["parent_checkpoint"] == str(parent)
