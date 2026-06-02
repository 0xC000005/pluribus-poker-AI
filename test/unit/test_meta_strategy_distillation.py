import json

import numpy as np
import torch

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.meta_strategy_distillation import train_meta_strategy_distillation
from poker_ai.research.mixed_policy_h2h import load_policy_adapter


def _write_dataset(path):
    observations = np.random.default_rng(7).normal(size=(16, N_FEATURES)).astype(np.float32)
    legal_masks = np.zeros((16, N_ACTIONS), dtype=np.bool_)
    legal_masks[:, 1] = True
    legal_masks[:, 8] = True
    actions = np.where(np.arange(16) % 2 == 0, 1, 8).astype(np.int64)
    np.savez(
        path,
        observations=observations,
        legal_masks=legal_masks,
        actions=actions,
        meta_strategy=np.array([0.6, 0.4], dtype=np.float32),
    )


def test_train_meta_strategy_distillation_exports_native_ppo_checkpoint(tmp_path):
    dataset = tmp_path / "meta_dataset.npz"
    checkpoint = tmp_path / "student.pt"
    metrics_json = tmp_path / "student.json"
    _write_dataset(dataset)

    metrics = train_meta_strategy_distillation(
        dataset,
        train_steps=4,
        batch_size=8,
        hidden_dim=16,
        seed=20260890,
        device="cpu",
        checkpoint_out=checkpoint,
        output_json=metrics_json,
    )

    adapter = load_policy_adapter(str(checkpoint), kind="native-ppo", device=torch.device("cpu"))
    legal_mask = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    probs = adapter.probs(np.zeros(N_FEATURES, dtype=np.float32), legal_mask, torch.device("cpu"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert metrics["algorithm"] == "native_meta_strategy_distillation"
    assert metrics["uses_slumbot_training_data"] is False
    assert payload["algorithm"] == "native_meta_strategy_distillation"
    assert payload["config"]["feature_mode"] == "flat"
    assert np.isclose(float(probs.sum()), 1.0)
    assert probs[1] > 0.0 and probs[8] > 0.0
    assert json.loads(metrics_json.read_text(encoding="utf-8"))["checkpoint_path"] == str(
        checkpoint
    )


def test_train_meta_strategy_distillation_cli_writes_json(monkeypatch, tmp_path):
    from scripts import train_meta_strategy_distillation as cli

    output = tmp_path / "metrics.json"
    checkpoint = tmp_path / "student.pt"
    calls = []

    def fake_train(**kwargs):
        calls.append(kwargs)
        return {
            "algorithm": "native_meta_strategy_distillation",
            "checkpoint_path": str(kwargs["checkpoint_out"]),
            "uses_slumbot_training_data": False,
        }

    monkeypatch.setattr(cli, "train_meta_strategy_distillation", fake_train)

    exit_code = cli.main(
        [
            "--dataset-npz",
            "dataset.npz",
            "--train-steps",
            "12",
            "--batch-size",
            "8",
            "--checkpoint-out",
            str(checkpoint),
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    assert calls[0]["dataset_npz"] == "dataset.npz"
    assert calls[0]["train_steps"] == 12
    assert calls[0]["batch_size"] == 8
    assert json.loads(output.read_text(encoding="utf-8"))["checkpoint_path"] == str(
        checkpoint
    )
