import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import ACTION_TO_INDEX, N_ACTIONS, N_FEATURES
from poker_ai.research.mixed_policy_h2h import load_policy_adapter
from poker_ai.research.xdo_policy_consumer import train_xdo_policy_consumer


def _write_toy_targets(path: Path) -> np.ndarray:
    features = np.zeros((2, N_FEATURES), dtype=np.float32)
    features[0, 0] = 1.0
    features[1, 10] = 1.0
    legal_masks = np.zeros((2, N_ACTIONS), dtype=np.float32)
    legal_masks[:, ACTION_TO_INDEX["fold"]] = 1.0
    legal_masks[:, ACTION_TO_INDEX["call"]] = 1.0
    target_probs = np.zeros((2, N_ACTIONS), dtype=np.float32)
    target_probs[0, ACTION_TO_INDEX["call"]] = 1.0
    target_probs[1, ACTION_TO_INDEX["fold"]] = 1.0
    PolicyTargetBuffer(features, legal_masks, target_probs).save_npz(path)
    return features


def test_train_xdo_policy_consumer_writes_npi_compatible_checkpoint(tmp_path):
    targets = tmp_path / "targets.npz"
    checkpoint = tmp_path / "consumer.pt"
    features = _write_toy_targets(targets)

    metrics = train_xdo_policy_consumer(
        targets,
        checkpoint,
        hidden_dim=32,
        n_steps=120,
        batch_size=2,
        device="cpu",
        seed=20260527,
        min_training_action_agreement=1.0,
    )

    assert metrics["passed"] is True
    assert metrics["training_action_agreement"] == 1.0
    adapter = load_policy_adapter(str(checkpoint), kind="npi", device=torch.device("cpu"))
    legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)
    legal_mask[ACTION_TO_INDEX["fold"]] = 1.0
    legal_mask[ACTION_TO_INDEX["call"]] = 1.0
    probs = adapter.probs(features[0], legal_mask, torch.device("cpu"))
    assert int(np.argmax(probs)) == ACTION_TO_INDEX["call"]


def test_train_xdo_policy_consumer_cli_writes_metrics(tmp_path):
    targets = tmp_path / "targets.npz"
    checkpoint = tmp_path / "consumer.pt"
    metrics_json = tmp_path / "metrics.json"
    _write_toy_targets(targets)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_xdo_policy_consumer.py",
            "--targets",
            str(targets),
            "--output",
            str(checkpoint),
            "--metrics-output",
            str(metrics_json),
            "--hidden-dim",
            "32",
            "--n-steps",
            "120",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--min-training-action-agreement",
            "1.0",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout)
    assert metrics["passed"] is True
    assert json.loads(metrics_json.read_text())["checkpoint"] == str(checkpoint)
