import json
from pathlib import Path

import numpy as np
import torch

from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.belief_probe import (
    BELIEF_DIM,
    N_HANDS,
    load_belief_probe_dataset,
    run_public_belief_probe,
)


def _write_checkpoint(path: Path) -> None:
    torch.manual_seed(0)
    net = ValueNetwork(N_FEATURES, 8, N_ACTIONS, n_layers=1, use_betting_history=True)
    torch.save(
        {
            "iteration": 1,
            "n_players": 2,
            "hidden_dim": 8,
            "n_layers": 1,
            "initial_chips": 20000,
            "uses_betting_history": True,
            "value_net": net.state_dict(),
        },
        path,
    )


def _write_targets_and_cases(tmp_path: Path) -> tuple[Path, Path]:
    features = np.zeros((1, N_FEATURES), dtype=np.float32)
    legal_masks = np.zeros((1, N_ACTIONS), dtype=np.float32)
    legal_masks[0, [1, 3, 8]] = 1.0
    target_probs = np.zeros((1, N_ACTIONS), dtype=np.float32)
    target_probs[0, [1, 3, 8]] = [0.2, 0.5, 0.3]
    targets_path = tmp_path / "targets.npz"
    PolicyTargetBuffer(features, legal_masks, target_probs).save_npz(targets_path)

    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "label": "river-open-check",
                        "hole_cards": ["Ac", "Kd"],
                        "board": ["2c", "7d", "Jh", "4s", "9c"],
                        "action_str": "ck/kk/kk/",
                        "client_pos": 0,
                        "source": "unit",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return targets_path, cases_path


def test_load_belief_probe_dataset_adds_raw_range_vectors(tmp_path):
    checkpoint = tmp_path / "range.pt"
    _write_checkpoint(checkpoint)
    targets, cases = _write_targets_and_cases(tmp_path)

    dataset = load_belief_probe_dataset(
        targets_npz=targets,
        cases_json=cases,
        range_checkpoint=checkpoint,
        device="cpu",
    )

    assert dataset.features.shape == (1, N_FEATURES)
    assert dataset.belief.shape == (1, BELIEF_DIM)
    assert dataset.legal_masks.shape == (1, N_ACTIONS)
    assert dataset.labels == ("river-open-check",)
    assert np.isclose(dataset.belief[0, :N_HANDS].sum(), 1.0)
    assert np.isclose(dataset.belief[0, N_HANDS:].sum(), 1.0)


def test_public_belief_probe_emits_feature_and_belief_metrics(tmp_path):
    checkpoint = tmp_path / "range.pt"
    _write_checkpoint(checkpoint)
    targets, cases = _write_targets_and_cases(tmp_path)

    metrics = run_public_belief_probe(
        train_targets_npz=targets,
        train_cases_json=cases,
        holdout_targets_npz=targets,
        holdout_cases_json=cases,
        range_checkpoint=checkpoint,
        device="cpu",
        hidden_dim=8,
        epochs=2,
        seed=0,
    )

    assert metrics["mode"] == "public_belief_probe"
    assert metrics["feature_dim"] == N_FEATURES
    assert metrics["belief_dim"] == BELIEF_DIM
    assert metrics["train_size"] == 1
    assert metrics["holdout_size"] == 1
    assert "base_holdout" in metrics
    assert "belief_holdout" in metrics
