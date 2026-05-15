import numpy as np
import torch

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.restricted_value_probe import (
    RestrictedValueProbeConfig,
    build_restricted_value_dataset,
    train_restricted_value_probe,
)


def test_build_restricted_value_dataset_shapes():
    dataset = build_restricted_value_dataset(
        n_roots=6,
        n_equity_samples=32,
        initial_chips=1000,
        seed=20260517,
    )

    assert dataset.features.shape == (6, N_FEATURES)
    assert dataset.targets.shape == (6, N_ACTIONS)
    assert dataset.legal_masks.shape == (6, N_ACTIONS)
    assert np.all(dataset.legal_masks.sum(axis=1) > 0)
    assert dataset.best_actions.shape == (6,)


def test_train_restricted_value_probe_smoke_learns_signal():
    metrics = train_restricted_value_probe(
        RestrictedValueProbeConfig(
            train_roots=32,
            holdout_roots=16,
            n_equity_samples=64,
            hidden_dim=32,
            n_layers=1,
            epochs=30,
            batch_size=16,
            device="cpu",
            seed=20260517,
        )
    )

    assert metrics["algorithm"] == "restricted_value_probe"
    assert metrics["promotion"] is False
    assert metrics["train_loss_final"] < metrics["train_loss_initial"]
    assert 0.0 <= metrics["holdout_top_action_match"] <= 1.0
    assert -1.0 <= metrics["holdout_mean_action_corr"] <= 1.0


def test_train_restricted_value_probe_can_save_checkpoint(tmp_path):
    checkpoint = tmp_path / "probe.pt"

    metrics = train_restricted_value_probe(
        RestrictedValueProbeConfig(
            train_roots=8,
            holdout_roots=4,
            n_equity_samples=16,
            hidden_dim=16,
            n_layers=1,
            epochs=1,
            batch_size=4,
            device="cpu",
            seed=20260517,
            output_checkpoint=str(checkpoint),
        )
    )

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert metrics["checkpoint"] == str(checkpoint)
    assert saved["hidden_dim"] == 16
    assert saved["n_layers"] == 1
    assert "value_net" in saved
