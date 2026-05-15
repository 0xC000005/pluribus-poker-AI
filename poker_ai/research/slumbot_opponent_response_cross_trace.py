"""Cross-trace Slumbot opponent-response action-likelihood diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from poker_ai.research.slumbot_opponent_response_probe import (
    _ProbeNet,
    _fit_temperature,
    _masked_cross_entropy,
    _metrics,
    _predict_logits,
    _probs_from_logits,
    _resolve_device,
    _split_by_hand,
    _standardize_pair,
    load_opponent_response_dataset,
)


def train_and_eval_cross_trace_response_probe(
    train_action_likelihood_json: str | Path,
    eval_action_likelihood_json: str | Path,
    *,
    holdout_fraction: float = 0.3,
    hidden_dim: int = 128,
    n_layers: int = 2,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | torch.device = "auto",
    seed: int = 20260515,
) -> dict[str, Any]:
    """Train on one Slumbot trace artifact and evaluate action likelihood on another."""
    source = load_opponent_response_dataset(train_action_likelihood_json)
    external = load_opponent_response_dataset(eval_action_likelihood_json)
    train_mask, calibration_mask = _split_by_hand(source.hand_indices, holdout_fraction)
    x_train, x_calibration, mean, std = _standardize_pair(
        source.features[train_mask],
        source.features[calibration_mask],
    )
    x_source = np.zeros_like(source.features, dtype=np.float32)
    x_source[train_mask] = x_train
    x_source[calibration_mask] = x_calibration
    x_external = ((external.features - mean) / std).astype(np.float32)

    resolved_device = _resolve_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _ProbeNet(source.features.shape[1], hidden_dim, n_layers).to(resolved_device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    source_x_t = torch.from_numpy(x_source).to(resolved_device)
    source_legal_t = torch.from_numpy(source.legal_masks).to(resolved_device)
    source_target_t = torch.from_numpy(source.target_probs).to(resolved_device)
    train_idx = np.flatnonzero(train_mask)

    for _ in range(max(1, int(epochs))):
        perm = np.random.permutation(train_idx)
        for start in range(0, len(perm), max(1, int(batch_size))):
            batch = perm[start : start + max(1, int(batch_size))]
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_cross_entropy(
                model(source_x_t[batch]),
                source_legal_t[batch],
                source_target_t[batch],
            )
            loss.backward()
            optimizer.step()

    source_logits = _predict_logits(model, x_source, resolved_device)
    external_logits = _predict_logits(model, x_external, resolved_device)
    temperature = _fit_temperature(
        source_logits,
        source.legal_masks,
        source.target_probs,
        calibration_mask,
        device=resolved_device,
    )
    source_probs = _probs_from_logits(source_logits, source.legal_masks)
    source_calibrated_probs = _probs_from_logits(
        source_logits,
        source.legal_masks,
        temperature=temperature,
    )
    external_probs = _probs_from_logits(external_logits, external.legal_masks)
    external_calibrated_probs = _probs_from_logits(
        external_logits,
        external.legal_masks,
        temperature=temperature,
    )
    external_mask = np.ones(external.features.shape[0], dtype=bool)
    external_metrics = _metrics(probs=external_probs, dataset=external, mask=external_mask)
    calibrated_external_metrics = _metrics(
        probs=external_calibrated_probs,
        dataset=external,
        mask=external_mask,
    )
    return {
        "mode": "slumbot_opponent_response_cross_trace",
        "train_source": str(train_action_likelihood_json),
        "eval_source": str(eval_action_likelihood_json),
        "device": str(resolved_device),
        "holdout_fraction": float(holdout_fraction),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "temperature": float(temperature),
        "source_n_records": int(source.features.shape[0]),
        "source_n_hands": int(len(set(int(x) for x in source.hand_indices))),
        "external_n_records": int(external.features.shape[0]),
        "external_n_hands": int(len(set(int(x) for x in external.hand_indices))),
        "source_train": _metrics(probs=source_probs, dataset=source, mask=train_mask),
        "source_calibration": _metrics(
            probs=source_probs,
            dataset=source,
            mask=calibration_mask,
        ),
        "calibrated_source_calibration": _metrics(
            probs=source_calibrated_probs,
            dataset=source,
            mask=calibration_mask,
        ),
        "external_eval": external_metrics,
        "calibrated_external_eval": calibrated_external_metrics,
        "probe_beats_model_on_external": bool(
            external_metrics["probe_minus_model_log_lift"] > 0.0
        ),
        "calibrated_probe_beats_model_on_external": bool(
            calibrated_external_metrics["probe_minus_model_log_lift"] > 0.0
        ),
        "calibrated_probe_beats_uniform_on_external": bool(
            calibrated_external_metrics["probe_mean_log_lift_vs_uniform"] > 0.0
        ),
        "passed": True,
    }


def write_metrics(metrics: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
