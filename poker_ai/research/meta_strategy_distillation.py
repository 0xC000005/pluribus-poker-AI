"""Supervised distillation of a local empirical-game meta-strategy policy."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.native_nfsp import resolve_device
from poker_ai.research.native_ppo_policy import _PolicyMLP


def _load_distillation_arrays(dataset_npz: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    path = Path(dataset_npz)
    data = np.load(path)
    for key in ("observations", "legal_masks", "actions"):
        if key not in data:
            raise ValueError(f"distillation dataset missing {key}")
    observations = np.asarray(data["observations"], dtype=np.float32)
    legal_masks = np.asarray(data["legal_masks"], dtype=bool)
    actions = np.asarray(data["actions"], dtype=np.int64)
    if observations.ndim != 2 or observations.shape[1] != N_FEATURES:
        raise ValueError("observations must have shape (n, N_FEATURES)")
    if legal_masks.shape != (observations.shape[0], N_ACTIONS):
        raise ValueError("legal_masks must have shape (n, N_ACTIONS)")
    if actions.shape != (observations.shape[0],):
        raise ValueError("actions must have shape (n,)")
    if observations.shape[0] <= 0:
        raise ValueError("distillation dataset is empty")
    if not np.all(legal_masks[np.arange(actions.shape[0]), actions]):
        raise ValueError("distillation actions must be legal")
    summary = {
        "dataset_npz": str(path),
        "dataset_transitions": int(observations.shape[0]),
        "dataset_meta_strategy": (
            np.asarray(data["meta_strategy"], dtype=np.float32).tolist()
            if "meta_strategy" in data
            else None
        ),
    }
    return observations, legal_masks, actions, summary


def _masked_logits(policy: torch.nn.Module, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    logits = policy(x)
    return logits.masked_fill(~legal_mask.to(dtype=torch.bool), -1.0e30)


def train_meta_strategy_distillation(
    dataset_npz: str | Path,
    *,
    train_steps: int = 1000,
    batch_size: int = 1024,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    seed: int = 20260890,
    device: str = "auto",
    checkpoint_out: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Train a native-ppo-compatible policy to imitate local meta-strategy actions."""
    observations, legal_masks, actions, dataset_summary = _load_distillation_arrays(dataset_npz)
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    policy = _PolicyMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(lr))
    x_all = torch.as_tensor(observations, dtype=torch.float32, device=resolved_device)
    mask_all = torch.as_tensor(legal_masks, dtype=torch.bool, device=resolved_device)
    y_all = torch.as_tensor(actions, dtype=torch.long, device=resolved_device)
    rng = np.random.default_rng(int(seed))
    n = int(observations.shape[0])
    losses: list[float] = []
    accuracies: list[float] = []
    started = time.perf_counter()
    for _step in range(int(train_steps)):
        batch_indices = rng.integers(0, n, size=min(int(batch_size), n))
        batch = torch.as_tensor(batch_indices, dtype=torch.long, device=resolved_device)
        logits = _masked_logits(policy, x_all[batch], mask_all[batch])
        loss = F.cross_entropy(logits, y_all[batch])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            accuracies.append(float((pred == y_all[batch]).float().mean().detach().cpu()))
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    with torch.no_grad():
        all_logits = _masked_logits(policy, x_all, mask_all)
        all_loss = F.cross_entropy(all_logits, y_all)
        all_pred = torch.argmax(all_logits, dim=1)
        all_accuracy = float((all_pred == y_all).float().mean().detach().cpu())
        probs = torch.softmax(all_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)), dim=1)
    metrics: dict[str, Any] = {
        "algorithm": "native_meta_strategy_distillation",
        "role": "supervised_local_population_policy_distillation",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": (
            "This is a local distillation of a solved empirical-game population "
            "policy. It is not promotion evidence without H2H support/off-support gates."
        ),
        **device_info,
        **dataset_summary,
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "hidden_dim": int(hidden_dim),
        "train_steps": int(train_steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
        "train_seconds": float(train_seconds),
        "steps_per_second": float(int(train_steps) / max(train_seconds, 1e-9)),
        "last_loss": float(losses[-1]) if losses else None,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "last_batch_accuracy": float(accuracies[-1]) if accuracies else None,
        "full_dataset_loss": float(all_loss.detach().cpu()),
        "full_dataset_accuracy": float(all_accuracy),
        "mean_policy_entropy": float(entropy.mean().detach().cpu()),
        "uses_slumbot_training_data": False,
        "promotion": False,
    }
    if checkpoint_out is not None:
        path = Path(checkpoint_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": metrics["algorithm"],
                "environment": metrics["environment"],
                "num_actions": N_ACTIONS,
                "num_features": N_FEATURES,
                "hidden_dim": int(hidden_dim),
                "policy_net_state_dict": policy.state_dict(),
                "config": {
                    "feature_mode": "flat",
                    "hidden_dim": int(hidden_dim),
                    "initial_chips": 20_000,
                    "max_steps_per_hand": 256,
                    "fsp_average_policy": False,
                },
                "metrics": metrics,
            },
            path,
        )
        metrics["checkpoint_path"] = str(path)
    if output_json is not None:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics
