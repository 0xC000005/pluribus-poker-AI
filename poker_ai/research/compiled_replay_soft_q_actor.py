"""Replay-trained soft-Q actor for compiled native population experience."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.compiled_joint_experience import (
    collect_compiled_joint_experience,
    load_compiled_joint_policy,
)
from poker_ai.research.native_nfsp import resolve_device
from poker_ai.research.native_ppo_policy import _PolicyMLP, _QMLP


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(metrics: dict[str, Any], output_json: str | Path | None) -> dict[str, Any]:
    if output_json is None:
        return metrics
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics["output_json"] = str(path)
    path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    return metrics


def _as_tensor_dataset(dataset: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    n_actions = int(dataset.get("num_actions", N_ACTIONS))
    n_features = int(dataset.get("num_features", N_FEATURES))
    if n_actions != N_ACTIONS:
        raise ValueError("dataset action count does not match native contract")
    if n_features != N_FEATURES:
        raise ValueError("dataset feature count does not match native contract")
    features = np.asarray(dataset["observations"], dtype=np.float32)
    masks = np.asarray(dataset["legal_masks"], dtype=np.bool_)
    actions = np.asarray(dataset["actions"], dtype=np.int64)
    returns = np.asarray(dataset["returns"], dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != N_FEATURES:
        raise ValueError("dataset observations must have shape (N, N_FEATURES)")
    if masks.shape != (features.shape[0], N_ACTIONS):
        raise ValueError("dataset legal_masks must have shape (N, N_ACTIONS)")
    if actions.shape != (features.shape[0],) or returns.shape != (features.shape[0],):
        raise ValueError("dataset actions and returns must match observations")
    if features.shape[0] <= 0:
        raise ValueError("dataset must contain at least one transition")
    legal_action_rows = masks[np.arange(actions.shape[0]), actions]
    if not bool(np.all(legal_action_rows)):
        raise ValueError("dataset contains an illegal recorded action")
    return {
        "features": torch.as_tensor(features, dtype=torch.float32, device=device),
        "legal_masks": torch.as_tensor(masks, dtype=torch.bool, device=device),
        "actions": torch.as_tensor(actions, dtype=torch.long, device=device),
        "returns": torch.as_tensor(returns, dtype=torch.float32, device=device),
    }


def train_soft_q_actor_from_compiled_replay(
    dataset: dict[str, Any],
    *,
    hidden_dim: int = 256,
    q_train_steps: int = 1000,
    policy_train_steps: int = 500,
    batch_size: int = 1024,
    lr: float = 1e-3,
    entropy_weight: float = 0.01,
    seed: int = 20260844,
    device: str = "auto",
    checkpoint_out: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Fit a Q function to replayed returns and distill a stochastic actor.

    This is a local off-policy falsifier, not a promotion path. The Q labels are
    terminal returns from the local simulator, not solver labels or Slumbot data.
    """

    if int(q_train_steps) <= 0:
        raise ValueError("q_train_steps must be positive")
    if int(policy_train_steps) <= 0:
        raise ValueError("policy_train_steps must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if float(entropy_weight) < 0.0:
        raise ValueError("entropy_weight must be non-negative")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    device_info = resolve_device(str(device))
    resolved_device = torch.device(device_info["resolved_device"])
    tensors = _as_tensor_dataset(dataset, resolved_device)
    n = int(tensors["actions"].shape[0])
    q_net = _QMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
    policy_net = _PolicyMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
    q_opt = torch.optim.Adam(q_net.parameters(), lr=float(lr))
    policy_opt = torch.optim.Adam(policy_net.parameters(), lr=float(lr))
    generator = torch.Generator(device=resolved_device.type)
    generator.manual_seed(int(seed) + 1000)

    q_losses: list[float] = []
    policy_losses: list[float] = []
    entropies: list[float] = []
    expected_qs: list[float] = []
    train_started = time.perf_counter()
    batch_n = min(int(batch_size), n)

    for _step in range(int(q_train_steps)):
        idx = torch.randint(n, (batch_n,), generator=generator, device=resolved_device)
        q_values = q_net(tensors["features"][idx])
        selected = q_values.gather(1, tensors["actions"][idx].view(-1, 1)).squeeze(1)
        loss = F.mse_loss(selected, tensors["returns"][idx])
        q_opt.zero_grad(set_to_none=True)
        loss.backward()
        q_opt.step()
        q_losses.append(float(loss.detach().cpu()))

    for parameter in q_net.parameters():
        parameter.requires_grad_(False)
    q_net.eval()

    for _step in range(int(policy_train_steps)):
        idx = torch.randint(n, (batch_n,), generator=generator, device=resolved_device)
        features = tensors["features"][idx]
        legal_masks = tensors["legal_masks"][idx]
        with torch.no_grad():
            q_values = q_net(features)
            q_values = torch.where(legal_masks, q_values, torch.zeros_like(q_values))
        logits = policy_net(features).masked_fill(~legal_masks, -1.0e30)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        expected_q = torch.sum(probs * q_values, dim=1).mean()
        safe_log_probs = torch.nan_to_num(log_probs, neginf=0.0)
        entropy = -torch.sum(
            torch.where(legal_masks, probs * safe_log_probs, torch.zeros_like(probs)),
            dim=1,
        ).mean()
        loss = -expected_q - float(entropy_weight) * entropy
        policy_opt.zero_grad(set_to_none=True)
        loss.backward()
        policy_opt.step()
        policy_losses.append(float(loss.detach().cpu()))
        entropies.append(float(entropy.detach().cpu()))
        expected_qs.append(float(expected_q.detach().cpu()))

    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_started
    returns_cpu = tensors["returns"].detach().cpu().numpy()
    metrics: dict[str, Any] = {
        "algorithm": "compiled_replay_soft_q_actor",
        "role": "native_compiled_replay_population_soft_q_actor",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": "Replay-trained local stochastic actor; not Slumbot or SOTA evidence.",
        **device_info,
        "seed": int(seed),
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "hidden_dim": int(hidden_dim),
        "q_train_steps": int(q_train_steps),
        "policy_train_steps": int(policy_train_steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "entropy_weight": float(entropy_weight),
        "dataset_algorithm": str(dataset.get("algorithm", "")),
        "dataset_backend": str(dataset.get("backend", "")),
        "dataset_n_hands": int(dataset.get("n_hands", 0)),
        "dataset_n_transitions": int(n),
        "dataset_initial_chips": int(dataset.get("initial_chips", 1000)),
        "dataset_max_steps_per_hand": int(dataset.get("max_steps_per_hand", 256)),
        "dataset_exploration_epsilon": float(dataset.get("exploration_epsilon", 0.0)),
        "dataset_needs_python_showdown": int(dataset.get("needs_python_showdown", 0)),
        "dataset_policy_kinds": list(dataset.get("policy_kinds", [])),
        "dataset_policy_checkpoints": list(dataset.get("policy_checkpoints", [])),
        "dataset_meta_strategy": np.asarray(dataset.get("meta_strategy", []), dtype=np.float32),
        "mean_return": float(np.mean(returns_cpu)) if returns_cpu.size else 0.0,
        "std_return": float(np.std(returns_cpu)) if returns_cpu.size else 0.0,
        "first_q_loss": q_losses[0] if q_losses else None,
        "last_q_loss": q_losses[-1] if q_losses else None,
        "first_policy_loss": policy_losses[0] if policy_losses else None,
        "last_policy_loss": policy_losses[-1] if policy_losses else None,
        "mean_policy_entropy": float(np.mean(entropies)) if entropies else None,
        "last_policy_entropy": entropies[-1] if entropies else None,
        "mean_policy_expected_q": float(np.mean(expected_qs)) if expected_qs else None,
        "last_policy_expected_q": expected_qs[-1] if expected_qs else None,
        "train_seconds": float(train_seconds),
        "updates_per_second": float((int(q_train_steps) + int(policy_train_steps)) / max(train_seconds, 1e-9)),
        "trained_environment_native": True,
        "native_action_projection": False,
        "rlcard_candidate": False,
        "uses_slumbot_data": False,
        "uses_slumbot_training_data": False,
        "uses_alphanlholdem_training_data": False,
        "uses_solver_labels": False,
        "promotion": False,
        "checkpoint_path": str(checkpoint_out) if checkpoint_out is not None else None,
    }
    metrics["passed"] = bool(
        q_losses
        and policy_losses
        and np.isfinite(q_losses[-1])
        and np.isfinite(policy_losses[-1])
        and int(dataset.get("needs_python_showdown", 0)) == 0
    )

    if checkpoint_out is not None:
        path = Path(checkpoint_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "compiled_replay_soft_q_actor",
                "role": "native_compiled_replay_population_soft_q_actor",
                "environment": "poker_ai:full_deck_hu_nlhe",
                "num_actions": N_ACTIONS,
                "num_features": N_FEATURES,
                "hidden_dim": int(hidden_dim),
                "policy_net_state_dict": policy_net.state_dict(),
                "q_net_state_dict": q_net.state_dict(),
                "config": {
                    "feature_mode": "flat",
                    "hidden_dim": int(hidden_dim),
                    "initial_chips": int(dataset.get("initial_chips", 1000)),
                    "max_steps_per_hand": int(dataset.get("max_steps_per_hand", 256)),
                    "train_environment": "poker_ai:full_deck_hu_nlhe",
                    "fsp_average_policy": False,
                    "dataset_algorithm": str(dataset.get("algorithm", "")),
                    "dataset_policy_kinds": list(dataset.get("policy_kinds", [])),
                    "dataset_policy_checkpoints": list(dataset.get("policy_checkpoints", [])),
                    "dataset_meta_strategy": np.asarray(
                        dataset.get("meta_strategy", []),
                        dtype=np.float32,
                    ).tolist(),
                },
                "metrics": metrics,
                "trained_environment_native": True,
                "native_action_projection": False,
                "rlcard_candidate": False,
                "uses_slumbot_data": False,
                "uses_slumbot_training_data": False,
                "uses_alphanlholdem_training_data": False,
                "uses_solver_labels": False,
            },
            path,
        )
    return _write_json(metrics, output_json)


def run_compiled_replay_soft_q_actor(
    *,
    policy_specs: Sequence[tuple[str, str]],
    meta_strategy: Sequence[float] | None = None,
    n_hands: int = 4096,
    collector_batch_size: int = 256,
    initial_chips: int = 20000,
    max_steps_per_hand: int = 256,
    exploration_epsilon: float = 0.1,
    hidden_dim: int = 256,
    q_train_steps: int = 1000,
    policy_train_steps: int = 500,
    train_batch_size: int = 1024,
    lr: float = 1e-3,
    entropy_weight: float = 0.01,
    seed: int = 20260844,
    device: str = "auto",
    checkpoint_out: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    if not policy_specs:
        raise ValueError("policy_specs must not be empty")
    device_info = resolve_device(str(device))
    resolved_device = torch.device(device_info["resolved_device"])
    policies = [
        load_compiled_joint_policy(checkpoint, kind=kind, device=resolved_device)
        for kind, checkpoint in policy_specs
    ]
    meta = (
        [1.0 / float(len(policy_specs))] * len(policy_specs)
        if meta_strategy is None
        else list(meta_strategy)
    )
    dataset = collect_compiled_joint_experience(
        policies,
        meta_strategy=meta,
        n_hands=int(n_hands),
        batch_size=int(collector_batch_size),
        seed=int(seed),
        initial_chips=int(initial_chips),
        max_steps_per_hand=int(max_steps_per_hand),
        exploration_epsilon=float(exploration_epsilon),
        device=resolved_device,
    )
    return train_soft_q_actor_from_compiled_replay(
        dataset,
        hidden_dim=int(hidden_dim),
        q_train_steps=int(q_train_steps),
        policy_train_steps=int(policy_train_steps),
        batch_size=int(train_batch_size),
        lr=float(lr),
        entropy_weight=float(entropy_weight),
        seed=int(seed),
        device=device_info["resolved_device"],
        checkpoint_out=checkpoint_out,
        output_json=output_json,
    )
