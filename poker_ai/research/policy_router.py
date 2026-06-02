"""Learned state-level routing over a local policy population."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from poker_ai.games.full_deck.state import N_FEATURES
from poker_ai.research.native_nfsp import resolve_device


class PolicyRouterNet(torch.nn.Module):
    """Predict per-policy values from a native poker observation."""

    def __init__(self, *, hidden_dim: int, n_policies: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES, int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), int(n_policies)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _require_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data:
        raise ValueError(f"joint-experience dataset is missing {key}")
    return np.asarray(data[key])


def _load_router_arrays(dataset_npz: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(dataset_npz))
    observations = _require_array(data, "observations").astype(np.float32, copy=False)
    behavior_policy_indices = _require_array(data, "behavior_policy_indices").astype(
        np.int64,
        copy=False,
    )
    returns = _require_array(data, "returns").astype(np.float32, copy=False)
    if observations.ndim != 2 or observations.shape[1] != N_FEATURES:
        raise ValueError("observations must have shape (n, N_FEATURES)")
    n = int(observations.shape[0])
    if behavior_policy_indices.shape != (n,) or returns.shape != (n,):
        raise ValueError("behavior_policy_indices and returns must match observations")
    if n <= 0:
        raise ValueError("joint-experience dataset must contain transitions")
    if not np.isfinite(returns).all():
        raise ValueError("returns must be finite")
    return {
        "observations": observations,
        "behavior_policy_indices": behavior_policy_indices,
        "returns": returns,
    }


def _normalize_member_specs(member_specs: Sequence[tuple[str, str]]) -> tuple[list[str], list[str]]:
    if not member_specs:
        raise ValueError("policy router requires at least one member policy")
    kinds: list[str] = []
    checkpoints: list[str] = []
    for kind, checkpoint in member_specs:
        kind_text = str(kind).strip()
        checkpoint_text = str(checkpoint).strip()
        if not kind_text or not checkpoint_text:
            raise ValueError("member policy kind and checkpoint must be non-empty")
        kinds.append(kind_text)
        checkpoints.append(checkpoint_text)
    return kinds, checkpoints


def train_policy_router_from_joint_experience(
    dataset_npz: str | Path,
    *,
    member_specs: Sequence[tuple[str, str]],
    hidden_dim: int = 128,
    train_steps: int = 1000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    seed: int = 20260990,
    device: str = "auto",
    checkpoint_out: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Fit a router value model from logged population trajectories.

    The target is the terminal local-simulator return observed when a behavior
    policy acted at that state. This is a contextual-bandit router over existing
    local policies, not Slumbot supervision or solver-label imitation.
    """

    if int(hidden_dim) <= 0:
        raise ValueError("hidden_dim must be positive")
    if int(train_steps) <= 0:
        raise ValueError("train_steps must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if float(lr) <= 0.0:
        raise ValueError("lr must be positive")

    member_kinds, member_checkpoints = _normalize_member_specs(member_specs)
    n_policies = len(member_kinds)
    arrays = _load_router_arrays(dataset_npz)
    observations = arrays["observations"]
    behavior_policy_indices = arrays["behavior_policy_indices"]
    returns = arrays["returns"]
    if int(behavior_policy_indices.min()) < 0 or int(behavior_policy_indices.max()) >= n_policies:
        raise ValueError("behavior_policy_indices exceed supplied member policies")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    device_info = resolve_device(str(device))
    resolved_device = torch.device(device_info["resolved_device"])
    net = PolicyRouterNet(hidden_dim=int(hidden_dim), n_policies=n_policies).to(resolved_device)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(lr))
    features = torch.as_tensor(observations, dtype=torch.float32, device=resolved_device)
    policy_indices = torch.as_tensor(
        behavior_policy_indices,
        dtype=torch.long,
        device=resolved_device,
    )
    targets = torch.as_tensor(returns, dtype=torch.float32, device=resolved_device)
    generator = torch.Generator(device=resolved_device.type)
    generator.manual_seed(int(seed) + 1000)
    n = int(features.shape[0])
    batch_n = min(int(batch_size), n)

    losses: list[float] = []
    started = time.perf_counter()
    for _step in range(int(train_steps)):
        idx = torch.randint(n, (batch_n,), generator=generator, device=resolved_device)
        values = net(features[idx])
        chosen_values = values.gather(1, policy_indices[idx].view(-1, 1)).squeeze(1)
        loss = F.mse_loss(chosen_values, targets[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started

    with torch.no_grad():
        all_values = net(features)
        predicted_policy_indices = torch.argmax(all_values, dim=1)
        behavior_match = (predicted_policy_indices == policy_indices).float().mean()
        chosen_values = all_values.gather(1, policy_indices.view(-1, 1)).squeeze(1)
        full_mse = F.mse_loss(chosen_values, targets)

    metrics: dict[str, Any] = {
        "algorithm": "policy_population_router",
        "role": "conflux_style_state_level_policy_router",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": "Learned local policy-population router; not promotion evidence without H2H gates.",
        **device_info,
        "dataset_npz": str(dataset_npz),
        "dataset_n_transitions": int(n),
        "num_features": N_FEATURES,
        "num_policies": int(n_policies),
        "member_policy_kinds": member_kinds,
        "member_checkpoints": member_checkpoints,
        "hidden_dim": int(hidden_dim),
        "train_steps": int(train_steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
        "train_seconds": float(train_seconds),
        "updates_per_second": float(int(train_steps) / max(train_seconds, 1e-9)),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "full_dataset_mse": float(full_mse.detach().cpu()),
        "behavior_policy_argmax_match": float(behavior_match.detach().cpu()),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "uses_slumbot_training_data": False,
        "uses_solver_labels": False,
        "trained_environment_native": True,
        "native_action_projection": False,
        "promotion": False,
        "passed": bool(losses and np.isfinite(losses[-1])),
    }
    if checkpoint_out is not None:
        path = Path(checkpoint_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": metrics["algorithm"],
                "environment": metrics["environment"],
                "num_features": N_FEATURES,
                "num_policies": int(n_policies),
                "hidden_dim": int(hidden_dim),
                "member_policy_kinds": member_kinds,
                "member_checkpoints": member_checkpoints,
                "router_state_dict": net.state_dict(),
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


def load_policy_router_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], PolicyRouterNet]:
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if str(payload.get("environment", "")) != "poker_ai:full_deck_hu_nlhe":
        raise ValueError("policy-router checkpoint must be native full-deck")
    if int(payload.get("num_features", -1)) != N_FEATURES:
        raise ValueError("policy-router feature count does not match native contract")
    n_policies = int(payload.get("num_policies", 0))
    hidden_dim = int(payload.get("hidden_dim", 0))
    if n_policies <= 0 or hidden_dim <= 0:
        raise ValueError("policy-router checkpoint has invalid dimensions")
    net = PolicyRouterNet(hidden_dim=hidden_dim, n_policies=n_policies).to(device)
    net.load_state_dict(payload["router_state_dict"])
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return dict(payload), net


def router_policy_scores(
    router: PolicyRouterNet,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        scores = router(x).detach().float().cpu().numpy()
    return scores[0].astype(np.float32, copy=False)
