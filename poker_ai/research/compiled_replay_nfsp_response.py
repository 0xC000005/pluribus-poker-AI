"""NFSP-style average response from compiled native population replay."""

from __future__ import annotations

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
from poker_ai.research.compiled_replay_soft_q_actor import (
    _as_tensor_dataset,
    _write_json,
)
from poker_ai.research.native_nfsp import resolve_device
from poker_ai.research.native_ppo_policy import _PolicyMLP, _QMLP


def _masked_soft_response(
    q_values: torch.Tensor,
    legal_masks: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    temp = max(float(temperature), 1.0e-6)
    logits = (q_values / temp).masked_fill(~legal_masks, -1.0e30)
    probs = torch.softmax(logits, dim=1)
    legal_mass = legal_masks.to(dtype=torch.float32).sum(dim=1, keepdim=True)
    uniform = legal_masks.to(dtype=torch.float32) / torch.clamp(legal_mass, min=1.0)
    bad = (
        (legal_mass.squeeze(1) <= 0)
        | (~torch.isfinite(probs).all(dim=1))
        | (probs.sum(dim=1) <= 0)
    )
    return torch.where(bad.unsqueeze(1), uniform, probs)


def train_nfsp_average_response_from_compiled_replay(
    dataset: dict[str, Any],
    *,
    hidden_dim: int = 256,
    q_train_steps: int = 1000,
    average_train_steps: int = 1000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    response_temperature: float = 1.0,
    seed: int = 20260850,
    device: str = "auto",
    checkpoint_out: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Fit Q from local replay and deploy a separate average-policy network."""

    if int(q_train_steps) <= 0:
        raise ValueError("q_train_steps must be positive")
    if int(average_train_steps) <= 0:
        raise ValueError("average_train_steps must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if float(response_temperature) <= 0.0:
        raise ValueError("response_temperature must be positive")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    device_info = resolve_device(str(device))
    resolved_device = torch.device(device_info["resolved_device"])
    tensors = _as_tensor_dataset(dataset, resolved_device)
    n = int(tensors["actions"].shape[0])
    q_net = _QMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
    avg_net = _PolicyMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
    q_opt = torch.optim.Adam(q_net.parameters(), lr=float(lr))
    avg_opt = torch.optim.Adam(avg_net.parameters(), lr=float(lr))
    generator = torch.Generator(device=resolved_device.type)
    generator.manual_seed(int(seed) + 1000)
    batch_n = min(int(batch_size), n)

    q_losses: list[float] = []
    avg_losses: list[float] = []
    avg_entropies: list[float] = []
    target_entropies: list[float] = []
    target_kls: list[float] = []
    started = time.perf_counter()

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

    for _step in range(int(average_train_steps)):
        idx = torch.randint(n, (batch_n,), generator=generator, device=resolved_device)
        features = tensors["features"][idx]
        legal_masks = tensors["legal_masks"][idx]
        with torch.no_grad():
            target_probs = _masked_soft_response(
                q_net(features),
                legal_masks,
                temperature=float(response_temperature),
            )
            safe_target_log = torch.log(torch.clamp(target_probs, min=1.0e-45))
            target_entropy = -torch.sum(target_probs * safe_target_log, dim=1).mean()
        logits = avg_net(features).masked_fill(~legal_masks, -1.0e30)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        safe_log_probs = torch.nan_to_num(log_probs, neginf=0.0)
        avg_entropy = -torch.sum(
            torch.where(legal_masks, probs * safe_log_probs, torch.zeros_like(probs)),
            dim=1,
        ).mean()
        loss = -torch.sum(target_probs * safe_log_probs, dim=1).mean()
        kl = torch.sum(
            target_probs * (safe_target_log - safe_log_probs),
            dim=1,
        ).mean()
        avg_opt.zero_grad(set_to_none=True)
        loss.backward()
        avg_opt.step()
        avg_losses.append(float(loss.detach().cpu()))
        avg_entropies.append(float(avg_entropy.detach().cpu()))
        target_entropies.append(float(target_entropy.detach().cpu()))
        target_kls.append(float(kl.detach().cpu()))

    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    returns_cpu = tensors["returns"].detach().cpu().numpy()
    metrics: dict[str, Any] = {
        "algorithm": "compiled_replay_nfsp_response",
        "role": "native_compiled_replay_nfsp_average_response",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": "NFSP-style replay average response; not Slumbot or SOTA evidence.",
        **device_info,
        "seed": int(seed),
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "hidden_dim": int(hidden_dim),
        "q_train_steps": int(q_train_steps),
        "average_train_steps": int(average_train_steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "response_temperature": float(response_temperature),
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
        "dataset_meta_strategy": np.asarray(
            dataset.get("meta_strategy", []),
            dtype=np.float32,
        ).tolist(),
        "mean_return": float(np.mean(returns_cpu)) if returns_cpu.size else 0.0,
        "std_return": float(np.std(returns_cpu)) if returns_cpu.size else 0.0,
        "first_q_loss": q_losses[0] if q_losses else None,
        "last_q_loss": q_losses[-1] if q_losses else None,
        "first_average_loss": avg_losses[0] if avg_losses else None,
        "last_average_loss": avg_losses[-1] if avg_losses else None,
        "mean_average_entropy": float(np.mean(avg_entropies)) if avg_entropies else None,
        "last_average_entropy": avg_entropies[-1] if avg_entropies else None,
        "mean_target_entropy": float(np.mean(target_entropies)) if target_entropies else None,
        "last_target_entropy": target_entropies[-1] if target_entropies else None,
        "mean_target_kl": float(np.mean(target_kls)) if target_kls else None,
        "last_target_kl": target_kls[-1] if target_kls else None,
        "train_seconds": float(train_seconds),
        "updates_per_second": float((int(q_train_steps) + int(average_train_steps)) / max(train_seconds, 1e-9)),
        "deployed_policy_object": "average_policy",
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
        and avg_losses
        and np.isfinite(q_losses[-1])
        and np.isfinite(avg_losses[-1])
        and int(dataset.get("needs_python_showdown", 0)) == 0
    )

    if checkpoint_out is not None:
        path = Path(checkpoint_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "compiled_replay_nfsp_response",
                "role": "native_compiled_replay_nfsp_average_response",
                "environment": "poker_ai:full_deck_hu_nlhe",
                "num_actions": N_ACTIONS,
                "num_features": N_FEATURES,
                "hidden_dim": int(hidden_dim),
                "q_net_state_dict": q_net.state_dict(),
                "policy_net_state_dict": avg_net.state_dict(),
                "avg_net_state_dict": avg_net.state_dict(),
                "config": {
                    "feature_mode": "flat",
                    "hidden_dim": int(hidden_dim),
                    "initial_chips": int(dataset.get("initial_chips", 1000)),
                    "max_steps_per_hand": int(dataset.get("max_steps_per_hand", 256)),
                    "train_environment": "poker_ai:full_deck_hu_nlhe",
                    "fsp_average_policy": True,
                    "deployed_policy_object": "average_policy",
                    "response_temperature": float(response_temperature),
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


def run_compiled_replay_nfsp_response(
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
    average_train_steps: int = 1000,
    train_batch_size: int = 1024,
    lr: float = 1e-3,
    response_temperature: float = 1.0,
    seed: int = 20260850,
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
    return train_nfsp_average_response_from_compiled_replay(
        dataset,
        hidden_dim=int(hidden_dim),
        q_train_steps=int(q_train_steps),
        average_train_steps=int(average_train_steps),
        batch_size=int(train_batch_size),
        lr=float(lr),
        response_temperature=float(response_temperature),
        seed=int(seed),
        device=device_info["resolved_device"],
        checkpoint_out=checkpoint_out,
        output_json=output_json,
    )
