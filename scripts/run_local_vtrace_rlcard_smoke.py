#!/usr/bin/env python3
"""Tiny local V-trace smoke on RLCard no-limit Hold'em trajectories."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.local_vtrace import masked_log_probs, vtrace_policy_value_loss  # noqa: E402
from poker_ai.research.native_nfsp import resolve_device  # noqa: E402
from poker_ai.research.rlcard_tianshou_ppo import (  # noqa: E402
    N_RLCARD_ACTIONS,
    N_RLCARD_FEATURES,
    RLCardNoLimitHoldemSingleAgentEnv,
)


class _ActorCritic(nn.Module):
    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(N_RLCARD_FEATURES, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.policy = nn.Linear(int(hidden_dim), N_RLCARD_ACTIONS)
        self.value = nn.Linear(int(hidden_dim), 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(obs)
        return self.policy(x), self.value(x).squeeze(-1)


def _write_metrics(metrics: dict[str, Any], output_json: str | Path | None) -> dict[str, Any]:
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics["output_json"] = str(path)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def run_smoke(
    *,
    n_envs: int = 8,
    unroll_length: int = 16,
    updates: int = 1,
    hidden_dim: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    seed: int = 20260528,
    device: str = "auto",
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    envs = [
        RLCardNoLimitHoldemSingleAgentEnv(seed=int(seed) + i, observation_format="tianshou")
        for i in range(int(n_envs))
    ]
    observations = [env.reset(seed=int(seed) + i)[0] for i, env in enumerate(envs)]
    model = _ActorCritic(hidden_dim=int(hidden_dim)).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    losses: list[float] = []
    illegal_probabilities: list[float] = []
    start = time.perf_counter()
    for _update in range(int(updates)):
        feature_steps: list[np.ndarray] = []
        mask_steps: list[np.ndarray] = []
        action_steps: list[np.ndarray] = []
        behavior_log_prob_steps: list[np.ndarray] = []
        reward_steps: list[np.ndarray] = []
        discount_steps: list[np.ndarray] = []
        for _t in range(int(unroll_length)):
            features_np = np.stack([obs["obs"] for obs in observations]).astype(np.float32)
            masks_np = np.stack([obs["mask"] for obs in observations]).astype(bool)
            features = torch.as_tensor(features_np, dtype=torch.float32, device=resolved_device)
            masks = torch.as_tensor(masks_np, dtype=torch.bool, device=resolved_device)
            with torch.no_grad():
                logits, _values = model(features)
                log_probs = masked_log_probs(logits, masks)
                probs = torch.exp(log_probs)
                actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
                behavior_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

            rewards = np.zeros((int(n_envs),), dtype=np.float32)
            discounts = np.full((int(n_envs),), float(gamma), dtype=np.float32)
            next_observations = []
            for env_i, env in enumerate(envs):
                obs, reward, terminated, truncated, _info = env.step(int(actions[env_i].cpu()))
                rewards[env_i] = float(reward)
                if terminated or truncated:
                    discounts[env_i] = 0.0
                    obs = env.reset(seed=int(seed) + 10_000 + env_i + _update * 1_000 + _t)[0]
                next_observations.append(obs)
            feature_steps.append(features_np)
            mask_steps.append(masks_np)
            action_steps.append(actions.detach().cpu().numpy().astype(np.int64))
            behavior_log_prob_steps.append(behavior_log_probs.detach().cpu().numpy().astype(np.float32))
            reward_steps.append(rewards)
            discount_steps.append(discounts)
            observations = next_observations

        features_t = torch.as_tensor(np.stack(feature_steps), dtype=torch.float32, device=resolved_device)
        masks_t = torch.as_tensor(np.stack(mask_steps), dtype=torch.bool, device=resolved_device)
        actions_t = torch.as_tensor(np.stack(action_steps), dtype=torch.long, device=resolved_device)
        behavior_log_probs_t = torch.as_tensor(
            np.stack(behavior_log_prob_steps),
            dtype=torch.float32,
            device=resolved_device,
        )
        rewards_t = torch.as_tensor(np.stack(reward_steps), dtype=torch.float32, device=resolved_device)
        discounts_t = torch.as_tensor(np.stack(discount_steps), dtype=torch.float32, device=resolved_device)
        flat_features = features_t.reshape(-1, N_RLCARD_FEATURES)
        logits_flat, values_flat = model(flat_features)
        logits_t = logits_flat.reshape(int(unroll_length), int(n_envs), N_RLCARD_ACTIONS)
        values_t = values_flat.reshape(int(unroll_length), int(n_envs))
        bootstrap_features = torch.as_tensor(
            np.stack([obs["obs"] for obs in observations]).astype(np.float32),
            dtype=torch.float32,
            device=resolved_device,
        )
        with torch.no_grad():
            _bootstrap_logits, bootstrap_value = model(bootstrap_features)
        loss, stats = vtrace_policy_value_loss(
            logits=logits_t,
            values=values_t,
            actions=actions_t,
            legal_mask=masks_t,
            behavior_action_log_probs=behavior_log_probs_t,
            rewards=rewards_t,
            discounts=discounts_t,
            bootstrap_value=bootstrap_value,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        illegal_probabilities.append(float(stats["illegal_action_probability"]))

    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    n_samples = int(n_envs) * int(unroll_length) * int(updates)
    metrics: dict[str, Any] = {
        "algorithm": "local_vtrace_rlcard_smoke",
        "role": "local_vtrace_learner_smoke",
        "environment": "rlcard:no-limit-holdem",
        "warning": "This is a learner plumbing smoke, not policy-strength evidence.",
        **device_info,
        "seed": int(seed),
        "n_envs": int(n_envs),
        "unroll_length": int(unroll_length),
        "updates": int(updates),
        "hidden_dim": int(hidden_dim),
        "lr": float(lr),
        "gamma": float(gamma),
        "n_samples": int(n_samples),
        "train_seconds": float(elapsed),
        "samples_per_second": float(n_samples / max(elapsed, 1e-9)),
        "last_loss": float(losses[-1]) if losses else None,
        "loss_is_finite": bool(losses and np.isfinite(losses[-1])),
        "illegal_action_probability": float(max(illegal_probabilities) if illegal_probabilities else 0.0),
        "trained_environment_native": True,
        "native_action_projection": False,
        "uses_slumbot_data": False,
        "uses_alphanlholdem_training_data": False,
        "promotion": False,
        "passed": bool(losses and np.isfinite(losses[-1]) and max(illegal_probabilities) == 0.0),
    }
    return _write_metrics(metrics, output_json)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--unroll-length", type=int, default=16)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    metrics = run_smoke(
        n_envs=args.n_envs,
        unroll_length=args.unroll_length,
        updates=args.updates,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
        device=args.device,
        output_json=args.output_json,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
