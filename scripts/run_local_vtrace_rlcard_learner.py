#!/usr/bin/env python3
"""Train a fresh RLCard-native 5-action checkpoint with local V-trace."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.local_vtrace import masked_log_probs, vtrace_policy_value_loss  # noqa: E402
from poker_ai.research.native_nfsp import resolve_device  # noqa: E402
from poker_ai.research.rlcard_tianshou_ppo import (  # noqa: E402
    N_RLCARD_ACTIONS,
    N_RLCARD_FEATURES,
    RLCardNoLimitHoldemSingleAgentEnv,
    _RLCardCritic,
    _RLCardMaskedActor,
)


def _write_metrics(metrics: dict[str, Any], output_json: str | Path | None) -> dict[str, Any]:
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics["output_json"] = str(path)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _actor_logits(actor: _RLCardMaskedActor, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    logits, _state = actor({"obs": features, "mask": masks})
    return logits


def _normalize_opponent_checkpoints(
    opponent_checkpoint: str | Path | None = None,
    opponent_checkpoints: Sequence[str | Path] | None = None,
) -> list[str]:
    checkpoints: list[str] = []
    if opponent_checkpoint is not None:
        checkpoints.append(str(opponent_checkpoint))
    if opponent_checkpoints is not None:
        checkpoints.extend(str(path) for path in opponent_checkpoints)
    return checkpoints


def _assign_opponent_checkpoints(
    opponent_checkpoints: Sequence[str],
    *,
    n_envs: int,
    seed: int,
) -> list[str | None]:
    checkpoints = [str(path) for path in opponent_checkpoints]
    if not checkpoints:
        return [None for _ in range(int(n_envs))]
    rng = random.Random(int(seed))
    assignments = [checkpoints[i % len(checkpoints)] for i in range(int(n_envs))]
    rng.shuffle(assignments)
    return assignments


def _load_checkpoint_in(
    checkpoint_in: str | Path | None,
    *,
    actor: _RLCardMaskedActor,
    critic: _RLCardCritic,
    hidden_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    if checkpoint_in is None:
        return {
            "initialized_from_checkpoint": False,
            "checkpoint_in": None,
            "checkpoint_in_algorithm": None,
            "checkpoint_in_contract": "same_environment_continuation_only",
        }
    path = Path(checkpoint_in)
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("environment") != "rlcard:no-limit-holdem":
        raise ValueError("checkpoint_in must be trained in rlcard:no-limit-holdem")
    if int(payload.get("num_actions", -1)) != N_RLCARD_ACTIONS:
        raise ValueError("checkpoint_in action count does not match RLCard contract")
    if int(payload.get("num_features", -1)) != N_RLCARD_FEATURES:
        raise ValueError("checkpoint_in feature count does not match RLCard contract")
    if int(payload.get("hidden_dim", -1)) != int(hidden_dim):
        raise ValueError("checkpoint_in hidden_dim does not match requested hidden_dim")
    actor.load_state_dict(payload["actor_state_dict"])
    if "critic_state_dict" in payload:
        critic.load_state_dict(payload["critic_state_dict"])
    return {
        "initialized_from_checkpoint": True,
        "checkpoint_in": str(path),
        "checkpoint_in_algorithm": str(payload.get("algorithm", "unknown")),
        "checkpoint_in_contract": "same_environment_continuation_only",
    }


def run_learner(
    *,
    n_envs: int = 8,
    unroll_length: int = 16,
    train_iterations: int = 4,
    hidden_dim: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    seed: int = 20260561,
    device: str = "auto",
    max_steps_per_hand: int = 256,
    opponent_checkpoint: str | Path | None = None,
    opponent_checkpoints: Sequence[str | Path] | None = None,
    opponent_device: str = "auto",
    checkpoint_in: str | Path | None = None,
    checkpoint_out: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Train a small RLCard-native V-trace actor/value checkpoint."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    opponent_checkpoint_list = _normalize_opponent_checkpoints(
        opponent_checkpoint=opponent_checkpoint,
        opponent_checkpoints=opponent_checkpoints,
    )
    if not opponent_checkpoint_list:
        effective_opponent_device = "cpu"
    elif str(opponent_device) == "auto":
        effective_opponent_device = str(resolved_device)
    else:
        effective_opponent_device = str(resolve_device(str(opponent_device))["resolved_device"])
    opponent_assignments = _assign_opponent_checkpoints(
        opponent_checkpoint_list,
        n_envs=int(n_envs),
        seed=int(seed),
    )

    envs = [
        RLCardNoLimitHoldemSingleAgentEnv(
            seed=int(seed) + env_i,
            observation_format="tianshou",
            max_steps_per_hand=int(max_steps_per_hand),
            opponent_checkpoint=opponent_assignments[env_i],
            opponent_device=effective_opponent_device,
        )
        for env_i in range(int(n_envs))
    ]
    observations = [env.reset(seed=int(seed) + env_i)[0] for env_i, env in enumerate(envs)]
    actor = _RLCardMaskedActor(hidden_dim=int(hidden_dim), device=resolved_device).to(resolved_device)
    critic = _RLCardCritic(hidden_dim=int(hidden_dim), device=resolved_device).to(resolved_device)
    checkpoint_info = _load_checkpoint_in(
        checkpoint_in,
        actor=actor,
        critic=critic,
        hidden_dim=int(hidden_dim),
        device=resolved_device,
    )
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=float(lr))

    losses: list[float] = []
    illegal_probabilities: list[float] = []
    n_samples = 0
    started = time.perf_counter()
    for update_i in range(int(train_iterations)):
        feature_steps: list[np.ndarray] = []
        mask_steps: list[np.ndarray] = []
        action_steps: list[np.ndarray] = []
        behavior_log_prob_steps: list[np.ndarray] = []
        reward_steps: list[np.ndarray] = []
        discount_steps: list[np.ndarray] = []
        for step_i in range(int(unroll_length)):
            features_np = np.stack([obs["obs"] for obs in observations]).astype(np.float32)
            masks_np = np.stack([obs["mask"] for obs in observations]).astype(bool)
            features = torch.as_tensor(features_np, dtype=torch.float32, device=resolved_device)
            masks = torch.as_tensor(masks_np, dtype=torch.bool, device=resolved_device)
            with torch.no_grad():
                logits = _actor_logits(actor, features, masks)
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
                    obs = env.reset(
                        seed=int(seed) + 10_000 + env_i + update_i * 1_000 + step_i
                    )[0]
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
        flat_masks = masks_t.reshape(-1, N_RLCARD_ACTIONS)
        logits_t = _actor_logits(actor, flat_features, flat_masks).reshape(
            int(unroll_length),
            int(n_envs),
            N_RLCARD_ACTIONS,
        )
        values_t = critic({"obs": flat_features}).reshape(int(unroll_length), int(n_envs))
        bootstrap_features = torch.as_tensor(
            np.stack([obs["obs"] for obs in observations]).astype(np.float32),
            dtype=torch.float32,
            device=resolved_device,
        )
        with torch.no_grad():
            bootstrap_value = critic({"obs": bootstrap_features}).reshape(int(n_envs))
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
        n_samples += int(n_envs) * int(unroll_length)

    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    max_illegal_probability = float(max(illegal_probabilities) if illegal_probabilities else 0.0)
    checkpoint_path = None if checkpoint_out is None else str(Path(checkpoint_out))
    metrics: dict[str, Any] = {
        "algorithm": "local_vtrace_rlcard",
        "role": "rlcard_native_vtrace_actor_learner",
        "environment": "rlcard:no-limit-holdem",
        "warning": "RLCard 5-action public-reference checkpoint; not native Slumbot evidence.",
        **device_info,
        "seed": int(seed),
        "num_actions": N_RLCARD_ACTIONS,
        "num_features": N_RLCARD_FEATURES,
        "n_envs": int(n_envs),
        "unroll_length": int(unroll_length),
        "train_iterations": int(train_iterations),
        "max_steps_per_hand": int(max_steps_per_hand),
        "hidden_dim": int(hidden_dim),
        "lr": float(lr),
        "gamma": float(gamma),
        **checkpoint_info,
        "opponent_kind": "rlcard-vtrace-or-ppo-population" if opponent_checkpoint_list else "random",
        "opponent_checkpoint": opponent_checkpoint_list[0] if len(opponent_checkpoint_list) == 1 else None,
        "opponent_checkpoints": opponent_checkpoint_list,
        "opponent_checkpoint_assignments": opponent_assignments,
        "opponent_population_size": int(len(opponent_checkpoint_list)),
        "opponent_device": effective_opponent_device if opponent_checkpoint_list else None,
        "population_training": bool(opponent_checkpoint_list),
        "n_samples": int(n_samples),
        "train_seconds": float(train_seconds),
        "samples_per_second": float(n_samples / max(train_seconds, 1e-9)),
        "last_loss": float(losses[-1]) if losses else None,
        "loss_is_finite": bool(losses and np.isfinite(losses[-1])),
        "illegal_action_probability": max_illegal_probability,
        "trained_environment_native": True,
        "native_action_projection": False,
        "rlcard_candidate": True,
        "uses_slumbot_data": False,
        "uses_alphanlholdem_training_data": False,
        "promotion": False,
        "checkpoint_path": checkpoint_path,
    }
    metrics["passed"] = bool(
        losses
        and np.isfinite(losses[-1])
        and max_illegal_probability == 0.0
        and n_samples > 0
    )

    if checkpoint_out is not None:
        path = Path(checkpoint_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "local_vtrace_rlcard",
                "environment": "rlcard:no-limit-holdem",
                "num_actions": N_RLCARD_ACTIONS,
                "num_features": N_RLCARD_FEATURES,
                "hidden_dim": int(hidden_dim),
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "metrics": metrics,
                "parent_checkpoint": None if checkpoint_in is None else str(Path(checkpoint_in)),
                "trained_environment_native": True,
                "native_action_projection": False,
                "rlcard_candidate": True,
                "uses_slumbot_data": False,
                "uses_alphanlholdem_training_data": False,
            },
            path,
        )
    return _write_metrics(metrics, output_json)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--unroll-length", type=int, default=16)
    parser.add_argument("--train-iterations", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=20260561)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps-per-hand", type=int, default=256)
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        action="append",
        dest="opponent_checkpoints",
        help="RLCard-native opponent checkpoint. Repeat for a local population mixture.",
    )
    parser.add_argument("--opponent-device", default="auto")
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    metrics = run_learner(
        n_envs=args.n_envs,
        unroll_length=args.unroll_length,
        train_iterations=args.train_iterations,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
        device=args.device,
        max_steps_per_hand=args.max_steps_per_hand,
        opponent_checkpoints=args.opponent_checkpoints,
        opponent_device=args.opponent_device,
        checkpoint_in=args.checkpoint_in,
        checkpoint_out=args.checkpoint_out,
        output_json=args.output_json,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
