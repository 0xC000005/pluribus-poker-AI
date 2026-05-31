"""Maintained-library response-oracle training from joint experience."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.native_nfsp import resolve_device


def _require_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data:
        raise ValueError(f"joint-experience dataset is missing {key}")
    return np.asarray(data[key])


def load_joint_experience_replay_buffer(dataset_npz: str | Path):
    """Load a joint-experience NPZ into a Tianshou replay buffer."""
    from tianshou.data import Batch, ReplayBuffer

    path = Path(dataset_npz)
    data = np.load(path)
    observations = _require_array(data, "observations").astype(np.float32, copy=False)
    next_observations = _require_array(data, "next_observations").astype(np.float32, copy=False)
    legal_masks = _require_array(data, "legal_masks").astype(bool, copy=False)
    next_legal_masks = _require_array(data, "next_legal_masks").astype(bool, copy=False)
    actions = _require_array(data, "actions").astype(np.int64, copy=False)
    rewards = _require_array(data, "returns").astype(np.float32, copy=False)
    dones = _require_array(data, "dones").astype(bool, copy=False)

    if observations.ndim != 2 or observations.shape[1] != N_FEATURES:
        raise ValueError("observations must have shape (n, N_FEATURES)")
    n_transitions = int(observations.shape[0])
    expected_shapes = {
        "next_observations": next_observations.shape == observations.shape,
        "legal_masks": legal_masks.shape == (n_transitions, N_ACTIONS),
        "next_legal_masks": next_legal_masks.shape == (n_transitions, N_ACTIONS),
        "actions": actions.shape == (n_transitions,),
        "returns": rewards.shape == (n_transitions,),
        "dones": dones.shape == (n_transitions,),
    }
    bad_shapes = [key for key, ok in expected_shapes.items() if not ok]
    if bad_shapes:
        raise ValueError("invalid joint-experience shapes: " + ", ".join(bad_shapes))
    if not np.all(legal_masks[np.arange(n_transitions), actions]):
        raise ValueError("joint-experience actions must be legal under legal_masks")

    replay = ReplayBuffer(size=max(n_transitions, 1))
    for transition_i in range(n_transitions):
        replay.add(
            Batch(
                obs=Batch(
                    obs=observations[transition_i],
                    mask=legal_masks[transition_i],
                ),
                act=int(actions[transition_i]),
                rew=float(rewards[transition_i]),
                terminated=bool(dones[transition_i]),
                truncated=False,
                obs_next=Batch(
                    obs=next_observations[transition_i],
                    mask=next_legal_masks[transition_i],
                ),
                info={},
            )
        )
    summary = {
        "dataset_npz": str(path),
        "n_transitions": n_transitions,
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "terminal_transitions": int(dones.sum()),
        "reward_mean": float(rewards.mean()) if n_transitions else 0.0,
        "reward_std": float(rewards.std()) if n_transitions else 0.0,
    }
    return replay, summary


def train_joint_experience_rainbow_response_oracle(
    dataset_npz: str | Path,
    *,
    updates: int = 100,
    batch_size: int = 256,
    hidden_dim: int = 256,
    num_atoms: int = 51,
    lr: float = 1e-3,
    gamma: float = 0.99,
    n_step: int = 1,
    target_update_freq: int = 50,
    seed: int = 20260527,
    device: str = "auto",
    checkpoint_out: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict:
    """Train a Tianshou Rainbow response-oracle checkpoint from joint experience.

    This is a local PSRO/XDO bridge. It does not promote the checkpoint; parent
    and population H2H gates must evaluate any resulting candidate.
    """
    from tianshou.algorithm.algorithm_base import policy_within_training_step
    from tianshou.algorithm.modelfree.c51 import C51Policy
    from tianshou.algorithm.modelfree.rainbow import RainbowDQN
    from tianshou.algorithm.optim import AdamOptimizerFactory

    from scripts.run_tianshou_rainbow_native_control import (
        NativeRainbowPokerEnv,
        _RainbowDistributionNet,
    )

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    replay, dataset_summary = load_joint_experience_replay_buffer(dataset_npz)
    if len(replay) <= 0:
        raise ValueError("joint-experience replay buffer is empty")

    base_env = NativeRainbowPokerEnv()
    model = _RainbowDistributionNet(
        hidden_dim=int(hidden_dim),
        num_atoms=int(num_atoms),
        device=resolved_device,
    ).to(resolved_device)
    policy = C51Policy(
        model=model,
        action_space=base_env.action_space,
        observation_space=base_env.observation_space,
        num_atoms=int(num_atoms),
        v_min=-1.0,
        v_max=1.0,
        eps_training=0.1,
        eps_inference=0.0,
    )
    policy.to(resolved_device)
    algorithm = RainbowDQN(
        policy=policy,
        optim=AdamOptimizerFactory(lr=float(lr)),
        gamma=float(gamma),
        n_step_return_horizon=int(n_step),
        target_update_freq=int(target_update_freq),
    )

    started = time.perf_counter()
    losses: list[float] = []
    for _ in range(int(updates)):
        with policy_within_training_step(algorithm.policy):
            stats = algorithm.update(replay, sample_size=min(int(batch_size), len(replay)))
        loss = getattr(stats, "loss", None)
        if loss is not None:
            losses.append(float(loss))
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started

    metrics = {
        "algorithm": "tianshou_rainbow_joint_experience_response_oracle",
        "role": "psro_response_oracle_from_joint_experience",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": (
            "This is an offline response-oracle training bridge from local "
            "joint experience, not promotion evidence."
        ),
        **device_info,
        **dataset_summary,
        "updates": int(updates),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "num_atoms": int(num_atoms),
        "lr": float(lr),
        "gamma": float(gamma),
        "n_step": int(n_step),
        "target_update_freq": int(target_update_freq),
        "train_seconds": float(train_seconds),
        "updates_per_second": float(int(updates) / max(train_seconds, 1e-9)),
        "last_loss": float(losses[-1]) if losses else None,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "uses_slumbot_training_data": False,
        "promotion": False,
        "league_eligible": False,
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
                "num_atoms": int(num_atoms),
                "model_state_dict": model.state_dict(),
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
