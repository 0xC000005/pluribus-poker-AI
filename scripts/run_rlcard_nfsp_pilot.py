#!/usr/bin/env python3
"""Run a tiny RLCard NFSP pilot against a random no-limit Hold'em agent.

This is a framework-control pilot. RLCard's no-limit environment has a 5-action
space, so these metrics are not Slumbot-parity evidence for the repo's 9-action
engine.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _to_float_list(values) -> list[float]:
    return [float(v) for v in values]


def resolve_device(requested_device: str = "auto") -> dict:
    requested = requested_device.strip().lower()
    cuda_available = bool(torch.cuda.is_available())
    if requested == "auto":
        resolved = "cpu"
        device_policy = "cpu_default_for_rlcard_framework_control"
    elif requested == "cuda":
        if not cuda_available:
            raise ValueError("CUDA requested but torch.cuda.is_available() is false")
        resolved = "cuda"
        device_policy = "explicit_cuda"
    elif requested == "cpu":
        resolved = "cpu"
        device_policy = "explicit_cpu"
    else:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return {
        "requested_device": requested,
        "resolved_device": resolved,
        "device_policy": device_policy,
        "torch_cuda_available": cuda_available,
        "torch_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "torch_device_name": (
            torch.cuda.get_device_name(0) if cuda_available else ""
        ),
    }


def run_pilot(
    *,
    train_episodes: int = 50,
    eval_games: int = 50,
    seed: int = 20260514,
    hidden_dim: int = 32,
    min_buffer_size_to_learn: int = 8,
    device: str = "auto",
    checkpoint_out: str | Path | None = None,
    opponent_kind: str = "random",
) -> dict:
    import rlcard
    from rlcard.agents import NFSPAgent, RandomAgent
    from rlcard.utils import reorganize, tournament

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_info = resolve_device(device)
    min_buffer_size_to_learn = int(min_buffer_size_to_learn)
    rl_batch_size = min(32, max(1, min_buffer_size_to_learn))
    q_batch_size = min(16, max(1, min_buffer_size_to_learn))
    opponent_kind = str(opponent_kind).strip().lower()
    if opponent_kind not in {"random", "nfsp-self-play"}:
        raise ValueError("opponent_kind must be one of: random, nfsp-self-play")

    env = rlcard.make("no-limit-holdem", config={"game_num_players": 2, "seed": seed})
    def _make_nfsp_agent() -> NFSPAgent:
        return NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=[hidden_dim],
            reservoir_buffer_capacity=1024,
            anticipatory_param=0.1,
            batch_size=rl_batch_size,
            train_every=1,
            min_buffer_size_to_learn=min_buffer_size_to_learn,
            q_replay_memory_size=1024,
            q_replay_memory_init_size=min_buffer_size_to_learn,
            q_batch_size=q_batch_size,
            q_train_every=1,
            q_mlp_layers=[hidden_dim],
            evaluate_with="average_policy",
            device=device_info["resolved_device"],
        )

    agent = _make_nfsp_agent()
    random_agent = RandomAgent(num_actions=env.num_actions)
    opponent_agent = _make_nfsp_agent() if opponent_kind == "nfsp-self-play" else random_agent
    env.set_agents([agent, opponent_agent])

    t0 = time.perf_counter()
    pre_payoffs = _to_float_list(tournament(env, int(eval_games)))
    pre_eval_seconds = time.perf_counter() - t0
    train_start = time.perf_counter()
    for _ in range(int(train_episodes)):
        agent.sample_episode_policy()
        if opponent_kind == "nfsp-self-play":
            opponent_agent.sample_episode_policy()
        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)
        train_agents = [agent, opponent_agent] if opponent_kind == "nfsp-self-play" else [agent]
        for player_id, train_agent in enumerate(train_agents):
            for transition in trajectories[player_id]:
                with contextlib.redirect_stdout(io.StringIO()):
                    train_agent.feed(transition)
    if device_info["resolved_device"] == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start
    post_eval_start = time.perf_counter()
    post_payoffs = _to_float_list(tournament(env, int(eval_games)))
    if device_info["resolved_device"] == "cuda":
        torch.cuda.synchronize()
    post_eval_seconds = time.perf_counter() - post_eval_start
    total_seconds = pre_eval_seconds + train_seconds + post_eval_seconds
    checkpoint_path: Path | None = None
    if checkpoint_out is not None:
        checkpoint_path = Path(checkpoint_out)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save_checkpoint(str(checkpoint_path.parent), filename=checkpoint_path.name)

    metrics: dict[str, Any] = {
        "algorithm": "nfsp",
        "role": "framework_control_pilot",
        "environment": "rlcard:no-limit-holdem",
        "warning": "RLCard no-limit Hold'em uses 5 actions; this is not the repo's 9-action Slumbot-parity environment.",
        "trained_environment_native": True,
        "native_action_projection": False,
        "opponent_kind": opponent_kind,
        "self_play_training": bool(opponent_kind == "nfsp-self-play"),
        **device_info,
        "seed": int(seed),
        "num_actions": int(env.num_actions),
        "min_buffer_size_to_learn": int(min_buffer_size_to_learn),
        "rl_batch_size": int(rl_batch_size),
        "q_batch_size": int(q_batch_size),
        "train_episodes": int(train_episodes),
        "eval_games": int(eval_games),
        "pre_payoffs": pre_payoffs,
        "post_payoffs": post_payoffs,
        "delta_player0": float(post_payoffs[0] - pre_payoffs[0]),
        "agent_total_t": int(agent.total_t),
        "opponent_total_t": (
            int(opponent_agent.total_t) if opponent_kind == "nfsp-self-play" else 0
        ),
        "pre_eval_seconds": float(pre_eval_seconds),
        "train_seconds": float(train_seconds),
        "post_eval_seconds": float(post_eval_seconds),
        "eval_seconds": float(pre_eval_seconds + post_eval_seconds),
        "total_seconds": float(total_seconds),
        "episodes_per_second": float(train_episodes / max(train_seconds, 1e-9)),
        "train_steps_per_second": float(agent.total_t / max(train_seconds, 1e-9)),
        "promotion": False,
    }
    if checkpoint_path is not None:
        metrics["checkpoint_out"] = str(checkpoint_path)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--min-buffer-size-to-learn", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--opponent-kind", choices=("random", "nfsp-self-play"), default="random")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = run_pilot(
        train_episodes=args.train_episodes,
        eval_games=args.eval_games,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        min_buffer_size_to_learn=args.min_buffer_size_to_learn,
        device=args.device,
        checkpoint_out=args.checkpoint_out,
        opponent_kind=args.opponent_kind,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
