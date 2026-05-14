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
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _to_float_list(values) -> list[float]:
    return [float(v) for v in values]


def run_pilot(
    *,
    train_episodes: int = 50,
    eval_games: int = 50,
    seed: int = 20260514,
    hidden_dim: int = 32,
    min_buffer_size_to_learn: int = 8,
) -> dict:
    import rlcard
    from rlcard.agents import NFSPAgent, RandomAgent
    from rlcard.utils import reorganize, tournament

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = rlcard.make("no-limit-holdem", config={"game_num_players": 2, "seed": seed})
    agent = NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=[hidden_dim],
        reservoir_buffer_capacity=1024,
        anticipatory_param=0.1,
        batch_size=32,
        train_every=1,
        min_buffer_size_to_learn=min_buffer_size_to_learn,
        q_replay_memory_size=1024,
        q_replay_memory_init_size=min_buffer_size_to_learn,
        q_batch_size=16,
        q_train_every=1,
        q_mlp_layers=[hidden_dim],
        evaluate_with="average_policy",
        device="cpu",
    )
    random_agent = RandomAgent(num_actions=env.num_actions)
    env.set_agents([agent, random_agent])

    pre_payoffs = _to_float_list(tournament(env, int(eval_games)))
    for _ in range(int(train_episodes)):
        agent.sample_episode_policy()
        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)
        for transition in trajectories[0]:
            with contextlib.redirect_stdout(io.StringIO()):
                agent.feed(transition)
    post_payoffs = _to_float_list(tournament(env, int(eval_games)))

    return {
        "algorithm": "nfsp",
        "role": "framework_control_pilot",
        "environment": "rlcard:no-limit-holdem",
        "warning": "RLCard no-limit Hold'em uses 5 actions; this is not the repo's 9-action Slumbot-parity environment.",
        "seed": int(seed),
        "num_actions": int(env.num_actions),
        "train_episodes": int(train_episodes),
        "eval_games": int(eval_games),
        "pre_payoffs": pre_payoffs,
        "post_payoffs": post_payoffs,
        "delta_player0": float(post_payoffs[0] - pre_payoffs[0]),
        "agent_total_t": int(agent.total_t),
        "promotion": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--min-buffer-size-to-learn", type=int, default=8)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = run_pilot(
        train_episodes=args.train_episodes,
        eval_games=args.eval_games,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        min_buffer_size_to_learn=args.min_buffer_size_to_learn,
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

