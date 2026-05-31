#!/usr/bin/env python3
"""Run plug-in Tianshou multi-agent PPO on native PettingZoo poker."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES  # noqa: E402
from poker_ai.research.native_nfsp import resolve_device  # noqa: E402
from poker_ai.research.native_pettingzoo import NativeNoLimitHoldemAECEnv  # noqa: E402
from scripts.run_tianshou_ppo_native_control import _Critic, _MaskedActor, _build_ppo_policy_from_actor  # noqa: E402


def _make_pettingzoo_env(seed: int, initial_chips: int, max_steps_per_hand: int):
    from tianshou.env import PettingZooEnv

    return PettingZooEnv(
        NativeNoLimitHoldemAECEnv(
            seed=int(seed),
            initial_chips=int(initial_chips),
            max_steps_per_hand=int(max_steps_per_hand),
        )
    )


def run_marl_ppo_control(
    *,
    rollout_steps: int = 1024,
    updates: int = 10,
    repeat: int = 2,
    batch_size: int = 256,
    hidden_dim: int = 128,
    replay_size: int = 20_000,
    num_envs: int = 1,
    vector_env_backend: str = "dummy",
    shared_policy: bool = True,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    initial_chips: int = 1000,
    max_steps_per_hand: int = 256,
    seed: int = 20260526,
    device: str = "auto",
    output_json: str | None = None,
    checkpoint_out: str | None = None,
) -> dict:
    from tianshou.algorithm.algorithm_base import policy_within_training_step
    from tianshou.algorithm.modelfree.ppo import PPO
    from tianshou.algorithm.multiagent.marl import MultiAgentOnPolicyAlgorithm
    from tianshou.algorithm.optim import AdamOptimizerFactory
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.env import DummyVectorEnv, SubprocVectorEnv

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    num_envs = max(1, int(num_envs))
    vector_env_backend = str(vector_env_backend)
    if vector_env_backend not in {"dummy", "subproc"}:
        raise ValueError("vector_env_backend must be one of: dummy, subproc")

    base_env = _make_pettingzoo_env(seed, initial_chips, max_steps_per_hand)
    env_fns = [
        partial(
            _make_pettingzoo_env,
            int(seed) + env_i,
            int(initial_chips),
            int(max_steps_per_hand),
        )
        for env_i in range(num_envs)
    ]
    vector_cls = DummyVectorEnv if vector_env_backend == "dummy" else SubprocVectorEnv
    env = vector_cls(env_fns)

    actors = {}
    critics = {}
    algorithms = []
    shared_actor = None
    shared_critic = None
    shared_algorithm = None
    if shared_policy:
        shared_actor = _MaskedActor(hidden_dim=int(hidden_dim), device=resolved_device).to(resolved_device)
        shared_critic = _Critic(hidden_dim=int(hidden_dim), device=resolved_device).to(resolved_device)
        shared_policy_obj = _build_ppo_policy_from_actor(
            actor=shared_actor,
            action_space=base_env.action_space,
            observation_space=base_env.observation_space,
        )
        shared_algorithm = PPO(
            policy=shared_policy_obj,
            critic=shared_critic,
            optim=AdamOptimizerFactory(lr=float(lr)),
            gamma=float(gamma),
            gae_lambda=float(gae_lambda),
            ent_coef=float(ent_coef),
            vf_coef=float(vf_coef),
            max_batchsize=int(batch_size),
        )
        shared_algorithm.to(resolved_device)

    for agent_id in base_env.agents:
        if shared_policy:
            assert shared_actor is not None
            assert shared_critic is not None
            assert shared_algorithm is not None
            actor = shared_actor
            critic = shared_critic
            algorithms.append(shared_algorithm)
        else:
            actor = _MaskedActor(hidden_dim=int(hidden_dim), device=resolved_device).to(resolved_device)
            critic = _Critic(hidden_dim=int(hidden_dim), device=resolved_device).to(resolved_device)
            policy = _build_ppo_policy_from_actor(
                actor=actor,
                action_space=base_env.action_space,
                observation_space=base_env.observation_space,
            )
            algorithm = PPO(
                policy=policy,
                critic=critic,
                optim=AdamOptimizerFactory(lr=float(lr)),
                gamma=float(gamma),
                gae_lambda=float(gae_lambda),
                ent_coef=float(ent_coef),
                vf_coef=float(vf_coef),
                max_batchsize=int(batch_size),
            )
            algorithm.to(resolved_device)
            algorithms.append(algorithm)
        actors[str(agent_id)] = actor
        critics[str(agent_id)] = critic

    marl_algorithm = MultiAgentOnPolicyAlgorithm(algorithms=algorithms, env=base_env)
    replay = VectorReplayBuffer(total_size=int(replay_size), buffer_num=num_envs)
    collector = Collector(marl_algorithm, env, replay, exploration_noise=True)

    train_start = time.perf_counter()
    collected_steps = 0
    update_train_times: list[float] = []
    for update_i in range(int(updates)):
        stats = collector.collect(
            n_step=max(int(rollout_steps), 1),
            reset_before_collect=update_i == 0,
        )
        collected_steps += int(getattr(stats, "n_collected_steps", rollout_steps))
        with policy_within_training_step(marl_algorithm.policy):
            train_stats = marl_algorithm.update(replay, batch_size=int(batch_size), repeat=int(repeat))
        train_time = getattr(train_stats, "train_time", None)
        if train_time is not None:
            update_train_times.append(float(train_time))
        replay.reset()
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start

    actor_state = shared_actor.state_dict() if shared_actor is not None else actors["player_0"].state_dict()
    critic_state = (
        shared_critic.state_dict() if shared_critic is not None else critics["player_0"].state_dict()
    )
    metrics = {
        "algorithm": "tianshou_marl_ppo",
        "role": "plug_in_multi_agent_policy_value_rl_control",
        "environment": "poker_ai:pettingzoo_full_deck_hu_nlhe",
        "warning": (
            "This is a maintained-library stochastic policy/value MARL control "
            "on the native PettingZoo adapter, not Slumbot/SOTA evidence."
        ),
        **device_info,
        "num_agents": len(base_env.agents),
        "agents": [str(agent) for agent in base_env.agents],
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "rollout_steps": int(rollout_steps),
        "train_steps": int(collected_steps),
        "updates": int(updates),
        "repeat": int(repeat),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "num_envs": int(num_envs),
        "vector_env_backend": vector_env_backend,
        "shared_policy": bool(shared_policy),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "ent_coef": float(ent_coef),
        "vf_coef": float(vf_coef),
        "train_seconds": float(train_seconds),
        "train_steps_per_second": float(collected_steps / max(train_seconds, 1e-9)),
        "mean_update_train_time": float(np.mean(update_train_times)) if update_train_times else None,
        "promotion": False,
        "league_eligible": False,
    }
    if checkpoint_out:
        path = Path(checkpoint_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": metrics["algorithm"],
                "environment": metrics["environment"],
                "num_actions": N_ACTIONS,
                "num_features": N_FEATURES,
                "hidden_dim": int(hidden_dim),
                "shared_policy": bool(shared_policy),
                "actor_state_dict": actor_state,
                "critic_state_dict": critic_state,
                "agent_actor_state_dicts": {
                    agent_id: actor.state_dict() for agent_id, actor in actors.items()
                },
                "agent_critic_state_dicts": {
                    agent_id: critic.state_dict() for agent_id, critic in critics.items()
                },
                "metrics": metrics,
            },
            path,
        )
        metrics["checkpoint_path"] = str(path)
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=20_000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--vector-env-backend", choices=("dummy", "subproc"), default="dummy")
    parser.add_argument("--shared-policy", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--max-steps-per-hand", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-json")
    parser.add_argument("--checkpoint-out")
    args = parser.parse_args(argv)
    metrics = run_marl_ppo_control(
        rollout_steps=args.rollout_steps,
        updates=args.updates,
        repeat=args.repeat,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        replay_size=args.replay_size,
        num_envs=args.num_envs,
        vector_env_backend=args.vector_env_backend,
        shared_policy=args.shared_policy,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        initial_chips=args.initial_chips,
        max_steps_per_hand=args.max_steps_per_hand,
        seed=args.seed,
        device=args.device,
        output_json=args.output_json,
        checkpoint_out=args.checkpoint_out,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
