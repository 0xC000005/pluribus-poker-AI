#!/usr/bin/env python3
"""Run plug-in Tianshou multi-agent Rainbow on native PettingZoo poker.

This is a maintained-library RL control. The repo supplies the native
PettingZoo adapter and metrics; Tianshou supplies the learning algorithms.
"""

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
from scripts.run_tianshou_rainbow_native_control import _RainbowDistributionNet  # noqa: E402


def _make_pettingzoo_env(
    seed: int,
    initial_chips: int,
    max_steps_per_hand: int,
    state_backend: str = "full-deck",
):
    from tianshou.env import PettingZooEnv

    return PettingZooEnv(
        NativeNoLimitHoldemAECEnv(
            seed=int(seed),
            initial_chips=int(initial_chips),
            max_steps_per_hand=int(max_steps_per_hand),
            state_backend=state_backend,
        )
    )


def _load_checkpoint_payload(checkpoint_in: str | None, resolved_device: torch.device) -> dict | None:
    if not checkpoint_in:
        return None
    payload = torch.load(checkpoint_in, map_location=resolved_device, weights_only=False)
    if int(payload.get("num_actions", -1)) != N_ACTIONS:
        raise ValueError("checkpoint action count does not match native full-deck contract")
    if int(payload.get("num_features", -1)) != N_FEATURES:
        raise ValueError("checkpoint feature count does not match native full-deck contract")
    return payload


def _state_dict_for_agent(payload: dict, agent_id: str) -> dict:
    if payload.get("shared_model_state_dict") is not None:
        return payload["shared_model_state_dict"]
    if "agent_model_state_dicts" in payload:
        agent_state_dicts = payload["agent_model_state_dicts"]
        if agent_id in agent_state_dicts:
            return agent_state_dicts[agent_id]
        if "player_0" in agent_state_dicts:
            return agent_state_dicts["player_0"]
        return agent_state_dicts[sorted(agent_state_dicts)[0]]
    if "model_state_dict" in payload:
        return payload["model_state_dict"]
    raise ValueError("checkpoint does not contain a loadable model state dict")


def run_marl_control(
    *,
    train_steps: int = 200,
    updates: int = 10,
    batch_size: int = 32,
    hidden_dim: int = 128,
    num_atoms: int = 51,
    warmup_steps: int | None = None,
    lr: float = 1e-3,
    gamma: float = 0.99,
    n_step: int = 1,
    target_update_freq: int = 50,
    replay_size: int = 10_000,
    num_envs: int = 1,
    vector_env_backend: str = "dummy",
    state_backend: str = "full-deck",
    shared_policy: bool = False,
    initial_chips: int = 1000,
    max_steps_per_hand: int = 256,
    seed: int = 20260526,
    device: str = "auto",
    output_json: str | None = None,
    checkpoint_in: str | None = None,
    checkpoint_out: str | None = None,
) -> dict:
    from tianshou.algorithm.algorithm_base import policy_within_training_step
    from tianshou.algorithm.modelfree.c51 import C51Policy
    from tianshou.algorithm.modelfree.rainbow import RainbowDQN
    from tianshou.algorithm.multiagent.marl import MultiAgentOffPolicyAlgorithm
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
    state_backend = str(state_backend)
    if state_backend not in {"full-deck", "fast-state"}:
        raise ValueError("state_backend must be one of: full-deck, fast-state")

    base_env = _make_pettingzoo_env(seed, initial_chips, max_steps_per_hand, state_backend)
    checkpoint_payload = _load_checkpoint_payload(checkpoint_in, resolved_device)
    env_fns = [
        partial(
            _make_pettingzoo_env,
            int(seed) + env_i,
            int(initial_chips),
            int(max_steps_per_hand),
            state_backend,
        )
        for env_i in range(num_envs)
    ]
    vector_cls = DummyVectorEnv if vector_env_backend == "dummy" else SubprocVectorEnv
    env = vector_cls(env_fns)

    models = {}
    algorithms = []
    shared_model = None
    shared_algorithm = None
    if shared_policy:
        shared_model = _RainbowDistributionNet(
            hidden_dim=int(hidden_dim),
            num_atoms=int(num_atoms),
            device=resolved_device,
        ).to(resolved_device)
        shared_policy_obj = C51Policy(
            model=shared_model,
            action_space=base_env.action_space,
            observation_space=base_env.observation_space,
            num_atoms=int(num_atoms),
            v_min=-1.0,
            v_max=1.0,
            eps_training=0.1,
            eps_inference=0.0,
        )
        shared_policy_obj.to(resolved_device)
        if checkpoint_payload is not None:
            shared_model.load_state_dict(_state_dict_for_agent(checkpoint_payload, "player_0"))
        shared_algorithm = RainbowDQN(
            policy=shared_policy_obj,
            optim=AdamOptimizerFactory(lr=float(lr)),
            gamma=float(gamma),
            n_step_return_horizon=int(n_step),
            target_update_freq=int(target_update_freq),
        )
    for agent_id in base_env.agents:
        if shared_policy:
            assert shared_model is not None
            assert shared_algorithm is not None
            model = shared_model
            algorithms.append(shared_algorithm)
        else:
            model = _RainbowDistributionNet(
                hidden_dim=int(hidden_dim),
                num_atoms=int(num_atoms),
                device=resolved_device,
            ).to(resolved_device)
            if checkpoint_payload is not None:
                model.load_state_dict(_state_dict_for_agent(checkpoint_payload, str(agent_id)))
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
            algorithms.append(
                RainbowDQN(
                    policy=policy,
                    optim=AdamOptimizerFactory(lr=float(lr)),
                    gamma=float(gamma),
                    n_step_return_horizon=int(n_step),
                    target_update_freq=int(target_update_freq),
                )
            )
        models[str(agent_id)] = model

    marl_algorithm = MultiAgentOffPolicyAlgorithm(algorithms=algorithms, env=base_env)
    replay = VectorReplayBuffer(total_size=int(replay_size), buffer_num=num_envs)
    collector = Collector(marl_algorithm, env, replay, exploration_noise=True)

    train_start = time.perf_counter()
    warmup = int(warmup_steps if warmup_steps is not None else max(batch_size, train_steps))
    warmup_stats = collector.collect(n_step=max(warmup, 1), reset_before_collect=True)
    collected_steps = int(getattr(warmup_stats, "n_collected_steps", warmup))
    update_train_times: list[float] = []
    for _ in range(int(updates)):
        collect_stats = collector.collect(n_step=max(int(train_steps), 1))
        collected_steps += int(getattr(collect_stats, "n_collected_steps", train_steps))
        with policy_within_training_step(marl_algorithm.policy):
            stats = marl_algorithm.update(replay, sample_size=min(int(batch_size), len(replay)))
        train_time = getattr(stats, "train_time", None)
        if train_time is not None:
            update_train_times.append(float(train_time))
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start

    metrics = {
        "algorithm": "tianshou_marl_rainbow_dqn",
        "role": "plug_in_multi_agent_rl_control",
        "environment": "poker_ai:pettingzoo_full_deck_hu_nlhe",
        "warning": (
            "This is a maintained-library multi-agent RL control on the native "
            "PettingZoo adapter, not Slumbot/SOTA evidence."
        ),
        **device_info,
        "num_agents": len(base_env.agents),
        "agents": [str(agent) for agent in base_env.agents],
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "train_steps": int(collected_steps),
        "updates": int(updates),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "num_atoms": int(num_atoms),
        "num_envs": int(num_envs),
        "vector_env_backend": vector_env_backend,
        "state_backend": state_backend,
        "shared_policy": bool(shared_policy),
        "n_step": int(n_step),
        "target_update_freq": int(target_update_freq),
        "train_seconds": float(train_seconds),
        "train_steps_per_second": float(collected_steps / max(train_seconds, 1e-9)),
        "mean_update_train_time": float(np.mean(update_train_times)) if update_train_times else None,
        "checkpoint_in": str(checkpoint_in) if checkpoint_in else None,
        "checkpoint_in_algorithm": (
            str(checkpoint_payload.get("algorithm")) if checkpoint_payload is not None else None
        ),
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
                "num_atoms": int(num_atoms),
                "shared_policy": bool(shared_policy),
                "shared_model_state_dict": (
                    shared_model.state_dict() if shared_model is not None else None
                ),
                "parent_checkpoint": str(checkpoint_in) if checkpoint_in else None,
                "agent_model_state_dicts": {
                    agent_id: model.state_dict() for agent_id, model in models.items()
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
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-atoms", type=int, default=51)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--target-update-freq", type=int, default=50)
    parser.add_argument("--replay-size", type=int, default=10_000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--vector-env-backend", choices=("dummy", "subproc"), default="dummy")
    parser.add_argument("--state-backend", choices=("full-deck", "fast-state"), default="full-deck")
    parser.add_argument("--shared-policy", action="store_true")
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--max-steps-per-hand", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-json")
    parser.add_argument("--checkpoint-in")
    parser.add_argument("--checkpoint-out")
    args = parser.parse_args(argv)
    metrics = run_marl_control(
        train_steps=args.train_steps,
        updates=args.updates,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_atoms=args.num_atoms,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        gamma=args.gamma,
        n_step=args.n_step,
        target_update_freq=args.target_update_freq,
        replay_size=args.replay_size,
        num_envs=args.num_envs,
        vector_env_backend=args.vector_env_backend,
        state_backend=args.state_backend,
        shared_policy=args.shared_policy,
        initial_chips=args.initial_chips,
        max_steps_per_hand=args.max_steps_per_hand,
        seed=args.seed,
        device=args.device,
        output_json=args.output_json,
        checkpoint_in=args.checkpoint_in,
        checkpoint_out=args.checkpoint_out,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
