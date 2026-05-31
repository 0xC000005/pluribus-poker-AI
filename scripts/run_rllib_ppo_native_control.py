#!/usr/bin/env python3
"""Run RLlib PPO on the native 9-action poker PettingZoo adapter.

RLlib supplies PPO, action masking, and multi-agent policy mapping. This repo
only supplies the native poker environment adapter and metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES  # noqa: E402
from poker_ai.research.rllib_native import RllibNativeNoLimitHoldemAECEnv  # noqa: E402


def _normalize_checkpoint_list(
    checkpoints: str | Path | Sequence[str | Path] | None,
) -> list[str]:
    if checkpoints is None:
        return []
    if isinstance(checkpoints, (str, Path)):
        return [str(checkpoints)]
    return [str(path) for path in checkpoints if str(path)]


def _resolve_checkpoint_list(
    checkpoints: str | Path | Sequence[str | Path] | None,
) -> list[str]:
    return [str(Path(path).resolve()) for path in _normalize_checkpoint_list(checkpoints)]


class RllibNativeResponseOracleEnv(gym.Env):
    """Single-agent RLlib env for learning against a frozen local population."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        from gymnasium import spaces

        from scripts.run_tianshou_rainbow_native_control import NativeRainbowPokerEnv

        cfg = dict(config or {})
        self._base = NativeRainbowPokerEnv(
            seed=int(cfg.get("seed", 20260725)),
            initial_chips=int(cfg.get("initial_chips", 1000)),
            max_steps_per_hand=int(cfg.get("max_steps_per_hand", 256)),
            opponent_kind=str(cfg.get("opponent_kind", "random")),
            opponent_checkpoint=_normalize_checkpoint_list(cfg.get("opponent_checkpoint")),
            opponent_device=str(cfg.get("opponent_device", "cpu")),
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "observations": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(N_FEATURES,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(0, 1, shape=(N_ACTIONS,), dtype=np.int8),
            }
        )
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self._base.reset(seed=seed, options=options)
        return self._convert_obs(obs), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._base.step(int(action))
        return self._convert_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return None

    def close(self) -> None:
        return self._base.close() if hasattr(self._base, "close") else None

    @staticmethod
    def _convert_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            "observations": np.asarray(obs["obs"], dtype=np.float32),
            "action_mask": np.asarray(obs["mask"], dtype=np.int8),
        }


def _result_value(result: dict[str, Any], key: str) -> float | None:
    value = result.get(key)
    if value is None:
        value = result.get("env_runners", {}).get(key)
    return float(value) if value is not None else None


def run_control(
    *,
    train_iterations: int = 1,
    train_batch_size: int = 128,
    minibatch_size: int = 64,
    num_epochs: int = 1,
    rollout_fragment_length: int = 64,
    num_env_runners: int = 0,
    num_cpus: int = 4,
    num_gpus: float = 0.0,
    hidden_dim: int = 64,
    initial_chips: int = 1000,
    max_steps_per_hand: int = 32,
    seed: int = 20260718,
    disable_env_checking: bool = True,
    opponent_kind: str = "self",
    opponent_checkpoint: str | Path | Sequence[str | Path] | None = None,
    opponent_device: str = "cpu",
    output_json: str | None = None,
    checkpoint_out: str | None = None,
) -> dict[str, Any]:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
    from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
        ActionMaskingTorchRLModule,
    )
    from ray.tune.registry import register_env

    normalized_opponent_kind = str(opponent_kind).strip().lower()
    opponent_checkpoints = _resolve_checkpoint_list(opponent_checkpoint)
    response_oracle = normalized_opponent_kind not in {"self", "self-play", "self_play"}
    env_name = (
        f"native_response_oracle_hu_nlhe_rllib_{int(seed)}"
        if response_oracle
        else f"native_hu_nlhe_rllib_{int(seed)}"
    )

    def _make_env(config):
        return PettingZooEnv(
            RllibNativeNoLimitHoldemAECEnv(
                seed=int(config.get("seed", seed)),
                initial_chips=int(config.get("initial_chips", initial_chips)),
                max_steps_per_hand=int(config.get("max_steps_per_hand", max_steps_per_hand)),
            )
        )

    def _make_response_env(config):
        return RllibNativeResponseOracleEnv(config)

    overall_start = time.perf_counter()
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        num_cpus=int(num_cpus),
        num_gpus=float(num_gpus),
        log_to_driver=False,
    )
    algo = None
    try:
        env_config = {
            "seed": int(seed),
            "initial_chips": int(initial_chips),
            "max_steps_per_hand": int(max_steps_per_hand),
        }
        if response_oracle:
            env_config.update(
                {
                    "opponent_kind": normalized_opponent_kind,
                    "opponent_checkpoint": opponent_checkpoints,
                    "opponent_device": str(opponent_device),
                }
            )
        config = (
            PPOConfig()
            .environment(
                env=env_name,
                env_config=env_config,
                disable_env_checking=bool(disable_env_checking),
            )
            .framework("torch")
            .env_runners(
                num_env_runners=int(num_env_runners),
                rollout_fragment_length=int(rollout_fragment_length),
            )
            .training(
                train_batch_size=int(train_batch_size),
                minibatch_size=int(minibatch_size),
                num_epochs=int(num_epochs),
            )
            .rl_module(
                rl_module_spec=RLModuleSpec(
                    module_class=ActionMaskingTorchRLModule,
                    model_config={
                        "head_fcnet_hiddens": [int(hidden_dim), int(hidden_dim)],
                        "head_fcnet_activation": "relu",
                    },
                )
            )
        )
        if not response_oracle:
            config = config.multi_agent(
                policies={"shared"},
                policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared",
            )
        try:
            config = config.resources(num_gpus=float(num_gpus))
        except Exception:
            pass

        register_env(env_name, _make_response_env if response_oracle else _make_env)
        algo = config.build_algo()
        setup_seconds = time.perf_counter() - overall_start

        train_start = time.perf_counter()
        result: dict[str, Any] = {}
        for _ in range(int(train_iterations)):
            result = algo.train()
        train_seconds = time.perf_counter() - train_start

        env_steps = _result_value(result, "num_env_steps_sampled_lifetime")
        episodes = _result_value(result, "num_episodes_lifetime")
        metrics: dict[str, Any] = {
            "algorithm": (
                "rllib_ppo_action_masked_response_oracle"
                if response_oracle
                else "rllib_ppo_action_masked"
            ),
            "role": (
                "maintained_library_population_response_oracle"
                if response_oracle
                else "maintained_library_multi_agent_rl_control"
            ),
            "environment": "poker_ai:rllib_pettingzoo_full_deck_hu_nlhe",
            "warning": (
                "RLlib PPO is a maintained-library RL control. This is not "
                "Slumbot/SOTA evidence without native H2H gates."
            ),
            "num_actions": N_ACTIONS,
            "num_features": N_FEATURES,
            "train_iterations": int(train_iterations),
            "train_batch_size": int(train_batch_size),
            "minibatch_size": int(minibatch_size),
            "num_epochs": int(num_epochs),
            "rollout_fragment_length": int(rollout_fragment_length),
            "num_env_runners": int(num_env_runners),
            "num_cpus": int(num_cpus),
            "num_gpus": float(num_gpus),
            "hidden_dim": int(hidden_dim),
            "disable_env_checking": bool(disable_env_checking),
            "uses_slumbot_training_data": False,
            "opponent_kind": normalized_opponent_kind,
            "opponent_checkpoints": opponent_checkpoints,
            "opponent_device": str(opponent_device),
            "fixed_opponent_population_size": int(len(opponent_checkpoints)),
            "train_opponent_mode": (
                "random_policy"
                if response_oracle and normalized_opponent_kind == "random"
                else "fixed_policy_population"
                if response_oracle and len(opponent_checkpoints) > 1
                else "fixed_policy"
                if response_oracle
                else "shared_self_play"
            ),
            "setup_seconds": float(setup_seconds),
            "train_seconds": float(train_seconds),
            "total_seconds": float(time.perf_counter() - overall_start),
            "env_steps_sampled": env_steps,
            "episodes": episodes,
            "steps_per_second": (
                float(env_steps / max(train_seconds, 1e-9)) if env_steps is not None else None
            ),
            "promotion": False,
            "league_eligible": False,
        }
        if checkpoint_out is not None:
            out_path = Path(checkpoint_out).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            metrics["checkpoint_path"] = str(algo.save_to_path(str(out_path)))
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()

    text = json.dumps(metrics, indent=2, sort_keys=True)
    if output_json is not None:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-iterations", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--rollout-fragment-length", type=int, default=64)
    parser.add_argument("--num-env-runners", type=int, default=0)
    parser.add_argument("--num-cpus", type=int, default=4)
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--max-steps-per-hand", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--opponent-kind",
        default="self",
        help=(
            "self for shared-policy multi-agent self-play, or random/native-nfsp/rainbow "
            "for single-agent response-oracle training against a frozen local opponent."
        ),
    )
    parser.add_argument("--opponent-checkpoint", action="append", default=[])
    parser.add_argument("--opponent-device", default="cpu")
    parser.add_argument(
        "--enable-env-checking",
        action="store_true",
        help=(
            "Let RLlib run its generic env pre-check. Disabled by default because "
            "the checker samples random illegal actions before the action-mask "
            "module is applied."
        ),
    )
    parser.add_argument("--output-json")
    parser.add_argument("--checkpoint-out")
    args = parser.parse_args(argv)

    metrics = run_control(
        train_iterations=args.train_iterations,
        train_batch_size=args.train_batch_size,
        minibatch_size=args.minibatch_size,
        num_epochs=args.num_epochs,
        rollout_fragment_length=args.rollout_fragment_length,
        num_env_runners=args.num_env_runners,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        hidden_dim=args.hidden_dim,
        initial_chips=args.initial_chips,
        max_steps_per_hand=args.max_steps_per_hand,
        seed=args.seed,
        disable_env_checking=not args.enable_env_checking,
        opponent_kind=args.opponent_kind,
        opponent_checkpoint=args.opponent_checkpoint,
        opponent_device=args.opponent_device,
        output_json=args.output_json,
        checkpoint_out=args.checkpoint_out,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
