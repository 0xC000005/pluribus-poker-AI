#!/usr/bin/env python3
"""Run a Ray RLlib IMPALA control on RLCard's no-limit Hold'em surface."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.alphanlholdem_benchmark import (  # noqa: E402
    AlphaNLHoldemTorchPolicy,
    PolicyDecision,
    evaluate_rlcard_reference_h2h,
)
from poker_ai.research.native_nfsp import resolve_device  # noqa: E402
from poker_ai.research.rlcard_tianshou_ppo import (  # noqa: E402
    N_RLCARD_ACTIONS,
    N_RLCARD_FEATURES,
    RLCardNoLimitHoldemSingleAgentEnv,
)


def _legal_actions_from_state(state: dict[str, Any]) -> list[int]:
    return sorted(int(getattr(action, "value", action)) for action in state["legal_actions"].keys())


def _rllib_observation_from_state(state: dict[str, Any]) -> dict[str, np.ndarray]:
    mask = np.zeros((N_RLCARD_ACTIONS,), dtype=np.float32)
    legal = _legal_actions_from_state(state)
    mask[legal] = 1.0
    return {
        "observations": np.asarray(state["obs"], dtype=np.float32),
        "action_mask": mask,
    }


class RLlibLivePolicy:
    policy_name = "rllib_impala_rlcard_live_policy"

    def __init__(self, algorithm) -> None:
        self.algorithm = algorithm

    def act_rlcard_state(self, state: dict[str, Any]) -> PolicyDecision:
        obs = _rllib_observation_from_state(state)
        raw_action = self.algorithm.compute_single_action(obs, explore=False)
        if isinstance(raw_action, tuple):
            raw_action = raw_action[0]
        action = int(raw_action)
        legal_actions = _legal_actions_from_state(state)
        if action not in legal_actions:
            action = int(legal_actions[0])
        logits = np.zeros((N_RLCARD_ACTIONS,), dtype=np.float32)
        masked = np.full((N_RLCARD_ACTIONS,), -np.inf, dtype=np.float32)
        masked[legal_actions] = 0.0
        return PolicyDecision(
            action=action,
            legal_actions=legal_actions,
            logits=logits,
            masked_logits=masked,
            policy_name=self.policy_name,
        )


def _write_metrics(metrics: dict[str, Any], output_json: str | Path | None) -> dict[str, Any]:
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics["output_json"] = str(path)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _checkpoint_result_path(checkpoint_result: Any) -> str:
    checkpoint = getattr(checkpoint_result, "checkpoint", checkpoint_result)
    path = getattr(checkpoint, "path", checkpoint)
    return str(path)


def _base_metrics(
    *,
    train_iterations: int,
    seed: int,
    hidden_dim: int,
    device: str,
) -> dict[str, Any]:
    device_info = resolve_device(device)
    return {
        "algorithm": "rllib_impala_rlcard",
        "role": "public_reference_impala_control",
        "environment": "rlcard:no-limit-holdem",
        "warning": (
            "RLlib IMPALA is a maintained-library public-reference control. "
            "It trains in RLCard's 5-action game and is not native Slumbot evidence."
        ),
        **device_info,
        "seed": int(seed),
        "train_iterations": int(train_iterations),
        "hidden_dim": int(hidden_dim),
        "num_actions": N_RLCARD_ACTIONS,
        "num_features": N_RLCARD_FEATURES,
        "trained_environment_native": True,
        "native_action_projection": False,
        "uses_slumbot_data": False,
        "uses_alphanlholdem_training_data": False,
        "promotion": False,
    }


def run_control(
    *,
    train_iterations: int = 1,
    rollout_fragment_length: int = 64,
    train_batch_size: int = 256,
    hidden_dim: int = 128,
    lr: float = 5e-4,
    entropy_coeff: float = 0.01,
    vf_loss_coeff: float = 0.5,
    eval_games_per_seat: int = 0,
    seed: int = 20260528,
    device: str = "auto",
    local_mode: bool = True,
    checkpoint_dir: str | Path | None = None,
    reference_weights: str | Path | None = None,
    output_json: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    metrics = _base_metrics(
        train_iterations=train_iterations,
        seed=seed,
        hidden_dim=hidden_dim,
        device=device,
    )
    metrics.update(
        {
            "rollout_fragment_length": int(rollout_fragment_length),
            "train_batch_size": int(train_batch_size),
            "lr": float(lr),
            "entropy_coeff": float(entropy_coeff),
            "vf_loss_coeff": float(vf_loss_coeff),
            "dry_run": bool(dry_run),
        }
    )
    if dry_run:
        return _write_metrics(metrics, output_json)
    if metrics["resolved_device"] == "cuda":
        raise RuntimeError(
            "RLlib IMPALA old-API action masking is currently CPU-only in this "
            "repo. The CUDA learner path failed V-trace shape checks; use "
            "--device cpu until a new-API masked RLModule or local V-trace "
            "learner is implemented."
        )

    import ray
    from ray.rllib.algorithms.impala import IMPALAConfig
    from ray.rllib.examples._old_api_stack.models.action_mask_model import TorchActionMaskModel
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env

    env_name = "rlcard_hunl_masked_public_reference"

    def _env_creator(env_config):
        return RLCardNoLimitHoldemSingleAgentEnv(
            seed=int(env_config.get("seed", seed)),
            max_steps_per_hand=int(env_config.get("max_steps_per_hand", 256)),
            observation_format="rllib",
        )

    register_env(env_name, _env_creator)
    ModelCatalog.register_custom_model("rlcard_torch_action_mask", TorchActionMaskModel)
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
    )
    metrics["local_mode_requested"] = bool(local_mode)
    metrics["local_mode_effective"] = False
    algorithm = None
    try:
        num_gpus = 1 if metrics["resolved_device"] == "cuda" else 0
        config = (
            IMPALAConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(env=env_name, env_config={"seed": int(seed)})
            .framework("torch")
            .resources(num_gpus=num_gpus)
            .env_runners(
                num_env_runners=0,
                rollout_fragment_length=int(rollout_fragment_length),
                batch_mode="truncate_episodes",
            )
            .training(
                vtrace=True,
                train_batch_size=int(train_batch_size),
                lr=float(lr),
                entropy_coeff=float(entropy_coeff),
                vf_loss_coeff=float(vf_loss_coeff),
            )
        )
        config.model = {
            "custom_model": "rlcard_torch_action_mask",
            "fcnet_hiddens": [int(hidden_dim), int(hidden_dim)],
            "fcnet_activation": "relu",
            "max_seq_len": 20,
            "custom_model_config": {},
        }
        train_start = time.perf_counter()
        algorithm = config.build_algo()
        train_results: list[dict[str, Any]] = []
        for _ in range(int(train_iterations)):
            result = algorithm.train()
            train_results.append(result)
        train_seconds = time.perf_counter() - train_start
        metrics.update(
            {
                "train_seconds": float(train_seconds),
                "iterations_per_second": float(train_iterations / max(train_seconds, 1e-9)),
            }
        )
        if train_results:
            last = train_results[-1]
            metrics["last_episode_reward_mean"] = last.get("episode_reward_mean")
            metrics["last_num_env_steps_sampled_lifetime"] = last.get(
                "num_env_steps_sampled_lifetime"
            )
            metrics["last_num_env_steps_trained_lifetime"] = last.get(
                "num_env_steps_trained_lifetime"
            )
        if checkpoint_dir is not None:
            checkpoint_path = algorithm.save(str(checkpoint_dir))
            metrics["checkpoint_path"] = _checkpoint_result_path(checkpoint_path)
        if eval_games_per_seat > 0:
            weights = (
                Path(reference_weights)
                if reference_weights is not None
                else REPO_ROOT / "reference_code" / "AlphaNLHoldem" / "weights" / "c_1048.pkl"
            )
            reference = AlphaNLHoldemTorchPolicy.from_pickle(weights, device=metrics["resolved_device"])
            h2h = evaluate_rlcard_reference_h2h(
                candidate=RLlibLivePolicy(algorithm),
                baseline=reference,
                games_per_seat=int(eval_games_per_seat),
                seed=int(seed) + 100_000,
            )
            metrics.update({f"h2h_{key}": value for key, value in h2h.items()})
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
    return _write_metrics(metrics, output_json)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-iterations", type=int, default=1)
    parser.add_argument("--rollout-fragment-length", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--vf-loss-coeff", type=float, default=0.5)
    parser.add_argument("--eval-games-per-seat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-mode", action="store_true", default=False)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--reference-weights")
    parser.add_argument("--output-json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    metrics = run_control(
        train_iterations=args.train_iterations,
        rollout_fragment_length=args.rollout_fragment_length,
        train_batch_size=args.train_batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        entropy_coeff=args.entropy_coeff,
        vf_loss_coeff=args.vf_loss_coeff,
        eval_games_per_seat=args.eval_games_per_seat,
        seed=args.seed,
        device=args.device,
        local_mode=args.local_mode,
        checkpoint_dir=args.checkpoint_dir,
        reference_weights=args.reference_weights,
        output_json=args.output_json,
        dry_run=args.dry_run,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
