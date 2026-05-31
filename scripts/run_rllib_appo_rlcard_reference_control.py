#!/usr/bin/env python3
"""Run a controlled Ray RLlib APPO bridge on RLCard no-limit Hold'em.

This is a maintained-library public-reference learner. It trains directly in
RLCard's 50bb/5-action game and must not be interpreted as native Slumbot
evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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


class APPOTimeoutError(TimeoutError):
    """Raised when a controlled APPO run exceeds its configured wall-clock cap."""


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


class RLlibAPPOLivePolicy:
    policy_name = "rllib_appo_rlcard_live_policy"

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


@contextmanager
def _time_limit(seconds: float | None) -> Iterator[None]:
    if seconds is None or float(seconds) <= 0:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def _raise_timeout(_signum, _frame):
        raise APPOTimeoutError(f"APPO run exceeded {float(seconds):.1f}s timeout")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


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


def _result_value(result: dict[str, Any], key: str) -> float | None:
    value = result.get(key)
    if value is None:
        value = result.get("env_runners", {}).get(key)
    if value is None:
        value = result.get("learners", {}).get(key)
    return float(value) if value is not None else None


def _base_metrics(
    *,
    train_iterations: int,
    seed: int,
    hidden_dim: int,
    device: str,
) -> dict[str, Any]:
    device_info = resolve_device(device)
    return {
        "algorithm": "rllib_appo_rlcard",
        "learner_family": "appo_vtrace",
        "role": "public_reference_appo_control",
        "environment": "rlcard:no-limit-holdem",
        "warning": (
            "RLlib APPO is a maintained-library public-reference control. "
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
        "passed": False,
    }


def _train_appo_in_process(
    metrics: dict[str, Any],
    *,
    rollout_fragment_length: int,
    train_batch_size: int,
    hidden_dim: int,
    lr: float,
    entropy_coeff: float,
    vf_loss_coeff: float,
    num_env_runners: int,
    num_cpus: int,
    eval_games_per_seat: int,
    seed: int,
    checkpoint_dir: str | Path | None,
    reference_weights: str | Path | None,
) -> dict[str, Any]:
    import ray
    from ray.rllib.algorithms.appo import APPOConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
        ActionMaskingTorchRLModule,
    )
    from ray.tune.registry import register_env

    env_name = f"rlcard_hunl_appo_masked_public_reference_{int(seed)}"

    def _env_creator(env_config):
        return RLCardNoLimitHoldemSingleAgentEnv(
            seed=int(env_config.get("seed", seed)),
            max_steps_per_hand=int(env_config.get("max_steps_per_hand", 256)),
            observation_format="rllib",
        )

    register_env(env_name, _env_creator)
    num_gpus = 1.0 if metrics["resolved_device"] == "cuda" else 0.0
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        num_cpus=int(num_cpus),
        num_gpus=float(num_gpus),
        log_to_driver=False,
    )
    algorithm = None
    try:
        config = (
            APPOConfig()
            .environment(
                env=env_name,
                env_config={"seed": int(seed)},
                disable_env_checking=True,
            )
            .framework("torch")
            .env_runners(
                num_env_runners=int(num_env_runners),
                rollout_fragment_length=int(rollout_fragment_length),
                batch_mode="truncate_episodes",
            )
            .training(
                vtrace=True,
                train_batch_size=int(train_batch_size),
                train_batch_size_per_learner=int(train_batch_size),
                circular_buffer_num_batches=1,
                circular_buffer_iterations_per_batch=1,
                lr=float(lr),
                entropy_coeff=float(entropy_coeff),
                vf_loss_coeff=float(vf_loss_coeff),
            )
            .reporting(
                min_time_s_per_iteration=0,
                min_sample_timesteps_per_iteration=int(train_batch_size),
                min_train_timesteps_per_iteration=0,
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
            .learners(num_learners=1, num_gpus_per_learner=float(num_gpus))
        )
        train_start = time.perf_counter()
        algorithm = config.build_algo()
        train_results: list[dict[str, Any]] = []
        for _ in range(int(metrics["train_iterations"])):
            train_results.append(algorithm.train())
        train_seconds = time.perf_counter() - train_start
        out: dict[str, Any] = {
            "status": "completed",
            "train_seconds": float(train_seconds),
            "iterations_per_second": float(metrics["train_iterations"] / max(train_seconds, 1e-9)),
            "passed": True,
        }
        if train_results:
            last = train_results[-1]
            env_steps = _result_value(last, "num_env_steps_sampled_lifetime")
            episodes = _result_value(last, "num_episodes_lifetime")
            out.update(
                {
                    "last_episode_reward_mean": last.get("episode_reward_mean"),
                    "last_num_env_steps_sampled_lifetime": env_steps,
                    "last_num_episodes_lifetime": episodes,
                    "train_env_steps_per_second": (
                        float(env_steps / max(train_seconds, 1e-9)) if env_steps is not None else None
                    ),
                }
            )
        if checkpoint_dir is not None:
            checkpoint_path = algorithm.save(str(checkpoint_dir))
            out["checkpoint_path"] = _checkpoint_result_path(checkpoint_path)
        if eval_games_per_seat > 0:
            weights = (
                Path(reference_weights)
                if reference_weights is not None
                else REPO_ROOT / "reference_code" / "AlphaNLHoldem" / "weights" / "c_1048.pkl"
            )
            reference = AlphaNLHoldemTorchPolicy.from_pickle(
                weights,
                device=metrics["resolved_device"],
            )
            h2h = evaluate_rlcard_reference_h2h(
                candidate=RLlibAPPOLivePolicy(algorithm),
                baseline=reference,
                games_per_seat=int(eval_games_per_seat),
                seed=int(seed) + 100_000,
            )
            out.update({f"h2h_{key}": value for key, value in h2h.items()})
            out["passed"] = bool(h2h.get("passed", True))
        return out
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()


def run_control(
    *,
    train_iterations: int = 1,
    rollout_fragment_length: int = 64,
    train_batch_size: int = 512,
    hidden_dim: int = 128,
    lr: float = 5e-4,
    entropy_coeff: float = 0.01,
    vf_loss_coeff: float = 0.5,
    num_env_runners: int = 0,
    num_cpus: int = 4,
    eval_games_per_seat: int = 0,
    seed: int = 20260528,
    device: str = "auto",
    checkpoint_dir: str | Path | None = None,
    reference_weights: str | Path | None = None,
    output_json: str | Path | None = None,
    timeout_seconds: float | None = 300.0,
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
            "num_env_runners": int(num_env_runners),
            "num_cpus": int(num_cpus),
            "eval_games_per_seat": int(eval_games_per_seat),
            "timeout_seconds": None if timeout_seconds is None else float(timeout_seconds),
            "dry_run": bool(dry_run),
        }
    )
    if dry_run:
        metrics["status"] = "dry_run"
        return _write_metrics(metrics, output_json)

    run_start = time.perf_counter()
    try:
        with _time_limit(timeout_seconds):
            metrics.update(
                _train_appo_in_process(
                    metrics,
                    rollout_fragment_length=rollout_fragment_length,
                    train_batch_size=train_batch_size,
                    hidden_dim=hidden_dim,
                    lr=lr,
                    entropy_coeff=entropy_coeff,
                    vf_loss_coeff=vf_loss_coeff,
                    num_env_runners=num_env_runners,
                    num_cpus=num_cpus,
                    eval_games_per_seat=eval_games_per_seat,
                    seed=seed,
                    checkpoint_dir=checkpoint_dir,
                    reference_weights=reference_weights,
                )
            )
    except APPOTimeoutError as exc:
        metrics.update(
            {
                "status": "timeout",
                "failure_reason": str(exc),
                "passed": False,
                "promotion": False,
            }
        )
    except Exception as exc:
        elapsed = time.perf_counter() - run_start
        if timeout_seconds is not None and elapsed >= float(timeout_seconds) * 0.9:
            metrics.update(
                {
                    "status": "timeout",
                    "failure_type": type(exc).__name__,
                    "failure_reason": str(exc) or f"APPO run exceeded {float(timeout_seconds):.1f}s timeout",
                    "passed": False,
                    "promotion": False,
                }
            )
            return _write_metrics(metrics, output_json)
        metrics.update(
            {
                "status": "failed",
                "failure_type": type(exc).__name__,
                "failure_reason": str(exc),
                "passed": False,
                "promotion": False,
            }
        )
    return _write_metrics(metrics, output_json)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-iterations", type=int, default=1)
    parser.add_argument("--rollout-fragment-length", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--vf-loss-coeff", type=float, default=0.5)
    parser.add_argument("--num-env-runners", type=int, default=0)
    parser.add_argument("--num-cpus", type=int, default=4)
    parser.add_argument("--eval-games-per-seat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--reference-weights",
        type=Path,
        default=REPO_ROOT / "reference_code" / "AlphaNLHoldem" / "weights" / "c_1048.pkl",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--worker-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _worker_argv(args: argparse.Namespace) -> list[str]:
    worker = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-run",
        "--train-iterations",
        str(args.train_iterations),
        "--rollout-fragment-length",
        str(args.rollout_fragment_length),
        "--train-batch-size",
        str(args.train_batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--lr",
        str(args.lr),
        "--entropy-coeff",
        str(args.entropy_coeff),
        "--vf-loss-coeff",
        str(args.vf_loss_coeff),
        "--num-env-runners",
        str(args.num_env_runners),
        "--num-cpus",
        str(args.num_cpus),
        "--eval-games-per-seat",
        str(args.eval_games_per_seat),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--reference-weights",
        str(args.reference_weights),
        "--timeout-seconds",
        "0",
    ]
    if args.checkpoint_dir is not None:
        worker.extend(["--checkpoint-dir", str(args.checkpoint_dir)])
    if args.output_json is not None:
        worker.extend(["--output-json", str(args.output_json)])
    return worker


def _run_worker_subprocess(cmd: list[str], *, timeout: float) -> int:
    process = subprocess.Popen(cmd, start_new_session=True)
    try:
        return int(process.wait(timeout=float(timeout)))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def run_supervised_worker(args: argparse.Namespace) -> int:
    timeout = float(args.timeout_seconds)
    try:
        return _run_worker_subprocess(_worker_argv(args), timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        metrics = _base_metrics(
            train_iterations=args.train_iterations,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            device=args.device,
        )
        metrics.update(
            {
                "rollout_fragment_length": int(args.rollout_fragment_length),
                "train_batch_size": int(args.train_batch_size),
                "lr": float(args.lr),
                "entropy_coeff": float(args.entropy_coeff),
                "vf_loss_coeff": float(args.vf_loss_coeff),
                "num_env_runners": int(args.num_env_runners),
                "num_cpus": int(args.num_cpus),
                "eval_games_per_seat": int(args.eval_games_per_seat),
                "timeout_seconds": timeout,
                "dry_run": False,
                "status": "timeout",
                "failure_type": type(exc).__name__,
                "failure_reason": f"APPO worker exceeded {timeout:.1f}s timeout",
                "passed": False,
                "promotion": False,
            }
        )
        _write_metrics(metrics, args.output_json)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.worker_run and not args.dry_run and args.timeout_seconds > 0:
        return run_supervised_worker(args)

    metrics = run_control(
        train_iterations=args.train_iterations,
        rollout_fragment_length=args.rollout_fragment_length,
        train_batch_size=args.train_batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        entropy_coeff=args.entropy_coeff,
        vf_loss_coeff=args.vf_loss_coeff,
        num_env_runners=args.num_env_runners,
        num_cpus=args.num_cpus,
        eval_games_per_seat=args.eval_games_per_seat,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        reference_weights=args.reference_weights,
        output_json=args.output_json,
        timeout_seconds=args.timeout_seconds,
        dry_run=args.dry_run,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics.get("passed", False) or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
