#!/usr/bin/env python3
"""Train a native 9-action checkpoint with compiled self-play V-trace updates."""

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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES  # noqa: E402
from poker_ai.research.local_vtrace import masked_log_probs, vtrace_policy_value_loss  # noqa: E402
from poker_ai.research.native_nfsp import resolve_device  # noqa: E402
from poker_ai.research.native_ppo_policy import _load_policy_network, _PolicyMLP, _ValueMLP  # noqa: E402
from poker_ai.research.native_rollout_substrate import (  # noqa: E402
    _collect_compiled_fast_policy_gradient_rollout,
)
from scripts.run_local_vtrace_compiled_native_smoke import (  # noqa: E402
    _pack_trajectories,
    _trajectory_indices,
)


def _write_metrics(metrics: dict[str, Any], output_json: str | Path | None) -> dict[str, Any]:
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics["output_json"] = str(path)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


class _RainbowQOpponent(torch.nn.Module):
    """Expose a Tianshou Rainbow distribution model as action-value logits."""

    def __init__(self, model: torch.nn.Module, *, num_atoms: int) -> None:
        super().__init__()
        self.model = model
        support = torch.linspace(-1.0, 1.0, int(num_atoms), dtype=torch.float32)
        self.register_buffer("support", support)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.model(features)
        distribution = output[0] if isinstance(output, tuple) else output
        return torch.sum(distribution * self.support.view(1, 1, -1), dim=-1)


class _RainbowDistributionNet(torch.nn.Module):
    def __init__(self, *, hidden_dim: int, num_atoms: int, device: torch.device) -> None:
        super().__init__()
        self.num_atoms = int(num_atoms)
        self.device = device
        self.net = torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES, int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim), N_ACTIONS * int(num_atoms)),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        logits = self.net(x).view(-1, N_ACTIONS, self.num_atoms)
        return torch.softmax(logits, dim=-1)


def _is_rainbow_payload(payload: dict[str, Any]) -> bool:
    algorithm = str(payload.get("algorithm", ""))
    return algorithm in {
        "tianshou_rainbow_dqn",
        "tianshou_marl_rainbow_dqn",
        "compiled_tianshou_rainbow_response_oracle",
    }


def _rainbow_state_dicts_by_seat_from_payload(payload: dict[str, Any]) -> dict[int, dict]:
    if payload.get("shared_model_state_dict") is not None:
        shared = payload["shared_model_state_dict"]
        return {0: shared, 1: shared}
    if "agent_model_state_dicts" in payload:
        agent_state_dicts = payload["agent_model_state_dicts"]
        fallback_key = "player_0" if "player_0" in agent_state_dicts else sorted(agent_state_dicts)[0]
        return {
            seat: agent_state_dicts.get(f"player_{seat}", agent_state_dicts[fallback_key])
            for seat in (0, 1)
        }
    if "model_state_dict" in payload:
        shared = payload["model_state_dict"]
        return {0: shared, 1: shared}
    raise ValueError("Rainbow checkpoint is missing a loadable model state dict")


def _load_rainbow_opponent(payload: dict[str, Any], device: torch.device) -> torch.nn.Module:
    if int(payload.get("num_actions", -1)) != N_ACTIONS:
        raise ValueError("Rainbow opponent action count does not match native full-deck contract")
    if int(payload.get("num_features", -1)) != N_FEATURES:
        raise ValueError("Rainbow opponent feature count does not match native full-deck contract")
    hidden_dim = int(payload.get("hidden_dim", 128))
    num_atoms = int(payload.get("num_atoms", 51))
    model = _RainbowDistributionNet(
        hidden_dim=hidden_dim,
        num_atoms=num_atoms,
        device=device,
    ).to(device)
    state_dicts = _rainbow_state_dicts_by_seat_from_payload(payload)
    model.load_state_dict(state_dicts.get(0, state_dicts[sorted(state_dicts)[0]]))
    model.eval()
    opponent = _RainbowQOpponent(model, num_atoms=num_atoms).to(device)
    opponent.eval()
    for parameter in opponent.parameters():
        parameter.requires_grad_(False)
    return opponent


def _summarize_opponent_kinds(kinds: list[str]) -> str | None:
    if not kinds:
        return None
    unique = sorted(set(kinds))
    return unique[0] if len(unique) == 1 else "mixed"


def _load_compiled_rollout_opponents(
    checkpoints: list[str | Path] | None,
    device: torch.device,
) -> tuple[list[torch.nn.Module], list[str]]:
    opponents: list[torch.nn.Module] = []
    kinds: list[str] = []
    for checkpoint in checkpoints or []:
        payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
        if _is_rainbow_payload(payload):
            opponents.append(_load_rainbow_opponent(payload, device))
            kinds.append("tianshou-rainbow")
            continue
        _payload, policy, feature_mode = _load_policy_network(
            str(checkpoint),
            device,
            strategy_source="auto",
        )
        if feature_mode != "flat":
            raise ValueError("compiled native V-trace opponents must use flat features")
        policy.eval()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        opponents.append(policy)
        kinds.append("native-ppo")
    return opponents, kinds


def _load_reference_policy(
    checkpoint: str | Path | None,
    device: torch.device,
) -> tuple[torch.nn.Module | None, str | None]:
    if checkpoint is None:
        return None, None
    policies, kinds = _load_compiled_rollout_opponents([checkpoint], device)
    if len(policies) != 1 or len(kinds) != 1:
        raise ValueError("reference_policy_checkpoint must load exactly one policy")
    reference = policies[0].to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return reference, kinds[0]


def _train_on_batch(
    *,
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    reference_policy: torch.nn.Module | None = None,
    reference_kl_weight: float = 0.0,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, np.ndarray],
    gamma: float,
    device: torch.device,
) -> tuple[float, dict[str, float], int, int]:
    trajectories = _trajectory_indices(batch)
    if not trajectories:
        return 0.0, {"illegal_action_probability": 0.0}, 0, 0
    packed = _pack_trajectories(batch, trajectories, gamma=float(gamma))
    features_t = torch.as_tensor(packed["features"], dtype=torch.float32, device=device)
    legal_masks_t = torch.as_tensor(packed["legal_masks"], dtype=torch.bool, device=device)
    actions_t = torch.as_tensor(packed["actions"], dtype=torch.long, device=device)
    behavior_log_probs_t = torch.as_tensor(
        packed["behavior_log_probs"],
        dtype=torch.float32,
        device=device,
    )
    rewards_t = torch.as_tensor(packed["rewards"], dtype=torch.float32, device=device)
    discounts_t = torch.as_tensor(packed["discounts"], dtype=torch.float32, device=device)
    valid_mask_t = torch.as_tensor(packed["valid_mask"], dtype=torch.bool, device=device)
    flat_features = features_t.reshape(-1, N_FEATURES)
    logits = policy_net(flat_features).reshape(features_t.shape[0], features_t.shape[1], N_ACTIONS)
    values = value_net(flat_features).reshape(features_t.shape[0], features_t.shape[1])
    bootstrap = torch.zeros((features_t.shape[1],), dtype=torch.float32, device=device)
    loss, stats = vtrace_policy_value_loss(
        logits=logits,
        values=values,
        actions=actions_t,
        legal_mask=legal_masks_t,
        behavior_action_log_probs=behavior_log_probs_t,
        rewards=rewards_t,
        discounts=discounts_t,
        bootstrap_value=bootstrap,
        valid_mask=valid_mask_t,
    )
    reference_kl = torch.zeros((), dtype=torch.float32, device=device)
    if reference_policy is not None and float(reference_kl_weight) > 0.0:
        with torch.no_grad():
            reference_logits = reference_policy(flat_features).reshape(
                features_t.shape[0],
                features_t.shape[1],
                N_ACTIONS,
            )
            reference_log_probs = masked_log_probs(reference_logits, legal_masks_t)
        log_probs = masked_log_probs(logits, legal_masks_t)
        probs = torch.exp(log_probs)
        valid = valid_mask_t.to(dtype=torch.float32, device=device)
        valid_count = valid.sum().clamp_min(1.0)
        safe_log_probs = torch.nan_to_num(log_probs, nan=0.0, posinf=0.0, neginf=0.0)
        safe_reference_log_probs = torch.nan_to_num(
            reference_log_probs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        kl_terms = probs * (safe_log_probs - safe_reference_log_probs)
        kl_terms = torch.where(legal_masks_t, kl_terms, torch.zeros_like(kl_terms))
        kl_terms = torch.nan_to_num(kl_terms, nan=0.0, posinf=0.0, neginf=0.0)
        per_row_kl = torch.sum(kl_terms, dim=-1)
        reference_kl = (per_row_kl * valid).sum() / valid_count
        loss = loss + float(reference_kl_weight) * reference_kl
        stats["reference_kl"] = float(reference_kl.detach().cpu())
        stats["reference_kl_weight"] = float(reference_kl_weight)
    else:
        stats["reference_kl"] = 0.0
        stats["reference_kl_weight"] = 0.0
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return (
        float(loss.detach().cpu()),
        {name: float(value) for name, value in stats.items()},
        int(valid_mask_t.sum().detach().cpu()),
        int(len(trajectories)),
    )


def run_learner(
    *,
    train_iterations: int = 4,
    games_per_iteration: int = 512,
    collector_batch_size: int = 128,
    max_steps_per_game: int = 64,
    initial_chips: int = 1000,
    hidden_dim: int = 128,
    lr: float = 3e-4,
    gamma: float = 1.0,
    seed: int = 20260559,
    device: str = "auto",
    opponent_checkpoints: list[str | Path] | None = None,
    reference_policy_checkpoint: str | Path | None = None,
    reference_kl_weight: float = 0.0,
    checkpoint_out: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Train a small native V-trace actor/value checkpoint.

    This is a native full-deck HUNL learner bridge. It deliberately writes a
    9-action checkpoint that is compatible with native H2H evaluators and is
    not a candidate for the RLCard/AlphaNLHoldem 5-action public-reference gate.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    policy_net = _PolicyMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
    value_net = _ValueMLP(int(hidden_dim), input_dim=N_FEATURES).to(resolved_device)
    opponent_policies, opponent_kinds = _load_compiled_rollout_opponents(
        opponent_checkpoints,
        resolved_device,
    )
    reference_policy, reference_kind = _load_reference_policy(
        reference_policy_checkpoint,
        resolved_device,
    )
    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(value_net.parameters()),
        lr=float(lr),
    )

    losses: list[float] = []
    illegal_probabilities: list[float] = []
    reference_kls: list[float] = []
    n_samples = 0
    n_trajectories = 0
    collector_steps = 0
    collector_seconds = 0.0
    compiled_needs_python_showdown = 0
    train_start = time.perf_counter()
    for iteration_i in range(int(train_iterations)):
        batch, collector_metrics = _collect_compiled_fast_policy_gradient_rollout(
            policy_net,
            n_games=int(games_per_iteration),
            batch_size=int(collector_batch_size),
            max_steps_per_game=int(max_steps_per_game),
            initial_chips=int(initial_chips),
            device=resolved_device,
            seed=int(seed) + iteration_i * 10_000,
            opponent_policies=opponent_policies if opponent_policies else None,
        )
        collector_steps += int(collector_metrics.get("steps", 0))
        collector_seconds += float(collector_metrics.get("seconds", 0.0))
        compiled_needs_python_showdown += int(collector_metrics.get("needs_python_showdown", 0))
        loss_value, stats, batch_samples, batch_trajectories = _train_on_batch(
            policy_net=policy_net,
            value_net=value_net,
            reference_policy=reference_policy,
            reference_kl_weight=float(reference_kl_weight),
            optimizer=optimizer,
            batch=batch,
            gamma=float(gamma),
            device=resolved_device,
        )
        if batch_samples > 0:
            losses.append(loss_value)
            illegal_probabilities.append(float(stats.get("illegal_action_probability", 0.0)))
            reference_kls.append(float(stats.get("reference_kl", 0.0)))
            n_samples += int(batch_samples)
            n_trajectories += int(batch_trajectories)
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start
    max_illegal_probability = float(max(illegal_probabilities) if illegal_probabilities else 0.0)
    checkpoint_path = None if checkpoint_out is None else str(Path(checkpoint_out))
    metrics: dict[str, Any] = {
        "algorithm": "local_vtrace_compiled_native",
        "role": "native_full_deck_vtrace_actor_learner",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": "Native 9-action learner checkpoint; not RLCard/AlphaNLHoldem strength evidence.",
        **device_info,
        "seed": int(seed),
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "train_iterations": int(train_iterations),
        "games_per_iteration": int(games_per_iteration),
        "collector_batch_size": int(collector_batch_size),
        "max_steps_per_game": int(max_steps_per_game),
        "initial_chips": int(initial_chips),
        "hidden_dim": int(hidden_dim),
        "lr": float(lr),
        "gamma": float(gamma),
        "population_training": bool(opponent_policies),
        "opponent_kind": _summarize_opponent_kinds(opponent_kinds),
        "opponent_kinds": list(opponent_kinds),
        "opponent_population_size": int(len(opponent_policies)),
        "opponent_checkpoints": [str(path) for path in (opponent_checkpoints or [])],
        "reference_regularized": bool(reference_policy is not None and float(reference_kl_weight) > 0.0),
        "reference_policy_checkpoint": (
            str(reference_policy_checkpoint) if reference_policy_checkpoint is not None else None
        ),
        "reference_policy_kind": reference_kind,
        "reference_kl_weight": float(reference_kl_weight),
        "mean_reference_kl": float(np.mean(reference_kls)) if reference_kls else None,
        "last_reference_kl": float(reference_kls[-1]) if reference_kls else None,
        "collector_backend": "compiled-fast-state",
        "trajectory_packing": "padded_vectorized",
        "collector_steps": int(collector_steps),
        "collector_seconds": float(collector_seconds),
        "compiled_needs_python_showdown": int(compiled_needs_python_showdown),
        "n_trajectories": int(n_trajectories),
        "n_samples": int(n_samples),
        "train_seconds": float(train_seconds),
        "samples_per_second": float(n_samples / max(train_seconds, 1e-9)),
        "last_loss": float(losses[-1]) if losses else None,
        "loss_is_finite": bool(losses and np.isfinite(losses[-1])),
        "illegal_action_probability": max_illegal_probability,
        "trained_environment_native": True,
        "native_action_projection": False,
        "rlcard_candidate": False,
        "uses_slumbot_data": False,
        "uses_alphanlholdem_training_data": False,
        "promotion": False,
        "checkpoint_path": checkpoint_path,
    }
    metrics["passed"] = bool(
        losses
        and np.isfinite(losses[-1])
        and max_illegal_probability == 0.0
        and compiled_needs_python_showdown == 0
        and n_samples > 0
    )

    if checkpoint_out is not None:
        path = Path(checkpoint_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "local_vtrace_compiled_native",
                "environment": "poker_ai:full_deck_hu_nlhe",
                "num_actions": N_ACTIONS,
                "num_features": N_FEATURES,
                "hidden_dim": int(hidden_dim),
                "policy_net_state_dict": policy_net.state_dict(),
                "value_net_state_dict": value_net.state_dict(),
                "config": {
                    "feature_mode": "flat",
                    "hidden_dim": int(hidden_dim),
                    "initial_chips": int(initial_chips),
                    "max_steps_per_hand": int(max_steps_per_game),
                    "rollout_backend": "compiled-fast-state",
                    "fsp_average_policy": False,
                    "train_environment": "poker_ai:full_deck_hu_nlhe",
                    "opponent_population_size": int(len(opponent_policies)),
                    "opponent_kinds": list(opponent_kinds),
                    "reference_policy_checkpoint": (
                        str(reference_policy_checkpoint)
                        if reference_policy_checkpoint is not None
                        else None
                    ),
                    "reference_policy_kind": reference_kind,
                    "reference_kl_weight": float(reference_kl_weight),
                },
                "metrics": metrics,
                "trained_environment_native": True,
                "native_action_projection": False,
                "rlcard_candidate": False,
                "uses_slumbot_data": False,
                "uses_alphanlholdem_training_data": False,
            },
            path,
        )
    return _write_metrics(metrics, output_json)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-iterations", type=int, default=4)
    parser.add_argument("--games-per-iteration", type=int, default=512)
    parser.add_argument("--collector-batch-size", type=int, default=128)
    parser.add_argument("--max-steps-per-game", type=int, default=64)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260559)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--opponent-checkpoint",
        action="append",
        dest="opponent_checkpoints",
        help="Frozen native-ppo-compatible opponent checkpoint. May be supplied multiple times.",
    )
    parser.add_argument(
        "--reference-policy-checkpoint",
        type=Path,
        help="Frozen local policy checkpoint used as a NashPG/MMD-style reference regularizer.",
    )
    parser.add_argument("--reference-kl-weight", type=float, default=0.0)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    metrics = run_learner(
        train_iterations=args.train_iterations,
        games_per_iteration=args.games_per_iteration,
        collector_batch_size=args.collector_batch_size,
        max_steps_per_game=args.max_steps_per_game,
        initial_chips=args.initial_chips,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
        device=args.device,
        opponent_checkpoints=args.opponent_checkpoints,
        reference_policy_checkpoint=args.reference_policy_checkpoint,
        reference_kl_weight=args.reference_kl_weight,
        checkpoint_out=args.checkpoint_out,
        output_json=args.output_json,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
