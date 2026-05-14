"""Native full-deck regularized policy-dynamics pilot.

This is a bounded R-NaD-style control branch for autoresearch. It keeps the
repo's full-deck 9-action contract and tests whether a direct regularized
policy update can beat the native NFSP control before any larger integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poker_ai.games.full_deck.state import INDEX_TO_ACTION, N_ACTIONS, N_FEATURES, new_game
from poker_ai.research.game_theoretic_rl import regularized_policy_update
from poker_ai.research.native_nfsp import (
    _MLP,
    _network_probs,
    get_legal_mask,
    masked_uniform,
    resolve_device,
    select_action,
)


@dataclass(frozen=True)
class NativeRegularizedPolicyConfig:
    train_episodes: int = 100
    eval_games: int = 100
    hidden_dim: int = 64
    batch_size: int = 128
    min_buffer_size_to_learn: int = 32
    lr: float = 1e-3
    policy_step_size: float = 0.35
    regularization_strength: float = 0.1
    buffer_capacity: int = 20_000
    initial_chips: int = 1000
    max_steps_per_hand: int = 256
    seed: int = 20260514
    device: str = "auto"
    checkpoint_path: str | None = None


class _PolicyTargetBuffer:
    def __init__(self, capacity: int = 20_000):
        self.capacity = int(capacity)
        self.items: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.n_seen = 0

    def add(
        self,
        feature: np.ndarray,
        legal_mask: np.ndarray,
        target_policy: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> None:
        item = (
            np.asarray(feature, dtype=np.float32),
            np.asarray(legal_mask, dtype=np.float32),
            np.asarray(target_policy, dtype=np.float32),
        )
        self.n_seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        replacement_idx = int(rng.integers(self.n_seen))
        if replacement_idx < self.capacity:
            self.items[replacement_idx] = item

    def sample(self, batch_size: int, rng: np.random.Generator):
        n = min(int(batch_size), len(self.items))
        indices = rng.choice(len(self.items), size=n, replace=False)
        return [self.items[int(i)] for i in indices]

    def __len__(self) -> int:
        return len(self.items)


def sampled_regularized_policy_target(
    current_policy: np.ndarray,
    legal_mask: np.ndarray,
    *,
    action: int,
    payoff: float,
    step_size: float = 0.35,
    regularization_strength: float = 0.1,
) -> np.ndarray:
    """Build one legal policy target from a sampled action payoff."""
    mask = np.asarray(legal_mask, dtype=np.float32).reshape(-1)
    advantages = np.zeros_like(mask, dtype=np.float32)
    action_idx = int(action)
    if action_idx < 0 or action_idx >= mask.shape[0] or mask[action_idx] <= 0:
        raise ValueError("sampled action must be legal")
    advantages[action_idx] = float(payoff)
    reference = masked_uniform(mask)
    return regularized_policy_update(
        current_policy,
        advantages,
        mask,
        reference_policy=reference,
        step_size=step_size,
        regularization_strength=regularization_strength,
    )


def _train_policy(
    policy_net: nn.Module,
    optimizer: optim.Optimizer,
    buffer: _PolicyTargetBuffer,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> float | None:
    if len(buffer) <= 0:
        return None
    batch = buffer.sample(batch_size, rng)
    features = torch.tensor(np.stack([b[0] for b in batch]), device=device)
    legal_masks = torch.tensor(np.stack([b[1] for b in batch]), device=device)
    targets = torch.tensor(np.stack([b[2] for b in batch]), device=device)
    logits = policy_net(features).masked_fill(legal_masks <= 0, -1e4)
    log_probs = nn.functional.log_softmax(logits, dim=1)
    loss = -(targets * log_probs).sum(dim=1).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def _play_hand(
    policy_net: nn.Module,
    cfg: NativeRegularizedPolicyConfig,
    rng: np.random.Generator,
    device: torch.device,
    *,
    opponent_random: bool = False,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray, int, np.ndarray]], list[float], int]:
    state = new_game(2, initial_chips=cfg.initial_chips)
    records: list[tuple[int, np.ndarray, np.ndarray, int, np.ndarray]] = []
    n_steps = 0
    while not state.is_terminal and n_steps < cfg.max_steps_per_hand:
        player = state.player_i
        features = state.to_feature_vector()
        legal_mask = get_legal_mask(state)
        if opponent_random and player == 1:
            probs = masked_uniform(legal_mask)
        else:
            probs = _network_probs(policy_net, features, legal_mask, device)
        action_idx = select_action(probs, legal_mask, rng=rng)
        records.append((player, features, legal_mask, action_idx, probs))
        state = state.apply_action(INDEX_TO_ACTION[action_idx])
        n_steps += 1
    payouts = [float(state.payout.get(i, 0)) / float(cfg.initial_chips) for i in range(2)]
    return records, payouts, n_steps


def run_native_regularized_policy_pilot(cfg: NativeRegularizedPolicyConfig | None = None) -> dict:
    cfg = cfg or NativeRegularizedPolicyConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device_info = resolve_device(cfg.device)
    device = torch.device(device_info["resolved_device"])

    policy_net = _MLP(cfg.hidden_dim).to(device)
    policy_opt = optim.Adam(policy_net.parameters(), lr=cfg.lr)
    target_buffer = _PolicyTargetBuffer(cfg.buffer_capacity)

    train_start = time.perf_counter()
    total_steps = 0
    last_policy_loss = None
    policy_updates = 0
    payoffs: list[float] = []
    for _ in range(int(cfg.train_episodes)):
        records, payouts, steps = _play_hand(policy_net, cfg, rng, device)
        total_steps += steps
        payoffs.append(payouts[0])
        for player, features, legal_mask, action_idx, current_policy in records:
            target_policy = sampled_regularized_policy_target(
                current_policy,
                legal_mask,
                action=action_idx,
                payoff=payouts[player],
                step_size=cfg.policy_step_size,
                regularization_strength=cfg.regularization_strength,
            )
            target_buffer.add(features, legal_mask, target_policy, rng=rng)
        if len(target_buffer) >= cfg.min_buffer_size_to_learn:
            last_policy_loss = _train_policy(
                policy_net,
                policy_opt,
                target_buffer,
                cfg.batch_size,
                rng,
                device,
            )
            policy_updates += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    eval_payoffs = []
    eval_steps = 0
    for _ in range(int(cfg.eval_games)):
        _, payouts, steps = _play_hand(policy_net, cfg, rng, device, opponent_random=True)
        eval_payoffs.append(payouts[0])
        eval_steps += steps
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start

    metrics = {
        "algorithm": "native_regularized_policy",
        "role": "native_game_theoretic_rl_pilot",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "warning": "Bounded R-NaD-style regularized policy-dynamics pilot; not a promoted poker agent.",
        **device_info,
        "num_actions": N_ACTIONS,
        "train_episodes": int(cfg.train_episodes),
        "eval_games": int(cfg.eval_games),
        "train_steps": int(total_steps),
        "eval_steps": int(eval_steps),
        "train_seconds": float(train_seconds),
        "eval_seconds": float(eval_seconds),
        "episodes_per_second": float(cfg.train_episodes / max(train_seconds, 1e-9)),
        "train_steps_per_second": float(total_steps / max(train_seconds, 1e-9)),
        "mean_train_payoff_p0": float(np.mean(payoffs)) if payoffs else 0.0,
        "mean_eval_payoff_p0_vs_random": float(np.mean(eval_payoffs)) if eval_payoffs else 0.0,
        "target_buffer_size": len(target_buffer),
        "policy_updates": int(policy_updates),
        "policy_step_size": float(cfg.policy_step_size),
        "regularization_strength": float(cfg.regularization_strength),
        "last_policy_loss": last_policy_loss,
        "promotion": False,
    }
    if cfg.checkpoint_path:
        path = Path(cfg.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics["checkpoint_path"] = str(path)
        torch.save(
            {
                "algorithm": metrics["algorithm"],
                "role": metrics["role"],
                "environment": metrics["environment"],
                "num_actions": N_ACTIONS,
                "num_features": N_FEATURES,
                "hidden_dim": int(cfg.hidden_dim),
                "policy_net_state_dict": policy_net.state_dict(),
                "config": {
                    "train_episodes": int(cfg.train_episodes),
                    "eval_games": int(cfg.eval_games),
                    "hidden_dim": int(cfg.hidden_dim),
                    "batch_size": int(cfg.batch_size),
                    "min_buffer_size_to_learn": int(cfg.min_buffer_size_to_learn),
                    "lr": float(cfg.lr),
                    "policy_step_size": float(cfg.policy_step_size),
                    "regularization_strength": float(cfg.regularization_strength),
                    "buffer_capacity": int(cfg.buffer_capacity),
                    "initial_chips": int(cfg.initial_chips),
                    "max_steps_per_hand": int(cfg.max_steps_per_hand),
                    "seed": int(cfg.seed),
                    "device": str(cfg.device),
                },
                "metrics": metrics,
            },
            path,
        )
    return metrics


def _load_policy_network(
    checkpoint_path: str,
    resolved_device: torch.device,
) -> tuple[dict, nn.Module]:
    payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    if int(payload.get("num_actions", -1)) != N_ACTIONS:
        raise ValueError("checkpoint action count does not match native full-deck action contract")
    if int(payload.get("num_features", -1)) != N_FEATURES:
        raise ValueError("checkpoint feature count does not match native full-deck feature contract")
    config_payload = payload.get("config", {})
    hidden_dim = int(payload.get("hidden_dim", config_payload.get("hidden_dim", 64)))
    policy_net = _MLP(hidden_dim).to(resolved_device)
    if "policy_net_state_dict" in payload:
        policy_net.load_state_dict(payload["policy_net_state_dict"])
    elif "avg_net_state_dict" in payload:
        policy_net.load_state_dict(payload["avg_net_state_dict"])
    else:
        raise ValueError("checkpoint does not contain a loadable policy network")
    policy_net.eval()
    return payload, policy_net


def evaluate_native_regularized_policy_head_to_head(
    candidate_checkpoint: str,
    baseline_checkpoint: str,
    *,
    n_games: int = 100,
    device: str = "auto",
    seed: int = 20260514,
) -> dict:
    """Evaluate two native policy checkpoints in paired duplicate-swapped hands."""
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    candidate_payload, candidate_policy = _load_policy_network(candidate_checkpoint, resolved_device)
    baseline_payload, baseline_policy = _load_policy_network(baseline_checkpoint, resolved_device)
    config_payload = candidate_payload.get("config", {})
    initial_chips = int(config_payload.get("initial_chips", 1000))
    max_steps_per_hand = int(config_payload.get("max_steps_per_hand", 256))
    candidate_payoffs: list[float] = []
    candidate_pair_payoffs: list[float] = []
    total_steps = 0

    eval_start = time.perf_counter()
    pair_i = 0
    while len(candidate_payoffs) < int(n_games):
        game_seed = int(seed) + pair_i
        action_seed = int(seed) + 1_000_000 + pair_i
        pair_payoffs: list[float] = []
        for candidate_seat in (0, 1):
            if len(candidate_payoffs) >= int(n_games):
                break
            random.seed(game_seed)
            np.random.seed(game_seed)
            torch.manual_seed(game_seed)
            rng = np.random.default_rng(action_seed)
            baseline_seat = 1 - candidate_seat
            policies = {
                candidate_seat: candidate_policy,
                baseline_seat: baseline_policy,
            }
            state = new_game(2, initial_chips=initial_chips)
            n_steps = 0
            while not state.is_terminal and n_steps < max_steps_per_hand:
                features = state.to_feature_vector()
                legal_mask = get_legal_mask(state)
                probs = _network_probs(policies[state.player_i], features, legal_mask, resolved_device)
                action_idx = select_action(probs, legal_mask, rng=rng)
                state = state.apply_action(INDEX_TO_ACTION[action_idx])
                n_steps += 1
            total_steps += n_steps
            payoff = float(state.payout.get(candidate_seat, 0)) / float(initial_chips)
            candidate_payoffs.append(payoff)
            pair_payoffs.append(payoff)
        if pair_payoffs:
            candidate_pair_payoffs.append(float(np.mean(pair_payoffs)))
        pair_i += 1
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start
    payoff_mean = float(np.mean(candidate_pair_payoffs)) if candidate_pair_payoffs else 0.0
    payoff_std = float(np.std(candidate_pair_payoffs, ddof=1)) if len(candidate_pair_payoffs) > 1 else 0.0
    payoff_se = payoff_std / float(np.sqrt(len(candidate_pair_payoffs))) if candidate_pair_payoffs else 0.0

    return {
        "algorithm": "native_regularized_policy_h2h",
        "role": "native_game_theoretic_rl_checkpoint_head_to_head",
        "environment": str(candidate_payload.get("environment", "poker_ai:full_deck_hu_nlhe")),
        "candidate_checkpoint": str(candidate_checkpoint),
        "baseline_checkpoint": str(baseline_checkpoint),
        "candidate_algorithm": str(candidate_payload.get("algorithm", "")),
        "baseline_algorithm": str(baseline_payload.get("algorithm", "")),
        **device_info,
        "num_actions": N_ACTIONS,
        "n_games": int(n_games),
        "n_pairs": int(len(candidate_pair_payoffs)),
        "eval_seconds": float(eval_seconds),
        "eval_steps": int(total_steps),
        "eval_games_per_second": float(n_games / max(eval_seconds, 1e-9)),
        "eval_steps_per_second": float(total_steps / max(eval_seconds, 1e-9)),
        "mean_candidate_payoff": payoff_mean,
        "std_candidate_payoff": payoff_std,
        "lower95_candidate_payoff": payoff_mean - 1.96 * payoff_se,
        "upper95_candidate_payoff": payoff_mean + 1.96 * payoff_se,
        "promotion": False,
    }
