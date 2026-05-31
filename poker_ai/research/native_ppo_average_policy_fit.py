"""Diagnostics for native PPO average-policy checkpoints."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn

from poker_ai.games.full_deck.state import INDEX_TO_ACTION, N_ACTIONS, N_FEATURES, new_game
from poker_ai.research.game_theoretic_rl import legal_softmax
from poker_ai.research.native_nfsp import _MLP, get_legal_mask, resolve_device, select_action


def _load_actor_and_average(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[dict, nn.Module, nn.Module]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(payload.get("num_actions", -1)) != N_ACTIONS:
        raise ValueError("checkpoint action count does not match native full-deck action contract")
    if int(payload.get("num_features", -1)) != N_FEATURES:
        raise ValueError("checkpoint feature count does not match native full-deck feature contract")
    if "policy_net_state_dict" not in payload:
        raise ValueError("checkpoint does not contain policy_net_state_dict")
    if "avg_net_state_dict" not in payload:
        raise ValueError("checkpoint does not contain avg_net_state_dict")
    config_payload = payload.get("config", {})
    hidden_dim = int(payload.get("hidden_dim", config_payload.get("hidden_dim", 64)))
    actor = _MLP(hidden_dim).to(device)
    average = _MLP(hidden_dim).to(device)
    actor.load_state_dict(payload["policy_net_state_dict"])
    average.load_state_dict(payload["avg_net_state_dict"])
    actor.eval()
    average.eval()
    return payload, actor, average


def _network_probs(net: nn.Module, features: np.ndarray, legal_mask: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32)).to(device)
        logits = net(x).detach().cpu().numpy().reshape(-1)
    return legal_softmax(logits, legal_mask)


def diagnose_average_policy_fit(
    checkpoint_path: str | Path,
    *,
    n_games: int = 100,
    device: str = "auto",
    seed: int = 20260521,
    rollout_source: str = "actor",
    initial_chips: int | None = None,
    max_steps_per_hand: int | None = None,
) -> dict:
    """Compare an FSP checkpoint's exported average policy to its actor."""
    if rollout_source not in {"actor", "average"}:
        raise ValueError("rollout_source must be one of: actor, average")
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    payload, actor, average = _load_actor_and_average(checkpoint_path, resolved_device)
    config_payload = payload.get("config", {})
    chips = int(initial_chips if initial_chips is not None else config_payload.get("initial_chips", 1000))
    max_steps = int(
        max_steps_per_hand
        if max_steps_per_hand is not None
        else config_payload.get("max_steps_per_hand", 256)
    )

    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    l1s: list[float] = []
    kls: list[float] = []
    top_matches = 0
    n_states = 0
    total_steps = 0
    actor_top_probs: list[float] = []
    avg_top_probs: list[float] = []
    rollout_net = actor if rollout_source == "actor" else average

    for _ in range(int(n_games)):
        state = new_game(2, initial_chips=chips)
        n_steps = 0
        while not state.is_terminal and n_steps < max_steps:
            features = state.to_feature_vector()
            legal_mask = get_legal_mask(state)
            actor_probs = _network_probs(actor, features, legal_mask, resolved_device)
            avg_probs = _network_probs(average, features, legal_mask, resolved_device)
            l1s.append(float(np.abs(actor_probs - avg_probs).sum()))
            kls.append(
                float(
                    np.sum(
                        actor_probs
                        * (
                            np.log(np.clip(actor_probs, 1e-9, 1.0))
                            - np.log(np.clip(avg_probs, 1e-9, 1.0))
                        )
                    )
                )
            )
            actor_top = int(np.argmax(actor_probs))
            avg_top = int(np.argmax(avg_probs))
            top_matches += int(actor_top == avg_top)
            actor_top_probs.append(float(actor_probs[actor_top]))
            avg_top_probs.append(float(avg_probs[actor_top]))
            n_states += 1

            rollout_probs = _network_probs(rollout_net, features, legal_mask, resolved_device)
            action_idx = select_action(rollout_probs, legal_mask, rng=rng)
            state = state.apply_action(INDEX_TO_ACTION[action_idx])
            n_steps += 1
        total_steps += n_steps

    if n_states <= 0:
        raise ValueError("no decision states sampled")
    return {
        "algorithm": "native_ppo_average_policy_fit",
        "checkpoint": str(checkpoint_path),
        **device_info,
        "n_games": int(n_games),
        "n_states": int(n_states),
        "rollout_source": str(rollout_source),
        "total_steps": int(total_steps),
        "mean_l1_actor_to_average": float(np.mean(l1s)),
        "mean_kl_actor_to_average": float(np.mean(kls)),
        "top_action_agreement": float(top_matches / max(n_states, 1)),
        "mean_actor_top_prob": float(np.mean(actor_top_probs)),
        "mean_average_prob_on_actor_top": float(np.mean(avg_top_probs)),
        "promotion": False,
        "passed": True,
    }
