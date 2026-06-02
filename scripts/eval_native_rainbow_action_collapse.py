#!/usr/bin/env python3
"""Evaluate native Rainbow action collapse on local simulator states."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.deep_cfr.fast_state import FastPokerState, new_fast_game  # noqa: E402
from poker_ai.games.full_deck.state import INDEX_TO_ACTION, N_ACTIONS, N_FEATURES  # noqa: E402
from poker_ai.research.action_collapse_gate import evaluate_action_collapse  # noqa: E402
from poker_ai.research.native_nfsp import resolve_device  # noqa: E402


class _RainbowDistributionNet(nn.Module):
    def __init__(self, *, hidden_dim: int, num_atoms: int, device: torch.device):
        super().__init__()
        self.num_atoms = int(num_atoms)
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), N_ACTIONS * int(num_atoms)),
        )

    def forward(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        logits = self.net(x).view(-1, N_ACTIONS, self.num_atoms)
        return torch.softmax(logits, dim=-1)


class _RainbowQPolicy(nn.Module):
    kind = "tianshou-rainbow"

    def __init__(self, model: nn.Module, *, num_atoms: int, algorithm: str):
        super().__init__()
        self.model = model
        self.algorithm = str(algorithm)
        self.register_buffer(
            "support",
            torch.linspace(-1.0, 1.0, int(num_atoms), dtype=torch.float32),
        )

    def forward(self, features):
        distribution = self.model(features)
        return torch.sum(distribution * self.support.view(1, 1, -1), dim=-1)


def _rainbow_state_dicts_by_seat_from_payload(payload: dict) -> dict[int, dict]:
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


def _load_rainbow_q_policy(checkpoint: str | Path, *, kind: str, device: torch.device) -> _RainbowQPolicy:
    normalized = str(kind).strip().lower().replace("_", "-")
    if normalized not in {"tianshou-rainbow", "rainbow"}:
        raise ValueError("only native tianshou-rainbow checkpoints are supported")
    payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    if int(payload.get("num_actions", -1)) != N_ACTIONS:
        raise ValueError("Rainbow checkpoint action count does not match native contract")
    if int(payload.get("num_features", -1)) != N_FEATURES:
        raise ValueError("Rainbow checkpoint feature count does not match native contract")
    hidden_dim = int(payload.get("hidden_dim", 128))
    num_atoms = int(payload.get("num_atoms", 51))
    model = _RainbowDistributionNet(
        hidden_dim=hidden_dim,
        num_atoms=num_atoms,
        device=device,
    ).to(device)
    state_dicts = _rainbow_state_dicts_by_seat_from_payload(payload)
    model.load_state_dict(state_dicts.get(0, state_dicts[sorted(state_dicts)[0]]))
    policy = _RainbowQPolicy(
        model,
        num_atoms=num_atoms,
        algorithm=str(payload.get("algorithm", "tianshou_rainbow_dqn")),
    ).to(device)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def _softmax_policy(q_values: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    legal = np.asarray(legal_mask, dtype=np.float64)
    if float(legal.sum()) <= 0.0:
        return np.ones(N_ACTIONS, dtype=np.float64) / float(N_ACTIONS)
    masked = np.where(legal > 0, q_values.astype(np.float64), -1.0e9)
    shifted = masked - float(np.max(masked))
    probs = np.exp(shifted) * legal
    total = float(probs.sum())
    return probs / total if total > 0.0 else legal / float(legal.sum())


def _sample_local_states(
    *,
    n_hands: int,
    max_steps_per_hand: int,
    initial_chips: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    streets: list[int] = []
    for _hand_i in range(int(n_hands)):
        state = new_fast_game(n_players=2, initial_chips=int(initial_chips))
        for _step_i in range(int(max_steps_per_hand)):
            if int(state.stage) >= FastPokerState.SHOWDOWN:
                break
            mask = state.get_legal_mask().astype(np.float32, copy=False)
            legal = np.flatnonzero(mask > 0)
            if legal.size == 0:
                break
            features.append(state.to_feature_vector().astype(np.float32, copy=False))
            masks.append(mask.copy())
            streets.append(int(state.stage))
            state.apply_action(int(rng.choice(legal)))
    if features:
        return (
            np.stack(features).astype(np.float32, copy=False),
            np.stack(masks).astype(np.float32, copy=False),
            np.asarray(streets, dtype=np.int64),
        )
    return (
        np.zeros((0, N_FEATURES), dtype=np.float32),
        np.zeros((0, N_ACTIONS), dtype=np.float32),
        np.zeros(0, dtype=np.int64),
    )


def evaluate_native_rainbow_action_collapse(
    *,
    checkpoint: str | Path,
    kind: str = "tianshou-rainbow",
    n_hands: int = 512,
    max_steps_per_hand: int = 64,
    initial_chips: int = 10000,
    seed: int = 20260810,
    device: str = "auto",
    max_top_action_fraction: float = 0.75,
    min_distinct_actions: int = 2,
) -> dict:
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    policy = _load_rainbow_q_policy(checkpoint, kind=kind, device=resolved_device)
    features, masks, streets = _sample_local_states(
        n_hands=n_hands,
        max_steps_per_hand=max_steps_per_hand,
        initial_chips=initial_chips,
        seed=seed,
    )
    if features.shape[0] == 0:
        raise RuntimeError("local state sampler produced no decision states")

    with torch.no_grad():
        q_values = policy(
            torch.as_tensor(features, dtype=torch.float32, device=resolved_device)
        )
        q_np = q_values.detach().cpu().numpy().astype(np.float64, copy=False)

    rng = np.random.default_rng(int(seed) + 17)
    greedy_counts: Counter[str] = Counter()
    stochastic_counts: Counter[str] = Counter()
    street_counts: Counter[str] = Counter()
    greedy_by_street: dict[str, Counter[str]] = {}
    stochastic_by_street: dict[str, Counter[str]] = {}
    allin_probs: list[float] = []
    top_probs: list[float] = []
    top_margins: list[float] = []
    for row_i, (q_row, mask_row) in enumerate(zip(q_np, masks, strict=True)):
        legal = mask_row > 0
        masked_q = np.where(legal, q_row, -1.0e9)
        greedy_action = int(np.argmax(masked_q))
        probs = _softmax_policy(q_row, mask_row)
        stochastic_action = int(rng.choice(np.arange(N_ACTIONS), p=probs))
        greedy_counts[INDEX_TO_ACTION[greedy_action]] += 1
        stochastic_counts[INDEX_TO_ACTION[stochastic_action]] += 1
        street = str(int(streets[row_i]))
        street_counts[street] += 1
        greedy_by_street.setdefault(street, Counter())[INDEX_TO_ACTION[greedy_action]] += 1
        stochastic_by_street.setdefault(street, Counter())[INDEX_TO_ACTION[stochastic_action]] += 1
        allin_probs.append(float(probs[8]) if bool(legal[8]) else 0.0)
        sorted_q = np.sort(masked_q[legal])
        top_probs.append(float(probs[greedy_action]))
        top_margins.append(float(sorted_q[-1] - sorted_q[-2]) if sorted_q.size >= 2 else 0.0)

    greedy_gate = evaluate_action_collapse(
        {"selected_action_counts": dict(greedy_counts)},
        max_top_action_fraction=max_top_action_fraction,
        min_distinct_actions=min_distinct_actions,
    )
    stochastic_gate = evaluate_action_collapse(
        {"selected_action_counts": dict(stochastic_counts)},
        max_top_action_fraction=max_top_action_fraction,
        min_distinct_actions=min_distinct_actions,
    )
    return {
        "algorithm": "native_rainbow_action_collapse_diagnostic",
        "role": "local_slumbot_free_deployment_mode_gate",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "uses_slumbot_training_data": False,
        "uses_slumbot_trace_data": False,
        "promotion": False,
        "checkpoint": str(checkpoint),
        "kind": policy.kind,
        "policy_algorithm": policy.algorithm,
        "n_hands": int(n_hands),
        "n_states": int(features.shape[0]),
        "initial_chips": int(initial_chips),
        "max_steps_per_hand": int(max_steps_per_hand),
        "seed": int(seed),
        "street_counts": dict(street_counts),
        "greedy_selected_action_counts_by_street": {
            street: dict(counts) for street, counts in sorted(greedy_by_street.items())
        },
        "stochastic_selected_action_counts_by_street": {
            street: dict(counts) for street, counts in sorted(stochastic_by_street.items())
        },
        "greedy": {
            "selected_action_counts": dict(greedy_counts),
            "gate": greedy_gate,
            "all_in_fraction": float(greedy_counts.get(INDEX_TO_ACTION[8], 0) / features.shape[0]),
        },
        "stochastic": {
            "selected_action_counts": dict(stochastic_counts),
            "gate": stochastic_gate,
            "all_in_fraction": float(stochastic_counts.get(INDEX_TO_ACTION[8], 0) / features.shape[0]),
        },
        "mean_all_in_probability": float(np.mean(allin_probs)),
        "mean_top_action_probability": float(np.mean(top_probs)),
        "mean_top_q_margin": float(np.mean(top_margins)),
        "passed": bool(greedy_gate["passed"] and stochastic_gate["passed"]),
        **device_info,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kind", default="tianshou-rainbow")
    parser.add_argument("--n-hands", type=int, default=512)
    parser.add_argument("--max-steps-per-hand", type=int, default=64)
    parser.add_argument("--initial-chips", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-top-action-fraction", type=float, default=0.75)
    parser.add_argument("--min-distinct-actions", type=int, default=2)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    metrics = evaluate_native_rainbow_action_collapse(
        checkpoint=args.checkpoint,
        kind=args.kind,
        n_hands=args.n_hands,
        max_steps_per_hand=args.max_steps_per_hand,
        initial_chips=args.initial_chips,
        seed=args.seed,
        device=args.device,
        max_top_action_fraction=args.max_top_action_fraction,
        min_distinct_actions=args.min_distinct_actions,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
