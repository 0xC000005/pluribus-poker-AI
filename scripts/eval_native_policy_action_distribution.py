#!/usr/bin/env python3
"""Diagnose local action concentration for any mixed native policy adapter."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.games.full_deck.state import INDEX_TO_ACTION, N_ACTIONS, new_game  # noqa: E402
from poker_ai.research.action_collapse_gate import evaluate_action_collapse  # noqa: E402
from poker_ai.research.mixed_policy_h2h import (  # noqa: E402
    SUPPORTED_POLICY_KINDS,
    load_policy_adapter,
)
from poker_ai.research.native_nfsp import get_legal_mask, resolve_device  # noqa: E402


def _safe_probs(adapter, features, legal_mask, device: torch.device) -> np.ndarray | None:
    try:
        probs = np.asarray(adapter.probs(features, legal_mask, device), dtype=np.float64)
    except Exception:
        return None
    if probs.shape != (N_ACTIONS,):
        return None
    probs = np.where(np.asarray(legal_mask, dtype=np.float64) > 0, probs, 0.0)
    total = float(probs.sum())
    if total <= 0.0:
        return None
    return probs / total


def evaluate_native_policy_action_distribution(
    *,
    checkpoint: str | Path,
    kind: str,
    n_hands: int = 512,
    max_steps_per_hand: int = 64,
    initial_chips: int = 20000,
    seed: int = 20260835,
    device: str = "auto",
    max_top_action_fraction: float = 0.75,
    min_distinct_actions: int = 2,
) -> dict:
    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    adapter = load_policy_adapter(checkpoint, kind=kind, device=resolved_device)
    rng = np.random.default_rng(int(seed))
    walk_rng = np.random.default_rng(int(seed) + 1009)

    selected_counts: Counter[str] = Counter()
    random_walk_counts: Counter[str] = Counter()
    selected_by_street: dict[str, Counter[str]] = {}
    entropies: list[float] = []
    top_probs: list[float] = []
    allin_probs: list[float] = []
    n_prob_rows = 0
    n_states = 0

    for _hand_i in range(int(n_hands)):
        state = new_game(2, initial_chips=int(initial_chips))
        for _step_i in range(int(max_steps_per_hand)):
            if state.is_terminal:
                break
            legal_mask = get_legal_mask(state).astype(np.float32, copy=False)
            legal = np.flatnonzero(legal_mask > 0)
            if legal.size == 0:
                break
            features = state.to_feature_vector().astype(np.float32, copy=False)
            probs = _safe_probs(adapter, features, legal_mask, resolved_device)
            if probs is not None:
                n_prob_rows += 1
                entropies.append(float(-np.sum(probs * np.log(np.clip(probs, 1.0e-12, 1.0)))))
                top_probs.append(float(np.max(probs)))
                allin_probs.append(float(probs[8]) if bool(legal_mask[8]) else 0.0)
            action_idx = adapter.select_action(
                state=state,
                features=features,
                legal_mask=legal_mask,
                device=resolved_device,
                rng=rng,
            )
            action_name = INDEX_TO_ACTION[int(action_idx)]
            selected_counts[action_name] += 1
            street = str(int(getattr(state, "stage", 0)))
            selected_by_street.setdefault(street, Counter())[action_name] += 1
            n_states += 1

            walk_action = int(walk_rng.choice(legal))
            random_walk_counts[INDEX_TO_ACTION[walk_action]] += 1
            state = state.apply_action(INDEX_TO_ACTION[walk_action])

    gate = evaluate_action_collapse(
        {"selected_action_counts": dict(selected_counts)},
        max_top_action_fraction=max_top_action_fraction,
        min_distinct_actions=min_distinct_actions,
    )
    return {
        "algorithm": "native_policy_action_distribution_diagnostic",
        "role": "local_slumbot_free_policy_distribution_gate",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "uses_slumbot_training_data": False,
        "uses_slumbot_trace_data": False,
        "promotion": False,
        "checkpoint": str(checkpoint),
        "kind": adapter.kind,
        "policy_algorithm": adapter.algorithm,
        "n_hands": int(n_hands),
        "n_states": int(n_states),
        "initial_chips": int(initial_chips),
        "max_steps_per_hand": int(max_steps_per_hand),
        "seed": int(seed),
        "probability_rows": int(n_prob_rows),
        "probability_metrics_available": bool(n_prob_rows == n_states and n_states > 0),
        "mean_entropy": float(np.mean(entropies)) if entropies else None,
        "mean_top_action_probability": float(np.mean(top_probs)) if top_probs else None,
        "mean_all_in_probability": float(np.mean(allin_probs)) if allin_probs else None,
        "selected_action_counts": dict(selected_counts),
        "selected_action_counts_by_street": {
            street: dict(counts) for street, counts in sorted(selected_by_street.items())
        },
        "random_walk_action_counts": dict(random_walk_counts),
        "gate": gate,
        "all_in_fraction": (
            float(selected_counts.get(INDEX_TO_ACTION[8], 0) / n_states) if n_states else 0.0
        ),
        "passed": bool(gate["passed"]),
        **device_info,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kind", choices=SUPPORTED_POLICY_KINDS, required=True)
    parser.add_argument("--n-hands", type=int, default=512)
    parser.add_argument("--max-steps-per-hand", type=int, default=64)
    parser.add_argument("--initial-chips", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260835)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-top-action-fraction", type=float, default=0.75)
    parser.add_argument("--min-distinct-actions", type=int, default=2)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    metrics = evaluate_native_policy_action_distribution(
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
