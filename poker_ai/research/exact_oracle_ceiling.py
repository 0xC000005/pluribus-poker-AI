"""Native one-step exact-oracle ceiling diagnostics.

This module asks a narrow question before more neural response-oracle training:
given the parent policy's continuation behavior, is there a better legal root
action on sampled public worlds? It is a ceiling diagnostic, not a deployable
policy or exploitability estimate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poker_ai.games.full_deck.state import INDEX_TO_ACTION
from poker_ai.research.mixed_policy_h2h import PolicyAdapter, load_policy_adapter
from poker_ai.research.native_nfsp import resolve_device
from poker_ai.research.public_action_rollout_value import (
    build_public_world_state,
    sample_public_worlds,
    sample_seeded_hero_cards,
    score_first_actions_across_worlds,
)


@dataclass(frozen=True)
class ExactOracleCeilingConfig:
    checkpoint: str
    kind: str = "tianshou-rainbow"
    n_roots: int = 8
    n_worlds: int = 16
    initial_chips: int = 1000
    small_blind: int = 50
    big_blind: int = 100
    max_steps_per_hand: int = 128
    seed: int = 20260527
    device: str = "auto"
    greedy_parent: bool = True
    greedy_continuation: bool = True
    min_roots: int = 1
    min_mean_oracle_gap: float = 0.0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(float(np.asarray(values, dtype=np.float64).mean()), 12)


def _action_name(action: int | str) -> str:
    if isinstance(action, str):
        return action
    return str(INDEX_TO_ACTION.get(int(action), int(action)))


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict):
        for key in ("rows", "records"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [dict(row) for row in rows]
    raise ValueError("records payload must be a list or contain rows/records")


def load_exact_oracle_ceiling_records(path: str | Path) -> list[dict[str, Any]]:
    """Load saved ceiling rows from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _records_from_payload(payload)


def summarize_exact_oracle_ceiling_records(
    records: list[dict[str, Any]],
    *,
    min_roots: int = 1,
    min_mean_oracle_gap: float = 0.0,
) -> dict[str, Any]:
    """Summarize whether the exact one-step oracle has useful decision signal."""
    rows = [dict(row) for row in records]
    parent_values = [float(row["parent_action_value"]) for row in rows]
    oracle_values = [float(row["oracle_action_value"]) for row in rows]
    oracle_gaps = [
        float(oracle) - float(parent)
        for parent, oracle in zip(parent_values, oracle_values, strict=False)
    ]
    truncated = int(sum(int(row.get("n_truncated_rollouts", 0)) for row in rows))
    finite = bool(
        np.isfinite(np.asarray(parent_values + oracle_values + oracle_gaps, dtype=np.float64)).all()
    )
    mean_gap = _mean(oracle_gaps)
    ceiling_positive = bool(mean_gap >= float(min_mean_oracle_gap))
    parent_actions = [_action_name(row["parent_action"]) for row in rows]
    oracle_actions = [_action_name(row["oracle_action"]) for row in rows]
    blockers: list[str] = []
    if len(rows) < int(min_roots):
        blockers.append("insufficient_roots")
    if truncated > 0:
        blockers.append("truncated_rollouts")
    if not finite:
        blockers.append("nonfinite_values")
    if not ceiling_positive:
        blockers.append("oracle_gap_below_threshold")
    match_rate = 0.0
    if rows:
        match_rate = round(
            float(np.mean([parent == oracle for parent, oracle in zip(parent_actions, oracle_actions, strict=False)])),
            12,
        )
    return {
        "mode": "native_exact_oracle_ceiling",
        "passed": bool(not blockers),
        "ceiling_positive": ceiling_positive,
        "promotable": False,
        "promotion": False,
        "promotion_blockers": [
            "ceiling_diagnostic_only",
            "requires_parent_population_h2h_after_training_candidate",
        ],
        "blockers": blockers,
        "n_roots": int(len(rows)),
        "min_roots": int(min_roots),
        "min_mean_oracle_gap": float(min_mean_oracle_gap),
        "mean_parent_action_value": _mean(parent_values),
        "mean_oracle_action_value": _mean(oracle_values),
        "mean_oracle_gap": mean_gap,
        "oracle_match_rate": match_rate,
        "parent_action_counts": dict(sorted(Counter(parent_actions).items())),
        "oracle_action_counts": dict(sorted(Counter(oracle_actions).items())),
        "n_truncated_rollouts": truncated,
        "rows": rows,
    }


def _adapter_action(
    adapter: PolicyAdapter,
    state: Any,
    device: torch.device,
    *,
    rng: np.random.Generator,
    greedy: bool,
) -> int:
    legal_mask = state.get_legal_mask()
    legal = np.flatnonzero(legal_mask > 0)
    if legal.size == 0:
        raise ValueError("state has no legal actions")
    features = state.to_feature_vector().astype(np.float32, copy=False)
    probs = np.asarray(adapter.probs(features, legal_mask, device), dtype=np.float64)
    legal_probs = np.asarray(probs[legal], dtype=np.float64)
    total = float(legal_probs.sum())
    if total <= 0.0:
        legal_probs = np.ones_like(legal_probs) / float(legal_probs.size)
    else:
        legal_probs = legal_probs / total
    if greedy:
        return int(legal[int(np.argmax(legal_probs))])
    return int(rng.choice(legal, p=legal_probs))


def policy_adapter_continuation_policy(
    adapter: PolicyAdapter,
    device: torch.device,
    *,
    greedy: bool,
):
    """Return a continuation policy using the same native policy adapter."""

    def _policy(state: Any, rng: np.random.Generator) -> int:
        return _adapter_action(adapter, state, device, rng=rng, greedy=greedy)

    return _policy


def evaluate_exact_oracle_ceiling(cfg: ExactOracleCeilingConfig) -> dict[str, Any]:
    """Evaluate parent root decisions against the sampled one-step oracle."""
    started = time.perf_counter()
    device_info = resolve_device(cfg.device)
    device = torch.device(device_info["resolved_device"])
    adapter = load_policy_adapter(cfg.checkpoint, kind=cfg.kind, device=device)
    continuation = policy_adapter_continuation_policy(
        adapter,
        device,
        greedy=cfg.greedy_continuation,
    )
    rows: list[dict[str, Any]] = []
    for root_idx in range(int(cfg.n_roots)):
        hero_cards = sample_seeded_hero_cards(seed=cfg.seed, root_idx=root_idx)
        worlds = sample_public_worlds(
            hero_cards=hero_cards,
            n_worlds=int(cfg.n_worlds),
            seed=int(cfg.seed) + 1009 * (root_idx + 1),
        )
        root = build_public_world_state(
            hero_cards=worlds[0].hero_cards,
            opponent_cards=worlds[0].opponent_cards,
            deck_tail=worlds[0].deck_tail,
            initial_chips=cfg.initial_chips,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
        )
        parent_action = _adapter_action(
            adapter,
            root,
            device,
            rng=np.random.default_rng(int(cfg.seed) + root_idx),
            greedy=cfg.greedy_parent,
        )
        scored = score_first_actions_across_worlds(
            worlds=worlds,
            continuation_policy=continuation,
            seed=int(cfg.seed) + 2003 * (root_idx + 1),
            max_steps_per_hand=int(cfg.max_steps_per_hand),
            initial_chips=int(cfg.initial_chips),
            small_blind=int(cfg.small_blind),
            big_blind=int(cfg.big_blind),
        )
        rows.append(
            {
                "root_idx": int(root_idx),
                "hero_cards": [int(card) for card in hero_cards],
                "parent_action": _action_name(parent_action),
                "oracle_action": _action_name(scored.best_action),
                "parent_action_value": float(scored.action_values[int(parent_action)]),
                "oracle_action_value": float(scored.best_action_value),
                "oracle_gap": float(
                    scored.best_action_value - scored.action_values[int(parent_action)]
                ),
                "action_values": {
                    _action_name(action): float(value)
                    for action, value in sorted(scored.action_values.items())
                },
                "action_standard_errors": {
                    _action_name(action): float(value)
                    for action, value in sorted(scored.action_standard_errors.items())
                },
                "n_truncated_rollouts": int(scored.n_truncated_rollouts),
            }
        )

    summary = summarize_exact_oracle_ceiling_records(
        rows,
        min_roots=int(cfg.min_roots),
        min_mean_oracle_gap=float(cfg.min_mean_oracle_gap),
    )
    summary.update(
        {
            "checkpoint": str(cfg.checkpoint),
            "kind": str(cfg.kind),
            "algorithm": str(adapter.algorithm),
            "n_worlds": int(cfg.n_worlds),
            "seed": int(cfg.seed),
            "initial_chips": int(cfg.initial_chips),
            "small_blind": int(cfg.small_blind),
            "big_blind": int(cfg.big_blind),
            "max_steps_per_hand": int(cfg.max_steps_per_hand),
            "greedy_parent": bool(cfg.greedy_parent),
            "greedy_continuation": bool(cfg.greedy_continuation),
            "device": device_info,
            "elapsed_seconds": float(time.perf_counter() - started),
            "uses_slumbot_training_data": False,
        }
    )
    return summary
