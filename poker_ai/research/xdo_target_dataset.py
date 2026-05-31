"""Build XDO-lite state-local policy target datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections import Counter

import numpy as np

from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import ACTION_TO_INDEX, N_ACTIONS, N_FEATURES
from poker_ai.research.public_action_rollout_value import (
    build_public_world_state,
    sample_public_worlds,
    sample_seeded_hero_cards,
)
from poker_ai.research.xdo_target_audit import audit_xdo_target_records


def _load_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"rows": payload}
    if not isinstance(payload, dict):
        raise ValueError("XDO target records must be a JSON list or object")
    if "rows" not in payload and "records" in payload:
        payload = dict(payload)
        payload["rows"] = payload["records"]
    if not isinstance(payload.get("rows"), list):
        raise ValueError("XDO target payload must contain rows")
    return payload


def _action_index(action: Any) -> int:
    if isinstance(action, str):
        if action in ACTION_TO_INDEX:
            return int(ACTION_TO_INDEX[action])
        return -1
    try:
        idx = int(action)
    except (TypeError, ValueError):
        return -1
    return idx if 0 <= idx < N_ACTIONS else -1


def _hero_cards(row: dict[str, Any], *, seed: int, root_idx: int) -> tuple[int, int]:
    cards = row.get("hero_cards")
    if cards is None:
        return sample_seeded_hero_cards(seed=seed, root_idx=root_idx)
    if len(cards) != 2:
        raise ValueError(f"row {root_idx} hero_cards must contain exactly two cards")
    return int(cards[0]), int(cards[1])


def _root_tensors(
    row: dict[str, Any],
    *,
    seed: int,
    initial_chips: int,
    small_blind: int,
    big_blind: int,
) -> tuple[np.ndarray, np.ndarray]:
    root_idx = int(row.get("root_idx", 0))
    hero_cards = _hero_cards(row, seed=seed, root_idx=root_idx)
    worlds = sample_public_worlds(
        hero_cards=hero_cards,
        n_worlds=1,
        seed=int(seed) + 1009 * (root_idx + 1),
    )
    root = build_public_world_state(
        hero_cards=worlds[0].hero_cards,
        opponent_cards=worlds[0].opponent_cards,
        deck_tail=worlds[0].deck_tail,
        initial_chips=int(initial_chips),
        small_blind=int(small_blind),
        big_blind=int(big_blind),
    )
    return (
        root.to_feature_vector().astype(np.float32, copy=False),
        root.get_legal_mask().astype(np.float32, copy=False),
    )


def _street_index(features: np.ndarray) -> int:
    street = np.asarray(features, dtype=np.float32)[104:108]
    if street.size != 4 or float(street.max(initial=0.0)) <= 0.0:
        return -1
    return int(np.argmax(street))


def build_xdo_policy_target_dataset(
    records_json: str | Path,
    output_npz: str | Path,
    *,
    output_json: str | Path | None = None,
    min_targets: int = 16,
    max_top_action_fraction: float = 0.75,
    min_mean_gap: float = 0.0,
    min_holdout_mean_gap: float = 0.0,
) -> dict[str, Any]:
    """Convert exact-oracle rows into a neural policy-target artifact.

    This is an export step only. It intentionally uses one-hot oracle targets
    and uniform per-row weights so later experiments test the information-state
    response object itself, not a hidden target weighting trick.
    """
    payload = _load_payload(records_json)
    rows = [dict(row) for row in payload["rows"]]
    seed = int(payload.get("seed", 20260527))
    initial_chips = int(payload.get("initial_chips", 1000))
    small_blind = int(payload.get("small_blind", 50))
    big_blind = int(payload.get("big_blind", 100))

    features = np.zeros((len(rows), N_FEATURES), dtype=np.float32)
    legal_masks = np.zeros((len(rows), N_ACTIONS), dtype=np.float32)
    target_probs = np.zeros((len(rows), N_ACTIONS), dtype=np.float32)
    weights = np.ones(len(rows), dtype=np.float32)
    root_indices = np.zeros(len(rows), dtype=np.int32)
    street_indices = np.full(len(rows), -1, dtype=np.int32)
    parent_actions = np.full(len(rows), -1, dtype=np.int32)
    oracle_actions = np.full(len(rows), -1, dtype=np.int32)
    oracle_gaps = np.zeros(len(rows), dtype=np.float32)
    action_values = np.full((len(rows), N_ACTIONS), np.nan, dtype=np.float32)

    blockers: list[str] = []
    for row_i, row in enumerate(rows):
        root_idx = int(row.get("root_idx", row_i))
        root_indices[row_i] = root_idx
        feature, mask = _root_tensors(
            row,
            seed=seed,
            initial_chips=initial_chips,
            small_blind=small_blind,
            big_blind=big_blind,
        )
        features[row_i] = feature
        legal_masks[row_i] = mask
        street_indices[row_i] = _street_index(feature)

        parent_idx = _action_index(row.get("parent_action"))
        oracle_idx = _action_index(row.get("oracle_action"))
        parent_actions[row_i] = parent_idx
        oracle_actions[row_i] = oracle_idx
        oracle_gaps[row_i] = float(row.get("oracle_gap", 0.0))
        if oracle_idx < 0:
            blockers.append("unknown_oracle_action")
        elif mask[oracle_idx] <= 0:
            blockers.append("oracle_action_illegal_at_reconstructed_root")
        else:
            target_probs[row_i, oracle_idx] = 1.0

        values = row.get("action_values", {})
        if isinstance(values, dict):
            for action, value in values.items():
                action_idx = _action_index(action)
                if action_idx >= 0:
                    action_values[row_i, action_idx] = float(value)

    audit = audit_xdo_target_records(
        rows,
        min_targets=int(min_targets),
        max_top_action_fraction=float(max_top_action_fraction),
        min_mean_gap=float(min_mean_gap),
        min_holdout_mean_gap=float(min_holdout_mean_gap),
    )
    unique_blockers = sorted(set(blockers + list(audit.get("blockers", []))))
    street_counts = Counter(str(int(street)) for street in street_indices if int(street) >= 0)
    coverage_blockers: list[str] = []
    if len(street_counts) < 4:
        coverage_blockers.append("insufficient_target_streets_for_whole_game_h2h")
    buffer = PolicyTargetBuffer(features, legal_masks, target_probs, weights)
    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=buffer.features,
        legal_masks=buffer.legal_masks,
        target_probs=buffer.target_probs,
        weights=buffer.weights,
        root_indices=root_indices,
        street_indices=street_indices,
        parent_actions=parent_actions,
        oracle_actions=oracle_actions,
        oracle_gaps=oracle_gaps,
        action_values=action_values,
    )

    metrics = {
        "mode": "xdo_lite_information_state_policy_target_dataset",
        "passed": bool(audit["passed"] and not blockers),
        "promotion": False,
        "promotable": False,
        "blockers": unique_blockers,
        "records_json": str(records_json),
        "output_npz": str(output_npz),
        "target_size": int(buffer.size),
        "n_targets": int(buffer.size),
        "feature_dim": int(buffer.features.shape[1]),
        "n_actions": int(buffer.target_probs.shape[1]),
        "seed": seed,
        "initial_chips": initial_chips,
        "small_blind": small_blind,
        "big_blind": big_blind,
        "target_weighting": "uniform",
        "target_street_counts": dict(sorted(street_counts.items())),
        "whole_game_h2h_ready": bool(not coverage_blockers),
        "coverage_blockers": coverage_blockers,
        "audit": audit,
        "promotion_blockers": [
            "target_dataset_only",
            "requires_neural_consumer_parent_population_h2h",
            "slumbot_heldout_confirmation_required",
        ],
        "uses_slumbot_training_data": False,
    }
    if output_json is not None:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics
