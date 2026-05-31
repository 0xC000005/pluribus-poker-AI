"""Early-street blueprint calibration diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from poker_ai.games.full_deck.state import N_ACTIONS
from poker_ai.research.evaluation import (
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import (  # noqa: E402
    action_to_slumbot,
    build_features,
    get_legal_mask_from_parsed,
    network_strategy,
    parse_action,
)
from poker_ai.research.policy_calibration import (  # noqa: E402
    _append_action,
    _resolve_device,
    _sample_cards,
    _visible_board,
)


STREET_NAMES = {
    0: "preflop",
    1: "flop",
    2: "turn",
    3: "river",
}


def _as_policy(values: Iterable[float], legal_mask: Iterable[float]) -> np.ndarray:
    policy = np.asarray(list(values), dtype=np.float64).reshape(N_ACTIONS)
    mask = np.asarray(list(legal_mask), dtype=np.float64).reshape(N_ACTIONS) > 0
    policy = np.where(mask, policy, 0.0)
    total = float(policy.sum())
    if total > 1e-12:
        return policy / total
    legal_total = int(mask.sum())
    if legal_total <= 0:
        raise ValueError("row has no legal actions")
    fallback = np.zeros(N_ACTIONS, dtype=np.float64)
    fallback[mask] = 1.0 / legal_total
    return fallback


def _top_counts(policies: np.ndarray) -> dict[str, int]:
    if policies.size == 0:
        return {}
    top = np.argmax(policies, axis=1)
    unique, counts = np.unique(top, return_counts=True)
    return {str(int(action)): int(count) for action, count in zip(unique, counts, strict=True)}


def _entropy(policy: np.ndarray) -> float:
    positive = policy[policy > 1e-12]
    return float(-(positive * np.log(positive)).sum()) if positive.size else 0.0


def _summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_rows": 0,
            "candidate_top_allin_rate": 0.0,
            "reference_top_allin_rate": 0.0,
            "mean_candidate_allin_prob": 0.0,
            "mean_reference_allin_prob": 0.0,
            "mean_l1_to_reference": 0.0,
            "top_action_agreement": 0.0,
            "candidate_top_action_counts": {},
            "reference_top_action_counts": {},
            "mean_candidate_entropy": 0.0,
            "mean_reference_entropy": 0.0,
        }
    candidate = np.asarray(
        [_as_policy(row["candidate_strategy"], row["legal_mask"]) for row in rows],
        dtype=np.float64,
    )
    reference = np.asarray(
        [_as_policy(row["reference_strategy"], row["legal_mask"]) for row in rows],
        dtype=np.float64,
    )
    candidate_top = np.argmax(candidate, axis=1)
    reference_top = np.argmax(reference, axis=1)
    return {
        "n_rows": int(len(rows)),
        "candidate_top_allin_rate": round(float(np.mean(candidate_top == 8)), 8),
        "reference_top_allin_rate": round(float(np.mean(reference_top == 8)), 8),
        "mean_candidate_allin_prob": round(float(np.mean(candidate[:, 8])), 8),
        "mean_reference_allin_prob": round(float(np.mean(reference[:, 8])), 8),
        "mean_l1_to_reference": round(float(np.abs(candidate - reference).sum(axis=1).mean()), 8),
        "top_action_agreement": round(float(np.mean(candidate_top == reference_top)), 8),
        "candidate_top_action_counts": _top_counts(candidate),
        "reference_top_action_counts": _top_counts(reference),
        "mean_candidate_entropy": round(float(np.mean([_entropy(row) for row in candidate])), 8),
        "mean_reference_entropy": round(float(np.mean([_entropy(row) for row in reference])), 8),
    }


def summarize_early_street_rows(
    rows: list[dict[str, Any]],
    *,
    max_candidate_preflop_allin_rate: float = 0.10,
    max_candidate_flop_allin_rate: float = 0.20,
    max_mean_l1_to_reference: float = 0.75,
) -> dict[str, Any]:
    """Summarize candidate/reference policy drift on shared early-street rows."""
    by_street: dict[str, Any] = {}
    for street in (0, 1):
        name = STREET_NAMES[street]
        by_street[name] = _summarize_subset(
            [row for row in rows if int(row.get("street", -1)) == street]
        )
    overall = _summarize_subset(rows)
    failed_checks: list[str] = []
    if by_street["preflop"]["candidate_top_allin_rate"] > float(max_candidate_preflop_allin_rate):
        failed_checks.append("preflop_allin_rate")
    if by_street["flop"]["candidate_top_allin_rate"] > float(max_candidate_flop_allin_rate):
        failed_checks.append("flop_allin_rate")
    if overall["mean_l1_to_reference"] > float(max_mean_l1_to_reference):
        failed_checks.append("mean_l1_to_reference")
    return {
        "mode": "early_street_blueprint_calibration",
        "promotion": False,
        "passed": not failed_checks,
        "n_rows": int(len(rows)),
        "max_candidate_preflop_allin_rate": float(max_candidate_preflop_allin_rate),
        "max_candidate_flop_allin_rate": float(max_candidate_flop_allin_rate),
        "max_mean_l1_to_reference": float(max_mean_l1_to_reference),
        "failed_checks": failed_checks,
        "overall": overall,
        "by_street": by_street,
    }


def _policy_from_checkpoint(
    loaded: Any,
    features: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
    *,
    strategy_source: str,
) -> np.ndarray:
    _action, strategy = network_strategy(
        loaded.value_net,
        features,
        legal_mask,
        device,
        strategy_source=strategy_source,
    )
    return np.asarray(strategy, dtype=np.float64).reshape(N_ACTIONS)


def evaluate_early_street_calibration(
    *,
    candidate_checkpoint: str | Path,
    reference_checkpoint: str | Path,
    state_source_checkpoint: str | Path | None = None,
    n_states: int = 512,
    seed: int = 0,
    candidate_strategy_source: str = "regret",
    reference_strategy_source: str = "regret",
    state_source_strategy_source: str = "regret",
    device: str | torch.device = "auto",
    max_hands: int | None = None,
    max_candidate_preflop_allin_rate: float = 0.10,
    max_candidate_flop_allin_rate: float = 0.20,
    max_mean_l1_to_reference: float = 0.75,
) -> dict[str, Any]:
    """Evaluate candidate/reference blueprint policies on identical early states."""
    resolved_device = _resolve_device(device)
    candidate = load_value_network_checkpoint(candidate_checkpoint, resolved_device)
    reference = load_value_network_checkpoint(reference_checkpoint, resolved_device)
    state_source_path = state_source_checkpoint or candidate_checkpoint
    state_source = load_value_network_checkpoint(state_source_path, resolved_device)
    assert_strategy_source_supported(candidate, candidate_strategy_source)
    assert_strategy_source_supported(reference, reference_strategy_source)
    assert_strategy_source_supported(state_source, state_source_strategy_source)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    attempted_hands = 0
    max_hands = int(max_hands or max(100, n_states * 5))
    while len(rows) < int(n_states) and attempted_hands < max_hands:
        attempted_hands += 1
        cards = _sample_cards(rng, 9)
        seat_holes = [cards[:2], cards[2:4]]
        board = cards[4:9]
        action_str = ""
        for _step in range(24):
            parsed = parse_action(action_str)
            if "error" in parsed:
                break
            street = int(parsed.get("st", -1))
            acting_pos = int(parsed.get("pos", -1))
            if acting_pos < 0 or street > 1:
                break
            visible_board = _visible_board(board, street)
            features = build_features(
                seat_holes[acting_pos],
                visible_board,
                action_str,
                acting_pos,
                parsed,
            )
            legal_mask = get_legal_mask_from_parsed(parsed, action_str, acting_pos)
            candidate_strategy = _policy_from_checkpoint(
                candidate,
                features,
                legal_mask,
                resolved_device,
                strategy_source=candidate_strategy_source,
            )
            reference_strategy = _policy_from_checkpoint(
                reference,
                features,
                legal_mask,
                resolved_device,
                strategy_source=reference_strategy_source,
            )
            rows.append(
                {
                    "street": int(street),
                    "street_name": STREET_NAMES.get(street, str(street)),
                    "action_str": action_str,
                    "acting_pos": int(acting_pos),
                    "legal_actions": [int(action) for action in np.flatnonzero(legal_mask > 0)],
                    "legal_mask": (legal_mask > 0).astype(np.float32).tolist(),
                    "candidate_strategy": candidate_strategy.round(8).tolist(),
                    "reference_strategy": reference_strategy.round(8).tolist(),
                }
            )
            if len(rows) >= int(n_states):
                break

            source_strategy = _policy_from_checkpoint(
                state_source,
                features,
                legal_mask,
                resolved_device,
                strategy_source=state_source_strategy_source,
            )
            legal_actions = np.flatnonzero(legal_mask > 0)
            probs = np.asarray([source_strategy[action] for action in legal_actions], dtype=np.float64)
            total = float(probs.sum())
            probs = probs / total if total > 1e-12 else np.ones_like(probs) / len(probs)
            action_idx = int(rng.choice(legal_actions, p=probs))
            increment = action_to_slumbot(action_idx, parsed, action_str, acting_pos)
            action_str = _append_action(action_str, increment, street)
            if increment.startswith("f") or action_idx == 8:
                break

    if len(rows) < int(n_states):
        raise RuntimeError(
            f"generated {len(rows)} early-street rows out of requested {n_states} "
            f"after {attempted_hands} hands"
        )

    metrics = summarize_early_street_rows(
        rows,
        max_candidate_preflop_allin_rate=max_candidate_preflop_allin_rate,
        max_candidate_flop_allin_rate=max_candidate_flop_allin_rate,
        max_mean_l1_to_reference=max_mean_l1_to_reference,
    )
    metrics.update(
        {
            "candidate_checkpoint": str(candidate_checkpoint),
            "reference_checkpoint": str(reference_checkpoint),
            "state_source_checkpoint": str(state_source_path),
            "candidate_strategy_source": candidate_strategy_source,
            "reference_strategy_source": reference_strategy_source,
            "state_source_strategy_source": state_source_strategy_source,
            "device": str(resolved_device),
            "seed": int(seed),
            "attempted_hands": int(attempted_hands),
            "rows": rows,
        }
    )
    return metrics
