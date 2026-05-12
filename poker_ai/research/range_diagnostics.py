"""Diagnostics for learned range-update likelihoods."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from poker_ai.research.evaluation import (
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import card_str_to_index, get_legal_mask_from_parsed, parse_action  # noqa: E402
from range_tracker import RangeTracker, update_tracker_from_actions  # noqa: E402


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _range_summary(prefix: str, values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    total = float(arr.sum())
    probs = arr / total if total > 0 else arr
    positive = probs[probs > 0]
    entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0
    denom = float(np.log(max(positive.size, 2))) if positive.size else 1.0
    sorted_probs = np.sort(probs)[::-1]
    return {
        f"{prefix}_support": int(positive.size),
        f"{prefix}_top1_mass": round(float(sorted_probs[0]) if sorted_probs.size else 0.0, 6),
        f"{prefix}_top10_mass": round(float(sorted_probs[:10].sum()) if sorted_probs.size else 0.0, 6),
        f"{prefix}_entropy": round(entropy, 6),
        f"{prefix}_normalized_entropy": round(entropy / denom if denom > 0 else 0.0, 6),
    }


def _action_dispersion(strategies: np.ndarray, legal_mask: np.ndarray) -> dict[str, Any]:
    legal = np.flatnonzero(legal_mask > 0)
    if legal.size == 0 or strategies.size == 0:
        return {
            "legal_actions": [],
            "mean_action_prob_std": 0.0,
            "max_action_prob_std": 0.0,
            "mean_action_prob_range": 0.0,
            "top_action_diversity": 0,
        }
    legal_probs = strategies[:, legal]
    stds = legal_probs.std(axis=0)
    ranges = legal_probs.max(axis=0) - legal_probs.min(axis=0)
    top_actions = legal[np.argmax(legal_probs, axis=1)]
    return {
        "legal_actions": [int(action) for action in legal],
        "mean_action_prob_std": round(float(stds.mean()), 6),
        "max_action_prob_std": round(float(stds.max()), 6),
        "mean_action_prob_range": round(float(ranges.mean()), 6),
        "top_action_diversity": int(np.unique(top_actions).size),
    }


def diagnose_range_likelihood(
    checkpoint: str | Path,
    cases: Iterable[ResolverBenchmarkCase],
    *,
    strategy_source: str = "regret",
    device: str | torch.device = "auto",
    max_cases: int | None = None,
) -> dict[str, Any]:
    """Measure whether range-update action likelihoods vary across hands."""
    resolved_device = _resolve_device(device)
    loaded = load_value_network_checkpoint(checkpoint, resolved_device)
    assert_strategy_source_supported(loaded, strategy_source)
    selected_cases = list(cases)
    if max_cases is not None:
        selected_cases = selected_cases[:max_cases]

    records: list[dict[str, Any]] = []
    for case in selected_cases:
        parsed = parse_action(case.action_str)
        if "error" in parsed:
            records.append({"label": case.label, "skipped": f"parse_error:{parsed['error']}"})
            continue
        street = int(parsed.get("st", -1))
        if street not in (2, 3):
            records.append({"label": case.label, "skipped": f"unsupported_street:{street}"})
            continue
        n_board = 4 if street == 2 else 5
        board_idx = [card_str_to_index(card) for card in case.board[:n_board]]
        our_cards_idx = [card_str_to_index(card) for card in case.hole_cards]
        tracker = RangeTracker(
            our_cards_idx,
            loaded.value_net,
            resolved_device,
            strategy_source=strategy_source,
        )
        update_tracker_from_actions(tracker, case.action_str, case.client_pos, board_idx)
        solver_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
        solver_hand_to_idx = {hand: i for i, hand in enumerate(solver_hands)}
        hero_range, villain_range = tracker.get_solver_ranges(solver_hands, solver_hand_to_idx)

        legal_mask = get_legal_mask_from_parsed(parsed, case.action_str, case.client_pos)
        hero_strategies = tracker._batch_blueprint(  # diagnostic use of local likelihood model
            tracker.hero_hands,
            board_idx,
            case.action_str,
            case.client_pos,
            parsed,
        )
        villain_pos = 1 - case.client_pos
        villain_strategies = tracker._batch_blueprint(
            tracker.opponent_hands,
            board_idx,
            case.action_str,
            villain_pos,
            parsed,
        )
        records.append(
            {
                "label": case.label,
                "street": street,
                "action_str": case.action_str,
                **_range_summary("hero_range", hero_range),
                **_range_summary("villain_range", villain_range),
                "hero_likelihood": _action_dispersion(hero_strategies, legal_mask),
                "villain_likelihood": _action_dispersion(villain_strategies, legal_mask),
            }
        )

    completed = [record for record in records if "skipped" not in record]
    summary: dict[str, Any] = {
        "mode": "range_likelihood_diagnostics",
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": loaded.metadata.get("checkpoint_iteration"),
        "strategy_source": strategy_source,
        "device": str(resolved_device),
        "n_cases": len(selected_cases),
        "n_completed": len(completed),
        "records": records,
        "passed": bool(completed),
    }
    for key in (
        "hero_range_normalized_entropy",
        "villain_range_normalized_entropy",
        "hero_range_top1_mass",
        "villain_range_top1_mass",
    ):
        values = [float(record[key]) for record in completed if key in record]
        if values:
            summary[f"mean_{key}"] = round(float(np.mean(values)), 6)
    for player in ("hero", "villain"):
        values = [
            float(record[f"{player}_likelihood"]["mean_action_prob_std"])
            for record in completed
        ]
        if values:
            summary[f"mean_{player}_likelihood_action_std"] = round(
                float(np.mean(values)),
                6,
            )
    return summary
