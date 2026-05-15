"""Fixed-state resolver A/B for calibrated Slumbot response ranges.

This module is diagnostic-only. It keeps the live Slumbot player unchanged and
asks whether a learned opponent-response model changes the ranges that feed
turn/river resolving on replayed Slumbot public states.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from poker_ai.deep_cfr.fast_state import N_ACTIONS
from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase
from poker_ai.research.slumbot_opponent_response_probe import (
    _predict_logits,
    _probs_from_logits,
)
from poker_ai.research.slumbot_opponent_response_range_ab import _train_response_model


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import (  # noqa: E402
    _compute_bets_before_street,
    card_str_to_index,
    parse_action,
)
from range_tracker import (  # noqa: E402
    RangeTracker,
    _build_features_batch,
    _get_legal_mask,
    map_slumbot_action_to_idx,
    update_tracker_from_actions,
    walk_actions,
)
from solver import solve_street  # noqa: E402


_TRACE_HAND_RE = re.compile(r"trace-hand(\d+)-")


@dataclass(frozen=True)
class ResponseRangeResult:
    solver_hands: tuple[tuple[int, int], ...]
    villain_range: np.ndarray
    n_opponent_updates: int
    n_revealed_board_cards: int


@dataclass(frozen=True)
class SolverRangeResult:
    solver_hands: tuple[tuple[int, int], ...]
    hero_range: np.ndarray
    villain_range: np.ndarray


def _normalize(values: np.ndarray) -> np.ndarray:
    out = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(out.sum())
    if total <= 1e-12:
        out = np.ones_like(out, dtype=np.float64)
        total = float(out.sum())
    return out / total


def _visible_board_for(prefix: str, board_idx: list[int]) -> list[int]:
    n_slashes = prefix.count("/")
    if n_slashes >= 3 and len(board_idx) >= 5:
        return list(board_idx[:5])
    if n_slashes >= 2 and len(board_idx) >= 4:
        return list(board_idx[:4])
    if n_slashes >= 1 and len(board_idx) >= 3:
        return list(board_idx[:3])
    return []


def _zero_blocked_hands(
    ranges: np.ndarray,
    hands: list[tuple[int, int]],
    blocked_cards: Iterable[int],
) -> None:
    blocked = set(int(card) for card in blocked_cards)
    if not blocked:
        return
    for idx, hand in enumerate(hands):
        if hand[0] in blocked or hand[1] in blocked:
            ranges[idx] = 0.0
    total = float(ranges.sum())
    if total > 0.0:
        ranges /= total


def _standardize_features(
    features: np.ndarray,
    mean: np.ndarray | float,
    std: np.ndarray | float,
) -> np.ndarray:
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    std_arr = np.where(std_arr > 1e-6, std_arr, 1.0)
    return ((features.astype(np.float32) - mean_arr) / std_arr).astype(np.float32)


def _solver_hands_for_board(board_idx: list[int]) -> list[tuple[int, int]]:
    remaining = sorted(set(range(52)) - set(int(card) for card in board_idx))
    return list(itertools.combinations(remaining, 2))


def response_villain_range_for_case(
    case: ResolverBenchmarkCase,
    *,
    model: torch.nn.Module,
    mean: np.ndarray | float,
    std: np.ndarray | float,
    temperature: float,
    device: torch.device,
) -> ResponseRangeResult:
    """Return Slumbot private-hand range from the learned response model."""
    parsed = parse_action(case.action_str)
    street = int(parsed.get("st", -1)) if "error" not in parsed else -1
    n_board = 5 if street == 3 else 4 if street == 2 else len(case.board)
    board_idx = [card_str_to_index(card) for card in case.board[:n_board]]
    our_cards_idx = [card_str_to_index(card) for card in case.hole_cards]
    opponent_cards = sorted(set(range(52)) - set(our_cards_idx))
    opponent_hands = list(itertools.combinations(opponent_cards, 2))
    opponent_range = np.ones(len(opponent_hands), dtype=np.float64) / float(
        len(opponent_hands)
    )
    opp_pos = 1 - int(case.client_pos)
    revealed = 0
    n_updates = 0

    for before, acting_pos, action_char, bet_to in walk_actions(case.action_str):
        current_board = _visible_board_for(before, board_idx)
        if len(current_board) > revealed:
            _zero_blocked_hands(opponent_range, opponent_hands, current_board[revealed:])
            revealed = len(current_board)
        if int(acting_pos) != opp_pos:
            continue
        parsed_before = parse_action(before)
        if "error" in parsed_before:
            continue
        action_data = map_slumbot_action_to_idx(
            action_char,
            int(bet_to),
            before,
            int(acting_pos),
            parsed_before,
        )
        features = _build_features_batch(
            opponent_hands,
            current_board,
            before,
            opp_pos,
            parsed_before,
        )
        x = _standardize_features(features, mean, std)
        logits = _predict_logits(model, x, device)
        legal_mask = _get_legal_mask(parsed_before, before, opp_pos).astype(np.float32)
        legal = np.broadcast_to(legal_mask, (len(opponent_hands), N_ACTIONS)).copy()
        probs = _probs_from_logits(logits, legal, temperature=float(temperature))
        likelihood = np.zeros(len(opponent_hands), dtype=np.float64)
        for action_idx, weight in action_data:
            likelihood += probs[:, int(action_idx)] * float(weight)
        eps = 0.01
        likelihood = (1.0 - eps) * likelihood + eps * (1.0 / N_ACTIONS)
        opponent_range *= likelihood
        opponent_range = _normalize(opponent_range)
        n_updates += 1

    final_board = _visible_board_for(case.action_str, board_idx)
    if len(final_board) > revealed:
        _zero_blocked_hands(opponent_range, opponent_hands, final_board[revealed:])
        revealed = len(final_board)

    solver_hands = _solver_hands_for_board(board_idx)
    hand_to_opp_idx = {hand: idx for idx, hand in enumerate(opponent_hands)}
    villain_range = np.zeros(len(solver_hands), dtype=np.float64)
    for solver_idx, hand in enumerate(solver_hands):
        opp_idx = hand_to_opp_idx.get(tuple(sorted(hand)))
        if opp_idx is not None:
            villain_range[solver_idx] = opponent_range[opp_idx]
    villain_range = _normalize(villain_range)
    return ResponseRangeResult(
        solver_hands=tuple(solver_hands),
        villain_range=villain_range,
        n_opponent_updates=n_updates,
        n_revealed_board_cards=revealed,
    )


def baseline_solver_ranges_for_case(
    case: ResolverBenchmarkCase,
    *,
    value_net: ValueNetwork,
    device: torch.device,
    strategy_source: str,
) -> SolverRangeResult:
    parsed = parse_action(case.action_str)
    street = int(parsed.get("st", -1)) if "error" not in parsed else -1
    n_board = 5 if street == 3 else 4 if street == 2 else len(case.board)
    board_idx = [card_str_to_index(card) for card in case.board[:n_board]]
    our_cards_idx = [card_str_to_index(card) for card in case.hole_cards]
    tracker = RangeTracker(
        our_cards_idx,
        value_net,
        device,
        strategy_source=strategy_source,
    )
    update_tracker_from_actions(tracker, case.action_str, int(case.client_pos), board_idx)
    solver_hands = _solver_hands_for_board(board_idx)
    hand_to_idx = {hand: idx for idx, hand in enumerate(solver_hands)}
    hero_range, villain_range = tracker.get_solver_ranges(solver_hands, hand_to_idx)
    return SolverRangeResult(
        solver_hands=tuple(solver_hands),
        hero_range=_normalize(hero_range),
        villain_range=_normalize(villain_range),
    )


def _strategy_vector(strategy: dict[int, float]) -> np.ndarray:
    out = np.zeros(N_ACTIONS, dtype=np.float64)
    for action_idx, prob in strategy.items():
        if 0 <= int(action_idx) < N_ACTIONS:
            out[int(action_idx)] = float(prob)
    total = float(out.sum())
    if total > 0.0:
        out /= total
    else:
        out[1] = 1.0
    return out


def _range_summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    probs = _normalize(values)
    positive = probs[probs > 0.0]
    entropy = -float(np.sum(positive * np.log(np.maximum(positive, 1e-12))))
    norm_entropy = entropy / math.log(max(int(positive.size), 2))
    return {
        f"{prefix}_top1_mass": float(np.max(positive)) if positive.size else 0.0,
        f"{prefix}_top10_mass": float(np.sort(positive)[::-1][:10].sum()) if positive.size else 0.0,
        f"{prefix}_normalized_entropy": float(norm_entropy),
    }


def true_hand_log_lift(
    solver_hands: Iterable[tuple[int, int]],
    values: np.ndarray,
    *,
    bot_hand: tuple[int, int],
) -> dict[str, float] | None:
    probs = _normalize(values)
    hands = [tuple(sorted(hand)) for hand in solver_hands]
    true_hand = tuple(sorted(int(card) for card in bot_hand))
    try:
        true_idx = hands.index(true_hand)
    except ValueError:
        return None
    positive = probs[probs > 0.0]
    if positive.size <= 0:
        return None
    true_prob = float(probs[true_idx])
    uniform_prob = 1.0 / float(positive.size)
    return {
        "true_hand_prob": true_prob,
        "uniform_prob": uniform_prob,
        "log_lift_vs_uniform": float(
            math.log(max(true_prob, 1e-12)) - math.log(max(uniform_prob, 1e-12))
        ),
        "true_hand_percentile": float(np.mean(positive <= true_prob)),
        "true_hand_rank": float(1 + np.sum(positive > true_prob)),
    }


def _hand_index_from_case(case: ResolverBenchmarkCase) -> int | None:
    match = _TRACE_HAND_RE.search(case.label)
    if match is None:
        return None
    return int(match.group(1))


def load_trace_bot_hands(trace_path: str | Path) -> dict[int, tuple[int, int]]:
    bot_hands: dict[int, tuple[int, int]] = {}
    with Path(trace_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") != "hand_result":
                continue
            cards = list(record.get("bot_hole_cards") or [])
            if len(cards) != 2:
                continue
            try:
                hand_index = int(record["hand_index"])
                bot_hands[hand_index] = tuple(
                    sorted(card_str_to_index(card) for card in cards)
                )
            except (KeyError, TypeError, ValueError):
                continue
    return bot_hands


def _solve_case_strategy(
    case: ResolverBenchmarkCase,
    *,
    hero_range: np.ndarray,
    villain_range: np.ndarray,
    solver_iterations: int,
    solver_backend: str,
    range_prune_threshold: float,
) -> dict[str, Any]:
    parsed = parse_action(case.action_str)
    if "error" in parsed:
        return {"action": 1, "strategy": np.eye(1, N_ACTIONS, 1)[0], "error": parsed["error"]}
    street = int(parsed.get("st", -1))
    n_board = 5 if street == 3 else 4
    board_idx = [card_str_to_index(card) for card in case.board[:n_board]]
    our_cards_idx = [card_str_to_index(card) for card in case.hole_cards]
    our_bet_pre, opp_bet_pre = _compute_bets_before_street(
        case.action_str,
        int(case.client_pos),
        target_street=street,
    )
    pot = our_bet_pre + opp_bet_pre
    hero_stack = 20000 - our_bet_pre
    villain_stack = 20000 - opp_bet_pre
    hero_first = int(case.client_pos) == 0
    street_parts = case.action_str.split("/")
    street_action = street_parts[street] if len(street_parts) > street else ""
    started = time.perf_counter()
    action, strategy, solver, node = solve_street(
        our_cards_idx,
        board_idx,
        pot,
        hero_stack,
        villain_stack,
        hero_first,
        action_str=street_action,
        n_iterations=int(solver_iterations),
        hero_range=hero_range,
        villain_range=villain_range,
        backend=solver_backend,
        range_prune_threshold=float(range_prune_threshold),
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "action": int(action),
        "strategy": _strategy_vector(strategy),
        "latency_ms": float(latency_ms),
        "solver_n_hands": int(getattr(solver, "n", 0)),
        "node_terminal": bool(node is None or getattr(node, "is_terminal", False)),
        "solve_ms": float(getattr(solver, "last_solve_ms", 0.0)),
    }


def evaluate_response_range_solver_ab(
    value_net: ValueNetwork,
    response_model: torch.nn.Module,
    response_mean: np.ndarray,
    response_std: np.ndarray,
    response_temperature: float,
    device: torch.device,
    *,
    cases: Iterable[ResolverBenchmarkCase],
    strategy_source: str,
    solver_iterations: int = 5,
    solver_backend: str = "auto",
    range_prune_threshold: float = 1e-4,
    bot_hands_by_hand_index: dict[int, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for case in cases:
        parsed = parse_action(case.action_str)
        if "error" in parsed or int(parsed.get("st", -1)) not in (2, 3):
            records.append({**asdict(case), "passed": False, "skipped": "invalid_case"})
            continue
        baseline_ranges = baseline_solver_ranges_for_case(
            case,
            value_net=value_net,
            device=device,
            strategy_source=strategy_source,
        )
        response_range = response_villain_range_for_case(
            case,
            model=response_model,
            mean=response_mean,
            std=response_std,
            temperature=response_temperature,
            device=device,
        )
        if baseline_ranges.solver_hands != response_range.solver_hands:
            records.append({**asdict(case), "passed": False, "skipped": "hand_index_mismatch"})
            continue
        baseline = _solve_case_strategy(
            case,
            hero_range=baseline_ranges.hero_range,
            villain_range=baseline_ranges.villain_range,
            solver_iterations=solver_iterations,
            solver_backend=solver_backend,
            range_prune_threshold=range_prune_threshold,
        )
        response = _solve_case_strategy(
            case,
            hero_range=baseline_ranges.hero_range,
            villain_range=response_range.villain_range,
            solver_iterations=solver_iterations,
            solver_backend=solver_backend,
            range_prune_threshold=range_prune_threshold,
        )
        l1 = float(np.abs(baseline["strategy"] - response["strategy"]).sum())
        record = {
            **asdict(case),
            "street": int(parsed.get("st", -1)),
            "passed": bool(np.isfinite(l1)),
            "baseline_action": int(baseline["action"]),
            "response_action": int(response["action"]),
            "action_agreement": bool(int(baseline["action"]) == int(response["action"])),
            "action_l1_drift": l1,
            "baseline_strategy": baseline["strategy"].round(6).tolist(),
            "response_strategy": response["strategy"].round(6).tolist(),
            "baseline_latency_ms": round(float(baseline["latency_ms"]), 3),
            "response_latency_ms": round(float(response["latency_ms"]), 3),
            "baseline_solve_ms": round(float(baseline["solve_ms"]), 3),
            "response_solve_ms": round(float(response["solve_ms"]), 3),
            "baseline_solver_n_hands": int(baseline["solver_n_hands"]),
            "response_solver_n_hands": int(response["solver_n_hands"]),
            "n_opponent_updates": int(response_range.n_opponent_updates),
            **_range_summary("baseline_villain_range", baseline_ranges.villain_range),
            **_range_summary("response_villain_range", response_range.villain_range),
        }
        hand_index = _hand_index_from_case(case)
        if bot_hands_by_hand_index is not None and hand_index is not None:
            bot_hand = bot_hands_by_hand_index.get(hand_index)
            if bot_hand is not None:
                baseline_truth = true_hand_log_lift(
                    baseline_ranges.solver_hands,
                    baseline_ranges.villain_range,
                    bot_hand=bot_hand,
                )
                response_truth = true_hand_log_lift(
                    response_range.solver_hands,
                    response_range.villain_range,
                    bot_hand=bot_hand,
                )
                if baseline_truth is not None and response_truth is not None:
                    for key, value in baseline_truth.items():
                        record[f"baseline_{key}"] = value
                    for key, value in response_truth.items():
                        record[f"response_{key}"] = value
                    record["delta_true_hand_log_lift"] = float(
                        response_truth["log_lift_vs_uniform"]
                        - baseline_truth["log_lift_vs_uniform"]
                    )
        records.append(record)

    evaluated = [record for record in records if record.get("passed") and "action_l1_drift" in record]
    l1_values = [float(record["action_l1_drift"]) for record in evaluated]
    action_agreement = [bool(record["action_agreement"]) for record in evaluated]
    truth_deltas = [
        float(record["delta_true_hand_log_lift"])
        for record in evaluated
        if "delta_true_hand_log_lift" in record
    ]
    return {
        "mode": "slumbot_response_range_solver_ab",
        "passed": bool(evaluated and all(record.get("passed") for record in records)),
        "promotion_blockers": [
            "response_range_solver_ab_is_offline_diagnostic_not_slumbot_confidence",
        ],
        "solver_iterations": int(solver_iterations),
        "solver_backend": solver_backend,
        "range_prune_threshold": float(range_prune_threshold),
        "strategy_source": strategy_source,
        "n_cases": len(records),
        "n_evaluated": len(evaluated),
        "action_agreement_rate": float(np.mean(action_agreement)) if action_agreement else 0.0,
        "mean_action_l1_drift": float(np.mean(l1_values)) if l1_values else None,
        "max_action_l1_drift": float(np.max(l1_values)) if l1_values else None,
        "baseline_allin_rate": (
            float(np.mean([record["baseline_action"] == 8 for record in evaluated]))
            if evaluated
            else 0.0
        ),
        "response_allin_rate": (
            float(np.mean([record["response_action"] == 8 for record in evaluated]))
            if evaluated
            else 0.0
        ),
        "n_true_hand_scored": len(truth_deltas),
        "mean_delta_true_hand_log_lift": (
            float(np.mean(truth_deltas)) if truth_deltas else None
        ),
        "true_hand_response_beats_baseline_rate": (
            float(np.mean([delta > 0.0 for delta in truth_deltas])) if truth_deltas else None
        ),
        "records": records,
    }


def train_response_model_from_action_likelihood(
    action_likelihood_json: str | Path,
    *,
    holdout_fraction: float,
    hidden_dim: int,
    n_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray, float, dict[str, Any]]:
    model, mean, std, temperature, _holdout_hands, metrics = _train_response_model(
        action_likelihood_json,
        holdout_fraction=holdout_fraction,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        seed=seed,
    )
    return model, mean, std, float(temperature), metrics
