"""Offline A/B for learned Slumbot opponent-response range updates."""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from poker_ai.research.evaluation import (
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)
from poker_ai.research.slumbot_opponent_response_probe import (
    _ProbeNet,
    _fit_temperature,
    _masked_cross_entropy,
    _predict_logits,
    _probs_from_logits,
    _split_by_hand,
    _standardize_pair,
    load_opponent_response_dataset,
)
from poker_ai.research.slumbot_trace_range_truth import (
    _aggregate,
    _decision_action_str,
    _last_decisions_by_hand,
    _read_trace,
    _score_true_hand,
    _street_name,
)


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import card_str_to_index  # noqa: E402
from range_tracker import (  # noqa: E402
    N_ACTIONS,
    _build_features_batch,
    _get_legal_mask,
    _parse_action,
    map_slumbot_action_to_idx,
    walk_actions,
)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _train_response_model(
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
) -> tuple[Any, np.ndarray, np.ndarray, float, set[int], dict[str, Any]]:
    dataset = load_opponent_response_dataset(action_likelihood_json)
    train_mask, holdout_mask = _split_by_hand(dataset.hand_indices, holdout_fraction)
    x_train, x_holdout, mean, std = _standardize_pair(
        dataset.features[train_mask],
        dataset.features[holdout_mask],
    )
    x_all = np.zeros_like(dataset.features, dtype=np.float32)
    x_all[train_mask] = x_train
    x_all[holdout_mask] = x_holdout

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _ProbeNet(dataset.features.shape[1], hidden_dim, n_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_t = torch.from_numpy(x_all).to(device)
    legal_t = torch.from_numpy(dataset.legal_masks).to(device)
    target_t = torch.from_numpy(dataset.target_probs).to(device)
    train_idx = np.flatnonzero(train_mask)

    for _ in range(max(1, int(epochs))):
        perm = np.random.permutation(train_idx)
        for start in range(0, len(perm), max(1, int(batch_size))):
            batch = perm[start : start + max(1, int(batch_size))]
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_cross_entropy(model(x_t[batch]), legal_t[batch], target_t[batch])
            loss.backward()
            optimizer.step()

    logits = _predict_logits(model, x_all, device)
    temperature = _fit_temperature(
        logits,
        dataset.legal_masks,
        dataset.target_probs,
        train_mask,
        device=device,
    )
    holdout_hands = set(int(x) for x in dataset.hand_indices[holdout_mask])
    return (
        model,
        mean,
        std,
        temperature,
        holdout_hands,
        {
            "n_records": int(dataset.features.shape[0]),
            "n_hands": int(len(set(int(x) for x in dataset.hand_indices))),
            "n_train_records": int(train_mask.sum()),
            "n_holdout_records": int(holdout_mask.sum()),
            "n_holdout_hands": int(len(holdout_hands)),
        },
    )


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
    blocked_cards: list[int],
) -> None:
    blocked = set(blocked_cards)
    if not blocked:
        return
    for idx, hand in enumerate(hands):
        if hand[0] in blocked or hand[1] in blocked:
            ranges[idx] = 0.0
    total = float(ranges.sum())
    if total > 0.0:
        ranges /= total


def _score_response_true_hand(
    *,
    model: Any,
    mean: np.ndarray,
    std: np.ndarray,
    temperature: float,
    device: torch.device,
    decision: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any] | None:
    bot_cards = list(result.get("bot_hole_cards") or [])
    hero_cards = list(decision.get("hole_cards") or [])
    board = list(decision.get("board") or [])
    if len(bot_cards) != 2 or len(hero_cards) != 2:
        return None
    visible = [*bot_cards, *hero_cards, *board]
    if len(visible) != len(set(visible)):
        return None
    try:
        client_pos = int(decision["client_pos"])
        hand_index = int(decision["hand_index"])
    except (KeyError, TypeError, ValueError):
        return None

    action_str = _decision_action_str(decision)
    hero_idx = [card_str_to_index(card) for card in hero_cards]
    board_idx = [card_str_to_index(card) for card in board]
    bot_hand = tuple(sorted(card_str_to_index(card) for card in bot_cards))
    opponent_cards = sorted(set(range(52)) - set(hero_idx))
    opponent_hands = list(itertools.combinations(opponent_cards, 2))
    hand_to_idx = {hand: idx for idx, hand in enumerate(opponent_hands)}
    true_idx = hand_to_idx.get(bot_hand)
    if true_idx is None:
        return None
    ranges = np.ones(len(opponent_hands), dtype=np.float64) / float(len(opponent_hands))
    revealed = 0
    opp_pos = 1 - client_pos

    for before, acting_pos, action_char, bet_to in walk_actions(action_str):
        current_board = _visible_board_for(before, board_idx)
        if len(current_board) > revealed:
            _zero_blocked_hands(ranges, opponent_hands, current_board[revealed:])
            revealed = len(current_board)
        if acting_pos != opp_pos:
            continue
        parsed_before = _parse_action(before)
        if "error" in parsed_before:
            continue
        action_data = map_slumbot_action_to_idx(
            action_char,
            bet_to,
            before,
            acting_pos,
            parsed_before,
        )
        features = _build_features_batch(
            opponent_hands,
            current_board,
            before,
            opp_pos,
            parsed_before,
        )
        x = ((features.astype(np.float32) - mean) / std).astype(np.float32)
        logits = _predict_logits(model, x, device)
        legal = np.zeros((len(opponent_hands), N_ACTIONS), dtype=np.float32)
        legal_mask = _get_legal_mask(parsed_before, before, opp_pos).astype(np.float32)
        legal[:] = legal_mask
        probs = _probs_from_logits(logits, legal, temperature=temperature)
        likelihood = np.zeros(len(opponent_hands), dtype=np.float64)
        for idx, weight in action_data:
            likelihood += probs[:, int(idx)] * float(weight)
        eps = 0.01
        likelihood = (1.0 - eps) * likelihood + eps * (1.0 / N_ACTIONS)
        ranges *= likelihood
        total = float(ranges.sum())
        if total > 0.0:
            ranges /= total

    final_board = _visible_board_for(action_str, board_idx)
    if len(final_board) > revealed:
        _zero_blocked_hands(ranges, opponent_hands, final_board[revealed:])

    positive = ranges[ranges > 0]
    if positive.size <= 0:
        return None
    true_prob = float(ranges[true_idx])
    uniform_prob = 1.0 / float(positive.size)
    eps = 1e-12
    log_lift = math.log(max(true_prob, eps)) - math.log(max(uniform_prob, eps))
    return {
        "hand_index": hand_index,
        "street": _street_name(decision),
        "street_index": decision.get("street_index"),
        "action_str": action_str,
        "client_pos": client_pos,
        "hole_cards": hero_cards,
        "bot_hole_cards": bot_cards,
        "board": board,
        "winnings": int(result.get("winnings", 0)),
        "true_hand_prob": true_prob,
        "uniform_prob": uniform_prob,
        "log_lift_vs_uniform": float(log_lift),
        "true_hand_percentile": float(np.mean(positive <= true_prob)),
        "true_hand_rank": int(1 + np.sum(positive > true_prob)),
        "range_support": int(positive.size),
        "range_top1_mass": float(np.max(positive)),
        "range_top10_mass": float(np.sort(positive)[::-1][:10].sum()),
    }


def _prefix_records(records: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [
        {
            f"{prefix}_{key}": value
            for key, value in record.items()
            if key
            in {
                "true_hand_prob",
                "uniform_prob",
                "log_lift_vs_uniform",
                "true_hand_percentile",
                "true_hand_rank",
                "range_support",
                "range_top1_mass",
                "range_top10_mass",
            }
        }
        for record in records
    ]


def evaluate_opponent_response_range_ab(
    checkpoint: str | Path,
    trace_path: str | Path,
    action_likelihood_json: str | Path,
    *,
    strategy_source: str = "regret",
    holdout_fraction: float = 0.3,
    hidden_dim: int = 128,
    n_layers: int = 2,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | torch.device = "auto",
    seed: int = 20260515,
    eval_split: str = "hand-heldout",
) -> dict[str, Any]:
    """Train a response probe and A/B its range updates on held-out trace hands."""
    if eval_split not in {"hand-heldout", "all"}:
        raise ValueError("eval_split must be 'hand-heldout' or 'all'")
    resolved_device = _resolve_device(device)
    (
        response_model,
        mean,
        std,
        temperature,
        holdout_hands,
        data_metrics,
    ) = _train_response_model(
        action_likelihood_json,
        holdout_fraction=holdout_fraction,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=resolved_device,
        seed=seed,
    )
    loaded = load_value_network_checkpoint(checkpoint, resolved_device)
    assert_strategy_source_supported(loaded, strategy_source)
    trace_records = _read_trace(trace_path)
    last_decisions = _last_decisions_by_hand(trace_records)
    baseline_records: list[dict[str, Any]] = []
    response_records: list[dict[str, Any]] = []
    paired_records: list[dict[str, Any]] = []
    skipped_results = 0
    eval_hands_seen: set[int] = set()

    for record in trace_records:
        if record.get("event") != "hand_result":
            continue
        try:
            hand_index = int(record["hand_index"])
        except (KeyError, TypeError, ValueError):
            skipped_results += 1
            continue
        if eval_split == "hand-heldout" and hand_index not in holdout_hands:
            continue
        eval_hands_seen.add(hand_index)
        decision = last_decisions.get(hand_index)
        if decision is None:
            skipped_results += 1
            continue
        baseline = _score_true_hand(
            value_net=loaded.value_net,
            device=resolved_device,
            strategy_source=strategy_source,
            decision=decision,
            result=record,
        )
        response = _score_response_true_hand(
            model=response_model,
            mean=mean,
            std=std,
            temperature=temperature,
            device=resolved_device,
            decision=decision,
            result=record,
        )
        if baseline is None or response is None:
            skipped_results += 1
            continue
        baseline_records.append(baseline)
        response_records.append(response)
        pair = {
            "hand_index": hand_index,
            "street": response["street"],
            "winnings": response["winnings"],
            "delta_log_lift": float(
                response["log_lift_vs_uniform"] - baseline["log_lift_vs_uniform"]
            ),
        }
        pair.update(_prefix_records([baseline], "baseline")[0])
        pair.update(_prefix_records([response], "response")[0])
        paired_records.append(pair)

    baseline_summary = _aggregate(baseline_records)
    response_summary = _aggregate(response_records)
    delta = float(
        response_summary["mean_log_lift_vs_uniform"]
        - baseline_summary["mean_log_lift_vs_uniform"]
    )
    return {
        "mode": "slumbot_opponent_response_range_ab",
        "checkpoint": str(checkpoint),
        "trace": str(trace_path),
        "action_likelihood": str(action_likelihood_json),
        "strategy_source": strategy_source,
        "device": str(resolved_device),
        "holdout_fraction": float(holdout_fraction),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "temperature": float(temperature),
        "eval_split": eval_split,
        "n_eval_hands": int(len(eval_hands_seen)),
        "n_holdout_hands": int(len(holdout_hands)),
        "n_skipped_results": skipped_results,
        **data_metrics,
        "baseline": baseline_summary,
        "response": response_summary,
        "delta_mean_log_lift": delta,
        "response_beats_baseline": bool(delta > 0.0),
        "by_street": {
            "baseline": {
                street: _aggregate([r for r in baseline_records if r["street"] == street])
                for street in ("preflop", "flop", "turn", "river", "unknown")
                if any(r["street"] == street for r in baseline_records)
            },
            "response": {
                street: _aggregate([r for r in response_records if r["street"] == street])
                for street in ("preflop", "flop", "turn", "river", "unknown")
                if any(r["street"] == street for r in response_records)
            },
        },
        "records": paired_records,
        "passed": bool(paired_records),
    }


def write_metrics(metrics: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
