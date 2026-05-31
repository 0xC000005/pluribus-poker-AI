#!/usr/bin/env python3
"""Duplicate-swapped H2H smoke for exact search-as-actor decisions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.deep_cfr.fast_state import FastPokerState, N_ACTIONS  # noqa: E402
from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402
from poker_ai.research.decision_value_actor import (  # noqa: E402
    _policy_for_root,
    regularized_action_value_target,
    score_first_actions_for_public_state,
    select_decision_value_behavior_action,
)
from poker_ai.research.evaluation import (  # noqa: E402
    BIG_BLIND,
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)
from poker_ai.research.native_nfsp import resolve_device  # noqa: E402
from poker_ai.research.public_action_rollout_value import (  # noqa: E402
    _action_name,
    build_public_world_state,
    checkpoint_continuation_policy,
    select_deployed_root_action,
)


Policy = Callable[[FastPokerState, np.random.Generator], int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        default="average-policy",
    )
    parser.add_argument("--n-games", type=int, default=20)
    parser.add_argument("--n-worlds", type=int, default=4)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--small-blind", type=int, default=50)
    parser.add_argument("--big-blind", type=int, default=100)
    parser.add_argument("--max-steps-per-hand", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--scorer-continuation",
        choices=("deployed", "search-guided"),
        default="deployed",
        help="Continuation used inside the local first-action scorer.",
    )
    parser.add_argument(
        "--continuation-search-worlds",
        type=int,
        default=1,
        help="Inner worlds for one-step search-guided continuation.",
    )
    parser.add_argument("--record-decisions", action="store_true")
    parser.add_argument("--max-decision-records", type=int, default=512)
    parser.add_argument("--output-json")
    return parser


def _state_from_deck(
    deck: np.ndarray,
    *,
    initial_chips: int,
    small_blind: int,
    big_blind: int,
) -> FastPokerState:
    deck_tuple = tuple(int(card) for card in deck)
    return build_public_world_state(
        hero_cards=(deck_tuple[0], deck_tuple[1]),
        opponent_cards=(deck_tuple[2], deck_tuple[3]),
        deck_tail=deck_tuple[4:],
        initial_chips=initial_chips,
        small_blind=small_blind,
        big_blind=big_blind,
    )


def _play_hand(
    *,
    deck: np.ndarray,
    policies: tuple[Policy, Policy],
    rng: np.random.Generator,
    initial_chips: int,
    small_blind: int,
    big_blind: int,
    max_steps_per_hand: int,
) -> tuple[float, bool]:
    state = _state_from_deck(
        deck,
        initial_chips=initial_chips,
        small_blind=small_blind,
        big_blind=big_blind,
    )
    steps = 0
    while not state.is_terminal and steps < int(max_steps_per_hand):
        player = int(state.current_player_i)
        if not bool(state.active[player]) or int(state.chips[player]) <= 0:
            state.apply_action(None)
            steps += 1
            continue
        action = int(policies[player](state, rng))
        mask = state.get_legal_mask()
        if action < 0 or action >= N_ACTIONS or mask[action] <= 0:
            raise RuntimeError(f"policy selected illegal action {action}")
        state.apply_action(action)
        steps += 1
    truncated = not state.is_terminal
    if truncated:
        return float(state.chips[0] - int(initial_chips)), True
    return float(state.payout[0]), False


def _ci95(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / math.sqrt(values.size))


def _street_name(stage: int) -> str:
    names = {
        FastPokerState.PREFLOP: "preflop",
        FastPokerState.FLOP: "flop",
        FastPokerState.TURN: "turn",
        FastPokerState.RIVER: "river",
        FastPokerState.SHOWDOWN: "showdown",
        FastPokerState.TERMINAL: "terminal",
    }
    return names.get(int(stage), str(int(stage)))


def _finite_action_value(values: dict[int, float], action: int) -> float | None:
    value = values.get(int(action))
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _summarize_decisions(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "n_recorded": 0,
            "disagreement_count": 0,
            "disagreement_fraction": 0.0,
            "mean_local_search_minus_deployed_value": 0.0,
            "by_street": {},
            "action_pair_counts": {},
        }

    disagreements = [
        record
        for record in records
        if record["search_action"] != record["deployed_action"]
    ]
    finite_improvements = [
        float(record["local_search_minus_deployed_value"])
        for record in records
        if record.get("local_search_minus_deployed_value") is not None
    ]
    by_street_values: dict[str, list[float]] = defaultdict(list)
    by_street_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    for record in records:
        street = str(record["street"])
        by_street_counts[street] += 1
        pair_counts[f"{record['deployed_action']}->{record['search_action']}"] += 1
        improvement = record.get("local_search_minus_deployed_value")
        if improvement is not None:
            by_street_values[street].append(float(improvement))

    return {
        "n_recorded": int(len(records)),
        "disagreement_count": int(len(disagreements)),
        "disagreement_fraction": float(len(disagreements) / len(records)),
        "mean_local_search_minus_deployed_value": _mean(finite_improvements),
        "by_street": {
            street: {
                "count": int(count),
                "mean_local_search_minus_deployed_value": _mean(
                    by_street_values.get(street, [])
                ),
            }
            for street, count in sorted(by_street_counts.items())
        },
        "action_pair_counts": {
            key: int(value) for key, value in sorted(pair_counts.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device_info = resolve_device(args.device)
    device = torch.device(device_info["resolved_device"])
    loaded = load_value_network_checkpoint(args.checkpoint, device)
    assert_strategy_source_supported(loaded, args.strategy_source)
    continuation = checkpoint_continuation_policy(
        loaded,
        device,
        strategy_source=args.strategy_source,
        greedy=True,
    )
    search_latencies_ms: list[float] = []
    continuation_search_latencies_ms: list[float] = []
    search_truncated_rollouts = 0
    continuation_search_truncated_rollouts = 0
    decision_counter = 0
    continuation_search_decision_counter = 0
    decision_records: list[dict[str, Any]] = []
    hand_context: dict[str, Any] = {"duplicate_pair": -1, "seat_role": "unset"}

    def deployed_policy(state: FastPokerState, _rng: np.random.Generator) -> int:
        advantages, strategy = _policy_for_root(
            loaded,
            state,
            device,
            strategy_source=args.strategy_source,
        )
        return select_deployed_root_action(
            advantages=advantages,
            strategy=strategy,
            legal_mask=state.get_legal_mask(),
            strategy_source=args.strategy_source,
            greedy=True,
        )

    def search_guided_continuation_policy(
        state: FastPokerState,
        rng: np.random.Generator,
    ) -> int:
        nonlocal continuation_search_decision_counter
        nonlocal continuation_search_truncated_rollouts
        started = time.perf_counter()
        continuation_search_decision_counter += 1
        _advantages, prior = _policy_for_root(
            loaded,
            state,
            device,
            strategy_source=args.strategy_source,
        )
        legal_mask = state.get_legal_mask().astype(np.float32, copy=False)
        result = score_first_actions_for_public_state(
            state,
            n_worlds=max(1, int(args.continuation_search_worlds)),
            continuation_policy=continuation,
            seed=(
                int(args.seed)
                + 7000001 * int(decision_counter)
                + 1009 * int(continuation_search_decision_counter)
            ),
            max_steps_per_hand=args.max_steps_per_hand,
        )
        action_values = np.full(N_ACTIONS, np.nan, dtype=np.float32)
        for action_idx, value in result.action_values.items():
            action_values[int(action_idx)] = float(value)
        target = regularized_action_value_target(
            prior,
            legal_mask,
            action_values,
            eta=args.eta,
        )
        action = select_decision_value_behavior_action(
            prior,
            target,
            legal_mask,
            rng,
            behavior_policy="search-improved",
            greedy=True,
        )
        continuation_search_truncated_rollouts += int(result.n_truncated_rollouts)
        continuation_search_latencies_ms.append(
            float((time.perf_counter() - started) * 1000.0)
        )
        return int(action)

    def search_policy(state: FastPokerState, rng: np.random.Generator) -> int:
        nonlocal decision_counter, search_truncated_rollouts
        started = time.perf_counter()
        decision_counter += 1
        advantages, prior = _policy_for_root(
            loaded,
            state,
            device,
            strategy_source=args.strategy_source,
        )
        legal_mask = state.get_legal_mask().astype(np.float32, copy=False)
        deployed_action = select_deployed_root_action(
            advantages=advantages,
            strategy=prior,
            legal_mask=legal_mask,
            strategy_source=args.strategy_source,
            greedy=True,
        )
        scorer_continuation: Policy = continuation
        if args.scorer_continuation == "search-guided":
            scorer_continuation = search_guided_continuation_policy
        result = score_first_actions_for_public_state(
            state,
            n_worlds=args.n_worlds,
            continuation_policy=scorer_continuation,
            seed=int(args.seed) + 1000003 * decision_counter,
            max_steps_per_hand=args.max_steps_per_hand,
        )
        action_values = np.full(N_ACTIONS, np.nan, dtype=np.float32)
        for action, value in result.action_values.items():
            action_values[int(action)] = float(value)
        target = regularized_action_value_target(
            prior,
            legal_mask,
            action_values,
            eta=args.eta,
        )
        action = select_decision_value_behavior_action(
            prior,
            target,
            legal_mask,
            rng,
            behavior_policy="search-improved",
            greedy=True,
        )
        if args.record_decisions and len(decision_records) < int(args.max_decision_records):
            search_value = _finite_action_value(result.action_values, action)
            deployed_value = _finite_action_value(result.action_values, deployed_action)
            oracle_value = float(result.best_action_value)
            local_delta = (
                None
                if search_value is None or deployed_value is None
                else float(search_value - deployed_value)
            )
            decision_records.append(
                {
                    "decision_index": int(decision_counter),
                    "duplicate_pair": int(hand_context["duplicate_pair"]),
                    "seat_role": str(hand_context["seat_role"]),
                    "player": int(state.current_player_i),
                    "street": _street_name(int(state.stage)),
                    "pot_total": float(state.pot_total),
                    "to_call": float(
                        max(float(np.max(state.bets)) - float(state.bets[state.current_player_i]), 0.0)
                    ),
                    "legal_actions": [
                        _action_name(int(idx)) for idx in np.flatnonzero(legal_mask > 0)
                    ],
                    "deployed_action": _action_name(int(deployed_action)),
                    "search_action": _action_name(int(action)),
                    "oracle_action": _action_name(int(result.best_action)),
                    "deployed_action_value": deployed_value,
                    "search_action_value": search_value,
                    "oracle_action_value": oracle_value,
                    "local_search_minus_deployed_value": local_delta,
                    "scorer_continuation": str(args.scorer_continuation),
                    "search_oracle_gap": (
                        None if search_value is None else float(oracle_value - search_value)
                    ),
                    "n_worlds": int(result.n_worlds),
                    "n_truncated_rollouts": int(result.n_truncated_rollouts),
                }
            )
        search_truncated_rollouts += int(result.n_truncated_rollouts)
        search_latencies_ms.append(float((time.perf_counter() - started) * 1000.0))
        return int(action)

    rng = np.random.default_rng(int(args.seed))
    paired: list[float] = []
    candidate_seat0: list[float] = []
    baseline_seat0: list[float] = []
    hand_truncations = 0
    for pair_idx in range(int(args.n_games)):
        deck = rng.permutation(52).astype(np.int8)
        hand_context["duplicate_pair"] = int(pair_idx)
        hand_context["seat_role"] = "candidate_seat0"
        cand, cand_truncated = _play_hand(
            deck=deck,
            policies=(search_policy, deployed_policy),
            rng=rng,
            initial_chips=args.initial_chips,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            max_steps_per_hand=args.max_steps_per_hand,
        )
        hand_context["seat_role"] = "candidate_seat1"
        base, base_truncated = _play_hand(
            deck=deck,
            policies=(deployed_policy, search_policy),
            rng=rng,
            initial_chips=args.initial_chips,
            small_blind=args.small_blind,
            big_blind=args.big_blind,
            max_steps_per_hand=args.max_steps_per_hand,
        )
        candidate_seat0.append(float(cand))
        baseline_seat0.append(float(base))
        paired.append(float((cand - base) / 2.0))
        hand_truncations += int(cand_truncated) + int(base_truncated)

    paired_arr = np.asarray(paired, dtype=np.float64)
    latency_arr = np.asarray(search_latencies_ms, dtype=np.float64)
    continuation_latency_arr = np.asarray(
        continuation_search_latencies_ms,
        dtype=np.float64,
    )
    ci95 = _ci95(paired_arr)
    avg = float(paired_arr.mean()) if paired_arr.size else 0.0
    metrics = {
        "mode": "decision_value_search_actor_h2h",
        "passed": bool(
            paired_arr.size == int(args.n_games)
            and np.isfinite(paired_arr).all()
            and hand_truncations == 0
            and search_truncated_rollouts == 0
            and continuation_search_truncated_rollouts == 0
        ),
        "promotable": False,
        "promotion_blockers": [
            "local_h2h_smoke_requires_larger_confidence_gate",
            "slumbot_confirmation_required",
        ],
        "checkpoint": args.checkpoint,
        "strategy_source": args.strategy_source,
        "scorer_continuation": str(args.scorer_continuation),
        "n_games": int(args.n_games * 2),
        "n_duplicate_pairs": int(args.n_games),
        "n_worlds": int(args.n_worlds),
        "continuation_search_worlds": int(args.continuation_search_worlds),
        "initial_chips": int(args.initial_chips),
        "seed": int(args.seed),
        "avg_chips_per_hand": avg,
        "ci95_chips_per_hand": ci95,
        "paired_delta_lower95_chips_per_hand": avg - ci95,
        "mbb_per_hand": avg / BIG_BLIND * 1000.0,
        "ci95_mbb_per_hand": ci95 / BIG_BLIND * 1000.0,
        "candidate_seat0_payouts": candidate_seat0,
        "baseline_seat0_payouts": baseline_seat0,
        "paired_deltas": paired,
        "hand_truncations": int(hand_truncations),
        "search_truncated_rollouts": int(search_truncated_rollouts),
        "continuation_search_truncated_rollouts": int(
            continuation_search_truncated_rollouts
        ),
        "n_search_decisions": int(latency_arr.size),
        "mean_search_decision_ms": float(latency_arr.mean()) if latency_arr.size else 0.0,
        "p95_search_decision_ms": (
            float(np.percentile(latency_arr, 95)) if latency_arr.size else 0.0
        ),
        "max_search_decision_ms": float(latency_arr.max()) if latency_arr.size else 0.0,
        "n_continuation_search_decisions": int(continuation_latency_arr.size),
        "mean_continuation_search_decision_ms": (
            float(continuation_latency_arr.mean())
            if continuation_latency_arr.size
            else 0.0
        ),
        "p95_continuation_search_decision_ms": (
            float(np.percentile(continuation_latency_arr, 95))
            if continuation_latency_arr.size
            else 0.0
        ),
        "max_continuation_search_decision_ms": (
            float(continuation_latency_arr.max())
            if continuation_latency_arr.size
            else 0.0
        ),
        "device": device_info,
    }
    if args.record_decisions:
        metrics["decision_summary"] = _summarize_decisions(decision_records)
        metrics["decision_records"] = decision_records
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
