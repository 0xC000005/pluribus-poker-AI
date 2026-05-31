"""Exact restricted response-oracle gates for local population improvement.

This diagnostic tests whether an exact sampled response actor can beat the
current local incumbent/population before training another neural imitator.
It is intentionally promotion-blocked: passing this gate is evidence to build
a response oracle, not evidence to deploy the oracle directly.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch

from poker_ai.deep_cfr.fast_state import FastPokerState, N_ACTIONS
from poker_ai.research.decision_value_actor import score_first_actions_for_public_state
from poker_ai.research.mixed_policy_h2h import PolicyAdapter, load_policy_adapter
from poker_ai.research.native_nfsp import resolve_device
from poker_ai.research.public_action_rollout_value import (
    _action_name,
    build_public_world_state,
)


Policy = Callable[[FastPokerState, np.random.Generator], int]


@dataclass(frozen=True)
class PolicySpec:
    kind: str
    checkpoint: str
    weight: float = 1.0


@dataclass(frozen=True)
class ExactResponseH2HConfig:
    baseline_checkpoint: str
    baseline_kind: str = "tianshou-rainbow"
    continuation_specs: tuple[PolicySpec, ...] = ()
    n_games: int = 100
    n_worlds: int = 4
    initial_chips: int = 1000
    small_blind: int = 50
    big_blind: int = 100
    max_steps_per_hand: int = 128
    seed: int = 20260527
    device: str = "auto"
    greedy_baseline: bool = True
    greedy_continuation: bool = True
    max_decision_records: int = 512
    min_games: int = 1
    min_lower95_response_payoff: float = 0.0


def _legal_uniform(legal_mask: np.ndarray) -> np.ndarray:
    legal = np.asarray(legal_mask, dtype=np.float64) > 0
    if not np.any(legal):
        raise ValueError("legal_mask must contain at least one legal action")
    probs = np.zeros(legal.shape[0], dtype=np.float64)
    probs[legal] = 1.0 / float(np.count_nonzero(legal))
    return probs


def mixture_policy_probs(
    adapters: Sequence[Any],
    features: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device | None,
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Return the legal normalized weighted average policy over adapters."""
    if not adapters:
        raise ValueError("adapters must not be empty")
    legal = np.asarray(legal_mask, dtype=np.float64) > 0
    if not np.any(legal):
        raise ValueError("legal_mask must contain at least one legal action")
    if weights is None:
        normalized_weights = np.full(len(adapters), 1.0 / float(len(adapters)))
    else:
        normalized_weights = np.asarray(weights, dtype=np.float64)
        if normalized_weights.shape != (len(adapters),):
            raise ValueError("weights must match adapters")
        if not np.all(np.isfinite(normalized_weights)) or float(normalized_weights.sum()) <= 0.0:
            raise ValueError("weights must be finite and have positive mass")
        normalized_weights = normalized_weights / float(normalized_weights.sum())

    mixed = np.zeros(legal.shape[0], dtype=np.float64)
    for weight, adapter in zip(normalized_weights, adapters, strict=True):
        probs = np.asarray(adapter.probs(features, legal_mask, device), dtype=np.float64)
        if probs.shape != mixed.shape:
            raise ValueError("adapter probability shape must match legal_mask")
        probs = np.where(legal, np.clip(probs, 0.0, None), 0.0)
        total = float(probs.sum())
        if total <= 1e-12 or not np.isfinite(total):
            probs = _legal_uniform(legal_mask)
        else:
            probs = probs / total
        mixed += float(weight) * probs

    mixed = np.where(legal, np.clip(mixed, 0.0, None), 0.0)
    total = float(mixed.sum())
    if total <= 1e-12 or not np.isfinite(total):
        return _legal_uniform(legal_mask).astype(np.float32)
    return (mixed / total).astype(np.float32)


def _select_policy_action(
    adapters: Sequence[PolicyAdapter],
    state: FastPokerState,
    *,
    device: torch.device,
    rng: np.random.Generator,
    greedy: bool,
    weights: Sequence[float] | None = None,
) -> int:
    legal_mask = state.get_legal_mask().astype(np.float32, copy=False)
    features = state.to_feature_vector().astype(np.float32, copy=False)
    probs = mixture_policy_probs(adapters, features, legal_mask, device, weights=weights)
    legal = np.flatnonzero(legal_mask > 0)
    if greedy:
        return int(legal[int(np.argmax(probs[legal]))])
    legal_probs = probs[legal].astype(np.float64, copy=False)
    total = float(legal_probs.sum())
    if total <= 0.0:
        legal_probs = np.ones_like(legal_probs) / float(legal_probs.size)
    else:
        legal_probs = legal_probs / total
    return int(rng.choice(legal, p=legal_probs))


def make_mixture_continuation_policy(
    adapters: Sequence[PolicyAdapter],
    *,
    device: torch.device,
    greedy: bool,
    weights: Sequence[float] | None = None,
) -> Policy:
    def _policy(state: FastPokerState, rng: np.random.Generator) -> int:
        return _select_policy_action(
            adapters,
            state,
            device=device,
            rng=rng,
            greedy=greedy,
            weights=weights,
        )

    return _policy


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return 0.0
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _ci(values: np.ndarray) -> tuple[float, float, float, float, float]:
    if values.size == 0 or not np.all(np.isfinite(values)):
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    se = std / float(math.sqrt(values.size)) if values.size > 1 else 0.0
    return mean, std, se, float(mean - 1.96 * se), float(mean + 1.96 * se)


def _summarize_decisions(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "n_recorded": 0,
            "disagreement_count": 0,
            "disagreement_fraction": 0.0,
            "mean_local_response_minus_baseline_value": 0.0,
            "by_street": {},
            "action_pair_counts": {},
        }
    disagreements = [
        record
        for record in records
        if str(record.get("baseline_action")) != str(record.get("response_action"))
    ]
    improvements = [
        float(record["local_response_minus_baseline_value"])
        for record in records
        if record.get("local_response_minus_baseline_value") is not None
        and np.isfinite(float(record["local_response_minus_baseline_value"]))
    ]
    by_street_counts: Counter[str] = Counter()
    by_street_values: dict[str, list[float]] = defaultdict(list)
    pair_counts: Counter[str] = Counter()
    for record in records:
        street = str(record.get("street", "unknown"))
        baseline = str(record.get("baseline_action", "unknown"))
        response = str(record.get("response_action", "unknown"))
        by_street_counts[street] += 1
        pair_counts[f"{baseline}->{response}"] += 1
        improvement = record.get("local_response_minus_baseline_value")
        if improvement is not None and np.isfinite(float(improvement)):
            by_street_values[street].append(float(improvement))

    return {
        "n_recorded": int(len(records)),
        "disagreement_count": int(len(disagreements)),
        "disagreement_fraction": float(len(disagreements) / len(records)),
        "mean_local_response_minus_baseline_value": _mean(improvements),
        "by_street": {
            street: {
                "count": int(count),
                "mean_local_response_minus_baseline_value": _mean(
                    by_street_values.get(street, [])
                ),
            }
            for street, count in sorted(by_street_counts.items())
        },
        "action_pair_counts": {
            key: int(value) for key, value in sorted(pair_counts.items())
        },
    }


def summarize_exact_response_h2h(
    response_payoffs: Sequence[float],
    decision_records: Sequence[dict[str, Any]],
    *,
    min_games: int = 1,
    min_lower95_response_payoff: float = 0.0,
) -> dict[str, Any]:
    """Summarize exact response H2H evidence and apply conservative blockers."""
    payoffs = np.asarray([float(value) for value in response_payoffs], dtype=np.float64)
    mean, std, se, lower95, upper95 = _ci(payoffs)
    blockers: list[str] = []
    if payoffs.size < int(min_games):
        blockers.append("insufficient_games")
    if payoffs.size == 0 or not np.all(np.isfinite(payoffs)):
        blockers.append("nonfinite_payoffs")
    n_truncated_rollouts = int(
        sum(int(record.get("n_truncated_rollouts", 0)) for record in decision_records)
    )
    if n_truncated_rollouts > 0:
        blockers.append("truncated_rollouts")
    if lower95 < float(min_lower95_response_payoff):
        blockers.append("lower95_below_threshold")
    return {
        "mode": "exact_restricted_response_h2h",
        "passed": bool(not blockers),
        "promotion": False,
        "promotable": False,
        "promotion_blockers": [
            "diagnostic_exact_response_not_deployable_policy",
            "full_population_equilibrium_gate_required",
            "slumbot_heldout_confirmation_required",
        ],
        "blockers": blockers,
        "n_games": int(payoffs.size),
        "mean_response_payoff": round(mean, 12),
        "std_response_payoff": round(std, 12),
        "se_response_payoff": round(se, 12),
        "lower95_response_payoff": round(lower95, 12),
        "upper95_response_payoff": round(upper95, 12),
        "min_games": int(min_games),
        "min_lower95_response_payoff": float(min_lower95_response_payoff),
        "n_truncated_rollouts": n_truncated_rollouts,
        "decision_summary": _summarize_decisions(decision_records),
    }


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
    if not state.is_terminal:
        return float(state.chips[0] - int(initial_chips)), True
    return float(state.payout[0]), False


def _finite_action_value(values: dict[int, float], action: int) -> float | None:
    value = values.get(int(action))
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def _load_adapters(
    specs: Sequence[PolicySpec],
    *,
    device: torch.device,
) -> tuple[list[PolicyAdapter], list[float]]:
    adapters: list[PolicyAdapter] = []
    weights: list[float] = []
    for spec in specs:
        adapters.append(
            load_policy_adapter(
                spec.checkpoint,
                kind=spec.kind,
                device=device,
            )
        )
        weights.append(float(spec.weight))
    return adapters, weights


def evaluate_exact_response_h2h(cfg: ExactResponseH2HConfig) -> dict[str, Any]:
    """Run duplicate-swapped H2H for exact sampled response vs baseline."""
    started = time.perf_counter()
    device_info = resolve_device(cfg.device)
    device = torch.device(device_info["resolved_device"])
    baseline_adapter = load_policy_adapter(
        cfg.baseline_checkpoint,
        kind=cfg.baseline_kind,
        device=device,
    )
    continuation_specs = cfg.continuation_specs or (
        PolicySpec(cfg.baseline_kind, cfg.baseline_checkpoint, 1.0),
    )
    continuation_adapters, continuation_weights = _load_adapters(
        continuation_specs,
        device=device,
    )
    continuation_policy = make_mixture_continuation_policy(
        continuation_adapters,
        device=device,
        greedy=cfg.greedy_continuation,
        weights=continuation_weights,
    )
    response_payoffs: list[float] = []
    candidate_seat0_payouts: list[float] = []
    baseline_seat0_payouts: list[float] = []
    decision_records: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    hand_truncations = 0
    response_decision_counter = 0
    search_truncated_rollouts = 0
    context: dict[str, Any] = {"duplicate_pair": -1, "seat_role": "unset"}

    def baseline_policy(state: FastPokerState, rng: np.random.Generator) -> int:
        return _select_policy_action(
            [baseline_adapter],
            state,
            device=device,
            rng=rng,
            greedy=cfg.greedy_baseline,
        )

    def exact_response_policy(state: FastPokerState, rng: np.random.Generator) -> int:
        nonlocal response_decision_counter, search_truncated_rollouts
        response_decision_counter += 1
        decision_started = time.perf_counter()
        legal_mask = state.get_legal_mask().astype(np.float32, copy=False)
        baseline_action = baseline_policy(state, rng)
        result = score_first_actions_for_public_state(
            state,
            n_worlds=cfg.n_worlds,
            continuation_policy=continuation_policy,
            seed=int(cfg.seed) + 1_000_003 * response_decision_counter,
            max_steps_per_hand=cfg.max_steps_per_hand,
        )
        action = int(result.best_action)
        response_value = _finite_action_value(result.action_values, action)
        baseline_value = _finite_action_value(result.action_values, baseline_action)
        local_delta = (
            None
            if response_value is None or baseline_value is None
            else float(response_value - baseline_value)
        )
        if len(decision_records) < int(cfg.max_decision_records):
            decision_records.append(
                {
                    "decision_index": int(response_decision_counter),
                    "duplicate_pair": int(context["duplicate_pair"]),
                    "seat_role": str(context["seat_role"]),
                    "player": int(state.current_player_i),
                    "street": _street_name(int(state.stage)),
                    "pot_total": float(state.pot_total),
                    "to_call": float(
                        max(
                            float(np.max(state.bets))
                            - float(state.bets[state.current_player_i]),
                            0.0,
                        )
                    ),
                    "legal_actions": [
                        _action_name(int(idx)) for idx in np.flatnonzero(legal_mask > 0)
                    ],
                    "baseline_action": _action_name(int(baseline_action)),
                    "response_action": _action_name(int(action)),
                    "baseline_action_value": baseline_value,
                    "response_action_value": response_value,
                    "local_response_minus_baseline_value": local_delta,
                    "n_worlds": int(result.n_worlds),
                    "n_truncated_rollouts": int(result.n_truncated_rollouts),
                }
            )
        search_truncated_rollouts += int(result.n_truncated_rollouts)
        latencies_ms.append(float((time.perf_counter() - decision_started) * 1000.0))
        return action

    rng = np.random.default_rng(int(cfg.seed))
    for pair_idx in range(int(cfg.n_games)):
        deck = rng.permutation(52).astype(np.int8)
        context["duplicate_pair"] = int(pair_idx)
        context["seat_role"] = "response_seat0"
        response_seat0, truncated0 = _play_hand(
            deck=deck,
            policies=(exact_response_policy, baseline_policy),
            rng=rng,
            initial_chips=cfg.initial_chips,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
            max_steps_per_hand=cfg.max_steps_per_hand,
        )
        context["seat_role"] = "response_seat1"
        baseline_seat0, truncated1 = _play_hand(
            deck=deck,
            policies=(baseline_policy, exact_response_policy),
            rng=rng,
            initial_chips=cfg.initial_chips,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
            max_steps_per_hand=cfg.max_steps_per_hand,
        )
        candidate_seat0_payouts.append(float(response_seat0))
        baseline_seat0_payouts.append(float(baseline_seat0))
        response_payoffs.append(
            float((response_seat0 - baseline_seat0) / 2.0 / float(cfg.initial_chips))
        )
        hand_truncations += int(truncated0) + int(truncated1)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    summary = summarize_exact_response_h2h(
        response_payoffs,
        decision_records,
        min_games=cfg.min_games,
        min_lower95_response_payoff=cfg.min_lower95_response_payoff,
    )
    blockers = list(summary["blockers"])
    if hand_truncations > 0 and "hand_truncations" not in blockers:
        blockers.append("hand_truncations")
    if search_truncated_rollouts > 0 and "truncated_rollouts" not in blockers:
        blockers.append("truncated_rollouts")
    latency_arr = np.asarray(latencies_ms, dtype=np.float64)
    summary.update(
        {
            "passed": bool(not blockers),
            "blockers": blockers,
            "algorithm": "exact_restricted_response_oracle_h2h",
            "environment": "poker_ai:full_deck_hu_nlhe",
            "num_actions": N_ACTIONS,
            "uses_slumbot_training_data": False,
            "baseline_checkpoint": str(cfg.baseline_checkpoint),
            "baseline_kind": str(cfg.baseline_kind),
            "continuation_specs": [
                {
                    "kind": spec.kind,
                    "checkpoint": spec.checkpoint,
                    "weight": float(spec.weight),
                }
                for spec in continuation_specs
            ],
            "n_duplicate_pairs": int(cfg.n_games),
            "n_worlds": int(cfg.n_worlds),
            "initial_chips": int(cfg.initial_chips),
            "seed": int(cfg.seed),
            "hand_truncations": int(hand_truncations),
            "search_truncated_rollouts": int(search_truncated_rollouts),
            "response_payoffs": response_payoffs,
            "candidate_seat0_payouts": candidate_seat0_payouts,
            "baseline_seat0_payouts": baseline_seat0_payouts,
            "decision_records": decision_records,
            "n_response_decisions": int(latency_arr.size),
            "mean_response_decision_ms": (
                float(latency_arr.mean()) if latency_arr.size else 0.0
            ),
            "p95_response_decision_ms": (
                float(np.percentile(latency_arr, 95)) if latency_arr.size else 0.0
            ),
            "max_response_decision_ms": (
                float(latency_arr.max()) if latency_arr.size else 0.0
            ),
            "eval_seconds": float(elapsed),
            **device_info,
        }
    )
    return summary
