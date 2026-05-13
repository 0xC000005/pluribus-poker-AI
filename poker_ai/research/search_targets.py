"""Build search-consistency policy targets for Deep CFR."""

from __future__ import annotations

import json
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from poker_ai.games.full_deck.state import N_ACTIONS
from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.research.evaluation import (
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)
from poker_ai.research.resolver_benchmark import (
    ResolverBenchmarkCase,
    SolverDecision,
    _solver_decision,
    default_benchmark_cases,
    load_cases_json,
)


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import (  # noqa: E402
    _compute_bets_before_street,
    action_to_slumbot,
    build_features,
    card_str_to_index,
    get_legal_mask_from_parsed,
    network_strategy,
    parse_action,
)
from range_tracker import RangeTracker, update_tracker_from_actions  # noqa: E402
from solver import solve_street, solver_action_to_slumbot  # noqa: E402


_SUITS = ("c", "d", "h", "s")
_RANKS = tuple("23456789TJQKA")
_BET_SIZES = (200, 300, 500, 800, 1200, 2000, 4000)


@dataclass(frozen=True)
class _RangeTargetContext:
    checkpoint: str
    strategy_source: str
    device: torch.device
    value_net: Any
    checkpoint_metadata: dict[str, Any]


def _card_to_str(card: int) -> str:
    return _RANKS[card // 4] + _SUITS[card % 4]


def _sample_cards(rng: np.random.Generator, n: int) -> list[str]:
    cards = rng.choice(52, size=n, replace=False)
    return [_card_to_str(int(card)) for card in cards]


def _sample_action_template(rng: np.random.Generator) -> tuple[int, str, int]:
    flop_bet = int(rng.choice(_BET_SIZES))
    turn_bet = int(rng.choice(_BET_SIZES))
    river_bet = int(rng.choice(_BET_SIZES))
    templates = (
        (2, "ck/kk/", 0),
        (2, f"ck/kk/b{turn_bet}", 1),
        (2, f"ck/b{flop_bet}c/", 0),
        (2, f"ck/b{flop_bet}c/b{turn_bet}", 1),
        (3, "ck/kk/kk/", 0),
        (3, f"ck/kk/kk/b{river_bet}", 1),
        (3, f"ck/b{flop_bet}c/kk/", 0),
        (3, f"ck/b{flop_bet}c/kk/b{river_bet}", 1),
        (3, f"ck/b{flop_bet}c/b{turn_bet}c/", 0),
    )
    return templates[int(rng.integers(0, len(templates)))]


def sample_resolver_cases(
    n_cases: int,
    *,
    seed: int = 0,
    source: str = "sampled",
) -> list[ResolverBenchmarkCase]:
    """Sample valid turn/river public states for resolver-target generation."""
    rng = np.random.default_rng(seed)
    cases: list[ResolverBenchmarkCase] = []
    attempts = 0
    while len(cases) < n_cases and attempts < max(100, n_cases * 20):
        attempts += 1
        street, action_str, client_pos = _sample_action_template(rng)
        cards = _sample_cards(rng, 7)
        hole_cards = tuple(cards[:2])
        board = tuple(cards[2:6] if street == 2 else cards[2:7])
        parsed = parse_action(action_str)
        if "error" in parsed:
            continue
        if int(parsed.get("st", -1)) != street or int(parsed.get("pos", -1)) != client_pos:
            continue
        label = f"{source}-{len(cases):04d}-street{street}"
        cases.append(
            ResolverBenchmarkCase(
                label=label,
                hole_cards=hole_cards,
                board=board,
                action_str=action_str,
                client_pos=client_pos,
                source=source,
            )
        )
    if len(cases) != n_cases:
        raise RuntimeError(f"generated {len(cases)} valid cases out of requested {n_cases}")
    return cases


def _visible_board(board: list[str], street: int) -> list[str]:
    if street <= 0:
        return []
    if street == 1:
        return board[:3]
    if street == 2:
        return board[:4]
    return board[:5]


def _sample_policy_action(
    value_net: Any,
    device: torch.device,
    *,
    hole_cards: list[str],
    board: list[str],
    action_str: str,
    acting_pos: int,
    parsed: dict,
    strategy_source: str,
    rng: np.random.Generator,
) -> int:
    features = build_features(
        hole_cards,
        board,
        action_str,
        acting_pos,
        parsed,
    )
    legal_mask = get_legal_mask_from_parsed(parsed, action_str, acting_pos)
    _, strategy = network_strategy(
        value_net,
        features,
        legal_mask,
        device,
        strategy_source=strategy_source,
    )
    legal_actions = np.flatnonzero(legal_mask > 0)
    if legal_actions.size == 0:
        return 1
    probs = np.asarray([strategy[action] for action in legal_actions], dtype=np.float64)
    total = float(probs.sum())
    if total > 0:
        probs /= total
    else:
        probs = np.ones_like(probs, dtype=np.float64) / len(probs)
    return int(rng.choice(legal_actions, p=probs))


def _append_action(action_str: str, increment: str, previous_street: int) -> str:
    next_action = action_str + increment
    if increment.startswith("f"):
        return next_action
    parsed_next = parse_action(next_action)
    if "error" in parsed_next:
        return next_action
    if (
        int(parsed_next.get("st", previous_street)) > previous_street
        and int(parsed_next.get("pos", -1)) >= 0
        and not next_action.endswith("/")
    ):
        return next_action + "/"
    return next_action


def sample_blueprint_resolver_cases(
    n_cases: int,
    *,
    blueprint_checkpoint: str | Path,
    seed: int = 0,
    strategy_source: str = "regret",
    device: str | torch.device = "auto",
    source: str = "blueprint_self_play",
    max_attempts: int | None = None,
    target_streets: Iterable[int] = (2, 3),
    stats: dict[str, Any] | None = None,
) -> list[ResolverBenchmarkCase]:
    """Sample turn/river cases from learned-policy Slumbot-format rollouts."""
    resolved_device = _resolve_device(device)
    loaded = load_value_network_checkpoint(blueprint_checkpoint, resolved_device)
    assert_strategy_source_supported(loaded, strategy_source)
    target_streets_tuple = tuple(sorted({int(street) for street in target_streets}))
    if not target_streets_tuple or any(street not in (2, 3) for street in target_streets_tuple):
        raise ValueError("target_streets must contain only turn(2) and/or river(3)")
    rng = np.random.default_rng(seed)
    max_attempts = max_attempts or max(200, n_cases * 200)
    cases: list[ResolverBenchmarkCase] = []
    attempts = 0
    while len(cases) < n_cases and attempts < max_attempts:
        attempts += 1
        target_street = int(target_streets_tuple[len(cases) % len(target_streets_tuple)])
        cards = _sample_cards(rng, 9)
        hero_hole = cards[:2]
        villain_hole = cards[2:4]
        board = cards[4:9]
        client_pos = int(rng.integers(0, 2))
        action_str = ""

        for _ in range(40):
            parsed = parse_action(action_str)
            if "error" in parsed:
                break
            street = int(parsed.get("st", -1))
            acting_pos = int(parsed.get("pos", -1))
            if acting_pos < 0 or street not in (0, 1, 2, 3):
                break
            visible_board = _visible_board(board, street)
            if street == target_street and acting_pos == client_pos:
                cases.append(
                    ResolverBenchmarkCase(
                        label=f"{source}-{len(cases):04d}-street{street}",
                        hole_cards=tuple(hero_hole),
                        board=tuple(visible_board),
                        action_str=action_str,
                        client_pos=client_pos,
                        source=source,
                    )
                )
                break

            acting_hole = hero_hole if acting_pos == client_pos else villain_hole
            action_idx = _sample_policy_action(
                loaded.value_net,
                resolved_device,
                hole_cards=acting_hole,
                board=visible_board,
                action_str=action_str,
                acting_pos=acting_pos,
                parsed=parsed,
                strategy_source=strategy_source,
                rng=rng,
            )
            increment = action_to_slumbot(action_idx, parsed, action_str, acting_pos)
            action_str = _append_action(action_str, increment, street)
            if increment.startswith("f"):
                break

    if stats is not None:
        stats["attempts"] = int(attempts)
        stats["requested_cases"] = int(n_cases)
        stats["generated_cases"] = int(len(cases))
        stats["success_rate"] = round(float(len(cases)) / max(float(attempts), 1.0), 6)
        stats["target_streets"] = [int(street) for street in target_streets_tuple]
    if len(cases) != n_cases:
        raise RuntimeError(
            f"generated {len(cases)} reachable cases out of requested {n_cases} "
            f"after {attempts} attempts"
        )
    return cases


def save_cases_json(cases: Iterable[ResolverBenchmarkCase], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cases": [
            {
                "label": case.label,
                "hole_cards": list(case.hole_cards),
                "board": list(case.board),
                "action_str": case.action_str,
                "client_pos": int(case.client_pos),
                "source": case.source,
            }
            for case in cases
        ]
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _normalized_legal_target(
    strategy: np.ndarray,
    legal_mask: np.ndarray,
    fallback_action: int,
) -> np.ndarray:
    target = np.asarray(strategy, dtype=np.float32) * (legal_mask > 0)
    total = float(target.sum())
    if total > 1e-8:
        return target / total
    fallback = np.zeros_like(target, dtype=np.float32)
    if 0 <= fallback_action < len(fallback) and legal_mask[fallback_action] > 0:
        fallback[fallback_action] = 1.0
        return fallback
    legal_total = float(legal_mask.sum())
    if legal_total <= 0:
        raise ValueError("cannot build target without legal actions")
    return legal_mask.astype(np.float32) / legal_total


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_range_target_context(
    checkpoint: str | Path | None,
    *,
    strategy_source: str,
    device: str | torch.device,
) -> _RangeTargetContext | None:
    if checkpoint is None:
        return None
    resolved_device = _resolve_device(device)
    loaded = load_value_network_checkpoint(checkpoint, resolved_device)
    assert_strategy_source_supported(loaded, strategy_source)
    return _RangeTargetContext(
        checkpoint=str(checkpoint),
        strategy_source=strategy_source,
        device=resolved_device,
        value_net=loaded.value_net,
        checkpoint_metadata=loaded.metadata,
    )


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


def _belief_conditioned_solver_decision(
    case: ResolverBenchmarkCase,
    parsed: dict,
    *,
    solver_iterations: int,
    solver_backend: str,
    range_context: _RangeTargetContext,
    range_prune_threshold: float,
) -> tuple[SolverDecision | None, dict[str, Any]]:
    street = int(parsed["st"])
    if street not in (2, 3):
        return None, {}

    n_board = 4 if street == 2 else 5
    board_idx = [card_str_to_index(card) for card in case.board[:n_board]]
    our_cards_idx = [card_str_to_index(card) for card in case.hole_cards]
    our_bet_pre, opp_bet_pre = _compute_bets_before_street(
        case.action_str,
        case.client_pos,
        target_street=street,
    )
    pot = our_bet_pre + opp_bet_pre
    hero_stack = 20000 - our_bet_pre
    villain_stack = 20000 - opp_bet_pre
    hero_first = case.client_pos == 0
    street_parts = case.action_str.split("/")
    street_action = street_parts[street] if len(street_parts) > street else ""

    tracker = RangeTracker(
        our_cards_idx,
        range_context.value_net,
        range_context.device,
        strategy_source=range_context.strategy_source,
    )
    update_tracker_from_actions(tracker, case.action_str, case.client_pos, board_idx)

    remaining = sorted(set(range(52)) - set(board_idx))
    solver_hands = list(itertools.combinations(remaining, 2))
    solver_hand_to_idx = {hand: i for i, hand in enumerate(solver_hands)}
    hero_range, villain_range = tracker.get_solver_ranges(solver_hands, solver_hand_to_idx)

    extra = {
        "range_mode": "belief_conditioned",
        "range_prune_threshold": float(range_prune_threshold),
        **_range_summary("hero_range", hero_range),
        **_range_summary("villain_range", villain_range),
    }

    started = time.perf_counter()
    _, strategy, solver, node = solve_street(
        our_cards_idx,
        board_idx,
        pot,
        hero_stack,
        villain_stack,
        hero_first,
        action_str=street_action,
        n_iterations=solver_iterations,
        hero_range=hero_range,
        villain_range=villain_range,
        backend=solver_backend,
        range_prune_threshold=range_prune_threshold,
    )

    strategy_vec = np.zeros(N_ACTIONS, dtype=np.float64)
    if node is None or node.is_terminal:
        action = 1
        strategy_vec[1] = 1.0
        increment = "k" if parsed["last_bet_size"] == 0 else "c"
        node_terminal = True
    else:
        for action_idx, prob in strategy.items():
            if 0 <= action_idx < len(strategy_vec):
                strategy_vec[action_idx] = float(prob)
        total = float(strategy_vec.sum())
        if total > 0:
            strategy_vec /= total
        else:
            strategy_vec[1] = 1.0
        action = int(np.argmax(strategy_vec))
        increment = solver_action_to_slumbot(action, node, solver, parsed)
        node_terminal = False

    latency_ms = (time.perf_counter() - started) * 1000.0
    extra["solver_n_hands"] = int(getattr(solver, "n", 0))
    extra["solver_full_n_hands"] = int(getattr(solver, "full_n", 0))
    return SolverDecision(action, increment, strategy_vec, latency_ms, node_terminal), extra


def _target_summary(targets: np.ndarray) -> dict[str, float]:
    top_actions = np.argmax(targets, axis=1)
    entropies = []
    for target in targets:
        positive = target[target > 0]
        entropies.append(float(-(positive * np.log(positive)).sum()) if positive.size else 0.0)
    return {
        "target_allin_rate": round(float(np.mean(top_actions == 8)), 6),
        "mean_target_allin_prob": round(float(np.mean(targets[:, 8])), 6),
        "mean_target_entropy": round(float(np.mean(entropies)), 6),
    }


def build_resolver_policy_targets(
    cases: Iterable[ResolverBenchmarkCase] | None = None,
    *,
    solver_iterations: int = 25,
    solver_backend: str = "auto",
    range_checkpoint: str | Path | None = None,
    range_strategy_source: str = "regret",
    range_device: str | torch.device = "auto",
    range_prune_threshold: float = 1e-4,
) -> tuple[PolicyTargetBuffer, dict]:
    """Build a small policy-target dataset from deterministic resolver cases."""
    cases = list(default_benchmark_cases() if cases is None else cases)
    range_context = _load_range_target_context(
        range_checkpoint,
        strategy_source=range_strategy_source,
        device=range_device,
    )
    features: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    target_probs: list[np.ndarray] = []
    records: list[dict] = []

    for case in cases:
        parsed = parse_action(case.action_str)
        if "error" in parsed:
            records.append(
                {
                    "label": case.label,
                    "skipped": f"parse_error:{parsed['error']}",
                }
            )
            continue
        feature_vec = build_features(
            list(case.hole_cards),
            list(case.board),
            case.action_str,
            case.client_pos,
            parsed,
        ).astype(np.float32)
        legal_mask = get_legal_mask_from_parsed(
            parsed,
            case.action_str,
            case.client_pos,
        ).astype(np.float32)
        extra_record: dict[str, Any] = {"range_mode": "uniform"}
        if range_context is None:
            solver = _solver_decision(
                case,
                parsed,
                solver_iterations=solver_iterations,
                solver_backend=solver_backend,
            )
        else:
            solver, extra_record = _belief_conditioned_solver_decision(
                case,
                parsed,
                solver_iterations=solver_iterations,
                solver_backend=solver_backend,
                range_context=range_context,
                range_prune_threshold=range_prune_threshold,
            )
        if solver is None:
            records.append({"label": case.label, "skipped": "unsupported_street"})
            continue

        target = _normalized_legal_target(
            solver.strategy,
            legal_mask,
            fallback_action=solver.action,
        )
        features.append(feature_vec)
        legal_masks.append(legal_mask)
        target_probs.append(target)
        records.append(
            {
                "label": case.label,
                "street": int(parsed["st"]),
                "solver_action": int(solver.action),
                "solver_increment": solver.increment,
                "solver_latency_ms": round(float(solver.latency_ms), 3),
                "target_entropy": round(
                    float(-(target[target > 0] * np.log(target[target > 0])).sum()),
                    6,
                ),
                **extra_record,
            }
        )

    if not features:
        raise RuntimeError("no resolver policy targets were generated")

    buffer = PolicyTargetBuffer(
        np.stack(features, axis=0),
        np.stack(legal_masks, axis=0),
        np.stack(target_probs, axis=0),
    )
    metadata = {
        "mode": (
            "belief_conditioned_resolver_policy_targets"
            if range_context is not None
            else "resolver_policy_targets"
        ),
        "solver_iterations": int(solver_iterations),
        "solver_backend": solver_backend,
        "range_enabled": range_context is not None,
        "n_cases": len(cases),
        "n_targets": int(buffer.size),
        "records": records,
        **_target_summary(buffer.target_probs),
    }
    if range_context is not None:
        metadata.update(
            {
                "range_checkpoint": range_context.checkpoint,
                "range_checkpoint_iteration": range_context.checkpoint_metadata.get(
                    "checkpoint_iteration"
                ),
                "range_strategy_source": range_context.strategy_source,
                "range_device": str(range_context.device),
                "range_prune_threshold": float(range_prune_threshold),
            }
        )
    return buffer, metadata


def save_resolver_policy_targets(
    output: str | Path,
    *,
    cases_json: str | Path | None = None,
    sampled_cases: int = 0,
    blueprint_cases: int = 0,
    blueprint_checkpoint: str | Path | None = None,
    blueprint_strategy_source: str = "regret",
    blueprint_device: str | torch.device = "auto",
    blueprint_max_attempts: int | None = None,
    blueprint_target_streets: Iterable[int] = (2, 3),
    seed: int = 0,
    solver_iterations: int = 25,
    solver_backend: str = "auto",
    range_checkpoint: str | Path | None = None,
    range_strategy_source: str = "regret",
    range_device: str | torch.device = "auto",
    range_prune_threshold: float = 1e-4,
) -> dict:
    selected_sources = sum(bool(item) for item in (cases_json, sampled_cases, blueprint_cases))
    if selected_sources > 1:
        raise ValueError("choose only one of cases_json, sampled_cases, or blueprint_cases")
    if cases_json:
        cases = load_cases_json(cases_json)
        case_source = str(cases_json)
    elif blueprint_cases:
        if blueprint_checkpoint is None:
            raise ValueError("blueprint_cases requires blueprint_checkpoint")
        blueprint_stats: dict[str, Any] = {}
        cases = sample_blueprint_resolver_cases(
            blueprint_cases,
            blueprint_checkpoint=blueprint_checkpoint,
            seed=seed,
            strategy_source=blueprint_strategy_source,
            device=blueprint_device,
            max_attempts=blueprint_max_attempts,
            target_streets=blueprint_target_streets,
            stats=blueprint_stats,
        )
        case_source = "blueprint_self_play"
    elif sampled_cases:
        cases = sample_resolver_cases(sampled_cases, seed=seed)
        case_source = "sampled"
    else:
        cases = default_benchmark_cases()
        case_source = "fixed_default"
    buffer, metadata = build_resolver_policy_targets(
        cases,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        range_checkpoint=range_checkpoint,
        range_strategy_source=range_strategy_source,
        range_device=range_device,
        range_prune_threshold=range_prune_threshold,
    )
    output = Path(output)
    buffer.save_npz(output)
    metadata_path = output.with_suffix(".json")
    cases_path = output.with_suffix(".cases.json")
    save_cases_json(cases, cases_path)
    metadata["output"] = str(output)
    metadata["metadata"] = str(metadata_path)
    metadata["cases_json"] = str(cases_path)
    metadata["case_source"] = case_source
    metadata["seed"] = int(seed) if sampled_cases else None
    if blueprint_cases:
        metadata["seed"] = int(seed)
        metadata["blueprint_checkpoint"] = str(blueprint_checkpoint)
        metadata["blueprint_strategy_source"] = blueprint_strategy_source
        metadata["blueprint_device"] = str(_resolve_device(blueprint_device))
        metadata["blueprint_max_attempts"] = blueprint_max_attempts
        metadata["blueprint_target_streets"] = [
            int(street) for street in blueprint_target_streets
        ]
        metadata["blueprint_sampling"] = blueprint_stats
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata
