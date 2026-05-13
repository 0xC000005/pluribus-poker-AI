"""Probe whether public-belief range inputs explain search value targets.

This diagnostic sits between the action-target public-belief probe and a full
trainer change. It asks a narrower question: when a turn/river subgame is solved
from learned ranges, does appending the raw public belief improve held-out
prediction of the searched hero value?
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import N_FEATURES
from poker_ai.research.belief_probe import (
    BELIEF_DIM,
    N_HANDS,
    _HAND_TO_INDEX,
    _case_range_vectors,
    _resolve_device,
    _standardize_pair,
)
from poker_ai.research.evaluation import (
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase, load_cases_json


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fast_cfr import (  # noqa: E402
    T_HERO_FOLD,
    T_SHOWDOWN,
    T_VILLAIN_FOLD,
    prune_hands,
)
from play_slumbot import (  # noqa: E402
    _compute_bets_before_street,
    card_str_to_index,
    parse_action,
)
from range_tracker import RangeTracker, update_tracker_from_actions  # noqa: E402
from solver import StreetSolver, _parse_nav, resolve_solver_backend  # noqa: E402


@dataclass(frozen=True)
class PublicBeliefValueDataset:
    features: np.ndarray
    belief: np.ndarray
    values: np.ndarray
    labels: tuple[str, ...]


@dataclass(frozen=True)
class PublicBeliefCFVDataset:
    features: np.ndarray
    belief: np.ndarray
    values: np.ndarray
    value_masks: np.ndarray
    labels: tuple[str, ...]


class _ValueProbeNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _CFVProbeNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = N_HANDS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _HandCFVProbeNet(nn.Module):
    def __init__(self, hidden_dim: int, *, use_belief: bool):
        super().__init__()
        self.use_belief = bool(use_belief)
        self.public = nn.Linear(N_FEATURES, hidden_dim)
        self.hand = nn.Linear(52, hidden_dim)
        self.belief = nn.Linear(BELIEF_DIM, hidden_dim) if use_belief else None
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        public_x: torch.Tensor,
        hand_x: torch.Tensor,
        belief_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.public(public_x) + self.hand(hand_x)
        if self.belief is not None:
            if belief_x is None:
                raise ValueError("belief_x is required when use_belief=True")
            hidden = hidden + self.belief(belief_x)
        return self.out(hidden).squeeze(-1)


_HAND_FEATURES = np.zeros((N_HANDS, 52), dtype=np.float32)
for _hand, _idx in _HAND_TO_INDEX.items():
    _HAND_FEATURES[_idx, int(_hand[0])] = 1.0
    _HAND_FEATURES[_idx, int(_hand[1])] = 1.0


def _strategy_for_hand(
    strategy_sum: np.ndarray,
    node_idx: int,
    actions: list[int],
    hand_idx: int,
) -> np.ndarray:
    sums = np.array([strategy_sum[node_idx, action, hand_idx] for action in actions])
    total = float(sums.sum())
    if total > 0:
        return (sums / total).astype(np.float64)
    return np.full(len(actions), 1.0 / max(len(actions), 1), dtype=np.float64)


def _strategy_matrix(
    strategy_sum: np.ndarray,
    node_idx: int,
    actions: list[int],
) -> np.ndarray:
    sums = np.stack([strategy_sum[node_idx, action] for action in actions], axis=0)
    totals = sums.sum(axis=0, keepdims=True)
    uniform = np.full_like(sums, 1.0 / max(len(actions), 1), dtype=np.float64)
    return np.where(totals > 0, sums / np.maximum(totals, 1e-12), uniform)


def _terminal_weighted_value(
    solver: StreetSolver,
    node_idx: int,
    hero_idx: int,
    villain_reach: np.ndarray,
) -> tuple[float, float]:
    valid_weights = np.asarray(villain_reach, dtype=np.float64) * solver.valid[hero_idx]
    mass = float(valid_weights.sum())
    if mass <= 1e-12:
        return 0.0, 0.0

    tree = solver._tree
    terminal_type = int(tree["terminal_type"][node_idx])
    hero_invested = float(solver.hero_stack_start - tree["stacks_h"][node_idx])
    villain_invested = float(solver.villain_stack_start - tree["stacks_v"][node_idx])

    if terminal_type == T_HERO_FOLD:
        return -hero_invested * mass, mass
    if terminal_type == T_VILLAIN_FOLD:
        return (float(solver.pot_start) + villain_invested) * mass, mass
    if terminal_type != T_SHOWDOWN:
        return 0.0, mass

    win_payoff = float(solver.pot_start) + villain_invested
    lose_payoff = -hero_invested
    tie_payoff = (float(solver.pot_start) + villain_invested - hero_invested) / 2.0
    payoff_by_villain = (
        solver.win_m[hero_idx] * win_payoff
        + solver.lose_m[hero_idx] * lose_payoff
        + solver.tie_m[hero_idx] * tie_payoff
    )
    return float(np.dot(valid_weights, payoff_by_villain)), mass


def compute_hero_hand_ev(
    solver: StreetSolver,
    node: Any,
    hero_hand: tuple[int, int] | list[int],
    villain_range: np.ndarray,
) -> float:
    """Return searched hero EV in chips for ``hero_hand`` at ``node``."""
    hand = tuple(sorted(int(card) for card in hero_hand))
    hero_idx = solver.hand_to_idx.get(hand)
    if hero_idx is None:
        raise ValueError(f"hero hand {hand} is not present in solver hand set")

    villain_reach = np.maximum(np.asarray(villain_range, dtype=np.float64), 0.0)
    if villain_reach.shape != (solver.n,):
        raise ValueError(
            f"villain_range shape {villain_reach.shape} does not match solver.n={solver.n}"
        )
    if float(villain_reach.sum()) <= 0:
        villain_reach = np.ones(solver.n, dtype=np.float64)

    tree = solver._tree
    node_idx_by_id = {id(item): i for i, item in enumerate(tree["all_nodes"])}

    def visit(current: Any, reach: np.ndarray) -> tuple[float, float]:
        node_idx = node_idx_by_id[id(current)]
        player = int(tree["player"][node_idx])
        if player == -1:
            return _terminal_weighted_value(solver, node_idx, hero_idx, reach)

        actions = list(tree["decision_actions"][node_idx])
        if not actions:
            return 0.0, 0.0
        if player == 0:
            probs = _strategy_for_hand(solver._strategy_sum, node_idx, actions, hero_idx)
            weighted_value = 0.0
            weighted_mass = 0.0
            for prob, action in zip(probs, actions, strict=True):
                child = tree["all_nodes"][int(tree["children"][node_idx, action])]
                child_value, child_mass = visit(child, reach)
                weighted_value += float(prob) * child_value
                weighted_mass += float(prob) * child_mass
            return weighted_value, weighted_mass

        strategy = _strategy_matrix(solver._strategy_sum, node_idx, actions)
        weighted_value = 0.0
        weighted_mass = 0.0
        for action_idx, action in enumerate(actions):
            next_reach = reach * strategy[action_idx]
            child = tree["all_nodes"][int(tree["children"][node_idx, action])]
            child_value, child_mass = visit(child, next_reach)
            weighted_value += child_value
            weighted_mass += child_mass
        return weighted_value, weighted_mass

    numerator, denominator = visit(node, villain_reach)
    return float(numerator / denominator) if denominator > 1e-12 else 0.0


def compute_hero_cfv_vector(
    solver: StreetSolver,
    node: Any,
    villain_range: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return hero counterfactual values for every solver hand at ``node``."""
    villain_reach = np.maximum(np.asarray(villain_range, dtype=np.float64), 0.0)
    if villain_reach.shape != (solver.n,):
        raise ValueError(
            f"villain_range shape {villain_reach.shape} does not match solver.n={solver.n}"
        )
    if float(villain_reach.sum()) <= 0:
        villain_reach = np.ones(solver.n, dtype=np.float64)

    tree = solver._tree
    node_idx_by_id = {id(item): i for i, item in enumerate(tree["all_nodes"])}

    def terminal(node_idx: int, reach: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid_weights = solver.valid.astype(np.float64) * reach.reshape(1, -1)
        denom = valid_weights.sum(axis=1)
        terminal_type = int(tree["terminal_type"][node_idx])
        hero_invested = float(solver.hero_stack_start - tree["stacks_h"][node_idx])
        villain_invested = float(solver.villain_stack_start - tree["stacks_v"][node_idx])

        if terminal_type == T_HERO_FOLD:
            return -hero_invested * denom, denom
        if terminal_type == T_VILLAIN_FOLD:
            return (float(solver.pot_start) + villain_invested) * denom, denom
        if terminal_type != T_SHOWDOWN:
            return np.zeros(solver.n, dtype=np.float64), denom

        win_payoff = float(solver.pot_start) + villain_invested
        lose_payoff = -hero_invested
        tie_payoff = (float(solver.pot_start) + villain_invested - hero_invested) / 2.0
        payoff = (
            solver.win_m.astype(np.float64) * win_payoff
            + solver.lose_m.astype(np.float64) * lose_payoff
            + solver.tie_m.astype(np.float64) * tie_payoff
        )
        return (payoff * valid_weights).sum(axis=1), denom

    def visit(current: Any, reach: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        node_idx = node_idx_by_id[id(current)]
        player = int(tree["player"][node_idx])
        if player == -1:
            return terminal(node_idx, reach)

        actions = list(tree["decision_actions"][node_idx])
        if not actions:
            return (
                np.zeros(solver.n, dtype=np.float64),
                np.zeros(solver.n, dtype=np.float64),
            )
        if player == 0:
            strategy = _strategy_matrix(solver._strategy_sum, node_idx, actions)
            total_num = np.zeros(solver.n, dtype=np.float64)
            total_den = np.zeros(solver.n, dtype=np.float64)
            for action_idx, action in enumerate(actions):
                child = tree["all_nodes"][int(tree["children"][node_idx, action])]
                child_num, child_den = visit(child, reach)
                total_num += strategy[action_idx] * child_num
                total_den += strategy[action_idx] * child_den
            return total_num, total_den

        strategy = _strategy_matrix(solver._strategy_sum, node_idx, actions)
        total_num = np.zeros(solver.n, dtype=np.float64)
        total_den = np.zeros(solver.n, dtype=np.float64)
        for action_idx, action in enumerate(actions):
            next_reach = reach * strategy[action_idx]
            child = tree["all_nodes"][int(tree["children"][node_idx, action])]
            child_num, child_den = visit(child, next_reach)
            total_num += child_num
            total_den += child_den
        return total_num, total_den

    numerator, denominator = visit(node, villain_reach)
    values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 1e-12,
    )
    return values.astype(np.float32), (denominator > 1e-12).astype(np.float32)


def _public_features(features: np.ndarray) -> np.ndarray:
    public = np.asarray(features, dtype=np.float32).copy()
    public[..., :52] = 0.0
    return public


def _case_value_target(
    case: ResolverBenchmarkCase,
    *,
    value_net: Any,
    device: torch.device,
    strategy_source: str,
    solver_iterations: int,
    solver_backend: str,
    range_prune_threshold: float,
    value_scale: float,
) -> tuple[float, dict[str, Any]]:
    parsed = parse_action(case.action_str)
    if "error" in parsed:
        raise ValueError(f"{case.label}: parse error: {parsed['error']}")
    street = int(parsed.get("st", -1))
    if street not in (2, 3):
        raise ValueError(f"{case.label}: expected turn/river street, got {street}")

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
        value_net,
        device,
        strategy_source=strategy_source,
    )
    update_tracker_from_actions(tracker, case.action_str, case.client_pos, board_idx)
    remaining = sorted(set(range(52)) - set(board_idx))
    full_hands = list(itertools.combinations(remaining, 2))
    full_hand_to_idx = {hand: i for i, hand in enumerate(full_hands)}
    hero_range, villain_range = tracker.get_solver_ranges(full_hands, full_hand_to_idx)

    active_indices = None
    hero_range_used = hero_range
    villain_range_used = villain_range
    if range_prune_threshold > 0:
        _, active_indices, hero_range_used, villain_range_used = prune_hands(
            full_hands,
            hero_range=hero_range,
            villain_range=villain_range,
            keep_hand=tuple(sorted(our_cards_idx)),
            threshold=range_prune_threshold,
        )

    backend, backend_device = resolve_solver_backend(solver_backend)
    started = time.perf_counter()
    solver = StreetSolver(
        board_idx,
        pot,
        hero_stack,
        villain_stack,
        hero_first,
        active_indices=active_indices,
    )
    solver.solve(
        n_iterations=solver_iterations,
        hero_range=hero_range_used,
        villain_range=villain_range_used,
        backend=backend,
        device=backend_device,
    )
    node = solver.navigate(_parse_nav(street_action, solver))
    if node is None:
        raise ValueError(f"{case.label}: solver could not navigate street action")

    ev_chips = compute_hero_hand_ev(
        solver,
        node,
        tuple(sorted(our_cards_idx)),
        np.asarray(villain_range_used, dtype=np.float32),
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    return ev_chips / float(value_scale), {
        "label": case.label,
        "street": street,
        "value_chips": round(float(ev_chips), 6),
        "value_scaled": round(float(ev_chips / float(value_scale)), 8),
        "solver_latency_ms": round(float(latency_ms), 3),
        "solver_n_hands": int(solver.n),
        "solver_full_n_hands": int(solver.full_n),
    }


def _case_cfv_target(
    case: ResolverBenchmarkCase,
    *,
    value_net: Any,
    device: torch.device,
    strategy_source: str,
    solver_iterations: int,
    solver_backend: str,
    range_prune_threshold: float,
    value_scale: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    parsed = parse_action(case.action_str)
    if "error" in parsed:
        raise ValueError(f"{case.label}: parse error: {parsed['error']}")
    street = int(parsed.get("st", -1))
    if street not in (2, 3):
        raise ValueError(f"{case.label}: expected turn/river street, got {street}")

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
        value_net,
        device,
        strategy_source=strategy_source,
    )
    update_tracker_from_actions(tracker, case.action_str, case.client_pos, board_idx)
    remaining = sorted(set(range(52)) - set(board_idx))
    full_hands = list(itertools.combinations(remaining, 2))
    full_hand_to_idx = {hand: i for i, hand in enumerate(full_hands)}
    hero_range, villain_range = tracker.get_solver_ranges(full_hands, full_hand_to_idx)

    active_indices = None
    hero_range_used = hero_range
    villain_range_used = villain_range
    if range_prune_threshold > 0:
        _, active_indices, hero_range_used, villain_range_used = prune_hands(
            full_hands,
            hero_range=hero_range,
            villain_range=villain_range,
            keep_hand=tuple(sorted(our_cards_idx)),
            threshold=range_prune_threshold,
        )

    backend, backend_device = resolve_solver_backend(solver_backend)
    started = time.perf_counter()
    solver = StreetSolver(
        board_idx,
        pot,
        hero_stack,
        villain_stack,
        hero_first,
        active_indices=active_indices,
    )
    solver.solve(
        n_iterations=solver_iterations,
        hero_range=hero_range_used,
        villain_range=villain_range_used,
        backend=backend,
        device=backend_device,
    )
    node = solver.navigate(_parse_nav(street_action, solver))
    if node is None:
        raise ValueError(f"{case.label}: solver could not navigate street action")

    local_values, local_mask = compute_hero_cfv_vector(
        solver,
        node,
        np.asarray(villain_range_used, dtype=np.float32),
    )
    values = np.zeros(N_HANDS, dtype=np.float32)
    mask = np.zeros(N_HANDS, dtype=np.float32)
    for local_idx, hand in enumerate(solver.hands):
        global_idx = _HAND_TO_INDEX[tuple(sorted(hand))]
        values[global_idx] = float(local_values[local_idx]) / float(value_scale)
        mask[global_idx] = float(local_mask[local_idx])

    latency_ms = (time.perf_counter() - started) * 1000.0
    active_values = values[mask > 0]
    return values, mask, {
        "label": case.label,
        "street": street,
        "value_mean": round(float(active_values.mean()) if active_values.size else 0.0, 8),
        "value_std": round(float(active_values.std()) if active_values.size else 0.0, 8),
        "value_mask_count": int(mask.sum()),
        "solver_latency_ms": round(float(latency_ms), 3),
        "solver_n_hands": int(solver.n),
        "solver_full_n_hands": int(solver.full_n),
    }


def load_public_belief_value_dataset(
    *,
    targets_npz: str | Path,
    cases_json: str | Path,
    range_checkpoint: str | Path,
    range_strategy_source: str = "regret",
    device: str | torch.device = "auto",
    solver_iterations: int = 25,
    solver_backend: str = "auto",
    range_prune_threshold: float = 1e-4,
    value_scale: float = 20000.0,
) -> tuple[PublicBeliefValueDataset, list[dict[str, Any]]]:
    resolved_device = _resolve_device(device)
    loaded = load_value_network_checkpoint(range_checkpoint, resolved_device)
    assert_strategy_source_supported(loaded, range_strategy_source)
    targets = PolicyTargetBuffer.from_npz(targets_npz)
    cases = load_cases_json(cases_json)
    if len(cases) != targets.size:
        raise ValueError(
            f"case count {len(cases)} does not match target count {targets.size}"
        )

    belief_rows = []
    values = []
    records = []
    labels = []
    for case in cases:
        hero_range, villain_range = _case_range_vectors(
            case,
            value_net=loaded.value_net,
            device=resolved_device,
            strategy_source=range_strategy_source,
        )
        value, record = _case_value_target(
            case,
            value_net=loaded.value_net,
            device=resolved_device,
            strategy_source=range_strategy_source,
            solver_iterations=solver_iterations,
            solver_backend=solver_backend,
            range_prune_threshold=range_prune_threshold,
            value_scale=value_scale,
        )
        belief_rows.append(np.concatenate([hero_range, villain_range], axis=0))
        values.append(value)
        records.append(record)
        labels.append(case.label)

    dataset = PublicBeliefValueDataset(
        features=targets.features.astype(np.float32, copy=False),
        belief=np.stack(belief_rows, axis=0).astype(np.float32, copy=False),
        values=np.asarray(values, dtype=np.float32),
        labels=tuple(labels),
    )
    return dataset, records


def load_public_belief_cfv_dataset(
    *,
    targets_npz: str | Path,
    cases_json: str | Path,
    range_checkpoint: str | Path,
    range_strategy_source: str = "regret",
    device: str | torch.device = "auto",
    solver_iterations: int = 25,
    solver_backend: str = "auto",
    range_prune_threshold: float = 0.0,
    value_scale: float = 20000.0,
) -> tuple[PublicBeliefCFVDataset, list[dict[str, Any]]]:
    resolved_device = _resolve_device(device)
    loaded = load_value_network_checkpoint(range_checkpoint, resolved_device)
    assert_strategy_source_supported(loaded, range_strategy_source)
    targets = PolicyTargetBuffer.from_npz(targets_npz)
    cases = load_cases_json(cases_json)
    if len(cases) != targets.size:
        raise ValueError(
            f"case count {len(cases)} does not match target count {targets.size}"
        )

    belief_rows = []
    values = []
    masks = []
    records = []
    labels = []
    for case in cases:
        hero_range, villain_range = _case_range_vectors(
            case,
            value_net=loaded.value_net,
            device=resolved_device,
            strategy_source=range_strategy_source,
        )
        value_vec, value_mask, record = _case_cfv_target(
            case,
            value_net=loaded.value_net,
            device=resolved_device,
            strategy_source=range_strategy_source,
            solver_iterations=solver_iterations,
            solver_backend=solver_backend,
            range_prune_threshold=range_prune_threshold,
            value_scale=value_scale,
        )
        belief_rows.append(np.concatenate([hero_range, villain_range], axis=0))
        values.append(value_vec)
        masks.append(value_mask)
        records.append(record)
        labels.append(case.label)

    dataset = PublicBeliefCFVDataset(
        features=_public_features(targets.features),
        belief=np.stack(belief_rows, axis=0).astype(np.float32, copy=False),
        values=np.stack(values, axis=0).astype(np.float32, copy=False),
        value_masks=np.stack(masks, axis=0).astype(np.float32, copy=False),
        labels=tuple(labels),
    )
    return dataset, records


def save_public_belief_value_dataset_cache(
    dataset: PublicBeliefValueDataset,
    records: list[dict[str, Any]],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=dataset.features.astype(np.float32, copy=False),
        belief=dataset.belief.astype(np.float32, copy=False),
        values=dataset.values.astype(np.float32, copy=False),
        labels=np.asarray(dataset.labels, dtype=str),
        records_json=np.asarray(json.dumps(records, sort_keys=True)),
    )


def load_public_belief_value_dataset_cache(
    path: str | Path,
) -> tuple[PublicBeliefValueDataset, list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as data:
        dataset = PublicBeliefValueDataset(
            features=data["features"].astype(np.float32, copy=False),
            belief=data["belief"].astype(np.float32, copy=False),
            values=data["values"].astype(np.float32, copy=False),
            labels=tuple(str(label) for label in data["labels"].tolist()),
        )
        records = json.loads(str(data["records_json"].item()))
    return dataset, records


def save_public_belief_cfv_dataset_cache(
    dataset: PublicBeliefCFVDataset,
    records: list[dict[str, Any]],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=dataset.features.astype(np.float32, copy=False),
        belief=dataset.belief.astype(np.float32, copy=False),
        values=dataset.values.astype(np.float32, copy=False),
        value_masks=dataset.value_masks.astype(np.float32, copy=False),
        labels=np.asarray(dataset.labels, dtype=str),
        records_json=np.asarray(json.dumps(records, sort_keys=True)),
    )


def load_public_belief_cfv_dataset_cache(
    path: str | Path,
) -> tuple[PublicBeliefCFVDataset, list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as data:
        dataset = PublicBeliefCFVDataset(
            features=data["features"].astype(np.float32, copy=False),
            belief=data["belief"].astype(np.float32, copy=False),
            values=data["values"].astype(np.float32, copy=False),
            value_masks=data["value_masks"].astype(np.float32, copy=False),
            labels=tuple(str(label) for label in data["labels"].tolist()),
        )
        records = json.loads(str(data["records_json"].item()))
    return dataset, records


def _load_or_build_value_dataset(
    *,
    targets_npz: str | Path,
    cases_json: str | Path,
    range_checkpoint: str | Path,
    range_strategy_source: str,
    device: str | torch.device,
    solver_iterations: int,
    solver_backend: str,
    range_prune_threshold: float,
    value_scale: float,
    cache_path: str | Path | None,
) -> tuple[PublicBeliefValueDataset, list[dict[str, Any]], bool]:
    if cache_path is not None and Path(cache_path).exists():
        dataset, records = load_public_belief_value_dataset_cache(cache_path)
        return dataset, records, True

    dataset, records = load_public_belief_value_dataset(
        targets_npz=targets_npz,
        cases_json=cases_json,
        range_checkpoint=range_checkpoint,
        range_strategy_source=range_strategy_source,
        device=device,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        range_prune_threshold=range_prune_threshold,
        value_scale=value_scale,
    )
    if cache_path is not None:
        save_public_belief_value_dataset_cache(dataset, records, cache_path)
    return dataset, records, False


def _load_or_build_cfv_dataset(
    *,
    targets_npz: str | Path,
    cases_json: str | Path,
    range_checkpoint: str | Path,
    range_strategy_source: str,
    device: str | torch.device,
    solver_iterations: int,
    solver_backend: str,
    range_prune_threshold: float,
    value_scale: float,
    cache_path: str | Path | None,
) -> tuple[PublicBeliefCFVDataset, list[dict[str, Any]], bool]:
    if cache_path is not None and Path(cache_path).exists():
        dataset, records = load_public_belief_cfv_dataset_cache(cache_path)
        return dataset, records, True

    dataset, records = load_public_belief_cfv_dataset(
        targets_npz=targets_npz,
        cases_json=cases_json,
        range_checkpoint=range_checkpoint,
        range_strategy_source=range_strategy_source,
        device=device,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        range_prune_threshold=range_prune_threshold,
        value_scale=value_scale,
    )
    if cache_path is not None:
        save_public_belief_cfv_dataset_cache(dataset, records, cache_path)
    return dataset, records, False


def _standardize_targets(
    train_y: np.ndarray,
    holdout_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(np.mean(train_y))
    std = float(np.std(train_y))
    if std <= 1e-6:
        std = 1.0
    return (
        ((train_y - mean) / std).astype(np.float32),
        ((holdout_y - mean) / std).astype(np.float32),
        mean,
        std,
    )


def _fit_value_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> _ValueProbeNet:
    torch.manual_seed(seed)
    model = _ValueProbeNet(train_x.shape[1], hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_t = torch.from_numpy(train_x).to(device)
    y_t = torch.from_numpy(train_y).to(device)
    model.train()
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x_t) - y_t) ** 2)
        loss.backward()
        optimizer.step()
    return model


def _predict_values(
    model: nn.Module,
    x: np.ndarray,
    *,
    target_mean: float,
    target_std: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(device)).cpu().numpy().astype(np.float32)
    return pred * float(target_std) + float(target_mean)


def _value_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "mae": round(float(np.mean(np.abs(err))), 8),
        "rmse": round(float(np.sqrt(np.mean(err**2))), 8),
        "bias": round(float(np.mean(err)), 8),
        "target_mean": round(float(np.mean(target)), 8),
        "target_std": round(float(np.std(target)), 8),
    }


def _standardize_masked_targets(
    train_y: np.ndarray,
    holdout_y: np.ndarray,
    train_mask: np.ndarray,
    holdout_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    del holdout_mask
    selected = train_y[train_mask > 0]
    mean = float(np.mean(selected)) if selected.size else 0.0
    std = float(np.std(selected)) if selected.size else 1.0
    if std <= 1e-6:
        std = 1.0
    return (
        ((train_y - mean) / std).astype(np.float32),
        ((holdout_y - mean) / std).astype(np.float32),
        mean,
        std,
    )


def _standardization_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(x, dtype=np.float32).mean(axis=0, keepdims=True)
    std = np.asarray(x, dtype=np.float32).std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32, copy=False)
    return mean.astype(np.float32, copy=False), std


def _apply_standardization(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) - mean) / std).astype(np.float32)


def _fit_cfv_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    *,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> _CFVProbeNet:
    torch.manual_seed(seed)
    model = _CFVProbeNet(train_x.shape[1], hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x_t = torch.from_numpy(train_x).to(device)
    y_t = torch.from_numpy(train_y).to(device)
    mask_t = torch.from_numpy(train_mask).to(device)
    denom = mask_t.sum().clamp(min=1.0)
    model.train()
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_t)
        loss = ((pred - y_t) ** 2 * mask_t).sum() / denom
        loss.backward()
        optimizer.step()
    return model


def _predict_cfv(
    model: nn.Module,
    x: np.ndarray,
    *,
    target_mean: float,
    target_std: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(device)).cpu().numpy().astype(np.float32)
    return pred * float(target_std) + float(target_mean)


def _cfv_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = mask > 0
    err = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if not np.any(selected):
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0, "target_mean": 0.0, "target_std": 0.0}
    err_selected = err[selected]
    target_selected = np.asarray(target, dtype=np.float64)[selected]
    return {
        "mae": round(float(np.mean(np.abs(err_selected))), 8),
        "rmse": round(float(np.sqrt(np.mean(err_selected**2))), 8),
        "bias": round(float(np.mean(err_selected)), 8),
        "target_mean": round(float(np.mean(target_selected)), 8),
        "target_std": round(float(np.std(target_selected)), 8),
    }


def _fit_hand_cfv_probe(
    dataset: PublicBeliefCFVDataset,
    target_z: np.ndarray,
    *,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    use_belief: bool,
) -> _HandCFVProbeNet:
    torch.manual_seed(seed)
    case_idx, hand_idx = np.nonzero(dataset.value_masks > 0)
    if case_idx.size == 0:
        raise ValueError("cannot train hand-CFV probe without valid labels")
    case_idx_t = torch.from_numpy(case_idx.astype(np.int64)).to(device)
    hand_idx_t = torch.from_numpy(hand_idx.astype(np.int64)).to(device)
    public_t = torch.from_numpy(dataset.features).to(device)
    belief_t = torch.from_numpy(dataset.belief).to(device)
    hand_t = torch.from_numpy(_HAND_FEATURES).to(device)
    target_t = torch.from_numpy(target_z[case_idx, hand_idx].astype(np.float32)).to(device)

    model = _HandCFVProbeNet(hidden_dim, use_belief=use_belief).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = int(case_idx_t.numel())
    batch_size = max(1, min(int(batch_size), n))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    model.train()
    for _ in range(max(1, int(epochs))):
        perm = torch.randperm(n, generator=generator, device=device)
        for start in range(0, n, batch_size):
            batch = perm[start : start + batch_size]
            c = case_idx_t.index_select(0, batch)
            h = hand_idx_t.index_select(0, batch)
            belief_batch = belief_t.index_select(0, c) if use_belief else None
            pred = model(
                public_t.index_select(0, c),
                hand_t.index_select(0, h),
                belief_batch,
            )
            loss = torch.mean((pred - target_t.index_select(0, batch)) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def _predict_hand_cfv(
    model: _HandCFVProbeNet,
    dataset: PublicBeliefCFVDataset,
    *,
    target_mean: float,
    target_std: float,
    batch_size: int,
    device: torch.device,
    use_belief: bool,
) -> np.ndarray:
    case_idx, hand_idx = np.nonzero(dataset.value_masks > 0)
    pred = np.zeros_like(dataset.values, dtype=np.float32)
    if case_idx.size == 0:
        return pred
    case_idx_t = torch.from_numpy(case_idx.astype(np.int64)).to(device)
    hand_idx_t = torch.from_numpy(hand_idx.astype(np.int64)).to(device)
    public_t = torch.from_numpy(dataset.features).to(device)
    belief_t = torch.from_numpy(dataset.belief).to(device)
    hand_t = torch.from_numpy(_HAND_FEATURES).to(device)
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(case_idx_t.numel()), max(1, int(batch_size))):
            c = case_idx_t[start : start + batch_size]
            h = hand_idx_t[start : start + batch_size]
            belief_batch = belief_t.index_select(0, c) if use_belief else None
            out = model(
                public_t.index_select(0, c),
                hand_t.index_select(0, h),
                belief_batch,
            )
            outputs.append(out.cpu().numpy().astype(np.float32))
    values = np.concatenate(outputs, axis=0) * float(target_std) + float(target_mean)
    pred[case_idx, hand_idx] = values
    return pred


def run_public_belief_value_probe(
    *,
    train_targets_npz: str | Path,
    train_cases_json: str | Path,
    holdout_targets_npz: str | Path,
    holdout_cases_json: str | Path,
    range_checkpoint: str | Path,
    range_strategy_source: str = "regret",
    device: str | torch.device = "auto",
    solver_iterations: int = 25,
    solver_backend: str = "auto",
    range_prune_threshold: float = 1e-4,
    value_scale: float = 20000.0,
    hidden_dim: int = 64,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 0,
    train_value_cache: str | Path | None = None,
    holdout_value_cache: str | Path | None = None,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    train, train_records, train_loaded_from_cache = _load_or_build_value_dataset(
        targets_npz=train_targets_npz,
        cases_json=train_cases_json,
        range_checkpoint=range_checkpoint,
        range_strategy_source=range_strategy_source,
        device=resolved_device,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        range_prune_threshold=range_prune_threshold,
        value_scale=value_scale,
        cache_path=train_value_cache,
    )
    holdout, holdout_records, holdout_loaded_from_cache = _load_or_build_value_dataset(
        targets_npz=holdout_targets_npz,
        cases_json=holdout_cases_json,
        range_checkpoint=range_checkpoint,
        range_strategy_source=range_strategy_source,
        device=resolved_device,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        range_prune_threshold=range_prune_threshold,
        value_scale=value_scale,
        cache_path=holdout_value_cache,
    )

    train_base, holdout_base = _standardize_pair(train.features, holdout.features)
    train_belief_input, holdout_belief_input = _standardize_pair(
        np.concatenate([train.features, train.belief], axis=1),
        np.concatenate([holdout.features, holdout.belief], axis=1),
    )
    train_y, _, target_mean, target_std = _standardize_targets(train.values, holdout.values)

    base_model = _fit_value_probe(
        train_base,
        train_y,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=resolved_device,
    )
    belief_model = _fit_value_probe(
        train_belief_input,
        train_y,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=resolved_device,
    )

    base_holdout_pred = _predict_values(
        base_model,
        holdout_base,
        target_mean=target_mean,
        target_std=target_std,
        device=resolved_device,
    )
    belief_holdout_pred = _predict_values(
        belief_model,
        holdout_belief_input,
        target_mean=target_mean,
        target_std=target_std,
        device=resolved_device,
    )
    base_holdout = _value_metrics(base_holdout_pred, holdout.values)
    belief_holdout = _value_metrics(belief_holdout_pred, holdout.values)
    holdout_mae_delta = round(float(base_holdout["mae"] - belief_holdout["mae"]), 8)
    holdout_rmse_delta = round(float(base_holdout["rmse"] - belief_holdout["rmse"]), 8)
    return {
        "mode": "public_belief_value_probe",
        "passed": bool(holdout_mae_delta > 0 and holdout_rmse_delta >= 0),
        "pass_criteria": "belief_holdout must improve MAE and not worsen RMSE",
        "device": str(resolved_device),
        "solver_iterations": int(solver_iterations),
        "solver_backend": solver_backend,
        "range_prune_threshold": float(range_prune_threshold),
        "value_scale": float(value_scale),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "range_checkpoint": str(range_checkpoint),
        "range_strategy_source": range_strategy_source,
        "train_value_cache": str(train_value_cache) if train_value_cache else None,
        "holdout_value_cache": str(holdout_value_cache) if holdout_value_cache else None,
        "train_loaded_from_cache": bool(train_loaded_from_cache),
        "holdout_loaded_from_cache": bool(holdout_loaded_from_cache),
        "train_size": int(train.features.shape[0]),
        "holdout_size": int(holdout.features.shape[0]),
        "feature_dim": int(N_FEATURES),
        "belief_dim": int(BELIEF_DIM),
        "base_holdout": base_holdout,
        "belief_holdout": belief_holdout,
        "holdout_mae_delta": holdout_mae_delta,
        "holdout_rmse_delta": holdout_rmse_delta,
        "train_value_records": train_records,
        "holdout_value_records": holdout_records,
    }


def run_public_belief_cfv_probe(
    *,
    train_targets_npz: str | Path,
    train_cases_json: str | Path,
    holdout_targets_npz: str | Path,
    holdout_cases_json: str | Path,
    range_checkpoint: str | Path,
    range_strategy_source: str = "regret",
    device: str | torch.device = "auto",
    solver_iterations: int = 25,
    solver_backend: str = "auto",
    range_prune_threshold: float = 0.0,
    value_scale: float = 20000.0,
    hidden_dim: int = 64,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 0,
    train_cfv_cache: str | Path | None = None,
    holdout_cfv_cache: str | Path | None = None,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    train, train_records, train_loaded_from_cache = _load_or_build_cfv_dataset(
        targets_npz=train_targets_npz,
        cases_json=train_cases_json,
        range_checkpoint=range_checkpoint,
        range_strategy_source=range_strategy_source,
        device=resolved_device,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        range_prune_threshold=range_prune_threshold,
        value_scale=value_scale,
        cache_path=train_cfv_cache,
    )
    holdout, holdout_records, holdout_loaded_from_cache = _load_or_build_cfv_dataset(
        targets_npz=holdout_targets_npz,
        cases_json=holdout_cases_json,
        range_checkpoint=range_checkpoint,
        range_strategy_source=range_strategy_source,
        device=resolved_device,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        range_prune_threshold=range_prune_threshold,
        value_scale=value_scale,
        cache_path=holdout_cfv_cache,
    )

    train_base, holdout_base = _standardize_pair(train.features, holdout.features)
    train_belief_input, holdout_belief_input = _standardize_pair(
        np.concatenate([train.features, train.belief], axis=1),
        np.concatenate([holdout.features, holdout.belief], axis=1),
    )
    train_y, _, target_mean, target_std = _standardize_masked_targets(
        train.values,
        holdout.values,
        train.value_masks,
        holdout.value_masks,
    )

    base_model = _fit_cfv_probe(
        train_base,
        train_y,
        train.value_masks,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=resolved_device,
    )
    belief_model = _fit_cfv_probe(
        train_belief_input,
        train_y,
        train.value_masks,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=resolved_device,
    )

    base_holdout_pred = _predict_cfv(
        base_model,
        holdout_base,
        target_mean=target_mean,
        target_std=target_std,
        device=resolved_device,
    )
    belief_holdout_pred = _predict_cfv(
        belief_model,
        holdout_belief_input,
        target_mean=target_mean,
        target_std=target_std,
        device=resolved_device,
    )
    base_holdout = _cfv_metrics(base_holdout_pred, holdout.values, holdout.value_masks)
    belief_holdout = _cfv_metrics(
        belief_holdout_pred,
        holdout.values,
        holdout.value_masks,
    )
    holdout_mae_delta = round(float(base_holdout["mae"] - belief_holdout["mae"]), 8)
    holdout_rmse_delta = round(float(base_holdout["rmse"] - belief_holdout["rmse"]), 8)
    return {
        "mode": "public_belief_cfv_probe",
        "passed": bool(holdout_mae_delta > 0 and holdout_rmse_delta >= 0),
        "pass_criteria": "belief_holdout must improve masked CFV MAE and not worsen RMSE",
        "device": str(resolved_device),
        "solver_iterations": int(solver_iterations),
        "solver_backend": solver_backend,
        "range_prune_threshold": float(range_prune_threshold),
        "value_scale": float(value_scale),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "range_checkpoint": str(range_checkpoint),
        "range_strategy_source": range_strategy_source,
        "train_cfv_cache": str(train_cfv_cache) if train_cfv_cache else None,
        "holdout_cfv_cache": str(holdout_cfv_cache) if holdout_cfv_cache else None,
        "train_loaded_from_cache": bool(train_loaded_from_cache),
        "holdout_loaded_from_cache": bool(holdout_loaded_from_cache),
        "train_size": int(train.features.shape[0]),
        "holdout_size": int(holdout.features.shape[0]),
        "feature_dim": int(N_FEATURES),
        "belief_dim": int(BELIEF_DIM),
        "target_dim": int(N_HANDS),
        "train_mask_count": int(train.value_masks.sum()),
        "holdout_mask_count": int(holdout.value_masks.sum()),
        "base_holdout": base_holdout,
        "belief_holdout": belief_holdout,
        "holdout_mae_delta": holdout_mae_delta,
        "holdout_rmse_delta": holdout_rmse_delta,
        "train_cfv_records": train_records,
        "holdout_cfv_records": holdout_records,
    }


def run_public_belief_hand_cfv_probe(
    *,
    train_targets_npz: str | Path,
    train_cases_json: str | Path,
    holdout_targets_npz: str | Path,
    holdout_cases_json: str | Path,
    range_checkpoint: str | Path,
    range_strategy_source: str = "regret",
    device: str | torch.device = "auto",
    solver_iterations: int = 25,
    solver_backend: str = "auto",
    range_prune_threshold: float = 0.0,
    value_scale: float = 20000.0,
    hidden_dim: int = 64,
    epochs: int = 30,
    batch_size: int = 8192,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 0,
    train_cfv_cache: str | Path | None = None,
    holdout_cfv_cache: str | Path | None = None,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    train, train_records, train_loaded_from_cache = _load_or_build_cfv_dataset(
        targets_npz=train_targets_npz,
        cases_json=train_cases_json,
        range_checkpoint=range_checkpoint,
        range_strategy_source=range_strategy_source,
        device=resolved_device,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        range_prune_threshold=range_prune_threshold,
        value_scale=value_scale,
        cache_path=train_cfv_cache,
    )
    holdout, holdout_records, holdout_loaded_from_cache = _load_or_build_cfv_dataset(
        targets_npz=holdout_targets_npz,
        cases_json=holdout_cases_json,
        range_checkpoint=range_checkpoint,
        range_strategy_source=range_strategy_source,
        device=resolved_device,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        range_prune_threshold=range_prune_threshold,
        value_scale=value_scale,
        cache_path=holdout_cfv_cache,
    )

    train_public, holdout_public = _standardize_pair(train.features, holdout.features)
    train_belief, holdout_belief = _standardize_pair(train.belief, holdout.belief)
    train = PublicBeliefCFVDataset(
        features=train_public,
        belief=train_belief,
        values=train.values,
        value_masks=train.value_masks,
        labels=train.labels,
    )
    holdout = PublicBeliefCFVDataset(
        features=holdout_public,
        belief=holdout_belief,
        values=holdout.values,
        value_masks=holdout.value_masks,
        labels=holdout.labels,
    )
    train_y, _, target_mean, target_std = _standardize_masked_targets(
        train.values,
        holdout.values,
        train.value_masks,
        holdout.value_masks,
    )

    base_model = _fit_hand_cfv_probe(
        train,
        train_y,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=resolved_device,
        use_belief=False,
    )
    belief_model = _fit_hand_cfv_probe(
        train,
        train_y,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=resolved_device,
        use_belief=True,
    )
    base_pred = _predict_hand_cfv(
        base_model,
        holdout,
        target_mean=target_mean,
        target_std=target_std,
        batch_size=batch_size,
        device=resolved_device,
        use_belief=False,
    )
    belief_pred = _predict_hand_cfv(
        belief_model,
        holdout,
        target_mean=target_mean,
        target_std=target_std,
        batch_size=batch_size,
        device=resolved_device,
        use_belief=True,
    )
    base_holdout = _cfv_metrics(base_pred, holdout.values, holdout.value_masks)
    belief_holdout = _cfv_metrics(belief_pred, holdout.values, holdout.value_masks)
    holdout_mae_delta = round(float(base_holdout["mae"] - belief_holdout["mae"]), 8)
    holdout_rmse_delta = round(float(base_holdout["rmse"] - belief_holdout["rmse"]), 8)
    return {
        "mode": "public_belief_hand_cfv_probe",
        "passed": bool(holdout_mae_delta > 0 and holdout_rmse_delta >= 0),
        "pass_criteria": "belief_holdout must improve shared hand-CFV MAE and not worsen RMSE",
        "device": str(resolved_device),
        "solver_iterations": int(solver_iterations),
        "solver_backend": solver_backend,
        "range_prune_threshold": float(range_prune_threshold),
        "value_scale": float(value_scale),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "range_checkpoint": str(range_checkpoint),
        "range_strategy_source": range_strategy_source,
        "train_cfv_cache": str(train_cfv_cache) if train_cfv_cache else None,
        "holdout_cfv_cache": str(holdout_cfv_cache) if holdout_cfv_cache else None,
        "train_loaded_from_cache": bool(train_loaded_from_cache),
        "holdout_loaded_from_cache": bool(holdout_loaded_from_cache),
        "train_size": int(train.features.shape[0]),
        "holdout_size": int(holdout.features.shape[0]),
        "feature_dim": int(N_FEATURES),
        "belief_dim": int(BELIEF_DIM),
        "hand_feature_dim": 52,
        "target_dim": int(N_HANDS),
        "train_mask_count": int(train.value_masks.sum()),
        "holdout_mask_count": int(holdout.value_masks.sum()),
        "base_holdout": base_holdout,
        "belief_holdout": belief_holdout,
        "holdout_mae_delta": holdout_mae_delta,
        "holdout_rmse_delta": holdout_rmse_delta,
        "train_cfv_records": train_records,
        "holdout_cfv_records": holdout_records,
    }


def train_public_belief_hand_cfv_checkpoint(
    *,
    train_targets_npz: str | Path,
    train_cases_json: str | Path,
    holdout_targets_npz: str | Path,
    holdout_cases_json: str | Path,
    range_checkpoint: str | Path,
    output_checkpoint: str | Path,
    range_strategy_source: str = "regret",
    device: str | torch.device = "auto",
    solver_iterations: int = 25,
    solver_backend: str = "auto",
    range_prune_threshold: float = 0.0,
    value_scale: float = 20000.0,
    hidden_dim: int = 64,
    epochs: int = 30,
    batch_size: int = 8192,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 0,
    train_cfv_cache: str | Path | None = None,
    holdout_cfv_cache: str | Path | None = None,
) -> dict[str, Any]:
    """Train and save the belief-conditioned shared hand-CFV model."""
    resolved_device = _resolve_device(device)
    train, train_records, train_loaded_from_cache = _load_or_build_cfv_dataset(
        targets_npz=train_targets_npz,
        cases_json=train_cases_json,
        range_checkpoint=range_checkpoint,
        range_strategy_source=range_strategy_source,
        device=resolved_device,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        range_prune_threshold=range_prune_threshold,
        value_scale=value_scale,
        cache_path=train_cfv_cache,
    )
    holdout, holdout_records, holdout_loaded_from_cache = _load_or_build_cfv_dataset(
        targets_npz=holdout_targets_npz,
        cases_json=holdout_cases_json,
        range_checkpoint=range_checkpoint,
        range_strategy_source=range_strategy_source,
        device=resolved_device,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        range_prune_threshold=range_prune_threshold,
        value_scale=value_scale,
        cache_path=holdout_cfv_cache,
    )

    public_mean, public_std = _standardization_stats(train.features)
    belief_mean, belief_std = _standardization_stats(train.belief)
    train = PublicBeliefCFVDataset(
        features=_apply_standardization(train.features, public_mean, public_std),
        belief=_apply_standardization(train.belief, belief_mean, belief_std),
        values=train.values,
        value_masks=train.value_masks,
        labels=train.labels,
    )
    holdout = PublicBeliefCFVDataset(
        features=_apply_standardization(holdout.features, public_mean, public_std),
        belief=_apply_standardization(holdout.belief, belief_mean, belief_std),
        values=holdout.values,
        value_masks=holdout.value_masks,
        labels=holdout.labels,
    )
    train_y, _, target_mean, target_std = _standardize_masked_targets(
        train.values,
        holdout.values,
        train.value_masks,
        holdout.value_masks,
    )
    model = _fit_hand_cfv_probe(
        train,
        train_y,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=resolved_device,
        use_belief=True,
    )
    holdout_pred = _predict_hand_cfv(
        model,
        holdout,
        target_mean=target_mean,
        target_std=target_std,
        batch_size=batch_size,
        device=resolved_device,
        use_belief=True,
    )
    holdout_metrics = _cfv_metrics(holdout_pred, holdout.values, holdout.value_masks)

    output_path = Path(output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mode": "public_belief_hand_cfv_checkpoint",
            "model_state": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "hidden_dim": int(hidden_dim),
            "use_belief": True,
            "feature_dim": int(N_FEATURES),
            "belief_dim": int(BELIEF_DIM),
            "hand_feature_dim": 52,
            "target_dim": int(N_HANDS),
            "public_mean": public_mean,
            "public_std": public_std,
            "belief_mean": belief_mean,
            "belief_std": belief_std,
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "value_scale": float(value_scale),
            "range_checkpoint": str(range_checkpoint),
            "range_strategy_source": range_strategy_source,
            "solver_iterations": int(solver_iterations),
            "solver_backend": solver_backend,
            "range_prune_threshold": float(range_prune_threshold),
            "train_targets_npz": str(train_targets_npz),
            "train_cases_json": str(train_cases_json),
            "train_cfv_cache": str(train_cfv_cache) if train_cfv_cache else None,
            "seed": int(seed),
        },
        output_path,
    )

    return {
        "mode": "public_belief_hand_cfv_checkpoint_train",
        "passed": bool(np.isfinite(holdout_metrics["mae"])),
        "checkpoint": str(output_path),
        "device": str(resolved_device),
        "solver_iterations": int(solver_iterations),
        "solver_backend": solver_backend,
        "range_prune_threshold": float(range_prune_threshold),
        "value_scale": float(value_scale),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "range_checkpoint": str(range_checkpoint),
        "range_strategy_source": range_strategy_source,
        "train_cfv_cache": str(train_cfv_cache) if train_cfv_cache else None,
        "holdout_cfv_cache": str(holdout_cfv_cache) if holdout_cfv_cache else None,
        "train_loaded_from_cache": bool(train_loaded_from_cache),
        "holdout_loaded_from_cache": bool(holdout_loaded_from_cache),
        "train_size": int(train.features.shape[0]),
        "holdout_size": int(holdout.features.shape[0]),
        "feature_dim": int(N_FEATURES),
        "belief_dim": int(BELIEF_DIM),
        "hand_feature_dim": 52,
        "target_dim": int(N_HANDS),
        "train_mask_count": int(train.value_masks.sum()),
        "holdout_mask_count": int(holdout.value_masks.sum()),
        "belief_holdout": holdout_metrics,
        "train_cfv_record_count": len(train_records),
        "holdout_cfv_record_count": len(holdout_records),
    }


def load_public_belief_hand_cfv_checkpoint(
    checkpoint: str | Path,
    device: str | torch.device = "auto",
) -> tuple[_HandCFVProbeNet, dict[str, Any]]:
    """Load a saved belief-conditioned hand-CFV checkpoint."""
    resolved_device = _resolve_device(device)
    payload = torch.load(checkpoint, map_location=resolved_device, weights_only=False)
    if payload.get("mode") != "public_belief_hand_cfv_checkpoint":
        raise ValueError(f"{checkpoint} is not a public-belief hand-CFV checkpoint")
    model = _HandCFVProbeNet(int(payload["hidden_dim"]), use_belief=True).to(resolved_device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def predict_public_belief_hand_cfv_checkpoint(
    checkpoint: str | Path,
    features: np.ndarray,
    belief: np.ndarray,
    value_masks: np.ndarray | None = None,
    *,
    device: str | torch.device = "auto",
    batch_size: int = 8192,
) -> np.ndarray:
    """Predict global hand CFVs from a saved public-belief hand-CFV checkpoint."""
    model, payload = load_public_belief_hand_cfv_checkpoint(checkpoint, device=device)
    return predict_public_belief_hand_cfv_model(
        model,
        payload,
        features,
        belief,
        value_masks,
        device=device,
        batch_size=batch_size,
    )


def predict_public_belief_hand_cfv_model(
    model: _HandCFVProbeNet,
    payload: dict[str, Any],
    features: np.ndarray,
    belief: np.ndarray,
    value_masks: np.ndarray | None = None,
    *,
    device: str | torch.device = "auto",
    batch_size: int = 8192,
) -> np.ndarray:
    """Predict global hand CFVs from a loaded hand-CFV model payload."""
    features = np.asarray(features, dtype=np.float32)
    belief = np.asarray(belief, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if belief.ndim == 1:
        belief = belief.reshape(1, -1)
    if features.shape[0] != belief.shape[0]:
        raise ValueError("features and belief must have the same number of rows")
    if value_masks is None:
        value_masks = np.ones((features.shape[0], N_HANDS), dtype=np.float32)
    else:
        value_masks = np.asarray(value_masks, dtype=np.float32)
    dataset = PublicBeliefCFVDataset(
        features=_apply_standardization(features, payload["public_mean"], payload["public_std"]),
        belief=_apply_standardization(belief, payload["belief_mean"], payload["belief_std"]),
        values=np.zeros((features.shape[0], N_HANDS), dtype=np.float32),
        value_masks=value_masks,
        labels=tuple(f"predict-{idx}" for idx in range(features.shape[0])),
    )
    return _predict_hand_cfv(
        model,
        dataset,
        target_mean=float(payload["target_mean"]),
        target_std=float(payload["target_std"]),
        batch_size=batch_size,
        device=_resolve_device(device),
        use_belief=True,
    )


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
