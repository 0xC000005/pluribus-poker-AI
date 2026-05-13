#!/usr/bin/env python3
"""Build dual-CFV targets from the actual CFR leaf-callback distribution."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.research.belief_probe import N_HANDS, _HAND_TO_INDEX
from poker_ai.research.belief_value_probe import load_public_belief_cfv_dataset_cache, save_metrics
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase, load_cases_json

from eval_joint_pbs_resolver_leaf_ab import (  # noqa: E402
    _action_path_to_street_string,
    _card_to_str,
    _global_board_mask,
    _local_ranges_from_belief,
    _normalize,
    _pre_street_prefix,
)
from eval_public_belief_dual_hand_cfv_probe import DualCFVDataset, _save_dual_cache  # noqa: E402
from play_slumbot import _compute_bets_before_street, build_features, card_str_to_index, parse_action  # noqa: E402
from solver import StreetSolver, resolve_solver_backend  # noqa: E402


def _leaf_evs_from_numerators(
    *,
    default_hero_values: np.ndarray,
    default_villain_values: np.ndarray,
    valid_m: np.ndarray,
    hero_reach: np.ndarray,
    villain_reach: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert CFR leaf numerators into per-hand EV labels and masks."""
    default_hero_values = np.asarray(default_hero_values, dtype=np.float32)
    default_villain_values = np.asarray(default_villain_values, dtype=np.float32)
    valid_m = np.asarray(valid_m, dtype=np.float32)
    hero_reach = np.asarray(hero_reach, dtype=np.float32)
    villain_reach = np.asarray(villain_reach, dtype=np.float32)
    hero_den = valid_m @ villain_reach
    villain_den = valid_m.T @ hero_reach
    hero_values = np.divide(
        default_hero_values,
        hero_den,
        out=np.zeros_like(default_hero_values, dtype=np.float32),
        where=hero_den > 1e-12,
    )
    villain_values = np.divide(
        default_villain_values,
        villain_den,
        out=np.zeros_like(default_villain_values, dtype=np.float32),
        where=villain_den > 1e-12,
    )
    return (
        hero_values.astype(np.float32, copy=False),
        villain_values.astype(np.float32, copy=False),
        (hero_den > 1e-12).astype(np.float32),
        (villain_den > 1e-12).astype(np.float32),
    )


@dataclass
class CallbackLeafCollector:
    case: ResolverBenchmarkCase
    board4: list[int]
    action_prefix: str
    value_scale: float
    rng: np.random.Generator
    max_states: int = 0
    features: list[np.ndarray] = field(default_factory=list)
    beliefs: list[np.ndarray] = field(default_factory=list)
    hero_values: list[np.ndarray] = field(default_factory=list)
    villain_values: list[np.ndarray] = field(default_factory=list)
    hero_masks: list[np.ndarray] = field(default_factory=list)
    villain_masks: list[np.ndarray] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    callback_calls: int = 0
    states_seen: int = 0
    fallback_showdowns: int = 0

    def __post_init__(self) -> None:
        self.board_str = [_card_to_str(card) for card in self.board4]
        self.board_mask = _global_board_mask(self.board4)

    def __call__(self, **kwargs):
        tree = kwargs["tree"]
        showdown_indices = np.asarray(kwargs["showdown_indices"], dtype=np.int32)
        hero_reach = np.asarray(kwargs["hero_reach"], dtype=np.float32)
        villain_reach = np.asarray(kwargs["villain_reach"], dtype=np.float32)
        valid_m = np.asarray(kwargs["valid_m"], dtype=np.float32)
        default_h = np.asarray(kwargs["default_hero_values"], dtype=np.float32)
        default_v = np.asarray(kwargs["default_villain_values"], dtype=np.float32)
        self.callback_calls += 1

        for row, node_idx in enumerate(showdown_indices.tolist()):
            action_str = self._terminal_action_str(tree, int(node_idx))
            parsed = parse_action(action_str)
            if "error" in parsed:
                self.fallback_showdowns += 1
                continue
            hero_belief = np.zeros(N_HANDS, dtype=np.float32)
            villain_belief = np.zeros(N_HANDS, dtype=np.float32)
            hero_belief[self._local_to_global] = hero_reach[row]
            villain_belief[self._local_to_global] = villain_reach[row]
            hero_belief = _normalize(hero_belief * self.board_mask)
            villain_belief = _normalize(villain_belief * self.board_mask)
            hero_local, villain_local, hero_mask_local, villain_mask_local = (
                _leaf_evs_from_numerators(
                    default_hero_values=default_h[row],
                    default_villain_values=default_v[row],
                    valid_m=valid_m,
                    hero_reach=hero_reach[row],
                    villain_reach=villain_reach[row],
                )
            )
            hero_values = np.zeros(N_HANDS, dtype=np.float32)
            villain_values = np.zeros(N_HANDS, dtype=np.float32)
            hero_masks = np.zeros(N_HANDS, dtype=np.float32)
            villain_masks = np.zeros(N_HANDS, dtype=np.float32)
            hero_values[self._local_to_global] = hero_local / float(self.value_scale)
            villain_values[self._local_to_global] = villain_local / float(self.value_scale)
            hero_masks[self._local_to_global] = hero_mask_local
            villain_masks[self._local_to_global] = villain_mask_local

            record = {
                "label": f"{self.case.label}-callback-leaf-{self.states_seen:06d}",
                "root_label": str(self.case.label),
                "action_str": action_str,
                "callback_call": int(self.callback_calls),
                "showdown_node_idx": int(node_idx),
                "hero_mask_count": int(hero_masks.sum()),
                "villain_mask_count": int(villain_masks.sum()),
            }
            self._append_or_reservoir(
                feature=build_features(
                    [],
                    self.board_str,
                    action_str,
                    self.case.client_pos,
                    parsed,
                ).astype(np.float32, copy=False),
                belief=np.concatenate([hero_belief, villain_belief]).astype(np.float32),
                hero_value=hero_values,
                villain_value=villain_values,
                hero_mask=hero_masks,
                villain_mask=villain_masks,
                label=str(record["label"]),
                record=record,
            )
            self.states_seen += 1
        return default_h, default_v

    def _append_or_reservoir(
        self,
        *,
        feature: np.ndarray,
        belief: np.ndarray,
        hero_value: np.ndarray,
        villain_value: np.ndarray,
        hero_mask: np.ndarray,
        villain_mask: np.ndarray,
        label: str,
        record: dict[str, Any],
    ) -> None:
        max_states = int(self.max_states)
        if max_states <= 0 or len(self.features) < max_states:
            self.features.append(feature)
            self.beliefs.append(belief)
            self.hero_values.append(hero_value)
            self.villain_values.append(villain_value)
            self.hero_masks.append(hero_mask)
            self.villain_masks.append(villain_mask)
            self.labels.append(label)
            self.records.append(record)
            return
        replacement = int(self.rng.integers(0, self.states_seen + 1))
        if replacement >= max_states:
            return
        self.features[replacement] = feature
        self.beliefs[replacement] = belief
        self.hero_values[replacement] = hero_value
        self.villain_values[replacement] = villain_value
        self.hero_masks[replacement] = hero_mask
        self.villain_masks[replacement] = villain_mask
        self.labels[replacement] = label
        self.records[replacement] = record

    def _terminal_action_str(self, tree: dict[str, Any], node_idx: int) -> str:
        street_action = _action_path_to_street_string(
            tree,
            node_idx,
            hero_stack_start=int(tree["stacks_h"][0]),
            villain_stack_start=int(tree["stacks_v"][0]),
        )
        return f"{self.action_prefix}/{street_action}" if self.action_prefix else street_action

    @property
    def solver_hands(self) -> list[tuple[int, int]]:
        return self._solver_hands

    @solver_hands.setter
    def solver_hands(self, hands: list[tuple[int, int]]) -> None:
        self._solver_hands = hands
        self._local_to_global = np.array(
            [_HAND_TO_INDEX[tuple(sorted(hand))] for hand in hands],
            dtype=np.int32,
        )


def _collect_case(
    case: ResolverBenchmarkCase,
    *,
    belief_row: np.ndarray,
    solver_iterations: int,
    solver_backend: str,
    value_scale: float,
    max_states_per_case: int,
    seed: int,
) -> CallbackLeafCollector | dict[str, Any]:
    parsed = parse_action(case.action_str)
    if "error" in parsed:
        return {"label": case.label, "skipped": parsed["error"]}
    street = int(parsed.get("st", -1))
    if street != 2:
        return {"label": case.label, "skipped": f"street:{street}"}
    board_idx = [card_str_to_index(card) for card in case.board[:4]]
    our_bet_pre, opp_bet_pre = _compute_bets_before_street(
        case.action_str,
        case.client_pos,
        target_street=street,
    )
    pot = our_bet_pre + opp_bet_pre
    hero_stack = 20000 - our_bet_pre
    villain_stack = 20000 - opp_bet_pre
    hero_first = case.client_pos == 0
    action_prefix = _pre_street_prefix(case.action_str, street)
    full_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
    hero_range, villain_range = _local_ranges_from_belief(belief_row, full_hands)
    backend, backend_device = resolve_solver_backend(solver_backend)
    if backend != "cpu":
        raise ValueError("callback leaf target generation requires CPU backend")
    solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    collector = CallbackLeafCollector(
        case=case,
        board4=board_idx,
        action_prefix=action_prefix,
        value_scale=value_scale,
        rng=np.random.default_rng(int(seed)),
        max_states=int(max_states_per_case),
    )
    collector.solver_hands = list(solver.hands)
    solver.solve(
        n_iterations=int(solver_iterations),
        hero_range=hero_range,
        villain_range=villain_range,
        backend=backend,
        device=backend_device,
        showdown_leaf_fn=collector,
    )
    return collector


def build_callback_leaf_targets(
    *,
    cases_json: str | Path,
    cfv_cache: str | Path,
    output: str | Path,
    start_index: int = 0,
    limit: int = 8,
    solver_iterations: int = 5,
    solver_backend: str = "cpu",
    value_scale: float = 20000.0,
    max_states_per_case: int = 2048,
    seed: int = 0,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    cases = load_cases_json(cases_json)
    base_dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if base_dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")
    start = max(0, min(int(start_index), len(cases)))
    n = max(0, min(int(limit), len(cases) - start))
    if n <= 0:
        raise ValueError("selected callback leaf target slice is empty")
    features: list[np.ndarray] = []
    beliefs: list[np.ndarray] = []
    hero_values: list[np.ndarray] = []
    villain_values: list[np.ndarray] = []
    hero_masks: list[np.ndarray] = []
    villain_masks: list[np.ndarray] = []
    labels: list[str] = []
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []
    for local_idx, case in enumerate(cases[start : start + n]):
        case_idx = start + local_idx
        result = _collect_case(
            case,
            belief_row=base_dataset.belief[case_idx],
            solver_iterations=solver_iterations,
            solver_backend=solver_backend,
            value_scale=value_scale,
            max_states_per_case=max_states_per_case,
            seed=int(seed) + case_idx,
        )
        if isinstance(result, dict):
            skipped.append(result)
            continue
        features.extend(result.features)
        beliefs.extend(result.beliefs)
        hero_values.extend(result.hero_values)
        villain_values.extend(result.villain_values)
        hero_masks.extend(result.hero_masks)
        villain_masks.extend(result.villain_masks)
        labels.extend(result.labels)
        records.extend(result.records)
        per_case.append(
            {
                "label": case.label,
                "states_seen": int(result.states_seen),
                "states_kept": int(len(result.labels)),
                "callback_calls": int(result.callback_calls),
                "fallback_showdowns": int(result.fallback_showdowns),
            }
        )
    if not features:
        raise RuntimeError("no callback leaf states were collected")
    dataset = DualCFVDataset(
        features=np.stack(features).astype(np.float32, copy=False),
        belief=np.stack(beliefs).astype(np.float32, copy=False),
        hero_values=np.stack(hero_values).astype(np.float32, copy=False),
        villain_values=np.stack(villain_values).astype(np.float32, copy=False),
        hero_masks=np.stack(hero_masks).astype(np.float32, copy=False),
        villain_masks=np.stack(villain_masks).astype(np.float32, copy=False),
        labels=tuple(labels),
    )
    _save_dual_cache(dataset, records, output)
    metrics = {
        "mode": "public_belief_callback_leaf_targets",
        "passed": True,
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "output": str(output),
        "start_index": int(start),
        "limit": int(n),
        "solver_iterations": int(solver_iterations),
        "solver_backend": str(solver_backend),
        "value_scale": float(value_scale),
        "max_states_per_case": int(max_states_per_case),
        "seed": int(seed),
        "n_cases": int(n),
        "n_skipped": int(len(skipped)),
        "n_states": int(dataset.features.shape[0]),
        "hero_label_count": int(dataset.hero_masks.sum()),
        "villain_label_count": int(dataset.villain_masks.sum()),
        "per_case": per_case,
        "skipped": skipped[:20],
    }
    if output_json is not None:
        save_metrics(metrics, output_json)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build dual-CFV labels from states actually queried by CFR leaf callbacks."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--solver-iterations", type=int, default=5)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--value-scale", type=float, default=20000.0)
    parser.add_argument("--max-states-per-case", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = build_callback_leaf_targets(
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        output=args.output,
        start_index=args.start_index,
        limit=args.limit,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        value_scale=args.value_scale,
        max_states_per_case=args.max_states_per_case,
        seed=args.seed,
        output_json=args.output_json,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
