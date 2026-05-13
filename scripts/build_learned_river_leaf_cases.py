#!/usr/bin/env python3
"""Export river leaf states reached by the turn resolver for CFV checking."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
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
from poker_ai.research.belief_value_probe import (
    PublicBeliefCFVDataset,
    load_public_belief_cfv_dataset_cache,
    save_public_belief_cfv_dataset_cache,
)
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase, load_cases_json
from poker_ai.research.search_targets import save_cases_json

from eval_learned_river_leaf_resolver_ab import (  # noqa: E402
    _action_path_to_street_string,
    _card_to_str,
    _global_board_mask,
    _is_descendant,
    _local_ranges_from_belief,
    _normalize,
    _pre_street_prefix,
)
from play_slumbot import (  # noqa: E402
    _compute_bets_before_street,
    build_features,
    card_str_to_index,
    parse_action,
)
from solver import StreetSolver, _parse_nav, resolve_solver_backend  # noqa: E402


class _RiverLeafCollector:
    def __init__(
        self,
        *,
        source_case: ResolverBenchmarkCase,
        board4: list[int],
        action_prefix: str,
        active_subtree_node_idx: int,
        solver_hands: list[tuple[int, int]],
        max_leaf_states: int,
        max_rivers_per_terminal: int = 0,
    ):
        self.source_case = source_case
        self.board4 = [int(card) for card in board4]
        self.action_prefix = action_prefix
        self.active_subtree_node_idx = int(active_subtree_node_idx)
        self.solver_hands = list(solver_hands)
        self.max_leaf_states = int(max_leaf_states)
        self.max_rivers_per_terminal = max(0, int(max_rivers_per_terminal))
        self.river_cards = sorted(set(range(52)) - set(self.board4))
        self.local_to_global = np.asarray(
            [_HAND_TO_INDEX[tuple(sorted(hand))] for hand in self.solver_hands],
            dtype=np.int32,
        )
        self.features: list[np.ndarray] = []
        self.beliefs: list[np.ndarray] = []
        self.cases: list[ResolverBenchmarkCase] = []
        self.records: list[dict[str, Any]] = []

    def __call__(self, **kwargs):
        tree = kwargs["tree"]
        showdown_indices = np.asarray(kwargs["showdown_indices"], dtype=np.int32)
        hero_reach = np.asarray(kwargs["hero_reach"], dtype=np.float32)
        villain_reach = np.asarray(kwargs["villain_reach"], dtype=np.float32)
        for row, node_idx in enumerate(showdown_indices.tolist()):
            if len(self.cases) >= self.max_leaf_states:
                break
            if not _is_descendant(tree, int(node_idx), self.active_subtree_node_idx):
                continue
            action_str = self._terminal_action_str(tree, int(node_idx))
            parsed = parse_action(action_str)
            if (
                "error" in parsed
                or int(parsed.get("st", -1)) != 3
                or int(parsed.get("pos", -1)) < 0
            ):
                continue
            emitted_for_terminal = 0
            for river_card in self.river_cards:
                if len(self.cases) >= self.max_leaf_states:
                    break
                if (
                    self.max_rivers_per_terminal
                    and emitted_for_terminal >= self.max_rivers_per_terminal
                ):
                    break
                board5 = [*self.board4, int(river_card)]
                board_mask = _global_board_mask(board5)
                local_legal = np.asarray(
                    [0.0 if river_card in hand else 1.0 for hand in self.solver_hands],
                    dtype=np.float32,
                )
                hero_belief = np.zeros(N_HANDS, dtype=np.float32)
                villain_belief = np.zeros(N_HANDS, dtype=np.float32)
                hero_belief[self.local_to_global] = hero_reach[row] * local_legal
                villain_belief[self.local_to_global] = villain_reach[row] * local_legal
                hero_belief = _normalize(hero_belief * board_mask)
                villain_belief = _normalize(villain_belief * board_mask)
                board_str = [_card_to_str(card) for card in board5]
                label = f"{self.source_case.label}-river-leaf-{len(self.cases):04d}"
                self.cases.append(
                    ResolverBenchmarkCase(
                        label=label,
                        hole_cards=self.source_case.hole_cards,
                        board=tuple(board_str),
                        action_str=action_str,
                        client_pos=self.source_case.client_pos,
                        source="turn_resolver_river_leaf",
                    )
                )
                self.features.append(
                    build_features(
                        [],
                        board_str,
                        action_str,
                        self.source_case.client_pos,
                        parsed,
                    )
                )
                self.beliefs.append(
                    np.concatenate([hero_belief, villain_belief]).astype(
                        np.float32,
                        copy=False,
                    )
                )
                self.records.append(
                    {
                        "label": label,
                        "source_case": self.source_case.label,
                        "source_action_str": self.source_case.action_str,
                        "leaf_action_str": action_str,
                        "river_card": _card_to_str(river_card),
                        "terminal_node_idx": int(node_idx),
                    }
                )
                emitted_for_terminal += 1
        return kwargs["default_hero_values"], kwargs["default_villain_values"]

    def _terminal_action_str(self, tree: dict[str, Any], node_idx: int) -> str:
        street_action = _action_path_to_street_string(
            tree,
            node_idx,
            hero_stack_start=int(tree["stacks_h"][0]),
            villain_stack_start=int(tree["stacks_v"][0]),
        )
        return f"{self.action_prefix}/{street_action}/" if self.action_prefix else f"{street_action}/"


def _collect_case(
    case: ResolverBenchmarkCase,
    *,
    belief_row: np.ndarray | None,
    max_leaf_states: int,
    max_rivers_per_terminal: int,
    solver_iterations: int,
    solver_backend: str,
) -> tuple[list[ResolverBenchmarkCase], list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    parsed = parse_action(case.action_str)
    if "error" in parsed or int(parsed.get("st", -1)) != 2:
        return [], [], [], []
    board_idx = [card_str_to_index(card) for card in case.board[:4]]
    our_bet_pre, opp_bet_pre = _compute_bets_before_street(
        case.action_str,
        case.client_pos,
        target_street=2,
    )
    pot = our_bet_pre + opp_bet_pre
    hero_stack = 20000 - our_bet_pre
    villain_stack = 20000 - opp_bet_pre
    hero_first = case.client_pos == 0
    street_action = case.action_str.split("/")[2] if len(case.action_str.split("/")) > 2 else ""
    action_prefix = _pre_street_prefix(case.action_str, 2)
    full_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
    hero_range, villain_range = _local_ranges_from_belief(belief_row, full_hands)
    backend, backend_device = resolve_solver_backend(solver_backend)
    if backend != "cpu":
        raise ValueError("river leaf collection requires the CPU callback backend")
    solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    node = solver.navigate(_parse_nav(street_action, solver))
    if node is None or node.is_terminal:
        return [], [], [], []
    active_idx = int(solver._tree["all_nodes"].index(node))
    collector = _RiverLeafCollector(
        source_case=case,
        board4=board_idx,
        action_prefix=action_prefix,
        active_subtree_node_idx=active_idx,
        solver_hands=list(solver.hands),
        max_leaf_states=max_leaf_states,
        max_rivers_per_terminal=max_rivers_per_terminal,
    )
    solver.solve(
        n_iterations=solver_iterations,
        hero_range=hero_range,
        villain_range=villain_range,
        backend=backend,
        device=backend_device,
        showdown_leaf_fn=collector,
    )
    return collector.cases, collector.features, collector.beliefs, collector.records


def _counter_summary(values: list[str], *, top_k: int = 10) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "n_unique": 0,
            "max_count": 0,
            "max_share": 0.0,
            "top": [],
        }
    counts = Counter(values)
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    max_count = int(top[0][1])
    return {
        "n": int(len(values)),
        "n_unique": int(len(counts)),
        "max_count": max_count,
        "max_share": round(float(max_count / len(values)), 6),
        "top": [{"key": str(key), "count": int(count)} for key, count in top],
    }


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
        "mean": round(float(arr.mean()), 6),
    }


def _summarize_leaf_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    terminal_keys = [
        f"{record.get('source_case', '')}:{record.get('terminal_node_idx', '')}"
        for record in records
    ]
    parsed_leaf_actions = [
        parse_action(str(record.get("leaf_action_str", ""))) for record in records
    ]
    valid_leaf_actions = [parsed for parsed in parsed_leaf_actions if "error" not in parsed]
    return {
        "sources": _counter_summary(
            [str(record.get("source_case", "")) for record in records]
        ),
        "terminals": _counter_summary(terminal_keys),
        "river_cards": _counter_summary(
            [str(record.get("river_card", "")) for record in records]
        ),
        "leaf_actions": _counter_summary(
            [str(record.get("leaf_action_str", "")) for record in records]
        ),
        "leaf_total_last_bet_to": _numeric_summary(
            [float(parsed["total_last_bet_to"]) for parsed in valid_leaf_actions]
        ),
        "leaf_street_last_bet_to": _numeric_summary(
            [float(parsed["street_last_bet_to"]) for parsed in valid_leaf_actions]
        ),
        "leaf_last_bet_size": _numeric_summary(
            [float(parsed["last_bet_size"]) for parsed in valid_leaf_actions]
        ),
        "leaf_action_parse_errors": int(len(records) - len(valid_leaf_actions)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export river leaf cases/ranges from turn resolver terminals."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max-leaf-states", type=int, default=16)
    parser.add_argument("--max-leaf-states-per-source", type=int, default=0)
    parser.add_argument("--max-rivers-per-terminal", type=int, default=0)
    parser.add_argument("--solver-iterations", type=int, default=1)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--output-cases", required=True)
    parser.add_argument("--output-cfv-cache", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    cases = load_cases_json(args.cases)
    base_dataset = None
    if args.cfv_cache:
        base_dataset, _ = load_public_belief_cfv_dataset_cache(args.cfv_cache)
        if base_dataset.features.shape[0] != len(cases):
            raise ValueError("case count does not match CFV cache rows")

    out_cases: list[ResolverBenchmarkCase] = []
    features: list[np.ndarray] = []
    beliefs: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    start_index = max(0, int(args.start_index))
    stop_index = max(start_index + 1, min(start_index + int(args.limit), len(cases)))
    per_source_cap = max(0, int(args.max_leaf_states_per_source))
    for idx in range(start_index, stop_index):
        case = cases[idx]
        if len(out_cases) >= int(args.max_leaf_states):
            break
        belief_row = base_dataset.belief[idx] if base_dataset is not None else None
        remaining = int(args.max_leaf_states) - len(out_cases)
        if per_source_cap:
            remaining = min(remaining, per_source_cap)
        case_items, case_features, case_beliefs, case_records = _collect_case(
            case,
            belief_row=belief_row,
            max_leaf_states=remaining,
            max_rivers_per_terminal=args.max_rivers_per_terminal,
            solver_iterations=args.solver_iterations,
            solver_backend=args.solver_backend,
        )
        out_cases.extend(case_items)
        features.extend(case_features)
        beliefs.extend(case_beliefs)
        records.extend(case_records)
    if not out_cases:
        raise RuntimeError("no river leaf states were collected")

    save_cases_json(out_cases, args.output_cases)
    dataset = PublicBeliefCFVDataset(
        features=np.stack(features).astype(np.float32, copy=False),
        belief=np.stack(beliefs).astype(np.float32, copy=False),
        values=np.zeros((len(out_cases), N_HANDS), dtype=np.float32),
        value_masks=np.zeros((len(out_cases), N_HANDS), dtype=np.float32),
        labels=tuple(case.label for case in out_cases),
    )
    save_public_belief_cfv_dataset_cache(dataset, records, args.output_cfv_cache)
    metrics = {
        "mode": "turn_resolver_river_leaf_case_export",
        "cases": str(args.cases),
        "cfv_cache": str(args.cfv_cache) if args.cfv_cache else None,
        "output_cases": str(args.output_cases),
        "output_cfv_cache": str(args.output_cfv_cache),
        "n_leaf_states": len(out_cases),
        "start_index": int(start_index),
        "limit": int(args.limit),
        "max_leaf_states": int(args.max_leaf_states),
        "max_leaf_states_per_source": int(per_source_cap),
        "max_rivers_per_terminal": int(args.max_rivers_per_terminal),
        "solver_iterations": int(args.solver_iterations),
        "solver_backend": args.solver_backend,
        "leaf_distribution": _summarize_leaf_records(records),
        "records": records,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
