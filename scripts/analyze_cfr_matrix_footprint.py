#!/usr/bin/env python3
"""Report matrix/fused CFR footprint for turn/river resolver cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.research.cfr_matrix_footprint import summarize_tree_for_matrix_cfr  # noqa: E402
from poker_ai.research.resolver_benchmark import (  # noqa: E402
    ResolverBenchmarkCase,
    default_benchmark_cases,
    load_cases_json,
)
from play_slumbot import _compute_bets_before_street, card_str_to_index, parse_action  # noqa: E402
from solver import StreetSolver  # noqa: E402


def _mib(n_bytes: int) -> float:
    return round(float(n_bytes) / (1024.0 * 1024.0), 6)


def _case_footprint(case: ResolverBenchmarkCase) -> dict:
    parsed = parse_action(case.action_str)
    if "error" in parsed:
        return {"label": case.label, "passed": False, "skipped": parsed["error"]}
    street = int(parsed.get("st", -1))
    if street not in (2, 3):
        return {"label": case.label, "passed": False, "skipped": f"unsupported_street:{street}"}
    n_board = 4 if street == 2 else 5
    board_idx = [card_str_to_index(card) for card in case.board[:n_board]]
    our_bet_pre, opp_bet_pre = _compute_bets_before_street(
        case.action_str,
        case.client_pos,
        target_street=street,
    )
    pot = int(our_bet_pre + opp_bet_pre)
    hero_stack = int(20000 - our_bet_pre)
    villain_stack = int(20000 - opp_bet_pre)
    hero_first = bool(case.client_pos == 0)
    solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    summary = summarize_tree_for_matrix_cfr(solver._tree, n_hands=solver.n)
    return {
        "label": case.label,
        "passed": True,
        "street": street,
        "board_cards": n_board,
        "pot": pot,
        "hero_stack": hero_stack,
        "villain_stack": villain_stack,
        "hero_first": hero_first,
        **summary,
        "memory_mib": {
            key: _mib(value)
            for key, value in summary["memory_bytes"].items()
        },
    }


def summarize_case_footprints(cases: list[ResolverBenchmarkCase]) -> dict:
    records = [_case_footprint(case) for case in cases]
    evaluated = [record for record in records if record.get("passed")]
    totals = [
        int(record["memory_bytes"]["solver_state_total"])
        for record in evaluated
    ]
    cfr_state = [
        int(record["memory_bytes"]["cfr_state"])
        for record in evaluated
    ]
    max_nodes = max((int(record["n_nodes"]) for record in evaluated), default=0)
    max_edges = max((int(record["n_edges"]) for record in evaluated), default=0)
    max_depth = max((int(record["max_depth"]) for record in evaluated), default=0)
    return {
        "mode": "cfr_matrix_footprint",
        "n_cases": len(records),
        "n_evaluated": len(evaluated),
        "passed": len(evaluated) == len(records),
        "max_nodes": max_nodes,
        "max_edges": max_edges,
        "max_depth": max_depth,
        "mean_solver_state_mib": _mib(int(mean(totals))) if totals else 0.0,
        "max_solver_state_mib": _mib(max(totals)) if totals else 0.0,
        "batch_solver_state_mib": _mib(sum(totals)) if totals else 0.0,
        "max_cfr_state_mib": _mib(max(cfr_state)) if cfr_state else 0.0,
        "records": records,
    }


def select_case_window(
    cases: list[ResolverBenchmarkCase],
    *,
    start_index: int = 0,
    max_cases: int | None = None,
) -> list[ResolverBenchmarkCase]:
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    selected = cases[int(start_index) :]
    if max_cases is not None:
        selected = selected[: int(max_cases)]
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate memory and tree shape for a future matrix/fused CFR boundary."
    )
    parser.add_argument("--cases-json")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    cases = load_cases_json(args.cases_json) if args.cases_json else default_benchmark_cases()
    cases = select_case_window(
        cases,
        start_index=args.start_index,
        max_cases=args.max_cases,
    )
    metrics = summarize_case_footprints(cases)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
