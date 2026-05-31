#!/usr/bin/env python3
"""Evaluate same-topology batched torch CFR against serial root decisions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fast_cfr import (  # noqa: E402
    get_average_strategy,
    solve_cfr_levelsync_torch,
    solve_cfr_levelsync_torch_batched_same_topology,
)
from play_slumbot import (  # noqa: E402
    _compute_bets_before_street,
    card_str_to_index,
    parse_action,
)
from poker_ai.research.resolver_benchmark import load_cases_json  # noqa: E402
from solver import StreetSolver, clear_terminal_matrix_cache  # noqa: E402
from fast_cfr import clear_torch_matrix_tensor_cache  # noqa: E402


def actual_hand_root_policy(
    strategy_sum: np.ndarray,
    *,
    hand_index: int,
    legal_actions: list[int],
    n_actions: int,
) -> np.ndarray:
    """Return dense root policy for one actual private hand."""
    policy = np.zeros(int(n_actions), dtype=np.float32)
    action_policy = get_average_strategy(strategy_sum, 0, legal_actions, int(hand_index))
    for action, prob in action_policy.items():
        policy[int(action)] = float(prob)
    total = float(policy.sum())
    if total > 0.0:
        policy /= total
    return policy


def summarize_actual_hand_root_parity(
    *,
    labels: list[str],
    serial_strategy_sums: list[np.ndarray],
    batched_strategy_sums: np.ndarray,
    hand_indices: list[int],
    legal_actions_by_root: list[list[int]],
    n_actions: int,
) -> dict:
    roots = []
    max_l1 = 0.0
    top_matches = 0
    for index, label in enumerate(labels):
        serial_policy = actual_hand_root_policy(
            serial_strategy_sums[index],
            hand_index=hand_indices[index],
            legal_actions=legal_actions_by_root[index],
            n_actions=n_actions,
        )
        batched_policy = actual_hand_root_policy(
            batched_strategy_sums[index],
            hand_index=hand_indices[index],
            legal_actions=legal_actions_by_root[index],
            n_actions=n_actions,
        )
        root_l1 = round(float(np.abs(serial_policy - batched_policy).sum()), 10)
        serial_top = int(serial_policy.argmax())
        batched_top = int(batched_policy.argmax())
        top_match = serial_top == batched_top
        max_l1 = max(max_l1, root_l1)
        top_matches += int(top_match)
        roots.append(
            {
                "label": str(label),
                "actual_root_l1": root_l1,
                "serial_top": serial_top,
                "batched_top": batched_top,
                "top_match": bool(top_match),
            }
        )
    return {
        "max_actual_root_l1": round(float(max_l1), 10),
        "top_matches": int(top_matches),
        "roots": roots,
    }


def summarize_teacher_relative_root_decisions(
    *,
    labels: list[str],
    teacher_policies: list[np.ndarray],
    serial_policies: list[np.ndarray],
    batched_policies: list[np.ndarray],
) -> dict:
    """Compare low-budget serial and batched root decisions against a teacher."""
    roots = []
    serial_matches = 0
    batched_matches = 0
    serial_l1_values = []
    batched_l1_values = []
    excess_values = []
    for label, teacher, serial, batched in zip(
        labels,
        teacher_policies,
        serial_policies,
        batched_policies,
        strict=True,
    ):
        teacher_top = int(np.argmax(teacher))
        serial_top = int(np.argmax(serial))
        batched_top = int(np.argmax(batched))
        serial_match = serial_top == teacher_top
        batched_match = batched_top == teacher_top
        serial_l1 = float(np.abs(serial - teacher).sum())
        batched_l1 = float(np.abs(batched - teacher).sum())
        excess_l1 = float(batched_l1 - serial_l1)
        serial_matches += int(serial_match)
        batched_matches += int(batched_match)
        serial_l1_values.append(serial_l1)
        batched_l1_values.append(batched_l1)
        excess_values.append(excess_l1)
        roots.append(
            {
                "label": str(label),
                "teacher_top": teacher_top,
                "serial_top": serial_top,
                "batched_top": batched_top,
                "serial_matches_teacher": bool(serial_match),
                "batched_matches_teacher": bool(batched_match),
                "serial_l1_to_teacher": round(serial_l1, 10),
                "batched_l1_to_teacher": round(batched_l1, 10),
                "batched_excess_l1_to_teacher": round(excess_l1, 10),
            }
        )
    return {
        "serial_teacher_top_matches": int(serial_matches),
        "batched_teacher_top_matches": int(batched_matches),
        "teacher_top_match_delta": int(batched_matches - serial_matches),
        "mean_serial_l1_to_teacher": round(float(np.mean(serial_l1_values)) if serial_l1_values else 0.0, 10),
        "mean_batched_l1_to_teacher": round(float(np.mean(batched_l1_values)) if batched_l1_values else 0.0, 10),
        "mean_batched_excess_l1_to_teacher": round(float(np.mean(excess_values)) if excess_values else 0.0, 10),
        "max_batched_excess_l1_to_teacher": round(float(max(excess_values)) if excess_values else 0.0, 10),
        "roots": roots,
    }


def _labels_for_group(shape_group: dict, max_cases: int | None) -> list[str]:
    labels: list[str] = []
    for chunk in shape_group.get("chunks", []):
        labels.extend(str(label) for label in chunk.get("labels", []))
    if max_cases is not None:
        labels = labels[: int(max_cases)]
    return labels


def _make_solver(case) -> StreetSolver:
    parsed = parse_action(case.action_str)
    if "error" in parsed:
        raise ValueError(f"cannot parse case {case.label}: {parsed['error']}")
    street = int(parsed.get("st", -1))
    if street not in (2, 3):
        raise ValueError(f"case {case.label} is not a turn/river resolver case: street={street}")
    n_board = 4 if street == 2 else 5
    board = [card_str_to_index(card) for card in case.board[:n_board]]
    our_bet_pre, opp_bet_pre = _compute_bets_before_street(
        case.action_str,
        case.client_pos,
        target_street=street,
    )
    return StreetSolver(
        board,
        int(our_bet_pre + opp_bet_pre),
        int(20000 - our_bet_pre),
        int(20000 - opp_bet_pre),
        bool(case.client_pos == 0),
    )


def _actual_hand_index(solver: StreetSolver, case) -> int:
    hand = tuple(sorted(card_str_to_index(card) for card in case.hole_cards))
    return int(solver.hand_to_idx[hand])


def _evaluate_group(
    *,
    group_index: int,
    shape_group: dict,
    cases_by_label: dict,
    iterations: int,
    device: str,
    max_cases_per_group: int | None,
    terminal_eval_mode: str,
    teacher_iterations: int | None,
) -> dict:
    labels = _labels_for_group(shape_group, max_cases_per_group)
    clear_terminal_matrix_cache()
    clear_torch_matrix_tensor_cache()
    cases = [cases_by_label[label] for label in labels]
    solvers = [_make_solver(case) for case in cases]
    if not solvers:
        raise ValueError(f"shape group {group_index} has no labels")

    solve_cfr_levelsync_torch(
        solvers[0]._tree,
        solvers[0].n,
        solvers[0].win_m,
        solvers[0].lose_m,
        solvers[0].tie_m,
        solvers[0].valid,
        solvers[0].pot_start,
        solvers[0].hero_stack_start,
        solvers[0].villain_stack_start,
        n_iterations=1,
        device=device,
    )
    if device == "cuda":
        torch.cuda.synchronize()

    serial_started = time.perf_counter()
    serial = []
    for solver in solvers:
        serial.append(
            solve_cfr_levelsync_torch(
                solver._tree,
                solver.n,
                solver.win_m,
                solver.lose_m,
                solver.tie_m,
                solver.valid,
                solver.pot_start,
                solver.hero_stack_start,
                solver.villain_stack_start,
                n_iterations=iterations,
                device=device,
            )
        )
    if device == "cuda":
        torch.cuda.synchronize()
    serial_sec = time.perf_counter() - serial_started

    teacher = []
    teacher_sec = None
    if teacher_iterations is not None:
        teacher_started = time.perf_counter()
        for solver in solvers:
            teacher.append(
                solve_cfr_levelsync_torch(
                    solver._tree,
                    solver.n,
                    solver.win_m,
                    solver.lose_m,
                    solver.tie_m,
                    solver.valid,
                    solver.pot_start,
                    solver.hero_stack_start,
                    solver.villain_stack_start,
                    n_iterations=int(teacher_iterations),
                    device=device,
                )
            )
        if device == "cuda":
            torch.cuda.synchronize()
        teacher_sec = time.perf_counter() - teacher_started

    batched_started = time.perf_counter()
    _, batched_strategy_sum = solve_cfr_levelsync_torch_batched_same_topology(
        [solver._tree for solver in solvers],
        solvers[0].n,
        [solver.win_m for solver in solvers],
        [solver.lose_m for solver in solvers],
        [solver.tie_m for solver in solvers],
        [solver.valid for solver in solvers],
        [solver.pot_start for solver in solvers],
        [solver.hero_stack_start for solver in solvers],
        [solver.villain_stack_start for solver in solvers],
        n_iterations=iterations,
        terminal_eval_mode=terminal_eval_mode,
        device=device,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    batched_sec = time.perf_counter() - batched_started

    parity = summarize_actual_hand_root_parity(
        labels=labels,
        serial_strategy_sums=[strategy for _, strategy in serial],
        batched_strategy_sums=batched_strategy_sum,
        hand_indices=[_actual_hand_index(solver, case) for solver, case in zip(solvers, cases, strict=True)],
        legal_actions_by_root=[sorted(solver.root.children.keys()) for solver in solvers],
        n_actions=int(solvers[0]._tree["n_actions"]),
    )
    hand_indices = [_actual_hand_index(solver, case) for solver, case in zip(solvers, cases, strict=True)]
    legal_actions_by_root = [sorted(solver.root.children.keys()) for solver in solvers]
    n_actions = int(solvers[0]._tree["n_actions"])
    teacher_relative = None
    if teacher:
        teacher_relative = summarize_teacher_relative_root_decisions(
            labels=labels,
            teacher_policies=[
                actual_hand_root_policy(
                    strategy,
                    hand_index=hand_index,
                    legal_actions=legal_actions,
                    n_actions=n_actions,
                )
                for (_, strategy), hand_index, legal_actions in zip(
                    teacher,
                    hand_indices,
                    legal_actions_by_root,
                    strict=True,
                )
            ],
            serial_policies=[
                actual_hand_root_policy(
                    strategy,
                    hand_index=hand_index,
                    legal_actions=legal_actions,
                    n_actions=n_actions,
                )
                for (_, strategy), hand_index, legal_actions in zip(
                    serial,
                    hand_indices,
                    legal_actions_by_root,
                    strict=True,
                )
            ],
            batched_policies=[
                actual_hand_root_policy(
                    batched_strategy_sum[index],
                    hand_index=hand_index,
                    legal_actions=legal_actions,
                    n_actions=n_actions,
                )
                for index, (hand_index, legal_actions) in enumerate(
                    zip(hand_indices, legal_actions_by_root, strict=True)
                )
            ],
        )
    return {
        "group_index": int(group_index),
        "shape_id": str(shape_group.get("shape_id", group_index)),
        "n_cases": int(len(labels)),
        "iterations": int(iterations),
        "device": device,
        "terminal_eval_mode": terminal_eval_mode,
        "n_nodes": int(solvers[0]._tree["n_nodes"]),
        "n_hands": int(solvers[0].n),
        "serial_sec": float(serial_sec),
        "teacher_iterations": int(teacher_iterations) if teacher_iterations is not None else None,
        "teacher_sec": float(teacher_sec) if teacher_sec is not None else None,
        "batched_sec": float(batched_sec),
        "serial_ms_per_root": float(1000.0 * serial_sec / max(len(labels), 1)),
        "batched_ms_per_root": float(1000.0 * batched_sec / max(len(labels), 1)),
        "speedup": float(serial_sec / batched_sec) if batched_sec > 0.0 else None,
        "teacher_relative": teacher_relative,
        **parity,
    }


def _parse_group_indices(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--topology-plan-json", required=True)
    parser.add_argument("--group-indices", default="0,1,2,3,4")
    parser.add_argument("--max-cases-per-group", type=int)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--terminal-eval-mode",
        default="batched",
        choices=("batched", "loop", "loop_showdown", "loop_folds"),
    )
    parser.add_argument("--teacher-iterations", type=int)
    parser.add_argument("--min-teacher-top-match-delta", type=int, default=0)
    parser.add_argument("--max-teacher-excess-l1", type=float)
    parser.add_argument("--max-root-l1", type=float, default=1e-3)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    plan = json.loads(Path(args.topology_plan_json).read_text(encoding="utf-8"))
    shape_plan = plan.get("shape_compatible_chunk_plan", plan)
    shape_groups = shape_plan["shape_groups"]
    cases_by_label = {case.label: case for case in load_cases_json(args.cases_json)}
    groups = [
        _evaluate_group(
            group_index=group_index,
            shape_group=shape_groups[group_index],
            cases_by_label=cases_by_label,
            iterations=args.iterations,
            device=device,
            max_cases_per_group=args.max_cases_per_group,
            terminal_eval_mode=args.terminal_eval_mode,
            teacher_iterations=args.teacher_iterations,
        )
        for group_index in _parse_group_indices(args.group_indices)
    ]
    if args.teacher_iterations is None:
        passed = all(
            group["top_matches"] == group["n_cases"]
            and group["max_actual_root_l1"] <= float(args.max_root_l1)
            and (group["speedup"] is not None and group["speedup"] >= float(args.min_speedup))
            for group in groups
        )
    else:
        passed = all(
            group["teacher_relative"] is not None
            and group["teacher_relative"]["teacher_top_match_delta"] >= int(args.min_teacher_top_match_delta)
            and (
                args.max_teacher_excess_l1 is None
                or group["teacher_relative"]["max_batched_excess_l1_to_teacher"]
                <= float(args.max_teacher_excess_l1)
            )
            and (group["speedup"] is not None and group["speedup"] >= float(args.min_speedup))
            for group in groups
        )
    teacher_relatives = [group["teacher_relative"] for group in groups if group["teacher_relative"] is not None]
    metrics = {
        "mode": (
            "same_topology_batched_cfr_teacher_relative_screen"
            if args.teacher_iterations is not None
            else "same_topology_batched_cfr_actual_hand_smoke"
        ),
        "passed": bool(passed),
        "n_groups": int(len(groups)),
        "total_roots": int(sum(group["n_cases"] for group in groups)),
        "total_top_matches": int(sum(group["top_matches"] for group in groups)),
        "total_serial_teacher_top_matches": int(
            sum(relative["serial_teacher_top_matches"] for relative in teacher_relatives)
        ),
        "total_batched_teacher_top_matches": int(
            sum(relative["batched_teacher_top_matches"] for relative in teacher_relatives)
        ),
        "total_teacher_top_match_delta": int(
            sum(relative["teacher_top_match_delta"] for relative in teacher_relatives)
        ),
        "max_batched_excess_l1_to_teacher": max(
            (relative["max_batched_excess_l1_to_teacher"] for relative in teacher_relatives),
            default=0.0,
        ),
        "max_actual_root_l1": max((group["max_actual_root_l1"] for group in groups), default=0.0),
        "min_speedup": min((group["speedup"] for group in groups if group["speedup"] is not None), default=0.0),
        "groups": groups,
    }
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
