#!/usr/bin/env python3
"""Benchmark ragged terminal matrix evaluation across heterogeneous roots."""

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

from play_slumbot import _compute_bets_before_street, card_str_to_index, parse_action  # noqa: E402
from poker_ai.research.ragged_terminal_eval import pad_ragged_2d_arrays  # noqa: E402
from poker_ai.research.resolver_benchmark import load_cases_json  # noqa: E402
from solver import StreetSolver  # noqa: E402


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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _terminal_count_and_matrix(solver: StreetSolver, terminal_kind: str) -> tuple[int, np.ndarray]:
    if terminal_kind == "showdown":
        return int(len(solver._tree["showdown_idx"])), solver.win_m.T.copy()
    if terminal_kind == "hero_fold":
        return int(len(solver._tree["hero_fold_idx"])), solver.valid.T.copy()
    if terminal_kind == "villain_fold":
        return int(len(solver._tree["villain_fold_idx"])), solver.valid.T.copy()
    raise ValueError("terminal_kind must be showdown, hero_fold, or villain_fold")


def evaluate_ragged_terminal(
    solvers: list[StreetSolver],
    *,
    terminal_kind: str,
    seed: int,
    repeats: int,
    device: str,
    max_abs_diff: float,
    min_speedup: float,
) -> dict:
    if not solvers:
        raise ValueError("solvers must be non-empty")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda is unavailable")
    rng = np.random.default_rng(int(seed))
    n_hands = int(solvers[0].n)
    terminal_shapes = [_terminal_count_and_matrix(solver, terminal_kind) for solver in solvers]
    reaches_np = [
        rng.random((terminal_rows, n_hands), dtype=np.float32)
        for terminal_rows, _ in terminal_shapes
    ]
    matrices_t = [
        torch.as_tensor(matrix, dtype=torch.float32, device=torch_device)
        for _, matrix in terminal_shapes
    ]
    reaches_t = [
        torch.as_tensor(reach, dtype=torch.float32, device=torch_device)
        for reach in reaches_np
    ]
    padded_np, mask_np = pad_ragged_2d_arrays(reaches_np)
    padded_t = torch.as_tensor(padded_np, dtype=torch.float32, device=torch_device)
    matrix_batch_t = torch.stack(matrices_t, dim=0)

    _sync(torch_device)
    serial_started = time.perf_counter()
    serial_outputs = None
    for _ in range(int(repeats)):
        serial_outputs = [reach @ matrix for reach, matrix in zip(reaches_t, matrices_t, strict=True)]
    _sync(torch_device)
    serial_sec = time.perf_counter() - serial_started

    _sync(torch_device)
    ragged_started = time.perf_counter()
    ragged_output = None
    for _ in range(int(repeats)):
        ragged_output = torch.bmm(padded_t, matrix_batch_t)
    _sync(torch_device)
    ragged_sec = time.perf_counter() - ragged_started

    assert serial_outputs is not None
    assert ragged_output is not None
    max_diff = 0.0
    for index, serial in enumerate(serial_outputs):
        rows = int(mask_np[index].sum())
        if rows == 0:
            continue
        diff = torch.max(torch.abs(serial - ragged_output[index, :rows, :])).item()
        max_diff = max(max_diff, float(diff))

    total_rows = int(sum(reach.shape[0] for reach in reaches_np))
    padded_rows = int(padded_np.shape[0] * padded_np.shape[1])
    speedup = float(serial_sec / ragged_sec) if ragged_sec > 0.0 else 0.0
    return {
        "mode": "ragged_terminal_eval_smoke",
        "passed": bool(max_diff <= float(max_abs_diff) and speedup >= float(min_speedup)),
        "device": str(torch_device),
        "terminal_kind": str(terminal_kind),
        "n_roots": int(len(solvers)),
        "n_hands": n_hands,
        "repeats": int(repeats),
        "total_terminal_rows": total_rows,
        "max_terminal_rows": int(padded_np.shape[1]),
        "padded_rows": padded_rows,
        "padding_fraction": round(float(1.0 - total_rows / max(padded_rows, 1)), 8),
        "serial_sec": float(serial_sec),
        "ragged_sec": float(ragged_sec),
        "serial_ms_per_repeat": float(1000.0 * serial_sec / max(int(repeats), 1)),
        "ragged_ms_per_repeat": float(1000.0 * ragged_sec / max(int(repeats), 1)),
        "speedup": speedup,
        "max_abs_diff": float(max_diff),
        "max_abs_diff_threshold": float(max_abs_diff),
        "min_speedup": float(min_speedup),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--start-index", type=int, default=128)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument(
        "--case-indices",
        help="Comma-separated case indices relative to --start-index. Overrides --limit.",
    )
    parser.add_argument(
        "--terminal-kind",
        default="showdown",
        choices=("showdown", "hero_fold", "villain_fold"),
    )
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-abs-diff", type=float, default=1e-3)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    cases = load_cases_json(args.cases_json)[int(args.start_index) :]
    if args.case_indices:
        indices = [int(part.strip()) for part in args.case_indices.split(",") if part.strip()]
        cases = [cases[index] for index in indices]
    else:
        cases = cases[: int(args.limit)]
    solvers = [_make_solver(case) for case in cases]
    metrics = evaluate_ragged_terminal(
        solvers,
        terminal_kind=args.terminal_kind,
        seed=args.seed,
        repeats=args.repeats,
        device=device,
        max_abs_diff=args.max_abs_diff,
        min_speedup=args.min_speedup,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
