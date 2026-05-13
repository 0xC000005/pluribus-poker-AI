#!/usr/bin/env python3
"""Evaluate a soft policy-head prior against a higher-iteration solver reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.deep_cfr.fast_state import N_ACTIONS
from poker_ai.research.evaluation import load_value_network_checkpoint
from poker_ai.research.resolver_benchmark import (
    _policy_decision,
    _solver_decision,
    load_cases_json,
)
from play_slumbot import parse_action
from poker_ai.research.belief_value_probe import save_metrics


def mix_strategy(base: np.ndarray, prior: np.ndarray, prior_weight: float) -> np.ndarray:
    weight = float(prior_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("prior_weight must be in [0, 1]")
    mixed = (1.0 - weight) * np.asarray(base, dtype=np.float64) + weight * np.asarray(
        prior,
        dtype=np.float64,
    )
    mixed = np.maximum(mixed, 0.0)
    total = float(mixed.sum())
    if total <= 0:
        out = np.zeros(N_ACTIONS, dtype=np.float64)
        out[1] = 1.0
        return out
    return mixed / total


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 8) if values else 0.0


def _rate(values: list[bool]) -> float:
    return round(float(np.mean(values)), 8) if values else 0.0


def eval_policy_prior_solver_budget(
    *,
    checkpoint: str | Path,
    cases_json: str | Path,
    device: str = "auto",
    low_iterations: int = 5,
    reference_iterations: int = 25,
    solver_backend: str = "cpu",
    prior_weight: float = 0.25,
    max_mixed_allin_rate: float = 0.05,
) -> dict[str, Any]:
    torch_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
    loaded = load_value_network_checkpoint(checkpoint, torch_device)
    cases = load_cases_json(cases_json)
    records = []
    for case in cases:
        parsed = parse_action(case.action_str)
        if "error" in parsed:
            records.append({"label": case.label, "passed": False, "skipped": parsed["error"]})
            continue
        low = _solver_decision(
            case,
            parsed,
            solver_iterations=low_iterations,
            solver_backend=solver_backend,
        )
        reference = _solver_decision(
            case,
            parsed,
            solver_iterations=reference_iterations,
            solver_backend=solver_backend,
        )
        if low is None or reference is None:
            records.append({"label": case.label, "passed": False, "skipped": "solver_skipped"})
            continue
        policy = _policy_decision(
            loaded.value_net,
            torch_device,
            case,
            parsed,
            strategy_source="policy_head",
        )
        mixed = mix_strategy(low.strategy, policy.strategy, prior_weight)
        records.append(
            {
                "label": case.label,
                "passed": True,
                "low_action": int(np.argmax(low.strategy)),
                "reference_action": int(np.argmax(reference.strategy)),
                "policy_action": int(np.argmax(policy.strategy)),
                "mixed_action": int(np.argmax(mixed)),
                "low_l1_to_reference": round(float(np.abs(low.strategy - reference.strategy).sum()), 8),
                "policy_l1_to_reference": round(
                    float(np.abs(policy.strategy - reference.strategy).sum()),
                    8,
                ),
                "mixed_l1_to_reference": round(float(np.abs(mixed - reference.strategy).sum()), 8),
                "low_allin_selected": bool(int(np.argmax(low.strategy)) == 8),
                "reference_allin_selected": bool(int(np.argmax(reference.strategy)) == 8),
                "policy_allin_selected": bool(int(np.argmax(policy.strategy)) == 8),
                "mixed_allin_selected": bool(int(np.argmax(mixed)) == 8),
                "low_agrees_with_reference": bool(
                    int(np.argmax(low.strategy)) == int(np.argmax(reference.strategy))
                ),
                "policy_agrees_with_reference": bool(
                    int(np.argmax(policy.strategy)) == int(np.argmax(reference.strategy))
                ),
                "mixed_agrees_with_reference": bool(
                    int(np.argmax(mixed)) == int(np.argmax(reference.strategy))
                ),
            }
        )
    evaluated = [record for record in records if record.get("passed")]
    low_l1 = [float(record["low_l1_to_reference"]) for record in evaluated]
    policy_l1 = [float(record["policy_l1_to_reference"]) for record in evaluated]
    mixed_l1 = [float(record["mixed_l1_to_reference"]) for record in evaluated]
    mixed_allin_rate = _rate([bool(record["mixed_allin_selected"]) for record in evaluated])
    return {
        "mode": "policy_prior_solver_budget",
        "checkpoint": str(checkpoint),
        "cases_json": str(cases_json),
        "device": str(torch_device),
        "solver_backend": solver_backend,
        "low_iterations": int(low_iterations),
        "reference_iterations": int(reference_iterations),
        "prior_weight": float(prior_weight),
        "max_mixed_allin_rate": float(max_mixed_allin_rate),
        "n_cases": int(len(records)),
        "n_evaluated": int(len(evaluated)),
        "passed": bool(
            evaluated
            and _mean(mixed_l1) < _mean(low_l1)
            and mixed_allin_rate <= float(max_mixed_allin_rate)
        ),
        "mean_low_l1_to_reference": _mean(low_l1),
        "mean_policy_l1_to_reference": _mean(policy_l1),
        "mean_mixed_l1_to_reference": _mean(mixed_l1),
        "low_action_agreement": _rate([bool(record["low_agrees_with_reference"]) for record in evaluated]),
        "policy_action_agreement": _rate(
            [bool(record["policy_agrees_with_reference"]) for record in evaluated]
        ),
        "mixed_action_agreement": _rate(
            [bool(record["mixed_agrees_with_reference"]) for record in evaluated]
        ),
        "low_allin_rate": _rate([bool(record["low_allin_selected"]) for record in evaluated]),
        "reference_allin_rate": _rate(
            [bool(record["reference_allin_selected"]) for record in evaluated]
        ),
        "policy_allin_rate": _rate([bool(record["policy_allin_selected"]) for record in evaluated]),
        "mixed_allin_rate": mixed_allin_rate,
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare low-iteration solver, policy head, and soft policy-prior mix against a stronger solver."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--low-iterations", type=int, default=5)
    parser.add_argument("--reference-iterations", type=int, default=25)
    parser.add_argument("--solver-backend", default="cpu")
    parser.add_argument("--prior-weight", type=float, default=0.25)
    parser.add_argument("--max-mixed-allin-rate", type=float, default=0.05)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = eval_policy_prior_solver_budget(
        checkpoint=args.checkpoint,
        cases_json=args.cases_json,
        device=args.device,
        low_iterations=args.low_iterations,
        reference_iterations=args.reference_iterations,
        solver_backend=args.solver_backend,
        prior_weight=args.prior_weight,
        max_mixed_allin_rate=args.max_mixed_allin_rate,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
