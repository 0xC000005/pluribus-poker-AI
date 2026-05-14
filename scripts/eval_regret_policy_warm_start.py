#!/usr/bin/env python3
"""Evaluate trained regret/policy field checkpoints as CFR+ warm starts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.deep_cfr.fast_state import N_ACTIONS
from poker_ai.research.belief_value_probe import load_public_belief_cfv_dataset_cache, save_metrics
from poker_ai.research.resolver_benchmark import SolverDecision, load_cases_json

from build_regret_policy_warm_start_targets import (  # noqa: E402
    _card_to_str,
    _node_fields,
    _node_legal_mask,
)
from eval_joint_pbs_policy_warm_start import (  # noqa: E402
    _case_slice,
    _illegal_mass,
    _kl_to_reference,
    _root_disjoint_audit,
    _solve_with_belief,
    _solver_context,
    _summarize_records,
)
from play_slumbot import build_features, parse_action  # noqa: E402
from solver import StreetSolver, _parse_nav  # noqa: E402
from train_regret_policy_warm_start import (  # noqa: E402
    RegretPolicyWarmStartDataset,
    RegretPolicyWarmStartNet,
    _predict_components,
    components_to_fields,
    make_model_inputs,
)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _resolve_payload_path(path: str | Path, *, checkpoint: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return Path(checkpoint).parent / candidate


def load_regret_policy_warm_start_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "auto",
) -> tuple[RegretPolicyWarmStartNet, dict[str, Any], torch.device]:
    resolved_device = _resolve_device(device)
    payload = torch.load(checkpoint, map_location=resolved_device, weights_only=False)
    if payload.get("mode") != "regret_policy_warm_start_checkpoint":
        raise ValueError(f"{checkpoint} is not a regret-policy warm-start checkpoint")
    model = RegretPolicyWarmStartNet(
        int(payload["input_dim"]),
        int(payload["hidden_dim"]),
        int(payload["n_layers"]),
    ).to(resolved_device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload, resolved_device


def build_regret_policy_field_warm_start(
    *,
    solver: StreetSolver,
    node: Any,
    predicted_regret: np.ndarray,
    predicted_strategy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Place predicted per-hand fields at one public node for a warm-start solve."""
    regret_rows = np.asarray(predicted_regret, dtype=np.float32)
    strategy_rows = np.asarray(predicted_strategy, dtype=np.float32)
    expected = (int(solver.n), N_ACTIONS)
    if regret_rows.shape != expected:
        raise ValueError(f"predicted_regret shape {regret_rows.shape} != {expected}")
    if strategy_rows.shape != expected:
        raise ValueError(f"predicted_strategy shape {strategy_rows.shape} != {expected}")
    legal = _node_legal_mask(node).reshape(1, N_ACTIONS).astype(np.float32, copy=False)
    regret_rows = np.maximum(regret_rows, 0.0) * legal
    strategy_rows = np.maximum(strategy_rows, 0.0) * legal
    node_idx = solver._tree["all_nodes"].index(node)
    shape = (
        solver._tree["n_nodes"],
        solver._tree["n_actions"],
        solver.n,
    )
    initial_regret = np.zeros(shape, dtype=np.float32)
    initial_strategy = np.zeros(shape, dtype=np.float32)
    initial_regret[node_idx, :N_ACTIONS, :] = regret_rows.T
    initial_strategy[node_idx, :N_ACTIONS, :] = strategy_rows.T
    return initial_regret, initial_strategy


def _predict_fields_for_solver_node(
    *,
    model: RegretPolicyWarmStartNet,
    payload: dict[str, Any],
    device: torch.device,
    case: Any,
    parsed: dict[str, Any],
    belief_row: np.ndarray,
    solver: StreetSolver,
    node: Any,
    low_regret: np.ndarray,
    low_strategy: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    legal_mask = _node_legal_mask(node)
    public_feature = build_features(
        [],
        list(case.board),
        case.action_str,
        int(case.client_pos),
        parsed,
    ).astype(np.float32, copy=False)
    policy_features = []
    for hand in solver.hands:
        hole_cards = [_card_to_str(int(hand[0])), _card_to_str(int(hand[1]))]
        policy_features.append(
            build_features(
                hole_cards,
                list(case.board),
                case.action_str,
                int(case.client_pos),
                parsed,
            ).astype(np.float32, copy=False)
        )
    n = len(policy_features)
    dataset = RegretPolicyWarmStartDataset(
        features=np.repeat(public_feature[np.newaxis, :], n, axis=0).astype(np.float32),
        policy_features=np.stack(policy_features).astype(np.float32),
        belief=np.repeat(np.asarray(belief_row, dtype=np.float32)[np.newaxis, :], n, axis=0),
        legal_masks=np.repeat(legal_mask[np.newaxis, :], n, axis=0).astype(np.float32),
        target_regret_sum=np.zeros((n, N_ACTIONS), dtype=np.float32),
        target_strategy_sum=np.zeros((n, N_ACTIONS), dtype=np.float32),
        target_probs=np.repeat(legal_mask[np.newaxis, :], n, axis=0).astype(np.float32),
        low_regret_sum=np.asarray(low_regret, dtype=np.float32),
        low_strategy_sum=np.asarray(low_strategy, dtype=np.float32),
        labels=tuple(str(i) for i in range(n)),
        root_labels=tuple(str(i) for i in range(n)),
    )
    x = make_model_inputs(
        dataset,
        public_mean=np.asarray(payload["public_mean"], dtype=np.float32),
        public_std=np.asarray(payload["public_std"], dtype=np.float32),
        belief_mean=np.asarray(payload["belief_mean"], dtype=np.float32),
        belief_std=np.asarray(payload["belief_std"], dtype=np.float32),
        include_low_state=bool(payload.get("include_low_state", False)),
    )
    regret_logits, strategy_logits, regret_log_mass, strategy_log_mass = _predict_components(
        model,
        x,
        batch_size=batch_size,
        device=device,
    )
    return components_to_fields(
        regret_logits,
        strategy_logits,
        regret_log_mass,
        strategy_log_mass,
        dataset.legal_masks,
    )


def _warm_start_decision(
    *,
    model: RegretPolicyWarmStartNet,
    payload: dict[str, Any],
    device: torch.device,
    case: Any,
    parsed: dict[str, Any],
    belief_row: np.ndarray,
    low_solver: StreetSolver,
    low_node: Any,
    solver_iterations: int,
    solver_backend: str,
    batch_size: int,
) -> tuple[StreetSolver, Any, SolverDecision] | None:
    started = time.perf_counter()
    board_idx, _our_cards_idx, pot, hero_stack, villain_stack, hero_first, street_action = _solver_context(
        case,
        parsed,
    )
    target_solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
    target_node = target_solver.navigate(_parse_nav(street_action, target_solver))
    if target_node is None or target_node.is_terminal:
        return None
    low_regret, low_strategy = _node_fields(low_solver, low_node)
    predicted_regret, predicted_strategy = _predict_fields_for_solver_node(
        model=model,
        payload=payload,
        device=device,
        case=case,
        parsed=parsed,
        belief_row=belief_row,
        solver=target_solver,
        node=target_node,
        low_regret=low_regret,
        low_strategy=low_strategy,
        batch_size=batch_size,
    )
    initial_regret, initial_strategy = build_regret_policy_field_warm_start(
        solver=target_solver,
        node=target_node,
        predicted_regret=predicted_regret,
        predicted_strategy=predicted_strategy,
    )
    solved = _solve_with_belief(
        case,
        parsed,
        belief_row=belief_row,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
        initial_regret_sum=initial_regret,
        initial_strategy_sum=initial_strategy,
    )
    if solved is None:
        return None
    solver, node, decision = solved
    total_ms = (time.perf_counter() - started) * 1000.0
    return solver, node, SolverDecision(
        decision.action,
        decision.increment,
        decision.strategy,
        float(total_ms),
        decision.node_terminal,
    )


def eval_regret_policy_warm_start(
    *,
    checkpoint: str | Path,
    cases_json: str | Path,
    cfv_cache: str | Path,
    device: str | torch.device = "auto",
    start_index: int = 128,
    limit: int = 64,
    low_iterations: int = 5,
    reference_iterations: int = 25,
    solver_backend: str = "cpu",
    train_labels_npz: str | Path | None = None,
    require_root_disjoint: bool = True,
    min_evaluated: int = 1,
    max_warm_latency_ratio: float = 2.0,
    batch_size: int = 8192,
) -> dict[str, Any]:
    model, payload, torch_device = load_regret_policy_warm_start_checkpoint(checkpoint, device=device)
    cases = load_cases_json(cases_json)
    base_dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if base_dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")
    selected_cases = _case_slice(cases, start_index=start_index, limit=limit)
    train_path = train_labels_npz
    if train_path is None and payload.get("train_npz"):
        train_path = _resolve_payload_path(payload["train_npz"], checkpoint=checkpoint)
    root_audit = _root_disjoint_audit(
        checkpoint=checkpoint,
        payload=payload,
        selected_cases=selected_cases,
        train_labels_npz=train_path,
        require_root_disjoint=require_root_disjoint,
    )
    records: list[dict[str, Any]] = []
    for local_idx, case in enumerate(selected_cases):
        case_idx = int(start_index) + int(local_idx)
        parsed = parse_action(case.action_str)
        if "error" in parsed:
            records.append({"label": case.label, "passed": False, "skipped": parsed["error"]})
            continue
        belief_row = base_dataset.belief[case_idx]
        low_solved = _solve_with_belief(
            case,
            parsed,
            belief_row=belief_row,
            solver_iterations=low_iterations,
            solver_backend=solver_backend,
        )
        reference_solved = _solve_with_belief(
            case,
            parsed,
            belief_row=belief_row,
            solver_iterations=reference_iterations,
            solver_backend=solver_backend,
        )
        if low_solved is None or reference_solved is None:
            records.append({"label": case.label, "passed": False, "skipped": "solver_skipped"})
            continue
        low_solver, low_node, low = low_solved
        reference_solver, reference_node, reference = reference_solved
        warm_solved = _warm_start_decision(
            model=model,
            payload=payload,
            device=torch_device,
            case=case,
            parsed=parsed,
            belief_row=belief_row,
            low_solver=low_solver,
            low_node=low_node,
            solver_iterations=low_iterations,
            solver_backend=solver_backend,
            batch_size=batch_size,
        )
        if warm_solved is None:
            records.append({"label": case.label, "passed": False, "skipped": "warm_start_skipped"})
            continue
        _warm_solver, warm_node, warm = warm_solved
        records.append(
            {
                "label": case.label,
                "passed": True,
                "low_action": int(np.argmax(low.strategy)),
                "reference_action": int(np.argmax(reference.strategy)),
                "warm_action": int(np.argmax(warm.strategy)),
                "low_l1_to_reference": round(float(np.abs(low.strategy - reference.strategy).sum()), 8),
                "warm_l1_to_reference": round(float(np.abs(warm.strategy - reference.strategy).sum()), 8),
                "low_kl_to_reference": _kl_to_reference(low.strategy, reference.strategy),
                "warm_kl_to_reference": _kl_to_reference(warm.strategy, reference.strategy),
                "low_allin_prob": round(float(low.strategy[8]), 8),
                "reference_allin_prob": round(float(reference.strategy[8]), 8),
                "warm_allin_prob": round(float(warm.strategy[8]), 8),
                "low_allin_selected": bool(int(np.argmax(low.strategy)) == 8),
                "reference_allin_selected": bool(int(np.argmax(reference.strategy)) == 8),
                "warm_allin_selected": bool(int(np.argmax(warm.strategy)) == 8),
                "low_illegal_mass": _illegal_mass(low.strategy, low_node),
                "reference_illegal_mass": _illegal_mass(reference.strategy, reference_node),
                "warm_illegal_mass": _illegal_mass(warm.strategy, warm_node),
                "low_latency_ms": round(float(low.latency_ms), 3),
                "reference_latency_ms": round(float(reference.latency_ms), 3),
                "warm_latency_ms": round(float(warm.latency_ms), 3),
                "low_agrees_with_reference": bool(
                    int(np.argmax(low.strategy)) == int(np.argmax(reference.strategy))
                ),
                "warm_agrees_with_reference": bool(
                    int(np.argmax(warm.strategy)) == int(np.argmax(reference.strategy))
                ),
            }
        )
    summary = _summarize_records(
        records,
        root_audit=root_audit,
        min_evaluated=min_evaluated,
        max_warm_latency_ratio=max_warm_latency_ratio,
    )
    return {
        "mode": "regret_policy_warm_start_solver_budget",
        "checkpoint": str(checkpoint),
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "device": str(torch_device),
        "start_index": int(start_index),
        "limit": int(limit),
        "low_iterations": int(low_iterations),
        "reference_iterations": int(reference_iterations),
        "solver_backend": solver_backend,
        "require_root_disjoint": bool(require_root_disjoint),
        "train_labels_npz": str(train_path or ""),
        "batch_size": int(batch_size),
        "root_disjoint_audit": root_audit,
        "n_cases": int(len(records)),
        **summary,
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare low-budget CFR+ to trained regret/policy-field warm-start CFR+."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-index", type=int, default=128)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--low-iterations", type=int, default=5)
    parser.add_argument("--reference-iterations", type=int, default=25)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--train-labels-npz")
    parser.add_argument("--no-require-root-disjoint", action="store_true")
    parser.add_argument("--min-evaluated", type=int, default=1)
    parser.add_argument("--max-warm-latency-ratio", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = eval_regret_policy_warm_start(
        checkpoint=args.checkpoint,
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        device=args.device,
        start_index=args.start_index,
        limit=args.limit,
        low_iterations=args.low_iterations,
        reference_iterations=args.reference_iterations,
        solver_backend=args.solver_backend,
        train_labels_npz=args.train_labels_npz,
        require_root_disjoint=not args.no_require_root_disjoint,
        min_evaluated=args.min_evaluated,
        max_warm_latency_ratio=args.max_warm_latency_ratio,
        batch_size=args.batch_size,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
