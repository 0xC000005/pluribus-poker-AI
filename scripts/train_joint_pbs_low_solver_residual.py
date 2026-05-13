#!/usr/bin/env python3
"""Train a root-disjoint low-solver residual policy diagnostic.

This is not a promoted playing policy. It tests whether a cheap low-iteration
resolver policy can be corrected toward high-budget joint-PBS policy targets
better than direct public-root generalization from scratch.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.belief_probe import (
    BELIEF_DIM,
    _HAND_TO_INDEX,
    _normalize_targets,
    _policy_metrics,
    _resolve_device,
)
from poker_ai.research.belief_value_probe import save_metrics
from poker_ai.research.resolver_benchmark import load_cases_json

from eval_joint_pbs_continuation_probe import (  # noqa: E402
    JointPBSDataset,
    _legal_uniform_policy,
    _masked_policy_loss,
    _standardize_joint_pair,
    load_joint_pbs_dataset,
)
from eval_joint_pbs_resolver_leaf_ab import _local_ranges_from_belief, _strategy_vector  # noqa: E402
from play_slumbot import (  # noqa: E402
    _compute_bets_before_street,
    card_str_to_index,
    parse_action,
)
from solver import StreetSolver, _parse_nav, resolve_solver_backend  # noqa: E402


_INDEX_TO_HAND = {int(index): tuple(hand) for hand, index in _HAND_TO_INDEX.items()}


class LowSolverResidualNet(nn.Module):
    """Predict delta logits added to log(low-solver-policy)."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        if int(n_layers) < 1:
            raise ValueError("n_layers must be positive")
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(n_layers)):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, N_ACTIONS))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_records_for_labels(
    metadata_json: str | Path,
    labels: tuple[str, ...],
) -> list[dict[str, Any]]:
    metadata = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
    records = [dict(item) for item in metadata.get("cut_records", [])]
    if len(records) != len(labels):
        raise ValueError(f"metadata record count {len(records)} != dataset labels {len(labels)}")
    record_labels = [str(item.get("label")) for item in records]
    if record_labels == list(labels):
        return records
    by_label = {str(item.get("label")): dict(item) for item in records}
    if len(by_label) != len(records):
        raise ValueError("metadata labels must be unique when order differs")
    try:
        return [by_label[label] for label in labels]
    except KeyError as exc:
        raise ValueError(f"metadata is missing record for label {exc.args[0]!r}") from exc


def _root_overlap_summary(
    train_records: list[dict[str, Any]],
    holdout_records: list[dict[str, Any]],
) -> dict[str, Any]:
    train_roots = {
        str(record.get("root_label"))
        for record in train_records
        if record.get("root_label") is not None
    }
    holdout_roots = {
        str(record.get("root_label"))
        for record in holdout_records
        if record.get("root_label") is not None
    }
    overlap = sorted(train_roots & holdout_roots)
    return {
        "train_root_count": int(len(train_roots)),
        "holdout_root_count": int(len(holdout_roots)),
        "overlap_count": int(len(overlap)),
        "overlap_sample": overlap[:10],
        "passed": bool(train_roots and holdout_roots and not overlap),
    }


def _residual_features(
    dataset: JointPBSDataset,
    low_policy: np.ndarray,
) -> np.ndarray:
    parts = [
        np.asarray(dataset.features, dtype=np.float32),
        np.asarray(dataset.policy_features[:, :52], dtype=np.float32),
        np.asarray(dataset.belief, dtype=np.float32),
        np.asarray(dataset.legal_masks, dtype=np.float32),
        np.asarray(low_policy, dtype=np.float32),
    ]
    n = parts[0].shape[0]
    expected = [
        (n, N_FEATURES),
        (n, 52),
        (n, BELIEF_DIM),
        (n, N_ACTIONS),
        (n, N_ACTIONS),
    ]
    for idx, (part, shape) in enumerate(zip(parts, expected, strict=True)):
        if part.shape != shape:
            raise ValueError(f"residual feature part {idx} shape {part.shape} != {shape}")
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _residual_logits(
    delta_logits: torch.Tensor,
    low_policy: torch.Tensor,
    legal_masks: torch.Tensor,
) -> torch.Tensor:
    legal = (legal_masks > 0).to(dtype=delta_logits.dtype)
    base = torch.log(low_policy.to(dtype=delta_logits.dtype).clamp(min=1e-6))
    logits = base + delta_logits
    return logits.masked_fill(legal <= 0, -1e4)


def _predict_residual_policy(
    model: LowSolverResidualNet,
    features: np.ndarray,
    low_policy: np.ndarray,
    legal_masks: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], max(1, int(batch_size))):
            stop = min(start + max(1, int(batch_size)), features.shape[0])
            x = torch.from_numpy(features[start:stop]).to(device)
            low = torch.from_numpy(low_policy[start:stop]).to(device)
            legal = torch.from_numpy(legal_masks[start:stop]).to(device)
            logits = _residual_logits(model(x), low, legal)
            preds.append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32))
    if not preds:
        return np.zeros((0, N_ACTIONS), dtype=np.float32)
    return _normalize_targets(np.concatenate(preds, axis=0), legal_masks)


def _case_by_label(cases_json: str | Path) -> tuple[dict[str, Any], list[Any]]:
    cases = load_cases_json(cases_json)
    by_label = {str(case.label): (idx, case) for idx, case in enumerate(cases)}
    if len(by_label) != len(cases):
        raise ValueError("case labels must be unique")
    return by_label, cases


def _compute_low_solver_policies(
    dataset: JointPBSDataset,
    records: list[dict[str, Any]],
    *,
    cases_json: str | Path,
    cfv_cache: str | Path,
    solver_iterations: int,
    solver_backend: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    from poker_ai.research.belief_value_probe import load_public_belief_cfv_dataset_cache

    if len(records) != dataset.features.shape[0]:
        raise ValueError("record count must match dataset rows")
    case_lookup, cases = _case_by_label(cases_json)
    base_dataset, _base_records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if base_dataset.features.shape[0] != len(cases):
        raise ValueError("CFV cache row count does not match cases")

    low_policy = _legal_uniform_policy(dataset)
    by_root: dict[str, list[int]] = defaultdict(list)
    for row_idx, record in enumerate(records):
        root = record.get("root_label")
        if root is not None and float(record.get("policy_weight", 0.0)) > 0.0:
            by_root[str(root)].append(row_idx)

    solved_roots = 0
    skipped_roots: list[dict[str, Any]] = []
    filled_rows = 0
    solver_ms: list[float] = []
    backend, backend_device = resolve_solver_backend(solver_backend)

    for root_label, row_indices in sorted(by_root.items()):
        lookup = case_lookup.get(root_label)
        if lookup is None:
            skipped_roots.append({"root_label": root_label, "reason": "missing_case"})
            continue
        case_idx, case = lookup
        parsed = parse_action(case.action_str)
        if "error" in parsed:
            skipped_roots.append({"root_label": root_label, "reason": parsed["error"]})
            continue
        street = int(parsed.get("st", -1))
        if street not in (2, 3):
            skipped_roots.append({"root_label": root_label, "reason": f"street:{street}"})
            continue
        board_count = 4 if street == 2 else 5
        board_idx = [card_str_to_index(card) for card in case.board[:board_count]]
        our_bet_pre, opp_bet_pre = _compute_bets_before_street(
            case.action_str,
            case.client_pos,
            target_street=street,
        )
        pot = int(our_bet_pre + opp_bet_pre)
        hero_stack = int(20000 - our_bet_pre)
        villain_stack = int(20000 - opp_bet_pre)
        hero_first = bool(case.client_pos == 0)
        street_action = (
            case.action_str.split("/")[street]
            if len(case.action_str.split("/")) > street
            else ""
        )
        solver_hands = list(itertools.combinations(sorted(set(range(52)) - set(board_idx)), 2))
        hero_range, villain_range = _local_ranges_from_belief(
            base_dataset.belief[int(case_idx)],
            solver_hands,
        )
        solver = StreetSolver(board_idx, pot, hero_stack, villain_stack, hero_first)
        solver.solve(
            n_iterations=int(solver_iterations),
            hero_range=hero_range,
            villain_range=villain_range,
            backend=backend,
            device=backend_device,
        )
        solver_ms.append(float(getattr(solver, "last_solve_ms", 0.0)))
        node = solver.navigate(_parse_nav(street_action, solver))
        if node is None or node.is_terminal:
            skipped_roots.append({"root_label": root_label, "reason": "terminal_or_missing_node"})
            continue
        solved_roots += 1
        for row_idx in row_indices:
            hand_index = int(records[row_idx].get("hand_index", -1))
            hand = _INDEX_TO_HAND.get(hand_index)
            if hand is None:
                continue
            strategy = _strategy_vector(solver.get_strategy(hand, node)).astype(np.float32)
            legal = dataset.legal_masks[row_idx : row_idx + 1]
            low_policy[row_idx] = _normalize_targets(strategy[np.newaxis, :], legal)[0]
            filled_rows += 1

    policy_rows = int(np.count_nonzero(dataset.policy_weights > 0.0))
    stats = {
        "solver_iterations": int(solver_iterations),
        "solver_backend": str(solver_backend),
        "root_count": int(len(by_root)),
        "solved_root_count": int(solved_roots),
        "skipped_root_count": int(len(skipped_roots)),
        "skipped_root_sample": skipped_roots[:10],
        "policy_row_count": int(policy_rows),
        "filled_row_count": int(filled_rows),
        "missing_policy_row_count": int(max(policy_rows - filled_rows, 0)),
        "mean_solver_ms": round(float(np.mean(solver_ms)), 6) if solver_ms else 0.0,
    }
    return low_policy.astype(np.float32, copy=False), stats


def train_joint_pbs_low_solver_residual(
    *,
    train_joint_npz: str | Path,
    holdout_joint_npz: str | Path,
    train_metadata_json: str | Path,
    holdout_metadata_json: str | Path,
    cases_json: str | Path,
    cfv_cache: str | Path,
    device: str | torch.device = "auto",
    low_solver_iterations: int = 5,
    solver_backend: str = "cpu",
    hidden_dim: int = 256,
    n_layers: int = 2,
    epochs: int = 20,
    batch_size: int = 8192,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    amp_dtype: str = "bf16",
    seed: int = 0,
    output_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    train_raw = load_joint_pbs_dataset(train_joint_npz, metadata_json=train_metadata_json)
    holdout_raw = load_joint_pbs_dataset(holdout_joint_npz, metadata_json=holdout_metadata_json)
    train_records = _load_records_for_labels(train_metadata_json, train_raw.labels)
    holdout_records = _load_records_for_labels(holdout_metadata_json, holdout_raw.labels)
    root_overlap = _root_overlap_summary(train_records, holdout_records)
    train, holdout, _public_mean, _public_std, _belief_mean, _belief_std = _standardize_joint_pair(
        train_raw,
        holdout_raw,
    )
    train_low, train_low_stats = _compute_low_solver_policies(
        train_raw,
        train_records,
        cases_json=cases_json,
        cfv_cache=cfv_cache,
        solver_iterations=low_solver_iterations,
        solver_backend=solver_backend,
    )
    holdout_low, holdout_low_stats = _compute_low_solver_policies(
        holdout_raw,
        holdout_records,
        cases_json=cases_json,
        cfv_cache=cfv_cache,
        solver_iterations=low_solver_iterations,
        solver_backend=solver_backend,
    )
    train_x = _residual_features(train, train_low)
    holdout_x = _residual_features(holdout, holdout_low)
    train_mask = train.policy_weights > 0.0
    holdout_mask = holdout.policy_weights > 0.0
    if not np.any(train_mask) or not np.any(holdout_mask):
        raise ValueError("residual diagnostic requires train and holdout policy labels")

    torch.manual_seed(int(seed))
    model = LowSolverResidualNet(train_x.shape[1], int(hidden_dim), int(n_layers)).to(resolved_device)
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    x_t = torch.from_numpy(train_x[train_mask]).to(resolved_device)
    low_t = torch.from_numpy(train_low[train_mask]).to(resolved_device)
    legal_t = torch.from_numpy(train.legal_masks[train_mask]).to(resolved_device)
    target_t = torch.from_numpy(train.target_probs[train_mask]).to(resolved_device)
    weight_t = torch.from_numpy(train.policy_weights[train_mask]).to(resolved_device)

    use_amp = resolved_device.type == "cuda" and amp_dtype != "none"
    amp_torch_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == "fp16")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == "fp16")
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(int(seed))
    losses: list[float] = []
    n = int(x_t.shape[0])
    batch_size = max(1, min(int(batch_size), n))
    model.train()
    for _epoch in range(max(1, int(epochs))):
        perm = torch.randperm(n, generator=generator, device=resolved_device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=resolved_device.type,
                dtype=amp_torch_dtype,
                enabled=use_amp,
            ):
                logits = _residual_logits(
                    model(x_t.index_select(0, idx)),
                    low_t.index_select(0, idx),
                    legal_t.index_select(0, idx),
                )
                loss = _masked_policy_loss(
                    logits,
                    legal_t.index_select(0, idx),
                    target_t.index_select(0, idx),
                    weight_t.index_select(0, idx),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

    pred = _predict_residual_policy(
        model,
        holdout_x,
        holdout_low,
        holdout.legal_masks,
        batch_size=batch_size,
        device=resolved_device,
    )
    low_metrics = _policy_metrics(
        holdout_low[holdout_mask],
        holdout.target_probs[holdout_mask],
        holdout.legal_masks[holdout_mask],
    )
    residual_metrics = _policy_metrics(
        pred[holdout_mask],
        holdout.target_probs[holdout_mask],
        holdout.legal_masks[holdout_mask],
    )
    uniform_metrics = _policy_metrics(
        _legal_uniform_policy(holdout)[holdout_mask],
        holdout.target_probs[holdout_mask],
        holdout.legal_masks[holdout_mask],
    )
    beats_low = bool(
        residual_metrics["mean_l1"] < low_metrics["mean_l1"]
        and residual_metrics["mean_kl"] < low_metrics["mean_kl"]
    )
    root_passed = bool(root_overlap["passed"])
    low_complete = bool(
        train_low_stats["missing_policy_row_count"] == 0
        and holdout_low_stats["missing_policy_row_count"] == 0
    )
    if output_checkpoint is not None:
        path = Path(output_checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mode": "joint_pbs_low_solver_residual",
                "model_state_dict": model.state_dict(),
                "input_dim": int(train_x.shape[1]),
                "hidden_dim": int(hidden_dim),
                "n_layers": int(n_layers),
                "low_solver_iterations": int(low_solver_iterations),
                "solver_backend": str(solver_backend),
                "train_joint_npz": str(train_joint_npz),
                "holdout_joint_npz": str(holdout_joint_npz),
            },
            path,
        )

    return {
        "mode": "joint_pbs_low_solver_residual",
        "passed": bool(root_passed and low_complete and beats_low),
        "pass_criteria": (
            "root-disjoint metadata, complete low-solver coverage, and learned "
            "low-policy residual must improve holdout L1/KL over the low solver"
        ),
        "device": str(resolved_device),
        "train_joint_npz": str(train_joint_npz),
        "holdout_joint_npz": str(holdout_joint_npz),
        "train_metadata_json": str(train_metadata_json),
        "holdout_metadata_json": str(holdout_metadata_json),
        "cases_json": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "low_solver_iterations": int(low_solver_iterations),
        "solver_backend": str(solver_backend),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "amp_dtype": str(amp_dtype),
        "seed": int(seed),
        "checkpoint": str(output_checkpoint) if output_checkpoint is not None else None,
        "train_size": int(train.features.shape[0]),
        "holdout_size": int(holdout.features.shape[0]),
        "input_dim": int(train_x.shape[1]),
        "root_disjoint_audit": root_overlap,
        "train_low_solver": train_low_stats,
        "holdout_low_solver": holdout_low_stats,
        "low_solver_policy": low_metrics,
        "residual_policy": residual_metrics,
        "legal_uniform_policy": uniform_metrics,
        "residual_beats_low_solver": bool(beats_low),
        "low_solver_complete": bool(low_complete),
        "final_train_loss": round(float(losses[-1]), 8) if losses else math.nan,
        "mean_train_loss": round(float(np.mean(losses)), 8) if losses else math.nan,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a low-solver residual correction on joint-PBS policy targets."
    )
    parser.add_argument("--train-joint", required=True)
    parser.add_argument("--holdout-joint", required=True)
    parser.add_argument("--train-metadata-json", required=True)
    parser.add_argument("--holdout-metadata-json", required=True)
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--low-solver-iterations", type=int, default=5)
    parser.add_argument("--solver-backend", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp-dtype", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-checkpoint")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = train_joint_pbs_low_solver_residual(
        train_joint_npz=args.train_joint,
        holdout_joint_npz=args.holdout_joint,
        train_metadata_json=args.train_metadata_json,
        holdout_metadata_json=args.holdout_metadata_json,
        cases_json=args.cases_json,
        cfv_cache=args.cfv_cache,
        device=args.device,
        low_solver_iterations=args.low_solver_iterations,
        solver_backend=args.solver_backend,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp_dtype=args.amp_dtype,
        seed=args.seed,
        output_checkpoint=args.output_checkpoint,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
