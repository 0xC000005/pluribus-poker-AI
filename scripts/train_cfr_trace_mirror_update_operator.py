#!/usr/bin/env python3
"""Train an amortized mirror-descent update operator from CFR traces."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_cfr_trace_predictor import _load_payload  # noqa: E402
from eval_cfr_trace_advantage_trust_region import (  # noqa: E402
    _center_scale_advantage,
    _legal_mask,
    _normalize_policy,
)
from poker_ai.research.belief_value_probe import save_metrics  # noqa: E402
from train_cfr_trace_delta_mlp import _advantage_context_row  # noqa: E402
from train_cfr_trace_policy_residual import (  # noqa: E402
    _feature_row,
    _mean,
    _rate,
    _record_map,
    _standardize,
    _top_match,
)


class MirrorUpdateOperator(nn.Module):
    """Predict a non-negative mirror-descent step size for one root trace."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, max_eta: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = int(input_dim)
        for _ in range(max(1, int(n_layers))):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)
        self.max_eta = float(max_eta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_eta * torch.sigmoid(self.net(x)).squeeze(-1)


def _device_name(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return requested


def mirror_update_policy(
    record: dict[str, Any],
    *,
    eta: float,
    min_base_prob: float = 1e-4,
) -> np.ndarray:
    """Apply one legal-mask mirror-descent update from the record advantage."""
    low_policy = np.asarray(record["strategy_policy"], dtype=np.float64).reshape(-1)
    legal_mask = _legal_mask(record, int(low_policy.shape[0]))
    legal_uniform = legal_mask / float(legal_mask.sum())
    base_policy = _normalize_policy(
        np.maximum(low_policy, float(min_base_prob) * legal_uniform),
        legal_mask,
    )
    advantage = _center_scale_advantage(
        np.asarray(record.get("counterfactual_advantage", []), dtype=np.float64),
        legal_mask,
        base_policy,
    )
    logits = np.log(np.clip(base_policy, 1e-12, 1.0)) + float(eta) * advantage
    logits = np.where(legal_mask > 0, logits, -1e9)
    logits = logits - float(np.max(logits[legal_mask > 0]))
    probs = np.exp(logits) * legal_mask
    return probs / max(float(probs.sum()), 1e-12)


def _common_labels(*maps: dict[str, dict[str, Any]]) -> list[str]:
    common = set(maps[0])
    for mapping in maps[1:]:
        common &= set(mapping)
    labels = sorted(common)
    if not labels:
        raise ValueError("no common labels across requested trace iterations")
    return labels


def _base_and_advantage(
    record: dict[str, Any],
    *,
    action_dim: int,
    min_base_prob: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    legal_mask = _legal_mask(record, action_dim)
    legal_uniform = legal_mask / float(legal_mask.sum())
    low_policy = np.asarray(record["strategy_policy"], dtype=np.float64).reshape(-1)
    base_policy = _normalize_policy(
        np.maximum(low_policy, float(min_base_prob) * legal_uniform),
        legal_mask,
    )
    advantage = _center_scale_advantage(
        np.asarray(record.get("counterfactual_advantage", []), dtype=np.float64),
        legal_mask,
        base_policy,
    )
    return base_policy, advantage, legal_mask


def _build_dataset(
    low_by_label: dict[str, dict[str, Any]],
    target_by_label: dict[str, dict[str, Any]],
    labels: list[str],
    *,
    min_base_prob: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action_dim = len(low_by_label[labels[0]]["strategy_policy"])
    features: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    advantages: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for label in labels:
        low_record = low_by_label[label]
        public_context = np.asarray(low_record.get("public_belief_features", ()), dtype=np.float64).reshape(-1)
        features.append(
            np.concatenate(
                [
                    _feature_row(low_record, action_dim),
                    public_context,
                    _advantage_context_row(low_record, action_dim),
                ]
            )
        )
        base, advantage, legal_mask = _base_and_advantage(
            low_record,
            action_dim=action_dim,
            min_base_prob=min_base_prob,
        )
        bases.append(base)
        advantages.append(advantage)
        masks.append(legal_mask)
        targets.append(np.asarray(target_by_label[label]["strategy_policy"], dtype=np.float64).reshape(-1))
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(bases, dtype=np.float64),
        np.asarray(advantages, dtype=np.float64),
        np.asarray(masks, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
    )


def _apply_update_torch(
    base_policy: torch.Tensor,
    advantage: torch.Tensor,
    legal_mask: torch.Tensor,
    eta: torch.Tensor,
) -> torch.Tensor:
    logits = torch.log(torch.clamp(base_policy, min=1e-12)) + eta.unsqueeze(-1) * advantage
    logits = logits.masked_fill(legal_mask <= 0, -1e9)
    logits = logits - logits.max(dim=1, keepdim=True).values
    probs = torch.exp(logits) * legal_mask
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _train_operator(
    x_train: np.ndarray,
    base_train: np.ndarray,
    advantage_train: np.ndarray,
    mask_train: np.ndarray,
    target_train: np.ndarray,
    *,
    hidden_dim: int,
    n_layers: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: str,
    max_eta: float,
) -> tuple[MirrorUpdateOperator, dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device_name = _device_name(device)
    model = MirrorUpdateOperator(
        input_dim=int(x_train.shape[1]),
        hidden_dim=int(hidden_dim),
        n_layers=int(n_layers),
        max_eta=float(max_eta),
    ).to(device_name)
    x = torch.as_tensor(x_train, dtype=torch.float32, device=device_name)
    base = torch.as_tensor(base_train, dtype=torch.float32, device=device_name)
    advantage = torch.as_tensor(advantage_train, dtype=torch.float32, device=device_name)
    mask = torch.as_tensor(mask_train, dtype=torch.float32, device=device_name)
    target = torch.as_tensor(target_train, dtype=torch.float32, device=device_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    n = int(x.shape[0])
    batch_size = max(1, min(int(batch_size), n))
    started = time.perf_counter()
    final_loss = 0.0
    for _epoch in range(int(epochs)):
        perm = torch.randperm(n, device=device_name)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            eta = model(x[idx])
            pred = _apply_update_torch(base[idx], advantage[idx], mask[idx], eta)
            denom = mask[idx].sum().clamp_min(1.0)
            loss = (((pred - target[idx]) * mask[idx]) ** 2).sum() / denom
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
    elapsed = time.perf_counter() - started
    return model, {
        "device": device_name,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "max_eta": float(max_eta),
        "train_seconds": round(float(elapsed), 6),
        "final_loss": round(float(final_loss), 10),
    }


def _predict_eta(model: MirrorUpdateOperator, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        eta = model(torch.as_tensor(x, dtype=torch.float32, device=device))
    return eta.detach().cpu().numpy().astype(np.float64)


def _evaluate(
    labels: list[str],
    low_by_label: dict[str, dict[str, Any]],
    target_by_label: dict[str, dict[str, Any]],
    uniform_by_label: dict[str, dict[str, Any]],
    reference_by_label: dict[str, dict[str, Any]],
    predicted_eta: np.ndarray,
    *,
    min_base_prob: float,
) -> dict[str, Any]:
    update_l1: list[float] = []
    update_to_target_l1: list[float] = []
    low_l1: list[float] = []
    low_to_target_l1: list[float] = []
    uniform_l1: list[float] = []
    update_match: list[bool] = []
    low_match: list[bool] = []
    uniform_match: list[bool] = []
    records: list[dict[str, Any]] = []
    for row_idx, label in enumerate(labels):
        low_record = low_by_label[label]
        eta = float(predicted_eta[row_idx])
        update_policy = mirror_update_policy(low_record, eta=eta, min_base_prob=min_base_prob)
        low_policy = np.asarray(low_record["strategy_policy"], dtype=np.float64)
        target = np.asarray(target_by_label[label]["strategy_policy"], dtype=np.float64)
        uniform_policy = np.asarray(uniform_by_label[label]["strategy_policy"], dtype=np.float64)
        reference = np.asarray(reference_by_label[label]["strategy_policy"], dtype=np.float64)
        update_value = float(np.abs(update_policy - reference).sum())
        update_target_value = float(np.abs(update_policy - target).sum())
        low_value = float(np.abs(low_policy - reference).sum())
        low_target_value = float(np.abs(low_policy - target).sum())
        uniform_value = float(np.abs(uniform_policy - reference).sum())
        update_l1.append(update_value)
        update_to_target_l1.append(update_target_value)
        low_l1.append(low_value)
        low_to_target_l1.append(low_target_value)
        uniform_l1.append(uniform_value)
        update_match.append(_top_match(update_policy, reference))
        low_match.append(_top_match(low_policy, reference))
        uniform_match.append(_top_match(uniform_policy, reference))
        records.append(
            {
                "label": label,
                "eta": round(eta, 8),
                "update_l1_to_reference": round(update_value, 8),
                "update_l1_to_target": round(update_target_value, 8),
                "low_l1_to_reference": round(low_value, 8),
                "low_l1_to_target": round(low_target_value, 8),
                "uniform_l1_to_reference": round(uniform_value, 8),
                "update_top_matches_reference": bool(update_match[-1]),
                "low_top_matches_reference": bool(low_match[-1]),
                "uniform_top_matches_reference": bool(uniform_match[-1]),
                "update_policy": update_policy.round(8).tolist(),
            }
        )
    mean_update = _mean(update_l1)
    mean_update_to_target = _mean(update_to_target_l1)
    mean_low = _mean(low_l1)
    mean_low_to_target = _mean(low_to_target_l1)
    mean_uniform = _mean(uniform_l1)
    target_fit_passed = mean_update_to_target < mean_low_to_target
    decision_passed = (
        mean_update < mean_low
        and mean_update < mean_uniform
        and _rate(update_match) >= _rate(uniform_match)
    )
    return {
        "target_fit_passed": bool(target_fit_passed),
        "decision_passed": bool(decision_passed),
        "mean_update_l1_to_reference": mean_update,
        "mean_update_l1_to_target": mean_update_to_target,
        "mean_low_l1_to_reference": mean_low,
        "mean_low_l1_to_target": mean_low_to_target,
        "mean_uniform_l1_to_reference": mean_uniform,
        "update_top_match_rate": _rate(update_match),
        "low_top_match_rate": _rate(low_match),
        "uniform_top_match_rate": _rate(uniform_match),
        "mean_eta": round(float(np.mean(predicted_eta)), 8) if len(predicted_eta) else 0.0,
        "max_eta": round(float(np.max(predicted_eta)), 8) if len(predicted_eta) else 0.0,
        "records": records,
    }


def fit_mirror_update_operator_from_payloads(
    train_payload: dict[str, Any],
    holdout_payload: dict[str, Any],
    *,
    low_trace_iteration: int = 5,
    target_trace_iteration: int = 24,
    uniform_trace_iteration: int = 10,
    reference_trace_iteration: int = 24,
    hidden_dim: int = 64,
    n_layers: int = 2,
    epochs: int = 400,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    seed: int = 20260521,
    device: str = "auto",
    max_eta: float = 8.0,
    min_base_prob: float = 1e-4,
) -> dict[str, Any]:
    train_low = _record_map(train_payload, low_trace_iteration)
    train_target = _record_map(train_payload, target_trace_iteration)
    train_labels = _common_labels(train_low, train_target)
    x_train, base_train, advantage_train, mask_train, target_train = _build_dataset(
        train_low,
        train_target,
        train_labels,
        min_base_prob=min_base_prob,
    )
    x_train, feature_mean, feature_std = _standardize(x_train)
    model, train_metrics = _train_operator(
        x_train,
        base_train,
        advantage_train,
        mask_train,
        target_train,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
        device=device,
        max_eta=max_eta,
    )

    holdout_low = _record_map(holdout_payload, low_trace_iteration)
    holdout_target = _record_map(holdout_payload, target_trace_iteration)
    holdout_uniform = _record_map(holdout_payload, uniform_trace_iteration)
    holdout_reference = _record_map(holdout_payload, reference_trace_iteration)
    holdout_labels = _common_labels(
        holdout_low,
        holdout_target,
        holdout_uniform,
        holdout_reference,
    )
    x_holdout, _base_holdout, _advantage_holdout, _mask_holdout, _target_holdout = _build_dataset(
        holdout_low,
        holdout_target,
        holdout_labels,
        min_base_prob=min_base_prob,
    )
    x_holdout, _, _ = _standardize(x_holdout, mean=feature_mean, std=feature_std)
    predicted_eta = _predict_eta(model, x_holdout, train_metrics["device"])
    holdout_metrics = _evaluate(
        holdout_labels,
        holdout_low,
        holdout_target,
        holdout_uniform,
        holdout_reference,
        predicted_eta,
        min_base_prob=min_base_prob,
    )
    passed = bool(
        holdout_metrics["target_fit_passed"]
        and holdout_metrics["decision_passed"]
        and holdout_metrics["mean_eta"] > 1e-6
    )
    return {
        "mode": "cfr_trace_mirror_update_operator",
        "promotion": False,
        "passed": passed,
        "target_fit_passed": bool(holdout_metrics["target_fit_passed"]),
        "decision_passed": bool(holdout_metrics["decision_passed"]),
        "low_trace_iteration": int(low_trace_iteration),
        "target_trace_iteration": int(target_trace_iteration),
        "uniform_trace_iteration": int(uniform_trace_iteration),
        "reference_trace_iteration": int(reference_trace_iteration),
        "min_base_prob": float(min_base_prob),
        "n_train": int(len(train_labels)),
        "n_holdout": int(len(holdout_labels)),
        "holdout": holdout_metrics,
        "model": {
            **train_metrics,
            "feature_mean": feature_mean.round(10).tolist(),
            "feature_std": feature_std.round(10).tolist(),
        },
    }


def fit_mirror_update_operator(
    *,
    train_trace_json: str | Path,
    holdout_trace_json: str | Path,
    low_trace_iteration: int = 5,
    target_trace_iteration: int = 24,
    uniform_trace_iteration: int = 10,
    reference_trace_iteration: int = 24,
    hidden_dim: int = 64,
    n_layers: int = 2,
    epochs: int = 400,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    seed: int = 20260521,
    device: str = "auto",
    max_eta: float = 8.0,
    min_base_prob: float = 1e-4,
) -> dict[str, Any]:
    metrics = fit_mirror_update_operator_from_payloads(
        _load_payload(train_trace_json),
        _load_payload(holdout_trace_json),
        low_trace_iteration=low_trace_iteration,
        target_trace_iteration=target_trace_iteration,
        uniform_trace_iteration=uniform_trace_iteration,
        reference_trace_iteration=reference_trace_iteration,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
        device=device,
        max_eta=max_eta,
        min_base_prob=min_base_prob,
    )
    metrics["train_trace_json"] = str(train_trace_json)
    metrics["holdout_trace_json"] = str(holdout_trace_json)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trace-json", required=True)
    parser.add_argument("--holdout-trace-json", required=True)
    parser.add_argument("--low-trace-iteration", type=int, default=5)
    parser.add_argument("--target-trace-iteration", type=int, default=24)
    parser.add_argument("--uniform-trace-iteration", type=int, default=10)
    parser.add_argument("--reference-trace-iteration", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-eta", type=float, default=8.0)
    parser.add_argument("--min-base-prob", type=float, default=1e-4)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    metrics = fit_mirror_update_operator(
        train_trace_json=args.train_trace_json,
        holdout_trace_json=args.holdout_trace_json,
        low_trace_iteration=args.low_trace_iteration,
        target_trace_iteration=args.target_trace_iteration,
        uniform_trace_iteration=args.uniform_trace_iteration,
        reference_trace_iteration=args.reference_trace_iteration,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        max_eta=args.max_eta,
        min_base_prob=args.min_base_prob,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
