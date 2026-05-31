"""Feature-level policy probe for trace-start hard states."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poker_ai.deep_cfr.networks import PolicyNetwork
from poker_ai.games.full_deck.state import N_ACTIONS
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase
from poker_ai.research.trace_start_contract import build_trace_start_observation


_HAND_RE = re.compile(r"trace-(hand[^-]+)-")


def value_soft_target(
    action_values: np.ndarray,
    *,
    legal_actions: list[int],
    temperature: float = 500.0,
) -> np.ndarray:
    """Convert counterfactual action values into a legal soft policy target."""
    values = np.asarray(action_values, dtype=np.float32)
    target = np.zeros_like(values, dtype=np.float32)
    legal = [int(action) for action in legal_actions if 0 <= int(action) < values.shape[0]]
    if not legal:
        raise ValueError("legal_actions must contain at least one valid action")
    temp = max(float(temperature), 1e-6)
    legal_values = values[legal] / temp
    legal_values = legal_values - float(np.max(legal_values))
    exp_values = np.exp(legal_values)
    probs = exp_values / max(float(exp_values.sum()), 1e-12)
    target[legal] = probs.astype(np.float32)
    return target


def _hand_key(record: dict[str, Any]) -> str:
    label = str(record.get("label", ""))
    match = _HAND_RE.search(label)
    if match:
        return match.group(1)
    return label


def split_trace_records_by_hand(
    records: list[dict[str, Any]],
    *,
    holdout_fraction: float = 0.33,
    seed: int = 20260526,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records by hand id so decisions from one hand never leak across splits."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_hand_key(record)].append(record)
    keys = sorted(grouped)
    rng = random.Random(int(seed))
    rng.shuffle(keys)
    n_holdout = int(round(len(keys) * float(holdout_fraction)))
    if len(keys) > 1:
        n_holdout = min(max(n_holdout, 1), len(keys) - 1)
    holdout_keys = set(keys[:n_holdout])
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for key in sorted(grouped):
        if key in holdout_keys:
            holdout.extend(grouped[key])
        else:
            train.extend(grouped[key])
    return train, holdout


def split_trace_records_by_source(
    records: list[dict[str, Any]],
    *,
    holdout_source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records by explicit trace source for leave-trace-out validation."""
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("_trace_source", "")) == str(holdout_source):
            holdout.append(record)
        else:
            train.append(record)
    if not train or not holdout:
        raise ValueError(f"holdout_source={holdout_source!r} did not produce both splits")
    return train, holdout


def _record_to_case(record: dict[str, Any]) -> ResolverBenchmarkCase:
    return ResolverBenchmarkCase(
        label=str(record["label"]),
        hole_cards=tuple(record["hole_cards"]),
        board=tuple(record["board"]),
        action_str=str(record["action_str"]),
        client_pos=int(record["client_pos"]),
        source=str(record.get("source", "trace_ev_gate")),
    )


def _features_and_targets(
    records: list[dict[str, Any]],
    *,
    source_flag: float,
    target_temperature: float,
) -> dict[str, np.ndarray]:
    features: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    baseline_actions: list[int] = []
    for record in records:
        obs = build_trace_start_observation(_record_to_case(record), source_flag=source_flag)
        values = np.asarray(record["counterfactual_action_values"], dtype=np.float32)
        if values.shape[0] != N_ACTIONS:
            raise ValueError(f"record {record.get('label')} has {values.shape[0]} action values")
        legal_actions = [int(action) for action in record["legal_actions"]]
        target = value_soft_target(
            values,
            legal_actions=legal_actions,
            temperature=target_temperature,
        )
        features.append(obs.features_with_context.astype(np.float32, copy=False))
        legal_masks.append(obs.legal_mask.astype(np.float32, copy=False))
        targets.append(target)
        action_values.append(values)
        baseline_actions.append(int(record.get("baseline_action", int(np.argmax(target)))))
    return {
        "features": np.asarray(features, dtype=np.float32),
        "legal_masks": np.asarray(legal_masks, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "action_values": np.asarray(action_values, dtype=np.float32),
        "baseline_actions": np.asarray(baseline_actions, dtype=np.int64),
    }


def _resolve_device(device: str) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _evaluate_policy(
    policy_net: PolicyNetwork,
    examples: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> dict[str, Any]:
    if examples["features"].shape[0] == 0:
        return {
            "n": 0,
            "mean_selected_ev": 0.0,
            "mean_baseline_ev": 0.0,
            "mean_selected_ev_delta_vs_baseline": 0.0,
            "selected_action_counts": {},
            "distinct_selected_actions": 0,
        }
    x = torch.from_numpy(examples["features"]).to(device)
    masks = torch.from_numpy(examples["legal_masks"]).to(device)
    with torch.no_grad():
        logits = policy_net(x).masked_fill(masks <= 0, -1e4)
        actions = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
    values = examples["action_values"]
    baseline_actions = examples["baseline_actions"]
    selected_ev = values[np.arange(values.shape[0]), actions]
    baseline_ev = values[np.arange(values.shape[0]), baseline_actions]
    masked_values = np.where(examples["legal_masks"] > 0, values, -np.inf)
    oracle_actions = np.argmax(masked_values, axis=1).astype(np.int64)
    oracle_ev = values[np.arange(values.shape[0]), oracle_actions]
    counts: dict[str, int] = {}
    for action in actions.tolist():
        counts[str(int(action))] = counts.get(str(int(action)), 0) + 1
    oracle_counts: dict[str, int] = {}
    for action in oracle_actions.tolist():
        oracle_counts[str(int(action))] = oracle_counts.get(str(int(action)), 0) + 1
    return {
        "n": int(values.shape[0]),
        "mean_selected_ev": float(np.mean(selected_ev)),
        "mean_baseline_ev": float(np.mean(baseline_ev)),
        "mean_selected_ev_delta_vs_baseline": float(np.mean(selected_ev - baseline_ev)),
        "mean_oracle_ev": float(np.mean(oracle_ev)),
        "mean_oracle_ev_delta_vs_baseline": float(np.mean(oracle_ev - baseline_ev)),
        "selected_action_counts": dict(sorted(counts.items())),
        "oracle_action_counts": dict(sorted(oracle_counts.items())),
        "distinct_selected_actions": int(len(counts)),
    }


def train_trace_start_policy_head(
    ev_gate_data: dict[str, Any],
    *,
    source_flag: float = 1.0,
    target_temperature: float = 500.0,
    holdout_fraction: float = 0.33,
    hidden_dim: int = 64,
    n_layers: int = 2,
    n_steps: int = 300,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 20260526,
    device: str = "auto",
    holdout_source: str | None = None,
) -> tuple[PolicyNetwork, dict[str, Any]]:
    """Train a diagnostic policy head from trace-start counterfactual values."""
    input_records = list(ev_gate_data.get("records") or [])
    records = [
        record
        for record in input_records
        if not record.get("skipped") and "counterfactual_action_values" in record
    ]
    if not records:
        raise ValueError("ev_gate_data must contain nonempty records")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch_device = _resolve_device(device)
    if holdout_source is None:
        train_records, holdout_records = split_trace_records_by_hand(
            records,
            holdout_fraction=holdout_fraction,
            seed=seed,
        )
        split_mode = "by_hand"
    else:
        train_records, holdout_records = split_trace_records_by_source(
            records,
            holdout_source=holdout_source,
        )
        split_mode = "by_trace_source"
    train_examples = _features_and_targets(
        train_records,
        source_flag=source_flag,
        target_temperature=target_temperature,
    )
    holdout_examples = _features_and_targets(
        holdout_records,
        source_flag=source_flag,
        target_temperature=target_temperature,
    )
    feature_dim = int(train_examples["features"].shape[1])
    policy_net = PolicyNetwork(
        input_dim=feature_dim,
        hidden_dim=int(hidden_dim),
        output_dim=N_ACTIONS,
        n_layers=int(n_layers),
    ).to(torch_device)
    optimizer = optim.Adam(policy_net.parameters(), lr=float(lr))
    x = torch.from_numpy(train_examples["features"]).to(torch_device)
    masks = torch.from_numpy(train_examples["legal_masks"]).to(torch_device)
    targets = torch.from_numpy(train_examples["targets"]).to(torch_device)
    n_train = int(x.shape[0])
    losses: list[float] = []
    for step in range(int(n_steps)):
        idx = torch.randint(0, n_train, (min(int(batch_size), n_train),), device=torch_device)
        logits = policy_net(x[idx]).masked_fill(masks[idx] <= 0, -1e4)
        log_probs = nn.functional.log_softmax(logits, dim=1)
        loss = -(targets[idx] * log_probs).sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or step == int(n_steps) - 1:
            losses.append(float(loss.detach().cpu()))

    train_eval = _evaluate_policy(policy_net, train_examples, device=torch_device)
    holdout_eval = _evaluate_policy(policy_net, holdout_examples, device=torch_device)
    metrics = {
        "mode": "trace_start_policy_head_probe",
        "passed": True,
        "promotion_blockers": [
            "feature_level_trace_policy_probe_is_not_deployable",
            "counterfactual_ev_targets_are_trace_derived",
        ],
        "n_records": len(records),
        "n_input_records": len(input_records),
        "n_filtered_records": len(input_records) - len(records),
        "n_train": len(train_records),
        "n_holdout": len(holdout_records),
        "feature_context_dim": feature_dim,
        "source_flag": float(source_flag),
        "target_temperature": float(target_temperature),
        "holdout_fraction": float(holdout_fraction),
        "holdout_source": holdout_source,
        "split_mode": split_mode,
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "device": str(torch_device),
        "initial_loss": float(losses[0]) if losses else 0.0,
        "final_loss": float(losses[-1]) if losses else 0.0,
        "train_mean_selected_ev_delta_vs_baseline": train_eval[
            "mean_selected_ev_delta_vs_baseline"
        ],
        "holdout_mean_selected_ev_delta_vs_baseline": holdout_eval[
            "mean_selected_ev_delta_vs_baseline"
        ],
        "train_eval": train_eval,
        "holdout_eval": holdout_eval,
        "decision_gate_passed": bool(
            holdout_eval["n"] > 0
            and holdout_eval["mean_selected_ev_delta_vs_baseline"] > 0.0
            and holdout_eval["distinct_selected_actions"] >= 2
        ),
    }
    return policy_net, metrics
