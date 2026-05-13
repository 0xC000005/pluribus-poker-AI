"""Utilities for reproducible train/holdout splits of search targets."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.research.belief_value_probe import (
    PublicBeliefCFVDataset,
    save_public_belief_cfv_dataset_cache,
)
from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase, load_cases_json


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import parse_action  # noqa: E402


@dataclass(frozen=True)
class SearchTargetSplitInput:
    targets_npz: Path
    cases_json: Path
    cfv_cache_npz: Path | None = None


@dataclass(frozen=True)
class SearchTargetSplitOutput:
    train_targets_npz: Path
    train_cases_json: Path
    holdout_targets_npz: Path
    holdout_cases_json: Path
    train_cfv_cache_npz: Path | None = None
    holdout_cfv_cache_npz: Path | None = None
    metadata_json: Path | None = None


def _case_to_json(case: ResolverBenchmarkCase) -> dict[str, Any]:
    return asdict(case)


def _save_cases(path: Path, cases: list[ResolverBenchmarkCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"cases": [_case_to_json(case) for case in cases]}, indent=2),
        encoding="utf-8",
    )


def _case_street(case: ResolverBenchmarkCase) -> int:
    parsed = parse_action(case.action_str)
    if "st" in parsed:
        street = int(parsed["st"])
        if street in (2, 3):
            return street
    if len(case.board) == 4:
        return 2
    if len(case.board) == 5:
        return 3
    raise ValueError(f"{case.label}: cannot infer target street")


def _load_cfv_cache(path: Path) -> tuple[PublicBeliefCFVDataset, list[dict[str, Any]]]:
    data = np.load(path, allow_pickle=False)
    records_json = str(data["records_json"].item()) if "records_json" in data.files else "[]"
    return (
        PublicBeliefCFVDataset(
            features=data["features"].astype(np.float32, copy=False),
            belief=data["belief"].astype(np.float32, copy=False),
            values=data["values"].astype(np.float32, copy=False),
            value_masks=data["value_masks"].astype(np.float32, copy=False),
            labels=tuple(str(item) for item in data["labels"].tolist()),
        ),
        json.loads(records_json),
    )


def _concat_inputs(
    inputs: Iterable[SearchTargetSplitInput],
) -> tuple[
    PolicyTargetBuffer,
    list[ResolverBenchmarkCase],
    PublicBeliefCFVDataset | None,
    list[dict[str, Any]] | None,
]:
    target_features = []
    target_masks = []
    target_probs = []
    target_weights = []
    cases: list[ResolverBenchmarkCase] = []
    cfv_features = []
    cfv_belief = []
    cfv_values = []
    cfv_masks = []
    cfv_labels = []
    cfv_records: list[dict[str, Any]] = []
    saw_cfv = False

    for item in inputs:
        buffer = PolicyTargetBuffer.from_npz(item.targets_npz)
        artifact_cases = load_cases_json(item.cases_json)
        if len(artifact_cases) != buffer.size:
            raise ValueError(
                f"{item.cases_json}: {len(artifact_cases)} cases for {buffer.size} targets"
            )

        target_features.append(buffer.features)
        target_masks.append(buffer.legal_masks)
        target_probs.append(buffer.target_probs)
        target_weights.append(buffer.weights)
        cases.extend(artifact_cases)

        if item.cfv_cache_npz is None:
            continue
        saw_cfv = True
        cfv, records = _load_cfv_cache(item.cfv_cache_npz)
        if cfv.values.shape[0] != buffer.size:
            raise ValueError(
                f"{item.cfv_cache_npz}: {cfv.values.shape[0]} CFV rows for {buffer.size} targets"
            )
        if cfv.features.shape != buffer.features.shape:
            raise ValueError(
                f"{item.cfv_cache_npz}: CFV feature shape {cfv.features.shape} "
                f"does not match target feature shape {buffer.features.shape}"
            )
        case_labels = tuple(case.label for case in artifact_cases)
        if cfv.labels != case_labels:
            raise ValueError(f"{item.cfv_cache_npz}: CFV labels do not match case order")
        cfv_features.append(cfv.features)
        cfv_belief.append(cfv.belief)
        cfv_values.append(cfv.values)
        cfv_masks.append(cfv.value_masks)
        cfv_labels.extend(cfv.labels)
        cfv_records.extend(records)

    targets = PolicyTargetBuffer(
        np.concatenate(target_features, axis=0),
        np.concatenate(target_masks, axis=0),
        np.concatenate(target_probs, axis=0),
        np.concatenate(target_weights, axis=0),
    )
    if not saw_cfv:
        return targets, cases, None, None
    if len(cfv_features) != len(target_features):
        raise ValueError("either provide CFV caches for every input artifact or for none")
    cfv_dataset = PublicBeliefCFVDataset(
        features=np.concatenate(cfv_features, axis=0).astype(np.float32, copy=False),
        belief=np.concatenate(cfv_belief, axis=0).astype(np.float32, copy=False),
        values=np.concatenate(cfv_values, axis=0).astype(np.float32, copy=False),
        value_masks=np.concatenate(cfv_masks, axis=0).astype(np.float32, copy=False),
        labels=tuple(cfv_labels),
    )
    return targets, cases, cfv_dataset, cfv_records


def _value_bins(values: np.ndarray, masks: np.ndarray, streets: np.ndarray, bins: int) -> np.ndarray:
    bins = max(1, int(bins))
    row_means = np.zeros(values.shape[0], dtype=np.float64)
    selected = masks > 0
    for idx in range(values.shape[0]):
        active = values[idx][selected[idx]]
        row_means[idx] = float(active.mean()) if active.size else 0.0

    out = np.zeros(values.shape[0], dtype=np.int64)
    for street in sorted(set(int(item) for item in streets.tolist())):
        street_idx = np.where(streets == street)[0]
        if street_idx.size <= 1 or bins <= 1:
            continue
        quantiles = np.quantile(
            row_means[street_idx],
            np.linspace(0.0, 1.0, bins + 1, dtype=np.float64),
        )
        edges = np.unique(quantiles[1:-1])
        if edges.size:
            out[street_idx] = np.searchsorted(edges, row_means[street_idx], side="right")
    return out


def _proportional_counts(capacities: dict[tuple[int, int], int], total: int) -> dict[tuple[int, int], int]:
    total = int(total)
    if total < 0:
        raise ValueError("total must be non-negative")
    capacity_sum = sum(capacities.values())
    if total > capacity_sum:
        raise ValueError(f"requested {total} rows from only {capacity_sum} available rows")
    if total == 0:
        return {key: 0 for key in capacities}

    raw = {key: total * count / capacity_sum for key, count in capacities.items()}
    counts = {key: min(int(np.floor(value)), capacities[key]) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    ranked = sorted(
        capacities,
        key=lambda key: (raw[key] - np.floor(raw[key]), capacities[key], key),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for key in ranked:
            if counts[key] >= capacities[key]:
                continue
            counts[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("failed to allocate proportional split counts")
    return counts


def _stratified_indices(
    *,
    streets: np.ndarray,
    value_bins: np.ndarray,
    train_size: int,
    holdout_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n_total = int(streets.shape[0])
    requested = int(train_size) + int(holdout_size)
    if train_size <= 0 or holdout_size <= 0:
        raise ValueError("train_size and holdout_size must both be positive")
    if requested > n_total:
        raise ValueError(f"requested {requested} rows from {n_total} available rows")

    rng = np.random.default_rng(int(seed))
    strata: dict[tuple[int, int], np.ndarray] = {}
    for street in sorted(set(int(item) for item in streets.tolist())):
        for bin_id in sorted(set(int(item) for item in value_bins[streets == street].tolist())):
            idx = np.where((streets == street) & (value_bins == bin_id))[0]
            if idx.size:
                strata[(street, bin_id)] = rng.permutation(idx)

    capacities = {key: int(value.size) for key, value in strata.items()}
    selected_counts = _proportional_counts(capacities, requested)
    holdout_counts = _proportional_counts(selected_counts, int(holdout_size))

    train_idx = []
    holdout_idx = []
    for key, shuffled in strata.items():
        selected = shuffled[: selected_counts[key]]
        n_holdout = holdout_counts[key]
        holdout_idx.extend(selected[:n_holdout].tolist())
        train_idx.extend(selected[n_holdout:].tolist())

    train_arr = np.asarray(train_idx, dtype=np.int64)
    holdout_arr = np.asarray(holdout_idx, dtype=np.int64)
    rng.shuffle(train_arr)
    rng.shuffle(holdout_arr)
    if train_arr.size != int(train_size) or holdout_arr.size != int(holdout_size):
        raise RuntimeError(
            f"split produced {train_arr.size}/{holdout_arr.size}, expected {train_size}/{holdout_size}"
        )

    metadata = {
        "seed": int(seed),
        "train_size": int(train_size),
        "holdout_size": int(holdout_size),
        "n_input_rows": n_total,
        "strata": [
            {
                "street": int(street),
                "value_bin": int(bin_id),
                "available": int(capacities[(street, bin_id)]),
                "selected": int(selected_counts[(street, bin_id)]),
                "train": int(selected_counts[(street, bin_id)] - holdout_counts[(street, bin_id)]),
                "holdout": int(holdout_counts[(street, bin_id)]),
            }
            for street, bin_id in sorted(strata)
        ],
    }
    return train_arr, holdout_arr, metadata


def _subset_targets(targets: PolicyTargetBuffer, indices: np.ndarray) -> PolicyTargetBuffer:
    return PolicyTargetBuffer(
        targets.features[indices],
        targets.legal_masks[indices],
        targets.target_probs[indices],
        targets.weights[indices],
    )


def _subset_cfv(dataset: PublicBeliefCFVDataset, indices: np.ndarray) -> PublicBeliefCFVDataset:
    return PublicBeliefCFVDataset(
        features=dataset.features[indices],
        belief=dataset.belief[indices],
        values=dataset.values[indices],
        value_masks=dataset.value_masks[indices],
        labels=tuple(dataset.labels[int(idx)] for idx in indices.tolist()),
    )


def split_search_targets_stratified(
    inputs: Iterable[SearchTargetSplitInput],
    output: SearchTargetSplitOutput,
    *,
    train_size: int,
    holdout_size: int,
    seed: int = 0,
    cfv_bins: int = 4,
) -> dict[str, Any]:
    """Write a deterministic stratified split for search-target diagnostics."""
    targets, cases, cfv_dataset, cfv_records = _concat_inputs(inputs)
    streets = np.asarray([_case_street(case) for case in cases], dtype=np.int64)
    if cfv_dataset is not None:
        bins = _value_bins(cfv_dataset.values, cfv_dataset.value_masks, streets, cfv_bins)
    else:
        top_actions = np.argmax(targets.target_probs, axis=1).astype(np.int64)
        bins = top_actions

    train_idx, holdout_idx, metadata = _stratified_indices(
        streets=streets,
        value_bins=bins,
        train_size=train_size,
        holdout_size=holdout_size,
        seed=seed,
    )

    _subset_targets(targets, train_idx).save_npz(output.train_targets_npz)
    _subset_targets(targets, holdout_idx).save_npz(output.holdout_targets_npz)
    _save_cases(output.train_cases_json, [cases[int(idx)] for idx in train_idx.tolist()])
    _save_cases(output.holdout_cases_json, [cases[int(idx)] for idx in holdout_idx.tolist()])

    if cfv_dataset is not None:
        if output.train_cfv_cache_npz is None or output.holdout_cfv_cache_npz is None:
            raise ValueError("CFV cache outputs are required when CFV cache inputs are provided")
        assert cfv_records is not None
        save_public_belief_cfv_dataset_cache(
            _subset_cfv(cfv_dataset, train_idx),
            [cfv_records[int(idx)] for idx in train_idx.tolist()],
            output.train_cfv_cache_npz,
        )
        save_public_belief_cfv_dataset_cache(
            _subset_cfv(cfv_dataset, holdout_idx),
            [cfv_records[int(idx)] for idx in holdout_idx.tolist()],
            output.holdout_cfv_cache_npz,
        )

    metadata = {
        **metadata,
        "cfv_bins": int(cfv_bins) if cfv_dataset is not None else None,
        "stratify_by": "street+cfv_mean_bin" if cfv_dataset is not None else "street+target_top_action",
        "train_targets_npz": str(output.train_targets_npz),
        "train_cases_json": str(output.train_cases_json),
        "holdout_targets_npz": str(output.holdout_targets_npz),
        "holdout_cases_json": str(output.holdout_cases_json),
        "train_cfv_cache_npz": str(output.train_cfv_cache_npz)
        if output.train_cfv_cache_npz
        else None,
        "holdout_cfv_cache_npz": str(output.holdout_cfv_cache_npz)
        if output.holdout_cfv_cache_npz
        else None,
    }
    if output.metadata_json is not None:
        output.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        output.metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata
