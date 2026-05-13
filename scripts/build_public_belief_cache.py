#!/usr/bin/env python3
"""Build a public-belief cache without extra CFV solving."""

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
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.research.belief_probe import (
    BELIEF_DIM,
    N_HANDS,
    _case_range_vectors,
    _resolve_device,
)
from poker_ai.research.belief_value_probe import (
    PublicBeliefCFVDataset,
    save_metrics,
    save_public_belief_cfv_dataset_cache,
)
from poker_ai.research.evaluation import (
    assert_strategy_source_supported,
    load_value_network_checkpoint,
)
from poker_ai.research.resolver_benchmark import load_cases_json


def _public_features(features: np.ndarray) -> np.ndarray:
    public = np.asarray(features, dtype=np.float32).copy()
    public[..., :52] = 0.0
    return public


def _range_summary(hero: np.ndarray, villain: np.ndarray) -> dict[str, float]:
    def side(values: np.ndarray, prefix: str) -> dict[str, float]:
        positive = values[values > 0]
        entropy = float(-(positive * np.log(np.maximum(positive, 1e-12))).sum())
        return {
            f"{prefix}_range_entropy": round(entropy, 6),
            f"{prefix}_range_support": int(positive.size),
            f"{prefix}_range_top1_mass": round(float(np.max(values)), 6),
            f"{prefix}_range_top10_mass": round(
                float(np.sort(values)[-10:].sum()) if values.size >= 10 else float(values.sum()),
                6,
            ),
        }

    return {**side(hero, "hero"), **side(villain, "villain")}


def build_public_belief_cache(
    *,
    targets_npz: str | Path,
    cases_json: str | Path,
    range_checkpoint: str | Path,
    output: str | Path,
    range_strategy_source: str = "regret",
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    loaded = load_value_network_checkpoint(range_checkpoint, resolved_device)
    assert_strategy_source_supported(loaded, range_strategy_source)
    targets = PolicyTargetBuffer.from_npz(targets_npz)
    cases = load_cases_json(cases_json)
    if len(cases) != targets.size:
        raise ValueError(f"case count {len(cases)} does not match target count {targets.size}")

    belief_rows = []
    records = []
    labels = []
    for case in cases:
        hero_range, villain_range = _case_range_vectors(
            case,
            value_net=loaded.value_net,
            device=resolved_device,
            strategy_source=range_strategy_source,
        )
        belief_rows.append(
            np.concatenate([hero_range, villain_range], axis=0).astype(
                np.float32,
                copy=False,
            )
        )
        labels.append(case.label)
        records.append(
            {
                "label": case.label,
                "mode": "public_belief_only",
                **_range_summary(hero_range, villain_range),
            }
        )

    n = int(targets.size)
    dataset = PublicBeliefCFVDataset(
        features=_public_features(targets.features),
        belief=np.stack(belief_rows, axis=0).astype(np.float32, copy=False),
        values=np.zeros((n, N_HANDS), dtype=np.float32),
        value_masks=np.zeros((n, N_HANDS), dtype=np.float32),
        labels=tuple(labels),
    )
    save_public_belief_cfv_dataset_cache(dataset, records, output)
    return {
        "mode": "public_belief_cache",
        "passed": True,
        "output": str(output),
        "targets_npz": str(targets_npz),
        "cases_json": str(cases_json),
        "range_checkpoint": str(range_checkpoint),
        "range_strategy_source": range_strategy_source,
        "device": str(resolved_device),
        "n_states": n,
        "feature_dim": int(targets.features.shape[1]),
        "belief_dim": int(BELIEF_DIM),
        "value_labels": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a public-belief cache for downstream dual-CFV labeling."
    )
    parser.add_argument("--targets", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--range-checkpoint", required=True)
    parser.add_argument(
        "--range-strategy-source",
        choices=("regret", "policy-head", "average-policy"),
        default="regret",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = build_public_belief_cache(
        targets_npz=args.targets,
        cases_json=args.cases,
        range_checkpoint=args.range_checkpoint,
        output=args.output,
        range_strategy_source=args.range_strategy_source,
        device=args.device,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
