"""Build search-consistency policy targets for Deep CFR."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.research.resolver_benchmark import (
    ResolverBenchmarkCase,
    _solver_decision,
    default_benchmark_cases,
    load_cases_json,
)


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import (  # noqa: E402
    build_features,
    get_legal_mask_from_parsed,
    parse_action,
)


def _normalized_legal_target(
    strategy: np.ndarray,
    legal_mask: np.ndarray,
    fallback_action: int,
) -> np.ndarray:
    target = np.asarray(strategy, dtype=np.float32) * (legal_mask > 0)
    total = float(target.sum())
    if total > 1e-8:
        return target / total
    fallback = np.zeros_like(target, dtype=np.float32)
    if 0 <= fallback_action < len(fallback) and legal_mask[fallback_action] > 0:
        fallback[fallback_action] = 1.0
        return fallback
    legal_total = float(legal_mask.sum())
    if legal_total <= 0:
        raise ValueError("cannot build target without legal actions")
    return legal_mask.astype(np.float32) / legal_total


def build_resolver_policy_targets(
    cases: Iterable[ResolverBenchmarkCase] | None = None,
    *,
    solver_iterations: int = 25,
    solver_backend: str = "auto",
) -> tuple[PolicyTargetBuffer, dict]:
    """Build a small policy-target dataset from deterministic resolver cases."""
    cases = list(default_benchmark_cases() if cases is None else cases)
    features: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    target_probs: list[np.ndarray] = []
    records: list[dict] = []

    for case in cases:
        parsed = parse_action(case.action_str)
        if "error" in parsed:
            records.append(
                {
                    "label": case.label,
                    "skipped": f"parse_error:{parsed['error']}",
                }
            )
            continue
        feature_vec = build_features(
            list(case.hole_cards),
            list(case.board),
            case.action_str,
            case.client_pos,
            parsed,
        ).astype(np.float32)
        legal_mask = get_legal_mask_from_parsed(
            parsed,
            case.action_str,
            case.client_pos,
        ).astype(np.float32)
        solver = _solver_decision(
            case,
            parsed,
            solver_iterations=solver_iterations,
            solver_backend=solver_backend,
        )
        if solver is None:
            records.append({"label": case.label, "skipped": "unsupported_street"})
            continue

        target = _normalized_legal_target(
            solver.strategy,
            legal_mask,
            fallback_action=solver.action,
        )
        features.append(feature_vec)
        legal_masks.append(legal_mask)
        target_probs.append(target)
        records.append(
            {
                "label": case.label,
                "street": int(parsed["st"]),
                "solver_action": int(solver.action),
                "solver_increment": solver.increment,
                "solver_latency_ms": round(float(solver.latency_ms), 3),
                "target_entropy": round(
                    float(-(target[target > 0] * np.log(target[target > 0])).sum()),
                    6,
                ),
            }
        )

    if not features:
        raise RuntimeError("no resolver policy targets were generated")

    buffer = PolicyTargetBuffer(
        np.stack(features, axis=0),
        np.stack(legal_masks, axis=0),
        np.stack(target_probs, axis=0),
    )
    metadata = {
        "mode": "resolver_policy_targets",
        "solver_iterations": int(solver_iterations),
        "solver_backend": solver_backend,
        "n_cases": len(cases),
        "n_targets": int(buffer.size),
        "records": records,
    }
    return buffer, metadata


def save_resolver_policy_targets(
    output: str | Path,
    *,
    cases_json: str | Path | None = None,
    solver_iterations: int = 25,
    solver_backend: str = "auto",
) -> dict:
    cases = load_cases_json(cases_json) if cases_json else default_benchmark_cases()
    buffer, metadata = build_resolver_policy_targets(
        cases,
        solver_iterations=solver_iterations,
        solver_backend=solver_backend,
    )
    output = Path(output)
    buffer.save_npz(output)
    metadata_path = output.with_suffix(".json")
    metadata["output"] = str(output)
    metadata["metadata"] = str(metadata_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata
