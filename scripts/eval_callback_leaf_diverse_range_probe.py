#!/usr/bin/env python3
"""Probe whether diverse opponent ranges can repair learned-leaf decision flips.

This is a diagnostic, not a gameplay path. It takes existing callback-leaf A/B
records, selects high-margin roots where the learned leaf flips the exact
solver's top action, then re-solves the learned leaf under generic opponent
range perturbations. The goal is to test whether the local failure points toward
a multi-valued/diverse-range depth-limit target before training another model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from poker_ai.research.belief_probe import N_HANDS, _resolve_device  # noqa: E402
from poker_ai.research.belief_value_probe import (  # noqa: E402
    load_public_belief_cfv_dataset_cache,
    save_metrics,
)
from poker_ai.research.resolver_benchmark import load_cases_json  # noqa: E402

from eval_public_belief_dcvn_leaf_ab import (  # noqa: E402
    PublicBeliefDCVNLeafCallback,
    _apply_leaf_predictions,
    _solve_case,
)
from eval_public_belief_dual_hand_cfv_probe import (  # noqa: E402
    load_public_belief_dual_hand_cfv_checkpoint,
    predict_public_belief_dual_hand_cfv_model_vectorized,
    project_dual_cfv_zero_sum,
)
from play_slumbot import build_features, parse_action  # noqa: E402


DEFAULT_PERTURBATIONS = (
    "original",
    "villain_temp_0.5",
    "villain_temp_2.0",
    "villain_uniform_mix_0.25",
    "villain_uniform_mix_0.50",
)

DEFAULT_VALUE_SET_AGGREGATIONS = ("mean", "hero_pessimistic", "villain_best_response")


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy()
    total = float(values.sum())
    if total <= 1e-12:
        return values
    return values / total


def perturb_public_belief(belief_row: np.ndarray, perturbation: str) -> np.ndarray:
    """Return a public-belief row with a generic villain-range perturbation."""
    row = np.asarray(belief_row, dtype=np.float32).copy()
    if row.shape != (2 * N_HANDS,):
        raise ValueError(f"belief row must have shape {(2 * N_HANDS,)}, got {row.shape}")
    row[:N_HANDS] = _normalize(row[:N_HANDS])
    if perturbation == "original":
        row[N_HANDS:] = _normalize(row[N_HANDS:])
        return row

    villain = row[N_HANDS:].copy()
    support = villain > 1e-12
    if perturbation.startswith("villain_temp_"):
        temp = float(perturbation.removeprefix("villain_temp_"))
        if temp <= 0.0:
            raise ValueError("temperature perturbation must be positive")
        adjusted = np.zeros_like(villain)
        adjusted[support] = villain[support] ** temp
        villain = adjusted
    elif perturbation.startswith("villain_uniform_mix_"):
        weight = float(perturbation.removeprefix("villain_uniform_mix_"))
        if not (0.0 <= weight <= 1.0):
            raise ValueError("uniform-mix perturbation must be in [0, 1]")
        uniform = np.zeros_like(villain)
        if int(support.sum()) > 0:
            uniform[support] = 1.0 / float(int(support.sum()))
        villain = (1.0 - weight) * villain + weight * uniform
    else:
        raise ValueError(f"unknown perturbation: {perturbation}")
    row[N_HANDS:] = _normalize(villain)
    return row


def aggregate_value_set_predictions(
    predictions: np.ndarray,
    mode: str,
    *,
    belief: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate variant predictions into one deployable diagnostic leaf value."""
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim != 4 or predictions.shape[1] != 2 or predictions.shape[3] != N_HANDS:
        raise ValueError("predictions must have shape (n_variants, 2, n_states, N_HANDS)")
    if mode == "mean":
        return predictions.mean(axis=0, dtype=np.float32)
    if mode == "hero_pessimistic":
        out = np.empty_like(predictions[0])
        out[0] = predictions[:, 0].min(axis=0)
        out[1] = predictions[:, 1].max(axis=0)
        return out
    if mode == "villain_best_response":
        if belief is None:
            raise ValueError("belief is required for villain_best_response aggregation")
        belief = np.asarray(belief, dtype=np.float32)
        if belief.ndim == 1:
            belief = belief.reshape(1, -1)
        if belief.shape != (predictions.shape[2], 2 * N_HANDS):
            raise ValueError(
                "belief must have shape "
                f"{(predictions.shape[2], 2 * N_HANDS)}, got {belief.shape}"
            )
        villain = np.stack([_normalize(row[N_HANDS:]) for row in belief]).astype(np.float32)
        scores = np.einsum("vsn,sn->vs", predictions[:, 1], villain)
        choices = np.argmax(scores, axis=0)
        out = np.empty_like(predictions[0])
        for state_idx, variant_idx in enumerate(choices.tolist()):
            out[:, state_idx, :] = predictions[int(variant_idx), :, state_idx, :]
        return out
    raise ValueError(f"unknown value-set aggregation: {mode}")


class DiverseRangeValueSetLeafCallback(PublicBeliefDCVNLeafCallback):
    """Aggregate multiple villain-range leaf predictions before CFR updates."""

    def __init__(
        self,
        *,
        perturbations: tuple[str, ...],
        value_set_aggregation: str,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.perturbations = tuple(perturbations)
        self.value_set_aggregation = str(value_set_aggregation)

    def __call__(self, **kwargs):
        tree = kwargs["tree"]
        showdown_indices = np.asarray(kwargs["showdown_indices"], dtype=np.int32)
        hero_reach = np.asarray(kwargs["hero_reach"], dtype=np.float32)
        villain_reach = np.asarray(kwargs["villain_reach"], dtype=np.float32)
        valid_m = np.asarray(kwargs["valid_m"], dtype=np.float32)
        out_h = np.asarray(kwargs["default_hero_values"], dtype=np.float32).copy()
        out_v = np.asarray(kwargs["default_villain_values"], dtype=np.float32).copy()
        self.stats.callback_calls += 1

        features: list[np.ndarray] = []
        beliefs: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        tasks: list[int] = []
        for row, node_idx in enumerate(showdown_indices.tolist()):
            action_str = self._terminal_action_str(tree, int(node_idx))
            parsed = parse_action(action_str)
            if "error" in parsed:
                self.stats.fallback_showdowns += 1
                continue
            hero_belief = np.zeros(N_HANDS, dtype=np.float32)
            villain_belief = np.zeros(N_HANDS, dtype=np.float32)
            hero_belief[self._local_to_global] = hero_reach[row]
            villain_belief[self._local_to_global] = villain_reach[row]
            hero_belief = _normalize(hero_belief * self.board_mask)
            villain_belief = _normalize(villain_belief * self.board_mask)
            features.append(
                build_features(
                    [],
                    self.board_str,
                    action_str,
                    self.case.client_pos,
                    parsed,
                )
            )
            beliefs.append(np.concatenate([hero_belief, villain_belief]).astype(np.float32))
            masks.append(self.board_mask)
            tasks.append(row)

        if not tasks:
            return out_h, out_v

        features_np = np.stack(features).astype(np.float32)
        base_beliefs = np.stack(beliefs).astype(np.float32)
        masks_np = np.stack(masks).astype(np.float32)
        started = time.perf_counter()
        variant_predictions: list[np.ndarray] = []
        for perturbation in self.perturbations:
            beliefs_np = np.stack(
                [perturb_public_belief(row, perturbation) for row in base_beliefs],
            ).astype(np.float32)
            pred = predict_public_belief_dual_hand_cfv_model_vectorized(
                self.model,
                self.payload,
                features_np,
                beliefs_np,
                masks_np,
                masks_np,
                device=self.device,
                state_batch_size=self.state_batch_size,
                hand_batch_size=self.hand_batch_size,
            )
            if self.project_zero_sum:
                pred = project_dual_cfv_zero_sum(pred, beliefs_np, masks_np, masks_np)
            variant_predictions.append(pred)
        pred = aggregate_value_set_predictions(
            np.stack(variant_predictions, axis=0),
            self.value_set_aggregation,
            belief=base_beliefs,
        )
        pred = pred * self.value_scale
        self.stats.prediction_ms += (time.perf_counter() - started) * 1000.0
        self.stats.prediction_states += int(len(tasks))

        out_h, out_v = _apply_leaf_predictions(
            out_h=out_h,
            out_v=out_v,
            pred=pred,
            tasks=tasks,
            local_to_global=self._local_to_global,
            valid_m=valid_m,
            hero_reach=hero_reach,
            villain_reach=villain_reach,
        )
        self.stats.replaced_showdowns += len(tasks)
        return out_h, out_v


def _strategy_margin(strategy: list[float] | np.ndarray) -> float:
    probs = np.sort(np.asarray(strategy, dtype=np.float64))[::-1]
    if probs.size == 0:
        return 0.0
    if probs.size == 1:
        return float(probs[0])
    return float(probs[0] - probs[1])


def load_high_margin_flip_records(
    leaf_ab_jsons: list[str | Path],
    *,
    min_margin: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Select high-confidence action flips from existing leaf A/B artifacts."""
    candidates: list[dict[str, Any]] = []
    for path_like in leaf_ab_jsons:
        path = Path(path_like)
        payload = json.loads(path.read_text(encoding="utf-8"))
        start_index = int(payload.get("start_index", 0))
        for offset, record in enumerate(payload.get("records", [])):
            if not (record.get("passed") and "action_l1_drift" in record):
                continue
            if bool(record.get("action_agreement", False)):
                continue
            margin = _strategy_margin(record.get("baseline_strategy", []))
            if margin < float(min_margin):
                continue
            candidates.append(
                {
                    **record,
                    "case_index": start_index + offset,
                    "source_json": str(path),
                    "baseline_margin": float(margin),
                }
            )
    candidates.sort(
        key=lambda item: (
            float(item.get("action_l1_drift", 0.0)),
            float(item.get("baseline_margin", 0.0)),
        ),
        reverse=True,
    )
    return candidates[: max(0, int(limit))]


def _plurality_action(actions: list[int]) -> int | None:
    if not actions:
        return None
    counts = Counter(actions)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def summarize_diverse_range_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "n_roots": 0,
            "original_mean_l1": None,
            "range_avg_mean_l1": None,
            "best_variant_oracle_mean_l1": None,
            "original_action_agreement": None,
            "range_avg_action_agreement": None,
            "best_variant_oracle_action_agreement": None,
            "plurality_action_agreement": None,
        }
    return {
        "n_roots": int(len(records)),
        "original_mean_l1": float(np.mean([record["original_l1"] for record in records])),
        "range_avg_mean_l1": float(np.mean([record["range_avg_l1"] for record in records])),
        "best_variant_oracle_mean_l1": float(
            np.mean([record["best_variant_l1"] for record in records])
        ),
        "original_action_agreement": float(
            np.mean(
                [
                    int(record["baseline_action"] == record["original_learned_action"])
                    for record in records
                ]
            )
        ),
        "range_avg_action_agreement": float(
            np.mean([int(record["range_avg_agree"]) for record in records])
        ),
        "best_variant_oracle_action_agreement": float(
            np.mean([int(record["best_variant_agree"]) for record in records])
        ),
        "plurality_action_agreement": float(
            np.mean([int(record["plurality_action_agree"]) for record in records])
        ),
    }


def summarize_integrated_value_sets(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        for item in record.get("integrated_value_sets", []):
            if item.get("passed"):
                by_mode.setdefault(str(item["aggregation"]), []).append(item)
    return {
        mode: {
            "n_roots": int(len(items)),
            "mean_l1": float(np.mean([float(item["l1"]) for item in items])),
            "action_agreement": float(np.mean([int(item["action_agree"]) for item in items])),
            "mean_solve_ms": float(np.mean([float(item["learned_solve_ms"]) for item in items])),
            "mean_leaf_prediction_ms": float(
                np.mean([float(item["leaf_prediction_ms"]) for item in items])
            ),
        }
        for mode, items in sorted(by_mode.items())
    }


def eval_callback_leaf_diverse_range_probe(
    *,
    checkpoint: str | Path,
    cases_json: str | Path,
    cfv_cache: str | Path,
    leaf_ab_jsons: list[str | Path],
    device: str = "auto",
    solver_iterations: int = 10,
    solver_backend: str = "cpu",
    value_scale: float = 20000.0,
    state_batch_size: int = 16,
    hand_batch_size: int = N_HANDS,
    perturbations: tuple[str, ...] = DEFAULT_PERTURBATIONS,
    value_set_aggregations: tuple[str, ...] = (),
    min_margin: float = 0.5,
    limit: int = 8,
    project_zero_sum: bool = False,
) -> dict[str, Any]:
    cases = load_cases_json(cases_json)
    dataset, _records = load_public_belief_cfv_dataset_cache(cfv_cache)
    if dataset.features.shape[0] != len(cases):
        raise ValueError("case count does not match CFV cache rows")
    selected = load_high_margin_flip_records(
        leaf_ab_jsons,
        min_margin=min_margin,
        limit=limit,
    )
    resolved_device = _resolve_device(device)
    model, payload = load_public_belief_dual_hand_cfv_checkpoint(
        checkpoint,
        device=resolved_device,
    )

    records: list[dict[str, Any]] = []
    for selected_record in selected:
        case_index = int(selected_record["case_index"])
        baseline_strategy = np.asarray(
            selected_record["baseline_strategy"],
            dtype=np.float32,
        )
        original_strategy = np.asarray(
            selected_record["learned_strategy"],
            dtype=np.float32,
        )
        variants: list[dict[str, Any]] = []
        for perturbation in perturbations:
            belief = perturb_public_belief(dataset.belief[case_index], perturbation)
            solved = _solve_case(
                cases[case_index],
                belief_row=belief,
                model=model,
                payload=payload,
                device=resolved_device,
                solver_iterations=solver_iterations,
                solver_backend=solver_backend,
                value_scale=value_scale,
                state_batch_size=state_batch_size,
                hand_batch_size=hand_batch_size,
                project_zero_sum=project_zero_sum,
            )
            if not (solved.get("passed") and "learned_strategy" in solved):
                variants.append(
                    {
                        "perturbation": perturbation,
                        "passed": False,
                        "error": solved.get("error") or solved.get("skipped") or "unknown",
                    }
                )
                continue
            strategy = np.asarray(solved["learned_strategy"], dtype=np.float32)
            variants.append(
                {
                    "perturbation": perturbation,
                    "passed": True,
                    "action": int(np.argmax(strategy)),
                    "strategy": strategy.round(6).tolist(),
                    "l1_to_original_baseline": float(np.abs(strategy - baseline_strategy).sum()),
                    "learned_solve_ms": solved.get("learned_solve_ms"),
                    "leaf_prediction_ms": solved.get("leaf_prediction_ms"),
                }
            )
        valid_variants = [variant for variant in variants if variant.get("passed")]
        if not valid_variants:
            continue
        variant_strategies = [
            np.asarray(variant["strategy"], dtype=np.float32) for variant in valid_variants
        ]
        range_avg = np.mean(variant_strategies, axis=0)
        baseline_action = int(np.argmax(baseline_strategy))
        best_variant = min(valid_variants, key=lambda item: float(item["l1_to_original_baseline"]))
        plurality_action = _plurality_action([int(variant["action"]) for variant in valid_variants])
        integrated_value_sets: list[dict[str, Any]] = []
        for aggregation in value_set_aggregations:
            solved = _solve_case(
                cases[case_index],
                belief_row=dataset.belief[case_index],
                model=model,
                payload=payload,
                device=resolved_device,
                solver_iterations=solver_iterations,
                solver_backend=solver_backend,
                value_scale=value_scale,
                state_batch_size=state_batch_size,
                hand_batch_size=hand_batch_size,
                project_zero_sum=project_zero_sum,
                leaf_callback_cls=DiverseRangeValueSetLeafCallback,
                leaf_callback_kwargs={
                    "perturbations": perturbations,
                    "value_set_aggregation": aggregation,
                },
            )
            if not (solved.get("passed") and "learned_strategy" in solved):
                integrated_value_sets.append(
                    {
                        "aggregation": aggregation,
                        "passed": False,
                        "error": solved.get("error") or solved.get("skipped") or "unknown",
                    }
                )
                continue
            strategy = np.asarray(solved["learned_strategy"], dtype=np.float32)
            action = int(np.argmax(strategy))
            integrated_value_sets.append(
                {
                    "aggregation": aggregation,
                    "passed": True,
                    "action": action,
                    "strategy": strategy.round(6).tolist(),
                    "l1": float(np.abs(strategy - baseline_strategy).sum()),
                    "action_agree": bool(action == baseline_action),
                    "learned_solve_ms": solved.get("learned_solve_ms"),
                    "leaf_prediction_ms": solved.get("leaf_prediction_ms"),
                }
            )
        records.append(
            {
                "label": selected_record.get("label"),
                "case_index": case_index,
                "source_json": selected_record.get("source_json"),
                "baseline_margin": float(selected_record["baseline_margin"]),
                "baseline_action": baseline_action,
                "original_learned_action": int(np.argmax(original_strategy)),
                "original_l1": float(np.abs(original_strategy - baseline_strategy).sum()),
                "range_avg_action": int(np.argmax(range_avg)),
                "range_avg_l1": float(np.abs(range_avg - baseline_strategy).sum()),
                "range_avg_agree": bool(int(np.argmax(range_avg)) == baseline_action),
                "best_variant_perturbation": str(best_variant["perturbation"]),
                "best_variant_action": int(best_variant["action"]),
                "best_variant_l1": float(best_variant["l1_to_original_baseline"]),
                "best_variant_agree": bool(int(best_variant["action"]) == baseline_action),
                "plurality_action": plurality_action,
                "plurality_action_agree": bool(plurality_action == baseline_action),
                "variants": variants,
                "integrated_value_sets": integrated_value_sets,
            }
        )
    summary = summarize_diverse_range_records(records)
    integrated_summary = summarize_integrated_value_sets(records)
    return {
        "mode": "callback_leaf_diverse_opponent_range_probe",
        "passed": bool(records),
        "promotion": False,
        "promotion_blockers": [
            "diagnostic_positive_control_only",
            "oracle_variant_is_not_deployable",
            "requires_value_set_or_search_integration_before_live_use",
        ],
        "checkpoint": str(checkpoint),
        "cases": str(cases_json),
        "cfv_cache": str(cfv_cache),
        "leaf_ab_jsons": [str(path) for path in leaf_ab_jsons],
        "device": str(resolved_device),
        "solver_iterations": int(solver_iterations),
        "solver_backend": str(solver_backend),
        "value_scale": float(value_scale),
        "state_batch_size": int(state_batch_size),
        "hand_batch_size": int(hand_batch_size),
        "project_zero_sum": bool(project_zero_sum),
        "perturbations": list(perturbations),
        "value_set_aggregations": list(value_set_aggregations),
        "integrated_value_set_summaries": integrated_summary,
        "min_margin": float(min_margin),
        "selection_limit": int(limit),
        "n_selected": int(len(selected)),
        **summary,
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cfv-cache", required=True)
    parser.add_argument("--leaf-ab-json", action="append", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--value-scale", type=float, default=20000.0)
    parser.add_argument("--state-batch-size", type=int, default=16)
    parser.add_argument("--hand-batch-size", type=int, default=N_HANDS)
    parser.add_argument("--perturbation", action="append", dest="perturbations")
    parser.add_argument(
        "--value-set-aggregation",
        action="append",
        choices=DEFAULT_VALUE_SET_AGGREGATIONS,
        dest="value_set_aggregations",
        help="Opt-in integrated leaf-value aggregation to test inside CFR.",
    )
    parser.add_argument("--min-margin", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--project-zero-sum", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = eval_callback_leaf_diverse_range_probe(
        checkpoint=args.checkpoint,
        cases_json=args.cases,
        cfv_cache=args.cfv_cache,
        leaf_ab_jsons=args.leaf_ab_json,
        device=args.device,
        solver_iterations=args.solver_iterations,
        solver_backend=args.solver_backend,
        value_scale=args.value_scale,
        state_batch_size=args.state_batch_size,
        hand_batch_size=args.hand_batch_size,
        perturbations=tuple(args.perturbations or DEFAULT_PERTURBATIONS),
        value_set_aggregations=tuple(args.value_set_aggregations or ()),
        min_margin=args.min_margin,
        limit=args.limit,
        project_zero_sum=args.project_zero_sum,
    )
    if args.output_json:
        save_metrics(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
