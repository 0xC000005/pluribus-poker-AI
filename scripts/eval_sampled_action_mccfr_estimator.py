#!/usr/bin/env python3
"""Evaluate a sampled-action MCCFR regret estimator on synthetic infosets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.sampled_action_mccfr import (  # noqa: E402
    full_regret,
    sampled_action_regret_estimate,
)


def _parse_samples(value: str) -> list[int]:
    samples = [int(part) for part in value.split(",") if part.strip()]
    if not samples or any(sample <= 0 for sample in samples):
        raise argparse.ArgumentTypeError("samples must be positive integers")
    return samples


def run_diagnostic(
    *,
    n_cases: int,
    n_repeats: int,
    n_actions: int,
    samples_per_estimate: list[int],
    uniform_mix: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    uniform_mix = float(np.clip(uniform_mix, 0.0, 1.0))
    results = {
        sample_count: {
            "sum_abs_bias": 0.0,
            "sum_l2_bias": 0.0,
            "sum_estimator_std": 0.0,
        }
        for sample_count in samples_per_estimate
    }

    for _ in range(n_cases):
        legal_mask = np.ones(n_actions, dtype=np.float32)
        values = rng.normal(loc=0.0, scale=1.0, size=n_actions).astype(np.float32)
        strategy = rng.dirichlet(np.ones(n_actions)).astype(np.float32)
        uniform = np.full(n_actions, 1.0 / n_actions, dtype=np.float32)
        sample_probs = (
            (1.0 - uniform_mix) * strategy + uniform_mix * uniform
        ).astype(np.float32)
        target = full_regret(values, strategy, legal_mask).astype(np.float64)

        for sample_count in samples_per_estimate:
            estimates = np.empty((n_repeats, n_actions), dtype=np.float32)
            for repeat_idx in range(n_repeats):
                actions = rng.choice(
                    n_actions,
                    size=sample_count,
                    replace=True,
                    p=sample_probs,
                )
                estimates[repeat_idx] = sampled_action_regret_estimate(
                    values,
                    strategy,
                    legal_mask,
                    sampled_actions=actions,
                    sample_probs=sample_probs,
                )
            mean_estimate = estimates.mean(axis=0).astype(np.float64)
            bias = mean_estimate - target
            results[sample_count]["sum_abs_bias"] += float(np.mean(np.abs(bias)))
            results[sample_count]["sum_l2_bias"] += float(np.sqrt(np.mean(bias ** 2)))
            results[sample_count]["sum_estimator_std"] += float(
                np.mean(estimates.std(axis=0))
            )

    metrics_by_sample_count = {}
    passed = True
    for sample_count, values_by_metric in results.items():
        abs_bias = values_by_metric["sum_abs_bias"] / n_cases
        l2_bias = values_by_metric["sum_l2_bias"] / n_cases
        estimator_std = values_by_metric["sum_estimator_std"] / n_cases
        metrics_by_sample_count[str(sample_count)] = {
            "mean_abs_bias": round(abs_bias, 6),
            "mean_l2_bias": round(l2_bias, 6),
            "mean_estimator_std": round(estimator_std, 6),
        }
        if abs_bias > 0.08:
            passed = False

    return {
        "mode": "sampled_action_mccfr_estimator_diagnostic",
        "passed": passed,
        "n_cases": int(n_cases),
        "n_repeats": int(n_repeats),
        "n_actions": int(n_actions),
        "samples_per_estimate": samples_per_estimate,
        "uniform_mix": uniform_mix,
        "seed": int(seed),
        "metrics_by_sample_count": metrics_by_sample_count,
        "promotion": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate sampled-action regret estimator bias and variance."
    )
    parser.add_argument("--n-cases", type=int, default=32)
    parser.add_argument("--n-repeats", type=int, default=4000)
    parser.add_argument("--n-actions", type=int, default=9)
    parser.add_argument("--samples-per-estimate", type=_parse_samples, default="1,2,4,8")
    parser.add_argument("--uniform-mix", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = run_diagnostic(
        n_cases=args.n_cases,
        n_repeats=args.n_repeats,
        n_actions=args.n_actions,
        samples_per_estimate=args.samples_per_estimate,
        uniform_mix=args.uniform_mix,
        seed=args.seed,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
