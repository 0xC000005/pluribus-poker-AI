#!/usr/bin/env python3
"""Evaluate sampled-action regret estimates on deterministic full-deck roots."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.games.full_deck.state import ACTION_TO_INDEX, N_ACTIONS, new_game  # noqa: E402
from poker_ai.research.restricted_action_value import (  # noqa: E402
    _set_private_cards_for_control,
    sample_seeded_hole_cards,
    score_legal_actions_by_showdown_equity,
)
from poker_ai.research.sampled_action_mccfr import (  # noqa: E402
    full_regret,
    sampled_action_regret_estimate,
)


def _parse_samples(value: str) -> list[int]:
    samples = [int(part) for part in value.split(",") if part.strip()]
    if not samples or any(sample <= 0 for sample in samples):
        raise argparse.ArgumentTypeError("samples must be positive integers")
    return samples


def _strategy_for_root(
    rng: np.random.Generator,
    legal_mask: np.ndarray,
    mode: str,
) -> np.ndarray:
    legal = legal_mask > 0.0
    if mode == "uniform":
        strategy = legal.astype(np.float32) / float(legal.sum())
    elif mode == "dirichlet":
        raw = rng.dirichlet(np.ones(int(legal.sum()))).astype(np.float32)
        strategy = np.zeros_like(legal_mask, dtype=np.float32)
        strategy[legal] = raw
    else:
        raise ValueError(f"unknown strategy mode: {mode}")
    return strategy


def run_diagnostic(
    *,
    n_roots: int,
    n_repeats: int,
    n_equity_samples: int,
    initial_chips: int,
    samples_per_estimate: list[int],
    strategy_mode: str,
    uniform_mix: float,
    seed: int,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
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

    legal_action_counts: list[int] = []
    for root_idx in range(max(1, int(n_roots))):
        state = new_game(2, initial_chips=int(initial_chips))
        _set_private_cards_for_control(
            state,
            0,
            list(sample_seeded_hole_cards(seed=int(seed), root_idx=root_idx)),
        )
        scored = score_legal_actions_by_showdown_equity(
            state,
            player_i=0,
            n_equity_samples=int(n_equity_samples),
            seed=int(seed) + 20_000 + root_idx,
        )

        action_values = np.zeros(N_ACTIONS, dtype=np.float32)
        legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)
        for action_name, value in scored["action_values"].items():
            idx = ACTION_TO_INDEX[action_name]
            action_values[idx] = float(value)
            legal_mask[idx] = 1.0
        legal_action_counts.append(int(legal_mask.sum()))

        strategy = _strategy_for_root(rng, legal_mask, strategy_mode)
        uniform = legal_mask / max(1.0, float(legal_mask.sum()))
        sample_probs = (
            (1.0 - uniform_mix) * strategy + uniform_mix * uniform
        ).astype(np.float32)
        sample_probs = sample_probs / float(sample_probs.sum())
        target = full_regret(action_values, strategy, legal_mask).astype(np.float64)

        legal_indices = np.flatnonzero(legal_mask > 0.0)
        for sample_count in samples_per_estimate:
            estimates = np.empty((n_repeats, N_ACTIONS), dtype=np.float32)
            for repeat_idx in range(n_repeats):
                actions = rng.choice(
                    legal_indices,
                    size=sample_count,
                    replace=True,
                    p=sample_probs[legal_indices],
                )
                estimates[repeat_idx] = sampled_action_regret_estimate(
                    action_values,
                    strategy,
                    legal_mask,
                    sampled_actions=actions,
                    sample_probs=sample_probs,
                )
            mean_estimate = estimates.mean(axis=0).astype(np.float64)
            legal_bias = mean_estimate[legal_indices] - target[legal_indices]
            legal_estimates = estimates[:, legal_indices]
            results[sample_count]["sum_abs_bias"] += float(np.mean(np.abs(legal_bias)))
            results[sample_count]["sum_l2_bias"] += float(
                np.sqrt(np.mean(legal_bias ** 2))
            )
            results[sample_count]["sum_estimator_std"] += float(
                np.mean(legal_estimates.std(axis=0))
            )

    metrics_by_sample_count = {}
    passed = True
    for sample_count, values_by_metric in results.items():
        abs_bias = values_by_metric["sum_abs_bias"] / n_roots
        l2_bias = values_by_metric["sum_l2_bias"] / n_roots
        estimator_std = values_by_metric["sum_estimator_std"] / n_roots
        metrics_by_sample_count[str(sample_count)] = {
            "mean_abs_bias": round(abs_bias, 6),
            "mean_l2_bias": round(l2_bias, 6),
            "mean_estimator_std": round(estimator_std, 6),
        }
        if abs_bias > 3.0:
            passed = False

    return {
        "mode": "sampled_action_full_deck_estimator_diagnostic",
        "passed": passed,
        "n_roots": int(n_roots),
        "n_repeats": int(n_repeats),
        "n_equity_samples": int(n_equity_samples),
        "initial_chips": int(initial_chips),
        "samples_per_estimate": samples_per_estimate,
        "strategy_mode": strategy_mode,
        "uniform_mix": uniform_mix,
        "seed": int(seed),
        "mean_legal_actions": round(float(np.mean(legal_action_counts)), 6),
        "metrics_by_sample_count": metrics_by_sample_count,
        "warning": "Restricted showdown action values; estimator math diagnostic only.",
        "promotion": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate sampled-action regret estimates on full-deck roots."
    )
    parser.add_argument("--n-roots", type=int, default=16)
    parser.add_argument("--n-repeats", type=int, default=2000)
    parser.add_argument("--n-equity-samples", type=int, default=256)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--samples-per-estimate", type=_parse_samples, default="1,2,4,8")
    parser.add_argument("--strategy-mode", choices=("uniform", "dirichlet"), default="dirichlet")
    parser.add_argument("--uniform-mix", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = run_diagnostic(
        n_roots=args.n_roots,
        n_repeats=args.n_repeats,
        n_equity_samples=args.n_equity_samples,
        initial_chips=args.initial_chips,
        samples_per_estimate=args.samples_per_estimate,
        strategy_mode=args.strategy_mode,
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
