#!/usr/bin/env python3
"""Evaluate sampled-action regret estimates on deterministic full-deck roots."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.deep_cfr.networks import ValueNetwork  # noqa: E402
from poker_ai.games.full_deck.state import ACTION_TO_INDEX, N_ACTIONS, N_FEATURES, new_game  # noqa: E402
from poker_ai.research.restricted_action_value import (  # noqa: E402
    _set_private_cards_for_control,
    sample_seeded_hole_cards,
    score_legal_actions_by_showdown_equity,
)
from poker_ai.research.sampled_action_mccfr import (  # noqa: E402
    full_regret,
    pps_without_replacement_inclusion_probs,
    sample_pps_without_replacement,
    sampled_action_regret_estimate,
    sampled_action_regret_estimate_without_replacement,
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


def _baseline_for_root(
    rng: np.random.Generator,
    action_values: np.ndarray,
    legal_mask: np.ndarray,
    mode: str,
    noise_scale: float,
) -> np.ndarray | None:
    legal = legal_mask > 0.0
    if mode == "zero":
        return None
    if mode == "legal-mean":
        baseline = np.zeros_like(action_values, dtype=np.float32)
        baseline[legal] = float(np.mean(action_values[legal]))
        return baseline
    if mode == "oracle":
        return action_values.astype(np.float32, copy=True)
    if mode == "noisy-oracle":
        baseline = action_values.astype(np.float32, copy=True)
        scale = float(np.std(action_values[legal])) * max(0.0, float(noise_scale))
        baseline[legal] += rng.normal(0.0, scale, size=int(legal.sum())).astype(np.float32)
        return baseline
    if mode == "checkpoint":
        return None
    raise ValueError(f"unknown baseline mode: {mode}")


def _load_baseline_checkpoint(path: str | None, device: torch.device) -> tuple[ValueNetwork | None, float]:
    if not path:
        return None, 1.0
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    hidden_dim = int(checkpoint.get("hidden_dim", 128))
    n_layers = int(checkpoint.get("n_layers", 2))
    model = ValueNetwork(
        N_FEATURES,
        hidden_dim=hidden_dim,
        output_dim=N_ACTIONS,
        n_layers=n_layers,
        use_betting_history=bool(checkpoint.get("uses_betting_history", True)),
    ).to(device)
    model.load_state_dict(checkpoint["value_net"])
    model.eval()
    return model, float(checkpoint.get("initial_chips", 1) or 1)


def _threshold_failures(
    metrics_by_sample_count: dict[str, dict[str, float]],
    *,
    max_mean_abs_bias: float | None,
    min_mean_estimate_top_action_match: float | None,
) -> list[str]:
    failures: list[str] = []
    for sample_count in sorted(metrics_by_sample_count, key=lambda item: int(item)):
        metrics = metrics_by_sample_count[sample_count]
        if max_mean_abs_bias is not None:
            mean_abs_bias = float(metrics["mean_abs_bias"])
            if mean_abs_bias > float(max_mean_abs_bias):
                failures.append(
                    f"sample_count={sample_count} mean_abs_bias "
                    f"{mean_abs_bias:.6f} > {float(max_mean_abs_bias):.6f}"
                )
        if min_mean_estimate_top_action_match is not None:
            top_match = float(metrics["mean_estimate_top_action_match_rate"])
            if top_match < float(min_mean_estimate_top_action_match):
                failures.append(
                    f"sample_count={sample_count} mean_estimate_top_action_match_rate "
                    f"{top_match:.6f} < {float(min_mean_estimate_top_action_match):.6f}"
                )
    return failures


def run_diagnostic(
    *,
    n_roots: int,
    n_repeats: int,
    n_equity_samples: int,
    initial_chips: int,
    samples_per_estimate: list[int],
    strategy_mode: str,
    uniform_mix: float,
    baseline_mode: str,
    baseline_noise_scale: float,
    baseline_checkpoint: str | None,
    max_mean_abs_bias: float | None,
    min_mean_estimate_top_action_match: float | None,
    sampling_mode: str,
    seed: int,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    uniform_mix = float(np.clip(uniform_mix, 0.0, 1.0))
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model, baseline_scale = _load_baseline_checkpoint(
        baseline_checkpoint,
        torch_device,
    )
    results = {
        sample_count: {
            "sum_abs_bias": 0.0,
            "sum_l2_bias": 0.0,
            "sum_estimator_std": 0.0,
            "sum_top_action_match_rate": 0.0,
            "sum_mean_estimate_top_action_match": 0.0,
        }
        for sample_count in samples_per_estimate
    }

    legal_action_counts: list[int] = []
    baseline_abs_errors: list[float] = []
    baseline_corrs: list[float] = []
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
        baseline_values = _baseline_for_root(
            rng,
            action_values,
            legal_mask,
            baseline_mode,
            baseline_noise_scale,
        )
        if baseline_model is not None:
            with torch.no_grad():
                features = torch.from_numpy(state.to_feature_vector()).to(
                    torch_device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                baseline_values = (
                    baseline_model(features)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    * float(baseline_scale)
                )
            baseline_values = np.where(legal_mask > 0.0, baseline_values, 0.0)
        if baseline_values is not None:
            legal_indices_for_baseline = np.flatnonzero(legal_mask > 0.0)
            baseline_abs_errors.append(
                float(
                    np.mean(
                        np.abs(
                            baseline_values[legal_indices_for_baseline]
                            - action_values[legal_indices_for_baseline]
                        )
                    )
                )
            )
            if legal_indices_for_baseline.size >= 2:
                baseline_slice = baseline_values[legal_indices_for_baseline]
                action_slice = action_values[legal_indices_for_baseline]
                if float(np.std(baseline_slice)) > 0.0 and float(np.std(action_slice)) > 0.0:
                    corr = np.corrcoef(baseline_slice, action_slice)[0, 1]
                    if np.isfinite(corr):
                        baseline_corrs.append(float(corr))

        legal_indices = np.flatnonzero(legal_mask > 0.0)
        for sample_count in samples_per_estimate:
            estimates = np.empty((n_repeats, N_ACTIONS), dtype=np.float32)
            inclusion_probs = None
            if sampling_mode == "without-replacement":
                inclusion_probs = pps_without_replacement_inclusion_probs(
                    sample_probs,
                    legal_mask,
                    sample_count=sample_count,
                )
            for repeat_idx in range(n_repeats):
                if sampling_mode == "with-replacement":
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
                        baseline_values=baseline_values,
                    )
                elif sampling_mode == "without-replacement":
                    actions = sample_pps_without_replacement(
                        rng,
                        sample_probs,
                        legal_mask,
                        sample_count=sample_count,
                    )
                    estimates[repeat_idx] = sampled_action_regret_estimate_without_replacement(
                        action_values,
                        strategy,
                        legal_mask,
                        sampled_actions=actions,
                        inclusion_probs=inclusion_probs,
                        baseline_values=baseline_values,
                    )
                else:
                    raise ValueError(f"unknown sampling mode: {sampling_mode}")
            mean_estimate = estimates.mean(axis=0).astype(np.float64)
            legal_bias = mean_estimate[legal_indices] - target[legal_indices]
            legal_estimates = estimates[:, legal_indices]
            target_top_local = int(np.argmax(target[legal_indices]))
            mean_estimate_top_local = int(np.argmax(mean_estimate[legal_indices]))
            estimate_top_local = np.argmax(legal_estimates, axis=1)
            results[sample_count]["sum_abs_bias"] += float(np.mean(np.abs(legal_bias)))
            results[sample_count]["sum_l2_bias"] += float(
                np.sqrt(np.mean(legal_bias ** 2))
            )
            results[sample_count]["sum_estimator_std"] += float(
                np.mean(legal_estimates.std(axis=0))
            )
            results[sample_count]["sum_top_action_match_rate"] += float(
                np.mean(estimate_top_local == target_top_local)
            )
            results[sample_count]["sum_mean_estimate_top_action_match"] += float(
                mean_estimate_top_local == target_top_local
            )

    metrics_by_sample_count = {}
    for sample_count, values_by_metric in results.items():
        abs_bias = values_by_metric["sum_abs_bias"] / n_roots
        l2_bias = values_by_metric["sum_l2_bias"] / n_roots
        estimator_std = values_by_metric["sum_estimator_std"] / n_roots
        metrics_by_sample_count[str(sample_count)] = {
            "mean_abs_bias": round(abs_bias, 6),
            "mean_l2_bias": round(l2_bias, 6),
            "mean_estimator_std": round(estimator_std, 6),
            "mean_top_action_match_rate": round(
                values_by_metric["sum_top_action_match_rate"] / n_roots,
                6,
            ),
            "mean_estimate_top_action_match_rate": round(
                values_by_metric["sum_mean_estimate_top_action_match"] / n_roots,
                6,
            ),
        }
    gate_failures = _threshold_failures(
        metrics_by_sample_count,
        max_mean_abs_bias=max_mean_abs_bias,
        min_mean_estimate_top_action_match=min_mean_estimate_top_action_match,
    )

    return {
        "mode": "sampled_action_full_deck_estimator_diagnostic",
        "passed": not gate_failures,
        "n_roots": int(n_roots),
        "n_repeats": int(n_repeats),
        "n_equity_samples": int(n_equity_samples),
        "initial_chips": int(initial_chips),
        "samples_per_estimate": samples_per_estimate,
        "strategy_mode": strategy_mode,
        "sampling_mode": sampling_mode,
        "uniform_mix": uniform_mix,
        "baseline_mode": baseline_mode,
        "baseline_noise_scale": float(baseline_noise_scale),
        "baseline_checkpoint": str(baseline_checkpoint or ""),
        "seed": int(seed),
        "mean_legal_actions": round(float(np.mean(legal_action_counts)), 6),
        "baseline_mean_abs_error": (
            round(float(np.mean(baseline_abs_errors)), 6)
            if baseline_abs_errors else None
        ),
        "baseline_mean_action_corr": (
            round(float(np.mean(baseline_corrs)), 6)
            if baseline_corrs else None
        ),
        "gate_thresholds": {
            "max_mean_abs_bias": max_mean_abs_bias,
            "min_mean_estimate_top_action_match": min_mean_estimate_top_action_match,
        },
        "gate_failures": gate_failures,
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
    parser.add_argument(
        "--sampling-mode",
        choices=("with-replacement", "without-replacement"),
        default="with-replacement",
    )
    parser.add_argument("--uniform-mix", type=float, default=0.25)
    parser.add_argument(
        "--baseline-mode",
        choices=("zero", "legal-mean", "oracle", "noisy-oracle"),
        default="zero",
    )
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--baseline-noise-scale", type=float, default=0.25)
    parser.add_argument("--max-mean-abs-bias", type=float, default=3.0)
    parser.add_argument("--min-mean-estimate-top-match", type=float)
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
        baseline_mode="checkpoint" if args.baseline_checkpoint else args.baseline_mode,
        baseline_noise_scale=args.baseline_noise_scale,
        baseline_checkpoint=args.baseline_checkpoint,
        max_mean_abs_bias=args.max_mean_abs_bias,
        min_mean_estimate_top_action_match=args.min_mean_estimate_top_match,
        sampling_mode=args.sampling_mode,
        seed=args.seed,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
