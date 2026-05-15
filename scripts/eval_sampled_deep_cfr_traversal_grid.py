#!/usr/bin/env python3
"""Run a grid of research-only sampled Deep CFR traversal probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.sampled_deep_cfr_traversal_probe import run_probe_grid  # noqa: E402


def _case_label(case: dict) -> str:
    return f"seed={case.get('seed')} initial_chips={case.get('initial_chips')}"


def evaluate_gate_failures(
    metrics: dict,
    *,
    min_top_match_rate: float | None = None,
    min_mean_speedup: float | None = None,
    max_mean_abs_bias: float | None = None,
    require_all_top_match: bool = False,
    max_case_mean_abs_bias: float | None = None,
) -> list[str]:
    failures: list[str] = []
    if (
        min_top_match_rate is not None
        and metrics["top_action_match_rate"] < min_top_match_rate
    ):
        failures.append(
            f"top_action_match_rate {metrics['top_action_match_rate']:.6f} "
            f"< {min_top_match_rate:.6f}"
        )
    if min_mean_speedup is not None and metrics["mean_speedup"] < min_mean_speedup:
        failures.append(
            f"mean_speedup {metrics['mean_speedup']:.6f} < {min_mean_speedup:.6f}"
        )
    if max_mean_abs_bias is not None and metrics["mean_abs_bias"] > max_mean_abs_bias:
        failures.append(
            f"mean_abs_bias {metrics['mean_abs_bias']:.6f} > {max_mean_abs_bias:.6f}"
        )
    if require_all_top_match:
        mismatches = [case for case in metrics["cases"] if not bool(case["top_action_match"])]
        if mismatches:
            failures.append(
                "top_action_match failed for "
                + ", ".join(_case_label(case) for case in mismatches)
            )
    if max_case_mean_abs_bias is not None:
        biased_cases = [
            case
            for case in metrics["cases"]
            if float(case["mean_abs_bias"]) > max_case_mean_abs_bias
        ]
        if biased_cases:
            failures.append(
                f"case mean_abs_bias exceeded {max_case_mean_abs_bias:.6f} for "
                + ", ".join(
                    f"{_case_label(case)} mean_abs_bias={float(case['mean_abs_bias']):.6f}"
                    for case in biased_cases
                )
            )
    return failures


def _parse_ints(value: str) -> list[int]:
    parsed = [int(part) for part in value.split(",") if part.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate tiny sampled-vs-exhaustive traversal probes."
    )
    parser.add_argument("--seeds", type=_parse_ints, default="20260525,20260526")
    parser.add_argument("--initial-chips-values", type=_parse_ints, default="300")
    parser.add_argument("--n-repeats", type=int, default=64)
    parser.add_argument("--n-reference-repeats", type=int)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument(
        "--sampling-mode",
        choices=("with-replacement", "without-replacement", "priority-without-replacement"),
        default="without-replacement",
    )
    parser.add_argument("--priority-forced-count", type=int, default=0)
    parser.add_argument(
        "--priority-source",
        choices=(
            "strategy",
            "advantage",
            "abs-advantage",
            "priority-model",
            "oracle-action-value",
        ),
        default="strategy",
    )
    parser.add_argument("--priority-checkpoint")
    parser.add_argument("--use-priority-baseline", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--uniform-mix", type=float, default=0.25)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--min-top-match-rate", type=float)
    parser.add_argument("--min-mean-speedup", type=float)
    parser.add_argument("--max-mean-abs-bias", type=float)
    parser.add_argument("--require-all-top-match", action="store_true")
    parser.add_argument("--max-case-mean-abs-bias", type=float)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)

    metrics = run_probe_grid(
        seeds=args.seeds,
        initial_chips_values=args.initial_chips_values,
        n_repeats=args.n_repeats,
        n_reference_repeats=args.n_reference_repeats,
        sample_count=args.sample_count,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        uniform_mix=args.uniform_mix,
        sampling_mode=args.sampling_mode,
        priority_forced_count=args.priority_forced_count,
        priority_source=args.priority_source,
        priority_checkpoint=args.priority_checkpoint,
        use_priority_baseline=args.use_priority_baseline,
        device=args.device,
    )
    failures = evaluate_gate_failures(
        metrics,
        min_top_match_rate=args.min_top_match_rate,
        min_mean_speedup=args.min_mean_speedup,
        max_mean_abs_bias=args.max_mean_abs_bias,
        require_all_top_match=args.require_all_top_match,
        max_case_mean_abs_bias=args.max_case_mean_abs_bias,
    )
    metrics["gate_thresholds"] = {
        "min_top_match_rate": args.min_top_match_rate,
        "min_mean_speedup": args.min_mean_speedup,
        "max_mean_abs_bias": args.max_mean_abs_bias,
        "require_all_top_match": args.require_all_top_match,
        "max_case_mean_abs_bias": args.max_case_mean_abs_bias,
    }
    metrics["gate_failures"] = failures
    metrics["passed"] = not failures
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
