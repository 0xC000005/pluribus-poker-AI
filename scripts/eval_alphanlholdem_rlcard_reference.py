#!/usr/bin/env python3
"""Run the isolated AlphaNLHoldem/RLCard reference benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.alphanlholdem_benchmark import (  # noqa: E402
    AlphaNLHoldemTorchPolicy,
    RandomLegalAlphaNLHoldemPolicy,
    RLCardAgentPolicy,
    evaluate_rlcard_reference_h2h,
)
from poker_ai.research.alphanlholdem_reference import inspect_alphanlholdem_reference  # noqa: E402
from poker_ai.research.rlcard_tianshou_ppo import RLCardPPOPolicy  # noqa: E402


def _load_policy(kind: str, *, weights: Path | None, seed: int, device: str):
    if kind == "alphanlholdem":
        if weights is None:
            raise ValueError("alphanlholdem policy requires weights")
        return AlphaNLHoldemTorchPolicy.from_pickle(weights, device=device)
    if kind == "rlcard-nfsp":
        if weights is None:
            raise ValueError("rlcard-nfsp policy requires --candidate-weights or --baseline-weights")
        return RLCardAgentPolicy.from_nfsp_checkpoint(weights, device=device)
    if kind in {"rlcard-ppo", "rlcard-vtrace"}:
        if weights is None:
            raise ValueError(f"{kind} policy requires --candidate-weights or --baseline-weights")
        return RLCardPPOPolicy.from_checkpoint(weights, device=device)
    if kind == "random":
        return RandomLegalAlphaNLHoldemPolicy(seed=seed)
    raise ValueError(f"unknown policy kind: {kind}")


def _reference_integrity_fields(reference_weights: Path) -> dict[str, object]:
    reference_dir = REPO_ROOT / "reference_code" / "AlphaNLHoldem"
    try:
        reference_weights.resolve().relative_to(reference_dir.resolve())
    except ValueError:
        return {
            "reference_integrity_checked": False,
            "reference_checkout_ignored": None,
            "tracked_external_reference_files": [],
            "reference_integrity_passed": False,
        }
    report = inspect_alphanlholdem_reference(
        reference_dir,
        native_num_actions=9,
        repo_root=REPO_ROOT,
    )
    return {
        "reference_integrity_checked": bool(report.get("reference_integrity_checked", False)),
        "reference_checkout_ignored": report.get("reference_checkout_ignored"),
        "tracked_external_reference_files": list(report.get("tracked_external_reference_files", [])),
        "reference_integrity_passed": bool(report.get("reference_integrity_passed", False)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-weights",
        type=Path,
        default=REPO_ROOT / "reference_code" / "AlphaNLHoldem" / "weights" / "c_1048.pkl",
    )
    parser.add_argument(
        "--candidate-weights",
        type=Path,
        help=(
            "Optional candidate checkpoint. Defaults to --reference-weights "
            "for alphanlholdem candidates; required for rlcard-nfsp/rlcard-ppo/rlcard-vtrace."
        ),
    )
    parser.add_argument(
        "--baseline-weights",
        type=Path,
        help=(
            "Optional baseline checkpoint. Defaults to --reference-weights "
            "for alphanlholdem baselines; required for rlcard-nfsp/rlcard-ppo/rlcard-vtrace."
        ),
    )
    policy_choices = ("random", "alphanlholdem", "rlcard-nfsp", "rlcard-ppo", "rlcard-vtrace")
    parser.add_argument("--candidate", choices=policy_choices, default="random")
    parser.add_argument("--baseline", choices=policy_choices, default="alphanlholdem")
    parser.add_argument("--games-per-seat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--min-lower95-candidate-payoff", type=float)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    candidate_weights = args.candidate_weights or (
        args.reference_weights if args.candidate == "alphanlholdem" else None
    )
    baseline_weights = args.baseline_weights or (
        args.reference_weights if args.baseline == "alphanlholdem" else None
    )

    candidate = _load_policy(
        args.candidate,
        weights=candidate_weights,
        seed=args.seed,
        device=args.device,
    )
    baseline = _load_policy(
        args.baseline,
        weights=baseline_weights,
        seed=args.seed + 100_000,
        device=args.device,
    )
    metrics = evaluate_rlcard_reference_h2h(
        candidate=candidate,
        baseline=baseline,
        games_per_seat=args.games_per_seat,
        seed=args.seed,
        max_steps=args.max_steps,
        min_lower95_candidate_payoff=args.min_lower95_candidate_payoff,
    )
    metrics.update(
        {
            "reference_weights": str(args.reference_weights),
            "candidate_weights": None if candidate_weights is None else str(candidate_weights),
            "baseline_weights": None if baseline_weights is None else str(baseline_weights),
            "candidate_kind": args.candidate,
            "baseline_kind": args.baseline,
            "device": args.device,
            "trained_environment_native": args.candidate in {
                "alphanlholdem",
                "rlcard-nfsp",
                "rlcard-ppo",
                "rlcard-vtrace",
            },
            "native_action_projection": False,
            "promotion": False,
            "native_slumbot_evidence": False,
            "uses_slumbot_data": False,
            "uses_alphanlholdem_training_data": False,
            **_reference_integrity_fields(args.reference_weights),
        }
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
