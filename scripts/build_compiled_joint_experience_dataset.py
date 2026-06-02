#!/usr/bin/env python3
"""Build a compiled local joint-experience dataset under a PSRO meta-policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.compiled_joint_experience import (  # noqa: E402
    collect_compiled_joint_experience,
    load_compiled_joint_policy,
)
from poker_ai.research.joint_experience import save_joint_experience_npz  # noqa: E402
from poker_ai.research.native_nfsp import resolve_device  # noqa: E402


def _parse_policy(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError("policy must be KIND:CHECKPOINT")
    kind, checkpoint = spec.split(":", 1)
    kind = kind.strip()
    checkpoint = checkpoint.strip()
    if not kind or not checkpoint:
        raise argparse.ArgumentTypeError("policy must be KIND:CHECKPOINT")
    return kind, checkpoint


def _parse_meta_strategy(text: str | None, n_policies: int) -> list[float]:
    if text is None:
        return [1.0 / float(n_policies)] * int(n_policies)
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != int(n_policies):
        raise argparse.ArgumentTypeError("meta-strategy length must match --policy count")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        help="Policy in KIND:CHECKPOINT format. May be supplied multiple times.",
    )
    parser.add_argument(
        "--meta-strategy",
        help="Comma-separated probabilities matching --policy order. Defaults to uniform.",
    )
    parser.add_argument("--n-hands", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--max-steps-per-hand", type=int, default=256)
    parser.add_argument(
        "--exploration-epsilon",
        type=float,
        default=0.0,
        help="Probability of replacing the selected policy action with a random legal action.",
    )
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path)
    args = parser.parse_args(argv)

    policy_specs = [_parse_policy(spec) for spec in args.policy]
    meta_strategy = _parse_meta_strategy(args.meta_strategy, len(policy_specs))
    device_info = resolve_device(args.device)
    resolved_device = torch.device(device_info["resolved_device"])
    policies = [
        load_compiled_joint_policy(checkpoint, kind=kind, device=resolved_device)
        for kind, checkpoint in policy_specs
    ]
    dataset = collect_compiled_joint_experience(
        policies,
        meta_strategy=meta_strategy,
        n_hands=args.n_hands,
        batch_size=args.batch_size,
        seed=args.seed,
        initial_chips=args.initial_chips,
        max_steps_per_hand=args.max_steps_per_hand,
        exploration_epsilon=args.exploration_epsilon,
        device=resolved_device,
    )
    summary = save_joint_experience_npz(
        dataset,
        args.output_npz,
        manifest_path=args.manifest_json,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
