#!/usr/bin/env python3
"""Emit a JSON plan for game-theoretic RL baseline experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.game_theoretic_rl import validate_baseline_plan  # noqa: E402


def build_plan_payload(
    algorithm: str,
    *,
    allow_control_baseline: bool = False,
) -> dict:
    plan = validate_baseline_plan(
        algorithm,
        allow_control_baseline=allow_control_baseline,
    )
    profile = plan.profile
    return {
        "algorithm": profile.name,
        "allowed_role": plan.allowed_role,
        "primary_role": profile.primary_role,
        "equilibrium_mechanism": profile.equilibrium_mechanism,
        "search_dependency": profile.search_dependency,
        "candidate_frameworks": list(profile.candidate_frameworks),
        "requires_equilibrium_layer": plan.requires_equilibrium_layer,
        "warning": profile.warning,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Describe how an RL algorithm can be used in poker autoresearch."
    )
    parser.add_argument("--algorithm", required=True)
    parser.add_argument(
        "--allow-control-baseline",
        action="store_true",
        help="Permit generic RL algorithms such as PPO/Rainbow as controls.",
    )
    args = parser.parse_args(argv)

    payload = build_plan_payload(
        args.algorithm,
        allow_control_baseline=args.allow_control_baseline,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

