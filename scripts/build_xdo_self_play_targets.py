#!/usr/bin/env python3
"""Build all-street XDO-lite targets from local parent self-play."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.mixed_policy_h2h import SUPPORTED_POLICY_KINDS, load_policy_adapter  # noqa: E402
from poker_ai.research.native_nfsp import resolve_device  # noqa: E402
from poker_ai.research.xdo_self_play_targets import collect_xdo_self_play_targets  # noqa: E402


def _parse_streets(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--policy-kind", choices=SUPPORTED_POLICY_KINDS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--n-states", type=int, default=64)
    parser.add_argument("--n-worlds", type=int, default=16)
    parser.add_argument("--target-streets", default="0,1,2,3")
    parser.add_argument("--required-streets", default="0,1,2,3")
    parser.add_argument("--min-rows-per-required-street", type=int, default=1)
    parser.add_argument("--max-hands", type=int, default=256)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--small-blind", type=int, default=50)
    parser.add_argument("--big-blind", type=int, default=100)
    parser.add_argument("--max-steps-per-hand", type=int, default=128)
    parser.add_argument("--collection-exploration-epsilon", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)

    device_info = resolve_device(args.device)
    resolved_device = torch.device(device_info["resolved_device"])
    adapter = load_policy_adapter(
        args.policy_checkpoint,
        kind=args.policy_kind,
        device=resolved_device,
    )
    metrics = collect_xdo_self_play_targets(
        adapter,
        args.output,
        output_json=args.output_json,
        n_states=args.n_states,
        n_worlds=args.n_worlds,
        target_streets=_parse_streets(args.target_streets),
        required_streets=_parse_streets(args.required_streets),
        min_rows_per_required_street=args.min_rows_per_required_street,
        max_hands=args.max_hands,
        initial_chips=args.initial_chips,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        max_steps_per_hand=args.max_steps_per_hand,
        collection_exploration_epsilon=args.collection_exploration_epsilon,
        device=resolved_device,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.require_pass and not bool(metrics.get("passed", False)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
