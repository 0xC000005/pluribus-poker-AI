#!/usr/bin/env python3
"""Evaluate a solved empirical-game meta-strategy against one native baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.mixed_policy_h2h import (  # noqa: E402
    SUPPORTED_POLICY_KINDS,
    evaluate_meta_strategy_head_to_head,
)


def _member_kind_overrides(values: Sequence[str]) -> dict[int, str]:
    overrides: dict[int, str] = {}
    for value in values:
        try:
            raw_index, kind = value.split(":", 1)
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError("--member-kind must use INDEX:KIND format") from exc
        if kind not in SUPPORTED_POLICY_KINDS:
            raise ValueError(f"unsupported policy kind for member {index}: {kind}")
        overrides[index] = kind
    return overrides


def _load_nonzero_support(
    empirical_game_json: Path,
    *,
    strategy_key: str,
    default_policy_kind: str,
    member_kind_overrides: dict[int, str],
    min_weight: float,
) -> tuple[list[str], list[str], list[float]]:
    payload = json.loads(empirical_game_json.read_text(encoding="utf-8"))
    policies = [str(path) for path in payload.get("policies", [])]
    meta_strategy = payload.get("meta_strategy", {})
    if meta_strategy.get("solved") is not True:
        raise ValueError("empirical game meta_strategy must be solved before H2H evaluation")
    weights = meta_strategy.get(strategy_key)
    if not isinstance(weights, list):
        raise ValueError(f"meta_strategy missing list field: {strategy_key}")
    if len(weights) != len(policies):
        raise ValueError("meta_strategy weights must match empirical-game policy count")

    checkpoints: list[str] = []
    kinds: list[str] = []
    support_weights: list[float] = []
    for index, (checkpoint, weight) in enumerate(zip(policies, weights, strict=True)):
        weight_value = float(weight)
        if weight_value <= float(min_weight):
            continue
        checkpoints.append(checkpoint)
        kinds.append(member_kind_overrides.get(index, default_policy_kind))
        support_weights.append(weight_value)
    if not checkpoints:
        raise ValueError("empirical meta-strategy has no support above min_weight")
    return checkpoints, kinds, support_weights


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empirical-game-json", type=Path, required=True)
    parser.add_argument(
        "--strategy-key",
        choices=("row_strategy", "column_strategy"),
        default="row_strategy",
    )
    parser.add_argument(
        "--default-policy-kind",
        choices=SUPPORTED_POLICY_KINDS,
        required=True,
    )
    parser.add_argument(
        "--member-kind",
        action="append",
        default=[],
        help="Override one empirical-game member kind with INDEX:KIND.",
    )
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--baseline-kind", choices=SUPPORTED_POLICY_KINDS, required=True)
    parser.add_argument("--n-games", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260849)
    parser.add_argument("--initial-chips", type=int)
    parser.add_argument("--max-steps-per-hand", type=int)
    parser.add_argument("--min-lower95-candidate-payoff", type=float)
    parser.add_argument(
        "--eval-state-backend",
        choices=("full-deck", "fast-state-canonical-deal"),
        default="fast-state-canonical-deal",
    )
    parser.add_argument("--min-weight", type=float, default=1e-12)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    checkpoints, kinds, weights = _load_nonzero_support(
        args.empirical_game_json,
        strategy_key=args.strategy_key,
        default_policy_kind=args.default_policy_kind,
        member_kind_overrides=_member_kind_overrides(args.member_kind),
        min_weight=args.min_weight,
    )
    metrics = evaluate_meta_strategy_head_to_head(
        candidate_checkpoints=checkpoints,
        candidate_kinds=kinds,
        candidate_weights=weights,
        baseline_checkpoint=args.baseline_checkpoint,
        baseline_kind=args.baseline_kind,
        n_games=args.n_games,
        device=args.device,
        seed=args.seed,
        initial_chips=args.initial_chips,
        max_steps_per_hand=args.max_steps_per_hand,
        min_lower95_candidate_payoff=args.min_lower95_candidate_payoff,
        eval_state_backend=args.eval_state_backend,
        meta_strategy_source=str(args.empirical_game_json),
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
