#!/usr/bin/env python3
"""Run the neural self-play policy-iteration contract smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.neural_policy_iteration import (  # noqa: E402
    NeuralPolicyIterationConfig,
    run_neural_policy_iteration_pilot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-play-hands", type=int, default=16)
    parser.add_argument("--max-self-play-hands", type=int)
    parser.add_argument("--min-searchable-self-play-states", type=int, default=0)
    parser.add_argument("--max-improvement-targets", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--parallel-self-play-hands", type=int, default=1)
    parser.add_argument(
        "--self-play-state-backend",
        choices=("full_deck", "fast"),
        default="full_deck",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--policy-sample-weighting",
        choices=("uniform", "inverse_target_top"),
        default="uniform",
    )
    parser.add_argument(
        "--teacher-mode",
        choices=(
            "legal_mixed",
            "public_belief_cfr",
            "public_world_rollout",
            "policy_public_world_rollout",
            "policy_public_state_rollout",
            "public_world_value",
            "public_world_value_prior",
            "sampled_state_value",
            "sampled_state_value_prior",
        ),
        default="legal_mixed",
    )
    parser.add_argument("--preferred-action", type=int, default=1)
    parser.add_argument("--preferred-action-prob", type=float, default=0.7)
    parser.add_argument("--cfr-iterations", type=int, default=5)
    parser.add_argument("--cfr-backend", default="cpu")
    parser.add_argument("--cfr-device")
    parser.add_argument(
        "--cfr-batch-roots",
        action="store_true",
        help="Opt into same-topology batched torch-levelsync CFR teacher solves.",
    )
    parser.add_argument("--cfr-batch-min-roots", type=int, default=2)
    parser.add_argument("--rollout-worlds", type=int, default=8)
    parser.add_argument("--rollout-temperature", type=float, default=100.0)
    parser.add_argument(
        "--rollout-continuation-policy",
        choices=("call", "uniform"),
        default="call",
    )
    parser.add_argument("--min-resolver-targets", type=int, default=0)
    parser.add_argument("--min-target-streets", type=int, default=0)
    parser.add_argument(
        "--max-target-top-action-fraction",
        type=float,
        default=1.0,
        help="Fail if one target top action exceeds this fraction; 1.0 disables the gate.",
    )
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--max-steps-per-hand", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--checkpoint-in",
        help="Initialize the policy/value actor from a previous neural policy-iteration checkpoint.",
    )
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--output-json")
    return parser


def build_config(args: argparse.Namespace) -> NeuralPolicyIterationConfig:
    return NeuralPolicyIterationConfig(
        self_play_hands=args.self_play_hands,
        max_self_play_hands=args.max_self_play_hands,
        min_searchable_self_play_states=args.min_searchable_self_play_states,
        max_improvement_targets=args.max_improvement_targets,
        train_steps=args.train_steps,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        parallel_self_play_hands=args.parallel_self_play_hands,
        self_play_state_backend=args.self_play_state_backend,
        lr=args.lr,
        value_loss_weight=args.value_loss_weight,
        policy_sample_weighting=args.policy_sample_weighting,
        teacher_mode=args.teacher_mode,
        preferred_action=args.preferred_action,
        preferred_action_prob=args.preferred_action_prob,
        cfr_iterations=args.cfr_iterations,
        cfr_backend=args.cfr_backend,
        cfr_device=args.cfr_device,
        cfr_batch_roots=args.cfr_batch_roots,
        cfr_batch_min_roots=args.cfr_batch_min_roots,
        rollout_worlds=args.rollout_worlds,
        rollout_temperature=args.rollout_temperature,
        rollout_continuation_policy=args.rollout_continuation_policy,
        min_resolver_targets=args.min_resolver_targets,
        min_target_streets=args.min_target_streets,
        max_target_top_action_fraction=args.max_target_top_action_fraction,
        initial_chips=args.initial_chips,
        max_steps_per_hand=args.max_steps_per_hand,
        seed=args.seed,
        device=args.device,
        initial_checkpoint_path=args.checkpoint_in,
        checkpoint_path=args.checkpoint_out,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = build_config(args)
    metrics = run_neural_policy_iteration_pilot(cfg)
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
