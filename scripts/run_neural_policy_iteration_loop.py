#!/usr/bin/env python3
"""Run repeated neural self-play policy-iteration generations."""

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
    evaluate_neural_policy_iteration_head_to_head,
    run_neural_policy_iteration_pilot,
)
from poker_ai.research.mixed_policy_h2h import (  # noqa: E402
    SUPPORTED_POLICY_KINDS,
    evaluate_mixed_policy_head_to_head,
)


def _parse_mixed_control_spec(spec: str) -> dict[str, str]:
    if ":" not in str(spec):
        raise argparse.ArgumentTypeError(
            "mixed control must use KIND:PATH, e.g. tianshou-rainbow:models/control.pt"
        )
    kind, checkpoint = str(spec).split(":", 1)
    kind = kind.strip()
    checkpoint = checkpoint.strip()
    if kind not in SUPPORTED_POLICY_KINDS:
        raise argparse.ArgumentTypeError(
            f"mixed control kind must be one of: {', '.join(SUPPORTED_POLICY_KINDS)}"
        )
    if not checkpoint:
        raise argparse.ArgumentTypeError("mixed control checkpoint path is empty")
    return {"kind": kind, "checkpoint": checkpoint}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-generations", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="neural_policy_iteration")
    parser.add_argument("--checkpoint-in")
    parser.add_argument(
        "--control-checkpoint",
        action="append",
        default=[],
        help="Same-format NPI checkpoint to evaluate every generation against.",
    )
    parser.add_argument(
        "--mixed-control-checkpoint",
        action="append",
        type=_parse_mixed_control_spec,
        default=[],
        help=(
            "External/native policy control to evaluate every generation against as KIND:PATH. "
            f"Kinds: {', '.join(SUPPORTED_POLICY_KINDS)}."
        ),
    )
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
    parser.add_argument("--cfr-batch-roots", action="store_true")
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
        help="Fail a generation if one target top action exceeds this fraction.",
    )
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--max-steps-per-hand", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--h2h-games", type=int, default=100)
    parser.add_argument(
        "--min-lower95-candidate-payoff",
        type=float,
        default=0.0,
        help="Default 0.0 makes the checkpoint league a positive-lower-bound gate.",
    )
    parser.add_argument(
        "--require-control-gate",
        action="store_true",
        help=(
            "Fail the loop unless at least one fixed or mixed control result is "
            "recorded for every generated checkpoint."
        ),
    )
    parser.add_argument("--output-json")
    return parser


def _generation_config(
    args: argparse.Namespace,
    *,
    generation: int,
    checkpoint_in: str | None,
    checkpoint_out: Path,
) -> NeuralPolicyIterationConfig:
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
        seed=int(args.seed) + int(generation),
        device=args.device,
        initial_checkpoint_path=checkpoint_in,
        checkpoint_path=str(checkpoint_out),
    )


def run_loop(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generations: list[dict] = []
    league_results: list[dict] = []
    control_results: list[dict] = []
    previous_checkpoint = args.checkpoint_in
    control_checkpoints = [str(path) for path in getattr(args, "control_checkpoint", [])]
    mixed_control_checkpoints = list(getattr(args, "mixed_control_checkpoint", []))

    for generation in range(max(int(args.n_generations), 0)):
        checkpoint_out = output_dir / f"{args.prefix}_gen{generation}.pt"
        cfg = _generation_config(
            args,
            generation=generation,
            checkpoint_in=previous_checkpoint,
            checkpoint_out=checkpoint_out,
        )
        metrics = run_neural_policy_iteration_pilot(cfg)
        generations.append(metrics)
        if int(args.h2h_games) > 0:
            for control_checkpoint in control_checkpoints:
                result = evaluate_neural_policy_iteration_head_to_head(
                    str(checkpoint_out),
                    str(control_checkpoint),
                    n_games=int(args.h2h_games),
                    device=args.device,
                    seed=int(args.seed) + 200_000 + generation,
                    min_lower95_candidate_payoff=args.min_lower95_candidate_payoff,
                )
                result["generation"] = int(generation)
                result["control_checkpoint"] = str(control_checkpoint)
                control_results.append(result)
            for control in mixed_control_checkpoints:
                result = evaluate_mixed_policy_head_to_head(
                    candidate_checkpoint=str(checkpoint_out),
                    candidate_kind="npi",
                    baseline_checkpoint=str(control["checkpoint"]),
                    baseline_kind=str(control["kind"]),
                    n_games=int(args.h2h_games),
                    device=args.device,
                    seed=int(args.seed) + 300_000 + generation,
                    min_lower95_candidate_payoff=args.min_lower95_candidate_payoff,
                )
                result["generation"] = int(generation)
                result["control_checkpoint"] = str(control["checkpoint"])
                result["control_kind"] = str(control["kind"])
                control_results.append(result)
        if previous_checkpoint and int(args.h2h_games) > 0:
            league_results.append(
                evaluate_neural_policy_iteration_head_to_head(
                    str(checkpoint_out),
                    str(previous_checkpoint),
                    n_games=int(args.h2h_games),
                    device=args.device,
                    seed=int(args.seed) + 100_000 + generation,
                    min_lower95_candidate_payoff=args.min_lower95_candidate_payoff,
                )
            )
        previous_checkpoint = str(checkpoint_out)

    control_gate_required = bool(getattr(args, "require_control_gate", False))
    required_control_results = int(args.n_generations) if control_gate_required else 0
    actual_control_results = int(len(control_results))
    gate_failures: list[str] = []
    if control_gate_required and actual_control_results < required_control_results:
        gate_failures.append("missing_required_control_results")

    passed = all(bool(gen.get("passed", True)) for gen in generations) and all(
        bool(result.get("passed", True)) for result in league_results
    ) and all(
        bool(result.get("passed", True)) for result in control_results
    ) and not gate_failures
    return {
        "algorithm": "neural_policy_iteration_generation_loop",
        "workflow": "alphazero_style_repeatable_policy_iteration",
        "role": "checkpoint_to_checkpoint_self_play_policy_improvement_loop",
        "uses_slumbot_training_data": False,
        "promotion": False,
        "passed": bool(passed),
        "n_generations": int(args.n_generations),
        "h2h_games": int(args.h2h_games),
        "control_gate_required": control_gate_required,
        "required_control_results": required_control_results,
        "actual_control_results": actual_control_results,
        "gate_failures": gate_failures,
        "initial_checkpoint_path": args.checkpoint_in,
        "control_checkpoints": control_checkpoints,
        "mixed_control_checkpoints": mixed_control_checkpoints,
        "generations": generations,
        "league_results": league_results,
        "control_results": control_results,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metrics = run_loop(args)
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
