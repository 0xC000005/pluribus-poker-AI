#!/usr/bin/env python3
"""Run the native full-deck policy-only PPO pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.native_ppo_policy import (  # noqa: E402
    NativePPOConfig,
    evaluate_native_ppo_policy_head_to_head,
    run_native_ppo_policy_pilot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rollout-episodes-per-update", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--entropy-anneal-to", type=float, default=None,
                        help="If set, linearly anneal entropy weight from --entropy-weight to this value over training (one principled schedule, not a sweep).")
    parser.add_argument(
        "--advantage-mode",
        choices=("value", "q_expected_mc", "q_expected_lambda", "q_expected_lambda_target"),
        default="value",
    )
    parser.add_argument(
        "--ppo-loss-mode",
        choices=("standard", "trinal_clip"),
        default="standard",
        help="PPO surrogate/loss variant. trinal_clip applies dual clipping for negative advantages and smooth critic loss.",
    )
    parser.add_argument(
        "--actor-update-mode",
        choices=("ppo", "neurd"),
        default="ppo",
        help="Actor update rule. neurd applies a direct legal-logit advantage update instead of the PPO probability-ratio surrogate.",
    )
    parser.add_argument(
        "--behavior-mode",
        choices=("policy", "q_boosted"),
        default="policy",
        help="Training behavior policy. q_boosted samples from the actor prior boosted by the learned Q critic.",
    )
    parser.add_argument("--q-boost-beta", type=float, default=1.0)
    parser.add_argument("--q-boost-min-prior", type=float, default=1e-6)
    parser.add_argument(
        "--centralized-q-critic",
        action="store_true",
        help="Use training-only full-state critic features for Q-expected PPO modes; actor remains observation-only.",
    )
    parser.add_argument("--fsp-average-policy", action="store_true")
    parser.add_argument("--average-policy-batch-size", type=int, default=512)
    parser.add_argument("--average-policy-memory-capacity", type=int, default=20_000)
    parser.add_argument("--trace-lambda", type=float, default=0.9)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument(
        "--feature-mode",
        choices=("flat", "raw_sequence"),
        default="flat",
        help="Actor observation surface. raw_sequence replaces aggregate action-count history with ordered action tokens and amounts.",
    )
    parser.add_argument(
        "--historical-opponent-interval",
        type=int,
        default=0,
        help="If >0, snapshot the actor every N episodes and train learner seats against sampled frozen historical opponents.",
    )
    parser.add_argument("--historical-opponent-capacity", type=int, default=8)
    parser.add_argument(
        "--historical-opponent-selection",
        choices=("fifo", "k_best"),
        default="fifo",
        help="How to trim the historical opponent pool when it exceeds capacity.",
    )
    parser.add_argument(
        "--external-opponent-checkpoint",
        help="Frozen local checkpoint controlling the non-learner seat during training.",
    )
    parser.add_argument(
        "--external-opponent-kind",
        choices=("auto", "native-ppo", "tianshou-rainbow"),
        default="auto",
        help="Kind of --external-opponent-checkpoint. Auto detects local native/Rainbow checkpoints.",
    )
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--max-steps-per-hand", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-json")
    parser.add_argument("--checkpoint-in")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--checkpoint-out")
    parser.add_argument(
        "--min-lower95-candidate-payoff",
        type=float,
        help="If set for H2H mode, exit nonzero unless lower95_candidate_payoff meets this threshold.",
    )
    parser.add_argument(
        "--strategy-source",
        choices=("auto", "actor", "average"),
        default="auto",
        help="Policy network to export for H2H. Auto uses average policy for FSP checkpoints.",
    )
    return parser


def build_config(argv: list[str] | None = None) -> NativePPOConfig:
    args = _parser().parse_args(argv)
    return NativePPOConfig(
        train_episodes=args.train_episodes,
        eval_games=args.eval_games,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        rollout_episodes_per_update=args.rollout_episodes_per_update,
        ppo_epochs=args.ppo_epochs,
        lr=args.lr,
        clip_epsilon=args.clip_epsilon,
        value_loss_weight=args.value_loss_weight,
        entropy_weight=args.entropy_weight,
        entropy_anneal_to=args.entropy_anneal_to,
        advantage_mode=args.advantage_mode,
        ppo_loss_mode=args.ppo_loss_mode,
        actor_update_mode=args.actor_update_mode,
        behavior_mode=args.behavior_mode,
        q_boost_beta=args.q_boost_beta,
        q_boost_min_prior=args.q_boost_min_prior,
        centralized_q_critic=args.centralized_q_critic,
        fsp_average_policy=args.fsp_average_policy,
        average_policy_batch_size=args.average_policy_batch_size,
        average_policy_memory_capacity=args.average_policy_memory_capacity,
        trace_lambda=args.trace_lambda,
        discount=args.discount,
        feature_mode=args.feature_mode,
        historical_opponent_interval=args.historical_opponent_interval,
        historical_opponent_capacity=args.historical_opponent_capacity,
        historical_opponent_selection=args.historical_opponent_selection,
        external_opponent_checkpoint=args.external_opponent_checkpoint,
        external_opponent_kind=args.external_opponent_kind,
        initial_chips=args.initial_chips,
        max_steps_per_hand=args.max_steps_per_hand,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.checkpoint_out,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.checkpoint_in and args.baseline_checkpoint:
        metrics = evaluate_native_ppo_policy_head_to_head(
            args.checkpoint_in,
            args.baseline_checkpoint,
            n_games=args.eval_games,
            device=args.device,
            seed=args.seed,
            min_lower95_candidate_payoff=args.min_lower95_candidate_payoff,
            strategy_source=args.strategy_source,
        )
    else:
        cfg = build_config(argv)
        metrics = run_native_ppo_policy_pilot(cfg)
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if metrics.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
