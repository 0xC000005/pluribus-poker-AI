#!/usr/bin/env python3
"""CLI entrypoint for the poker autoresearch workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.research.autoresearch import (  # noqa: E402
    audit_objective_alignment,
    close_cycle,
    commit_ready_report,
    continuous,
    enqueue_candidate_comparison,
    enqueue_candidate_promotion_gate,
    enqueue_callback_calibration_audit,
    enqueue_cfr_budget_frontier,
    enqueue_cfr_matrix_footprint,
    enqueue_falsification_ladder,
    enqueue_failure_synthesis,
    enqueue_gpu_training,
    enqueue_methodology_review,
    enqueue_paradigm_innovation_review,
    enqueue_resolver_benchmark,
    enqueue_sd_cfr_mixture_falsification,
    enqueue_slumbot_smoke,
    enqueue_cycle,
    enqueue_warm_start_resolver_gate,
    init_state,
    git_changed_paths,
    new_cycle,
    readiness_report,
    research_drift_status,
    register_research_knob,
    run_gate,
    set_incumbent,
    set_research_phase,
    synthesis_status,
    write_review_manifest,
)


def _emit(data: dict) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run safe poker autoresearch workflow gates and cycles."
    )
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT),
        help="Repository root containing autoresearch-session/ state.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create local workflow state.")
    init.add_argument("--force", action="store_true", help="Overwrite existing state files.")

    subparsers.add_parser("status", help="Print workflow readiness as JSON.")
    subparsers.add_parser(
        "drift-status",
        help="Review recent research history for objective drift before the next queued cycle.",
    )

    incumbent = subparsers.add_parser("set-incumbent", help="Record incumbent checkpoint.")
    incumbent.add_argument("--checkpoint", required=True)
    incumbent.add_argument("--reason", required=True)

    phase = subparsers.add_parser("set-phase", help="Set the autoresearch phase guard.")
    phase.add_argument("--phase", required=True)
    phase.add_argument("--reason", required=True)

    subparsers.add_parser("synthesis-status", help="Report whether failure synthesis is due.")

    gate = subparsers.add_parser("gate", help="Run one configured workflow gate.")
    gate.add_argument("name", help="Gate name, for example tier0.")
    gate.add_argument("--run-dir", help="Optional directory for metrics.json.")

    new = subparsers.add_parser("new-cycle", help="Open a new research cycle.")
    new.add_argument("--hypothesis", required=True)
    new.add_argument("--cycle-type", default="experiment")
    new.add_argument("--failure-class", default="eval_invalid")
    new.add_argument("--gate", default="tier0")

    enqueue = subparsers.add_parser("enqueue", help="Add a queued research cycle.")
    enqueue.add_argument("--hypothesis", required=True)
    enqueue.add_argument("--cycle-type", default="experiment")
    enqueue.add_argument("--failure-class", default="eval_invalid")
    enqueue.add_argument("--gate", default="tier0")

    compare = subparsers.add_parser(
        "enqueue-compare",
        help="Create and queue a candidate-vs-incumbent local comparison gate.",
    )
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--baseline")
    compare.add_argument("--n-games", type=int, default=500)
    compare.add_argument("--seeds", default="20260511,20260512,20260513")
    compare.add_argument("--device", default="auto")
    compare.add_argument("--timeout-seconds", type=int, default=2400)
    compare.add_argument(
        "--head-to-head",
        action="store_true",
        help="Queue duplicate-swapped model-vs-model evaluation instead of vs-random deltas.",
    )
    compare.add_argument(
        "--strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        default="regret",
        help="Use advantage regret matching or the trained average-strategy policy head.",
    )
    compare.add_argument(
        "--candidate-strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        help="Override --strategy-source for the candidate only.",
    )
    compare.add_argument(
        "--baseline-strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        help="Override --strategy-source for the baseline only.",
    )

    promotion = subparsers.add_parser(
        "enqueue-promotion-gate",
        help="Create and queue the dual-surface pre-Slumbot candidate promotion gate.",
    )
    promotion.add_argument("--rlcard-reference-json", required=True)
    promotion.add_argument("--native-h2h-json", required=True)
    promotion.add_argument("--empirical-game-json")
    promotion.add_argument("--native-candidate-checkpoint")
    promotion.add_argument("--min-lower95", type=float, default=0.0)
    promotion.add_argument("--min-candidate-support", type=float, default=1.0e-9)
    promotion.add_argument("--timeout-seconds", type=int, default=300)

    slumbot = subparsers.add_parser(
        "enqueue-slumbot",
        help="Create and queue a sparse live Slumbot smoke gate for one model.",
    )
    slumbot.add_argument("--model", required=True)
    slumbot.add_argument(
        "--model-kind",
        choices=("auto", "deep-cfr", "tianshou-rainbow"),
        default="auto",
    )
    slumbot.add_argument("--hands", type=int, default=5)
    slumbot.add_argument("--greedy", action="store_true")
    slumbot.add_argument("--no-allin", action="store_true")
    slumbot.add_argument("--no-solver", action="store_true")
    slumbot.add_argument(
        "--solver-backend",
        choices=(
            "auto",
            "cpu",
            "torch-cuda",
            "torch-cpu",
            "torch-levelsync-cuda",
            "torch-levelsync-cpu",
            "segmented-cuda",
            "segmented-cpu",
        ),
        default="auto",
    )
    slumbot.add_argument(
        "--solver-budget-profile",
        choices=("live", "frontier-live", "fast-live"),
        default="live",
    )
    slumbot.add_argument(
        "--strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        default="regret",
    )
    slumbot.add_argument("--timeout-seconds", type=int, default=600)

    resolver = subparsers.add_parser(
        "enqueue-resolver",
        help="Create and queue a fixed public-state resolver benchmark gate.",
    )
    resolver.add_argument("--model", required=True)
    resolver.add_argument("--solver-iterations", type=int, default=25)
    resolver.add_argument(
        "--solver-backend",
        choices=(
            "auto",
            "cpu",
            "torch-cuda",
            "torch-cpu",
            "torch-levelsync-cuda",
            "torch-levelsync-cpu",
            "segmented-cuda",
            "segmented-cpu",
        ),
        default="auto",
    )
    resolver.add_argument("--max-cases", type=int)
    resolver.add_argument("--device", default="auto")
    resolver.add_argument("--timeout-seconds", type=int, default=1200)

    frontier = subparsers.add_parser(
        "enqueue-cfr-budget-frontier",
        help="Create and queue a root-disjoint exact CFR budget frontier gate.",
    )
    frontier.add_argument("--cases", required=True)
    frontier.add_argument("--cfv-cache", required=True)
    frontier.add_argument("--budgets", required=True)
    frontier.add_argument("--start-index", type=int, default=128)
    frontier.add_argument("--limit", type=int, default=64)
    frontier.add_argument("--reference-iterations", type=int, default=25)
    frontier.add_argument(
        "--solver-backend",
        choices=(
            "cpu",
            "cpu-levelsync",
            "auto",
            "torch-cuda",
            "torch-cpu",
            "torch-levelsync-cuda",
            "torch-levelsync-cpu",
            "segmented-cuda",
            "segmented-cpu",
        ),
        default="torch-levelsync-cuda",
    )
    frontier.add_argument(
        "--solver-update",
        choices=("cfr_plus", "dcfr_plus", "pdcfr_plus"),
        default="cfr_plus",
    )
    frontier.add_argument("--min-evaluated", type=int, default=1)
    frontier.add_argument("--output-json")
    frontier.add_argument("--timeout-seconds", type=int, default=3600)

    footprint = subparsers.add_parser(
        "enqueue-cfr-matrix-footprint",
        help="Create and queue a matrix/fused CFR footprint and chunk-plan gate.",
    )
    footprint.add_argument("--cases", required=True)
    footprint.add_argument("--start-index", type=int, default=128)
    footprint.add_argument("--max-cases", type=int)
    footprint.add_argument("--chunk-memory-cap-mib", type=float)
    footprint.add_argument("--output-json")
    footprint.add_argument("--timeout-seconds", type=int, default=900)

    warm_start = subparsers.add_parser(
        "enqueue-warm-start-resolver",
        help="Create and queue the root-disjoint neural warm-start resolver A/B gate.",
    )
    warm_start.add_argument("--checkpoint", required=True)
    warm_start.add_argument("--cases", required=True)
    warm_start.add_argument("--cfv-cache", required=True)
    warm_start.add_argument("--train-labels-npz")
    warm_start.add_argument("--start-index", type=int, default=128)
    warm_start.add_argument("--limit", type=int, default=64)
    warm_start.add_argument("--low-iterations", type=int, default=5)
    warm_start.add_argument(
        "--baseline-iterations",
        type=int,
        help="Optional uniform CFR+ budget to compare against regret-policy warm starts.",
    )
    warm_start.add_argument("--reference-iterations", type=int, default=25)
    warm_start.add_argument("--solver-backend", choices=("cpu", "auto"), default="cpu")
    warm_start.add_argument("--device", default="auto")
    warm_start.add_argument("--regret-mass-scale", type=float, default=1.0)
    warm_start.add_argument("--strategy-mass", type=float, default=0.0)
    warm_start.add_argument("--min-evaluated", type=int, default=32)
    warm_start.add_argument("--max-warm-latency-ratio", type=float, default=2.0)
    warm_start.add_argument("--output-json")
    warm_start.add_argument("--timeout-seconds", type=int, default=3600)

    falsify = subparsers.add_parser(
        "enqueue-falsification",
        help="Create and queue objective-audit, incumbent-comparison, and resolver counter-tests.",
    )
    falsify.add_argument("--candidate", required=True)
    falsify.add_argument("--mechanism", required=True)
    falsify.add_argument("--baseline")
    falsify.add_argument("--n-games", type=int, default=500)
    falsify.add_argument("--seeds", default="20260511,20260512,20260513")
    falsify.add_argument("--device", default="auto")
    falsify.add_argument("--changed-path", action="append", default=[])
    falsify.add_argument("--solver-iterations", type=int, default=25)
    falsify.add_argument(
        "--solver-backend",
        choices=(
            "auto",
            "cpu",
            "torch-cuda",
            "torch-cpu",
            "torch-levelsync-cuda",
            "torch-levelsync-cpu",
            "segmented-cuda",
            "segmented-cpu",
        ),
        default="auto",
    )
    falsify.add_argument("--max-resolver-cases", type=int)
    falsify.add_argument(
        "--strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        default="regret",
    )
    falsify.add_argument("--timeout-seconds", type=int, default=3600)

    mixture_falsify = subparsers.add_parser(
        "enqueue-sd-cfr-mixture-falsification",
        help="Create and queue a local falsification gate for a fixed SD-CFR checkpoint mixture.",
    )
    mixture_falsify.add_argument("--candidate-glob", action="append", default=[])
    mixture_falsify.add_argument("--candidate-checkpoint", action="append", default=[])
    mixture_falsify.add_argument("--mechanism", required=True)
    mixture_falsify.add_argument("--baseline")
    mixture_falsify.add_argument("--n-games", type=int, default=500)
    mixture_falsify.add_argument("--seeds", default="20260511,20260512,20260513")
    mixture_falsify.add_argument("--device", default="auto")
    mixture_falsify.add_argument("--changed-path", action="append", default=[])
    mixture_falsify.add_argument("--review-dir")
    mixture_falsify.add_argument(
        "--strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        default="regret",
    )
    mixture_falsify.add_argument("--timeout-seconds", type=int, default=3600)

    review = subparsers.add_parser(
        "enqueue-review",
        help="Create and queue a methodology review gate with verifier and related-work artifacts.",
    )
    review.add_argument("--subject", required=True)
    review.add_argument("--trigger", required=True)
    review.add_argument("--claim", required=True)
    review.add_argument("--timeout-seconds", type=int, default=600)

    synthesis = subparsers.add_parser(
        "enqueue-synthesis",
        help="Create and queue a failure-synthesis gate after repeated experiments.",
    )
    synthesis.add_argument("--subject", required=True)
    synthesis.add_argument("--timeout-seconds", type=int, default=600)

    innovation = subparsers.add_parser(
        "enqueue-innovation-review",
        help="Create and queue a paradigm-innovation review gate.",
    )
    innovation.add_argument("--subject", required=True)
    innovation.add_argument("--anomaly", required=True)
    innovation.add_argument("--timeout-seconds", type=int, default=600)

    manifest = subparsers.add_parser(
        "write-review-manifest",
        help="Write a tracked digest manifest for an ignored review bundle.",
    )
    manifest.add_argument("--review-dir", required=True)

    calibration = subparsers.add_parser(
        "enqueue-calibration-audit",
        help="Create and queue a callback-state DCVN calibration audit.",
    )
    calibration.add_argument("--train-dual-cache", required=True)
    calibration.add_argument("--holdout-dual-cache", required=True)
    calibration.add_argument("--supervised-metrics")
    calibration.add_argument("--leaf-ab")
    calibration.add_argument("--output-json")
    calibration.add_argument("--timeout-seconds", type=int, default=900)

    knob = subparsers.add_parser(
        "add-knob",
        help="Register one persistent research knob with a mechanism and removal criterion.",
    )
    knob.add_argument("--name", required=True)
    knob.add_argument("--default", required=True)
    knob.add_argument("--failure-class", required=True)
    knob.add_argument("--mechanism", required=True)
    knob.add_argument("--rationale", required=True)
    knob.add_argument("--removal-criterion", required=True)
    knob.add_argument("--review-dir", required=True)

    audit = subparsers.add_parser(
        "objective-audit",
        help="Check changed files for protected evaluation-surface drift.",
    )
    audit.add_argument(
        "--changed-path",
        action="append",
        default=[],
        help="Repository-relative changed path. Repeat for multiple paths.",
    )
    audit.add_argument("--review-dir", help="Completed methodology review bundle.")
    audit.add_argument("--base-ref", help="Optional git ref to diff against when paths are omitted.")
    audit.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow an audit with no changed paths. Without this, empty audits fail.",
    )

    commit_ready = subparsers.add_parser(
        "commit-ready",
        help="Report whether the current research batch is ready for a natural commit.",
    )
    commit_ready.add_argument("--changed-path", action="append", default=[])
    commit_ready.add_argument("--base-ref")
    commit_ready.add_argument("--review-dir")
    commit_ready.add_argument("--allow-empty", action="store_true")

    train = subparsers.add_parser(
        "enqueue-train",
        help="Create and queue a GPU Deep CFR candidate-training gate.",
    )
    train.add_argument("--n-iterations", type=int, default=10)
    train.add_argument("--n-traversals", type=int, default=1000)
    train.add_argument("--n-training-steps", type=int, default=1000)
    train.add_argument("--buffer-capacity", type=int, default=2_000_000)
    train.add_argument("--hidden-dim", type=int, default=512)
    train.add_argument("--n-layers", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--traversal-pool-max-slots", type=int, default=1_000_000)
    train.add_argument("--traversal-slots-per-traversal", type=int, default=7000)
    train.add_argument("--use-frontier-indexing", action="store_true")
    train.add_argument("--policy-slots-per-traversal", type=int, default=64)
    train.add_argument("--max-pool-exhausted-per-traversal", type=float)
    train.add_argument("--max-overflow-chunk-fraction", type=float)
    train.add_argument("--max-rejected-traversal-chunks", type=int)
    train.add_argument("--min-traversals-per-second", type=float)
    train.add_argument("--average-strategy-weight", type=float, default=0.0)
    train.add_argument("--average-strategy-memory-capacity", type=int, default=0)
    train.add_argument("--average-strategy-batch-size", type=int, default=0)
    train.add_argument("--average-strategy-targets")
    train.add_argument("--search-targets")
    train.add_argument("--search-target-weight", type=float, default=0.0)
    train.add_argument("--search-target-batch-size", type=int, default=0)
    train.add_argument("--save-dir")
    train.add_argument("--prefix", default="candidate")
    train.add_argument("--save-every", type=int, default=0)
    train.add_argument("--save-replay-buffers", action="store_true")
    train.add_argument("--require-replay-buffer-resume", action="store_true")
    train.add_argument("--resume")
    train.add_argument("--eval-games", type=int, default=0)
    train.add_argument("--seed", type=int)
    train.add_argument(
        "--auto-compare",
        action="store_true",
        help="After training passes, queue incumbent head-to-head comparisons for emitted checkpoints.",
    )
    train.add_argument("--compare-n-games", type=int, default=500)
    train.add_argument("--compare-seeds", default="20260511,20260512,20260513")
    train.add_argument("--compare-device", default="auto")
    train.add_argument("--compare-timeout-seconds", type=int, default=2400)
    train.add_argument(
        "--compare-strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        default="regret",
        help="Strategy source to use for auto-queued checkpoint comparisons.",
    )
    train.add_argument(
        "--compare-candidate-strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        help="Override --compare-strategy-source for the trained candidate only.",
    )
    train.add_argument(
        "--compare-baseline-strategy-source",
        choices=("regret", "policy-head", "average-policy", "policy-head-covered"),
        help="Override --compare-strategy-source for the incumbent baseline only.",
    )
    train.add_argument("--timeout-seconds", type=int, default=7200)

    close = subparsers.add_parser("close-cycle", help="Close the active research cycle.")
    close.add_argument("--run-id", required=True)
    close.add_argument("--outcome", required=True, choices=["passed", "failed", "blocked"])
    close.add_argument("--failure-class", required=True)
    close.add_argument("--metrics-path")
    close.add_argument("--summary", required=True)

    loop = subparsers.add_parser("continuous", help="Run queued cycles until stopped.")
    loop.add_argument("--max-cycles", type=int)
    loop.add_argument("--sleep-seconds", type=float, default=30.0)
    loop.add_argument(
        "--max-idle-checks",
        type=int,
        help="Dry-run helper: stop after this many empty-queue checks.",
    )
    loop.add_argument(
        "--continue-on-mechanism-fail",
        action="store_true",
        help=(
            "Treat failed mechanism gates as soft failures: document the failed "
            "cycle, queue synthesis plus innovation review, and continue."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root)

    if args.command == "init":
        _emit(init_state(root, force=args.force))
        return 0

    if args.command == "status":
        report = readiness_report(root)
        _emit(report)
        return 0 if report["ready"] else 2

    if args.command == "drift-status":
        state_path = root / "autoresearch-session" / "poker_state.json"
        next_cycle = None
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            queue = state.get("hypothesis_queue", [])
            next_cycle = queue[0] if queue else None
        _emit(research_drift_status(root, next_cycle=next_cycle))
        return 0

    if args.command == "set-incumbent":
        _emit(set_incumbent(root, args.checkpoint, reason=args.reason))
        return 0

    if args.command == "set-phase":
        _emit(set_research_phase(root, phase=args.phase, reason=args.reason))
        return 0

    if args.command == "synthesis-status":
        _emit(synthesis_status(root))
        return 0

    if args.command == "gate":
        result = run_gate(root, args.name, run_dir=args.run_dir)
        _emit(result)
        return 0 if result["passed"] else 1

    if args.command == "new-cycle":
        _emit(
            new_cycle(
                root,
                hypothesis=args.hypothesis,
                cycle_type=args.cycle_type,
                failure_class=args.failure_class,
                gate=args.gate,
            )
        )
        return 0

    if args.command == "enqueue":
        _emit(
            enqueue_cycle(
                root,
                hypothesis=args.hypothesis,
                cycle_type=args.cycle_type,
                failure_class=args.failure_class,
                gate=args.gate,
            )
        )
        return 0

    if args.command == "enqueue-compare":
        _emit(
            enqueue_candidate_comparison(
                root,
                args.candidate,
                baseline_checkpoint=args.baseline,
                n_games=args.n_games,
                seeds=args.seeds,
                device=args.device,
                timeout_seconds=args.timeout_seconds,
                head_to_head=args.head_to_head,
                strategy_source=args.strategy_source,
                candidate_strategy_source=args.candidate_strategy_source,
                baseline_strategy_source=args.baseline_strategy_source,
            )
        )
        return 0

    if args.command == "enqueue-promotion-gate":
        _emit(
            enqueue_candidate_promotion_gate(
                root,
                rlcard_reference_json=args.rlcard_reference_json,
                native_h2h_json=args.native_h2h_json,
                empirical_game_json=args.empirical_game_json,
                native_candidate_checkpoint=args.native_candidate_checkpoint,
                min_lower95=args.min_lower95,
                min_candidate_support=args.min_candidate_support,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "enqueue-slumbot":
        _emit(
            enqueue_slumbot_smoke(
                root,
                args.model,
                model_kind=args.model_kind,
                hands=args.hands,
                greedy=args.greedy,
                no_allin=args.no_allin,
                no_solver=args.no_solver,
                strategy_source=args.strategy_source,
                solver_backend=args.solver_backend,
                solver_budget_profile=args.solver_budget_profile,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "enqueue-resolver":
        _emit(
            enqueue_resolver_benchmark(
                root,
                args.model,
                solver_iterations=args.solver_iterations,
                solver_backend=args.solver_backend,
                max_cases=args.max_cases,
                device=args.device,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "enqueue-cfr-budget-frontier":
        _emit(
            enqueue_cfr_budget_frontier(
                root,
                cases_json=args.cases,
                cfv_cache=args.cfv_cache,
                budgets=args.budgets,
                start_index=args.start_index,
                limit=args.limit,
                reference_iterations=args.reference_iterations,
                solver_backend=args.solver_backend,
                solver_update=args.solver_update,
                min_evaluated=args.min_evaluated,
                output_json=args.output_json,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "enqueue-cfr-matrix-footprint":
        _emit(
            enqueue_cfr_matrix_footprint(
                root,
                cases_json=args.cases,
                start_index=args.start_index,
                max_cases=args.max_cases,
                chunk_memory_cap_mib=args.chunk_memory_cap_mib,
                output_json=args.output_json,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "enqueue-warm-start-resolver":
        _emit(
            enqueue_warm_start_resolver_gate(
                root,
                checkpoint=args.checkpoint,
                cases_json=args.cases,
                cfv_cache=args.cfv_cache,
                train_labels_npz=args.train_labels_npz,
                start_index=args.start_index,
                limit=args.limit,
                low_iterations=args.low_iterations,
                baseline_iterations=args.baseline_iterations,
                reference_iterations=args.reference_iterations,
                solver_backend=args.solver_backend,
                device=args.device,
                regret_mass_scale=args.regret_mass_scale,
                strategy_mass=args.strategy_mass,
                min_evaluated=args.min_evaluated,
                max_warm_latency_ratio=args.max_warm_latency_ratio,
                output_json=args.output_json,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "enqueue-falsification":
        _emit(
            enqueue_falsification_ladder(
                root,
                args.candidate,
                mechanism=args.mechanism,
                baseline_checkpoint=args.baseline,
                n_games=args.n_games,
                seeds=args.seeds,
                device=args.device,
                changed_paths=args.changed_path,
                solver_iterations=args.solver_iterations,
                solver_backend=args.solver_backend,
                max_resolver_cases=args.max_resolver_cases,
                strategy_source=args.strategy_source,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "enqueue-sd-cfr-mixture-falsification":
        _emit(
            enqueue_sd_cfr_mixture_falsification(
                root,
                candidate_globs=args.candidate_glob,
                candidate_checkpoints=args.candidate_checkpoint,
                mechanism=args.mechanism,
                baseline_checkpoint=args.baseline,
                n_games=args.n_games,
                seeds=args.seeds,
                device=args.device,
                changed_paths=args.changed_path,
                review_dir=args.review_dir,
                strategy_source=args.strategy_source,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "enqueue-review":
        _emit(
            enqueue_methodology_review(
                root,
                subject=args.subject,
                trigger=args.trigger,
                claim=args.claim,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "enqueue-synthesis":
        _emit(
            enqueue_failure_synthesis(
                root,
                subject=args.subject,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "enqueue-innovation-review":
        _emit(
            enqueue_paradigm_innovation_review(
                root,
                subject=args.subject,
                anomaly=args.anomaly,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "write-review-manifest":
        _emit(write_review_manifest(root, args.review_dir))
        return 0

    if args.command == "enqueue-calibration-audit":
        _emit(
            enqueue_callback_calibration_audit(
                root,
                train_dual_cache=args.train_dual_cache,
                holdout_dual_cache=args.holdout_dual_cache,
                supervised_metrics=args.supervised_metrics,
                leaf_ab=args.leaf_ab,
                output_json=args.output_json,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "add-knob":
        _emit(
            register_research_knob(
                root,
                name=args.name,
                default=args.default,
                failure_class=args.failure_class,
                mechanism=args.mechanism,
                rationale=args.rationale,
                removal_criterion=args.removal_criterion,
                review_dir=args.review_dir,
            )
        )
        return 0

    if args.command == "objective-audit":
        changed_paths = args.changed_path
        if not changed_paths:
            changed_paths = git_changed_paths(root, base_ref=args.base_ref)
            if not changed_paths and not args.allow_empty:
                print(
                    "No changed paths found; pass --allow-empty for an explicit no-op audit.",
                    file=sys.stderr,
                )
                return 2
        result = audit_objective_alignment(
            root,
            changed_paths=changed_paths,
            review_dir=args.review_dir,
        )
        _emit(result)
        return 0 if result["passed"] else 1

    if args.command == "commit-ready":
        result = commit_ready_report(
            root,
            changed_paths=args.changed_path or None,
            base_ref=args.base_ref,
            review_dir=args.review_dir,
            allow_empty=args.allow_empty,
        )
        _emit(result)
        return 0 if result["ready"] else 1

    if args.command == "enqueue-train":
        _emit(
            enqueue_gpu_training(
                root,
                n_iterations=args.n_iterations,
                n_traversals=args.n_traversals,
                n_training_steps=args.n_training_steps,
                buffer_capacity=args.buffer_capacity,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                batch_size=args.batch_size,
                traversal_pool_max_slots=args.traversal_pool_max_slots,
                traversal_slots_per_traversal=args.traversal_slots_per_traversal,
                use_frontier_indexing=args.use_frontier_indexing,
                policy_slots_per_traversal=args.policy_slots_per_traversal,
                max_pool_exhausted_per_traversal=args.max_pool_exhausted_per_traversal,
                max_overflow_chunk_fraction=args.max_overflow_chunk_fraction,
                max_rejected_traversal_chunks=args.max_rejected_traversal_chunks,
                min_traversals_per_second=args.min_traversals_per_second,
                average_strategy_weight=args.average_strategy_weight,
                average_strategy_memory_capacity=args.average_strategy_memory_capacity,
                average_strategy_batch_size=args.average_strategy_batch_size,
                average_strategy_targets=args.average_strategy_targets,
                search_targets=args.search_targets,
                search_target_weight=args.search_target_weight,
                search_target_batch_size=args.search_target_batch_size,
                save_dir=args.save_dir,
                prefix=args.prefix,
                save_every=args.save_every,
                save_replay_buffers=args.save_replay_buffers,
                require_replay_buffer_resume=args.require_replay_buffer_resume,
                resume=args.resume,
                eval_games=args.eval_games,
                auto_compare=args.auto_compare,
                compare_n_games=args.compare_n_games,
                compare_seeds=args.compare_seeds,
                compare_device=args.compare_device,
                compare_timeout_seconds=args.compare_timeout_seconds,
                compare_strategy_source=args.compare_strategy_source,
                compare_candidate_strategy_source=args.compare_candidate_strategy_source,
                compare_baseline_strategy_source=args.compare_baseline_strategy_source,
                seed=args.seed,
                timeout_seconds=args.timeout_seconds,
            )
        )
        return 0

    if args.command == "close-cycle":
        _emit(
            close_cycle(
                root,
                args.run_id,
                outcome=args.outcome,
                failure_class=args.failure_class,
                metrics_path=args.metrics_path,
                summary=args.summary,
            )
        )
        return 0

    if args.command == "continuous":
        result = continuous(
            root,
            max_cycles=args.max_cycles,
            sleep_seconds=args.sleep_seconds,
            max_idle_checks=args.max_idle_checks,
            continue_on_mechanism_fail=args.continue_on_mechanism_fail,
        )
        _emit(result)
        return 1 if result["stopped_reason"] in {"not_ready", "gate_failed"} else 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
