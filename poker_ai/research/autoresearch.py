"""Autoresearch workflow primitives for poker AI experiments.

The module intentionally stays independent from trainer internals. It provides
the durable loop mechanics: local state, gates, metrics, logs, and STOP-file
handling.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from glob import glob
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fnmatch import fnmatch
from hashlib import sha256
from pathlib import Path
from typing import Callable, Iterable, Sequence


SESSION_DIR = "autoresearch-session"
RUNS_DIR = "poker_runs"
REVIEWS_DIR = "poker_reviews"
GOAL_FILE = "poker_goal.json"
STATE_FILE = "poker_state.json"
KNOBS_FILE = "poker_knobs.tsv"
STOP_FILE = "STOP"
RESEARCH_LOG = "RESEARCH_LOG.md"
REVIEW_MANIFESTS_DIR = Path("docs") / "research_protocols" / "poker_review_manifests"
ALLOWED_REVIEW_DECISIONS = {"proceed", "revise", "abandon", "gather_more_evidence"}
HARD_STOP_FAILURE_CLASSES = {
    "eval_invalid",
    "evaluation_invalid",
    "objective_drift",
    "benchmark_hacking",
    "action_mapping",
    "legal_mask",
    "readiness",
}
LOCAL_TARGET_CONSUMER_TERMS = (
    "xdo",
    "npi",
    "target consumer",
    "policy consumer",
    "search target",
    "local target",
    "exact-oracle target",
    "policy-continuation",
    "supervised imitation",
    "imitator",
)
RL_RESPONSE_ORACLE_TERMS = (
    "response oracle",
    "rainbow",
    "ppo",
    "nfsp",
    "fsp",
    "marl",
    "ippo",
    "tianshou",
    "agilerl",
    "openspiel",
    "stochastic neural actor",
    "empirical-game meta-policy",
)
ALLOWED_RESEARCH_PHASES = {
    "open_research",
    "callback_state_calibration_debug",
    "exact_gpu_resolving",
    "neural_regret_field_resolving",
    "self_play_policy_improvement",
}
PROTECTED_EVAL_SURFACES = [
    "scripts/poker_autoresearch.py",
    "scripts/poker_autoresearch_eval.py",
    "scripts/poker_autoresearch_slumbot.py",
    "scripts/run_neural_policy_iteration_loop.py",
    "scripts/eval_mixed_policy_h2h.py",
    "scripts/poker_resolver_benchmark.py",
    "scripts/play_slumbot.py",
    "scripts/solver.py",
    "scripts/fast_cfr.py",
    "scripts/eval_cfr_budget_frontier.py",
    "scripts/eval_solver_budget_profiles.py",
    "scripts/eval_solver_budget_selective_escalation.py",
    "scripts/eval_solver_budget_boundary_predictor.py",
    "scripts/eval_slumbot_response_range_ev_gate.py",
    "scripts/analyze_cfr_trace_sequence_predictor.py",
    "scripts/run_frozen_best_response.py",
    "scripts/eval_restricted_action_values.py",
    "scripts/poker_objective_audit.py",
    "poker_ai/research/autoresearch.py",
    "poker_ai/research/mixed_policy_h2h.py",
    "poker_ai/research/promotion.py",
    "test/unit/test_network_mask.py",
    "test/unit/test_slumbot_mapping.py",
    "test/unit/test_legal_mask_parity.py",
    "test/unit/test_poker_autoresearch_eval.py",
]
PROTECTED_EVAL_SURFACE_PATTERNS = [
    "scripts/eval_*.py",
    "scripts/analyze_*trace*.py",
    "scripts/build_*trace*.py",
    "scripts/*slumbot*.py",
    "scripts/*resolver*.py",
    "scripts/*objective_audit*.py",
    "scripts/*methodology_review*.py",
    "scripts/*synthesis_review*.py",
    "poker_ai/research/*trace*.py",
    "poker_ai/research/*slumbot*.py",
    "poker_ai/research/*resolver*.py",
    "poker_ai/research/*promotion*.py",
    "test/unit/test_*slumbot*.py",
    "test/unit/test_*resolver*.py",
    "test/unit/test_*parity*.py",
    "test/unit/test_*autoresearch*.py",
]
KNOB_COLUMNS = [
    "name",
    "status",
    "default",
    "failure_class",
    "mechanism",
    "rationale",
    "removal_criterion",
    "review_id",
    "created_at",
    "retired_at",
]
FORCE_SYNC_GOAL_KEYS = {
    "objective",
    "primary_metric",
}
FORCE_SYNC_GOAL_SECTION_KEYS = {
    "objective_alignment_policy": {
        "active_method_target",
        "learning_philosophy",
        "long_term_objective",
        "neural_policy_iteration_policy",
        "promotion_requires",
        "slumbot_validation_policy",
        "self_play_league_policy",
        "visible_metric_rule",
    },
    "agent_goal_contract": {
        "current_frontier",
        "success_criteria",
        "blocked_pivots",
        "hard_stop_rules",
    },
    "mechanism_brief_policy": {
        "required_for",
        "required_fields",
        "reject_if_missing",
    },
    "research_phase": {
        "self_play_policy_improvement",
    },
}


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    seconds: float


CommandRunner = Callable[[Sequence[str], float | None], CommandResult]


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _session(root: Path) -> Path:
    return root / SESSION_DIR


def _goal_path(root: Path) -> Path:
    return _session(root) / GOAL_FILE


def _state_path(root: Path) -> Path:
    return _session(root) / STATE_FILE


def _knobs_path(root: Path) -> Path:
    return _session(root) / KNOBS_FILE


def _runs_path(root: Path) -> Path:
    return _session(root) / RUNS_DIR


def _reviews_path(root: Path) -> Path:
    return _session(root) / REVIEWS_DIR


def _review_manifest_dir(root: Path) -> Path:
    return root / REVIEW_MANIFESTS_DIR


def _stop_path(root: Path) -> Path:
    return _session(root) / STOP_FILE


def _log_path(root: Path) -> Path:
    return root / RESEARCH_LOG


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_mode(path: Path) -> str:
    """Return lightweight checkpoint mode metadata when it is cheaply available."""
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("mode", ""))


def _warm_start_evaluator_script(checkpoint: Path) -> str:
    mode = _checkpoint_mode(checkpoint)
    if mode == "regret_policy_warm_start_checkpoint":
        return "scripts/eval_regret_policy_warm_start.py"
    return "scripts/eval_joint_pbs_policy_warm_start.py"


def _first_numeric_key(payload: object, keys: set[str]) -> tuple[str, float] | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys and isinstance(value, (int, float)):
                return key, float(value)
            found = _first_numeric_key(value, keys)
            if found is not None:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = _first_numeric_key(value, keys)
            if found is not None:
                return found
    return None


def _internal_self_play_league_evidence(root: Path) -> dict:
    state = _read_json(_state_path(root))
    lower95_keys = {
        "best_worst_lower95",
        "best_worst_lower95_chips_per_hand",
        "lower95_chips_per_hand",
        "paired_delta_lower95_chips_per_hand_across_seeds",
    }
    for record in reversed(state.get("history", [])):
        if record.get("outcome") != "passed":
            continue
        descriptor = " ".join(
            str(record.get(key, ""))
            for key in ("gate", "type", "hypothesis", "summary")
        ).lower()
        if not any(token in descriptor for token in ("self-play", "self_play", "league")):
            continue
        metrics_path = Path(str(record.get("run_dir", ""))) / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            try:
                metrics = _read_json(metrics_path)
            except json.JSONDecodeError:
                metrics = {}
        lower95 = _first_numeric_key(metrics, lower95_keys)
        if lower95 is not None and lower95[1] > 0.0:
            return {
                "passed": True,
                "run_id": record.get("run_id"),
                "gate": record.get("gate"),
                "metric": lower95[0],
                "value": lower95[1],
            }
    return {
        "passed": False,
        "reason": "no passed self-play checkpoint league with positive lower95 evidence",
    }


def _candidate_promotion_gate_evidence(root: Path) -> dict:
    """Return the latest passed dual-surface pre-Slumbot promotion evidence."""
    state = _read_json(_state_path(root))
    for record in reversed(state.get("history", [])):
        if record.get("outcome") != "passed":
            continue
        descriptor = " ".join(
            str(record.get(key, ""))
            for key in ("gate", "type", "hypothesis", "summary")
        ).lower()
        if "promotion" not in descriptor or "slumbot" not in descriptor:
            continue
        metrics_path = Path(str(record.get("run_dir", ""))) / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = _read_json(metrics_path)
        except json.JSONDecodeError:
            continue
        if (
            metrics.get("algorithm") == "poker_candidate_promotion_gate"
            and bool(metrics.get("passed", False))
            and bool(metrics.get("slumbot_confidence_eligible", False))
        ):
            return {
                "passed": True,
                "run_id": record.get("run_id"),
                "gate": record.get("gate"),
                "metrics_path": str(metrics_path),
                "promotion_blockers": list(metrics.get("promotion_blockers", [])),
            }
    return {
        "passed": False,
        "reason": (
            "no passed pre-Slumbot candidate promotion gate with RLCard reference, "
            "native H2H, and empirical-game evidence"
        ),
    }


def _append_missing(items: list, defaults: list) -> list:
    seen = set(items)
    for item in defaults:
        if item not in seen:
            items.append(item)
            seen.add(item)
    return items


def _merge_default_dict(existing: dict, defaults: dict) -> dict:
    for key, default_value in defaults.items():
        if key not in existing:
            existing[key] = default_value
            continue
        if isinstance(existing[key], dict) and isinstance(default_value, dict):
            _merge_default_dict(existing[key], default_value)
        elif isinstance(existing[key], list) and isinstance(default_value, list):
            _append_missing(existing[key], default_value)
    return existing


def _sync_goal_defaults(goal: dict, default_goal: dict) -> dict:
    for key, default_value in default_goal.items():
        if key in FORCE_SYNC_GOAL_KEYS:
            goal[key] = default_value
            continue
        if key == "gates":
            goal.setdefault("gates", {})
            for gate_name, gate_config in default_value.items():
                if gate_name not in goal["gates"]:
                    goal["gates"][gate_name] = gate_config
                    continue
                existing = goal["gates"][gate_name]
                existing.setdefault("description", gate_config.get("description", ""))
                existing.setdefault("timeout_seconds", gate_config.get("timeout_seconds"))
                existing.setdefault("commands", [])
                existing_commands = {tuple(command) for command in existing["commands"]}
                for command in gate_config.get("commands", []):
                    if tuple(command) not in existing_commands:
                        existing["commands"].append(command)
                        existing_commands.add(tuple(command))
            continue
        if key == "constraints":
            goal[key] = _append_missing(list(goal.get(key, [])), default_value)
            continue
        if isinstance(default_value, dict):
            merged = _merge_default_dict(goal.setdefault(key, {}), default_value)
            for forced_key in FORCE_SYNC_GOAL_SECTION_KEYS.get(key, set()):
                if forced_key in default_value:
                    merged[forced_key] = default_value[forced_key]
            continue
        goal.setdefault(key, default_value)
    return goal


def _migrate_knobs_file(path: Path) -> None:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        path.write_text("\t".join(KNOB_COLUMNS) + "\n", encoding="utf-8")
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t") if lines else []
    if header == KNOB_COLUMNS:
        return

    migrated = ["\t".join(KNOB_COLUMNS)]
    for line in lines[1:]:
        if not line.strip():
            continue
        record = dict(zip(header, line.split("\t"), strict=False))
        row = {
            "name": record.get("name", ""),
            "status": record.get("status") or "active",
            "default": record.get("default", ""),
            "failure_class": record.get("failure_class", ""),
            "mechanism": record.get("mechanism", ""),
            "rationale": record.get("rationale", ""),
            "removal_criterion": record.get("removal_criterion", ""),
            "review_id": record.get("review_id", ""),
            "created_at": record.get("created_at", ""),
            "retired_at": record.get("retired_at", ""),
        }
        migrated.append("\t".join(_tsv_clean(row.get(column, "")) for column in KNOB_COLUMNS))
    path.write_text("\n".join(migrated) + "\n", encoding="utf-8")


def _project_python(root: str | Path | None = None) -> str:
    """Return the Python executable that should run repo-local gates."""
    candidates = []
    if root is not None:
        candidates.append(Path(root) / ".venv" / "bin" / "python")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "python")
    candidates.append(Path(sys.executable))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _default_goal(root: str | Path | None = None) -> dict:
    python = _project_python(root)
    return {
        "objective": (
            "Continuously run the poker autoresearch outer loop until explicit "
            "user stop or real promotion evidence is achieved. Develop and "
            "falsify a tabula-rasa game-theoretic neural self-play method for "
            "full-deck heads-up no-limit hold'em: train stochastic policy/value "
            "networks from fresh weights using only the simulator, observations, "
            "legal actions, and rewards; prefer R-NaD/NashPG/MMD-style "
            "regularized self-play when plain policy-gradient learners cycle; "
            "and add a candidate only if it beats parent and population gates "
            "before any Slumbot confidence run. Slumbot remains held-out "
            "evaluation only, with tiny smoke allowed for integration and "
            "catastrophic-transfer checks. The method must stay elegant, "
            "novel, bitter lesson aligned, personal-PC trainable, and free of "
            "hand-crafted poker-strategy rules or default solver-label imitation."
        ),
        "constraints": [
            "keep generated checkpoints and raw run artifacts out of git",
            "do not add opponent-specific or street-specific human strategy rules",
            "do not use solver-generated policy labels as the default training target for the mainline learner",
            "do not use Slumbot as a primary promotion signal before self-play league gates pass",
            "train a fresh environment-native model per card/action environment; transfer the training schema, not checkpoints or action mappings",
            "treat explicit opponent ranges as diagnostic scaffolding unless a counterfactual-EV gate passes",
            "run Tier 0 integrity before promoting any result",
            "classify every failed or inconclusive cycle",
            "batch commits by research objective rather than by individual gate",
            "run methodology review before method, promotion, or persistent knob changes",
            "run a paradigm innovation review before repeating a failed mechanism family",
        ],
        "agent_goal_contract": {
            "north_star": (
                "Build a heads-up no-limit hold'em agent whose stochastic neural "
                "policy improves through tabula-rasa local self-play/population competition, "
                "then validates externally on Slumbot only after internal parent "
                "and empirical-game population gates pass."
            ),
            "current_frontier": {
                "mechanism": (
                    "R-NaD-style tabula-rasa neural self-play scaled from exact "
                    "small poker games to the native full-deck 9-action game, "
                    "with NashPG/MMD-style regularized policy-gradient variants "
                    "allowed as reviewed successors"
                ),
                "why_now": (
                    "The current AlphaHoldem-style PPO branch failed the exact "
                    "small-NLHE hardening gate, while R-NaD reduced exact "
                    "NashConv reliably. The next work should scale the aligned "
                    "regularized self-play learner rather than continue broad "
                    "method-shopping or solver-label target fitting."
                ),
                "decision_object": (
                    "stochastic neural policies, native empirical-game payoffs, "
                    "meta-strategy support, parent/population H2H lower bounds, "
                    "and small-game exact NashConv diagnostics"
                ),
            },
            "success_criteria": [
                "candidate self-play checkpoint beats its parent/incumbent with positive lower95",
                "complete native empirical game keeps the candidate in meta-strategy support",
                "candidate does not lose to saved local controls with positive-confidence evidence",
                "candidate remains legal and traversal-valid with zero rejected chunks in fidelity-gated runs",
                "external Slumbot confidence validation is attempted only after repeated internal population gates pass",
                "public RLCard and native 9-action evidence come from separately trained environment-native candidates, not cross-environment checkpoint adaptation",
            ],
            "completion_rules": {
                "mechanism_failure_is_not_completion": True,
                "complete_only_on_promotion_evidence": (
                    "A goal is complete only after a candidate clears internal "
                    "self-play league gates and held-out Slumbot transfer "
                    "confidence, or the user explicitly changes the objective."
                ),
                "soft_pivot_is_part_of_workflow": (
                    "A failed mechanism retires that inner hypothesis, then "
                    "triggers synthesis, related-work review, paradigm "
                    "innovation review, and the next principled mechanism."
                ),
            },
            "soft_pivot_rules": [
                "after a failed decision-impact mechanism gate, document the result and queue failure synthesis",
                "after repeated same-failure-class results, queue paradigm innovation review and related-work research",
                "treat pivot review as continuation of the outer goal, not as completion",
                "choose the next mechanism from the synthesis and innovation review before more experiments",
            ],
            "held_out_validation": [
                "Slumbot confidence runs are held-out external validation, not the optimization target",
                "tiny Slumbot smoke is allowed for integration and catastrophic-transfer checks only before internal league promotion",
            ],
            "blocked_pivots": [
                "do not tune loss-weight, target count, hidden size, or selector thresholds after a failed mechanism without a positive root-decision gate",
                "do not weaken evaluation, parser, legal-mask, or promotion surfaces to improve visible metrics",
                "do not promote policy patches that only fit supervised or solver labels without self-play league impact",
            ],
            "hard_stop_rules": [
                "stop only for explicit user STOP, readiness failure, invalid evaluation, benchmark hacking, or unsafe objective drift",
                "stop if a proposed change primarily optimizes a visible benchmark instead of a mechanism",
                "stop if a control run has rejected traversal chunks or unmatched compute budget",
            ],
        },
        "commit_policy": {
            "mode": "batch_by_research_objective",
            "commit_during_continuous": False,
            "natural_boundaries": [
                "workflow feature complete",
                "experiment batch complete",
                "methodology review complete",
                "documentation synchronized",
            ],
            "required_commit_body": [
                "objective",
                "files changed",
                "tests or gates run",
                "key result",
                "review or related-work status",
            ],
        },
        "review_policy": {
            "required_for": [
                "methodology_review_required",
                "mechanism_review_required",
                "checkpoint promotion",
                "architecture or objective change",
                "evaluation protocol change",
                "persistent research knob addition",
                "surprising or contradictory result interpretation",
            ],
            "requires_independent_verifier": True,
            "requires_related_work": True,
            "requires_mechanism_review": True,
            "requires_review_manifest": True,
            "requires_review_scope": True,
            "requires_decision_impact_statement": True,
            "mechanism_review_fields": [
                "learned object",
                "search boundary",
                "train distribution",
                "eval distribution",
                "falsifier",
                "pass action",
                "fail action",
                "related-work delta",
            ],
            "review_scope_fields": [
                "changed_paths",
                "protected_hits",
                "mechanism",
                "expected_gate",
                "decision_impact",
                "decision_impact_gate",
                "fallback_if_no_decision_impact",
                "pass_action",
                "fail_action",
            ],
            "decision_impact_rule": (
                "Every new CUDA/search primitive must explicitly state how it "
                "gets the project closer to stronger root decisions per "
                "millisecond or better self-play checkpoint-league strength. "
                "If that path is indirect, the review must name the next "
                "decision-impact gate and the fallback if impact is absent."
            ),
        },
        "mechanism_brief_policy": {
            "required_for": [
                "methodology_review",
                "paradigm_innovation_review",
                "persistent_knob",
                "new_training_objective",
                "new_search_or_resolver_primitive",
            ],
            "required_fields": [
                "decision_object",
                "where_consumed",
                "matched_control",
                "primary_decision_gate",
                "retirement_criterion",
                "flexibility_boundary",
                "anti_benchmark_hack",
                "neural_policy_role",
                "cfr_role",
                "stochastic_policy_contract",
            ],
            "reject_if_missing": True,
            "rule": (
                "A research action may stay flexible in implementation, but it "
                "must pin down the object being improved, the matched control, "
                "and the result that retires the mechanism before protected "
                "methodology changes are accepted."
            ),
        },
        "synthesis_policy": {
            "experiments_per_synthesis": 5,
            "required_fields": [
                "current causal model",
                "retired hypotheses",
                "live hypotheses",
                "single next test",
            ],
            "rule": (
                "After every five non-review experiments, pause expansion and "
                "write a failure synthesis before adding a new experiment family."
            ),
        },
        "innovation_policy": {
            "required_after_consecutive_failures": 2,
            "selection_rule": (
                "Use Bayesian surprise and root-cause anomaly value to choose "
                "one mechanism-level novelty sprint; do not select by easiest "
                "visible benchmark gain."
            ),
            "required_fields": [
                "Anomaly ledger",
                "Current-practice limit",
                "First-principles reduction",
                "Cross-paradigm analogy",
                "Novel mechanism",
                "Bitter-lesson alignment",
                "Smallest decisive test",
                "Falsifier",
            ],
            "thought_experiment_fields": [
                "Mechanism stress test",
                "Failure thought experiment",
                "Transfer thought experiment",
                "Compute thought experiment",
            ],
            "rule": (
                "After repeated failures in the same mechanism family, force a "
                "novelty review with online related work, first-principles "
                "analysis, paradigm alternatives, ideation, and thought "
                "experiments before another scale run or knob."
            ),
        },
        "research_phase": {
            "current": "self_play_policy_improvement",
            "reason": (
                "default after trace-specific and detached target mechanisms "
                "failed transfer; local self-play is the mainline and Slumbot "
                "is evaluation-only"
            ),
            "set_at": None,
            "allowed_phases": sorted(ALLOWED_RESEARCH_PHASES),
            "neural_regret_field_resolving": {
                "problem": (
                    "Hard learned CFV leaf/successor replacement and direct policy "
                    "imitation repeatedly fit local targets without improving "
                    "root-disjoint resolver behavior."
                ),
                "approved_method": (
                    "Learn a public-belief regret/policy initializer that warm-starts "
                    "CFR+/resolving; use search as the correction operator rather than "
                    "letting the network replace the solver."
                ),
                "allowed_actions": [
                    "methodology_review",
                    "mechanism_review",
                    "paradigm_innovation_review",
                    "failure_synthesis",
                    "objective_audit",
                    "implement_solver_warm_start",
                    "export_teacher_regret_field_targets",
                    "train_regret_field_initializer",
                    "eval_warm_start_resolver_gate",
                ],
                "blocked_actions": [
                    "hard_value_leaf_replacement_as_mainline",
                    "hard_successor_cut_replacement_as_mainline",
                    "final_distribution_policy_mixing_sweeps",
                    "new_model_size_sweeps_without_root_disjoint_gate",
                    "slumbot_confidence_before_warm_start_resolver_gate",
                ],
                "gate": (
                    "On root-disjoint public states, a neural-warm-started low-budget "
                    "resolver must move closer than the same-budget vanilla resolver "
                    "to a higher-budget teacher on root action L1/KL and top-action "
                    "agreement, without illegal actions or latency regression."
                ),
            },
            "callback_state_calibration_debug": {
                "problem": (
                    "The 4-root callback-state DCVN smoke passed, but the "
                    "32-train/16-holdout scale-up failed supervised baselines "
                    "and learned-leaf A/B."
                ),
                "allowed_actions": [
                    "callback_state_calibration_audit",
                    "methodology_review",
                    "mechanism_review",
                    "paradigm_innovation_review",
                    "failure_synthesis",
                    "objective_audit",
                ],
                "blocked_actions": [
                    "gpu_deep_cfr_training",
                    "slumbot_smoke",
                    "new_model_size_or_search_knob",
                ],
            },
            "exact_gpu_resolving": {
                "problem": (
                    "Static learned warm-starts and shallow trace controllers "
                    "lost to spending the same latency on exact CFR+ updates."
                ),
                "approved_method": (
                    "Use exact GPU CFR+/resolving as the current search baseline; "
                    "learned components must beat the CUDA budget frontier by root "
                    "decision quality per millisecond before becoming mainline."
                ),
                "allowed_actions": [
                    "methodology_review",
                    "mechanism_review",
                    "paradigm_innovation_review",
                    "failure_synthesis",
                    "objective_audit",
                    "solver_budget_frontier",
                    "solver_latency_profile",
                    "self_play_league_training",
                ],
                "blocked_actions": [
                    "static_warm_start_resolver_gate_as_mainline",
                    "another_warm_start_mass_sweep",
                    "policy_mixing_after_solve",
                    "slumbot_confidence_before_internal_league_pass",
                ],
                "gate": (
                    "Exact CUDA search changes must be judged by root-disjoint "
                    "budget frontiers, illegal-mass parity, latency, and downstream "
                    "self-play league evidence before any Slumbot confidence claim."
                ),
            },
            "self_play_policy_improvement": {
                "problem": (
                    "A maintained Tianshou Rainbow response oracle became the "
                    "best local plug-in RL incumbent, but the next identical "
                    "response-oracle generation failed its parent gate."
                ),
                "approved_method": (
                    "Use a stochastic neural policy/value actor as the player: "
                    "freeze the current local incumbent, solve the native "
                    "empirical game over local self-play checkpoints, train "
                    "maintained-library neural response "
                    "oracles against the empirical-game meta-policy or a reviewed "
                    "population approximation, and promote only by positive "
                    "parent plus population H2H lower-bound evidence before "
                    "held-out Slumbot validation."
                ),
                "allowed_actions": [
                    "methodology_review",
                    "mechanism_review",
                    "paradigm_innovation_review",
                    "failure_synthesis",
                    "objective_audit",
                    "gpu_deep_cfr_training",
                    "self_play_league_training",
                    "solver_budget_frontier",
                    "solver_latency_profile",
                ],
                "blocked_actions": [
                    "slumbot_confidence_before_internal_league_pass",
                    "slumbot_trace_training_data",
                    "repeat_identical_single_checkpoint_response_oracle_after_parent_gate_failure",
                    "always_on_cfr_argmax_as_mainline",
                    "response_range_as_live_opponent_model",
                    "post_hoc_policy_calibration_as_mainline",
                    "detached_trace_target_injection_as_mainline",
                ],
                "gate": (
                    "Candidates must beat parent/incumbent and the native "
                    "empirical-game population with positive lower-bound evidence "
                    "before any Slumbot confidence run."
                ),
            },
        },
        "knob_policy": {
            "max_active_knobs": 5,
            "rule": (
                "Every persistent knob needs a mechanism, one primary variable, "
                "and a removal criterion. Broad sweeps are rejected."
            ),
        },
        "architecture_policy": {
            "rule": (
                "Modern neural architectures are allowed and expected when they "
                "improve the learned search primitive. Capacity, attention, set "
                "encoders, recurrence, and mixed precision must be justified by a "
                "root-disjoint train/test split and a resolver-behavior gate."
            ),
            "approved_roles": [
                "exact GPU resolving baseline",
                "root-disjoint budget frontier",
                "public-belief encoder",
                "private-card set encoder",
                "action-sequence encoder",
                "regret/policy initializer",
                "uncertainty or calibration head for search warm starts",
            ],
            "blocked_roles": [
                "bigger network as a substitute for search evidence",
                "policy argmax patch without resolver correction",
                "architecture sweep scored only by local smoke or target fit",
            ],
        },
        "objective_alignment_policy": {
            "long_term_objective": (
                "Develop an elegant, novel, compute-efficient Texas hold'em method "
                "whose self-play checkpoint league strength improves over time on "
                "personal-PC hardware, then transfers to SOTA-style Slumbot and "
                "public baseline performance by using the available GPU for local "
                "tabula-rasa self-play, neural learning, and equilibrium-oriented "
                "regularized RL rather than Slumbot-specific fitting or solver-label "
                "imitation."
            ),
            "active_method_target": (
                "R-NaD-style tabula-rasa neural self-play scaled to the native "
                "full-deck 9-action game: start from fresh policy/value networks, "
                "train through self-play trajectories using simulator observations, "
                "legal actions, and rewards, compare against parent/population H2H "
                "gates, and keep Slumbot as held-out evaluation after internal "
                "progress."
            ),
            "learning_philosophy": (
                "Prefer tabula-rasa neural self-play adapted to imperfect "
                "information. A stochastic neural policy should remain the "
                "deployable player. R-NaD/NashPG/MMD-style regularized self-play "
                "is aligned because it learns from self-play trajectories while "
                "controlling multi-agent cycling. CFR/resolving remain acceptable "
                "as evaluators, diagnostics, or optional general search controls, "
                "but not as the default supervised target source or a Slumbot-"
                "specific fitting path."
            ),
            "population_improvement_policy": {
                "active_loop": "tabula_rasa_regularized_self_play_population_league",
                "current_local_incumbent_source": "autoresearch-session/poker_state.json.incumbent_checkpoint",
                "response_oracle_rule": "maintained-library response oracles are controls or reviewed successors, not the default mainline learner",
                "meta_policy_rule": "use the native empirical game to judge population support after self-play training, not as a Slumbot selector",
                "drift_guard": (
                    "Run the research-log drift guard before every queued "
                    "experiment; repeated local target-consumer/search-label "
                    "transfer failures require review, and the mainline should "
                    "return to tabula-rasa self-play instead of another detached "
                    "label-fitting variant."
                ),
                "slumbot_rule": "tiny smoke only until repeated parent/population gates pass; never use Slumbot as training data or selector",
                "required_ladder": [
                    "candidate versus parent/incumbent",
                    "candidate versus empirical-game population/meta-policy",
                    "candidate versus saved local controls such as NFSP, NPI, and Rainbow variants",
                    "complete empirical-game matrix with no missing required pairs",
                    "tiny held-out Slumbot smoke only after local promotion, then larger Slumbot confidence only after repeat progress",
                ],
                "blocked_drift": [
                    "do not repeat identical single-checkpoint response-oracle training",
                    "do not ignore a parent-gate failure from the previous response-oracle generation",
                "do not train from Slumbot hands, traces, or revealed cards",
                "do not adapt checkpoints across card/action environments for promotion; train fresh per environment and transfer only the general learning schema",
                "do not replace maintained RL learners with local PPO/Rainbow/NFSP/PSRO internals without methodology review",
                    "do not promote a checkpoint that only beats one frozen target but fails the broader population",
                ],
                "current_negative_evidence": (
                    "The second same-style Tianshou Rainbow response oracle lost "
                    "to its parent; population exposure or meta-policy training "
                    "must change before another response-oracle attempt."
                ),
            },
            "neural_policy_iteration_policy": {
                "current_status": "active_as_tabula_rasa_regularized_self_play",
                "neural_policy_role": "main_stochastic_actor",
                "cfr_role": "evaluator_or_optional_policy_improvement_control_not_default_teacher",
                "deploy_policy_rule": "sample_mixed_strategy_not_argmax_by_default",
                "generic_rl_algorithm_policy": "plug_in_maintained_libraries_only",
                "local_code_boundary": (
                    "Implement native poker environment adapters, legal masks, "
                    "checkpoint loaders, evaluators, empirical-game gates, and "
                    "batching/profiling glue locally; do not implement generic "
                    "PPO/Rainbow/NFSP/PSRO learner internals unless maintained "
                    "libraries cannot preserve the native poker contract and a "
                    "methodology review approves the exception."
                ),
                "control_gate_rule": "require_fixed_or_mixed_controls_per_generation",
                "required_loop_flag": "--require-control-gate",
                "required_control_kinds": [
                    "same-format-npi",
                    "native-nfsp",
                    "tianshou-rainbow",
                    "tianshou-ppo",
                ],
                "training_loop": [
                    "fresh stochastic neural policy/value networks play local self-play trajectories",
                    "learner receives simulator observations, legal action masks, and terminal rewards",
                    "R-NaD/NashPG/MMD-style regularized self-play update improves the network without Slumbot or human data",
                    "updated stochastic network returns to self-play and is judged by parent/population gates",
                ],
                "blocked_drift": [
                    "always run fixed-budget CFR as the entire player",
                    "treat detached solver labels as the main training objective",
                    "deploy deterministic argmax policy without a reviewed exploitability gate",
                    "use PPO/Rainbow/Gym/PettingZoo as benchmark tuning detached from poker equilibrium pressure",
                    "hand-roll generic PPO/Rainbow/NFSP/PSRO internals when Tianshou/RLCard/OpenSpiel/AgileRL can be adapted",
                ],
                "allowed_infrastructure": [
                    "Gymnasium",
                    "PettingZoo",
                    "RLCard",
                    "OpenSpiel",
                    "AgileRL",
                    "Tianshou PPO/Rainbow",
                    "maintained NFSP controls",
                ],
            },
            "self_play_league_policy": {
                "primary_role": "promotion_gate",
                "metric": "checkpoint league positive lower95 versus incumbent or previous checkpoint",
                "required_ladder": [
                    "candidate versus previous checkpoint",
                    "candidate versus current incumbent",
                    "candidate versus empirical-game population/meta-policy",
                    "candidate versus native self-play controls such as NFSP/PPO/Deep CFR",
                    "fixed-state resolver diagnostics for solver-coupled changes",
                    "held-out Slumbot validation only after internal league gates pass",
                ],
                "slumbot_role": (
                    "Held-out external validation and integration benchmark after "
                    "self-play league progress is established."
                ),
                "blocked_before_internal_pass": (
                    "Slumbot chip-rate claims, Slumbot-specific response patches, "
                    "or promotion based on sparse live API smokes."
                ),
                "progress_plot_fields": [
                    "checkpoint_iteration",
                    "opponent_or_baseline",
                    "strategy_source",
                    "n_games",
                    "seeds",
                    "mean_chips_per_hand",
                    "lower95_chips_per_hand",
                    "upper95_chips_per_hand",
                    "promotion_blockers",
                ],
            },
            "slumbot_validation_policy": {
                "max_smoke_hands_before_internal_pass": 50,
                "allowed_without_internal_pass": (
                    "Sparse live Slumbot integration smokes up to the hand cap "
                    "may check API parsing, latency, and trace fields only."
                ),
                "confidence_requires": [
                    "passed pre-Slumbot candidate promotion gate covering RLCard AlphaNLHoldem reference, native 9-action H2H, and empirical-game support",
                    "passed self-play checkpoint league with positive lower95 evidence",
                    "candidate versus incumbent or previous checkpoint evidence",
                    "native self-play controls when applicable",
                    "fixed-state resolver diagnostics for solver-coupled changes",
                ],
                "blocked_without_internal_pass": [
                    "Slumbot confidence runs",
                    "Slumbot chip-rate promotion claims",
                    "Slumbot-specific response/range/action patches",
                ],
            },
            "explicit_range_policy": {
                "default_role": "diagnostic_teacher_only",
                "allowed_roles": [
                    "belief-state falsification",
                    "counterfactual-EV replay evaluator",
                    "debugging Slumbot trace likelihood failures",
                ],
                "blocked_roles": [
                    "default live opponent model",
                    "street-specific range patch",
                    "Slumbot-only exploit prior",
                    "training data or target source",
                    "checkpoint selector",
                    "promotion target based only on revealed-hand likelihood",
                ],
                "mainline_replacement": (
                    "local self-play representation learned from public state, "
                    "private cards, actions, and rewards; any belief-like state "
                    "must arise inside the general self-play/search loop"
                ),
                "promotion_rule": (
                    "Explicit range interventions remain diagnostic unless a completed "
                    "methodology review reframes them as general locally trained "
                    "mechanisms and they pass matched local self-play gates before "
                    "held-out Slumbot evaluation."
                ),
            },
            "protected_surfaces": PROTECTED_EVAL_SURFACES,
            "protected_surface_patterns": PROTECTED_EVAL_SURFACE_PATTERNS,
            "protected_surface_rule": (
                "Evaluation harnesses, Slumbot adapters, promotion logic, seed "
                "lists, parsers, and parity tests are immutable during ordinary "
                "experiments. Changing them requires completed methodology review "
                "and benchmark-hacking audit artifacts."
            ),
            "promotion_requires": [
                "checkpoint league positive lower95 versus incumbent or previous checkpoint",
                "native self-play control comparison when applicable",
                "fixed-state resolver diagnostics",
                "held-out Slumbot validation after internal league pass",
                "objective-alignment audit",
            ],
            "visible_metric_rule": (
                "Local random, sparse Slumbot, and smoke metrics are diagnostics, "
                "not promotion targets. Do not optimize solely for visible smoke "
                "gates or Slumbot chip noise before the self-play league ladder passes."
            ),
        },
        "primary_metric": "checkpoint_league_lower_95_ci_chips_per_hand_vs_incumbent",
        "hard_stop_conditions": [
            "STOP file exists",
            "Tier 0 integrity gate fails",
            "metric output is missing or non-mechanical",
            "Slumbot credentials, network access, or API limits block evaluation",
            "cycle would overwrite an incumbent checkpoint",
            "cycle requires a hand-crafted opponent rule",
            "cycle treats Slumbot chips as primary promotion evidence before self-play league pass",
            "cycle treats hard learned value cuts or policy argmax imitation as a "
            "mainline method without a completed failure synthesis and resolver gate",
        ],
        "gates": {
            "tier0": {
                "description": "Fast integrity tests for masks, Slumbot mapping, and feature parity.",
                "timeout_seconds": 600,
                "commands": [
                    [
                        python,
                        "-m",
                        "pytest",
                        "-q",
                        "test/unit/test_network_mask.py",
                        "test/unit/test_slumbot_mapping.py",
                        "test/unit/test_legal_mask_parity.py",
                    ],
                    [
                        "/usr/bin/env",
                        "NUMBA_ENABLE_CUDASIM=1",
                        python,
                        "scripts/test_feature_encoding.py",
                    ]
                ],
            },
            "eval-local": {
                "description": "Small local fixed-seed checkpoint evaluation against random opponents.",
                "timeout_seconds": 900,
                "commands": [
                    [
                        python,
                        "scripts/poker_autoresearch_eval.py",
                        "--checkpoint",
                        "models/slumbot_2p_iter1000.pt",
                        "--n-games",
                        "32",
                        "--device",
                        "auto",
                        "--seed",
                        "20260510",
                    ]
                ],
            },
            "eval-local-confidence": {
                "description": "Larger fixed-seed checkpoint evaluation for a less noisy local baseline.",
                "timeout_seconds": 1800,
                "commands": [
                    [
                        python,
                        "scripts/poker_autoresearch_eval.py",
                        "--checkpoint",
                        "models/slumbot_2p_iter1000.pt",
                        "--n-games",
                        "1000",
                        "--device",
                        "auto",
                        "--seed",
                        "20260511",
                    ]
                ],
            },
            "eval-local-multiseed": {
                "description": "Multi-seed local checkpoint evaluation to reduce single-seed variance.",
                "timeout_seconds": 2400,
                "commands": [
                    [
                        python,
                        "scripts/poker_autoresearch_eval.py",
                        "--checkpoint",
                        "models/slumbot_2p_iter1000.pt",
                        "--n-games",
                        "1000",
                        "--device",
                        "auto",
                        "--seeds",
                        "20260511,20260512,20260513",
                    ]
                ],
            },
            "eval-incumbent-self-compare": {
                "description": (
                    "Local comparison smoke that verifies candidate-vs-incumbent "
                    "delta metrics without treating random-opponent wins as promotable."
                ),
                "timeout_seconds": 2400,
                "commands": [
                    [
                        python,
                        "scripts/poker_autoresearch_eval.py",
                        "--checkpoint",
                        "models/slumbot_2p_iter1000.pt",
                        "--baseline-checkpoint",
                        "models/slumbot_2p_iter1000.pt",
                        "--n-games",
                        "500",
                        "--device",
                        "auto",
                        "--seeds",
                        "20260511,20260512,20260513",
                    ]
                ],
            },
            "eval-head-to-head-self-compare": {
                "description": (
                    "Duplicate-swapped local model-vs-model self-compare. This "
                    "validates the stronger Tier 2 evaluation path before using "
                    "it for candidate checkpoints."
                ),
                "timeout_seconds": 2400,
                "commands": [
                    [
                        python,
                        "scripts/poker_autoresearch_eval.py",
                        "--checkpoint",
                        "models/slumbot_2p_iter1000.pt",
                        "--baseline-checkpoint",
                        "models/slumbot_2p_iter1000.pt",
                        "--head-to-head",
                        "--n-games",
                        "200",
                        "--device",
                        "auto",
                        "--seeds",
                        "20260511,20260512,20260513",
                    ]
                ],
            },
            "eval-self-play-league-smoke": {
                "description": (
                    "Checkpoint-league evaluator smoke. Duplicate-swapped "
                    "model-vs-model self-comparison validates the self-play league "
                    "path before it is used to promote candidate checkpoints."
                ),
                "timeout_seconds": 2400,
                "commands": [
                    [
                        python,
                        "scripts/poker_autoresearch_eval.py",
                        "--checkpoint",
                        "models/slumbot_2p_iter1000.pt",
                        "--baseline-checkpoint",
                        "models/slumbot_2p_iter1000.pt",
                        "--head-to-head",
                        "--n-games",
                        "200",
                        "--device",
                        "auto",
                        "--seeds",
                        "20260511,20260512,20260513",
                    ]
                ],
            },
            "native_rollout_parity_and_5x_throughput": {
                "description": (
                    "Native-style rollout substrate gate: deterministic replay "
                    "parity against full_deck/state.py plus at least 5x environment-step "
                    "throughput before learner integration."
                ),
                "timeout_seconds": 1200,
                "commands": [
                    [
                        python,
                        "scripts/eval_native_rollout_substrate.py",
                        "--n-parity-games",
                        "32",
                        "--parity-max-steps",
                        "96",
                        "--n-benchmark-games",
                        "512",
                        "--benchmark-max-steps",
                        "128",
                        "--initial-chips",
                        "1000",
                        "--seed",
                        "20260752",
                        "--min-speedup",
                        "5.0",
                        "--output-json",
                        "autoresearch-session/native_rollout_substrate/native_rollout_parity_5x.json",
                    ]
                ],
            },
            "eval-resolver-fixed-states": {
                "description": (
                    "Fixed public-state turn/river benchmark for blueprint-vs-resolver "
                    "legality, latency, action drift, and learned-advantage proxies."
                ),
                "timeout_seconds": 1200,
                "commands": [
                    [
                        python,
                        "scripts/poker_resolver_benchmark.py",
                        "--checkpoint",
                        "models/slumbot_2p_iter1000.pt",
                        "--device",
                        "auto",
                        "--solver-iterations",
                        "25",
                    ]
                ],
            },
            "eval-response-range-counterfactual-ev": {
                "description": (
                    "Diagnostic replay gate that asks whether calibrated Slumbot "
                    "response ranges improve counterfactual action EV, not just "
                    "revealed-hand likelihood or top-action agreement."
                ),
                "timeout_seconds": 1800,
                "commands": [
                    [
                        python,
                        "scripts/eval_slumbot_response_range_ev_gate.py",
                        "--checkpoint",
                        "models/autoresearch_gpu_20260515T111750Z/avg_strategy_candidate_final.pt",
                        "--strategy-source",
                        "average-policy",
                        "--cases-json",
                        "autoresearch-session/slumbot_trace_cases/20260515T134500Z-session3-resolver-cases.json",
                        "--trace",
                        "autoresearch-session/slumbot_traces/slumbot-candidate-trace-20260515T134500Z-avg-strategy-noallin-fullhist-500h-session3.jsonl",
                        "--action-likelihood",
                        "autoresearch-session/slumbot_trace_cases/20260515T125000Z-avg-strategy-noallin-fullhist-500h-action-likelihood.json",
                        "--output-json",
                        "autoresearch-session/slumbot_trace_cases/response_range_counterfactual_ev_gate.json",
                        "--max-cases",
                        "16",
                        "--solver-iterations",
                        "5",
                        "--evaluator-iterations",
                        "10",
                    ]
                ],
            },
            "slumbot-smoke": {
                "description": "Tiny live Slumbot API smoke with conservative diagnostics settings.",
                "timeout_seconds": 600,
                "commands": [
                    [
                        python,
                        "scripts/poker_autoresearch_slumbot.py",
                        "--model",
                        "models/slumbot_2p_iter1000.pt",
                        "--hands",
                        "5",
                        "--greedy",
                        "--no-allin",
                        "--no-solver",
                    ]
                ],
            },
            "slumbot-solver-smoke": {
                "description": "Small live Slumbot smoke with turn/river solver enabled.",
                "timeout_seconds": 900,
                "commands": [
                    [
                        python,
                        "scripts/poker_autoresearch_slumbot.py",
                        "--model",
                        "models/slumbot_2p_iter1000.pt",
                        "--hands",
                        "10",
                        "--greedy",
                        "--no-allin",
                    ]
                ],
            }
        },
    }


def _default_state() -> dict:
    return {
        "status": "ready",
        "created_at": _now(),
        "updated_at": _now(),
        "incumbent_checkpoint": None,
        "active_cycle": None,
        "last_metrics": {},
        "history": [],
        "hypothesis_queue": [
            {
                "id": "initial_tier0_integrity",
                "type": "experiment",
                "hypothesis": "Tier 0 integrity should pass before unattended autoresearch.",
                "failure_class": "eval_invalid",
                "gate": "tier0",
            }
        ],
    }


def init_state(root: str | Path, force: bool = False) -> dict:
    root = Path(root)
    session = _session(root)
    runs = _runs_path(root)
    reviews = _reviews_path(root)
    session.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    reviews.mkdir(parents=True, exist_ok=True)
    _review_manifest_dir(root).mkdir(parents=True, exist_ok=True)

    default_goal = _default_goal(root)
    goal_path = _goal_path(root)
    if force or not goal_path.exists():
        _write_json(goal_path, default_goal)
    else:
        goal = _read_json(goal_path)
        _write_json(goal_path, _sync_goal_defaults(goal, default_goal))

    state_path = _state_path(root)
    if force or not state_path.exists():
        _write_json(state_path, _default_state())

    knobs = _knobs_path(root)
    if force or not knobs.exists():
        knobs.write_text("\t".join(KNOB_COLUMNS) + "\n", encoding="utf-8")
    else:
        _migrate_knobs_file(knobs)

    log = _log_path(root)
    if force or not log.exists():
        log.write_text(
            "# Research Log\n\n"
            "## Poker Autoresearch Initialized\n\n"
            f"- Timestamp: {_now()}\n"
            "- Purpose: maintain an auditable trail for HEAD research cycles.\n\n",
            encoding="utf-8",
        )

    return readiness_report(root)


def readiness_report(root: str | Path) -> dict:
    root = Path(root)
    required = [
        _goal_path(root),
        _state_path(root),
        _knobs_path(root),
        _reviews_path(root),
        _runs_path(root),
        _log_path(root),
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    stop_requested = _stop_path(root).exists()
    return {
        "ready": not missing and not stop_requested,
        "missing": missing,
        "stop_requested": stop_requested,
        "stop_file": str(_stop_path(root)),
        "session_dir": str(_session(root)),
    }


def set_incumbent(root: str | Path, checkpoint: str | Path, *, reason: str) -> dict:
    root = Path(root)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Incumbent checkpoint does not exist: {checkpoint_path}")

    state = _read_json(_state_path(root))
    incumbent = {
        "checkpoint": str(checkpoint_path),
        "reason": reason,
        "set_at": _now(),
    }
    state["incumbent_checkpoint"] = incumbent
    state["updated_at"] = _now()
    _write_json(_state_path(root), state)
    return incumbent


def run_command(command: Sequence[str], timeout_seconds: float | None = None) -> CommandResult:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(command),
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return CommandResult(
            command=list(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            seconds=round(time.monotonic() - started, 3),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return CommandResult(
            command=list(command),
            returncode=124,
            stdout=stdout,
            stderr=stderr + f"\nTimed out after {timeout_seconds} seconds.",
            seconds=round(time.monotonic() - started, 3),
        )


def _command_lists(commands: Iterable[Sequence[str]]) -> list[list[str]]:
    return [list(command) for command in commands]


def _command_metric(result: CommandResult) -> dict:
    metric = asdict(result)
    stdout = result.stdout.strip()
    if stdout:
        try:
            metric["stdout_json"] = json.loads(stdout)
        except json.JSONDecodeError:
            pass
    return metric


def run_gate(
    root: str | Path,
    gate: str,
    *,
    run_dir: str | Path | None = None,
    runner: CommandRunner = run_command,
    commands: Iterable[Sequence[str]] | None = None,
) -> dict:
    root = Path(root)
    goal = _read_json(_goal_path(root))
    gate_config = goal.get("gates", {}).get(gate)
    if gate_config is None and commands is None:
        raise ValueError(f"Unknown gate: {gate}")

    timeout_seconds = None if gate_config is None else gate_config.get("timeout_seconds")
    selected_commands = _command_lists(commands or gate_config.get("commands", []))
    if not selected_commands:
        raise ValueError(f"Gate {gate} has no commands configured.")

    started_at = _now()
    results = [runner(command, timeout_seconds) for command in selected_commands]
    passed = all(result.returncode == 0 for result in results)
    metrics = {
        "gate": gate,
        "passed": passed,
        "started_at": started_at,
        "finished_at": _now(),
        "commands": [_command_metric(result) for result in results],
    }

    if run_dir is not None:
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        _write_json(run_path / "metrics.json", metrics)

    return metrics


def _slug(text: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in text)
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts[:8]) or "cycle"


def _tsv_clean(value: str) -> str:
    return str(value).replace("\t", " ").replace("\n", " ").strip()


def _looks_like_broad_sweep(*values: str) -> bool:
    text = " ".join(str(value).lower() for value in values)
    if any(
        marker in text
        for marker in (
            "grid search",
            "hyperparameter sweep",
            "parameter sweep",
            "broad sweep",
        )
    ):
        return True
    default = str(values[0]).strip()
    return any(marker in default for marker in ("[", "]", "{", "}", ",")) or ".." in default


def _unique_gate_name(root: Path, base_name: str) -> str:
    goal_path = _goal_path(root)
    if not goal_path.exists():
        return base_name
    existing = set(_read_json(goal_path).get("gates", {}))
    if base_name not in existing:
        return base_name
    suffix = 2
    while f"{base_name}-{suffix}" in existing:
        suffix += 1
    return f"{base_name}-{suffix}"


def _unique_child_dir(parent: Path, base_name: str) -> Path:
    candidate = parent / base_name
    if not candidate.exists():
        return candidate
    suffix = 2
    while (parent / f"{base_name}-{suffix}").exists():
        suffix += 1
    return parent / f"{base_name}-{suffix}"


def register_research_knob(
    root: str | Path,
    *,
    name: str,
    default: str,
    failure_class: str,
    mechanism: str,
    rationale: str,
    removal_criterion: str,
    review_dir: str | Path | None = None,
    max_active: int | None = None,
) -> dict:
    """Register one persistent research knob with a mechanism and budget."""
    root = Path(root)
    _assert_phase_allows_action(
        root,
        "new_model_size_or_search_knob",
        details=f"{name} {failure_class} {mechanism} {rationale}",
    )
    if review_dir is None:
        raise RuntimeError(
            "Persistent research knobs require a completed methodology review."
        )
    review_result = validate_methodology_review(
        _resolve_existing_path(root, review_dir, label="Methodology review")
    )
    if not review_result["passed"]:
        raise RuntimeError(
            "Persistent research knob review artifacts are incomplete: "
            + "; ".join(review_result["errors"])
        )
    fields = {
        "name": name,
        "default": default,
        "failure_class": failure_class,
        "mechanism": mechanism,
        "rationale": rationale,
        "removal_criterion": removal_criterion,
    }
    missing = [key for key, value in fields.items() if not str(value).strip()]
    if missing:
        raise ValueError(f"Missing required knob fields: {', '.join(missing)}")
    if _looks_like_broad_sweep(default, mechanism, rationale):
        raise ValueError(
            "Research knobs must define one mechanism with a single default, "
            "not a broad sweep."
        )

    goal = _read_json(_goal_path(root))
    budget = int(max_active or goal.get("knob_policy", {}).get("max_active_knobs", 5))
    knob_path = _knobs_path(root)
    _migrate_knobs_file(knob_path)
    lines = knob_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t") if lines else []
    records = [dict(zip(header, line.split("\t"), strict=False)) for line in lines[1:] if line]
    active = [record for record in records if record.get("status", "active") == "active"]
    if any(record.get("name") == name for record in active):
        raise RuntimeError(f"Active research knob already exists: {name}")
    if len(active) >= budget:
        raise RuntimeError(
            f"Cannot add {name}: active research knob budget is {budget}."
        )

    row = {
        "name": name,
        "status": "active",
        "default": default,
        "failure_class": failure_class,
        "mechanism": mechanism,
        "rationale": rationale,
        "removal_criterion": removal_criterion,
        "review_id": Path(review_dir).name,
        "created_at": _now(),
        "retired_at": "",
    }
    if not header or "status" not in header:
        header = KNOB_COLUMNS
        knob_path.write_text("\t".join(header) + "\n", encoding="utf-8")
    with knob_path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(_tsv_clean(row.get(column, "")) for column in header) + "\n")
    return row


def new_cycle(
    root: str | Path,
    *,
    hypothesis: str,
    cycle_type: str,
    failure_class: str,
    gate: str,
) -> dict:
    root = Path(root)
    state = _read_json(_state_path(root))
    if state.get("active_cycle") is not None:
        raise RuntimeError("Cannot create a new cycle while another cycle is active.")

    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{_slug(hypothesis)}"
    run_dir = _runs_path(root) / run_id
    cycle = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "hypothesis": hypothesis,
        "type": cycle_type,
        "failure_class": failure_class,
        "gate": gate,
        "started_at": _now(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "cycle.json", cycle)

    state["active_cycle"] = cycle
    state["updated_at"] = _now()
    _write_json(_state_path(root), state)
    return cycle


def enqueue_cycle(
    root: str | Path,
    *,
    hypothesis: str,
    cycle_type: str,
    failure_class: str,
    gate: str,
    postprocess: dict | None = None,
) -> dict:
    root = Path(root)
    state = _read_json(_state_path(root))
    item = {
        "id": f"{_now().replace(':', '').replace('-', '')}-{_slug(hypothesis)}",
        "type": cycle_type,
        "hypothesis": hypothesis,
        "failure_class": failure_class,
        "gate": gate,
    }
    if postprocess is not None:
        item["postprocess"] = postprocess
    state.setdefault("hypothesis_queue", []).append(item)
    state["updated_at"] = _now()
    _write_json(_state_path(root), state)
    return item


def set_research_phase(root: str | Path, *, phase: str, reason: str) -> dict:
    """Set the workflow phase used by guardrails."""
    if phase not in ALLOWED_RESEARCH_PHASES:
        raise ValueError(
            "phase must be one of: " + ", ".join(sorted(ALLOWED_RESEARCH_PHASES))
        )
    if not str(reason).strip():
        raise ValueError("reason is required")
    root = Path(root)
    goal = _read_json(_goal_path(root))
    phase_record = goal.setdefault("research_phase", {})
    phase_record["current"] = phase
    phase_record["reason"] = reason
    phase_record["set_at"] = _now()
    phase_record["allowed_phases"] = sorted(ALLOWED_RESEARCH_PHASES)
    default_phase_record = _default_goal(root)["research_phase"]
    if phase in default_phase_record:
        phase_record.setdefault(phase, default_phase_record[phase])
    _write_json(_goal_path(root), goal)
    return phase_record


def _assert_phase_allows_action(
    root: Path,
    action: str,
    *,
    details: str = "",
) -> None:
    goal_path = _goal_path(root)
    if not goal_path.exists():
        return
    goal = _read_json(goal_path)
    phase = goal.get("research_phase", {}).get("current", "open_research")
    review_actions = {
        "methodology_review",
        "mechanism_review",
        "failure_synthesis",
        "objective_audit",
    }
    if phase == "neural_regret_field_resolving":
        if action in review_actions:
            return
        if action == "new_model_size_or_search_knob":
            lowered = details.lower()
            warm_start_terms = (
                "regret",
                "warm",
                "initializer",
                "initialization",
                "public-belief",
                "public belief",
                "resolver",
            )
            if any(term in lowered for term in warm_start_terms):
                return
        if action in {
            "gpu_deep_cfr_training",
            "slumbot_smoke",
            "new_model_size_or_search_knob",
        }:
            raise RuntimeError(
                "Research phase neural_regret_field_resolving blocks this action. "
                "First implement and pass the root-disjoint warm-start resolver "
                f"gate. Blocked action: {action}."
            )
        return
    if phase == "exact_gpu_resolving":
        if action in review_actions:
            return
        if action == "eval_warm_start_resolver_gate":
            raise RuntimeError(
                "Research phase exact_gpu_resolving blocks this action. static "
                "warm-start resolver gates are retired as mainline unless a "
                "methodology review reopens them against the CUDA budget frontier. "
                f"Blocked action: {action}."
            )
        if action == "new_model_size_or_search_knob":
            lowered = details.lower()
            stale_terms = ("warm", "initializer", "mix", "policy prior", "eta")
            if any(term in lowered for term in stale_terms):
                raise RuntimeError(
                    "Research phase exact_gpu_resolving blocks this action. "
                    "Static warm-start or policy-mixing knobs must first beat "
                    f"the exact CUDA budget frontier. Blocked action: {action}."
                )
        return
    if phase != "callback_state_calibration_debug":
        return
    if action in review_actions | {"callback_state_calibration_audit"}:
        return
    if action == "new_model_size_or_search_knob":
        lowered = details.lower()
        guarded_terms = (
            "model",
            "hidden",
            "layer",
            "capacity",
            "architecture",
            "search",
            "solver",
            "slumbot",
        )
        if not any(term in lowered for term in guarded_terms):
            return
    raise RuntimeError(
        "Research phase callback_state_calibration_debug blocks this action. "
        "Run a callback-state calibration audit or synthesis before expanding "
        f"the experiment surface. Blocked action: {action}."
    )


def synthesis_status(root: str | Path) -> dict:
    """Return whether the workflow is due for failure synthesis."""
    root = Path(root)
    goal = _read_json(_goal_path(root))
    state = _read_json(_state_path(root))
    interval = int(goal.get("synthesis_policy", {}).get("experiments_per_synthesis", 5))
    history = state.get("history", [])
    last_synthesis_idx = -1
    for idx, record in enumerate(history):
        if record.get("type") == "synthesis":
            last_synthesis_idx = idx
    since = [
        record
        for record in history[last_synthesis_idx + 1 :]
        if record.get("type") != "synthesis"
        and "review" not in str(record.get("type", ""))
    ]
    return {
        "due": len(since) >= interval,
        "experiments_since_synthesis": len(since),
        "interval": interval,
        "required_fields": goal.get("synthesis_policy", {}).get("required_fields", []),
        "last_synthesis_run_id": (
            history[last_synthesis_idx].get("run_id") if last_synthesis_idx >= 0 else None
        ),
    }


def enqueue_failure_synthesis(
    root: str | Path,
    *,
    subject: str,
    timeout_seconds: int = 600,
) -> dict:
    """Queue a synthesis review that compresses recent failures into a causal model."""
    root = Path(root)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    review_dir = _unique_child_dir(
        _reviews_path(root),
        f"{timestamp}-{_slug(subject)}-synthesis",
    )
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "synthesis.md").write_text(
        "# Failure Synthesis\n\n"
        f"Subject: {subject}\n\n"
        "Required fields:\n"
        "- Current causal model: TODO\n"
        "- Retired hypotheses: TODO\n"
        "- Live hypotheses: TODO\n"
        "- Single next test: TODO\n\n"
        "Verdict: PENDING\n",
        encoding="utf-8",
    )
    _write_json(
        review_dir / "decision.json",
        {
            "decision": "pending",
            "reason": "PENDING",
            "sources": [],
        },
    )

    gate_name = _unique_gate_name(
        root,
        f"failure-synthesis-{timestamp}-{_slug(subject)}",
    )
    python = _project_python(root)
    command = [
        python,
        "scripts/poker_synthesis_review.py",
        "--synthesis-dir",
        str(review_dir),
        "--require-complete",
    ]
    goal = _read_json(_goal_path(root))
    goal.setdefault("gates", {})[gate_name] = {
        "description": (
            "Validate that recent failures were compressed into a causal model "
            "before adding another experiment family."
        ),
        "timeout_seconds": timeout_seconds,
        "commands": [command],
    }
    _write_json(_goal_path(root), goal)
    item = enqueue_cycle(
        root,
        hypothesis=(
            f"Failure synthesis for {subject} should identify the causal model "
            "and one next falsifier before further expansion."
        ),
        cycle_type="synthesis",
        failure_class="eval_invalid",
        gate=gate_name,
    )
    item["synthesis_dir"] = str(review_dir)
    state = _read_json(_state_path(root))
    state["hypothesis_queue"][-1]["synthesis_dir"] = str(review_dir)
    state["updated_at"] = _now()
    _write_json(_state_path(root), state)
    return item


def _write_paradigm_innovation_templates(
    innovation_dir: Path,
    *,
    subject: str,
    anomaly: str,
) -> None:
    innovation_dir.mkdir(parents=True, exist_ok=True)
    (innovation_dir / "innovation.md").write_text(
        "# Paradigm Innovation Review\n\n"
        f"Subject: {subject}\n\n"
        f"Anomaly ledger: {anomaly or 'TODO'}\n"
        "Current-practice limit: TODO\n"
        "First-principles reduction: TODO\n"
        "Cross-paradigm analogy: TODO\n"
        "Novel mechanism: TODO\n"
        "Bitter-lesson alignment: TODO\n"
        "Smallest decisive test: TODO\n"
        "Falsifier: TODO\n\n"
        "Verdict: PENDING\n",
        encoding="utf-8",
    )
    (innovation_dir / "thought_experiments.md").write_text(
        "# Thought Experiments\n\n"
        "Mechanism stress test: TODO\n"
        "Failure thought experiment: TODO\n"
        "Transfer thought experiment: TODO\n"
        "Compute thought experiment: TODO\n",
        encoding="utf-8",
    )
    (innovation_dir / "related_work.md").write_text(
        "# Related Work\n\n"
        "Title: TODO\n"
        "Source URL: TODO\n"
        "Source type: TODO primary / secondary / no suitable primary source found\n"
        "Transfers to this codebase: TODO\n"
        "Does not transfer: TODO\n"
        "Novelty delta: TODO\n"
        "Smallest local test: TODO\n",
        encoding="utf-8",
    )
    _write_json(
        innovation_dir / "decision.json",
        {
            "schema_version": 1,
            "decision": "pending",
            "reason": "PENDING",
            "sources": [],
        },
    )


def enqueue_paradigm_innovation_review(
    root: str | Path,
    *,
    subject: str,
    anomaly: str,
    timeout_seconds: int = 600,
) -> dict:
    """Queue a novelty review before expanding a repeatedly failed paradigm."""
    root = Path(root)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    innovation_dir = _unique_child_dir(
        _reviews_path(root),
        f"{timestamp}-{_slug(subject)}-innovation",
    )
    _write_paradigm_innovation_templates(
        innovation_dir,
        subject=subject,
        anomaly=anomaly,
    )

    gate_name = _unique_gate_name(
        root,
        f"paradigm-innovation-{timestamp}-{_slug(subject)}",
    )
    python = _project_python(root)
    command = [
        python,
        "scripts/poker_innovation_review.py",
        "--innovation-dir",
        str(innovation_dir),
        "--require-complete",
    ]
    goal = _read_json(_goal_path(root))
    goal.setdefault("gates", {})[gate_name] = {
        "description": (
            "Validate novelty research, first-principles reduction, thought "
            "experiments, and related work before another paradigm expansion."
        ),
        "timeout_seconds": timeout_seconds,
        "commands": [command],
    }
    _write_json(_goal_path(root), goal)
    item = enqueue_cycle(
        root,
        hypothesis=(
            f"Paradigm innovation review for {subject} should convert the "
            "repeated anomaly into one falsifiable mechanism-level sprint."
        ),
        cycle_type="innovation_review",
        failure_class="research_direction",
        gate=gate_name,
    )
    item["innovation_dir"] = str(innovation_dir)
    item["requires_related_work"] = True
    item["requires_first_principles"] = True
    item["requires_thought_experiments"] = True
    state = _read_json(_state_path(root))
    state["hypothesis_queue"][-1]["innovation_dir"] = str(innovation_dir)
    state["hypothesis_queue"][-1]["requires_related_work"] = True
    state["hypothesis_queue"][-1]["requires_first_principles"] = True
    state["hypothesis_queue"][-1]["requires_thought_experiments"] = True
    state["updated_at"] = _now()
    _write_json(_state_path(root), state)
    return item


def _write_methodology_review_templates(
    review_dir: Path,
    *,
    subject: str,
    trigger: str,
    claim: str,
) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "review.md").write_text(
        "# Independent Verification\n\n"
        f"Subject: {subject}\n\n"
        f"Trigger: {trigger}\n\n"
        f"Claim under review: {claim}\n\n"
        "Required process:\n"
        "- Invoke the independent verifier before accepting the claim.\n"
        "- Inspect source files and artifacts directly.\n"
        "- Separate code bugs, measurement flaws, methodology flaws, and reporting flaws.\n\n"
        "Verdict: PENDING\n\n"
        "Findings:\n"
        "- TODO\n",
        encoding="utf-8",
    )
    (review_dir / "related_work.md").write_text(
        "# Related Work\n\n"
        f"Diagnostic question: {claim}\n\n"
        "Title: TODO\n"
        "Source URL: TODO\n"
        "Source type: TODO primary / secondary / no suitable primary source found\n"
        "Transfers to this codebase: TODO\n"
        "Does not transfer: TODO\n"
        "Smallest local test: TODO\n",
        encoding="utf-8",
    )
    (review_dir / "benchmark_audit.md").write_text(
        "# Benchmark-Hacking Audit\n\n"
        f"Claim under review: {claim}\n\n"
        "Team role: benchmark-hacking auditor.\n\n"
        "Protected surfaces:\n"
        "- Evaluation harnesses, Slumbot adapters, promotion logic, seed lists, "
        "parsers, and parity tests.\n\n"
        "Required checks:\n"
        "- TODO: list changed files and identify protected-surface changes.\n"
        "- TODO: confirm no tests, parsers, legal masks, opponent adapters, or "
        "promotion blockers were weakened to improve the metric.\n"
        "- TODO: identify which metrics are diagnostic and which are promotion gates.\n"
        "- TODO: state why the result should transfer to Slumbot instead of only "
        "the visible local benchmark.\n\n"
        "Verdict: PENDING\n",
        encoding="utf-8",
    )
    (review_dir / "mechanism_review.md").write_text(
        "# Mechanism Review\n\n"
        f"Claim under review: {claim}\n\n"
        "Required fields:\n"
        "- Learned object: TODO\n"
        "- Search boundary: TODO\n"
        "- Train distribution: TODO\n"
        "- Eval distribution: TODO\n"
        "- Falsifier: TODO\n"
        "- Pass action: TODO\n"
        "- Fail action: TODO\n"
        "- Related-work delta: TODO\n\n"
        "Verdict: PENDING\n",
        encoding="utf-8",
    )
    (review_dir / "team_review.md").write_text(
        "# Review Team Routing\n\n"
        "- Research lead: owns `decision.json` and final go/no-go.\n"
        "- Independent verifier: fills `review.md` from local artifacts.\n"
        "- Literature scout: fills `related_work.md` from primary sources.\n"
        "- Benchmark auditor: fills `benchmark_audit.md` and checks objective drift.\n\n"
        "Use separate sub-agents for these roles when available. The files are "
        "the source of truth, not chat memory.\n",
        encoding="utf-8",
    )
    _write_json(
        review_dir / "review_scope.json",
        {
            "changed_paths": [],
            "protected_hits": [],
            "mechanism": "PENDING",
            "mechanism_brief": {
                "decision_object": "PENDING",
                "where_consumed": "PENDING",
                "matched_control": "PENDING",
                "primary_decision_gate": "PENDING",
                "retirement_criterion": "PENDING",
                "flexibility_boundary": "PENDING",
                "anti_benchmark_hack": "PENDING",
                "neural_policy_role": "PENDING",
                "cfr_role": "PENDING",
                "stochastic_policy_contract": "PENDING",
            },
            "expected_gate": "PENDING",
            "decision_impact": "PENDING",
            "decision_impact_gate": "PENDING",
            "fallback_if_no_decision_impact": "PENDING",
            "pass_action": "PENDING",
            "fail_action": "PENDING",
        },
    )
    _write_json(
        review_dir / "decision.json",
        {
            "schema_version": 2,
            "decision": "pending",
            "reason": "PENDING",
            "sources": [],
        },
    )


def enqueue_methodology_review(
    root: str | Path,
    *,
    subject: str,
    trigger: str,
    claim: str,
    timeout_seconds: int = 600,
) -> dict:
    """Create a methodology review bundle and queue its completion gate."""
    root = Path(root)
    _assert_phase_allows_action(root, "methodology_review")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    review_dir = _unique_child_dir(
        _reviews_path(root),
        f"{timestamp}-{_slug(subject)}",
    )
    _write_methodology_review_templates(
        review_dir,
        subject=subject,
        trigger=trigger,
        claim=claim,
    )

    gate_name = _unique_gate_name(
        root,
        f"methodology-review-{timestamp}-{_slug(subject)}",
    )
    python = _project_python(root)
    command = [
        python,
        "scripts/poker_methodology_review.py",
        "--review-dir",
        str(review_dir),
        "--require-complete",
    ]
    goal = _read_json(_goal_path(root))
    goal.setdefault("gates", {})[gate_name] = {
        "description": (
            "Validate that independent verification and related-work review "
            "artifacts are complete before acting on a methodology decision."
        ),
        "timeout_seconds": timeout_seconds,
        "commands": [command],
    }
    _write_json(_goal_path(root), goal)
    item = enqueue_cycle(
        root,
        hypothesis=(
            f"Methodology review for {subject} should verify the claim and "
            "include related work before the next research action."
        ),
        cycle_type="methodology_review",
        failure_class="eval_invalid",
        gate=gate_name,
    )
    item["review_dir"] = str(review_dir)
    item["requires_independent_verifier"] = True
    item["requires_related_work"] = True
    item["requires_benchmark_audit"] = True
    item["requires_mechanism_review"] = True
    state = _read_json(_state_path(root))
    state["hypothesis_queue"][-1]["review_dir"] = str(review_dir)
    state["hypothesis_queue"][-1]["requires_independent_verifier"] = True
    state["hypothesis_queue"][-1]["requires_related_work"] = True
    state["hypothesis_queue"][-1]["requires_benchmark_audit"] = True
    state["hypothesis_queue"][-1]["requires_mechanism_review"] = True
    state["updated_at"] = _now()
    _write_json(_state_path(root), state)
    return item


def _review_file_digest(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": path.name,
        "sha256": sha256(data).hexdigest(),
        "bytes": len(data),
    }


def write_review_manifest(root: str | Path, review_dir: str | Path) -> dict:
    """Write a tracked digest manifest for an ignored review bundle."""
    root = Path(root)
    review_dir = Path(review_dir)
    if not review_dir.is_absolute():
        review_dir = root / review_dir
    if not review_dir.exists():
        raise FileNotFoundError(f"Review directory does not exist: {review_dir}")
    manifest_dir = _review_manifest_dir(root)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    files = [
        path
        for path in sorted(review_dir.iterdir())
        if path.is_file() and path.name.endswith((".md", ".json"))
    ]
    decision_path = review_dir / "decision.json"
    decision = _read_json(decision_path) if decision_path.exists() else {}
    manifest = {
        "review_id": review_dir.name,
        "review_dir": str(review_dir.relative_to(root)) if review_dir.is_relative_to(root) else str(review_dir),
        "written_at": _now(),
        "decision": decision.get("decision"),
        "reason": decision.get("reason"),
        "sources": decision.get("sources", []),
        "files": [_review_file_digest(path) for path in files],
    }
    manifest_path = manifest_dir / f"{review_dir.name}.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path.relative_to(root))
    return manifest


def validate_methodology_review(review_dir: str | Path) -> dict:
    """Validate independent-verifier, related-work, and decision artifacts."""
    review_dir = Path(review_dir)
    errors: list[str] = []
    review_path = review_dir / "review.md"
    related_path = review_dir / "related_work.md"
    benchmark_path = review_dir / "benchmark_audit.md"
    mechanism_path = review_dir / "mechanism_review.md"
    decision_path = review_dir / "decision.json"
    scope_path = review_dir / "review_scope.json"
    for path in (
        review_path,
        related_path,
        benchmark_path,
        mechanism_path,
        scope_path,
        decision_path,
    ):
        if not path.exists():
            errors.append(f"Missing required artifact: {path.name}")

    review_text = review_path.read_text(encoding="utf-8") if review_path.exists() else ""
    if "PENDING" in review_text or "TODO" in review_text:
        errors.append("review.md is still pending")
    if "Verdict:" not in review_text:
        errors.append("review.md must include a Verdict line")

    related_text = related_path.read_text(encoding="utf-8") if related_path.exists() else ""
    if "PENDING" in related_text or "TODO" in related_text:
        errors.append("related_work.md is still pending")
    if "http://" not in related_text and "https://" not in related_text:
        errors.append("related_work.md must cite at least one source URL")
    for required in (
        "Source type:",
        "Transfers to this codebase:",
        "Does not transfer:",
        "Smallest local test:",
    ):
        if required not in related_text:
            errors.append(f"related_work.md must include {required}")
    lowered_related = related_text.lower()
    if "source type: primary" not in lowered_related and "no suitable primary source found" not in lowered_related:
        errors.append(
            "related_work.md must identify a primary source or explain that no suitable primary source was found"
        )

    benchmark_text = benchmark_path.read_text(encoding="utf-8") if benchmark_path.exists() else ""
    if "PENDING" in benchmark_text or "TODO" in benchmark_text:
        errors.append("benchmark_audit.md is still pending")
    if "Verdict:" not in benchmark_text:
        errors.append("benchmark_audit.md must include a Verdict line")

    mechanism_text = mechanism_path.read_text(encoding="utf-8") if mechanism_path.exists() else ""
    if "PENDING" in mechanism_text or "TODO" in mechanism_text:
        errors.append("mechanism_review.md is still pending")
    for required in (
        "Learned object:",
        "Search boundary:",
        "Train distribution:",
        "Eval distribution:",
        "Falsifier:",
        "Pass action:",
        "Fail action:",
        "Related-work delta:",
    ):
        if required not in mechanism_text:
            errors.append(f"mechanism_review.md must include {required}")
    if "Verdict:" not in mechanism_text:
        errors.append("mechanism_review.md must include a Verdict line")

    decision: dict = {}
    if scope_path.exists():
        try:
            scope = _read_json(scope_path)
            mechanism_brief = scope.get("mechanism_brief")
            if not isinstance(mechanism_brief, dict):
                errors.append("review_scope.json must include mechanism_brief object")
                mechanism_brief = {}
            else:
                for required in (
                    "decision_object",
                    "where_consumed",
                    "matched_control",
                    "primary_decision_gate",
                    "retirement_criterion",
                    "flexibility_boundary",
                    "anti_benchmark_hack",
                    "neural_policy_role",
                    "cfr_role",
                    "stochastic_policy_contract",
                ):
                    value = mechanism_brief.get(required)
                    if value is None:
                        errors.append(
                            f"review_scope.json mechanism_brief must include {required}"
                        )
                    elif isinstance(value, str) and value == "PENDING":
                        errors.append(
                            f"review_scope.json mechanism_brief {required} is still pending"
                        )
            for required in (
                "changed_paths",
                "protected_hits",
                "mechanism",
                "expected_gate",
                "decision_impact",
                "decision_impact_gate",
                "fallback_if_no_decision_impact",
                "pass_action",
                "fail_action",
            ):
                if required not in scope:
                    errors.append(f"review_scope.json must include {required}")
                elif isinstance(scope[required], str) and scope[required] == "PENDING":
                    errors.append(f"review_scope.json {required} is still pending")
            for required in (
                "decision_impact",
                "decision_impact_gate",
                "fallback_if_no_decision_impact",
            ):
                value = str(scope.get(required, "")).lower()
                if required == "decision_impact" and value:
                    if not any(
                        token in value
                        for token in (
                            "root decision",
                            "root action",
                            "self-play",
                            "checkpoint league",
                            "decision quality",
                            "decisions per millisecond",
                        )
                    ):
                        errors.append(
                            "review_scope.json decision_impact must connect to "
                            "root decisions or self-play league strength"
                        )
        except json.JSONDecodeError as exc:
            errors.append(f"review_scope.json is invalid JSON: {exc}")
    if decision_path.exists():
        try:
            decision = _read_json(decision_path)
        except json.JSONDecodeError as exc:
            errors.append(f"decision.json is invalid JSON: {exc}")
    if decision:
        if decision.get("decision") not in ALLOWED_REVIEW_DECISIONS:
            errors.append(
                "decision.json decision must be one of: "
                + ", ".join(sorted(ALLOWED_REVIEW_DECISIONS))
            )
        if not str(decision.get("reason", "")).strip() or decision.get("reason") == "PENDING":
            errors.append("decision.json must include a non-pending reason")
        if not decision.get("sources"):
            errors.append("decision.json must include at least one source")

    return {
        "passed": not errors,
        "review_dir": str(review_dir),
        "errors": errors,
        "decision": decision.get("decision"),
    }


def validate_failure_synthesis(synthesis_dir: str | Path) -> dict:
    """Validate a failure-synthesis bundle."""
    synthesis_dir = Path(synthesis_dir)
    errors: list[str] = []
    synthesis_path = synthesis_dir / "synthesis.md"
    decision_path = synthesis_dir / "decision.json"
    if not synthesis_path.exists():
        errors.append("Missing required artifact: synthesis.md")
    if not decision_path.exists():
        errors.append("Missing required artifact: decision.json")
    text = synthesis_path.read_text(encoding="utf-8") if synthesis_path.exists() else ""
    if "PENDING" in text or "TODO" in text:
        errors.append("synthesis.md is still pending")
    for required in (
        "Current causal model:",
        "Retired hypotheses:",
        "Live hypotheses:",
        "Single next test:",
    ):
        if required not in text:
            errors.append(f"synthesis.md must include {required}")
    if "Verdict:" not in text:
        errors.append("synthesis.md must include a Verdict line")

    decision: dict = {}
    if decision_path.exists():
        try:
            decision = _read_json(decision_path)
        except json.JSONDecodeError as exc:
            errors.append(f"decision.json is invalid JSON: {exc}")
    if decision:
        if decision.get("decision") not in ALLOWED_REVIEW_DECISIONS:
            errors.append(
                "decision.json decision must be one of: "
                + ", ".join(sorted(ALLOWED_REVIEW_DECISIONS))
            )
        if not str(decision.get("reason", "")).strip() or decision.get("reason") == "PENDING":
            errors.append("decision.json must include a non-pending reason")
    return {
        "passed": not errors,
        "synthesis_dir": str(synthesis_dir),
        "errors": errors,
        "decision": decision.get("decision"),
    }


def validate_paradigm_innovation_review(innovation_dir: str | Path) -> dict:
    """Validate a paradigm-innovation review bundle."""
    innovation_dir = Path(innovation_dir)
    errors: list[str] = []
    innovation_path = innovation_dir / "innovation.md"
    thought_path = innovation_dir / "thought_experiments.md"
    related_path = innovation_dir / "related_work.md"
    decision_path = innovation_dir / "decision.json"
    for path in (innovation_path, thought_path, related_path, decision_path):
        if not path.exists():
            errors.append(f"Missing required artifact: {path.name}")

    innovation_text = innovation_path.read_text(encoding="utf-8") if innovation_path.exists() else ""
    if "PENDING" in innovation_text or "TODO" in innovation_text:
        errors.append("innovation.md is still pending")
    for required in (
        "Anomaly ledger:",
        "Current-practice limit:",
        "First-principles reduction:",
        "Cross-paradigm analogy:",
        "Novel mechanism:",
        "Bitter-lesson alignment:",
        "Smallest decisive test:",
        "Falsifier:",
    ):
        if required not in innovation_text:
            errors.append(f"innovation.md must include {required}")
    if "Verdict:" not in innovation_text:
        errors.append("innovation.md must include a Verdict line")

    thought_text = thought_path.read_text(encoding="utf-8") if thought_path.exists() else ""
    if "PENDING" in thought_text or "TODO" in thought_text:
        errors.append("thought_experiments.md is still pending")
    for required in (
        "Mechanism stress test:",
        "Failure thought experiment:",
        "Transfer thought experiment:",
        "Compute thought experiment:",
    ):
        if required not in thought_text:
            errors.append(f"thought_experiments.md must include {required}")

    related_text = related_path.read_text(encoding="utf-8") if related_path.exists() else ""
    if "PENDING" in related_text or "TODO" in related_text:
        errors.append("related_work.md is still pending")
    if "http://" not in related_text and "https://" not in related_text:
        errors.append("related_work.md must cite at least one source URL")
    for required in (
        "Source type:",
        "Transfers to this codebase:",
        "Does not transfer:",
        "Novelty delta:",
        "Smallest local test:",
    ):
        if required not in related_text:
            errors.append(f"related_work.md must include {required}")
    lowered_related = related_text.lower()
    if "source type: primary" not in lowered_related and "no suitable primary source found" not in lowered_related:
        errors.append(
            "related_work.md must identify a primary source or explain that no suitable primary source was found"
        )

    decision: dict = {}
    if decision_path.exists():
        try:
            decision = _read_json(decision_path)
        except json.JSONDecodeError as exc:
            errors.append(f"decision.json is invalid JSON: {exc}")
    if decision:
        if decision.get("decision") not in ALLOWED_REVIEW_DECISIONS:
            errors.append(
                "decision.json decision must be one of: "
                + ", ".join(sorted(ALLOWED_REVIEW_DECISIONS))
            )
        if not str(decision.get("reason", "")).strip() or decision.get("reason") == "PENDING":
            errors.append("decision.json must include a non-pending reason")
        if not decision.get("sources"):
            errors.append("decision.json must include at least one source")
    return {
        "passed": not errors,
        "innovation_dir": str(innovation_dir),
        "errors": errors,
        "decision": decision.get("decision"),
    }


def git_changed_paths(root: str | Path, base_ref: str | None = None) -> list[str]:
    """Return repo-relative changed paths from unstaged, staged, and optional base diff."""
    root = Path(root)
    commands = [
        ["git", "diff", "--name-only"],
        ["git", "diff", "--cached", "--name-only"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ]
    if base_ref:
        commands.append(["git", "diff", "--name-only", base_ref, "--"])

    paths: set[str] = set()
    for command in commands:
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"git command failed: {command}")
        paths.update(path.strip() for path in result.stdout.splitlines() if path.strip())
    return sorted(paths)


def _is_protected_path(
    path: str,
    protected_surfaces: Iterable[str],
    protected_patterns: Iterable[str] | None = None,
) -> bool:
    normalized = path.replace("\\", "/").lstrip("./")
    for protected in protected_surfaces:
        prefix = protected.replace("\\", "/").rstrip("/")
        if normalized == prefix or normalized.startswith(f"{prefix}/"):
            return True
    for pattern in protected_patterns or ():
        normalized_pattern = pattern.replace("\\", "/").lstrip("./")
        if fnmatch(normalized, normalized_pattern):
            return True
    return False


def _review_manifest_path(root: Path, review_dir: str | Path) -> Path:
    review_path = Path(review_dir)
    review_id = review_path.name
    return _review_manifest_dir(root) / f"{review_id}.json"


def _validate_review_manifest(root: Path, review_dir: str | Path) -> dict:
    manifest_path = _review_manifest_path(root, review_dir)
    if not manifest_path.exists():
        return {
            "passed": False,
            "manifest_path": str(manifest_path.relative_to(root)),
            "errors": ["Missing tracked review manifest."],
        }
    try:
        manifest = _read_json(manifest_path)
    except json.JSONDecodeError as exc:
        return {
            "passed": False,
            "manifest_path": str(manifest_path.relative_to(root)),
            "errors": [f"Review manifest is invalid JSON: {exc}"],
        }
    decision = manifest.get("decision")
    errors = []
    if decision not in ALLOWED_REVIEW_DECISIONS - {"abandon"}:
        errors.append("Tracked review manifest decision is not usable for protected changes.")
    return {
        "passed": not errors,
        "manifest_path": str(manifest_path.relative_to(root)),
        "errors": errors,
        "decision": decision,
    }


def _validate_review_scope(
    root: Path,
    review_dir: str | Path,
    changed_paths: Sequence[str],
    protected_hits: Sequence[str],
) -> dict:
    review_path = Path(review_dir)
    if not review_path.is_absolute():
        review_path = root / review_path
    scope_path = review_path / "review_scope.json"
    if not scope_path.exists():
        return {
            "passed": False,
            "scope_path": str(scope_path),
            "errors": ["Missing review_scope.json for protected-surface audit."],
        }
    try:
        scope = _read_json(scope_path)
    except json.JSONDecodeError as exc:
        return {
            "passed": False,
            "scope_path": str(scope_path),
            "errors": [f"review_scope.json is invalid JSON: {exc}"],
        }
    scoped = {
        str(path).replace("\\", "/").lstrip("./")
        for path in scope.get("changed_paths", [])
    }
    scoped_protected = {
        str(path).replace("\\", "/").lstrip("./")
        for path in scope.get("protected_hits", [])
    }
    missing_changed = [path for path in changed_paths if path not in scoped]
    missing_protected = [path for path in protected_hits if path not in scoped_protected]
    errors = []
    if missing_changed:
        errors.append(
            "review_scope.json does not cover changed paths: "
            + ", ".join(missing_changed)
        )
    if missing_protected:
        errors.append(
            "review_scope.json does not cover protected hits: "
            + ", ".join(missing_protected)
        )
    return {
        "passed": not errors,
        "scope_path": str(scope_path),
        "errors": errors,
        "changed_paths": sorted(scoped),
        "protected_hits": sorted(scoped_protected),
    }


def audit_objective_alignment(
    root: str | Path,
    *,
    changed_paths: Iterable[str],
    review_dir: str | Path | None = None,
) -> dict:
    """Reject objective drift unless protected-surface changes are reviewed."""
    root = Path(root)
    goal = _read_json(_goal_path(root))
    policy = goal.get("objective_alignment_policy", {})
    default_review_policy = _default_goal(root).get("review_policy", {})
    review_policy = dict(goal.get("review_policy", {}))
    for hard_required in ("requires_review_manifest", "requires_review_scope"):
        review_policy[hard_required] = bool(review_policy.get(hard_required)) or bool(
            default_review_policy.get(hard_required)
        )
    protected_surfaces = _append_missing(
        list(policy.get("protected_surfaces", [])),
        PROTECTED_EVAL_SURFACES,
    )
    protected_patterns = _append_missing(
        list(policy.get("protected_surface_patterns", [])),
        PROTECTED_EVAL_SURFACE_PATTERNS,
    )
    changed = sorted({str(path).replace("\\", "/").lstrip("./") for path in changed_paths})
    protected_hits = [
        path
        for path in changed
        if _is_protected_path(path, protected_surfaces, protected_patterns)
    ]
    errors: list[str] = []
    review_result: dict | None = None
    manifest_result: dict | None = None
    scope_result: dict | None = None

    if protected_hits:
        if review_dir is None:
            errors.append(
                "Protected evaluation surfaces changed without a completed "
                "methodology review."
            )
        else:
            review_result = validate_methodology_review(review_dir)
            if not review_result["passed"]:
                errors.append(
                    "Protected evaluation surfaces changed but review artifacts "
                    "are incomplete."
                )
                errors.extend(review_result["errors"])
            if review_policy.get("requires_review_manifest", False):
                manifest_result = _validate_review_manifest(root, review_dir)
                if not manifest_result["passed"]:
                    errors.append(
                        "Protected evaluation surfaces changed without a tracked review manifest."
                    )
                    errors.extend(manifest_result["errors"])
            if review_policy.get("requires_review_scope", False):
                scope_result = _validate_review_scope(
                    root,
                    review_dir,
                    changed,
                    protected_hits,
                )
                if not scope_result["passed"]:
                    errors.extend(scope_result["errors"])

    return {
        "passed": not errors,
        "changed_paths": changed,
        "protected_hits": protected_hits,
        "errors": errors,
        "review": review_result,
        "review_manifest": manifest_result,
        "review_scope": scope_result,
    }


def _resolve_existing_path(root: Path, path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    resolved = candidate if candidate.is_absolute() else root / candidate
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def enqueue_gpu_training(
    root: str | Path,
    *,
    n_iterations: int = 10,
    n_traversals: int = 1000,
    n_training_steps: int = 1000,
    buffer_capacity: int = 2_000_000,
    hidden_dim: int = 512,
    n_layers: int = 4,
    batch_size: int = 4096,
    traversal_pool_max_slots: int = 1_000_000,
    traversal_slots_per_traversal: int = 7000,
    use_frontier_indexing: bool = False,
    policy_slots_per_traversal: int = 64,
    max_pool_exhausted_per_traversal: float | None = None,
    max_overflow_chunk_fraction: float | None = None,
    max_rejected_traversal_chunks: int | None = None,
    min_traversals_per_second: float | None = None,
    average_strategy_weight: float = 0.0,
    average_strategy_memory_capacity: int = 0,
    average_strategy_batch_size: int = 0,
    average_strategy_targets: str | Path | None = None,
    search_targets: str | Path | None = None,
    search_target_weight: float = 0.0,
    search_target_batch_size: int = 0,
    save_dir: str | Path | None = None,
    prefix: str = "candidate",
    save_every: int = 0,
    save_replay_buffers: bool = False,
    require_replay_buffer_resume: bool = False,
    resume: str | Path | None = None,
    eval_games: int = 0,
    auto_compare: bool = False,
    compare_n_games: int = 500,
    compare_seeds: str = "20260511,20260512,20260513",
    compare_device: str = "auto",
    compare_timeout_seconds: int = 2400,
    compare_head_to_head: bool = True,
    compare_strategy_source: str = "regret",
    compare_candidate_strategy_source: str | None = None,
    compare_baseline_strategy_source: str | None = None,
    seed: int | None = None,
    timeout_seconds: int = 7200,
) -> dict:
    """Create and queue a GPU Deep CFR candidate-training gate."""
    root = Path(root)
    _assert_phase_allows_action(root, "gpu_deep_cfr_training")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if save_dir is None:
        save_dir = root / "models" / f"autoresearch_gpu_{timestamp}"
    else:
        save_dir = Path(save_dir)
        if not save_dir.is_absolute():
            save_dir = root / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        resume = _resolve_existing_path(root, resume, label="Resume checkpoint")
    if search_targets:
        search_targets = _resolve_existing_path(root, search_targets, label="Search targets")
    if average_strategy_targets:
        average_strategy_targets = _resolve_existing_path(
            root,
            average_strategy_targets,
            label="Average-strategy targets",
        )

    gate_name = _unique_gate_name(
        root,
        f"train-gpu-deep-cfr-{timestamp}-{_slug(prefix)}",
    )
    python = _project_python(root)
    command = [
        python,
        "scripts/poker_autoresearch_train.py",
        "--n-iterations",
        str(n_iterations),
        "--n-traversals",
        str(n_traversals),
        "--n-training-steps",
        str(n_training_steps),
        "--buffer-capacity",
        str(buffer_capacity),
        "--hidden-dim",
        str(hidden_dim),
        "--n-layers",
        str(n_layers),
        "--batch-size",
        str(batch_size),
        "--traversal-pool-max-slots",
        str(traversal_pool_max_slots),
        "--traversal-slots-per-traversal",
        str(traversal_slots_per_traversal),
        "--policy-slots-per-traversal",
        str(policy_slots_per_traversal),
        "--average-strategy-weight",
        str(average_strategy_weight),
        "--average-strategy-memory-capacity",
        str(average_strategy_memory_capacity),
        "--average-strategy-batch-size",
        str(average_strategy_batch_size),
        "--search-target-weight",
        str(search_target_weight),
        "--search-target-batch-size",
        str(search_target_batch_size),
        "--save-dir",
        str(save_dir),
        "--prefix",
        prefix,
        "--save-every",
        str(save_every),
        "--eval-games",
        str(eval_games),
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    if use_frontier_indexing:
        command.append("--use-frontier-indexing")
    if max_pool_exhausted_per_traversal is not None:
        command.extend([
            "--max-pool-exhausted-per-traversal",
            str(max_pool_exhausted_per_traversal),
        ])
    if max_overflow_chunk_fraction is not None:
        command.extend([
            "--max-overflow-chunk-fraction",
            str(max_overflow_chunk_fraction),
        ])
    if max_rejected_traversal_chunks is not None:
        command.extend([
            "--max-rejected-traversal-chunks",
            str(max_rejected_traversal_chunks),
        ])
    if min_traversals_per_second is not None:
        command.extend([
            "--min-traversals-per-second",
            str(min_traversals_per_second),
        ])
    if resume:
        command.extend(["--resume", str(resume)])
    if save_replay_buffers:
        command.append("--save-replay-buffers")
    if require_replay_buffer_resume:
        command.append("--require-replay-buffer-resume")
    if search_targets:
        command.extend(["--search-targets", str(search_targets)])
    if average_strategy_targets:
        command.extend(["--average-strategy-targets", str(average_strategy_targets)])

    goal = _read_json(_goal_path(root))
    goal.setdefault("gates", {})[gate_name] = {
        "description": (
            "One-off GPU Deep CFR training job that emits candidate checkpoint "
            "path and throughput metrics as JSON."
        ),
        "timeout_seconds": timeout_seconds,
        "commands": [command],
        "search_target_weight": float(search_target_weight),
        "search_targets": str(search_targets) if search_targets else "",
        "average_strategy_weight": float(average_strategy_weight),
        "average_strategy_targets": (
            str(average_strategy_targets) if average_strategy_targets else ""
        ),
        "save_replay_buffers": bool(save_replay_buffers),
        "require_replay_buffer_resume": bool(require_replay_buffer_resume),
    }
    _write_json(_goal_path(root), goal)
    postprocess = None
    if auto_compare:
        postprocess = {
            "type": "compare_training_checkpoints",
            "n_games": int(compare_n_games),
            "seeds": compare_seeds,
            "device": compare_device,
            "timeout_seconds": int(compare_timeout_seconds),
            "head_to_head": bool(compare_head_to_head),
            "strategy_source": compare_strategy_source,
            "candidate_strategy_source": (
                compare_candidate_strategy_source or compare_strategy_source
            ),
            "baseline_strategy_source": (
                compare_baseline_strategy_source or compare_strategy_source
            ),
        }
    return enqueue_cycle(
        root,
        hypothesis=(
            f"GPU Deep CFR training should produce {prefix}_final.pt with "
            "machine-readable throughput metrics."
        ),
        cycle_type="experiment",
        failure_class="compute_efficiency",
        gate=gate_name,
        postprocess=postprocess,
    )


def enqueue_warm_start_resolver_gate(
    root: str | Path,
    *,
    checkpoint: str | Path,
    cases_json: str | Path,
    cfv_cache: str | Path,
    train_labels_npz: str | Path | None = None,
    start_index: int = 128,
    limit: int = 64,
    low_iterations: int = 5,
    baseline_iterations: int | None = None,
    reference_iterations: int = 25,
    solver_backend: str = "cpu",
    device: str = "auto",
    regret_mass_scale: float = 1.0,
    strategy_mass: float = 0.0,
    min_evaluated: int = 32,
    max_warm_latency_ratio: float = 2.0,
    output_json: str | Path | None = None,
    timeout_seconds: int = 3600,
) -> dict:
    """Queue the root-disjoint neural warm-start resolver A/B gate."""
    root = Path(root)
    _assert_phase_allows_action(root, "eval_warm_start_resolver_gate")
    checkpoint_path = _resolve_existing_path(root, checkpoint, label="Warm-start checkpoint")
    cases_path = _resolve_existing_path(root, cases_json, label="Warm-start cases JSON")
    cache_path = _resolve_existing_path(root, cfv_cache, label="Warm-start CFV cache")
    train_path = (
        _resolve_existing_path(root, train_labels_npz, label="Warm-start train labels")
        if train_labels_npz is not None
        else None
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if output_json is None:
        output_json = _runs_path(root) / f"{timestamp}-warm-start-resolver-gate" / "metrics.json"
    else:
        output_json = Path(output_json)
        if not output_json.is_absolute():
            output_json = root / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)

    gate_name = _unique_gate_name(
        root,
        f"warm-start-resolver-{timestamp}-{_slug(checkpoint_path.stem)}",
    )
    python = _project_python(root)
    evaluator_script = _warm_start_evaluator_script(checkpoint_path)
    command = [
        python,
        evaluator_script,
        "--checkpoint",
        str(checkpoint_path),
        "--cases",
        str(cases_path),
        "--cfv-cache",
        str(cache_path),
        "--device",
        device,
        "--start-index",
        str(start_index),
        "--limit",
        str(limit),
        "--low-iterations",
        str(low_iterations),
        "--reference-iterations",
        str(reference_iterations),
        "--solver-backend",
        solver_backend,
        "--min-evaluated",
        str(min_evaluated),
        "--max-warm-latency-ratio",
        str(max_warm_latency_ratio),
        "--output-json",
        str(output_json),
    ]
    if evaluator_script == "scripts/eval_joint_pbs_policy_warm_start.py":
        if baseline_iterations is not None:
            raise ValueError(
                "baseline_iterations is currently supported only for "
                "regret-policy warm-start checkpoints."
            )
        command.extend(
            [
                "--regret-mass-scale",
                str(regret_mass_scale),
                "--strategy-mass",
                str(strategy_mass),
            ]
        )
    elif baseline_iterations is not None:
        command.extend(["--baseline-iterations", str(baseline_iterations)])
    if train_path is not None:
        command.extend(["--train-labels-npz", str(train_path)])

    goal = _read_json(_goal_path(root))
    goal.setdefault("gates", {})[gate_name] = {
        "description": (
            "Root-disjoint neural regret-field resolving gate: compare vanilla "
            "low-budget CFR+ against neural-warm-start low-budget CFR+ using a "
            "higher-budget resolver teacher."
        ),
        "timeout_seconds": int(timeout_seconds),
        "commands": [command],
        "checkpoint": str(checkpoint_path),
        "checkpoint_mode": _checkpoint_mode(checkpoint_path),
        "evaluator_script": evaluator_script,
        "cases_json": str(cases_path),
        "cfv_cache": str(cache_path),
        "output_json": str(output_json),
        "low_iterations": int(low_iterations),
        "baseline_iterations": (
            int(baseline_iterations) if baseline_iterations is not None else None
        ),
        "reference_iterations": int(reference_iterations),
    }
    _write_json(_goal_path(root), goal)
    return enqueue_cycle(
        root,
        hypothesis=(
            "Neural warm-started low-budget resolving should be closer than "
            "vanilla low-budget resolving to the higher-budget teacher on "
            "root-disjoint public states."
        ),
        cycle_type="warm_start_resolver_gate",
        failure_class="search_quality",
        gate=gate_name,
    )


def _comma_ints(values: Sequence[int] | str) -> str:
    if isinstance(values, str):
        parsed = [int(part.strip()) for part in values.split(",") if part.strip()]
    else:
        parsed = [int(value) for value in values]
    if not parsed or min(parsed) <= 0:
        raise ValueError("integer list must contain positive values")
    return ",".join(str(value) for value in parsed)


def enqueue_cfr_budget_frontier(
    root: str | Path,
    *,
    cases_json: str | Path,
    cfv_cache: str | Path,
    budgets: Sequence[int] | str,
    start_index: int = 128,
    limit: int = 64,
    reference_iterations: int = 25,
    solver_backend: str = "torch-levelsync-cuda",
    solver_update: str = "cfr_plus",
    min_evaluated: int = 1,
    output_json: str | Path | None = None,
    timeout_seconds: int = 3600,
) -> dict:
    """Queue a root-disjoint exact CFR budget frontier gate."""
    root = Path(root)
    _assert_phase_allows_action(root, "solver_budget_frontier")
    cases_path = _resolve_existing_path(root, cases_json, label="CFR frontier cases JSON")
    cache_path = _resolve_existing_path(root, cfv_cache, label="CFR frontier CFV cache")
    budget_list = _comma_ints(budgets)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if output_json is None:
        output_json = _runs_path(root) / f"{timestamp}-cfr-budget-frontier" / "metrics.json"
    else:
        output_json = Path(output_json)
        if not output_json.is_absolute():
            output_json = root / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)

    gate_name = _unique_gate_name(
        root,
        f"cfr-budget-frontier-{timestamp}",
    )
    python = _project_python(root)
    command = [
        python,
        "scripts/eval_cfr_budget_frontier.py",
        "--cases",
        str(cases_path),
        "--cfv-cache",
        str(cache_path),
        "--budgets",
        budget_list,
        "--start-index",
        str(start_index),
        "--limit",
        str(limit),
        "--reference-iterations",
        str(reference_iterations),
        "--solver-backend",
        solver_backend,
        "--solver-update",
        solver_update,
        "--min-evaluated",
        str(min_evaluated),
        "--output-json",
        str(output_json),
    ]

    goal = _read_json(_goal_path(root))
    goal.setdefault("gates", {})[gate_name] = {
        "description": (
            "Root-disjoint exact CFR budget frontier. This gate compares lower "
            "CFR budgets against a higher-budget teacher and records root "
            "decision quality, illegal mass, and latency."
        ),
        "timeout_seconds": int(timeout_seconds),
        "commands": [command],
        "cases_json": str(cases_path),
        "cfv_cache": str(cache_path),
        "output_json": str(output_json),
        "budgets": budget_list,
        "reference_iterations": int(reference_iterations),
        "solver_backend": solver_backend,
        "solver_update": solver_update,
    }
    _write_json(_goal_path(root), goal)
    return enqueue_cycle(
        root,
        hypothesis=(
            "Exact GPU CFR budget frontier should preserve legal root decisions "
            "and improve quality as budget increases on root-disjoint public states."
        ),
        cycle_type="solver_budget_frontier",
        failure_class="search_quality",
        gate=gate_name,
    )


def enqueue_cfr_matrix_footprint(
    root: str | Path,
    *,
    cases_json: str | Path,
    start_index: int = 128,
    max_cases: int | None = None,
    chunk_memory_cap_mib: float | None = None,
    output_json: str | Path | None = None,
    timeout_seconds: int = 900,
) -> dict:
    """Queue a matrix/fused CFR footprint and memory-chunk planning gate."""
    root = Path(root)
    _assert_phase_allows_action(root, "solver_latency_profile")
    cases_path = _resolve_existing_path(root, cases_json, label="CFR matrix cases JSON")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if output_json is None:
        output_json = _runs_path(root) / f"{timestamp}-cfr-matrix-footprint" / "metrics.json"
    else:
        output_json = Path(output_json)
        if not output_json.is_absolute():
            output_json = root / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)

    gate_name = _unique_gate_name(root, f"cfr-matrix-footprint-{timestamp}")
    python = _project_python(root)
    command = [
        python,
        "scripts/analyze_cfr_matrix_footprint.py",
        "--cases-json",
        str(cases_path),
        "--start-index",
        str(start_index),
        "--output-json",
        str(output_json),
    ]
    if max_cases is not None:
        command.extend(["--max-cases", str(max_cases)])
    if chunk_memory_cap_mib is not None:
        command.extend(["--chunk-memory-cap-mib", str(float(chunk_memory_cap_mib))])

    goal = _read_json(_goal_path(root))
    goal.setdefault("gates", {})[gate_name] = {
        "description": (
            "Matrix/fused CFR feasibility gate. This estimates per-root solver "
            "state memory and, when requested, emits an order-preserving "
            "memory-capped chunk plan before any fused solver implementation."
        ),
        "timeout_seconds": int(timeout_seconds),
        "commands": [command],
        "cases_json": str(cases_path),
        "output_json": str(output_json),
        "start_index": int(start_index),
        "max_cases": int(max_cases) if max_cases is not None else None,
        "chunk_memory_cap_mib": (
            float(chunk_memory_cap_mib) if chunk_memory_cap_mib is not None else None
        ),
    }
    _write_json(_goal_path(root), goal)
    return enqueue_cycle(
        root,
        hypothesis=(
            "CFR matrix footprint should yield a memory-capped chunk plan before "
            "attempting fused exact GPU resolving."
        ),
        cycle_type="exact_gpu_solver_planning",
        failure_class="compute_efficiency",
        gate=gate_name,
    )


def _first_stdout_json(metrics: dict) -> dict | None:
    for command_metric in metrics.get("commands", []):
        payload = command_metric.get("stdout_json")
        if isinstance(payload, dict):
            return payload
    return None


def _training_checkpoint_paths(payload: dict) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for record in payload.get("checkpoints", []):
        if isinstance(record, dict):
            path = record.get("path")
        else:
            path = record
        if not isinstance(path, str) or not path:
            continue
        if path not in seen:
            paths.append(path)
            seen.add(path)
    final = payload.get("checkpoint")
    if isinstance(final, str) and final and final not in seen:
        paths.append(final)
    return paths


def _postprocess_completed_cycle(root: Path, item: dict, metrics: dict) -> list[dict]:
    postprocess = item.get("postprocess") or {}
    if not metrics.get("passed"):
        return []
    if postprocess.get("type") != "compare_training_checkpoints":
        return []

    payload = _first_stdout_json(metrics)
    if payload is None or payload.get("mode") != "autoresearch_gpu_deep_cfr_train":
        return []

    queued: list[dict] = []
    for checkpoint in _training_checkpoint_paths(payload):
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path
        if not checkpoint_path.exists():
            continue
        queued.append(
            enqueue_candidate_comparison(
                root,
                checkpoint_path,
                n_games=int(postprocess.get("n_games", 500)),
                seeds=str(postprocess.get("seeds", "20260511,20260512,20260513")),
                device=str(postprocess.get("device", "auto")),
                timeout_seconds=int(postprocess.get("timeout_seconds", 2400)),
                head_to_head=bool(postprocess.get("head_to_head", True)),
                strategy_source=str(postprocess.get("strategy_source", "regret")),
                candidate_strategy_source=str(
                    postprocess.get(
                        "candidate_strategy_source",
                        postprocess.get("strategy_source", "regret"),
                    )
                ),
                baseline_strategy_source=str(
                    postprocess.get(
                        "baseline_strategy_source",
                        postprocess.get("strategy_source", "regret"),
                    )
                ),
            )
        )
    return queued


def _is_soft_mechanism_failure(item: dict) -> bool:
    cycle_type = str(item.get("type", ""))
    failure_class = str(item.get("failure_class", ""))
    if not failure_class or failure_class == "none":
        return False
    if failure_class in HARD_STOP_FAILURE_CLASSES:
        return False
    if (
        cycle_type == "synthesis"
        or "review" in cycle_type
        or "audit" in cycle_type
    ):
        return False
    return True


def _postprocess_failed_mechanism_cycle(root: Path, item: dict) -> list[dict]:
    """Queue synthesis and pivot review after a failed mechanism gate."""
    subject = f"{item.get('failure_class', 'mechanism failure')} after {item.get('gate', 'gate')}"
    anomaly = (
        f"Gate {item.get('gate', 'unknown')} failed for hypothesis: "
        f"{item.get('hypothesis', 'unknown hypothesis')}. "
        "Treat this as evidence to synthesize and pivot, not as project completion."
    )
    return [
        enqueue_failure_synthesis(root, subject=subject),
        enqueue_paradigm_innovation_review(
            root,
            subject=f"pivot after {item.get('failure_class', 'mechanism failure')}",
            anomaly=anomaly,
        ),
    ]


def enqueue_candidate_comparison(
    root: str | Path,
    candidate_checkpoint: str | Path,
    *,
    baseline_checkpoint: str | Path | None = None,
    n_games: int = 500,
    seeds: str = "20260511,20260512,20260513",
    device: str = "auto",
    timeout_seconds: int = 2400,
    head_to_head: bool = False,
    strategy_source: str = "regret",
    candidate_strategy_source: str | None = None,
    baseline_strategy_source: str | None = None,
) -> dict:
    """Create and queue a one-off candidate-vs-incumbent local comparison gate."""
    root = Path(root)
    candidate = _resolve_existing_path(root, candidate_checkpoint, label="Candidate checkpoint")
    state = _read_json(_state_path(root))
    candidate_strategy_source = candidate_strategy_source or strategy_source
    baseline_strategy_source = baseline_strategy_source or strategy_source

    if baseline_checkpoint is None:
        incumbent = state.get("incumbent_checkpoint") or {}
        baseline_checkpoint = incumbent.get("checkpoint")
        if baseline_checkpoint is None:
            raise RuntimeError("No baseline checkpoint provided and no incumbent is recorded.")
    baseline = _resolve_existing_path(root, baseline_checkpoint, label="Baseline checkpoint")

    gate_name = _unique_gate_name(root, (
        f"eval-candidate-compare-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{_slug(candidate.stem)}"
    ))
    python = _project_python(root)
    command = [
        python,
        "scripts/poker_autoresearch_eval.py",
        "--checkpoint",
        str(candidate),
        "--baseline-checkpoint",
        str(baseline),
        "--n-games",
        str(n_games),
        "--device",
        device,
        "--seeds",
        seeds,
    ]
    if head_to_head:
        command.append("--head-to-head")
    if candidate_strategy_source == baseline_strategy_source:
        if candidate_strategy_source != "regret":
            command.extend(["--strategy-source", candidate_strategy_source])
    else:
        command.extend(["--candidate-strategy-source", candidate_strategy_source])
        command.extend(["--baseline-strategy-source", baseline_strategy_source])
    command.append("--require-positive-lower95")

    if candidate_strategy_source == baseline_strategy_source:
        strategy_description = f"{candidate_strategy_source} strategy source"
    else:
        strategy_description = (
            f"candidate {candidate_strategy_source} vs baseline "
            f"{baseline_strategy_source} strategy sources"
        )

    gate_config = {
        "description": (
            "One-off local candidate-vs-incumbent comparison. This gate emits "
            "delta metrics and fails unless the lower 95% comparison bound is "
            "positive. It is not a standalone strategy-strength proof."
        ),
        "timeout_seconds": timeout_seconds,
        "commands": [command],
    }

    goal = _read_json(_goal_path(root))
    goal.setdefault("gates", {})[gate_name] = gate_config
    _write_json(_goal_path(root), goal)
    return enqueue_cycle(
        root,
        hypothesis=(
            f"Candidate checkpoint {candidate.name} should improve local "
            f"comparison metrics against incumbent {baseline.name} without "
            f"claiming local-only promotion using {strategy_description}."
        ),
        cycle_type="experiment",
        failure_class="strategy_quality",
        gate=gate_name,
    )


def enqueue_candidate_promotion_gate(
    root: str | Path,
    *,
    rlcard_reference_json: str | Path,
    native_h2h_json: str | Path,
    empirical_game_json: str | Path | None = None,
    native_candidate_checkpoint: str | Path | None = None,
    min_lower95: float = 0.0,
    min_candidate_support: float = 1.0e-9,
    timeout_seconds: int = 300,
) -> dict:
    """Create and queue the dual-surface pre-Slumbot promotion gate."""
    root = Path(root)
    rlcard_path = _resolve_existing_path(
        root,
        rlcard_reference_json,
        label="RLCard AlphaNLHoldem reference metrics",
    )
    native_path = _resolve_existing_path(
        root,
        native_h2h_json,
        label="Native 9-action H2H metrics",
    )
    empirical_path = (
        None
        if empirical_game_json is None
        else _resolve_existing_path(root, empirical_game_json, label="Native empirical-game metrics")
    )
    native_candidate_path = (
        None
        if native_candidate_checkpoint is None
        else _resolve_existing_path(
            root,
            native_candidate_checkpoint,
            label="Native candidate checkpoint",
        )
    )

    gate_name = _unique_gate_name(
        root,
        f"candidate-promotion-gate-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
    )
    output_json = _session(root) / "candidate_promotion" / f"{gate_name}.json"
    python = _project_python(root)
    command = [
        python,
        "scripts/eval_candidate_promotion_gate.py",
        "--rlcard-reference-json",
        str(rlcard_path),
        "--native-h2h-json",
        str(native_path),
        "--min-lower95",
        str(float(min_lower95)),
        "--min-candidate-support",
        str(float(min_candidate_support)),
        "--output-json",
        str(output_json),
    ]
    if empirical_path is not None:
        command.extend(["--empirical-game-json", str(empirical_path)])
    if native_candidate_path is not None:
        command.extend(["--native-candidate-checkpoint", str(native_candidate_path)])

    goal = _read_json(_goal_path(root))
    goal.setdefault("gates", {})[gate_name] = {
        "description": (
            "Pre-Slumbot candidate promotion gate. Requires RLCard AlphaNLHoldem "
            "reference evidence, native 9-action H2H evidence, native empirical-game "
            "support, no benchmark data leakage, and no cross-environment action projection."
        ),
        "timeout_seconds": int(timeout_seconds),
        "commands": [command],
        "promotion_role": "pre_slumbot_confidence_gate",
        "requires_rlcard_reference_pass": True,
        "requires_native_h2h_pass": True,
        "requires_empirical_game_pass": True,
        "blocks_slumbot_confidence_until_passed": True,
        "output_json": str(output_json),
    }
    _write_json(_goal_path(root), goal)
    return enqueue_cycle(
        root,
        hypothesis=(
            "Candidate should clear the dual-surface pre-Slumbot promotion gate "
            "before any held-out Slumbot confidence evaluation."
        ),
        cycle_type="candidate_promotion_gate",
        failure_class="promotion_evidence",
        gate=gate_name,
    )


def enqueue_slumbot_smoke(
    root: str | Path,
    model: str | Path,
    *,
    model_kind: str = "auto",
    hands: int = 5,
    greedy: bool = True,
    no_allin: bool = True,
    no_solver: bool = True,
    strategy_source: str = "regret",
    solver_backend: str = "auto",
    solver_budget_profile: str = "live",
    timeout_seconds: int = 600,
) -> dict:
    """Create and queue a sparse live Slumbot smoke for a candidate checkpoint."""
    root = Path(root)
    _assert_phase_allows_action(root, "slumbot_smoke")
    model_path = _resolve_existing_path(root, model, label="Slumbot model checkpoint")
    goal = _read_json(_goal_path(root))
    slumbot_policy = goal.get("objective_alignment_policy", {}).get(
        "slumbot_validation_policy", {}
    )
    smoke_cap = int(slumbot_policy.get("max_smoke_hands_before_internal_pass", 50))
    requires_internal_league = int(hands) > smoke_cap
    internal_league_evidence = _internal_self_play_league_evidence(root)
    candidate_promotion_evidence = _candidate_promotion_gate_evidence(root)
    if requires_internal_league and not internal_league_evidence["passed"]:
        raise RuntimeError(
            "Slumbot confidence spend requires a passed self-play checkpoint league "
            f"with positive lower95 evidence before queueing {hands} hands. "
            f"Evidence status: {internal_league_evidence.get('reason', 'missing')}."
        )
    if requires_internal_league and not candidate_promotion_evidence["passed"]:
        raise RuntimeError(
            "Slumbot confidence spend requires a passed pre-Slumbot candidate promotion gate "
            "covering the RLCard AlphaNLHoldem reference surface, native 9-action H2H, "
            f"and empirical-game evidence before queueing {hands} hands. "
            f"Evidence status: {candidate_promotion_evidence.get('reason', 'missing')}."
        )
    gate_name = _unique_gate_name(root, (
        f"slumbot-candidate-smoke-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{_slug(model_path.stem)}"
    ))
    trace_path = _session(root) / "slumbot_traces" / f"{gate_name}.jsonl"
    python = _project_python(root)
    command = [
        python,
        "scripts/poker_autoresearch_slumbot.py",
        "--model",
        str(model_path),
        "--hands",
        str(hands),
        "--timeout-seconds",
        str(timeout_seconds),
        "--trace-jsonl",
        str(trace_path),
    ]
    if model_kind != "auto":
        command.extend(["--model-kind", str(model_kind)])
    if greedy:
        command.append("--greedy")
    if no_allin:
        command.append("--no-allin")
    if no_solver:
        command.append("--no-solver")
    if not no_solver and solver_backend != "auto":
        command.extend(["--solver-backend", solver_backend])
    if not no_solver and solver_budget_profile != "live":
        command.extend(["--solver-budget-profile", solver_budget_profile])
    if strategy_source != "regret":
        command.extend(["--strategy-source", strategy_source])

    goal.setdefault("gates", {})[gate_name] = {
        "description": (
            "One-off sparse live Slumbot smoke for a candidate checkpoint. "
            "This is an integration and distribution-shift diagnostic, not a "
            "promotion gate by itself."
        ),
        "timeout_seconds": timeout_seconds,
        "commands": [command],
        "promotion_role": (
            "held_out_external_validation"
            if requires_internal_league
            else "integration_smoke_only"
        ),
        "model_kind": str(model_kind),
        "requires_internal_league_pass": requires_internal_league,
        "internal_league_evidence": internal_league_evidence,
        "candidate_promotion_evidence": candidate_promotion_evidence,
    }
    _write_json(_goal_path(root), goal)
    return enqueue_cycle(
        root,
        hypothesis=(
            f"Candidate checkpoint {model_path.name} should produce parsed live "
            "Slumbot metrics under the sparse smoke budget."
        ),
        cycle_type="experiment",
        failure_class="distribution_shift",
        gate=gate_name,
    )


def enqueue_resolver_benchmark(
    root: str | Path,
    model: str | Path,
    *,
    solver_iterations: int = 25,
    solver_backend: str = "auto",
    max_cases: int | None = None,
    device: str = "auto",
    timeout_seconds: int = 1200,
) -> dict:
    """Create and queue a fixed public-state resolver benchmark for a checkpoint."""
    root = Path(root)
    model_path = _resolve_existing_path(root, model, label="Resolver benchmark model checkpoint")
    gate_name = _unique_gate_name(root, (
        f"resolver-candidate-benchmark-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{_slug(model_path.stem)}"
    ))
    python = _project_python(root)
    command = [
        python,
        "scripts/poker_resolver_benchmark.py",
        "--checkpoint",
        str(model_path),
        "--device",
        device,
        "--solver-iterations",
        str(solver_iterations),
        "--solver-backend",
        solver_backend,
    ]
    if max_cases is not None:
        command.extend(["--max-cases", str(max_cases)])

    goal = _read_json(_goal_path(root))
    goal.setdefault("gates", {})[gate_name] = {
        "description": (
            "One-off fixed public-state resolver benchmark for a candidate checkpoint. "
            "This catches legality, latency, and blueprint-vs-resolver drift before "
            "spending Slumbot hands."
        ),
        "timeout_seconds": timeout_seconds,
        "commands": [command],
    }
    _write_json(_goal_path(root), goal)
    return enqueue_cycle(
        root,
        hypothesis=(
            f"Candidate checkpoint {model_path.name} should pass fixed turn/river "
            "resolver legality and drift diagnostics before live Slumbot evaluation."
        ),
        cycle_type="experiment",
        failure_class="search_quality",
        gate=gate_name,
    )


def enqueue_falsification_ladder(
    root: str | Path,
    candidate_checkpoint: str | Path,
    *,
    mechanism: str,
    baseline_checkpoint: str | Path | None = None,
    n_games: int = 500,
    seeds: str = "20260511,20260512,20260513",
    device: str = "auto",
    changed_paths: Iterable[str] | None = None,
    solver_iterations: int = 25,
    solver_backend: str = "auto",
    max_resolver_cases: int | None = None,
    strategy_source: str = "regret",
    timeout_seconds: int = 3600,
) -> dict:
    """Create and queue a promotion falsification ladder for one candidate."""
    if not mechanism.strip():
        raise ValueError("Falsification ladder requires a mechanism claim.")

    root = Path(root)
    candidate = _resolve_existing_path(root, candidate_checkpoint, label="Candidate checkpoint")
    state = _read_json(_state_path(root))
    if baseline_checkpoint is None:
        incumbent = state.get("incumbent_checkpoint") or {}
        baseline_checkpoint = incumbent.get("checkpoint")
        if baseline_checkpoint is None:
            raise RuntimeError("No baseline checkpoint provided and no incumbent is recorded.")
    baseline = _resolve_existing_path(root, baseline_checkpoint, label="Baseline checkpoint")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    gate_name = _unique_gate_name(
        root,
        f"falsification-ladder-{timestamp}-{_slug(candidate.stem)}",
    )
    python = _project_python(root)
    audit_command = [
        python,
        "scripts/poker_objective_audit.py",
    ]
    paths = list(changed_paths or [])
    if paths:
        for changed_path in paths:
            audit_command.extend(["--changed-path", str(changed_path)])
    else:
        audit_command.extend(["--base-ref", "HEAD"])

    compare_command = [
        python,
        "scripts/poker_autoresearch_eval.py",
        "--checkpoint",
        str(candidate),
        "--baseline-checkpoint",
        str(baseline),
        "--n-games",
        str(n_games),
        "--device",
        device,
        "--seeds",
        seeds,
        "--head-to-head",
        "--require-positive-lower95",
    ]
    if strategy_source != "regret":
        compare_command.extend(["--strategy-source", strategy_source])

    resolver_command = [
        python,
        "scripts/poker_resolver_benchmark.py",
        "--checkpoint",
        str(candidate),
        "--device",
        device,
        "--solver-iterations",
        str(solver_iterations),
        "--solver-backend",
        solver_backend,
    ]
    if max_resolver_cases is not None:
        resolver_command.extend(["--max-cases", str(max_resolver_cases)])

    goal = _read_json(_goal_path(root))
    goal.setdefault("gates", {})[gate_name] = {
        "description": (
            "Promotion falsification ladder: objective-drift audit, paired "
            "incumbent head-to-head comparison, and fixed-state resolver "
            "diagnostics. Passing this gate does not by itself promote a model; "
            "it blocks weak or benchmark-hacked candidates before Slumbot spend."
        ),
        "timeout_seconds": timeout_seconds,
        "commands": [audit_command, compare_command, resolver_command],
        "mechanism": mechanism,
        "candidate_checkpoint": str(candidate),
        "baseline_checkpoint": str(baseline),
    }
    _write_json(_goal_path(root), goal)
    item = enqueue_cycle(
        root,
        hypothesis=(
            f"Candidate checkpoint {candidate.name} should survive falsification "
            f"of mechanism: {mechanism}"
        ),
        cycle_type="falsification",
        failure_class="strategy_quality",
        gate=gate_name,
    )
    item["mechanism"] = mechanism
    state = _read_json(_state_path(root))
    state["hypothesis_queue"][-1]["mechanism"] = mechanism
    state["updated_at"] = _now()
    _write_json(_state_path(root), state)
    return item


def enqueue_sd_cfr_mixture_falsification(
    root: str | Path,
    *,
    candidate_globs: Iterable[str | Path] | None = None,
    candidate_checkpoints: Iterable[str | Path] | None = None,
    mechanism: str,
    baseline_checkpoint: str | Path | None = None,
    n_games: int = 500,
    seeds: str = "20260511,20260512,20260513",
    device: str = "auto",
    changed_paths: Iterable[str] | None = None,
    review_dir: str | Path | None = None,
    strategy_source: str = "regret",
    timeout_seconds: int = 3600,
) -> dict:
    """Create and queue a local falsification gate for a fixed SD-CFR checkpoint mixture."""
    if not mechanism.strip():
        raise ValueError("SD-CFR mixture falsification requires a mechanism claim.")

    root = Path(root)
    resolved_globs: list[str] = []
    for pattern in candidate_globs or []:
        path_pattern = Path(pattern)
        resolved = path_pattern if path_pattern.is_absolute() else root / path_pattern
        matches = sorted(glob(str(resolved)))
        if not matches:
            raise FileNotFoundError(f"Candidate checkpoint glob matched no files: {resolved}")
        resolved_globs.append(str(resolved))

    resolved_checkpoints = [
        str(_resolve_existing_path(root, checkpoint, label="Candidate checkpoint"))
        for checkpoint in candidate_checkpoints or []
    ]
    if not resolved_globs and not resolved_checkpoints:
        raise ValueError("Provide at least one candidate glob or candidate checkpoint.")

    state = _read_json(_state_path(root))
    if baseline_checkpoint is None:
        incumbent = state.get("incumbent_checkpoint") or {}
        baseline_checkpoint = incumbent.get("checkpoint")
        if baseline_checkpoint is None:
            raise RuntimeError("No baseline checkpoint provided and no incumbent is recorded.")
    baseline = _resolve_existing_path(root, baseline_checkpoint, label="Baseline checkpoint")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    gate_name = _unique_gate_name(root, f"sd-cfr-mixture-falsification-{timestamp}")
    python = _project_python(root)
    audit_command = [
        python,
        "scripts/poker_objective_audit.py",
    ]
    if review_dir is not None:
        audit_command.extend(["--review-dir", str(review_dir)])
    paths = list(changed_paths or [])
    if paths:
        for changed_path in paths:
            audit_command.extend(["--changed-path", str(changed_path)])
    else:
        audit_command.extend(["--base-ref", "HEAD"])

    output_path = _session(root) / "sd_cfr_mixture" / f"{gate_name}.json"
    compare_command = [
        python,
        "scripts/eval_sd_cfr_mixture.py",
        "--baseline-checkpoint",
        str(baseline),
        "--n-games",
        str(n_games),
        "--device",
        device,
        "--seeds",
        seeds,
        "--strategy-source",
        strategy_source,
        "--output",
        str(output_path),
    ]
    for pattern in resolved_globs:
        compare_command.extend(["--candidate-glob", pattern])
    for checkpoint in resolved_checkpoints:
        compare_command.extend(["--candidate-checkpoint", checkpoint])

    goal = _read_json(_goal_path(root))
    goal.setdefault("gates", {})[gate_name] = {
        "description": (
            "SD-CFR checkpoint-mixture falsification gate: objective-drift audit "
            "plus duplicate-swapped local H2H using fixed candidate globs or "
            "checkpoints. This blocks weak mixture candidates before Slumbot spend."
        ),
        "timeout_seconds": timeout_seconds,
        "commands": [audit_command, compare_command],
        "mechanism": mechanism,
        "candidate_globs": resolved_globs,
        "candidate_checkpoints": resolved_checkpoints,
        "baseline_checkpoint": str(baseline),
        "output_json": str(output_path),
    }
    _write_json(_goal_path(root), goal)
    item = enqueue_cycle(
        root,
        hypothesis=(
            "Fixed SD-CFR checkpoint mixture should survive local falsification "
            f"of mechanism: {mechanism}"
        ),
        cycle_type="falsification",
        failure_class="strategy_quality",
        gate=gate_name,
    )
    item["mechanism"] = mechanism
    state = _read_json(_state_path(root))
    state["hypothesis_queue"][-1]["mechanism"] = mechanism
    state["updated_at"] = _now()
    _write_json(_state_path(root), state)
    return item


def enqueue_callback_calibration_audit(
    root: str | Path,
    *,
    train_dual_cache: str | Path,
    holdout_dual_cache: str | Path,
    supervised_metrics: str | Path | None = None,
    leaf_ab: str | Path | None = None,
    output_json: str | Path | None = None,
    timeout_seconds: int = 900,
) -> dict:
    """Queue the callback-state target calibration audit phase."""
    root = Path(root)
    _assert_phase_allows_action(root, "callback_state_calibration_audit")
    train = _resolve_existing_path(root, train_dual_cache, label="Train callback cache")
    holdout = _resolve_existing_path(root, holdout_dual_cache, label="Holdout callback cache")
    supervised = (
        _resolve_existing_path(root, supervised_metrics, label="Supervised metrics")
        if supervised_metrics
        else None
    )
    leaf = _resolve_existing_path(root, leaf_ab, label="Leaf A/B metrics") if leaf_ab else None
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if output_json is None:
        output_json = (
            _runs_path(root)
            / f"{timestamp}-callback-state-calibration-audit"
            / "metrics.json"
        )
    else:
        output_json = Path(output_json)
        if not output_json.is_absolute():
            output_json = root / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)

    gate_name = _unique_gate_name(
        root,
        f"callback-state-calibration-audit-{timestamp}",
    )
    python = _project_python(root)
    command = [
        python,
        "scripts/poker_callback_calibration_audit.py",
        "--train-dual-cache",
        str(train),
        "--holdout-dual-cache",
        str(holdout),
        "--output-json",
        str(output_json),
    ]
    if supervised is not None:
        command.extend(["--supervised-metrics", str(supervised)])
    if leaf is not None:
        command.extend(["--leaf-ab", str(leaf)])

    goal = _read_json(_goal_path(root))
    goal.setdefault("gates", {})[gate_name] = {
        "description": (
            "Callback-state DCVN calibration audit: summarize target variance, "
            "reach skew, support coverage, and failed integration metrics before "
            "adding capacity or Slumbot runs."
        ),
        "timeout_seconds": timeout_seconds,
        "commands": [command],
        "phase": "callback_state_calibration_debug",
    }
    _write_json(_goal_path(root), goal)
    item = enqueue_cycle(
        root,
        hypothesis=(
            "Callback-state calibration audit should explain whether target "
            "variance, reach skew, or loss weighting likely caused the DCVN "
            "scale-up failure."
        ),
        cycle_type="calibration_audit",
        failure_class="callback_state_scale_generalization_gap",
        gate=gate_name,
    )
    item["calibration_audit_output"] = str(output_json)
    state = _read_json(_state_path(root))
    state["hypothesis_queue"][-1]["calibration_audit_output"] = str(output_json)
    state["updated_at"] = _now()
    _write_json(_state_path(root), state)
    return item


def append_research_log(
    root: str | Path,
    *,
    cycle: dict,
    outcome: str,
    failure_class: str,
    summary: str,
    metrics: dict,
    metrics_path: str | Path | None = None,
) -> None:
    root = Path(root)
    log = _log_path(root)
    if not log.exists():
        log.write_text("# Research Log\n\n", encoding="utf-8")

    key_metrics = {"passed": metrics.get("passed"), "gate": metrics.get("gate")}
    payload = _first_stdout_json(metrics)
    if payload:
        for key in (
            "mode",
            "avg_chips_per_hand",
            "lower95_chips_per_hand",
            "paired_delta_lower95_chips_per_hand_across_seeds",
            "ci95_chips_per_hand",
            "mbb_per_hand",
            "iters_per_hour",
            "traversals_per_second",
            "avg_iter_seconds",
            "seconds_per_hand",
            "strategy_source",
            "decision",
        ):
            if key in payload:
                key_metrics[key] = payload[key]
    metrics_location = "not recorded"
    if metrics_path is not None:
        path = Path(metrics_path)
        if path.is_absolute():
            try:
                metrics_location = str(path.relative_to(root))
            except ValueError:
                metrics_location = str(path)
        else:
            metrics_location = str(path)
    entry = (
        f"## {cycle['run_id']} - {outcome}\n\n"
        f"- Timestamp: {_now()}\n"
        f"- Type: {cycle['type']}\n"
        f"- Gate: {cycle['gate']}\n"
        f"- Hypothesis: {cycle['hypothesis']}\n"
        f"- Failure class: {failure_class}\n"
        f"- Summary: {summary}\n"
        f"- Metrics file: {metrics_location}\n"
        f"- Key metrics: `{json.dumps(key_metrics, sort_keys=True)}`\n\n"
    )
    with log.open("a", encoding="utf-8") as handle:
        handle.write(entry)


def close_cycle(
    root: str | Path,
    run_id: str,
    *,
    outcome: str,
    failure_class: str,
    metrics_path: str | Path | None = None,
    summary: str,
) -> dict:
    root = Path(root)
    state = _read_json(_state_path(root))
    cycle = state.get("active_cycle")
    if cycle is None or cycle.get("run_id") != run_id:
        raise RuntimeError(f"Active cycle does not match run_id {run_id}.")

    metrics = {}
    if metrics_path is not None:
        metrics = _read_json(Path(metrics_path))

    closed = {
        **cycle,
        "outcome": outcome,
        "closed_at": _now(),
        "failure_class": failure_class,
        "summary": summary,
    }
    state["active_cycle"] = None
    state["last_metrics"] = metrics
    state.setdefault("history", []).append(closed)
    state["updated_at"] = _now()
    _write_json(_state_path(root), state)
    append_research_log(
        root,
        cycle=cycle,
        outcome=outcome,
        failure_class=failure_class,
        summary=summary,
        metrics=metrics,
        metrics_path=metrics_path,
    )
    return closed


def _latest_log_heading(root: Path) -> str | None:
    log_path = _log_path(root)
    if not log_path.exists():
        return None
    headings = [
        line.strip()
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("## ")
    ]
    return headings[-1] if headings else None


def commit_ready_report(
    root: str | Path,
    *,
    changed_paths: Iterable[str] | None = None,
    base_ref: str | None = None,
    review_dir: str | Path | None = None,
    allow_empty: bool = False,
) -> dict:
    """Summarize whether the current research batch is ready for a natural commit."""
    root = Path(root)
    if changed_paths is None:
        changed = git_changed_paths(root, base_ref=base_ref)
    else:
        changed = sorted({str(path).replace("\\", "/").lstrip("./") for path in changed_paths})
    objective = audit_objective_alignment(
        root,
        changed_paths=changed,
        review_dir=review_dir,
    )
    synthesis = synthesis_status(root)
    state = _read_json(_state_path(root))
    blockers: list[str] = []
    if not changed and not allow_empty:
        blockers.append("no changed paths; use allow_empty only for a deliberate no-op report")
    if not objective["passed"]:
        blockers.append("objective audit failed")
    if synthesis.get("due"):
        blockers.append("failure synthesis is due before another research commit")
    if state.get("active_cycle") is not None:
        blockers.append("an active autoresearch cycle is still open")
    if any(item.get("type") == "methodology_review" for item in state.get("hypothesis_queue", [])):
        blockers.append("methodology review is queued but not completed")

    return {
        "ready": not blockers,
        "changed_paths": changed,
        "objective_audit": objective,
        "synthesis": synthesis,
        "active_cycle": state.get("active_cycle"),
        "queued_cycles": len(state.get("hypothesis_queue", [])),
        "latest_research_log_entry": _latest_log_heading(root),
        "blockers": blockers,
    }


def _cycle_can_run_when_synthesis_due(item: dict) -> bool:
    cycle_type = str(item.get("type", ""))
    return (
        cycle_type == "synthesis"
        or "review" in cycle_type
        or "audit" in cycle_type
    )


def _cycle_text(item: dict) -> str:
    fields = [
        item.get("type", ""),
        item.get("gate", ""),
        item.get("failure_class", ""),
        item.get("hypothesis", ""),
        item.get("summary", ""),
    ]
    return " ".join(str(field).lower() for field in fields if field is not None)


def _is_local_target_consumer_family(item: dict) -> bool:
    text = _cycle_text(item)
    if not any(term in text for term in LOCAL_TARGET_CONSUMER_TERMS):
        return False
    if "audit" in text or "review" in text or "synthesis" in text:
        return False
    if "response oracle" in text and not ("target consumer" in text or "npi" in text or "xdo" in text):
        return False
    return True


def _is_primary_rl_response_oracle_family(item: dict) -> bool:
    text = _cycle_text(item)
    return any(term in text for term in RL_RESPONSE_ORACLE_TERMS)


def _failed_local_target_consumer_records(history: Sequence[dict], *, limit: int = 12) -> list[dict]:
    failures: list[dict] = []
    for record in list(history)[-int(limit) :]:
        if record.get("outcome") != "failed":
            continue
        if record.get("failure_class") not in {"mechanism_transfer", "strategy_quality"}:
            continue
        if _is_local_target_consumer_family(record):
            failures.append(record)
    return failures


def research_drift_status(
    root: str | Path,
    *,
    next_cycle: dict | None = None,
    recent_limit: int = 12,
    failure_threshold: int = 2,
) -> dict:
    """Return whether the next item repeats a failed mechanism family.

    This is a lightweight research-log guard. The structured state history is
    the machine-readable mirror of ``RESEARCH_LOG.md``; use it to stop obvious
    philosophy drift before another experiment starts.
    """
    root = Path(root)
    state = _read_json(_state_path(root))
    history = state.get("history", [])
    recent_failures = _failed_local_target_consumer_records(
        history,
        limit=int(recent_limit),
    )
    blockers: list[str] = []
    matched_family = None
    requires_review = False
    blocked = False

    if next_cycle is not None and _cycle_can_run_when_synthesis_due(next_cycle):
        return {
            "blocked": False,
            "requires_review": False,
            "matched_family": None,
            "recent_failed_same_family_count": len(recent_failures),
            "recent_failed_run_ids": [str(record.get("run_id", "")) for record in recent_failures],
            "latest_research_log_entry": _latest_log_heading(root),
            "blockers": [],
        }

    if (
        next_cycle is not None
        and _is_local_target_consumer_family(next_cycle)
        and not _is_primary_rl_response_oracle_family(next_cycle)
        and len(recent_failures) >= int(failure_threshold)
    ):
        matched_family = "local_target_consumer"
        requires_review = True
        blocked = True
        blockers.append(
            "review the research log before repeating local target-consumer/search-label "
            "experiments; recent failures show this family has not transferred to "
            "whole-game parent/incumbent H2H"
        )

    return {
        "blocked": blocked,
        "requires_review": requires_review,
        "matched_family": matched_family,
        "recent_failed_same_family_count": len(recent_failures),
        "recent_failed_run_ids": [str(record.get("run_id", "")) for record in recent_failures],
        "latest_research_log_entry": _latest_log_heading(root),
        "blockers": blockers,
    }


def _pop_next_cycle_for_synthesis_state(
    queue: list[dict],
    *,
    synthesis_due: bool,
) -> tuple[dict | None, list[dict]]:
    if not synthesis_due or _cycle_can_run_when_synthesis_due(queue[0]):
        item = queue.pop(0)
        return item, queue

    for idx, queued in enumerate(queue[1:], start=1):
        if _cycle_can_run_when_synthesis_due(queued):
            item = queue.pop(idx)
            return item, queue
    return None, queue


def _queued_review_like_cycle_is_complete(item: dict) -> bool:
    """Return True for queued review artifacts that already validate complete."""
    try:
        cycle_type = str(item.get("type", ""))
        if cycle_type == "methodology_review" and item.get("review_dir"):
            return bool(validate_methodology_review(item["review_dir"]).get("passed"))
        if cycle_type == "synthesis" and item.get("synthesis_dir"):
            return bool(validate_failure_synthesis(item["synthesis_dir"]).get("passed"))
        if cycle_type == "innovation_review" and item.get("innovation_dir"):
            return bool(validate_paradigm_innovation_review(item["innovation_dir"]).get("passed"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return False


def _drop_completed_review_like_queue_items(queue: list[dict]) -> tuple[list[dict], list[dict]]:
    pending: list[dict] = []
    skipped: list[dict] = []
    for item in queue:
        if _queued_review_like_cycle_is_complete(item):
            skipped.append(item)
        else:
            pending.append(item)
    return pending, skipped


def continuous(
    root: str | Path,
    *,
    max_cycles: int | None = None,
    runner: CommandRunner = run_command,
    sleep_seconds: float = 30.0,
    max_idle_checks: int | None = None,
    continue_on_mechanism_fail: bool = False,
) -> dict:
    root = Path(root)
    cycles_completed = 0
    idle_checks = 0
    continued_after_failures = 0

    while max_cycles is None or cycles_completed < max_cycles:
        if _stop_path(root).exists():
            return {"cycles_completed": cycles_completed, "stopped_reason": "stop_file"}

        report = readiness_report(root)
        if not report["ready"]:
            return {
                "cycles_completed": cycles_completed,
                "stopped_reason": "not_ready",
                "readiness": report,
            }

        state = _read_json(_state_path(root))
        queue = state.get("hypothesis_queue", [])
        queue, skipped_completed = _drop_completed_review_like_queue_items(queue)
        if skipped_completed:
            state["hypothesis_queue"] = queue
            state["updated_at"] = _now()
            _write_json(_state_path(root), state)
        if not queue:
            if max_cycles is not None:
                result = {"cycles_completed": cycles_completed, "stopped_reason": "empty_queue"}
                if skipped_completed:
                    result["skipped_completed_cycles"] = len(skipped_completed)
                return result
            idle_checks += 1
            if max_idle_checks is not None and idle_checks >= max_idle_checks:
                result = {"cycles_completed": cycles_completed, "stopped_reason": "idle_limit"}
                if skipped_completed:
                    result["skipped_completed_cycles"] = len(skipped_completed)
                return result
            time.sleep(sleep_seconds)
            continue
        idle_checks = 0

        synthesis = synthesis_status(root)
        item, queue = _pop_next_cycle_for_synthesis_state(
            queue,
            synthesis_due=bool(synthesis["due"]),
        )
        if item is None:
            return {
                "cycles_completed": cycles_completed,
                "stopped_reason": "synthesis_due",
                "synthesis": synthesis,
                "next_cycle": queue[0],
            }
        drift = research_drift_status(root, next_cycle=item)
        if drift.get("blocked"):
            queue.insert(0, item)
            review_item: dict | None = None
            for idx, queued in enumerate(queue[1:], start=1):
                if _cycle_can_run_when_synthesis_due(queued):
                    review_item = queue.pop(idx)
                    break
            if review_item is None:
                return {
                    "cycles_completed": cycles_completed,
                    "stopped_reason": "objective_drift_review_required",
                    "drift": drift,
                    "next_cycle": item,
                }
            item = review_item

        state["hypothesis_queue"] = queue
        state["updated_at"] = _now()
        _write_json(_state_path(root), state)

        cycle = new_cycle(
            root,
            hypothesis=item["hypothesis"],
            cycle_type=item["type"],
            failure_class=item["failure_class"],
            gate=item["gate"],
        )
        metrics = run_gate(root, item["gate"], run_dir=cycle["run_dir"], runner=runner)
        if metrics["passed"]:
            queued_followups = _postprocess_completed_cycle(root, item, metrics)
            if queued_followups:
                metrics["postprocessed"] = {
                    "queued_followup_cycles": queued_followups,
                }
                _write_json(Path(cycle["run_dir"]) / "metrics.json", metrics)
        elif continue_on_mechanism_fail and _is_soft_mechanism_failure(item):
            queued_followups = _postprocess_failed_mechanism_cycle(root, item)
            continued_after_failures += 1
            metrics["postprocessed"] = {
                "continued_after_mechanism_failure": True,
                "queued_followup_cycles": queued_followups,
            }
            _write_json(Path(cycle["run_dir"]) / "metrics.json", metrics)
        outcome = "passed" if metrics["passed"] else "failed"
        close_cycle(
            root,
            cycle["run_id"],
            outcome=outcome,
            failure_class="none" if metrics["passed"] else item["failure_class"],
            metrics_path=Path(cycle["run_dir"]) / "metrics.json",
            summary=f"Gate {item['gate']} {outcome}.",
        )
        cycles_completed += 1

        if not metrics["passed"]:
            if continue_on_mechanism_fail and _is_soft_mechanism_failure(item):
                if max_cycles is None:
                    time.sleep(sleep_seconds)
                continue
            return {"cycles_completed": cycles_completed, "stopped_reason": "gate_failed"}
        if max_cycles is None:
            time.sleep(sleep_seconds)

    result = {"cycles_completed": cycles_completed, "stopped_reason": "max_cycles"}
    if continued_after_failures:
        result["continued_after_failures"] = continued_after_failures
    return result
