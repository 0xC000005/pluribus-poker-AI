import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import poker_ai.research.autoresearch as autoresearch
from poker_ai.research.autoresearch import (
    CommandResult,
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
    enqueue_paradigm_innovation_review,
    enqueue_methodology_review,
    enqueue_resolver_benchmark,
    enqueue_sd_cfr_mixture_falsification,
    enqueue_slumbot_smoke,
    enqueue_cycle,
    enqueue_warm_start_resolver_gate,
    init_state,
    new_cycle,
    readiness_report,
    register_research_knob,
    run_gate,
    set_incumbent,
    set_research_phase,
    synthesis_status,
    validate_methodology_review,
    validate_paradigm_innovation_review,
    validate_failure_synthesis,
    write_review_manifest,
)


def _ok_runner(command, timeout_seconds=None):
    return CommandResult(
        command=list(command),
        returncode=0,
        stdout="5 passed\n",
        stderr="",
        seconds=0.25,
    )


def _json_runner(command, timeout_seconds=None):
    return CommandResult(
        command=list(command),
        returncode=0,
        stdout='{"avg_chips_per_hand": 12.5, "passed": true}\n',
        stderr="",
        seconds=0.1,
    )


def _failed_runner(command, timeout_seconds=None):
    return CommandResult(
        command=list(command),
        returncode=1,
        stdout='{"passed": false, "reason": "controlled failure"}\n',
        stderr="controlled failure\n",
        seconds=0.1,
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_tiny_callback_cache(path: Path, *, root_label: str) -> None:
    features = np.zeros((2, 3), dtype=np.float32)
    belief = np.array(
        [
            [0.7, 0.3, 0.0, 0.4, 0.6, 0.0],
            [0.2, 0.8, 0.0, 0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    hero_values = np.array([[0.1, -0.2, 0.0], [0.3, -0.4, 0.0]], dtype=np.float32)
    villain_values = -hero_values
    masks = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.float32)
    records = [
        {"root_label": root_label, "action_str": "ck/b100c", "label": f"{root_label}-0"},
        {"root_label": root_label, "action_str": "ck/b200c", "label": f"{root_label}-1"},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        features=features,
        belief=belief,
        hero_values=hero_values,
        villain_values=villain_values,
        hero_masks=masks,
        villain_masks=masks,
        labels=np.array([f"{root_label}-0", f"{root_label}-1"]),
        records_json=json.dumps(records),
    )


def _set_open_research(root: Path) -> None:
    set_research_phase(root, phase="open_research", reason="test generic queue path")


def test_default_goal_includes_native_rollout_substrate_gate(tmp_path):
    init_state(tmp_path)

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    gate = goal["gates"]["native_rollout_parity_and_5x_throughput"]
    command_text = " ".join(gate["commands"][0])

    assert "scripts/eval_native_rollout_substrate.py" in command_text
    assert "--min-speedup 5.0" in command_text


def _set_neural_regret_phase(root: Path) -> None:
    set_research_phase(
        root,
        phase="neural_regret_field_resolving",
        reason="test legacy neural regret warm-start path",
    )


def _cli_set_open_research(script: Path, root: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "set-phase",
            "--phase",
            "open_research",
            "--reason",
            "test generic queue path",
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def _cli_set_neural_regret_phase(script: Path, root: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "set-phase",
            "--phase",
            "neural_regret_field_resolving",
            "--reason",
            "test legacy neural regret warm-start path",
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def _complete_review(root: Path, subject: str = "Method change") -> Path:
    queued = enqueue_methodology_review(
        root,
        subject=subject,
        trigger="method_change",
        claim="The reviewed change preserves evaluation hardness.",
    )
    review_dir = Path(queued["review_dir"])
    (review_dir / "review.md").write_text(
        "# Independent Verification\n\nVerdict: PASS\n\nArtifacts checked.\n",
        encoding="utf-8",
    )
    (review_dir / "related_work.md").write_text(
        "# Related Work\n\n"
        "Title: Deep CFR\n"
        "Source URL: https://arxiv.org/abs/1811.00164\n"
        "Source type: primary\n"
        "Transfers to this codebase: validates learned regret approximation.\n"
        "Does not transfer: does not validate Slumbot-specific patches.\n"
        "Smallest local test: root-disjoint resolver gate.\n",
        encoding="utf-8",
    )
    (review_dir / "benchmark_audit.md").write_text(
        "# Benchmark-Hacking Audit\n\nVerdict: PASS\n\nNo benchmark weakening.\n",
        encoding="utf-8",
    )
    (review_dir / "mechanism_review.md").write_text(
        "# Mechanism Review\n\n"
        "Learned object: public-belief model.\n"
        "Search boundary: turn/river resolver.\n"
        "Train distribution: train trace sessions.\n"
        "Eval distribution: held-out trace sessions.\n"
        "Falsifier: no held-out range-likelihood improvement.\n"
        "Pass action: allow opt-in integration gate.\n"
        "Fail action: keep incumbent path unchanged.\n"
        "Related-work delta: aligns with public-belief search literature.\n\n"
        "Verdict: PASS\n",
        encoding="utf-8",
    )
    (review_dir / "review_scope.json").write_text(
        json.dumps(
            {
                "changed_paths": ["scripts/play_slumbot.py"],
                "protected_hits": ["scripts/play_slumbot.py"],
                "mechanism": "public-belief integration gate",
                "mechanism_brief": {
                    "decision_object": "root action values and belief-conditioned policy",
                    "where_consumed": "turn/river resolver before live deployment",
                    "matched_control": "same-budget incumbent resolver without the new object",
                    "primary_decision_gate": "root-disjoint resolver A/B",
                    "retirement_criterion": "retire if held-out root decisions do not improve",
                    "flexibility_boundary": "architecture may change, but eval gates and controls may not",
                    "anti_benchmark_hack": "Slumbot is held out until internal gates pass",
                    "neural_policy_role": "main stochastic actor or unchanged evaluator-preserving path",
                    "cfr_role": "policy-improvement teacher or unchanged resolver control",
                    "stochastic_policy_contract": "do not deploy deterministic argmax without review",
                },
                "expected_gate": "held-out trace and resolver A/B",
                "decision_impact": (
                    "The change must improve root decisions per millisecond "
                    "or self-play checkpoint league strength before promotion."
                ),
                "decision_impact_gate": "root-disjoint resolver A/B then self-play league",
                "fallback_if_no_decision_impact": "leave incumbent path unchanged",
                "pass_action": "run bounded live opt-in check",
                "fail_action": "leave incumbent path unchanged",
            }
        ),
        encoding="utf-8",
    )
    (review_dir / "decision.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "decision": "proceed",
                "reason": "Review artifacts support a bounded opt-in diagnostic.",
                "sources": ["https://arxiv.org/abs/1811.00164"],
            }
        ),
        encoding="utf-8",
    )
    return review_dir


def test_init_state_creates_resumable_files_and_initial_queue(tmp_path):
    init_state(tmp_path)

    session = tmp_path / "autoresearch-session"
    assert (session / "poker_goal.json").is_file()
    assert (session / "poker_state.json").is_file()
    assert (session / "poker_knobs.tsv").is_file()
    assert (session / "poker_runs").is_dir()
    assert (session / "poker_reviews").is_dir()
    assert (tmp_path / "RESEARCH_LOG.md").is_file()

    state = _read_json(session / "poker_state.json")
    assert state["status"] == "ready"
    assert state["active_cycle"] is None
    assert state["hypothesis_queue"][0]["gate"] == "tier0"
    goal = _read_json(session / "poker_goal.json")
    assert "tier0" in goal["gates"]
    tier0_commands = goal["gates"]["tier0"]["commands"]
    assert any("scripts/test_feature_encoding.py" in command for command in tier0_commands)
    assert "eval-local" in goal["gates"]
    assert "eval-local-confidence" in goal["gates"]
    assert "eval-local-multiseed" in goal["gates"]
    assert "eval-incumbent-self-compare" in goal["gates"]
    assert "eval-head-to-head-self-compare" in goal["gates"]
    assert "eval-self-play-league-smoke" in goal["gates"]
    assert "eval-resolver-fixed-states" in goal["gates"]
    assert "slumbot-smoke" in goal["gates"]
    assert "slumbot-solver-smoke" in goal["gates"]
    assert goal["commit_policy"]["mode"] == "batch_by_research_objective"
    assert "methodology_review_required" in goal["review_policy"]["required_for"]
    assert goal["knob_policy"]["max_active_knobs"] == 5
    assert goal["agent_goal_contract"]["completion_rules"][
        "mechanism_failure_is_not_completion"
    ]
    assert "failure synthesis" in " ".join(
        goal["agent_goal_contract"]["soft_pivot_rules"]
    )
    assert "explicit user STOP" in " ".join(
        goal["agent_goal_contract"]["hard_stop_rules"]
    )
    assert "scripts/play_slumbot.py" in goal["objective_alignment_policy"]["protected_surfaces"]
    assert "scripts/eval_*.py" in goal["objective_alignment_policy"]["protected_surface_patterns"]
    assert "bitter lesson" in goal["objective"].lower()
    assert "tabula-rasa game-theoretic neural self-play" in goal["objective"].lower()
    assert "default solver-label imitation" in goal["objective"].lower()
    assert "stochastic neural policy" in goal["objective_alignment_policy"]["learning_philosophy"].lower()
    assert "r-nad/nashpg/mmd-style regularized self-play" in goal["objective_alignment_policy"]["learning_philosophy"].lower()
    assert "not as the default supervised target source" in goal["objective_alignment_policy"]["learning_philosophy"].lower()
    assert "k-best/historical population learning" in goal["objective_alignment_policy"]["active_method_target"].lower()
    assert "fresh stochastic policy/value networks" in goal["objective_alignment_policy"]["active_method_target"].lower()
    population_policy = goal["objective_alignment_policy"]["population_improvement_policy"]
    assert population_policy["active_loop"] == "alphaholdem_faithful_historical_k_best_population_learning"
    assert population_policy["current_local_incumbent_source"] == "autoresearch-session/poker_state.json.incumbent_checkpoint"
    assert population_policy["archive_policy"] == "maintain_k_best_and_historical_checkpoint_archive"
    assert population_policy["selection_rule"] == "train_against_full_archive_or_meta_strategy_not_single_parent"
    assert "do not repeat identical single-checkpoint response-oracle training" in population_policy["blocked_drift"]
    assert "do not repeat uniform-support response training without a new empirical-game objective" in population_policy["blocked_drift"]
    assert "research-log drift guard" in population_policy["drift_guard"].lower()
    assert "local target-consumer" in population_policy["drift_guard"].lower()
    assert "population-weak" in population_policy["drift_guard"].lower()
    assert "candidate versus empirical-game population/meta-policy" in population_policy["required_ladder"]
    assert "candidate versus prior support member that caused the last lower95 failure" in population_policy["required_ladder"]
    neural_policy = goal["objective_alignment_policy"]["neural_policy_iteration_policy"]
    assert neural_policy["current_status"] == "active_as_tabula_rasa_regularized_self_play"
    assert neural_policy["neural_policy_role"] == "main_stochastic_actor"
    assert neural_policy["cfr_role"] == "evaluator_or_optional_policy_improvement_control_not_default_teacher"
    assert neural_policy["deploy_policy_rule"] == "sample_mixed_strategy_not_argmax_by_default"
    assert neural_policy["generic_rl_algorithm_policy"] == "plug_in_maintained_libraries_only"
    assert "do not implement generic" in neural_policy["local_code_boundary"]
    assert neural_policy["control_gate_rule"] == "require_fixed_or_mixed_controls_per_generation"
    assert neural_policy["required_loop_flag"] == "--require-control-gate"
    assert "tianshou-rainbow" in neural_policy["required_control_kinds"]
    assert any("detached solver labels" in drift for drift in neural_policy["blocked_drift"])
    assert any("hand-roll generic" in drift for drift in neural_policy["blocked_drift"])
    assert "Tianshou PPO/Rainbow" in neural_policy["allowed_infrastructure"]
    league_policy = goal["objective_alignment_policy"]["self_play_league_policy"]
    assert league_policy["primary_role"] == "promotion_gate"
    assert "held-out external validation" in league_policy["slumbot_role"].lower()
    assert "Slumbot" in league_policy["blocked_before_internal_pass"]
    slumbot_policy = goal["objective_alignment_policy"]["slumbot_validation_policy"]
    assert slumbot_policy["max_smoke_hands_before_internal_pass"] == 50
    assert "self-play checkpoint league" in " ".join(slumbot_policy["confidence_requires"])
    assert "integration" in slumbot_policy["allowed_without_internal_pass"].lower()
    assert "checkpoint league positive lower95" in " ".join(goal["objective_alignment_policy"]["promotion_requires"])
    assert "held-out Slumbot validation" in " ".join(goal["objective_alignment_policy"]["promotion_requires"])
    assert goal["primary_metric"] == "checkpoint_league_lower_95_ci_chips_per_hand_vs_incumbent"
    assert "scripts/poker_autoresearch.py" in goal["objective_alignment_policy"]["protected_surfaces"]
    assert "scripts/run_neural_policy_iteration_loop.py" in goal["objective_alignment_policy"]["protected_surfaces"]
    assert "scripts/eval_mixed_policy_h2h.py" in goal["objective_alignment_policy"]["protected_surfaces"]
    assert "poker_ai/research/mixed_policy_h2h.py" in goal["objective_alignment_policy"]["protected_surfaces"]
    assert "scripts/eval_cfr_budget_frontier.py" in goal["objective_alignment_policy"]["protected_surfaces"]
    assert "scripts/eval_solver_budget_profiles.py" in goal["objective_alignment_policy"]["protected_surfaces"]
    assert "scripts/eval_solver_budget_selective_escalation.py" in goal["objective_alignment_policy"]["protected_surfaces"]
    assert "scripts/eval_solver_budget_boundary_predictor.py" in goal["objective_alignment_policy"]["protected_surfaces"]
    assert "scripts/analyze_cfr_trace_sequence_predictor.py" in goal["objective_alignment_policy"]["protected_surfaces"]
    assert "scripts/eval_slumbot_response_range_ev_gate.py" in goal["objective_alignment_policy"]["protected_surfaces"]
    assert "eval-response-range-counterfactual-ev" in goal["gates"]
    explicit_range = goal["objective_alignment_policy"]["explicit_range_policy"]
    assert explicit_range["default_role"] == "diagnostic_teacher_only"
    assert "local self-play" in explicit_range["mainline_replacement"].lower()
    assert "held-out Slumbot evaluation" in explicit_range["promotion_rule"]
    assert "scripts/run_frozen_best_response.py" in goal["objective_alignment_policy"]["protected_surfaces"]
    assert "scripts/eval_restricted_action_values.py" in goal["objective_alignment_policy"]["protected_surfaces"]
    assert goal["review_policy"]["requires_mechanism_review"] is True
    assert goal["review_policy"]["requires_review_manifest"] is True
    assert goal["review_policy"]["requires_review_scope"] is True
    assert goal["review_policy"]["requires_decision_impact_statement"] is True
    assert "decision_impact" in goal["review_policy"]["review_scope_fields"]
    assert "decision_impact_gate" in goal["review_policy"]["review_scope_fields"]
    assert "fallback_if_no_decision_impact" in goal["review_policy"]["review_scope_fields"]
    assert goal["synthesis_policy"]["experiments_per_synthesis"] == 5
    assert goal["innovation_policy"]["required_after_consecutive_failures"] == 2
    assert "Bayesian surprise" in goal["innovation_policy"]["selection_rule"]
    assert "first-principles reduction" in " ".join(
        goal["innovation_policy"]["required_fields"]
    ).lower()
    assert goal["research_phase"]["current"] == "self_play_policy_improvement"
    assert (
        "local self-play"
        in goal["research_phase"]["self_play_policy_improvement"]["approved_method"]
    )
    assert "regret/policy initializer" in goal["architecture_policy"]["approved_roles"]
    assert (tmp_path / "docs" / "research_protocols" / "poker_review_manifests").is_dir()


def test_default_goal_contract_names_current_frontier_and_mechanism_brief(tmp_path):
    init_state(tmp_path)

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    contract = goal["agent_goal_contract"]
    assert "alphaholdem-faithful historical/k-best population learning" in contract["current_frontier"]["mechanism"].lower()
    assert "small-game exact nashconv diagnostics" in contract["current_frontier"]["decision_object"].lower()
    assert "self-play checkpoint" in " ".join(contract["success_criteria"]).lower()
    assert "empirical game" in " ".join(contract["success_criteria"]).lower()
    assert "slumbot" in " ".join(contract["held_out_validation"]).lower()
    assert "loss-weight" in " ".join(contract["blocked_pivots"]).lower()
    assert "benchmark" in " ".join(contract["hard_stop_rules"]).lower()

    mechanism_policy = goal["mechanism_brief_policy"]
    assert mechanism_policy["required_for"] == [
        "methodology_review",
        "paradigm_innovation_review",
        "persistent_knob",
        "new_training_objective",
        "new_search_or_resolver_primitive",
    ]
    assert mechanism_policy["required_fields"] == [
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
    ]
    assert mechanism_policy["reject_if_missing"] is True


def test_methodology_review_rejects_missing_neural_policy_iteration_brief_fields(tmp_path):
    init_state(tmp_path)
    review_dir = _complete_review(tmp_path)
    scope_path = review_dir / "review_scope.json"
    scope = _read_json(scope_path)
    scope["mechanism_brief"].pop("neural_policy_role", None)
    scope["mechanism_brief"].pop("cfr_role", None)
    scope["mechanism_brief"].pop("stochastic_policy_contract", None)
    scope_path.write_text(json.dumps(scope), encoding="utf-8")

    result = validate_methodology_review(review_dir)

    assert result["passed"] is False
    assert any("neural_policy_role" in error for error in result["errors"])
    assert any("cfr_role" in error for error in result["errors"])
    assert any("stochastic_policy_contract" in error for error in result["errors"])


def test_methodology_review_rejects_missing_structured_mechanism_brief(tmp_path):
    init_state(tmp_path)
    review_dir = _complete_review(tmp_path)
    scope_path = review_dir / "review_scope.json"
    scope = _read_json(scope_path)
    scope.pop("mechanism_brief", None)
    scope_path.write_text(json.dumps(scope), encoding="utf-8")

    result = validate_methodology_review(review_dir)

    assert result["passed"] is False
    assert any("mechanism_brief" in error for error in result["errors"])


def test_init_state_prefers_repo_venv_python_for_default_gates(tmp_path):
    repo_python = tmp_path / ".venv" / "bin" / "python"
    repo_python.parent.mkdir(parents=True)
    repo_python.write_text("#!/bin/sh\n", encoding="utf-8")

    init_state(tmp_path)

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    assert goal["gates"]["tier0"]["commands"][0][0] == str(repo_python)
    assert goal["gates"]["eval-local"]["commands"][0][0] == str(repo_python)


def test_init_state_syncs_missing_default_gates_without_overwriting_history(tmp_path):
    init_state(tmp_path)
    goal_path = tmp_path / "autoresearch-session" / "poker_goal.json"
    goal = _read_json(goal_path)
    goal["gates"].pop("eval-local-confidence")
    goal["primary_metric"] = "lower_95_ci_mbb_per_hand_vs_incumbent"
    goal["objective_alignment_policy"]["visible_metric_rule"] = (
        "Local random and smoke metrics are diagnostics, not promotion targets."
    )
    goal["research_phase"]["self_play_policy_improvement"]["approved_method"] = (
        "Train from local self-play with optional generic search helpers."
    )
    goal_path.write_text(json.dumps(goal), encoding="utf-8")

    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["history"].append({"run_id": "kept"})
    state_path.write_text(json.dumps(state), encoding="utf-8")

    init_state(tmp_path)

    synced_goal = _read_json(goal_path)
    synced_state = _read_json(state_path)
    assert "eval-local-confidence" in synced_goal["gates"]
    assert (
        synced_goal["primary_metric"]
        == "checkpoint_league_lower_95_ci_chips_per_hand_vs_incumbent"
    )
    assert "Slumbot chip noise" in synced_goal["objective_alignment_policy"]["visible_metric_rule"]
    assert (
        "stochastic neural policy/value actor"
        in synced_goal["research_phase"]["self_play_policy_improvement"]["approved_method"]
    )
    assert synced_state["history"] == [{"run_id": "kept"}]


def test_init_state_migrates_existing_policy_and_knob_files(tmp_path):
    init_state(tmp_path)
    goal_path = tmp_path / "autoresearch-session" / "poker_goal.json"
    goal = _read_json(goal_path)
    del goal["commit_policy"]
    del goal["review_policy"]
    del goal["knob_policy"]
    goal["constraints"] = goal["constraints"][:1]
    goal_path.write_text(json.dumps(goal), encoding="utf-8")
    knob_path = tmp_path / "autoresearch-session" / "poker_knobs.tsv"
    knob_path.write_text(
        "name\tdefault\tfailure_class\trationale\tremoval_criterion\n"
        "legacy\t1\tstrategy_quality\told rationale\tretire when false\n",
        encoding="utf-8",
    )

    init_state(tmp_path)

    migrated_goal = _read_json(goal_path)
    assert migrated_goal["commit_policy"]["mode"] == "batch_by_research_objective"
    assert migrated_goal["review_policy"]["requires_related_work"] is True
    assert migrated_goal["review_policy"]["requires_mechanism_review"] is True
    assert migrated_goal["review_policy"]["requires_review_scope"] is True
    assert migrated_goal["synthesis_policy"]["experiments_per_synthesis"] == 5
    assert migrated_goal["knob_policy"]["max_active_knobs"] == 5
    assert (
        "run methodology review before method, promotion, or persistent knob changes"
        in migrated_goal["constraints"]
    )
    knob_lines = knob_path.read_text(encoding="utf-8").splitlines()
    assert knob_lines[0].startswith("name\tstatus\tdefault\tfailure_class\tmechanism")
    assert knob_lines[1].startswith("legacy\tactive\t1\tstrategy_quality\t")


def test_init_state_syncs_missing_default_commands_without_overwriting_history(tmp_path):
    init_state(tmp_path)
    goal_path = tmp_path / "autoresearch-session" / "poker_goal.json"
    goal = _read_json(goal_path)
    goal["gates"]["tier0"]["commands"] = [
        command
        for command in goal["gates"]["tier0"]["commands"]
        if "scripts/test_feature_encoding.py" not in command
    ]
    goal_path.write_text(json.dumps(goal), encoding="utf-8")

    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["history"].append({"run_id": "kept"})
    state_path.write_text(json.dumps(state), encoding="utf-8")

    init_state(tmp_path)

    synced_goal = _read_json(goal_path)
    tier0_commands = synced_goal["gates"]["tier0"]["commands"]
    synced_state = _read_json(state_path)
    assert any("scripts/test_feature_encoding.py" in command for command in tier0_commands)
    assert synced_state["history"] == [{"run_id": "kept"}]


def test_readiness_report_detects_missing_state_and_stop_file(tmp_path):
    missing = readiness_report(tmp_path)
    assert missing["ready"] is False
    assert "autoresearch-session/poker_goal.json" in missing["missing"]

    init_state(tmp_path)
    ready = readiness_report(tmp_path)
    assert ready["ready"] is True
    assert ready["stop_requested"] is False

    (tmp_path / "autoresearch-session" / "STOP").write_text("pause\n", encoding="utf-8")
    stopped = readiness_report(tmp_path)
    assert stopped["ready"] is False
    assert stopped["stop_requested"] is True


def test_run_gate_writes_machine_readable_metrics(tmp_path):
    init_state(tmp_path)
    run_dir = tmp_path / "autoresearch-session" / "poker_runs" / "cycle_001"
    run_dir.mkdir()

    result = run_gate(tmp_path, "tier0", run_dir=run_dir, runner=_ok_runner)

    assert result["passed"] is True
    assert result["gate"] == "tier0"
    assert result["commands"][0]["returncode"] == 0
    assert result["commands"][0]["seconds"] == 0.25
    metrics = _read_json(run_dir / "metrics.json")
    assert metrics["passed"] is True
    assert metrics["commands"][0]["stdout"] == "5 passed\n"


def test_run_gate_parses_json_stdout_into_metrics(tmp_path):
    init_state(tmp_path)
    run_dir = tmp_path / "autoresearch-session" / "poker_runs" / "cycle_002"
    run_dir.mkdir()

    result = run_gate(tmp_path, "tier0", run_dir=run_dir, runner=_json_runner)

    assert result["commands"][0]["stdout_json"]["avg_chips_per_hand"] == 12.5
    metrics = _read_json(run_dir / "metrics.json")
    assert metrics["commands"][0]["stdout_json"]["passed"] is True


def test_new_and_close_cycle_update_state_and_research_log(tmp_path):
    init_state(tmp_path)
    cycle = new_cycle(
        tmp_path,
        hypothesis="Tier 0 should pass before unattended work.",
        cycle_type="experiment",
        failure_class="eval_invalid",
        gate="tier0",
    )

    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    assert state["active_cycle"]["run_id"] == cycle["run_id"]
    assert (Path(cycle["run_dir"]) / "cycle.json").is_file()

    metrics_path = Path(cycle["run_dir"]) / "metrics.json"
    metrics_path.write_text(json.dumps({"passed": True, "gate": "tier0"}), encoding="utf-8")
    close_cycle(
        tmp_path,
        cycle["run_id"],
        outcome="passed",
        failure_class="none",
        metrics_path=metrics_path,
        summary="Tier 0 integrity gate passed.",
    )

    closed_state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    assert closed_state["active_cycle"] is None
    assert closed_state["history"][-1]["run_id"] == cycle["run_id"]
    assert closed_state["last_metrics"]["passed"] is True
    log_text = (tmp_path / "RESEARCH_LOG.md").read_text(encoding="utf-8")
    assert "Tier 0 integrity gate passed." in log_text
    assert "- Metrics file:" in log_text
    assert '"commands"' not in log_text


def test_close_cycle_accepts_external_metrics_path(tmp_path):
    init_state(tmp_path)
    cycle = new_cycle(
        tmp_path,
        hypothesis="External metrics artifacts can still be referenced.",
        cycle_type="experiment",
        failure_class="eval_invalid",
        gate="tier0",
    )
    external_metrics = tmp_path.parent / "external_metrics.json"
    external_metrics.write_text(json.dumps({"passed": True, "gate": "tier0"}), encoding="utf-8")

    close_cycle(
        tmp_path,
        cycle["run_id"],
        outcome="passed",
        failure_class="none",
        metrics_path=external_metrics,
        summary="External metrics path recorded.",
    )

    log_text = (tmp_path / "RESEARCH_LOG.md").read_text(encoding="utf-8")
    assert str(external_metrics) in log_text


def test_continuous_processes_one_queued_cycle_and_honors_stop_file(tmp_path):
    init_state(tmp_path)

    summary = continuous(tmp_path, max_cycles=1, runner=_ok_runner, sleep_seconds=0.0)
    assert summary["cycles_completed"] == 1
    assert summary["stopped_reason"] == "max_cycles"

    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    assert state["history"][-1]["outcome"] == "passed"

    (tmp_path / "autoresearch-session" / "STOP").write_text("pause\n", encoding="utf-8")
    stopped = continuous(tmp_path, max_cycles=1, runner=_ok_runner, sleep_seconds=0.0)
    assert stopped["cycles_completed"] == 0
    assert stopped["stopped_reason"] == "stop_file"


def test_unbounded_continuous_idles_when_queue_is_empty(tmp_path):
    init_state(tmp_path)
    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["hypothesis_queue"] = []
    state_path.write_text(json.dumps(state), encoding="utf-8")

    summary = continuous(
        tmp_path,
        runner=_ok_runner,
        sleep_seconds=0.0,
        max_idle_checks=1,
    )
    assert summary["cycles_completed"] == 0
    assert summary["stopped_reason"] == "idle_limit"


def test_continuous_continue_on_mechanism_fail_queues_pivot_followups(tmp_path):
    init_state(tmp_path)
    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["hypothesis_queue"] = []
    state_path.write_text(json.dumps(state), encoding="utf-8")
    enqueue_cycle(
        tmp_path,
        hypothesis="Bounded learned hook should improve held-out root decisions.",
        cycle_type="experiment",
        failure_class="search_state_update_misalignment",
        gate="tier0",
    )

    summary = continuous(
        tmp_path,
        max_cycles=1,
        runner=_failed_runner,
        sleep_seconds=0.0,
        continue_on_mechanism_fail=True,
    )

    assert summary["cycles_completed"] == 1
    assert summary["stopped_reason"] == "max_cycles"
    assert summary["continued_after_failures"] == 1
    state_after = _read_json(state_path)
    assert state_after["history"][-1]["outcome"] == "failed"
    queued_types = [item["type"] for item in state_after["hypothesis_queue"]]
    assert queued_types[:2] == ["synthesis", "innovation_review"]
    metrics_path = Path(state_after["last_metrics"]["commands"][0]["command"][0]).parent
    assert metrics_path
    run_metrics = _read_json(
        Path(state_after["history"][-1]["run_dir"]) / "metrics.json"
    )
    assert run_metrics["postprocessed"]["continued_after_mechanism_failure"] is True
    assert len(run_metrics["postprocessed"]["queued_followup_cycles"]) == 2


def test_continuous_continue_on_mechanism_fail_still_stops_on_integrity_failure(tmp_path):
    init_state(tmp_path)
    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["hypothesis_queue"] = []
    state_path.write_text(json.dumps(state), encoding="utf-8")
    enqueue_cycle(
        tmp_path,
        hypothesis="Tier 0 integrity failure should stop the unattended loop.",
        cycle_type="experiment",
        failure_class="eval_invalid",
        gate="tier0",
    )

    summary = continuous(
        tmp_path,
        max_cycles=1,
        runner=_failed_runner,
        sleep_seconds=0.0,
        continue_on_mechanism_fail=True,
    )

    assert summary["cycles_completed"] == 1
    assert summary["stopped_reason"] == "gate_failed"
    state_after = _read_json(state_path)
    assert state_after["hypothesis_queue"] == []


def test_enqueue_cycle_adds_work_without_opening_active_cycle(tmp_path):
    init_state(tmp_path)
    queued = enqueue_cycle(
        tmp_path,
        hypothesis="Run Tier 0 again after queue support.",
        cycle_type="experiment",
        failure_class="eval_invalid",
        gate="tier0",
    )

    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    assert state["active_cycle"] is None
    assert state["hypothesis_queue"][-1]["id"] == queued["id"]
    assert state["hypothesis_queue"][-1]["hypothesis"] == "Run Tier 0 again after queue support."


def test_enqueue_methodology_review_creates_templates_and_related_work_gate(tmp_path):
    init_state(tmp_path)

    queued = enqueue_methodology_review(
        tmp_path,
        subject="Policy-head Slumbot transfer interpretation",
        trigger="surprising_slumbot_result",
        claim="The policy-head checkpoint transfers to Slumbot.",
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    gate = goal["gates"][queued["gate"]]
    command = gate["commands"][0]
    review_dir = Path(queued["review_dir"])
    assert queued["gate"].startswith("methodology-review-")
    assert queued["requires_independent_verifier"] is True
    assert queued["requires_related_work"] is True
    assert queued["requires_benchmark_audit"] is True
    assert state["hypothesis_queue"][-1]["gate"] == queued["gate"]
    assert (review_dir / "review.md").is_file()
    assert (review_dir / "related_work.md").is_file()
    assert (review_dir / "benchmark_audit.md").is_file()
    assert (review_dir / "mechanism_review.md").is_file()
    assert (review_dir / "team_review.md").is_file()
    assert (review_dir / "review_scope.json").is_file()
    assert (review_dir / "decision.json").is_file()
    assert "scripts/poker_methodology_review.py" in command
    assert "--require-complete" in command


def test_methodology_review_validator_requires_independent_review_and_related_work(tmp_path):
    init_state(tmp_path)
    queued = enqueue_methodology_review(
        tmp_path,
        subject="New search objective",
        trigger="method_change",
        claim="Changing the search target is justified.",
    )
    review_dir = Path(queued["review_dir"])
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_methodology_review.py"

    pending = subprocess.run(
        [sys.executable, str(script), "--review-dir", str(review_dir), "--require-complete"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert pending.returncode == 1
    assert "review.md is still pending" in pending.stderr
    assert "benchmark_audit.md is still pending" in pending.stderr
    assert "mechanism_review.md is still pending" in pending.stderr

    (review_dir / "review.md").write_text(
        "# Independent Verification\n\n"
        "Verdict: PARTIAL\n\n"
        "Checked actual artifacts and found mixed support.\n",
        encoding="utf-8",
    )
    (review_dir / "related_work.md").write_text(
        "# Related Work\n\n"
        "Title: Deep CFR\n"
        "Source URL: https://arxiv.org/abs/1811.00164\n"
        "Source type: primary\n"
        "Transfers to this codebase: canonical learned regret baseline.\n"
        "Does not transfer: does not prove Slumbot transfer.\n"
        "Smallest local test: root-disjoint resolver A/B.\n",
        encoding="utf-8",
    )
    (review_dir / "benchmark_audit.md").write_text(
        "# Benchmark-Hacking Audit\n\n"
        "Protected surfaces checked: no eval harness weakening.\n\n"
        "Verdict: PASS\n",
        encoding="utf-8",
    )
    (review_dir / "mechanism_review.md").write_text(
        "# Mechanism Review\n\n"
        "- Learned object: public-belief value network.\n"
        "- Search boundary: resolver leaf state.\n"
        "- Train distribution: sampled public roots.\n"
        "- Eval distribution: root-disjoint heldout public roots.\n"
        "- Falsifier: fails heldout leaf A/B drift.\n"
        "- Pass action: queue limited implementation.\n"
        "- Fail action: revise target distribution.\n"
        "- Related-work delta: differs from Deep CFR by training values at search boundary.\n\n"
        "Verdict: PASS\n",
        encoding="utf-8",
    )
    (review_dir / "decision.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "decision": "gather_more_evidence",
                "reason": "Local and live evidence disagree.",
                "sources": ["https://arxiv.org/abs/1811.00164"],
            }
        ),
        encoding="utf-8",
    )
    (review_dir / "review_scope.json").write_text(
        json.dumps(
                {
                    "changed_paths": ["poker_ai/deep_cfr/networks.py"],
                    "protected_hits": [],
                    "mechanism": "search target change",
                    "mechanism_brief": {
                        "decision_object": "held-out root policy and action-value quality",
                        "where_consumed": "resolver target generation and candidate training",
                        "matched_control": "same-budget resolver A/B without the target change",
                        "primary_decision_gate": "root-disjoint resolver A/B",
                        "retirement_criterion": "retire if root decisions do not improve",
                        "flexibility_boundary": "target implementation may change, gate semantics may not",
                        "anti_benchmark_hack": "promotion still requires checkpoint league evidence",
                        "neural_policy_role": "main stochastic actor trained from self-play",
                        "cfr_role": "mixed-strategy policy-improvement teacher",
                        "stochastic_policy_contract": "train/deploy mixed strategies, not default argmax",
                    },
                    "expected_gate": "root-disjoint resolver A/B",
                    "decision_impact": (
                        "The search target must improve held-out root decisions "
                        "per millisecond before it can affect checkpoint league promotion."
                ),
                "decision_impact_gate": "root-disjoint resolver A/B",
                "fallback_if_no_decision_impact": "revise target distribution",
                "pass_action": "queue bounded implementation",
                "fail_action": "revise target",
            }
        ),
        encoding="utf-8",
    )

    complete = subprocess.run(
        [sys.executable, str(script), "--review-dir", str(review_dir), "--require-complete"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert complete.returncode == 0
    assert json.loads(complete.stdout)["passed"] is True


def test_write_review_manifest_creates_tracked_digest_for_ignored_bundle(tmp_path):
    init_state(tmp_path)
    queued = enqueue_methodology_review(
        tmp_path,
        subject="Callback-state DCVN calibration",
        trigger="mechanism_change",
        claim="A calibration audit should precede model scaling.",
    )
    review_dir = Path(queued["review_dir"])
    (review_dir / "review.md").write_text(
        "# Independent Verification\n\nVerdict: PASS\n\nArtifacts checked.\n",
        encoding="utf-8",
    )
    (review_dir / "related_work.md").write_text(
        "# Related Work\n\n"
        "Title: ReBeL\n"
        "Source URL: https://arxiv.org/abs/2007.13544\n"
        "Source type: primary\n"
        "Transfers to this codebase: public-belief search framing.\n"
        "Does not transfer: full implementation cost is not validated here.\n"
        "Smallest local test: methodology manifest digest.\n",
        encoding="utf-8",
    )
    (review_dir / "benchmark_audit.md").write_text(
        "# Benchmark-Hacking Audit\n\nVerdict: PASS\n\nNo protected-surface drift.\n",
        encoding="utf-8",
    )
    (review_dir / "mechanism_review.md").write_text(
        "# Mechanism Review\n\n"
        "Learned object: callback-state dual counterfactual values.\n"
        "Search boundary: CFR showdown leaf callback.\n"
        "Train distribution: callback states queried by train roots.\n"
        "Eval distribution: root-disjoint callback states.\n"
        "Falsifier: worse than zero-CFV baseline.\n"
        "Pass action: test calibrated loss.\n"
        "Fail action: abandon scale-up.\n"
        "Related-work delta: follows value-at-search-boundary papers.\n\n"
        "Verdict: PASS\n",
        encoding="utf-8",
    )
    (review_dir / "decision.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "decision": "revise",
                "reason": "Need calibration before scaling.",
                "sources": ["https://arxiv.org/abs/2007.13544"],
            }
        ),
        encoding="utf-8",
    )
    (review_dir / "review_scope.json").write_text(
        json.dumps(
            {
                "changed_paths": ["poker_ai/research/autoresearch.py"],
                "protected_hits": [],
                "mechanism": "review manifest digest",
                "expected_gate": "manifest write",
                "decision_impact": (
                    "The manifest gate preserves the review trail for root decision "
                    "and self-play league promotion claims."
                ),
                "decision_impact_gate": "objective audit",
                "fallback_if_no_decision_impact": "keep review local only",
                "pass_action": "use manifest in objective audit",
                "fail_action": "keep review local only",
            }
        ),
        encoding="utf-8",
    )

    manifest = write_review_manifest(tmp_path, review_dir)

    manifest_path = tmp_path / manifest["manifest_path"]
    saved = _read_json(manifest_path)
    assert manifest_path.is_file()
    assert saved["decision"] == "revise"
    assert any(record["path"] == "mechanism_review.md" for record in saved["files"])
    assert all(len(record["sha256"]) == 64 for record in saved["files"])


def test_synthesis_status_and_enqueue_failure_synthesis(tmp_path):
    init_state(tmp_path)
    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["history"] = [
        {"run_id": f"run-{idx}", "type": "experiment", "outcome": "failed"}
        for idx in range(5)
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    status = synthesis_status(tmp_path)
    queued = enqueue_failure_synthesis(tmp_path, subject="callback-state failures")

    assert status["due"] is True
    assert status["experiments_since_synthesis"] == 5
    assert queued["type"] == "synthesis"
    synthesis_dir = Path(queued["synthesis_dir"])
    pending = validate_failure_synthesis(synthesis_dir)
    assert pending["passed"] is False
    assert "synthesis.md is still pending" in pending["errors"]


def test_enqueue_paradigm_innovation_review_requires_novelty_artifacts(tmp_path):
    init_state(tmp_path)

    queued = enqueue_paradigm_innovation_review(
        tmp_path,
        subject="learned search update operator",
        anomaly="oracle eta helps but learned selectors fail held-out roots",
    )

    assert queued["type"] == "innovation_review"
    review_dir = Path(queued["innovation_dir"])
    assert (review_dir / "innovation.md").is_file()
    assert (review_dir / "thought_experiments.md").is_file()
    assert (review_dir / "related_work.md").is_file()
    assert (review_dir / "decision.json").is_file()

    pending = validate_paradigm_innovation_review(review_dir)
    assert pending["passed"] is False
    assert "innovation.md is still pending" in pending["errors"]
    assert "thought_experiments.md is still pending" in pending["errors"]
    assert "related_work.md is still pending" in pending["errors"]

    (review_dir / "innovation.md").write_text(
        "# Paradigm Innovation Review\n\n"
        "Anomaly ledger: oracle eta helps while selectors fail root-disjoint roots.\n"
        "Current-practice limit: static trace-to-policy predictors underfit search dynamics.\n"
        "First-principles reduction: learn a closed-loop search-update operator.\n"
        "Cross-paradigm analogy: control feedback stabilizes iterative updates.\n"
        "Novel mechanism: recurrent update field consumed inside the resolver.\n"
        "Bitter-lesson alignment: reusable learned computation replaces hand rules.\n"
        "Smallest decisive test: root-disjoint CFR5+operator must beat CFR10 L1.\n"
        "Falsifier: it fails held-out root decision L1 or top-action agreement.\n"
        "Verdict: proceed\n",
        encoding="utf-8",
    )
    (review_dir / "thought_experiments.md").write_text(
        "# Thought Experiments\n\n"
        "Mechanism stress test: if the operator sees an adversarial range, it should abstain.\n"
        "Failure thought experiment: if it only predicts final policy, it should lose to CFR10.\n"
        "Transfer thought experiment: it should hold on root-disjoint Slumbot traces.\n"
        "Compute thought experiment: it should shift work from CPU rollout to CUDA batched updates.\n",
        encoding="utf-8",
    )
    (review_dir / "related_work.md").write_text(
        "# Related Work\n\n"
        "Title: AutoDiscovery\n"
        "Source URL: https://arxiv.org/abs/2507.00310\n"
        "Source type: primary\n"
        "Transfers to this codebase: use surprise to choose mechanism-level anomalies.\n"
        "Does not transfer: does not validate poker solver updates directly.\n"
        "Novelty delta: closed-loop search update is different from static selector tuning.\n"
        "Smallest local test: root-disjoint resolver A/B against CFR10.\n",
        encoding="utf-8",
    )
    (review_dir / "decision.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "decision": "proceed",
                "reason": "The novelty sprint has a falsifiable resolver gate.",
                "sources": ["https://arxiv.org/abs/2507.00310"],
            }
        ),
        encoding="utf-8",
    )

    passed = validate_paradigm_innovation_review(review_dir)
    assert passed["passed"] is True
    assert passed["decision"] == "proceed"


def test_synthesis_status_does_not_count_innovation_reviews_as_experiments(tmp_path):
    init_state(tmp_path)
    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["history"] = [
        {"type": "innovation_review", "run_id": "innovation"},
        {"type": "experiment", "run_id": "experiment-1"},
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    status = synthesis_status(tmp_path)

    assert status["experiments_since_synthesis"] == 1
    assert status["due"] is False


def test_continuous_stops_when_synthesis_is_due_before_more_experiments(tmp_path):
    init_state(tmp_path)
    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["history"] = [
        {"run_id": f"run-{idx}", "type": "experiment", "outcome": "failed"}
        for idx in range(5)
    ]
    state["hypothesis_queue"] = [
        {
            "id": "next-exp",
            "type": "experiment",
            "hypothesis": "Another model run should not start before synthesis.",
            "failure_class": "strategy_quality",
            "gate": "tier0",
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = continuous(tmp_path, max_cycles=1, runner=_ok_runner, sleep_seconds=0.0)

    assert result["cycles_completed"] == 0
    assert result["stopped_reason"] == "synthesis_due"
    state_after = _read_json(state_path)
    assert state_after["hypothesis_queue"][0]["id"] == "next-exp"


def test_continuous_allows_synthesis_when_synthesis_is_due(tmp_path):
    init_state(tmp_path)
    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["history"] = [
        {"run_id": f"run-{idx}", "type": "experiment", "outcome": "failed"}
        for idx in range(5)
    ]
    state["hypothesis_queue"] = []
    state_path.write_text(json.dumps(state), encoding="utf-8")
    enqueue_failure_synthesis(tmp_path, subject="five failed experiments")

    result = continuous(tmp_path, max_cycles=1, runner=_ok_runner, sleep_seconds=0.0)

    assert result == {"cycles_completed": 1, "stopped_reason": "max_cycles"}


def test_continuous_runs_queued_synthesis_ahead_of_blocked_experiment(tmp_path):
    init_state(tmp_path)
    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["history"] = [
        {"run_id": f"run-{idx}", "type": "experiment", "outcome": "failed"}
        for idx in range(5)
    ]
    state["hypothesis_queue"] = [
        {
            "id": "next-exp",
            "type": "experiment",
            "hypothesis": "This experiment should wait until synthesis runs.",
            "failure_class": "strategy_quality",
            "gate": "tier0",
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    enqueue_failure_synthesis(tmp_path, subject="five failed experiments")

    result = continuous(tmp_path, max_cycles=1, runner=_ok_runner, sleep_seconds=0.0)

    state_after = _read_json(state_path)
    assert result == {"cycles_completed": 1, "stopped_reason": "max_cycles"}
    assert state_after["history"][-1]["type"] == "synthesis"
    assert state_after["hypothesis_queue"][0]["id"] == "next-exp"


def test_continuous_skips_completed_queued_synthesis_before_next_cycle(tmp_path):
    init_state(tmp_path)
    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    synthesis_dir = (
        tmp_path
        / "autoresearch-session"
        / "poker_reviews"
        / "already-complete-synthesis"
    )
    synthesis_dir.mkdir(parents=True)
    (synthesis_dir / "synthesis.md").write_text(
        "# Failure Synthesis\n\n"
        "- Current causal model: already complete\n"
        "- Retired hypotheses: stale synthesis queue entry\n"
        "- Live hypotheses: run the next queued cycle\n"
        "- Single next test: normal queued experiment\n"
        "\nVerdict: REVISE\n",
        encoding="utf-8",
    )
    (synthesis_dir / "decision.json").write_text(
        json.dumps({"decision": "revise", "reason": "complete", "sources": []}),
        encoding="utf-8",
    )
    state = _read_json(state_path)
    state["hypothesis_queue"] = [
        {
            "id": "completed-synthesis",
            "type": "synthesis",
            "hypothesis": "This completed synthesis should not replay.",
            "failure_class": "eval_invalid",
            "gate": "tier0",
            "synthesis_dir": str(synthesis_dir),
        },
        {
            "id": "next-exp",
            "type": "experiment",
            "hypothesis": "This experiment should run after stale synthesis is skipped.",
            "failure_class": "strategy_quality",
            "gate": "tier0",
        },
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = continuous(tmp_path, max_cycles=1, runner=_ok_runner, sleep_seconds=0.0)

    state_after = _read_json(state_path)
    assert result == {"cycles_completed": 1, "stopped_reason": "max_cycles"}
    assert state_after["history"][-1]["type"] == "experiment"
    assert state_after["history"][-1]["hypothesis"] == (
        "This experiment should run after stale synthesis is skipped."
    )
    assert state_after["hypothesis_queue"] == []


def _xdo_target_drift_history():
    return [
        {
            "run_id": "run-xdo-1",
            "type": "experiment",
            "outcome": "failed",
            "failure_class": "mechanism_transfer",
            "gate": "duplicate_swapped_parent_h2h",
            "hypothesis": "All-street XDO NPI policy consumer should transfer local search targets to parent H2H.",
            "summary": "NPI consumer fit the target dataset but failed parent H2H.",
        },
        {
            "run_id": "run-xdo-2",
            "type": "experiment",
            "outcome": "failed",
            "failure_class": "mechanism_transfer",
            "gate": "empirical_game_meta_strategy",
            "hypothesis": "Closed-loop XDO policy consumer should add useful population value.",
            "summary": "Empirical game reverted to the frozen Rainbow incumbent.",
        },
    ]


def test_research_drift_status_blocks_repeated_target_consumer_after_log_failures(tmp_path):
    init_state(tmp_path)
    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["history"] = _xdo_target_drift_history()
    state_path.write_text(json.dumps(state), encoding="utf-8")

    report = autoresearch.research_drift_status(
        tmp_path,
        next_cycle={
            "type": "experiment",
            "failure_class": "mechanism_transfer",
            "gate": "xdo-target-consumer-parent-h2h",
            "hypothesis": "Another XDO NPI target consumer should improve after small target changes.",
        },
    )

    assert report["blocked"] is True
    assert report["requires_review"] is True
    assert report["matched_family"] == "local_target_consumer"
    assert "review the research log" in report["blockers"][0]


def test_continuous_stops_on_objective_drift_before_repeated_target_consumer(tmp_path):
    init_state(tmp_path)
    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["history"] = _xdo_target_drift_history()
    state["hypothesis_queue"] = [
        {
            "id": "repeat-xdo",
            "type": "experiment",
            "hypothesis": "Repeat the XDO NPI target consumer with another small target variant.",
            "failure_class": "mechanism_transfer",
            "gate": "xdo-target-consumer-parent-h2h",
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = continuous(tmp_path, max_cycles=1, runner=_ok_runner, sleep_seconds=0.0)

    assert result["cycles_completed"] == 0
    assert result["stopped_reason"] == "objective_drift_review_required"
    state_after = _read_json(state_path)
    assert state_after["hypothesis_queue"][0]["id"] == "repeat-xdo"


def test_continuous_allows_rl_response_oracle_after_target_consumer_drift(tmp_path):
    init_state(tmp_path)
    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["history"] = _xdo_target_drift_history()
    state["hypothesis_queue"] = [
        {
            "id": "rl-response",
            "type": "experiment",
            "hypothesis": "Maintained Tianshou Rainbow response oracle should train as the stochastic neural actor against the empirical-game meta-policy.",
            "failure_class": "strategy_quality",
            "gate": "tier0",
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = continuous(tmp_path, max_cycles=1, runner=_ok_runner, sleep_seconds=0.0)

    assert result == {"cycles_completed": 1, "stopped_reason": "max_cycles"}
    state_after = _read_json(state_path)
    assert state_after["history"][-1]["run_id"] != "rl-response"
    assert state_after["history"][-1]["hypothesis"].startswith("Maintained Tianshou Rainbow response oracle")


def test_research_phase_blocks_training_and_slumbot_until_calibration_audit(tmp_path):
    init_state(tmp_path)
    set_research_phase(
        tmp_path,
        phase="callback_state_calibration_debug",
        reason="scale-up failed",
    )
    model = tmp_path / "models" / "candidate.pt"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")

    try:
        enqueue_gpu_training(tmp_path, n_iterations=1, n_traversals=1)
    except RuntimeError as exc:
        assert "callback_state_calibration_debug blocks" in str(exc)
    else:
        raise AssertionError("training should be blocked during calibration debug")

    try:
        enqueue_slumbot_smoke(tmp_path, model)
    except RuntimeError as exc:
        assert "callback_state_calibration_debug blocks" in str(exc)
    else:
        raise AssertionError("Slumbot smoke should be blocked during calibration debug")

    train = tmp_path / "train.npz"
    holdout = tmp_path / "holdout.npz"
    _write_tiny_callback_cache(train, root_label="train-root")
    _write_tiny_callback_cache(holdout, root_label="holdout-root")
    queued = enqueue_callback_calibration_audit(
        tmp_path,
        train_dual_cache=train,
        holdout_dual_cache=holdout,
    )
    assert queued["type"] == "calibration_audit"


def test_neural_regret_phase_blocks_old_training_path_but_allows_warm_start_knob(tmp_path):
    init_state(tmp_path)
    set_research_phase(
        tmp_path,
        phase="neural_regret_field_resolving",
        reason="test legacy neural regret guard",
    )
    model = tmp_path / "models" / "candidate.pt"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")

    try:
        enqueue_gpu_training(tmp_path, n_iterations=1, n_traversals=1)
    except RuntimeError as exc:
        assert "neural_regret_field_resolving blocks" in str(exc)
    else:
        raise AssertionError("generic GPU training should be blocked in neural regret phase")

    try:
        enqueue_slumbot_smoke(tmp_path, model)
    except RuntimeError as exc:
        assert "neural_regret_field_resolving blocks" in str(exc)
    else:
        raise AssertionError("Slumbot smoke should be blocked before warm-start resolver gate")

    row = register_research_knob(
        tmp_path,
        name="regret_field_initializer_strength",
        default="1.0",
        failure_class="search_quality",
        mechanism="Test one regret warm-start initializer strength inside the resolver.",
        rationale="Isolates the neural public-belief warm-start mechanism.",
        removal_criterion="Retire if the warm-start resolver gate does not improve over vanilla CFR.",
        review_dir=_complete_review(tmp_path, subject="Regret field knob"),
    )
    assert row["name"] == "regret_field_initializer_strength"
    assert row["review_id"]


def test_enqueue_warm_start_resolver_gate_is_allowed_in_neural_regret_phase(tmp_path):
    init_state(tmp_path)
    set_research_phase(
        tmp_path,
        phase="neural_regret_field_resolving",
        reason="test legacy neural regret warm-start gate",
    )
    checkpoint = tmp_path / "models" / "joint.pt"
    cases = tmp_path / "cases.json"
    cache = tmp_path / "cache.npz"
    train = tmp_path / "train.npz"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    cases.write_text('{"cases": []}', encoding="utf-8")
    cache.write_bytes(b"cache")
    train.write_bytes(b"train")

    queued = enqueue_warm_start_resolver_gate(
        tmp_path,
        checkpoint=checkpoint,
        cases_json=cases,
        cfv_cache=cache,
        train_labels_npz=train,
        start_index=7,
        limit=11,
        low_iterations=3,
        reference_iterations=9,
        min_evaluated=5,
        output_json="autoresearch-session/warm/metrics.json",
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    command = goal["gates"][queued["gate"]]["commands"][0]
    assert queued["type"] == "warm_start_resolver_gate"
    assert state["hypothesis_queue"][-1]["gate"] == queued["gate"]
    assert "scripts/eval_joint_pbs_policy_warm_start.py" in command
    assert "--train-labels-npz" in command
    assert str(train) in command
    assert "--min-evaluated" in command
    assert "5" in command


def test_exact_gpu_resolving_phase_blocks_retired_warm_start_gate(tmp_path):
    init_state(tmp_path)
    phase = set_research_phase(
        tmp_path,
        phase="exact_gpu_resolving",
        reason="matched-latency warm starts lost to uniform CUDA CFR",
    )
    checkpoint = tmp_path / "models" / "warm.pt"
    cases = tmp_path / "cases.json"
    cache = tmp_path / "cache.npz"
    train = tmp_path / "train.npz"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    cases.write_text('{"cases": []}', encoding="utf-8")
    cache.write_bytes(b"cache")
    train.write_bytes(b"train")

    assert phase["current"] == "exact_gpu_resolving"
    assert "exact GPU" in phase["exact_gpu_resolving"]["approved_method"]

    try:
        enqueue_warm_start_resolver_gate(
            tmp_path,
            checkpoint=checkpoint,
            cases_json=cases,
            cfv_cache=cache,
            train_labels_npz=train,
            start_index=7,
            limit=11,
            low_iterations=3,
            reference_iterations=9,
            min_evaluated=5,
            output_json="autoresearch-session/warm/metrics.json",
        )
    except RuntimeError as exc:
        assert "exact_gpu_resolving blocks" in str(exc)
        assert "static warm-start" in str(exc)
    else:
        raise AssertionError("retired warm-start resolver gate should be blocked")


def test_objective_audit_blocks_protected_surface_without_review(tmp_path):
    init_state(tmp_path)

    result = audit_objective_alignment(
        tmp_path,
        changed_paths=["scripts/play_slumbot.py", "poker_ai/deep_cfr/networks.py"],
    )

    assert result["passed"] is False
    assert result["protected_hits"] == ["scripts/play_slumbot.py"]
    assert "Protected evaluation surfaces changed" in result["errors"][0]


def test_objective_audit_protects_pattern_matched_eval_scripts(tmp_path):
    init_state(tmp_path)

    result = audit_objective_alignment(
        tmp_path,
        changed_paths=["scripts/eval_new_resolver_gate.py"],
    )

    assert result["passed"] is False
    assert result["protected_hits"] == ["scripts/eval_new_resolver_gate.py"]


def test_objective_audit_enforces_new_defaults_even_with_stale_goal(tmp_path):
    init_state(tmp_path)
    goal_path = tmp_path / "autoresearch-session" / "poker_goal.json"
    goal = _read_json(goal_path)
    surfaces = goal["objective_alignment_policy"]["protected_surfaces"]
    goal["objective_alignment_policy"]["protected_surfaces"] = [
        path for path in surfaces if path != "scripts/poker_autoresearch.py"
    ]
    goal_path.write_text(json.dumps(goal), encoding="utf-8")

    result = audit_objective_alignment(
        tmp_path,
        changed_paths=["scripts/poker_autoresearch.py"],
    )

    assert result["passed"] is False
    assert result["protected_hits"] == ["scripts/poker_autoresearch.py"]


def test_objective_audit_protects_new_npi_mixed_control_surfaces_with_stale_goal(tmp_path):
    init_state(tmp_path)
    goal_path = tmp_path / "autoresearch-session" / "poker_goal.json"
    goal = _read_json(goal_path)
    goal["objective_alignment_policy"]["protected_surfaces"] = [
        path
        for path in goal["objective_alignment_policy"]["protected_surfaces"]
        if path
        not in {
            "scripts/run_neural_policy_iteration_loop.py",
            "scripts/eval_mixed_policy_h2h.py",
            "poker_ai/research/mixed_policy_h2h.py",
        }
    ]
    goal_path.write_text(json.dumps(goal), encoding="utf-8")

    result = audit_objective_alignment(
        tmp_path,
        changed_paths=[
            "scripts/run_neural_policy_iteration_loop.py",
            "scripts/eval_mixed_policy_h2h.py",
            "poker_ai/research/mixed_policy_h2h.py",
        ],
    )

    assert result["passed"] is False
    assert result["protected_hits"] == [
        "poker_ai/research/mixed_policy_h2h.py",
        "scripts/eval_mixed_policy_h2h.py",
        "scripts/run_neural_policy_iteration_loop.py",
    ]


def test_objective_audit_allows_protected_surface_with_completed_review(tmp_path):
    init_state(tmp_path)
    queued = enqueue_methodology_review(
        tmp_path,
        subject="Slumbot parser update",
        trigger="evaluation_protocol_change",
        claim="Parser update preserves evaluation hardness.",
    )
    review_dir = Path(queued["review_dir"])
    (review_dir / "review.md").write_text(
        "# Independent Verification\n\nVerdict: PASS\n\nArtifacts inspected.\n",
        encoding="utf-8",
    )
    (review_dir / "related_work.md").write_text(
        "# Related Work\n\n"
        "Title: Deep CFR\n"
        "Source URL: https://arxiv.org/abs/1811.00164\n"
        "Source type: primary\n"
        "Transfers to this codebase: learned regret approximation baseline.\n"
        "Does not transfer: does not validate parser changes alone.\n"
        "Smallest local test: transcript replay parity.\n",
        encoding="utf-8",
    )
    (review_dir / "benchmark_audit.md").write_text(
        "# Benchmark-Hacking Audit\n\nVerdict: PASS\n\nNo benchmark weakening.\n",
        encoding="utf-8",
    )
    (review_dir / "mechanism_review.md").write_text(
        "# Mechanism Review\n\n"
        "Learned object: parser output only.\n"
        "Search boundary: live Slumbot adapter.\n"
        "Train distribution: not applicable.\n"
        "Eval distribution: live Slumbot hands.\n"
        "Falsifier: replayed transcript mismatch.\n"
        "Pass action: allow parser fix.\n"
        "Fail action: block evaluation change.\n"
        "Related-work delta: no algorithmic method change.\n\n"
        "Verdict: PASS\n",
        encoding="utf-8",
    )
    (review_dir / "decision.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "decision": "proceed",
                "reason": "Parser fix keeps metric semantics unchanged.",
                "sources": ["https://arxiv.org/abs/1811.00164"],
            }
        ),
        encoding="utf-8",
    )
    (review_dir / "review_scope.json").write_text(
        json.dumps(
                {
                    "changed_paths": ["scripts/play_slumbot.py"],
                    "protected_hits": ["scripts/play_slumbot.py"],
                    "mechanism": "Slumbot parser preservation",
                    "mechanism_brief": {
                        "decision_object": "Slumbot trace parsing and root action semantics",
                        "where_consumed": "live Slumbot adapter and replay audits",
                        "matched_control": "transcript replay parity before and after parser change",
                        "primary_decision_gate": "transcript replay parity",
                        "retirement_criterion": "block if parser changes action semantics",
                        "flexibility_boundary": "parser internals may change, metric semantics may not",
                        "anti_benchmark_hack": "parser changes cannot alter promotion thresholds",
                        "neural_policy_role": "unchanged; parser must not select policy actions",
                        "cfr_role": "unchanged; parser must not alter resolver semantics",
                        "stochastic_policy_contract": "preserve stochastic policy evaluation semantics",
                    },
                    "expected_gate": "transcript replay parity",
                    "decision_impact": (
                        "The parser change must preserve root action evaluation semantics "
                        "so Slumbot traces do not corrupt self-play league promotion."
                ),
                "decision_impact_gate": "transcript replay parity",
                "fallback_if_no_decision_impact": "block evaluation change",
                "pass_action": "allow parser fix",
                "fail_action": "block evaluation change",
            }
        ),
        encoding="utf-8",
    )
    write_review_manifest(tmp_path, review_dir)

    result = audit_objective_alignment(
        tmp_path,
        changed_paths=["scripts/play_slumbot.py"],
        review_dir=review_dir,
    )

    assert result["passed"] is True
    assert result["protected_hits"] == ["scripts/play_slumbot.py"]


def test_objective_audit_requires_review_manifest_for_protected_surface(tmp_path):
    init_state(tmp_path)
    review_dir = _complete_review(tmp_path, subject="Protected change without manifest")

    result = audit_objective_alignment(
        tmp_path,
        changed_paths=["scripts/play_slumbot.py"],
        review_dir=review_dir,
    )

    assert result["passed"] is False
    assert any("tracked review manifest" in error for error in result["errors"])


def test_methodology_review_requires_decision_impact_scope_fields(tmp_path):
    init_state(tmp_path)
    review_dir = _complete_review(tmp_path, subject="Missing decision impact")
    scope_path = review_dir / "review_scope.json"
    scope = _read_json(scope_path)
    scope.pop("decision_impact", None)
    scope.pop("decision_impact_gate", None)
    scope.pop("fallback_if_no_decision_impact", None)
    scope_path.write_text(json.dumps(scope), encoding="utf-8")

    result = validate_methodology_review(review_dir)

    assert result["passed"] is False
    assert any("decision_impact" in error for error in result["errors"])
    assert any("decision_impact_gate" in error for error in result["errors"])
    assert any("fallback_if_no_decision_impact" in error for error in result["errors"])


def test_objective_audit_requires_review_scope_to_cover_changed_path(tmp_path):
    init_state(tmp_path)
    review_dir = _complete_review(tmp_path, subject="Wrong scope")
    write_review_manifest(tmp_path, review_dir)

    result = audit_objective_alignment(
        tmp_path,
        changed_paths=["scripts/eval_new_resolver_gate.py"],
        review_dir=review_dir,
    )

    assert result["passed"] is False
    assert any("review_scope.json" in error for error in result["errors"])


def test_objective_audit_enforces_review_scope_even_with_stale_goal(tmp_path):
    init_state(tmp_path)
    review_dir = _complete_review(tmp_path, subject="Stale review policy")
    write_review_manifest(tmp_path, review_dir)
    goal_path = tmp_path / "autoresearch-session" / "poker_goal.json"
    goal = _read_json(goal_path)
    goal["review_policy"]["requires_review_scope"] = False
    goal_path.write_text(json.dumps(goal), encoding="utf-8")

    result = audit_objective_alignment(
        tmp_path,
        changed_paths=["scripts/eval_new_resolver_gate.py"],
        review_dir=review_dir,
    )

    assert result["passed"] is False
    assert any("review_scope.json" in error for error in result["errors"])


def test_register_research_knob_requires_completed_review(tmp_path):
    init_state(tmp_path)
    _set_open_research(tmp_path)

    try:
        register_research_knob(
            tmp_path,
            name="unreviewed_knob",
            default="1",
            failure_class="strategy_quality",
            mechanism="Test one isolated mechanism.",
            rationale="Should require a completed review.",
            removal_criterion="Retire on failed gate.",
        )
    except RuntimeError as exc:
        assert "methodology review" in str(exc)
    else:
        raise AssertionError("Expected unreviewed knob to be rejected.")


def test_register_research_knob_requires_mechanism_and_enforces_budget(tmp_path):
    init_state(tmp_path)
    _set_open_research(tmp_path)

    for i in range(5):
        register_research_knob(
            tmp_path,
            name=f"knob_{i}",
            default="1",
            failure_class="strategy_quality",
            mechanism="Test one isolated mechanism.",
            rationale="Needed to isolate a falsifiable failure mode.",
            removal_criterion="Retire if the mechanism is falsified.",
            review_dir=_complete_review(tmp_path, subject=f"Knob {i}"),
        )

    try:
        register_research_knob(
            tmp_path,
            name="knob_5",
            default="1",
            failure_class="strategy_quality",
            mechanism="Would exceed active knob budget.",
            rationale="This should be rejected.",
            removal_criterion="N/A",
            review_dir=_complete_review(tmp_path, subject="Knob 5"),
        )
    except RuntimeError as exc:
        assert "active research knob budget" in str(exc)
    else:
        raise AssertionError("Expected active knob budget enforcement.")


def test_register_research_knob_rejects_sweep_shaped_defaults(tmp_path):
    init_state(tmp_path)
    _set_open_research(tmp_path)

    try:
        register_research_knob(
            tmp_path,
            name="optimizer_grid",
            default="[0.0001,0.001,0.01]",
            failure_class="train_fit",
            mechanism="Hyperparameter sweep over optimizer settings.",
            rationale="This would tune a benchmark instead of testing a mechanism.",
            removal_criterion="Retire after best value is found.",
            review_dir=_complete_review(tmp_path, subject="Optimizer grid"),
        )
    except ValueError as exc:
        assert "single default" in str(exc)
    else:
        raise AssertionError("Expected sweep-shaped default to be rejected.")


def test_enqueue_candidate_comparison_creates_named_gate_from_incumbent(tmp_path):
    init_state(tmp_path)
    candidate = tmp_path / "models" / "candidate.pt"
    incumbent = tmp_path / "models" / "incumbent.pt"
    candidate.parent.mkdir()
    candidate.write_bytes(b"candidate")
    incumbent.write_bytes(b"incumbent")
    set_incumbent(tmp_path, incumbent, reason="baseline")

    queued = enqueue_candidate_comparison(
        tmp_path,
        candidate,
        n_games=12,
        seeds="1,2",
        device="cpu",
        head_to_head=True,
        strategy_source="policy-head",
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    gate = goal["gates"][queued["gate"]]
    command = gate["commands"][0]
    assert queued["gate"].startswith("eval-candidate-compare-")
    assert state["hypothesis_queue"][-1]["gate"] == queued["gate"]
    assert str(candidate) in command
    assert str(incumbent) in command
    assert "--baseline-checkpoint" in command
    assert "--head-to-head" in command
    assert "--require-positive-lower95" in command
    assert "--strategy-source" in command
    assert "policy-head" in command
    assert "12" in command
    assert "1,2" in command


def test_enqueue_candidate_promotion_gate_creates_dual_surface_gate(tmp_path):
    init_state(tmp_path)
    rlcard = tmp_path / "autoresearch-session" / "alphanlholdem_reference" / "rlcard.json"
    native = tmp_path / "autoresearch-session" / "native_rollout_substrate" / "native.json"
    empirical = tmp_path / "autoresearch-session" / "empirical_game" / "matrix.json"
    native_candidate = tmp_path / "models" / "native_candidate.pt"
    for path in (rlcard, native, empirical, native_candidate):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8") if path.suffix == ".json" else path.write_bytes(b"pt")

    queued = enqueue_candidate_promotion_gate(
        tmp_path,
        rlcard_reference_json=rlcard,
        native_h2h_json=native,
        empirical_game_json=empirical,
        native_candidate_checkpoint=native_candidate,
        min_lower95=0.05,
        min_candidate_support=0.2,
        timeout_seconds=321,
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    gate = goal["gates"][queued["gate"]]
    command = gate["commands"][0]
    assert queued["gate"].startswith("candidate-promotion-gate-")
    assert queued["type"] == "candidate_promotion_gate"
    assert state["hypothesis_queue"][-1]["gate"] == queued["gate"]
    assert "scripts/eval_candidate_promotion_gate.py" in command
    assert "--rlcard-reference-json" in command
    assert str(rlcard) in command
    assert "--native-h2h-json" in command
    assert str(native) in command
    assert "--empirical-game-json" in command
    assert str(empirical) in command
    assert "--native-candidate-checkpoint" in command
    assert str(native_candidate) in command
    assert "--min-lower95" in command
    assert "0.05" in command
    assert "--min-candidate-support" in command
    assert "0.2" in command
    assert gate["promotion_role"] == "pre_slumbot_confidence_gate"
    assert gate["requires_rlcard_reference_pass"] is True
    assert gate["requires_native_h2h_pass"] is True
    assert gate["requires_empirical_game_pass"] is True
    assert gate["timeout_seconds"] == 321


def test_enqueue_slumbot_smoke_creates_candidate_live_gate(tmp_path):
    init_state(tmp_path)
    _set_open_research(tmp_path)
    model = tmp_path / "models" / "candidate.pt"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")

    queued = enqueue_slumbot_smoke(
        tmp_path,
        model,
        model_kind="tianshou-rainbow",
        hands=7,
        greedy=True,
        no_allin=True,
        no_solver=True,
        strategy_source="policy-head",
        timeout_seconds=123,
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    gate = goal["gates"][queued["gate"]]
    command = gate["commands"][0]
    assert queued["gate"].startswith("slumbot-candidate-smoke-")
    assert state["hypothesis_queue"][-1]["gate"] == queued["gate"]
    assert str(model) in command
    assert "7" in command
    assert "--model-kind" in command
    assert "tianshou-rainbow" in command
    assert "--greedy" in command
    assert "--no-allin" in command
    assert "--no-solver" in command
    assert "--strategy-source" in command
    assert "policy-head" in command
    assert "--trace-jsonl" in command
    trace_path = command[command.index("--trace-jsonl") + 1]
    assert trace_path.endswith(".jsonl")
    assert "slumbot_traces" in trace_path
    assert "123" in command
    assert gate["promotion_role"] == "integration_smoke_only"
    assert gate["model_kind"] == "tianshou-rainbow"
    assert gate["requires_internal_league_pass"] is False


def test_enqueue_slumbot_confidence_requires_internal_league_pass(tmp_path):
    init_state(tmp_path)
    _set_open_research(tmp_path)
    model = tmp_path / "models" / "candidate.pt"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")

    try:
        enqueue_slumbot_smoke(
            tmp_path,
            model,
            hands=300,
            no_solver=False,
        )
    except RuntimeError as exc:
        assert "self-play checkpoint league" in str(exc)
    else:
        raise AssertionError("Slumbot confidence spend should require an internal league pass")

    run_dir = tmp_path / "autoresearch-session" / "poker_runs" / "league-pass"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "passed": True,
                "best_worst_lower95_chips_per_hand": 0.25,
            }
        ),
        encoding="utf-8",
    )
    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["history"].append(
        {
            "run_id": "league-pass",
            "run_dir": str(run_dir),
            "gate": "native-policy-league-root-disjoint",
            "type": "self_play_league",
            "outcome": "passed",
            "summary": "Checkpoint league positive lower95 versus incumbent.",
        }
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")

    try:
        enqueue_slumbot_smoke(
            tmp_path,
            model,
            hands=300,
            no_solver=False,
        )
    except RuntimeError as exc:
        assert "pre-Slumbot candidate promotion gate" in str(exc)
    else:
        raise AssertionError("Slumbot confidence spend should require dual-surface promotion gate")

    promotion_dir = tmp_path / "autoresearch-session" / "poker_runs" / "promotion-pass"
    promotion_dir.mkdir(parents=True)
    (promotion_dir / "metrics.json").write_text(
        json.dumps(
            {
                "algorithm": "poker_candidate_promotion_gate",
                "passed": True,
                "slumbot_confidence_eligible": True,
                "promotion_blockers": [],
            }
        ),
        encoding="utf-8",
    )
    state = _read_json(state_path)
    state["history"].append(
        {
            "run_id": "promotion-pass",
            "run_dir": str(promotion_dir),
            "gate": "pre-slumbot-candidate-promotion-gate",
            "type": "candidate_promotion_gate",
            "outcome": "passed",
            "summary": "RLCard reference, native H2H, and empirical-game evidence passed.",
        }
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")

    queued = enqueue_slumbot_smoke(
        tmp_path,
        model,
        hands=300,
        no_solver=False,
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    gate = goal["gates"][queued["gate"]]
    assert gate["promotion_role"] == "held_out_external_validation"
    assert gate["requires_internal_league_pass"] is True
    assert gate["internal_league_evidence"]["passed"] is True
    assert gate["candidate_promotion_evidence"]["passed"] is True


def test_enqueue_slumbot_smoke_uses_unique_gate_names_with_same_timestamp(
    tmp_path,
    monkeypatch,
):
    init_state(tmp_path)
    _set_open_research(tmp_path)
    model = tmp_path / "models" / "candidate.pt"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")

    class FixedDateTime:
        @classmethod
        def now(cls, tz):
            return datetime(2026, 5, 11, 19, 29, 29, tzinfo=UTC)

    monkeypatch.setattr(autoresearch, "datetime", FixedDateTime)

    first = enqueue_slumbot_smoke(tmp_path, model, strategy_source="policy-head")
    second = enqueue_slumbot_smoke(tmp_path, model, strategy_source="regret")

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    assert first["gate"] != second["gate"]
    assert first["gate"] in goal["gates"]
    assert second["gate"] in goal["gates"]
    assert [item["gate"] for item in state["hypothesis_queue"][-2:]] == [
        first["gate"],
        second["gate"],
    ]


def test_enqueue_warm_start_resolver_selects_regret_policy_evaluator_for_regret_checkpoint(tmp_path):
    init_state(tmp_path)
    _set_neural_regret_phase(tmp_path)
    checkpoint = tmp_path / "models" / "regret_policy.pt"
    checkpoint.parent.mkdir()
    torch.save({"mode": "regret_policy_warm_start_checkpoint"}, checkpoint)
    cases = tmp_path / "cases.json"
    cases.write_text("[]", encoding="utf-8")
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"npz")
    train_labels = tmp_path / "train_labels.npz"
    train_labels.write_bytes(b"npz")

    queued = enqueue_warm_start_resolver_gate(
        tmp_path,
        checkpoint=checkpoint,
        cases_json=cases,
        cfv_cache=cache,
        train_labels_npz=train_labels,
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    command = goal["gates"][queued["gate"]]["commands"][0]
    assert "scripts/eval_regret_policy_warm_start.py" in command
    assert "--regret-mass-scale" not in command
    assert "--strategy-mass" not in command


def test_enqueue_warm_start_resolver_passes_uniform_budget_baseline_for_regret_checkpoint(tmp_path):
    init_state(tmp_path)
    _set_neural_regret_phase(tmp_path)
    checkpoint = tmp_path / "models" / "regret_policy.pt"
    checkpoint.parent.mkdir()
    torch.save({"mode": "regret_policy_warm_start_checkpoint"}, checkpoint)
    cases = tmp_path / "cases.json"
    cases.write_text("[]", encoding="utf-8")
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"npz")

    queued = enqueue_warm_start_resolver_gate(
        tmp_path,
        checkpoint=checkpoint,
        cases_json=cases,
        cfv_cache=cache,
        baseline_iterations=10,
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    gate = goal["gates"][queued["gate"]]
    command = gate["commands"][0]
    assert "scripts/eval_regret_policy_warm_start.py" in command
    assert "--baseline-iterations" in command
    assert command[command.index("--baseline-iterations") + 1] == "10"
    assert gate["baseline_iterations"] == 10


def test_enqueue_warm_start_resolver_keeps_joint_pbs_evaluator_for_joint_checkpoint(tmp_path):
    init_state(tmp_path)
    _set_neural_regret_phase(tmp_path)
    checkpoint = tmp_path / "models" / "joint_pbs.pt"
    checkpoint.parent.mkdir()
    torch.save({"mode": "joint_pbs_low_solver_residual"}, checkpoint)
    cases = tmp_path / "cases.json"
    cases.write_text("[]", encoding="utf-8")
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"npz")

    queued = enqueue_warm_start_resolver_gate(
        tmp_path,
        checkpoint=checkpoint,
        cases_json=cases,
        cfv_cache=cache,
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    command = goal["gates"][queued["gate"]]["commands"][0]
    assert "scripts/eval_joint_pbs_policy_warm_start.py" in command
    assert "--regret-mass-scale" in command
    assert "--strategy-mass" in command


def test_enqueue_resolver_benchmark_creates_candidate_gate(tmp_path):
    init_state(tmp_path)
    model = tmp_path / "models" / "candidate.pt"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")

    queued = enqueue_resolver_benchmark(
        tmp_path,
        model,
        solver_iterations=9,
        max_cases=2,
        device="cpu",
        timeout_seconds=321,
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    gate = goal["gates"][queued["gate"]]
    command = gate["commands"][0]
    assert queued["gate"].startswith("resolver-candidate-benchmark-")
    assert state["hypothesis_queue"][-1]["gate"] == queued["gate"]
    assert str(model) in command
    assert "scripts/poker_resolver_benchmark.py" in command
    assert "9" in command
    assert "2" in command
    assert "321" == str(gate["timeout_seconds"])


def test_enqueue_cfr_budget_frontier_creates_exact_gpu_gate(tmp_path):
    init_state(tmp_path)
    cases = tmp_path / "cases.json"
    cache = tmp_path / "cache.npz"
    cases.write_text("[]", encoding="utf-8")
    cache.write_bytes(b"npz")

    queued = enqueue_cfr_budget_frontier(
        tmp_path,
        cases_json=cases,
        cfv_cache=cache,
        budgets=[25, 50],
        reference_iterations=75,
        start_index=4,
        limit=8,
        solver_backend="torch-levelsync-cuda",
        min_evaluated=8,
        timeout_seconds=1234,
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    gate = goal["gates"][queued["gate"]]
    command = gate["commands"][0]
    assert queued["gate"].startswith("cfr-budget-frontier-")
    assert queued["type"] == "solver_budget_frontier"
    assert state["hypothesis_queue"][-1]["gate"] == queued["gate"]
    assert "scripts/eval_cfr_budget_frontier.py" in command
    assert "--budgets" in command
    assert "25,50" in command
    assert "--solver-backend" in command
    assert "torch-levelsync-cuda" in command
    assert gate["timeout_seconds"] == 1234


def test_enqueue_cfr_matrix_footprint_creates_chunk_plan_gate(tmp_path):
    init_state(tmp_path)
    cases = tmp_path / "cases.json"
    cases.write_text("[]", encoding="utf-8")

    queued = enqueue_cfr_matrix_footprint(
        tmp_path,
        cases_json=cases,
        start_index=16,
        max_cases=32,
        chunk_memory_cap_mib=4096,
        timeout_seconds=222,
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    gate = goal["gates"][queued["gate"]]
    command = gate["commands"][0]
    assert queued["gate"].startswith("cfr-matrix-footprint-")
    assert queued["type"] == "exact_gpu_solver_planning"
    assert state["hypothesis_queue"][-1]["gate"] == queued["gate"]
    assert "scripts/analyze_cfr_matrix_footprint.py" in command
    assert "--chunk-memory-cap-mib" in command
    assert "4096.0" in command
    assert "--max-cases" in command
    assert "32" in command
    assert gate["timeout_seconds"] == 222


def test_enqueue_falsification_ladder_creates_countertest_gate(tmp_path):
    init_state(tmp_path)
    candidate = tmp_path / "models" / "candidate.pt"
    incumbent = tmp_path / "models" / "incumbent.pt"
    candidate.parent.mkdir()
    candidate.write_bytes(b"candidate")
    incumbent.write_bytes(b"incumbent")
    set_incumbent(tmp_path, incumbent, reason="baseline")

    queued = enqueue_falsification_ladder(
        tmp_path,
        candidate,
        mechanism="search-distilled policy targets reduce transfer loss",
        n_games=24,
        seeds="11,12",
        max_resolver_cases=2,
        device="cpu",
        changed_paths=["poker_ai/deep_cfr/networks.py"],
        strategy_source="policy-head",
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    gate = goal["gates"][queued["gate"]]
    commands = gate["commands"]
    assert queued["gate"].startswith("falsification-ladder-")
    assert queued["type"] == "falsification"
    assert queued["mechanism"] == "search-distilled policy targets reduce transfer loss"
    assert state["hypothesis_queue"][-1]["gate"] == queued["gate"]
    assert "scripts/poker_objective_audit.py" in commands[0]
    assert "--changed-path" in commands[0]
    assert "poker_ai/deep_cfr/networks.py" in commands[0]
    assert "scripts/poker_autoresearch_eval.py" in commands[1]
    assert "--head-to-head" in commands[1]
    assert "--require-positive-lower95" in commands[1]
    assert "--strategy-source" in commands[1]
    assert "policy-head" in commands[1]
    assert "24" in commands[1]
    assert "11,12" in commands[1]
    assert "scripts/poker_resolver_benchmark.py" in commands[2]
    assert "--max-cases" in commands[2]
    assert "2" in commands[2]


def test_enqueue_sd_cfr_mixture_falsification_creates_local_gate(tmp_path):
    init_state(tmp_path)
    run_dir = tmp_path / "models" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "candidate_iter_50.pt").write_bytes(b"candidate-50")
    (run_dir / "candidate_iter_100.pt").write_bytes(b"candidate-100")
    baseline = tmp_path / "models" / "incumbent.pt"
    baseline.write_bytes(b"incumbent")
    set_incumbent(tmp_path, baseline, reason="baseline")
    review_dir = tmp_path / "autoresearch-session" / "poker_reviews" / "review-1"
    review_dir.mkdir(parents=True)

    queued = enqueue_sd_cfr_mixture_falsification(
        tmp_path,
        candidate_globs=[str(run_dir / "candidate_iter_*.pt")],
        mechanism="fixed SD-CFR checkpoint mixture preserves regret/value coupling",
        n_games=24,
        seeds="11,12",
        device="cpu",
        changed_paths=["scripts/play_slumbot.py"],
        review_dir=review_dir,
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    gate = goal["gates"][queued["gate"]]
    commands = gate["commands"]
    assert queued["gate"].startswith("sd-cfr-mixture-falsification-")
    assert queued["type"] == "falsification"
    assert state["hypothesis_queue"][-1]["gate"] == queued["gate"]
    assert "scripts/poker_objective_audit.py" in commands[0]
    assert "--review-dir" in commands[0]
    assert str(review_dir) in commands[0]
    assert "--changed-path" in commands[0]
    assert "scripts/play_slumbot.py" in commands[0]
    assert "scripts/eval_sd_cfr_mixture.py" in commands[1]
    assert "--candidate-glob" in commands[1]
    assert str(run_dir / "candidate_iter_*.pt") in commands[1]
    assert "--baseline-checkpoint" in commands[1]
    assert str(baseline) in commands[1]
    assert "24" in commands[1]
    assert "11,12" in commands[1]
    assert gate["candidate_globs"] == [str(run_dir / "candidate_iter_*.pt")]


def test_enqueue_gpu_training_creates_candidate_gate(tmp_path):
    init_state(tmp_path)
    _set_open_research(tmp_path)

    queued = enqueue_gpu_training(
        tmp_path,
        n_iterations=2,
        n_traversals=3,
        n_training_steps=4,
        buffer_capacity=5000,
        hidden_dim=64,
        n_layers=2,
        batch_size=128,
        save_dir="models/train_gate",
        prefix="probe",
        eval_games=5,
        max_pool_exhausted_per_traversal=0.0,
        max_overflow_chunk_fraction=0.0,
        max_rejected_traversal_chunks=0,
        min_traversals_per_second=100.0,
        save_replay_buffers=True,
        timeout_seconds=678,
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    gate = goal["gates"][queued["gate"]]
    command = gate["commands"][0]
    assert queued["gate"].startswith("train-gpu-deep-cfr-")
    assert state["hypothesis_queue"][-1]["gate"] == queued["gate"]
    assert "scripts/poker_autoresearch_train.py" in command
    assert "--save-dir" in command
    assert str(tmp_path / "models" / "train_gate") in command
    assert "--prefix" in command
    assert "probe" in command
    assert "--max-pool-exhausted-per-traversal" in command
    assert "0.0" in command
    assert "--max-overflow-chunk-fraction" in command
    assert "--max-rejected-traversal-chunks" in command
    assert "0" in command
    assert "--min-traversals-per-second" in command
    assert "100.0" in command
    assert "--save-replay-buffers" in command
    assert gate["save_replay_buffers"] is True
    slots_arg = command.index("--traversal-slots-per-traversal")
    assert command[slots_arg + 1] == "7000"
    assert "678" == str(gate["timeout_seconds"])


def test_enqueue_gpu_training_can_request_periodic_checkpoint_comparisons(tmp_path):
    init_state(tmp_path)
    _set_open_research(tmp_path)
    baseline = tmp_path / "models" / "incumbent.pt"
    baseline.parent.mkdir()
    baseline.write_bytes(b"checkpoint")
    set_incumbent(tmp_path, baseline, reason="baseline")

    queued = enqueue_gpu_training(
        tmp_path,
        n_iterations=4,
        n_traversals=3,
        save_dir="models/train_gate",
        prefix="probe",
        save_every=2,
        traversal_slots_per_traversal=2500,
        traversal_pool_max_slots=1_000_000,
        auto_compare=True,
        compare_n_games=7,
        compare_seeds="1,2",
        compare_timeout_seconds=99,
        compare_strategy_source="policy-head",
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    gate = goal["gates"][queued["gate"]]
    command = gate["commands"][0]
    queued_item = state["hypothesis_queue"][-1]
    assert "--save-every" in command
    assert "2" in command
    assert "--traversal-slots-per-traversal" in command
    assert "2500" in command
    assert "--traversal-pool-max-slots" in command
    assert "1000000" in command
    assert queued_item["postprocess"] == {
        "type": "compare_training_checkpoints",
        "n_games": 7,
        "seeds": "1,2",
        "device": "auto",
        "timeout_seconds": 99,
        "head_to_head": True,
        "strategy_source": "policy-head",
        "candidate_strategy_source": "policy-head",
        "baseline_strategy_source": "policy-head",
    }


def test_enqueue_gpu_training_auto_compare_can_split_candidate_and_baseline_sources(tmp_path):
    init_state(tmp_path)
    _set_open_research(tmp_path)
    baseline = tmp_path / "models" / "incumbent.pt"
    baseline.parent.mkdir()
    baseline.write_bytes(b"checkpoint")
    set_incumbent(tmp_path, baseline, reason="baseline")

    queued = enqueue_gpu_training(
        tmp_path,
        n_iterations=2,
        n_traversals=3,
        save_dir="models/train_gate",
        prefix="avg_policy_probe",
        save_every=1,
        auto_compare=True,
        compare_n_games=11,
        compare_seeds="5,6",
        compare_candidate_strategy_source="average-policy",
        compare_baseline_strategy_source="regret",
    )

    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    queued_item = state["hypothesis_queue"][-1]
    assert queued["gate"] == queued_item["gate"]
    assert queued_item["postprocess"]["candidate_strategy_source"] == "average-policy"
    assert queued_item["postprocess"]["baseline_strategy_source"] == "regret"


def test_enqueue_gpu_training_can_request_frontier_indexing(tmp_path):
    init_state(tmp_path)
    _set_open_research(tmp_path)

    queued = enqueue_gpu_training(
        tmp_path,
        n_iterations=1,
        n_traversals=2,
        save_dir="models/train_gate",
        prefix="frontier",
        use_frontier_indexing=True,
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    command = goal["gates"][queued["gate"]]["commands"][0]
    assert "--use-frontier-indexing" in command


def test_enqueue_gpu_training_can_request_seed(tmp_path):
    init_state(tmp_path)
    _set_open_research(tmp_path)

    queued = enqueue_gpu_training(
        tmp_path,
        n_iterations=1,
        n_traversals=2,
        save_dir="models/train_gate",
        prefix="seeded",
        seed=20260515,
    )

    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    command = goal["gates"][queued["gate"]]["commands"][0]
    seed_arg = command.index("--seed")
    assert command[seed_arg + 1] == "20260515"


def test_continuous_queues_comparisons_for_saved_training_checkpoints(tmp_path):
    init_state(tmp_path)
    _set_open_research(tmp_path)
    baseline = tmp_path / "models" / "incumbent.pt"
    baseline.parent.mkdir()
    baseline.write_bytes(b"checkpoint")
    set_incumbent(tmp_path, baseline, reason="baseline")

    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["hypothesis_queue"] = []
    state_path.write_text(json.dumps(state), encoding="utf-8")

    save_dir = tmp_path / "models" / "train_gate"
    enqueue_gpu_training(
        tmp_path,
        n_iterations=2,
        n_traversals=3,
        save_dir=save_dir,
        prefix="probe",
        save_every=1,
        auto_compare=True,
        compare_n_games=11,
        compare_seeds="5,6",
        compare_strategy_source="policy-head",
    )

    def training_runner(command, timeout_seconds=None):
        iter_path = save_dir / "probe_iter_1.pt"
        final_path = save_dir / "probe_final.pt"
        save_dir.mkdir(parents=True, exist_ok=True)
        iter_path.write_bytes(b"iter")
        final_path.write_bytes(b"final")
        return CommandResult(
            command=list(command),
            returncode=0,
            stdout=json.dumps(
                {
                    "passed": True,
                    "mode": "autoresearch_gpu_deep_cfr_train",
                    "checkpoint": str(final_path),
                    "checkpoints": [
                        {"path": str(iter_path), "kind": "periodic", "iteration": 1},
                        {"path": str(final_path), "kind": "final", "iteration": 2},
                    ],
                }
            ),
            stderr="",
            seconds=0.1,
        )

    result = continuous(tmp_path, max_cycles=1, runner=training_runner, sleep_seconds=0)

    assert result == {"cycles_completed": 1, "stopped_reason": "max_cycles"}
    state = _read_json(state_path)
    compare_gates = [item["gate"] for item in state["hypothesis_queue"]]
    assert len(compare_gates) == 2
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    commands = [goal["gates"][gate]["commands"][0] for gate in compare_gates]
    assert str(save_dir / "probe_iter_1.pt") in commands[0]
    assert str(save_dir / "probe_final.pt") in commands[1]
    assert all("--head-to-head" in command for command in commands)
    assert all("--strategy-source" in command for command in commands)
    assert all("policy-head" in command for command in commands)
    assert all("11" in command for command in commands)
    assert all("5,6" in command for command in commands)


def test_continuous_queues_split_strategy_source_comparison_commands(tmp_path):
    init_state(tmp_path)
    _set_open_research(tmp_path)
    baseline = tmp_path / "models" / "incumbent.pt"
    baseline.parent.mkdir()
    baseline.write_bytes(b"checkpoint")
    set_incumbent(tmp_path, baseline, reason="baseline")

    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["hypothesis_queue"] = []
    state_path.write_text(json.dumps(state), encoding="utf-8")

    save_dir = tmp_path / "models" / "train_gate"
    enqueue_gpu_training(
        tmp_path,
        n_iterations=2,
        n_traversals=3,
        save_dir=save_dir,
        prefix="avg_policy_probe",
        save_every=1,
        auto_compare=True,
        compare_n_games=11,
        compare_seeds="5,6",
        compare_candidate_strategy_source="average-policy",
        compare_baseline_strategy_source="regret",
    )

    def training_runner(command, timeout_seconds=None):
        iter_path = save_dir / "avg_policy_probe_iter_1.pt"
        save_dir.mkdir(parents=True, exist_ok=True)
        iter_path.write_bytes(b"iter")
        return CommandResult(
            command=list(command),
            returncode=0,
            stdout=json.dumps(
                {
                    "passed": True,
                    "mode": "autoresearch_gpu_deep_cfr_train",
                    "checkpoints": [
                        {"path": str(iter_path), "kind": "periodic", "iteration": 1},
                    ],
                }
            ),
            stderr="",
            seconds=0.1,
        )

    result = continuous(tmp_path, max_cycles=1, runner=training_runner, sleep_seconds=0)

    assert result == {"cycles_completed": 1, "stopped_reason": "max_cycles"}
    state = _read_json(state_path)
    compare_gate = state["hypothesis_queue"][-1]["gate"]
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    command = goal["gates"][compare_gate]["commands"][0]
    assert "--candidate-strategy-source" in command
    assert command[command.index("--candidate-strategy-source") + 1] == "average-policy"
    assert "--baseline-strategy-source" in command
    assert command[command.index("--baseline-strategy-source") + 1] == "regret"
    assert "--strategy-source" not in command


def test_set_incumbent_records_checkpoint_metadata(tmp_path):
    init_state(tmp_path)
    checkpoint = tmp_path / "models" / "candidate.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")

    result = set_incumbent(
        tmp_path,
        checkpoint,
        reason="best available local Slumbot-track checkpoint",
    )

    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    assert result["checkpoint"] == str(checkpoint)
    assert result["reason"] == "best available local Slumbot-track checkpoint"
    assert state["incumbent_checkpoint"]["checkpoint"] == str(checkpoint)


def test_cli_init_and_status_emit_json(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"

    init_result = subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert init_result.returncode == 0
    assert json.loads(init_result.stdout)["ready"] is True

    status_result = subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "status"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert status_result.returncode == 0
    assert json.loads(status_result.stdout)["ready"] is True


def test_cli_drift_status_reports_queued_target_consumer_blocker(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["history"] = _xdo_target_drift_history()
    state["hypothesis_queue"] = [
        {
            "id": "repeat-xdo",
            "type": "experiment",
            "hypothesis": "Repeat the XDO NPI target consumer with another small target variant.",
            "failure_class": "mechanism_transfer",
            "gate": "xdo-target-consumer-parent-h2h",
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "drift-status"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["blocked"] is True
    assert payload["matched_family"] == "local_target_consumer"


def test_cli_enqueue_compare_creates_gate(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    candidate = tmp_path / "models" / "candidate.pt"
    incumbent = tmp_path / "models" / "incumbent.pt"
    candidate.parent.mkdir()
    candidate.write_bytes(b"candidate")
    incumbent.write_bytes(b"incumbent")

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "set-incumbent",
            "--checkpoint",
            str(incumbent),
            "--reason",
            "baseline",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "enqueue-compare",
            "--candidate",
            str(candidate),
            "--n-games",
            "8",
            "--seeds",
            "3,4",
            "--device",
            "cpu",
            "--head-to-head",
            "--strategy-source",
            "policy-head",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    queued = json.loads(result.stdout)
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    assert queued["gate"] in goal["gates"]
    assert "--head-to-head" in goal["gates"][queued["gate"]]["commands"][0]
    assert "policy-head" in goal["gates"][queued["gate"]]["commands"][0]


def test_cli_enqueue_slumbot_creates_gate(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    model = tmp_path / "models" / "candidate.pt"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    _cli_set_open_research(script, tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "enqueue-slumbot",
            "--model",
            str(model),
            "--model-kind",
            "tianshou-rainbow",
            "--hands",
            "3",
            "--greedy",
            "--no-allin",
            "--no-solver",
            "--strategy-source",
            "policy-head",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    queued = json.loads(result.stdout)
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    assert queued["gate"] in goal["gates"]
    command = goal["gates"][queued["gate"]]["commands"][0]
    assert "--model-kind" in command
    assert "tianshou-rainbow" in command
    assert "--no-solver" in command
    assert "policy-head" in command


def test_cli_enqueue_slumbot_accepts_torch_levelsync_cuda_backend(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    model = tmp_path / "models" / "candidate.pt"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    _cli_set_open_research(script, tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "enqueue-slumbot",
            "--model",
            str(model),
            "--hands",
            "2",
            "--solver-backend",
            "torch-levelsync-cuda",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    queued = json.loads(result.stdout)
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    command = goal["gates"][queued["gate"]]["commands"][0]
    assert "--solver-backend" in command
    assert "torch-levelsync-cuda" in command


def test_cli_enqueue_slumbot_accepts_fast_live_budget_profile(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    model = tmp_path / "models" / "candidate.pt"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    _cli_set_open_research(script, tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "enqueue-slumbot",
            "--model",
            str(model),
            "--solver-backend",
            "torch-levelsync-cuda",
            "--solver-budget-profile",
            "fast-live",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    queued = json.loads(result.stdout)
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    command = goal["gates"][queued["gate"]]["commands"][0]
    assert "--solver-budget-profile" in command
    assert "fast-live" in command


def test_cli_enqueue_slumbot_accepts_frontier_live_budget_profile(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    model = tmp_path / "models" / "candidate.pt"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    _cli_set_open_research(script, tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "enqueue-slumbot",
            "--model",
            str(model),
            "--solver-backend",
            "torch-levelsync-cuda",
            "--solver-budget-profile",
            "frontier-live",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    queued = json.loads(result.stdout)
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    command = goal["gates"][queued["gate"]]["commands"][0]
    assert "--solver-budget-profile" in command
    assert "frontier-live" in command


def test_cli_enqueue_resolver_benchmark_creates_gate(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    model = tmp_path / "models" / "candidate.pt"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    _cli_set_open_research(script, tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "enqueue-resolver",
            "--model",
            str(model),
            "--solver-iterations",
            "8",
            "--max-cases",
            "1",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    queued = json.loads(result.stdout)
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    assert queued["gate"] in goal["gates"]
    assert "--max-cases" in goal["gates"][queued["gate"]]["commands"][0]


def test_cli_enqueue_cfr_budget_frontier_creates_gate(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    cases = tmp_path / "cases.json"
    cache = tmp_path / "cache.npz"
    cases.write_text("[]", encoding="utf-8")
    cache.write_bytes(b"cache")

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "enqueue-cfr-budget-frontier",
            "--cases",
            str(cases),
            "--cfv-cache",
            str(cache),
            "--budgets",
            "10,20",
            "--reference-iterations",
            "40",
            "--solver-backend",
            "torch-levelsync-cpu",
            "--min-evaluated",
            "4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    queued = json.loads(result.stdout)
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    command = goal["gates"][queued["gate"]]["commands"][0]
    assert "scripts/eval_cfr_budget_frontier.py" in command
    assert "10,20" in command
    assert "torch-levelsync-cpu" in command


def test_cli_enqueue_cfr_matrix_footprint_creates_gate(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    cases = tmp_path / "cases.json"
    cases.write_text("[]", encoding="utf-8")

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "enqueue-cfr-matrix-footprint",
            "--cases",
            str(cases),
            "--start-index",
            "8",
            "--max-cases",
            "16",
            "--chunk-memory-cap-mib",
            "4096",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    queued = json.loads(result.stdout)
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    command = goal["gates"][queued["gate"]]["commands"][0]
    assert "scripts/analyze_cfr_matrix_footprint.py" in command
    assert "--chunk-memory-cap-mib" in command
    assert "4096.0" in command


def test_cli_enqueue_warm_start_resolver_creates_gate_in_neural_regret_phase(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    checkpoint = tmp_path / "models" / "joint.pt"
    cases = tmp_path / "cases.json"
    cache = tmp_path / "cache.npz"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    cases.write_text('{"cases": []}', encoding="utf-8")
    cache.write_bytes(b"cache")

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    _cli_set_neural_regret_phase(script, tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "enqueue-warm-start-resolver",
            "--checkpoint",
            str(checkpoint),
            "--cases",
            str(cases),
            "--cfv-cache",
            str(cache),
            "--limit",
            "4",
            "--min-evaluated",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    queued = json.loads(result.stdout)
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    command = goal["gates"][queued["gate"]]["commands"][0]
    assert "scripts/eval_joint_pbs_policy_warm_start.py" in command
    assert "--limit" in command
    assert "4" in command


def test_cli_enqueue_warm_start_resolver_accepts_uniform_budget_baseline(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    checkpoint = tmp_path / "models" / "regret.pt"
    cases = tmp_path / "cases.json"
    cache = tmp_path / "cache.npz"
    checkpoint.parent.mkdir()
    torch.save({"mode": "regret_policy_warm_start_checkpoint"}, checkpoint)
    cases.write_text('{"cases": []}', encoding="utf-8")
    cache.write_bytes(b"cache")

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    _cli_set_neural_regret_phase(script, tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "enqueue-warm-start-resolver",
            "--checkpoint",
            str(checkpoint),
            "--cases",
            str(cases),
            "--cfv-cache",
            str(cache),
            "--baseline-iterations",
            "10",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    queued = json.loads(result.stdout)
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    command = goal["gates"][queued["gate"]]["commands"][0]
    assert "scripts/eval_regret_policy_warm_start.py" in command
    assert "--baseline-iterations" in command
    assert "10" in command


def test_cli_enqueue_review_creates_review_gate(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    _cli_set_open_research(script, tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "enqueue-review",
            "--subject",
            "New search objective",
            "--trigger",
            "method_change",
            "--claim",
            "The new search objective is justified.",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    queued = json.loads(result.stdout)
    assert Path(queued["review_dir"]).is_dir()
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    assert queued["gate"] in goal["gates"]


def test_cli_phase_and_calibration_audit_commands(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    train = tmp_path / "train.npz"
    holdout = tmp_path / "holdout.npz"
    _write_tiny_callback_cache(train, root_label="train-root")
    _write_tiny_callback_cache(holdout, root_label="holdout-root")

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    phase = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "set-phase",
            "--phase",
            "callback_state_calibration_debug",
            "--reason",
            "scale-up failed",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert phase.returncode == 0
    assert json.loads(phase.stdout)["current"] == "callback_state_calibration_debug"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "enqueue-calibration-audit",
            "--train-dual-cache",
            str(train),
            "--holdout-dual-cache",
            str(holdout),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    queued = json.loads(result.stdout)
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    assert queued["gate"] in goal["gates"]
    assert "scripts/poker_callback_calibration_audit.py" in goal["gates"][queued["gate"]]["commands"][0]


def test_callback_calibration_audit_script_summarizes_tiny_cache(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_callback_calibration_audit.py"
    train = tmp_path / "train.npz"
    holdout = tmp_path / "holdout.npz"
    output = tmp_path / "metrics.json"
    _write_tiny_callback_cache(train, root_label="train-root")
    _write_tiny_callback_cache(holdout, root_label="holdout-root")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--train-dual-cache",
            str(train),
            "--holdout-dual-cache",
            str(holdout),
            "--output-json",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    metrics = _read_json(output)
    assert metrics["mode"] == "callback_state_calibration_audit"
    assert metrics["passed"] is True
    assert metrics["train"]["n_states"] == 2
    assert metrics["holdout"]["n_roots"] == 1


def test_cli_add_knob_records_governed_knob(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    _cli_set_open_research(script, tmp_path)
    review_dir = _complete_review(tmp_path, subject="CLI knob")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "add-knob",
            "--name",
            "search_target_mix",
            "--default",
            "0.0",
            "--failure-class",
            "search_quality",
            "--mechanism",
            "Test whether search-distilled targets reduce live transfer loss.",
            "--rationale",
            "One variable isolates the search target mechanism.",
            "--removal-criterion",
            "Retire if Slumbot transfer remains negative after confirmation.",
            "--review-dir",
            str(review_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["name"] == "search_target_mix"
    assert payload["review_id"] == review_dir.name
    knob_text = (tmp_path / "autoresearch-session" / "poker_knobs.tsv").read_text(
        encoding="utf-8"
    )
    assert "search_target_mix" in knob_text


def test_cli_objective_audit_blocks_protected_surface(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    _cli_set_open_research(script, tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "objective-audit",
            "--changed-path",
            "scripts/play_slumbot.py",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["protected_hits"] == ["scripts/play_slumbot.py"]


def test_cli_objective_audit_discovers_git_changes_and_requires_allow_empty(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, text=True, check=True)
    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=tmp_path, capture_output=True, text=True, check=True)
    (tmp_path / "scripts").mkdir(exist_ok=True)
    changed = tmp_path / "scripts" / "eval_new_gate.py"
    changed.write_text("print('changed')\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "objective-audit"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["changed_paths"] == ["scripts/eval_new_gate.py"]
    assert payload["protected_hits"] == ["scripts/eval_new_gate.py"]

    changed.unlink()
    empty = subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "objective-audit"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert empty.returncode == 2
    assert "No changed paths" in empty.stderr

    allowed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "objective-audit",
            "--allow-empty",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert allowed.returncode == 0
    assert json.loads(allowed.stdout)["changed_paths"] == []


def test_commit_ready_report_summarizes_git_review_and_synthesis_state(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, text=True, check=True)
    init_state(tmp_path)
    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["history"] = [
        {"run_id": f"run-{idx}", "type": "experiment", "outcome": "failed"}
        for idx in range(5)
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=tmp_path, capture_output=True, text=True, check=True)
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "eval_new_gate.py").write_text("print('x')\n", encoding="utf-8")

    report = commit_ready_report(tmp_path)

    assert report["ready"] is False
    assert report["changed_paths"] == ["scripts/eval_new_gate.py"]
    assert report["objective_audit"]["protected_hits"] == ["scripts/eval_new_gate.py"]
    assert report["synthesis"]["due"] is True
    assert any("synthesis is due" in blocker for blocker in report["blockers"])


def test_cli_enqueue_falsification_ladder_creates_gate(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    candidate = tmp_path / "models" / "candidate.pt"
    incumbent = tmp_path / "models" / "incumbent.pt"
    candidate.parent.mkdir()
    candidate.write_bytes(b"candidate")
    incumbent.write_bytes(b"incumbent")

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "set-incumbent",
            "--checkpoint",
            str(incumbent),
            "--reason",
            "baseline",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "enqueue-falsification",
            "--candidate",
            str(candidate),
            "--mechanism",
            "search-distilled policy targets reduce transfer loss",
            "--n-games",
            "16",
            "--max-resolver-cases",
            "1",
            "--changed-path",
            "poker_ai/deep_cfr/networks.py",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    queued = json.loads(result.stdout)
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    assert queued["gate"] in goal["gates"]
    assert len(goal["gates"][queued["gate"]]["commands"]) == 3


def test_cli_enqueue_sd_cfr_mixture_falsification_creates_gate(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    run_dir = tmp_path / "models" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "candidate_iter_50.pt").write_bytes(b"candidate-50")
    baseline = tmp_path / "models" / "incumbent.pt"
    baseline.write_bytes(b"incumbent")

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "set-incumbent",
            "--checkpoint",
            str(baseline),
            "--reason",
            "baseline",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "enqueue-sd-cfr-mixture-falsification",
            "--candidate-glob",
            str(run_dir / "candidate_iter_*.pt"),
            "--mechanism",
            "fixed mixture preserves regret/value coupling",
            "--n-games",
            "16",
            "--changed-path",
            "scripts/play_slumbot.py",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    queued = json.loads(result.stdout)
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    commands = goal["gates"][queued["gate"]]["commands"]
    assert queued["gate"].startswith("sd-cfr-mixture-falsification-")
    assert "scripts/eval_sd_cfr_mixture.py" in commands[1]
    assert "--candidate-glob" in commands[1]


def test_cli_enqueue_train_creates_gate(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    search_targets = tmp_path / "targets.npz"
    search_targets.write_bytes(b"placeholder")

    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    _cli_set_open_research(script, tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "enqueue-train",
            "--n-iterations",
            "2",
            "--n-traversals",
            "3",
            "--prefix",
            "probe",
            "--save-every",
            "1",
            "--auto-compare",
            "--compare-n-games",
            "6",
            "--compare-strategy-source",
            "policy-head",
            "--search-targets",
            str(search_targets),
            "--search-target-weight",
            "0.05",
            "--search-target-batch-size",
            "7",
            "--timeout-seconds",
            "55",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    queued = json.loads(result.stdout)
    goal = _read_json(tmp_path / "autoresearch-session" / "poker_goal.json")
    command = goal["gates"][queued["gate"]]["commands"][0]
    assert "scripts/poker_autoresearch_train.py" in command
    assert "probe" in command
    assert "--save-every" in command
    assert "1" in command
    assert "--search-targets" in command
    assert str(search_targets) in command
    assert "--search-target-weight" in command
    assert "0.05" in command
    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    assert state["hypothesis_queue"][-1]["postprocess"]["n_games"] == 6
    assert state["hypothesis_queue"][-1]["postprocess"]["strategy_source"] == "policy-head"


def test_cli_enqueue_train_accepts_split_auto_compare_strategy_sources(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"
    subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    _cli_set_open_research(script, tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "enqueue-train",
            "--n-iterations",
            "2",
            "--n-traversals",
            "3",
            "--prefix",
            "avg_policy_probe",
            "--save-every",
            "1",
            "--auto-compare",
            "--compare-candidate-strategy-source",
            "average-policy",
            "--compare-baseline-strategy-source",
            "regret",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    state = _read_json(tmp_path / "autoresearch-session" / "poker_state.json")
    postprocess = state["hypothesis_queue"][-1]["postprocess"]
    assert postprocess["candidate_strategy_source"] == "average-policy"
    assert postprocess["baseline_strategy_source"] == "regret"
