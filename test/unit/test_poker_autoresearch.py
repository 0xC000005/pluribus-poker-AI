import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import poker_ai.research.autoresearch as autoresearch
from poker_ai.research.autoresearch import (
    CommandResult,
    audit_objective_alignment,
    close_cycle,
    continuous,
    enqueue_candidate_comparison,
    enqueue_falsification_ladder,
    enqueue_gpu_training,
    enqueue_methodology_review,
    enqueue_resolver_benchmark,
    enqueue_slumbot_smoke,
    enqueue_cycle,
    init_state,
    new_cycle,
    readiness_report,
    register_research_knob,
    run_gate,
    set_incumbent,
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


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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
    assert "eval-resolver-fixed-states" in goal["gates"]
    assert "slumbot-smoke" in goal["gates"]
    assert "slumbot-solver-smoke" in goal["gates"]
    assert goal["commit_policy"]["mode"] == "batch_by_research_objective"
    assert "methodology_review_required" in goal["review_policy"]["required_for"]
    assert goal["knob_policy"]["max_active_knobs"] == 5
    assert "scripts/play_slumbot.py" in goal["objective_alignment_policy"]["protected_surfaces"]


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
    goal_path.write_text(json.dumps(goal), encoding="utf-8")

    state_path = tmp_path / "autoresearch-session" / "poker_state.json"
    state = _read_json(state_path)
    state["history"].append({"run_id": "kept"})
    state_path.write_text(json.dumps(state), encoding="utf-8")

    init_state(tmp_path)

    synced_goal = _read_json(goal_path)
    synced_state = _read_json(state_path)
    assert "eval-local-confidence" in synced_goal["gates"]
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
    assert (review_dir / "team_review.md").is_file()
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

    (review_dir / "review.md").write_text(
        "# Independent Verification\n\n"
        "Verdict: PARTIAL\n\n"
        "Checked actual artifacts and found mixed support.\n",
        encoding="utf-8",
    )
    (review_dir / "related_work.md").write_text(
        "# Related Work\n\n"
        "- [Deep CFR](https://arxiv.org/abs/1811.00164): canonical baseline.\n",
        encoding="utf-8",
    )
    (review_dir / "benchmark_audit.md").write_text(
        "# Benchmark-Hacking Audit\n\n"
        "Protected surfaces checked: no eval harness weakening.\n\n"
        "Verdict: PASS\n",
        encoding="utf-8",
    )
    (review_dir / "decision.json").write_text(
        json.dumps(
            {
                "decision": "gather_more_evidence",
                "reason": "Local and live evidence disagree.",
                "sources": ["https://arxiv.org/abs/1811.00164"],
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


def test_objective_audit_blocks_protected_surface_without_review(tmp_path):
    init_state(tmp_path)

    result = audit_objective_alignment(
        tmp_path,
        changed_paths=["scripts/play_slumbot.py", "poker_ai/deep_cfr/networks.py"],
    )

    assert result["passed"] is False
    assert result["protected_hits"] == ["scripts/play_slumbot.py"]
    assert "Protected evaluation surfaces changed" in result["errors"][0]


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
        "# Related Work\n\n- [Deep CFR](https://arxiv.org/abs/1811.00164)\n",
        encoding="utf-8",
    )
    (review_dir / "benchmark_audit.md").write_text(
        "# Benchmark-Hacking Audit\n\nVerdict: PASS\n\nNo benchmark weakening.\n",
        encoding="utf-8",
    )
    (review_dir / "decision.json").write_text(
        json.dumps(
            {
                "decision": "proceed",
                "reason": "Parser fix keeps metric semantics unchanged.",
                "sources": ["https://arxiv.org/abs/1811.00164"],
            }
        ),
        encoding="utf-8",
    )

    result = audit_objective_alignment(
        tmp_path,
        changed_paths=["scripts/play_slumbot.py"],
        review_dir=review_dir,
    )

    assert result["passed"] is True
    assert result["protected_hits"] == ["scripts/play_slumbot.py"]


def test_register_research_knob_requires_mechanism_and_enforces_budget(tmp_path):
    init_state(tmp_path)

    for i in range(5):
        register_research_knob(
            tmp_path,
            name=f"knob_{i}",
            default="1",
            failure_class="strategy_quality",
            mechanism="Test one isolated mechanism.",
            rationale="Needed to isolate a falsifiable failure mode.",
            removal_criterion="Retire if the mechanism is falsified.",
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
        )
    except RuntimeError as exc:
        assert "active research knob budget" in str(exc)
    else:
        raise AssertionError("Expected active knob budget enforcement.")


def test_register_research_knob_rejects_sweep_shaped_defaults(tmp_path):
    init_state(tmp_path)

    try:
        register_research_knob(
            tmp_path,
            name="optimizer_grid",
            default="[0.0001,0.001,0.01]",
            failure_class="train_fit",
            mechanism="Hyperparameter sweep over optimizer settings.",
            rationale="This would tune a benchmark instead of testing a mechanism.",
            removal_criterion="Retire after best value is found.",
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


def test_enqueue_slumbot_smoke_creates_candidate_live_gate(tmp_path):
    init_state(tmp_path)
    model = tmp_path / "models" / "candidate.pt"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")

    queued = enqueue_slumbot_smoke(
        tmp_path,
        model,
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
    assert "--greedy" in command
    assert "--no-allin" in command
    assert "--no-solver" in command
    assert "--strategy-source" in command
    assert "policy-head" in command
    assert "123" in command


def test_enqueue_slumbot_smoke_uses_unique_gate_names_with_same_timestamp(
    tmp_path,
    monkeypatch,
):
    init_state(tmp_path)
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


def test_enqueue_gpu_training_creates_candidate_gate(tmp_path):
    init_state(tmp_path)

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
    assert "678" == str(gate["timeout_seconds"])


def test_enqueue_gpu_training_can_request_periodic_checkpoint_comparisons(tmp_path):
    init_state(tmp_path)
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
    }


def test_continuous_queues_comparisons_for_saved_training_checkpoints(tmp_path):
    init_state(tmp_path)
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
    assert "--no-solver" in goal["gates"][queued["gate"]]["commands"][0]
    assert "policy-head" in goal["gates"][queued["gate"]]["commands"][0]


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


def test_cli_enqueue_review_creates_review_gate(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"

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


def test_cli_add_knob_records_governed_knob(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch.py"

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
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["name"] == "search_target_mix"
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
