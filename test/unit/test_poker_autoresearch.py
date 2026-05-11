import json
import subprocess
import sys
from pathlib import Path

from poker_ai.research.autoresearch import (
    CommandResult,
    close_cycle,
    continuous,
    enqueue_candidate_comparison,
    enqueue_resolver_benchmark,
    enqueue_slumbot_smoke,
    enqueue_cycle,
    init_state,
    new_cycle,
    readiness_report,
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
    assert "123" in command


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
