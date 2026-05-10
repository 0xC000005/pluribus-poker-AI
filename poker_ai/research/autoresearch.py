"""Autoresearch workflow primitives for poker AI experiments.

The module intentionally stays independent from trainer internals. It provides
the durable loop mechanics: local state, gates, metrics, logs, and STOP-file
handling.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence


SESSION_DIR = "autoresearch-session"
RUNS_DIR = "poker_runs"
GOAL_FILE = "poker_goal.json"
STATE_FILE = "poker_state.json"
KNOBS_FILE = "poker_knobs.tsv"
STOP_FILE = "STOP"
RESEARCH_LOG = "RESEARCH_LOG.md"


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


def _stop_path(root: Path) -> Path:
    return _session(root) / STOP_FILE


def _log_path(root: Path) -> Path:
    return root / RESEARCH_LOG


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_goal() -> dict:
    return {
        "objective": (
            "Train a learned full-deck poker engine on a personal PC that uses "
            "search efficiently during play and improves against Slumbot without "
            "hand-crafted poker-strategy rules."
        ),
        "constraints": [
            "keep generated checkpoints and raw run artifacts out of git",
            "do not add opponent-specific or street-specific human strategy rules",
            "run Tier 0 integrity before promoting any result",
            "classify every failed or inconclusive cycle",
        ],
        "primary_metric": "lower_95_ci_mbb_per_hand_vs_incumbent",
        "hard_stop_conditions": [
            "STOP file exists",
            "Tier 0 integrity gate fails",
            "metric output is missing or non-mechanical",
            "Slumbot credentials, network access, or API limits block evaluation",
            "cycle would overwrite an incumbent checkpoint",
            "cycle requires a hand-crafted opponent rule",
        ],
        "gates": {
            "tier0": {
                "description": "Fast integrity tests for masks, Slumbot mapping, and feature parity.",
                "timeout_seconds": 600,
                "commands": [
                    [
                        sys.executable,
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
                        sys.executable,
                        "scripts/test_feature_encoding.py",
                    ]
                ],
            },
            "eval-local": {
                "description": "Small local fixed-seed checkpoint evaluation against random opponents.",
                "timeout_seconds": 900,
                "commands": [
                    [
                        sys.executable,
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
                        sys.executable,
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
                        sys.executable,
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
                        sys.executable,
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
                        sys.executable,
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
            "slumbot-smoke": {
                "description": "Tiny live Slumbot API smoke with conservative diagnostics settings.",
                "timeout_seconds": 600,
                "commands": [
                    [
                        sys.executable,
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
                        sys.executable,
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
    session.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)

    default_goal = _default_goal()
    goal_path = _goal_path(root)
    if force or not goal_path.exists():
        _write_json(goal_path, default_goal)
    else:
        goal = _read_json(goal_path)
        goal.setdefault("gates", {})
        for gate_name, gate_config in default_goal["gates"].items():
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
        _write_json(goal_path, goal)

    state_path = _state_path(root)
    if force or not state_path.exists():
        _write_json(state_path, _default_state())

    knobs = _knobs_path(root)
    if force or not knobs.exists():
        knobs.write_text(
            "name\tdefault\tfailure_class\trationale\tremoval_criterion\n",
            encoding="utf-8",
        )

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
    state.setdefault("hypothesis_queue", []).append(item)
    state["updated_at"] = _now()
    _write_json(_state_path(root), state)
    return item


def _resolve_existing_path(root: Path, path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    resolved = candidate if candidate.is_absolute() else root / candidate
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


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
) -> dict:
    """Create and queue a one-off candidate-vs-incumbent local comparison gate."""
    root = Path(root)
    candidate = _resolve_existing_path(root, candidate_checkpoint, label="Candidate checkpoint")
    state = _read_json(_state_path(root))

    if baseline_checkpoint is None:
        incumbent = state.get("incumbent_checkpoint") or {}
        baseline_checkpoint = incumbent.get("checkpoint")
        if baseline_checkpoint is None:
            raise RuntimeError("No baseline checkpoint provided and no incumbent is recorded.")
    baseline = _resolve_existing_path(root, baseline_checkpoint, label="Baseline checkpoint")

    gate_name = (
        f"eval-candidate-compare-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{_slug(candidate.stem)}"
    )
    command = [
        sys.executable,
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

    gate_config = {
        "description": (
            "One-off local candidate-vs-incumbent comparison. This gate emits "
            "delta metrics and promotion blockers; it is not a standalone "
            "strategy-strength proof."
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
            "claiming local-only promotion."
        ),
        cycle_type="experiment",
        failure_class="strategy_quality",
        gate=gate_name,
    )


def enqueue_slumbot_smoke(
    root: str | Path,
    model: str | Path,
    *,
    hands: int = 5,
    greedy: bool = True,
    no_allin: bool = True,
    no_solver: bool = True,
    timeout_seconds: int = 600,
) -> dict:
    """Create and queue a sparse live Slumbot smoke for a candidate checkpoint."""
    root = Path(root)
    model_path = _resolve_existing_path(root, model, label="Slumbot model checkpoint")
    gate_name = (
        f"slumbot-candidate-smoke-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{_slug(model_path.stem)}"
    )
    command = [
        sys.executable,
        "scripts/poker_autoresearch_slumbot.py",
        "--model",
        str(model_path),
        "--hands",
        str(hands),
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    if greedy:
        command.append("--greedy")
    if no_allin:
        command.append("--no-allin")
    if no_solver:
        command.append("--no-solver")

    goal = _read_json(_goal_path(root))
    goal.setdefault("gates", {})[gate_name] = {
        "description": (
            "One-off sparse live Slumbot smoke for a candidate checkpoint. "
            "This is an integration and distribution-shift diagnostic, not a "
            "promotion gate by itself."
        ),
        "timeout_seconds": timeout_seconds,
        "commands": [command],
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


def append_research_log(
    root: str | Path,
    *,
    cycle: dict,
    outcome: str,
    failure_class: str,
    summary: str,
    metrics: dict,
) -> None:
    root = Path(root)
    log = _log_path(root)
    if not log.exists():
        log.write_text("# Research Log\n\n", encoding="utf-8")

    metric_summary = json.dumps(metrics, sort_keys=True)
    entry = (
        f"## {cycle['run_id']} - {outcome}\n\n"
        f"- Timestamp: {_now()}\n"
        f"- Type: {cycle['type']}\n"
        f"- Gate: {cycle['gate']}\n"
        f"- Hypothesis: {cycle['hypothesis']}\n"
        f"- Failure class: {failure_class}\n"
        f"- Summary: {summary}\n"
        f"- Metrics: `{metric_summary}`\n\n"
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
    )
    return closed


def continuous(
    root: str | Path,
    *,
    max_cycles: int | None = None,
    runner: CommandRunner = run_command,
    sleep_seconds: float = 30.0,
    max_idle_checks: int | None = None,
) -> dict:
    root = Path(root)
    cycles_completed = 0
    idle_checks = 0

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
        if not queue:
            if max_cycles is not None:
                return {"cycles_completed": cycles_completed, "stopped_reason": "empty_queue"}
            idle_checks += 1
            if max_idle_checks is not None and idle_checks >= max_idle_checks:
                return {"cycles_completed": cycles_completed, "stopped_reason": "idle_limit"}
            time.sleep(sleep_seconds)
            continue
        idle_checks = 0

        item = queue.pop(0)
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
            return {"cycles_completed": cycles_completed, "stopped_reason": "gate_failed"}
        if max_cycles is None:
            time.sleep(sleep_seconds)

    return {"cycles_completed": cycles_completed, "stopped_reason": "max_cycles"}
