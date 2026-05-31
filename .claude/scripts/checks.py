"""Deterministic governance checks: hard-stop taxonomy, objective-drift guard,
synthesis-due status, and the gate-runner.

Faithful spec-first ports from ``poker_ai/research/autoresearch.py``. Imports the
session state layer as a sibling module (``gov.py`` and the tests put
``.claude/scripts/`` on sys.path). Golden-tested in
``test/unit/test_governance_port.py``.

(The review-bundle + objective-alignment audit cluster is intentionally NOT here —
it is reached only via the protected review scripts and is deferred to Phase-7
retirement prep; see TaskList #10.)
"""
from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from session import goal_path, log_path, read_json, state_path

# --- taxonomy / family terms ---------------------------------------------------
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
ALLOWED_REVIEW_DECISIONS = {"proceed", "revise", "abandon", "gather_more_evidence"}
ALLOWED_RESEARCH_PHASES = {
    "open_research",
    "callback_state_calibration_debug",
    "exact_gpu_resolving",
    "neural_regret_field_resolving",
    "self_play_policy_improvement",
}


# --- cycle classifiers ---------------------------------------------------------
def cycle_text(item: dict) -> str:
    fields = [
        item.get("type", ""),
        item.get("gate", ""),
        item.get("failure_class", ""),
        item.get("hypothesis", ""),
        item.get("summary", ""),
    ]
    return " ".join(str(field).lower() for field in fields if field is not None)


def cycle_can_run_when_synthesis_due(item: dict) -> bool:
    cycle_type = str(item.get("type", ""))
    return cycle_type == "synthesis" or "review" in cycle_type or "audit" in cycle_type


def is_soft_mechanism_failure(item: dict) -> bool:
    cycle_type = str(item.get("type", ""))
    failure_class = str(item.get("failure_class", ""))
    if not failure_class or failure_class == "none":
        return False
    if failure_class in HARD_STOP_FAILURE_CLASSES:
        return False
    if cycle_type == "synthesis" or "review" in cycle_type or "audit" in cycle_type:
        return False
    return True


def is_local_target_consumer_family(item: dict) -> bool:
    text = cycle_text(item)
    if not any(term in text for term in LOCAL_TARGET_CONSUMER_TERMS):
        return False
    if "audit" in text or "review" in text or "synthesis" in text:
        return False
    if "response oracle" in text and not (
        "target consumer" in text or "npi" in text or "xdo" in text
    ):
        return False
    return True


def is_primary_rl_response_oracle_family(item: dict) -> bool:
    text = cycle_text(item)
    return any(term in text for term in RL_RESPONSE_ORACLE_TERMS)


def failed_local_target_consumer_records(
    history: Sequence[dict], *, limit: int = 12
) -> list[dict]:
    failures: list[dict] = []
    for record in list(history)[-int(limit):]:
        if record.get("outcome") != "failed":
            continue
        if record.get("failure_class") not in {"mechanism_transfer", "strategy_quality"}:
            continue
        if is_local_target_consumer_family(record):
            failures.append(record)
    return failures


def latest_log_heading(root: Path) -> str | None:
    path = log_path(root)
    if not path.exists():
        return None
    headings = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("## ")
    ]
    return headings[-1] if headings else None


def research_drift_status(
    root: str | Path,
    *,
    next_cycle: dict | None = None,
    recent_limit: int = 12,
    failure_threshold: int = 2,
) -> dict:
    """Whether the next item repeats a failed mechanism family (objective-drift guard)."""
    root = Path(root)
    state = read_json(state_path(root))
    history = state.get("history", [])
    recent_failures = failed_local_target_consumer_records(history, limit=int(recent_limit))
    blockers: list[str] = []
    matched_family = None
    requires_review = False
    blocked = False

    if next_cycle is not None and cycle_can_run_when_synthesis_due(next_cycle):
        return {
            "blocked": False,
            "requires_review": False,
            "matched_family": None,
            "recent_failed_same_family_count": len(recent_failures),
            "recent_failed_run_ids": [str(r.get("run_id", "")) for r in recent_failures],
            "latest_research_log_entry": latest_log_heading(root),
            "blockers": [],
        }

    if (
        next_cycle is not None
        and is_local_target_consumer_family(next_cycle)
        and not is_primary_rl_response_oracle_family(next_cycle)
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
        "recent_failed_run_ids": [str(r.get("run_id", "")) for r in recent_failures],
        "latest_research_log_entry": latest_log_heading(root),
        "blockers": blockers,
    }


def pop_next_cycle_for_synthesis_state(
    queue: list[dict], *, synthesis_due: bool
) -> tuple[dict | None, list[dict]]:
    if not synthesis_due or cycle_can_run_when_synthesis_due(queue[0]):
        return queue.pop(0), queue
    for idx, queued in enumerate(queue[1:], start=1):
        if cycle_can_run_when_synthesis_due(queued):
            return queue.pop(idx), queue
    return None, queue


def synthesis_status(root: str | Path) -> dict:
    """Whether the workflow is due for failure synthesis (experiments since last)."""
    root = Path(root)
    goal = read_json(goal_path(root))
    state = read_json(state_path(root))
    interval = int(goal.get("synthesis_policy", {}).get("experiments_per_synthesis", 5))
    history = state.get("history", [])
    last_synthesis_idx = -1
    for idx, record in enumerate(history):
        if record.get("type") == "synthesis":
            last_synthesis_idx = idx
    since = [
        record
        for record in history[last_synthesis_idx + 1:]
        if record.get("type") != "synthesis" and "review" not in str(record.get("type", ""))
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


def circuit_breaker_status(root: str | Path, *, max_consecutive_failures: int = 3) -> dict:
    """Circuit breaker: halt after N consecutive failed cycles.

    A trailing run of >= max_consecutive_failures 'failed' cycles in history trips
    the breaker. (Best-practice addition vs the engine; the autonomous driver calls
    this each cycle and touches the STOP file when it halts.)
    """
    root = Path(root)
    state = read_json(state_path(root))
    consecutive = 0
    for record in reversed(state.get("history", [])):
        if record.get("outcome") == "failed":
            consecutive += 1
        else:
            break
    halt = consecutive >= int(max_consecutive_failures)
    return {
        "halt": halt,
        "consecutive_failures": consecutive,
        "threshold": int(max_consecutive_failures),
        "reason": (
            f"{consecutive} consecutive failed cycles (>= {max_consecutive_failures})"
            if halt
            else ""
        ),
    }


# --- gate runner ---------------------------------------------------------------
@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    seconds: float


CommandRunner = Callable[[Sequence[str], "float | None"], CommandResult]


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


def command_lists(commands: Iterable[Sequence[str]]) -> list[list[str]]:
    return [list(command) for command in commands]


def command_metric(result: CommandResult) -> dict:
    import json

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
    from session import now, write_json

    root = Path(root)
    goal = read_json(goal_path(root))
    gate_config = goal.get("gates", {}).get(gate)
    if gate_config is None and commands is None:
        raise ValueError(f"Unknown gate: {gate}")
    timeout_seconds = None if gate_config is None else gate_config.get("timeout_seconds")
    selected_commands = command_lists(commands or gate_config.get("commands", []))
    if not selected_commands:
        raise ValueError(f"Gate {gate} has no commands configured.")
    started_at = now()
    results = [runner(command, timeout_seconds) for command in selected_commands]
    passed = all(result.returncode == 0 for result in results)
    metrics = {
        "gate": gate,
        "passed": passed,
        "started_at": started_at,
        "finished_at": now(),
        "commands": [command_metric(result) for result in results],
    }
    if run_dir is not None:
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        write_json(run_path / "metrics.json", metrics)
    return metrics
