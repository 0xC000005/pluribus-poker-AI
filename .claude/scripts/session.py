"""Autoresearch session state: paths, JSON I/O, and the cycle state-machine.

Self-contained (stdlib only) so it can be loaded standalone by path. Faithful
ports from ``poker_ai/research/autoresearch.py`` of the deterministic state layer
and cycle mutators — these are precision tasks (a corrupt write wedges the loop),
so they stay tested code rather than LLM JSON hand-editing. Golden-tested in
``test/unit/test_governance_port.py``.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

SESSION_DIR = "autoresearch-session"
RUNS_DIR = "poker_runs"
REVIEWS_DIR = "poker_reviews"
GOAL_FILE = "poker_goal.json"
STATE_FILE = "poker_state.json"
KNOBS_FILE = "poker_knobs.tsv"
STOP_FILE = "STOP"
RESEARCH_LOG = "RESEARCH_LOG.md"
REVIEW_MANIFESTS_DIR = Path("docs") / "research_protocols" / "poker_review_manifests"
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

_LOG_KEY_METRICS = (
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
)


def now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def session_dir(root: Path) -> Path:
    return root / SESSION_DIR


def goal_path(root: Path) -> Path:
    return session_dir(root) / GOAL_FILE


def state_path(root: Path) -> Path:
    return session_dir(root) / STATE_FILE


def knobs_path(root: Path) -> Path:
    return session_dir(root) / KNOBS_FILE


def runs_path(root: Path) -> Path:
    return session_dir(root) / RUNS_DIR


def reviews_path(root: Path) -> Path:
    return session_dir(root) / REVIEWS_DIR


def review_manifest_dir(root: Path) -> Path:
    return root / REVIEW_MANIFESTS_DIR


def stop_path(root: Path) -> Path:
    return session_dir(root) / STOP_FILE


def log_path(root: Path) -> Path:
    return root / RESEARCH_LOG


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def slug(text: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in text)
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts[:8]) or "cycle"


def first_stdout_json(metrics: dict) -> dict | None:
    for command_metric in metrics.get("commands", []):
        payload = command_metric.get("stdout_json")
        if isinstance(payload, dict):
            return payload
    return None


def default_state() -> dict:
    return {
        "status": "ready",
        "created_at": now(),
        "updated_at": now(),
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


def readiness_report(root: str | Path) -> dict:
    root = Path(root)
    required = [
        goal_path(root),
        state_path(root),
        knobs_path(root),
        reviews_path(root),
        runs_path(root),
        log_path(root),
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    stop_requested = stop_path(root).exists()
    return {
        "ready": not missing and not stop_requested,
        "missing": missing,
        "stop_requested": stop_requested,
        "stop_file": str(stop_path(root)),
        "session_dir": str(session_dir(root)),
    }


def set_incumbent(root: str | Path, checkpoint: str | Path, *, reason: str) -> dict:
    root = Path(root)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Incumbent checkpoint does not exist: {checkpoint_path}")
    state = read_json(state_path(root))
    incumbent = {"checkpoint": str(checkpoint_path), "reason": reason, "set_at": now()}
    state["incumbent_checkpoint"] = incumbent
    state["updated_at"] = now()
    write_json(state_path(root), state)
    return incumbent


def new_cycle(
    root: str | Path, *, hypothesis: str, cycle_type: str, failure_class: str, gate: str
) -> dict:
    root = Path(root)
    state = read_json(state_path(root))
    if state.get("active_cycle") is not None:
        raise RuntimeError("Cannot create a new cycle while another cycle is active.")
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{slug(hypothesis)}"
    run_dir = runs_path(root) / run_id
    cycle = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "hypothesis": hypothesis,
        "type": cycle_type,
        "failure_class": failure_class,
        "gate": gate,
        "started_at": now(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "cycle.json", cycle)
    state["active_cycle"] = cycle
    state["updated_at"] = now()
    write_json(state_path(root), state)
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
    state = read_json(state_path(root))
    item = {
        "id": f"{now().replace(':', '').replace('-', '')}-{slug(hypothesis)}",
        "type": cycle_type,
        "hypothesis": hypothesis,
        "failure_class": failure_class,
        "gate": gate,
    }
    if postprocess is not None:
        item["postprocess"] = postprocess
    state.setdefault("hypothesis_queue", []).append(item)
    state["updated_at"] = now()
    write_json(state_path(root), state)
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
    log = log_path(root)
    if not log.exists():
        log.write_text("# Research Log\n\n", encoding="utf-8")
    key_metrics = {"passed": metrics.get("passed"), "gate": metrics.get("gate")}
    payload = first_stdout_json(metrics)
    if payload:
        for key in _LOG_KEY_METRICS:
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
        f"- Timestamp: {now()}\n"
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
    state = read_json(state_path(root))
    cycle = state.get("active_cycle")
    if cycle is None or cycle.get("run_id") != run_id:
        raise RuntimeError(f"Active cycle does not match run_id {run_id}.")
    metrics: dict = {}
    if metrics_path is not None:
        metrics = read_json(Path(metrics_path))
    closed = {
        **cycle,
        "outcome": outcome,
        "closed_at": now(),
        "failure_class": failure_class,
        "summary": summary,
    }
    state["active_cycle"] = None
    state["last_metrics"] = metrics
    state.setdefault("history", []).append(closed)
    state["updated_at"] = now()
    write_json(state_path(root), state)
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
