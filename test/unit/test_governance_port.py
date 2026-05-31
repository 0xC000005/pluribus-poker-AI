"""Golden parity tests for the lean Claude-native governance core.

Pins the bundled scripts under ``.claude/scripts/`` (loaded by path, as the guard
hook does) against the legacy engine (``poker_ai/research/autoresearch.py``) so the
reimplementation cannot silently diverge from the behaviour it replaces.

Kept core: surfaces (protected registry), session (state io + cycle mutators),
checks (drift guard, synthesis, hard-stop taxonomy, gate-runner). The dropped
over-port (goal sync/init/snapshot, unused metric helpers) has no tests here.

Named test_governance_port.py (not test_*autoresearch*.py) so it does not match the
protected ``test/unit/test_*autoresearch*.py`` surface and stays editable.
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from poker_ai.research import autoresearch as eng

_SCRIPTS = Path(__file__).resolve().parents[2] / ".claude" / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import checks  # noqa: E402
import session  # noqa: E402
import surfaces  # noqa: E402


def _strip_timestamps(state: dict) -> dict:
    out = dict(state)
    out.pop("created_at", None)
    out.pop("updated_at", None)
    return out


# ---------------------------------------------------------------- surfaces.py ---
def test_surfaces_superset_of_engine():
    # surfaces is the guard's single source of truth: it must include every legacy
    # engine surface/pattern (no dropped protection) and may add more (the
    # Claude-native self-protection — see test_governance_core_is_protected).
    assert set(eng.PROTECTED_EVAL_SURFACES) <= set(surfaces.PROTECTED_EVAL_SURFACES)
    assert set(eng.PROTECTED_EVAL_SURFACE_PATTERNS) <= set(surfaces.PROTECTED_EVAL_SURFACE_PATTERNS)


_ENGINE_PROTECTED_SAMPLES = [
    "scripts/solver.py",
    "scripts/poker_objective_audit.py",
    "scripts/eval_anything.py",
    "scripts/build_turn_river_trace.py",
    "scripts/analyze_x_trace_y.py",
    "scripts/foo_resolver_bar.py",
    "poker_ai/research/promotion.py",
    "poker_ai/research/x_slumbot_y.py",
    "test/unit/test_slumbot_mapping.py",
    "test/unit/test_xyz_parity.py",
    "test/unit/test_poker_autoresearch_eval.py",
    "./scripts/solver.py",
]
_UNPROTECTED_SAMPLES = ["poker_ai/deep_cfr/networks.py", "README.md", "poker_ai/games/full_deck/state.py"]


@pytest.mark.parametrize("path", _ENGINE_PROTECTED_SAMPLES)
def test_is_protected_covers_engine(path):
    # Superset direction: whatever the engine protects, surfaces must protect too.
    if eng._is_protected_path(path, eng.PROTECTED_EVAL_SURFACES, eng.PROTECTED_EVAL_SURFACE_PATTERNS):
        assert surfaces.is_protected(path)


@pytest.mark.parametrize("path", _UNPROTECTED_SAMPLES)
def test_clearly_unprotected(path):
    assert not surfaces.is_protected(path)


def test_every_explicit_surface_is_protected():
    for surface in surfaces.PROTECTED_EVAL_SURFACES:
        assert surfaces.is_protected(surface), surface


def test_governance_core_is_protected():
    # The live governance core + its parity proof are self-protected (auditor-finding fix):
    # editing them trips the guard, so the golden tests can't be silently weakened.
    for path in (
        ".claude/scripts/surfaces.py",
        ".claude/scripts/checks.py",
        ".claude/scripts/session.py",
        ".claude/scripts/gov.py",
        ".claude/hooks/guard_protected_surfaces.py",
        "test/unit/test_governance_port.py",
    ):
        assert surfaces.is_protected(path), path


# ----------------------------------------------------------------- session.py ---
def test_layout_constants_match_engine():
    assert session.SESSION_DIR == eng.SESSION_DIR
    assert session.GOAL_FILE == eng.GOAL_FILE
    assert session.STATE_FILE == eng.STATE_FILE
    assert session.KNOBS_FILE == eng.KNOBS_FILE
    assert session.STOP_FILE == eng.STOP_FILE
    assert session.RESEARCH_LOG == eng.RESEARCH_LOG
    assert session.KNOB_COLUMNS == eng.KNOB_COLUMNS


def test_default_state_structure_matches_engine():
    ported, engine = session.default_state(), eng._default_state()
    assert ported.keys() == engine.keys()
    assert _strip_timestamps(ported) == _strip_timestamps(engine)


def test_slug_matches_engine():
    for text in ["New Search Objective!", "a", "", "Tier 0 -- x!!", "  spaced  ", "z" * 200]:
        assert session.slug(text) == eng._slug(text)


def test_first_stdout_json_matches_engine():
    for sample in [
        {"commands": [{"stdout_json": {"a": 1}}]},
        {"commands": [{"x": 1}, {"stdout_json": {"b": 2}}]},
        {"commands": [{"stdout_json": [1, 2]}]},
        {"commands": []},
        {},
    ]:
        assert session.first_stdout_json(sample) == eng._first_stdout_json(sample)


def _make_session(tmp_path: Path) -> Path:
    sess = tmp_path / "autoresearch-session"
    (sess / "poker_runs").mkdir(parents=True)
    (sess / "poker_reviews").mkdir(parents=True)
    (sess / "poker_goal.json").write_text("{}")
    (sess / "poker_state.json").write_text(json.dumps(session.default_state()))
    (sess / "poker_knobs.tsv").write_text("name\n")
    (tmp_path / "RESEARCH_LOG.md").write_text("# log\n")
    return sess


def test_readiness_report_matches_engine(tmp_path):
    assert session.readiness_report(tmp_path) == eng.readiness_report(tmp_path)
    sess = _make_session(tmp_path)
    assert session.readiness_report(tmp_path) == eng.readiness_report(tmp_path)
    assert session.readiness_report(tmp_path)["ready"] is True
    (sess / "STOP").write_text("")
    assert session.readiness_report(tmp_path) == eng.readiness_report(tmp_path)
    assert session.readiness_report(tmp_path)["ready"] is False


def _seed_state(root: Path) -> Path:
    sess = root / "autoresearch-session"
    sess.mkdir(parents=True)
    (sess / "poker_state.json").write_text(json.dumps(session.default_state()))
    return root


def test_set_incumbent_matches_engine(tmp_path):
    _seed_state(tmp_path)
    with pytest.raises(FileNotFoundError):
        session.set_incumbent(tmp_path, tmp_path / "nope.pt", reason="x")
    ckpt = tmp_path / "model.pt"
    ckpt.write_text("x")
    out = session.set_incumbent(tmp_path, ckpt, reason="best")
    assert out["checkpoint"] == str(ckpt) and out["reason"] == "best" and "set_at" in out


def _norm_cycle(cycle: dict) -> dict:
    out = dict(cycle)
    for key in ("run_id", "started_at", "closed_at", "run_dir"):
        out.pop(key, None)
    return out


def test_new_cycle_matches_engine(tmp_path):
    g, e = _seed_state(tmp_path / "g"), _seed_state(tmp_path / "e")
    kw = dict(hypothesis="Test H", cycle_type="experiment", failure_class="eval_invalid", gate="tier0")
    gc, ec = session.new_cycle(g, **kw), eng.new_cycle(e, **kw)
    assert _norm_cycle(gc) == _norm_cycle(ec)
    assert (Path(gc["run_dir"]) / "cycle.json").exists()
    with pytest.raises(RuntimeError):
        session.new_cycle(g, hypothesis="x", cycle_type="experiment", failure_class="none", gate="tier0")


def test_enqueue_cycle_matches_engine(tmp_path):
    g, e = _seed_state(tmp_path / "g"), _seed_state(tmp_path / "e")
    kw = dict(hypothesis="H", cycle_type="experiment", failure_class="eval_invalid", gate="tier0")
    gi, ei = session.enqueue_cycle(g, **kw), eng.enqueue_cycle(e, **kw)
    gi.pop("id")
    ei.pop("id")
    assert gi == ei


def test_close_cycle_matches_engine(tmp_path):
    g, e = _seed_state(tmp_path / "g"), _seed_state(tmp_path / "e")
    kw = dict(hypothesis="H", cycle_type="experiment", failure_class="eval_invalid", gate="tier0")
    gc, ec = session.new_cycle(g, **kw), eng.new_cycle(e, **kw)
    gclosed = session.close_cycle(g, gc["run_id"], outcome="passed", failure_class="none", summary="ok")
    eclosed = eng.close_cycle(e, ec["run_id"], outcome="passed", failure_class="none", summary="ok")
    assert _norm_cycle(gclosed) == _norm_cycle(eclosed)
    state = json.loads((g / "autoresearch-session" / "poker_state.json").read_text())
    assert state["active_cycle"] is None and len(state["history"]) == 1


def test_append_research_log_format_matches_engine(tmp_path):
    g, e = tmp_path / "g", tmp_path / "e"
    g.mkdir()
    e.mkdir()
    cycle = {"run_id": "RID", "type": "experiment", "gate": "tier0", "hypothesis": "H"}
    metrics = {"passed": True, "gate": "tier0", "commands": [{"stdout_json": {"avg_chips_per_hand": 1.5, "mode": "x"}}]}
    kw = dict(cycle=cycle, outcome="passed", failure_class="none", summary="ok", metrics=metrics)
    session.append_research_log(g, **kw)
    eng.append_research_log(e, **kw)

    def strip_ts(text: str) -> str:
        return "\n".join(line for line in text.splitlines() if not line.startswith("- Timestamp:"))

    assert strip_ts((g / "RESEARCH_LOG.md").read_text()) == strip_ts((e / "RESEARCH_LOG.md").read_text())


# ------------------------------------------------------------------ checks.py ---
def test_taxonomy_constants_match_engine():
    assert checks.HARD_STOP_FAILURE_CLASSES == eng.HARD_STOP_FAILURE_CLASSES
    assert checks.LOCAL_TARGET_CONSUMER_TERMS == eng.LOCAL_TARGET_CONSUMER_TERMS
    assert checks.RL_RESPONSE_ORACLE_TERMS == eng.RL_RESPONSE_ORACLE_TERMS
    assert checks.ALLOWED_REVIEW_DECISIONS == eng.ALLOWED_REVIEW_DECISIONS
    assert checks.ALLOWED_RESEARCH_PHASES == eng.ALLOWED_RESEARCH_PHASES


_CYCLE_ITEMS = [
    {"type": "experiment", "failure_class": "strategy_quality"},
    {"type": "experiment", "failure_class": "eval_invalid"},
    {"type": "experiment", "failure_class": "none"},
    {"type": "methodology_review", "failure_class": "strategy_quality"},
    {"type": "synthesis", "failure_class": "compute_efficiency"},
    {"type": "objective_audit", "failure_class": "x"},
    {"type": "experiment", "hypothesis": "XDO target consumer search target"},
    {"type": "experiment", "hypothesis": "PPO response oracle rainbow"},
    {"type": "experiment", "hypothesis": "npi response oracle"},
    {"type": "methodology_review", "hypothesis": "xdo"},
]


@pytest.mark.parametrize("item", _CYCLE_ITEMS)
def test_cycle_classifiers_match_engine(item):
    assert checks.is_soft_mechanism_failure(item) == eng._is_soft_mechanism_failure(item)
    assert checks.cycle_can_run_when_synthesis_due(item) == eng._cycle_can_run_when_synthesis_due(item)
    assert checks.is_local_target_consumer_family(item) == eng._is_local_target_consumer_family(item)
    assert checks.is_primary_rl_response_oracle_family(item) == eng._is_primary_rl_response_oracle_family(item)
    assert checks.cycle_text(item) == eng._cycle_text(item)


def test_failed_local_target_consumer_records_matches_engine():
    history = [
        {"run_id": "a", "outcome": "failed", "failure_class": "strategy_quality", "type": "experiment", "hypothesis": "xdo target consumer"},
        {"run_id": "b", "outcome": "passed", "failure_class": "strategy_quality", "type": "experiment", "hypothesis": "xdo"},
        {"run_id": "c", "outcome": "failed", "failure_class": "eval_invalid", "type": "experiment", "hypothesis": "xdo"},
        {"run_id": "d", "outcome": "failed", "failure_class": "mechanism_transfer", "type": "experiment", "hypothesis": "search target"},
    ]
    assert checks.failed_local_target_consumer_records(history) == eng._failed_local_target_consumer_records(history)


def _drift_root(tmp_path, n=2):
    sess = tmp_path / "autoresearch-session"
    sess.mkdir(parents=True)
    history = [
        {"run_id": f"r{i}", "outcome": "failed", "failure_class": "strategy_quality",
         "type": "experiment", "hypothesis": "xdo search target"}
        for i in range(n)
    ]
    (sess / "poker_state.json").write_text(json.dumps({"history": history}))
    (tmp_path / "RESEARCH_LOG.md").write_text("# Log\n\n## Latest Heading\n")
    return tmp_path


@pytest.mark.parametrize("next_cycle", [
    None,
    {"type": "experiment", "hypothesis": "xdo search target"},
    {"type": "methodology_review", "hypothesis": "xdo"},
    {"type": "experiment", "hypothesis": "ppo response oracle"},
])
def test_research_drift_status_matches_engine(tmp_path, next_cycle):
    root = _drift_root(tmp_path)
    assert checks.research_drift_status(root, next_cycle=next_cycle) == eng.research_drift_status(root, next_cycle=next_cycle)


def test_research_drift_status_blocks_repeat_family(tmp_path):
    root = _drift_root(tmp_path, n=2)
    result = checks.research_drift_status(root, next_cycle={"type": "experiment", "hypothesis": "xdo search target"})
    assert result["blocked"] is True and result["matched_family"] == "local_target_consumer"


def test_pop_next_cycle_for_synthesis_state_matches_engine():
    base = [{"type": "experiment"}, {"type": "methodology_review"}, {"type": "experiment"}]
    for due in (False, True):
        ported = checks.pop_next_cycle_for_synthesis_state(copy.deepcopy(base), synthesis_due=due)
        engine = eng._pop_next_cycle_for_synthesis_state(copy.deepcopy(base), synthesis_due=due)
        assert ported == engine


def test_synthesis_status_matches_engine(tmp_path):
    sess = tmp_path / "autoresearch-session"
    sess.mkdir(parents=True)
    (sess / "poker_goal.json").write_text(
        json.dumps({"synthesis_policy": {"experiments_per_synthesis": 3, "required_fields": ["current causal model"]}})
    )
    history = [
        {"type": "experiment", "run_id": "e1"},
        {"type": "synthesis", "run_id": "s1"},
        {"type": "experiment", "run_id": "e2"},
        {"type": "methodology_review", "run_id": "r1"},
        {"type": "experiment", "run_id": "e3"},
    ]
    (sess / "poker_state.json").write_text(json.dumps({"history": history}))
    assert checks.synthesis_status(tmp_path) == eng.synthesis_status(tmp_path)


def _write_goal(tmp_path: Path, goal: dict) -> None:
    sess = tmp_path / "autoresearch-session"
    sess.mkdir(parents=True, exist_ok=True)
    (sess / "poker_goal.json").write_text(json.dumps(goal))


def test_run_gate_parity_with_stub_runner(tmp_path):
    _write_goal(tmp_path, {"gates": {"g": {"timeout_seconds": 5, "commands": [["echo", "a"], ["echo", "b"]]}}})

    def gov_runner(cmd, timeout):
        return checks.CommandResult(command=list(cmd), returncode=0, stdout='{"k": 1}', stderr="", seconds=0.0)

    def eng_runner(cmd, timeout):
        return eng.CommandResult(command=list(cmd), returncode=0, stdout='{"k": 1}', stderr="", seconds=0.0)

    ported = checks.run_gate(tmp_path, "g", runner=gov_runner)
    engine = eng.run_gate(tmp_path, "g", runner=eng_runner)
    for metrics in (ported, engine):
        metrics.pop("started_at")
        metrics.pop("finished_at")
    assert ported == engine
    assert ported["passed"] is True and ported["commands"][0]["stdout_json"] == {"k": 1}


def test_run_gate_fails_when_any_command_nonzero(tmp_path):
    _write_goal(tmp_path, {"gates": {"g": {"commands": [["a"], ["b"]]}}})

    def runner(cmd, timeout):
        rc = 1 if list(cmd) == ["b"] else 0
        return checks.CommandResult(command=list(cmd), returncode=rc, stdout="", stderr="", seconds=0.0)

    assert checks.run_gate(tmp_path, "g", runner=runner)["passed"] is False


def test_run_gate_unknown_gate_raises(tmp_path):
    _write_goal(tmp_path, {"gates": {}})
    with pytest.raises(ValueError):
        checks.run_gate(tmp_path, "nope")


def test_run_command_parity_real():
    ported = checks.run_command(["echo", "hi"])
    engine = eng.run_command(["echo", "hi"])
    assert ported.returncode == engine.returncode == 0
    assert ported.stdout == engine.stdout == "hi\n"
    assert ported.command == engine.command == ["echo", "hi"]


# ---------------------------------------------------- guard hook (end-to-end) ---
_REPO = Path(__file__).resolve().parents[2]
_HOOK = _REPO / ".claude" / "hooks" / "guard_protected_surfaces.py"


def _run_guard(file_path: str):
    payload = json.dumps({"tool_input": {"file_path": file_path}})
    env = {**os.environ, "POKER_AI_ALLOW_PROTECTED": "0"}
    return subprocess.run(
        [sys.executable, str(_HOOK)], input=payload, capture_output=True, text=True, env=env
    )


def test_guard_hook_blocks_protected_end_to_end():
    # Pins the hook's ACTUAL blocking behaviour so a future surfaces-path drift
    # cannot silently fail-open (a missing-registry load returns exit 0 by design).
    blocked = _run_guard(str(_REPO / "scripts" / "solver.py"))
    assert blocked.returncode == 2, blocked.stderr
    assert "BLOCKED" in blocked.stderr
    # self-protection: the governance core is now guarded end-to-end too
    assert _run_guard(str(_REPO / ".claude" / "scripts" / "checks.py")).returncode == 2
    # a genuinely-unprotected file is still allowed
    assert _run_guard(str(_REPO / "poker_ai" / "deep_cfr" / "networks.py")).returncode == 0


# ------------------------------------------------------ circuit breaker (new) ---
def _state_root(root: Path, outcomes: list[str]) -> Path:
    sess = root / "autoresearch-session"
    sess.mkdir(parents=True)
    (sess / "poker_state.json").write_text(json.dumps({"history": [{"outcome": o} for o in outcomes]}))
    return root


def test_circuit_breaker_status_behavior(tmp_path):
    assert checks.circuit_breaker_status(_state_root(tmp_path / "a", ["failed", "failed", "failed"]), max_consecutive_failures=3)["halt"] is True
    assert checks.circuit_breaker_status(_state_root(tmp_path / "b", ["failed", "passed", "failed"]), max_consecutive_failures=2)["halt"] is False
    trailing = checks.circuit_breaker_status(_state_root(tmp_path / "c", ["failed", "failed", "passed"]), max_consecutive_failures=1)
    assert trailing["halt"] is False and trailing["consecutive_failures"] == 0
    tail = checks.circuit_breaker_status(_state_root(tmp_path / "d", ["passed", "failed", "failed"]), max_consecutive_failures=2)
    assert tail["halt"] is True and tail["consecutive_failures"] == 2


def test_circuit_check_trips_stop_via_dispatcher(tmp_path):
    root = _state_root(tmp_path / "cb", ["failed", "failed", "failed"])
    gov = _REPO / ".claude" / "scripts" / "gov.py"
    result = subprocess.run(
        [sys.executable, str(gov), "--root", str(root), "circuit-check", "--max-failures", "2"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["halt"] is True and payload.get("tripped_stop") is True
    assert (root / "autoresearch-session" / "STOP").exists()
