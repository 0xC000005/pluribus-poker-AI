import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_cfr_budget_frontier as frontier  # noqa: E402
import eval_solver_budget_profiles as profiles  # noqa: E402
from eval_cfr_budget_frontier import summarize_budget_frontier_records  # noqa: E402


def _record():
    return {
        "passed": True,
        "reference_action": 1,
        "reference_allin_prob": 0.05,
        "reference_allin_selected": False,
        "reference_illegal_mass": 0.0,
        "budgets": {
            "5": {
                "l1_to_reference": 0.52,
                "kl_to_reference": 0.26,
                "action": 0,
                "allin_prob": 0.12,
                "allin_selected": False,
                "illegal_mass": 0.0,
                "latency_ms": 180.0,
            },
            "10": {
                "l1_to_reference": 0.36,
                "kl_to_reference": 0.13,
                "action": 1,
                "allin_prob": 0.08,
                "allin_selected": False,
                "illegal_mass": 0.0,
                "latency_ms": 350.0,
            },
        },
    }


def test_budget_frontier_summary_compares_quality_and_latency_by_budget():
    metrics = summarize_budget_frontier_records([_record()], budgets=[5, 10])

    assert metrics["passed"] is True
    assert metrics["n_evaluated"] == 1
    assert metrics["best_l1_budget"] == 10
    assert metrics["budgets"]["5"]["mean_l1_to_reference"] == 0.52
    assert metrics["budgets"]["10"]["action_agreement"] == 1.0
    assert metrics["budgets"]["10"]["latency_ratio_to_min_budget"] == 1.94444444


def test_budget_frontier_summary_fails_when_any_budget_has_illegal_mass():
    record = _record()
    record["budgets"]["10"]["illegal_mass"] = 0.25

    metrics = summarize_budget_frontier_records([record], budgets=[5, 10])

    assert metrics["passed"] is False
    assert metrics["max_illegal_mass"] == 0.25


def test_budget_frontier_summary_fails_when_reference_has_illegal_mass():
    record = _record()
    record["reference_illegal_mass"] = 0.5

    metrics = summarize_budget_frontier_records([record], budgets=[5, 10])

    assert metrics["passed"] is False
    assert metrics["max_illegal_mass"] == 0.5


def test_case_budget_frontier_reuses_one_solver_for_all_budgets(monkeypatch):
    node = SimpleNamespace(children={0: None, 8: None}, is_terminal=False)
    built = []

    class FakeSolver:
        def __init__(self, *args, **kwargs):
            self.solve_calls = []
            self.last_iterations = None
            built.append(self)

        def solve(self, *, n_iterations, **kwargs):
            self.solve_calls.append(n_iterations)
            self.last_iterations = n_iterations

        def navigate(self, nav):
            return node

    def fake_decision(case, parsed, solver, current_node, *, latency_ms=None):
        assert current_node is node
        allin_prob = solver.last_iterations / 100.0
        strategy = np.array([1.0 - allin_prob, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, allin_prob])
        return SimpleNamespace(strategy=strategy, latency_ms=float(solver.last_iterations))

    monkeypatch.setattr(
        frontier,
        "_solver_context",
        lambda case, parsed: ([0, 1, 2, 3], [], 100, 1000, 1000, True, "ck/"),
    )
    monkeypatch.setattr(frontier, "_local_ranges_from_belief", lambda belief, hands: (None, None))
    monkeypatch.setattr(frontier, "_strategy_decision", fake_decision)
    monkeypatch.setattr(frontier, "_parse_nav", lambda action_str, solver: action_str)

    record = frontier._solve_case_budget_frontier(
        case=SimpleNamespace(label="case-0"),
        parsed={"st": 2},
        belief_row=np.ones(1),
        budgets=[5, 10],
        reference_iterations=25,
        solver_backend="cpu",
        solver_update="cfr_plus",
        solver_factory=FakeSolver,
    )

    assert len(built) == 1
    assert built[0].solve_calls == [25, 5, 10]
    assert record["passed"] is True
    assert record["reference_allin_prob"] == 0.25
    assert record["budgets"]["10"]["latency_ms"] == 10.0


def test_solver_budget_profile_iterations_match_live_policy():
    case = SimpleNamespace(action_str="ck/kk/", client_pos=0)
    parsed = {"st": 2, "street_last_bet_to": 0}

    result = profiles.profile_iterations_for_case(case, parsed)

    assert result == {"live": 250, "fast-live": 150}

    pressure_case = SimpleNamespace(action_str="ck/kk/b600", client_pos=1)
    pressure_parsed = {"st": 2, "street_last_bet_to": 600}

    pressure_result = profiles.profile_iterations_for_case(pressure_case, pressure_parsed)

    assert pressure_result == {"live": 350, "fast-live": 250}


def test_solver_budget_profile_summary_reports_drift_and_latency():
    records = [
        {
            "passed": True,
            "profiles": {
                "live": {
                    "action": 1,
                    "strategy": [0.1, 0.9, 0, 0, 0, 0, 0, 0, 0],
                    "latency_ms": 500.0,
                    "iterations": 150,
                    "illegal_mass": 0.0,
                },
                "fast-live": {
                    "action": 1,
                    "strategy": [0.2, 0.8, 0, 0, 0, 0, 0, 0, 0],
                    "latency_ms": 350.0,
                    "iterations": 100,
                    "illegal_mass": 0.0,
                },
            },
        },
        {
            "passed": True,
            "profiles": {
                "live": {
                    "action": 8,
                    "strategy": [0.0, 0.0, 0, 0, 0, 0, 0, 0, 1.0],
                    "latency_ms": 800.0,
                    "iterations": 250,
                    "illegal_mass": 0.0,
                },
                "fast-live": {
                    "action": 1,
                    "strategy": [0.0, 1.0, 0, 0, 0, 0, 0, 0, 0.0],
                    "latency_ms": 400.0,
                    "iterations": 150,
                    "illegal_mass": 0.0,
                },
            },
        },
    ]

    metrics = profiles.summarize_profile_records(records)

    assert metrics["passed"] is True
    assert metrics["n_evaluated"] == 2
    assert metrics["fast_live_action_agreement"] == 0.5
    assert metrics["fast_live_mean_l1_to_live"] == 1.1
    assert metrics["live_mean_latency_ms"] == 650.0
    assert metrics["fast_live_mean_latency_ms"] == 375.0
    assert metrics["fast_live_latency_ratio_to_live"] == 0.57692308
