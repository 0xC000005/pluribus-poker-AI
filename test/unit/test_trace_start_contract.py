import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from poker_ai.research.resolver_benchmark import ResolverBenchmarkCase


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from play_slumbot import build_features, get_legal_mask_from_parsed, parse_action  # noqa: E402


def _sample_case() -> ResolverBenchmarkCase:
    return ResolverBenchmarkCase(
        label="trace-hand3-decision1-street2",
        hole_cards=("Kc", "Js"),
        board=("Jc", "5s", "3d", "9d"),
        action_str="b200c/b100c/",
        client_pos=0,
        source="slumbot_trace",
    )


def test_trace_start_observation_matches_slumbot_feature_and_mask():
    from poker_ai.research.trace_start_contract import build_trace_start_observation

    case = _sample_case()
    observation = build_trace_start_observation(case, source_flag=1.0)
    parsed = parse_action(case.action_str)

    expected_features = build_features(
        list(case.hole_cards),
        list(case.board),
        case.action_str,
        case.client_pos,
        parsed,
    )
    expected_mask = get_legal_mask_from_parsed(parsed, case.action_str, case.client_pos)

    assert observation.contract_kind == "observation_only"
    assert observation.label == case.label
    assert observation.street == 2
    assert observation.client_pos == 0
    assert observation.action_str == case.action_str
    np.testing.assert_allclose(observation.features, expected_features)
    np.testing.assert_allclose(observation.legal_mask, expected_mask)


def test_trace_start_source_flag_is_explicit_context_not_feature_rewrite(tmp_path):
    from poker_ai.research.trace_start_contract import load_trace_start_observations

    case = _sample_case()
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "label": case.label,
                        "hole_cards": list(case.hole_cards),
                        "board": list(case.board),
                        "action_str": case.action_str,
                        "client_pos": case.client_pos,
                        "source": case.source,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    observations = load_trace_start_observations(cases_path, source_flag=0.75)

    assert len(observations) == 1
    obs = observations[0]
    assert obs.source == "slumbot_trace"
    assert obs.source_flag == 0.75
    assert obs.features_with_context.shape == (obs.features.shape[0] + 1,)
    np.testing.assert_allclose(obs.features_with_context[:-1], obs.features)
    assert obs.features_with_context[-1] == 0.75


def test_trace_start_rejects_malformed_action():
    from poker_ai.research.trace_start_contract import build_trace_start_observation

    case = ResolverBenchmarkCase(
        label="bad-action",
        hole_cards=("Kc", "Js"),
        board=("Jc", "5s", "3d", "9d"),
        action_str="b200x",
        client_pos=0,
        source="slumbot_trace",
    )

    try:
        build_trace_start_observation(case)
    except ValueError as exc:
        assert "invalid Slumbot action" in str(exc)
    else:
        raise AssertionError("malformed Slumbot action should be rejected")


def test_validate_trace_start_contract_cli_writes_metrics(tmp_path):
    case = _sample_case()
    cases_path = tmp_path / "cases.json"
    output_json = tmp_path / "metrics.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "label": case.label,
                        "hole_cards": list(case.hole_cards),
                        "board": list(case.board),
                        "action_str": case.action_str,
                        "client_pos": case.client_pos,
                        "source": case.source,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_trace_start_contract.py",
            "--cases",
            str(cases_path),
            "--output-json",
            str(output_json),
            "--source-flag",
            "0.5",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout)
    assert metrics["passed"] is True
    assert metrics["n_observations"] == 1
    assert metrics["feature_dim"] == 126
    assert metrics["feature_context_dim"] == 127
    assert metrics["street_counts"] == {"2": 1}
    assert metrics["source_flag"] == 0.5
    assert metrics["action_mapping"]["n_bet_actions"] == 2
    assert metrics["action_mapping"]["ambiguous_bet_actions"] == 1
    assert metrics["action_mapping"]["hard_mappable_bet_rate"] == 0.5
    assert json.loads(output_json.read_text(encoding="utf-8")) == metrics
