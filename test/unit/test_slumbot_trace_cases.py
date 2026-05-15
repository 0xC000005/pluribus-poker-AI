import json
import subprocess
import sys
from pathlib import Path

from poker_ai.research.slumbot_trace_cases import extract_resolver_cases_from_trace


def test_extract_resolver_cases_from_trace_keeps_valid_turn_and_river_decisions(tmp_path):
    trace = tmp_path / "trace.jsonl"
    records = [
        {
            "event": "decision",
            "source": "policy",
            "hand_index": 1,
            "street_index": 0,
            "hole_cards": ["Ac", "Kd"],
            "board": [],
            "action_str": "",
            "client_pos": 0,
        },
        {
            "event": "decision",
            "source": "policy",
            "hand_index": 2,
            "street_index": 2,
            "hole_cards": ["Ac", "Kd"],
            "board": ["2c", "7d", "Jh", "4s"],
            "action_str": "ck/kk/",
            "client_pos": 0,
        },
        {
            "event": "decision",
            "source": "policy",
            "hand_index": 3,
            "street_index": 3,
            "hole_cards": ["Qs", "Qd"],
            "board": ["2h", "8c", "Td", "3s", "9c"],
            "action_str": "ck/kk/kk/",
            "client_pos": 0,
        },
        {
            "event": "decision",
            "source": "policy",
            "hand_index": 4,
            "street_index": 3,
            "hole_cards": ["Qs", "Qd"],
            "board": ["2h", "8c", "Td"],
            "action_str": "ck/kk/kk/",
            "client_pos": 0,
        },
        {
            "event": "hand_result",
            "hand_index": 5,
            "hole_cards": ["As", "Ah"],
            "board": ["2h", "8c", "Td", "3s", "9c"],
        },
    ]
    trace.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    cases = extract_resolver_cases_from_trace(trace)

    assert [case.label for case in cases] == [
        "trace-hand2-decision1-street2",
        "trace-hand3-decision2-street3",
    ]
    assert cases[0].hole_cards == ("Ac", "Kd")
    assert cases[0].board == ("2c", "7d", "Jh", "4s")
    assert cases[0].action_str == "ck/kk/"
    assert cases[0].client_pos == 0
    assert cases[0].source == "slumbot_trace"


def test_extract_resolver_cases_from_trace_respects_limit(tmp_path):
    trace = tmp_path / "trace.jsonl"
    record = {
        "event": "decision",
        "source": "policy",
        "street_index": 2,
        "hole_cards": ["Ac", "Kd"],
        "board": ["2c", "7d", "Jh", "4s"],
        "action_str": "ck/kk/",
        "client_pos": 0,
    }
    trace.write_text(
        "\n".join(json.dumps({**record, "hand_index": idx}) for idx in range(3)),
        encoding="utf-8",
    )

    cases = extract_resolver_cases_from_trace(trace, limit=2)

    assert len(cases) == 2


def test_build_slumbot_trace_resolver_cases_cli_writes_cases_json(tmp_path):
    trace = tmp_path / "trace.jsonl"
    output = tmp_path / "cases.json"
    trace.write_text(
        json.dumps(
            {
                "event": "decision",
                "source": "policy",
                "hand_index": 7,
                "street_index": 2,
                "hole_cards": ["Ac", "Kd"],
                "board": ["2c", "7d", "Jh", "4s"],
                "action_str": "ck/kk/",
                "client_pos": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "build_slumbot_trace_resolver_cases.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--trace", str(trace), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["cases"][0]["label"] == "trace-hand7-decision1-street2"
