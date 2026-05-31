import json
import subprocess
import sys

from poker_ai.research.traversal_compaction import (
    analyze_traversal_compaction_potential,
    load_traversal_profiles,
)


def _profile(**overrides):
    profile = {
        "traversal_mean_allocated_to_live_ratio": 7.5,
        "traversal_max_allocated_to_live_ratio": 8.0,
        "traversal_mean_slots_per_traversal": 360.0,
        "traversal_mean_max_nonterminal_slots_per_traversal": 48.0,
        "traversal_max_nonterminal_slots_per_traversal": 80.0,
        "traversal_overflow_chunk_fraction": 0.0,
        "traversal_pool_exhausted_per_traversal": 0.0,
        "traversal_pool_exhausted_nodes": 0,
        "traverse_seconds": 0.75,
        "traversals_per_second": 5300.0,
    }
    profile.update(overrides)
    return profile


def test_compaction_gate_passes_when_clean_traversal_has_large_slot_waste():
    report = analyze_traversal_compaction_potential([_profile()])

    assert report["passed"] is True
    assert report["fidelity_clean"] is True
    assert report["compaction_promising"] is True
    assert report["estimated_slot_reduction_fraction"] > 0.8
    assert "active-frontier compaction" in report["next_action"]


def test_compaction_gate_blocks_unclean_accepted_traversal():
    report = analyze_traversal_compaction_potential(
        [
            _profile(
                traversal_overflow_chunk_fraction=0.25,
                traversal_pool_exhausted_per_traversal=12.0,
                traversal_pool_exhausted_nodes=100,
            )
        ]
    )

    assert report["passed"] is False
    assert report["fidelity_clean"] is False
    assert report["compaction_promising"] is True
    assert "not fidelity-clean" in report["failures"][0]


def test_compaction_gate_rejects_small_slot_waste():
    report = analyze_traversal_compaction_potential(
        [_profile(traversal_mean_allocated_to_live_ratio=1.1)]
    )

    assert report["passed"] is False
    assert report["fidelity_clean"] is True
    assert report["compaction_promising"] is False
    assert "too small" in report["failures"][0]


def test_load_traversal_profiles_defaults_to_measured_profiles(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps(
            {
                "warmup_profiles": [_profile(traversal_mean_allocated_to_live_ratio=3.0)],
                "profiles": [_profile(traversal_mean_allocated_to_live_ratio=6.0)],
            }
        ),
        encoding="utf-8",
    )

    profiles = load_traversal_profiles(path)
    all_profiles = load_traversal_profiles(path, include_warmup=True)

    assert len(profiles) == 1
    assert profiles[0]["traversal_mean_allocated_to_live_ratio"] == 6.0
    assert len(all_profiles) == 2


def test_compaction_cli_writes_report(tmp_path):
    input_path = tmp_path / "benchmark.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(json.dumps({"profiles": [_profile()]}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_traversal_compaction_potential.py",
            "--input-json",
            str(input_path),
            "--output-json",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert "traversal_compaction_potential" in result.stdout
    assert report["passed"] is True
    assert report["input_json"] == str(input_path)
