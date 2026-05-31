import json
import subprocess
import sys

import pytest
import torch

from poker_ai.research.frontier_compaction_smoke import (
    compact_indices_prefix_sum,
    run_frontier_index_interop_smoke,
    run_frontier_compaction_smoke,
)


def test_compact_indices_prefix_sum_matches_torch_nonzero_order():
    mask = torch.tensor([False, True, True, False, True, False])

    indices = compact_indices_prefix_sum(mask)

    assert indices.tolist() == [1, 2, 4]
    assert torch.equal(indices, torch.nonzero(mask).flatten())


def test_frontier_compaction_smoke_passes_on_cpu_tiny_case():
    report = run_frontier_compaction_smoke(
        n_rows=256,
        feature_dim=8,
        live_density=0.25,
        work_repeats=1,
        repeats=2,
        device="cpu",
        seed=20260515,
        min_slot_reduction_fraction=0.5,
    )

    assert report["mode"] == "frontier_compaction_smoke"
    assert report["passed"] is True
    assert report["correct"] is True
    assert report["stable_order"] is True
    assert report["slot_reduction_fraction"] >= 0.5
    assert report["promotion"] is False


def test_frontier_compaction_smoke_rejects_low_slot_reduction():
    report = run_frontier_compaction_smoke(
        n_rows=128,
        feature_dim=4,
        live_density=0.95,
        work_repeats=1,
        repeats=1,
        device="cpu",
        seed=20260516,
        min_slot_reduction_fraction=0.5,
    )

    assert report["passed"] is False
    assert any("slot reduction" in failure for failure in report["failures"])


def test_frontier_compaction_cli_writes_report(tmp_path):
    output_path = tmp_path / "frontier.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_frontier_compaction_smoke.py",
            "--n-rows",
            "256",
            "--feature-dim",
            "8",
            "--live-density",
            "0.25",
            "--work-repeats",
            "1",
            "--repeats",
            "2",
            "--device",
            "cpu",
            "--output-json",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert "frontier_compaction_smoke" in result.stdout
    assert report["passed"] is True


def test_frontier_compaction_rejects_invalid_mask_shape():
    with pytest.raises(ValueError, match="one-dimensional"):
        compact_indices_prefix_sum(torch.ones((2, 2), dtype=torch.bool))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_frontier_index_interop_smoke_tiny_cuda_case():
    report = run_frontier_index_interop_smoke(
        n_rows=4096,
        live_density=0.2,
        repeats=1,
        device="cuda",
        seed=20260517,
        min_slot_reduction_fraction=0.5,
    )

    assert report["mode"] == "frontier_index_interop_smoke"
    assert report["passed"] is True
    assert report["correct_indices"] is True
    assert report["correct_gather"] is True
    assert report["promotion"] is False
