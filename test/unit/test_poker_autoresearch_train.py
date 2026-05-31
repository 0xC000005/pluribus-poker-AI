from argparse import Namespace

from scripts.poker_autoresearch_train import (
    evaluate_training_gate_failures,
    summarize_resume_checkpoint_metadata,
)


def test_training_gate_fails_on_rejected_traversal_chunks():
    args = Namespace(
        max_pool_exhausted_per_traversal=0.0,
        max_overflow_chunk_fraction=0.0,
        max_rejected_traversal_chunks=0,
        min_traversals_per_second=100.0,
    )

    failures = evaluate_training_gate_failures(
        args,
        {
            "traversal_pool_exhausted_per_traversal": 0.0,
            "traversal_overflow_chunk_fraction": 0.0,
            "traversal_rejected_chunks": 2,
        },
        traversals_per_second=250.0,
    )

    assert any("traversal_rejected_chunks" in failure for failure in failures)


def test_training_gate_allows_rejected_traversal_chunks_without_threshold():
    args = Namespace(
        max_pool_exhausted_per_traversal=0.0,
        max_overflow_chunk_fraction=0.0,
        max_rejected_traversal_chunks=None,
        min_traversals_per_second=100.0,
    )

    failures = evaluate_training_gate_failures(
        args,
        {
            "traversal_pool_exhausted_per_traversal": 0.0,
            "traversal_overflow_chunk_fraction": 0.0,
            "traversal_rejected_chunks": 2,
        },
        traversals_per_second=250.0,
    )

    assert failures == []


def test_resume_checkpoint_metadata_marks_compact_checkpoint_as_model_warm_start():
    metadata = summarize_resume_checkpoint_metadata(
        {
            "iteration": 100,
            "buffer_sizes": [4_000_000, 4_000_000],
            "has_average_policy_net": False,
        },
        resume_path="models/incumbent.pt",
    )

    assert metadata["resume_used"] is True
    assert metadata["resume_checkpoint"] == "models/incumbent.pt"
    assert metadata["resume_checkpoint_iteration"] == 100
    assert metadata["resume_checkpoint_buffer_sizes"] == [4_000_000, 4_000_000]
    assert metadata["resume_restored_replay_buffers"] is False
    assert metadata["resume_semantics"] == "model_warm_start_no_replay_buffers"
