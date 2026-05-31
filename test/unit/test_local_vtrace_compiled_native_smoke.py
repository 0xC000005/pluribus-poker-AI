def test_local_vtrace_compiled_native_smoke_uses_compiled_rollout(tmp_path):
    from scripts.run_local_vtrace_compiled_native_smoke import run_smoke

    output = tmp_path / "compiled_vtrace.json"
    metrics = run_smoke(
        n_games=4,
        collector_batch_size=4,
        max_steps_per_game=16,
        updates=1,
        hidden_dim=16,
        seed=27,
        device="cpu",
        output_json=output,
    )

    assert output.exists()
    assert metrics["algorithm"] == "local_vtrace_compiled_native_smoke"
    assert metrics["environment"] == "poker_ai:full_deck_hu_nlhe"
    assert metrics["collector_backend"] == "compiled-fast-state"
    assert metrics["trajectory_packing"] == "padded_vectorized"
    assert metrics["uses_slumbot_data"] is False
    assert metrics["uses_alphanlholdem_training_data"] is False
    assert metrics["native_action_projection"] is False
    assert metrics["compiled_needs_python_showdown"] == 0
    assert metrics["n_trajectories"] > 0
    assert metrics["n_samples"] > 0
    assert metrics["illegal_action_probability"] == 0.0
    assert metrics["loss_is_finite"] is True
    assert metrics["passed"] is True
