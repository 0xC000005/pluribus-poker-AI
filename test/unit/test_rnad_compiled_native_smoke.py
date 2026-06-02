def test_rnad_compiled_native_smoke_uses_native_compiled_rollout(tmp_path):
    from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES
    from scripts.run_rnad_compiled_native_smoke import main
    from poker_ai.research.native_rnad import run_compiled_native_rnad_smoke

    output = tmp_path / "rnad_compiled_native.json"
    metrics = run_compiled_native_rnad_smoke(
        n_games=4,
        collector_batch_size=4,
        max_steps_per_game=32,
        updates=1,
        hidden_dim=16,
        seed=31,
        device="cpu",
    )

    assert metrics["algorithm"] == "rnad_compiled_native_smoke"
    assert metrics["environment"] == "poker_ai:full_deck_hu_nlhe"
    assert metrics["collector_backend"] == "compiled-fast-state"
    assert metrics["trajectory_contract"] == "poker_ai.rnad.collector.Trajectory"
    assert metrics["num_actions"] == N_ACTIONS == 9
    assert metrics["num_features"] == N_FEATURES
    assert metrics["trained_environment_native"] is True
    assert metrics["uses_slumbot_data"] is False
    assert metrics["uses_alphanlholdem_training_data"] is False
    assert metrics["native_action_projection"] is False
    assert metrics["promotion"] is False
    assert metrics["compiled_needs_python_showdown"] == 0
    assert metrics["illegal_records"] == 0
    assert metrics["n_samples"] > 0
    assert metrics["n_trajectories"] > 0
    assert metrics["rnad_loss_is_finite"] is True
    assert metrics["passed"] is True

    exit_code = main(
        [
            "--n-games",
            "4",
            "--collector-batch-size",
            "4",
            "--max-steps-per-game",
            "32",
            "--updates",
            "1",
            "--hidden-dim",
            "16",
            "--seed",
            "31",
            "--device",
            "cpu",
            "--output-json",
            str(output),
        ]
    )
    assert exit_code == 0
    assert output.exists()
