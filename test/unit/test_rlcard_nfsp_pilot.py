from scripts.run_rlcard_nfsp_pilot import run_pilot


def test_run_pilot_returns_framework_control_metrics():
    metrics = run_pilot(
        train_episodes=2,
        eval_games=2,
        seed=11,
        hidden_dim=16,
        min_buffer_size_to_learn=100,
    )

    assert metrics["environment"] == "rlcard:no-limit-holdem"
    assert metrics["num_actions"] == 5
    assert metrics["role"] == "framework_control_pilot"
    assert metrics["algorithm"] == "nfsp"
    assert len(metrics["pre_payoffs"]) == 2
    assert len(metrics["post_payoffs"]) == 2
    assert metrics["train_episodes"] == 2
    assert metrics["agent_total_t"] >= 0
