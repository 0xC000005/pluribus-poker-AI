import numpy as np
import subprocess
import sys

from poker_ai.research.native_rollout_substrate import (
    _build_policy_inference_net,
    _collect_batched_fast_policy_gradient_rollout,
    benchmark_batched_fast_self_play_collection,
    collect_compiled_fast_self_play,
    collect_batched_fast_self_play,
    run_batched_fast_policy_gradient_pilot,
    run_batched_fast_policy_fit_smoke,
    run_fast_state_policy_inference_throughput,
    run_fast_state_parity_check,
    run_fast_state_policy_replay_parity,
    run_rollout_throughput,
    summarize_native_rollout_gate,
)


def test_summarize_native_rollout_gate_requires_parity_and_speedup():
    failed_parity = summarize_native_rollout_gate(
        parity={"passed": False, "checked_steps": 3, "mismatches": ["mask"]},
        baseline_steps_per_second=100.0,
        candidate_steps_per_second=1000.0,
        min_speedup=5.0,
    )
    assert failed_parity["passed"] is False
    assert failed_parity["parity_passed"] is False

    failed_speed = summarize_native_rollout_gate(
        parity={"passed": True, "checked_steps": 3, "mismatches": []},
        baseline_steps_per_second=100.0,
        candidate_steps_per_second=250.0,
        min_speedup=5.0,
    )
    assert failed_speed["passed"] is False
    assert failed_speed["speedup"] == 2.5

    failed_policy = summarize_native_rollout_gate(
        parity={"passed": True, "checked_steps": 3, "mismatches": []},
        policy_parity={"passed": False, "checked_decisions": 3, "mismatches": ["action"]},
        baseline_steps_per_second=100.0,
        candidate_steps_per_second=600.0,
        min_speedup=5.0,
    )
    assert failed_policy["passed"] is False
    assert failed_policy["policy_parity_passed"] is False

    passed = summarize_native_rollout_gate(
        parity={"passed": True, "checked_steps": 3, "mismatches": []},
        policy_parity={"passed": True, "checked_decisions": 3, "mismatches": []},
        baseline_steps_per_second=100.0,
        candidate_steps_per_second=600.0,
        min_speedup=5.0,
    )
    assert passed["passed"] is True
    assert passed["speedup_passed"] is True


def test_fast_state_replay_parity_matches_canonical_full_deck():
    metrics = run_fast_state_parity_check(
        n_games=3,
        max_steps_per_game=24,
        initial_chips=1000,
        seed=20260750,
    )

    assert metrics["passed"] is True
    assert metrics["checked_steps"] > 0
    assert metrics["mismatches"] == []
    assert metrics["feature_max_abs_diff"] <= 0.0


def test_fast_state_policy_replay_parity_matches_canonical_decisions():
    metrics = run_fast_state_policy_replay_parity(
        n_games=4,
        max_steps_per_game=32,
        initial_chips=1000,
        seed=20260788,
    )

    assert metrics["passed"] is True
    assert metrics["checked_decisions"] > 0
    assert metrics["action_mismatches"] == 0
    assert metrics["mismatches"] == []
    assert metrics["feature_max_abs_diff"] <= 0.0


def test_fast_state_policy_replay_parity_matches_random_future_deals():
    def shove_then_call(_features, legal_mask, *, step_i, **_kwargs):
        if step_i == 0 and legal_mask[8] > 0:
            return 8
        if legal_mask[1] > 0:
            return 1
        legal = np.flatnonzero(legal_mask > 0)
        return int(legal[0])

    metrics = run_fast_state_policy_replay_parity(
        n_games=4,
        max_steps_per_game=8,
        initial_chips=1000,
        seed=20260792,
        deterministic_deals=False,
        action_selector=shove_then_call,
    )

    assert metrics["passed"] is True
    assert metrics["checked_decisions"] > 0
    assert metrics["mismatches"] == []


def test_rollout_throughput_reports_same_contract_rates():
    baseline = run_rollout_throughput(
        backend="python-full-deck",
        n_games=2,
        max_steps_per_game=8,
        initial_chips=1000,
        seed=20260751,
    )
    candidate = run_rollout_throughput(
        backend="fast-state",
        n_games=2,
        max_steps_per_game=8,
        initial_chips=1000,
        seed=20260751,
    )

    assert baseline["backend"] == "python-full-deck"
    assert candidate["backend"] == "fast-state"
    assert baseline["steps"] > 0
    assert candidate["steps"] > 0
    assert np.isfinite(baseline["steps_per_second"])
    assert np.isfinite(candidate["steps_per_second"])


def test_batched_fast_state_policy_inference_reduces_forward_calls():
    sequential = run_fast_state_policy_inference_throughput(
        mode="sequential",
        n_games=8,
        batch_size=4,
        max_steps_per_game=8,
        hidden_dim=16,
        device="cpu",
        seed=20260805,
    )
    batched = run_fast_state_policy_inference_throughput(
        mode="batched",
        n_games=8,
        batch_size=4,
        max_steps_per_game=8,
        hidden_dim=16,
        device="cpu",
        seed=20260805,
    )

    assert sequential["steps"] > 0
    assert batched["steps"] > 0
    assert sequential["policy_forward_calls"] == sequential["steps"]
    assert batched["policy_forward_calls"] < batched["steps"]
    assert batched["mean_decisions_per_forward"] > 1.0
    assert sequential["resolved_device"] == "cpu"
    assert batched["resolved_device"] == "cpu"


def test_batched_fast_self_play_collector_returns_legal_training_arrays():
    result = collect_batched_fast_self_play(
        mode="batched",
        n_games=12,
        batch_size=6,
        max_steps_per_game=12,
        hidden_dim=16,
        device="cpu",
        seed=20260807,
    )
    batch = result["batch"]
    metrics = result["metrics"]

    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["backend"] == "fast-state"
    assert metrics["steps"] > 0
    assert batch["features"].shape == (metrics["steps"], 126)
    assert batch["legal_masks"].shape == (metrics["steps"], 9)
    assert batch["actions"].shape == (metrics["steps"],)
    assert batch["rewards"].shape == (metrics["steps"],)
    assert batch["payoffs"].shape == (12, 2)
    assert np.all(batch["legal_masks"][np.arange(metrics["steps"]), batch["actions"]] > 0)
    assert set(np.unique(batch["players"]).tolist()).issubset({0, 1})
    assert metrics["policy_forward_calls"] < metrics["steps"]
    assert metrics["mean_decisions_per_forward"] > 1.0


def test_compiled_fast_self_play_collector_returns_legal_training_arrays():
    result = collect_compiled_fast_self_play(
        n_games=12,
        batch_size=6,
        max_steps_per_game=12,
        hidden_dim=16,
        device="cpu",
        seed=20260835,
    )
    batch = result["batch"]
    metrics = result["metrics"]

    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["backend"] == "compiled-fast-state"
    assert metrics["steps"] > 0
    assert batch["features"].shape == (metrics["steps"], 126)
    assert batch["legal_masks"].shape == (metrics["steps"], 9)
    assert batch["actions"].shape == (metrics["steps"],)
    assert batch["rewards"].shape == (metrics["steps"],)
    assert batch["payoffs"].shape == (12, 2)
    assert np.all(batch["legal_masks"][np.arange(metrics["steps"]), batch["actions"]] > 0)
    assert set(np.unique(batch["players"]).tolist()).issubset({0, 1})
    assert metrics["policy_forward_calls"] < metrics["steps"]
    assert metrics["mean_decisions_per_forward"] > 1.0
    assert metrics["needs_python_showdown"] == 0


def test_batched_fast_self_play_benchmark_matches_sequential_decisions():
    metrics = benchmark_batched_fast_self_play_collection(
        n_games=12,
        batch_size=6,
        max_steps_per_game=12,
        hidden_dim=16,
        device="cpu",
        seed=20260808,
    )

    assert metrics["sequential"]["steps"] == metrics["batched"]["steps"]
    assert metrics["sequential"]["action_checksum"] == metrics["batched"]["action_checksum"]
    assert metrics["sequential"]["payoff_checksum"] == metrics["batched"]["payoff_checksum"]
    assert metrics["batched"]["policy_forward_calls"] < metrics["sequential"]["policy_forward_calls"]
    assert metrics["batched_forward_call_reduction"] > 1.0


def test_batched_fast_policy_fit_smoke_consumes_collector_batch():
    metrics = run_batched_fast_policy_fit_smoke(
        n_games=24,
        collector_batch_size=8,
        train_batch_size=32,
        max_steps_per_game=12,
        hidden_dim=32,
        train_steps=30,
        lr=0.01,
        device="cpu",
        seed=20260810,
    )

    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["collector_steps"] > 0
    assert metrics["train_steps"] == 30
    assert metrics["initial_loss"] > metrics["final_loss"]
    assert metrics["samples_per_second"] > 0.0


def test_batched_fast_policy_gradient_pilot_writes_native_ppo_compatible_checkpoint(tmp_path):
    checkpoint = tmp_path / "batched_pg.pt"

    metrics = run_batched_fast_policy_gradient_pilot(
        train_iterations=2,
        games_per_iteration=16,
        collector_batch_size=8,
        train_batch_size=32,
        train_epochs=1,
        max_steps_per_game=12,
        hidden_dim=32,
        lr=0.001,
        entropy_weight=0.01,
        device="cpu",
        seed=20260811,
        checkpoint_path=str(checkpoint),
    )

    assert metrics["algorithm"] == "batched_fast_policy_gradient"
    assert metrics["uses_slumbot_training_data"] is False
    assert metrics["collector_backend"] == "fast-state"
    assert metrics["total_collector_steps"] > 0
    assert metrics["checkpoint_path"] == str(checkpoint)
    assert checkpoint.exists()

    import torch
    from poker_ai.research.mixed_policy_h2h import load_policy_adapter

    adapter = load_policy_adapter(str(checkpoint), kind="native-ppo", device=torch.device("cpu"))
    legal_mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    probs = adapter.probs(np.zeros(126, dtype=np.float32), legal_mask, torch.device("cpu"))
    assert probs.shape == (9,)
    assert np.isclose(float(probs.sum()), 1.0)
    assert np.all(probs[legal_mask <= 0] == 0.0)


def test_batched_fast_policy_gradient_pilot_supports_history_opponents():
    metrics = run_batched_fast_policy_gradient_pilot(
        train_iterations=3,
        games_per_iteration=16,
        collector_batch_size=8,
        train_batch_size=32,
        train_epochs=1,
        max_steps_per_game=12,
        hidden_dim=32,
        lr=0.001,
        entropy_weight=0.01,
        device="cpu",
        seed=20260813,
        history_opponent_interval=1,
        history_opponent_capacity=2,
    )

    assert metrics["train_opponent_mode"] == "history_population"
    assert metrics["history_opponent_interval"] == 1
    assert metrics["history_opponent_capacity"] == 2
    assert metrics["history_snapshots_added"] == 3
    assert 1 <= metrics["history_population_size"] <= 2
    assert metrics["opponent_controlled_steps"] > 0
    assert metrics["learner_controlled_steps"] > 0
    assert metrics["total_collector_steps"] == metrics["learner_controlled_steps"]
    assert metrics["total_env_steps"] >= metrics["total_collector_steps"]


def test_compiled_policy_gradient_pilot_uses_compiled_rollout_backend(tmp_path):
    checkpoint = tmp_path / "compiled_pg.pt"

    metrics = run_batched_fast_policy_gradient_pilot(
        train_iterations=2,
        games_per_iteration=16,
        collector_batch_size=8,
        train_batch_size=32,
        train_epochs=1,
        max_steps_per_game=12,
        hidden_dim=32,
        lr=0.001,
        entropy_weight=0.01,
        device="cpu",
        seed=20260836,
        checkpoint_path=str(checkpoint),
        history_opponent_interval=1,
        history_opponent_capacity=2,
        rollout_backend="compiled",
    )

    assert metrics["rollout_backend"] == "compiled"
    assert metrics["collector_backend"] == "compiled-fast-state"
    assert metrics["compiled_needs_python_showdown"] == 0
    assert metrics["train_opponent_mode"] == "history_population"
    assert metrics["history_population_size"] > 0
    assert metrics["total_collector_steps"] > 0
    assert checkpoint.exists()


def test_batched_fast_policy_gradient_pilot_supports_value_baseline(tmp_path):
    checkpoint = tmp_path / "batched_pg_value.pt"

    metrics = run_batched_fast_policy_gradient_pilot(
        train_iterations=2,
        games_per_iteration=16,
        collector_batch_size=8,
        train_batch_size=32,
        train_epochs=1,
        max_steps_per_game=12,
        hidden_dim=32,
        lr=0.001,
        entropy_weight=0.01,
        value_loss_weight=0.5,
        device="cpu",
        seed=20260815,
        checkpoint_path=str(checkpoint),
    )

    assert metrics["uses_value_baseline"] is True
    assert metrics["value_loss_weight"] == 0.5
    assert metrics["value_updates"] > 0
    assert metrics["first_value_loss"] is not None
    assert metrics["last_value_loss"] is not None
    assert checkpoint.exists()

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "value_net_state_dict" in payload


def test_batched_fast_policy_gradient_pilot_supports_q_boosting_update(tmp_path):
    checkpoint = tmp_path / "batched_pg_q_boost.pt"

    metrics = run_batched_fast_policy_gradient_pilot(
        train_iterations=2,
        games_per_iteration=16,
        collector_batch_size=8,
        train_batch_size=32,
        train_epochs=1,
        max_steps_per_game=12,
        hidden_dim=32,
        lr=0.001,
        entropy_weight=0.01,
        q_boost_lambda=0.8,
        q_loss_weight=1.0,
        ppo_clip_epsilon=0.2,
        device="cpu",
        seed=20260819,
        checkpoint_path=str(checkpoint),
    )

    assert metrics["uses_q_boosting"] is True
    assert metrics["q_boost_lambda"] == 0.8
    assert metrics["ppo_clip_epsilon"] == 0.2
    assert metrics["q_updates"] > 0
    assert metrics["first_q_loss"] is not None
    assert metrics["last_q_loss"] is not None
    assert checkpoint.exists()

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "q_net_state_dict" in payload


def test_batched_fast_policy_gradient_rollout_exposes_trajectory_metadata():
    import torch

    device = torch.device("cpu")
    policy = _build_policy_inference_net(hidden_dim=16, device=device)
    batch, metrics = _collect_batched_fast_policy_gradient_rollout(
        policy,
        n_games=8,
        batch_size=4,
        max_steps_per_game=12,
        initial_chips=1000,
        device=device,
        seed=20260817,
    )

    n = int(batch["actions"].shape[0])
    assert n > 0
    assert batch["old_log_probs"].shape == (n,)
    assert batch["critic_features"].shape[0] == n
    assert batch["critic_features"].shape[1] > batch["features"].shape[1]
    assert np.all(np.isfinite(batch["old_log_probs"]))
    assert batch["next_decision_indices"].shape == (n,)
    assert batch["dones"].shape == (n,)
    assert set(np.unique(batch["dones"]).tolist()).issubset({0.0, 1.0})
    assert metrics["trajectory_links"] == int(np.sum(batch["next_decision_indices"] >= 0))

    for idx, next_idx in enumerate(batch["next_decision_indices"]):
        if int(next_idx) < 0:
            continue
        assert int(next_idx) < n
        assert batch["game_indices"][idx] == batch["game_indices"][next_idx]
        assert batch["players"][idx] == batch["players"][next_idx]
        assert batch["step_indices"][idx] < batch["step_indices"][next_idx]


def test_batched_fast_policy_gradient_pilot_supports_centralized_q_critic(tmp_path):
    checkpoint = tmp_path / "batched_pg_cq.pt"

    metrics = run_batched_fast_policy_gradient_pilot(
        train_iterations=2,
        games_per_iteration=16,
        collector_batch_size=8,
        train_batch_size=32,
        train_epochs=1,
        max_steps_per_game=12,
        hidden_dim=32,
        lr=0.001,
        entropy_weight=0.01,
        q_boost_lambda=0.8,
        q_loss_weight=1.0,
        ppo_clip_epsilon=0.2,
        centralized_q_critic=True,
        device="cpu",
        seed=20260821,
        checkpoint_path=str(checkpoint),
    )

    assert metrics["uses_q_boosting"] is True
    assert metrics["uses_centralized_q_critic"] is True
    assert metrics["q_input_dim"] > 126
    assert metrics["q_updates"] > 0
    assert checkpoint.exists()

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["q_input_dim"] == metrics["q_input_dim"]
    assert payload["config"]["centralized_q_critic"] is True


def test_eval_native_rollout_substrate_cli_writes_json(tmp_path):
    output = tmp_path / "native_rollout_gate.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_native_rollout_substrate.py",
            "--n-parity-games",
            "2",
            "--parity-max-steps",
            "12",
            "--n-benchmark-games",
            "2",
            "--benchmark-max-steps",
            "8",
            "--min-speedup",
            "0.1",
            "--output-json",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert '"policy_parity"' in output.read_text(encoding="utf-8")


def test_eval_native_rollout_substrate_cli_can_include_batched_collector(tmp_path):
    output = tmp_path / "native_rollout_with_collector.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_native_rollout_substrate.py",
            "--n-parity-games",
            "2",
            "--parity-max-steps",
            "12",
            "--n-benchmark-games",
            "2",
            "--benchmark-max-steps",
            "8",
            "--min-speedup",
            "0.1",
            "--include-batched-collector-benchmark",
            "--collector-games",
            "8",
            "--collector-batch-size",
            "4",
            "--collector-hidden-dim",
            "16",
            "--collector-device",
            "cpu",
            "--output-json",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert '"batched_self_play_collection_benchmark"' in text
    assert '"same_decision_trace": true' in text


def test_eval_native_rollout_substrate_cli_can_include_compiled_collector(tmp_path):
    output = tmp_path / "native_rollout_with_compiled_collector.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_native_rollout_substrate.py",
            "--n-parity-games",
            "2",
            "--parity-max-steps",
            "8",
            "--n-benchmark-games",
            "2",
            "--benchmark-max-steps",
            "8",
            "--min-speedup",
            "0.1",
            "--include-compiled-collector-benchmark",
            "--collector-games",
            "12",
            "--collector-batch-size",
            "6",
            "--collector-hidden-dim",
            "16",
            "--collector-device",
            "cpu",
            "--output-json",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert '"compiled_self_play_collection"' in text
    assert '"backend": "compiled-fast-state"' in text
    assert '"needs_python_showdown": 0' in text


def test_eval_native_rollout_substrate_cli_can_include_policy_fit_smoke(tmp_path):
    output = tmp_path / "native_rollout_with_policy_fit.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_native_rollout_substrate.py",
            "--n-parity-games",
            "2",
            "--parity-max-steps",
            "12",
            "--n-benchmark-games",
            "2",
            "--benchmark-max-steps",
            "8",
            "--min-speedup",
            "0.1",
            "--include-batched-policy-fit-smoke",
            "--policy-fit-games",
            "12",
            "--policy-fit-collector-batch-size",
            "4",
            "--policy-fit-train-batch-size",
            "16",
            "--policy-fit-hidden-dim",
            "16",
            "--policy-fit-train-steps",
            "10",
            "--policy-fit-device",
            "cpu",
            "--output-json",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert '"batched_policy_fit_smoke"' in text
    assert '"loss_decreased": true' in text


def test_eval_native_rollout_substrate_cli_can_run_batched_policy_gradient(tmp_path):
    output = tmp_path / "native_rollout_with_pg.json"
    checkpoint = tmp_path / "batched_pg.pt"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_native_rollout_substrate.py",
            "--n-parity-games",
            "2",
            "--parity-max-steps",
            "12",
            "--n-benchmark-games",
            "2",
            "--benchmark-max-steps",
            "8",
            "--min-speedup",
            "0.1",
            "--include-batched-policy-gradient-pilot",
            "--pg-train-iterations",
            "1",
            "--pg-games-per-iteration",
            "8",
            "--pg-collector-batch-size",
            "4",
            "--pg-train-batch-size",
            "16",
            "--pg-train-epochs",
            "1",
            "--pg-hidden-dim",
            "16",
            "--pg-device",
            "cpu",
            "--pg-checkpoint-out",
            str(checkpoint),
            "--output-json",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert '"batched_policy_gradient_pilot"' in text
    assert checkpoint.exists()


def test_eval_native_rollout_substrate_cli_can_run_batched_policy_gradient_with_history_population(
    tmp_path,
):
    output = tmp_path / "native_rollout_with_pg_history.json"
    checkpoint = tmp_path / "batched_pg_history.pt"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_native_rollout_substrate.py",
            "--n-parity-games",
            "2",
            "--parity-max-steps",
            "12",
            "--n-benchmark-games",
            "2",
            "--benchmark-max-steps",
            "8",
            "--min-speedup",
            "0.1",
            "--include-batched-policy-gradient-pilot",
            "--pg-train-iterations",
            "2",
            "--pg-games-per-iteration",
            "8",
            "--pg-collector-batch-size",
            "4",
            "--pg-train-batch-size",
            "16",
            "--pg-train-epochs",
            "1",
            "--pg-hidden-dim",
            "16",
            "--pg-device",
            "cpu",
            "--pg-history-opponent-interval",
            "1",
            "--pg-history-opponent-capacity",
            "2",
            "--pg-checkpoint-out",
            str(checkpoint),
            "--output-json",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert '"train_opponent_mode": "history_population"' in text
    assert '"history_population_size"' in text
    assert checkpoint.exists()


def test_eval_native_rollout_substrate_cli_can_run_compiled_policy_gradient(
    tmp_path,
):
    output = tmp_path / "native_rollout_with_compiled_pg.json"
    checkpoint = tmp_path / "compiled_pg_history.pt"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_native_rollout_substrate.py",
            "--n-parity-games",
            "2",
            "--parity-max-steps",
            "12",
            "--n-benchmark-games",
            "2",
            "--benchmark-max-steps",
            "8",
            "--min-speedup",
            "0.1",
            "--include-batched-policy-gradient-pilot",
            "--pg-rollout-backend",
            "compiled",
            "--pg-train-iterations",
            "2",
            "--pg-games-per-iteration",
            "8",
            "--pg-collector-batch-size",
            "4",
            "--pg-train-batch-size",
            "16",
            "--pg-train-epochs",
            "1",
            "--pg-hidden-dim",
            "16",
            "--pg-device",
            "cpu",
            "--pg-history-opponent-interval",
            "1",
            "--pg-history-opponent-capacity",
            "2",
            "--pg-checkpoint-out",
            str(checkpoint),
            "--output-json",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert '"rollout_backend": "compiled"' in text
    assert '"collector_backend": "compiled-fast-state"' in text
    assert '"compiled_needs_python_showdown": 0' in text
    assert checkpoint.exists()


def test_eval_native_rollout_substrate_cli_can_run_batched_policy_gradient_with_value_baseline(
    tmp_path,
):
    output = tmp_path / "native_rollout_with_pg_value.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_native_rollout_substrate.py",
            "--n-parity-games",
            "2",
            "--parity-max-steps",
            "12",
            "--n-benchmark-games",
            "2",
            "--benchmark-max-steps",
            "8",
            "--min-speedup",
            "0.1",
            "--include-batched-policy-gradient-pilot",
            "--pg-train-iterations",
            "1",
            "--pg-games-per-iteration",
            "8",
            "--pg-collector-batch-size",
            "4",
            "--pg-train-batch-size",
            "16",
            "--pg-train-epochs",
            "1",
            "--pg-hidden-dim",
            "16",
            "--pg-value-loss-weight",
            "0.5",
            "--pg-device",
            "cpu",
            "--output-json",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert '"uses_value_baseline": true' in text
    assert '"value_loss_weight": 0.5' in text


def test_eval_native_rollout_substrate_cli_can_run_batched_policy_gradient_with_q_boosting(
    tmp_path,
):
    output = tmp_path / "native_rollout_with_pg_q_boost.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_native_rollout_substrate.py",
            "--n-parity-games",
            "2",
            "--parity-max-steps",
            "12",
            "--n-benchmark-games",
            "2",
            "--benchmark-max-steps",
            "8",
            "--min-speedup",
            "0.1",
            "--include-batched-policy-gradient-pilot",
            "--pg-train-iterations",
            "1",
            "--pg-games-per-iteration",
            "8",
            "--pg-collector-batch-size",
            "4",
            "--pg-train-batch-size",
            "16",
            "--pg-train-epochs",
            "1",
            "--pg-hidden-dim",
            "16",
            "--pg-q-boost-lambda",
            "0.8",
            "--pg-q-loss-weight",
            "1.0",
            "--pg-ppo-clip-epsilon",
            "0.2",
            "--pg-device",
            "cpu",
            "--output-json",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert '"uses_q_boosting": true' in text
    assert '"q_boost_lambda": 0.8' in text


def test_eval_native_rollout_substrate_cli_can_run_batched_policy_gradient_with_centralized_q(
    tmp_path,
):
    output = tmp_path / "native_rollout_with_pg_cq.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_native_rollout_substrate.py",
            "--n-parity-games",
            "2",
            "--parity-max-steps",
            "12",
            "--n-benchmark-games",
            "2",
            "--benchmark-max-steps",
            "8",
            "--min-speedup",
            "0.1",
            "--include-batched-policy-gradient-pilot",
            "--pg-train-iterations",
            "1",
            "--pg-games-per-iteration",
            "8",
            "--pg-collector-batch-size",
            "4",
            "--pg-train-batch-size",
            "16",
            "--pg-train-epochs",
            "1",
            "--pg-hidden-dim",
            "16",
            "--pg-q-boost-lambda",
            "0.8",
            "--pg-ppo-clip-epsilon",
            "0.2",
            "--pg-centralized-q-critic",
            "--pg-device",
            "cpu",
            "--output-json",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert '"uses_centralized_q_critic": true' in text
