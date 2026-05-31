import json
from pathlib import Path

import numpy as np
import pytest
import torch

from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES


pytest.importorskip("gymnasium")

from scripts.run_tianshou_rainbow_native_control import (  # noqa: E402
    NativeRainbowPokerEnv,
    evaluate_rainbow_checkpoint_vs_neural_policy_iteration,
    evaluate_rainbow_checkpoint_vs_rainbow,
    evaluate_rainbow_checkpoint_vs_native_nfsp,
    _rainbow_state_dicts_by_seat_from_payload,
    run_control,
)


def test_native_rainbow_env_exposes_observation_and_legal_mask():
    env = NativeRainbowPokerEnv(seed=7, initial_chips=1000, max_steps_per_hand=32)

    obs, info = env.reset(seed=7)

    assert obs["obs"].shape == (N_FEATURES,)
    assert obs["mask"].shape == (N_ACTIONS,)
    assert obs["mask"].dtype == np.bool_
    assert obs["mask"].sum() >= 2
    assert info["seat"] == 0


def test_native_rainbow_env_supports_fast_state_backend():
    env = NativeRainbowPokerEnv(
        seed=107,
        initial_chips=1000,
        max_steps_per_hand=32,
        state_backend="fast-state",
    )

    obs, info = env.reset(seed=107)
    legal = int(np.flatnonzero(obs["mask"])[0])
    next_obs, _reward, _terminated, _truncated, step_info = env.step(legal)

    assert info["state_backend"] == "fast-state"
    assert step_info["state_backend"] == "fast-state"
    assert obs["obs"].shape == (N_FEATURES,)
    assert next_obs["obs"].shape == (N_FEATURES,)
    assert next_obs["mask"].shape == (N_ACTIONS,)


def test_native_rainbow_env_rejects_illegal_action_with_terminal_penalty():
    env = NativeRainbowPokerEnv(seed=8, initial_chips=1000, max_steps_per_hand=32)
    obs, _info = env.reset(seed=8)
    illegal = int(np.flatnonzero(~obs["mask"])[0])

    _next_obs, reward, terminated, truncated, info = env.step(illegal)

    assert terminated is True
    assert truncated is False
    assert reward < 0
    assert info["illegal_action"] is True


def test_native_rainbow_env_supports_fixed_native_nfsp_opponent(tmp_path):
    from poker_ai.research.native_nfsp import NativeNFSPConfig, run_native_nfsp_pilot

    nfsp_checkpoint = tmp_path / "nfsp.pt"
    run_native_nfsp_pilot(
        NativeNFSPConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=8,
            min_buffer_size_to_learn=100,
            device="cpu",
            seed=25,
            checkpoint_path=str(nfsp_checkpoint),
        )
    )
    env = NativeRainbowPokerEnv(
        seed=26,
        initial_chips=1000,
        max_steps_per_hand=32,
        opponent_kind="native-nfsp",
        opponent_checkpoint=str(nfsp_checkpoint),
    )

    obs, info = env.reset(seed=26)
    legal = int(np.flatnonzero(obs["mask"])[0])
    next_obs, _reward, _terminated, _truncated, step_info = env.step(legal)

    assert info["opponent_kind"] == "native-nfsp"
    assert step_info["opponent_kind"] == "native-nfsp"
    assert next_obs["obs"].shape == (N_FEATURES,)
    assert next_obs["mask"].shape == (N_ACTIONS,)


def test_native_rainbow_env_allows_explicit_opponent_device(tmp_path):
    from poker_ai.research.native_nfsp import NativeNFSPConfig, run_native_nfsp_pilot

    nfsp_checkpoint = tmp_path / "nfsp.pt"
    run_native_nfsp_pilot(
        NativeNFSPConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=8,
            min_buffer_size_to_learn=100,
            device="cpu",
            seed=125,
            checkpoint_path=str(nfsp_checkpoint),
        )
    )

    env = NativeRainbowPokerEnv(
        seed=126,
        initial_chips=1000,
        max_steps_per_hand=32,
        opponent_kind="native-nfsp",
        opponent_checkpoint=str(nfsp_checkpoint),
        opponent_device="cpu",
    )

    _obs, info = env.reset(seed=126)

    assert info["opponent_device"] == "cpu"
    assert env._opponent_device.type == "cpu"


def test_native_rainbow_env_supports_fixed_rainbow_opponent(tmp_path):
    pytest.importorskip("tianshou")

    opponent_checkpoint = tmp_path / "opponent_rainbow.pt"
    run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=30,
        checkpoint_out=str(opponent_checkpoint),
    )
    env = NativeRainbowPokerEnv(
        seed=31,
        initial_chips=1000,
        max_steps_per_hand=32,
        opponent_kind="rainbow",
        opponent_checkpoint=str(opponent_checkpoint),
    )

    obs, info = env.reset(seed=31)
    legal = int(np.flatnonzero(obs["mask"])[0])
    next_obs, _reward, _terminated, _truncated, step_info = env.step(legal)

    assert info["opponent_kind"] == "rainbow"
    assert step_info["opponent_kind"] == "rainbow"
    assert next_obs["obs"].shape == (N_FEATURES,)
    assert next_obs["mask"].shape == (N_ACTIONS,)


def test_native_rainbow_env_samples_from_rainbow_checkpoint_league(tmp_path):
    pytest.importorskip("tianshou")

    first = tmp_path / "first_rainbow.pt"
    second = tmp_path / "second_rainbow.pt"
    for idx, checkpoint in enumerate((first, second)):
        run_control(
            train_steps=4,
            updates=1,
            batch_size=4,
            hidden_dim=16,
            num_atoms=11,
            warmup_steps=4,
            eval_games=1,
            device="cpu",
            seed=40 + idx,
            checkpoint_out=str(checkpoint),
        )
    env = NativeRainbowPokerEnv(
        seed=42,
        initial_chips=1000,
        max_steps_per_hand=32,
        opponent_kind="rainbow",
        opponent_checkpoint=[str(first), str(second)],
    )

    seen = set()
    for reset_seed in range(42, 52):
        _obs, info = env.reset(seed=reset_seed)
        seen.add(info["opponent_checkpoint"])

    assert seen == {str(first), str(second)}


def test_tianshou_rainbow_control_supports_subproc_rainbow_opponent(tmp_path):
    pytest.importorskip("tianshou")

    opponent_checkpoint = tmp_path / "opponent_rainbow.pt"
    output = tmp_path / "subproc_vs_rainbow.json"
    run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=43,
        checkpoint_out=str(opponent_checkpoint),
    )

    metrics = run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=44,
        num_envs=2,
        vector_env_backend="subproc",
        opponent_kind="rainbow",
        opponent_checkpoint=str(opponent_checkpoint),
        output_json=str(output),
    )

    assert metrics["vector_env_backend"] == "subproc"
    assert metrics["vector_env_context"] == "spawn"
    assert metrics["opponent_kind"] == "rainbow"
    assert metrics["train_steps"] >= 8
    assert metrics["total_seconds"] >= metrics["train_seconds"]


def test_tianshou_rainbow_native_control_smoke_writes_metrics(tmp_path):
    pytest.importorskip("tianshou")
    output = tmp_path / "rainbow.json"
    checkpoint = tmp_path / "rainbow.pt"

    metrics = run_control(
        train_steps=8,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=8,
        eval_games=2,
        device="cpu",
        seed=9,
        output_json=str(output),
        checkpoint_out=str(checkpoint),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "tianshou_rainbow_dqn"
    assert payload["environment"] == "poker_ai:full_deck_hu_nlhe_single_agent_control"
    assert payload["num_actions"] == 9
    assert payload["role"] == "rl_control_baseline"
    assert payload["promotion"] is False
    assert payload["league_eligible"] is False
    assert payload["train_steps"] >= 8
    assert payload["updates"] == 1
    assert payload["setup_seconds"] >= 0.0
    assert payload["total_seconds"] >= payload["train_seconds"]
    assert Path(metrics["checkpoint_path"]).exists()


def test_tianshou_rainbow_native_control_supports_dummy_vector_env(tmp_path):
    pytest.importorskip("tianshou")

    metrics = run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=14,
        num_envs=2,
        vector_env_backend="dummy",
        output_json=str(tmp_path / "vector.json"),
    )

    assert metrics["num_envs"] == 2
    assert metrics["vector_env_backend"] == "dummy"
    assert metrics["train_steps"] >= 8


def test_tianshou_rainbow_control_supports_multiple_learner_updates_per_collect(tmp_path):
    pytest.importorskip("tianshou")

    metrics = run_control(
        train_steps=4,
        updates=2,
        updates_per_collect=3,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=140,
        output_json=str(tmp_path / "multi_update.json"),
    )

    assert metrics["collect_iterations"] == 2
    assert metrics["updates_per_collect"] == 3
    assert metrics["learner_updates"] == 6
    assert metrics["train_steps"] >= 12


def test_tianshou_rainbow_control_records_fast_state_backend(tmp_path):
    pytest.importorskip("tianshou")

    metrics = run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        state_backend="fast-state",
        device="cpu",
        seed=20260753,
        output_json=str(tmp_path / "fast_state_backend.json"),
    )

    assert metrics["state_backend"] == "fast-state"
    assert metrics["train_steps"] >= 8


def test_tianshou_rainbow_native_control_trains_against_fixed_native_nfsp(tmp_path):
    pytest.importorskip("tianshou")
    from poker_ai.research.native_nfsp import NativeNFSPConfig, run_native_nfsp_pilot

    nfsp_checkpoint = tmp_path / "nfsp.pt"
    output = tmp_path / "rainbow_vs_fixed.json"
    run_native_nfsp_pilot(
        NativeNFSPConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=8,
            min_buffer_size_to_learn=100,
            device="cpu",
            seed=27,
            checkpoint_path=str(nfsp_checkpoint),
        )
    )

    metrics = run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=28,
        opponent_kind="native-nfsp",
        opponent_checkpoint=str(nfsp_checkpoint),
        output_json=str(output),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["opponent_kind"] == "native-nfsp"
    assert payload["opponent_checkpoint"] == str(nfsp_checkpoint)
    assert payload["opponent_device"] == "cpu"
    assert metrics["train_steps"] >= 4


def test_tianshou_rainbow_native_control_reports_auto_opponent_device(tmp_path):
    pytest.importorskip("tianshou")
    from poker_ai.research.native_nfsp import NativeNFSPConfig, run_native_nfsp_pilot

    nfsp_checkpoint = tmp_path / "nfsp.pt"
    run_native_nfsp_pilot(
        NativeNFSPConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=8,
            min_buffer_size_to_learn=100,
            device="cpu",
            seed=127,
            checkpoint_path=str(nfsp_checkpoint),
        )
    )

    metrics = run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=128,
        opponent_kind="native-nfsp",
        opponent_checkpoint=str(nfsp_checkpoint),
        opponent_device="auto",
        output_json=str(tmp_path / "rainbow_vs_fixed_auto_device.json"),
    )

    assert metrics["opponent_device_requested"] == "auto"
    assert metrics["opponent_device"] == "cpu"


def test_tianshou_rainbow_native_control_trains_against_fixed_rainbow(tmp_path):
    pytest.importorskip("tianshou")

    opponent_checkpoint = tmp_path / "opponent_rainbow.pt"
    output = tmp_path / "rainbow_vs_rainbow.json"
    run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=32,
        checkpoint_out=str(opponent_checkpoint),
    )

    metrics = run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=33,
        opponent_kind="rainbow",
        opponent_checkpoint=str(opponent_checkpoint),
        output_json=str(output),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["opponent_kind"] == "rainbow"
    assert payload["opponent_checkpoint"] == str(opponent_checkpoint)
    assert metrics["train_steps"] >= 4


def test_tianshou_rainbow_native_control_trains_against_rainbow_league(tmp_path):
    pytest.importorskip("tianshou")

    first = tmp_path / "first_rainbow.pt"
    second = tmp_path / "second_rainbow.pt"
    output = tmp_path / "rainbow_vs_league.json"
    for idx, checkpoint in enumerate((first, second)):
        run_control(
            train_steps=4,
            updates=1,
            batch_size=4,
            hidden_dim=16,
            num_atoms=11,
            warmup_steps=4,
            eval_games=1,
            device="cpu",
            seed=44 + idx,
            checkpoint_out=str(checkpoint),
        )

    metrics = run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=46,
        opponent_kind="rainbow",
        opponent_checkpoint=[str(first), str(second)],
        output_json=str(output),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["opponent_kind"] == "rainbow"
    assert payload["opponent_checkpoint"] is None
    assert payload["opponent_checkpoints"] == [str(first), str(second)]
    assert metrics["train_steps"] >= 4


def test_tianshou_rainbow_training_can_initialize_from_checkpoint(tmp_path):
    pytest.importorskip("tianshou")

    parent = tmp_path / "parent_rainbow.pt"
    child = tmp_path / "child_rainbow.pt"
    output = tmp_path / "child.json"
    run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=20260775,
        checkpoint_out=str(parent),
    )

    metrics = run_control(
        train_steps=0,
        updates=0,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=0,
        eval_games=1,
        device="cpu",
        seed=20260776,
        checkpoint_in=str(parent),
        checkpoint_out=str(child),
        output_json=str(output),
    )

    parent_payload = torch.load(parent, map_location="cpu", weights_only=False)
    child_payload = torch.load(child, map_location="cpu", weights_only=False)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["checkpoint_in"] == str(parent)
    assert metrics["checkpoint_in"] == str(parent)
    assert child_payload["parent_checkpoint"] == str(parent)
    for key, tensor in parent_payload["model_state_dict"].items():
        assert torch.equal(tensor, child_payload["model_state_dict"][key])


def test_tianshou_rainbow_validates_fixed_opponent_checkpoint_before_subproc_spawn(tmp_path):
    pytest.importorskip("tianshou")

    with pytest.raises(ValueError, match="opponent_checkpoint does not exist"):
        run_control(
            train_steps=4,
            updates=1,
            batch_size=4,
            hidden_dim=16,
            num_atoms=11,
            warmup_steps=4,
            eval_games=1,
            device="cpu",
            seed=29,
            num_envs=2,
            vector_env_backend="subproc",
            opponent_kind="native-nfsp",
            opponent_checkpoint=str(tmp_path / "nfsp.pt"),
        )


def test_tianshou_rainbow_native_control_cuda_smoke_when_available(tmp_path):
    pytest.importorskip("tianshou")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    metrics = run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cuda",
        seed=10,
        output_json=str(tmp_path / "cuda.json"),
    )

    assert metrics["resolved_device"] == "cuda"
    assert metrics["train_steps"] >= 4


def test_tianshou_rainbow_checkpoint_h2h_against_native_nfsp(tmp_path):
    pytest.importorskip("tianshou")
    from poker_ai.research.native_nfsp import NativeNFSPConfig, run_native_nfsp_pilot

    rainbow_checkpoint = tmp_path / "rainbow.pt"
    nfsp_checkpoint = tmp_path / "nfsp.pt"
    run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=11,
        checkpoint_out=str(rainbow_checkpoint),
    )
    run_native_nfsp_pilot(
        NativeNFSPConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=8,
            min_buffer_size_to_learn=100,
            device="cpu",
            seed=12,
            checkpoint_path=str(nfsp_checkpoint),
        )
    )

    metrics = evaluate_rainbow_checkpoint_vs_native_nfsp(
        str(rainbow_checkpoint),
        str(nfsp_checkpoint),
        n_games=4,
        device="cpu",
        seed=13,
    )

    assert metrics["algorithm"] == "tianshou_rainbow_vs_native_nfsp_h2h"
    assert metrics["candidate_checkpoint"] == str(rainbow_checkpoint)
    assert metrics["baseline_checkpoint"] == str(nfsp_checkpoint)
    assert metrics["n_games"] == 4
    assert metrics["n_pairs"] == 2
    assert metrics["promotion"] is False


def test_tianshou_rainbow_checkpoint_h2h_against_npi_checkpoint(tmp_path):
    pytest.importorskip("tianshou")
    from poker_ai.research.neural_policy_iteration import (
        NeuralPolicyIterationConfig,
        run_neural_policy_iteration_pilot,
    )

    rainbow_checkpoint = tmp_path / "rainbow.pt"
    npi_checkpoint = tmp_path / "npi.pt"
    run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=19,
        checkpoint_out=str(rainbow_checkpoint),
    )
    run_neural_policy_iteration_pilot(
        NeuralPolicyIterationConfig(
            self_play_hands=2,
            max_improvement_targets=2,
            train_steps=1,
            hidden_dim=16,
            batch_size=2,
            device="cpu",
            seed=20,
            checkpoint_path=str(npi_checkpoint),
        )
    )

    metrics = evaluate_rainbow_checkpoint_vs_neural_policy_iteration(
        str(rainbow_checkpoint),
        str(npi_checkpoint),
        n_games=4,
        device="cpu",
        seed=21,
    )

    assert metrics["algorithm"] == "tianshou_rainbow_vs_neural_policy_iteration_h2h"
    assert metrics["candidate_checkpoint"] == str(rainbow_checkpoint)
    assert metrics["baseline_checkpoint"] == str(npi_checkpoint)
    assert metrics["n_games"] == 4
    assert metrics["n_pairs"] == 2
    assert metrics["promotion"] is False


def test_tianshou_rainbow_cli_can_compare_against_npi_checkpoint(tmp_path):
    pytest.importorskip("tianshou")
    from poker_ai.research.neural_policy_iteration import (
        NeuralPolicyIterationConfig,
        run_neural_policy_iteration_pilot,
    )
    from scripts.run_tianshou_rainbow_native_control import main as rainbow_main

    rainbow_checkpoint = tmp_path / "rainbow.pt"
    npi_checkpoint = tmp_path / "npi.pt"
    output = tmp_path / "rainbow_vs_npi.json"
    run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=22,
        checkpoint_out=str(rainbow_checkpoint),
    )
    run_neural_policy_iteration_pilot(
        NeuralPolicyIterationConfig(
            self_play_hands=2,
            max_improvement_targets=2,
            train_steps=1,
            hidden_dim=16,
            batch_size=2,
            device="cpu",
            seed=23,
            checkpoint_path=str(npi_checkpoint),
        )
    )

    exit_code = rainbow_main(
        [
            "--checkpoint-in",
            str(rainbow_checkpoint),
            "--baseline-checkpoint",
            str(npi_checkpoint),
            "--baseline-kind",
            "npi",
            "--eval-games",
            "4",
            "--device",
            "cpu",
            "--seed",
            "24",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "tianshou_rainbow_vs_neural_policy_iteration_h2h"
    assert payload["baseline_checkpoint"] == str(npi_checkpoint)


def test_tianshou_rainbow_checkpoint_h2h_against_rainbow_checkpoint(tmp_path):
    pytest.importorskip("tianshou")

    candidate = tmp_path / "candidate_rainbow.pt"
    baseline = tmp_path / "baseline_rainbow.pt"
    run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=34,
        checkpoint_out=str(candidate),
    )
    run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=35,
        checkpoint_out=str(baseline),
    )

    metrics = evaluate_rainbow_checkpoint_vs_rainbow(
        str(candidate),
        str(baseline),
        n_games=4,
        device="cpu",
        seed=36,
    )

    assert metrics["algorithm"] == "tianshou_rainbow_vs_tianshou_rainbow_h2h"
    assert metrics["candidate_checkpoint"] == str(candidate)
    assert metrics["baseline_checkpoint"] == str(baseline)
    assert metrics["n_pairs"] == 2
    assert metrics["promotion"] is False


def test_tianshou_rainbow_checkpoint_h2h_supports_fast_state_backend(tmp_path):
    pytest.importorskip("tianshou")

    checkpoint = tmp_path / "rainbow.pt"
    run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=20260780,
        checkpoint_out=str(checkpoint),
    )

    metrics = evaluate_rainbow_checkpoint_vs_rainbow(
        str(checkpoint),
        str(checkpoint),
        n_games=4,
        device="cpu",
        seed=20260781,
        eval_state_backend="fast-state",
    )

    assert metrics["eval_state_backend"] == "fast-state"
    assert metrics["n_pairs"] == 2
    assert metrics["mean_candidate_payoff"] == pytest.approx(0.0)


def test_tianshou_rainbow_h2h_canonical_fast_backend_matches_full_deck(tmp_path):
    pytest.importorskip("tianshou")

    candidate = tmp_path / "candidate_rainbow.pt"
    baseline = tmp_path / "baseline_rainbow.pt"
    run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=20260789,
        checkpoint_out=str(candidate),
    )
    run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=20260790,
        checkpoint_out=str(baseline),
    )

    full_deck = evaluate_rainbow_checkpoint_vs_rainbow(
        str(candidate),
        str(baseline),
        n_games=6,
        device="cpu",
        seed=20260791,
        eval_state_backend="full-deck",
    )
    canonical_fast = evaluate_rainbow_checkpoint_vs_rainbow(
        str(candidate),
        str(baseline),
        n_games=6,
        device="cpu",
        seed=20260791,
        eval_state_backend="fast-state-canonical-deal",
    )

    assert canonical_fast["eval_state_backend"] == "fast-state-canonical-deal"
    assert canonical_fast["mean_candidate_payoff"] == pytest.approx(
        full_deck["mean_candidate_payoff"]
    )
    assert canonical_fast["std_candidate_payoff"] == pytest.approx(
        full_deck["std_candidate_payoff"]
    )
    assert canonical_fast["eval_steps"] == full_deck["eval_steps"]


def test_rainbow_state_dicts_by_seat_preserves_independent_agents():
    p0 = {"weight": torch.tensor([0.0])}
    p1 = {"weight": torch.tensor([1.0])}

    by_seat = _rainbow_state_dicts_by_seat_from_payload(
        {
            "agent_model_state_dicts": {
                "player_0": p0,
                "player_1": p1,
            }
        }
    )

    assert by_seat[0] is p0
    assert by_seat[1] is p1


def test_rainbow_h2h_uses_seat_specific_policy_maps(monkeypatch):
    import scripts.run_tianshou_rainbow_native_control as rainbow_mod

    used_policies = []

    def fake_loader(checkpoint_path, _resolved_device):
        prefix = Path(checkpoint_path).name
        return (
            {"algorithm": "fake", "metrics": {"initial_chips": 1000}},
            {0: f"{prefix}:p0", 1: f"{prefix}:p1"},
        )

    def fake_action(policy, _features, legal_mask):
        used_policies.append(policy)
        if legal_mask[1] > 0:
            return 1
        return int(np.flatnonzero(legal_mask > 0)[0])

    monkeypatch.setattr(rainbow_mod, "_load_rainbow_checkpoint_policy_map", fake_loader)
    monkeypatch.setattr(rainbow_mod, "_rainbow_greedy_action", fake_action)

    evaluate_rainbow_checkpoint_vs_rainbow(
        "candidate.pt",
        "baseline.pt",
        n_games=4,
        device="cpu",
        seed=20260793,
        eval_state_backend="fast-state-canonical-deal",
    )

    assert "candidate.pt:p0" in used_policies
    assert "candidate.pt:p1" in used_policies
    assert "baseline.pt:p0" in used_policies
    assert "baseline.pt:p1" in used_policies


def test_tianshou_rainbow_cli_can_compare_against_rainbow_checkpoint(tmp_path):
    pytest.importorskip("tianshou")
    from scripts.run_tianshou_rainbow_native_control import main as rainbow_main

    candidate = tmp_path / "candidate_rainbow.pt"
    baseline = tmp_path / "baseline_rainbow.pt"
    output = tmp_path / "rainbow_vs_rainbow.json"
    run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=37,
        checkpoint_out=str(candidate),
    )
    run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=38,
        checkpoint_out=str(baseline),
    )

    exit_code = rainbow_main(
        [
            "--checkpoint-in",
            str(candidate),
            "--baseline-checkpoint",
            str(baseline),
            "--baseline-kind",
            "rainbow",
            "--eval-games",
            "4",
            "--device",
            "cpu",
            "--seed",
            "39",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "tianshou_rainbow_vs_tianshou_rainbow_h2h"
    assert payload["baseline_checkpoint"] == str(baseline)


def test_tianshou_rainbow_cli_can_compare_with_fast_state_eval_backend(tmp_path):
    pytest.importorskip("tianshou")
    from scripts.run_tianshou_rainbow_native_control import main as rainbow_main

    checkpoint = tmp_path / "rainbow.pt"
    output = tmp_path / "rainbow_fast_eval.json"
    run_control(
        train_steps=4,
        updates=1,
        batch_size=4,
        hidden_dim=16,
        num_atoms=11,
        warmup_steps=4,
        eval_games=1,
        device="cpu",
        seed=20260782,
        checkpoint_out=str(checkpoint),
    )

    exit_code = rainbow_main(
        [
            "--checkpoint-in",
            str(checkpoint),
            "--baseline-checkpoint",
            str(checkpoint),
            "--baseline-kind",
            "rainbow",
            "--eval-games",
            "4",
            "--device",
            "cpu",
            "--seed",
            "20260783",
            "--eval-state-backend",
            "fast-state",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["eval_state_backend"] == "fast-state"
    assert payload["mean_candidate_payoff"] == pytest.approx(0.0)


def test_tianshou_ppo_native_control_smoke_uses_vector_env(tmp_path):
    pytest.importorskip("tianshou")
    from scripts.run_tianshou_ppo_native_control import run_control as run_ppo_control

    output = tmp_path / "ppo.json"
    checkpoint = tmp_path / "ppo.pt"
    metrics = run_ppo_control(
        rollout_steps=8,
        updates=1,
        repeat=1,
        batch_size=4,
        hidden_dim=16,
        num_envs=2,
        vector_env_backend="dummy",
        eval_games=2,
        device="cpu",
        seed=15,
        output_json=str(output),
        checkpoint_out=str(checkpoint),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "tianshou_ppo"
    assert payload["role"] == "rl_control_baseline"
    assert payload["environment"] == "poker_ai:full_deck_hu_nlhe_single_agent_control"
    assert payload["num_actions"] == 9
    assert payload["num_envs"] == 2
    assert payload["vector_env_backend"] == "dummy"
    assert payload["promotion"] is False
    assert Path(metrics["checkpoint_path"]).exists()


def test_tianshou_ppo_checkpoint_h2h_against_native_nfsp(tmp_path):
    pytest.importorskip("tianshou")
    from poker_ai.research.native_nfsp import NativeNFSPConfig, run_native_nfsp_pilot
    from scripts.run_tianshou_ppo_native_control import (
        evaluate_ppo_checkpoint_vs_native_nfsp,
        run_control as run_ppo_control,
    )

    ppo_checkpoint = tmp_path / "ppo.pt"
    nfsp_checkpoint = tmp_path / "nfsp.pt"
    run_ppo_control(
        rollout_steps=4,
        updates=1,
        repeat=1,
        batch_size=4,
        hidden_dim=16,
        num_envs=1,
        eval_games=1,
        device="cpu",
        seed=16,
        checkpoint_out=str(ppo_checkpoint),
    )
    run_native_nfsp_pilot(
        NativeNFSPConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=8,
            min_buffer_size_to_learn=100,
            device="cpu",
            seed=17,
            checkpoint_path=str(nfsp_checkpoint),
        )
    )

    metrics = evaluate_ppo_checkpoint_vs_native_nfsp(
        str(ppo_checkpoint),
        str(nfsp_checkpoint),
        n_games=4,
        device="cpu",
        seed=18,
    )

    assert metrics["algorithm"] == "tianshou_ppo_vs_native_nfsp_h2h"
    assert metrics["candidate_checkpoint"] == str(ppo_checkpoint)
    assert metrics["baseline_checkpoint"] == str(nfsp_checkpoint)
    assert metrics["n_games"] == 4
    assert metrics["n_pairs"] == 2
    assert metrics["promotion"] is False
