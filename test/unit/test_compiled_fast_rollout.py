import numpy as np
import subprocess
import sys

from poker_ai.deep_cfr.fast_state import N_ACTIONS, FastPokerState, new_fast_game
from poker_ai.research.compiled_fast_rollout import (
    CompiledFastStateBatch,
    compiled_apply_actions,
    compiled_feature_vectors,
    compiled_legal_masks,
    run_compiled_fast_transition_benchmark,
)


def _seeded_fast_states(n_games: int, *, seed: int = 20260830) -> list[FastPokerState]:
    states = []
    for game_i in range(n_games):
        np.random.seed(seed + game_i)
        states.append(new_fast_game(2, initial_chips=1000))
    return states


def _assert_batch_matches_fast_states(batch: CompiledFastStateBatch, states: list[FastPokerState]) -> None:
    masks = compiled_legal_masks(batch)
    assert masks.shape == (len(states), N_ACTIONS)
    for row_i, state in enumerate(states):
        if state.is_terminal:
            continue
        assert int(batch.current_players()[row_i]) == int(state.current_player_i)
        assert np.array_equal(masks[row_i] > 0, state.get_legal_mask() > 0)
        assert int(batch.stage[row_i]) == int(state.stage)
        assert int(batch.n_raises[row_i]) == int(state.n_raises)
        assert int(batch.player_i_index[row_i]) == int(state._player_i_index)
        assert int(batch.n_actions[row_i]) == int(state.n_actions)
        assert int(batch.n_players_started_round[row_i]) == int(state.n_players_started_round)
        assert int(batch.pot_total[row_i]) == int(state.pot_total)
        assert np.array_equal(batch.chips[row_i], state.chips)
        assert np.array_equal(batch.bets[row_i], state.bets)
        assert np.array_equal(batch.active[row_i], state.active)
        assert np.array_equal(batch.community[row_i], state.community)
        assert int(batch.deck_cursor[row_i]) == int(state.deck_cursor)
        assert np.array_equal(batch.history[row_i], state.history)


def test_compiled_batch_transitions_match_fast_state_before_showdown():
    states = _seeded_fast_states(6)
    batch = CompiledFastStateBatch.from_fast_states(states)

    for step_i in range(6):
        _assert_batch_matches_fast_states(batch, states)
        actions = np.full(len(states), -1, dtype=np.int16)
        masks = compiled_legal_masks(batch)
        for row_i, state in enumerate(states):
            if state.is_terminal:
                continue
            legal = np.flatnonzero(masks[row_i] > 0)
            assert legal.size > 0
            if step_i % 3 == 1 and 2 in legal:
                action = 2
            elif 1 in legal:
                action = 1
            else:
                action = int(legal[0])
            actions[row_i] = action
            state.apply_action(action)

        result = compiled_apply_actions(batch, actions)
        assert result["needs_python_showdown"] == 0
        assert result["applied"] > 0

    _assert_batch_matches_fast_states(batch, states)


def test_compiled_batch_showdown_payout_matches_fast_state_call_down():
    states = _seeded_fast_states(8, seed=20260831)
    batch = CompiledFastStateBatch.from_fast_states(states)

    for _ in range(16):
        actions = np.full(len(states), -1, dtype=np.int16)
        masks = compiled_legal_masks(batch)
        for row_i, state in enumerate(states):
            if state.is_terminal:
                continue
            actions[row_i] = 1 if masks[row_i, 1] > 0 else int(np.flatnonzero(masks[row_i] > 0)[0])
            state.apply_action(int(actions[row_i]))
        result = compiled_apply_actions(batch, actions)
        assert result["needs_python_showdown"] == 0
        if all(state.is_terminal for state in states):
            break

    assert all(state.is_terminal for state in states)
    assert not np.any(batch.needs_python_showdown)
    for row_i, state in enumerate(states):
        assert int(batch.stage[row_i]) in {FastPokerState.SHOWDOWN, FastPokerState.TERMINAL}
        assert np.array_equal(batch.chips[row_i], state.chips)
        assert np.array_equal(batch.bets[row_i], state.bets)
        assert int(batch.pot_total[row_i]) == int(state.pot_total)


def test_compiled_batch_random_full_hands_match_fast_state_terminal_payouts():
    rng = np.random.default_rng(20260833)
    states = _seeded_fast_states(24, seed=20260833)
    batch = CompiledFastStateBatch.from_fast_states(states)

    for _ in range(64):
        actions = np.full(len(states), -1, dtype=np.int16)
        masks = compiled_legal_masks(batch)
        for row_i, state in enumerate(states):
            if state.is_terminal:
                continue
            legal = np.flatnonzero(masks[row_i] > 0)
            assert legal.size > 0
            action = int(rng.choice(legal))
            actions[row_i] = action
            state.apply_action(action)
        result = compiled_apply_actions(batch, actions)
        assert result["needs_python_showdown"] == 0
        if all(state.is_terminal for state in states):
            break

    assert all(state.is_terminal for state in states)
    for row_i, state in enumerate(states):
        assert np.array_equal(batch.chips[row_i], state.chips)
        assert np.array_equal(batch.bets[row_i], state.bets)
        assert np.array_equal(batch.active[row_i], state.active)
        assert int(batch.pot_total[row_i]) == int(state.pot_total)


def test_compiled_feature_vectors_match_fast_state_observations():
    states = _seeded_fast_states(10, seed=20260834)
    batch = CompiledFastStateBatch.from_fast_states(states)

    for step_i in range(8):
        features = compiled_feature_vectors(batch)
        assert features.shape == (len(states), 126)
        for row_i, state in enumerate(states):
            if state.is_terminal:
                continue
            assert np.array_equal(features[row_i], state.to_feature_vector())

        actions = np.full(len(states), -1, dtype=np.int16)
        masks = compiled_legal_masks(batch)
        for row_i, state in enumerate(states):
            if state.is_terminal:
                continue
            legal = np.flatnonzero(masks[row_i] > 0)
            action = 2 if step_i % 4 == 0 and 2 in legal else int(legal[min(step_i, legal.size - 1)])
            actions[row_i] = action
            state.apply_action(action)
        compiled_apply_actions(batch, actions)


def test_compiled_transition_benchmark_reports_parity_and_speedup_fields():
    metrics = run_compiled_fast_transition_benchmark(
        n_games=16,
        max_steps_per_game=12,
        initial_chips=1000,
        seed=20260832,
        min_speedup=1.1,
    )

    assert metrics["backend"] == "compiled-fast-state"
    assert metrics["parity"]["passed"] is True
    assert metrics["baseline"]["steps"] > 0
    assert metrics["candidate"]["steps"] > 0
    assert metrics["speedup"] > 0.0
    assert "passed" in metrics
    assert metrics["terminal_payout_supported"] is True
    assert metrics["fallback_free"] is True


def test_eval_native_rollout_substrate_cli_can_include_compiled_transition(tmp_path):
    output = tmp_path / "compiled_transition_gate.json"

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
            "--include-compiled-transition-benchmark",
            "--compiled-benchmark-games",
            "16",
            "--compiled-max-steps",
            "12",
            "--compiled-min-speedup",
            "1.1",
            "--output-json",
            str(output),
        ],
        cwd=".",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode in {0, 1}
    assert output.exists()
    import json

    metrics = json.loads(output.read_text())
    compiled = metrics["compiled_transition_benchmark"]
    assert compiled["backend"] == "compiled-fast-state"
    assert compiled["parity"]["passed"] is True
    assert "training_integration_allowed" in compiled
