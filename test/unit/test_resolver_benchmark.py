import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES
from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.research.belief_probe import N_HANDS, _HAND_TO_INDEX
from poker_ai.research.resolver_benchmark import (
    ResolverBenchmarkCase,
    run_resolver_benchmark,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_learned_river_leaf_resolver_ab import (  # noqa: E402
    _action_path_to_street_string,
    _global_board_mask,
    _local_ranges_from_belief,
)
from eval_joint_pbs_resolver_cut_ab import (  # noqa: E402
    StructuralCutRiskPredictor,
    _case_slice,
    _frontier_action_shape,
    _load_static_belief_by_label,
    _select_successor_cut_node_records,
    _successor_cut_node_indices,
)
from eval_policy_prior_solver_budget import mix_strategy  # noqa: E402
from eval_policy_warm_start_solver_budget import build_policy_warm_start  # noqa: E402
from eval_regret_oracle_warm_start import build_regret_oracle_warm_start  # noqa: E402
from train_policy_residual_combiner import (  # noqa: E402
    make_combiner_features,
    masked_softmax_np,
    policy_metrics,
)
from build_learned_river_leaf_cases import (  # noqa: E402
    _rotated_cards,
    _summarize_leaf_records,
)
from build_regret_policy_warm_start_targets import normalize_action_rows  # noqa: E402
from solver import StreetSolver  # noqa: E402


def _small_checkpoint(path: Path) -> None:
    checkpoint = {
        "iteration": 9,
        "n_players": 2,
        "hidden_dim": 16,
        "n_layers": 1,
        "initial_chips": 20000,
        "value_net": {
            "net.0.weight": torch.randn(16, N_FEATURES),
            "net.0.bias": torch.randn(16),
            "net.2.weight": torch.randn(N_ACTIONS, 16),
            "net.2.bias": torch.randn(N_ACTIONS),
        },
    }
    torch.save(checkpoint, path)


def test_resolver_benchmark_reports_fixed_state_policy_and_solver_metrics():
    torch.manual_seed(0)
    value_net = ValueNetwork(N_FEATURES, 16, N_ACTIONS, n_layers=1)
    case = ResolverBenchmarkCase(
        label="river-check",
        hole_cards=("Ac", "Kd"),
        board=("2c", "7d", "Jh", "4s", "9c"),
        action_str="ck/kk/kk/",
        client_pos=0,
    )

    metrics = run_resolver_benchmark(
        value_net,
        torch.device("cpu"),
        cases=[case],
        solver_iterations=1,
    )

    assert metrics["passed"] is True
    assert metrics["n_cases"] == 1
    assert metrics["n_solver_cases"] == 1
    result = metrics["cases"][0]
    assert result["label"] == "river-check"
    assert 0 <= result["blueprint_action"] < N_ACTIONS
    assert 0 <= result["solver_action"] < N_ACTIONS
    assert result["blueprint_action_legal"] is True
    assert 0 <= result["blueprint_no_allin_action"] < N_ACTIONS
    assert result["blueprint_no_allin_action_legal"] is True
    assert isinstance(result["blueprint_allin_selected"], bool)
    assert isinstance(result["allin_removed_action_changed"], bool)
    assert 0 <= result["policy_head_action"] < N_ACTIONS
    assert result["policy_head_action_legal"] is True
    assert isinstance(result["policy_head_allin_selected"], bool)
    assert 0.0 <= result["policy_head_action_l1_drift"] <= 2.0
    assert result["solver_action_legal"] is True
    assert result["solver_latency_ms"] >= 0.0
    assert 0.0 <= result["action_l1_drift"] <= 2.0
    assert isinstance(result["advantage_delta_proxy"], float)
    assert "blueprint_allin_rate" in metrics
    assert "no_allin_changed_rate" in metrics
    assert "policy_head_allin_rate" in metrics
    assert "solver_allin_rate" in metrics
    assert "policy_head_solver_allin_gap" in metrics
    assert "policy_head_mean_allin_prob" in metrics
    assert "solver_mean_allin_prob" in metrics
    assert "policy_head_solver_allin_prob_gap" in metrics
    assert "policy_head_mean_action_l1_drift" in metrics
    assert "policy_head_behavior_passed" in metrics
    assert metrics["mechanical_passed"] is True

    gated = run_resolver_benchmark(
        value_net,
        torch.device("cpu"),
        cases=[case],
        solver_iterations=1,
        enforce_policy_head_behavior_gate=True,
        max_policy_head_allin_rate=-1.0,
    )
    assert gated["mechanical_passed"] is True
    assert gated["policy_head_behavior_passed"] is False
    assert gated["passed"] is False

    relative_gate = run_resolver_benchmark(
        value_net,
        torch.device("cpu"),
        cases=[case],
        solver_iterations=1,
        enforce_policy_head_behavior_gate=True,
        max_policy_head_solver_allin_gap=2.0,
        max_policy_head_mean_l1_drift=3.0,
    )
    assert relative_gate["policy_head_behavior_gate"]["mode"] == "solver_gap"
    assert relative_gate["policy_head_behavior_passed"] is True

    probability_gate = run_resolver_benchmark(
        value_net,
        torch.device("cpu"),
        cases=[case],
        solver_iterations=1,
        enforce_policy_head_behavior_gate=True,
        max_policy_head_solver_allin_prob_gap=2.0,
        max_policy_head_mean_l1_drift=3.0,
    )
    assert probability_gate["policy_head_behavior_gate"]["mode"] == "solver_prob_gap"
    assert probability_gate["policy_head_behavior_passed"] is True


def test_street_solver_omits_under_minimum_raise_buckets():
    solver = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )

    assert 2 not in solver.root.children
    assert 3 in solver.root.children


def test_policy_prior_mix_strategy_normalizes_and_validates_weight():
    base = np.zeros(N_ACTIONS, dtype=np.float32)
    prior = np.zeros(N_ACTIONS, dtype=np.float32)
    base[1] = 1.0
    prior[8] = 1.0

    mixed = mix_strategy(base, prior, 0.25)

    assert mixed[1] == pytest.approx(0.75)
    assert mixed[8] == pytest.approx(0.25)
    assert mixed.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="prior_weight"):
        mix_strategy(base, prior, 1.5)


def test_street_solver_showdown_leaf_callback_can_reproduce_default():
    base = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    hooked = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    calls = []

    def passthrough_leaf(**kwargs):
        calls.append(kwargs["showdown_indices"].shape[0])
        return kwargs["default_hero_values"], kwargs["default_villain_values"]

    base.solve(n_iterations=2)
    hooked.solve(n_iterations=2, showdown_leaf_fn=passthrough_leaf)

    assert calls
    np.testing.assert_allclose(hooked._regret_sum, base._regret_sum, atol=1e-5)
    np.testing.assert_allclose(hooked._strategy_sum, base._strategy_sum, atol=1e-5)


def test_street_solver_zero_warm_start_reproduces_default():
    base = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    warmed = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    shape = (
        warmed._tree["n_nodes"],
        warmed._tree["n_actions"],
        warmed.n,
    )
    zeros = np.zeros(shape, dtype=np.float32)

    base.solve(n_iterations=2)
    warmed.solve(
        n_iterations=2,
        initial_regret_sum=zeros,
        initial_strategy_sum=zeros,
    )

    np.testing.assert_allclose(warmed._regret_sum, base._regret_sum, atol=1e-5)
    np.testing.assert_allclose(warmed._strategy_sum, base._strategy_sum, atol=1e-5)


def test_street_solver_strategy_warm_start_is_visible_without_iterations():
    solver = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    shape = (solver._tree["n_nodes"], solver._tree["n_actions"], solver.n)
    initial_strategy = np.zeros(shape, dtype=np.float32)
    hand = solver.hands[0]
    action = max(solver.root.children)
    initial_strategy[0, action, 0] = 1.0

    solver.solve(n_iterations=0, initial_strategy_sum=initial_strategy)

    strategy = solver.get_strategy(hand, solver.root)
    assert strategy[action] == pytest.approx(1.0)


def test_street_solver_warm_start_rejects_bad_shape():
    solver = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )

    with pytest.raises(ValueError, match="initial_regret_sum shape"):
        solver.solve(n_iterations=0, initial_regret_sum=np.zeros((1, 1, 1), dtype=np.float32))


def test_build_policy_warm_start_seeds_only_selected_node():
    solver = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    policy = np.zeros((solver._tree["n_actions"], solver.n), dtype=np.float32)
    policy[1] = 0.25
    policy[max(solver.root.children)] = 0.75

    regret, strategy = build_policy_warm_start(
        solver=solver,
        node_idx=0,
        policy_by_action_hand=policy,
        regret_mass=100.0,
        strategy_mass=2.0,
    )

    assert regret.shape == strategy.shape
    assert regret[0, 1, 0] == pytest.approx(25.0)
    assert strategy[0, max(solver.root.children), 0] == pytest.approx(1.5)
    assert regret[1:].sum() == pytest.approx(0.0)


def test_build_regret_oracle_warm_start_copies_teacher_regrets_only_selected_node():
    reference = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    target = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    shape = (
        reference._tree["n_nodes"],
        reference._tree["n_actions"],
        reference.n,
    )
    reference._regret_sum = np.zeros(shape, dtype=np.float32)
    reference._strategy_sum = np.zeros(shape, dtype=np.float32)
    seeded_action = max(reference.root.children)
    reference._regret_sum[0, seeded_action, 0] = 3.0
    reference._strategy_sum[0, seeded_action, 0] = 0.25

    initial_regret, initial_strategy = build_regret_oracle_warm_start(
        reference_solver=reference,
        reference_node=reference.root,
        target_solver=target,
        target_node=target.root,
        regret_scale=2.0,
    )

    assert initial_regret[0, seeded_action, 0] == pytest.approx(6.0)
    assert initial_strategy[0, seeded_action, 0] == pytest.approx(0.25)
    assert np.count_nonzero(initial_regret[1:]) == 0
    assert np.count_nonzero(initial_strategy[1:]) == 0


def test_normalize_action_rows_masks_illegal_and_falls_back_to_legal_uniform():
    legal_mask = np.zeros(N_ACTIONS, dtype=np.float32)
    legal_mask[[1, 8]] = 1.0
    values = np.zeros((2, N_ACTIONS), dtype=np.float32)
    values[0, 1] = 3.0
    values[0, 8] = 1.0
    values[0, 2] = 99.0

    normalized = normalize_action_rows(values, legal_mask)

    assert normalized[0, 1] == pytest.approx(0.75)
    assert normalized[0, 8] == pytest.approx(0.25)
    assert normalized[0, 2] == pytest.approx(0.0)
    assert normalized[1, 1] == pytest.approx(0.5)
    assert normalized[1, 8] == pytest.approx(0.5)


def test_policy_residual_combiner_features_validate_shapes():
    features = np.zeros((2, N_FEATURES), dtype=np.float32)
    masks = np.ones((2, N_ACTIONS), dtype=np.float32)
    low = np.full((2, N_ACTIONS), 1.0 / N_ACTIONS, dtype=np.float32)
    policy = low.copy()

    combined = make_combiner_features(features, masks, low, policy)

    assert combined.shape == (2, N_FEATURES + 3 * N_ACTIONS)
    with pytest.raises(ValueError, match="combiner feature part"):
        make_combiner_features(features[:1], masks, low, policy)


def test_policy_residual_combiner_metrics_mask_and_count_allin():
    logits = np.zeros((2, N_ACTIONS), dtype=np.float32)
    logits[0, 8] = 5.0
    logits[0, 1] = 4.0
    logits[1, 1] = 5.0
    masks = np.ones((2, N_ACTIONS), dtype=np.float32)
    masks[0, 8] = 0.0
    target = np.zeros((2, N_ACTIONS), dtype=np.float32)
    target[:, 1] = 1.0

    probs = masked_softmax_np(logits, masks)
    metrics = policy_metrics(probs, target, masks)

    assert probs[0, 8] == pytest.approx(0.0)
    assert metrics["top_allin_rate"] == pytest.approx(0.0)
    assert metrics["top_action_agreement"] == pytest.approx(1.0)


def test_street_solver_showdown_leaf_callback_rejects_torch_backend():
    solver = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )

    with pytest.raises(ValueError, match="CPU CFR backend"):
        solver.solve(n_iterations=1, backend="torch-cpu", showdown_leaf_fn=lambda **_: None)


def test_street_solver_cut_node_callback_stops_descendant_updates():
    solver = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    cut_node = solver.root.children[1]
    cut_idx = solver._tree["all_nodes"].index(cut_node)
    descendant = cut_node.children[1]
    descendant_idx = solver._tree["all_nodes"].index(descendant)
    calls = []

    def zero_cut(**kwargs):
        calls.append(kwargs["cut_indices"].copy())
        n_cut = kwargs["cut_indices"].shape[0]
        n_hands = kwargs["hero_reach"].shape[1]
        return (
            np.zeros((n_cut, n_hands), dtype=np.float32),
            np.zeros((n_cut, n_hands), dtype=np.float32),
        )

    solver.solve(n_iterations=2, cut_node_indices=[cut_idx], cut_node_fn=zero_cut)

    assert len(calls) == 2
    assert calls[0].tolist() == [cut_idx]
    np.testing.assert_allclose(solver._strategy_sum[cut_idx], 0.0)
    np.testing.assert_allclose(solver._strategy_sum[descendant_idx], 0.0)


def test_street_solver_cut_node_callback_rejects_bad_shape():
    solver = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    cut_idx = solver._tree["all_nodes"].index(solver.root.children[1])

    def bad_cut(**kwargs):
        n_hands = kwargs["hero_reach"].shape[1]
        return (
            np.zeros((2, n_hands), dtype=np.float32),
            np.zeros((1, n_hands), dtype=np.float32),
        )

    with pytest.raises(ValueError, match="cut_node_fn hero values shape"):
        solver.solve(n_iterations=1, cut_node_indices=[cut_idx], cut_node_fn=bad_cut)


def test_street_solver_cut_node_callback_rejects_torch_backend():
    solver = StreetSolver(
        board=[0, 1, 2, 3, 4],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    cut_idx = solver._tree["all_nodes"].index(solver.root.children[1])

    with pytest.raises(ValueError, match="CPU CFR backend"):
        solver.solve(
            n_iterations=1,
            backend="torch-cpu",
            cut_node_indices=[cut_idx],
            cut_node_fn=lambda **_: None,
        )


def test_street_solver_trace_node_callback_observes_without_changing_strategy():
    baseline = StreetSolver(
        board=[0, 1, 2, 3],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    traced = StreetSolver(
        board=[0, 1, 2, 3],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    trace_idx = traced._tree["all_nodes"].index(traced.root.children[1])
    calls = []

    def trace_nodes(**kwargs):
        calls.append(kwargs)
        assert kwargs["hero_reach"].shape == (1, traced.n)
        assert kwargs["villain_values"].shape == (1, traced.n)

    baseline.solve(n_iterations=2, backend="cpu")
    traced.solve(
        n_iterations=2,
        backend="cpu",
        trace_node_indices=[trace_idx],
        trace_node_fn=trace_nodes,
    )

    hand = traced.hands[0]
    assert len(calls) == 2
    assert calls[0]["iteration"] == 0
    assert calls[1]["iteration"] == 1
    assert baseline.get_strategy(hand) == traced.get_strategy(hand)


def test_successor_cut_node_indices_only_returns_nonterminal_children():
    solver = StreetSolver(
        board=[0, 1, 2, 3],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    cut_indices = _successor_cut_node_indices(solver, solver.root)

    assert cut_indices
    assert all(not solver._tree["all_nodes"][idx].is_terminal for idx in cut_indices)
    terminal_children = [
        child for child in solver.root.children.values() if child.is_terminal
    ]
    assert all(solver._tree["all_nodes"].index(child) not in cut_indices for child in terminal_children)


def test_successor_cut_node_indices_can_filter_frontier_shape():
    solver = StreetSolver(
        board=[0, 1, 2, 3],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    all_indices = _successor_cut_node_indices(solver, solver.root)
    bet_indices = _successor_cut_node_indices(solver, solver.root, min_bet_count=1)
    check_indices = _successor_cut_node_indices(
        solver,
        solver.root,
        target_action_shapes=("k",),
    )

    assert _frontier_action_shape("b400c/b1200") == "bc/b"
    assert bet_indices
    assert check_indices
    assert set(bet_indices).isdisjoint(check_indices)
    assert set(bet_indices).issubset(all_indices)


def test_structural_risk_predictor_filters_successor_cuts_before_solve():
    solver = StreetSolver(
        board=[0, 1, 2, 3],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    predictor = StructuralCutRiskPredictor(
        numeric_fields=(
            "actor_to_act",
            "bet_count",
            "client_pos",
            "cut_pos",
            "legal_action_count",
        ),
        categorical_fields=("action_shape",),
        category_vocab={"action_shape": ["k"]},
        numeric_mean=np.zeros(5, dtype=np.float64),
        numeric_std=np.ones(5, dtype=np.float64),
        weights=np.asarray([np.log(0.1 + 1e-6), 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        abstention_cut=0.2,
        abstention_rule="test_rule",
    )

    candidates, selected = _select_successor_cut_node_records(
        solver,
        solver.root,
        action_prefix="",
        client_pos=0,
        min_bet_count=0,
        target_action_shapes=(),
        risk_predictor=predictor,
    )

    assert candidates
    assert selected
    assert any(
        record["risk_decision"] == "risk_rejected_exact_fallback"
        for record in candidates
    )
    assert all(record["risk_decision"] == "selected_for_learned_value" for record in selected)


def test_static_belief_loader_maps_labels(tmp_path):
    path = tmp_path / "beliefs.npz"
    belief = np.zeros((2, 2 * N_HANDS), dtype=np.float32)
    belief[0, 0] = 1.0
    belief[1, 1] = 1.0
    np.savez_compressed(path, labels=np.asarray(["a", "b"]), belief=belief)

    loaded = _load_static_belief_by_label(path)

    assert set(loaded) == {"a", "b"}
    assert loaded["a"].shape == (2 * N_HANDS,)
    assert loaded["b"][1] == 1.0


def test_successor_cut_ab_case_slice_supports_heldout_offsets():
    cases = [
        ResolverBenchmarkCase(
            label=f"case-{idx}",
            hole_cards=("Ac", "Kd"),
            board=("2c", "7d", "Jh", "4s"),
            action_str="ck/kk/",
            client_pos=0,
        )
        for idx in range(5)
    ]

    assert [case.label for case in _case_slice(cases, start_index=2, limit=2)] == [
        "case-2",
        "case-3",
    ]
    assert [case.label for case in _case_slice(cases, start_index=4, limit=0)] == [
        "case-4"
    ]
    with pytest.raises(ValueError, match="start_index"):
        _case_slice(cases, start_index=-1, limit=1)


def test_learned_leaf_action_path_reconstructs_turn_sequence():
    solver = StreetSolver(
        board=[0, 1, 2, 3],
        pot=200,
        hero_stack=20000,
        villain_stack=20000,
        hero_first=True,
    )
    node = solver.root.children[1].children[1]
    node_idx = solver._tree["all_nodes"].index(node)

    street = _action_path_to_street_string(
        solver._tree,
        node_idx,
        hero_stack_start=solver.hero_stack_start,
        villain_stack_start=solver.villain_stack_start,
    )

    assert street == "kk"


def test_learned_leaf_range_and_mask_helpers_normalize_legal_hands():
    solver_hands = [(0, 1), (4, 5)]
    belief = np.zeros(2 * N_HANDS, dtype=np.float32)
    belief[_HAND_TO_INDEX[(0, 1)]] = 2.0
    belief[N_HANDS + _HAND_TO_INDEX[(4, 5)]] = 3.0

    hero, villain = _local_ranges_from_belief(belief, solver_hands)
    board_mask = _global_board_mask([0, 1, 2, 3, 4])

    np.testing.assert_allclose(hero, [1.0, 0.0])
    np.testing.assert_allclose(villain, [0.0, 1.0])
    assert int(board_mask.sum()) == 47 * 46 // 2


def test_river_leaf_record_summary_reports_dataset_skew():
    records = [
        {
            "source_case": "turn-a",
            "leaf_action_str": "ck/kk/kk/",
            "river_card": "2c",
            "terminal_node_idx": 7,
        },
        {
            "source_case": "turn-a",
            "leaf_action_str": "ck/kk/kk/",
            "river_card": "3d",
            "terminal_node_idx": 7,
        },
        {
            "source_case": "turn-b",
            "leaf_action_str": "ck/kk/b400",
            "river_card": "2c",
            "terminal_node_idx": 9,
        },
    ]

    summary = _summarize_leaf_records(records)

    assert summary["sources"]["n_unique"] == 2
    assert summary["sources"]["max_share"] == pytest.approx(2 / 3)
    assert summary["terminals"]["n_unique"] == 2
    assert summary["river_cards"]["top"][0] == {"key": "2c", "count": 2}
    assert summary["leaf_action_shapes"]["top"][0] == {"key": "ck/kk/kk/", "count": 2}
    assert summary["leaf_bet_counts"]["top"][0] == {"key": "0", "count": 2}
    assert summary["leaf_action_parse_errors"] == 0
    assert summary["leaf_total_last_bet_to"]["max"] == 500.0


def test_river_leaf_card_rotation_is_deterministic_and_keyed():
    cards = [1, 2, 3, 4, 5]

    first = _rotated_cards(cards, key="seed:a")
    second = _rotated_cards(cards, key="seed:a")
    other = _rotated_cards(cards, key="seed:b")

    assert first == second
    assert sorted(first) == cards
    assert sorted(other) == cards
    assert first != cards or other != cards


def test_resolver_benchmark_cli_emits_json_for_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "candidate.pt"
    _small_checkpoint(checkpoint_path)
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_resolver_benchmark.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint_path),
            "--device",
            "cpu",
            "--max-cases",
            "1",
            "--solver-iterations",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    metrics = json.loads(result.stdout)
    assert metrics["passed"] is True
    assert metrics["checkpoint_iteration"] == 9
    assert metrics["n_cases"] == 1
    assert metrics["promotion_blockers"] == [
        "fixed_state_resolver_benchmark_is_not_slumbot_confidence"
    ]
