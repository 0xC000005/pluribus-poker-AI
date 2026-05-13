import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fast_cfr import build_tree_arrays
from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research import belief_value_probe as bvp
from poker_ai.research.belief_value_probe import compute_hero_hand_ev
from eval_public_belief_dual_hand_cfv_probe import (
    DualCFVDataset,
    _DualHandCFVProbeNet,
    load_public_belief_dual_hand_cfv_checkpoint,
    predict_public_belief_dual_hand_cfv_ensemble,
    predict_public_belief_dual_hand_cfv_model,
    predict_public_belief_dual_hand_cfv_model_vectorized,
    project_dual_cfv_zero_sum,
)
from analyze_dual_cfv_cache_errors import (
    enrich_leaf_record,
    group_error_records,
    merge_record_metadata,
)
from build_joint_pbs_continuation_targets import build_joint_payload
from solver import Node


def _write_checkpoint(path: Path) -> None:
    torch.manual_seed(0)
    net = ValueNetwork(N_FEATURES, 8, N_ACTIONS, n_layers=1, use_betting_history=True)
    torch.save(
        {
            "iteration": 1,
            "n_players": 2,
            "hidden_dim": 8,
            "n_layers": 1,
            "initial_chips": 20000,
            "uses_betting_history": True,
            "value_net": net.state_dict(),
        },
        path,
    )


def _write_targets_and_cases(tmp_path: Path) -> tuple[Path, Path]:
    features = np.zeros((1, N_FEATURES), dtype=np.float32)
    legal_masks = np.ones((1, N_ACTIONS), dtype=np.float32)
    target_probs = np.ones((1, N_ACTIONS), dtype=np.float32) / float(N_ACTIONS)
    targets_path = tmp_path / "targets.npz"
    PolicyTargetBuffer(features, legal_masks, target_probs).save_npz(targets_path)

    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "label": "river-open-check",
                        "hole_cards": ["Ac", "Kd"],
                        "board": ["2c", "7d", "Jh", "4s", "9c"],
                        "action_str": "ck/kk/kk/",
                        "client_pos": 0,
                        "source": "unit",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return targets_path, cases_path


def test_compute_hero_hand_ev_uses_average_strategy():
    root = Node(player=0, pot=100, stacks=(100, 100), to_call=10, n_raises=0)
    fold = Node(
        player=-1,
        pot=100,
        stacks=(90, 100),
        to_call=0,
        n_raises=0,
        terminal_type="hero_fold",
    )
    showdown = Node(
        player=-1,
        pot=120,
        stacks=(90, 90),
        to_call=0,
        n_raises=0,
        terminal_type="showdown",
    )
    root.children[0] = fold
    root.children[1] = showdown
    tree = build_tree_arrays(root)

    strategy_sum = np.zeros((tree["n_nodes"], tree["n_actions"], 2), dtype=np.float32)
    strategy_sum[0, 0, 0] = 1.0
    strategy_sum[0, 1, 0] = 3.0
    valid = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    solver = SimpleNamespace(
        _tree=tree,
        _strategy_sum=strategy_sum,
        hands=[(0, 1), (2, 3)],
        hand_to_idx={(0, 1): 0, (2, 3): 1},
        n=2,
        valid=valid,
        win_m=valid.copy(),
        lose_m=np.zeros_like(valid),
        tie_m=np.zeros_like(valid),
        pot_start=100,
        hero_stack_start=100,
        villain_stack_start=100,
    )

    ev = compute_hero_hand_ev(solver, root, (0, 1), np.array([0.0, 1.0]))

    assert ev == 80.0


def test_compute_hero_cfv_vector_matches_hand_ev():
    root = Node(player=0, pot=100, stacks=(100, 100), to_call=10, n_raises=0)
    fold = Node(
        player=-1,
        pot=100,
        stacks=(90, 100),
        to_call=0,
        n_raises=0,
        terminal_type="hero_fold",
    )
    showdown = Node(
        player=-1,
        pot=120,
        stacks=(90, 90),
        to_call=0,
        n_raises=0,
        terminal_type="showdown",
    )
    root.children[0] = fold
    root.children[1] = showdown
    tree = build_tree_arrays(root)

    strategy_sum = np.zeros((tree["n_nodes"], tree["n_actions"], 2), dtype=np.float32)
    strategy_sum[0, 0, 0] = 1.0
    strategy_sum[0, 1, 0] = 3.0
    valid = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    solver = SimpleNamespace(
        _tree=tree,
        _strategy_sum=strategy_sum,
        hands=[(0, 1), (2, 3)],
        hand_to_idx={(0, 1): 0, (2, 3): 1},
        n=2,
        valid=valid,
        win_m=valid.copy(),
        lose_m=np.zeros_like(valid),
        tie_m=np.zeros_like(valid),
        pot_start=100,
        hero_stack_start=100,
        villain_stack_start=100,
    )

    values, mask = bvp.compute_hero_cfv_vector(solver, root, np.array([0.0, 1.0]))

    assert values[0] == 80.0
    assert mask[0] == 1.0


def test_compute_villain_cfv_vector_uses_hero_reach_and_strategy():
    root = Node(player=0, pot=100, stacks=(100, 100), to_call=10, n_raises=0)
    fold = Node(
        player=-1,
        pot=100,
        stacks=(90, 100),
        to_call=0,
        n_raises=0,
        terminal_type="hero_fold",
    )
    showdown = Node(
        player=-1,
        pot=120,
        stacks=(90, 90),
        to_call=0,
        n_raises=0,
        terminal_type="showdown",
    )
    root.children[0] = fold
    root.children[1] = showdown
    tree = build_tree_arrays(root)

    strategy_sum = np.zeros((tree["n_nodes"], tree["n_actions"], 2), dtype=np.float32)
    strategy_sum[0, 0, 0] = 1.0
    strategy_sum[0, 1, 0] = 3.0
    valid = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    solver = SimpleNamespace(
        _tree=tree,
        _strategy_sum=strategy_sum,
        hands=[(0, 1), (2, 3)],
        hand_to_idx={(0, 1): 0, (2, 3): 1},
        n=2,
        valid=valid,
        win_m=valid.copy(),
        lose_m=np.zeros_like(valid),
        tie_m=np.zeros_like(valid),
        pot_start=100,
        hero_stack_start=100,
        villain_stack_start=100,
    )

    values, mask = bvp.compute_villain_cfv_vector(solver, root, np.array([1.0, 0.0]))

    assert values[1] == 20.0
    assert mask[1] == 1.0


def test_dual_cfv_probe_accepts_bottlenecked_separate_heads():
    model = _DualHandCFVProbeNet(
        8,
        use_belief=True,
        head_mode="separate",
        belief_bottleneck_dim=4,
    )
    public_x = torch.zeros((3, N_FEATURES), dtype=torch.float32)
    hand_x = torch.zeros((3, 52), dtype=torch.float32)
    player_x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=torch.float32,
    )
    belief_x = torch.zeros((3, bvp.BELIEF_DIM), dtype=torch.float32)

    out = model(public_x, hand_x, player_x, belief_x)

    assert out.shape == (3,)


def test_dual_cfv_probe_accepts_deepset_card_encoder():
    model = _DualHandCFVProbeNet(
        8,
        use_belief=True,
        head_mode="separate",
        belief_bottleneck_dim=4,
        card_encoder="deepset",
    )
    public_x = torch.zeros((3, N_FEATURES), dtype=torch.float32)
    public_x[:, 52:57] = 1.0
    hand_x = torch.zeros((3, 52), dtype=torch.float32)
    hand_x[:, :2] = 1.0
    player_x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=torch.float32,
    )
    belief_x = torch.zeros((3, bvp.BELIEF_DIM), dtype=torch.float32)

    out = model(public_x, hand_x, player_x, belief_x)

    assert out.shape == (3,)


def test_dual_cfv_checkpoint_round_trip_predicts_both_players(tmp_path):
    model = _DualHandCFVProbeNet(
        8,
        use_belief=True,
        head_mode="separate",
        belief_bottleneck_dim=4,
    )
    checkpoint = tmp_path / "dual_cfv.pt"
    torch.save(
        {
            "mode": "public_belief_dual_hand_cfv_checkpoint",
            "model_state": model.state_dict(),
            "hidden_dim": 8,
            "head_mode": "separate",
            "belief_bottleneck_dim": 4,
            "public_mean": np.zeros((1, N_FEATURES), dtype=np.float32),
            "public_std": np.ones((1, N_FEATURES), dtype=np.float32),
            "belief_mean": np.zeros((1, bvp.BELIEF_DIM), dtype=np.float32),
            "belief_std": np.ones((1, bvp.BELIEF_DIM), dtype=np.float32),
            "target_mean": 0.0,
            "target_std": 1.0,
        },
        checkpoint,
    )
    loaded, payload = load_public_belief_dual_hand_cfv_checkpoint(
        checkpoint,
        device="cpu",
    )
    pred = predict_public_belief_dual_hand_cfv_model(
        loaded,
        payload,
        np.zeros((2, N_FEATURES), dtype=np.float32),
        np.zeros((2, bvp.BELIEF_DIM), dtype=np.float32),
        np.ones((2, bvp.N_HANDS), dtype=np.float32),
        np.ones((2, bvp.N_HANDS), dtype=np.float32),
        device="cpu",
        batch_size=512,
    )

    assert pred.shape == (2, 2, bvp.N_HANDS)


def test_dual_cfv_checkpoint_ensemble_averages_predictions(tmp_path):
    model = _DualHandCFVProbeNet(
        8,
        use_belief=True,
        head_mode="separate",
        belief_bottleneck_dim=4,
    )
    checkpoints = []
    for idx, target_mean in enumerate((1.0, 3.0)):
        checkpoint = tmp_path / f"dual_cfv_{idx}.pt"
        torch.save(
            {
                "mode": "public_belief_dual_hand_cfv_checkpoint",
                "model_state": model.state_dict(),
                "hidden_dim": 8,
                "head_mode": "separate",
                "belief_bottleneck_dim": 4,
                "public_mean": np.zeros((1, N_FEATURES), dtype=np.float32),
                "public_std": np.ones((1, N_FEATURES), dtype=np.float32),
                "belief_mean": np.zeros((1, bvp.BELIEF_DIM), dtype=np.float32),
                "belief_std": np.ones((1, bvp.BELIEF_DIM), dtype=np.float32),
                "target_mean": target_mean,
                "target_std": 0.0,
            },
            checkpoint,
        )
        checkpoints.append(checkpoint)

    pred = predict_public_belief_dual_hand_cfv_ensemble(
        checkpoints,
        np.zeros((1, N_FEATURES), dtype=np.float32),
        np.zeros((1, bvp.BELIEF_DIM), dtype=np.float32),
        np.ones((1, bvp.N_HANDS), dtype=np.float32),
        np.ones((1, bvp.N_HANDS), dtype=np.float32),
        device="cpu",
        batch_size=512,
    )

    assert pred.shape == (2, 1, bvp.N_HANDS)
    assert np.allclose(pred, 2.0)


def test_dual_cfv_vectorized_prediction_matches_pairwise():
    torch.manual_seed(3)
    model = _DualHandCFVProbeNet(
        8,
        use_belief=True,
        head_mode="separate",
        belief_bottleneck_dim=4,
        card_encoder="deepset",
    )
    payload = {
        "public_mean": np.zeros((1, N_FEATURES), dtype=np.float32),
        "public_std": np.ones((1, N_FEATURES), dtype=np.float32),
        "belief_mean": np.zeros((1, bvp.BELIEF_DIM), dtype=np.float32),
        "belief_std": np.ones((1, bvp.BELIEF_DIM), dtype=np.float32),
        "target_mean": 0.25,
        "target_std": 1.5,
    }
    features = np.zeros((2, N_FEATURES), dtype=np.float32)
    features[:, 52:57] = 1.0
    belief = np.zeros((2, bvp.BELIEF_DIM), dtype=np.float32)
    belief[:, :10] = 0.1
    hero_masks = np.zeros((2, bvp.N_HANDS), dtype=np.float32)
    villain_masks = np.zeros((2, bvp.N_HANDS), dtype=np.float32)
    hero_masks[:, :5] = 1.0
    villain_masks[:, 5:10] = 1.0

    pairwise = predict_public_belief_dual_hand_cfv_model(
        model,
        payload,
        features,
        belief,
        hero_masks,
        villain_masks,
        device="cpu",
        batch_size=7,
    )
    vectorized = predict_public_belief_dual_hand_cfv_model_vectorized(
        model,
        payload,
        features,
        belief,
        hero_masks,
        villain_masks,
        device="cpu",
        state_batch_size=1,
        hand_batch_size=4,
    )

    np.testing.assert_allclose(vectorized, pairwise, rtol=1e-5, atol=1e-5)


def test_dual_cfv_zero_sum_projection_removes_range_weighted_residual():
    pred = np.zeros((2, 2, bvp.N_HANDS), dtype=np.float32)
    pred[0, 0, 0] = 3.0
    pred[1, 0, 1] = 1.0
    pred[0, 1, 2] = 5.0
    belief = np.zeros((2, bvp.BELIEF_DIM), dtype=np.float32)
    belief[0, 0] = 1.0
    belief[0, bvp.N_HANDS + 1] = 1.0
    masks = np.zeros((2, bvp.N_HANDS), dtype=np.float32)
    masks[0, 0] = 1.0
    masks[0, 1] = 1.0

    projected = project_dual_cfv_zero_sum(pred, belief, masks, masks)

    assert projected[0, 0, 0] == 1.0
    assert projected[1, 0, 1] == -1.0
    assert projected[0, 0, 2] == 0.0
    assert projected[0, 1, 2] == 5.0
    residual = projected[0, 0, 0] + projected[1, 0, 1]
    assert residual == 0.0


def test_dual_cfv_cache_error_attribution_groups_worst_rows():
    features = np.zeros((2, 3), dtype=np.float32)
    belief = np.zeros((2, bvp.BELIEF_DIM), dtype=np.float32)
    belief[:, 0] = 1.0
    belief[:, bvp.N_HANDS] = 1.0
    hero_values = np.zeros((2, bvp.N_HANDS), dtype=np.float32)
    villain_values = np.zeros((2, bvp.N_HANDS), dtype=np.float32)
    hero_values[0, 0] = 1.0
    villain_values[0, 0] = -1.0
    hero_values[1, 0] = 3.0
    villain_values[1, 0] = -3.0
    masks = np.zeros((2, bvp.N_HANDS), dtype=np.float32)
    masks[:, 0] = 1.0
    dataset = DualCFVDataset(
        features=features,
        belief=belief,
        hero_values=hero_values,
        villain_values=villain_values,
        hero_masks=masks,
        villain_masks=masks,
        labels=("low", "high"),
    )
    pred = np.zeros((2, 2, bvp.N_HANDS), dtype=np.float32)
    records = [
        {"leaf_action_str": "ck/kk/"},
        {"leaf_action_str": "ck/b200c/"},
    ]

    groups = group_error_records(
        pred,
        dataset,
        records,
        group_by=("leaf_action_str",),
        top_k=1,
    )

    assert [group["key"] for group in groups] == ["ck/b200c/", "ck/kk/"]
    assert groups[0]["model"]["mae"] == 3.0
    assert groups[0]["zero_baseline"]["mae"] == 3.0
    assert groups[0]["worst_states"][0]["label"] == "high"


def test_dual_cfv_cache_metadata_merge_keeps_solver_fields():
    merged = merge_record_metadata(
        [{"label": "leaf-a", "solver_latency_ms": 12.0}],
        [{"label": "leaf-a", "leaf_action_str": "ck/b200c/", "solver_latency_ms": 99.0}],
    )

    assert merged == [
        {
            "label": "leaf-a",
            "leaf_action_shape": "ck/bc/",
            "leaf_action_str": "ck/b200c/",
            "leaf_bet_count": 1,
            "leaf_parse_ok": True,
            "leaf_last_bet_size": 0.0,
            "leaf_street_last_bet_to": 0.0,
            "leaf_total_last_bet_to": 300.0,
            "solver_latency_ms": 12.0,
        }
    ]


def test_dual_cfv_cache_leaf_record_enrichment_strips_bet_amounts():
    enriched = enrich_leaf_record({"leaf_action_str": "ck/b200c/b150b525c/"})

    assert enriched["leaf_action_shape"] == "ck/bc/bbc/"
    assert enriched["leaf_bet_count"] == 3
    assert enriched["leaf_parse_ok"] is True


def test_joint_pbs_continuation_builder_requires_feature_alignment():
    features = np.zeros((2, N_FEATURES), dtype=np.float32)
    legal_masks = np.ones((2, N_ACTIONS), dtype=np.float32)
    target_probs = np.zeros((2, N_ACTIONS), dtype=np.float32)
    target_probs[:, 1] = 1.0
    policy = PolicyTargetBuffer(features, legal_masks, target_probs)
    belief = np.zeros((2, bvp.BELIEF_DIM), dtype=np.float32)
    values = np.zeros((2, bvp.N_HANDS), dtype=np.float32)
    masks = np.zeros((2, bvp.N_HANDS), dtype=np.float32)
    masks[:, :3] = 1.0
    dual = DualCFVDataset(
        features=features.copy(),
        belief=belief,
        hero_values=values,
        villain_values=values,
        hero_masks=masks,
        villain_masks=masks,
        labels=("a", "b"),
    )

    payload, metadata = build_joint_payload(
        policy,
        dual,
        labels=dual.labels,
        feature_atol=1e-6,
    )

    assert payload["target_probs"].shape == (2, N_ACTIONS)
    assert payload["hero_values"].shape == (2, bvp.N_HANDS)
    assert metadata["n_states"] == 2
    assert metadata["label_count"] == 12
    assert metadata["policy_target_mean_entropy"] == 0.0

    misaligned = DualCFVDataset(
        features=features + 1.0,
        belief=belief,
        hero_values=values,
        villain_values=values,
        hero_masks=masks,
        villain_masks=masks,
        labels=("a", "b"),
    )
    with pytest.raises(ValueError, match="misaligned"):
        build_joint_payload(policy, misaligned, labels=misaligned.labels, feature_atol=1e-6)


def test_public_belief_value_probe_emits_metrics(tmp_path, monkeypatch):
    checkpoint = tmp_path / "range.pt"
    _write_checkpoint(checkpoint)
    targets, cases = _write_targets_and_cases(tmp_path)

    def fake_value_target(case, **kwargs):
        return 0.125, {
            "label": case.label,
            "street": 3,
            "value_chips": 2500.0,
            "value_scaled": 0.125,
            "solver_latency_ms": 0.0,
            "solver_n_hands": 1,
            "solver_full_n_hands": 1,
        }

    monkeypatch.setattr(bvp, "_case_value_target", fake_value_target)
    metrics = bvp.run_public_belief_value_probe(
        train_targets_npz=targets,
        train_cases_json=cases,
        holdout_targets_npz=targets,
        holdout_cases_json=cases,
        range_checkpoint=checkpoint,
        device="cpu",
        hidden_dim=8,
        epochs=2,
        seed=0,
    )

    assert metrics["mode"] == "public_belief_value_probe"
    assert metrics["feature_dim"] == N_FEATURES
    assert metrics["belief_dim"] == bvp.BELIEF_DIM
    assert metrics["train_size"] == 1
    assert metrics["holdout_size"] == 1
    assert "base_holdout" in metrics
    assert "belief_holdout" in metrics
    assert metrics["holdout_value_records"][0]["value_scaled"] == 0.125


def test_public_belief_value_probe_reuses_value_cache(tmp_path, monkeypatch):
    checkpoint = tmp_path / "range.pt"
    _write_checkpoint(checkpoint)
    targets, cases = _write_targets_and_cases(tmp_path)
    train_cache = tmp_path / "train_value_cache.npz"
    holdout_cache = tmp_path / "holdout_value_cache.npz"

    def fake_value_target(case, **kwargs):
        return 0.125, {
            "label": case.label,
            "street": 3,
            "value_chips": 2500.0,
            "value_scaled": 0.125,
            "solver_latency_ms": 0.0,
            "solver_n_hands": 1,
            "solver_full_n_hands": 1,
        }

    monkeypatch.setattr(bvp, "_case_value_target", fake_value_target)
    first = bvp.run_public_belief_value_probe(
        train_targets_npz=targets,
        train_cases_json=cases,
        holdout_targets_npz=targets,
        holdout_cases_json=cases,
        range_checkpoint=checkpoint,
        device="cpu",
        hidden_dim=8,
        epochs=1,
        seed=0,
        train_value_cache=train_cache,
        holdout_value_cache=holdout_cache,
    )
    assert train_cache.exists()
    assert holdout_cache.exists()
    assert first["train_loaded_from_cache"] is False
    assert first["holdout_loaded_from_cache"] is False

    def fail_value_target(case, **kwargs):
        raise AssertionError("value labels should have been loaded from cache")

    monkeypatch.setattr(bvp, "_case_value_target", fail_value_target)
    second = bvp.run_public_belief_value_probe(
        train_targets_npz=targets,
        train_cases_json=cases,
        holdout_targets_npz=targets,
        holdout_cases_json=cases,
        range_checkpoint=checkpoint,
        device="cpu",
        hidden_dim=8,
        epochs=1,
        seed=1,
        train_value_cache=train_cache,
        holdout_value_cache=holdout_cache,
    )

    assert second["train_loaded_from_cache"] is True
    assert second["holdout_loaded_from_cache"] is True
    assert second["holdout_value_records"][0]["value_scaled"] == 0.125


def test_public_belief_cfv_probe_reuses_cache(tmp_path, monkeypatch):
    checkpoint = tmp_path / "range.pt"
    _write_checkpoint(checkpoint)
    targets, cases = _write_targets_and_cases(tmp_path)
    train_cache = tmp_path / "train_cfv_cache.npz"
    holdout_cache = tmp_path / "holdout_cfv_cache.npz"

    def fake_cfv_target(case, **kwargs):
        values = np.zeros(bvp.N_HANDS, dtype=np.float32)
        mask = np.zeros(bvp.N_HANDS, dtype=np.float32)
        values[:4] = 0.25
        mask[:4] = 1.0
        return values, mask, {
            "label": case.label,
            "street": 3,
            "value_mean": 0.25,
            "value_std": 0.0,
            "value_mask_count": 4,
            "solver_latency_ms": 0.0,
            "solver_n_hands": 4,
            "solver_full_n_hands": 4,
        }

    monkeypatch.setattr(bvp, "_case_cfv_target", fake_cfv_target)
    first = bvp.run_public_belief_cfv_probe(
        train_targets_npz=targets,
        train_cases_json=cases,
        holdout_targets_npz=targets,
        holdout_cases_json=cases,
        range_checkpoint=checkpoint,
        device="cpu",
        hidden_dim=8,
        epochs=1,
        seed=0,
        train_cfv_cache=train_cache,
        holdout_cfv_cache=holdout_cache,
    )
    assert train_cache.exists()
    assert holdout_cache.exists()
    assert first["train_loaded_from_cache"] is False
    assert first["target_dim"] == bvp.N_HANDS
    assert first["train_mask_count"] == 4

    def fail_cfv_target(case, **kwargs):
        raise AssertionError("CFV labels should have been loaded from cache")

    monkeypatch.setattr(bvp, "_case_cfv_target", fail_cfv_target)
    second = bvp.run_public_belief_cfv_probe(
        train_targets_npz=targets,
        train_cases_json=cases,
        holdout_targets_npz=targets,
        holdout_cases_json=cases,
        range_checkpoint=checkpoint,
        device="cpu",
        hidden_dim=8,
        epochs=1,
        seed=1,
        train_cfv_cache=train_cache,
        holdout_cfv_cache=holdout_cache,
    )

    assert second["train_loaded_from_cache"] is True
    assert second["holdout_loaded_from_cache"] is True
    assert second["holdout_cfv_records"][0]["value_mask_count"] == 4


def test_public_belief_hand_cfv_probe_uses_cached_pair_labels(tmp_path, monkeypatch):
    checkpoint = tmp_path / "range.pt"
    _write_checkpoint(checkpoint)
    targets, cases = _write_targets_and_cases(tmp_path)
    train_cache = tmp_path / "train_cfv_cache.npz"
    holdout_cache = tmp_path / "holdout_cfv_cache.npz"

    def fake_cfv_target(case, **kwargs):
        values = np.zeros(bvp.N_HANDS, dtype=np.float32)
        mask = np.zeros(bvp.N_HANDS, dtype=np.float32)
        values[:4] = np.linspace(0.1, 0.4, 4, dtype=np.float32)
        mask[:4] = 1.0
        return values, mask, {
            "label": case.label,
            "street": 3,
            "value_mean": 0.25,
            "value_std": 0.111803,
            "value_mask_count": 4,
            "solver_latency_ms": 0.0,
            "solver_n_hands": 4,
            "solver_full_n_hands": 4,
        }

    monkeypatch.setattr(bvp, "_case_cfv_target", fake_cfv_target)
    metrics = bvp.run_public_belief_hand_cfv_probe(
        train_targets_npz=targets,
        train_cases_json=cases,
        holdout_targets_npz=targets,
        holdout_cases_json=cases,
        range_checkpoint=checkpoint,
        device="cpu",
        hidden_dim=8,
        epochs=1,
        batch_size=2,
        seed=0,
        train_cfv_cache=train_cache,
        holdout_cfv_cache=holdout_cache,
    )

    assert metrics["mode"] == "public_belief_hand_cfv_probe"
    assert metrics["hand_feature_dim"] == 52
    assert metrics["train_mask_count"] == 4
    assert metrics["holdout_mask_count"] == 4
    assert metrics["train_loaded_from_cache"] is False

    def fail_cfv_target(case, **kwargs):
        raise AssertionError("CFV labels should have been loaded from cache")

    monkeypatch.setattr(bvp, "_case_cfv_target", fail_cfv_target)
    cached = bvp.run_public_belief_hand_cfv_probe(
        train_targets_npz=targets,
        train_cases_json=cases,
        holdout_targets_npz=targets,
        holdout_cases_json=cases,
        range_checkpoint=checkpoint,
        device="cpu",
        hidden_dim=8,
        epochs=1,
        batch_size=2,
        seed=1,
        train_cfv_cache=train_cache,
        holdout_cfv_cache=holdout_cache,
    )

    assert cached["train_loaded_from_cache"] is True
    assert cached["holdout_loaded_from_cache"] is True


def test_public_belief_hand_cfv_checkpoint_round_trips_cached_model(tmp_path, monkeypatch):
    checkpoint = tmp_path / "range.pt"
    _write_checkpoint(checkpoint)
    targets, cases = _write_targets_and_cases(tmp_path)
    train_cache = tmp_path / "train_cfv_cache.npz"
    holdout_cache = tmp_path / "holdout_cfv_cache.npz"
    model_path = tmp_path / "hand_cfv.pt"

    def fake_cfv_target(case, **kwargs):
        values = np.zeros(bvp.N_HANDS, dtype=np.float32)
        mask = np.zeros(bvp.N_HANDS, dtype=np.float32)
        values[:4] = np.linspace(0.1, 0.4, 4, dtype=np.float32)
        mask[:4] = 1.0
        return values, mask, {
            "label": case.label,
            "street": 3,
            "value_mean": 0.25,
            "value_std": 0.111803,
            "value_mask_count": 4,
            "solver_latency_ms": 0.0,
            "solver_n_hands": 4,
            "solver_full_n_hands": 4,
        }

    monkeypatch.setattr(bvp, "_case_cfv_target", fake_cfv_target)
    metrics = bvp.train_public_belief_hand_cfv_checkpoint(
        train_targets_npz=targets,
        train_cases_json=cases,
        holdout_targets_npz=targets,
        holdout_cases_json=cases,
        range_checkpoint=checkpoint,
        output_checkpoint=model_path,
        device="cpu",
        hidden_dim=8,
        epochs=1,
        batch_size=2,
        seed=0,
        train_cfv_cache=train_cache,
        holdout_cfv_cache=holdout_cache,
    )

    assert model_path.exists()
    assert metrics["mode"] == "public_belief_hand_cfv_checkpoint_train"
    assert metrics["belief_holdout"]["mae"] >= 0.0

    cache = np.load(holdout_cache, allow_pickle=False)
    pred = bvp.predict_public_belief_hand_cfv_checkpoint(
        model_path,
        cache["features"],
        cache["belief"],
        cache["value_masks"],
        device="cpu",
        batch_size=2,
    )

    assert pred.shape == cache["values"].shape
    assert np.all(pred[cache["value_masks"] <= 0] == 0.0)
