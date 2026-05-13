import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
