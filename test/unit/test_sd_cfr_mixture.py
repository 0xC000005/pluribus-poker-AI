import numpy as np
import torch

import poker_ai.research.sd_cfr_mixture as sd_cfr_mixture
from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.sd_cfr_mixture import (
    discover_checkpoint_paths,
    evaluate_checkpoint_mixture_across_seeds,
    evaluate_checkpoint_mixture_head_to_head,
    load_checkpoint_policy_set,
)


def _write_checkpoint(path, *, iteration: int):
    net = ValueNetwork(N_FEATURES, hidden_dim=16, output_dim=N_ACTIONS, n_layers=1)
    torch.save(
        {
            "value_net": net.state_dict(),
            "hidden_dim": 16,
            "n_layers": 1,
            "n_players": 2,
            "initial_chips": 1000,
            "iteration": iteration,
        },
        path,
    )


def test_discover_checkpoint_paths_accepts_absolute_globs(tmp_path):
    first = tmp_path / "mix_iter_1.pt"
    second = tmp_path / "mix_iter_3.pt"
    _write_checkpoint(first, iteration=1)
    _write_checkpoint(second, iteration=3)

    paths = discover_checkpoint_paths([str(tmp_path / "mix_iter_*.pt")])

    assert paths == [first, second]


def test_checkpoint_policy_set_weights_by_iteration(tmp_path):
    first = tmp_path / "candidate_iter_1.pt"
    second = tmp_path / "candidate_iter_3.pt"
    _write_checkpoint(first, iteration=1)
    _write_checkpoint(second, iteration=3)

    policy_set = load_checkpoint_policy_set([second, first], torch.device("cpu"))

    assert [meta["checkpoint_iteration"] for meta in policy_set.metadata] == [1, 3]
    np.testing.assert_allclose(policy_set.weights, np.array([0.25, 0.75]))


def test_checkpoint_mixture_head_to_head_returns_finite_metrics(tmp_path):
    first = tmp_path / "candidate_iter_1.pt"
    second = tmp_path / "candidate_iter_3.pt"
    baseline = tmp_path / "baseline.pt"
    _write_checkpoint(first, iteration=1)
    _write_checkpoint(second, iteration=3)
    _write_checkpoint(baseline, iteration=3)

    metrics = evaluate_checkpoint_mixture_head_to_head(
        [first, second],
        baseline,
        torch.device("cpu"),
        n_games=2,
        seed=20260512,
        initial_chips=1000,
    )

    assert metrics["passed"] is True
    assert metrics["mode"] == "sd_cfr_checkpoint_mixture_head_to_head"
    assert metrics["n_games"] == 4
    assert np.isfinite(metrics["avg_chips_per_hand"])


def test_checkpoint_mixture_across_seeds_requires_positive_lower95(monkeypatch):
    def fake_run(*args, **kwargs):
        seed = int(kwargs["seed"])
        avg_by_seed = {1: -10.0, 2: 90.0, 3: 10.0}
        return {
            "passed": True,
            "mode": "sd_cfr_checkpoint_mixture_head_to_head",
            "n_games": 200,
            "n_players": 2,
            "initial_chips": 1000,
            "seed": seed,
            "strategy_source": "regret",
            "avg_chips_per_hand": avg_by_seed[seed],
            "paired_delta_lower95_chips_per_hand": avg_by_seed[seed] - 1.0,
            "promotable": False,
            "promotion_blockers": ["local_head_to_head_requires_slumbot_confirmation"],
        }

    monkeypatch.setattr(
        sd_cfr_mixture,
        "evaluate_checkpoint_mixture_head_to_head",
        fake_run,
    )

    metrics = evaluate_checkpoint_mixture_across_seeds(
        ["candidate.pt"],
        "baseline.pt",
        torch.device("cpu"),
        n_games=100,
        seeds=[1, 2, 3],
    )

    assert metrics["lower95_chips_per_hand_across_seeds"] < 0.0
    assert metrics["passed"] is False
    assert "local_across_seed_lower95_not_positive" in metrics["promotion_blockers"]
