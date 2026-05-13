import json
import subprocess
import sys
from pathlib import Path

import torch

from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES
from poker_ai.research.evaluation import (
    _strategies_from_network,
    assert_strategy_source_supported,
    compare_checkpoint_metrics,
    evaluate_value_nets_head_to_head,
    evaluate_value_net_vs_random,
    load_value_network_checkpoint,
)


class _PolicyOnlyProbeNet(torch.nn.Module):
    def forward(self, features):
        raise AssertionError("policy-head evaluation should not call forward()")

    def forward_with_policy(self, features):
        advantages = torch.zeros((features.shape[0], N_ACTIONS), dtype=torch.float32)
        logits = torch.zeros_like(advantages)
        logits[:, 1] = 10.0
        return advantages, logits


class _CoveredPolicyProbeNet(torch.nn.Module):
    def forward(self, features):
        advantages = torch.zeros((features.shape[0], N_ACTIONS), dtype=torch.float32)
        advantages[:, 8] = 10.0
        return advantages

    def forward_with_policy(self, features):
        advantages = self.forward(features)
        logits = torch.zeros_like(advantages)
        logits[:, 1] = 10.0
        return advantages, logits


class _AveragePolicyProbeNet(torch.nn.Module):
    def forward(self, features):
        logits = torch.zeros((features.shape[0], N_ACTIONS), dtype=torch.float32)
        logits[:, 1] = 10.0
        return logits


class _AveragePolicyOnlyProbeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.average_policy_net = _AveragePolicyProbeNet()

    def forward(self, features):
        raise AssertionError("average-policy evaluation should not call value forward()")

    def forward_with_policy(self, features):
        raise AssertionError("average-policy evaluation should not call policy head")


def test_load_value_network_checkpoint_accepts_legacy_sequential_keys(tmp_path):
    checkpoint = {
        "iteration": 7,
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
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)

    loaded = load_value_network_checkpoint(path, torch.device("cpu"))

    assert loaded.metadata["iteration"] == 7
    assert loaded.metadata["n_players"] == 2
    assert loaded.metadata["initial_chips"] == 20000
    assert loaded.metadata["has_policy_head"] is False
    assert loaded.metadata["uses_betting_history"] is False
    assert loaded.value_net.hidden_dim == 16


def test_load_value_network_checkpoint_loads_average_policy_net(tmp_path):
    from poker_ai.deep_cfr.networks import PolicyNetwork, ValueNetwork

    value_net = ValueNetwork(N_FEATURES, 16, N_ACTIONS, n_layers=1)
    average_policy_net = PolicyNetwork(N_FEATURES, 16, N_ACTIONS, n_layers=1)
    checkpoint = {
        "iteration": 7,
        "n_players": 2,
        "hidden_dim": 16,
        "n_layers": 1,
        "initial_chips": 20000,
        "uses_betting_history": True,
        "value_net": value_net.state_dict(),
        "average_policy_net": average_policy_net.state_dict(),
    }
    path = tmp_path / "average_policy.pt"
    torch.save(checkpoint, path)

    loaded = load_value_network_checkpoint(path, torch.device("cpu"))

    assert loaded.metadata["has_average_policy_net"] is True
    assert loaded.metadata["uses_betting_history"] is True
    assert loaded.average_policy_net is not None
    assert getattr(loaded.value_net, "average_policy_net") is loaded.average_policy_net


def test_load_value_network_checkpoint_exposes_policy_calibration_metadata(tmp_path):
    from poker_ai.deep_cfr.networks import ValueNetwork

    value_net = ValueNetwork(N_FEATURES, 16, N_ACTIONS, n_layers=1)
    checkpoint = {
        "iteration": 7,
        "n_players": 2,
        "hidden_dim": 16,
        "n_layers": 1,
        "initial_chips": 20000,
        "value_net": value_net.state_dict(),
        "policy_calibration": {"target_streets": [2, 3]},
    }
    path = tmp_path / "policy_calibrated.pt"
    torch.save(checkpoint, path)

    loaded = load_value_network_checkpoint(path, torch.device("cpu"))

    assert loaded.metadata["policy_calibration"]["target_streets"] == [2, 3]
    assert loaded.value_net.policy_calibration["target_streets"] == [2, 3]


def test_strategy_source_guard_rejects_legacy_policy_head_checkpoint(tmp_path):
    checkpoint = {
        "iteration": 7,
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
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)
    loaded = load_value_network_checkpoint(path, torch.device("cpu"))

    try:
        assert_strategy_source_supported(loaded, "policy-head")
    except RuntimeError as exc:
        assert "does not contain a trained policy head" in str(exc)
    else:
        raise AssertionError("legacy checkpoint should reject policy-head evaluation")


def test_strategy_source_guard_rejects_missing_average_policy_checkpoint(tmp_path):
    checkpoint = {
        "iteration": 7,
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
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)
    loaded = load_value_network_checkpoint(path, torch.device("cpu"))

    try:
        assert_strategy_source_supported(loaded, "average-policy")
    except RuntimeError as exc:
        assert "does not contain a trained average policy net" in str(exc)
    else:
        raise AssertionError("legacy checkpoint should reject average-policy evaluation")


def test_strategy_source_guard_rejects_policy_head_covered_without_metadata(tmp_path):
    from poker_ai.deep_cfr.networks import ValueNetwork

    value_net = ValueNetwork(N_FEATURES, 16, N_ACTIONS, n_layers=1)
    checkpoint = {
        "iteration": 7,
        "n_players": 2,
        "hidden_dim": 16,
        "n_layers": 1,
        "initial_chips": 20000,
        "value_net": value_net.state_dict(),
    }
    path = tmp_path / "policy.pt"
    torch.save(checkpoint, path)
    loaded = load_value_network_checkpoint(path, torch.device("cpu"))

    try:
        assert_strategy_source_supported(loaded, "policy-head-covered")
    except RuntimeError as exc:
        assert "target_streets" in str(exc)
    else:
        raise AssertionError("covered policy source should require coverage metadata")


def test_policy_head_covered_routes_by_feature_street():
    features = torch.zeros((2, N_FEATURES), dtype=torch.float32).numpy()
    features[0, 104] = 1.0  # preflop: unsupported, use regret fallback
    features[1, 106] = 1.0  # turn: covered, use policy logits
    mask = torch.zeros((N_ACTIONS,), dtype=torch.float32).numpy()
    mask[[1, 8]] = 1.0

    strategies = _strategies_from_network(
        _CoveredPolicyProbeNet(),
        features,
        [mask, mask],
        torch.device("cpu"),
        strategy_source="policy-head-covered",
        checkpoint_metadata={"policy_calibration": {"target_streets": [2, 3]}},
    )

    assert strategies[0][8] > 0.99
    assert strategies[1][1] > 0.99


def test_evaluate_value_net_vs_random_returns_ci_metrics():
    checkpoint = {
        "iteration": 0,
        "n_players": 2,
        "initial_chips": 1000,
    }
    from poker_ai.deep_cfr.networks import ValueNetwork

    value_net = ValueNetwork(N_FEATURES, 32, N_ACTIONS, n_layers=1)
    metrics = evaluate_value_net_vs_random(
        value_net,
        torch.device("cpu"),
        n_games=8,
        n_players=2,
        initial_chips=1000,
        seed=123,
        checkpoint_metadata=checkpoint,
    )

    assert metrics["n_games"] == 8
    assert metrics["n_players"] == 2
    assert metrics["initial_chips"] == 1000
    assert isinstance(metrics["avg_chips_per_hand"], float)
    assert isinstance(metrics["ci95_chips_per_hand"], float)
    assert metrics["checkpoint_iteration"] == 0


def test_evaluate_value_net_vs_random_can_use_policy_head_source():
    checkpoint = {
        "iteration": 0,
        "n_players": 2,
        "initial_chips": 1000,
    }

    metrics = evaluate_value_net_vs_random(
        _PolicyOnlyProbeNet(),
        torch.device("cpu"),
        n_games=4,
        n_players=2,
        initial_chips=1000,
        seed=123,
        checkpoint_metadata=checkpoint,
        strategy_source="policy-head",
    )

    assert metrics["passed"] is True
    assert metrics["strategy_source"] == "policy-head"


def test_evaluate_value_net_vs_random_can_use_average_policy_source():
    checkpoint = {
        "iteration": 0,
        "n_players": 2,
        "initial_chips": 1000,
    }

    metrics = evaluate_value_net_vs_random(
        _AveragePolicyOnlyProbeNet(),
        torch.device("cpu"),
        n_games=4,
        n_players=2,
        initial_chips=1000,
        seed=123,
        checkpoint_metadata=checkpoint,
        strategy_source="average-policy",
    )

    assert metrics["passed"] is True
    assert metrics["strategy_source"] == "average-policy"


def test_evaluate_value_nets_head_to_head_self_compare_is_zero_with_swapped_seats():
    from poker_ai.deep_cfr.networks import ValueNetwork

    value_net = ValueNetwork(N_FEATURES, 32, N_ACTIONS, n_layers=1)
    metadata = {
        "checkpoint": "self.pt",
        "checkpoint_iteration": 0,
        "hidden_dim": 32,
        "n_layers": 1,
    }

    metrics = evaluate_value_nets_head_to_head(
        value_net,
        value_net,
        torch.device("cpu"),
        n_games=6,
        initial_chips=1000,
        seed=123,
        candidate_metadata=metadata,
        baseline_metadata=metadata,
    )

    assert metrics["passed"] is True
    assert metrics["mode"] == "duplicate_swapped_head_to_head"
    assert metrics["avg_chips_per_hand"] == 0.0
    assert metrics["paired_delta_lower95_chips_per_hand"] == 0.0
    assert metrics["promotable"] is False


def test_evaluate_value_nets_head_to_head_can_use_policy_head_source():
    metadata = {
        "checkpoint": "self.pt",
        "checkpoint_iteration": 0,
        "hidden_dim": 32,
        "n_layers": 1,
    }

    metrics = evaluate_value_nets_head_to_head(
        _PolicyOnlyProbeNet(),
        _PolicyOnlyProbeNet(),
        torch.device("cpu"),
        n_games=4,
        initial_chips=1000,
        seed=123,
        candidate_metadata=metadata,
        baseline_metadata=metadata,
        strategy_source="policy-head",
    )

    assert metrics["passed"] is True
    assert metrics["strategy_source"] == "policy-head"


def test_evaluate_value_nets_head_to_head_can_use_average_policy_source():
    metadata = {
        "checkpoint": "self.pt",
        "checkpoint_iteration": 0,
        "hidden_dim": 32,
        "n_layers": 1,
    }

    metrics = evaluate_value_nets_head_to_head(
        _AveragePolicyOnlyProbeNet(),
        _AveragePolicyOnlyProbeNet(),
        torch.device("cpu"),
        n_games=4,
        initial_chips=1000,
        seed=123,
        candidate_metadata=metadata,
        baseline_metadata=metadata,
        strategy_source="average-policy",
    )

    assert metrics["passed"] is True
    assert metrics["strategy_source"] == "average-policy"


def test_evaluate_value_nets_head_to_head_can_mix_strategy_sources():
    metadata = {
        "checkpoint": "self.pt",
        "checkpoint_iteration": 0,
        "hidden_dim": 32,
        "n_layers": 1,
    }
    from poker_ai.deep_cfr.networks import ValueNetwork

    baseline = ValueNetwork(N_FEATURES, 32, N_ACTIONS, n_layers=1)

    metrics = evaluate_value_nets_head_to_head(
        _AveragePolicyOnlyProbeNet(),
        baseline,
        torch.device("cpu"),
        n_games=4,
        initial_chips=1000,
        seed=123,
        candidate_metadata=metadata,
        baseline_metadata=metadata,
        strategy_source="regret",
        candidate_strategy_source="average-policy",
        baseline_strategy_source="regret",
    )

    assert metrics["passed"] is True
    assert metrics["strategy_source"] == "mixed"
    assert metrics["candidate_strategy_source"] == "average-policy"
    assert metrics["baseline_strategy_source"] == "regret"


def test_eval_cli_emits_json_for_legacy_checkpoint(tmp_path):
    checkpoint = {
        "iteration": 3,
        "n_players": 2,
        "hidden_dim": 16,
        "n_layers": 1,
        "initial_chips": 1000,
        "value_net": {
            "net.0.weight": torch.randn(16, N_FEATURES),
            "net.0.bias": torch.randn(16),
            "net.2.weight": torch.randn(N_ACTIONS, 16),
            "net.2.bias": torch.randn(N_ACTIONS),
        },
    }
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(checkpoint, checkpoint_path)
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch_eval.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint_path),
            "--n-games",
            "6",
            "--device",
            "cpu",
            "--seed",
            "7",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    metrics = json.loads(result.stdout)
    assert metrics["checkpoint_iteration"] == 3
    assert metrics["n_games"] == 6
    assert metrics["passed"] is True


def test_eval_cli_aggregates_multiple_seeds(tmp_path):
    checkpoint = {
        "iteration": 4,
        "n_players": 2,
        "hidden_dim": 16,
        "n_layers": 1,
        "initial_chips": 1000,
        "value_net": {
            "net.0.weight": torch.randn(16, N_FEATURES),
            "net.0.bias": torch.randn(16),
            "net.2.weight": torch.randn(N_ACTIONS, 16),
            "net.2.bias": torch.randn(N_ACTIONS),
        },
    }
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(checkpoint, checkpoint_path)
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch_eval.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint_path),
            "--n-games",
            "4",
            "--device",
            "cpu",
            "--seeds",
            "11,12",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    metrics = json.loads(result.stdout)
    assert metrics["passed"] is True
    assert metrics["n_runs"] == 2
    assert [run["seed"] for run in metrics["runs"]] == [11, 12]
    assert isinstance(metrics["avg_chips_per_hand"], float)


def test_eval_cli_accepts_policy_head_strategy_source(tmp_path):
    from poker_ai.deep_cfr.networks import ValueNetwork

    value_net = ValueNetwork(N_FEATURES, 16, N_ACTIONS, n_layers=1)
    checkpoint = {
        "iteration": 6,
        "n_players": 2,
        "hidden_dim": 16,
        "n_layers": 1,
        "initial_chips": 1000,
        "value_net": value_net.state_dict(),
    }
    checkpoint_path = tmp_path / "policy.pt"
    torch.save(checkpoint, checkpoint_path)
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch_eval.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint_path),
            "--n-games",
            "4",
            "--device",
            "cpu",
            "--seed",
            "7",
            "--strategy-source",
            "policy-head",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    metrics = json.loads(result.stdout)
    assert metrics["passed"] is True
    assert metrics["strategy_source"] == "policy-head"


def test_compare_checkpoint_metrics_keeps_local_result_non_promotable():
    candidate = {
        "passed": True,
        "avg_chips_per_hand": 15.0,
        "runs": [
            {"seed": 1, "avg_chips_per_hand": 10.0},
            {"seed": 2, "avg_chips_per_hand": 20.0},
        ],
        "checkpoint": "candidate.pt",
    }
    baseline = {
        "passed": True,
        "avg_chips_per_hand": 10.0,
        "runs": [
            {"seed": 1, "avg_chips_per_hand": 8.0},
            {"seed": 2, "avg_chips_per_hand": 12.0},
        ],
        "checkpoint": "baseline.pt",
    }

    metrics = compare_checkpoint_metrics(candidate, baseline)

    assert metrics["passed"] is True
    assert metrics["delta_avg_chips_per_hand"] == 5.0
    assert metrics["paired_delta_chips_per_hand"] == [2.0, 8.0]
    assert metrics["promotable"] is False
    assert "local_random_comparison_is_not_strategy_quality" in metrics["promotion_blockers"]


def test_eval_cli_compares_candidate_to_baseline_checkpoint(tmp_path):
    checkpoint = {
        "iteration": 5,
        "n_players": 2,
        "hidden_dim": 16,
        "n_layers": 1,
        "initial_chips": 1000,
        "value_net": {
            "net.0.weight": torch.randn(16, N_FEATURES),
            "net.0.bias": torch.randn(16),
            "net.2.weight": torch.randn(N_ACTIONS, 16),
            "net.2.bias": torch.randn(N_ACTIONS),
        },
    }
    candidate_path = tmp_path / "candidate.pt"
    baseline_path = tmp_path / "baseline.pt"
    torch.save(checkpoint, candidate_path)
    torch.save(checkpoint, baseline_path)
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch_eval.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(candidate_path),
            "--baseline-checkpoint",
            str(baseline_path),
            "--n-games",
            "4",
            "--device",
            "cpu",
            "--seeds",
            "21,22",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    metrics = json.loads(result.stdout)
    assert metrics["passed"] is True
    assert metrics["candidate"]["checkpoint"] == str(candidate_path)
    assert metrics["baseline"]["checkpoint"] == str(baseline_path)
    assert metrics["delta_avg_chips_per_hand"] == 0.0
    assert metrics["promotable"] is False


def test_eval_cli_runs_head_to_head_self_compare(tmp_path):
    checkpoint = {
        "iteration": 6,
        "n_players": 2,
        "hidden_dim": 16,
        "n_layers": 1,
        "initial_chips": 1000,
        "value_net": {
            "net.0.weight": torch.randn(16, N_FEATURES),
            "net.0.bias": torch.randn(16),
            "net.2.weight": torch.randn(N_ACTIONS, 16),
            "net.2.bias": torch.randn(N_ACTIONS),
        },
    }
    candidate_path = tmp_path / "candidate.pt"
    baseline_path = tmp_path / "baseline.pt"
    torch.save(checkpoint, candidate_path)
    torch.save(checkpoint, baseline_path)
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch_eval.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(candidate_path),
            "--baseline-checkpoint",
            str(baseline_path),
            "--head-to-head",
            "--n-games",
            "4",
            "--device",
            "cpu",
            "--seeds",
            "31,32",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    metrics = json.loads(result.stdout)
    assert metrics["passed"] is True
    assert metrics["mode"] == "duplicate_swapped_head_to_head"
    assert metrics["avg_chips_per_hand"] == 0.0
    assert metrics["promotable"] is False


def test_eval_cli_can_require_positive_head_to_head_lower95(tmp_path):
    checkpoint = {
        "iteration": 6,
        "n_players": 2,
        "hidden_dim": 16,
        "n_layers": 1,
        "initial_chips": 1000,
        "value_net": {
            "net.0.weight": torch.randn(16, N_FEATURES),
            "net.0.bias": torch.randn(16),
            "net.2.weight": torch.randn(N_ACTIONS, 16),
            "net.2.bias": torch.randn(N_ACTIONS),
        },
    }
    candidate_path = tmp_path / "candidate.pt"
    baseline_path = tmp_path / "baseline.pt"
    torch.save(checkpoint, candidate_path)
    torch.save(checkpoint, baseline_path)
    script = Path(__file__).resolve().parents[2] / "scripts" / "poker_autoresearch_eval.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(candidate_path),
            "--baseline-checkpoint",
            str(baseline_path),
            "--head-to-head",
            "--require-positive-lower95",
            "--n-games",
            "4",
            "--device",
            "cpu",
            "--seeds",
            "31,32",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "positive lower95" in result.stderr
    metrics = json.loads(result.stdout)
    assert metrics["passed"] is False
    assert metrics["positive_lower95_required"] is True
    assert metrics["paired_delta_lower95_chips_per_hand_across_seeds"] == 0.0
    assert "comparison_requires_positive_lower95" in metrics["promotion_blockers"]
