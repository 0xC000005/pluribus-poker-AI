import json
import sys
import types

import numpy as np
import pytest
import torch

from poker_ai.games.full_deck.state import new_game
from poker_ai.research.mixed_policy_h2h import (
    PolicyAdapter,
    SUPPORTED_POLICY_KINDS,
    evaluate_loaded_policies_head_to_head,
    load_policy_adapter,
    _resolve_rllib_module_checkpoint,
)
from poker_ai.research.native_nfsp import get_legal_mask


def _first_legal_policy(_features, legal_mask, _device):
    probs = np.zeros_like(legal_mask, dtype=np.float32)
    probs[int(np.flatnonzero(legal_mask)[0])] = 1.0
    return probs


def test_evaluate_loaded_policies_head_to_head_reports_native_contract():
    candidate = PolicyAdapter(
        kind="fake",
        checkpoint_path="candidate.pt",
        algorithm="fake_policy",
        action_probs_fn=_first_legal_policy,
    )
    baseline = PolicyAdapter(
        kind="fake",
        checkpoint_path="baseline.pt",
        algorithm="fake_policy",
        action_probs_fn=_first_legal_policy,
    )

    metrics = evaluate_loaded_policies_head_to_head(
        candidate,
        baseline,
        n_games=4,
        seed=13,
        initial_chips=100,
        max_steps_per_hand=16,
        device="cpu",
    )

    assert metrics["algorithm"] == "mixed_native_policy_h2h"
    assert metrics["environment"] == "poker_ai:full_deck_hu_nlhe"
    assert metrics["num_actions"] == 9
    assert metrics["candidate_checkpoint"] == "candidate.pt"
    assert metrics["baseline_checkpoint"] == "baseline.pt"
    assert metrics["n_games"] == 4
    assert np.isfinite(metrics["mean_candidate_payoff"])


def test_evaluate_loaded_policies_head_to_head_applies_lower95_gate():
    candidate = PolicyAdapter(
        kind="fake",
        checkpoint_path="candidate.pt",
        algorithm="fake_policy",
        action_probs_fn=_first_legal_policy,
    )
    baseline = PolicyAdapter(
        kind="fake",
        checkpoint_path="baseline.pt",
        algorithm="fake_policy",
        action_probs_fn=_first_legal_policy,
    )

    metrics = evaluate_loaded_policies_head_to_head(
        candidate,
        baseline,
        n_games=2,
        seed=14,
        initial_chips=100,
        max_steps_per_hand=8,
        device="cpu",
        min_lower95_candidate_payoff=999.0,
    )

    assert metrics["min_lower95_candidate_payoff"] == 999.0
    assert metrics["passed"] is False


def test_evaluate_loaded_policies_head_to_head_fast_canonical_matches_full_deck():
    candidate = PolicyAdapter(
        kind="fake",
        checkpoint_path="candidate.pt",
        algorithm="fake_policy",
        action_probs_fn=_first_legal_policy,
    )
    baseline = PolicyAdapter(
        kind="fake",
        checkpoint_path="baseline.pt",
        algorithm="fake_policy",
        action_probs_fn=_first_legal_policy,
    )

    full_deck = evaluate_loaded_policies_head_to_head(
        candidate,
        baseline,
        n_games=6,
        seed=20260796,
        initial_chips=100,
        max_steps_per_hand=16,
        device="cpu",
        eval_state_backend="full-deck",
    )
    fast_canonical = evaluate_loaded_policies_head_to_head(
        candidate,
        baseline,
        n_games=6,
        seed=20260796,
        initial_chips=100,
        max_steps_per_hand=16,
        device="cpu",
        eval_state_backend="fast-state-canonical-deal",
    )

    assert fast_canonical["eval_state_backend"] == "fast-state-canonical-deal"
    assert fast_canonical["mean_candidate_payoff"] == pytest.approx(
        full_deck["mean_candidate_payoff"]
    )
    assert fast_canonical["eval_steps"] == full_deck["eval_steps"]


def test_load_policy_adapter_supports_native_ppo_average_policy(monkeypatch):
    import poker_ai.research.native_ppo_policy as native_ppo

    fake_policy = object()

    monkeypatch.setattr(
        native_ppo,
        "_load_policy_network",
        lambda checkpoint_path, resolved_device, *, strategy_source="auto": (
            {
                "algorithm": "native_ppo_policy",
                "config": {
                    "initial_chips": 777,
                    "max_steps_per_hand": 33,
                    "fsp_average_policy": True,
                },
            },
            fake_policy,
            "flat",
        ),
    )
    monkeypatch.setattr(
        native_ppo,
        "_network_probs",
        lambda policy, features, legal_mask, resolved_device: np.array(
            [0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4],
            dtype=np.float32,
        ),
    )

    adapter = load_policy_adapter("native_ppo.pt", kind="native-ppo", device=torch.device("cpu"))
    legal_mask = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    probs = adapter.probs(np.zeros(126, dtype=np.float32), legal_mask, torch.device("cpu"))

    assert "native-ppo" in SUPPORTED_POLICY_KINDS
    assert adapter.kind == "native-ppo"
    assert adapter.algorithm == "native_ppo_policy"
    assert adapter.initial_chips == 777
    assert adapter.max_steps_per_hand == 33
    assert np.allclose(probs, np.array([0, 0.6, 0, 0, 0, 0, 0, 0, 0.4], dtype=np.float32))


def test_load_policy_adapter_supports_native_ppo_raw_sequence_stateful_action(monkeypatch):
    import poker_ai.research.native_ppo_policy as native_ppo

    fake_policy = object()
    captured_feature_modes = []

    monkeypatch.setattr(
        native_ppo,
        "_load_policy_network",
        lambda checkpoint_path, resolved_device, *, strategy_source="auto": (
            {
                "algorithm": "native_ppo_policy",
                "config": {
                    "initial_chips": 100,
                    "max_steps_per_hand": 16,
                    "feature_mode": "raw_sequence",
                    "fsp_average_policy": True,
                },
            },
            fake_policy,
            "raw_sequence",
        ),
    )

    def fake_policy_features(state, feature_mode):
        captured_feature_modes.append(feature_mode)
        return np.zeros(999, dtype=np.float32)

    monkeypatch.setattr(native_ppo, "_policy_feature_vector", fake_policy_features)
    monkeypatch.setattr(
        native_ppo,
        "_network_probs",
        lambda policy, features, legal_mask, resolved_device: legal_mask / legal_mask.sum(),
    )

    adapter = load_policy_adapter("native_ppo_raw.pt", kind="native-ppo", device=torch.device("cpu"))
    state = new_game(2, initial_chips=100)
    legal_mask = get_legal_mask(state).astype(np.float32)
    action = adapter.select_action(
        state=state,
        features=state.to_feature_vector().astype(np.float32),
        legal_mask=legal_mask,
        device=torch.device("cpu"),
        rng=np.random.default_rng(20260798),
    )

    assert adapter.kind == "native-ppo"
    assert captured_feature_modes == ["raw_sequence"]
    assert legal_mask[action] > 0


def test_load_policy_adapter_supports_tianshou_rainbow_one_hot(monkeypatch):
    pytest.importorskip("gymnasium")
    from scripts import run_tianshou_rainbow_native_control as rainbow_control

    fake_policy = object()

    monkeypatch.setattr(
        rainbow_control,
        "_load_rainbow_checkpoint_policy",
        lambda checkpoint_path, resolved_device: (
            {
                "algorithm": "tianshou_rainbow_dqn",
                "metrics": {"initial_chips": 777, "max_steps_per_hand": 33},
            },
            fake_policy,
        ),
    )
    monkeypatch.setattr(
        rainbow_control,
        "_rainbow_greedy_action",
        lambda policy, features, legal_mask: int(np.flatnonzero(legal_mask)[-1]),
    )

    adapter = load_policy_adapter("rainbow.pt", kind="tianshou-rainbow", device=torch.device("cpu"))
    legal_mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    probs = adapter.probs(np.zeros(126, dtype=np.float32), legal_mask, torch.device("cpu"))

    assert adapter.kind == "tianshou-rainbow"
    assert adapter.algorithm == "tianshou_rainbow_dqn"
    assert adapter.initial_chips == 777
    assert adapter.max_steps_per_hand == 33
    assert np.allclose(probs, np.array([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32))


def test_load_policy_adapter_supports_tianshou_ppo_probabilities(monkeypatch):
    pytest.importorskip("gymnasium")
    from scripts import run_tianshou_ppo_native_control as ppo_control

    fake_policy = object()

    monkeypatch.setattr(
        ppo_control,
        "_load_ppo_checkpoint_policy",
        lambda checkpoint_path, resolved_device: (
            {
                "algorithm": "tianshou_ppo",
                "metrics": {"initial_chips": 888, "max_steps_per_hand": 44},
            },
            fake_policy,
        ),
    )
    monkeypatch.setattr(
        ppo_control,
        "_ppo_action_probs",
        lambda policy, features, legal_mask: np.array(
            [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75],
            dtype=np.float32,
        ),
    )

    adapter = load_policy_adapter("ppo.pt", kind="tianshou-ppo", device=torch.device("cpu"))
    legal_mask = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    probs = adapter.probs(np.zeros(126, dtype=np.float32), legal_mask, torch.device("cpu"))

    assert adapter.kind == "tianshou-ppo"
    assert adapter.algorithm == "tianshou_ppo"
    assert adapter.initial_chips == 888
    assert adapter.max_steps_per_hand == 44
    assert np.allclose(probs, np.array([0, 0.25, 0, 0, 0, 0, 0, 0, 0.75], dtype=np.float32))


def test_load_policy_adapter_supports_agilerl_ippo_stateful_action(monkeypatch):
    agilerl_module = types.ModuleType("agilerl")
    algorithms_module = types.ModuleType("agilerl.algorithms")

    class _FakeAgent:
        training_mode = True

        def set_training_mode(self, mode):
            self.training_mode = bool(mode)

    class _FakeIPPO:
        @staticmethod
        def load(checkpoint_path, device):
            assert checkpoint_path == "agilerl.pt"
            assert device == "cpu"
            return _FakeAgent()

    algorithms_module.IPPO = _FakeIPPO
    monkeypatch.setitem(sys.modules, "agilerl", agilerl_module)
    monkeypatch.setitem(sys.modules, "agilerl.algorithms", algorithms_module)

    from scripts import run_agilerl_ippo_native_control as agilerl_control

    monkeypatch.setattr(
        agilerl_control,
        "_agilerl_checkpoint_action",
        lambda agent, *, player_i, features, legal_mask: int(np.flatnonzero(legal_mask)[-1]),
    )

    adapter = load_policy_adapter("agilerl.pt", kind="agilerl-ippo", device=torch.device("cpu"))
    state = new_game(2, initial_chips=999)
    legal_mask = get_legal_mask(state).astype(np.float32)
    features = state.to_feature_vector().astype(np.float32)
    rng = np.random.default_rng(123)

    action = adapter.select_action(
        state=state,
        features=features,
        legal_mask=legal_mask,
        device=torch.device("cpu"),
        rng=rng,
    )
    probs = adapter.probs(features, legal_mask, torch.device("cpu"))

    assert adapter.kind == "agilerl-ippo"
    assert adapter.algorithm == "agilerl_ippo"
    assert adapter.initial_chips == 1000
    assert adapter.max_steps_per_hand == 256
    assert action == int(np.flatnonzero(legal_mask)[-1])
    assert np.allclose(probs, legal_mask / legal_mask.sum())


def test_load_policy_adapter_supports_rllib_ppo_checkpoint(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    rl_module_module = types.ModuleType("ray.rllib.core.rl_module")
    module_dir = tmp_path / "algo_ckpt" / "learner_group" / "learner" / "rl_module" / "shared"
    module_dir.mkdir(parents=True)

    class _FakeRllibModule:
        @staticmethod
        def from_checkpoint(path):
            assert path == str(module_dir.resolve())
            return _FakeRllibModule()

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.training = False
            return self

        def forward_inference(self, batch):
            obs = batch["obs"]
            assert set(obs) == {"observations", "action_mask"}
            assert obs["observations"].shape == (1, 126)
            return {
                "action_dist_inputs": torch.tensor(
                    [[0.0, 1.0, 99.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]],
                    dtype=torch.float32,
                )
            }

    rl_module_module.RLModule = _FakeRllibModule
    monkeypatch.setitem(sys.modules, "ray.rllib.core.rl_module", rl_module_module)

    adapter = load_policy_adapter("algo_ckpt", kind="rllib-ppo", device=torch.device("cpu"))
    legal_mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    probs = adapter.probs(np.zeros(126, dtype=np.float32), legal_mask, torch.device("cpu"))

    assert "rllib-ppo" in SUPPORTED_POLICY_KINDS
    assert adapter.kind == "rllib-ppo"
    assert adapter.algorithm == "rllib_ppo_action_masked"
    assert probs[2] == 0.0
    assert int(np.argmax(probs)) == 8
    assert np.isclose(float(probs.sum()), 1.0)


def test_resolve_rllib_module_checkpoint_accepts_single_agent_default_policy(tmp_path):
    module_dir = tmp_path / "algo_ckpt" / "learner_group" / "learner" / "rl_module" / "default_policy"
    module_dir.mkdir(parents=True)

    assert _resolve_rllib_module_checkpoint(tmp_path / "algo_ckpt") == module_dir.resolve()


def test_eval_mixed_policy_h2h_cli_writes_json(monkeypatch, tmp_path):
    from scripts import eval_mixed_policy_h2h as cli

    output = tmp_path / "mixed.json"
    expected = {
        "algorithm": "mixed_native_policy_h2h",
        "candidate_checkpoint": "a.pt",
        "baseline_checkpoint": "b.pt",
    }
    monkeypatch.setattr(
        cli,
        "evaluate_mixed_policy_head_to_head",
        lambda **kwargs: {**expected, "eval_state_backend": kwargs["eval_state_backend"]},
    )

    exit_code = cli.main(
        [
            "--candidate",
            "a.pt",
            "--candidate-kind",
            "tianshou-rainbow",
            "--baseline",
            "b.pt",
            "--baseline-kind",
            "native-nfsp",
            "--eval-state-backend",
            "fast-state-canonical-deal",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["candidate_checkpoint"] == expected["candidate_checkpoint"]
    assert payload["eval_state_backend"] == "fast-state-canonical-deal"


def test_eval_mixed_policy_h2h_cli_returns_nonzero_on_failed_gate(monkeypatch, tmp_path):
    from scripts import eval_mixed_policy_h2h as cli

    output = tmp_path / "mixed_failed.json"
    monkeypatch.setattr(
        cli,
        "evaluate_mixed_policy_head_to_head",
        lambda **_kwargs: {
            "algorithm": "mixed_native_policy_h2h",
            "candidate_checkpoint": "a.pt",
            "baseline_checkpoint": "b.pt",
            "passed": False,
        },
    )

    exit_code = cli.main(
        [
            "--candidate",
            "a.pt",
            "--candidate-kind",
            "tianshou-rainbow",
            "--baseline",
            "b.pt",
            "--baseline-kind",
            "native-nfsp",
            "--min-lower95-candidate-payoff",
            "0",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False
