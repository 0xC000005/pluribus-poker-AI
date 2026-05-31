import json
import subprocess

import numpy as np
import pytest

from poker_ai.research.alphanlholdem_benchmark import (
    AlphaNLHoldemTorchPolicy,
    RLCardAgentPolicy,
    _current_rlcard_state,
    encode_reference_observation,
    summarize_h2h_payoffs,
)
from poker_ai.research.alphanlholdem_reference import (
    inspect_alphanlholdem_reference,
)


def _write_reference_tree(root):
    (root / "agi").mkdir(parents=True)
    (root / "weights").mkdir()
    (root / "readme.md").write_text(
        "\n".join(
            [
                "This is an implementation of a self-play non-limit texas holdem ai.",
                "it's not a offical implementation of Alpha Holdem.",
                "rlcard's nl-holdem env was used.",
                "It's a 50bb 1v1 env, not the standard 100bb ACPC one.",
                "Weights of all checkpoints in the process of a week of training",
            ]
        ),
        encoding="utf-8",
    )
    (root / "LICENSE.txt").write_text("GNU AFFERO GENERAL PUBLIC LICENSE", encoding="utf-8")
    (root / "requirements.txt").write_text(
        "\n".join(["tensorflow==1.15.2", "ray==0.8.3", "rlcard==1.1.0"]),
        encoding="utf-8",
    )
    (root / "agi" / "nl_holdem_env.py").write_text(
        "\n".join(
            [
                "import rlcard",
                "class NlHoldemEnvWrapper:",
                "    def __init__(self):",
                "        self.env = rlcard.make('no-limit-holdem')",
                "        self.action_num = 5",
            ]
        ),
        encoding="utf-8",
    )
    (root / "agi" / "nl_holdem_net.py").write_text(
        "class NlHoldemNet:\n    pass\n",
        encoding="utf-8",
    )
    (root / "agi" / "league.py").write_text(
        "def kbsp(win_rates, k=5):\n    return win_rates\n",
        encoding="utf-8",
    )
    (root / "weights" / "c_1048.pkl").write_bytes(b"not-a-real-pickle")


def test_inspect_alphanlholdem_reference_reports_separate_rlcard_surface(tmp_path):
    _write_reference_tree(tmp_path)

    report = inspect_alphanlholdem_reference(tmp_path, native_num_actions=9)

    assert report["present"] is True
    assert report["source_name"] == "AlphaNLHoldem"
    assert report["official_alpha_holdem"] is False
    assert report["rlcard_reference_surface"] is True
    assert report["native_slumbot_surface"] is False
    assert report["action_count"] == 5
    assert report["native_num_actions"] == 9
    assert report["direct_native_action_match"] is False
    assert report["stack_depth"] == "50bb"
    assert report["bundled_checkpoint_present"] is True
    assert report["license_risk"] == "agpl_reference_only"
    assert report["safe_tracked_import"] is False
    assert report["recommendation"] == "reference_only_build_isolated_rlcard_benchmark"


def test_inspect_alphanlholdem_reference_missing_checkout_blocks_benchmark(tmp_path):
    missing = tmp_path / "missing"

    report = inspect_alphanlholdem_reference(missing, native_num_actions=9)

    assert report["present"] is False
    assert report["ready_for_rlcard_benchmark"] is False
    assert "reference checkout is missing" in report["blockers"]


def test_probe_alphanlholdem_reference_cli_writes_json(tmp_path):
    from scripts import probe_alphanlholdem_reference as cli

    reference_dir = tmp_path / "AlphaNLHoldem"
    _write_reference_tree(reference_dir)
    output = tmp_path / "probe.json"

    exit_code = cli.main(
        [
            "--reference-dir",
            str(reference_dir),
            "--native-num-actions",
            "9",
            "--require-checkpoint",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready_for_rlcard_benchmark"] is True
    assert payload["promotion"] is False
    assert payload["benchmark_surfaces"] == [
        "rlcard_alphanlholdem_reference",
        "native_9_action_local_league",
        "held_out_slumbot",
    ]


def test_inspect_alphanlholdem_reference_verifies_ignored_checkout(tmp_path):
    repo = tmp_path / "repo"
    reference_dir = repo / "reference_code" / "AlphaNLHoldem"
    reference_dir.mkdir(parents=True)
    (repo / ".gitignore").write_text("reference_code/*\n!reference_code/README.md\n", encoding="utf-8")
    (repo / "reference_code" / "README.md").write_text("tracked note\n", encoding="utf-8")
    _write_reference_tree(reference_dir)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", ".gitignore", "reference_code/README.md"], cwd=repo, check=True)

    report = inspect_alphanlholdem_reference(reference_dir, native_num_actions=9, repo_root=repo)

    assert report["reference_checkout_ignored"] is True
    assert report["tracked_external_reference_files"] == []
    assert report["reference_integrity_passed"] is True
    assert report["ready_for_rlcard_benchmark"] is True


def test_inspect_alphanlholdem_reference_blocks_tracked_agpl_checkout_files(tmp_path):
    repo = tmp_path / "repo"
    reference_dir = repo / "reference_code" / "AlphaNLHoldem"
    reference_dir.mkdir(parents=True)
    (repo / ".gitignore").write_text("reference_code/*\n!reference_code/README.md\n", encoding="utf-8")
    (repo / "reference_code" / "README.md").write_text("tracked note\n", encoding="utf-8")
    _write_reference_tree(reference_dir)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "add", ".gitignore", "reference_code/README.md"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "add", "-f", "reference_code/AlphaNLHoldem/agi/nl_holdem_env.py"],
        cwd=repo,
        check=True,
    )

    report = inspect_alphanlholdem_reference(reference_dir, native_num_actions=9, repo_root=repo)

    assert report["reference_checkout_ignored"] is True
    assert report["tracked_external_reference_files"] == [
        "reference_code/AlphaNLHoldem/agi/nl_holdem_env.py"
    ]
    assert report["reference_integrity_passed"] is False
    assert report["ready_for_rlcard_benchmark"] is False
    assert "tracked AlphaNLHoldem reference files present" in report["blockers"]


def test_encode_reference_observation_matches_alphanlholdem_shapes():
    raw_obs = {
        "hand": ["SA", "D2"],
        "public_cards": ["C3", "H4", "S5", "D6"],
        "legal_actions": [type("Action", (), {"value": 0})(), type("Action", (), {"value": 4})()],
        "stakes": [98, 101],
        "stage": type("Stage", (), {"value": 2})(),
        "current_player": 1,
    }
    history = [
        [(0, 3, [0, 1, 3, 4])],
        [(1, 4, [0, 1, 4])],
        [],
        [],
    ]

    obs = encode_reference_observation(raw_obs, history, current_player=1, action_num=5)

    assert obs["card_info"].shape == (4, 13, 6)
    assert obs["action_info"].shape == (4, 5, 25)
    assert obs["extra_info"].tolist() == [98.0, 101.0]
    assert obs["legal_moves"].tolist() == [1.0, 0.0, 0.0, 0.0, 1.0]
    assert obs["card_info"][3, 12, 0] == 1.0  # SA in hole-card channel
    assert obs["card_info"][1, 4, 2] == 1.0  # D6 in turn channel
    assert obs["action_info"][0, 3, 0] == 1.0
    assert obs["action_info"][3, 4, 6] == 1.0


def _zero_reference_weights():
    shapes = {
        "card_conv0b0/kernel": (3, 3, 6, 16),
        "card_conv0b0/bias": (16,),
        "history_conv0b0/kernel": (3, 3, 25, 16),
        "history_conv0b0/bias": (16,),
        "card_conv0b0_2/kernel": (3, 3, 16, 16),
        "card_conv0b0_2/bias": (16,),
        "history_conv0b0_2/kernel": (3, 3, 16, 16),
        "history_conv0b0_2/bias": (16,),
        "card_conv1b0/kernel": (3, 3, 16, 32),
        "card_conv1b0/bias": (32,),
        "history_conv1b0/kernel": (3, 3, 16, 32),
        "history_conv1b0/bias": (32,),
        "card_conv1b0_2/kernel": (3, 3, 32, 32),
        "card_conv1b0_2/bias": (32,),
        "history_conv1b0_2/kernel": (3, 3, 32, 32),
        "history_conv1b0_2/bias": (32,),
        "card_conv1b1/kernel": (3, 3, 32, 32),
        "card_conv1b1/bias": (32,),
        "history_conv1b1/kernel": (3, 3, 32, 32),
        "history_conv1b1/bias": (32,),
        "card_conv1b1_2/kernel": (3, 3, 32, 32),
        "card_conv1b1_2/bias": (32,),
        "history_conv1b1_2/kernel": (3, 3, 32, 32),
        "history_conv1b1_2/bias": (32,),
        "card_conv2b0/kernel": (3, 3, 32, 64),
        "card_conv2b0/bias": (64,),
        "history_conv2b0/kernel": (3, 3, 32, 64),
        "history_conv2b0/bias": (64,),
        "card_conv2b0_s/kernel": (1, 1, 32, 64),
        "card_conv2b0_s/bias": (64,),
        "card_conv2b0_2/kernel": (3, 3, 64, 64),
        "card_conv2b0_2/bias": (64,),
        "history_conv2b0_s/kernel": (1, 1, 32, 64),
        "history_conv2b0_s/bias": (64,),
        "history_conv2b0_2/kernel": (3, 3, 64, 64),
        "history_conv2b0_2/bias": (64,),
        "card_conv2b1/kernel": (3, 3, 64, 64),
        "card_conv2b1/bias": (64,),
        "history_conv2b1/kernel": (3, 3, 64, 64),
        "history_conv2b1/bias": (64,),
        "card_conv2b1_s/kernel": (1, 1, 64, 64),
        "card_conv2b1_s/bias": (64,),
        "card_conv2b1_2/kernel": (3, 3, 64, 64),
        "card_conv2b1_2/bias": (64,),
        "history_conv2b1_s/kernel": (1, 1, 64, 64),
        "history_conv2b1_s/bias": (64,),
        "history_conv2b1_2/kernel": (3, 3, 64, 64),
        "history_conv2b1_2/bias": (64,),
        "extra_fc/kernel": (64, 16),
        "extra_fc/bias": (16,),
        "fc_1/kernel": (144, 256),
        "fc_1/bias": (256,),
        "fc_2/kernel": (256, 128),
        "fc_2/bias": (128,),
        "fc_3/kernel": (128, 64),
        "fc_3/bias": (64,),
        "conv_fuse/kernel": (64, 5),
        "conv_fuse/bias": (5,),
        "value_out/kernel": (64, 1),
        "value_out/bias": (1,),
    }
    return {
        f"oppo_policy/{name}": np.zeros(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }


def test_alphanlholdem_torch_policy_masks_illegal_actions():
    weights = _zero_reference_weights()
    weights["oppo_policy/conv_fuse/bias"] = np.asarray([0, 10, 0, 0, 5], dtype=np.float32)
    policy = AlphaNLHoldemTorchPolicy.from_weights(weights)
    obs = {
        "card_info": np.zeros((4, 13, 6), dtype=np.float32),
        "action_info": np.zeros((4, 5, 25), dtype=np.float32),
        "extra_info": np.zeros((2,), dtype=np.float32),
        "legal_moves": np.asarray([1, 0, 0, 0, 1], dtype=np.float32),
    }

    decision = policy.act(obs)

    assert decision.action == 4
    assert decision.legal_actions == [0, 4]
    assert decision.policy_name == "alphanlholdem_torch_checkpoint"
    assert decision.logits.shape == (5,)


def test_summarize_h2h_payoffs_reports_confidence_bounds():
    metrics = summarize_h2h_payoffs([1.0, -0.5, 0.5, 0.0], games_per_seat=2)

    assert metrics["games"] == 4
    assert metrics["games_per_seat"] == 2
    assert metrics["mean_candidate_payoff"] == 0.25
    assert metrics["promotion"] is False
    assert metrics["lower95_candidate_payoff"] < metrics["upper95_candidate_payoff"]


def test_summarize_h2h_payoffs_applies_lower95_gate():
    passed = summarize_h2h_payoffs(
        [2.0, 2.0, 2.0, 2.0],
        games_per_seat=2,
        min_lower95_candidate_payoff=1.0,
    )
    failed = summarize_h2h_payoffs(
        [0.0, 0.0, 0.0, 0.0],
        games_per_seat=2,
        min_lower95_candidate_payoff=1.0,
    )

    assert passed["passed"] is True
    assert passed["min_lower95_candidate_payoff"] == 1.0
    assert failed["passed"] is False


def test_eval_alphanlholdem_cli_accepts_separate_candidate_and_baseline_weights(monkeypatch, tmp_path):
    from scripts import eval_alphanlholdem_rlcard_reference as cli

    candidate_weights = tmp_path / "candidate.pkl"
    baseline_weights = tmp_path / "baseline.pkl"
    candidate_weights.write_bytes(b"candidate")
    baseline_weights.write_bytes(b"baseline")
    output = tmp_path / "h2h.json"
    load_calls = []

    class _Policy:
        def __init__(self, name):
            self.policy_name = name

    def fake_load_policy(kind, *, weights, seed, device):
        load_calls.append((kind, weights, seed, device))
        return _Policy(f"{kind}:{weights.name}")

    def fake_eval(**kwargs):
        return {
            "algorithm": "alphanlholdem_rlcard_reference_h2h",
            "candidate_policy": kwargs["candidate"].policy_name,
            "baseline_policy": kwargs["baseline"].policy_name,
            "games": kwargs["games_per_seat"] * 2,
            "promotion": False,
        }

    monkeypatch.setattr(cli, "_load_policy", fake_load_policy)
    monkeypatch.setattr(cli, "evaluate_rlcard_reference_h2h", fake_eval)
    monkeypatch.setattr(
        cli,
        "inspect_alphanlholdem_reference",
        lambda reference_dir, native_num_actions, **kwargs: {
            "reference_integrity_checked": True,
            "reference_checkout_ignored": True,
            "tracked_external_reference_files": [],
            "reference_integrity_passed": True,
            "ready_for_rlcard_benchmark": True,
        },
    )

    exit_code = cli.main(
        [
            "--candidate",
            "alphanlholdem",
            "--candidate-weights",
            str(candidate_weights),
            "--baseline",
            "alphanlholdem",
            "--baseline-weights",
            str(baseline_weights),
            "--games-per-seat",
            "3",
            "--seed",
            "11",
            "--device",
            "cpu",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    assert load_calls[0][:2] == ("alphanlholdem", candidate_weights)
    assert load_calls[1][:2] == ("alphanlholdem", baseline_weights)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["candidate_weights"] == str(candidate_weights)
    assert payload["baseline_weights"] == str(baseline_weights)
    assert payload["games"] == 6
    assert payload["reference_integrity_checked"] is True
    assert payload["reference_checkout_ignored"] is True
    assert payload["tracked_external_reference_files"] == []
    assert payload["reference_integrity_passed"] is True


def test_eval_alphanlholdem_cli_rejects_cross_env_projection_candidate():
    from scripts import eval_alphanlholdem_rlcard_reference as cli

    with pytest.raises(ValueError, match="unknown policy kind"):
        cli._load_policy(
            "projected-native-uniform",
            weights=None,
            seed=1,
            device="cpu",
        )


def test_rlcard_agent_policy_uses_raw_rlcard_state_and_masks_actions():
    class _Agent:
        def eval_step(self, state):
            assert state["obs"].shape == (4,)
            return 3, {"probs": {state["raw_legal_actions"][0]: 0.25, state["raw_legal_actions"][1]: 0.75}}

    action0 = type("Action", (), {"value": 1})()
    action1 = type("Action", (), {"value": 3})()
    policy = RLCardAgentPolicy(_Agent(), policy_name="rlcard_fake")
    decision = policy.act_rlcard_state(
        {
            "obs": np.zeros((4,), dtype=np.float32),
            "legal_actions": {1: None, 3: None},
            "raw_legal_actions": [action0, action1],
        }
    )

    assert decision.action == 3
    assert decision.legal_actions == [1, 3]
    assert decision.policy_name == "rlcard_fake"
    assert np.isneginf(decision.masked_logits[0])
    assert np.isneginf(decision.masked_logits[2])
    assert np.isneginf(decision.masked_logits[4])


def test_current_rlcard_state_uses_state_from_rlcard_tuple():
    state = {"obs": np.zeros((4,), dtype=np.float32)}

    assert _current_rlcard_state((state, 1)) is state


def test_eval_alphanlholdem_cli_accepts_rlcard_nfsp_candidate(monkeypatch, tmp_path):
    from scripts import eval_alphanlholdem_rlcard_reference as cli

    candidate_weights = tmp_path / "candidate.pt"
    baseline_weights = tmp_path / "baseline.pkl"
    candidate_weights.write_bytes(b"candidate")
    baseline_weights.write_bytes(b"baseline")
    output = tmp_path / "h2h.json"
    load_calls = []

    class _Policy:
        def __init__(self, name):
            self.policy_name = name

    def fake_load_policy(kind, *, weights, seed, device):
        load_calls.append((kind, weights, seed, device))
        return _Policy(kind)

    def fake_eval(**kwargs):
        return {
            "algorithm": "alphanlholdem_rlcard_reference_h2h",
            "candidate_policy": kwargs["candidate"].policy_name,
            "baseline_policy": kwargs["baseline"].policy_name,
            "games": kwargs["games_per_seat"] * 2,
            "promotion": False,
        }

    monkeypatch.setattr(cli, "_load_policy", fake_load_policy)
    monkeypatch.setattr(cli, "evaluate_rlcard_reference_h2h", fake_eval)

    exit_code = cli.main(
        [
            "--candidate",
            "rlcard-nfsp",
            "--candidate-weights",
            str(candidate_weights),
            "--baseline",
            "alphanlholdem",
            "--baseline-weights",
            str(baseline_weights),
            "--games-per-seat",
            "2",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    assert load_calls[0][:2] == ("rlcard-nfsp", candidate_weights)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["candidate_kind"] == "rlcard-nfsp"
    assert payload["trained_environment_native"] is True
    assert payload["native_action_projection"] is False


def test_eval_alphanlholdem_cli_returns_nonzero_when_lower95_gate_fails(monkeypatch, tmp_path):
    from scripts import eval_alphanlholdem_rlcard_reference as cli

    weights = tmp_path / "weights.pkl"
    weights.write_bytes(b"weights")

    class _Policy:
        policy_name = "fake"

    monkeypatch.setattr(cli, "_load_policy", lambda *_args, **_kwargs: _Policy())
    monkeypatch.setattr(
        cli,
        "evaluate_rlcard_reference_h2h",
        lambda **_kwargs: {
            "algorithm": "alphanlholdem_rlcard_reference_h2h",
            "lower95_candidate_payoff": -1.0,
            "passed": False,
            "promotion": False,
        },
    )

    exit_code = cli.main(
        [
            "--candidate",
            "alphanlholdem",
            "--candidate-weights",
            str(weights),
            "--baseline",
            "alphanlholdem",
            "--baseline-weights",
            str(weights),
            "--min-lower95-candidate-payoff",
            "0.0",
        ]
    )

    assert exit_code == 1
