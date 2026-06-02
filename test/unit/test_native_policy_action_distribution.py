import json

import numpy as np

from poker_ai.research.mixed_policy_h2h import PolicyAdapter


def test_native_policy_action_distribution_cli_writes_metrics(monkeypatch, tmp_path):
    from scripts import eval_native_policy_action_distribution as script

    def fake_loader(checkpoint, kind, device):
        assert str(checkpoint) == "fake.pt"
        assert kind == "native-ppo"

        def probs(_features, legal_mask, _device):
            legal = np.asarray(legal_mask, dtype=np.float32)
            weights = np.arange(1, legal.shape[0] + 1, dtype=np.float32) * legal
            return weights / float(weights.sum())

        return PolicyAdapter(
            kind="native-ppo",
            checkpoint_path="fake.pt",
            algorithm="fake_policy",
            action_probs_fn=probs,
        )

    monkeypatch.setattr(script, "load_policy_adapter", fake_loader)
    output = tmp_path / "dist.json"

    exit_code = script.main(
        [
            "--checkpoint",
            "fake.pt",
            "--kind",
            "native-ppo",
            "--n-hands",
            "2",
            "--max-steps-per-hand",
            "2",
            "--initial-chips",
            "20000",
            "--seed",
            "20260835",
            "--device",
            "cpu",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "native_policy_action_distribution_diagnostic"
    assert payload["initial_chips"] == 20000
    assert payload["n_states"] > 0
    assert payload["probability_metrics_available"] is True
    assert payload["gate"]["n_selected_actions"] == payload["n_states"]
