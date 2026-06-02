import numpy as np
import torch


def test_native_rainbow_action_collapse_detects_greedy_top_action(monkeypatch, tmp_path):
    from scripts import eval_native_rainbow_action_collapse as script

    class FakePolicy:
        kind = "tianshou-rainbow"
        algorithm = "fake"

        def eval(self):
            return self

        def __call__(self, features):
            scores = torch.zeros((features.shape[0], script.N_ACTIONS), dtype=torch.float32)
            scores[:, 8] = 1.0
            return scores

    features = np.zeros((4, 126), dtype=np.float32)
    masks = np.ones((4, script.N_ACTIONS), dtype=np.float32)
    streets = np.zeros(4, dtype=np.int64)
    monkeypatch.setattr(script, "_load_rainbow_q_policy", lambda *args, **kwargs: FakePolicy())
    monkeypatch.setattr(script, "_sample_local_states", lambda **kwargs: (features, masks, streets))

    metrics = script.evaluate_native_rainbow_action_collapse(
        checkpoint=tmp_path / "fake.pt",
        device="cpu",
    )

    assert metrics["greedy"]["selected_action_counts"]["all_in"] == 4
    assert metrics["greedy"]["gate"]["passed"] is False
    assert metrics["passed"] is False
