import json
import subprocess
import sys
from pathlib import Path

import torch

from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES
from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.research.slumbot_opponent_response_range_ab import (
    evaluate_opponent_response_range_ab,
)


def _small_checkpoint(path: Path) -> None:
    value_net = ValueNetwork(N_FEATURES, 16, N_ACTIONS, n_layers=1)
    torch.save(
        {
            "iteration": 3,
            "n_players": 2,
            "hidden_dim": 16,
            "n_layers": 1,
            "initial_chips": 20000,
            "value_net": value_net.state_dict(),
        },
        path,
    )


def _write_trace(path: Path) -> None:
    bot_hands = [
        (1, ["Qs", "Qd"]),
        (2, ["Jh", "Td"]),
        (3, ["9s", "8s"]),
        (4, ["7h", "7d"]),
        (5, ["6s", "5s"]),
        (6, ["4h", "4d"]),
    ]
    rows = []
    for hand_index, bot_cards in bot_hands:
        rows.append(
            {
                "event": "decision",
                "source": "policy",
                "hand_index": hand_index,
                "client_pos": 0,
                "hole_cards": ["Ac", "Kd"],
                "board": [],
                "action_str": "b200",
                "full_action_str": "b200",
                "street_index": 0,
            }
        )
        rows.append(
            {
                "event": "hand_result",
                "hand_index": hand_index,
                "client_pos": 0,
                "hole_cards": ["Ac", "Kd"],
                "board": [],
                "bot_hole_cards": bot_cards,
                "winnings": 100 if hand_index % 2 == 0 else -100,
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_action_likelihood(path: Path) -> None:
    records = []
    bot_hands = [
        (1, ["Qs", "Qd"], 3),
        (2, ["Jh", "Td"], 3),
        (3, ["9s", "8s"], 3),
        (4, ["7h", "7d"], 3),
        (5, ["6s", "5s"], 3),
        (6, ["4h", "4d"], 3),
    ]
    for hand_index, bot_cards, action_idx in bot_hands:
        records.append(
            {
                "hand_index": hand_index,
                "client_pos": 0,
                "hole_cards": ["Ac", "Kd"],
                "bot_hole_cards": bot_cards,
                "board": [],
                "visible_board": [],
                "action_str_before": "",
                "action_char": "b",
                "bet_to": 200,
                "mapped_actions": [{"action_idx": action_idx, "weight": 1.0}],
                "action_prob": 0.1,
                "uniform_prob": 0.125,
                "log_lift_vs_uniform": -0.2,
            }
        )
    path.write_text(
        json.dumps({"mode": "slumbot_trace_opponent_action_likelihood", "records": records}),
        encoding="utf-8",
    )


def test_evaluate_opponent_response_range_ab_scores_holdout_hands(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    trace = tmp_path / "trace.jsonl"
    action_likelihood = tmp_path / "action_likelihood.json"
    _small_checkpoint(checkpoint)
    _write_trace(trace)
    _write_action_likelihood(action_likelihood)

    metrics = evaluate_opponent_response_range_ab(
        checkpoint,
        trace,
        action_likelihood,
        strategy_source="regret",
        holdout_fraction=0.5,
        hidden_dim=16,
        n_layers=1,
        epochs=3,
        batch_size=4,
        device="cpu",
        seed=7,
    )

    assert metrics["mode"] == "slumbot_opponent_response_range_ab"
    assert metrics["n_holdout_hands"] == 3
    assert metrics["baseline"]["n"] == metrics["response"]["n"] == 3
    assert "mean_log_lift_vs_uniform" in metrics["baseline"]
    assert "mean_log_lift_vs_uniform" in metrics["response"]
    assert "delta_mean_log_lift" in metrics
    assert all("response_true_hand_prob" in record for record in metrics["records"])


def test_evaluate_opponent_response_range_ab_can_score_all_trace_hands(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    trace = tmp_path / "trace.jsonl"
    action_likelihood = tmp_path / "action_likelihood.json"
    _small_checkpoint(checkpoint)
    _write_trace(trace)
    _write_action_likelihood(action_likelihood)

    metrics = evaluate_opponent_response_range_ab(
        checkpoint,
        trace,
        action_likelihood,
        strategy_source="regret",
        holdout_fraction=0.5,
        eval_split="all",
        hidden_dim=16,
        n_layers=1,
        epochs=3,
        batch_size=4,
        device="cpu",
        seed=7,
    )

    assert metrics["eval_split"] == "all"
    assert metrics["baseline"]["n"] == metrics["response"]["n"] == 6
    assert metrics["n_eval_hands"] == 6


def test_eval_slumbot_opponent_response_range_ab_cli_writes_metrics(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    trace = tmp_path / "trace.jsonl"
    action_likelihood = tmp_path / "action_likelihood.json"
    output = tmp_path / "range_ab.json"
    _small_checkpoint(checkpoint)
    _write_trace(trace)
    _write_action_likelihood(action_likelihood)
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "eval_slumbot_opponent_response_range_ab.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--trace",
            str(trace),
            "--action-likelihood",
            str(action_likelihood),
            "--output",
            str(output),
            "--holdout-fraction",
            "0.5",
            "--epochs",
            "3",
            "--hidden-dim",
            "16",
            "--n-layers",
            "1",
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mode"] == "slumbot_opponent_response_range_ab"
    assert payload["response"]["n"] == 3
