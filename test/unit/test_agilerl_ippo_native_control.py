import json
from pathlib import Path

import pytest

from poker_ai.research.native_nfsp import NativeNFSPConfig, run_native_nfsp_pilot


pytest.importorskip("agilerl")
pytest.importorskip("pettingzoo")


from scripts.run_agilerl_ippo_native_control import (  # noqa: E402
    evaluate_agilerl_ippo_checkpoint_vs_native_nfsp,
    run_agilerl_ippo_control,
)


def test_agilerl_ippo_smoke_writes_plugin_control_metrics(tmp_path):
    output = tmp_path / "agilerl_ippo.json"

    metrics = run_agilerl_ippo_control(
        max_steps=8,
        evo_steps=8,
        learn_step=8,
        eval_steps=4,
        hidden_dim=16,
        batch_size=8,
        update_epochs=1,
        seed=20260688,
        device="cpu",
        verbose=False,
        output_json=str(output),
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "agilerl_ippo"
    assert payload["role"] == "plug_in_multi_agent_policy_value_rl_control"
    assert payload["environment"] == "poker_ai:pettingzoo_full_deck_hu_nlhe"
    assert payload["library"] == "agilerl"
    assert payload["num_actions"] == 9
    assert payload["num_features"] == 126
    assert payload["num_agents"] == 2
    assert payload["max_steps"] == 8
    assert payload["evo_steps"] == 8
    assert payload["promotion"] is False
    assert metrics["train_seconds"] >= 0.0
    assert Path(output).exists()


def test_agilerl_ippo_checkpoint_evaluates_against_native_nfsp(tmp_path):
    agile_checkpoint = tmp_path / "agilerl_ippo.pt"
    nfsp_checkpoint = tmp_path / "native_nfsp.pt"

    run_agilerl_ippo_control(
        max_steps=8,
        evo_steps=8,
        learn_step=8,
        eval_steps=4,
        hidden_dim=16,
        batch_size=8,
        update_epochs=1,
        seed=20260689,
        device="cpu",
        verbose=False,
        checkpoint_out=str(agile_checkpoint),
    )
    run_native_nfsp_pilot(
        NativeNFSPConfig(
            train_episodes=2,
            eval_games=1,
            hidden_dim=16,
            batch_size=8,
            min_buffer_size_to_learn=100,
            device="cpu",
            seed=20260690,
            checkpoint_path=str(nfsp_checkpoint),
        )
    )

    metrics = evaluate_agilerl_ippo_checkpoint_vs_native_nfsp(
        candidate_checkpoint=str(agile_checkpoint),
        baseline_checkpoint=str(nfsp_checkpoint),
        n_games=2,
        device="cpu",
        seed=20260691,
    )

    assert metrics["algorithm"] == "agilerl_ippo_vs_native_nfsp_h2h"
    assert metrics["candidate_algorithm"] == "agilerl_ippo"
    assert metrics["baseline_algorithm"].startswith("native_nfsp")
    assert metrics["n_games"] == 2
    assert metrics["num_actions"] == 9
    assert metrics["promotion"] is False
