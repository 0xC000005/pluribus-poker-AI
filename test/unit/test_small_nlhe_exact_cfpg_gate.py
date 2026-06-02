import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_exact_counterfactual_rows_have_policy_centered_advantages():
    pytest.importorskip("pyspiel")
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "scripts"))

    from scripts.run_rnad_tabular_gate import Tree
    from scripts.run_small_nlhe_baseline_hardening_gate import _obs_lookup
    from scripts.run_small_nlhe_exact_cfpg_neural_gate import (
        _counterfactual_training_rows,
    )
    from poker_ai.rnad.small_nlhe import load_small_nlhe

    game = load_small_nlhe()
    tree = Tree(game)
    by = _obs_lookup(game)
    uniform_pi = [
        {action: 1.0 / len(actions) for action in actions}
        for actions in tree.iset_actions
    ]

    rows = _counterfactual_training_rows(
        game=game,
        tree=tree,
        by=by,
        pi=uniform_pi,
        advantage_temperature=1.0,
    )

    assert len(rows) == tree.n_iset
    for row in rows:
        legal = np.asarray(row["legal"], dtype=np.float64)
        current = np.asarray(row["current_policy"], dtype=np.float64)
        advantage = np.asarray(row["advantage"], dtype=np.float64)
        target = np.asarray(row["target_policy"], dtype=np.float64)

        assert row["cf_reach_weight"] >= 0.0
        assert np.isclose(target.sum(), 1.0)
        assert np.all(target[legal <= 0.0] == 0.0)
        assert abs(float(np.sum(current * advantage))) < 1e-9


def test_reference_regularization_moves_targets_toward_reference():
    pytest.importorskip("pyspiel")
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "scripts"))

    from scripts.run_rnad_tabular_gate import Tree
    from scripts.run_small_nlhe_baseline_hardening_gate import _obs_lookup
    from scripts.run_small_nlhe_exact_cfpg_neural_gate import (
        _counterfactual_training_rows,
    )
    from poker_ai.rnad.small_nlhe import load_small_nlhe

    game = load_small_nlhe()
    tree = Tree(game)
    by = _obs_lookup(game)
    current_pi = []
    reference_pi = []
    for actions in tree.iset_actions:
        first, *rest = list(actions)
        if rest:
            current = {int(first): 0.8}
            for action in rest:
                current[int(action)] = 0.2 / len(rest)
        else:
            current = {int(first): 1.0}
        reference = {int(action): 1.0 / len(actions) for action in actions}
        current_pi.append(current)
        reference_pi.append(reference)

    unregularized = _counterfactual_training_rows(
        game=game,
        tree=tree,
        by=by,
        pi=current_pi,
        advantage_temperature=1.0,
    )
    regularized = _counterfactual_training_rows(
        game=game,
        tree=tree,
        by=by,
        pi=current_pi,
        reference_pi=reference_pi,
        reference_regularization_weight=0.2,
        advantage_temperature=1.0,
    )

    unreg_distance = np.mean(
        [
            np.abs(
                row["target_policy"]
                - np.asarray(
                    [reference_pi[idx].get(a, 0.0) for a in range(game.num_distinct_actions())],
                    dtype=np.float64,
                )
            ).sum()
            for idx, row in enumerate(unregularized)
        ]
    )
    reg_distance = np.mean(
        [
            np.abs(
                row["target_policy"]
                - np.asarray(
                    [reference_pi[idx].get(a, 0.0) for a in range(game.num_distinct_actions())],
                    dtype=np.float64,
                )
            ).sum()
            for idx, row in enumerate(regularized)
        ]
    )

    assert reg_distance < unreg_distance
    assert regularized[0]["reference_regularization_weight"] == 0.2


def test_small_nlhe_exact_cfpg_gate_tiny_cli(tmp_path):
    pytest.importorskip("pyspiel")
    out = tmp_path / "exact_cfpg_gate.json"
    repo = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_small_nlhe_exact_cfpg_neural_gate.py",
            "--steps",
            "1",
            "--eval-every",
            "1",
            "--seeds",
            "1",
            "--layers",
            "8",
            "--fit-epochs-per-step",
            "1",
            "--output-json",
            str(out),
        ],
        check=False,
        cwd=repo,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())

    assert data["gate"] == "small_nlhe_exact_counterfactual_neural_pg"
    assert data["algorithm"] == "exact_counterfactual_neural_policy_gradient"
    assert data["uses_slumbot_training_data"] is False
    assert data["promotion"] is False
    assert data["decision"]["small_game_exact_only"] is True
    assert data["fingerprint"]["uniform_nashconv"] == pytest.approx(1.7, abs=1e-3)
    assert len(data["arms"]["exact_cfpg"]["runs"]) == 1
    assert data["arms"]["exact_cfpg"]["runs"][0]["loss_last"]["n_information_sets"] > 0
