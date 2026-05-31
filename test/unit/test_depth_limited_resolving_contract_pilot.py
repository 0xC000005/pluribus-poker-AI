import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_depth_limited_resolving_contract_pilot import (  # noqa: E402
    DepthLimitedContractConfig,
    run_depth_limited_resolving_contract_pilot,
)


def test_depth_limited_contract_uses_callback_leaf_cache_for_train_and_eval(tmp_path, monkeypatch):
    calls = []

    def fake_build_callback_leaf_targets(**kwargs):
        calls.append(("collect", kwargs))
        return {
            "passed": True,
            "output": str(kwargs["output"]),
            "n_states": 12,
            "n_cases": kwargs["limit"],
            "n_skipped": 0,
        }

    def fake_train_checkpoint(**kwargs):
        calls.append(("train", kwargs))
        assert kwargs["train_dual_cache"] == tmp_path / "contract_train_callback_leaf.npz"
        assert kwargs["holdout_dual_cache"] == tmp_path / "contract_holdout_callback_leaf.npz"
        return {
            "passed": True,
            "checkpoint": str(kwargs["output_checkpoint"]),
            "belief_holdout": {"mae": 0.1},
            "zero_baseline": {"mae": 0.2},
        }

    def fake_eval_leaf_ab(**kwargs):
        calls.append(("eval", kwargs))
        assert kwargs["checkpoint"] == tmp_path / "contract_model.pt"
        return {
            "passed": True,
            "n_leaf_applied": 4,
            "leaf_action_agreement_rate": 1.0,
            "leaf_mean_action_l1_drift": 0.0,
        }

    monkeypatch.setattr(
        "run_depth_limited_resolving_contract_pilot.build_callback_leaf_targets",
        fake_build_callback_leaf_targets,
    )
    monkeypatch.setattr(
        "run_depth_limited_resolving_contract_pilot.train_public_belief_dual_hand_cfv_checkpoint",
        fake_train_checkpoint,
    )
    monkeypatch.setattr(
        "run_depth_limited_resolving_contract_pilot.eval_public_belief_dcvn_leaf_ab",
        fake_eval_leaf_ab,
    )

    metrics = run_depth_limited_resolving_contract_pilot(
        DepthLimitedContractConfig(
            cases_json=tmp_path / "cases.json",
            cfv_cache=tmp_path / "belief_cache.npz",
            output_dir=tmp_path,
            train_start_index=0,
            train_limit=2,
            holdout_start_index=2,
            holdout_limit=1,
            solver_iterations=3,
            epochs=5,
            device="cpu",
        )
    )

    assert metrics["passed"] is True
    assert metrics["contract"] == "callback_leaf_train_to_callback_leaf_inference"
    assert metrics["promotion"] is False
    assert [name for name, _kwargs in calls] == ["collect", "collect", "train", "eval"]
    assert calls[0][1]["output"] == tmp_path / "contract_train_callback_leaf.npz"
    assert calls[1][1]["output"] == tmp_path / "contract_holdout_callback_leaf.npz"


def test_depth_limited_contract_fails_if_leaf_eval_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "run_depth_limited_resolving_contract_pilot.build_callback_leaf_targets",
        lambda **kwargs: {"passed": True, "output": str(kwargs["output"]), "n_states": 1},
    )
    monkeypatch.setattr(
        "run_depth_limited_resolving_contract_pilot.train_public_belief_dual_hand_cfv_checkpoint",
        lambda **kwargs: {"passed": True, "checkpoint": str(kwargs["output_checkpoint"])},
    )
    monkeypatch.setattr(
        "run_depth_limited_resolving_contract_pilot.eval_public_belief_dcvn_leaf_ab",
        lambda **kwargs: {"passed": False, "leaf_action_agreement_rate": 0.0},
    )

    metrics = run_depth_limited_resolving_contract_pilot(
        DepthLimitedContractConfig(
            cases_json=tmp_path / "cases.json",
            cfv_cache=tmp_path / "belief_cache.npz",
            output_dir=tmp_path,
        )
    )

    assert metrics["passed"] is False
    assert metrics["leaf_eval_passed"] is False


def test_depth_limited_contract_uses_leaf_eval_not_supervised_fit_as_decision_gate(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "run_depth_limited_resolving_contract_pilot.build_callback_leaf_targets",
        lambda **kwargs: {"passed": True, "output": str(kwargs["output"]), "n_states": 1},
    )
    monkeypatch.setattr(
        "run_depth_limited_resolving_contract_pilot.train_public_belief_dual_hand_cfv_checkpoint",
        lambda **kwargs: {"passed": False, "checkpoint": str(kwargs["output_checkpoint"])},
    )
    monkeypatch.setattr(
        "run_depth_limited_resolving_contract_pilot.eval_public_belief_dcvn_leaf_ab",
        lambda **kwargs: {
            "passed": True,
            "n_leaf_applied": 1,
            "leaf_action_agreement_rate": 1.0,
        },
    )

    metrics = run_depth_limited_resolving_contract_pilot(
        DepthLimitedContractConfig(
            cases_json=tmp_path / "cases.json",
            cfv_cache=tmp_path / "belief_cache.npz",
            output_dir=tmp_path,
        )
    )

    assert metrics["passed"] is True
    assert metrics["train_supervised_fit_passed"] is False
    assert metrics["leaf_eval_passed"] is True
