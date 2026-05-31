import pytest


pytest.importorskip("tianshou")

from scripts.run_tianshou_ppo_native_control import (  # noqa: E402
    _MaskedActor,
    _build_ppo_policy_from_actor,
    _load_ppo_checkpoint_policy,
    _make_env,
    run_control,
)
from scripts.run_tianshou_rainbow_native_control import NativeRainbowPokerEnv  # noqa: E402


def test_ppo_policy_builder_allows_stochastic_eval_mode():
    env = NativeRainbowPokerEnv(seed=1)
    actor = _MaskedActor(hidden_dim=16, device="cpu")

    stochastic_policy = _build_ppo_policy_from_actor(
        actor,
        action_space=env.action_space,
        observation_space=env.observation_space,
        deterministic_eval=False,
    )
    deterministic_policy = _build_ppo_policy_from_actor(
        actor,
        action_space=env.action_space,
        observation_space=env.observation_space,
        deterministic_eval=True,
    )

    assert stochastic_policy.deterministic_eval is False
    assert deterministic_policy.deterministic_eval is True


def test_ppo_checkpoint_loader_preserves_requested_eval_mode(tmp_path):
    checkpoint = tmp_path / "ppo.pt"
    run_control(
        rollout_steps=8,
        updates=1,
        repeat=1,
        batch_size=4,
        hidden_dim=16,
        eval_games=1,
        device="cpu",
        seed=20260670,
        checkpoint_out=str(checkpoint),
    )

    _payload, stochastic_policy = _load_ppo_checkpoint_policy(
        str(checkpoint),
        resolved_device="cpu",
        deterministic_eval=False,
    )
    _payload, deterministic_policy = _load_ppo_checkpoint_policy(
        str(checkpoint),
        resolved_device="cpu",
        deterministic_eval=True,
    )

    assert stochastic_policy.deterministic_eval is False
    assert deterministic_policy.deterministic_eval is True


def test_ppo_run_control_records_fixed_opponent_configuration(tmp_path):
    checkpoint = tmp_path / "ppo.pt"
    metrics = run_control(
        rollout_steps=8,
        updates=1,
        repeat=1,
        batch_size=4,
        hidden_dim=16,
        eval_games=1,
        device="cpu",
        seed=20260671,
        opponent_kind="random",
        opponent_checkpoint=["parent.pt", "population.pt"],
        opponent_device="cpu",
        checkpoint_out=str(checkpoint),
    )

    assert metrics["opponent_kind"] == "random"
    assert metrics["opponent_checkpoints"] == ["parent.pt", "population.pt"]
    assert metrics["opponent_device_requested"] == "cpu"
    assert metrics["opponent_device"] is None
    assert checkpoint.exists()


def test_ppo_subproc_checkpoint_opponents_use_spawn_context(monkeypatch):
    import scripts.run_tianshou_ppo_native_control as ppo_script
    import tianshou.env as tianshou_env

    class FakeNativeEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSubprocVectorEnv:
        def __init__(self, env_fns, context=None):
            self.env_fns = env_fns
            self.context = context

    monkeypatch.setattr(ppo_script, "NativeRainbowPokerEnv", FakeNativeEnv)
    monkeypatch.setattr(tianshou_env, "SubprocVectorEnv", FakeSubprocVectorEnv)

    env, _replay_cls = _make_env(
        seed=1,
        initial_chips=1000,
        max_steps_per_hand=16,
        num_envs=2,
        vector_env_backend="subproc",
        opponent_kind="rainbow",
        opponent_checkpoint=["opponent.pt"],
        opponent_device="cpu",
    )

    assert env.context == "spawn"
