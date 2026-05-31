"""Numerical parity + convergence tests for the PyTorch R-NaD port (poker_ai/rnad).

Correctness strategy (the whole point of porting next to a working reference):
1. PARITY — feed IDENTICAL random trajectory tensors to the canonical jax reference
   (scripts/vendor/rnad.py) and the torch port; assert the loss components and the v-trace /
   nerd / loss_v outputs match to tight tolerance. This catches re-implementation bugs.
2. CONVERGENCE SMOKE — a short Kuhn run must reduce exact NashConv (sanity that the assembled
   learner actually learns). The heavy Leduc convergence + speed validation is the script
   scripts/run_rnad_torch_leduc.py.

Reference functions are pure jax and run on CPU here (tiny tensors), so no GPU needed.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]

torch.manual_seed(0)
np.random.seed(0)

# ---- import the torch port ----
from poker_ai.rnad import functional as F  # noqa: E402


def _load_jax_reference():
    """Import scripts/vendor/rnad.py (canonical jax R-NaD). Skip the module if jax is unavailable."""
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    spec = importlib.util.spec_from_file_location("vendor_rnad", str(ROOT / "scripts" / "vendor" / "rnad.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return jax, jnp, mod


def _random_trajectory(T=5, B=4, A=3, P=2, seed=0):
    """Random but well-formed [T,B,...] trajectory tensors (numpy float64)."""
    rng = np.random.RandomState(seed)
    logits = rng.randn(T, B, A)
    legal = np.ones((T, B, A), dtype=np.float64)
    # make some actions illegal (but keep >=1 legal per row)
    illegal = rng.rand(T, B, A) < 0.25
    illegal[..., 0] = False  # action 0 always legal
    legal[illegal] = 0.0
    player_id = rng.randint(0, P, size=(T, B)).astype(np.float64)
    valid = (rng.rand(T, B) > 0.2).astype(np.float64)
    rewards = rng.randn(T, B, P)
    # behavior policy mu (legal, normalized)
    mu = np.exp(logits) * legal
    mu = mu / mu.sum(axis=-1, keepdims=True)
    # sampled action one-hot (a legal action)
    action = np.zeros((T, B), dtype=np.int64)
    for t in range(T):
        for b in range(B):
            legal_idx = np.where(legal[t, b] > 0)[0]
            action[t, b] = rng.choice(legal_idx)
    action_oh = np.zeros((T, B, A), dtype=np.float64)
    for t in range(T):
        action_oh[t, np.arange(B), action[t]] = 1.0
    v = rng.randn(T, B, 1)
    return dict(logits=logits, legal=legal, player_id=player_id, valid=valid,
               rewards=rewards, mu=mu, action_oh=action_oh, v=v, A=A, B=B, T=T, P=P)


# ============================== PARITY TESTS ==============================

def test_legal_policy_and_log_policy_parity():
    jax, jnp, ref = _load_jax_reference()
    d = _random_trajectory()
    logits, legal = d["logits"], d["legal"]

    t_pi = F.legal_policy(torch.tensor(logits), torch.tensor(legal)).numpy()
    t_logpi = F.legal_log_policy(torch.tensor(logits), torch.tensor(legal)).numpy()
    j_pi = np.asarray(ref._legal_policy(jnp.array(logits), jnp.array(legal)))
    j_logpi = np.asarray(ref.legal_log_policy(jnp.array(logits), jnp.array(legal)))

    assert np.allclose(t_pi, j_pi, atol=1e-6), np.abs(t_pi - j_pi).max()
    assert np.allclose(t_logpi, j_logpi, atol=1e-6), np.abs(t_logpi - j_logpi).max()


def test_has_played_parity():
    jax, jnp, ref = _load_jax_reference()
    d = _random_trajectory()
    for player in range(d["P"]):
        t = F.has_played(torch.tensor(d["valid"]), torch.tensor(d["player_id"]), player).numpy()
        j = np.asarray(ref._has_played(jnp.array(d["valid"]), jnp.array(d["player_id"]), player))
        assert np.allclose(t, j, atol=1e-9), (player, np.abs(t - j).max())


def test_policy_ratio_and_player_others_parity():
    jax, jnp, ref = _load_jax_reference()
    d = _random_trajectory()
    pi = d["mu"]  # any valid policy
    t_pr = F.policy_ratio(torch.tensor(pi), torch.tensor(d["mu"]), torch.tensor(d["action_oh"]),
                          torch.tensor(d["valid"])).numpy()
    j_pr = np.asarray(ref._policy_ratio(jnp.array(pi), jnp.array(d["mu"]),
                                        jnp.array(d["action_oh"]), jnp.array(d["valid"])))
    assert np.allclose(t_pr, j_pr, atol=1e-6), np.abs(t_pr - j_pr).max()

    for player in range(d["P"]):
        t_po = F.player_others(torch.tensor(d["player_id"]), torch.tensor(d["valid"]), player).numpy()
        j_po = np.asarray(ref._player_others(jnp.array(d["player_id"]), jnp.array(d["valid"]), player))
        assert np.allclose(t_po, j_po, atol=1e-9), np.abs(t_po - j_po).max()


def test_vtrace_parity():
    """The hardest piece: mixed-player v-trace reverse scan must match the jax lax.scan exactly."""
    jax, jnp, ref = _load_jax_reference()
    d = _random_trajectory()
    A, eta, c, lam = d["A"], 0.2, 1.0, 1.0
    # build a reward-transform log policy like the real loss does (use mu's log here)
    log_reg = np.log(np.clip(d["mu"], 1e-12, None)) * d["legal"]
    pi = d["mu"]

    for player in range(d["P"]):
        po = ref._player_others(jnp.array(d["player_id"]), jnp.array(d["valid"]), player)
        j_vt, j_hp, j_lo = ref.v_trace(
            jnp.array(d["v"]), jnp.array(d["valid"]), jnp.array(d["player_id"]),
            jnp.array(d["mu"]), jnp.array(pi), jnp.array(log_reg), po,
            jnp.array(d["action_oh"]), jnp.array(d["rewards"][:, :, player]), player,
            eta=eta, lambda_=lam, c=c, rho=np.inf)
        po_t = F.player_others(torch.tensor(d["player_id"]), torch.tensor(d["valid"]), player)
        t_vt, t_hp, t_lo = F.v_trace(
            torch.tensor(d["v"]), torch.tensor(d["valid"]), torch.tensor(d["player_id"]),
            torch.tensor(d["mu"]), torch.tensor(pi), torch.tensor(log_reg), po_t,
            torch.tensor(d["action_oh"]), torch.tensor(d["rewards"][:, :, player]), player,
            eta=eta, lambda_=lam, c=c, rho=float("inf"))
        assert np.allclose(np.asarray(j_vt), t_vt.numpy(), atol=1e-6), ("v_target", player, np.abs(np.asarray(j_vt)-t_vt.numpy()).max())
        assert np.allclose(np.asarray(j_hp), t_hp.numpy(), atol=1e-9), ("has_played", player)
        assert np.allclose(np.asarray(j_lo), t_lo.numpy(), atol=1e-6), ("learning_output", player, np.abs(np.asarray(j_lo)-t_lo.numpy()).max())


def test_loss_v_and_nerd_parity():
    jax, jnp, ref = _load_jax_reference()
    d = _random_trajectory()
    P = d["P"]
    v = d["v"]
    v_targets = [d["v"] + 0.1 * i for i in range(P)]
    masks = [d["valid"] * (d["player_id"] == k) for k in range(P)]

    t_lv = F.get_loss_v([torch.tensor(v)] * P, [torch.tensor(x) for x in v_targets],
                        [torch.tensor(m) for m in masks]).item()
    j_lv = float(ref.get_loss_v([jnp.array(v)] * P, [jnp.array(x) for x in v_targets],
                                [jnp.array(m) for m in masks]))
    assert abs(t_lv - j_lv) < 1e-6, (t_lv, j_lv)

    # nerd
    logit = d["logits"]
    pi = d["mu"]
    q_vr = [d["rewards"][:, :, :1] * 0 + np.random.RandomState(k).randn(*pi.shape) for k in range(P)]
    is_c = [np.ones((d["T"], d["B"], 1)) for _ in range(P)]
    t_ln = F.get_loss_nerd([torch.tensor(logit)] * P, [torch.tensor(pi)] * P,
                           [torch.tensor(q) for q in q_vr], torch.tensor(d["valid"]),
                           torch.tensor(d["player_id"]), torch.tensor(d["legal"]),
                           [torch.tensor(x) for x in is_c], clip=10000.0, threshold=2.0).item()
    j_ln = float(ref.get_loss_nerd([jnp.array(logit)] * P, [jnp.array(pi)] * P,
                                   [jnp.array(q) for q in q_vr], jnp.array(d["valid"]),
                                   jnp.array(d["player_id"]), jnp.array(d["legal"]),
                                   [jnp.array(x) for x in is_c], clip=10000.0, threshold=2.0))
    assert abs(t_ln - j_ln) < 1e-5, (t_ln, j_ln)


def test_entropy_schedule_parity():
    jax, jnp, ref = _load_jax_reference()
    t_sched = F.EntropySchedule(sizes=(7, 5), repeats=(2, 1))
    j_sched = ref.EntropySchedule(sizes=(7, 5), repeats=(2, 1))
    for step in range(0, 40):
        ta, tu = t_sched(step)
        ja, ju = j_sched(step)
        # jax computes alpha in float32; the torch port uses float64 -> compare at f32 precision.
        assert abs(ta - float(ja)) < 1e-6, (step, ta, ja)
        assert bool(tu) == bool(ju), (step, tu, ju)


# ============================== CONVERGENCE SMOKE ==============================

def test_kuhn_convergence_smoke():
    """A short Kuhn run must reduce exact NashConv (the assembled learner actually learns)."""
    pytest.importorskip("pyspiel")
    from open_spiel.python.algorithms import exploitability  # noqa
    from poker_ai.rnad import RNaDConfig, RNaDSolver, LeducTreeCollector
    import pyspiel
    from open_spiel.python import policy as policy_lib

    col = LeducTreeCollector("kuhn_poker", device="cpu")
    cfg = RNaDConfig(batch_size=256, trajectory_max=10, policy_network_layers=(64, 64),
                     learning_rate=0.005, entropy_schedule_size=(100,), entropy_schedule_repeats=(1,),
                     seed=1)
    solver = RNaDSolver(cfg, col, device="cpu")

    game = pyspiel.load_game("kuhn_poker")

    def nashconv():
        tp = policy_lib.TabularPolicy(game)
        for I, row in tp.state_lookup.items():
            # map infoset string -> obs/legal via the tree
            pass
        # Use the solver policy via action_probabilities over tabular states.
        tp = _solver_to_tabular(game, solver)
        return exploitability.nash_conv(game, tp)

    nc0 = nashconv()
    for _ in range(800):
        solver.step()
    nc1 = nashconv()
    # Full convergence (NashConv -> ~0.11 in 2000 steps) is validated by
    # scripts/run_rnad_torch_leduc.py; here we just assert clear learning in a fast smoke window.
    assert nc1 < nc0, f"NashConv did not decrease: {nc0:.3f} -> {nc1:.3f}"
    assert nc1 < nc0 * 0.6, f"NashConv barely moved: {nc0:.3f} -> {nc1:.3f}"


def _solver_to_tabular(game, solver):
    """Build an OpenSpiel TabularPolicy from the solver's target-net policy."""
    from open_spiel.python import policy as policy_lib
    tp = policy_lib.TabularPolicy(game)
    # Walk states; for each info state, query the net on its info_state_tensor.
    states = _all_decision_states(game)
    obs_by_key = {}
    for st in states:
        key = st.information_state_string()
        if key not in obs_by_key:
            obs_by_key[key] = (np.asarray(st.information_state_tensor(), dtype=np.float32),
                               np.asarray(st.legal_actions_mask(), dtype=np.float32))
    for key, idx in tp.state_lookup.items():
        if key in obs_by_key:
            obs, legal = obs_by_key[key]
            pi = solver.action_probabilities(obs[None, :], legal[None, :])[0]
            row = tp.action_probability_array[idx]
            row[:] = 0.0
            row[: len(pi)] = pi
    return tp


def _all_decision_states(game):
    out = []
    stack = [game.new_initial_state()]
    seen = set()
    while stack:
        s = stack.pop()
        if s.is_terminal():
            continue
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                c = s.clone(); c.apply_action(a); stack.append(c)
            continue
        out.append(s)
        for a in s.legal_actions():
            c = s.clone(); c.apply_action(a)
            if c.history_str() not in seen:
                seen.add(c.history_str()); stack.append(c)
    return out
