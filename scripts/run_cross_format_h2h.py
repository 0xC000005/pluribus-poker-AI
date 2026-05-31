#!/usr/bin/env python3
"""Cross-format duplicate-swapped H2H: native-PPO/NeuRD policies vs the tianshou C51
Rainbow incumbent (which the native league eval cannot load).

Reuses the EXACT duplicate-swapped loop from
native_ppo_policy.evaluate_native_ppo_policy_head_to_head (seeded deck, both seat
orientations) but parameterized by per-policy action callables so a reconstructed
C51 greedy policy can play. Diagnostic only; no protected-surface edit; no Slumbot.

Policy specs:  native:<ckpt>  |  rainbow:<ckpt>  |  random
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from poker_ai.games.full_deck.state import INDEX_TO_ACTION, new_game
from poker_ai.research.native_nfsp import get_legal_mask, select_action
from poker_ai.research.native_ppo_policy import (
    _PolicyMLP, _network_probs, _policy_feature_vector, _load_policy_network,
)


def _rainbow_q_fn(ckpt_path: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_atoms = int(ck["num_atoms"]); n_act = int(ck["num_actions"]); hid = int(ck["hidden_dim"])
    n_feat = int(ck["num_features"])
    net = nn.Sequential(nn.Linear(n_feat, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                        nn.Linear(hid, n_act * n_atoms)).to(device)
    raw_sd = ck["shared_model_state_dict"]
    sd = {(k[4:] if k.startswith("net.") else k): v for k, v in raw_sd.items()}
    net.load_state_dict(sd); net.eval()
    z = torch.linspace(-1.0, 1.0, n_atoms, device=device)  # C51 support v_min=-1, v_max=1

    def act(state, rng):
        feat = torch.tensor(_policy_feature_vector(state, "flat"), dtype=torch.float32, device=device)
        legal = get_legal_mask(state)
        with torch.no_grad():
            logits = net(feat).view(n_act, n_atoms)
            p = torch.softmax(logits, dim=1)
            q = (p * z.unsqueeze(0)).sum(dim=1).cpu().numpy()
        q = np.where(np.asarray(legal) > 0, q, -1e9)
        return int(np.argmax(q))  # greedy (DQN eval policy)
    return act


def _native_act_fn(ckpt_path: str, device: torch.device):
    _payload, policy_net, feature_mode = _load_policy_network(ckpt_path, device, strategy_source="auto")

    def act(state, rng):
        feat = _policy_feature_vector(state, feature_mode)
        legal = get_legal_mask(state)
        probs = _network_probs(policy_net, feat, legal, device)
        return int(select_action(probs, legal, rng=rng))
    return act


def _random_act_fn():
    def act(state, rng):
        legal = np.asarray(get_legal_mask(state), dtype=np.float64)
        p = legal / legal.sum()
        return int(rng.choice(len(p), p=p))
    return act


def build_act(spec: str, device: torch.device):
    if spec == "random":
        return _random_act_fn()
    kind, path = spec.split(":", 1)
    if kind == "rainbow":
        return _rainbow_q_fn(path, device)
    if kind == "native":
        return _native_act_fn(path, device)
    raise ValueError(f"bad spec {spec}")


def duplicate_swapped_h2h(cand_act, base_act, n_games, seed, initial_chips=20000, max_steps=256):
    cand_pair = []; cand_payoffs = []
    pair_i = 0
    while len(cand_payoffs) < n_games:
        game_seed = seed + pair_i
        action_seed = seed + 1_000_000 + pair_i
        pair = []
        for cand_seat in (0, 1):
            if len(cand_payoffs) >= n_games:
                break
            random.seed(game_seed); np.random.seed(game_seed); torch.manual_seed(game_seed)
            rng = np.random.default_rng(action_seed)
            acts = {cand_seat: cand_act, 1 - cand_seat: base_act}
            state = new_game(2, initial_chips=initial_chips)
            ns = 0
            while not state.is_terminal and ns < max_steps:
                a = acts[state.player_i](state, rng)
                state = state.apply_action(INDEX_TO_ACTION[a]); ns += 1
            payoff = float(state.payout.get(cand_seat, 0)) / float(initial_chips)
            cand_payoffs.append(payoff); pair.append(payoff)
        if pair:
            cand_pair.append(float(np.mean(pair)))
        pair_i += 1
    mean = float(np.mean(cand_pair)); std = float(np.std(cand_pair, ddof=1)) if len(cand_pair) > 1 else 0.0
    se = std / float(np.sqrt(len(cand_pair))) if cand_pair else 0.0
    return {"mean": mean, "lower95": mean - 1.96 * se, "upper95": mean + 1.96 * se, "std": std, "n": len(cand_payoffs)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="native:<ckpt> | rainbow:<ckpt> | random")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--n-games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260529)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    cand = build_act(args.candidate, device); base = build_act(args.baseline, device)
    r = duplicate_swapped_h2h(cand, base, int(args.n_games), int(args.seed))
    out = {"candidate": args.candidate, "baseline": args.baseline, **r}
    print(json.dumps(out, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
