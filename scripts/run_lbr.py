#!/usr/bin/env python3
"""Local Best Response (LBR) exploitability LOWER BOUND for full-deck HU NLHE.

This is a correctness-critical research instrument. A prior learned-Q variant
(BR-LB) was BROKEN: it ranked an always-all-in shover as LESS exploitable than a
trained policy (inverted sign). The whole point of this rewrite is the
POSITIVE-CONTROL GATE below; do not trust an LBR number until the controls pass.

WHAT LBR MEASURES
-----------------
LBR plays a one-ply best response against a fixed TARGET policy and reports the
mean chips it wins (in mbb/g). Because the responder is only one-ply (it assumes
both players check/call to showdown after its action), LBR's value is a LOWER
BOUND on the target's true exploitability:

  * a HIGH LBR value PROVES the target is NOT near-Nash (it is exploitable by at
    least this much) -> verdict REFUTED.
  * a LOW LBR value is CONSISTENT WITH but is NOT PROOF OF near-Nash, because a
    smarter (multi-ply) responder might still find more exploitation
    -> verdict SUGGESTIVE_NEAR_NASH (never "near-Nash confirmed").

ALGORITHM (one-ply LBR with a Monte-Carlo playout leaf)
-------------------------------------------------------
Simulate N hands with DUPLICATE-SWAP (each deck seed played twice, LBR in seat 0
then seat 1; the pair is averaged -> the unit of variance, killing card luck).

LBR maintains a BELIEF over the target's hidden hand: uniform over all C(50,2)
hands consistent with LBR's two known cards, Bayes-updated at every TARGET
decision node by belief(h) *= target_policy(action_taken | state_with_h), with a
0.01 floor then renormalize (mirrors RangeTracker). The board narrows the belief
(hands containing a revealed card get zero weight).

At each LBR decision node, for each legal action a, EV(a) is estimated over the
belief and over sampled board runouts:
  * Apply LBR's action a (on a clone with the candidate opponent hand h injected).
  * If the target now faces the action, it FOLDS with prob
    target_policy(fold | state_after_a, h); on fold LBR wins the current pot
    (REAL chip delta read from the engine by forcing the fold).
  * Otherwise apply the ONE-PLY LEAF: force check/call to showdown, deal a fresh
    consistent runout, and read the REAL terminal payout from
    FastPokerState.payout()[lbr_seat]. We NEVER hand-derive a win*pot/2 formula
    (that formula is the bug that breaks naive LBRs).
LBR picks argmax_a EV(a), applies it to the real hand, and the realized hand
payoff (averaged over duplicate pairs) is the reported lower bound.

UNITS: payoff is in fraction-of-stack (FastPokerState.payout returns chip deltas;
we divide by initial_chips). mbb/g = payoff_in_stacks * initial_chips/big_blind *
1000. With initial_chips=20000, big_blind=100 -> * 200000.

This file is NEW and edits no existing/protected surface. It imports only
``regret_match`` from scripts.range_tracker (NOT RangeTracker, NOT its
Slumbot-string feature encoder, which produces wrong features 108-125 here).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from poker_ai.deep_cfr.fast_state import (
    N_ACTIONS,
    new_fast_game,
)
from scripts.range_tracker import regret_match

# ---------------------------------------------------------------------------
# Units / table constants (HU NLHE).
# ---------------------------------------------------------------------------
INITIAL_CHIPS = 20000
SMALL_BLIND = 50
BIG_BLIND = 100
# payoff_in_stacks -> mbb/g
MBB_PER_STACK = (INITIAL_CHIPS / BIG_BLIND) * 1000.0  # = 200000.0

FOLD = 0
CALL = 1
ALLIN = 8

ACTION_SETS = {
    "fc": (0, 1),
    "fcpa": (0, 1, 5, 8),
    "full9": tuple(range(9)),
}

# Verdict thresholds (mbb/g) from the spec.
REFUTED_LOWER95 = 200.0
SUGGESTIVE_UPPER95 = 500.0

BELIEF_FLOOR = 0.01  # matches RangeTracker's eps mixing floor.


# ---------------------------------------------------------------------------
# Target policy abstraction.
#
# A target policy is a callable: probs_batch(states_features, legal_mask) is NOT
# used directly; instead each target exposes ``action_probs(state)`` returning a
# (9,) distribution over the 9-action space for a given FastPokerState (the state
# already carries the target's hole cards). This single interface drives BOTH the
# target's real move and the Bayesian belief likelihood.
# ---------------------------------------------------------------------------


class TargetPolicy:
    """Maps a FastPokerState (from the target's perspective) -> (9,) prob dist.

    ``action_probs`` must return a distribution supported only on legal actions
    and summing to 1. Subclasses implement ``_logits_or_q_batch`` for batched
    belief evaluation when possible; the default falls back to per-state calls.
    """

    name = "abstract"

    def action_probs(self, state) -> np.ndarray:
        raise NotImplementedError

    def action_probs_batch(self, states: list) -> np.ndarray:
        """Default: loop. Subclasses override for batched network inference."""
        return np.stack([self.action_probs(s) for s in states], axis=0)


def _uniform_over_legal(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float64)
    total = mask.sum()
    if total <= 0:
        # Degenerate (no legal action) — should not happen at a decision node.
        out = np.zeros(N_ACTIONS, dtype=np.float64)
        return out
    return mask / total


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked softmax over legal actions.

    This is R-NaD's / native-ppo's ACTUAL inference policy (legal_policy) and is
    VERIFIED to match poker_ai.research.mixed_policy_h2h.load_policy_adapter's
    decode bit-for-bit (max abs diff 0.0). The earlier use of regret_match here
    measured a DIFFERENT (sharper) policy than the one the network actually plays
    in the gauntlet — a decode bug that inflated LBR exploitability.
    """
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    z = np.where(mask > 0, logits, -1e30)
    z = z - z.max()
    e = np.exp(z) * (mask > 0)
    s = e.sum()
    return e / s if s > 0 else _uniform_over_legal(mask)


class ScriptedTarget(TargetPolicy):
    """Deterministic / uniform scripted control policies.

    kind:
      shover           -> all-in (8) if legal else call (1)
      always_fold      -> fold (0) if legal else call (1)
      calling_station  -> call (1) always (1 is always legal at a decision node)
      uniform          -> uniform over legal actions
    """

    def __init__(self, kind: str):
        assert kind in {"shover", "always_fold", "calling_station", "uniform"}
        self.kind = kind
        self.name = kind

    def action_probs(self, state) -> np.ndarray:
        mask = state.get_legal_mask().astype(np.float64)
        probs = np.zeros(N_ACTIONS, dtype=np.float64)
        if self.kind == "uniform":
            return _uniform_over_legal(mask)
        if self.kind == "shover":
            a = ALLIN if mask[ALLIN] > 0 else CALL
        elif self.kind == "always_fold":
            a = FOLD if mask[FOLD] > 0 else CALL
        else:  # calling_station
            a = CALL if mask[CALL] > 0 else int(np.argmax(mask))
        probs[a] = 1.0
        return probs


class TightEquityTarget(TargetPolicy):
    """A code-defined near-equilibrium REFERENCE policy (no learning, no tuning).

    Purpose: a stand-in that is genuinely LESS exploitable than the degenerate
    controls (shover / calling_station / uniform), so the LBR ordering invariant
    ``LBR(shover) > LBR(reference)`` is both meaningful and robust. Empirically a
    pure shover and a uniform-random agent are about EQUALLY exploitable under a
    one-ply LBR (a shover denies LBR its fold-equity lever), so uniform is a poor
    "near-Nash" reference. This policy instead folds losing hands and never
    bloats the pot, denying LBR both bluff and thin-value exploitation.

    Rules (deterministic):
      * Preflop facing a bet: continue (call) iff the hand's precomputed equity
        vs a uniform opponent range >= ``preflop_threshold``, else fold.
      * Postflop facing a bet: continue iff the made-hand strength is in the top
        ~half (evaluator rank <= ``postflop_rank_cutoff``), else fold.
      * Facing no bet: check (action 1). Never raises (so LBR gets no free value
        from over-aggression).
    It is NOT claimed to be Nash — only strictly tighter/sounder than the
    degenerate controls, which is all the ordering gate needs.
    """

    # Lazily-built shared preflop equity table: {(c1<c2): equity_vs_uniform}.
    _PREFLOP_EQUITY: dict | None = None

    def __init__(self, preflop_threshold: float = 0.58,
                 postflop_rank_cutoff: int = 2200):
        self.preflop_threshold = float(preflop_threshold)
        self.postflop_rank_cutoff = int(postflop_rank_cutoff)
        self.name = f"tight_equity(th={preflop_threshold})"
        if TightEquityTarget._PREFLOP_EQUITY is None:
            TightEquityTarget._PREFLOP_EQUITY = _build_preflop_equity_table()

    def _continue_prob_for_hand(self, hand, board, to_call: int) -> float:
        """1.0 if the policy continues with ``hand``, else 0.0 (fold)."""
        if to_call <= 0:
            return 1.0  # check
        board_cards = [int(c) for c in board if int(c) >= 0]
        if not board_cards:
            key = (min(hand), max(hand))
            eq = TightEquityTarget._PREFLOP_EQUITY.get(key, 0.5)
            return 1.0 if eq >= self.preflop_threshold else 0.0
        # Postflop: made-hand strength via the evaluator (low rank = strong).
        from poker_ai.deep_cfr.fast_state import CARD_INDEX_TO_EVAL_CARD, _EVALUATOR
        be = [int(CARD_INDEX_TO_EVAL_CARD[c]) for c in board_cards]
        rank = _EVALUATOR.evaluate(
            be, [int(CARD_INDEX_TO_EVAL_CARD[int(hand[0])]),
                 int(CARD_INDEX_TO_EVAL_CARD[int(hand[1])])]
        )
        return 1.0 if rank <= self.postflop_rank_cutoff else 0.0

    def action_probs(self, state) -> np.ndarray:
        mask = state.get_legal_mask().astype(np.float64)
        pi = state.current_player_i
        hand = (int(state.hole_cards[pi, 0]), int(state.hole_cards[pi, 1]))
        biggest = int(state.bets.max())
        to_call = biggest - int(state.bets[pi])
        probs = np.zeros(N_ACTIONS, dtype=np.float64)
        cont = self._continue_prob_for_hand(hand, state.community, to_call)
        if to_call <= 0:
            probs[CALL] = 1.0  # check
        elif cont > 0.5 and mask[CALL] > 0:
            probs[CALL] = 1.0
        elif mask[FOLD] > 0:
            probs[FOLD] = 1.0
        else:
            probs[CALL] = 1.0
        return probs


def _build_preflop_equity_table(n_rollouts: int = 400, seed: int = 0) -> dict:
    """Equity-vs-uniform for all C(52,2) hands, computed per 169 canonical class.

    One-time cost ~1s. Card index = (rank-2)*4 + suit_idx.
    """
    from poker_ai.deep_cfr.fast_state import CARD_INDEX_TO_EVAL_CARD, _EVALUATOR

    rng = np.random.default_rng(seed)

    def canon(c1: int, c2: int):
        r1, s1 = c1 // 4, c1 % 4
        r2, s2 = c2 // 4, c2 % 4
        hi, lo = max(r1, r2), min(r1, r2)
        suited = (s1 == s2) and (r1 != r2)
        return (hi, lo, suited)

    class_cache: dict = {}

    def class_equity(c1: int, c2: int) -> float:
        key = canon(c1, c2)
        if key in class_cache:
            return class_cache[key]
        used = {c1, c2}
        deck = [c for c in range(52) if c not in used]
        wins = 0.0
        h_eval = [int(CARD_INDEX_TO_EVAL_CARD[c1]), int(CARD_INDEX_TO_EVAL_CARD[c2])]
        for _ in range(n_rollouts):
            idx = rng.choice(len(deck), size=7, replace=False)
            opp = [deck[idx[0]], deck[idx[1]]]
            board = [deck[idx[i]] for i in range(2, 7)]
            be = [int(CARD_INDEX_TO_EVAL_CARD[c]) for c in board]
            hr = _EVALUATOR.evaluate(be, h_eval)
            orr = _EVALUATOR.evaluate(
                be, [int(CARD_INDEX_TO_EVAL_CARD[c]) for c in opp]
            )
            wins += 1.0 if hr < orr else (0.5 if hr == orr else 0.0)
        class_cache[key] = wins / n_rollouts
        return class_cache[key]

    table: dict = {}
    for c1 in range(52):
        for c2 in range(c1 + 1, 52):
            table[(c1, c2)] = class_equity(c1, c2)
    return table


class NativePolicyTarget(TargetPolicy):
    """A trained native-PPO/NeuRD policy network (logits -> regret_match)."""

    def __init__(self, ckpt_path: str, device: torch.device):
        from poker_ai.research.native_ppo_policy import _load_policy_network

        payload, net, feature_mode = _load_policy_network(
            ckpt_path, device, strategy_source="auto"
        )
        if str(feature_mode) != "flat":
            raise ValueError(
                "run_lbr only supports feature_mode='flat' (to_feature_vector); "
                f"checkpoint uses {feature_mode!r}"
            )
        self.net = net
        self.device = device
        self.name = f"native:{Path(ckpt_path).name}"

    def _logits_batch(self, feats: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(feats.astype(np.float32)).to(self.device)
            return self.net(x).detach().cpu().numpy().reshape(len(feats), N_ACTIONS)

    def action_probs(self, state) -> np.ndarray:
        feat = state.to_feature_vector()[None, :]
        logits = self._logits_batch(feat)[0]
        mask = state.get_legal_mask().astype(np.float64)
        return _masked_softmax(logits, mask).astype(np.float64)

    def action_probs_batch(self, states: list) -> np.ndarray:
        feats = np.stack([s.to_feature_vector() for s in states], axis=0)
        logits = self._logits_batch(feats)
        out = np.zeros((len(states), N_ACTIONS), dtype=np.float64)
        for i, s in enumerate(states):
            mask = s.get_legal_mask().astype(np.float64)
            out[i] = _masked_softmax(logits[i], mask)
        return out


class RainbowQTarget(TargetPolicy):
    """A tianshou C51 Rainbow Q-network opponent.

    Rainbow outputs Q-values, NOT policy logits. The behavior policy for the
    belief likelihood is GREEDY one-hot over argmax(Q) among legal actions, with
    a small 0.01/9 epsilon floor on each legal action (never pass Q through
    regret_match).
    """

    def __init__(self, ckpt_path: str, device: torch.device):
        import torch.nn as nn

        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        n_atoms = int(ck["num_atoms"])
        n_act = int(ck["num_actions"])
        hid = int(ck["hidden_dim"])
        n_feat = int(ck["num_features"])
        if n_act != N_ACTIONS:
            raise ValueError("Rainbow checkpoint action count mismatch")
        net = nn.Sequential(
            nn.Linear(n_feat, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, n_act * n_atoms),
        ).to(device)
        raw_sd = ck["shared_model_state_dict"]
        sd = {(k[4:] if k.startswith("net.") else k): v for k, v in raw_sd.items()}
        net.load_state_dict(sd)
        net.eval()
        self.net = net
        self.n_atoms = n_atoms
        self.device = device
        self.support = torch.linspace(-1.0, 1.0, n_atoms, device=device)
        self.name = f"rainbow:{Path(ckpt_path).name}"

    def _q_batch(self, feats: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(feats.astype(np.float32)).to(self.device)
            logits = self.net(x).view(len(feats), N_ACTIONS, self.n_atoms)
            p = torch.softmax(logits, dim=2)
            q = (p * self.support.view(1, 1, -1)).sum(dim=2)
            return q.detach().cpu().numpy()

    @staticmethod
    def _greedy_eps(q: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=np.float64)
        legal = mask > 0
        eps = 0.01 / N_ACTIONS
        probs = np.zeros(N_ACTIONS, dtype=np.float64)
        probs[legal] = eps
        q_masked = np.where(legal, q, -1e30)
        best = int(np.argmax(q_masked))
        probs[best] += 1.0 - eps * int(legal.sum())
        # Renormalize defensively.
        s = probs.sum()
        return probs / s if s > 0 else _uniform_over_legal(mask)

    def action_probs(self, state) -> np.ndarray:
        q = self._q_batch(state.to_feature_vector()[None, :])[0]
        return self._greedy_eps(q, state.get_legal_mask())

    def action_probs_batch(self, states: list) -> np.ndarray:
        feats = np.stack([s.to_feature_vector() for s in states], axis=0)
        q = self._q_batch(feats)
        out = np.zeros((len(states), N_ACTIONS), dtype=np.float64)
        for i, s in enumerate(states):
            out[i] = self._greedy_eps(q[i], s.get_legal_mask())
        return out


class AdapterTarget(TargetPolicy):
    """Wraps mixed_policy_h2h.load_policy_adapter for any gauntlet kind.

    Guarantees the LBR decodes the target EXACTLY as the gauntlet H2H did
    (policy-router, native-nfsp, etc.). Per-state .probs; batch loops it.
    """

    def __init__(self, kind: str, ckpt_path: str, device: torch.device):
        from poker_ai.research.mixed_policy_h2h import load_policy_adapter

        self.adapter = load_policy_adapter(ckpt_path, kind=kind, device=device)
        self.device = device
        self.name = f"{kind}:{Path(ckpt_path).name}"

    def action_probs(self, state) -> np.ndarray:
        p = self.adapter.probs(
            state.to_feature_vector(), state.get_legal_mask(), self.device
        )
        return np.asarray(p, dtype=np.float64)


def build_target(spec: str, device: torch.device) -> TargetPolicy:
    """Resolve a target spec into a TargetPolicy.

    Specs: shover | always_fold | calling_station | uniform | tight
           native:<ckpt> | rainbow:<ckpt> | router:<ckpt> | nfsp:<ckpt>
    """
    if spec in {"shover", "always_fold", "calling_station", "uniform"}:
        return ScriptedTarget(spec)
    if spec in {"tight", "tight_equity"}:
        return TightEquityTarget()
    if ":" in spec:
        kind, path = spec.split(":", 1)
        if kind == "native":
            return NativePolicyTarget(path, device)
        if kind == "rainbow":
            return RainbowQTarget(path, device)
        if kind == "router":
            return AdapterTarget("policy-router", path, device)
        if kind == "nfsp":
            return AdapterTarget("native-nfsp", path, device)
    raise ValueError(
        f"Unknown target spec {spec!r}. Supported: shover, always_fold, "
        "calling_station, uniform, tight, native:<ckpt>, rainbow:<ckpt>, "
        "router:<ckpt>, nfsp:<ckpt>"
    )


# ---------------------------------------------------------------------------
# Belief over the target's hidden hand.
# ---------------------------------------------------------------------------


class Belief:
    """Distribution over the target's two hidden cards.

    Starts uniform over every C(50,2) hand that excludes LBR's two known cards.
    Board cards revealed during the hand zero out colliding hands. Target actions
    Bayes-narrow the distribution.
    """

    def __init__(self, lbr_cards: tuple[int, int]):
        opp_cards = [c for c in range(52) if c not in lbr_cards]
        self.hands = list(itertools.combinations(opp_cards, 2))
        self.hand_arr = np.asarray(self.hands, dtype=np.int8)  # (M, 2)
        self.idx = {h: i for i, h in enumerate(self.hands)}
        self.w = np.ones(len(self.hands), dtype=np.float64) / len(self.hands)
        self._dead = set(lbr_cards)

    def observe_board(self, board_cards) -> None:
        new = [int(c) for c in board_cards if int(c) >= 0 and int(c) not in self._dead]
        if not new:
            return
        self._dead.update(new)
        new_set = set(new)
        for i, h in enumerate(self.hands):
            if self.w[i] > 0 and (h[0] in new_set or h[1] in new_set):
                self.w[i] = 0.0
        s = self.w.sum()
        if s > 0:
            self.w /= s

    def support_indices(self) -> np.ndarray:
        return np.nonzero(self.w > 0)[0]

    def bayes_update(self, likelihood: np.ndarray) -> None:
        """likelihood: (M,) P(action_taken | hand). Floor then renormalize."""
        like = (1.0 - BELIEF_FLOOR) * likelihood + BELIEF_FLOOR * (1.0 / N_ACTIONS)
        self.w = self.w * like
        s = self.w.sum()
        if s > 0:
            self.w /= s
        else:
            # Belief collapsed (numerical) — reset to uniform over live support.
            live = np.zeros_like(self.w)
            live[self.support_indices()] = 1.0
            ls = live.sum()
            self.w = live / ls if ls > 0 else np.ones_like(self.w) / len(self.w)


# ---------------------------------------------------------------------------
# Engine helpers operating on FastPokerState clones with injected hands.
# ---------------------------------------------------------------------------


def _inject_hand(state, seat: int, hand: tuple[int, int]):
    """Clone ``state`` and set ``seat``'s hole cards to ``hand``."""
    s = state.copy()
    s.hole_cards[seat, 0] = hand[0]
    s.hole_cards[seat, 1] = hand[1]
    return s


def _rewrite_runout_deck(state, lbr_seat: int, opp_hand: tuple[int, int], rng) -> None:
    """Rewrite ``state.deck_order`` so future board deals draw a consistent runout.

    The engine's pre-shuffled deck still holds the *original* opponent cards, so
    after we inject ``opp_hand`` a naive runout could deal a board card that
    duplicates a hole card. We replace the undrawn tail with a fresh shuffle of
    every card not already used (both hole hands + revealed board).
    """
    used = set()
    used.update(int(c) for c in state.hole_cards[lbr_seat] if c >= 0)
    used.add(int(opp_hand[0]))
    used.add(int(opp_hand[1]))
    used.update(int(c) for c in state.community if c >= 0)
    remaining = [c for c in range(52) if c not in used]
    rng.shuffle(remaining)
    cursor = int(state.deck_cursor)
    new_deck = np.zeros(52, dtype=np.int8)
    # Prefix [0:cursor) is already consumed; its contents no longer matter.
    take = min(len(remaining), 52 - cursor)
    new_deck[cursor:cursor + take] = np.asarray(remaining[:take], dtype=np.int8)
    state.deck_order = new_deck


def _force_checkcall_to_terminal(state, max_steps: int = 64) -> None:
    """Drive ``state`` to terminal by forcing check/call (action 1) for all."""
    steps = 0
    while not state.is_terminal and steps < max_steps:
        state.apply_action(CALL)
        steps += 1


def _payout_if_fold(state, folding_seat: int, lbr_seat: int) -> float:
    """Chip-delta-in-stacks for ``lbr_seat`` if ``folding_seat`` folds now."""
    s = state.copy()
    # It must currently be folding_seat's turn.
    s.apply_action(FOLD)
    return float(s.payout[lbr_seat]) / INITIAL_CHIPS


# ---------------------------------------------------------------------------
# Core: one-ply LBR EV for a single action over the belief, with MC runouts.
# ---------------------------------------------------------------------------


def _ev_of_action(
    real_state,
    lbr_seat: int,
    action: int,
    belief: Belief,
    target: TargetPolicy,
    sample_hands: np.ndarray,
    sample_idx: np.ndarray,
    rng,
    n_runouts: int,
) -> float:
    """Estimate EV (in stacks, from lbr_seat) of LBR taking ``action`` now.

    sample_hands: (K,2) candidate opponent hands sampled from belief.
    sample_idx:   their indices into belief (unused but kept for clarity).
    """
    target_seat = 1 - lbr_seat
    total = 0.0
    count = 0
    for hand in sample_hands:
        hand_t = (int(hand[0]), int(hand[1]))
        # Average the action's value over independent runouts. The runout deck
        # MUST be rewritten BEFORE applying LBR's action, because the action may
        # immediately deal board cards (e.g. an all-in that closes betting) and
        # the engine's pre-shuffled deck still holds the injected hand's cards,
        # which would create duplicate-card showdowns.
        acc = 0.0
        for _ in range(n_runouts):
            s_after = _inject_hand(real_state, target_seat, hand_t)
            _rewrite_runout_deck(s_after, lbr_seat, hand_t, rng)
            s_after.apply_action(action)

            if s_after.is_terminal:
                # LBR's action ended the hand (e.g. LBR folds, or betting closed
                # and the board was dealt to showdown). Read the real payout.
                acc += float(s_after.payout[lbr_seat]) / INITIAL_CHIPS
                continue

            actor = s_after.current_player_i
            if actor == target_seat:
                # Target faces LBR's action: folds with prob p_fold(hand).
                probs = target.action_probs(s_after)
                mask = s_after.get_legal_mask()
                p_fold = float(probs[FOLD]) if mask[FOLD] > 0 else 0.0
                fold_val = (
                    _payout_if_fold(s_after, target_seat, lbr_seat)
                    if p_fold > 0.0
                    else 0.0
                )
                leaf = s_after.copy()
                _force_checkcall_to_terminal(leaf)
                cont_val = float(leaf.payout[lbr_seat]) / INITIAL_CHIPS
                acc += p_fold * fold_val + (1.0 - p_fold) * cont_val
            else:
                # LBR acts again / no target choice: one-ply leaf to showdown.
                leaf = s_after.copy()
                _force_checkcall_to_terminal(leaf)
                acc += float(leaf.payout[lbr_seat]) / INITIAL_CHIPS
        total += acc / n_runouts
        count += 1
    return total / max(count, 1)


def _both_active(state) -> bool:
    return int(state.active.sum()) >= 2


def _all_active_committed_after(pre_state, action: int) -> bool:
    """True if applying ``action`` in ``pre_state`` settles all betting into an
    all-in showdown (both players active and committed, with board still to deal).

    We re-simulate on a clone to inspect the post-state's commitment without
    randomizing the board (deck untouched), then verify it is a non-fold,
    not-yet-fully-resolved all-in line.
    """
    if action == FOLD:
        return False
    probe = pre_state.copy()
    probe.apply_action(action)
    if not _both_active(probe):
        return False  # the action ended the hand by fold elsewhere
    # All still-active players must be all-in (no chips left to bet).
    for i in range(probe.n_players):
        if probe.active[i] and int(probe.chips[i]) > 0:
            return False
    return True


def _expected_showdown_value_via_action(
    committed_pre, action: int, lbr_seat: int, lbr_hand, opp_seat: int,
    opp_hand, rng, k: int,
) -> float:
    """Average lbr payoff (stacks) over k fresh runouts of an all-in showdown.

    ``committed_pre`` is the state JUST BEFORE the committing ``action``; we
    rewrite the runout deck (using the REAL hole cards) and re-apply the action
    so the auto-dealt board is freshly randomized each iteration. Chips/bets are
    fixed by the action, so this is an unbiased estimate of the realized line's
    value, with the board card-luck averaged out.
    """
    acc = 0.0
    for _ in range(k):
        leaf = committed_pre.copy()
        _rewrite_runout_deck(leaf, lbr_seat, tuple(int(c) for c in opp_hand), rng)
        leaf.apply_action(action)
        _force_checkcall_to_terminal(leaf)
        acc += float(leaf.payout[lbr_seat]) / INITIAL_CHIPS
    return acc / k


def _sample_belief_hands(belief: Belief, k: int, rng):
    """Sample k opponent-hand indices ~ belief (with replacement)."""
    support = belief.support_indices()
    if len(support) == 0:
        return np.zeros((0, 2), dtype=np.int8), np.zeros(0, dtype=np.int64)
    w = belief.w[support]
    w = w / w.sum()
    k = min(k, max(len(support), 1))
    chosen = rng.choice(support, size=k, replace=True, p=w)
    return belief.hand_arr[chosen], chosen


# ---------------------------------------------------------------------------
# Play one hand: LBR (seat lbr_seat) vs target. Returns lbr payoff in stacks.
# ---------------------------------------------------------------------------


def play_lbr_hand(
    lbr_seat: int,
    target: TargetPolicy,
    action_set: tuple[int, ...],
    game_seed: int,
    rng,
    n_belief_samples: int,
    n_runouts: int,
    max_steps: int = 128,
) -> float:
    np.random.seed(game_seed)  # FastPokerState shuffles via np.random.permutation
    state = new_fast_game(
        n_players=2,
        small_blind=SMALL_BLIND,
        big_blind=BIG_BLIND,
        initial_chips=INITIAL_CHIPS,
    )
    target_seat = 1 - lbr_seat
    lbr_cards = (int(state.hole_cards[lbr_seat, 0]), int(state.hole_cards[lbr_seat, 1]))
    belief = Belief(lbr_cards)

    steps = 0
    while not state.is_terminal and steps < max_steps:
        steps += 1
        belief.observe_board(state.community)
        actor = state.current_player_i

        if actor == target_seat:
            # Target's real move (uses its true hole cards) + Bayes belief update.
            real_probs = target.action_probs(state)
            mask = state.get_legal_mask()
            real_action = _draw_action(real_probs, mask, rng)

            # Likelihood of the chosen action under every live belief hand.
            support = belief.support_indices()
            if len(support) > 0:
                states = [
                    _inject_hand(state, target_seat, tuple(int(c) for c in belief.hand_arr[i]))
                    for i in support
                ]
                probs_batch = target.action_probs_batch(states)
                like = np.zeros(len(belief.hands), dtype=np.float64)
                like[support] = probs_batch[:, real_action]
                belief.bayes_update(like)

            chosen = real_action
        else:
            # LBR's decision: argmax EV over legal actions in this action set.
            mask = state.get_legal_mask()
            legal_in_set = [a for a in action_set if mask[a] > 0]
            if not legal_in_set:
                legal_in_set = [a for a in range(N_ACTIONS) if mask[a] > 0]
            sample_hands, sample_idx = _sample_belief_hands(belief, n_belief_samples, rng)
            if len(sample_hands) == 0:
                # No live belief — fall back to a check/call.
                chosen = CALL if mask[CALL] > 0 else legal_in_set[0]
            else:
                best_ev = -np.inf
                chosen = legal_in_set[0]
                for a in legal_in_set:
                    ev = _ev_of_action(
                        state, lbr_seat, a, belief, target,
                        sample_hands, sample_idx, rng, n_runouts,
                    )
                    if ev > best_ev:
                        best_ev = ev
                        chosen = a

        # Variance reduction: if applying ``chosen`` settles all betting into an
        # all-in showdown (both committed, undealt board to come), return the
        # EXPECTED payoff over fresh runouts using the REAL hole cards instead of
        # one sampled board. This is unbiased (chips are fixed; only the board is
        # random) and kills the dominant all-in card-luck variance.
        pre = state.copy()
        if _all_active_committed_after(pre, chosen):
            lbr_hand = (int(pre.hole_cards[lbr_seat, 0]),
                        int(pre.hole_cards[lbr_seat, 1]))
            opp_hand = (int(pre.hole_cards[target_seat, 0]),
                        int(pre.hole_cards[target_seat, 1]))
            return _expected_showdown_value_via_action(
                pre, chosen, lbr_seat, lbr_hand, target_seat, opp_hand,
                rng, max(n_runouts * 8, 16),
            )
        state.apply_action(chosen)

    belief.observe_board(state.community)
    return float(state.payout[lbr_seat]) / INITIAL_CHIPS


def _draw_action(probs: np.ndarray, mask: np.ndarray, rng) -> int:
    """Sample a legal action from a (possibly one-hot) distribution."""
    p = np.asarray(probs, dtype=np.float64) * (np.asarray(mask) > 0)
    s = p.sum()
    if s <= 0:
        legal = np.nonzero(np.asarray(mask) > 0)[0]
        return int(rng.choice(legal))
    p = p / s
    return int(rng.choice(N_ACTIONS, p=p))


# ---------------------------------------------------------------------------
# Duplicate-swapped LBR run for one action set.
# ---------------------------------------------------------------------------


def run_lbr_action_set(
    target: TargetPolicy,
    action_set: tuple[int, ...],
    n_pairs: int,
    seed: int,
    n_belief_samples: int,
    n_runouts: int,
) -> dict:
    """Run n_pairs duplicate-swapped hands; return mbb/g stats (pairs as unit)."""
    pair_means = []
    for p in range(n_pairs):
        game_seed = seed + p
        seat_payoffs = []
        for lbr_seat in (0, 1):
            # Independent action RNG per (seed, seat) so belief sampling and
            # runouts are reproducible but distinct across the swap.
            rng = np.random.default_rng(seed + 1_000_000 + p * 2 + lbr_seat)
            payoff = play_lbr_hand(
                lbr_seat, target, action_set, game_seed, rng,
                n_belief_samples, n_runouts,
            )
            seat_payoffs.append(payoff)
        pair_means.append(float(np.mean(seat_payoffs)))

    arr = np.asarray(pair_means, dtype=np.float64)
    mean_stacks = float(arr.mean()) if len(arr) else 0.0
    std_stacks = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    se_stacks = std_stacks / np.sqrt(len(arr)) if len(arr) else 0.0
    mean = mean_stacks * MBB_PER_STACK
    se = se_stacks * MBB_PER_STACK
    return {
        "mean_mbb_g": mean,
        "lower95": mean - 1.96 * se,
        "upper95": mean + 1.96 * se,
        "n": int(len(arr)),
    }


def run_lbr(
    target: TargetPolicy,
    action_sets,
    n_pairs: int,
    seed: int,
    n_belief_samples: int,
    n_runouts: int,
) -> dict:
    results = {}
    for name in action_sets:
        results[name] = run_lbr_action_set(
            target, ACTION_SETS[name], n_pairs, seed, n_belief_samples, n_runouts,
        )
    return results


def _verdict(full9: dict | None) -> str:
    if full9 is None:
        return "AMBIGUOUS"
    n = full9["n"]
    if n >= 500 and full9["lower95"] >= REFUTED_LOWER95:
        return "REFUTED"
    if full9["mean_mbb_g"] <= 0.0 and full9["upper95"] < SUGGESTIVE_UPPER95:
        return "SUGGESTIVE_NEAR_NASH"
    return "AMBIGUOUS"


INTERPRETATION = (
    "LBR is a one-ply LOWER BOUND on exploitability. A HIGH value (REFUTED) "
    "proves the target is NOT near-Nash; it is exploitable by at least this "
    "much. A LOW value (SUGGESTIVE_NEAR_NASH) is CONSISTENT WITH but is NOT "
    "PROOF OF near-Nash, because a stronger multi-ply responder could exploit "
    "more. Never read a low LBR as 'near-Nash confirmed'."
)


# ---------------------------------------------------------------------------
# Positive-control battery.
# ---------------------------------------------------------------------------


def run_positive_controls(
    device: torch.device,
    n_pairs: int,
    seed: int,
    n_belief_samples: int,
    n_runouts: int,
    trained_spec: str | None = None,
) -> dict:
    """Run the LBR sign/ordering gate. Returns a dict with gate_passed: bool.

    Controls (full9): shover, always_fold, calling_station must each yield
    LBR > 0 with lower95 > 0.

    Ordering (the invariant the broken BR-LB inverted): a maximally-bad
    degenerate must score HIGH while a genuinely-tighter policy scores LOWER.
    We use the code-defined ``TightEquityTarget`` as the near-Nash stand-in,
    because empirically a pure shover and a uniform-random agent are about
    EQUALLY exploitable under one-ply LBR (the shover removes LBR's fold-equity
    lever), so uniform is NOT a reliable "less-exploitable" reference. The gate
    asserts LBR(shover) > LBR(tight) and LBR(calling_station) > LBR(tight).
    ``uniform`` is still measured and reported. ``trained_spec`` (optional) is
    an extra real-policy line; it is reported but does NOT change the gate.
    """
    out = {}
    controls = ["shover", "always_fold", "calling_station", "uniform"]
    for kind in controls:
        tgt = ScriptedTarget(kind)
        res = run_lbr_action_set(
            tgt, ACTION_SETS["full9"], n_pairs, seed, n_belief_samples, n_runouts,
        )
        out[kind] = res

    out["tight"] = run_lbr_action_set(
        TightEquityTarget(), ACTION_SETS["full9"], n_pairs, seed,
        n_belief_samples, n_runouts,
    )

    if trained_spec is not None:
        tgt = build_target(trained_spec, device)
        out["trained"] = run_lbr_action_set(
            tgt, ACTION_SETS["full9"], n_pairs, seed, n_belief_samples, n_runouts,
        )

    near_nash_key = "tight"

    checks = {
        "shover_positive": bool(
            out["shover"]["mean_mbb_g"] > 0 and out["shover"]["lower95"] > 0
        ),
        "always_fold_positive": bool(
            out["always_fold"]["mean_mbb_g"] > 0
            and out["always_fold"]["lower95"] > 0
        ),
        "calling_station_positive": bool(
            out["calling_station"]["mean_mbb_g"] > 0
            and out["calling_station"]["lower95"] > 0
        ),
        "ordering_shover_gt_nearnash": bool(
            out["shover"]["mean_mbb_g"] > out[near_nash_key]["mean_mbb_g"]
        ),
        "ordering_station_gt_nearnash": bool(
            out["calling_station"]["mean_mbb_g"] > out[near_nash_key]["mean_mbb_g"]
        ),
    }
    gate_passed = all(checks.values())
    return {
        "results": out,
        "checks": checks,
        "near_nash_key": near_nash_key,
        "gate_passed": bool(gate_passed),
        "config": {
            "n_pairs": n_pairs,
            "seed": seed,
            "n_belief_samples": n_belief_samples,
            "n_runouts": n_runouts,
        },
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--target",
        help="Target spec: shover | always_fold | calling_station | uniform | "
        "native:<ckpt> | rainbow:<ckpt>",
    )
    ap.add_argument("--n-hands", type=int, default=500,
                    help="Number of duplicate PAIRS to simulate.")
    ap.add_argument("--action-sets", default="fc,fcpa,full9",
                    help="Comma list among: fc, fcpa, full9.")
    ap.add_argument("--belief-samples", type=int, default=24,
                    help="Opponent hands sampled from belief per LBR decision.")
    ap.add_argument("--runouts", type=int, default=4,
                    help="Board runouts averaged per leaf evaluation.")
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-json")
    ap.add_argument("--positive-controls", action="store_true",
                    help="Run the positive-control gate battery instead.")
    ap.add_argument("--trained-spec", default=None,
                    help="Optional real-policy spec for the ordering check.")
    args = ap.parse_args(argv)

    device = torch.device(args.device)

    if args.positive_controls:
        pc = run_positive_controls(
            device, args.n_hands, args.seed, args.belief_samples,
            args.runouts, trained_spec=args.trained_spec,
        )
        out = {
            "mode": "positive_controls",
            "positive_controls": pc,
            "interpretation": INTERPRETATION,
        }
        print(json.dumps(out, indent=2))
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(json.dumps(out, indent=2))
        return 0 if pc["gate_passed"] else 1

    if not args.target:
        ap.error("--target is required unless --positive-controls is set")

    action_sets = [s.strip() for s in args.action_sets.split(",") if s.strip()]
    for s in action_sets:
        if s not in ACTION_SETS:
            ap.error(f"unknown action set {s!r}")

    target = build_target(args.target, device)
    res = run_lbr(
        target, action_sets, args.n_hands, args.seed,
        args.belief_samples, args.runouts,
    )

    # Always also run the positive-control gate so a measurement carries proof
    # that the instrument was sign-correct under this config.
    pc = run_positive_controls(
        device, max(1, min(args.n_hands, 50)), args.seed,
        args.belief_samples, args.runouts,
    )

    full9 = res.get("full9")
    out = {
        "target": target.name,
        "action_sets": {
            k: res[k] for k in ("fc", "fcpa", "full9") if k in res
        },
        "verdict": _verdict(full9),
        "interpretation": INTERPRETATION,
        "config": {
            "n_pairs": args.n_hands,
            "seed": args.seed,
            "belief_samples": args.belief_samples,
            "runouts": args.runouts,
            "initial_chips": INITIAL_CHIPS,
            "big_blind": BIG_BLIND,
        },
        "positive_controls": {**pc, "note": "fast smoke-config gate alongside "
                              "the measurement; rerun --positive-controls with "
                              "N>=500 for the full gate."},
    }
    print(json.dumps(out, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
