"""Tests for the M2a AlphaHoldem-style card/action encoder (poker_ai/rnad/encoder.py)."""
from __future__ import annotations

import numpy as np
import torch

from poker_ai.deep_cfr.fast_state import new_fast_game
from poker_ai.rnad.encoder import CardActionEncoder, obs_to_card_planes
from poker_ai.rnad.network import RNaDNetwork


def _idx_to_suit_rankpos(card_idx: int) -> tuple[int, int]:
    return card_idx % 4, card_idx // 4  # card_idx = (rank-2)*4 + suit


def test_card_plane_decode_matches_cards():
    np.random.seed(3)
    s = new_fast_game(n_players=2, small_blind=50, big_blind=100, initial_chips=20000)
    # advance to a flop+ so the board is populated
    rng = np.random.RandomState(0)
    for _ in range(6):
        if s.is_terminal:
            break
        legal = np.nonzero(s.get_legal_mask() > 0)[0]
        s.apply_action(int(legal[1]))  # check/call
    obs = torch.from_numpy(s.to_feature_vector().astype(np.float32))
    planes = obs_to_card_planes(obs)  # [3,4,13]
    assert planes.shape == (3, 4, 13)

    hole_idx = [int(c) for c in s.hole_cards[s.current_player_i] if c >= 0]
    # NOTE: to_feature_vector encodes the ACTING player's hole cards; just check counts + board.
    assert float(planes[1].sum()) == float(len([c for c in s.community if c >= 0]))  # board count
    assert float(planes[0].sum()) == 2.0  # exactly two hole cards
    # board plane positions match community card indices exactly.
    for c in s.community:
        c = int(c)
        if c < 0:
            continue
        suit, rankpos = _idx_to_suit_rankpos(c)
        assert float(planes[1, suit, rankpos]) == 1.0
    # union channel == OR of hole and board.
    assert torch.equal(planes[2], torch.clamp(planes[0] + planes[1], max=1.0))


def test_decode_specific_indices():
    obs = torch.zeros(126)
    obs[12] = 1.0   # hole: suit 0, rankpos 3
    obs[40] = 1.0   # hole: suit 0, rankpos 10
    obs[52 + 51] = 1.0  # board: card 51 -> suit 3, rankpos 12
    p = obs_to_card_planes(obs)
    assert float(p[0, 0, 3]) == 1.0 and float(p[0, 0, 10]) == 1.0 and float(p[0].sum()) == 2.0
    assert float(p[1, 3, 12]) == 1.0 and float(p[1].sum()) == 1.0


def test_encoder_forward_shapes_and_grad():
    enc = CardActionEncoder(obs_dim=126, out_dim=128)
    # batched [B,126] and trajectory [T,B,126]
    for shape in [(16, 126), (5, 16, 126)]:
        obs = torch.rand(*shape, requires_grad=True)
        out = enc(obs)
        assert out.shape == (*shape[:-1], 128)
        out.sum().backward()
        assert obs.grad is not None
        # at least one conv weight has a gradient
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())


def test_rnad_network_with_encoder():
    enc = CardActionEncoder(obs_dim=126, out_dim=128)
    net = RNaDNetwork(126, 9, encoder=enc)
    obs = torch.rand(7, 126)
    legal = torch.zeros(7, 9)
    legal[:, :3] = 1.0  # fold/call/first-raise legal
    pi, v, log_pi, logit = net(obs, legal)
    assert pi.shape == (7, 9) and v.shape == (7, 1) and logit.shape == (7, 9)
    s = pi.sum(-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    assert float((pi * (legal <= 0)).sum()) < 1e-6  # mass only on legal


def test_flat_path_unchanged():
    net = RNaDNetwork(126, 9)  # encoder=None -> original flat MLP
    assert isinstance(net.torso, torch.nn.Sequential)
    obs = torch.rand(4, 126)
    legal = torch.ones(4, 9)
    pi, v, log_pi, logit = net(obs, legal)
    assert pi.shape == (4, 9) and v.shape == (4, 1)
