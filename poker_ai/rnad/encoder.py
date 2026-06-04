"""AlphaHoldem-style pseudo-Siamese encoder for the R-NaD net (M2a).

Replaces the flat-MLP torso in RNaDNetwork with a CARD-PLANE CNN branch + a
non-card MLP branch, fused into an embedding. Motivation: the flat 126-d feature
vector conflates rank/suit structure that a CNN extracts naturally; the flat-MLP
blueprint's LBR exploitability is ceiling'd at ~12-14K mbb/g across budget/batch/
clip/reset (RESEARCH_LOG 20260603T220157Z), and representation is the only
untested lever. Card encoding follows AlphaHoldem (Zhao et al. 2022).

Decodes the EXISTING 126-d feature contract (no upstream change): obs[0:52]=hole,
obs[52:104]=board (binary, card_idx=(rank-2)*4+suit_idx, suit=i%4, rankpos=i//4),
obs[104:126]=round one-hot + scalars + action history. So all other consumers of
the 126-d vector are untouched; the encoder just reads it differently.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_SUITS = 4
N_RANKS = 13
N_CARD_SLOTS = 52  # 4*13
HOLE_SLICE = slice(0, 52)
BOARD_SLICE = slice(52, 104)
NONCARD_START = 104


def obs_to_card_planes(obs: torch.Tensor) -> torch.Tensor:
    """Decode obs[..., 126] -> card planes [..., 3, 4, 13].

    Channel 0 = hole, 1 = board, 2 = hole|board. plane[suit, rankpos] layout
    (card_idx i -> suit=i%4, rankpos=i//4, since card_idx=(rank-2)*4+suit).
    Handles arbitrary leading dims (e.g. [T, B]).
    """
    lead = obs.shape[:-1]
    hole = obs[..., HOLE_SLICE]
    board = obs[..., BOARD_SLICE]
    # 52 -> (rankpos=13, suit=4) since index = rankpos*4 + suit; transpose -> (suit, rankpos).
    hole_p = hole.reshape(*lead, N_RANKS, N_SUITS).transpose(-1, -2)
    board_p = board.reshape(*lead, N_RANKS, N_SUITS).transpose(-1, -2)
    both_p = torch.clamp(hole_p + board_p, max=1.0)
    return torch.stack([hole_p, board_p, both_p], dim=-3)


class CardActionEncoder(nn.Module):
    """Card-plane CNN + non-card MLP, fused to a fixed-size embedding (torso replacement)."""

    def __init__(
        self,
        obs_dim: int = 126,
        out_dim: int = 256,
        conv_channels: tuple[int, ...] = (32, 64),
        noncard_hidden: tuple[int, ...] = (128,),
    ):
        super().__init__()
        if obs_dim < NONCARD_START:
            raise ValueError(f"obs_dim {obs_dim} too small for card layout (need >= {NONCARD_START})")
        self.obs_dim = int(obs_dim)
        self.out_dim = int(out_dim)
        self.n_noncard = self.obs_dim - NONCARD_START

        conv: list[nn.Module] = []
        prev = 3
        for ch in conv_channels:
            conv.append(nn.Conv2d(prev, ch, kernel_size=3, padding=1))
            conv.append(nn.ReLU())
            prev = ch
        self.card_conv = nn.Sequential(*conv)
        self.card_flat_dim = prev * N_SUITS * N_RANKS

        mlp: list[nn.Module] = []
        p = self.n_noncard
        for h in noncard_hidden:
            mlp.append(nn.Linear(p, h))
            mlp.append(nn.ReLU())
            p = h
        self.noncard_mlp = nn.Sequential(*mlp)
        self.noncard_out = p

        self.fuse = nn.Sequential(
            nn.Linear(self.card_flat_dim + self.noncard_out, self.out_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        lead = obs.shape[:-1]
        planes = obs_to_card_planes(obs)                     # [..., 3, 4, 13]
        x = planes.reshape(-1, 3, N_SUITS, N_RANKS)
        x = self.card_conv(x).reshape(x.shape[0], -1)        # [N, card_flat_dim]
        nc = obs[..., NONCARD_START:self.obs_dim].reshape(-1, self.n_noncard)
        nc = self.noncard_mlp(nc)                            # [N, noncard_out]
        h = self.fuse(torch.cat([x, nc], dim=-1))            # [N, out_dim]
        return h.reshape(*lead, self.out_dim)
