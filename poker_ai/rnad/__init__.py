"""Game-agnostic R-NaD (Regularized Nash Dynamics / DeepNash) in PyTorch.

A faithful port of the canonical DeepMind R-NaD recipe (vendored at scripts/vendor/rnad.py)
to a reusable harness: a game-agnostic learner core (`functional`, `network`, `solver`) plus
pluggable trajectory collectors (`collector`). Validated on the simplified card game (Leduc/Kuhn)
where exact NashConv is computable; the same core later targets full-NLHE via the cuda collector.

The learner consumes only abstract batched trajectory tensors ([T, B, ...]): obs, legal mask,
player_id, valid, rewards, sampled action one-hot, behavior policy. Nothing here is poker-specific.
"""
from poker_ai.rnad.network import RNaDNetwork
from poker_ai.rnad.solver import RNaDConfig, RNaDSolver
from poker_ai.rnad.collector import Trajectory, LeducTreeCollector, GPUTreeCollector

__all__ = [
    "RNaDNetwork",
    "RNaDConfig",
    "RNaDSolver",
    "Trajectory",
    "LeducTreeCollector",
    "GPUTreeCollector",
]
