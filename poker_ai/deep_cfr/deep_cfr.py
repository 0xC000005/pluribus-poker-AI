"""Single Deep CFR (SD-CFR) algorithm.

Implements external-sampling MCCFR where regrets are stored in a neural
network instead of a table. The value network is retrained from scratch
each CFR iteration using samples collected during game tree traversal.

Reference: Brown et al., "Deep Counterfactual Regret Minimization" (2019)
           https://arxiv.org/abs/1811.00164
"""
from __future__ import annotations

import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poker_ai.deep_cfr.buffer import ReservoirBuffer
from poker_ai.deep_cfr.networks import ValueNetwork
from poker_ai.deep_cfr.policy_targets import (
    PolicyTargetBuffer,
    PolicyReservoirBuffer,
    masked_policy_cross_entropy,
)
from poker_ai.games.full_deck.state import (
    N_ACTIONS,
    N_FEATURES,
    ACTION_TO_INDEX,
    PokerState,
    new_game,
)

logger = logging.getLogger("poker_ai.deep_cfr")


def regret_match(advantages: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Convert advantage values to a strategy via regret matching.

    Parameters
    ----------
    advantages : ndarray of shape (N_ACTIONS,)
        Predicted advantages from the value network.
    legal_mask : ndarray of shape (N_ACTIONS,)
        Binary mask: 1 for legal actions, 0 for illegal.

    Returns
    -------
    strategy : ndarray of shape (N_ACTIONS,)
        Probability distribution over actions.
    """
    # Zero out illegal actions.
    positive = np.maximum(advantages, 0) * legal_mask
    total = positive.sum()
    if total > 0:
        return positive / total
    # Uniform over legal actions.
    return legal_mask / legal_mask.sum()


def get_legal_mask(state: PokerState) -> np.ndarray:
    """Get a binary mask of legal actions for the current state."""
    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    for action in state.legal_actions:
        if action is not None and action in ACTION_TO_INDEX:
            mask[ACTION_TO_INDEX[action]] = 1.0
    return mask


def traverse(
    state: PokerState,
    traverser: int,
    value_net: ValueNetwork,
    buffer: ReservoirBuffer,
    iteration: int,
    device: torch.device,
    strategy_buffer: PolicyReservoirBuffer | None = None,
) -> float:
    """External-sampling MCCFR traversal.

    Recursively walks the game tree. At the traverser's decision nodes,
    all actions are explored and counterfactual regrets are computed and
    stored in the buffer. At opponent nodes, a single action is sampled
    from the current strategy.

    Parameters
    ----------
    state : PokerState
        Current game state.
    traverser : int
        Index of the player we're computing regrets for.
    value_net : ValueNetwork
        Current value network (used to get strategy at opponent nodes).
    buffer : ReservoirBuffer
        Buffer to store (features, iteration, advantages) samples.
    iteration : int
        Current CFR iteration number.
    device : torch.device
        Device for neural network inference.

    Returns
    -------
    value : float
        Expected value of this state for the traverser.
    """
    # Terminal node: return payoff.
    if state.is_terminal:
        return float(state.payout[traverser])

    current_player = state.player_i

    # If current player is inactive (folded), skip them.
    if not state.current_player.is_active:
        return traverse(
            state.apply_action(None), traverser, value_net, buffer,
            iteration, device, strategy_buffer,
        )

    features = state.to_feature_vector()
    legal_mask = get_legal_mask(state)
    legal_actions = [a for a in state.legal_actions if a is not None]

    if current_player == traverser:
        # Traverser node: explore ALL legal actions.
        advantages_pred = value_net.predict(
            torch.from_numpy(features), device
        ).cpu().numpy()
        strategy = regret_match(advantages_pred, legal_mask)

        action_values = {}
        for action in legal_actions:
            child_state = state.apply_action(action)
            action_values[action] = traverse(
                child_state, traverser, value_net, buffer, iteration, device,
                strategy_buffer,
            )

        # Compute counterfactual regrets.
        state_value = sum(
            strategy[ACTION_TO_INDEX[a]] * action_values[a]
            for a in legal_actions
        )
        regrets = np.zeros(N_ACTIONS, dtype=np.float32)
        for action in legal_actions:
            idx = ACTION_TO_INDEX[action]
            regrets[idx] = action_values[action] - state_value

        # Normalize regrets to [-1, 1] range for stable NN training.
        # Use initial stack size for scale (consistent with fast_traverse).
        try:
            init_chips = float(getattr(state, "_initial_n_chips", 10000))
        except Exception:
            init_chips = 10000.0
        if init_chips <= 0:
            init_chips = 1.0
        regrets /= init_chips
        # Add sample to buffer.
        buffer.add(features, iteration, regrets)
        return state_value

    else:
        # Opponent node: sample a single action from current strategy.
        advantages_pred = value_net.predict(
            torch.from_numpy(features), device
        ).cpu().numpy()
        strategy = regret_match(advantages_pred, legal_mask)
        if strategy_buffer is not None:
            strategy_buffer.add(
                features,
                legal_mask,
                strategy,
                weight=float(max(iteration, 1)),
            )

        # Sample action.
        action_probs = [strategy[ACTION_TO_INDEX[a]] for a in legal_actions]
        action_probs = np.array(action_probs, dtype=np.float64)
        action_probs /= action_probs.sum()  # Re-normalize for safety.
        action = np.random.choice(legal_actions, p=action_probs)

        child_state = state.apply_action(action)
        return traverse(
            child_state, traverser, value_net, buffer, iteration, device,
            strategy_buffer,
        )


def train_value_network(
    buffer: ReservoirBuffer,
    input_dim: int = N_FEATURES,
    hidden_dim: int = 256,
    output_dim: int = N_ACTIONS,
    n_epochs: int = 2000,
    batch_size: int = 2048,
    lr: float = 0.001,
    device: torch.device | None = None,
    n_layers: int = 2,
    policy_target_buffer: PolicyTargetBuffer | None = None,
    policy_target_weight: float = 0.0,
    policy_target_batch_size: int | None = None,
    average_strategy_buffer: PolicyTargetBuffer | PolicyReservoirBuffer | None = None,
    average_strategy_weight: float = 0.0,
    average_strategy_batch_size: int | None = None,
) -> ValueNetwork:
    """Train a new value network from scratch on the buffer contents.

    The network is initialized with random weights each time (not
    fine-tuned from the previous iteration). This yields better
    convergence per the Deep CFR paper.

    Parameters
    ----------
    buffer : ReservoirBuffer
    input_dim, hidden_dim, output_dim : int
    n_epochs : int
        Number of SGD training steps (not full epochs over data).
    batch_size : int
    lr : float
    device : torch.device
    n_layers : int
        Number of hidden layers in the value network.

    Returns
    -------
    net : ValueNetwork
        Freshly trained value network.
    """
    if device is None:
        device = torch.device("cpu")

    if device.type == "cuda":
        # Allow tensor cores to accelerate dense matmuls on modern NVIDIA GPUs.
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    net = ValueNetwork(
        input_dim, hidden_dim, output_dim, n_layers=n_layers
    ).to(device)

    use_cuda = device.type == "cuda"
    use_amp = use_cuda
    amp_dtype = (
        torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    # Fused optimizer is substantially faster on CUDA for many small steps.
    if use_cuda:
        try:
            optimizer = optim.AdamW(
                net.parameters(), lr=lr, weight_decay=0.0, fused=True
            )
        except (TypeError, RuntimeError):
            optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(net.parameters(), lr=lr)

    # Compile the model graph when available; this reduces Python/kernel
    # launch overhead in the per-step training loop.
    compile_enabled = os.getenv("POKER_AI_COMPILE_VALUE_NET", "0").lower() in {
        "1", "true", "yes", "on",
    }
    if use_cuda and compile_enabled and hasattr(torch, "compile"):
        try:
            net = torch.compile(net, mode="reduce-overhead")
        except Exception:
            logger.warning("torch.compile unavailable for value net; using eager mode.")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    policy_target_weight = float(policy_target_weight)
    use_policy_targets = (
        policy_target_buffer is not None
        and policy_target_weight > 0
        and getattr(policy_target_buffer, "size", 0) > 0
    )
    policy_target_batch_size = int(policy_target_batch_size or batch_size)
    average_strategy_weight = float(average_strategy_weight)
    use_average_strategy_targets = (
        average_strategy_buffer is not None
        and average_strategy_weight > 0
        and getattr(average_strategy_buffer, "size", 0) > 0
    )
    average_strategy_batch_size = int(average_strategy_batch_size or batch_size)

    net.train()
    total_loss = 0.0
    for step in range(n_epochs):
        features, iterations, advantages = buffer.sample_batch(
            batch_size, device
        )
        # Compact GPU replay caches may store targets in half precision to fit
        # larger buffers in VRAM. Keep loss targets in fp32 for stable scaling.
        if advantages.dtype != torch.float32:
            advantages = advantages.float()
        if iterations.dtype != torch.float32:
            iterations = iterations.float()

        # Weight samples by iteration (linear CFR weighting).
        # Later iterations get higher weight.
        weights = iterations / iterations.max().clamp(min=1)
        weights = weights.unsqueeze(1)  # (batch, 1)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            adv_pred, pol_logits = net.forward_with_policy(features)
            # Advantage loss (MSE on regrets).
            loss_adv = (weights * (adv_pred - advantages) ** 2).mean()

            loss = loss_adv
            if use_average_strategy_targets:
                avg_batch = average_strategy_buffer.sample_batch(
                    average_strategy_batch_size,
                    device,
                )
                _, avg_logits = net.forward_with_policy(avg_batch.features)
                loss_avg = masked_policy_cross_entropy(
                    avg_logits,
                    avg_batch.legal_masks,
                    avg_batch.target_probs,
                    weights=avg_batch.weights,
                )
                loss = loss + average_strategy_weight * loss_avg
            else:
                # Fallback for legacy runs without traversal-collected strategy
                # memory: train a weak policy head from positive sampled regrets.
                with torch.no_grad():
                    pos = torch.clamp(advantages, min=0.0)
                    denom = pos.sum(dim=1, keepdim=True).clamp(min=1e-8)
                    policy_target = pos / denom
                logp = torch.log_softmax(pol_logits, dim=1)
                loss_pol = -(weights * (policy_target * logp).sum(dim=1)).mean()
                loss = loss + 0.1 * loss_pol
            if use_policy_targets:
                target_batch = policy_target_buffer.sample_batch(
                    policy_target_batch_size,
                    device,
                )
                _, target_logits = net.forward_with_policy(target_batch.features)
                loss_search = masked_policy_cross_entropy(
                    target_logits,
                    target_batch.legal_masks,
                    target_batch.target_probs,
                    weights=target_batch.weights,
                )
                loss = loss + policy_target_weight * loss_search

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(n_epochs, 1)
    logger.info(f"Value network trained: avg_loss={avg_loss:.6f}")
    return net


class DeepCFRTrainer:
    """Orchestrates the Single Deep CFR training loop.

    Each iteration:
    1. For each player, traverse the game tree collecting regret samples
    2. Retrain the value network from scratch on accumulated samples

    Parameters
    ----------
    n_players : int
        Number of players (default: 2).
    buffer_capacity : int
        Maximum samples in the reservoir buffer.
    hidden_dim : int
        Hidden layer size for the value network.
    batch_size : int
        Mini-batch size for training.
    lr : float
        Learning rate.
    n_training_steps : int
        SGD steps per value network training.
    n_traversals : int
        Number of game tree traversals per player per iteration.
    device : torch.device or None
        GPU/CPU device. Auto-detected if None.
    """

    def __init__(
        self,
        n_players: int = 2,
        buffer_capacity: int = 2_000_000,
        hidden_dim: int = 256,
        batch_size: int = 2048,
        lr: float = 0.001,
        n_training_steps: int = 2000,
        n_traversals: int = 500,
        device: torch.device | None = None,
        policy_target_buffer: PolicyTargetBuffer | None = None,
        policy_target_weight: float = 0.0,
        policy_target_batch_size: int | None = None,
        average_strategy_memory_capacity: int | None = None,
        average_strategy_weight: float = 0.0,
        average_strategy_batch_size: int | None = None,
    ):
        self.n_players = n_players
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lr = lr
        self.n_training_steps = n_training_steps
        self.n_traversals = n_traversals
        self.policy_target_buffer = policy_target_buffer
        self.policy_target_weight = float(policy_target_weight)
        self.policy_target_batch_size = policy_target_batch_size
        self.average_strategy_weight = float(average_strategy_weight)
        self.average_strategy_batch_size = average_strategy_batch_size
        self.strategy_buffer = PolicyReservoirBuffer(
            int(average_strategy_memory_capacity or buffer_capacity)
        )

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        # One buffer per player.
        self.buffers: List[ReservoirBuffer] = [
            ReservoirBuffer(buffer_capacity) for _ in range(n_players)
        ]
        # Single shared value network (SD-CFR).
        self.value_net = ValueNetwork(
            N_FEATURES, hidden_dim, N_ACTIONS
        ).to(self.device)
        self.iteration = 0

    def run_iteration(self):
        """Run one full CFR iteration.

        For each player, perform n_traversals game tree traversals,
        then retrain the value network from scratch.
        """
        self.iteration += 1
        self.value_net.eval()

        # Traverse for each player.
        for player_i in range(self.n_players):
            for _ in range(self.n_traversals):
                state = new_game(self.n_players)
                traverse(
                    state,
                    traverser=player_i,
                    value_net=self.value_net,
                    buffer=self.buffers[player_i],
                    iteration=self.iteration,
                    device=self.device,
                    strategy_buffer=(
                        self.strategy_buffer if self.average_strategy_weight > 0 else None
                    ),
                )

        # Combine all player buffers for training.
        combined = self._combine_buffers()

        if len(combined) > 0:
            self.value_net = train_value_network(
                buffer=combined,
                hidden_dim=self.hidden_dim,
                n_epochs=self.n_training_steps,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self.device,
                policy_target_buffer=self.policy_target_buffer,
                policy_target_weight=self.policy_target_weight,
                policy_target_batch_size=self.policy_target_batch_size,
                average_strategy_buffer=self.strategy_buffer,
                average_strategy_weight=self.average_strategy_weight,
                average_strategy_batch_size=self.average_strategy_batch_size,
            )

    def _combine_buffers(self) -> ReservoirBuffer:
        """Merge all player buffers into a single buffer for training."""
        total_size = sum(len(b) for b in self.buffers)
        combined = ReservoirBuffer(total_size)
        for buf in self.buffers:
            for i in range(buf.size):
                combined.add(
                    buf.features[i],
                    int(buf.iterations[i]),
                    buf.advantages[i],
                )
        return combined

    def get_strategy(self, state: PokerState) -> np.ndarray:
        """Get the current strategy for a game state.

        Parameters
        ----------
        state : PokerState

        Returns
        -------
        strategy : ndarray of shape (N_ACTIONS,)
            Probability distribution over actions.
        """
        self.value_net.eval()
        features = state.to_feature_vector()
        legal_mask = get_legal_mask(state)
        advantages = self.value_net.predict(
            torch.from_numpy(features), self.device
        ).cpu().numpy()
        return regret_match(advantages, legal_mask)

    def save(self, path: str):
        """Save the trainer state to disk."""
        torch.save(
            {
                "value_net": self.value_net.state_dict(),
                "iteration": self.iteration,
                "n_players": self.n_players,
                "hidden_dim": self.hidden_dim,
                "average_strategy_target_size": int(self.strategy_buffer.size),
                "average_strategy_weight": self.average_strategy_weight,
                "buffer_sizes": [len(b) for b in self.buffers],
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")

    @classmethod
    def load(cls, path: str, device: torch.device | None = None) -> DeepCFRTrainer:
        """Load a trainer from a checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        device : torch.device, optional

        Returns
        -------
        trainer : DeepCFRTrainer
        """
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        trainer = cls(
            n_players=checkpoint["n_players"],
            hidden_dim=checkpoint["hidden_dim"],
            device=device,
        )
        trainer.value_net.load_state_dict(checkpoint["value_net"])
        trainer.iteration = checkpoint["iteration"]
        logger.info(
            f"Loaded checkpoint from {path} (iteration {trainer.iteration})"
        )
        return trainer
