"""Local evaluation helpers for poker autoresearch gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poker_ai.deep_cfr.deep_cfr import regret_match
from poker_ai.deep_cfr.fast_state import N_ACTIONS, N_FEATURES
from poker_ai.deep_cfr.networks import ValueNetwork, PolicyNetwork
from poker_ai.deep_cfr.vectorized_env import VectorizedPokerEnv


BIG_BLIND = 100


@dataclass(frozen=True)
class LoadedValueNetwork:
    value_net: ValueNetwork
    metadata: dict[str, Any]
    average_policy_net: PolicyNetwork | None = None


def remap_legacy_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map legacy ``net.<idx>`` checkpoints to ``trunk`` + ``adv_head``."""
    if not any(key.startswith("net.") for key in state):
        return state

    linear_indices = sorted(
        {int(key.split(".")[1]) for key in state if key.startswith("net.")}
    )
    *trunk_indices, head_index = linear_indices
    trunk_index_set = set(trunk_indices)
    remapped: dict[str, torch.Tensor] = {}

    for key, value in state.items():
        if not key.startswith("net."):
            remapped[key] = value
            continue
        _, idx_text, param = key.split(".", 2)
        idx = int(idx_text)
        if idx == head_index:
            remapped[f"adv_head.{param}"] = value
        elif idx in trunk_index_set:
            remapped[f"trunk.{idx}.{param}"] = value
    return remapped


def load_value_network_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> LoadedValueNetwork:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = int(checkpoint.get("hidden_dim", 256))
    n_layers = int(checkpoint.get("n_layers", 2))
    n_players = int(checkpoint.get("n_players", 2))
    initial_chips = int(checkpoint.get("initial_chips", 10000))
    uses_betting_history = bool(checkpoint.get("uses_betting_history", False))

    value_net = ValueNetwork(
        N_FEATURES,
        hidden_dim,
        N_ACTIONS,
        n_layers=n_layers,
        use_betting_history=uses_betting_history,
    ).to(device)
    state = remap_legacy_state_dict(checkpoint["value_net"])
    missing, unexpected = value_net.load_state_dict(state, strict=False)
    allowed_missing = {
        "policy_head.weight",
        "policy_head.bias",
        "seq_proj.weight",
        "seq_proj.bias",
    }
    disallowed_missing = sorted(set(missing) - allowed_missing)
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "Checkpoint is incompatible: "
            f"missing={disallowed_missing}, unexpected={list(unexpected)}"
        )
    has_policy_head = not {
        "policy_head.weight",
        "policy_head.bias",
    }.intersection(missing)
    average_policy_net = None
    if checkpoint.get("average_policy_net") is not None:
        average_policy_net = PolicyNetwork(
            N_FEATURES,
            hidden_dim,
            N_ACTIONS,
            n_layers=n_layers,
            use_betting_history=uses_betting_history,
        ).to(device)
        average_policy_net.load_state_dict(checkpoint["average_policy_net"])
        average_policy_net.eval()
        value_net.average_policy_net = average_policy_net
    value_net.eval()

    return LoadedValueNetwork(
        value_net=value_net,
        metadata={
            "checkpoint": str(checkpoint_path),
            "iteration": checkpoint.get("iteration"),
            "checkpoint_iteration": checkpoint.get("iteration"),
            "n_players": n_players,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "initial_chips": initial_chips,
            "has_policy_head": has_policy_head,
            "has_average_policy_net": average_policy_net is not None,
            "uses_betting_history": uses_betting_history,
        },
        average_policy_net=average_policy_net,
    )


def assert_strategy_source_supported(
    loaded: LoadedValueNetwork,
    strategy_source: str,
) -> None:
    if strategy_source == "policy-head" and not loaded.metadata.get("has_policy_head"):
        checkpoint = loaded.metadata.get("checkpoint", "<unknown>")
        raise RuntimeError(
            f"Checkpoint {checkpoint} does not contain a trained policy head; "
            "use --strategy-source regret or retrain/create an incumbent with "
            "policy_head weights."
        )
    if (
        strategy_source == "average-policy"
        and not loaded.metadata.get("has_average_policy_net")
    ):
        checkpoint = loaded.metadata.get("checkpoint", "<unknown>")
        raise RuntimeError(
            f"Checkpoint {checkpoint} does not contain a trained average policy net; "
            "use --strategy-source regret or train with average-strategy collection."
        )
    if strategy_source not in {"regret", "policy-head", "average-policy"}:
        raise ValueError(f"Unknown strategy source: {strategy_source}")


def _strategies_from_network(
    value_net: ValueNetwork,
    features: np.ndarray,
    masks: list[np.ndarray],
    device: torch.device,
    *,
    strategy_source: str,
) -> list[np.ndarray]:
    """Return legal action probabilities from the requested learned source."""
    feature_tensor = torch.from_numpy(features).to(device)
    if strategy_source == "regret":
        with torch.no_grad():
            advantages = value_net(feature_tensor).cpu().numpy()
        return [regret_match(advantages[i], masks[i]) for i in range(len(masks))]

    if strategy_source == "policy-head":
        with torch.no_grad():
            _, logits_t = value_net.forward_with_policy(feature_tensor)
        logits = logits_t.cpu().numpy().astype(np.float64)
    elif strategy_source == "average-policy":
        average_policy_net = getattr(value_net, "average_policy_net", None)
        if average_policy_net is None:
            raise RuntimeError("average-policy strategy source requires average_policy_net")
        average_policy_net.eval()
        with torch.no_grad():
            logits_t = average_policy_net(feature_tensor)
        logits = logits_t.cpu().numpy().astype(np.float64)
    else:
        raise ValueError(f"Unknown strategy source: {strategy_source}")

    if strategy_source in {"policy-head", "average-policy"}:
        strategies: list[np.ndarray] = []
        for i, mask in enumerate(masks):
            masked_logits = np.where(mask > 0, logits[i], -1e9)
            shifted = masked_logits - np.max(masked_logits)
            probs = np.exp(shifted) * mask
            total = probs.sum()
            if total > 0:
                strategies.append(probs / total)
            else:
                strategies.append(mask / mask.sum())
        return strategies

    raise ValueError(f"Unknown strategy source: {strategy_source}")


def _evaluate_payouts_vs_random(
    value_net: ValueNetwork,
    device: torch.device,
    *,
    n_games: int,
    n_players: int,
    initial_chips: int,
    strategy_source: str,
) -> np.ndarray:
    env = VectorizedPokerEnv(n_games, n_players, initial_chips=initial_chips)
    env.reset()
    value_net.eval()

    max_steps = n_games * 50
    step_count = 0

    while not env.done.all() and step_count < max_steps:
        step_count += 1

        for i in range(n_games):
            if env.done[i]:
                continue
            state = env.states[i]
            while not state.is_terminal and not state.active[state.current_player_i]:
                child = state.copy()
                child.apply_action(None)
                env.states[i] = child
                state = child
                if state.is_terminal:
                    env.done[i] = True

        if env.done.all():
            break

        agent_indices: list[int] = []
        random_indices: list[int] = []
        for i in range(n_games):
            if env.done[i]:
                continue
            if env.states[i].current_player_i == 0:
                agent_indices.append(i)
            else:
                random_indices.append(i)

        for i in random_indices:
            mask = env.states[i].get_legal_mask()
            legal = np.where(mask > 0)[0]
            env.step_single(i, int(np.random.choice(legal)))

        if agent_indices:
            features = np.stack([env.states[i].to_feature_vector() for i in agent_indices])
            masks = [env.states[i].get_legal_mask() for i in agent_indices]
            strategies = _strategies_from_network(
                value_net,
                features,
                masks,
                device,
                strategy_source=strategy_source,
            )

            for j, i in enumerate(agent_indices):
                legal = np.where(masks[j] > 0)[0]
                probs = np.array([strategies[j][action] for action in legal], dtype=np.float64)
                probs /= probs.sum()
                env.step_single(i, int(np.random.choice(legal, p=probs)))

    return env.get_payouts(0)


def _step_model_group(
    env: VectorizedPokerEnv,
    indices: list[int],
    value_net: ValueNetwork,
    device: torch.device,
    *,
    strategy_source: str,
) -> None:
    if not indices:
        return
    features = np.stack([env.states[i].to_feature_vector() for i in indices])
    masks = [env.states[i].get_legal_mask() for i in indices]
    strategies = _strategies_from_network(
        value_net,
        features,
        masks,
        device,
        strategy_source=strategy_source,
    )

    for j, env_index in enumerate(indices):
        legal = np.where(masks[j] > 0)[0]
        probs = np.array([strategies[j][action] for action in legal], dtype=np.float64)
        probs /= probs.sum()
        env.step_single(env_index, int(np.random.choice(legal, p=probs)))


def _evaluate_head_to_head_payouts(
    player0_net: ValueNetwork,
    player1_net: ValueNetwork,
    device: torch.device,
    *,
    n_games: int,
    initial_chips: int,
    player0_strategy_source: str,
    player1_strategy_source: str,
) -> np.ndarray:
    env = VectorizedPokerEnv(n_games, 2, initial_chips=initial_chips)
    env.reset()
    player0_net.eval()
    player1_net.eval()

    max_steps = n_games * 50
    step_count = 0

    while not env.done.all() and step_count < max_steps:
        step_count += 1

        for i in range(n_games):
            if env.done[i]:
                continue
            state = env.states[i]
            while not state.is_terminal and not state.active[state.current_player_i]:
                child = state.copy()
                child.apply_action(None)
                env.states[i] = child
                state = child
                if state.is_terminal:
                    env.done[i] = True

        if env.done.all():
            break

        player0_indices: list[int] = []
        player1_indices: list[int] = []
        for i in range(n_games):
            if env.done[i]:
                continue
            if env.states[i].current_player_i == 0:
                player0_indices.append(i)
            else:
                player1_indices.append(i)

        _step_model_group(
            env,
            player0_indices,
            player0_net,
            device,
            strategy_source=player0_strategy_source,
        )
        _step_model_group(
            env,
            player1_indices,
            player1_net,
            device,
            strategy_source=player1_strategy_source,
        )

    return env.get_payouts(0)


def evaluate_value_net_vs_random(
    value_net: ValueNetwork,
    device: torch.device,
    *,
    n_games: int,
    n_players: int,
    initial_chips: int,
    seed: int,
    checkpoint_metadata: dict[str, Any],
    strategy_source: str = "regret",
) -> dict[str, Any]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    payouts = _evaluate_payouts_vs_random(
        value_net,
        device,
        n_games=n_games,
        n_players=n_players,
        initial_chips=initial_chips,
        strategy_source=strategy_source,
    )
    avg = float(payouts.mean())
    std = float(payouts.std(ddof=1)) if len(payouts) > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(len(payouts))) if len(payouts) > 1 else 0.0
    return {
        "passed": bool(np.isfinite(payouts).all()),
        "n_games": int(n_games),
        "n_players": int(n_players),
        "initial_chips": int(initial_chips),
        "seed": int(seed),
        "strategy_source": strategy_source,
        "avg_chips_per_hand": avg,
        "ci95_chips_per_hand": ci95,
        "lower95_chips_per_hand": avg - ci95,
        "mbb_per_hand": avg / BIG_BLIND * 1000.0,
        "ci95_mbb_per_hand": ci95 / BIG_BLIND * 1000.0,
        "win_rate": float(np.mean(payouts > 0)),
        "loss_rate": float(np.mean(payouts < 0)),
        "checkpoint": checkpoint_metadata.get("checkpoint"),
        "checkpoint_iteration": checkpoint_metadata.get(
            "checkpoint_iteration", checkpoint_metadata.get("iteration")
        ),
        "hidden_dim": checkpoint_metadata.get("hidden_dim"),
        "n_layers": checkpoint_metadata.get("n_layers"),
    }


def evaluate_value_nets_head_to_head(
    candidate_net: ValueNetwork,
    baseline_net: ValueNetwork,
    device: torch.device,
    *,
    n_games: int,
    initial_chips: int,
    seed: int,
    candidate_metadata: dict[str, Any],
    baseline_metadata: dict[str, Any],
    strategy_source: str = "regret",
    candidate_strategy_source: str | None = None,
    baseline_strategy_source: str | None = None,
) -> dict[str, Any]:
    candidate_strategy_source = candidate_strategy_source or strategy_source
    baseline_strategy_source = baseline_strategy_source or strategy_source
    np.random.seed(seed)
    torch.manual_seed(seed)
    candidate_seat0 = _evaluate_head_to_head_payouts(
        candidate_net,
        baseline_net,
        device,
        n_games=n_games,
        initial_chips=initial_chips,
        player0_strategy_source=candidate_strategy_source,
        player1_strategy_source=baseline_strategy_source,
    )

    np.random.seed(seed)
    torch.manual_seed(seed)
    baseline_seat0 = _evaluate_head_to_head_payouts(
        baseline_net,
        candidate_net,
        device,
        n_games=n_games,
        initial_chips=initial_chips,
        player0_strategy_source=baseline_strategy_source,
        player1_strategy_source=candidate_strategy_source,
    )
    paired = (candidate_seat0 - baseline_seat0) / 2.0
    avg = float(paired.mean())
    std = float(paired.std(ddof=1)) if len(paired) > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(len(paired))) if len(paired) > 1 else 0.0
    return {
        "passed": bool(np.isfinite(paired).all()),
        "mode": "duplicate_swapped_head_to_head",
        "n_games": int(n_games * 2),
        "n_duplicate_pairs": int(n_games),
        "n_players": 2,
        "initial_chips": int(initial_chips),
        "seed": int(seed),
        "strategy_source": (
            strategy_source
            if candidate_strategy_source == baseline_strategy_source == strategy_source
            else "mixed"
        ),
        "candidate_strategy_source": candidate_strategy_source,
        "baseline_strategy_source": baseline_strategy_source,
        "avg_chips_per_hand": avg,
        "ci95_chips_per_hand": ci95,
        "paired_delta_lower95_chips_per_hand": avg - ci95,
        "mbb_per_hand": avg / BIG_BLIND * 1000.0,
        "ci95_mbb_per_hand": ci95 / BIG_BLIND * 1000.0,
        "candidate_checkpoint": candidate_metadata.get("checkpoint"),
        "baseline_checkpoint": baseline_metadata.get("checkpoint"),
        "candidate_iteration": candidate_metadata.get("checkpoint_iteration"),
        "baseline_iteration": baseline_metadata.get("checkpoint_iteration"),
        "candidate_hidden_dim": candidate_metadata.get("hidden_dim"),
        "baseline_hidden_dim": baseline_metadata.get("hidden_dim"),
        "candidate_n_layers": candidate_metadata.get("n_layers"),
        "baseline_n_layers": baseline_metadata.get("n_layers"),
        "promotable": False,
        "promotion_blockers": [
            "local_head_to_head_requires_slumbot_confirmation",
        ],
    }


def aggregate_seed_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate fixed-seed local evaluations into one metrics block."""
    if not runs:
        raise ValueError("Cannot aggregate an empty run list.")
    if len(runs) == 1:
        return runs[0]

    avgs = [float(run["avg_chips_per_hand"]) for run in runs]
    mean = sum(avgs) / len(avgs)
    variance = (
        sum((value - mean) ** 2 for value in avgs) / (len(avgs) - 1)
        if len(avgs) > 1
        else 0.0
    )
    ci95 = 1.96 * math.sqrt(variance) / math.sqrt(len(avgs))
    first = runs[0]
    metrics = {
        "passed": all(bool(run["passed"]) for run in runs),
        "n_runs": len(runs),
        "n_games_per_run": int(first["n_games"]),
        "n_games": sum(int(run["n_games"]) for run in runs),
        "n_players": int(first["n_players"]),
        "initial_chips": int(first["initial_chips"]),
        "avg_chips_per_hand": mean,
        "ci95_chips_per_hand_across_seeds": ci95,
        "lower95_chips_per_hand_across_seeds": mean - ci95,
        "runs": runs,
    }
    for key in (
        "checkpoint",
        "checkpoint_iteration",
        "hidden_dim",
        "n_layers",
        "mode",
        "candidate_checkpoint",
        "baseline_checkpoint",
        "candidate_iteration",
        "baseline_iteration",
        "candidate_hidden_dim",
        "baseline_hidden_dim",
        "candidate_n_layers",
        "baseline_n_layers",
        "promotable",
        "promotion_blockers",
        "strategy_source",
        "candidate_strategy_source",
        "baseline_strategy_source",
    ):
        if key in first and first[key] is not None:
            metrics[key] = first[key]
    if "paired_delta_lower95_chips_per_hand" in first:
        metrics["paired_delta_lower95_chips_per_hand_across_seeds"] = (
            metrics["lower95_chips_per_hand_across_seeds"]
        )
    return metrics


def _paired_seed_deltas(candidate: dict[str, Any], baseline: dict[str, Any]) -> list[float]:
    candidate_runs = candidate.get("runs") or []
    baseline_runs = baseline.get("runs") or []
    if not candidate_runs or len(candidate_runs) != len(baseline_runs):
        return []
    deltas: list[float] = []
    for candidate_run, baseline_run in zip(candidate_runs, baseline_runs, strict=True):
        if candidate_run.get("seed") != baseline_run.get("seed"):
            return []
        deltas.append(
            float(candidate_run["avg_chips_per_hand"])
            - float(baseline_run["avg_chips_per_hand"])
        )
    return deltas


def compare_checkpoint_metrics(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Compare two local random-opponent evaluations without promoting strength.

    This is deliberately a mechanical guardrail. It can show regressions or
    large candidate deltas under the same local protocol, but promotion still
    requires harder incumbent/Slumbot evidence.
    """
    delta_avg = float(candidate["avg_chips_per_hand"]) - float(baseline["avg_chips_per_hand"])
    paired_deltas = _paired_seed_deltas(candidate, baseline)
    metrics: dict[str, Any] = {
        "passed": bool(candidate.get("passed")) and bool(baseline.get("passed")),
        "candidate": candidate,
        "baseline": baseline,
        "delta_avg_chips_per_hand": delta_avg,
        "promotable": False,
        "promotion_blockers": [
            "local_random_comparison_is_not_strategy_quality",
            "requires_incumbent_or_slumbot_confidence_gate",
        ],
    }

    if paired_deltas:
        mean = sum(paired_deltas) / len(paired_deltas)
        variance = (
            sum((value - mean) ** 2 for value in paired_deltas) / (len(paired_deltas) - 1)
            if len(paired_deltas) > 1
            else 0.0
        )
        ci95 = 1.96 * math.sqrt(variance) / math.sqrt(len(paired_deltas))
        metrics.update(
            {
                "paired_delta_chips_per_hand": paired_deltas,
                "paired_delta_avg_chips_per_hand": mean,
                "paired_delta_ci95_chips_per_hand": ci95,
                "paired_delta_lower95_chips_per_hand": mean - ci95,
            }
        )

    return metrics
