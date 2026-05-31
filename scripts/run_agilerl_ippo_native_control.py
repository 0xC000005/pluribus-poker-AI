#!/usr/bin/env python3
"""Run plug-in AgileRL IPPO on the native PettingZoo poker adapter."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_ai.games.full_deck.state import (  # noqa: E402
    INDEX_TO_ACTION,
    N_ACTIONS,
    N_FEATURES,
    new_game,
)
from poker_ai.research.agilerl_contract import (  # noqa: E402
    make_agilerl_native_parallel_env,
    _package_version,
)
from poker_ai.research.native_nfsp import (  # noqa: E402
    _load_native_checkpoint_networks,
    _network_probs,
    get_legal_mask,
    resolve_device,
    select_action,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run_agilerl_ippo_control(
    *,
    max_steps: int = 1024,
    evo_steps: int = 256,
    eval_steps: int = 128,
    eval_loop: int = 1,
    population_size: int = 1,
    learn_step: int = 256,
    batch_size: int = 128,
    hidden_dim: int = 128,
    update_epochs: int = 4,
    lr: float = 1e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    initial_chips: int = 1000,
    max_steps_per_hand: int = 256,
    seed: int = 20260527,
    device: str = "auto",
    verbose: bool = False,
    output_json: str | None = None,
    checkpoint_out: str | None = None,
) -> dict[str, Any]:
    """Train a tiny AgileRL IPPO control on the native 9-action poker env.

    This function intentionally delegates PPO/IPPO implementation details to
    AgileRL. Local code only adapts the environment contract and records a
    non-promotion control artifact.
    """
    from agilerl.training.train_multi_agent_on_policy import train_multi_agent_on_policy
    from agilerl.utils.utils import create_population

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_info = resolve_device(device)
    resolved_device = str(device_info["resolved_device"])

    env = make_agilerl_native_parallel_env(
        seed=int(seed),
        initial_chips=int(initial_chips),
        max_steps_per_hand=int(max_steps_per_hand),
    )
    observation_spaces = {
        str(agent): env.observation_space(agent) for agent in env.possible_agents
    }
    action_spaces = {str(agent): env.action_space(agent) for agent in env.possible_agents}
    init_hp = {
        "AGENT_IDS": [str(agent) for agent in env.possible_agents],
        "BATCH_SIZE": int(batch_size),
        "LR": float(lr),
        "LEARN_STEP": int(learn_step),
        "GAMMA": float(gamma),
        "GAE_LAMBDA": float(gae_lambda),
        "ENT_COEF": float(ent_coef),
        "VF_COEF": float(vf_coef),
        "UPDATE_EPOCHS": int(update_epochs),
    }
    net_config = {
        "encoder_config": {
            "hidden_size": [int(hidden_dim), int(hidden_dim)],
            "init_layers": False,
        },
        "head_config": {"hidden_size": [int(hidden_dim)], "init_layers": False},
    }

    train_start = time.perf_counter()
    population = create_population(
        algo="IPPO",
        net_config=net_config,
        INIT_HP=init_hp,
        observation_space=observation_spaces,
        action_space=action_spaces,
        population_size=int(population_size),
        num_envs=1,
        device=resolved_device,
    )
    trained_population, fitnesses = train_multi_agent_on_policy(
        env=env,
        env_name="poker_ai_native_hu_nlhe",
        algo="IPPO",
        pop=population,
        sum_scores=True,
        INIT_HP=init_hp,
        max_steps=int(max_steps),
        evo_steps=int(evo_steps),
        eval_steps=int(eval_steps),
        eval_loop=int(eval_loop),
        verbose=bool(verbose),
    )
    if resolved_device == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start

    checkpoint_path = None
    if checkpoint_out:
        checkpoint_path = str(checkpoint_out)
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        trained_population[0].save_checkpoint(path)

    metrics: dict[str, Any] = {
        "algorithm": "agilerl_ippo",
        "library": "agilerl",
        "role": "plug_in_multi_agent_policy_value_rl_control",
        "environment": "poker_ai:pettingzoo_full_deck_hu_nlhe",
        "warning": (
            "This is a maintained-library IPPO smoke on the native 9-action "
            "PettingZoo adapter, not Slumbot/SOTA evidence."
        ),
        **device_info,
        "agilerl_version": _package_version("agilerl"),
        "pettingzoo_version": _package_version("pettingzoo"),
        "num_agents": int(len(env.possible_agents)),
        "agents": [str(agent) for agent in env.possible_agents],
        "num_actions": N_ACTIONS,
        "num_features": N_FEATURES,
        "max_steps": int(max_steps),
        "evo_steps": int(evo_steps),
        "eval_steps": int(eval_steps),
        "eval_loop": int(eval_loop),
        "population_size": int(population_size),
        "learn_step": int(learn_step),
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "update_epochs": int(update_epochs),
        "lr": float(lr),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "ent_coef": float(ent_coef),
        "vf_coef": float(vf_coef),
        "fitnesses": _json_safe(fitnesses),
        "agent_steps": _json_safe([getattr(agent, "steps", None) for agent in trained_population]),
        "agent_fitness": _json_safe(
            [getattr(agent, "fitness", None) for agent in trained_population]
        ),
        "train_seconds": float(train_seconds),
        "train_steps_per_second": float(max_steps / max(train_seconds, 1e-9)),
        "checkpoint_path": checkpoint_path,
        "promotion": False,
        "league_eligible": False,
    }

    text = json.dumps(metrics, indent=2, sort_keys=True)
    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return metrics


def _agilerl_checkpoint_action(
    agent: Any,
    *,
    player_i: int,
    features: np.ndarray,
    legal_mask: np.ndarray,
) -> int:
    agent_name = f"player_{int(player_i)}"
    observations = {agent_name: np.asarray(features, dtype=np.float32)}
    infos = {
        agent_name: {
            "action_mask": np.asarray(legal_mask, dtype=np.int8).tolist(),
        }
    }
    actions, _log_probs, _entropy, _values = agent.get_action(observations, infos=infos)
    action_value = np.asarray(actions[agent_name]).reshape(-1)[0]
    return int(action_value)


def evaluate_agilerl_ippo_checkpoint_vs_native_nfsp(
    *,
    candidate_checkpoint: str,
    baseline_checkpoint: str,
    n_games: int = 100,
    device: str = "auto",
    seed: int = 20260527,
) -> dict[str, Any]:
    """Duplicate-swapped H2H: AgileRL IPPO checkpoint versus native NFSP."""
    from agilerl.algorithms import IPPO

    device_info = resolve_device(device)
    resolved_device = torch.device(device_info["resolved_device"])
    candidate_agent = IPPO.load(str(candidate_checkpoint), device=str(resolved_device))
    candidate_agent.set_training_mode(False)
    baseline_payload, _baseline_q, baseline_avg = _load_native_checkpoint_networks(
        str(baseline_checkpoint),
        resolved_device,
    )
    baseline_config = baseline_payload.get("config", {})
    initial_chips = int(baseline_config.get("initial_chips", 1000))
    max_steps_per_hand = int(baseline_config.get("max_steps_per_hand", 256))

    candidate_payoffs: list[float] = []
    candidate_pair_payoffs: list[float] = []
    total_steps = 0
    invalid_candidate_actions = 0
    eval_start = time.perf_counter()
    pair_i = 0
    while len(candidate_payoffs) < int(n_games):
        game_seed = int(seed) + pair_i
        action_seed = int(seed) + 1_000_000 + pair_i
        pair_payoffs: list[float] = []
        for candidate_seat in (0, 1):
            if len(candidate_payoffs) >= int(n_games):
                break
            random.seed(game_seed)
            np.random.seed(game_seed)
            torch.manual_seed(game_seed)
            rng = np.random.default_rng(action_seed + candidate_seat)
            state = new_game(2, initial_chips=initial_chips)
            n_steps = 0
            while not state.is_terminal and n_steps < max_steps_per_hand:
                features = state.to_feature_vector().astype(np.float32, copy=False)
                legal_mask = get_legal_mask(state)
                if int(state.player_i) == int(candidate_seat):
                    action_idx = _agilerl_checkpoint_action(
                        candidate_agent,
                        player_i=int(state.player_i),
                        features=features,
                        legal_mask=legal_mask,
                    )
                    if (
                        action_idx < 0
                        or action_idx >= N_ACTIONS
                        or not bool(legal_mask[action_idx])
                    ):
                        invalid_candidate_actions += 1
                        action_idx = select_action(
                            legal_mask / max(float(legal_mask.sum()), 1.0),
                            legal_mask,
                            rng=rng,
                        )
                else:
                    probs = _network_probs(baseline_avg, features, legal_mask, resolved_device)
                    action_idx = select_action(probs, legal_mask, rng=rng)
                state = state.apply_action(INDEX_TO_ACTION[action_idx])
                n_steps += 1
            total_steps += n_steps
            payoff = float(state.payout.get(candidate_seat, 0)) / float(initial_chips)
            candidate_payoffs.append(payoff)
            pair_payoffs.append(payoff)
        if pair_payoffs:
            candidate_pair_payoffs.append(float(np.mean(pair_payoffs)))
        pair_i += 1
    if resolved_device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start

    payoff_mean = float(np.mean(candidate_pair_payoffs)) if candidate_pair_payoffs else 0.0
    payoff_std = (
        float(np.std(candidate_pair_payoffs, ddof=1))
        if len(candidate_pair_payoffs) > 1
        else 0.0
    )
    payoff_se = (
        payoff_std / float(np.sqrt(len(candidate_pair_payoffs)))
        if candidate_pair_payoffs
        else 0.0
    )
    return {
        "algorithm": "agilerl_ippo_vs_native_nfsp_h2h",
        "role": "plug_in_multi_agent_policy_value_rl_control",
        "environment": "poker_ai:full_deck_hu_nlhe",
        "candidate_checkpoint": str(candidate_checkpoint),
        "baseline_checkpoint": str(baseline_checkpoint),
        "candidate_algorithm": "agilerl_ippo",
        "candidate_eval_mode": "agilerl_masked_policy_sample",
        "baseline_algorithm": str(baseline_payload.get("algorithm", "native_nfsp_dqn")),
        **device_info,
        "num_actions": N_ACTIONS,
        "n_games": int(n_games),
        "n_pairs": int(len(candidate_pair_payoffs)),
        "eval_seconds": float(eval_seconds),
        "eval_steps": int(total_steps),
        "eval_games_per_second": float(n_games / max(eval_seconds, 1e-9)),
        "mean_candidate_payoff": payoff_mean,
        "std_candidate_payoff": payoff_std,
        "se_candidate_payoff": payoff_se,
        "lower95_candidate_payoff": float(payoff_mean - 1.96 * payoff_se),
        "upper95_candidate_payoff": float(payoff_mean + 1.96 * payoff_se),
        "invalid_candidate_actions": int(invalid_candidate_actions),
        "promotion": False,
        "warning": (
            "This is a native duplicate-swapped H2H adapter for an AgileRL "
            "checkpoint. It is evidence only at sufficient game count and "
            "after parent/control gates."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--evo-steps", type=int, default=256)
    parser.add_argument("--eval-steps", type=int, default=128)
    parser.add_argument("--eval-loop", type=int, default=1)
    parser.add_argument("--population-size", type=int, default=1)
    parser.add_argument("--learn-step", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--max-steps-per-hand", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--checkpoint-in")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--checkpoint-out")
    args = parser.parse_args(argv)

    if args.checkpoint_in or args.baseline_checkpoint:
        if not args.checkpoint_in or not args.baseline_checkpoint:
            parser.error("--checkpoint-in and --baseline-checkpoint must be supplied together")
        metrics = evaluate_agilerl_ippo_checkpoint_vs_native_nfsp(
            candidate_checkpoint=args.checkpoint_in,
            baseline_checkpoint=args.baseline_checkpoint,
            n_games=args.eval_steps,
            device=args.device,
            seed=args.seed,
        )
        if args.output_json:
            out = Path(args.output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        metrics = run_agilerl_ippo_control(
            max_steps=args.max_steps,
            evo_steps=args.evo_steps,
            eval_steps=args.eval_steps,
            eval_loop=args.eval_loop,
            population_size=args.population_size,
            learn_step=args.learn_step,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            update_epochs=args.update_epochs,
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            initial_chips=args.initial_chips,
            max_steps_per_hand=args.max_steps_per_hand,
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
            output_json=args.output_json,
            checkpoint_out=args.checkpoint_out,
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
