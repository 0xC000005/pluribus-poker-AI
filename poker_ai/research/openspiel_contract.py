"""OpenSpiel compatibility probes for native poker autoresearch.

The probe is deliberately narrow: it records whether a maintained OpenSpiel
poker game can preserve this repo's native action contract before any training
run treats OpenSpiel evidence as transferable.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable
from typing import Any


DEFAULT_GAME_NAMES = ("kuhn_poker", "leduc_poker", "universal_poker")
NATIVE_NUM_ACTIONS = 9
UNIVERSAL_POKER_ACTION_ABSTRACTIONS = (
    ("default_fcpa", "universal_poker"),
    (
        "fchpa",
        "universal_poker("
        "bettingAbstraction=fchpa,"
        "numPlayers=2,numRounds=4,numSuits=4,numRanks=13,"
        "numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,blind=50 100,firstPlayer=1 1 1 1"
        ")",
    ),
    (
        "fullgame_holdem",
        "universal_poker("
        "bettingAbstraction=fullgame,"
        "numPlayers=2,numRounds=4,numSuits=4,numRanks=13,"
        "numHoleCards=2,numBoardCards=0 3 1 1,"
        "stack=20000 20000,blind=50 100,firstPlayer=1 1 1 1"
        ")",
    ),
)


def _jsonable_parameters(parameters: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in dict(parameters).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[str(key)] = value
        else:
            result[str(key)] = str(value)
    return result


def _game_summary(pyspiel_module: Any, name: str, native_num_actions: int) -> dict[str, Any]:
    try:
        game = pyspiel_module.load_game(name)
        game_type = game.get_type()
        num_actions = int(game.num_distinct_actions())
        return {
            "name": str(name),
            "loadable": True,
            "short_name": str(getattr(game_type, "short_name", name)),
            "num_players": int(game.num_players()),
            "num_distinct_actions": num_actions,
            "matches_native_num_actions": num_actions == int(native_num_actions),
            "parameters": _jsonable_parameters(game.get_parameters()),
        }
    except Exception as exc:  # pragma: no cover - exercised by optional deps.
        return {
            "name": str(name),
            "loadable": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "matches_native_num_actions": False,
        }


def _universal_poker_abstraction_summary(
    pyspiel_module: Any,
    *,
    native_num_actions: int,
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for label, game_string in UNIVERSAL_POKER_ACTION_ABSTRACTIONS:
        summary = _game_summary(pyspiel_module, game_string, native_num_actions)
        summary["label"] = str(label)
        summary["game_string"] = str(game_string)
        variants.append(summary)
    return variants


def summarize_openspiel_contracts(
    pyspiel_module: Any,
    *,
    native_num_actions: int = NATIVE_NUM_ACTIONS,
    game_names: Iterable[str] = DEFAULT_GAME_NAMES,
    psro_v2_available: bool | None = None,
) -> dict[str, Any]:
    """Summarize OpenSpiel poker games against the native 9-action contract."""
    registered = set(pyspiel_module.registered_names())
    requested_games = [str(name) for name in game_names]
    games = [
        _game_summary(pyspiel_module, name, native_num_actions)
        for name in requested_games
        if name in registered
    ]
    universal = next((game for game in games if game["name"] == "universal_poker"), None)
    abstraction_variants = (
        _universal_poker_abstraction_summary(
            pyspiel_module,
            native_num_actions=int(native_num_actions),
        )
        if "universal_poker" in registered
        else []
    )
    direct_match = bool(
        universal
        and universal.get("loadable")
        and universal.get("matches_native_num_actions")
    )
    any_variant_match = any(
        bool(variant.get("loadable") and variant.get("matches_native_num_actions"))
        for variant in abstraction_variants
    )
    if direct_match:
        recommendation = "candidate_adapter_spike"
    elif universal and universal.get("loadable"):
        recommendation = "reference_only_action_mismatch"
    else:
        recommendation = "install_or_game_registration_needed"
    if direct_match or any_variant_match:
        contract_risk = "candidate_native_match_needs_transfer_gate"
    elif universal and universal.get("loadable"):
        contract_risk = "action_abstraction_mismatch"
    else:
        contract_risk = "universal_poker_unavailable"
    return {
        "algorithm": "openspiel_poker_contract_probe",
        "native_num_actions": int(native_num_actions),
        "requested_games": requested_games,
        "registered_poker_games": sorted(name for name in registered if "poker" in name.lower()),
        "has_universal_poker": "universal_poker" in registered,
        "has_psro_v2": bool(psro_v2_available)
        if psro_v2_available is not None
        else importlib.util.find_spec("open_spiel.python.algorithms.psro_v2") is not None,
        "games": games,
        "universal_poker_action_abstractions": abstraction_variants,
        "openspiel_native_contract_risk": contract_risk,
        "direct_native_action_match": direct_match,
        "recommendation": recommendation,
        "promotion": False,
        "warning": (
            "OpenSpiel poker evidence is reference-only unless a game preserves "
            "the native 9-action Slumbot-facing contract or transfers through native gates."
        ),
    }


def probe_installed_openspiel(
    *,
    native_num_actions: int = NATIVE_NUM_ACTIONS,
    game_names: Iterable[str] = DEFAULT_GAME_NAMES,
) -> dict[str, Any]:
    """Import pyspiel if installed and summarize the available poker contracts."""
    try:
        import pyspiel  # type: ignore[import-not-found]
    except Exception as exc:
        return {
            "algorithm": "openspiel_poker_contract_probe",
            "native_num_actions": int(native_num_actions),
            "games": [],
            "has_universal_poker": False,
            "has_psro_v2": False,
            "direct_native_action_match": False,
            "recommendation": "install_or_game_registration_needed",
            "promotion": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return summarize_openspiel_contracts(
        pyspiel,
        native_num_actions=native_num_actions,
        game_names=game_names,
    )
