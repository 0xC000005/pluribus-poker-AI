import json

from poker_ai.research.openspiel_contract import summarize_openspiel_contracts


class _FakeGameType:
    def __init__(self, short_name):
        self.short_name = short_name


class _FakeGame:
    def __init__(self, *, name, players, actions, parameters):
        self._name = name
        self._players = players
        self._actions = actions
        self._parameters = parameters

    def num_players(self):
        return self._players

    def num_distinct_actions(self):
        return self._actions

    def get_type(self):
        return _FakeGameType(self._name)

    def get_parameters(self):
        return dict(self._parameters)


class _FakePySpiel:
    @staticmethod
    def registered_names():
        return ["kuhn_poker", "universal_poker", "other_game"]

    @staticmethod
    def load_game(name):
        if name == "kuhn_poker":
            return _FakeGame(name=name, players=2, actions=2, parameters={"players": 2})
        if name == "universal_poker":
            return _FakeGame(
                name=name,
                players=2,
                actions=4,
                parameters={"betting": "nolimit", "bettingAbstraction": "fcpa"},
            )
        raise ValueError(name)


def test_summarize_openspiel_contracts_flags_default_action_mismatch():
    summary = summarize_openspiel_contracts(
        _FakePySpiel,
        native_num_actions=9,
        game_names=("kuhn_poker", "universal_poker"),
        psro_v2_available=True,
    )

    assert summary["has_universal_poker"] is True
    assert summary["has_psro_v2"] is True
    assert summary["direct_native_action_match"] is False
    assert summary["recommendation"] == "reference_only_action_mismatch"
    universal = next(game for game in summary["games"] if game["name"] == "universal_poker")
    assert universal["num_distinct_actions"] == 4
    assert universal["matches_native_num_actions"] is False
    assert universal["parameters"]["bettingAbstraction"] == "fcpa"


def test_summarize_openspiel_contracts_reports_universal_poker_abstractions():
    class AbstractionPySpiel(_FakePySpiel):
        @staticmethod
        def load_game(name):
            if "bettingAbstraction=fchpa" in name:
                return _FakeGame(
                    name="universal_poker",
                    players=2,
                    actions=5,
                    parameters={"bettingAbstraction": "fchpa"},
                )
            if "bettingAbstraction=fullgame" in name:
                return _FakeGame(
                    name="universal_poker",
                    players=2,
                    actions=20001,
                    parameters={"bettingAbstraction": "fullgame"},
                )
            return _FakePySpiel.load_game(name)

    summary = summarize_openspiel_contracts(
        AbstractionPySpiel,
        native_num_actions=9,
        game_names=("universal_poker",),
        psro_v2_available=True,
    )

    variants = {
        variant["label"]: variant
        for variant in summary["universal_poker_action_abstractions"]
    }
    assert variants["default_fcpa"]["num_distinct_actions"] == 4
    assert variants["default_fcpa"]["matches_native_num_actions"] is False
    assert variants["fchpa"]["num_distinct_actions"] == 5
    assert variants["fullgame_holdem"]["num_distinct_actions"] == 20001
    assert summary["openspiel_native_contract_risk"] == "action_abstraction_mismatch"


def test_summarize_openspiel_contracts_marks_candidate_when_actions_match():
    class MatchingPySpiel(_FakePySpiel):
        @staticmethod
        def load_game(name):
            if name == "universal_poker":
                return _FakeGame(
                    name=name,
                    players=2,
                    actions=9,
                    parameters={"betting": "nolimit", "bettingAbstraction": "custom"},
                )
            return _FakePySpiel.load_game(name)

    summary = summarize_openspiel_contracts(
        MatchingPySpiel,
        native_num_actions=9,
        game_names=("universal_poker",),
        psro_v2_available=False,
    )

    assert summary["direct_native_action_match"] is True
    assert summary["recommendation"] == "candidate_adapter_spike"


def test_probe_openspiel_contract_cli_writes_json(monkeypatch, tmp_path):
    from scripts import probe_openspiel_poker_contract as cli

    output = tmp_path / "openspiel_contract.json"
    expected = {
        "algorithm": "openspiel_poker_contract_probe",
        "recommendation": "reference_only_action_mismatch",
        "promotion": False,
    }
    monkeypatch.setattr(
        cli,
        "probe_installed_openspiel",
        lambda **_kwargs: dict(expected),
    )

    exit_code = cli.main(["--native-num-actions", "9", "--output-json", str(output)])

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
