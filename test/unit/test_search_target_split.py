import json

import numpy as np

from poker_ai.deep_cfr.policy_targets import PolicyTargetBuffer
from poker_ai.games.full_deck.state import N_ACTIONS, N_FEATURES
from poker_ai.research.belief_probe import N_HANDS
from poker_ai.research.belief_value_probe import (
    PublicBeliefCFVDataset,
    save_public_belief_cfv_dataset_cache,
)
from poker_ai.research.search_target_split import (
    SearchTargetSplitInput,
    SearchTargetSplitOutput,
    split_search_targets_stratified,
)


def _write_artifact(tmp_path, name: str, *, n_turn: int, n_river: int, offset: float):
    n = n_turn + n_river
    features = np.zeros((n, N_FEATURES), dtype=np.float32)
    legal = np.zeros((n, N_ACTIONS), dtype=np.float32)
    legal[:, [1, 2]] = 1.0
    probs = np.zeros((n, N_ACTIONS), dtype=np.float32)
    probs[:, 2] = 1.0
    targets = tmp_path / f"{name}.npz"
    PolicyTargetBuffer(features, legal, probs).save_npz(targets)

    cases = []
    for idx in range(n_turn):
        cases.append(
            {
                "label": f"{name}-turn-{idx}",
                "hole_cards": ["Ac", "Kd"],
                "board": ["2c", "7d", "Jh", "4s"],
                "action_str": "ck/kk/",
                "client_pos": 0,
                "source": "unit",
            }
        )
    for idx in range(n_river):
        cases.append(
            {
                "label": f"{name}-river-{idx}",
                "hole_cards": ["Ac", "Kd"],
                "board": ["2c", "7d", "Jh", "4s", "9c"],
                "action_str": "ck/kk/kk/",
                "client_pos": 0,
                "source": "unit",
            }
        )
    cases_path = tmp_path / f"{name}.cases.json"
    cases_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")

    values = np.zeros((n, N_HANDS), dtype=np.float32)
    masks = np.zeros((n, N_HANDS), dtype=np.float32)
    masks[:, :4] = 1.0
    for row in range(n):
        values[row, :4] = offset + row
    cfv = PublicBeliefCFVDataset(
        features=features,
        belief=np.zeros((n, N_HANDS * 2), dtype=np.float32),
        values=values,
        value_masks=masks,
        labels=tuple(item["label"] for item in cases),
    )
    cfv_path = tmp_path / f"{name}.cfv.npz"
    save_public_belief_cfv_dataset_cache(
        cfv,
        [
            {"label": item["label"], "value_mean": float(values[i, :4].mean())}
            for i, item in enumerate(cases)
        ],
        cfv_path,
    )
    return targets, cases_path, cfv_path


def test_split_search_targets_stratifies_and_preserves_cache_alignment(tmp_path):
    first = _write_artifact(tmp_path, "a", n_turn=4, n_river=4, offset=0.0)
    second = _write_artifact(tmp_path, "b", n_turn=4, n_river=4, offset=100.0)

    metadata = split_search_targets_stratified(
        [
            SearchTargetSplitInput(*first),
            SearchTargetSplitInput(*second),
        ],
        SearchTargetSplitOutput(
            train_targets_npz=tmp_path / "train.npz",
            train_cases_json=tmp_path / "train.cases.json",
            holdout_targets_npz=tmp_path / "holdout.npz",
            holdout_cases_json=tmp_path / "holdout.cases.json",
            train_cfv_cache_npz=tmp_path / "train.cfv.npz",
            holdout_cfv_cache_npz=tmp_path / "holdout.cfv.npz",
            metadata_json=tmp_path / "split.json",
        ),
        train_size=8,
        holdout_size=4,
        seed=17,
        cfv_bins=2,
    )

    train = PolicyTargetBuffer.from_npz(tmp_path / "train.npz")
    holdout = PolicyTargetBuffer.from_npz(tmp_path / "holdout.npz")
    train_cases = json.loads((tmp_path / "train.cases.json").read_text())["cases"]
    holdout_cases = json.loads((tmp_path / "holdout.cases.json").read_text())["cases"]
    train_cache = np.load(tmp_path / "train.cfv.npz", allow_pickle=False)
    holdout_cache = np.load(tmp_path / "holdout.cfv.npz", allow_pickle=False)

    assert train.size == 8
    assert holdout.size == 4
    assert len(train_cases) == 8
    assert len(holdout_cases) == 4
    assert train_cache["values"].shape[0] == 8
    assert holdout_cache["values"].shape[0] == 4
    assert tuple(train_cache["labels"].tolist()) == tuple(
        item["label"] for item in train_cases
    )
    assert tuple(holdout_cache["labels"].tolist()) == tuple(
        item["label"] for item in holdout_cases
    )
    assert metadata["stratify_by"] == "street+cfv_mean_bin"
    assert sum(item["holdout"] for item in metadata["strata"]) == 4


def test_split_search_targets_can_filter_to_river_only(tmp_path):
    first = _write_artifact(tmp_path, "a", n_turn=4, n_river=4, offset=0.0)
    second = _write_artifact(tmp_path, "b", n_turn=4, n_river=4, offset=100.0)

    metadata = split_search_targets_stratified(
        [
            SearchTargetSplitInput(*first),
            SearchTargetSplitInput(*second),
        ],
        SearchTargetSplitOutput(
            train_targets_npz=tmp_path / "river_train.npz",
            train_cases_json=tmp_path / "river_train.cases.json",
            holdout_targets_npz=tmp_path / "river_holdout.npz",
            holdout_cases_json=tmp_path / "river_holdout.cases.json",
            train_cfv_cache_npz=tmp_path / "river_train.cfv.npz",
            holdout_cfv_cache_npz=tmp_path / "river_holdout.cfv.npz",
        ),
        train_size=4,
        holdout_size=2,
        seed=23,
        cfv_bins=2,
        target_streets=(3,),
    )

    train_cases = json.loads((tmp_path / "river_train.cases.json").read_text())["cases"]
    holdout_cases = json.loads((tmp_path / "river_holdout.cases.json").read_text())["cases"]

    assert metadata["target_streets"] == [3]
    assert len(train_cases) == 4
    assert len(holdout_cases) == 2
    assert all(len(case["board"]) == 5 for case in [*train_cases, *holdout_cases])
