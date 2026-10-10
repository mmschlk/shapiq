"""Input-feature variation keeps retraining player identities and legacy recipes explicit."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from shapiq_benchmark import materialize, models, qualification
from shapiq_benchmark.families import make_family
from shapiq_benchmark.partitioned import _filter_game
from shapiq_benchmark.prepare import prepare
from shapiq_benchmark.report import merge_results, public_preparation
from shapiq_benchmark.runner import load_snapshot

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def small_data(monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    """Use the real 30-column catalog entry with tiny in-memory data and no downloads."""
    rng = np.random.default_rng(19)
    x = rng.normal(size=(100, 30))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    monkeypatch.setattr(models, "load_raw_dataset", lambda _: (x, y, [f"f{i}" for i in range(30)]))
    return x


@pytest.mark.parametrize("family", ["data_valuation", "dataset_valuation"])
def test_input_width_preserves_row_players_and_splits(small_data: np.ndarray, family: str) -> None:
    """Adding model inputs changes neither training players nor disjoint evaluation pools."""
    constructed = [
        make_family(
            family,
            dataset="breast_cancer",
            n_players=3,
            model_profile="linear",
            instance_seed=2,
            input_features=width,
            feature_rule="nested",
        )
        for width in (12, 24)
    ]
    (small_game, small), (large_game, large) = constructed
    assert small_game.n_players == large_game.n_players == 3
    assert set(small["feature_indices"]) < set(large["feature_indices"])
    assert len(small["feature_indices"]) == small["input_features"] == 12
    assert len(large["feature_indices"]) == large["input_features"] == 24
    assert small["model_key"] != large["model_key"]
    for key in ("train_indices", "test_indices", "validation_indices"):
        assert small[key] == large[key]
    assert set(small["train_indices"]).isdisjoint(small["test_indices"])
    assert set(small["train_indices"]).isdisjoint(small["validation_indices"])
    positions = [large["feature_indices"].index(i) for i in small["feature_indices"]]
    if family == "dataset_valuation":
        assert small["group_indices"] == large["group_indices"]
        for group in range(3):
            np.testing.assert_array_equal(
                small_game.data_sets[group], large_game.data_sets[group][:, positions]
            )
        assert sorted(i for group in small["group_indices"] for i in group) == sorted(
            small["train_indices"]
        )
    else:
        np.testing.assert_array_equal(small_game.x_train, large_game.x_train[:, positions])
    json.dumps([small, large], allow_nan=False)


@pytest.mark.parametrize("family", ["feature_selection", "local_baseline"])
def test_feature_player_dimensions_are_nested(small_data: np.ndarray, family: str) -> None:
    """Feature players are inputs; nested subsets retain the same row split."""
    constructed = [
        make_family(
            family,
            dataset="breast_cancer",
            n_players=count,
            model_profile="linear",
            instance_seed=2,
            feature_rule="nested",
        )
        for count in (3, 5)
    ]
    (small_game, small), (large_game, large) = constructed
    assert (small_game.n_players, large_game.n_players) == (3, 5)
    assert set(small["feature_indices"]) < set(large["feature_indices"])
    assert small["feature_rule"] == large["feature_rule"] == "nested"
    for key in ("train_indices", "test_indices", "validation_indices"):
        assert small[key] == large[key]
    assert "input_features" not in small and "input_features" not in large


@pytest.mark.parametrize("family", ["feature_selection", "data_valuation", "dataset_valuation"])
def test_omitted_options_keep_legacy_features_and_model_identity(
    small_data: np.ndarray, family: str
) -> None:
    """Opt-in dimensions do not change existing choice-based features or default cache keys."""
    options = {
        "dataset": "breast_cancer",
        "n_players": 3,
        "model_profile": "linear",
        "instance_seed": 2,
    }
    original, metadata = make_family(family, **options)
    explicit, explicit_metadata = make_family(family, feature_rule="all", **options)
    width = 3 if family == "feature_selection" else 12
    expected = np.sort(np.random.default_rng(2).choice(30, width, replace=False)).tolist()
    assert metadata["feature_indices"] == expected
    assert "feature_rule" not in metadata and "input_features" not in metadata
    assert metadata == explicit_metadata
    prepared = models.prepare_model("breast_cancer", width, 2, "linear")
    assert metadata["model_key"] == prepared.metadata["model_key"]
    coalitions = ((np.arange(8)[:, None] >> np.arange(3)) & 1).astype(bool)
    np.testing.assert_array_equal(original(coalitions), explicit(coalitions))


@pytest.mark.parametrize(
    "family,extra",
    [
        ("feature_selection", {"input_features": 12}),
        ("local_baseline", {"feature_rule": "continuous"}),
        ("data_valuation", {"input_features": 0}),
        ("data_valuation", {"input_features": True}),
        ("dataset_valuation", {"input_features": 31}),
        ("data_valuation", {"feature_rule": "continuous"}),
    ],
)
def test_invalid_dimension_options_fail_before_loading(
    monkeypatch: pytest.MonkeyPatch, family: str, extra: dict
) -> None:
    """Unsupported combinations cannot silently load data or choose a different game."""

    def unexpected_load(_):
        pytest.fail("Invalid dimensions reached the dataset loader")

    monkeypatch.setattr(models, "load_raw_dataset", unexpected_load)
    with pytest.raises(ValueError):
        make_family(family, dataset="breast_cancer", n_players=3, model_profile="linear", **extra)


@pytest.mark.parametrize("extra", [{"input_features": 12}, {"feature_rule": "nested"}])
def test_dimension_options_require_profile(extra: dict) -> None:
    """Legacy non-profiled construction cannot silently ignore explicit settings."""
    with pytest.raises(ValueError, match="require a model profile"):
        make_family("data_valuation", **extra)


def test_dimensions_survive_real_snapshot_serialization(
    small_data: np.ndarray, tmp_path: Path
) -> None:
    """The actual small-table path authenticates independent feature widths in the snapshot."""
    families = [
        {
            "id": f"grouped-{width}",
            "family": "dataset_valuation",
            "dataset": "breast_cancer",
            "n_players": 3,
            "model_profile": "linear",
            "input_features": width,
            "feature_rule": "nested",
        }
        for width in (12, 24)
    ]
    suite = {
        "families": families,
        "targets": [{"index": "SV", "order": 1}],
        "methods": ["KernelSHAP"],
        "relative_budgets": [1, 2],
        "seeds": [0],
        "game_seeds": [2],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    output = tmp_path / "snapshot"
    snapshot = prepare(path, output)
    assert len(snapshot["games"]) == 2
    assert {g["metadata"]["input_features"] for g in snapshot["games"]} == {12, 24}
    assert {g["metadata"]["feature_rule"] for g in snapshot["games"]} == {"nested"}
    assert len({g["artifact"] for g in snapshot["games"]}) == 2
    assert all(g["n_players"] == 3 for g in snapshot["games"])
    assert load_snapshot(output)[0] == snapshot
    result = {
        "schema_version": 1,
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_provenance": {},
        "run_provenance": {},
        "suite": snapshot["suite"],
        "games": snapshot["games"],
        "methods": {},
        "records": [],
    }
    result_path = tmp_path / "results.json"
    result_path.write_text(json.dumps(result))
    public = merge_results([result_path])
    for game in public["games"]:
        filtered = _filter_game(game, 0, [3, 6], set())
        assert filtered["metadata"]["input_features"] in (12, 24)
        assert filtered["metadata"]["feature_rule"] == "nested"


def test_pilot_and_chunk_forward_dimensions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pilot and independent chunk construction carry the same explicit recipe fields."""
    calls = []

    class CheapGame:
        n_players = 13

        def __call__(self, coalitions):
            return coalitions.sum(axis=1).astype(float)

    def factory(name, **kwargs):
        calls.append((name, kwargs))
        return CheapGame(), {
            "input_features": kwargs["input_features"],
            "feature_rule": kwargs["feature_rule"],
        }

    monkeypatch.setattr(materialize, "make_family", factory)
    monkeypatch.setattr(qualification, "make_family", factory)
    spec = {
        "id": "grouped-wide",
        "family": "dataset_valuation",
        "dataset": "breast_cancer",
        "n_players": 13,
        "model_profile": "linear",
        "input_features": 24,
        "feature_rule": "nested",
    }
    pilot = qualification._pilot(spec, 2, str(tmp_path / "models"))
    chunk = materialize.prepare_family_chunk(spec, 2, 0, tmp_path)
    assert pilot["status"] == "measured"
    assert len(calls) == 2
    for family, options in calls:
        assert family == "dataset_valuation"
        assert options["input_features"] == 24 and options["feature_rule"] == "nested"
        assert options["n_players"] == 13 and options["instance_seed"] == 2
    with np.load(chunk, allow_pickle=False) as stored:
        manifest = json.loads(str(stored["manifest"]))
    assert manifest["identity"]["spec"] == spec
    assert manifest["metadata"] == {"input_features": 24, "feature_rule": "nested"}


def test_public_exclusions_keep_explicit_dimensions() -> None:
    """Excluded widths remain distinguishable without exposing private pilot details."""
    spec = {
        "id": "wide",
        "family": "dataset_valuation",
        "n_players": 3,
        "dataset": "breast_cancer",
        "model_profile": "linear",
        "input_features": 24,
        "feature_rule": "nested",
    }
    suite = {
        "preparation_preflight": {"families": []},
        "preparation_exclusions": [
            {"spec": spec, "reason": "model_not_better_than_validation_dummy", "instances": []}
        ],
    }
    public = public_preparation(suite)
    assert public["preparation_exclusions"][0]["spec"] == spec


@pytest.mark.parametrize(
    "spec",
    [
        {"family": "feature_selection", "model_profile": "linear", "input_features": 12},
        {"family": "data_valuation", "input_features": 12},
        {"family": "data_valuation", "model_profile": "linear", "input_features": True},
        {"family": "local_baseline", "model_profile": "linear", "feature_rule": "continuous"},
    ],
)
def test_invalid_materialization_dimensions_make_no_artifacts(tmp_path: Path, spec: dict) -> None:
    """Reject unsupported dimension recipes at the schema boundary before any oracle work."""
    with pytest.raises(ValueError):
        materialize.prepare_families(
            [{"id": "invalid", "dataset": "breast_cancer", "n_players": 3, **spec}],
            [{"index": "SV", "order": 1}],
            tmp_path,
        )
    assert not list(tmp_path.iterdir())
