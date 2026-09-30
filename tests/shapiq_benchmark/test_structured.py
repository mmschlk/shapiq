"""Structured truth matches the same callable game, without exponential large tables."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from shapiq.explainer.nn.games.knn import KNNExplainerGame
from shapiq.tree.interventional.game import InterventionalGame
from shapiq_benchmark.games import load_game, prepare_structured
from shapiq_benchmark.prepare import prepare
from shapiq_benchmark.runner import run

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def structured(tmp_path: Path) -> tuple[dict, Path]:
    """Prepare both natural feature count and training-example player definitions."""
    specs = [
        {"id": "tree-sv", "oracle": "tree", "index": "SV", "order": 1},
        {"id": "tree-pairs", "oracle": "tree", "index": "k-SII", "order": 2},
        {"id": "knn", "oracle": "knn", "index": "SV", "order": 1, "n_players": 128},
    ]
    suite = {
        "name": "test",
        "games": specs,
        "methods": ["KernelSHAP"],
        "budgets": [16],
        "seeds": [0],
    }
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    output = tmp_path / "snapshot"
    return prepare(path, output), output


def test_structured_exact_truth_and_live_reconstruction(structured: tuple[dict, Path]) -> None:
    """Small enumeration checks pass and reconstructed live games preserve endpoints."""
    snapshot, root = structured
    assert [game["n_players"] for game in snapshot["games"]] == [30, 30, 128]
    for game in snapshot["games"]:
        assert game["metadata"]["small_validation_max_error"] < 1e-10
        assert game["metadata"].get("active_players", 0) <= game["n_players"]
        with np.load(root / game["artifact"]) as artifact:
            assert "values" not in artifact.files
        oracle = load_game(game, root)
        endpoints = oracle(
            np.array([np.zeros(game["n_players"]), np.ones(game["n_players"])], dtype=bool)
        )
        assert endpoints[0] == pytest.approx(game["truth"]["baseline"])
        assert endpoints[1] - endpoints[0] == pytest.approx(sum(game["truth"]["values"]))


def test_reconstructed_model_and_version_are_checked(structured: tuple[dict, Path]) -> None:
    """A changed model or dependency cannot masquerade as the frozen oracle."""
    snapshot, root = structured
    game = copy.deepcopy(snapshot["games"][0])
    game["metadata"]["model_sha256"] = "wrong"
    with pytest.raises(ValueError, match="differs from the frozen"):
        load_game(game, root)
    game["metadata"]["sklearn_version"] = "unknown"
    with pytest.raises(ValueError, match="scikit-learn version"):
        load_game(game, root)


def test_incompatible_method_is_explicit(structured: tuple[dict, Path], tmp_path: Path) -> None:
    """SV-only methods cannot silently compete in the interaction panel."""
    _, root = structured
    result = run(root, tmp_path / "results")
    row = next(row for row in result["records"] if row["game_id"] == "tree-pairs")
    assert row["status"] == "unsupported"
    assert row["queries"] == 0
    assert row["seconds"] is None
    assert all(row["timing_scope"] != "estimator_with_table_oracle" for row in result["records"])


@pytest.mark.parametrize("ids", [["same", "same"], ["../outside"], []])
def test_invalid_game_names_rejected_before_writes(tmp_path: Path, ids: list[str]) -> None:
    """IDs cannot overwrite other games or escape the artifact directory."""
    output = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="filename-safe"):
        prepare_structured([{"id": name} for name in ids], output)
    assert not output.exists()


def test_knn_truth_rewards_the_heldout_label(structured: tuple[dict, Path]) -> None:
    """Data valuation utility rewards the true label, even when it is class zero."""
    snapshot, root = structured
    game = snapshot["games"][2]
    oracle = load_game(game, root)
    assert isinstance(oracle, KNNExplainerGame)
    assert oracle.model.classes_[oracle.class_index] == game["metadata"]["point_label"]
    assert game["metadata"]["point_label"] == 0
    expected = oracle.model.predict_proba(oracle.x[None])[0, oracle.class_index]
    assert oracle(np.ones((1, game["n_players"]), dtype=bool))[0] == pytest.approx(expected)


@pytest.mark.parametrize("dataset,n_players", [("breast_cancer", 30), ("digits", 64)])
@pytest.mark.parametrize("index", ["SV", "k-SII", "SII", "STII", "FSII", "FBII"])
def test_all_tree_targets_keep_true_class_and_exact_truth(
    tmp_path: Path, dataset: str, n_players: int, index: str
) -> None:
    """Every tree target qualifies against enumeration on its eight-feature counterpart."""
    spec = {
        "id": "tree",
        "oracle": "tree",
        "dataset": dataset,
        "index": index,
        "order": 1 if index == "SV" else 2,
        "instance_seed": 2,
    }
    game = prepare_structured([spec], tmp_path)[0]
    assert game["n_players"] == n_players
    assert game["metadata"]["small_validation_max_error"] < 1e-8
    assert game["metadata"]["active_players"] > 15
    assert (
        0
        < len(game["truth"]["coordinates"])
        <= (n_players if index == "SV" else n_players * (n_players + 1) // 2)
    )
    oracle = load_game(game, tmp_path)
    assert isinstance(oracle, InterventionalGame)
    assert isinstance(oracle.model, RandomForestClassifier)
    assert oracle.model.classes_[game["metadata"]["class_index"]] == game["metadata"]["point_label"]
    with np.load(tmp_path / game["artifact"], allow_pickle=False) as arrays:
        probability = oracle.model.predict_proba(arrays["point"][None])[
            0, game["metadata"]["class_index"]
        ]
    endpoints = oracle(np.array([np.zeros(n_players), np.ones(n_players)], dtype=bool))
    assert endpoints[0] == pytest.approx(game["truth"]["baseline"])
    assert endpoints[1] == pytest.approx(probability)


def test_construction_seeds_change_games_and_reload_deterministically(tmp_path: Path) -> None:
    """A replicate changes the fitted game, while case identity remains the same."""
    specs = [
        {
            "id": f"tree-i{seed}",
            "basecase_id": "tree",
            "oracle": "tree",
            "index": "SV",
            "order": 1,
            "instance_seed": seed,
        }
        for seed in range(4)
    ]
    games = prepare_structured(specs, tmp_path)
    assert len({game["metadata"]["model_sha256"] for game in games}) == 4
    assert len({game["metadata"]["point_row"] for game in games}) == 4
    assert len({game["metadata"]["cluster_id"] for game in games}) == 4
    assert {game["metadata"]["case_id"] for game in games} == {"tree"}
    repeated = prepare_structured([specs[2]], tmp_path / "repeat")[0]
    assert repeated == games[2]
    coalitions = np.random.default_rng(7).integers(0, 2, size=(16, 30)).astype(bool)
    np.testing.assert_array_equal(
        load_game(repeated, tmp_path / "repeat")(coalitions),
        load_game(games[2], tmp_path)(coalitions),
    )


@pytest.mark.parametrize("n_players", [128, 256])
def test_large_knn_instances_change_selected_training_players(
    tmp_path: Path, n_players: int
) -> None:
    """The construction seed changes the valued sample and held-out point."""
    games = prepare_structured(
        [
            {
                "id": f"knn-i{seed}",
                "oracle": "knn",
                "index": "SV",
                "order": 1,
                "n_players": n_players,
                "instance_seed": seed,
            }
            for seed in range(4)
        ],
        tmp_path,
    )
    selected_rows = []
    for game in games:
        assert game["n_players"] == n_players
        assert len(game["truth"]["values"]) == n_players
        assert game["metadata"]["small_validation_max_error"] < 1e-10
        oracle = load_game(game, tmp_path)
        assert isinstance(oracle, KNNExplainerGame)
        assert oracle.model.classes_[oracle.class_index] == game["metadata"]["point_label"]
        with np.load(tmp_path / game["artifact"], allow_pickle=False) as arrays:
            selected_rows.append(arrays["selected_rows"])
            np.testing.assert_array_equal(
                arrays["selected_rows"], arrays["train_indices"][:n_players]
            )
    assert all(not np.array_equal(selected_rows[0], rows) for rows in selected_rows[1:])
    assert games[0]["metadata"]["point_row"] != games[1]["metadata"]["point_row"]


@pytest.mark.parametrize(
    "dataset,n_players",
    [("breast_cancer", n) for n in (16, 32, 64)] + [("digits", n) for n in (16, 32, 64, 512)],
)
@pytest.mark.parametrize("seed", range(4))
def test_stratified_knn_preserves_classes_and_exact_truth(
    tmp_path: Path, dataset: str, n_players: int, seed: int
) -> None:
    """Small new valued subsets retain all labels without altering the held-out truth."""
    spec = {
        "id": "knn",
        "oracle": "knn",
        "dataset": dataset,
        "index": "SV",
        "order": 1,
        "n_players": n_players,
        "instance_seed": seed,
        "row_selection": "stratified",
    }
    game = prepare_structured([spec], tmp_path)[0]
    assert game["metadata"]["row_selection"] == "stratified"
    assert game["metadata"]["small_validation_max_error"] < 1e-10
    with np.load(tmp_path / game["artifact"], allow_pickle=False) as arrays:
        assert len(np.unique(arrays["selected_rows"])) == n_players
        assert set(arrays["selected_rows"]) <= set(arrays["train_indices"])
        assert game["metadata"]["point_row"] not in arrays["selected_rows"]
        assert len(np.unique(arrays["y_train"])) == (10 if dataset == "digits" else 2)
        assert game["metadata"]["point_label"] in arrays["y_train"]
    oracle = load_game(game, tmp_path)
    endpoints = oracle(np.array([np.zeros(n_players), np.ones(n_players)], dtype=bool))
    assert endpoints[1] - endpoints[0] == pytest.approx(sum(game["truth"]["values"]))


@pytest.mark.parametrize("dataset,n_players", [("breast_cancer", 30), ("digits", 64)])
def test_product_kernel_arrays_reload_without_refitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset: str, n_players: int
) -> None:
    """An actual fitted binary RBF model has exact SV and a pickle-free oracle."""
    spec = {
        "id": "product",
        "oracle": "product_kernel",
        "dataset": dataset,
        "index": "SV",
        "order": 1,
        "instance_seed": 1,
    }
    game = prepare_structured([spec], tmp_path)[0]
    repeated = prepare_structured([spec], tmp_path / "repeat")[0]
    assert repeated == game
    assert game["n_players"] == n_players
    assert game["metadata"]["small_validation_max_error"] < 1e-10
    assert game["metadata"]["nonzero_coefficients"] > 20
    assert len(game["metadata"]["classes"]) == 2

    def forbid_fit(*args: object, **kwargs: object) -> None:
        pytest.fail("Product oracle reload must use frozen arrays, not refit SVC.")

    monkeypatch.setattr("shapiq_benchmark.games.SVC.fit", forbid_fit)
    oracle = load_game(game, tmp_path)
    endpoints = oracle(np.array([np.zeros(n_players), np.ones(n_players)], dtype=bool))
    assert endpoints[0] == pytest.approx(game["truth"]["baseline"])
    assert endpoints[1] - endpoints[0] == pytest.approx(sum(game["truth"]["values"]))
    coalitions = np.random.default_rng(9).integers(0, 2, size=(16, n_players)).astype(bool)
    np.testing.assert_array_equal(
        oracle(coalitions), load_game(repeated, tmp_path / "repeat")(coalitions)
    )
    with np.load(tmp_path / game["artifact"], allow_pickle=False) as arrays:
        assert arrays["support_vectors"].shape[1] == n_players
        assert all(not arrays[name].dtype.hasobject for name in arrays.files)
