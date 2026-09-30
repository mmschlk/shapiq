"""Representative shipped-game recipes preserve boundedness and payoff semantics."""

from __future__ import annotations

import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from shapiq_benchmark.families import FAMILY_CATALOG, make_family


@pytest.mark.parametrize("players", [11, 20])
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.filterwarnings(
    "ignore:Number of distinct clusters:sklearn.exceptions.ConvergenceWarning"
)
def test_digits_cluster_singletons_have_defined_scores(players: int, seed: int) -> None:
    """Blank border pixels cannot form a valid clustering game on their own."""
    from shapiq_benchmark.families import _dataset

    with threadpool_limits(limits=1):
        game, metadata = make_family(
            "cluster", dataset="digits", n_players=players, instance_seed=seed
        )
        x, _, train, _, _ = _dataset("digits", seed)
        selected = metadata["feature_indices"]
        assert len(selected) == players
        assert (np.ptp(x[train[:128]][:, selected], axis=0) > 0).all()
        assert "nonconstant" in metadata["parameters"]["feature_rule"]
        assert np.isfinite(game(np.eye(players, dtype=bool))).all()


@pytest.mark.parametrize("name", FAMILY_CATALOG)
def test_recipe_is_bounded_reproducible_and_explicit(name: str) -> None:
    """Fresh recipes reproduce sampled realizations; deterministic ones ignore batching."""
    for dependency in FAMILY_CATALOG[name]["dependencies"]:
        pytest.importorskip(dependency)
    with threadpool_limits(limits=1):
        game, metadata = make_family(name)
        assert 1 <= game.n_players <= 8
        coalitions = (
            np.random.default_rng(123).integers(0, 2, size=(5, game.n_players)).astype(bool)
        )
        coalitions[0], coalitions[-1] = False, True
        values = game(coalitions)
        assert values.shape == (5,)
        assert np.all(np.isfinite(values))
        json.dumps(metadata, allow_nan=False)
        recreated, repeated_metadata = make_family(name)
        assert metadata == repeated_metadata
        np.testing.assert_allclose(values, recreated(coalitions), rtol=1e-12, atol=1e-12)
        if not metadata["stochastic_frozen"]:
            np.testing.assert_allclose(values, game(coalitions), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(values[::-1], game(coalitions[::-1]), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(
                values,
                np.concatenate([game(row[None]) for row in coalitions]),
                rtol=1e-12,
                atol=1e-12,
            )
        if not metadata["synthetic"]:
            assert metadata["dataset"] in ("california_housing", "iris")
            assert len(metadata["data_sha256"]) == 64
            assert set(metadata["train_indices"]).isdisjoint(metadata["test_indices"])
            if metadata["player_unit"] == "feature":
                assert game.n_players == (4 if metadata["dataset"] == "iris" else 8)


def test_dataset_players_are_full_training_groups() -> None:
    """Grouped valuation retains all real training rows with disjoint players."""
    game, metadata = make_family("dataset_valuation")
    groups = metadata["group_indices"]
    assert game.n_players == len(groups) == 8
    assert sorted(row for group in groups for row in group) == sorted(metadata["train_indices"])
    assert all(len(group) == 64 for group in groups)


def test_baseline_companion_preserves_original_case() -> None:
    """The fixed forest companion retains the original zero game and point selection."""
    original, original_metadata = make_family("local_baseline")
    companion, metadata = make_family("local_baseline_forest")
    coalitions = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(bool)
    np.testing.assert_array_equal(original(coalitions), np.zeros(256))
    values = companion(coalitions)
    assert np.all(np.isfinite(values))
    assert np.ptp(values) > 0
    for key in ("point_row", "background_indices", "train_indices", "test_indices"):
        assert metadata[key] == original_metadata[key]
    assert metadata["model"] == "RandomForestRegressor"
    assert metadata["model_parameters"]["n_estimators"] == 8
    assert metadata["model_parameters"]["max_depth"] == 4
    assert metadata["model_parameters"]["random_state"] == 0
    assert metadata["model_parameters"]["n_jobs"] == 1
    assert metadata["application_family"] == "local_explanation"


def test_knn_uses_held_out_true_label_and_fixed_denominator() -> None:
    """The row utility targets the selected real point's class, with exactly k denominator."""
    game, metadata = make_family("knn")
    assert metadata["class_index"] == metadata["point_label"]
    assert metadata["point_row"] in metadata["test_indices"]
    for row in range(game.n_players):
        coalition = np.zeros((1, game.n_players), dtype=bool)
        coalition[0, row] = True
        assert game(coalition)[0] == (
            1 / 3 if game.y_train_indices[row] == metadata["point_label"] else 0
        )


def test_synthetics_are_separate_diagnostics() -> None:
    """Synthetic payoff fixtures cannot silently enter the real-data headline."""
    assert {name for name, info in FAMILY_CATALOG.items() if info["synthetic"]} == {
        "unanimity",
        "soum",
        "dummy",
        "random",
    }
    assert FAMILY_CATALOG["local_marginal"]["application_family"] == "local_explanation"
    assert FAMILY_CATALOG["interventional_tree"]["application_family"] == "local_explanation"
    assert FAMILY_CATALOG["dataset_valuation"]["application_family"] == "data_valuation"


@pytest.mark.parametrize(
    "name", ["local_baseline_forest", "data_valuation", "product_kernel", "knn"]
)
def test_construction_seeds_change_instances_reproducibly(name: str) -> None:
    """Four construction seeds change data/model inputs, not only estimator sampling."""
    metadata_by_seed = []
    with threadpool_limits(limits=1):
        for seed in range(4):
            game, metadata = make_family(name, instance_seed=seed)
            repeated, repeated_metadata = make_family(name, instance_seed=seed)
            assert metadata == repeated_metadata
            assert metadata["instance_seed"] == seed
            assert set(metadata["train_indices"]).isdisjoint(metadata["test_indices"])
            coalitions = np.random.default_rng(55).integers(0, 2, size=(8, game.n_players))
            np.testing.assert_allclose(game(coalitions), repeated(coalitions), rtol=0, atol=0)
            metadata_by_seed.append(metadata)
    assert len({tuple(m["train_indices"]) for m in metadata_by_seed}) == 4
    assert len({m["point_row"] for m in metadata_by_seed}) == 4


@pytest.mark.parametrize("name", ["dummy", "unanimity"])
def test_fixed_synthetic_recipes_vary_interaction_support(name: str) -> None:
    """Changing construction seeds must not emit four copies of a fixed diagnostic."""
    coalitions = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(bool)
    values = [make_family(name, instance_seed=seed)[0](coalitions) for seed in range(4)]
    assert len({value.tobytes() for value in values}) == 4


@pytest.mark.parametrize(
    ("name", "dataset", "players", "features"),
    [
        ("local_baseline_forest", "diabetes", 10, 10),
        ("uncertainty", "wine", 6, 6),
        ("uncertainty", "wine", 12, 12),
        ("data_valuation", "diabetes", 4, 10),
        ("dataset_valuation", "diabetes", 4, 10),
        ("ensemble", "diabetes", 4, 10),
        ("forest_ensemble", "diabetes", 12, 10),
        ("knn", "wine", 4, 13),
        ("weighted_knn", "wine", 12, 13),
    ],
)
def test_explicit_recipe_preserves_player_unit_and_selected_data(
    name: str, dataset: str, players: int, features: int
) -> None:
    """Feature counts and row/group/model counts must not be confused."""
    with threadpool_limits(limits=1):
        game, metadata = make_family(name, dataset=dataset, n_players=players, instance_seed=2)
        repeated, again = make_family(name, dataset=dataset, n_players=players, instance_seed=2)
        assert game.n_players == players
        assert metadata == again and metadata["dataset"] == dataset
        assert len(metadata["feature_indices"]) == len(metadata["feature_names"]) == features
        assert len(set(metadata["feature_indices"])) == features
        assert not set(metadata["train_indices"]) & set(metadata["test_indices"])
        coalitions = np.random.default_rng(0).integers(0, 2, (8, players)).astype(bool)
        values = game(coalitions)
        assert np.isfinite(values).all()
        np.testing.assert_array_equal(values, repeated(coalitions))
        if name == "dataset_valuation":
            assert len(metadata["group_indices"]) == players
            assert sorted(i for group in metadata["group_indices"] for i in group) == sorted(
                metadata["train_indices"]
            )
        if name in ("knn", "weighted_knn"):
            assert len(metadata["train_indices"]) == players


def test_classification_feature_selection_is_seeded_and_reproducible() -> None:
    """Each construction records its actual subset of the native thirteen wine features."""
    selected = []
    for seed in range(4):
        _, metadata = make_family("uncertainty", dataset="wine", n_players=6, instance_seed=seed)
        selected.append(tuple(metadata["feature_indices"]))
        assert len(selected[-1]) == 6 and all(0 <= i < 13 for i in selected[-1])
    assert len(set(selected)) == 4


def test_configured_tnn_radius_uses_training_geometry_only() -> None:
    """Dimension changes use a declared training-only radius, preserving the legacy rule."""
    from scipy.spatial.distance import pdist
    from sklearn.preprocessing import StandardScaler

    from shapiq_benchmark.families import _dataset

    _, legacy = make_family("tnn")
    assert legacy["model_parameters"]["radius"] == 2.0
    x, _, train, _, _ = _dataset("wine", 2)
    distances = pdist(StandardScaler().fit_transform(x[train]))
    expected = np.median(distances[distances > 0])
    for players in (4, 12):
        game, metadata = make_family("tnn", dataset="wine", n_players=players, instance_seed=2)
        assert game.n_players == players
        assert metadata["model_parameters"]["radius"] == expected
        assert metadata["parameters"]["radius"] == expected
        assert "training rows" in metadata["parameters"]["radius_rule"]
        assert metadata["preprocessing_fit_indices"] == train.tolist()


@pytest.mark.parametrize("players", [True, 0, 21, 2.5])
def test_explicit_player_count_rejects_invalid_or_unbounded_sizes(players: object) -> None:
    """The adapter must never quietly materialize exponentially larger games."""
    with pytest.raises(ValueError, match="n_players"):
        make_family("uncertainty", dataset="wine", n_players=players)


@pytest.mark.parametrize("players", [13, 16, 20])
def test_larger_recipe_uses_real_features(players: int) -> None:
    """The raised cap retains native columns and finite classifier payoffs."""
    with threadpool_limits(limits=1):
        game, metadata = make_family(
            "local_baseline", dataset="breast_cancer", n_players=players, instance_seed=2
        )
        assert game.n_players == players
        assert len(set(metadata["feature_indices"])) == players
        coalitions = np.random.default_rng(0).integers(0, 2, (8, players)).astype(bool)
        assert np.isfinite(game(coalitions)).all()


@pytest.mark.parametrize("name", ["local_gaussian", "local_copula"])
@pytest.mark.parametrize("players", [11, 12])
def test_larger_gaussian_recipes_explain_wine_class_probability(name: str, players: int) -> None:
    """Continuous-feature classification respects Gaussian imputation and output semantics."""
    from sklearn.tree import DecisionTreeClassifier

    from shapiq_benchmark.families import _dataset

    game, metadata = make_family(name, dataset="wine", n_players=players, instance_seed=2)
    x, y, train, test, _ = _dataset("wine", 2)
    x = x[:, metadata["feature_indices"]]
    model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=2).fit(
        x[train], y[train]
    )
    assert game.n_players == players
    assert metadata["model"] == "DecisionTreeClassifier"
    assert metadata["class_index"] == 1 and metadata["output_scale"] == "class probability"
    assert model.classes_[1] == 1
    assert game(np.ones((1, players), dtype=bool))[0] == pytest.approx(
        model.predict_proba(x[test[:1]])[0, 1]
    )
    assert np.isfinite(game(np.zeros((1, players), dtype=bool))).all()


@pytest.mark.parametrize(
    "name",
    [
        "local_baseline",
        "local_baseline_forest",
        "local_marginal",
        "global_fidelity",
        "pathdependent_tree",
        "interventional_tree",
    ],
)
def test_classification_explanation_uses_probability(name: str) -> None:
    """Classifier explanations target a declared probability, never an ordinal class number."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    from shapiq_benchmark.families import _dataset

    game, metadata = make_family(name, dataset="wine", n_players=11, instance_seed=1)
    x, y, train, test, _ = _dataset("wine", 1)
    x = x[:, metadata["feature_indices"]]
    if name == "local_baseline_forest":
        model = RandomForestClassifier(n_estimators=8, max_depth=4, random_state=1, n_jobs=1)
    else:
        model = DecisionTreeClassifier(
            max_depth=4 if name == "interventional_tree" else 3, min_samples_leaf=5, random_state=1
        )
    model.fit(x[train], y[train])
    assert metadata["class_index"] == 1 and metadata["output_scale"] == "class probability"
    if name == "global_fidelity":
        np.testing.assert_array_equal(
            game.model(x[test[:5]]), model.predict_proba(x[test[:5]])[:, 1]
        )
    else:
        actual = game(np.ones((1, 11), dtype=bool))[0]
        if game.normalize:
            actual += game.normalization_value
        assert actual == pytest.approx(model.predict_proba(x[test[:1]])[0, 1])


@pytest.mark.parametrize("name", ["feature_selection", "data_valuation", "dataset_valuation"])
def test_classifier_retraining_uses_accuracy_even_for_one_class(name: str) -> None:
    """Singleton training coalitions remain valid classifier games with a bounded utility."""
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier

    from shapiq_benchmark.families import _dataset

    game, metadata = make_family(name, dataset="wine", n_players=11, instance_seed=1)
    x, y, _, _, _ = _dataset("wine", 1)
    x = x[:, metadata["feature_indices"]]
    train, test = np.array(metadata["train_indices"]), np.array(metadata["test_indices"])
    coalition = np.zeros((1, 11), dtype=bool)
    coalition[0, 0] = True
    if name == "feature_selection":
        train_x, test_x, labels = x[train, :1], x[test, :1], y[train]
    else:
        selected = train[:1] if name == "data_valuation" else np.array(metadata["group_indices"][0])
        train_x, test_x, labels = x[selected], x[test], y[selected]
    model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=1).fit(
        train_x, labels
    )
    expected = accuracy_score(y[test], model.predict(test_x))
    assert game(coalition)[0] == pytest.approx(expected)
    assert game(np.zeros((1, 11), dtype=bool))[0] == 0
    assert metadata["output_scale"] == "accuracy"


@pytest.mark.parametrize("name", ["ensemble", "forest_ensemble"])
def test_classifier_ensemble_uses_majority_vote_accuracy(name: str) -> None:
    """The shipped classification ensemble aggregates votes rather than numeric class averages."""
    from scipy.stats import mode
    from sklearn.metrics import accuracy_score

    with threadpool_limits(limits=1):
        game, metadata = make_family(name, dataset="wine", n_players=11, instance_seed=1)
        expected = accuracy_score(game._y_test, mode(game.predictions, axis=0)[0].ravel())
        assert game(np.ones((1, 11), dtype=bool))[0] == pytest.approx(expected)
        assert game.dataset_type == "classification" and metadata["output_scale"] == "accuracy"


def test_digits_gaussian_selects_training_eligible_columns() -> None:
    """Binary/constant pixels are filtered without consulting labels or the explained point."""
    from shapiq_benchmark.families import _dataset, feature_subset

    x, _, train, _, _ = _dataset("digits", 2)
    eligible = np.array([i for i in range(x.shape[1]) if len(np.unique(x[train, i])) > 2])
    expected = eligible[feature_subset(x[:, eligible], 12, 2)]
    for name in ("local_gaussian", "local_copula"):
        game, metadata = make_family(name, dataset="digits", n_players=12, instance_seed=2)
        assert metadata["feature_indices"] == expected.tolist()
        assert "training columns" in metadata["parameters"]["feature_rule"]
        assert np.isfinite(game(np.ones((1, 12), dtype=bool))).all()


def test_binary_classification_product_kernel_matches_decision_score() -> None:
    """Kernel payoffs explain binary SVC scores, not probabilities or multiclass label codes."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from shapiq_benchmark.families import _dataset

    game, metadata = make_family(
        "product_kernel", dataset="breast_cancer", n_players=11, instance_seed=1
    )
    x, y, train, test, _ = _dataset("breast_cancer", 1)
    x = x[:, metadata["feature_indices"]]
    scaler = StandardScaler().fit(x[train])
    model = SVC(kernel="rbf", gamma="scale").fit(scaler.transform(x[train[:128]]), y[train[:128]])
    assert game(np.ones((1, 11), dtype=bool))[0] == pytest.approx(
        model.decision_function(scaler.transform(x[test[:1]]))[0]
    )
    assert metadata["output_scale"] == "binary SVC decision score"
    with pytest.raises(ValueError, match="two classes"):
        make_family("product_kernel", dataset="wine", n_players=11)
