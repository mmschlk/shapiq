"""Semantic tests for the machine learning game families."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import (
    entropy as scipy_entropy,
    mode,
)
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shapiq_games import (
    ClusterExplanation,
    DatasetValuation,
    DataValuation,
    EnsembleSelection,
    FeatureSelection,
    GlobalConfoundingXAI,
    ImageClassifier,
    LocalConfoundingXAI,
    RandomForestEnsembleSelection,
    SentimentAnalysis,
    TabularGlobalExplanation,
    TabularLocalExplanation,
    TabularUncertaintyExplanation,
    UnsupervisedData,
)
from shapiq_games._training import resolve_metric, resolve_row_metric
from shapiq_games.unsupervised import _entropy, total_correlation
from tests.shapiq_games.helpers import (
    FakeSentimentPipeline,
    is_installed,
    mean_brightness_classifier,
    skip_if_no_skimage,
)


@pytest.fixture(scope="module")
def data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 4))
    y_class = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    y_reg = x[:, 0] - 2 * x[:, 2]
    return {
        "x_train": x[:300],
        "x_test": x[300:],
        "yc_train": y_class[:300],
        "yc_test": y_class[300:],
        "yr_train": y_reg[:300],
        "yr_test": y_reg[300:],
    }


def test_tabular_local_explanation_full_coalition_is_the_prediction(data) -> None:
    model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(
        data["x_train"], data["yr_train"]
    )
    game = TabularLocalExplanation(model, data["x_train"], x=data["x_test"][0], normalize=False)
    assert game.class_index is None
    assert game(game.grand_coalition)[0] == pytest.approx(model.predict(data["x_test"][:1])[0])
    assert game(game.empty_coalition)[0] == pytest.approx(game.empty_value)
    assert game(game.grand_coalition)[0] == pytest.approx(game.original_model_output)


def test_tabular_local_explanation_classifier_and_callable(data) -> None:
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(
        data["x_train"], data["yc_train"]
    )
    game = TabularLocalExplanation(model, data["x_train"], x=5, normalize=False)
    assert game.class_index == 1
    assert game(game.grand_coalition)[0] == pytest.approx(
        model.predict_proba(data["x_train"][5:6])[0, 1]
    )
    game = TabularLocalExplanation(lambda z: z.sum(axis=1), data["x_train"], x=0, normalize=False)
    assert game.class_index is None
    assert game(game.grand_coalition)[0] == pytest.approx(data["x_train"][0].sum())
    with pytest.raises(ValueError, match="Unknown imputer"):
        TabularLocalExplanation(model, data["x_train"], imputer="nearest")  # type: ignore[arg-type]
    with pytest.raises(IndexError, match="out of range"):
        TabularLocalExplanation(model, data["x_train"], x=10_000)


def test_tabular_local_explanation_takes_point_and_seed_from_an_imputer(data) -> None:
    """An imputer brings its own point and seed, and the game stays centered and batch-free."""
    from shapiq.imputer import GaussianImputer, GenerativeConditionalImputer

    model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(
        data["x_train"], data["yr_train"]
    )
    # the Gaussian imputer sets no empty prediction and draws samples coalition after coalition
    imputer = GaussianImputer(
        model=model.predict, data=data["x_train"], x=data["x_test"][3], sample_size=20,
        random_state=0,
    )  # fmt: skip
    game = TabularLocalExplanation(model, data["x_train"], imputer=imputer)
    np.testing.assert_array_equal(game.x, data["x_test"][3])
    assert game.random_state == 0
    assert game(game.empty_coalition)[0] == 0.0
    coalition = np.array([[True, False, False, False]])
    alone = game(coalition)[0]
    assert game(np.vstack([[False, True, True, False], coalition[0]]))[1] == alone

    conditional = GenerativeConditionalImputer(
        model=model.predict, data=data["x_train"], x=data["x_test"][3], random_state=7,
        normalize=False,
    )  # fmt: skip
    expected = conditional.value_function(coalition)[0]  # drawn with a fresh generator seeded 7
    game = TabularLocalExplanation(model, data["x_train"], imputer=conditional, normalize=False)
    assert game(coalition)[0] == pytest.approx(expected)


def test_an_imputer_that_samples_must_be_seeded(data) -> None:
    from shapiq.imputer import GaussianImputer

    model = DecisionTreeRegressor(max_depth=2, random_state=0).fit(
        data["x_train"], data["yr_train"]
    )
    unseeded = GaussianImputer(model=model.predict, data=data["x_train"], x=data["x_test"][0])
    with pytest.raises(ValueError, match="random_state"):
        TabularLocalExplanation(model, data["x_train"], imputer=unseeded)


def test_nan_baseline_passes_absent_features_as_missing_values(data) -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(max_iter=20, random_state=0).fit(
        data["x_train"], data["yr_train"]
    )
    point = data["x_test"][0]
    game = TabularLocalExplanation(
        model, data["x_train"], x=point, imputer="baseline", baseline=np.nan, normalize=False
    )
    masked = point.copy()
    masked[[1, 3]] = np.nan
    coalition = np.array([[True, False, True, False]])
    assert game(coalition)[0] == pytest.approx(model.predict(masked[None])[0])
    assert game(game.empty_coalition)[0] == pytest.approx(model.predict(np.full((1, 4), np.nan))[0])


def test_baseline_is_the_background_mean_or_given_values(data) -> None:
    def model(x: np.ndarray) -> np.ndarray:
        return x.sum(axis=1)

    x, point = data["x_train"], data["x_test"][0]
    coalition = np.array([[True, False, True, False]])
    mean = TabularLocalExplanation(model, x, x=point, imputer="baseline", normalize=False)
    expected = point[[0, 2]].sum() + x[:, [1, 3]].mean(axis=0).sum()
    assert mean(coalition)[0] == pytest.approx(expected)
    values = np.array([10.0, 20.0, 30.0, 40.0])
    given = TabularLocalExplanation(model, x, x=point, imputer="baseline", baseline=values)
    assert given(coalition)[0] + given.normalization_value == pytest.approx(
        point[[0, 2]].sum() + 60.0
    )
    with pytest.raises(ValueError, match="one value or one per feature"):
        TabularLocalExplanation(model, x, x=point, imputer="baseline", baseline=[1.0, 2.0])
    with pytest.raises(ValueError, match="baseline applies to imputer='baseline'"):
        TabularLocalExplanation(model, x, x=point, baseline=np.nan)


def test_tabpfn_reads_inf_as_missing_only_with_passthrough(
    data, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TabPFN v3 (tabpfn>=8.1) reads +inf as missing when built with PASSTHROUGH_INF."""
    import importlib.metadata
    import sys
    import types

    class TabPFNRegressor:
        def __init__(self, inference_config: dict | None = None) -> None:
            self.inference_config = inference_config

        def predict(self, x: np.ndarray) -> np.ndarray:  # the sum of the features it can read
            return np.where(np.isinf(x), 0.0, x).sum(axis=1)

    TabPFNRegressor.__module__ = "tabpfn"
    fake = types.ModuleType("tabpfn")
    fake.TabPFNRegressor = TabPFNRegressor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tabpfn", fake)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "9.1.0")
    x, point = data["x_train"], data["x_test"][0]
    with pytest.raises(ValueError, match="PASSTHROUGH_INF"):
        TabularLocalExplanation(TabPFNRegressor(), x, x=point, imputer="baseline", baseline=np.inf)
    model = TabPFNRegressor({"PASSTHROUGH_INF": True})
    game = TabularLocalExplanation(
        model, x, x=point, imputer="baseline", baseline=np.inf, normalize=False
    )
    assert game(np.array([[True, False, True, False]]))[0] == pytest.approx(point[[0, 2]].sum())
    assert game(game.empty_coalition)[0] == 0.0
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "6.4.1")
    with pytest.raises(ValueError, match="tabpfn 8.1"):
        TabularLocalExplanation(model, x, x=point, imputer="baseline", baseline=np.inf)


def test_tabular_global_explanation_explains_loss_reduction(data) -> None:
    model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(
        data["x_train"], data["yr_train"]
    )
    game = TabularGlobalExplanation(model, data["x_test"], n_samples=50, normalize=False)
    assert game(game.grand_coalition)[0] == pytest.approx(0.0)  # no loss with all features
    assert game(game.empty_coalition)[0] < 0.0
    # the model never splits on feature 3, so it explains nothing on its own
    assert 3 not in model.tree_.feature
    single = game(np.eye(4, dtype=bool)) - game(game.empty_coalition)[0]
    assert single[3] == pytest.approx(0.0, abs=1e-12)
    assert single[0] > 0 and single[2] > 0


def test_feature_selection_empty_and_full_coalition(data) -> None:
    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    game = FeatureSelection(
        model, data["x_train"], data["yc_train"], data["x_test"], data["yc_test"],
        task="classification", normalize=False,
    )  # fmt: skip
    majority = max(data["yc_train"].mean(), 1 - data["yc_train"].mean())
    majority_label = int(data["yc_train"].mean() > 0.5)
    assert game(game.empty_coalition)[0] == pytest.approx(
        np.mean(data["yc_test"] == majority_label)
    )
    assert 0.0 < majority < 1.0
    fitted = DecisionTreeClassifier(max_depth=3, random_state=0).fit(
        data["x_train"], data["yc_train"]
    )
    assert game(game.grand_coalition)[0] == pytest.approx(
        fitted.score(data["x_test"], data["yc_test"])
    )


def test_valuation_accuracy_uses_labels_not_column_indices(data) -> None:
    """A training subset with one class predicts that class (regression test of the old games)."""
    model = DecisionTreeClassifier(random_state=0)
    game = DataValuation(
        model, data["x_train"][:10], data["yc_train"][:10], data["x_test"], data["yc_test"],
        task="classification", normalize=False,
    )  # fmt: skip
    ones = np.flatnonzero(data["yc_train"][:10] == 1)
    coalition = np.zeros((1, 10), dtype=bool)
    coalition[0, ones] = True
    assert game(coalition)[0] == pytest.approx(np.mean(data["yc_test"] == 1))
    assert game(game.empty_coalition)[0] == 0.0


def test_retraining_games_are_deterministic_with_unseeded_models(data) -> None:
    """A model that leaves its random_state unset is seeded by the game's random_state."""
    forest = RandomForestClassifier(n_estimators=5)
    game = FeatureSelection(
        forest, data["x_train"], data["yc_train"], data["x_test"], data["yc_test"]
    )
    assert len({float(game(game.grand_coalition)[0]) for _ in range(3)}) == 1
    valuation = DataValuation(
        DecisionTreeClassifier(), data["x_train"][:20], data["yc_train"][:20], data["x_test"],
        data["yc_test"],
    )  # fmt: skip
    assert len({float(valuation(valuation.grand_coalition)[0]) for _ in range(3)}) == 1
    assert forest.random_state is None  # the caller's model is left alone


def test_valuation_survives_coalitions_a_model_cannot_fit() -> None:
    """Coalitions with too few rows or a subset of the classes still have a value."""
    x, y = make_classification(
        n_samples=120, n_features=4, n_informative=3, n_redundant=0, n_classes=3, random_state=0
    )
    knn = DataValuation(KNeighborsClassifier(n_neighbors=3), x[:6], y[:6], x[60:], y[60:])
    assert np.isfinite(knn(np.array([[True, True, False, False, False, False]]))).all()
    if is_installed("xgboost"):  # XGBoost wants the labels 0, ..., k - 1 of the coalition
        import xgboost as xgb

        rows = np.concatenate([np.flatnonzero(y == 0)[:3], np.flatnonzero(y == 2)[:3]])
        model = xgb.XGBClassifier(n_estimators=3, n_jobs=1)
        game = DataValuation(model, x[rows], y[rows], x[60:], y[60:], normalize=False)
        assert 0.0 < game(game.grand_coalition)[0] <= 1.0


def test_dataset_valuation_accepts_boolean_masks_and_rejects_overlaps(data) -> None:
    model = DecisionTreeRegressor(random_state=0)
    source = np.arange(300) % 3 == 0
    game = DatasetValuation(
        model, data["x_train"], data["yr_train"], data["x_test"], data["yr_test"],
        groups=[source, ~source],
    )  # fmt: skip
    np.testing.assert_array_equal(game.groups[0], np.flatnonzero(source))
    with pytest.raises(ValueError, match="disjoint"):
        DatasetValuation(
            model, data["x_train"], data["yr_train"], data["x_test"], data["yr_test"],
            groups=[np.arange(10), np.arange(5, 20)],
        )  # fmt: skip


def test_dataset_valuation_groups(data) -> None:
    model = DecisionTreeRegressor(random_state=0)
    game = DatasetValuation(
        model, data["x_train"], data["yr_train"], data["x_test"], data["yr_test"],
        task="regression", n_players=4, player_sizes="increasing", random_state=0,
    )  # fmt: skip
    sizes = [group.size for group in game.groups]
    assert sum(sizes) == 300
    assert sizes == sorted(sizes)
    assert np.unique(np.concatenate(game.groups)).size == 300  # disjoint and complete
    explicit = DatasetValuation(
        model, data["x_train"], data["yr_train"], data["x_test"], data["yr_test"],
        task="regression", groups=[np.arange(100), np.arange(100, 300)],
    )  # fmt: skip
    assert explicit.n_players == 2


def test_ensemble_selection_single_members_and_votes(data) -> None:
    members = [
        DecisionTreeClassifier(max_depth=depth, random_state=0).fit(
            data["x_train"], data["yc_train"]
        )
        for depth in (1, 2, 3)
    ]
    game = EnsembleSelection(
        members, data["x_test"], data["yc_test"], task="classification", normalize=False
    )
    values = game(np.eye(3, dtype=bool))
    for member, value in zip(members, values, strict=True):
        assert value == pytest.approx(member.score(data["x_test"], data["yc_test"]))
    assert game.player_name_lookup["0_DecisionTreeClassifier"] == 0
    regression = EnsembleSelection(
        [LinearRegression().fit(data["x_train"], data["yr_train"])] * 2,
        data["x_test"], data["yr_test"], task="regression", normalize=False,
    )  # fmt: skip
    assert regression(regression.grand_coalition)[0] == pytest.approx(1.0)
    with pytest.raises(TypeError, match="random forest"):
        RandomForestEnsembleSelection(
            LinearRegression(), data["x_test"], data["yr_test"], task="regression"
        )


def test_ensemble_selection_equals_votes_and_scores_coalition_by_coalition(data) -> None:
    """The vectorized game equals scipy's majority vote and scikit-learn's metrics, bit for bit."""
    x, y = make_classification(n_samples=300, n_classes=3, n_informative=3, random_state=0)
    members = [
        DecisionTreeClassifier(max_depth=depth, random_state=0).fit(x[:200], y[:200])
        for depth in (1, 2, 3, 4, 5)
    ]
    game = EnsembleSelection(members, x[200:], y[200:], empty_value=0.25, normalize=False)
    coalitions = np.random.default_rng(0).random((64, 5)) < 0.5
    votes = np.stack([member.predict(x[200:]) for member in members])
    expected = [
        accuracy_score(y[200:], mode(votes[coalition], axis=0).mode) if coalition.any() else 0.25
        for coalition in coalitions
    ]
    np.testing.assert_array_equal(game(coalitions), expected)

    regressors = [
        DecisionTreeRegressor(max_depth=depth, random_state=0).fit(
            data["x_train"], data["yr_train"]
        )
        for depth in (1, 2, 3, 4, 5)
    ]
    regression = EnsembleSelection(regressors, data["x_test"], data["yr_test"], normalize=False)
    predictions = np.stack([member.predict(data["x_test"]) for member in regressors])
    expected = [
        r2_score(data["yr_test"], predictions[coalition].mean(axis=0)) if coalition.any() else 0.0
        for coalition in coalitions
    ]
    np.testing.assert_array_equal(regression(coalitions), expected)


def test_labels_as_a_column_vector_give_the_same_games(data) -> None:
    """Labels of shape (n, 1), e.g. ``df[["target"]].to_numpy()``, mean the same as shape (n,)."""
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    rng = np.random.default_rng(0)
    for game_class, x_train, y_train in (
        (FeatureSelection, data["x_train"], data["yc_train"]),
        (DataValuation, data["x_train"][:6], data["yc_train"][:6]),  # six points as players
    ):
        flat = game_class(tree, x_train, y_train, data["x_test"], data["yc_test"])
        column = game_class(
            tree, x_train, y_train[:, None], data["x_test"], data["yc_test"][:, None]
        )
        coalitions = rng.random((8, flat.n_players)) < 0.5
        np.testing.assert_array_equal(column(coalitions), flat(coalitions))
    coalitions = rng.random((8, 3)) < 0.5
    members = [
        DecisionTreeRegressor(max_depth=depth, random_state=0).fit(
            data["x_train"], data["yr_train"]
        )
        for depth in (1, 2, 3)
    ]
    flat = EnsembleSelection(members, data["x_test"], data["yr_test"])
    column = EnsembleSelection(members, data["x_test"], data["yr_test"][:, None])
    np.testing.assert_array_equal(column(coalitions), flat(coalitions))
    with pytest.raises(ValueError, match="one target"):
        EnsembleSelection(members, data["x_test"], np.ones((len(data["x_test"]), 2)))


@pytest.mark.parametrize(
    ("name", "reference"),
    [
        ("accuracy", accuracy_score),
        ("r2", r2_score),
        ("neg_mse", lambda y, p: -mean_squared_error(y, p)),
        ("neg_mae", lambda y, p: -mean_absolute_error(y, p)),
    ],
)
def test_named_metrics_are_those_of_scikit_learn_bit_for_bit(name, reference) -> None:
    rng = np.random.default_rng(0)
    task = "classification" if name == "accuracy" else "regression"
    if task == "classification":
        targets = [rng.integers(0, 3, size=50), np.array(["a", "b"])[rng.integers(0, 2, size=50)]]
    else:  # also a constant target and integer targets
        targets = [rng.normal(size=50) * 1e3, np.full(50, 2.0), rng.integers(0, 3, size=50)]
    for y in targets:
        other = y + rng.normal(size=50) if task == "regression" else np.roll(y, 1)
        predictions = np.stack([y, rng.permutation(y), other])
        expected = [float(reference(y, p)) for p in predictions]
        np.testing.assert_array_equal(resolve_row_metric(name, task)(y, predictions), expected)
        assert [resolve_metric(name, task)(y, p) for p in predictions] == expected


def test_ensemble_selection_with_any_labels(data) -> None:
    labels = np.array(["no", "yes"])
    members = [
        DecisionTreeClassifier(max_depth=depth, random_state=0).fit(
            data["x_train"], labels[data["yc_train"]]
        )
        for depth in (1, 3)
    ]
    game = EnsembleSelection(members, data["x_test"], labels[data["yc_test"]], normalize=False)
    assert game(game.grand_coalition)[0] > 0.8

    # the trees of a forest predict indices into the forest's classes, here 1 and 2
    forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(
        data["x_train"], data["yc_train"] + 1
    )
    forest_game = RandomForestEnsembleSelection(
        forest, data["x_test"], data["yc_test"] + 1, normalize=False
    )
    assert forest_game(forest_game.grand_coalition)[0] == pytest.approx(
        forest.score(data["x_test"], data["yc_test"] + 1), abs=0.05
    )
    with pytest.raises(ValueError, match="does not know"):
        RandomForestEnsembleSelection(forest, data["x_test"], data["yc_test"] + 5)
    # the members of other ensembles may predict labels instead of indices into classes_
    from sklearn.ensemble import AdaBoostClassifier

    boosted = AdaBoostClassifier(n_estimators=3, random_state=0).fit(
        data["x_train"], data["yc_train"]
    )
    with pytest.raises(TypeError, match="random forest"):
        RandomForestEnsembleSelection(boosted, data["x_test"], data["yc_test"])


def test_uncertainty_decomposition(data) -> None:
    forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(
        data["x_train"], data["yc_train"]
    )
    values = {
        kind: TabularUncertaintyExplanation(
            forest, data["x_train"], x=1, uncertainty=kind, normalize=False
        )
        for kind in ("total", "aleatoric", "epistemic")
    }
    full = {kind: game(game.grand_coalition)[0] for kind, game in values.items()}
    assert full["total"] == pytest.approx(full["aleatoric"] + full["epistemic"])
    with pytest.raises(TypeError, match="RandomForestClassifier"):
        TabularUncertaintyExplanation(
            DecisionTreeClassifier().fit(data["x_train"], data["yc_train"]), data["x_train"]
        )
    # an imputer object would explain its own model instead of the forest's uncertainty
    from shapiq.imputer import BaselineImputer

    imputer = BaselineImputer(model=forest.predict, data=data["x_train"], x=data["x_train"][1])
    with pytest.raises(TypeError, match="imputer"):
        TabularUncertaintyExplanation(forest, data["x_train"], imputer=imputer)


def test_clustering_and_unsupervised_games(data) -> None:
    game = ClusterExplanation(data["x_train"], n_clusters=2, normalize=False)
    assert game(game.empty_coalition)[0] == 0.0
    assert game(game.grand_coalition)[0] > 0.0
    with pytest.raises(ValueError, match="method"):
        ClusterExplanation(data["x_train"], method="dbscan")  # type: ignore[arg-type]

    duplicated = np.column_stack(
        [data["x_train"][:, 0], data["x_train"][:, 0], data["x_train"][:, 1]]
    )
    game = UnsupervisedData(duplicated, n_bins=5)
    np.testing.assert_allclose(game(np.eye(3, dtype=bool)), 0.0)  # single features
    assert (
        game(np.array([[1, 1, 0]], dtype=bool))[0] > 0.0
    )  # duplicated features share all information
    discrete = np.array([[0, 0], [1, 1], [0, 0], [1, 1]])
    assert total_correlation(discrete) == pytest.approx(np.log(2))


def test_entropies_are_those_of_scipy_bit_for_bit() -> None:
    """The fast entropy and joint entropy reproduce scipy and ``np.unique(axis=0)`` exactly."""
    rng = np.random.default_rng(0)
    for size in (1, 2, 7, 100, 10_000):
        counts = rng.integers(1, 1000, size=size)
        assert _entropy(counts) == float(scipy_entropy(counts))
    data = rng.integers(-3, 4, size=(2000, 6))
    data[:, 2] = rng.integers(0, 10**15, size=2000)  # codes that would overflow int64 together
    expected = sum(
        float(scipy_entropy(np.unique(column, return_counts=True)[1])) for column in data.T
    ) - float(scipy_entropy(np.unique(data, axis=0, return_counts=True)[1]))
    assert total_correlation(data) == expected


def test_confounding_game_with_a_linear_regressor() -> None:
    rng = np.random.default_rng(0)
    covariates = rng.normal(size=(300, 4))
    treatment = (rng.random(300) < 1 / (1 + np.exp(-covariates[:, 0]))).astype(float)
    outcome = covariates[:, 0] + covariates[:, 1] + treatment * (1.0 + covariates[:, 2])
    game = GlobalConfoundingXAI(covariates, treatment, outcome, regressor=LinearRegression)
    # adjusting for every covariate reproduces the reference effect
    assert game(game.grand_coalition)[0] == pytest.approx(0.0, abs=1e-10)
    naive = game.Y[game.A == 1].mean() - game.Y[game.A == 0].mean()
    assert game(game.empty_coalition)[0] == pytest.approx(game.tau_hat.mean() - naive)


@skip_if_no_skimage
def test_image_classifier_superpixels_cover_every_player() -> None:
    image = np.random.default_rng(0).integers(0, 255, (60, 60, 3), dtype=np.uint8)
    game = ImageClassifier(
        image, model=mean_brightness_classifier, n_superpixels=9, normalize=False
    )
    assert set(np.unique(game.regions)) == set(range(game.n_players))
    expected = mean_brightness_classifier(image[None])[0, game.class_index]
    assert game(game.grand_coalition)[0] == pytest.approx(expected)
    grayscale = ImageClassifier(image[..., 0], model=mean_brightness_classifier, n_superpixels=4)
    assert grayscale.image.shape == (60, 60, 3)
    with pytest.raises(ValueError, match="vision transformer"):
        ImageClassifier(image, model=mean_brightness_classifier, revision="main")


def test_image_classifier_forwards_the_vit_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """The revision reaches the Hugging Face loader."""
    calls: list[dict] = []

    class FakeViT:
        def __init__(self, image: np.ndarray, n_players: int, **kwargs: object) -> None:
            calls.append(kwargs)
            self.image, self.n_players = image, n_players
            self.regions = np.arange(64).reshape(8, 8) % n_players
            self.categories = [f"class {i}" for i in range(4)]

        def __call__(self, coalitions: np.ndarray) -> np.ndarray:  # class 3: the visible share
            probabilities = np.zeros((coalitions.shape[0], 4))
            probabilities[:, 3] = coalitions.mean(axis=1)
            return probabilities

    import shapiq_games.vision.image_classifier as module

    monkeypatch.setattr(module, "ViTTokenModel", FakeViT)
    game = ImageClassifier(np.zeros((8, 8, 3), np.uint8), "vit_16_patches", revision="v1")
    assert calls[0]["revision"] == "v1"
    assert game.class_name == "class 3"


def test_sentiment_analysis_with_a_fake_pipeline() -> None:
    pipeline = FakeSentimentPipeline()
    game = SentimentAnalysis("good good bad plot", classifier=pipeline, normalize=False)
    assert game.n_players == 4
    assert game.original_model_output == pytest.approx(0.2)  # 2 * 0.6 - 1
    # masking the two 'good' tokens leaves one 'bad': a negative score
    assert game(np.array([[0, 0, 1, 1]], dtype=bool))[0] == pytest.approx(-0.2)
    # the score is continuous: an undecided classifier scores 0, not +-0.5
    assert game(np.array([[1, 0, 1, 1]], dtype=bool))[0] == pytest.approx(0.0)
    # padded batches can round a text differently: one text per forward pass
    assert set(pipeline.batch_sizes) == {1}
    removed = SentimentAnalysis(
        "good bad", classifier=pipeline, mask_strategy="remove", normalize=False
    )
    assert removed(np.array([[1, 0]], dtype=bool))[0] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="mask_strategy"):
        SentimentAnalysis("good", classifier=pipeline, mask_strategy="drop")  # type: ignore[arg-type]


def test_sentiment_analysis_with_other_labels_and_tokenizers() -> None:
    labelled = FakeSentimentPipeline(labels=("LABEL_1", "LABEL_0"))
    game = SentimentAnalysis(
        "good good bad", classifier=labelled, positive_label="LABEL_1", normalize=False
    )
    assert game.original_model_output == pytest.approx(0.2)
    with pytest.raises(ValueError, match="no label 'POSITIVE'"):
        SentimentAnalysis("good", classifier=labelled)
    # special tokens are left out of the players, whatever the tokenizer adds
    assert game.n_players == 3

    no_mask = FakeSentimentPipeline()
    no_mask.tokenizer.mask_token_id = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="no mask token"):
        SentimentAnalysis("good", classifier=no_mask)
    removed = SentimentAnalysis("good bad", classifier=no_mask, mask_strategy="remove")
    assert removed.n_players == 2


def test_task_is_inferred_from_the_model() -> None:
    """The games read the task off the model, so it need not be passed."""
    x, y = make_classification(n_samples=80, n_features=4, random_state=0)
    tree = DecisionTreeClassifier(random_state=0)
    inferred = DataValuation(tree, x[:6], y[:6], x[40:], y[40:])
    explicit = DataValuation(tree, x[:6], y[:6], x[40:], y[40:], task="classification")
    assert inferred.task == "classification"
    coalitions = np.random.default_rng(0).random((8, 6)) < 0.5
    np.testing.assert_allclose(inferred(coalitions), explicit(coalitions))

    regressor = DecisionTreeRegressor(random_state=0)
    target = x[:, 0] * 2.0
    assert (
        FeatureSelection(regressor, x[:40], target[:40], x[40:], target[40:]).task == "regression"
    )
    members = [DecisionTreeClassifier(max_depth=d, random_state=0).fit(x, y) for d in (1, 2, 3)]
    assert EnsembleSelection(members, x[40:], y[40:]).task == "classification"
    with pytest.raises(ValueError, match="task must be"):
        FeatureSelection(tree, x[:40], y[:40], x[40:], y[40:], task="ranking")


def test_confounding_games_fit_their_own_reference_effect() -> None:
    """Without tau_hat, the reference effects come from an S-learner on all covariates."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    treatment = (rng.random(200) < 0.5).astype(float)
    outcome = x[:, 0] + treatment * (1.0 + x[:, 1]) + rng.normal(scale=0.1, size=200)
    game = GlobalConfoundingXAI(x, treatment, outcome, regressor=LinearRegression)

    s_learner = LinearRegression().fit(np.column_stack([x, treatment]), outcome)
    effect = s_learner.predict(np.column_stack([x, np.ones(200)])) - s_learner.predict(
        np.column_stack([x, np.zeros(200)])
    )
    np.testing.assert_allclose(game.tau_hat, effect)
    local = LocalConfoundingXAI(x, treatment, outcome, unit=3, regressor=LinearRegression)
    assert local.n_players == 4
    np.testing.assert_allclose(local.unit, x[3])


@pytest.mark.skipif(not is_installed("tabpfn"), reason="tabpfn is not installed")
def test_tabpfn_regressor_is_v2_unless_a_version_is_chosen() -> None:
    """The causal games' default regressor needs no license token; building downloads nothing."""
    from shapiq_games.causal import tabpfn_regressor

    model = tabpfn_regressor()
    assert model.model_path.endswith("tabpfn-v2-regressor.ckpt")
    assert model.n_estimators == 1
    assert model.inference_config == {"REGRESSION_Y_PREPROCESS_TRANSFORMS": (None,)}
    paper = tabpfn_regressor(version="v2.5", inference_config={"PASSTHROUGH_INF": True})
    assert "v2.5" in paper.model_path
    assert paper.inference_config == {
        "REGRESSION_Y_PREPROCESS_TRANSFORMS": (None,),
        "PASSTHROUGH_INF": True,
    }
    with pytest.raises(ValueError, match="has no TabPFN 'v0.9'"):
        tabpfn_regressor(version="v0.9")
