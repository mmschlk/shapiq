"""Semantic tests for the model-specific games (trees, nearest neighbors, product kernels)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor

from shapiq.explainer.product_kernel.conversion import convert_svm
from shapiq_games._base import predicts_row_by_row
from shapiq_games.kernel import ProductKernelGame
from shapiq_games.nn import KNNGame, ThresholdNNGame, WeightedKNNGame
from shapiq_games.nn.weighted_knn import quantize_weights
from shapiq_games.tree import InterventionalTreeGame, PathDependentTreeGame
from tests.shapiq_games.helpers import is_installed


@pytest.fixture(scope="module")
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 4))
    return x, x[:, 0] + 2 * x[:, 1] * x[:, 2]


@pytest.fixture(scope="module")
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 4))
    return x, (x[:, 0] + x[:, 1] > 0).astype(int)


def test_path_dependent_game_matches_the_model(regression_data, classification_data) -> None:
    x, y = regression_data
    tree = DecisionTreeRegressor(max_depth=5, random_state=0).fit(x, y)
    game = PathDependentTreeGame(tree, x[0], normalize=False)
    assert game(game.grand_coalition)[0] == pytest.approx(tree.predict(x[:1])[0])
    # without features, the tree predicts its sample-weighted mean leaf value
    assert game(game.empty_coalition)[0] == pytest.approx(y.mean())

    x, y = classification_data
    forest = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0).fit(x, y)
    game = PathDependentTreeGame(forest, x[0])
    assert game.class_index == 1
    full = game(game.grand_coalition)[0] + game.normalization_value
    assert full == pytest.approx(forest.predict_proba(x[:1])[0, 1])


def test_path_dependent_game_does_not_depend_on_the_batch(regression_data) -> None:
    """The game evaluates a batch in chunks of coalitions: the same floats for any chunking."""
    x, y = regression_data
    forest = RandomForestClassifier(n_estimators=3, random_state=0).fit(x, y > 0)
    game = PathDependentTreeGame(forest, x[0])
    coalitions = np.random.default_rng(0).random((1500, 4)) < 0.5
    together = game(coalitions)
    in_parts = np.concatenate([game(coalitions[i : i + 100]) for i in range(0, 1500, 100)])
    np.testing.assert_array_equal(together, in_parts)
    one_by_one = np.concatenate([game(coalitions[i : i + 1]) for i in range(50)])
    np.testing.assert_array_equal(together[:50], one_by_one)


@pytest.mark.skipif(not is_installed("xgboost"), reason="xgboost is not installed")
def test_path_dependent_game_of_xgboost_matches_tree_shap(regression_data) -> None:
    """XGBoost stores node weights as float32; the game computes the shares in float64."""
    import xgboost

    from shapiq import ExactComputer, TreeExplainer

    x, y = regression_data
    model = xgboost.XGBRegressor(n_estimators=10, max_depth=4, n_jobs=1).fit(x, y)
    game = PathDependentTreeGame(model, x[0], normalize=False)
    shapley = ExactComputer(game, n_players=4)("SV", 1).get_n_order_values(1)
    expected = TreeExplainer(model, index="SV", max_order=1).explain(x[0])
    np.testing.assert_allclose(shapley, expected.get_n_order_values(1), rtol=0, atol=1e-10)


def test_only_tree_models_predict_row_by_row() -> None:
    forest = RandomForestClassifier(n_estimators=2)
    assert predicts_row_by_row(forest)
    assert predicts_row_by_row(forest.predict_proba)  # a bound method of a tree model
    assert not predicts_row_by_row(SVC())  # BLAS can round a row differently in larger calls
    assert not predicts_row_by_row(lambda rows: rows.sum(axis=1))
    # gradient boosting starts from its init model, here a linear one
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression

    assert predicts_row_by_row(GradientBoostingRegressor())
    assert not predicts_row_by_row(GradientBoostingRegressor(init=LinearRegression()))


def test_interventional_game_matches_its_definition(regression_data) -> None:
    x, y = regression_data
    tree = DecisionTreeRegressor(max_depth=4, random_state=0).fit(x, y)
    reference = x[:20]
    game = InterventionalTreeGame(tree, reference, x[50])
    coalition = np.array([True, False, True, False])
    expected = tree.predict(np.where(coalition, x[50], reference)).mean()
    assert game(coalition.reshape(1, -1))[0] == pytest.approx(expected)
    assert game.normalization_value == 0.0  # not normalized by default


@pytest.mark.skipif(not is_installed("xgboost"), reason="xgboost is not installed")
def test_interventional_game_uses_margins_for_boosted_classifiers(classification_data) -> None:
    import xgboost as xgb

    x, y = classification_data
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=0, n_jobs=1).fit(x, y)
    game = InterventionalTreeGame(model, x[:10], x[0])
    margin = model.get_booster().predict(xgb.DMatrix(x[:1]), output_margin=True)[0]
    assert game(game.grand_coalition)[0] == pytest.approx(margin, rel=1e-6)


@pytest.mark.skipif(not is_installed("xgboost"), reason="xgboost is not installed")
def test_interventional_game_accepts_boosters_trained_on_dataframes(classification_data) -> None:
    import pandas as pd
    import xgboost as xgb

    x, y = classification_data
    frame = pd.DataFrame(x, columns=["a", "b", "c", "d"])
    model = xgb.XGBClassifier(n_estimators=3, max_depth=2, n_jobs=1).fit(frame, y)
    game = InterventionalTreeGame(model, x[:10], x[0])
    margin = model.predict(frame.iloc[:1], output_margin=True)[0]
    assert game(game.grand_coalition)[0] == pytest.approx(margin, rel=1e-6)


@pytest.mark.skipif(not is_installed("lightgbm"), reason="lightgbm is not installed")
def test_tree_games_explain_the_raw_scores_of_lightgbm_boosters(classification_data) -> None:
    """A raw ``Booster`` is no classifier, but its class_index still selects the class."""
    import lightgbm as lgb

    x, _ = classification_data
    labels = np.digitize(x[:, 0] + x[:, 1] * x[:, 2], [-0.5, 0.5])
    params = {"objective": "multiclass", "num_class": 3, "num_leaves": 8, "verbose": -1}
    booster = lgb.train(params, lgb.Dataset(x, labels), num_boost_round=5)
    raw = booster.predict(x[:1], raw_score=True)[0]
    for class_index in (0, 2):
        for game in (
            PathDependentTreeGame(booster, x[0], class_index=class_index, normalize=False),
            InterventionalTreeGame(booster, x[:10], x[0], class_index=class_index),
        ):
            assert game.class_index == class_index
            assert game(game.grand_coalition)[0] == pytest.approx(raw[class_index])
    with pytest.raises(ValueError, match="pass the class_index"):
        InterventionalTreeGame(booster, x[:10], x[0])
    binary = lgb.train(
        {"objective": "binary", "verbose": -1}, lgb.Dataset(x, labels > 0), num_boost_round=5
    )
    game = InterventionalTreeGame(binary, x[:10], x[0])
    assert game(game.grand_coalition)[0] == pytest.approx(binary.predict(x[:1], raw_score=True)[0])


def _line_knn(k: int) -> tuple[KNeighborsClassifier, np.ndarray]:
    """Training points at 1, 2, 3, 4 with labels 1, 0, 1, 1; explained point at 0."""
    x_train = np.array([[1.0], [2.0], [3.0], [4.0]])
    model = KNeighborsClassifier(n_neighbors=k).fit(x_train, np.array([1, 0, 1, 1]))
    return model, np.array([0.0])


def test_nn_games_check_the_class_indices_of_the_model() -> None:
    model, x = _line_knn(k=2)
    model._y = np.array(["a", "b", "c", "d"])
    with pytest.raises(TypeError, match="dtype.+_y"):
        KNNGame(model, x)
    model._y = np.array([[1, 0], [0, 1], [1, 1], [1, 0]])
    with pytest.raises(ValueError, match="[Mm]ulti-output"):
        KNNGame(model, x)


def test_knn_games_check_the_weights_of_the_model() -> None:
    """Each game describes one weighting (as its explainer); the other one is refused."""
    model, x = _line_knn(k=2)
    with pytest.raises(ValueError, match="weights='distance'"):
        WeightedKNNGame(model, x)
    model.set_params(weights="distance")
    with pytest.raises(ValueError, match="weights='uniform'"):
        KNNGame(model, x)
    assert WeightedKNNGame(model, x).n_players == 4


def test_knn_game_counts_nearest_neighbors_of_the_class() -> None:
    model, x = _line_knn(k=2)
    game = KNNGame(model, x, class_index=1)
    coalitions = np.array(
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0]], dtype=bool
    )
    # the two nearest players of each coalition, and the share of class 1 among k=2 slots
    np.testing.assert_allclose(game(coalitions), [0.0, 0.5, 0.5, 1.0, 0.0])


def test_threshold_nn_game_defaults_to_uniform_utility() -> None:
    x_train = np.array([[0.1], [0.2], [5.0]])
    model = RadiusNeighborsClassifier(radius=1.0).fit(x_train, np.array([1, 0, 1]))
    game = ThresholdNNGame(model, np.array([0.0]), class_index=1)
    coalitions = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=bool)
    # outside the radius only the third point; empty neighborhoods give 1 / n_classes
    np.testing.assert_allclose(game(coalitions), [0.5, 0.5, 0.5, 1.0])


def test_weighted_knn_quantization_and_binary_games() -> None:
    np.testing.assert_allclose(quantize_weights(np.array([0.3, 0.26, 1.0]), 2), [0.25, 0.25, 1.0])
    model = KNeighborsClassifier(n_neighbors=2, weights="distance")
    model.fit(np.array([[1.0], [2.0], [3.0]]), np.array([0, 1, 2]))
    game = WeightedKNNGame(model, np.array([0.0]), class_index=0, n_bits=3)
    assert game.other_classes == [1, 2]
    assert game(game.empty_coalition)[0] == 0.0
    # only the explained class present: it wins every binary game
    assert game(np.array([[1, 0, 0]], dtype=bool))[0] == 1.0


def test_product_kernel_game_matches_the_decision_function(classification_data) -> None:
    x, y = classification_data
    svc = SVC(kernel="rbf", gamma=0.5).fit(x[:100], y[:100])
    game = ProductKernelGame(svc, x[0])
    assert game(game.grand_coalition)[0] == pytest.approx(svc.decision_function(x[:1])[0])
    converted = convert_svm(svc)
    assert game(game.empty_coalition)[0] == pytest.approx(
        converted.alpha.sum() + converted.intercept
    )
    # the converted model gives the same game
    np.testing.assert_allclose(
        ProductKernelGame(converted, x[0])(np.eye(4, dtype=bool)), game(np.eye(4, dtype=bool))
    )

    svr = SVR(kernel="rbf").fit(x[:100], x[:100, 0])
    game = ProductKernelGame(svr, x[1])
    assert game(game.grand_coalition)[0] == pytest.approx(svr.predict(x[1:2])[0])


def test_product_kernel_game_resolves_the_kernel_width(regression_data) -> None:
    """Each factor of the product kernel has one feature: gamma=None is gamma=1."""
    x, y = regression_data
    converted = convert_svm(SVR(kernel="rbf", gamma=1.0).fit(x[:60], y[:60]))
    unit = ProductKernelGame(converted, x[0], normalize=False)
    coalitions = np.random.default_rng(0).random((20, 4)) < 0.5
    for gamma in (None, np.array([1.0])):
        converted.gamma = gamma
        game = ProductKernelGame(converted, x[0], normalize=False)
        np.testing.assert_array_equal(game(coalitions), unit(coalitions))
    converted.gamma = np.ones(4)
    with pytest.raises(NotImplementedError, match="one length scale"):
        ProductKernelGame(converted, x[0])


def test_product_kernel_game_rejects_other_models(regression_data) -> None:
    with pytest.raises(TypeError, match="Unsupported model"):
        ProductKernelGame(DecisionTreeRegressor(), np.zeros(3))
    # shapiq's conversion drops the target scaling of normalize_y
    x, y = regression_data
    scaled = GaussianProcessRegressor(normalize_y=True).fit(x[:50], y[:50])
    with pytest.raises(ValueError, match="normalize_y"):
        ProductKernelGame(scaled, x[0])
