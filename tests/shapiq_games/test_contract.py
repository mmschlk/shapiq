"""The game contract, checked for every game family.

Every game must be a proper, deterministic set function: the value of a coalition may not depend
on call order, batch composition, repetition, the instance (given the same arguments), or whether
the coalition is passed as a boolean or an integer 0/1 matrix. The games are built from objects;
the setups that build them from names are tested in ``tests/shapiq_benchmark/test_setups.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import shapiq_games as sg
from tests.shapiq_games.helpers import (
    FakeSentimentPipeline,
    is_installed,
    mean_brightness_classifier,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq import Game


_RNG = np.random.default_rng(0)
_X = _RNG.normal(size=(240, 4))
_Y_REG = _X[:, 0] * _X[:, 1] + _X[:, 2]
_Y_CLF = (_X[:, 0] + _X[:, 1] > 0).astype(int)
_X_TRAIN, _X_TEST = _X[:200], _X[200:]
_Y_REG_TRAIN, _Y_REG_TEST = _Y_REG[:200], _Y_REG[200:]
_Y_CLF_TRAIN, _Y_CLF_TEST = _Y_CLF[:200], _Y_CLF[200:]


def _tree_regressor() -> DecisionTreeRegressor:
    return DecisionTreeRegressor(max_depth=3, random_state=0).fit(_X_TRAIN, _Y_REG_TRAIN)


def _forest_classifier(n_estimators: int = 5) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=3, random_state=0).fit(
        _X_TRAIN, _Y_CLF_TRAIN
    )


def _knn(**params: object) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=3, **params).fit(_X_TRAIN[:8], _Y_CLF_TRAIN[:8])


def _treatment_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    treatment = (_RNG.random(200) < 1 / (1 + np.exp(-_X_TRAIN[:, 0]))).astype(float)
    outcome = _X_TRAIN[:, 0] + treatment * (1.0 + _X_TRAIN[:, 1])
    return _X_TRAIN, treatment, outcome


_TREATMENT = _treatment_data()


def _image_game() -> Game:
    image = np.random.default_rng(0).integers(0, 255, (48, 48, 3), dtype=np.uint8)
    return sg.ImageClassifier(image, model=mean_brightness_classifier, n_superpixels=6)


# name -> (factory, centered: whether v(empty) == 0); every game is built from objects
GAMES: dict[str, tuple[Callable[[], Game], bool]] = {
    "dummy": (lambda: sg.DummyGame(5, interaction=(1, 2)), False),
    "unanimity": (lambda: sg.UnanimityGame(np.array([1, 0, 1, 0])), True),
    "soum": (lambda: sg.SOUM(6, 10, random_state=3, normalize=True), True),
    "random_table": (lambda: sg.RandomTableGame(5, random_state=1, normalize=True), True),
    "local_xai_marginal": (
        lambda: sg.LocalExplanation(_tree_regressor(), _X_TRAIN[:50], x=_X_TEST[0]),
        True,
    ),
    "local_xai_baseline": (
        lambda: sg.LocalExplanation(
            _forest_classifier(), _X_TRAIN[:50], x=_X_TEST[1], imputer="baseline"
        ),
        True,
    ),
    "local_xai_conditional": (
        lambda: sg.LocalExplanation(
            _tree_regressor(), _X_TRAIN, x=_X_TEST[2], imputer="conditional", random_state=0
        ),
        True,
    ),
    "local_xai_index": (lambda: sg.LocalExplanation(_tree_regressor(), _X, x=3), True),
    "local_xai_missing": (
        lambda: sg.LocalExplanation(_tree_regressor(), _X_TRAIN, x=_X_TEST[4], imputer="missing"),
        True,
    ),
    "global_xai": (lambda: sg.GlobalExplanation(_tree_regressor(), _X_TEST), True),
    "feature_selection": (
        lambda: sg.FeatureSelection(
            DecisionTreeClassifier(max_depth=3, random_state=0),
            _X_TRAIN[:100],
            _Y_CLF_TRAIN[:100],
            _X_TEST,
            _Y_CLF_TEST,
        ),
        True,
    ),
    "data_valuation": (
        lambda: sg.DataValuation(
            DecisionTreeClassifier(random_state=0),
            _X_TRAIN[:8],
            _Y_CLF_TRAIN[:8],
            _X_TEST,
            _Y_CLF_TEST,
            empty_value=0.5,
        ),
        True,
    ),
    "dataset_valuation": (
        lambda: sg.DatasetValuation(
            LinearRegression(),
            _X_TRAIN,
            _Y_REG_TRAIN,
            _X_TEST,
            _Y_REG_TEST,
            n_players=5,
            player_sizes="increasing",
            random_state=0,
        ),
        False,
    ),
    "ensemble_selection": (
        lambda: sg.EnsembleSelection(
            [
                LogisticRegression().fit(_X_TRAIN, _Y_CLF_TRAIN),
                DecisionTreeClassifier(max_depth=2, random_state=0).fit(_X_TRAIN, _Y_CLF_TRAIN),
                _knn(),
                _forest_classifier(3),
            ],
            _X_TEST,
            _Y_CLF_TEST,
        ),
        False,
    ),
    "random_forest_ensemble_selection": (
        lambda: sg.RandomForestEnsembleSelection.from_forest(
            _forest_classifier(), _X_TEST, _Y_CLF_TEST
        ),
        False,
    ),
    "uncertainty": (
        lambda: sg.UncertaintyExplanation(_forest_classifier(), _X_TRAIN[:50], _X_TEST[0]),
        True,
    ),
    "clustering": (lambda: sg.ClusterExplanation(_X, n_clusters=3, random_state=0), False),
    "unsupervised": (lambda: sg.UnsupervisedData(_X_TRAIN, n_bins=5), True),
    "global_confounding": (
        lambda: sg.GlobalConfoundingXAI(*_TREATMENT, regressor=LinearRegression),
        False,
    ),
    "local_confounding": (
        lambda: sg.LocalConfoundingXAI(*_TREATMENT, unit=2, mode="sq", regressor=LinearRegression),
        False,
    ),
    "path_dependent_tree": (
        lambda: sg.PathDependentTreeGame(_forest_classifier(), _X_TEST[0]),
        True,
    ),
    "interventional_tree": (
        lambda: sg.InterventionalTreeGame(
            _tree_regressor(), _X_TRAIN[:30], _X_TEST[0], normalize=True
        ),
        True,
    ),
    "knn": (lambda: sg.KNNGame(_knn(), _X_TEST[0]), True),
    "weighted_knn": (
        lambda: sg.WeightedKNNGame(_knn(weights="distance"), _X_TEST[0], n_bits=3),
        True,
    ),
    "threshold_nn": (
        lambda: sg.ThresholdNNGame(
            RadiusNeighborsClassifier(radius=1.5).fit(_X_TRAIN[:8], _Y_CLF_TRAIN[:8]),
            _X_TEST[0],
        ),
        False,
    ),
    "product_kernel": (
        lambda: sg.ProductKernelGame(
            SVC(kernel="rbf", gamma=0.3).fit(_X_TRAIN[:100], _Y_CLF_TRAIN[:100]),
            _X_TEST[0],
            normalize=True,
        ),
        True,
    ),
    "sentiment": (
        lambda: sg.SentimentAnalysis(
            "a good movie but a bad ending good", classifier=FakeSentimentPipeline()
        ),
        True,
    ),
}

if is_installed("skimage"):
    GAMES["image_callable"] = (_image_game, True)


def _coalitions(n_players: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    coalitions = rng.random((8, n_players)) < 0.5
    coalitions[0] = False
    coalitions[1] = True
    return coalitions


@pytest.fixture(params=sorted(GAMES), scope="module")
def game_pair(request: pytest.FixtureRequest) -> tuple[str, Game, Game, bool]:
    """Two instances of the same game, built with the same arguments."""
    factory, centered = GAMES[request.param]
    return request.param, factory(), factory(), centered


def _fresh(name: str) -> Game:
    """A new instance: values a game caches (e.g. the confounding games) cannot hide a change."""
    return GAMES[name][0]()


def test_values_are_finite_and_shaped(game_pair: tuple[str, Game, Game, bool]) -> None:
    _, game, _, _ = game_pair
    coalitions = _coalitions(game.n_players)
    values = game(coalitions)
    assert values.shape == (coalitions.shape[0],)
    assert np.all(np.isfinite(values))


def test_repeated_evaluation_is_identical(game_pair: tuple[str, Game, Game, bool]) -> None:
    _, game, _, _ = game_pair
    coalitions = _coalitions(game.n_players)
    np.testing.assert_array_equal(game(coalitions), game(coalitions))


def test_batch_composition_does_not_matter(game_pair: tuple[str, Game, Game, bool]) -> None:
    name, game, _, _ = game_pair
    coalitions = _coalitions(game.n_players)
    batched = game(coalitions)
    reversed_batch = _fresh(name)(coalitions[::-1])[::-1]
    single = _fresh(name)
    one_by_one = np.array([single(coalition.reshape(1, -1))[0] for coalition in coalitions])
    np.testing.assert_allclose(batched, reversed_batch, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(batched, one_by_one, rtol=1e-12, atol=1e-12)


def test_same_arguments_give_same_game(game_pair: tuple[str, Game, Game, bool]) -> None:
    _, game, other, _ = game_pair
    coalitions = _coalitions(game.n_players)
    np.testing.assert_allclose(game(coalitions), other(coalitions), rtol=1e-12, atol=1e-12)


def test_integer_coalitions_equal_boolean_coalitions(
    game_pair: tuple[str, Game, Game, bool],
) -> None:
    name, game, _, _ = game_pair
    coalitions = _coalitions(game.n_players)
    np.testing.assert_allclose(game(coalitions), _fresh(name)(coalitions.astype(int)))


def test_centered_games_vanish_on_the_empty_coalition(
    game_pair: tuple[str, Game, Game, bool],
) -> None:
    _, game, _, centered = game_pair
    if not centered:
        pytest.skip("this game is not centered")
    assert game(game.empty_coalition)[0] == pytest.approx(0.0, abs=1e-12)


def _ignores_feature_3(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + 2 * x[:, 1] * x[:, 2]


def _tree_without_feature_3() -> DecisionTreeRegressor:
    """A tree that cannot split on feature 3: it is constant in the training data."""
    x = _X_TRAIN.copy()
    x[:, 3] = 0.0
    return DecisionTreeRegressor(max_depth=4, random_state=0).fit(x, _ignores_feature_3(x))


def _booster_without_feature_3() -> HistGradientBoostingRegressor:
    """A booster that reads missing values and never splits on the constant feature 3."""
    x = _X_TRAIN.copy()
    x[:, 3] = 0.0
    model = HistGradientBoostingRegressor(max_iter=30, random_state=0)
    return model.fit(x, _ignores_feature_3(x))


# games of a model that ignores feature 3, whose exact Shapley value must therefore be zero (not
# the conditional imputer: conditioning on an ignored feature changes the sampled background, so
# observational values need not vanish)
NULL_PLAYER_GAMES: dict[str, Callable[[], Game]] = {
    "local_xai_marginal": lambda: sg.LocalExplanation(_ignores_feature_3, _X_TRAIN, x=_X_TEST[0]),
    "local_xai_baseline": lambda: sg.LocalExplanation(
        _ignores_feature_3, _X_TRAIN, x=_X_TEST[0], imputer="baseline"
    ),
    "local_xai_missing": lambda: sg.LocalExplanation(
        _booster_without_feature_3(), _X_TRAIN, x=_X_TEST[0], imputer="missing"
    ),
    "global_xai": lambda: sg.GlobalExplanation(_tree_without_feature_3(), _X_TEST),
    "path_dependent_tree": lambda: sg.PathDependentTreeGame(_tree_without_feature_3(), _X_TEST[0]),
    "interventional_tree": lambda: sg.InterventionalTreeGame(
        _tree_without_feature_3(), _X_TRAIN[:20], _X_TEST[0]
    ),
}


@pytest.mark.parametrize("name", sorted(NULL_PLAYER_GAMES))
def test_a_feature_the_model_ignores_is_a_null_player(name: str) -> None:
    game = NULL_PLAYER_GAMES[name]()
    shapley = game.exact_values(index="SV", order=1)
    assert shapley[(3,)] == pytest.approx(0.0, abs=1e-10)
    assert max(abs(shapley[(i,)]) for i in range(3)) > 1e-6  # the other features matter
