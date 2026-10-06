"""The game contract, checked for every game family.

Every game must be a proper, deterministic set function: the value of a coalition may not depend
on call order, batch composition, repetition, the instance (given the same arguments), or whether
the coalition is passed as a boolean or an integer 0/1 matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

import shapiq_games as sg
from tests.shapiq_games.helpers import (
    FakeSentimentPipeline,
    is_installed,
    mean_brightness_classifier,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq import Game


def _object_local_explanation() -> Game:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    model = DecisionTreeRegressor(max_depth=3, random_state=0).fit(x, x[:, 0] * x[:, 1])
    return sg.LocalExplanation(model, x, x=3, random_state=0)


def _image_game() -> Game:
    image = np.random.default_rng(0).integers(0, 255, (48, 48, 3), dtype=np.uint8)
    return sg.ImageClassifier(image, model=mean_brightness_classifier, n_superpixels=6)


# name -> (factory, centered: whether v(empty) == 0)
GAMES: dict[str, tuple[Callable[[], Game], bool]] = {
    "dummy": (lambda: sg.DummyGame(5, interaction=(1, 2)), False),
    "unanimity": (lambda: sg.UnanimityGame(np.array([1, 0, 1, 0])), True),
    "soum": (lambda: sg.SOUM(6, 10, random_state=3, normalize=True), True),
    "random_table": (lambda: sg.RandomTableGame(5, random_state=1, normalize=True), True),
    "local_xai_marginal": (
        lambda: sg.LocalExplanation.from_config(dataset="xor", model="decision_tree"),
        True,
    ),
    "local_xai_baseline": (
        lambda: sg.LocalExplanation.from_config(
            dataset="condind", model="random_forest", imputer="baseline"
        ),
        True,
    ),
    "local_xai_conditional": (
        lambda: sg.LocalExplanation.from_config(
            dataset="sphere", model="decision_tree", imputer="conditional", n_background=200
        ),
        True,
    ),
    "local_xai_objects": (_object_local_explanation, True),
    "global_xai": (
        lambda: sg.GlobalExplanation.from_config(dataset="xor", model="decision_tree"),
        True,
    ),
    "feature_selection": (
        lambda: sg.FeatureSelection.from_config(dataset="breast_cancer", n_train=100),
        True,
    ),
    "data_valuation": (
        lambda: sg.DataValuation.from_config(dataset="xor", n_players=8, empty_value=0.5),
        True,
    ),
    "dataset_valuation": (
        lambda: sg.DatasetValuation.from_config(
            dataset="independentlinear60",
            dataset_params={"n_samples": 300},
            n_players=5,
            player_sizes="increasing",
        ),
        False,
    ),
    "ensemble_selection": (
        lambda: sg.EnsembleSelection.from_config(
            dataset="breast_cancer", members=["linear", "decision_tree", "knn", "random_forest"]
        ),
        False,
    ),
    "random_forest_ensemble_selection": (
        lambda: sg.RandomForestEnsembleSelection.from_config(dataset="xor", n_members=5),
        False,
    ),
    "uncertainty": (
        lambda: sg.UncertaintyExplanation.from_config(dataset="breast_cancer"),
        True,
    ),
    "clustering": (
        lambda: sg.ClusterExplanation.from_config(dataset="group", n_samples=200),
        False,
    ),
    "unsupervised": (lambda: sg.UnsupervisedData.from_config(dataset="xor"), True),
    "global_confounding": (
        lambda: sg.GlobalConfoundingXAI.from_config(n=200, regressor=LinearRegression),
        False,
    ),
    "local_confounding": (
        lambda: sg.LocalConfoundingXAI.from_config(
            n=200, unit=2, mode="sq", regressor=LinearRegression
        ),
        False,
    ),
    "path_dependent_tree": (
        lambda: sg.PathDependentTreeGame.from_config(dataset="xor", model="random_forest"),
        True,
    ),
    "interventional_tree": (
        lambda: sg.InterventionalTreeGame.from_config(
            dataset="breast_cancer", model="decision_tree", normalize=True
        ),
        True,
    ),
    "knn": (lambda: sg.KNNGame.from_config(dataset="xor", n_train=8), True),
    "weighted_knn": (
        lambda: sg.WeightedKNNGame.from_config(dataset="breast_cancer", n_train=8, n_bits=3),
        True,
    ),
    "threshold_nn": (
        lambda: sg.ThresholdNNGame.from_config(
            dataset="xor", n_train=8, model_params={"radius": 1.0}
        ),
        False,
    ),
    "product_kernel": (
        lambda: sg.ProductKernelGame.from_config(
            dataset="breast_cancer", model="svm", n_train=100, normalize=True
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
    _, game, _, _ = game_pair
    coalitions = _coalitions(game.n_players)
    batched = game(coalitions)
    reversed_batch = game(coalitions[::-1])[::-1]
    one_by_one = np.array([game(coalition.reshape(1, -1))[0] for coalition in coalitions])
    np.testing.assert_allclose(batched, reversed_batch, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(batched, one_by_one, rtol=1e-12, atol=1e-12)


def test_same_arguments_give_same_game(game_pair: tuple[str, Game, Game, bool]) -> None:
    _, game, other, _ = game_pair
    coalitions = _coalitions(game.n_players)
    np.testing.assert_allclose(game(coalitions), other(coalitions), rtol=1e-12, atol=1e-12)


def test_integer_coalitions_equal_boolean_coalitions(
    game_pair: tuple[str, Game, Game, bool],
) -> None:
    _, game, _, _ = game_pair
    coalitions = _coalitions(game.n_players)
    np.testing.assert_allclose(game(coalitions), game(coalitions.astype(int)))


def test_centered_games_vanish_on_the_empty_coalition(
    game_pair: tuple[str, Game, Game, bool],
) -> None:
    _, game, _, centered = game_pair
    if not centered:
        pytest.skip("this game is not centered")
    assert game(game.empty_coalition)[0] == pytest.approx(0.0, abs=1e-12)


def test_configured_games_have_stable_fingerprints(
    game_pair: tuple[str, Game, Game, bool],
) -> None:
    _, game, other, _ = game_pair
    fingerprint = getattr(game, "fingerprint", None)
    if fingerprint is None:
        pytest.skip("this game was not built from a configuration")
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 16
    assert fingerprint == other.fingerprint


def test_fingerprint_changes_with_configuration() -> None:
    first = sg.KNNGame.from_config(dataset="xor", n_train=8, random_state=0)
    second = sg.KNNGame.from_config(dataset="xor", n_train=8, random_state=1)
    third = sg.KNNGame.from_config(dataset="xor", n_train=8, random_state=0)
    assert first.fingerprint != second.fingerprint
    assert first.fingerprint == third.fingerprint
    assert first.config is not None
    assert first.config["random_state"] == 0
