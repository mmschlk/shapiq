from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shapiq.explainer.base import Explainer

if TYPE_CHECKING:
    from shapiq.explainer.nn.games.base import NNExplainerGameBase

import numpy as np
import pytest

from shapiq.explainer.nn import KNNExplainer, ThresholdNNExplainer, WeightedKNNExplainer
from shapiq.explainer.nn.games.knn import KNNExplainerGame
from shapiq.explainer.nn.games.tnn import TNNExplainerGame
from shapiq.explainer.nn.games.wknn import WeightedKNNExplainerGame


@pytest.mark.parametrize(
    "model_fixture, explainer_cls",
    [
        ("sklearn_knn_model", KNNExplainer),
        ("sklearn_wknn_model", WeightedKNNExplainer),
        ("sklearn_tnn_model", ThresholdNNExplainer),
    ],
)
def test_select_explainer(
    model_fixture: str,
    explainer_cls: type[Explainer],
    request: pytest.FixtureRequest,
):
    model = request.getfixturevalue(model_fixture)
    explainer = Explainer(model, class_index=0, max_order=1)

    assert isinstance(explainer, explainer_cls)


@pytest.mark.parametrize(
    "model_fixture, game_cls, explainer_cls, extra_kwargs",
    [
        ("sklearn_knn_model", KNNExplainerGame, KNNExplainer, {}),
        ("sklearn_wknn_model", WeightedKNNExplainerGame, WeightedKNNExplainer, {"n_bits": 3}),
        ("sklearn_tnn_model", TNNExplainerGame, ThresholdNNExplainer, {}),
    ],
)
def test_sv_values_agree_with_ground_truth_game(
    model_fixture: str,
    game_cls: type[NNExplainerGameBase],
    explainer_cls: type[Explainer],
    extra_kwargs: dict[str, Any],
    request: pytest.FixtureRequest,
    background_clf_dataset_small,
):
    model = request.getfixturevalue(model_fixture)
    X, y = background_clf_dataset_small
    n_classes = np.max(y) + 1

    rng = np.random.default_rng(seed=43)
    X_test = rng.multivariate_normal(np.mean(X, axis=0), np.cov(X, rowvar=False), size=10)

    for x_test in X_test:
        for class_index in range(n_classes):
            ground_truth_game = game_cls(model, x_test, class_index, **extra_kwargs)
            iv_expected = ground_truth_game.exact_values(index="SV", order=1)

            explainer = explainer_cls(model, class_index=class_index, **extra_kwargs)
            iv = explainer.explain(x_test)

            interactions = iv.interactions.keys()
            iv_expected_array = np.array([iv_expected.interactions[ia] for ia in interactions])
            iv_array = np.array([iv.interactions[ia] for ia in interactions])

            assert np.allclose(iv_expected_array, iv_array)


def test_knn_small_n(sklearn_knn_model, background_clf_dataset_small):
    """Test the KNNExplainer in the case where N < k."""
    X, y = background_clf_dataset_small
    X = X[:2]
    y = y[:2]
    n_classes = np.max(y) + 1

    for x_test in X:
        for class_index in range(n_classes):
            ground_truth_game = KNNExplainerGame(sklearn_knn_model, x_test, class_index)
            iv_expected = ground_truth_game.exact_values("SV", 1)
            knn_explainer = KNNExplainer(sklearn_knn_model, class_index=class_index)
            iv = knn_explainer.explain(x_test)

            interactions = iv.interactions.keys()
            iv_expected_array = np.array([iv_expected.interactions[ia] for ia in interactions])
            iv_array = np.array([iv.interactions[ia] for ia in interactions])

            assert np.allclose(iv_expected_array, iv_array)
