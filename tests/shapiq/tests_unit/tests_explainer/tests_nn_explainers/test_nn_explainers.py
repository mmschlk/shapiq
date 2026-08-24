from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shapiq.explainer.base import Explainer

if TYPE_CHECKING:
    from shapiq.explainer.nn.games.base import NNExplainerGameBase

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

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
    explainer = Explainer(model, class_index=0, index="SV", max_order=1)

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


@pytest.mark.parametrize(
    "weights, explainer_cls, game_cls",
    [
        ("uniform", KNNExplainer, KNNExplainerGame),
        ("distance", WeightedKNNExplainer, WeightedKNNExplainerGame),
    ],
)
def test_knn_small_training_set_rejected(
    weights: str,
    explainer_cls: type[Explainer],
    game_cls: type[NNExplainerGameBase],
    background_clf_dataset_small,
):
    """Models fitted on fewer training samples than ``n_neighbors`` are rejected at construction.

    scikit-learn itself refuses to predict with such models ("Expected n_neighbors <=
    n_samples_fit"), so there is no model behavior to explain.
    """
    X, y = background_clf_dataset_small
    model = KNeighborsClassifier(n_neighbors=3, weights=weights)
    model.fit(X[:2], y[:2])

    with pytest.raises(ValueError, match="n_neighbors <= n_samples_fit"):
        explainer_cls(model, class_index=0)

    with pytest.raises(ValueError, match="n_neighbors <= n_samples_fit"):
        game_cls(model, X[0], 0)
