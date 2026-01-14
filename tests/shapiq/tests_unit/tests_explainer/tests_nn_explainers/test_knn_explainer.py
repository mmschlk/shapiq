from __future__ import annotations

import numpy as np

from shapiq.explainer.nn import KNNExplainer
from shapiq.explainer.nn.games.knn import KNNExplainerGame


def test_knn(sklearn_knn_model, background_clf_dataset_small):
    X, y = background_clf_dataset_small
    n_classes = np.max(y) + 1

    # Generate random test points similar to train data (the distribution doesn't really matter but this ensures we
    # cover a good range of test data)
    rng = np.random.default_rng(seed=43)
    X_test = rng.multivariate_normal(np.mean(X, axis=0), np.cov(X, rowvar=False), size=10)

    for x_test in X_test:
        for class_index in range(n_classes):
            ground_truth_game = KNNExplainerGame(sklearn_knn_model, x_test, class_index)
            iv_expected = ground_truth_game.exact_values("SV", 1)
            knn_explainer = KNNExplainer(sklearn_knn_model, class_index=class_index)
            iv = knn_explainer.explain(x_test)

            interactions = iv.interactions.keys()
            iv_expected_array = np.array([iv_expected.interactions[ia] for ia in interactions])
            iv_array = np.array([iv.interactions[ia] for ia in interactions])

            assert np.allclose(iv_expected_array, iv_array)


def test_knn_small_n(sklearn_knn_model, background_clf_dataset_small):
    """Test the case where N < k."""
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
