from __future__ import annotations

import numpy as np

from shapiq.explainer.nn import KNNExplainer

# TODO(Zaphoood): Apparently, shapiq_games.benchmark is deprecated, where should NN benchmarks go instead?  # noqa: TD003
from shapiq_games.benchmark.nn_xai.benchmark_knn import KNNExplainerXAI


def test_knn(sklearn_knn_model, background_clf_dataset_small):
    X, y = background_clf_dataset_small
    n_classes = np.max(y) + 1

    x_test = X[0]

    for class_index in range(n_classes):
        ground_truth_game = KNNExplainerXAI(sklearn_knn_model, x_test, class_index)
        iv_expected = ground_truth_game.exact_values("SV", 1)
        knn_explainer = KNNExplainer(sklearn_knn_model, class_index=class_index)
        iv = knn_explainer.explain(x_test)

        interactions = iv.interactions.keys()
        iv_expected_array = np.array([iv_expected.interactions[ia] for ia in interactions])
        iv_array = np.array([iv.interactions[ia] for ia in interactions])

        assert np.allclose(iv_expected_array, iv_array)
