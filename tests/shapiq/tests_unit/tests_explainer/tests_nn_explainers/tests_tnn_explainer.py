from __future__ import annotations

import numpy as np

from shapiq.explainer.nn import ThresholdNNExplainer
from shapiq_games.benchmark.nn_xai.benchmark_tnn import TNNExplainerXAI


def test_tnn(sklearn_tnn_model, background_clf_dataset_small):
    X, y = background_clf_dataset_small
    n_classes = np.max(y) + 1

    rng = np.random.default_rng(seed=43)
    X_test = rng.multivariate_normal(np.mean(X, axis=0), np.cov(X, rowvar=False), size=10)

    for x_test in X_test:
        for class_index in range(n_classes):
            ground_truth_game = TNNExplainerXAI(sklearn_tnn_model, x_test, class_index)
            iv_expected = ground_truth_game.exact_values("SV", 1)
            knn_explainer = ThresholdNNExplainer(sklearn_tnn_model, class_index=class_index)
            iv = knn_explainer.explain(x_test)

            interactions = iv.interactions.keys()
            iv_expected_array = np.array([iv_expected.interactions[ia] for ia in interactions])
            iv_array = np.array([iv.interactions[ia] for ia in interactions])

            assert np.allclose(iv_expected_array, iv_array)
