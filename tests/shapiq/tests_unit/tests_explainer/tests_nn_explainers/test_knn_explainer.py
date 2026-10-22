from __future__ import annotations

import pytest

from shapiq.explainer.nn import KNNExplainer


class TestErrorHandling:
    def test_invalid_sklearn_weights(self, sklearn_wknn_model) -> None:
        with pytest.raises(ValueError, match="must use weights='uniform'"):
            KNNExplainer(sklearn_wknn_model)

    def test_n_neighbors_invalid_type(self, sklearn_knn_model) -> None:
        sklearn_knn_model.n_neighbors = "Not an int"
        with pytest.raises(TypeError, match="Expected .*n_neighbors to be int"):
            KNNExplainer(sklearn_knn_model)
