from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from shapiq_games.nn import KNNGame


class TestErrorHandling:
    def test_X_train_data_type_checks(
        self, sklearn_knn_model, background_clf_dataset_small
    ) -> None:
        X, _ = background_clf_dataset_small
        x = X[0]

        model = deepcopy(sklearn_knn_model)
        model._fit_X = "This is not a numpy array"

        with pytest.raises(TypeError, match="_fit_X"):
            KNNGame(model, x, 0)

        model = deepcopy(sklearn_knn_model)
        model._fit_X = np.array(["Not", "an", "array", "of", "numbers"])

        with pytest.raises(TypeError, match="dtype.+_fit_X"):
            KNNGame(model, x, 0)

        # Integer features should be converted to floating
        model = deepcopy(sklearn_knn_model)
        model._fit_X = np.array([1, 2, 3], dtype=np.int32)
        explainer = KNNGame(model, x, 0)
        assert np.issubdtype(explainer.X_train.dtype, np.floating)

    def test_y_train_data_type_checks(
        self, sklearn_knn_model, background_clf_dataset_small
    ) -> None:
        X, _ = background_clf_dataset_small
        x = X[0]

        model = deepcopy(sklearn_knn_model)
        model._y = "This is not a numpy array"

        with pytest.raises(TypeError, match="_y"):
            KNNGame(model, x, 0)

        model = deepcopy(sklearn_knn_model)
        model._y = np.array(["Not", "an", "array", "of", "numbers"])

        with pytest.raises(TypeError, match="dtype.+_y"):
            KNNGame(model, x, 0)

        model = deepcopy(sklearn_knn_model)
        model._y = np.array([[1, 2, 3], [4, 5, 6]])

        with pytest.raises(ValueError, match="[Mm]ulti-output.+classifiers"):
            KNNGame(model, x, 0)
