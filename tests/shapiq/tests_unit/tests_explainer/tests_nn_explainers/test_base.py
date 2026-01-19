from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from shapiq.explainer.nn import KNNExplainer
from shapiq.explainer.nn.base import _sklearn_model_get_private_attribute


class TestErrorHandling:
    def test_X_train_data_type_checks(self, sklearn_knn_model) -> None:
        model = deepcopy(sklearn_knn_model)
        model._fit_X = "This is not a numpy array"

        with pytest.raises(TypeError, match="_fit_X"):
            KNNExplainer(model)

        model = deepcopy(sklearn_knn_model)
        model._fit_X = np.array(["Not", "an", "array", "of", "numbers"])

        with pytest.raises(TypeError, match="dtype.+_fit_X"):
            KNNExplainer(model)

        # Integer features should be converted to floating
        model = deepcopy(sklearn_knn_model)
        model._fit_X = np.array([1, 2, 3], dtype=np.int32)
        explainer = KNNExplainer(model)
        assert np.issubdtype(explainer.X_train.dtype, np.floating)

    def test_y_train_data_type_checks(self, sklearn_knn_model) -> None:
        model = deepcopy(sklearn_knn_model)
        model._y = "This is not a numpy array"

        with pytest.raises(TypeError, match="_y"):
            KNNExplainer(model)

        model = deepcopy(sklearn_knn_model)
        model._y = np.array(["Not", "an", "array", "of", "numbers"])

        with pytest.raises(TypeError, match="dtype.+_y"):
            KNNExplainer(model)

        model = deepcopy(sklearn_knn_model)
        model._y = np.array([[1, 2, 3], [4, 5, 6]])

        with pytest.raises(ValueError, match="[Mm]ulti-output.+classifiers"):
            KNNExplainer(model)

    def test_sklearn_model_get_private_attribute(self, sklearn_knn_model) -> None:
        with pytest.raises(ValueError, match="must start with underscore"):
            _sklearn_model_get_private_attribute(sklearn_knn_model, "this_is_a_public_attr")

        with pytest.raises(AttributeError, match="Failed to access private attribute"):
            _sklearn_model_get_private_attribute(sklearn_knn_model, "_this_attr_doesnt_exist")
