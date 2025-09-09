"""Tests functionality that is common to both Gaussian imputers."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.imputer import GaussianImputer, GaussianCopulaImputer
from shapiq.imputer.gaussian_imputer_exceptions import CategoricalFeatureError



@pytest.mark.parametrize("model_class", [GaussianImputer, GaussianCopulaImputer])
class TestInputValidation:
    """Tests that input data is handled correctly and that malformed data is detected successfully."""

    def test_categorical_feature_check_only_continuous(self, dummy_model, model_class) -> None:
        """Test that no error is raised when all features are continuous with >2 unique values."""
        data = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        x = np.array([[2.0, 3.0, 4.0]])
        # Should not raise error
        model_class(model=dummy_model, data=data, x=x)

    def test_categorical_feature_check_binary_column(self, dummy_model, model_class) -> None:
        """Test that an exception is raised if the background data has column containing only two unique values."""
        data = np.array(
            [
                [1.0, 0, 3.0],
                [2.0, 1, 4.0],
                [3.0, 0, 5.0],
            ]
        )
        x = np.array([2.0, 1.0, 4.0])
        with pytest.raises(CategoricalFeatureError) as exc:
            model_class(model=dummy_model, data=data, x=x)
        msg = str(exc.value)
        # The second column (index 1) should be flagged as categorical, so 'f2' should be in the message
        assert "f2" in msg
        # The first and third columns should not be flagged, so 'f1' and 'f3' should not be in the message
        assert "f1" not in msg
        assert "f3" not in msg

    def test_categorical_feature_check_string(self, dummy_model, model_class) -> None:
        """Test that an exception is raised if the background data has a column containing string values."""
        data = np.array(
            [
                [1.0, "a", 3.0],
                [2.0, "b", 4.0],
                [3.0, "a", 5.0],
            ],
            dtype=object,
        )
        x = np.array([2.0, "b", 4.0], dtype=object)
        with pytest.raises(CategoricalFeatureError) as exc:
            model_class(model=dummy_model, data=data, x=x)
        msg = str(exc.value)
        assert "f2" in msg
        assert "f1" not in msg
        assert "f3" not in msg

    def test_categorical_feature_check_mixed(self, dummy_model, model_class) -> None:
        """Test that an exception is raised if the background data has binary and string-valued columns."""
        data = np.array(
            [
                [1.0, 0, "a", 3.0],
                [2.0, 1, "b", 4.0],
                [3.0, 0, "a", 5.0],
            ],
            dtype=object,
        )
        x = np.array([2.0, 1.0, "b", 4.0], dtype=object)
        with pytest.raises(CategoricalFeatureError) as exc:
            model_class(model=dummy_model, data=data, x=x)
        msg = str(exc.value)
        # Both f2 (index 1) and f3 (index 2) must be mentioned
        assert "f2" in msg
        assert "f3" in msg
        # f1 and f4 must not appear
        assert "f1" not in msg
        assert "f4" not in msg

    def test_x_explain_shapes(self, dummy_model, model_class):
        """Tests that an explain point can be passed both as a vector and a matrix with one row; both when passing in the constructor and when calling the fit() method."""
        data = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )
        x_explain = np.array([3, 2, 1])
        coalitions = np.array([[False, True, False]])

        imputer = model_class(model=dummy_model, data=data, x=x_explain.copy())
        imputer.value_function(coalitions)

        imputer = model_class(model=dummy_model, data=data, x=np.atleast_2d(x_explain.copy()))
        imputer.value_function(coalitions)

        imputer = model_class(model=dummy_model, data=data)
        imputer.fit(x_explain.copy())
        imputer.value_function(coalitions)

        imputer = model_class(model=dummy_model, data=data)
        imputer.fit(np.atleast_2d(x_explain.copy()))
        imputer.value_function(coalitions)

    def test_empty_background_data(self, dummy_model, model_class) -> None:
        """Test that a ValueError is raised when instantiating the imputer with an empty background data array."""
        data = np.array([])
        x = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="data.*empty"):
            model_class(model=dummy_model, data=data, x=x)
