"""This test module tests the tabpfn imputer object."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq import TabPFNImputer
from shapiq.explainer.utils import get_predict_function_and_model_type
from tests.markers import skip_if_no_tabpfn


@skip_if_no_tabpfn
@pytest.mark.external_libraries
def test_tabpfn_imputer(tabpfn_classification_problem):
    """Test the TabPFNImputer class."""
    import tabpfn

    # setup
    model, data, labels, x_test = tabpfn_classification_problem
    assert isinstance(model, tabpfn.TabPFNClassifier)
    if model.n_features_in_ == data.shape[1]:
        model.fit(data, labels)
    assert model.n_features_in_ == data.shape[1]
    assert not hasattr(model, "_shapiq_predict_function")

    # setup the tabpfn imputer
    prediction_function, _ = get_predict_function_and_model_type(model)
    imputer = TabPFNImputer(
        model=model,
        x_train=data,
        y_train=labels,
        x_test=x_test,
        predict_function=prediction_function,
    )
    imputer.fit(x=x_test[0])

    # test the imputer
    out_1 = imputer(np.asarray([True, True, True]))  # 3 features should now been fitted
    out_2 = imputer(np.asarray([True, True, False]))  # 2 features should now been fitted
    out_3 = imputer(np.asarray([False, True, False]))  # 1 feature should now been fitted
    assert out_1 != out_2
    assert out_1 != out_3
    assert out_2 != out_3


@skip_if_no_tabpfn
@pytest.mark.external_libraries
def test_empty_prediction(tabpfn_classification_problem):
    """Tests the TabPFNImputer with a manual empty prediction."""
    import tabpfn

    # setup
    model, data, labels, x_test = tabpfn_classification_problem
    assert isinstance(model, tabpfn.TabPFNClassifier)
    if model.n_features_in_ == data.shape[1]:
        model.fit(data, labels)
    assert model.n_features_in_ == data.shape[1]
    assert not hasattr(model, "_shapiq_predict_function")

    manual_empty_prediction = 1000

    # setup the tabpfn imputer
    prediction_function, _ = get_predict_function_and_model_type(model)
    imputer = TabPFNImputer(
        model=model,
        x_train=data,
        y_train=labels,
        x_test=x_test,
        predict_function=prediction_function,
        empty_prediction=manual_empty_prediction,
    )

    output = imputer(np.asarray([False, False, False]))
    assert output[0] == manual_empty_prediction


@skip_if_no_tabpfn
@pytest.mark.external_libraries
def test_tabpfn_imputer_validation(tabpfn_classification_problem):
    """Test that the TabPFNImputer raises a ValueError if no predict function is provided."""
    import tabpfn

    # setup
    model, data, labels, x_test = tabpfn_classification_problem
    assert isinstance(model, tabpfn.TabPFNClassifier)
    if model.n_features_in_ == data.shape[1]:
        model.fit(data, labels)
    assert model.n_features_in_ == data.shape[1]
    assert not hasattr(model, "_shapiq_predict_function")

    # no prediction function
    with pytest.raises(ValueError):
        _ = TabPFNImputer(
            model=model,
            x_train=data,
            y_train=labels,
            x_test=x_test,
            predict_function=None,
        )

    # no x_test and no empty prediction
    with pytest.raises(ValueError):

        def pred_fun(model, x):
            return model.predict_proba(x)[0]

        _ = TabPFNImputer(model=model, x_train=data, y_train=labels, predict_function=pred_fun)
