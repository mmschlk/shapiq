"""This test module contains all tests for the baseline imputer module of the shapiq package."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.games.imputer import BaselineImputer


def test_baseline_init_background():
    """Test the initialization of the baseline imputer."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[0])

    data = np.array([[1, 2, "a"], [2, 3, "a"], [3, 4, "b"]], dtype=object)
    x = np.array([1, 2, 3])
    imputer = BaselineImputer(
        model=model,
        data=data,
        x=x,
        random_state=42,
    )
    assert np.array_equal(imputer.baseline_values, np.array([[2, 3, "a"]], dtype=object))

    baseline_values = np.zeros((1, 3))
    imputer.init_background(data=baseline_values)
    assert np.array_equal(imputer.baseline_values, baseline_values)


def test_baseline_imputer_with_model(dt_reg_model, background_reg_dataset):
    """Test the baseline imputer with a real model."""
    # create a coalitions
    data, target = background_reg_dataset
    x = data[0]
    coalitions = [
        [False for _ in range(data.shape[1])],
        [False for _ in range(data.shape[1])],
        [True for _ in range(data.shape[1])],
    ]
    coalitions[1][0] = True  # first feature is present
    coalitions = np.array(coalitions)

    imputer = BaselineImputer(
        model=dt_reg_model.predict,
        data=data,
        x=x,
        random_state=42,
        normalize=False,
    )
    assert np.array_equal(imputer.x[0], x)
    assert imputer.sample_size == 1
    assert imputer.random_state == 42
    assert imputer.n_features == data.shape[1]
    imputed_values = imputer(coalitions)
    assert len(imputed_values) == 3


def test_baseline_imputer_background_computation(background_reg_dataset):
    """Checks weather the baseline values from data are computed correctly with the mean/mode."""

    # set up a dummy model function that returns zeros
    def model_cat(x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[0])

    # make a feature into a categorical feature
    data, target = background_reg_dataset
    data = np.copy(data).astype(object)
    data[:, 0] = np.random.choice(["a", "b", "c"], size=data.shape[0])
    data[:, 1] = np.random.choice(["a", "b", "c"], size=data.shape[0])
    x = data[0]

    with pytest.warns(UserWarning):  # we expect the warning that feature 1 is not numerical
        imputer = BaselineImputer(
            model=model_cat,
            data=data,
            x=x,
            random_state=42,
            categorical_features=[0],  # only tell the imputer the first feature is categorical
        )

    # check if the categorical feature is correctly identified
    assert imputer._cat_features == [0, 1]

    # check if modes are correct
    for feature in imputer._cat_features:
        val, count = np.unique(data[:, feature], return_counts=True)
        mode = val[np.argmax(count)]
        assert imputer.baseline_values[0, feature] == mode

    # check if the mean is correct
    assert imputer.baseline_values[0, 2] == np.mean(data[:, 2])


def test_baseline_imputer_init():
    """Test the initialization of the marginal imputer."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    # get np data set of 10 rows and 3 columns of random numbers
    n_features = 3
    data = np.random.rand(10, n_features)

    # init with a baseline vector
    imputer = BaselineImputer(
        model=model,
        data=np.zeros((n_features,)),  # baseline vector of shape (n_features,)
        x=np.ones((1, n_features)),
        random_state=42,
    )
    assert imputer.sample_size == 1  # sample size is always 1 for baseline imputer
    assert imputer.random_state == 42
    assert imputer.n_features == 3

    # call with two inputs
    imputed_values = imputer(np.array([[False, False, False], [True, False, True]]))
    assert len(imputed_values) == 2
    assert imputed_values[0] == imputer.empty_prediction

    # test without x
    x = np.random.rand(1, 3)
    imputer = BaselineImputer(
        model=model,
        data=data,
        x=None,
        random_state=42,
    )
    assert imputer._x is None
    imputer.fit(x)
    assert np.array_equal(imputer.x, x)
    assert imputer.n_features == 3
    assert imputer.random_state == 42
    imputer.fit(x=np.ones((n_features,)))  # test with vector
    assert np.array_equal(imputer.x, np.ones((1, n_features)))
