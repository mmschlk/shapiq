"""This test module contains all tests for the baseline imputer module of the shapiq package."""

import numpy as np

from shapiq.games.imputer import BaselineImputer


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
    assert imputer._random_state == 42
    assert imputer._n_features == 3

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
    assert imputer._n_features == 3
    assert imputer._random_state == 42
    imputer.fit(x=np.ones((n_features,)))  # test with vector
    assert np.array_equal(imputer.x, np.ones((1, n_features)))

    def model_cat(x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[0])

    data = np.asarray([["a", "b", 1], ["c", "d", 2], ["e", "f", 3]])
    categorical_features = [0]  # only first specified
    imputer = BaselineImputer(
        model=model_cat,
        data=data,
        categorical_features=categorical_features,
        random_state=42,
    )
    assert imputer._cat_features == [0]
