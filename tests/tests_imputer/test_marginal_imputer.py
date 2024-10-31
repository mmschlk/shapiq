"""This test module contains all tests for the marginal imputer module of the shapiq package."""

import numpy as np
import pytest

from shapiq.games.imputer import MarginalImputer


def test_marginal_imputer_init():
    """Test the initialization of the marginal imputer."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    # get np data set of 10 rows and 3 columns of random numbers
    data = np.random.rand(10, 3)

    imputer = MarginalImputer(
        model=model,
        data=data,
        sample_size=10,
        random_state=42,
    )
    assert imputer.sample_size == 10
    assert imputer.random_state == 42
    assert imputer.n_features == 3

    # test with x
    x = np.random.rand(1, 3)
    imputer = MarginalImputer(
        model=model,
        data=data,
        x=x,
        random_state=42,
    )
    assert np.array_equal(imputer._x, x)
    assert imputer.n_features == 3
    assert imputer.random_state == 42

    # check with categorical features and a wrong numerical feature

    def model_cat(x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[0])

    data = np.asarray([["a", "b", 1], ["c", "d", 2], ["e", "f", 3]])
    categorical_features = [0]  # only first specified
    imputer = MarginalImputer(
        model=model_cat,
        data=data,
        categorical_features=categorical_features,
        random_state=42,
    )
    assert imputer._cat_features == [0]


def test_marginal_imputer_value_function():
    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    # get np data set of 10 rows and 3 columns of random numbers
    data = np.random.rand(10, 3)

    imputer = MarginalImputer(
        model=model,
        data=data,
        x=np.ones((1, 3)),
        sample_size=8,
        random_state=42,
    )

    imputed_values = imputer(np.array([[True, False, True], [False, True, False]]))
    assert len(imputed_values) == 2


def test_joint_marginal_distribution():
    """Test weather the marginal imputer correctly samples replacement values."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    data = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    data_as_tuples = [tuple(row) for row in data]
    data = np.array(data)
    x = np.array([1, 1, 1])

    imputer = MarginalImputer(
        model=model,
        data=data,
        x=x,
        sample_size=3,
        random_state=42,
        joint_marginal_distribution=False,
    )
    replacement_data_independent = imputer._sample_replacement_data(3)
    print(replacement_data_independent)

    imputer = MarginalImputer(
        model=model,
        data=data,
        x=x,
        sample_size=3,
        random_state=42,
        joint_marginal_distribution=True,
    )
    replacement_data_joint = imputer._sample_replacement_data(3)
    for i in range(3):
        assert tuple(replacement_data_joint[i]) in data_as_tuples
        # the below only works because of the random seed (might break in future)
        assert tuple(replacement_data_independent[i]) not in data_as_tuples


def test_raise_warning():

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    # get np data set of 10 rows and 3 columns of random numbers
    data = np.random.rand(10, 3)

    with pytest.warns(DeprecationWarning):
        _ = MarginalImputer(
            model=model,
            data=data,
            sample_replacements=False,  # deprecated
            sample_size=10,
            random_state=42,
        )
